"""ChunkKdaFwdFinalize 的 GPU 双标杆。

golden 为独立 FP64 两项矩阵乘。control 由
``KDA_FINALIZE_GPU_REFERENCE=contract|upstream_strict`` 选择，默认 upstream_strict：

- contract：无 fla 依赖的改写 Triton kernel，消费完整 Aqk，固定配置；
- upstream_strict：调用已安装 fla 的真实 ``chunk_gla_fwd_o_gk`` 原函数，
  传 q=qg_scaled、g=0、scale=1；仅接受有效 chunk 内上三角为零的 Aqk。

两种 control 均不自动回退到 Torch。contract 删除了上游的 gate/scale、
下三角 mask、graph 分支和 autotune，不能代表原版 GPU 性能。
upstream_strict 保留上游 gate/scale 路径、输出初始化和调优。
scale=1 的乘法能否被编译器消除应以生成代码为准。
输入布局转换和元数据准备在 ATK GPU control 初始化时执行一次；输出布局
转换仍在调用内。源码身份和计时边界经 export_custom_data 留证。
"""

from __future__ import annotations

import os
import hashlib
import inspect
from pathlib import Path
from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # 无 triton 的环境仍可使用 FP64 torch 真值路径
    triton = None
    tl = None

CHUNK_SIZE = 64
HEAD_DIM = 128


# ---------------------------------------------------------------------------
# 布局 / 索引辅助：与 executor_chunk_kda_fwd_finalize.py 中的同名函数保持一致
#（本文件自包含，不 import executor / atk）。
# ---------------------------------------------------------------------------

def _head_major(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(0) if value.ndim == 3 else value


def _output_layout(value: torch.Tensor, layout: str) -> torch.Tensor:
    """head-major `[B,HV,T,V]` -> BNSD / BSND / NTD / TND。"""
    if layout == "BNSD":
        return value.contiguous()
    if layout == "BSND":
        return value.permute(0, 2, 1, 3).contiguous()
    if layout == "NTD":
        return value[0].contiguous()
    return value[0].permute(1, 0, 2).contiguous()


def _token_major(value: torch.Tensor) -> torch.Tensor:
    """head-major `[B,HV,T,D]`（或 `[HV,T,D]`）-> 上游 kernel 的 token-major `[B,T,HV,D]`。"""
    if value.ndim == 3:
        value = value.unsqueeze(0)
    return value.permute(0, 2, 1, 3).contiguous()


def _chunk_map(batch: int, tokens: int, cu: Optional[tuple[int, ...]]):
    if cu is None:
        return [[(start, min(start + CHUNK_SIZE, tokens), start // CHUNK_SIZE)
                 for start in range(0, tokens, CHUNK_SIZE)] for _ in range(batch)]
    chunks = []
    chunk_id = 0
    for begin, end in zip(cu, cu[1:]):
        for start in range(begin, end, CHUNK_SIZE):
            chunks.append((start, min(start + CHUNK_SIZE, end), chunk_id))
            chunk_id += 1
    return [chunks]


def _canonical_chunk_indices(cu: tuple[int, ...]) -> list[int]:
    """sequence-major 的 (seq, chunk) 平铺对，与 executor `_prepare` 的 canonical 校验同源。"""
    return [element
            for sequence, (left, right) in enumerate(zip(cu, cu[1:]))
            for chunk in range((right - left + CHUNK_SIZE - 1) // CHUNK_SIZE)
            for element in (sequence, chunk)]


# ---------------------------------------------------------------------------
# 抽取版 Triton kernel：上游 fla/ops/gla/chunk.py::chunk_gla_fwd_kernel_o
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit(do_not_specialize=['T'])
    def _chunk_kda_fwd_finalize_kernel(
        q,
        v,
        h,
        o,
        A,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        STATE_V_FIRST: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        # 上游 unflatten_program_id(tl.cdiv(V, BV)) 内联。
        pid = tl.program_id(0).to(tl.int64)
        NV = tl.cdiv(V, BV)
        i_v, i_t = pid % NV, pid // NV
        i_bh = tl.program_id(1).to(tl.int64)
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)
        if IS_VARLEN:
            i_tg = i_t.to(tl.int64)
            i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = eos - bos
            NT = tl.cdiv(T, BT)
        else:
            NT = tl.cdiv(T, BT)
            i_tg = (i_b * NT + i_t).to(tl.int64)
            bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

        # 上游在此处构造下三角 mask m_s；NPU 契约不 mask aqk，已删除。

        q += (bos * H + i_h) * K
        v += (bos * HV + i_hv) * V
        o += (bos * HV + i_hv) * V
        h += (i_tg * HV + i_hv).to(tl.int64) * K * V
        A += (bos * HV + i_hv) * BT

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        o_t = i_t * BT + tl.arange(0, BT)
        o_v = i_v * BV + tl.arange(0, BV)
        o_i = tl.arange(0, BT)
        m_t = o_t < T
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        m_A = m_t[:, None] & (o_i[None, :] < BT)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_qk = m_t[:, None] & m_k[None, :]
            p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
            if STATE_V_FIRST:
                p_h = h + o_v[:, None] * K + o_k[None, :]
                m_h = m_v[:, None] & m_k[None, :]
            else:
                p_h = h + o_k[:, None] * V + o_v[None, :]
                m_h = m_k[:, None] & m_v[None, :]

            # [BT, BK]：qg_scaled 已含 scale 与 gate（上游为 b_q * exp2(b_g)）。
            b_qg = tl.load(p_q, mask=m_qk, other=0.0)
            b_h = tl.load(p_h, mask=m_h, other=0.0)
            if STATE_V_FIRST:
                b_o = tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype), b_o)
            else:
                b_o = tl.dot(b_qg, b_h.to(b_qg.dtype), b_o)
        # 上游在此处 b_o *= scale；scale 已折叠进 qg_scaled。
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        p_o = o + o_t[:, None] * (HV * V) + o_v[None, :]
        p_A = A + o_t[:, None] * (HV * BT) + o_i[None, :]
        # [BT, BV]
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        # [BT, BT]：契约直接消费 aqk 全矩阵（上游 tl.where(m_s, ...) 已删除）；
        # 尾 chunk 的 j≥rows 列由 b_v 的行 mask（m_t）自动中和。
        b_A = tl.load(p_A, mask=m_A, other=0.0).to(b_v.dtype)
        b_o = tl.dot(b_A, b_v, b_o)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)


# ---------------------------------------------------------------------------
# 参考实现
# ---------------------------------------------------------------------------

def _torch_reference(inputs: Any, dtype: torch.dtype) -> torch.Tensor:
    """FP64 真值 / 契约精度 torch 标杆：镜像 executor 的 ``run_cpu``，但设备跟随输入。"""
    high_precision = dtype == torch.float64
    qg = _head_major(inputs.qg_scaled.to(torch.bfloat16)).to(dtype)
    aqk = _head_major(inputs.aqk.to(torch.bfloat16)).to(dtype)
    v_new = _head_major(inputs.v_new.to(torch.bfloat16)).to(dtype)
    h = inputs.h.to(torch.bfloat16).to(dtype)
    result = torch.zeros((inputs.batch, inputs.heads, inputs.tokens, HEAD_DIM),
                         dtype=dtype, device=qg.device)
    for batch_id, chunks in enumerate(_chunk_map(inputs.batch, inputs.tokens, inputs.cu_seqlens)):
        for begin, end, chunk_id in chunks:
            rows = end - begin
            state = h[batch_id, chunk_id]
            if inputs.state_v_first:
                state = state.transpose(-1, -2)
            qh = torch.matmul(qg[batch_id, :, begin:end], state)
            av = torch.matmul(aqk[batch_id, :, begin:end, :rows],
                              v_new[batch_id, :, begin:end])
            value = qh + av
            result[batch_id, :, begin:end] = value if high_precision else value.to(torch.bfloat16).float()
    return _output_layout(result, inputs.output_layout)


def _prepare_token_inputs(inputs: Any) -> dict[str, Any]:
    """准备 GPU control 共用的 token-major 输入和 canonical 元数据。"""
    device = inputs.qg_scaled.device
    if inputs.cu_seqlens is not None:
        cu = torch.tensor(inputs.cu_seqlens, dtype=torch.int64, device=device)
        flat_indices = (list(inputs.chunk_indices) if inputs.chunk_indices is not None
                        else _canonical_chunk_indices(inputs.cu_seqlens))
        ci = torch.tensor(flat_indices, dtype=torch.int64, device=device).view(-1, 2)
    else:
        cu = ci = None
    return dict(q=_token_major(inputs.qg_scaled), A=_token_major(inputs.aqk),
                v=_token_major(inputs.v_new), h=inputs.h.contiguous(),
                cu_seqlens=cu, chunk_indices=ci)


def _launch_contract(prepared: dict[str, Any], state_v_first: bool) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton 不可用；contract 标杆不会回退到 Torch")
    q, A, v, h = (prepared[name] for name in ("q", "A", "v", "h"))
    cu, ci = prepared["cu_seqlens"], prepared["chunk_indices"]
    B, T, HV, K = q.shape
    V = v.shape[-1]
    is_varlen = cu is not None
    NT = int(ci.shape[0]) if is_varlen else triton.cdiv(T, CHUNK_SIZE)
    # 上游注释：请确保 zeros，因为 vllm 会使用 padding v。
    o = torch.zeros_like(v)
    grid = (triton.cdiv(V, 128) * NT, B * HV)
    _chunk_kda_fwd_finalize_kernel[grid](
        q, v, h, o, A, cu, ci,
        T,
        H=HV,
        HV=HV,
        K=K,
        V=V,
        BT=CHUNK_SIZE,
        BK=64,
        BV=128,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
        num_warps=4,
        num_stages=3,
    )
    return o


def _triton_finalize(inputs: Any) -> torch.Tensor:
    """保留解释器寻址自测入口；正式 control 经 prepare_gpu_reference。"""
    o = _launch_contract(_prepare_token_inputs(inputs), inputs.state_v_first)
    return _output_layout(o.permute(0, 2, 1, 3), inputs.output_layout)


def _require_cuda_inputs(inputs: Any) -> None:
    tensors = (inputs.qg_scaled, inputs.aqk, inputs.v_new, inputs.h)
    device = tensors[0].device
    if device.type != "cuda" or any(value.device != device for value in tensors):
        raise ValueError("GPU control 的四个输入必须在同一 CUDA 设备")
    if any(value.dtype != torch.bfloat16 for value in tensors):
        raise ValueError("GPU control 的四个输入必须为 BF16")


def _validate_upstream_aqk(inputs: Any) -> None:
    """只检查实际参与计算的上三角；不修改输入或忽略有效非零值。"""
    # Packed 序列边界可能不对齐64，必须由每个真实 chunk 计算局部行号。
    local_rows, valid_columns = [], []
    for begin, end, _ in _chunk_map(inputs.batch, inputs.tokens, inputs.cu_seqlens)[0]:
        rows = end - begin
        local_rows.extend(range(rows))
        valid_columns.extend([rows] * rows)
    aqk = _head_major(inputs.aqk)
    row = torch.tensor(local_rows, device=aqk.device)[:, None]
    limit = torch.tensor(valid_columns, device=aqk.device)[:, None]
    col = torch.arange(CHUNK_SIZE, device=aqk.device)[None, :]
    upper = (col > row) & (col < limit)
    if torch.any((aqk != 0) & upper[None, None]).item():
        raise ValueError(
            "upstream_strict 只接受 chunk 内 r<j<rows 为零的 Aqk；"
            "当前输入具有有效上三角贡献。请使用 contract，或另备同源因果输入；"
            "不能仅在 GPU 端对 Aqk 做 tril。")


def _upstream_kwargs(inputs: Any) -> dict[str, Any]:
    prepared = _prepare_token_inputs(inputs)
    return dict(**prepared, g=torch.zeros_like(prepared["q"], dtype=torch.float32),
                scale=1.0, state_v_first=inputs.state_v_first,
                chunk_size=CHUNK_SIZE, use_graph=False)


def _load_upstream():
    try:
        from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
    except ImportError as exc:
        raise RuntimeError(
            "upstream_strict 需要在 GPU Python 环境安装原版 flash-linear-attention；"
            "不会回退到抽取 kernel 或 Torch") from exc
    # @dispatch 仅负责后端路由；固定到该函数的原始 Triton 实现。
    function = inspect.unwrap(chunk_gla_fwd_o_gk)
    path = Path(inspect.getsourcefile(function)).resolve()
    if function.__name__ != "chunk_gla_fwd_o_gk" or path.parts[-4:] != ("fla", "ops", "gla", "chunk.py"):
        raise RuntimeError(f"不是预期的原版 GLA Python 入口：{path}")
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return function, dict(upstream_source_path=str(path), upstream_source_sha256=source_sha,
                          upstream_dispatch="unwrapped_default_triton")


class GpuReference:
    """ATK 固定输入的 GPU control；准备操作和有效域检查在计时前完成。"""

    def __init__(self, inputs: Any, reference: Optional[str] = None):
        self.reference = reference or os.environ.get("KDA_FINALIZE_GPU_REFERENCE", "upstream_strict")
        if self.reference not in ("contract", "upstream_strict"):
            raise ValueError("KDA_FINALIZE_GPU_REFERENCE 必须为 contract 或 upstream_strict")
        _require_cuda_inputs(inputs)
        self.layout = inputs.output_layout
        self.state_v_first = inputs.state_v_first
        self.provenance = {}
        if self.reference == "upstream_strict":
            _validate_upstream_aqk(inputs)
            self.function, self.provenance = _load_upstream()
            self.prepared = _upstream_kwargs(inputs)
        else:
            if triton is None:
                raise RuntimeError("triton 不可用；contract 标杆不会回退到 Torch")
            self.prepared = _prepare_token_inputs(inputs)

    def __call__(self) -> torch.Tensor:
        if self.reference == "upstream_strict":
            o = self.function(**self.prepared)
        else:
            o = _launch_contract(self.prepared, self.state_v_first)
        return _output_layout(o.permute(0, 2, 1, 3), self.layout)

    def export_custom_data(self) -> dict[str, Any]:
        return dict(reference=("gpu_upstream_chunk_gla_fwd_o_gk" if self.reference == "upstream_strict"
                               else "gpu_contract_adapted_triton"),
                    gpu_reference_mode=self.reference,
                    input_preparation="once_in_init_excluded_from_control_call",
                    control_call_scope="output_zero+matmul_kernel+output_layout;not_kernel_only",
                    gpu_reference_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    **self.provenance)


def prepare_gpu_reference(inputs: Any, reference: Optional[str] = None) -> GpuReference:
    return GpuReference(inputs, reference)


def run_gpu(inputs: Any, high_precision: bool = True, reference: Optional[str] = None) -> torch.Tensor:
    """GPU 参考节点统一入口。

    ``inputs`` 为 executor ``Inputs`` 的 duck-typed 对象（本文件不 import
    executor / atk）。``high_precision=True`` 跑 FP64 torch 真值（golden）；
    ``False`` 跑所选同精度 GPU control；执行失败直接抛错。
    反复计时应预先调用 prepare_gpu_reference，避免计入输入准备。
    """
    if high_precision:
        return _torch_reference(inputs, torch.float64)
    return prepare_gpu_reference(inputs, reference)()


if __name__ == "__main__":
    # 自测：FP64 真值 / 契约精度 torch 两路对比 + Triton kernel 逻辑验证。
    # 注意：TRITON_INTERPRET=1 的解释器对 bf16 的 tl.dot 数值错乱（numpy 后端
    # 无 bf16），因此 CPU 解释模式下改用 fp32 张量喂 kernel（kernel 对 dtype
    # 不敏感），与手写 fp32 数学（不做最终 bf16 舍入）对比，验证寻址 / mask /
    # varlen / state_v_first / 布局逻辑；bf16 同精度路只在真 GPU 上可比。
    import argparse
    import types

    parser = argparse.ArgumentParser(description="gpu.py 自测")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", choices=("contract", "upstream_strict"), default="contract")
    parser.add_argument("--causal-aqk", action="store_true",
                        help="仅对本次新生成的自测输入构造 chunk 内下三角 Aqk")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interpret = device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1"
    print(f"device={device}  interpret={interpret}  "
          f"triton={'ok' if triton is not None else 'missing'}")

    torch.manual_seed(args.seed)

    def make_inputs(layout, batch, heads, tokens, cu, state_v_first, dtype=torch.bfloat16):
        chunks = len(_chunk_map(batch, tokens, cu)[0])
        head_only = layout in ("NTD", "TND")

        def rand(shape):
            return (torch.rand(shape, device=device) * 2 - 1).to(dtype)

        def shape_of(*tail):
            return (heads, *tail) if head_only else (batch, heads, *tail)

        inputs = types.SimpleNamespace(
            qg_scaled=rand(shape_of(tokens, HEAD_DIM)),
            aqk=rand(shape_of(tokens, CHUNK_SIZE)),
            v_new=rand(shape_of(tokens, HEAD_DIM)),
            h=rand((batch, chunks, heads, HEAD_DIM, HEAD_DIM)),
            cu_seqlens=cu,
            chunk_indices=(tuple(_canonical_chunk_indices(cu)) if cu is not None else None),
            output_layout=layout,
            state_v_first=state_v_first,
            batch=batch,
            heads=heads,
            tokens=tokens,
        )
        if args.causal_aqk:
            aqk = _head_major(inputs.aqk)
            for batch_id, chunk_map in enumerate(_chunk_map(batch, tokens, cu)):
                for begin, end, _ in chunk_map:
                    rows = end - begin
                    aqk[batch_id, :, begin:end, :rows] = torch.tril(aqk[batch_id, :, begin:end, :rows])
        return inputs

    def manual_fp32(case_inputs):
        """手写 fp32 契约数学（不做最终 bf16 舍入），与解释模式下的 kernel 同口径。"""
        qg = _head_major(case_inputs.qg_scaled).float()
        aqk = _head_major(case_inputs.aqk).float()
        v_new = _head_major(case_inputs.v_new).float()
        h = case_inputs.h.float()
        result = torch.zeros((case_inputs.batch, case_inputs.heads, case_inputs.tokens, HEAD_DIM),
                             dtype=torch.float32, device=qg.device)
        for batch_id, chunks in enumerate(_chunk_map(case_inputs.batch, case_inputs.tokens,
                                                     case_inputs.cu_seqlens)):
            for begin, end, chunk_id in chunks:
                rows = end - begin
                state = h[batch_id, chunk_id]
                if case_inputs.state_v_first:
                    state = state.transpose(-1, -2)
                result[batch_id, :, begin:end] = (
                    torch.matmul(qg[batch_id, :, begin:end], state)
                    + torch.matmul(aqk[batch_id, :, begin:end, :rows],
                                   v_new[batch_id, :, begin:end]))
        return _output_layout(result, case_inputs.output_layout)

    cases = [
        ("dense  BNSD       ", ("BNSD", 2, 4, 160, None, False)),
        ("dense  BSND vfirst", ("BSND", 1, 2, 96, None, True)),
        ("varlen NTD        ", ("NTD", 1, 4, 160, (0, 37, 101, 160), False)),
        ("varlen TND vfirst", ("TND", 1, 2, 160, (0, 64, 160), True)),
    ]
    for name, spec in cases:
        case_inputs = make_inputs(*spec)
        truth = run_gpu(case_inputs, high_precision=True).float()
        reference = _torch_reference(case_inputs, torch.float32).float()
        line = (f"[{name}] torch32-vs-fp64 max|diff|="
                f"{(reference - truth).abs().max().item():.3e}")
        if interpret:
            if args.reference != "contract":
                raise ValueError("解释器只能验证 contract 寻址，不能验证原版 CUDA 上游")
            fp32_inputs = types.SimpleNamespace(**{
                key: (value.float() if torch.is_tensor(value) else value)
                for key, value in vars(case_inputs).items()})
            kernel = _triton_finalize(fp32_inputs).float()
            line += (f"  fp32-kernel-vs-manual max|diff|="
                     f"{(kernel - manual_fp32(fp32_inputs)).abs().max().item():.3e}")
        else:
            control = run_gpu(case_inputs, high_precision=False, reference=args.reference).float()
            line += (f"  control-vs-fp64 max|diff|={(control - truth).abs().max().item():.3e}"
                     f"  control-vs-torch32 max|diff|={(control - reference).abs().max().item():.3e}")
        print(line)
