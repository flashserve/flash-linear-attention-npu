"""fused_recurrent_rwkv8 (WKV7) 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
CPU 标杆内嵌在本文件中（ATK 规范：executor 自包含，不跨目录 import 金标）；
它与 fla/ops/ascendc/rwkv8/fused_recurrent_rwkv8/tests/pta/golden.py 是同一份
逻辑的两份拷贝——修改金标算法时两处必须同步。

递推公式（per head，state (V,K)，RWKV 朝向：行 = v/q 侧，列 = k/z 侧）：
    sa    = state @ z_t
    state = state * decay_t[None, :] + sa[:, None] * b_t[None, :]
            + v_t[:, None] * k_t[None, :]
    y_t   = state @ (q_t * scale)
decay = exp(-exp(w))（w 为 log 域衰减参数，对齐 wkv7_cuda.cu:21）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, NamedTuple, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _case_spec,
    _finite_tuple,
    _marker_device,
    _orig_dtype,
)

OP_NAME = "fused_recurrent_rwkv8"


def _rwkv8_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool) -> dict[str, Any]:
    """结构化造数（与仓外 ascendc/scripts/gen_data.py 同配方）：
    q/k/v ~ randn；w = -rand*2-0.1（保证 decay ∈ (0,1)）；
    kk = L2normalize(randn)，z = -kk，b = kk * randn。
    禁止无约束 randn 造 z/b（delta-rule 状态会指数爆炸）。

    生成在 CPU 上按 seed 确定，先量化到用例 dtype 再转到计算 dtype
    （CPU 高精度标杆 fp64 / 其余路径为用例原始 dtype），保证两个
    ATK 节点看到完全一致的输入。
    """
    dtype_name = str(spec.get("dtype", "fp32"))
    calc_dtype = torch.float64 if high_precision else _orig_dtype(dtype_name)
    seed = int(spec.get("seed", 42))
    B, H, T, K, V = (int(spec[x]) for x in ("B", "H", "T", "K", "V"))

    def _make(shape, offset, factory):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + offset)
        data = factory(shape, gen)
        return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)

    q = _make((B, H, T, K), 1, lambda s, g: torch.randn(s, generator=g))
    w = _make((B, H, T, K), 2, lambda s, g: -torch.rand(s, generator=g) * 2.0 - 0.1)
    k = _make((B, H, T, K), 3, lambda s, g: torch.randn(s, generator=g))
    v = _make((B, H, T, V), 4, lambda s, g: torch.randn(s, generator=g))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 5)
    kk = torch.nn.functional.normalize(torch.randn((B, H, T, K), generator=gen), p=2, dim=-1)
    z = (-kk).to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)
    b = (kk * torch.randn((B, H, T, K), generator=gen)).to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)

    initial_state = None
    if bool(spec.get("initial_state", False)):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 6)
        initial_state = torch.randn((B, H, K, V), generator=gen)
        initial_state = initial_state.to(torch.float64 if high_precision else torch.float32).to(device)

    return {
        "q": q, "w": w, "k": k, "v": v, "z": z, "b": b,
        "initial_state": initial_state,
        "scale": float(spec.get("scale", 1.0)),
        "chunk_len": int(spec.get("chunk_len", 16)),
        "output_chunk_state": bool(spec.get("output_chunk_state", False)),
        "output_sa": bool(spec.get("output_sa", False)),
    }


# ---- CPU 标杆（内嵌拷贝，源头见 tests/pta/golden.py）----

class _Rwkv8Result(NamedTuple):
    o: torch.Tensor
    s: Optional[torch.Tensor]
    sa: Optional[torch.Tensor]


def _wkv7_decay(w: torch.Tensor) -> torch.Tensor:
    """log 域衰减参数 -> 衰减值：decay = exp(-exp(w))，与 wkv7_cuda.cu:21 一致。"""
    return torch.exp(-torch.exp(w))


def fused_recurrent_rwkv8_reference(
    q: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_chunk_state: bool = False,
    output_sa: bool = False,
    chunk_len: int = 16,
) -> _Rwkv8Result:
    """逐 token 递推参考实现（einsum 版）。输入 fp64 时全程 fp64 累加，
    否则 fp32 累加；o 回投到输入 dtype，s/sa 为计算精度。"""
    orig_dtype = q.dtype
    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    q, w, k, v, z, b = (x.to(calc_dtype) for x in (q, w, k, v, z, b))
    B, H, T, K = q.shape
    V = v.shape[-1]

    if initial_state is None:
        state = q.new_zeros(B, H, V, K)
    else:
        state = initial_state.to(calc_dtype).transpose(-1, -2).clone()  # 接口 (K,V) → 内部 (V,K)
    decay = _wkv7_decay(w)  # (B, H, T, K)

    o = torch.empty(B, H, T, V, dtype=calc_dtype)
    sa_out = torch.empty(B, H, T, V, dtype=calc_dtype) if output_sa else None
    s_snaps = []
    for t in range(T):
        sa = torch.einsum('bhij,bhj->bhi', state, z[:, :, t])
        if output_sa:
            sa_out[:, :, t] = sa
        state = (state * decay[:, :, t].unsqueeze(-2)
                 + sa.unsqueeze(-1) * b[:, :, t].unsqueeze(-2)
                 + v[:, :, t].unsqueeze(-1) * k[:, :, t].unsqueeze(-2))
        o[:, :, t] = torch.einsum('bhij,bhj->bhi', state, q[:, :, t] * scale)
        if output_chunk_state and (t + 1) % chunk_len == 0:
            s_snaps.append((t // chunk_len, state.transpose(-1, -2)))

    o = o.to(orig_dtype)
    if output_chunk_state:
        s = q.new_zeros(B, H, T // chunk_len, K, V)
        for slot, snap in s_snaps:
            s[:, :, slot] = snap
    else:
        s = None
    return _Rwkv8Result(o, s, sa_out)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """Run the CPU reference at original or fp64 precision."""
    inputs = _rwkv8_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    res = fused_recurrent_rwkv8_reference(
        inputs["q"], inputs["w"], inputs["k"], inputs["v"], inputs["z"], inputs["b"],
        scale=inputs["scale"],
        initial_state=inputs["initial_state"],
        output_chunk_state=inputs["output_chunk_state"],
        output_sa=inputs["output_sa"],
        chunk_len=inputs["chunk_len"],
    )
    return (res.o, res.s, res.sa)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = _rwkv8_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.fused_recurrent_rwkv8(
        inputs["q"], inputs["w"], inputs["k"], inputs["v"], inputs["z"], inputs["b"],
        scale=inputs["scale"],
        initial_state=inputs["initial_state"],
        output_chunk_state=inputs["output_chunk_state"],
        output_sa=inputs["output_sa"],
        chunk_len=inputs["chunk_len"],
    )


@register("executor_fused_recurrent_rwkv8")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
