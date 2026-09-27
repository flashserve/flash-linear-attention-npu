"""示例文件：tests/atk/op_name/executor_op_name.py

说明：文件名按本仓 tests/atk/README.md 的 `executor_<算子>.py`；内容是 ATK 的 `api_type`
执行插件，按 ATK 仓 skill/atk-quality-guard 与 references/atk_user_guide.md 编写
（skill 只提供内容要求，不改命名）。

注意事项：
  1. `@register` 名称必须与 op_name.yaml 顶层 `api_type` 一致（本示例：`executor_op_name`）。
  2. 同一插件内按 `self.device` 分支：`"npu"` 走算子真实接口（本仓稳定入口
     `fla_npu.ops.ascendc.op_name`，发布物路径用 `FLA_NPU_PUBLISH_PATH` 注入），
     `"cpu"` 走 PyTorch 原生 golden。
  3. **用例输入就是算子真实 inputs/attr**：x、g、a_log、initial_state、cu_seqlens、chunk_indices、
     layout、chunk_size、scale、epsilon、return_saved，逐项与用例 JSON 的 inputs 同名。
     不要从 case_spec / low_precision_marker / fp32_marker 之类的占位参数里解析用例语义。
  4. 张量由 ATK 按 backend 生成（NPU 节点生成在 NPU、CPU 节点生成在 CPU），executor 不再搬运设备；
     标量 Attr 可能以 0 维张量或 bytes 传入，先用 `_scalar` 转成 Python 标量再下传。
  5. 变长元数据必须归一：ATK 生成的是随机 INT64，需要在本文件里变成合法前缀和，并按 chunk_size
     重算 chunk_indices。归一逻辑只写一份，NPU 与 CPU 两条路径共用，避免两路切分口径不一致。
  6. CPU golden 用 PyTorch 原生算子实现；低精度输入先转 fp32 计算，最后 cast 回目标 dtype。
  7. `with_output=True` 时才返回结果；返回值就是算子的真实输出 y、state、x_norm
     （`return_saved=False` 时后两项为 None），个数/shape/dtype 要能支撑双标杆比较。
  8. 本文件不修改算子源码或 ATK 框架代码；精度失败按 troubleshooting.md 区分类别后报告，
     不在这里改判结论。
"""

from __future__ import annotations

import os
import sys

import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# 算子发布物路径（Phase B 由环境变量注入），例如 custom OPP 的 python 包目录。
PUBLISH_PATH = os.environ.get("FLA_NPU_PUBLISH_PATH", "")

# layout → (搬到 [..., H, T, D] 规范形的 permute, token 轴)。D 轴在所有 layout 下都是最后一维；
# 规范形把 head 轴固定到 -3、token 轴固定到 -2，CPU 标杆只按一种形状写。
_LAYOUT = {
    "BSND": ((0, 2, 1, 3), 1),   # [B,T,H,D] → [B,H,T,D]
    "BNSD": ((0, 1, 2, 3), 2),   # [B,H,T,D] → [B,H,T,D]
    "TND": ((1, 0, 2), 0),       # [T,H,D]   → [H,T,D]
    "NTD": ((0, 1, 2), 1),       # [N,T,D]   → [N,T,D]
}


def _inverse_perm(perm):
    return tuple(sorted(range(len(perm)), key=perm.__getitem__))


def _normalize_varlen(cu_seqlens, total_tokens, chunk_size):
    """把随机 INT64 元数据归一成合法变长元数据，并按 chunk_size 重算 chunk_indices。

    cu_seqlens：从 0 开始、单调不减、末元素等于总 token 数。
    chunk_indices：每个序列内 (序列号, 序列内 chunk 号) 的列表，两列 INT64。
    """
    raw = cu_seqlens.reshape(-1).to(torch.int64).abs().clamp(min=1)
    ends = torch.cumsum(raw, dim=0).clamp(max=int(total_tokens))
    ends[-1] = int(total_tokens)
    seq = torch.unique_consecutive(
        torch.cat([torch.zeros(1, dtype=torch.int64, device=ends.device), ends]))
    if seq.numel() < 2:
        seq = torch.tensor([0, int(total_tokens)], dtype=torch.int64, device=ends.device)

    rows = []
    for index in range(seq.numel() - 1):
        start, stop = int(seq[index]), int(seq[index + 1])
        chunk_count = (max(stop - start, 0) + chunk_size - 1) // chunk_size
        rows.extend((index, chunk) for chunk in range(chunk_count))
    indices = torch.tensor(rows or [(0, 0)], dtype=torch.int64, device=ends.device)
    return seq, indices.reshape(-1, 2)


def _chunk_ranges(cu_seqlens, total_tokens, chunk_size):
    """返回 [(start, stop), ...]；有 cu_seqlens 时 chunk 不跨序列边界。"""
    if cu_seqlens is None:
        return [(start, min(start + chunk_size, total_tokens))
                for start in range(0, total_tokens, chunk_size)]
    segments = [int(value) for value in cu_seqlens.reshape(-1).tolist()]
    ranges = []
    for start, stop in zip(segments[:-1], segments[1:]):
        ranges.extend((cursor, min(cursor + chunk_size, stop))
                      for cursor in range(start, stop, chunk_size))
    return ranges


def forward_v2(x, g, a_log=None, initial_state=None, cu_seqlens=None, chunk_indices=None,
               layout="BSND", scale=1.0, chunk_size=64, epsilon=1.0e-6, return_saved=False):
    """NPU 路径：调用算子真实接口，输入就是用例声明的真实张量与 Attr。"""
    if PUBLISH_PATH:
        sys.path.insert(0, PUBLISH_PATH)
    from fla_npu.ops.ascendc import op_name

    return op_name(x, g, a_log=a_log, initial_state=initial_state,
                   cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                   layout=layout, scale=scale, chunk_size=chunk_size,
                   epsilon=epsilon, return_saved=return_saved)


def forward_golden_v2(x, g, a_log=None, initial_state=None, cu_seqlens=None, chunk_indices=None,
                      layout="BSND", scale=1.0, chunk_size=64, epsilon=1.0e-6,
                      return_saved=False):
    """CPU 标杆：PyTorch 原生实现（低精度先转 fp32，再 cast 回目标 dtype）。"""
    del chunk_indices  # 只用于 NPU 路径；CPU 标杆从归一后的 cu_seqlens 自己推导切分
    perm, token_axis = _LAYOUT[layout]
    total_tokens = int(x.shape[token_axis])
    chunk_size = int(chunk_size)

    x_c = x.permute(perm).to(torch.float32)
    g_c = g.permute(perm).to(torch.float32)
    dim = int(x_c.shape[-1])
    norm = torch.rsqrt((x_c * x_c).sum(dim=-1, keepdim=True) / float(dim) + float(epsilon))

    # a_log 非空时门控系数是 exp(a_log) * scale；为空时直接用 scale。
    coefficient = float(scale)
    if a_log is not None:
        head = a_log.to(torch.float32).reshape(*([1] * (g_c.dim() - 2)), -1, 1)
        coefficient = torch.exp(head) * float(scale)
    gate = g_c * coefficient

    # carry 是每个 (head, D) 位上的累计值：chunk 之间续算，序列之间不续算。
    # initial_state 的 B/H 必须跟 x 对齐；varlen 下 B=1，reshape 后与规范形的 head 维一致。
    carry = None
    if initial_state is not None:
        carry = initial_state.to(torch.float32).reshape(
            *x_c.shape[:-2], int(x_c.shape[-1]))
    scan = torch.zeros_like(x_c)
    ends = []
    for start, stop in _chunk_ranges(cu_seqlens, total_tokens, chunk_size):
        part = torch.cumsum(gate[..., start:stop], dim=-1)
        part = part.unsqueeze(-1).expand(*part.shape, dim)
        if carry is not None:
            part = part + carry.unsqueeze(-2)
        else:
            carry = torch.zeros_like(part[..., -1, :])
        scan[..., start:stop, :] = part
        carry = part[..., -1, :].contiguous()
        ends.append(carry)

    back = _inverse_perm(perm)
    y = (x_c * scan * norm).to(x.dtype).permute(back).contiguous()
    if not return_saved:
        return y, None, None
    state = torch.stack(ends, dim=-2).to(x.dtype) if ends else None   # [..., H, chunk_count, D]
    x_norm = norm.expand(x_c.shape).permute(back).contiguous()     # 与 x 同 shape，FP32
    return y, state, x_norm


def _argument(input_data, name, default=None):
    """按用例 JSON/YAML 里 inputs 的 name 取本次用例的实参。"""
    for container in (getattr(input_data, "kwargs", None) or {},
                      getattr(input_data, "method_kwargs", None) or {}):
        if name in container:
            return container[name]
    for item in (getattr(input_data, "args", None) or ()):
        if isinstance(item, dict) and name in item:
            return item[name]
    return default


def _scalar(value, default=None):
    """ATK 可能用 0 维张量或 bytes 传 Attr，这里统一成 Python 标量。"""
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else default
    return value


@register("executor_op_name")
class OpNameExecutor(BaseApi):
    """ATK 执行插件：注册名与用例 `api_type` 一致，按 device 分支。"""

    def __call__(self, input_data, with_output: bool = False):
        call_kwargs = dict(
            x=_argument(input_data, "x"),
            g=_argument(input_data, "g"),
            a_log=_argument(input_data, "a_log"),
            initial_state=_argument(input_data, "initial_state"),
            cu_seqlens=_argument(input_data, "cu_seqlens"),
            chunk_indices=_argument(input_data, "chunk_indices"),
            layout=str(_scalar(_argument(input_data, "layout"), "BSND")),
            scale=float(_scalar(_argument(input_data, "scale"), 1.0)),
            chunk_size=int(_scalar(_argument(input_data, "chunk_size"), 64)),
            epsilon=float(_scalar(_argument(input_data, "epsilon"), 1.0e-6)),
            return_saved=bool(_scalar(_argument(input_data, "return_saved"), False)),
        )
        if call_kwargs["x"] is None or call_kwargs["g"] is None:
            raise RuntimeError("executor_op_name: 用例缺少 x/g 输入。")

        # 变长元数据归一：NPU 与 CPU 共用同一份结果，避免两路切分口径不一致。
        if call_kwargs["cu_seqlens"] is not None:
            token_axis = _LAYOUT[call_kwargs["layout"]][1]
            total_tokens = int(call_kwargs["x"].shape[token_axis])
            call_kwargs["cu_seqlens"], call_kwargs["chunk_indices"] = _normalize_varlen(
                call_kwargs["cu_seqlens"], total_tokens, call_kwargs["chunk_size"])

        if self.device == "npu":
            outputs = forward_v2(**call_kwargs)
        elif self.device == "cpu":
            outputs = forward_golden_v2(**call_kwargs)
        else:
            raise RuntimeError(f"Unsupported device: {self.device}")

        return outputs if with_output else None
