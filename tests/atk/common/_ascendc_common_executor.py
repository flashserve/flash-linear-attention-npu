"""AscendC ATK 基础公共工具。

本文件只保留各算子 executor 共用的轻量工具函数。具体输入构造、CPU 标杆、
run_cpu、run_npu 和 FunctionApi 均放在各算子自己的 ATK 目录中。
"""

from __future__ import annotations

import json
from typing import Any, Tuple

import torch

try:
    import numpy as np

    # ATK 读取 numpy 标量序列化输入时需要把这些类型加入 PyTorch 安全列表。
    torch.serialization.add_safe_globals(
        [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float32)),
            type(np.dtype(np.float64)),
            type(np.dtype(np.int32)),
            type(np.dtype(np.int64)),
        ]
    )
except (AttributeError, ImportError):
    pass


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}
_RCP_LN2 = 1.4426950408889634


def _to_python(value: Any) -> Any:
    """把 ATK 传入的 Tensor/bytes 标量转成普通 Python 对象。"""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _case_spec(input_data, op_name: str) -> dict[str, Any]:
    """读取 case_spec；若用例未显式提供，则从 attr 输入兜底组装。"""
    raw = _to_python(input_data.kwargs.get("case_spec"))
    if raw:
        if isinstance(raw, str):
            spec = json.loads(raw)
        elif isinstance(raw, dict):
            spec = raw
        else:
            raise TypeError(f"case_spec 类型不支持：{type(raw)!r}")
    else:
        spec = {
            key: _to_python(value)
            for key, value in input_data.kwargs.items()
            if not isinstance(value, torch.Tensor)
        }
    spec.setdefault("op", op_name)
    spec.setdefault("dtype", "bf16")
    return spec


def _marker_device(input_data) -> torch.device:
    """取 ATK marker tensor 所在设备，用于在 NPU 节点上构造真实输入。"""
    for value in input_data.kwargs.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _orig_dtype(name: str) -> torch.dtype:
    """把 case_spec 中的 dtype 名称转成 torch dtype。"""
    return _DTYPE_MAP.get(str(name).lower(), torch.bfloat16)


def _calc_dtype(name: str, high_precision: bool) -> torch.dtype:
    """CPU 高精度标杆使用 fp64，其余路径使用用例声明的原始精度。"""
    if high_precision:
        return torch.float64
    return _orig_dtype(name)


def _randn(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device, seed: int, scale: float = 0.05):
    """生成确定性正态分布输入；先量化到原始 dtype，再转到计算 dtype。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.randn(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * float(scale)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)


def _rand(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device, seed: int, low: float = 0.05, high: float = 0.95):
    """生成确定性均匀分布输入；常用于 beta 这类正值参数。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32)
    data = data * (float(high) - float(low)) + float(low)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)


def _zeros(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device):
    """生成指定 dtype/device 的零张量。"""
    data = torch.zeros(tuple(int(x) for x in shape), dtype=calc_dtype, device=device)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype)


def _gate(shape, calc_dtype: torch.dtype, device: torch.device, seed: int):
    """生成沿 T 维单调递减的 GDN gate，避免 exp 溢出。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * 0.01 + 0.001
    data = -torch.cumsum(data, dim=-1)
    return data.to(calc_dtype).to(device)


def _kda_gate(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device, seed: int):
    """生成 KDA gate，沿 token 维做 chunk 前的稳定递减。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * 0.01 + 0.001
    data = -torch.cumsum(data, dim=-2)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)


def _int_tensor(values, device: torch.device, dtype=torch.int64):
    """生成与输入同设备的整型元数据 Tensor。"""
    return torch.tensor(values, dtype=dtype, device=device)


def _chunks(total: int, chunk_size: int):
    """按 chunk_size 生成左闭右开的 token 范围。"""
    total = int(total)
    chunk_size = int(chunk_size)
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def _num_chunks(total: int, chunk_size: int) -> int:
    """计算定长场景的 chunk 数。"""
    return (int(total) + int(chunk_size) - 1) // int(chunk_size)


def _finite_tuple(outputs) -> Tuple[torch.Tensor, ...]:
    """过滤 None 输出，并在 ATK 读取前检查浮点输出是否有限。"""
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    visible = []
    for output in outputs:
        if output is None or not isinstance(output, torch.Tensor):
            continue
        check = output.detach()
        if check.is_floating_point() and not torch.isfinite(check.float()).all().item():
            raise RuntimeError("输出包含 NaN 或 Inf")
        visible.append(output)
    return tuple(visible)
