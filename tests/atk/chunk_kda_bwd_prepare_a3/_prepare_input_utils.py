from typing import Any, Tuple
import torch
_DTYPE_MAP = {"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32,"fp64":torch.float64}

def _marker_device(input_data) -> torch.device:
    """取 ATK marker tensor 所在设备，用于在 NPU 节点上构造真实输入。"""
    for value in input_data.kwargs.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")

def _orig_dtype(name: str) -> torch.dtype:
    """把 case_spec 中的 dtype 名称转成 torch dtype。"""
    return _DTYPE_MAP.get(str(name).lower(), torch.bfloat16)

def _randn(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device, seed: int, scale: float = 0.05):
    """生成确定性正态分布输入；先量化到原始 dtype，再转到计算 dtype。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.randn(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * float(scale)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)

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
