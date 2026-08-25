"""solve_tri 的 ATK executor。

输入生成、CPU 标杆 (MCH+MBH 算法)、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
支持 bsnd 和 tnd 两种 layout。
"""

from __future__ import annotations

import json
import random
import sys
import os
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch

# 添加父目录到 sys.path 以导入公共模块
_PARENT_DIR = str(Path(__file__).resolve().parents[1])
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


# ============================================================================
# 从 _ascendc_common_executor 复制必要的工具函数（避免导入问题）
# ============================================================================
_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def _to_python(value):
    """把 ATK 传入的 Tensor/bytes 标量转成普通 Python 对象。"""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _case_spec(input_data, op_name: str):
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


def _marker_device(input_data):
    """取 ATK marker tensor 所在设备，用于在 NPU 节点上构造真实输入。"""
    for value in input_data.kwargs.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _orig_dtype(name: str):
    """把 case_spec 中的 dtype 名称转成 torch dtype。"""
    return _DTYPE_MAP.get(str(name).lower(), torch.bfloat16)


def _calc_dtype(name: str, high_precision: bool):
    """CPU 高精度标杆使用 fp32（与 NPU MCH+MBH 精度对齐），其余路径使用用例声明的原始精度。"""
    if high_precision:
        return torch.float32
    return _orig_dtype(name)


def _finite_tuple(outputs):
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

OP_NAME = "solve_tri"


# ============================================================================
# 输入生成
# ============================================================================
def _make_lower_tri_block(actual_size: int, chunk_size: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """生成严格下三角块（对角线为0）。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    block = torch.randn(actual_size, chunk_size, generator=gen, dtype=torch.float32) * 0.1
    # 只保留 actual_size x actual_size 的下三角部分
    block[:, :actual_size] = torch.tril(block[:, :actual_size], diagonal=-1)
    # 其余列置零
    if chunk_size > actual_size:
        block[:, actual_size:] = 0
    return block.to(dtype)


def _generate_cu_seqlens(num_seqs: int, total_T: int, chunk_size: int, seed: int) -> List[int]:
    """生成 TND 模式的 cu_seqlens。
    
    每个序列至少 chunk_size 长度，确保每个序列至少有一个完整 chunk。
    如果 total_T 不够分配，自动减少 num_seqs。
    """
    random.seed(seed)
    if num_seqs == 1:
        return [0, total_T]
    
    # 确保每个序列至少有 chunk_size 长度
    min_len = chunk_size
    max_seqs = total_T // min_len
    if max_seqs < num_seqs:
        num_seqs = max(1, max_seqs)
    
    if num_seqs == 1:
        return [0, total_T]
    
    remaining = total_T - num_seqs * min_len
    seq_lens = [min_len] * num_seqs
    # 随机分配剩余长度
    for _ in range(remaining):
        idx = random.randint(0, num_seqs - 1)
        seq_lens[idx] += 1
    
    cu_seqlens = [0]
    for length in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return cu_seqlens


def _prepare_chunk_indices(cu_seqlens: List[int], chunk_size: int) -> List[int]:
    """生成 TND 模式的 chunk_indices (flatten)。"""
    indices = []
    num_seqs = len(cu_seqlens) - 1
    for seq_idx in range(num_seqs):
        bos = cu_seqlens[seq_idx]
        eos = cu_seqlens[seq_idx + 1]
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            indices.extend([seq_idx, chunk_idx])
    return indices


def build_inputs_bsnd(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    """构造 BSND 格式输入。"""
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    orig_dtype = _orig_dtype(dtype_name)
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, H, T, chunk_size = (int(spec[x]) for x in ("B", "H", "T", "chunk_size"))
    
    # BSND: [B, T, H, chunk_size]
    x = torch.zeros(B, T, H, chunk_size, dtype=calc_dtype, device="cpu")
    num_chunks = (T + chunk_size - 1) // chunk_size
    
    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual_size = e - s
                block_seed = seed + b * 10000 + h * 1000 + c
                block = _make_lower_tri_block(actual_size, chunk_size, calc_dtype, block_seed)
                x[b, s:e, h, :] = block
    
    # 先转到原始精度再转到计算精度（模拟量化）
    x = x.to(orig_dtype).to(calc_dtype).to(device)
    return {"x": x, "cu_seqlens": None, "chunk_indices": None, "layout": "bsnd"}


def build_inputs_tnd(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    """构造 TND 格式输入。"""
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    orig_dtype = _orig_dtype(dtype_name)
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    H, T, chunk_size = (int(spec[x]) for x in ("H", "T", "chunk_size"))
    num_seqs = int(spec.get("num_seqs", 1))
    
    cu_seqlens = _generate_cu_seqlens(num_seqs, T, chunk_size, seed)
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
    total_T = cu_seqlens[-1]
    
    # TND: [total_T, H, chunk_size]
    x = torch.zeros(total_T, H, chunk_size, dtype=calc_dtype, device="cpu")
    
    for seq_idx in range(num_seqs):
        bos = cu_seqlens[seq_idx]
        eos = cu_seqlens[seq_idx + 1]
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                actual_size = e - s
                block_seed = seed + seq_idx * 100000 + h * 1000 + c
                block = _make_lower_tri_block(actual_size, chunk_size, calc_dtype, block_seed)
                x[s:e, h, :] = block
    
    x = x.to(orig_dtype).to(calc_dtype).to(device)
    return {"x": x, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices, "layout": "tnd"}


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    """根据 layout 构造输入。"""
    layout = str(spec.get("layout", "bsnd")).lower()
    if layout == "tnd":
        return build_inputs_tnd(spec, device, high_precision)
    else:
        return build_inputs_bsnd(spec, device, high_precision)


# ============================================================================
# MCH+MBH 算法实现 (CPU Golden)
# ============================================================================
def _to_low_precision(x: np.ndarray, use_bf16: bool) -> np.ndarray:
    """转换到低精度 (fp16 或 bf16 模拟)。"""
    if use_bf16:
        return torch.from_numpy(x.astype(np.float32)).bfloat16().float().numpy()
    return x.astype(np.float16)


def _mch_invert_16x16(block: np.ndarray, use_bf16: bool = False, use_fp32: bool = False) -> np.ndarray:
    """MCH 算法求 16x16 块的逆。
    
    计算 (I + A)^{-1}，其中 A 是严格下三角矩阵。
    使用迭代公式：X = I - A, Y = A^2, 然后 X = X @ Y + X, Y = Y @ Y 迭代 3 次。
    """
    n = block.shape[0]
    
    if use_fp32:
        # FP32 高精度模式
        A = block.astype(np.float32)
        I = np.eye(n, dtype=np.float32)
        Y = A @ A
        X = I - A
        for _ in range(3):
            X = X @ Y + X
            Y = Y @ Y
        return X
    
    # 低精度模式
    def to_low(x):
        return _to_low_precision(x, use_bf16)
    
    def matmul_low(a, b):
        result = a.astype(np.float32) @ b.astype(np.float32)
        return to_low(result)
    
    A = to_low(block)
    I = to_low(np.eye(n, dtype=np.float32))
    neg_I = to_low(-np.eye(n, dtype=np.float32))

    Y = matmul_low(A, A)
    acc = I.astype(np.float32) @ I.astype(np.float32)
    acc += neg_I.astype(np.float32) @ A.astype(np.float32)
    X = to_low(acc)

    for iter_idx in range(3):
        acc = X.astype(np.float32) @ Y.astype(np.float32)
        acc += X.astype(np.float32) @ I.astype(np.float32)
        X = to_low(acc)
        if iter_idx < 2:
            Y = matmul_low(Y, Y)

    return X


def _extract_block_diagonal(matrix: np.ndarray, block_size: int, start: int) -> np.ndarray:
    """提取块对角矩阵。"""
    n = matrix.shape[0]
    result = np.zeros_like(matrix)
    num_blocks = n // block_size
    for blk in range(start, num_blocks, 2):
        r0 = blk * block_size
        r1 = r0 + block_size
        result[r0:r1, r0:r1] = matrix[r0:r1, r0:r1]
    return result


def _mbh_recursive_merge(x_low: np.ndarray, m_low: np.ndarray, matrix_size: int, 
                         use_bf16: bool = False, use_fp32: bool = False) -> np.ndarray:
    """MBH 递归合并算法。"""
    FRAC = 16
    n = matrix_size
    
    if use_fp32:
        I = np.eye(n, dtype=np.float32)
        M_neg = -m_low.astype(np.float32)
        X = x_low.copy()
        block_size = FRAC
        while block_size < n:
            driving = _extract_block_diagonal(X, block_size, 1)
            other = _extract_block_diagonal(X, block_size, 0)
            Y_result = I + driving @ M_neg
            X = Y_result @ other + driving
            block_size *= 2
        return X
    
    def to_low(x):
        return _to_low_precision(x, use_bf16)
    
    I = to_low(np.eye(n, dtype=np.float32))
    M_neg = to_low(-m_low.astype(np.float32))
    X = x_low.copy()

    block_size = FRAC
    while block_size < n:
        driving = _extract_block_diagonal(X, block_size, 1)
        other = _extract_block_diagonal(X, block_size, 0)

        acc = I.astype(np.float32) @ I.astype(np.float32)
        acc += driving.astype(np.float32) @ M_neg.astype(np.float32)
        Y_result = to_low(acc)

        acc = Y_result.astype(np.float32) @ other.astype(np.float32)
        acc += I.astype(np.float32) @ driving.astype(np.float32)
        X = to_low(acc)

        block_size *= 2

    return X


def _inverse_block_mch_mbh(block: np.ndarray, matrix_size: int, 
                           use_bf16: bool = False, use_fp32: bool = False) -> np.ndarray:
    """使用 MCH+MBH 算法求块的逆。
    
    Args:
        block: 输入块，shape [actual_size, actual_size]
        matrix_size: chunk_size，用于 padding
        use_bf16: 是否使用 bf16 模拟
        use_fp32: 是否使用 fp32 高精度
    
    Returns:
        逆矩阵块
    """
    FRAC = 16
    n = block.shape[0]  # actual size
    
    # Padding 到 matrix_size
    if n < matrix_size:
        padded = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        padded[:n, :n] = block
        block = padded
        padded_size = matrix_size
    else:
        padded_size = n
    
    if use_fp32:
        block_work = block.astype(np.float32)
    else:
        block_work = _to_low_precision(block, use_bf16)

    if padded_size <= FRAC:
        result = _mch_invert_16x16(block_work, use_bf16, use_fp32)
        return result[:n, :n]

    # MCH: 对每个 16x16 对角块求逆
    num_fracs = padded_size // FRAC
    x_mch = np.zeros((padded_size, padded_size), dtype=np.float32)
    for i in range(num_fracs):
        r0 = i * FRAC
        r1 = r0 + FRAC
        sub_block = block_work[r0:r1, r0:r1]
        x_mch[r0:r1, r0:r1] = _mch_invert_16x16(sub_block, use_bf16, use_fp32)

    if not use_fp32:
        x_mch = _to_low_precision(x_mch, use_bf16)
    
    # MBH: 递归合并
    result = _mbh_recursive_merge(x_mch, block_work, padded_size, use_bf16, use_fp32)
    return result[:n, :n]


# ============================================================================
# CPU 标杆 (Golden)
# ============================================================================
def _solve_tri_ref_bsnd(x: torch.Tensor, chunk_size: int, high_precision: bool) -> torch.Tensor:
    """BSND 格式的 MCH+MBH 标杆。
    
    chunk_size=64 时 NPU 走全 FP32 路径（无中间截断），其他 chunk_size 走低精度路径。
    """
    x_np = x.detach().cpu().float().numpy()
    B, T, H, _ = x_np.shape
    
    use_bf16 = (x.dtype == torch.bfloat16)
    # chunk_size=64 时 NPU 全程 fp32，CPU golden 也用 fp32
    # 其他 chunk_size 时 NPU 走低精度路径，CPU golden 模拟低精度
    use_fp32 = (chunk_size == 64) or high_precision
    
    result = np.zeros_like(x_np, dtype=np.float32)
    num_chunks = (T + chunk_size - 1) // chunk_size
    
    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual_size = e - s
                block = x_np[b, s:e, h, :actual_size]
                inv_block = _inverse_block_mch_mbh(block, chunk_size, use_bf16, use_fp32)
                result[b, s:e, h, :actual_size] = inv_block
    
    if high_precision:
        return torch.from_numpy(result).to(torch.float32)
    return torch.from_numpy(result).to(x.dtype)


def _solve_tri_ref_tnd(x: torch.Tensor, cu_seqlens: List[int], chunk_size: int, high_precision: bool) -> torch.Tensor:
    """TND 格式的 MCH+MBH 标杆。
    
    chunk_size=64 时 NPU 走全 FP32 路径（无中间截断），其他 chunk_size 走低精度路径。
    """
    x_np = x.detach().cpu().float().numpy()
    total_T, H, _ = x_np.shape
    
    use_bf16 = (x.dtype == torch.bfloat16)
    use_fp32 = (chunk_size == 64) or high_precision
    
    result = np.zeros_like(x_np, dtype=np.float32)
    num_seqs = len(cu_seqlens) - 1
    
    for seq_idx in range(num_seqs):
        bos = cu_seqlens[seq_idx]
        eos = cu_seqlens[seq_idx + 1]
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                actual_size = e - s
                block = x_np[s:e, h, :actual_size]
                inv_block = _inverse_block_mch_mbh(block, chunk_size, use_bf16, use_fp32)
                result[s:e, h, :actual_size] = inv_block
    
    if high_precision:
        return torch.from_numpy(result).to(torch.float32)
    return torch.from_numpy(result).to(x.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 标杆 (MCH+MBH 算法)。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    x = inputs["x"]
    layout = inputs["layout"]
    chunk_size = int(spec["chunk_size"])
    
    if layout == "tnd":
        return _solve_tri_ref_tnd(x, inputs["cu_seqlens"], chunk_size, high_precision)
    else:
        return _solve_tri_ref_bsnd(x, chunk_size, high_precision)


def _mask_output(out: torch.Tensor, inputs: dict, chunk_size: int) -> torch.Tensor:
    """对输出 tensor 的 padding 区域清零。
    
    NPU kernel 对尾块 (actual_size < chunk_size) 的 padding 列可能产生非零残余，
    需要清零以确保 ATK 全 tensor 比较时不受 padding 区域影响。
    """
    layout = inputs["layout"]
    cu_seqlens = inputs.get("cu_seqlens")
    
    if layout == "bsnd":
        B, T, H, _ = out.shape
        num_chunks = (T + chunk_size - 1) // chunk_size
        last_actual = T - (num_chunks - 1) * chunk_size
        if last_actual < chunk_size:
            s = (num_chunks - 1) * chunk_size
            out[:, s:, :, last_actual:] = 0
    elif layout == "tnd" and cu_seqlens is not None:
        num_seqs = len(cu_seqlens) - 1
        for seq_idx in range(num_seqs):
            bos = cu_seqlens[seq_idx]
            eos = cu_seqlens[seq_idx + 1]
            seq_len = eos - bos
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            last_chunk_start = bos + (num_chunks - 1) * chunk_size
            last_actual = eos - last_chunk_start
            if last_actual < chunk_size:
                out[last_chunk_start:eos, :, last_actual:] = 0
    return out


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    out = ascendc.solve_tri(
        inputs["x"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        layout=inputs["layout"]
    )
    # 对 padding 区域清零，与 golden 保持一致
    return _mask_output(out, inputs, int(spec["chunk_size"]))


@register("executor_solve_tri")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)
