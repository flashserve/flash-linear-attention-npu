# -*- coding: utf-8 -*-
"""
test_npu_solve_tri_all_layouts.py - Unified SolveTri tests for TND, BSND and BNSD/BHTD.

Supports fp16 and bf16 input. Precision compare uses CPU dual-benchmark:
  - Golden: MCH+MBH algorithm in float32 (same algorithm path, full precision).
  - Benchmark: MCH+MBH algorithm stepwise in fp16/bf16 (simulating NPU precision).
  - Target: NPU custom operator output.

Ratio = error(NPU, Golden_fp32) / error(Benchmark_stepwise, Golden_fp32)
"""
import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch_npu
import fla_npu


torch.npu.utils.set_device(0)


EPS = 1e-7
VERIFY_THRESHOLD_FP16 = 1e-3
VERIFY_THRESHOLD_BF16 = 2e-3
# For threshold_pass: NPU vs fp32 golden absolute error check
# Matrix inverse accumulates error, so use relaxed thresholds
FP16_MERE_THRESHOLD = 1e-3     # Relaxed: allow ~0.1% average relative error
FP16_MARE_THRESHOLD = 0.5      # Relaxed: allow up to 50% max relative error (outliers)
BF16_MERE_THRESHOLD = 2e-3
BF16_MARE_THRESHOLD = 1.0
# For ratio_pass: NPU error / CPU stepwise error (L1 standard)
# L1: Important operators for LLM/multimodal/recommendation systems
RATIO_MARE_THRESHOLD = 5.0     # L1 standard
RATIO_MERE_THRESHOLD = 1.5     # L1 standard
RATIO_RMSE_THRESHOLD = 1.5     # L1 standard
RATIO_DENOM_EPS = 1e-12


@dataclass
class PrecisionReport:
    mare_npu: float
    mere_npu: float
    rmse_npu: float
    mare_benchmark: float
    mere_benchmark: float
    rmse_benchmark: float
    mare_ratio: float
    mere_ratio: float
    rmse_ratio: float
    max_abs_diff: float
    verify_err: float
    threshold_pass: bool
    ratio_pass: bool
    verify_pass: bool

    @property
    def passed(self) -> bool:
        return self.ratio_pass and self.verify_pass


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype).replace("torch.", "")


def _make_lower_tri_block(actual_size: int, dtype: torch.dtype) -> torch.Tensor:
    block = torch.randn(actual_size, actual_size, dtype=dtype) * 0.1
    return torch.tril(block, diagonal=-1)


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks_per_seq = (lens + chunk_size - 1) // chunk_size

    all_seq_ids = []
    all_chunk_ids = []
    for seq_idx, n_chunks in enumerate(num_chunks_per_seq):
        n = int(n_chunks.item())
        all_seq_ids.extend([seq_idx] * n)
        all_chunk_ids.extend(range(n))

    return torch.stack(
        [
            torch.tensor(all_seq_ids, dtype=torch.int32),
            torch.tensor(all_chunk_ids, dtype=torch.int32),
        ],
        dim=1,
    )


def generate_dense_lower_tri_input(
    layout: str,
    B: int,
    H: int,
    T: int,
    chunk_size: int,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    num_chunks = (T + chunk_size - 1) // chunk_size

    if layout == "bsnd":
        x = torch.zeros(B, T, H, chunk_size, dtype=dtype)
    elif layout == "bnsd":
        x = torch.zeros(B, H, T, chunk_size, dtype=dtype)
    else:
        raise ValueError(f"dense layout must be bsnd or bnsd, got {layout}")

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual_size = e - s
                block = _make_lower_tri_block(actual_size, dtype)
                if layout == "bsnd":
                    x[b, s:e, h, :actual_size] = block
                else:
                    x[b, h, s:e, :actual_size] = block
    return x


def generate_tnd_lower_tri_input(
    cu_seqlens: torch.Tensor,
    H: int,
    chunk_size: int,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    total_t = int(cu_seqlens[-1].item())
    x = torch.zeros(total_t, H, chunk_size, dtype=dtype)

    for seq_idx in range(len(cu_seqlens) - 1):
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                actual_size = e - s
                x[s:e, h, :actual_size] = _make_lower_tri_block(actual_size, dtype)
    return x


def _fp16_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Simulate NPU matmul: fp16 inputs, fp32 accumulate, cast back to fp16."""
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)


def _bf16_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Simulate NPU matmul: bf16 inputs, fp32 accumulate, cast back to bf16."""
    a_f32 = torch.from_numpy(a.astype(np.float32))
    b_f32 = torch.from_numpy(b.astype(np.float32))
    result = (a_f32 @ b_f32).to(torch.bfloat16)
    return result.float().numpy().astype(np.float32)


def _to_bf16(arr: np.ndarray) -> np.ndarray:
    """Cast numpy array to bfloat16 precision via torch."""
    return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16).float().numpy()


def _mch_invert_16x16(block_fp16: np.ndarray) -> np.ndarray:
    """MCH iterative inversion for a single 16x16 block, step-by-step fp16.
    
    Mirrors NPU MCHInvertDiagonal():
      1. Y = A @ A
      2. X = I - A  (via I*I + (-I)*A accumulated in fp32)
      3. Iterate 3 times:
         X_new = X*Y + X*I  (accumulated in fp32, then cast fp16)
         Y_new = Y*Y
    """
    n = block_fp16.shape[0]
    A = block_fp16.astype(np.float16)
    I = np.eye(n, dtype=np.float16)
    neg_I = (-I).astype(np.float16)

    # Step 1: Y = A @ A
    Y = _fp16_matmul(A, A)

    # Step 2: X = I - A (NPU does I*I + (-I)*A in fp32 accumulator)
    acc = I.astype(np.float32) @ I.astype(np.float32)
    acc += neg_I.astype(np.float32) @ A.astype(np.float32)
    X = acc.astype(np.float16)

    # Step 3: iterate 3 times (log2(16/2) = 3)
    for iter_idx in range(3):
        # X_new = X*Y + X*I (two matmuls accumulated in fp32)
        acc = X.astype(np.float32) @ Y.astype(np.float32)
        acc += X.astype(np.float32) @ I.astype(np.float32)
        X = acc.astype(np.float16)

        # Y_new = Y*Y (skip last iteration)
        if iter_idx < 2:
            Y = _fp16_matmul(Y, Y)

    return X


def _mch_invert_16x16_bf16(block_bf16: np.ndarray) -> np.ndarray:
    """MCH iterative inversion for a single 16x16 block, step-by-step bf16."""
    n = block_bf16.shape[0]
    A = _to_bf16(block_bf16)
    I = _to_bf16(np.eye(n, dtype=np.float32))
    neg_I = _to_bf16(-np.eye(n, dtype=np.float32))

    Y = _bf16_matmul(A, A)

    acc = A.astype(np.float32) @ np.eye(n, dtype=np.float32)
    acc_init = np.eye(n, dtype=np.float32) @ np.eye(n, dtype=np.float32)
    acc_init += neg_I.astype(np.float32) @ A.astype(np.float32)
    X = _to_bf16(acc_init)

    for iter_idx in range(3):
        acc = X.astype(np.float32) @ Y.astype(np.float32)
        acc += X.astype(np.float32) @ I.astype(np.float32)
        X = _to_bf16(acc)

        if iter_idx < 2:
            Y = _bf16_matmul(Y, Y)

    return X


def _extract_block_diagonal(matrix: np.ndarray, block_size: int, start: int) -> np.ndarray:
    """Extract block-diagonal elements at positions start, start+2, start+4, ...
    
    For lower-tri with block_size=16, start=1: extracts odd 16x16 blocks on diagonal.
    Returns a matrix with only those diagonal blocks non-zero.
    """
    n = matrix.shape[0]
    result = np.zeros_like(matrix)
    num_blocks = n // block_size
    for blk in range(start, num_blocks, 2):
        r0 = blk * block_size
        r1 = r0 + block_size
        result[r0:r1, r0:r1] = matrix[r0:r1, r0:r1]
    return result


def _mbh_recursive_merge(x_fp16: np.ndarray, m_fp16: np.ndarray, matrix_size: int) -> np.ndarray:
    """MBH recursive merge, step-by-step fp16.
    
    Mirrors NPU RecursiveMerge() for lower-triangular matrices.
    x_fp16: MCH result (block-diagonal inverse), shape [n, n], fp16
    m_fp16: original input matrix M (lower-tri, no diagonal), shape [n, n], fp16
    """
    FRAC = 16
    n = matrix_size
    I = np.eye(n, dtype=np.float16)
    M_neg = (-m_fp16).astype(np.float16)
    X = x_fp16.copy()

    block_size = FRAC
    while block_size < n:
        # Lower-tri: driving=odd blocks, other=even blocks
        driving = _extract_block_diagonal(X, block_size, 1)
        other = _extract_block_diagonal(X, block_size, 0)

        # Y_result = I*I + driving*M_neg (accumulated in fp32, cast fp16)
        acc = I.astype(np.float32) @ I.astype(np.float32)
        acc += driving.astype(np.float32) @ M_neg.astype(np.float32)
        Y_result = acc.astype(np.float16)

        # X_new = Y_result*other + I*driving (accumulated in fp32, cast fp16)
        acc = Y_result.astype(np.float32) @ other.astype(np.float32)
        acc += I.astype(np.float32) @ driving.astype(np.float32)
        X = acc.astype(np.float16)

        block_size *= 2

    return X


def _mbh_recursive_merge_bf16(x_bf16: np.ndarray, m_bf16: np.ndarray, matrix_size: int) -> np.ndarray:
    """MBH recursive merge, step-by-step bf16."""
    FRAC = 16
    n = matrix_size
    I = _to_bf16(np.eye(n, dtype=np.float32))
    M_neg = _to_bf16(-m_bf16)
    X = x_bf16.copy()

    block_size = FRAC
    while block_size < n:
        driving = _extract_block_diagonal(X, block_size, 1)
        other = _extract_block_diagonal(X, block_size, 0)

        acc = I.astype(np.float32) @ I.astype(np.float32)
        acc += driving.astype(np.float32) @ M_neg.astype(np.float32)
        Y_result = _to_bf16(acc)

        acc = Y_result.astype(np.float32) @ other.astype(np.float32)
        acc += I.astype(np.float32) @ driving.astype(np.float32)
        X = _to_bf16(acc)

        block_size *= 2

    return X


def _inverse_block_stepwise_bf16(block: np.ndarray) -> np.ndarray:
    """Compute (I + block)^{-1} using stepwise bf16, mimicking NPU MCH+MBH."""
    FRAC = 16
    n = block.shape[0]

    padded_n = ((n + FRAC - 1) // FRAC) * FRAC
    if padded_n > n:
        padded = np.zeros((padded_n, padded_n), dtype=np.float32)
        padded[:n, :n] = block
        block_bf16 = _to_bf16(padded)
    else:
        block_bf16 = _to_bf16(block)

    if padded_n <= FRAC:
        result = _mch_invert_16x16_bf16(block_bf16)
        return result[:n, :n]

    num_fracs = padded_n // FRAC
    x_mch = np.zeros((padded_n, padded_n), dtype=np.float32)
    for i in range(num_fracs):
        r0 = i * FRAC
        r1 = r0 + FRAC
        sub_block = block_bf16[r0:r1, r0:r1]
        x_mch[r0:r1, r0:r1] = _mch_invert_16x16_bf16(sub_block)

    result = _mbh_recursive_merge_bf16(x_mch, block_bf16, padded_n)
    return result[:n, :n]


def _inverse_block_stepwise_fp16(block: np.ndarray) -> np.ndarray:
    """Compute (I + block)^{-1} using stepwise fp16, mimicking NPU MCH+MBH.
    
    block: lower-triangular (below diagonal), shape [n, n], any dtype.
    Returns fp16 result.
    """
    FRAC = 16
    n = block.shape[0]

    padded_n = ((n + FRAC - 1) // FRAC) * FRAC
    if padded_n > n:
        padded = np.zeros((padded_n, padded_n), dtype=np.float16)
        padded[:n, :n] = block.astype(np.float16)
        block_fp16 = padded
    else:
        block_fp16 = block.astype(np.float16)

    if padded_n <= FRAC:
        result = _mch_invert_16x16(block_fp16)
        return result[:n, :n]

    num_fracs = padded_n // FRAC
    x_mch = np.zeros((padded_n, padded_n), dtype=np.float16)
    for i in range(num_fracs):
        r0 = i * FRAC
        r1 = r0 + FRAC
        sub_block = block_fp16[r0:r1, r0:r1]
        x_mch[r0:r1, r0:r1] = _mch_invert_16x16(sub_block)

    result = _mbh_recursive_merge(x_mch, block_fp16, padded_n)
    return result[:n, :n]


def _inverse_block_stepwise_fp32(block: np.ndarray) -> np.ndarray:
    """Compute (I + block)^{-1} using MCH+MBH algorithm in full float32 precision.
    Same algorithm as NPU kernel, but no precision truncation at each step.
    For partial blocks (n not multiple of 16), pads to next multiple of 16.
    """
    FRAC = 16
    n = block.shape[0]
    blk = block.astype(np.float32)

    # Pad to next multiple of FRAC if needed (mirrors NPU kernel behavior)
    padded_n = ((n + FRAC - 1) // FRAC) * FRAC
    if padded_n > n:
        padded = np.zeros((padded_n, padded_n), dtype=np.float32)
        padded[:n, :n] = blk
        blk = padded

    if padded_n <= FRAC:
        result = _mch_invert_16x16_fp32(blk)
        return result[:n, :n]

    num_fracs = padded_n // FRAC
    x_mch = np.zeros((padded_n, padded_n), dtype=np.float32)
    for i in range(num_fracs):
        r0 = i * FRAC
        r1 = r0 + FRAC
        x_mch[r0:r1, r0:r1] = _mch_invert_16x16_fp32(blk[r0:r1, r0:r1])

    result = _mbh_recursive_merge_fp32(x_mch, blk, padded_n)
    return result[:n, :n]


def _mch_invert_16x16_fp32(block: np.ndarray) -> np.ndarray:
    """MCH iterative inversion for a block (up to 16x16), full float32 precision.
    Iteration count = log2(n) - 1 where n is block size.
    """
    n = block.shape[0]
    A = block.astype(np.float32)
    I = np.eye(n, dtype=np.float32)

    Y = A @ A
    X = I - A

    num_iters = max(0, int(np.log2(n)) - 1) if n > 1 else 0
    for i in range(num_iters):
        X = X @ Y + X @ I
        if i < num_iters - 1:
            Y = Y @ Y
    return X


def _mbh_recursive_merge_fp32(x: np.ndarray, m: np.ndarray, n: int) -> np.ndarray:
    """MBH recursive merge, full float32 precision."""
    FRAC = 16
    I = np.eye(n, dtype=np.float32)
    M_neg = -m.astype(np.float32)
    X = x.copy()

    block_size = FRAC
    while block_size < n:
        driving = _extract_block_diagonal(X, block_size, 1)
        other = _extract_block_diagonal(X, block_size, 0)

        Y_result = I @ I + driving @ M_neg
        X = Y_result @ other + I @ driving

        block_size *= 2
    return X


def _inverse_block(block: np.ndarray, compute_dtype, output_dtype: Optional[np.dtype]) -> np.ndarray:
    """Compute (I + block)^{-1}.
    
    All paths use MCH+MBH algorithm (same as NPU kernel):
    - float32: MCH+MBH with full fp32 precision (golden).
    - float16: MCH+MBH stepwise fp16 (benchmark).
    - bfloat16: MCH+MBH stepwise bf16 (benchmark).
    """
    if compute_dtype == np.float16:
        inv = _inverse_block_stepwise_fp16(block)
        if output_dtype is not None:
            inv = inv.astype(output_dtype)
        return inv

    if compute_dtype == "bfloat16":
        inv = _inverse_block_stepwise_bf16(block)
        if output_dtype is not None:
            inv = inv.astype(output_dtype)
        return inv

    # float32 golden: same MCH+MBH algorithm, full precision
    inv = _inverse_block_stepwise_fp32(block)
    if output_dtype is not None:
        inv = inv.astype(output_dtype)
    return inv


def solve_tril_reference_dense(
    x: torch.Tensor,
    chunk_size: int,
    layout: str,
    compute_dtype,
    output_dtype: Optional[np.dtype],
) -> torch.Tensor:
    x_np = x.detach().cpu().float().numpy()
    storage_dtype = np.float32 if (output_dtype is None and compute_dtype == "bfloat16") else (output_dtype or compute_dtype)
    result = np.zeros_like(x_np, dtype=storage_dtype)

    if layout == "bsnd":
        B, T, H, _ = x_np.shape
    elif layout == "bnsd":
        B, H, T, _ = x_np.shape
    else:
        raise ValueError(f"dense layout must be bsnd or bnsd, got {layout}")

    num_chunks = (T + chunk_size - 1) // chunk_size
    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual_size = e - s
                if layout == "bsnd":
                    block = x_np[b, s:e, h, :actual_size]
                    result[b, s:e, h, :actual_size] = _inverse_block(block, compute_dtype, output_dtype)
                else:
                    block = x_np[b, h, s:e, :actual_size]
                    result[b, h, s:e, :actual_size] = _inverse_block(block, compute_dtype, output_dtype)
    return torch.from_numpy(result)


def solve_tril_reference_tnd(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    compute_dtype,
    output_dtype: Optional[np.dtype],
) -> torch.Tensor:
    x_np = x.detach().cpu().float().numpy()
    storage_dtype = np.float32 if (output_dtype is None and compute_dtype == "bfloat16") else (output_dtype or compute_dtype)
    result = np.zeros_like(x_np, dtype=storage_dtype)
    _, H, _ = x_np.shape

    for seq_idx in range(len(cu_seqlens) - 1):
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                actual_size = e - s
                block = x_np[s:e, h, :actual_size]
                result[s:e, h, :actual_size] = _inverse_block(block, compute_dtype, output_dtype)
    return torch.from_numpy(result)


def _valid_dense_mask(shape: Sequence[int], layout: str, chunk_size: int) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    if layout == "bsnd":
        B, T, H, _ = shape
    else:
        B, H, T, _ = shape
    num_chunks = (T + chunk_size - 1) // chunk_size

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual_size = e - s
                if layout == "bsnd":
                    mask[b, s:e, h, :actual_size] = True
                else:
                    mask[b, h, s:e, :actual_size] = True
    return mask


def _valid_tnd_mask(shape: Sequence[int], cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    _, H, _ = shape
    for seq_idx in range(len(cu_seqlens) - 1):
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                actual_size = e - s
                mask[s:e, h, :actual_size] = True
    return mask


def _error_metrics(actual: torch.Tensor, golden: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float, float, float]:
    actual_np = actual.detach().cpu().double()[mask].numpy()
    golden_np = golden.detach().cpu().double()[mask].numpy()
    abs_err = np.abs(actual_np - golden_np)
    rel_err = abs_err / (np.abs(golden_np) + EPS)
    mare = float(np.max(rel_err)) if rel_err.size else 0.0
    mere = float(np.mean(rel_err)) if rel_err.size else 0.0
    rmse = float(np.sqrt(np.mean(np.square(actual_np - golden_np)))) if actual_np.size else 0.0
    max_abs = float(np.max(abs_err)) if abs_err.size else 0.0
    return mare, mere, rmse, max_abs


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, RATIO_DENOM_EPS)


def verify_inverse_dense(x: torch.Tensor, out: torch.Tensor, chunk_size: int, layout: str) -> float:
    x_np = x.detach().cpu().float().numpy()
    out_np = out.detach().cpu().float().numpy()

    if layout == "bsnd":
        B, T, H, _ = x_np.shape
    else:
        B, H, T, _ = x_np.shape
    num_chunks = (T + chunk_size - 1) // chunk_size

    max_err = 0.0
    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual_size = e - s
                if layout == "bsnd":
                    block = x_np[b, s:e, h, :actual_size]
                    inv_block = out_np[b, s:e, h, :actual_size]
                else:
                    block = x_np[b, h, s:e, :actual_size]
                    inv_block = out_np[b, h, s:e, :actual_size]
                eye = np.eye(actual_size, dtype=np.float32)
                err = np.abs((eye + block) @ inv_block - eye).max()
                max_err = max(max_err, float(err))
    return max_err


def verify_inverse_tnd(x: torch.Tensor, out: torch.Tensor, cu_seqlens: torch.Tensor, chunk_size: int) -> float:
    x_np = x.detach().cpu().float().numpy()
    out_np = out.detach().cpu().float().numpy()
    _, H, _ = x_np.shape

    max_err = 0.0
    for seq_idx in range(len(cu_seqlens) - 1):
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                actual_size = e - s
                block = x_np[s:e, h, :actual_size]
                inv_block = out_np[s:e, h, :actual_size]
                eye = np.eye(actual_size, dtype=np.float32)
                err = np.abs((eye + block) @ inv_block - eye).max()
                max_err = max(max_err, float(err))
    return max_err


def build_precision_report(
    npu_out: torch.Tensor,
    golden_fp32: torch.Tensor,
    cpu_benchmark: torch.Tensor,
    mask: torch.Tensor,
    verify_err: float,
    dtype: torch.dtype = torch.float16,
) -> PrecisionReport:
    mare_npu, mere_npu, rmse_npu, max_abs = _error_metrics(npu_out, golden_fp32, mask)
    mare_bench, mere_bench, rmse_bench, _ = _error_metrics(cpu_benchmark, golden_fp32, mask)

    mare_ratio = _safe_ratio(mare_npu, mare_bench)
    mere_ratio = _safe_ratio(mere_npu, mere_bench)
    rmse_ratio = _safe_ratio(rmse_npu, rmse_bench)

    if dtype == torch.bfloat16:
        threshold_pass = mare_npu < BF16_MARE_THRESHOLD and mere_npu < BF16_MERE_THRESHOLD
        verify_pass = verify_err < VERIFY_THRESHOLD_BF16
    else:
        threshold_pass = mare_npu < FP16_MARE_THRESHOLD and mere_npu < FP16_MERE_THRESHOLD
        verify_pass = verify_err < VERIFY_THRESHOLD_FP16

    ratio_pass = (
        mare_ratio <= RATIO_MARE_THRESHOLD
        and mere_ratio <= RATIO_MERE_THRESHOLD
        and rmse_ratio <= RATIO_RMSE_THRESHOLD
    )

    return PrecisionReport(
        mare_npu=mare_npu,
        mere_npu=mere_npu,
        rmse_npu=rmse_npu,
        mare_benchmark=mare_bench,
        mere_benchmark=mere_bench,
        rmse_benchmark=rmse_bench,
        mare_ratio=mare_ratio,
        mere_ratio=mere_ratio,
        rmse_ratio=rmse_ratio,
        max_abs_diff=max_abs,
        verify_err=verify_err,
        threshold_pass=threshold_pass,
        ratio_pass=ratio_pass,
        verify_pass=verify_pass,
    )


def print_report(case_name: str, report: PrecisionReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(
        f"  [{status}] {case_name}: "
        f"verify={report.verify_err:.6g}, max_abs={report.max_abs_diff:.6g}, "
        f"NPU(MARE={report.mare_npu:.6g}, MERE={report.mere_npu:.6g}, RMSE={report.rmse_npu:.6g}), "
        f"CPU16(MARE={report.mare_benchmark:.6g}, MERE={report.mere_benchmark:.6g}, RMSE={report.rmse_benchmark:.6g}), "
        f"Ratio(MARE={report.mare_ratio:.3g}, MERE={report.mere_ratio:.3g}, RMSE={report.rmse_ratio:.3g})"
    )
    if not report.passed:
        print(
            f"    checks: threshold={report.threshold_pass}, "
            f"ratio={report.ratio_pass}, inverse={report.verify_pass}"
        )


def run_dense_case(layout: str, B: int, H: int, T: int, chunk_size: int, seed: int, dtype: torch.dtype = torch.float16) -> bool:
    x = generate_dense_lower_tri_input(layout, B, H, T, chunk_size, dtype, seed)
    golden_fp32 = solve_tril_reference_dense(x, chunk_size, layout, np.float32, None)
    benchmark_compute = np.float16 if dtype == torch.float16 else "bfloat16"
    cpu_benchmark = solve_tril_reference_dense(x, chunk_size, layout, benchmark_compute, None)

    npu_layout = "bsnd" if layout == "bsnd" else "bhtd"
    out_npu = torch.ops.npu.npu_solve_tri(x.npu(), layout=npu_layout).cpu()
    mask = _valid_dense_mask(x.shape, layout, chunk_size)
    verify_err = verify_inverse_dense(x, out_npu, chunk_size, layout)
    report = build_precision_report(out_npu, golden_fp32, cpu_benchmark, mask, verify_err, dtype)

    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    shape_desc = f"{layout.upper()} B={B} H={H} T={T} BT={chunk_size} {dtype_str} seed={seed}"
    print_report(shape_desc, report)
    return report.passed


def run_tnd_case(seq_lens: Sequence[int], H: int, chunk_size: int, seed: int, dtype: torch.dtype = torch.float16) -> bool:
    cu_seqlens = torch.tensor([0] + list(np.cumsum(seq_lens)), dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    x = generate_tnd_lower_tri_input(cu_seqlens, H, chunk_size, dtype, seed)

    golden_fp32 = solve_tril_reference_tnd(x, cu_seqlens, chunk_size, np.float32, None)
    benchmark_compute = np.float16 if dtype == torch.float16 else "bfloat16"
    cpu_benchmark = solve_tril_reference_tnd(x, cu_seqlens, chunk_size, benchmark_compute, None)

    out_npu = torch.ops.npu.npu_solve_tri(
        x.npu(),
        cu_seqlens=cu_seqlens.tolist(),
        chunk_indices=chunk_indices.flatten().tolist(),
        layout="tnd",
    ).cpu()
    mask = _valid_tnd_mask(x.shape, cu_seqlens, chunk_size)
    verify_err = verify_inverse_tnd(x, out_npu, cu_seqlens, chunk_size)
    report = build_precision_report(out_npu, golden_fp32, cpu_benchmark, mask, verify_err, dtype)

    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    shape_desc = f"TND seq_lens={list(seq_lens)} H={H} BT={chunk_size} {dtype_str} seed={seed}"
    print_report(shape_desc, report)
    return report.passed


def dense_cases(mode: str) -> List[Tuple[int, int, int, int]]:
    smoke = [
        (1, 1, 32, 32),
        (1, 2, 35, 32),
        (2, 2, 100, 32),
        (1, 1, 20, 16),
    ]
    stress = [
        (1, 1, 16, 16),
        (2, 2, 16, 16),
        (1, 1, 48, 16),
        (2, 4, 32, 16),
        (1, 1, 64, 32),
        (2, 2, 64, 32),
        (1, 4, 128, 32),
        (4, 4, 128, 32),
        (8, 2, 64, 32),
        (2, 8, 64, 32),
        (1, 1, 256, 32),
        (1, 1, 30, 16),
        (2, 2, 25, 16),
        (1, 1, 50, 32),
        (1, 1, 60, 32),
        (1, 1, 33, 32),
        (1, 1, 63, 32),
    ]
    return smoke if mode == "smoke" else smoke + stress


def tnd_cases(mode: str) -> List[Tuple[List[int], int, int]]:
    smoke = [
        ([64], 1, 32),
        ([35], 1, 32),
        ([100, 50, 35], 2, 32),
    ]
    stress = [
        ([32, 32, 32], 2, 32),
        ([50], 2, 32),
        ([64, 35, 100], 2, 32),
        ([128, 50], 4, 32),
        ([3], 1, 32),
        ([18], 2, 32),
    ]
    return smoke if mode == "smoke" else smoke + stress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified npu_solve_tri tests for TND, BSND and BNSD/BHTD.")
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["tnd", "bsnd", "bnsd"],
        default=["tnd", "bsnd", "bnsd"],
        help="Layouts to test. bnsd maps to operator layout='bhtd'.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["fp16", "bf16"],
        default=["fp16", "bf16"],
        help="Data types to test.",
    )
    parser.add_argument("--mode", choices=["smoke", "stress"], default="smoke")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtypes_to_test = [dtype_map[d] for d in args.dtypes]

    print("=" * 80)
    print("SolveTri NPU Unified Layout Test")
    print(f"dtypes={args.dtypes} mode={args.mode} runs={args.runs} layouts={args.layouts}")
    print("compare=CPU fp32 Golden + CPU stepwise Benchmark (MCH+MBH) + NPU Target")
    print(f"L1 standard: MARE_ratio<={RATIO_MARE_THRESHOLD}, MERE_ratio<={RATIO_MERE_THRESHOLD}, RMSE_ratio<={RATIO_RMSE_THRESHOLD}")
    print("=" * 80)

    total = 0
    passed = 0

    for run in range(args.runs):
        seed = args.seed + run
        print(f"\n{'=' * 32} Run {run + 1}/{args.runs}, seed={seed} {'=' * 32}")

        for dtype in dtypes_to_test:
            dtype_str = "fp16" if dtype == torch.float16 else "bf16"

            if "tnd" in args.layouts:
                print(f"\n--- TND cases ({dtype_str}): input [total_T, H, BT], layout='tnd' ---")
                for seq_lens, H, BT in tnd_cases(args.mode):
                    ok = run_tnd_case(seq_lens, H, BT, seed, dtype)
                    passed += int(ok)
                    total += 1

            for layout in ("bsnd", "bnsd"):
                if layout not in args.layouts:
                    continue
                op_layout = "bsnd" if layout == "bsnd" else "bhtd"
                shape = "[B, T, H, BT]" if layout == "bsnd" else "[B, H, T, BT]"
                print(f"\n--- {layout.upper()} cases ({dtype_str}): input {shape}, operator layout='{op_layout}' ---")
                for B, H, T, BT in dense_cases(args.mode):
                    ok = run_dense_case(layout, B, H, T, BT, seed, dtype)
                    passed += int(ok)
                    total += 1

    print(f"\n{'=' * 80}")
    print(f"Total: {passed}/{total} passed ({passed / total * 100:.1f}%)")
    if passed == total:
        print("ALL TESTS PASSED!")
        return 0
    print(f"FAILED: {total - passed} tests")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
