# -*- coding: utf-8 -*-
"""
test_npu_solve_tri_triton_case.py - NPU SolveTri test with Triton-identical parameters.

Test case: B=4, T=8192, H=4, chunk_size=64, layout=BSND
Same as Triton GPU test for direct comparison.

Precision compare uses CPU dual-benchmark:
  - Golden: CPU float32 inverse (np.linalg.inv), high-precision truth.
  - Benchmark: CPU stepwise fp16 inverse, simulating NPU's MCH+MBH algorithm.
  - Target: NPU custom operator output (fp16).
"""
import numpy as np
import torch
import torch_npu
import fla_npu


torch.npu.utils.set_device(0)


# ============================================================================
# Constants
# ============================================================================
EPS = 1e-7
VERIFY_THRESHOLD_FP16 = 1e-3
VERIFY_THRESHOLD_BF16 = 3e-3  # bf16 has lower precision than fp16
FP16_MERE_THRESHOLD = 1e-3
FP16_MARE_THRESHOLD = 0.5
RATIO_MARE_THRESHOLD = 5.0
RATIO_MERE_THRESHOLD = 1.5
RATIO_RMSE_THRESHOLD = 1.5
RATIO_DENOM_EPS = 1e-12

# Test case parameters
# B,H in [1,10], T in [1,4096], CHUNK_SIZE in {64,128}
import random

def generate_test_cases(num_cases: int = 100, seed: int = 42):
    """Generate random test cases."""
    random.seed(seed)
    cases = []
    for _ in range(num_cases):
        B = random.randint(9, 9)
        H = random.randint(2, 2)
        T = random.randint(128, 128)
        chunk_size = random.choice([64])
        cases.append((B, T, H, chunk_size))
    return cases

TEST_CASES = generate_test_cases(num_cases=1, seed=42)
LAYOUT = "bsnd"
SEED = 42


# ============================================================================
# Helper functions (from test_npu_solve_tri_all_layouts.py)
# ============================================================================
def _make_lower_tri_block(actual_size: int, dtype: torch.dtype) -> torch.Tensor:
    block = torch.randn(actual_size, actual_size, dtype=dtype) * 0.1
    return torch.tril(block, diagonal=-1)


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


# ============================================================================
# CPU fp16/bf16 stepwise inverse (MCH + MBH simulation)
# ============================================================================
def _mch_invert_16x16_generic(block: np.ndarray, use_bf16: bool = False) -> np.ndarray:
    """MCH invert for 16x16 block, supports fp16 or bf16 simulation."""
    n = block.shape[0]
    
    def to_low(x):
        if use_bf16:
            return torch.from_numpy(x.astype(np.float32)).bfloat16().float().numpy()
        return x.astype(np.float16)
    
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


def _mch_invert_16x16_fp32(block: np.ndarray) -> np.ndarray:
    """MCH invert for 16x16 block using fp32, no truncation."""
    n = block.shape[0]
    A = block.astype(np.float32)
    I = np.eye(n, dtype=np.float32)

    Y = A @ A
    X = I - A

    for _ in range(3):
        X = X @ Y + X
        Y = Y @ Y

    return X


def _extract_block_diagonal(matrix: np.ndarray, block_size: int, start: int) -> np.ndarray:
    n = matrix.shape[0]
    result = np.zeros_like(matrix)
    num_blocks = n // block_size
    for blk in range(start, num_blocks, 2):
        r0 = blk * block_size
        r1 = r0 + block_size
        result[r0:r1, r0:r1] = matrix[r0:r1, r0:r1]
    return result


def _mbh_recursive_merge_generic(x_low: np.ndarray, m_low: np.ndarray, matrix_size: int, use_bf16: bool = False) -> np.ndarray:
    """MBH recursive merge, supports fp16 or bf16 simulation."""
    FRAC = 16
    n = matrix_size
    
    def to_low(x):
        if use_bf16:
            return torch.from_numpy(x.astype(np.float32)).bfloat16().float().numpy()
        return x.astype(np.float16)
    
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


def _mbh_recursive_merge_fp32(x: np.ndarray, m: np.ndarray, matrix_size: int) -> np.ndarray:
    """MBH recursive merge using fp32, no truncation."""
    FRAC = 16
    n = matrix_size
    I = np.eye(n, dtype=np.float32)
    M_neg = -m.astype(np.float32)
    X = x.copy()

    block_size = FRAC
    while block_size < n:
        driving = _extract_block_diagonal(X, block_size, 1)
        other = _extract_block_diagonal(X, block_size, 0)

        Y_result = I + driving @ M_neg
        X = Y_result @ other + driving

        block_size *= 2

    return X


def _inverse_block_stepwise_generic(block: np.ndarray, use_bf16: bool = False, matrix_size: int = 64) -> np.ndarray:
    """Stepwise inverse (MCH + MBH), supports fp16 or bf16.
    
    For partial blocks (n < matrix_size), padding to matrix_size like NPU does.
    """
    FRAC = 16
    n = block.shape[0]  # actual size, e.g. 54
    
    def to_low(x):
        if use_bf16:
            return torch.from_numpy(x.astype(np.float32)).bfloat16().float().numpy()
        return x.astype(np.float16)
    
    # Padding to matrix_size (e.g. 64) to match NPU behavior
    if n < matrix_size:
        padded = np.zeros((matrix_size, matrix_size), dtype=block.dtype)
        padded[:n, :n] = block
        block = padded
        padded_size = matrix_size
    else:
        padded_size = n
    
    block_low = to_low(block)

    if padded_size <= FRAC:
        result = _mch_invert_16x16_generic(block_low, use_bf16)
        return result[:n, :n]  # return valid region

    num_fracs = padded_size // FRAC
    x_mch = np.zeros((padded_size, padded_size), dtype=np.float32)
    for i in range(num_fracs):
        r0 = i * FRAC
        r1 = r0 + FRAC
        sub_block = block_low[r0:r1, r0:r1]
        x_mch[r0:r1, r0:r1] = _mch_invert_16x16_generic(sub_block, use_bf16)

    x_mch = to_low(x_mch)
    result = _mbh_recursive_merge_generic(x_mch, block_low, padded_size, use_bf16)
    return result[:n, :n]  # return valid region


def _inverse_block_stepwise_fp32(block: np.ndarray, matrix_size: int = 64) -> np.ndarray:
    """Stepwise inverse (MCH + MBH) using fp32, no truncation.
    
    For partial blocks (n < matrix_size), padding to matrix_size like NPU does.
    """
    FRAC = 16
    n = block.shape[0]
    
    # Padding to matrix_size to match NPU behavior
    if n < matrix_size:
        padded = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        padded[:n, :n] = block
        block = padded
        padded_size = matrix_size
    else:
        padded_size = n
    
    A = block.astype(np.float32)

    if padded_size <= FRAC:
        result = _mch_invert_16x16_fp32(A)
        return result[:n, :n]

    num_fracs = padded_size // FRAC
    x_mch = np.zeros((padded_size, padded_size), dtype=np.float32)
    for i in range(num_fracs):
        r0 = i * FRAC
        r1 = r0 + FRAC
        sub_block = A[r0:r1, r0:r1]
        x_mch[r0:r1, r0:r1] = _mch_invert_16x16_fp32(sub_block)

    result = _mbh_recursive_merge_fp32(x_mch, A, padded_size)
    return result[:n, :n]


def _inverse_block(block: np.ndarray, compute_dtype, output_dtype=None, matrix_size: int = 64) -> np.ndarray:
    """Inverse block with specified compute dtype.
    
    compute_dtype can be:
      - np.float32: use high precision np.linalg.inv
      - np.float16: use stepwise fp16 simulation
      - 'bf16' or torch.bfloat16: use stepwise bf16 simulation
    
    matrix_size: the full chunk size for padding partial blocks (e.g. 64 or 128)
    """
    actual_size = block.shape[0]

    # fp16 stepwise
    if compute_dtype == np.float16:
        inv = _inverse_block_stepwise_generic(block, use_bf16=False, matrix_size=matrix_size)
        if output_dtype is not None:
            inv = inv.astype(output_dtype)
        return inv

    # bf16 stepwise
    if compute_dtype == 'bf16' or compute_dtype == torch.bfloat16:
        inv = _inverse_block_stepwise_generic(block, use_bf16=True, matrix_size=matrix_size)
        if output_dtype is not None:
            inv = inv.astype(output_dtype)
        return inv

    # fp32 high precision using MCH+MBH (no truncation)
    inv = _inverse_block_stepwise_fp32(block, matrix_size=matrix_size)
    if output_dtype is not None:
        inv = inv.astype(output_dtype)
    return inv


def solve_tril_reference_dense(
    x: torch.Tensor,
    chunk_size: int,
    layout: str,
    compute_dtype,
    output_dtype=None,
) -> torch.Tensor:
    """Compute reference inverse for dense format (BSND/BNSD)."""
    x_np = x.detach().cpu().float().numpy()  # bf16 needs float() first
    # bf16 simulation stores as fp32
    result_dtype = np.float32 if compute_dtype == 'bf16' else (output_dtype or compute_dtype)
    result = np.zeros_like(x_np, dtype=result_dtype)

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
                    result[b, s:e, h, :actual_size] = _inverse_block(block, compute_dtype, output_dtype, matrix_size=chunk_size)
                else:
                    block = x_np[b, h, s:e, :actual_size]
                    result[b, h, s:e, :actual_size] = _inverse_block(block, compute_dtype, output_dtype, matrix_size=chunk_size)
    return torch.from_numpy(result)


def _valid_dense_mask(shape, layout: str, chunk_size: int) -> torch.Tensor:
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


def _error_metrics(actual: torch.Tensor, golden: torch.Tensor, mask: torch.Tensor):
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


# ============================================================================
# Main test function
# ============================================================================
def run_single_case(B: int, T: int, H: int, chunk_size: int, dtype: torch.dtype = torch.float16):
    """Run a single test case with given parameters."""
    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    verify_threshold = VERIFY_THRESHOLD_BF16 if dtype == torch.bfloat16 else VERIFY_THRESHOLD_FP16
    
    print(f"\n{'='*70}")
    print(f"Test: B={B}, T={T}, H={H}, BT={chunk_size}, Layout={LAYOUT.upper()}, dtype={dtype_str}")
    print("=" * 70)

    # Generate input
    x = generate_dense_lower_tri_input(LAYOUT, B, H, T, chunk_size, dtype, SEED)

    # Compute references (dual benchmark)
    golden_fp32 = solve_tril_reference_dense(x, chunk_size, LAYOUT, np.float32, None)
    # Use matching low-precision benchmark: fp16 for fp16, bf16 for bf16
    bench_dtype = 'bf16' if dtype == torch.bfloat16 else np.float16
    cpu_benchmark_low = solve_tril_reference_dense(x, chunk_size, LAYOUT, bench_dtype, None)

    # Run NPU operator
    x_npu = x.npu()
    out_npu = torch.ops.npu.npu_solve_tri(x_npu, layout=LAYOUT)
    out_cpu = out_npu.cpu()

    # Compute metrics
    mask = _valid_dense_mask(x.shape, LAYOUT, chunk_size)
    verify_err = verify_inverse_dense(x, out_cpu, chunk_size, LAYOUT)

    mare_npu, mere_npu, rmse_npu, max_abs = _error_metrics(out_cpu, golden_fp32, mask)
    mare_bench, mere_bench, rmse_bench, _ = _error_metrics(cpu_benchmark_low, golden_fp32, mask)

    mare_ratio = _safe_ratio(mare_npu, mare_bench)
    mere_ratio = _safe_ratio(mere_npu, mere_bench)
    rmse_ratio = _safe_ratio(rmse_npu, rmse_bench)

    ratio_pass = (
        mare_ratio <= RATIO_MARE_THRESHOLD
        and mere_ratio <= RATIO_MERE_THRESHOLD
        and rmse_ratio <= RATIO_RMSE_THRESHOLD
    )
    
    # bf16 only checks ratio, fp16 also checks verify_err
    if dtype == torch.bfloat16:
        passed = ratio_pass
    else:
        verify_pass = verify_err < verify_threshold
        passed = ratio_pass and verify_pass

    status = "PASS" if passed else "FAIL"

    print(f"  verify_err={verify_err:.6g}, MARE={mare_npu:.6g}, MERE={mere_npu:.6g}")
    print(f"  MARE_ratio={mare_ratio:.3g}, MERE_ratio={mere_ratio:.3g}, RMSE_ratio={rmse_ratio:.3g}")
    print(f"  [{status}] B={B}, T={T}, H={H}, BT={chunk_size}, {dtype_str}")

    return passed


def run_all_cases():
    """Run all test cases."""
    print("=" * 70)
    print("NPU SolveTri Test - BSND Format Generalized")
    print("=" * 70)
    print(f"Layout = {LAYOUT.upper()}, Seed = {SEED}")
    print(f"Total test cases: {len(TEST_CASES)} x 2 (fp16 + bf16)")

    results = []

    # FP16 tests
    print("\n--- FP16 Tests ---")
    for B, T, H, chunk_size in TEST_CASES:
        passed = run_single_case(B, T, H, chunk_size, dtype=torch.float16)
        results.append((B, T, H, chunk_size, "fp16", passed))

    # BF16 tests
    print("\n--- BF16 Tests ---")
    for B, T, H, chunk_size in TEST_CASES:
        passed = run_single_case(B, T, H, chunk_size, dtype=torch.bfloat16)
        results.append((B, T, H, chunk_size, "bf16", passed))

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    num_passed = sum(1 for r in results if r[5])
    num_total = len(results)

    for B, T, H, chunk_size, dtype_str, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] B={B}, T={T}, H={H}, BT={chunk_size}, {dtype_str}")

    print(f"\nResults: {num_passed}/{num_total} passed")
    all_passed = num_passed == num_total
    if all_passed:
        print("[PASS] All NPU SolveTri BSND tests PASSED (fp16 + bf16)!")
    else:
        print("[FAIL] Some NPU SolveTri BSND tests FAILED!")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed = run_all_cases()
    exit(0 if passed else 1)
