"""
test.py - Test SolveTri operator on NPU.

Computes (I + A)^{-1} where A is strictly lower triangular.
Compares NPU output against CPU golden (numpy inverse).

支持四种数据布局：
  - bsnd: [B, T, H, BT]  (单 chunk 内数据不连续)
  - bnsd: [B, H, T, BT]  (单 chunk 内数据连续, BSND 的转置)
  - tnd:  [total_T, H, BT]  (变长序列, 单 chunk 内数据不连续)
  - ntd:  [H, total_T, BT]  (变长序列, 单 chunk 内数据连续, TND 的转置)
"""
import torch
import torch_npu
import numpy as np
from fla_npu.ops import ascendc as ascendc_ops

torch.npu.utils.set_device(0)


# ============================================================================
# Golden / verify helpers
# ============================================================================
def _inverse_block(block_np):
    """CPU golden: compute (I + A)^{-1} for a single chunk."""
    n = block_np.shape[0]
    eye = np.eye(n, dtype=np.float32)
    M = eye + block_np
    return np.linalg.inv(M)


def _make_lower_tri_block(actual_size, dtype):
    """Generate random strictly lower triangular block."""
    block = torch.randn(actual_size, actual_size, dtype=dtype) * 0.1
    return torch.tril(block, diagonal=-1)


def verify_inverse_dense(A, result, layout, chunk_size, atol=1e-3):
    """Verify (I+A) * result ≈ I for dense layouts (bsnd/bnsd)."""
    A_f = A.float().numpy()
    R_f = result.float().numpy()
    if layout == "bsnd":
        B, T, H, BT = A_f.shape
    else:  # bnsd
        B, H, T, BT = A_f.shape
    num_chunks = (T + chunk_size - 1) // chunk_size
    max_err = 0.0

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual = e - s
                if layout == "bsnd":
                    block = A_f[b, s:e, h, :actual]
                    inv_block = R_f[b, s:e, h, :actual]
                else:
                    block = A_f[b, h, s:e, :actual]
                    inv_block = R_f[b, h, s:e, :actual]
                eye = np.eye(actual, dtype=np.float32)
                err = np.abs((eye + block) @ inv_block - eye).max()
                max_err = max(max_err, err)
    return max_err < atol, max_err


def verify_inverse_varlen(A, result, layout, chunk_size, cu_seqlens, chunk_indices, atol=1e-3):
    """Verify (I+A) * result ≈ I for varlen layouts (tnd/ntd)."""
    A_f = A.float().numpy()
    R_f = result.float().numpy()
    if layout == "tnd":
        total_T, H, BT = A_f.shape
    else:  # ntd
        H, total_T, BT = A_f.shape
    num_seqs = len(cu_seqlens) - 1
    total_chunks = len(chunk_indices) // 2
    max_err = 0.0

    for chunk_idx in range(total_chunks):
        seq_idx = chunk_indices[chunk_idx * 2]
        chunk_in_seq = chunk_indices[chunk_idx * 2 + 1]
        bos = cu_seqlens[seq_idx]
        eos = cu_seqlens[seq_idx + 1]
        seq_len = eos - bos
        s = bos + chunk_in_seq * chunk_size
        e = min(s + chunk_size, eos)
        actual = e - s
        for h in range(H):
            if layout == "tnd":
                block = A_f[s:e, h, :actual]
                inv_block = R_f[s:e, h, :actual]
            else:  # ntd
                block = A_f[h, s:e, :actual]
                inv_block = R_f[h, s:e, :actual]
            eye = np.eye(actual, dtype=np.float32)
            err = np.abs((eye + block) @ inv_block - eye).max()
            max_err = max(max_err, err)
    return max_err < atol, max_err


# ============================================================================
# Dense layout (bsnd/bnsd)
# ============================================================================
def generate_dense_input(B, H, T, BT, layout, dtype=torch.float16, seed=42):
    """Generate random strictly lower triangular input for bsnd/bnsd."""
    torch.manual_seed(seed)
    num_chunks = (T + BT - 1) // BT
    if layout == "bsnd":
        A = torch.zeros(B, T, H, BT, dtype=dtype)
    else:  # bnsd
        A = torch.zeros(B, H, T, BT, dtype=dtype)

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * BT
                e = min(s + BT, T)
                actual = e - s
                block = _make_lower_tri_block(actual, dtype)
                if layout == "bsnd":
                    A[b, s:e, h, :actual] = block
                else:
                    A[b, h, s:e, :actual] = block
    return A


def solve_tril_golden_dense(A, layout, BT):
    """CPU golden for dense layouts."""
    A_np = A.float().numpy()
    if layout == "bsnd":
        B, T, H, _ = A_np.shape
    else:
        B, H, T, _ = A_np.shape
    num_chunks = (T + BT - 1) // BT
    result = np.zeros_like(A_np)

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * BT
                e = min(s + BT, T)
                actual = e - s
                if layout == "bsnd":
                    block = A_np[b, s:e, h, :actual]
                else:
                    block = A_np[b, h, s:e, :actual]
                inv = _inverse_block(block)
                if layout == "bsnd":
                    result[b, s:e, h, :actual] = inv
                else:
                    result[b, h, s:e, :actual] = inv
    return torch.from_numpy(result).to(A.dtype)


def test_dense(B, H, T, BT, layout, dtype=torch.float16):
    """Run one dense test case (bsnd or bnsd)."""
    print(f"  Test: layout={layout}, B={B}, H={H}, T={T}, BT={BT}, dtype={dtype}")
    A = generate_dense_input(B, H, T, BT, layout, dtype)
    golden = solve_tril_golden_dense(A, layout, BT)

    A_npu = A.npu()
    out_npu = ascendc_ops.npu_solve_tri(A_npu, layout=layout)
    out_cpu = out_npu.cpu()

    diff = (out_cpu.float() - golden.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    passed, verify_err = verify_inverse_dense(A, out_cpu, layout, BT)
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}, verify_err={verify_err:.6f}")
    return passed


# ============================================================================
# Varlen layout (tnd/ntd)
# ============================================================================
def prepare_chunk_indices(seq_lens, chunk_size):
    """Build chunk_indices flat list from seq_lens."""
    indices = []
    for seq_idx, seq_len in enumerate(seq_lens):
        n = (seq_len + chunk_size - 1) // chunk_size
        for c in range(n):
            indices.extend([seq_idx, c])
    return indices


def generate_varlen_input(seq_lens, H, BT, layout, dtype=torch.float16, seed=42):
    """Generate random strictly lower triangular input for tnd/ntd."""
    torch.manual_seed(seed)
    total_T = sum(seq_lens)
    if layout == "tnd":
        A = torch.zeros(total_T, H, BT, dtype=dtype)
    else:  # ntd
        A = torch.zeros(H, total_T, BT, dtype=dtype)

    cu_seqlens = [0]
    for s in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + s)

    num_seqs = len(seq_lens)
    for seq_idx in range(num_seqs):
        bos = cu_seqlens[seq_idx]
        eos = cu_seqlens[seq_idx + 1]
        seq_len = eos - bos
        num_chunks = (seq_len + BT - 1) // BT
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * BT
                e = min(s + BT, eos)
                actual = e - s
                block = _make_lower_tri_block(actual, dtype)
                if layout == "tnd":
                    A[s:e, h, :actual] = block
                else:
                    A[h, s:e, :actual] = block
    return A, cu_seqlens


def solve_tril_golden_varlen(A, layout, BT, cu_seqlens, chunk_indices):
    """CPU golden for varlen layouts."""
    A_np = A.float().numpy()
    if layout == "tnd":
        total_T, H, _ = A_np.shape
    else:
        H, total_T, _ = A_np.shape
    result = np.zeros_like(A_np)
    total_chunks = len(chunk_indices) // 2

    for chunk_idx in range(total_chunks):
        seq_idx = chunk_indices[chunk_idx * 2]
        chunk_in_seq = chunk_indices[chunk_idx * 2 + 1]
        bos = cu_seqlens[seq_idx]
        eos = cu_seqlens[seq_idx + 1]
        s = bos + chunk_in_seq * BT
        e = min(s + BT, eos)
        actual = e - s
        for h in range(H):
            if layout == "tnd":
                block = A_np[s:e, h, :actual]
            else:
                block = A_np[h, s:e, :actual]
            inv = _inverse_block(block)
            if layout == "tnd":
                result[s:e, h, :actual] = inv
            else:
                result[h, s:e, :actual] = inv
    return torch.from_numpy(result).to(A.dtype)


def test_varlen(seq_lens, H, BT, layout, dtype=torch.float16):
    """Run one varlen test case (tnd or ntd)."""
    print(f"  Test: layout={layout}, seq_lens={seq_lens}, H={H}, BT={BT}, dtype={dtype}")
    A, cu_seqlens = generate_varlen_input(seq_lens, H, BT, layout, dtype)
    chunk_indices = prepare_chunk_indices(seq_lens, BT)
    golden = solve_tril_golden_varlen(A, layout, BT, cu_seqlens, chunk_indices)

    A_npu = A.npu()
    out_npu = ascendc_ops.npu_solve_tri(
        A_npu,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout=layout,
    )
    out_cpu = out_npu.cpu()

    diff = (out_cpu.float() - golden.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    passed, verify_err = verify_inverse_varlen(
        A, out_cpu, layout, BT, cu_seqlens, chunk_indices
    )
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}, verify_err={verify_err:.6f}")
    return passed


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("SolveTri NPU Test (4 layouts: bsnd/bnsd/tnd/ntd)")
    print("=" * 60)

    results = []

    # ---- Dense: BSND ----
    print("\n--- Dense layout: BSND [B, T, H, BT] ---")
    for B, H, T, BT in [
        (1, 1, 16, 16),
        (2, 4, 32, 16),
        (1, 2, 32, 32),
        (2, 4, 64, 32),
        (1, 2, 64, 64),
        (2, 4, 128, 64),
        (1, 2, 128, 128),
    ]:
        results.append(test_dense(B, H, T, BT, "bsnd"))

    # ---- Dense: BNSD (BSND 的转置, 单 chunk 内数据连续) ----
    print("\n--- Dense layout: BNSD [B, H, T, BT] (contiguous) ---")
    for B, H, T, BT in [
        (1, 1, 16, 16),
        (2, 4, 32, 16),
        (1, 2, 32, 32),
        (2, 4, 64, 32),
        (1, 2, 64, 64),
        (2, 4, 128, 64),
        (1, 2, 128, 128),
    ]:
        results.append(test_dense(B, H, T, BT, "bnsd"))

    # ---- Varlen: TND ----
    print("\n--- Varlen layout: TND [total_T, H, BT] ---")
    for seq_lens, H, BT in [
        ([64], 1, 32),
        ([32, 32, 32], 2, 32),
        ([45], 1, 32),
        ([100, 50, 35], 2, 32),
        ([64], 1, 64),
        ([100, 80], 2, 64),
        ([128], 1, 128),
        ([100, 80, 60], 2, 128),
    ]:
        results.append(test_varlen(seq_lens, H, BT, "tnd"))

    # ---- Varlen: NTD (TND 的转置, 单 chunk 内数据连续) ----
    print("\n--- Varlen layout: NTD [H, total_T, BT] (contiguous) ---")
    for seq_lens, H, BT in [
        ([64], 1, 32),
        ([32, 32, 32], 2, 32),
        ([45], 1, 32),
        ([100, 50, 35], 2, 32),
        ([64], 1, 64),
        ([100, 80], 2, 64),
        ([128], 1, 128),
        ([100, 80, 60], 2, 128),
    ]:
        results.append(test_varlen(seq_lens, H, BT, "ntd"))

    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(results)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
