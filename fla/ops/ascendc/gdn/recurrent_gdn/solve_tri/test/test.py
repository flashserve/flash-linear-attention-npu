"""
test.py - Test SolveTri operator on NPU.

Computes (I + A)^{-1} where A is strictly lower triangular.
Compares NPU output against CPU golden (numpy inverse).
"""
import torch
import torch_npu
import numpy as np
import fla_npu
from fla_npu.ops.ascendc import npu_solve_tri

torch.npu.utils.set_device(0)


def solve_tril_golden(A_tensor):
    """CPU golden: compute (I + A)^{-1} for each chunk."""
    A = A_tensor.float().numpy()
    B, H, T, BT = A.shape
    num_chunks = T // BT
    result = np.zeros_like(A)

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                row_start = c * BT
                row_end = row_start + BT
                block = A[b, h, row_start:row_end, :BT]
                eye = np.eye(BT, dtype=np.float32)
                M = eye + block
                M_inv = np.linalg.inv(M)
                result[b, h, row_start:row_end, :BT] = M_inv

    return torch.from_numpy(result).to(A_tensor.dtype)


def generate_lower_tri_input(B, H, T, BT, dtype=torch.float16):
    """Generate random strictly lower triangular input."""
    A = torch.randn(B, H, T, BT, dtype=dtype) * 0.1
    num_chunks = T // BT
    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                row_start = c * BT
                for i in range(BT):
                    for j in range(i, BT):
                        A[b, h, row_start + i, j] = 0.0
    return A


def verify_inverse(A, result, atol=1e-3):
    """Verify (I+A) * result ≈ I."""
    A_f = A.float().numpy()
    R_f = result.float().numpy()
    B, H, T, BT = A_f.shape
    num_chunks = T // BT
    max_err = 0.0

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * BT
                block = A_f[b, h, s:s+BT, :BT]
                inv_block = R_f[b, h, s:s+BT, :BT]
                eye = np.eye(BT, dtype=np.float32)
                product = (eye + block) @ inv_block
                err = np.abs(product - eye).max()
                max_err = max(max_err, err)

    return max_err < atol, max_err


def test_solve_tri(B, H, T, BT, dtype=torch.float16):
    """Run one test case."""
    print(f"  Test: B={B}, H={H}, T={T}, BT={BT}, dtype={dtype}")

    A = generate_lower_tri_input(B, H, T, BT, dtype)
    golden = solve_tril_golden(A)

    # Call NPU operator
    A_npu = A.npu()
    out_npu = npu_solve_tri(A_npu, layout="bhtd")
    out_cpu = out_npu.cpu()

    # Compare with golden
    diff = (out_cpu.float() - golden.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Verify inverse property
    passed, verify_err = verify_inverse(A, out_cpu)
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}, verify_err={verify_err:.6f}")
    return passed


def test_mch_fp32_intermediates(dtype):
    """Cover a large finite inverse while MCH intermediates stay in FP32."""
    B, H, T, BT = 1, 1, 64, 64
    A = torch.zeros(B, H, T, BT, dtype=dtype)
    block = torch.tril(torch.full((16, 16), 1.45, dtype=dtype), diagonal=-1)
    A[0, 0, :16, :16] = block

    golden = solve_tril_golden(A)
    out = npu_solve_tri(A.npu(), layout="bhtd").cpu()
    finite = torch.isfinite(out).all().item()
    max_diff = (out.float() - golden.float()).abs().max().item()
    passed = finite and max_diff < 2e-2
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] {dtype} MCH FP32 intermediates: "
        f"finite={finite}, max_diff={max_diff:.6f}"
    )
    return passed


def test_fp16_partial_chunk():
    """Cover the unaligned DataCopyPad path for a short final chunk."""
    A = torch.zeros(1, 70, 1, 64, dtype=torch.float16)
    generator = torch.Generator().manual_seed(20260726)
    tail = torch.tril(
        torch.randn(6, 6, generator=generator, dtype=torch.float32) * 0.1,
        diagonal=-1,
    ).half()
    A[0, 64:70, 0, :6] = tail

    out_full = npu_solve_tri(A.npu(), layout="bsnd").cpu()
    out = out_full[0, 64:70, 0, :6].float()
    golden = torch.linalg.inv(torch.eye(6, dtype=torch.float64) + tail.double()).float()
    finite = torch.isfinite(out).all().item()
    max_diff = (out - golden).abs().max().item()
    row = torch.arange(70).remainder(64)
    col = torch.arange(64)
    valid = col.unsqueeze(0) <= row.unsqueeze(1)
    invalid_zero = bool((out_full.masked_select(~valid.view(1, 70, 1, 64)) == 0).all())
    passed = finite and invalid_zero and max_diff < 2e-2
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] FP16 partial chunk: finite={finite}, "
        f"invalid_zero={invalid_zero}, max_diff={max_diff:.6f}"
    )
    return passed


def test_tnd_invalid_region_zero_fill():
    """The stable adapter must zero upper triangles and short-tail padding."""
    cu_seqlens = [0, 70, 135]
    chunk_size = 64
    heads = 2
    x = torch.zeros(135, heads, chunk_size, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(20260727)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        for chunk_start in range(bos, eos, chunk_size):
            valid = min(chunk_size, eos - chunk_start)
            block = torch.tril(
                torch.randn(
                    valid,
                    valid,
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.05,
                diagonal=-1,
            ).bfloat16()
            x[chunk_start : chunk_start + valid, :, :valid] = block.unsqueeze(1)

    chunk_indices = []
    for sequence_index, (bos, eos) in enumerate(
        zip(cu_seqlens[:-1], cu_seqlens[1:])
    ):
        for chunk_index in range((eos - bos + chunk_size - 1) // chunk_size):
            chunk_indices.extend([sequence_index, chunk_index])

    out = npu_solve_tri(
        x.npu(),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout="tnd",
    ).cpu()
    local_rows = []
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        local_rows.extend(index % chunk_size for index in range(eos - bos))
    row = torch.tensor(local_rows)
    col = torch.arange(chunk_size)
    valid = col.unsqueeze(0) <= row.unsqueeze(1)
    invalid = out.masked_select(~valid.unsqueeze(1))
    invalid_zero = bool((invalid == 0).all())
    valid_finite = bool(torch.isfinite(out.masked_select(valid.unsqueeze(1))).all())
    passed = invalid_zero and valid_finite
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] BF16 TND invalid-region zero fill: "
        f"invalid_zero={invalid_zero}, valid_finite={valid_finite}"
    )
    return passed


def generate_tnd_input(seq_lens, heads, chunk_size, dtype):
    """Generate packed TND input and a float32 inverse reference."""
    total_tokens = sum(seq_lens)
    x = torch.zeros(total_tokens, heads, chunk_size, dtype=dtype)
    golden = torch.zeros(total_tokens, heads, chunk_size, dtype=torch.float32)
    cu_seqlens = [0]
    chunk_indices = []

    bos = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        for chunk_idx, chunk_start in enumerate(range(0, seq_len, chunk_size)):
            valid_size = min(chunk_size, seq_len - chunk_start)
            chunk_indices.extend([seq_idx, chunk_idx])
            for head_idx in range(heads):
                block = torch.tril(
                    torch.randn(valid_size, valid_size, dtype=torch.float32) * 0.01,
                    diagonal=-1,
                )
                row_start = bos + chunk_start
                x[row_start:row_start + valid_size, head_idx, :valid_size] = block.to(dtype)
                golden[row_start:row_start + valid_size, head_idx, :valid_size] = torch.linalg.inv(
                    torch.eye(valid_size, dtype=torch.float32) + block
                )
        bos += seq_len
        cu_seqlens.append(bos)

    return x, golden, cu_seqlens, chunk_indices


def test_solve_tri_tnd_tail(chunk_size, tail_size, dtype):
    """Verify a non-16-aligned final TND chunk without reading past the input."""
    seq_lens = [chunk_size, tail_size]
    heads = 2
    x, golden, cu_seqlens, chunk_indices = generate_tnd_input(
        seq_lens,
        heads,
        chunk_size,
        dtype,
    )

    # Keep NaNs immediately after the logical input. A tail loader that rounds the
    # final diagonal block up to 16x16 will read these guard rows and poison output.
    backing = torch.full(
        (x.shape[0] + 16, heads, chunk_size),
        float("nan"),
        dtype=dtype,
        device="npu",
    )
    x_npu = backing[:x.shape[0]]
    x_npu.copy_(x.npu())
    assert x_npu.is_contiguous()

    out = npu_solve_tri(
        x_npu,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout="tnd",
    ).float().cpu()

    max_diff = 0.0
    all_finite = True
    bos = 0
    for seq_len in seq_lens:
        for chunk_start in range(0, seq_len, chunk_size):
            valid_size = min(chunk_size, seq_len - chunk_start)
            row_start = bos + chunk_start
            actual_block = out[row_start:row_start + valid_size, :, :valid_size]
            golden_block = golden[row_start:row_start + valid_size, :, :valid_size]
            all_finite = all_finite and bool(torch.isfinite(actual_block).all().item())
            max_diff = max(max_diff, float((actual_block - golden_block).abs().max().item()))
        bos += seq_len

    tolerance = 5e-3 if dtype == torch.float16 else 2e-2
    passed = all_finite and max_diff <= tolerance
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] TND chunk_size={chunk_size}, tail_size={tail_size}, dtype={dtype}, "
        f"finite={all_finite}, max_diff={max_diff:.6f}, tolerance={tolerance:.6f}"
    )
    return passed


def main():
    print("=" * 60)
    print("SolveTri NPU Test")
    print("=" * 60)

    test_cases = [
        # B, H, T, BT
        (1, 1, 16, 16),
        (2, 4, 32, 16),
        (1, 2, 32, 32),
        (2, 4, 64, 32),
        (1, 2, 64, 64),
        (2, 4, 128, 64),
        (1, 2, 128, 128),
    ]

    results = []
    for B, H, T, BT in test_cases:
        passed = test_solve_tri(B, H, T, BT)
        results.append(passed)

    results.append(test_mch_fp32_intermediates(torch.float16))
    results.append(test_mch_fp32_intermediates(torch.bfloat16))
    results.append(test_fp16_partial_chunk())
    results.append(test_tnd_invalid_region_zero_fill())

    tnd_tail_cases = [
        (64, 39, torch.bfloat16),
        (64, 63, torch.float16),
        (128, 95, torch.bfloat16),
        (128, 127, torch.float16),
    ]
    for chunk_size, tail_size, dtype in tnd_tail_cases:
        passed = test_solve_tri_tnd_tail(chunk_size, tail_size, dtype)
        results.append(passed)

    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(results)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All tests PASSED!")
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
