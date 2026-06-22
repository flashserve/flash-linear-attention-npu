"""
test_npu_solve_tril.py - Test SolveTril custom operator via torch.ops.npu
Supports BHTD [B,H,T,BT], BSND [B,T,H,BT], and TND [total_T,H,BT] layouts.
"""
import torch
import torch_npu
import numpy as np
import fla_npu

torch.npu.utils.set_device(0)


def solve_tril_golden(A_tensor, chunk_size, layout="bhtd"):
    """CPU golden: compute (I + A)^{-1} for each chunk, support non-aligned T."""
    A = A_tensor.float().numpy()
    if layout == "bhtd":
        B, H, T, BT = A.shape
    else:
        B, T, H, BT = A.shape
    num_chunks = (T + chunk_size - 1) // chunk_size
    result = np.zeros_like(A)

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual_size = e - s
                if layout == "bhtd":
                    block = A[b, h, s:e, :actual_size]
                else:
                    block = A[b, s:e, h, :actual_size]
                eye = np.eye(actual_size, dtype=np.float32)
                M = eye + block
                M_inv = np.linalg.inv(M)
                if layout == "bhtd":
                    result[b, h, s:e, :actual_size] = M_inv
                else:
                    result[b, s:e, h, :actual_size] = M_inv
    return torch.from_numpy(result).half()


def generate_lower_tri_input(B, H, T, chunk_size, dtype=torch.float16, seed=42, layout="bhtd"):
    """Generate random strictly lower triangular input."""
    torch.manual_seed(seed)
    if layout == "bhtd":
        A = torch.zeros(B, H, T, chunk_size, dtype=dtype)
    else:
        A = torch.zeros(B, T, H, chunk_size, dtype=dtype)
    num_chunks = (T + chunk_size - 1) // chunk_size

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual_size = e - s
                one_chunk = torch.randn(actual_size, actual_size, dtype=dtype) * 0.1
                for i in range(actual_size):
                    for j in range(i, actual_size):
                        one_chunk[i, j] = 0.0
                if layout == "bhtd":
                    A[b, h, s:e, :actual_size] = one_chunk
                else:
                    A[b, s:e, h, :actual_size] = one_chunk
    return A


def test_solve_tril(B, H, T, chunk_size, layout="bhtd", dtype=torch.float16):
    """Run one test case."""
    torch.manual_seed(42)
    A = generate_lower_tri_input(B, H, T, chunk_size, dtype, layout=layout)
    golden = solve_tril_golden(A, chunk_size, layout=layout)

    A_npu = A.npu()
    out_npu = torch.ops.npu.npu_solve_tril(A_npu, chunk_size, layout=layout)
    out_cpu = out_npu.cpu()

    num_chunks = (T + chunk_size - 1) // chunk_size

    diff = (out_cpu.float() - golden.float()).abs()
    max_diff = diff.max().item()

    A_np = A.float().numpy()
    R_np = out_cpu.float().numpy()
    max_verify_err = 0.0
    print(f"\n--- Test (layout={layout}, B={B}, H={H}, T={T}, BT={chunk_size}, numChunks={num_chunks}) ---")
    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                actual_size = e - s
                if layout == "bhtd":
                    block = A_np[b, h, s:e, :actual_size]
                    inv_block = R_np[b, h, s:e, :actual_size]
                else:
                    block = A_np[b, s:e, h, :actual_size]
                    inv_block = R_np[b, s:e, h, :actual_size]
                eye = np.eye(actual_size, dtype=np.float32)
                product = (eye + block) @ inv_block
                err = np.abs(product - eye).max()
                max_verify_err = max(max_verify_err, err)

                golden_chunk = golden[b, h, s:e, :actual_size].float().numpy() if layout == "bhtd" else golden[b, s:e, h, :actual_size].float().numpy()
                chunk_diff = np.abs(inv_block - golden_chunk).max()
                is_partial = (c == num_chunks - 1 and actual_size < chunk_size)
                partial_tag = f" [partial {actual_size}x{actual_size}]" if is_partial else ""
                cst = "OK" if err < 1e-3 else "FAIL"
                print(f"  [{cst}] b={b} h={h} c={c}{partial_tag}: max_diff={chunk_diff:.6f}, verify_err={err:.6f}")

    passed = max_verify_err < 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] layout={layout} B={B} H={H} T={T} BT={chunk_size}: "
          f"max_diff={max_diff:.6f}, verify_err={max_verify_err:.6f}")
    return passed


def test_solve_tril_varlen(seq_lens, H, chunk_size, dtype=torch.float16):
    """Test THD varlen format [total_T, H, BT]."""
    total_T = sum(seq_lens)
    cu_seqlens = torch.tensor([0] + list(np.cumsum(seq_lens)), dtype=torch.int32)
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks_per_seq = (lens + chunk_size - 1) // chunk_size

    all_seq_ids = []
    all_chunk_ids = []
    for seq_idx, n_chunks in enumerate(num_chunks_per_seq):
        n = n_chunks.item()
        all_seq_ids.extend([seq_idx] * n)
        all_chunk_ids.extend(range(n))
    chunk_indices = torch.stack([
        torch.tensor(all_seq_ids, dtype=torch.int32),
        torch.tensor(all_chunk_ids, dtype=torch.int32)
    ], dim=1)

    torch.manual_seed(42)
    A = torch.zeros(total_T, H, chunk_size, dtype=dtype)
    num_seqs = len(seq_lens)
    for seq_idx in range(num_seqs):
        bos = cu_seqlens[seq_idx].item()
        eos = cu_seqlens[seq_idx + 1].item()
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                actual_size = e - s
                one_chunk = torch.randn(actual_size, actual_size, dtype=dtype) * 0.1
                for i in range(actual_size):
                    for j in range(i, actual_size):
                        one_chunk[i, j] = 0.0
                A[s:e, h, :actual_size] = one_chunk

    A_np = A.float().numpy()
    golden = np.zeros_like(A_np)
    for seq_idx in range(num_seqs):
        bos = cu_seqlens[seq_idx].item()
        eos = cu_seqlens[seq_idx + 1].item()
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                actual_size = e - s
                block = A_np[s:e, h, :actual_size]
                eye = np.eye(actual_size, dtype=np.float32)
                M = eye + block
                M_inv = np.linalg.inv(M)
                golden[s:e, h, :actual_size] = M_inv
    golden = torch.from_numpy(golden).half()

    A_npu = A.npu()
    cu_seqlens_list = cu_seqlens.tolist()
    chunk_indices_flat = chunk_indices.flatten().tolist()

    out_npu = torch.ops.npu.npu_solve_tril(A_npu, chunk_size,
                                           cu_seqlens=cu_seqlens_list,
                                           chunk_indices=chunk_indices_flat,
                                           layout="tnd")
    out_cpu = out_npu.cpu()

    max_verify_err = 0.0
    total_chunks = chunk_indices.shape[0]
    print(f"\n--- Varlen test (seqs={seq_lens}, H={H}, BT={chunk_size}, "
          f"total_T={total_T}, total_chunks={total_chunks}) ---")

    for seq_idx in range(num_seqs):
        bos = cu_seqlens[seq_idx].item()
        eos = cu_seqlens[seq_idx + 1].item()
        seq_len = eos - bos
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                actual_size = e - s
                block = A_np[s:e, h, :actual_size]
                inv_block = out_cpu[s:e, h, :actual_size].float().numpy()
                eye = np.eye(actual_size, dtype=np.float32)
                product = (eye + block) @ inv_block
                err = np.abs(product - eye).max()
                max_verify_err = max(max_verify_err, err)

                golden_chunk = golden[s:e, h, :actual_size].float().numpy()
                chunk_diff = np.abs(inv_block - golden_chunk).max()
                is_partial = (actual_size < chunk_size)
                partial_tag = f" [partial {actual_size}x{actual_size}]" if is_partial else ""
                cst = "OK" if err < 1e-3 else "FAIL"
                print(f"  [{cst}] seq={seq_idx} h={h} c={c}{partial_tag}: "
                      f"max_diff={chunk_diff:.6f}, verify_err={err:.6f}")

    passed = max_verify_err < 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] varlen seqs={seq_lens}, H={H}, BT={chunk_size}: "
          f"max_verify_err={max_verify_err:.6f}")
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("SolveTril NPU Test (torch.ops.npu)")
    print("=" * 60)

    results = []

    # BHTD layout tests
    print("\n--- BHTD layout [B, H, T, BT] ---")
    results.append(test_solve_tril(1, 1, 16, 16, layout="bhtd"))
    results.append(test_solve_tril(2, 2, 32, 32, layout="bhtd"))
    results.append(test_solve_tril(1, 1, 64, 32, layout="bhtd"))
    results.append(test_solve_tril(1, 1, 100, 32, layout="bhtd"))
    results.append(test_solve_tril(1, 1, 50, 32, layout="bhtd"))
    results.append(test_solve_tril(2, 2, 100, 32, layout="bhtd"))

    # BSND layout tests
    print("\n--- BSND layout [B, T, H, BT] ---")
    results.append(test_solve_tril(1, 2, 64, 32, layout="bsnd"))
    results.append(test_solve_tril(2, 2, 64, 32, layout="bsnd"))
    results.append(test_solve_tril(1, 1, 35, 32, layout="bsnd"))
    results.append(test_solve_tril(2, 2, 100, 32, layout="bsnd"))

    # Varlen TND tests
    print("\n--- TND varlen layout [total_T, H, BT] ---")
    results.append(test_solve_tril_varlen([64], 1, 32))
    results.append(test_solve_tril_varlen([32, 32, 32], 2, 32))
    results.append(test_solve_tril_varlen([100, 50, 35], 2, 32))
    results.append(test_solve_tril_varlen([35], 1, 32))
    results.append(test_solve_tril_varlen([3], 1, 32))
    results.append(test_solve_tril_varlen([18], 2, 32))

    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    if all(results):
        print("All tests PASSED!")
        exit(0)
    else:
        print("Some tests FAILED!")
        exit(1)
