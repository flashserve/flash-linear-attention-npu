"""
test_npu_solve_tri_ascend950.py - Test SolveTri custom operator on ascend950 via torch.ops.npu

由仓1 test_npu_solve_tril.py 适配而来。仓2 算子接口：
    npu_solve_tri(x, *, cu_seqlens=None, chunk_indices=None, layout="bsnd")
  —— 无 chunk_size 入参，chunk_size 由 x 末维 BT 隐式确定；每个 chunk 块为 BT×BT。
支持 BSND [B,T,H,BT] 与 TND [total_T,H,BT]（变长）两种布局。

用例覆盖：
  - 仓1 原有全部用例（BSND + TND varlen，BT=32）。
  - chunk_size ∈ {16,32,64,128} × 定长(BSND)/变长(TND) × 有尾块/无尾块。
"""
import torch
import torch_npu
import numpy as np
import fla_npu

torch.npu.utils.set_device(0)


def solve_tril_golden(A_tensor, chunk_size, layout="bsnd"):
    """CPU golden: 逐 chunk 计算 (I + A)^{-1}，支持非对齐 T（尾块）。"""
    A = A_tensor.float().numpy()
    if layout == "tnd":
        T, H, BT = A.shape
        B = 1
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
                if layout == "tnd":
                    block = A[s:e, h, :actual_size]
                else:
                    block = A[b, s:e, h, :actual_size]
                eye = np.eye(actual_size, dtype=np.float32)
                M = eye + block
                M_inv = np.linalg.inv(M)
                if layout == "tnd":
                    result[s:e, h, :actual_size] = M_inv
                else:
                    result[b, s:e, h, :actual_size] = M_inv
    return torch.from_numpy(result).half()


def generate_lower_tri_input(B, H, T, chunk_size, dtype=torch.float16, seed=42, layout="bsnd"):
    """生成随机严格下三角输入（每 chunk 块严格下三角，尾块只填 actual_size×actual_size）。"""
    torch.manual_seed(seed)
    if layout == "tnd":
        A = torch.zeros(T, H, chunk_size, dtype=dtype)
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
                if layout == "tnd":
                    A[s:e, h, :actual_size] = one_chunk
                else:
                    A[b, s:e, h, :actual_size] = one_chunk
    return A


def test_solve_tri(B, H, T, chunk_size, layout="bsnd", dtype=torch.float16):
    """定长用例（BSND / BHTD）。chunk_size = 输入末维 BT。"""
    torch.manual_seed(42)
    A = generate_lower_tri_input(B, H, T, chunk_size, dtype, layout=layout)
    golden = solve_tril_golden(A, chunk_size, layout=layout)

    A_npu = A.npu()
    # 仓2 接口：chunk_size 由 x 末维隐式确定，定长无需 cu_seqlens / chunk_indices
    out_npu = torch.ops.npu.npu_solve_tri(A_npu, cu_seqlens=None, chunk_indices=None, layout=layout)
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
                if layout == "tnd":
                    block = A_np[s:e, h, :actual_size]
                    inv_block = R_np[s:e, h, :actual_size]
                    golden_chunk = golden[s:e, h, :actual_size].float().numpy()
                else:
                    block = A_np[b, s:e, h, :actual_size]
                    inv_block = R_np[b, s:e, h, :actual_size]
                    golden_chunk = golden[b, s:e, h, :actual_size].float().numpy()
                eye = np.eye(actual_size, dtype=np.float32)
                product = (eye + block) @ inv_block
                err = np.abs(product - eye).max()
                max_verify_err = max(max_verify_err, err)

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


def test_solve_tri_varlen(seq_lens, H, chunk_size, dtype=torch.float16):
    """变长 TND 用例 [total_T, H, BT]。chunk_size = 输入末维 BT。"""
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

    out_npu = torch.ops.npu.npu_solve_tri(A_npu,
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
    print("SolveTri NPU Test (ascend950, torch.ops.npu)")
    print("=" * 60)

    results = []

    # ================================================================
    # Group 1: 仓1 原有全部用例（BT=32）
    # ================================================================
    print("\n########## [Group 1] repo1 original cases (BT=32) ##########")
    print("\n--- BSND layout [B, T, H, BT] ---")
    results.append(test_solve_tri(1, 2, 40, 32, layout="bsnd"))
    results.append(test_solve_tri(2, 2, 64, 32, layout="bsnd"))
    results.append(test_solve_tri(1, 1, 35, 32, layout="bsnd"))
    results.append(test_solve_tri(2, 2, 100, 32, layout="bsnd"))
    print("\n--- Network Cases layout [B, T, H, BT] ---")
    results.append(test_solve_tri(1, 4, 32768, 64, layout="bsnd"))
    results.append(test_solve_tri(1, 4, 32768, 128, layout="bsnd"))
    results.append(test_solve_tri(8, 8, 4096, 64, layout="bsnd"))
    results.append(test_solve_tri(8, 8, 4096, 128, layout="bsnd"))

    print("\n--- TND varlen layout [total_T, H, BT] ---")
    results.append(test_solve_tri_varlen([64], 1, 32))
    results.append(test_solve_tri_varlen([32, 32, 32], 2, 32))
    results.append(test_solve_tri_varlen([45], 1, 32))
    results.append(test_solve_tri_varlen([100, 50, 35], 2, 32))
    results.append(test_solve_tri_varlen([4, 18], 1, 32))
    results.append(test_solve_tri_varlen([35], 1, 32))
    results.append(test_solve_tri_varlen([3], 1, 32))
    results.append(test_solve_tri_varlen([18], 2, 32))

    # ================================================================
    # Group 2: chunk_size ∈ {16,32,64,128} × 定长/变长 × 有尾块/无尾块
    #   - tb = 大尾块（cs>16 时 ChunkAlign→cs，触发 MBH 尾块路径），恒 < cs
    #   - ts = 小尾块（ChunkAlign→16，触发 MCH-only 尾块 + cur<cs 写回路径）
    # ================================================================
    print("\n########## [Group 2] chunk_size sweep x fixed/varlen x tail/no-tail ##########")
    for cs in (16, 32, 64, 128):
        tb = (cs // 2 + 8) if cs > 16 else 8   # 大尾块（< cs）
        ts = 8                                  # 小尾块（→ cur=16）

        print(f"\n=== chunk_size (BT) = {cs} ===")

        # ---- 定长 BSND ----
        print(f"[BSND fixed, no-tail]  T={2 * cs}")
        results.append(test_solve_tri(2, 2, 2 * cs, cs, layout="bsnd"))
        print(f"[BSND fixed, with-tail] T={2 * cs + tb} (tail={tb})")
        results.append(test_solve_tri(2, 2, 2 * cs + tb, cs, layout="bsnd"))

        # ---- 变长 TND ----
        print(f"[TND varlen, no-tail]  seqs=[{2 * cs},{cs}]")
        results.append(test_solve_tri_varlen([2 * cs, cs], 2, cs))
        print(f"[TND varlen, with-tail] seqs=[{2 * cs + tb},{cs + ts}] (tails={tb},{ts})")
        results.append(test_solve_tri_varlen([2 * cs + tb, cs + ts], 2, cs))

    print(f"\n{'=' * 60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    if all(results):
        print("All tests PASSED!")
        exit(0)
    else:
        print("Some tests FAILED!")
        exit(1)
