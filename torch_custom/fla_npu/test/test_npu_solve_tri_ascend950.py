"""
test_npu_solve_tri_ascend950.py - Test SolveTri custom operator on ascend950 via torch.ops.npu

整算子测试（MCH + MBH）：
    solve_tri 接口为 (x, cu_seqlens, chunk_indices, layout)，chunk_size 由 x 末维隐式确定。
    算子对严格下三角 A 计算完整的 (I + A)^{-1}（与 triton 参考约定一致）。

    算法分两段：MCH 求 16×16 对角块逆（块对角逆），MBH 递归合并非对角块得到完整下三角逆。
"""
import torch
import torch_npu
import numpy as np
import fla_npu

torch.npu.utils.set_device(0)

FRAC = 16  # 叶子块大小（MCH 对角块）


def solve_tri_golden(block):
    """完整下三角逆 golden：(I + A)^{-1}，A 为 BT×BT 严格下三角 (float32 numpy)。"""
    bt = block.shape[0]
    eye = np.eye(bt, dtype=np.float32)
    return np.linalg.inv(eye + block)


def generate_lower_tri_block(bt, dtype=torch.float16, seed=42):
    """生成单个 BT×BT 的严格下三角矩阵。"""
    torch.manual_seed(seed)
    m = torch.randn(bt, bt, dtype=dtype) * 0.1
    for i in range(bt):
        for j in range(i, bt):
            m[i, j] = 0.0
    return m


def test_solve_tri(BT, layout="bhtd", dtype=torch.float16, seed=42):
    """整算子单 tile 测试：B=H=1, T=BT，单个 chunk。"""
    B, H, T = 1, 1, BT

    block = generate_lower_tri_block(BT, dtype, seed)
    block_np = block.float().numpy()
    golden_np = solve_tri_golden(block_np)

    if layout == "bhtd":
        A = block.reshape(B, H, T, BT)
    else:
        A = block.reshape(B, T, H, BT)

    A_npu = A.npu()

    out_npu = torch.ops.npu.npu_solve_tri(
        A_npu,
        cu_seqlens=None,
        chunk_indices=None,
        layout=layout,
    )
    out_cpu = out_npu.cpu()

    if layout == "bhtd":
        inv_block = out_cpu[0, 0].float().numpy()
    else:
        inv_block = out_cpu[0, :, 0].float().numpy()

    max_diff = np.abs(inv_block - golden_np).max()
    eye = np.eye(BT, dtype=np.float32)
    product = (eye + block_np) @ inv_block
    verify_err = np.abs(product - eye).max()

    passed = verify_err < 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] layout={layout} BT={BT}: "
          f"max_diff={max_diff:.6f}, verify_err={verify_err:.6f}")

    if (not passed) and BT <= 64:
        nf = BT // FRAC
        print(f"      --- block-wise diagnosis (BT={BT}, {nf}x{nf} blocks of 16) ---")
        for bi in range(nf):
            row_err = []
            for bj in range(nf):
                o = inv_block[bi*FRAC:(bi+1)*FRAC, bj*FRAC:(bj+1)*FRAC]
                g = golden_np[bi*FRAC:(bi+1)*FRAC, bj*FRAC:(bj+1)*FRAC]
                tag = "diag" if bi == bj else ("low " if bi > bj else "up  ")
                row_err.append(f"({bi},{bj}){tag} err={np.abs(o-g).max():.4f}")
            print("        " + " | ".join(row_err))
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("SolveTri NPU Test (ascend950) - whole operator (torch.ops.npu)")
    print("=" * 60)

    results = []

    print("\n--- BHTD layout [B=1, H=1, T=BT, BT] ---")
    for bt in (16, 32, 64, 128):
        results.append(test_solve_tri(bt, layout="bhtd"))

    print("\n--- BSND layout [B=1, T=BT, H=1, BT] ---")
    for bt in (16, 32, 64, 128):
        results.append(test_solve_tri(bt, layout="bsnd"))

    print("\n--- BT=128, multiple seeds ---")
    for sd in (1, 7, 123):
        results.append(test_solve_tri(128, layout="bhtd", seed=sd))

    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    if all(results):
        print("All tests PASSED!")
        exit(0)
    else:
        print("Some tests FAILED!")
        exit(1)
