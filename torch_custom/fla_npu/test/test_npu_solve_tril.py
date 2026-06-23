"""
test_npu_solve_tril.py - Test SolveTril custom operator via torch.ops.npu

MBH 调试模式（SOLVE_TRIL_MBH_DEBUG_ONLY）：
    当前算子已屏蔽 MCH 实现，MCH 的输出作为接口入参传入，用于单独调试 MBH 模块。
    新增三个可选接口入参（均为 BT×BT 的 float16 ND 矩阵）：
      - mch_output      : MCH 模块输出（块对角逆矩阵，含 BT/16 个 16×16 对角逆块，
                          非对角块为 0）。等价于原 MCHInvertDiagonal() 的输出。
      - zero_matrix     : 全 0 矩阵
      - identity_matrix : 单位矩阵 I
    算子内部：X <- mch_output，-A 由原始输入 x 与 -I 计算，随后执行 MBH 递归组装，
    输出完整的 (I + A)^{-1}。

    说明：mch_output / zero_matrix / identity_matrix 在 kernel 中按"单个 BT×BT 矩阵"
    （基地址）读取，因此本调试脚本聚焦"单 tile"场景（B=H=1，T==BT），逐个验证
    BT=16/32/64/128 下 MBH 模块的精度。
"""
import torch
import torch_npu
import numpy as np
import fla_npu

torch.npu.utils.set_device(0)

FRAC = 16  # 叶子块大小（MCH 对角块）


def build_mch_output(block):
    """构造 MCH 模块输出：块对角矩阵，每个 16×16 对角块为 (I16 + A_block)^{-1}，
    非对角块为 0。block 为 BT×BT 的严格下三角矩阵 (float32 numpy)。"""
    bt = block.shape[0]
    mch = np.zeros((bt, bt), dtype=np.float32)
    num_fracs = (bt + FRAC - 1) // FRAC
    for f in range(num_fracs):
        s = f * FRAC
        e = min(s + FRAC, bt)
        sz = e - s
        diag = block[s:e, s:e]
        eye = np.eye(sz, dtype=np.float32)
        mch[s:e, s:e] = np.linalg.inv(eye + diag)
    return mch


def solve_tril_golden_block(block):
    """单个 BT×BT 块的 golden：(I + A)^{-1}。"""
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


def test_solve_tril_mbh(BT, layout="bhtd", dtype=torch.float16, seed=42):
    """MBH 调试单 tile 测试：B=H=1, T=BT，单个 chunk。"""
    B, H, T, chunk_size = 1, 1, BT, BT

    block = generate_lower_tri_block(BT, dtype, seed)              # BT×BT 严格下三角 A
    block_np = block.float().numpy()

    # 构造接口入参
    mch_np = build_mch_output(block_np)                            # MCH 输出（块对角逆）
    zero_np = np.zeros((BT, BT), dtype=np.float32)
    eye_np = np.eye(BT, dtype=np.float32)
    golden_np = solve_tril_golden_block(block_np)                 # 完整 (I+A)^{-1}

    # 组织为算子输入张量（BHTD: [B,H,T,BT] / BSND: [B,T,H,BT]）
    if layout == "bhtd":
        A = block.reshape(B, H, T, chunk_size)
    else:
        A = block.reshape(B, T, H, chunk_size)

    A_npu = A.npu()
    mch_npu = torch.from_numpy(mch_np).half().npu()
    zero_npu = torch.from_numpy(zero_np).half().npu()
    eye_npu = torch.from_numpy(eye_np).half().npu()

    out_npu = torch.ops.npu.npu_solve_tril(
        A_npu, chunk_size,
        mch_output=mch_npu,
        zero_matrix=zero_npu,
        identity_matrix=eye_npu,
        layout=layout,
    )
    out_cpu = out_npu.cpu()

    if layout == "bhtd":
        inv_block = out_cpu[0, 0].float().numpy()
    else:
        inv_block = out_cpu[0, :, 0].float().numpy()

    # ===== 诊断3专用：当 kernel 编译为 PASSTHROUGH==3 时，输出应为 MNEG=-A。=====
    # 设环境变量 MBH_DIAG_MNEG=1 时，把输出与 -A / -A^T / 0 比对，定位 MNEG 计算问题。
    import os
    if os.environ.get("MBH_DIAG_MNEG") == "1":
        negA = -block_np
        negAT = -block_np.T
        e_negA = np.abs(inv_block - negA).max()
        e_negAT = np.abs(inv_block - negAT).max()
        e_zero = np.abs(inv_block).max()
        tri_low = np.abs(np.tril(inv_block, -1)).max()
        tri_up = np.abs(np.triu(inv_block, 1)).max()
        print(f"      [MNEG diag BT={BT}] vs(-A)={e_negA:.4f}  vs(-A^T)={e_negAT:.4f}  "
              f"maxabs={e_zero:.4f}  lowTriMax={tri_low:.4f}  upTriMax={tri_up:.4f}")

    # 校验 1：与 CPU golden 的最大差
    max_diff = np.abs(inv_block - golden_np).max()
    # 校验 2：(I + A) @ out ≈ I
    eye = np.eye(BT, dtype=np.float32)
    product = (eye + block_np) @ inv_block
    verify_err = np.abs(product - eye).max()

    passed = verify_err < 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] MBH layout={layout} BT={BT}: "
          f"max_diff={max_diff:.6f}, verify_err={verify_err:.6f}")

    # ===== 诊断：逐 16×16 子块对比，定位 MBH 输出哪一块出错 =====
    # out_block_err : 算子输出 vs golden 的逐块最大误差
    # diag_vs_mch   : 输出对角块 vs 注入的 mch_out 对角块（应一致，因 MBH 保留对角逆）
    # 仅在失败且块数不太多时打印，避免刷屏。
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
        # 对角块是否等于注入的 mch_out（MBH 应原样保留对角逆块）
        for bi in range(nf):
            o = inv_block[bi*FRAC:(bi+1)*FRAC, bi*FRAC:(bi+1)*FRAC]
            m = mch_np[bi*FRAC:(bi+1)*FRAC, bi*FRAC:(bi+1)*FRAC]
            print(f"        diag block {bi}: out-vs-mchOut max={np.abs(o-m).max():.4f}, "
                  f"out[0,0]={o[0,0]:.4f} mch[0,0]={m[0,0]:.4f}")
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("SolveTril NPU Test - MBH module debug (torch.ops.npu)")
    print("=" * 60)

    results = []

    # MBH 模块单 tile 调试：BT=16/32/64/128（128 为主目标）
    print("\n--- MBH debug, BHTD layout [B=1, H=1, T=BT, BT] ---")
    for bt in (16, 32, 64, 128):
        results.append(test_solve_tril_mbh(bt, layout="bhtd"))

    print("\n--- MBH debug, BSND layout [B=1, T=BT, H=1, BT] ---")
    for bt in (16, 32, 64, 128):
        results.append(test_solve_tril_mbh(bt, layout="bsnd"))

    # 不同随机种子，增强覆盖
    print("\n--- MBH debug, BT=128, multiple seeds ---")
    for sd in (1, 7, 123):
        results.append(test_solve_tril_mbh(128, layout="bhtd", seed=sd))

    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    if all(results):
        print("All MBH tests PASSED!")
        exit(0)
    else:
        print("Some MBH tests FAILED!")
        exit(1)
