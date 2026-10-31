/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged_policy.h
 * \brief 架构分档（arch35 = A5/950，dav-3510；arch22 = A2/A3，910B/910_93）与按 arch 的策略开关。
 */

#ifndef PREF_PROCESS_FWD_KERNEL_MERGED_POLICY_H
#define PREF_PROCESS_FWD_KERNEL_MERGED_POLICY_H

namespace GDN {

// ---------------- 目标 arch 分档 ----------------
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define PPFM_ARCH_IS_950 1
// 950：有 L0C→UB 直连通道（A2 优化方案可用）
#ifndef PPFM_VTMP_UB
#define PPFM_VTMP_UB 1
#endif
#else
#define PPFM_ARCH_IS_950 0
// 910B/910_93(A2/A3)：cube↔vector 必须经 GM，无 L0C→UB 通道
#ifndef PPFM_VTMP_UB
#define PPFM_VTMP_UB 0
#endif
#endif
// 跨核 flag 模式：950 用 0x4（同 block 内 AIC↔AIV，每子核 slot），910B 用 0x2
#if PPFM_ARCH_IS_950
constexpr int32_t PPFM_XCORE_MODE = 0x4;
#else
constexpr int32_t PPFM_XCORE_MODE = 0x2;
#endif

// AIV UB 上限（编译期自检用）：950 = 256 KiB，A2/910B = 192 KiB
#if PPFM_ARCH_IS_950
constexpr int32_t PPFM_UB_CAP_BYTES = 256 * 1024;
#else
constexpr int32_t PPFM_UB_CAP_BYTES = 192 * 1024;
#endif

// 省掉 dH 每 chunk 的 GM 往返；0 = 旧路径（AIC 写 dHF_/dHF1_ GM，AIV 回读）。
// 注意： SPLIT_M 的语义是「M 方向对半、两半分别写进两个 AIV 子核各自 bank
// 的同一偏移」，
//   所以 AIV 侧按**连续半区**（子核 i = 行 [i*K/2, (i+1)*K/2)）取自己那半。
#ifndef PPFM_DH_CV
#if PPFM_ARCH_IS_950
#define PPFM_DH_CV 1
#else
#define PPFM_DH_CV 0
#endif
#endif

// **T2 也走 L0C→UB**（与 dH 同一套 SPLIT_M 落点）。
// 动机不只是性能：把 dH 挪进 UB 之后，T2 成了 m 链上**唯一**剩下的 "AIC→AIV 经 GM" 边，
// 而 950 上这条边历史上就是薄弱点（validation 的九组排除实验）。
// 顺带省掉每 chunk「T2 写 GM 64 KiB + 回读 64 KiB」。
#ifndef PPFM_T2_CV
#if PPFM_ARCH_IS_950
#define PPFM_T2_CV 1
#else
#define PPFM_T2_CV 0
#endif
#endif

// 950：实测只用跨核 flag 就够（并且去掉 DCCI 后竞态由 3/6 降到 1/6），默认 0。
// 910B/910_93：实测 AIC 会读到 AIV 尚未对其他核可见的 bf16(h)（chunk0 的 h≡0 探针
//   仍得到 1.6e-2 的 h），故先按 A2 老做法启用 DCCI/DSB；若后续定位到更精确的边，
//   可只保留必要的那一条。
#ifndef PPFM_LEGACY_CACHEOPS
#if PPFM_ARCH_IS_950
#define PPFM_LEGACY_CACHEOPS 0
#else
#define PPFM_LEGACY_CACHEOPS 1
#endif
#endif

// ---- C3 实验（A2 专用；默认 = 现状，950 机器码按构造不变）----
// 动机：A2 的 DCCI/DSB 目前"不分生产方"地全开（AIC 每 chunk 10 次整缓存 DCCI + AIV 6 次
//   + 4 次 DSB），而 950 的实测记录是"去掉 DCCI 后竞态由 3/6 降到 1/6"。
//   把 A2 的缓存操作按**数据生产方**拆成两组，做 4 组 A/B（默认都是 1 = 现状）：
//     W 组 = AIC 抬 flag 前，对**自己算出的 C** 做的 DCCI/DSB（vTmp / t1Bf_ / dH / T2 及末尾 DSB）
//     R 组 = AIV 读 **AIC 算出的 C** 之前做的失效（dHF_/dHF1_/t2F_/t2F1_/vTmpF_/t1F_）
//   保留不动的是"消费 AIV 产物"的那一半（AIC 读 wBf_/kBf_/hBf_/mBf_/lBf_/vNewBf_ 前的 DCCI，
//   以及 AivSetToAic 的 DSB）—— 那一半有明确故障证据（AIC 读到尚未可见的 bf16(h)）。
//   实验矩阵：W1R1（基线）/ W0R1 / W1R0 / W0R0，判据 = PROC S≥40 的失败率 + L1 + perf。
#ifndef PPFM_A2_CACHEOPS_W
//   **结论（2026-10-08，234 上机）**：W0R1（=A2 默认）在 60 进程位级口径下残差 1/60，
//   基线 W1R1 是 9/60 ⇒ 去掉 W 组的 DCCI 没有让竞态变差，反而随 C7/C9 一起变好。
#define PPFM_A2_CACHEOPS_W (PPFM_ARCH_IS_950 ? 1 : 0)
#endif
#ifndef PPFM_A2_CACHEOPS_R
#define PPFM_A2_CACHEOPS_R 1
#endif
// 注意：下面两个宏在 950 上必须恒为 0（= 与 PPFM_LEGACY_CACHEOPS 同源），
//       以保证 950 的预处理输出与 kernel .o 逐字节不变（L-A）。
#if PPFM_ARCH_IS_950
#define PPFM_KEEP_CACHEOPS_W 0
#define PPFM_KEEP_CACHEOPS_R 0
#else
#define PPFM_KEEP_CACHEOPS_W (PPFM_LEGACY_CACHEOPS && PPFM_A2_CACHEOPS_W)
#define PPFM_KEEP_CACHEOPS_R (PPFM_LEGACY_CACHEOPS && PPFM_A2_CACHEOPS_R)
#endif

}  // namespace GDN

#endif  // PREF_PROCESS_FWD_KERNEL_MERGED_POLICY_H
