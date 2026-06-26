/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef SOLVE_TRIL_COMMON_H
#define SOLVE_TRIL_COMMON_H

#include "kernel_operator.h"

// ========== Ascend950 平台检测 / UB 优化开关 ==========
// 经实测：Ascend950(arch35) 上把"辅助矩阵在 AIC 核的 UB 上自生成 + UB↔L1 直通"
// 的方案跑不通——AIC(cube)核访问 UB / 执行向量指令(Duplicate/Muls/DataCopy↔UB)
// 得到的是 0/垃圾值（BT=16 输出恒为 0 即为佐证）。因此暂时关闭该 UB 优化路径，
// 在 950 上沿用经验证可用的 910b GM 通路（AIV 生成 I/-I/Zero 到 GM，AIC 经
// GM 加载 + GM scratch 中转）。KERNEL_TYPE_MIX_AIC_1_2 在 950 上同样可用。
//
// 待确认 arch35 的 AIC 核确实可访问 UB 后，再将本宏改回基于
// (__CCE_AICORE__ == 310) 的检测以启用 UB 优化。
#define SOLVE_TRIL_PLATFORM_ASCEND950 0

// ========== arch3510 UB 优化开关（Ascend950 性能优化）==========
// 利用 arch3510 架构特性把 MBH 递归的 GM 中转换成 UB 中转，由 cube(AIC)+vector(AIV) 协作：
//   - AIC：Mmad；递归结果 Fixpipe(L0C->UB) 暂存（task1）。
//   - AIV：MCH 输出 GM->UB(nd2nz) 暂存；每轮把所需块经 raw UB->L1 提取到 L1（task2）。
//     （UB 位于 AIV，相关 DataCopy 必须在 AIV 执行；cube 仅经 Fixpipe 写 UB。）
// 仅在 arch3510(__NPU_ARCH__==3510) 上默认开启（Fixpipe L0C->UB / GM-UB nd2nz / raw UB->L1
// 均为该架构指令）；其它架构(910b 等)自动为 0，回退到已验证 11/11 的 GM 通路。
#ifndef SOLVE_TRIL_MBH_UB_OPT
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define SOLVE_TRIL_MBH_UB_OPT 1
#else
#define SOLVE_TRIL_MBH_UB_OPT 0
#endif
#endif

#if SOLVE_TRIL_PLATFORM_ASCEND950
// Ascend950: 纯 AIC 模式，无 AIC↔AIV 同步需求
// 辅助矩阵在 UB 上生成，无需 GM workspace slot
#else
// AIC/AIV 同步标志常量
constexpr uint64_t SYNC_AIV_AIC_FLAG_SOLVE = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG_SOLVE = 5;

// GM 共享 workspace slot（ND 行优先）
constexpr int32_t GM_WS_I    = 0;
constexpr int32_t GM_WS_INEG = 1;
constexpr int32_t GM_WS_ZERO = 2;
constexpr int32_t GM_NUM_SHARED_SLOTS = 3;

// AIV 核生成辅助矩阵的参数
constexpr int32_t ROWS_PER_AIV_CORE = 8;
constexpr int32_t DIAG_BLOCK_ELEMS  = ROWS_PER_AIV_CORE * 16;  // 8x16

// 8x16 ND 块中对角 mask（偶数条带）
// 对角在 col 0..7: elem = i*16 + i
constexpr uint64_t DIAG_MASK_8X16_EVEN[2] = {
    0x0008000400020001ULL,
    0x0080004000200010ULL
};
#endif

#endif  // SOLVE_TRIL_COMMON_H
