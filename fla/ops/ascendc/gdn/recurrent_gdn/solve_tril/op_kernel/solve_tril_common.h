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

// ========== MBH 诊断开关（默认关闭）==========
// 打开后，BT>16 的调试路径跳过 RecursiveMerge，仅做 X×I 写回（X=mch_out）。
// 期望输出 == mch_out（块对角逆，非对角块为 0）。用于隔离"多分形 matmul + 写回"
// 是否正确：若输出对角块==mch_out 且非对角==0，则 matmul/store 正常，bug 在
// ExtractBlocksToSlot / L0CToSlot / L0C 累加；否则多分形 matmul 或写回本身有问题。
#ifndef SOLVE_TRIL_MBH_PASSTHROUGH
#define SOLVE_TRIL_MBH_PASSTHROUGH 0
#endif

// ========== UB-opt 卡死定位探针（默认 0=关闭）==========
// UB-opt 路径在 950 上始终卡死在 setup（BT=16 前无输出），多轮修复无效，故加分级早退探针定位：
//   =1：Process() 一进入即 return（Resource 已构造、Init 已跑）。若仍卡死 -> 卡在 Resource 构造/Init/launch；
//       若完成（不超时，结果错）-> 卡在 Process 体内（SyncAll 或 compute）。
//   =2：执行到 SyncAll 之后 return（cube：SyncAll 后；vector：aux-gen+SyncAll 后）。
//       若 =1 完成而 =2 卡死 -> 卡在 SyncAll/aux-gen；若 =2 完成 -> 卡在 SyncAll 之后（PrepareConstants/compute）。
//   =3：cube 执行到 PrepareConstants 之后 return。隔离 PrepareConstants vs tile 计算。
//   =0：完整执行。定位完成后移除本探针。
#ifndef SOLVE_TRIL_UBOPT_DIAG
#define SOLVE_TRIL_UBOPT_DIAG 0
#endif

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

#if SOLVE_TRIL_MBH_UB_OPT
// UB 优化的 AIC<->AIV 跨核同步 flag。1:2 MIX 下 FFTS 计数语义（已查 SDK 与参考算子证实）：
//   - mode 0x2 的 CrossCoreSetFlag 是【按 AIV subcore 扇出】：AIC 一次 Set 给【两个 subcore】各 +1；
//     每个 subcore 一次 Set 只给 AIC +1。每个 CrossCoreWaitFlag 消费(−1)一个计数。
//   - 故方向 1->2（AIC->AIV）：用【1 个 flag、AIC Set 1 次】即可让两个 subcore 各 Wait 1 次满足。
//     方向 2->1（AIV->AIC）：必须【每 subcore 一个独立 flag】，两 subcore 各 Set 自己的，AIC 各 Wait 一次。
//     （之前两 subcore 共用一个 aivReady、AIC 只 Wait 一次 -> 计数泄漏 -> ≤15 次后冻结。参考算子
//      chunk_gated_delta_rule_fwd_h 即用 vec2Done[2]={3,4} 两个 flag、cube 等两次。）
//   - cube 侧 Set 必须用 cube pipe（PIPE_FIX/PIPE_MTE2/PIPE_MTE1/PIPE_M）；AIV 侧用 vector pipe
//     （PIPE_MTE3 等）。cube 用 PIPE_MTE3 会编译成空操作（MTE3 非 cube pipe）-> 必死锁。
// 含义：
//   - UBOPT_FLAG_AIC_READY：AIC -> 两 AIV subcore。“常量就绪(L0)/本层结果已 Fixpipe 写回 xUB_(L>=1)”。
//   - UBOPT_FLAG_AIV_READY_0/1：AIV subcore0/1 -> AIC。“本层 drv/oth 已提取到 L1，可做 Mmad”。
// 【flag id 选择】避开：aux-gen 握手 SYNC_*_FLAG_SOLVE=3/5；catlass barrier 预留 7~10；
//   SDK SyncAll 占用 11/12/13/14（dav_3510 sync_impl.h）。用低位空闲 id 0/1/2。
constexpr uint16_t UBOPT_FLAG_AIC_READY   = 0;  // AIC -> 两 subcore（单 flag 扇出）
constexpr uint16_t UBOPT_FLAG_AIV_READY_0 = 1;  // AIV subcore0 -> AIC
constexpr uint16_t UBOPT_FLAG_AIV_READY_1 = 2;  // AIV subcore1 -> AIC
#endif
#endif

#endif  // SOLVE_TRIL_COMMON_H
