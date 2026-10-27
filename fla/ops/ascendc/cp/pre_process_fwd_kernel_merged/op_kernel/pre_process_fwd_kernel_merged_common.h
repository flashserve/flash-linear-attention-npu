/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged.cpp
 * \brief CP 前处理算子 kernel（v2：Cube/MIX 版，AIC 4 个 bf16 matmul + AIV elementwise）。
 *
 * 一个工作项 = 一条链（段 n × value-head hv），链内逐 chunk 串行。所有中间量放在
 * 本核 user workspace（GM），AIC/AIV 用 CrossCore flag 串联：
 *
 *   AIV  prologue : h←0（fp32+bf16）、m←I（fp32+bf16）        → kFlagState
 *   AIV  stage c  : 载入 W/k/left/v（bf16，尾块零填充）+ dg/decay → kFlagInputs
 *   AIC  ①③       : vTmp = W_c@bf16(h)、T1 = W_c@bf16(m)      → kFlagHalf1
 *   AIV           : v_new = (v - vTmp)·dg → bf16(v_new)；bf16(T1) → kFlagVNew
 *   AIC  ②        : dH = k_c^T @ bf16(v_new)                  → kFlagDH
 *   AIV           : h = decay⊙h + dH（同步存 bf16(h)）          → kFlagHUpd
 *   AIC  ④        : T2 = left^T @ bf16(T1)                    → kFlagT2
 *   AIV           : m = decay⊙m - T2（同步存 bf16(m)）          → kFlagState（下一 chunk）
 *
 * 语义与竞品/标杆对齐（见 docs/api.md、tests/atk/pre_process_fwd_kernel_merged/scripts/）：
 *   * h 项用**未加门控的 k**，门控只作用在 v_new 上；m 项用 left = k·2^(g_last-g_t)；
 *   * h/m 的衰减：USE_G 为标量 2^(g_last)，USE_GK 为逐 k 的 2^(gk_last[k])；
 *   * m 链在 FP32 上做（这里的乘积把 m 量化到 bf16，与竞品 default/TF32 口径一致，
 *     在模型同构数据下实测 matched=1.000000）。
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

// 架构分档与按 arch 的开关默认值（TilingKey/同步协议相关的常量也在这里）
#include "pre_process_fwd_kernel_merged_policy.h"

// CATLASS 的 arch 选择必须在包含 catlass 头之前给出（同 chunk_scaled_dot_kkt 的写法）
#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
// A5 的 L0C→UB 直连（手写 tile 级 mmad 用；A2/A3 用 PackedTileCopyTla 落 GM）
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "pre_process_fwd_kernel_merged_struct.h"



#ifndef PREF_PROCESS_FWD_KERNEL_MERGED_COMMON_H
#define PREF_PROCESS_FWD_KERNEL_MERGED_COMMON_H

namespace GDN {
using namespace AscendC;

// 行缩放标量预取（位级不变）
#ifndef PPFM_ROW_PREFETCH
#define PPFM_ROW_PREFETCH 1
#endif

// glast 走 UB（省每 chunk 两次 GM 标量读）
#ifndef PPFM_GLAST_UB
#define PPFM_GLAST_UB 1
#endif

// K2-vf: KDA 逐行 decay 用 VF/RegBase（位级不变）
#ifndef PPFM_KDA_VF_ROWS
#define PPFM_KDA_VF_ROWS 1
#endif

// K1-a: KDA 逐行 decay 的标量预取（先把一批 factor 读进寄存器数组，再整批 Muls）。
// 算术与舍入顺序完全不变 ⇒ 位级不变。
// 0=关（回退逐行 GetValue+Muls），其它=每批预取行数。
// 只作用于 KDA 的逐行 decay 分支（A2/A3 唯一可用的等价改造；950 走 RegBase VF 路径，
// 该分支在 950 上被预处理器整段丢弃 ⇒ 本宏不影响 950 的机器码）。
#ifndef PPFM_KDA_ROW_BATCH
#define PPFM_KDA_ROW_BATCH 16
#endif

#if PPFM_KDA_ROW_BATCH > 0
#define PPFM_KDA_ROW_PREFETCH 1
#else
#define PPFM_KDA_ROW_PREFETCH 0
#endif

// P0-1k: KDA 的状态更新三合一（逐行 decay 缩放 + 加/减 dH 合成一趟 RegBase；位级不变）
#ifndef PPFM_STATE_FUSE
#define PPFM_STATE_FUSE 1
#endif

// P1-1: left = cast_bf16(cast_fp32(k) · dg[row]) 合成一趟 RegBase（位级不变；仅 950）
#ifndef PPFM_LEFT_FUSE
#define PPFM_LEFT_FUSE 1
#endif

// P2-1: KDA(USE_GK) 下 left ≡ k ⇒ 复用 k 的 GM 槽，AIV 不再重复写一份 lBf_（位级不变）
#ifndef PPFM_KDA_LEFT_ALIAS
#define PPFM_KDA_LEFT_ALIAS 1
#endif

// epilogue 逐行 hm 写合并成一次跨步 DataCopy
#ifndef PPFM_EP_MERGE
#define PPFM_EP_MERGE 1
#endif

// m=I 整块构造（行间零栅栏）
#ifndef PPFM_M_INIT_BLOCK
#define PPFM_M_INIT_BLOCK 1
#endif

#ifndef PPFM_A2_BLOCKWISE
#define PPFM_A2_BLOCKWISE 1
#endif

constexpr int32_t CV_BT = 64;
constexpr int32_t CV_K = 128;
constexpr int32_t CV_V = 128;
constexpr int32_t CV_LANES = 8;

// ---------------- 每核 GM 暂存区（字节偏移）----------------
// [实验] 把 h 的 fp32 状态从偏移 0 挪到最后：CATLASS matmul 可能在本核 workspace 起始处
// 使用自己的暂存区，之前 hF32_（偏移 0 起）恒被冲成 0，而更靠后的 hBf_ 是对的。
constexpr int64_t WS_M_F32 = 0;                        // m  [K,K] fp32 65536
constexpr int64_t WS_H_BF = WS_M_F32 + 65536;          // bf16(h) [K,V] 32768
constexpr int64_t WS_M_BF = WS_H_BF + 32768;           // bf16(m) [K,K] 32768
constexpr int64_t WS_W_BF = WS_M_BF + 32768;           // W_c  [BT,K] bf16 16384
constexpr int64_t WS_K_BF = WS_W_BF + 16384;           // k_c  [BT,K] bf16 16384
constexpr int64_t WS_L_BF = WS_K_BF + 16384;           // left [BT,K] bf16 16384
constexpr int64_t WS_V_BF = WS_L_BF + 16384;           // v_c  [BT,V] bf16 16384
constexpr int64_t WS_VTMP_F32 = WS_V_BF + 16384;       // W@h   [BT,V] fp32 32768
constexpr int64_t WS_VNEW_BF = WS_VTMP_F32 + 32768;    // bf16(v_new) [BT,V] 16384
constexpr int64_t WS_DH_F32 = WS_VNEW_BF + 16384;      // dH [K,V] fp32 65536
constexpr int64_t WS_T1_F32 = WS_DH_F32 + 65536;       // T1 [BT,K] fp32 32768
constexpr int64_t WS_T1_BF = WS_T1_F32 + 32768;        // bf16(T1) [BT,K] 16384
constexpr int64_t WS_T2_F32 = WS_T1_BF + 16384;        // T2 [K,K] fp32 65536
// dH / T2 各双缓冲一份：AIC 写 chunk c 用的那份，AIV 在 chunk c 读 c-1 写的那份
constexpr int64_t WS_DH_F32_1 = WS_T2_F32 + 65536;     // dH 第二份 65536
constexpr int64_t WS_T2_F32_1 = WS_DH_F32_1 + 65536;   // T2 第二份 65536
constexpr int64_t WS_GATE = WS_T2_F32_1 + 65536;       // dg[BT] + decay[K] 1024
constexpr int64_t WS_H_F32 = WS_GATE + 4096;           // h  [K,V] fp32 65536（最后一段）
// kBf_/lBf_ 的第二槽（chunk 奇偶选槽）
constexpr int64_t WS_K_BF_1 = WS_H_F32 + 65536;        // k_c 第二槽 16384
constexpr int64_t WS_L_BF_1 = WS_K_BF_1 + 16384;       // left 第二槽 16384
// m 链 fp32 化（PPFM_M_CHAIN_FP32）用的 Kw = left^T @ W_c（[K,K] fp32）
constexpr int64_t WS_KW_F32 = WS_L_BF_1 + 16384;       // Kw [K,K] fp32 65536
constexpr int64_t WS_GATE_DG = 0;
constexpr int64_t WS_GATE_DECAY = CV_BT * 4;

// ---------------- CrossCore flag（MIX 内 AIC <-> 2×AIV）----------------
// 两个平台的同步模型不同（见 PPFM_XCORE_MODE / PPFM_ARCH_IS_950）：
//   * 950 用 mode 0x4：flag ID 按 subblock 分槽（第二个子核 = id + 16），AIC 侧显式
//     wait/set 两个 slot。与仓内 arch35 算子的约定一致
//     （见 chunk_fwd_h/op_kernel/chunk_fwd_h_policy.h、kda/chunk_kda_fwd/.../fwd_h.h）。
//     实测反例：用 A2/A3 风格的 `CrossCoreSetFlag<0x2, ...>` 时，950 上先干完的那个
//     子核就会把 AIC 放行，AIC 的 mm1/mm3 读到"只写了一半"的 h/m bf16 状态，
//     表现为 h 半边约一半行数据错、m 基本对（GDN 快路径暴露，KDA 慢路径看不出来）。
//   * 910B/910_93 用 mode 0x2：AIC 与「本 block 的 2 个 AIV」是集合同步 —— AIC 的一次
//     set 对本 block 两个 AIV 同时置起；两个 AIV 都 set 同一个 ID 才算 AIC 侧事件置起。
//     因此 AIC 侧每轮只需一对 set/wait（多 set/多 wait 会让 flag 计数失衡）。
constexpr uint16_t PPFM_SUBFLAG_STRIDE = 16;   // AIV 子核 1 的 slot 偏移
constexpr uint16_t kFlagInputs = 1;   // AIV -> AIC：本 chunk staging 就位
constexpr uint16_t kFlagState = 2;    // AIV -> AIC：h/m 状态就位
constexpr uint16_t kFlagHalf1 = 3;    // AIC -> AIV：vTmp 与 T1 就位
constexpr uint16_t kFlagVNew = 4;     // AIV -> AIC：bf16(v_new) 与 bf16(T1) 就位
constexpr uint16_t kFlagDH = 5;       // AIC -> AIV：dH 与 T2 就位
// （dH 走 L0C->UB）：dH 的 UB 槽「归还」信用（AIV -> AIC）。槽 j 用 id + j，
// AIC 侧按 subblock 各等一次（id 与 id+PPFM_SUBFLAG_STRIDE），
// 与仓内 arch35 的 MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN/CV_SUBBLOCK_FLAG_STRIDE 同构。
constexpr uint16_t kFlagDhFree = 8;
// T2 的 UB 槽归还信用（AIV -> AIC），单槽。
constexpr uint16_t kFlagT2Free = 10;

// AIV 子核：写自己本地 slot（硬件按子核自动映射到 id / id+16）
__aicore__ inline void AivSetToAic(uint16_t id)
{
    // set_intra_block 不保证"之前的搬运已落地"，这里显式排空 MTE3
    PipeBarrier<PIPE_MTE3>();
    // 对称于 AIC 侧：本核写出的 GM（kBf_/wBf_/lBf_/vNewBf_/t1Bf_/状态）也要先对其他核可见
#if PPFM_LEGACY_CACHEOPS
    DataSyncBarrier<MemDsbT::DDR>();
#endif
    CrossCoreSetFlag<PPFM_XCORE_MODE, PIPE_MTE3>(id);
}

__aicore__ inline void AivWaitFromAic(uint16_t id)
{
    // 950：用 PIPE_S 排队 —— 等待指令卡住后续指令的发射（否则后面的 MTE2/V 会先跑）。
    // 910B/910_93：消费方是 MTE2（从 GM 回读 cube 的 C），按 A2 惯用法在 MTE2 上排队
    //   （见仓内 arch22 的 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(..._READY_FLAG)`）。
    //   实测 A2 上用 PIPE_S 等待时，随后的 MTE2 会提前发射，读到"半新半旧"的 C：
    //   表现为 h 半边误差 ~1.5e-2 且逐次小幅跳动（v=0 的探针本应得到 h≡0）。
#if PPFM_ARCH_IS_950
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_S>(id);
#else
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_MTE2>(id);
#endif
}

// AIC：消费 AIV 侧的通知
//   950 (mode 0x4)：flag ID 按 subblock 分槽，两个 AIV 子核各自 set 自己的 ID ⇒ 这里要等两次
//   910B/910_93 (mode 0x2)：AIC 与「本 block 的 2 个 AIV」是**集合同步**——两个 AIV 必须
//     都 set 同一个 ID，事件才会对 AIC 置起 ⇒ 这里只等一次
//   （约定出处：仓内 chunk_fwd_h/op_kernel/chunk_fwd_h_policy.h 的 FwdHAicPeerFlag 注释）
__aicore__ inline void AicWaitFromAiv(uint16_t id)
{
#if PPFM_ARCH_IS_950
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_S>(id);
    CrossCoreWaitFlag<0x4, PIPE_S>(static_cast<uint16_t>(id + PPFM_SUBFLAG_STRIDE));
#else
    // 910B：AIC 侧消费方同样是 MTE2（把 AIV 写好的 k/w/h/m/vNew 搬进 L1/L0）
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_MTE2>(id);
#endif
}

// AIC -> AIV：
//   950：两个 slot 都要置位，否则只会唤醒一个子核；
//   910B/910_93：0x2 下一次 set 即对本 block 的两个 AIV 同时置起，只能 set 一次
//   （多 set 会让计数失衡 —— 单条 flag 连续 set 超过 15 次会挂死，见 catlass
// cross_core_sync.hpp）
__aicore__ inline void AicSetToAiv(uint16_t id)
{
    PipeBarrier<PIPE_FIX>();
    CrossCoreSetFlag<PPFM_XCORE_MODE, PIPE_FIX>(id);
#if PPFM_ARCH_IS_950
    CrossCoreSetFlag<PPFM_XCORE_MODE, PIPE_FIX>(static_cast<uint16_t>(id + PPFM_SUBFLAG_STRIDE));
#endif
}

// 手写 tile 路径下 L1 的两个槽（A 在前、B 在后），单位字节。
// 两个槽都必须落在 64 KiB 边界上：A 槽（128x128 bf16）实际只占 32 KiB，若把 B 紧跟
// 在 32 KiB 处，L1→L0B 的 LoadData 就落在非 64 KiB 对齐的 L1 基址上 —— 数值仍然正确，
// 但 950 的 mssanitizer（CANN 9.1.0 与 9.2.0-beta.1 表现一致）会报
//   ERROR: misaligned access of size 512 at 0x3 on L1 ... block aic(0-3)
// 调用栈落在 catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp 的 AscendC::LoadData，
// 并连带把 kernel 参数块的两条读也报成 illegal read，使 ATK 内存检测判 Failed。
// 槽间距取 64 KiB 后 mssanitizer 全清（精度 111/111、确定性 5/5、性能不变）；
// fp32 专用槽（128 / 192 KiB）本来就按 64 KiB 对齐，这里只是把 bf16 槽补齐到同一约定。
constexpr int32_t TILED_L1_A_OFF = 0;
constexpr int32_t TILED_L1_B_OFF = 64 * 1024;
// L1A/L1B 的「容量形状」：必须与 BlockMmad 的 L1_TILE_M/K/N 一致（zZ/nZ 分形布局的
// stride 由 originShape 决定，用实际 (m,k) 构造会让 GM→L1 的落点与 L1→L0 的读点错位，
// 表现为 mmad 读到空 L0、C 恒为 0）。
constexpr int32_t TILED_L1_CAP_M = 128;
constexpr int32_t TILED_L1_CAP_K = 128;
constexpr int32_t TILED_L1_CAP_N = 128;
// fp32（PPFM_M_CHAIN_FP32）专用 L1 偏移：Ky 128x128 fp32 = 64 KiB、m 窗口 128x128 fp32 = 64 KiB，
// 不能沿用 bf16 的 0 / 64 KiB（会互相覆盖）。
constexpr int32_t TILED_L1_A_F32_OFF = 128 * 1024;
constexpr int32_t TILED_L1_B_F32_OFF = 192 * 1024;
// 手写 tile 级 mmad 开关：1=用 TileMmadTla 手拼，0=退回 BlockMmadTla
// （已验证） A1 数值已对齐（2026-09-28，241 device6 实测）：
//   - 只 tile mm1（SEL=1）时 m 半边与基线逐位一致，只 tile mm3（SEL=2）时 h 半边逐位一致
//     ⇒ 两个 matmul 各自的 tile 结果与 BlockMmad 等价；
//   - 全 tile 时 5 轮 smoke 有 3 轮命中**已知的 GDN h 跨核可见性窗口**（TILE=0 基线 5/5
// 干净），
//     误差幅度 1.17~1.53 随机跳动，属时序放大，待 A2（A5 L0C→UB 直连）结构性消除。
//   根因（曾表现为 h 半边错、m≈decay·I）：`CopyL0CToGmTla` 的 4 参调用会误选
//   `(dst, src, l0Batch, dstNdStride)` 批处理变体，l0Batch=0 ⇒ fixpipe 一个块都不搬，
//   C 恒为 workspace 初值 0。必须走 3 参 `(dst, src, unitFlag)`。
#ifndef PPFM_TILE_MMAD
#define PPFM_TILE_MMAD 1
#endif
// A2 优化方案（mm1 的 C 由 fixpipe SPLIT_M 直落 UB）开关。
//   0 = A5 主线（C 落 GM，已验证）；1 = UB 落点（首次测量 h 半边崩，UB
// 语义待测量确认）。
#ifndef PPFM_VTMP_UB
#define PPFM_VTMP_UB 1
#endif
// UB 语义诊断（默认 0）：1=UB 落点同时再写一份 GM，并在 AIV 侧回采探针
#ifndef PPFM_VTMP_UB_DIAG
#define PPFM_VTMP_UB_DIAG 0
#endif
// （默认 950=1 / A2A3=0）：dH 的 C 由 fixpipe 直落 UB（L0C->UB，SPLIT_M），
// （默认 1）：回收 UB 布局里"按子核切两份"的冗余。
// 依据（实测）：`SPLIT_M` 的 fixpipe 把 C 的 M 两半分别写进两个 AIV 子核
// **各自 bank 的同一偏移**，且两个子核从**同一个 UB 偏移**读到了**不同**的数据
// （位级一致 ⇒ 只能是各自 bank）。⇒ 凡"两个子核都会写"的 scratch 都不需要
// 再按 subIdx_ 切成两份，单份即可（省 ~39 KiB）。0 = 改用历史的两份布局。
#ifndef PPFM_UB_SHARE
#define PPFM_UB_SHARE 1
#endif
#if PPFM_UB_SHARE
#define PPFM_NSLOT 1
#else
#define PPFM_NSLOT PPFM_SUB
#endif
// （默认 1）：**h 状态常驻 UB**，去掉 h 的 fp32 GM 往返。
// 依据：已证每个 AIV 子核有独立 UB bank，而状态更新的行分配（起）是**连续半区**
// ⇒ 子核 i 只需要自己那 64 行 × cb_ 的 fp32（cb_=128 时 32 KiB/子核），不是整张 [K,V]。
// 省掉每 chunk「h fp32 读 64 KiB + 写 64 KiB」，只保留 AIC 需要的 bf16(h) 落盘。
#ifndef PPFM_H_UB
#define PPFM_H_UB 1
#endif
// h 常驻 UB 依赖两件事，二者都由 提供：① 状态更新的**连续半区**行分配；
// ② dH 已经在 UB 里（否则还要额外一份 UB 拷贝）。没有 L0C→UB 的 A2/A3 上自动关闭。
// （2026-10-08）：**已启用**（`PPFM_H_UB_A2=1`）——原来的两个阻塞都已修掉：
//   ① 行归属竞态（prologue/epilogue 与状态更新的行分区不一致，见 arch22 的修复）；
//   ② `UB_STATE_ROWS` 的判据用错宏 ⇒ A2 打开 H_UB 时 stateBlkF_/extBlkF_ 越界 16 KiB；
//   实现：那边 dH 仍从 GM 回读进 `extBlkF_` 暂存，h 本体不再往返 GM
//   （省每 chunk「h fp32 读 64 KiB + 写 64 KiB」）；910B 的 `ub_size=262144`
//   （按保守的 192 KiB 读也够）也放得下这 32 KiB。
//   历史记录（当时的阻塞）：910B 上实测它**数值全对但非确定**——
//     同一 kernel 连跑 4 次 dump，两两比较有 2/3 次出现差异（每次 3/5 个用例，
//     64 个元素、bf16 舍入量级，集中在 `m` 半边某一行）。而 （本改动的父版本）
//     连跑 4 次**完全确定**。⇒ 它改变了时序，把 A2 上"T2 走 GM"那条边的残余窗口
//     顶到了表面（与 950 的 之前同源）。**先把那条边收口，再打开这个开关。**
#ifndef PPFM_H_UB_A2
#define PPFM_H_UB_A2 1
#endif
#if PPFM_H_UB && !PPFM_DH_CV && !PPFM_H_UB_A2
#undef PPFM_H_UB
#define PPFM_H_UB 0
#endif
// （默认 1，仅 950）：**m 状态也常驻 UB**。对 h 做过同样的事（省 128 KiB/chunk 的
// fp32 往返）；m 的往返量完全一样，用 回收出来的 32 KiB 放下。
// 前提：状态相位的**连续半区**行分配（起有）⇒ 每个 AIV 子核只持有自己那 64 行。
#ifndef PPFM_M_UB
#define PPFM_M_UB 1
#endif
// 约束：
//   ① 必须 H_UB —— epilogue 的 M_UB 读取分支挂在 H_UB(+EP_MERGE) 里；
//   ② 状态更新的行块必须正好等于一个半区（PPFM_SBRB == CV_K/PPFM_SUB），
//      否则 mUb_ 的本地偏移与 epilogue 的连续半区对不上（下面有 static_assert）。
// （2026-10-09）：A2/A3 的非 CV 路径也实现了 m 常驻 UB（原来是 950 CV 专属），
//   因此不再要求 DH_CV；A2 侧可用 PPFM_M_UB_A2=0 单独关掉做 A/B。
#if PPFM_M_UB && !PPFM_H_UB
#undef PPFM_M_UB
#define PPFM_M_UB 0
#endif
#ifndef PPFM_M_UB_A2
#define PPFM_M_UB_A2 1
#endif
#if PPFM_M_UB && !PPFM_ARCH_IS_950 && !PPFM_M_UB_A2
#undef PPFM_M_UB
#define PPFM_M_UB 0
#endif
// ---------------- m 链精度（PPFM_M_CHAIN_FP32，默认 950 打开）----------------
// 契约（docs/design.md §1.3 第 3 条）与上游 H20（`AFFINE_CHAIN_PRECISION` 默认 `ieee`，
// 快档 `tf32x3` 也是三次分裂乘）都要求 **m 链的乘积在 FP32 上完成**。旧实现在 cube 上用
// 结合律 `T2 = left^T @ bf16(W_c @ bf16(m))`，把 fp32 的 `m` 状态与中间量 `t1` 各量化成
// bf16（2^-9）——实测 m 半边有效精度比契约差约 1000×（case 81: 6.26e-05 vs 2.03e-06），
// ATK `cv_fused_double_benchmark` 的 gk 档因此判为"小值域数错误差比例 3.63 > 2.0"。
// 打开后（与契约 S2/S4 逐条对应）：
//   ③ Kw[K,K] = FP32(left^T @ W_c)      —— bf16×bf16、FP32 累加（与契约一致）
//   ④ T2[K,K] = FP32(Kw @ m)            —— **fp32×fp32**（950 的 cube 原生支持 fp32 MMA；
//                                          见 third_party/catlass/examples/43_ascend950_basic_matmul）
// AIV 侧算法不变：`m` 的 fp32 状态本来就每 chunk 落 `mF32_`（本开关下 M_UB 分支也要落）。
// **默认关闭**：交付口径与 CP 组兄弟算子 `pre_process_bwd_kernel_merged` 对齐 ——
// `m` 链的 `Kw = LᵀW` 与兄弟的链式量同级，都按**模型 dtype** 入 Cube（bf16 操作数 + fp32 累加），
// 验收也按模型 dtype 判（见 `tests/atk/pre_process_fwd_kernel_merged/README.md`「精度口径」）。
// 打开它是 "IEEE FP32 m 链" 的替代实现（`ATK cv_fused_double_benchmark` 也能 111/111 过），
// 代价实测 **+65~80%**（950 板端 `msprof`：T1024 85.8 → 146.3 µs、T4096 294.3 → 531.5 µs），
// 因此不做交付默认；需要时显式 `-DPPFM_M_CHAIN_FP32=1`。
#ifndef PPFM_M_CHAIN_FP32
#define PPFM_M_CHAIN_FP32 0
#endif
// —— A2（arch22 / 910B·910_93）：m 链**保持 bf16**。这与 950 的默认一致，也正是本次的交付口径
//    ——对齐 CP 组兄弟算子 `pre_process_bwd_kernel_merged`：链上中间量按模型 dtype 入 Cube，
//    验收同样按模型 dtype 判（见 `tests/atk/pre_process_fwd_kernel_merged/README.md` 的「精度口径」）。
//    2026-10-08 在 234 上量过 A2 的实际水平（T=256、HK=2、HV=8，对比仓内 reference 契约）：
//      m 半边 absmax 3.4e-05 / absmean 1.9e-06（h 半边 9.5e-07）
//    ⇒ 与 950 的 bf16 m 链同级，按 `mixed_tolerance_bm` + `output_dtype_overrides{bf16}` 判达标，
//      因此 A2 不需要再做 m 链的 fp32/双量化等价实现。
//    （历史记录：A2 的「IEEE FP32 等价」原型 =「AIC 落 fp32 T1 → AIV 拆 hi/lo → 4 次 MMAD」曾
//      实现过一版，端到端实测不成立（m 半边 absmax 0.947，而 m 量级本身才 0.94），已按上述口径
//      撤下；若将来 A2 也要过严格的 `cv_fused_double_benchmark`，原型留档在 commit `62ece9e6`。）
// 注意：**不动 `PPFM_T2_CV` / `PPFM_M_UB`**。T2 仍然落 UB 单槽（只是改由 fp32 的
// `RunMmadNTF32Ub` 写），m 仍然常驻 UB；`PPFM_M_UB` 分支额外把 fp32 的 m 行块落
// `mF32_`，供 cube 的 ④' 直接读（见 arch35 的 vector/cube）。
// 临时诊断开关：1=在 prologue 给 AIC 要写的 C 缓冲预置哨兵（见 ProcessChain）
#ifndef PPFM_SENTINEL_PROBE
#define PPFM_SENTINEL_PROBE 0
#endif
// 临时诊断开关：1=把 AIV 读到的 vTmp 第 0 行搬到 hm 的 m 半边第 0 行（chain0/head0）
#ifndef PPFM_RD_PROBE
#define PPFM_RD_PROBE 0
#endif
// KDA 的逐 k 衰减（decay[k] = 2^gk_last[k]）是否走向量化实现。
// 0 = 原来的逐点 SetValue + Exp2Scalar（每 chunk 128 次"标量写 + 2 次全栅栏 + Exp + 标量读"）
// 1 = 整块 DataCopy gk_last → ×ln2 → Exp（与逐点版本逐位等价，L1 位级门禁验证）
#ifndef PPFM_KDA_DECAY_VEC
#define PPFM_KDA_DECAY_VEC 1
#endif
// T1（mm3 的 C）是否由 fixpipe 直接按 bf16 落 GM（=1）——省掉 AIV 侧"读回 t1F_(fp32) →
// Cast → 写 t1Bf_" 的整条回路（每 chunk 32KB 读 + 16KB 写 + 一次 32K 元素的 Cast）。
// 前提：fixpipe 的 fp32→bf16 量化与原来的 CAST_RINT 等价（L1 位级门禁验证）；
// 仅在 PPFM_TILE_MMAD=1（手写 tile 路径）下生效。
#ifndef PPFM_T1_FIXPIPE_BF16
#define PPFM_T1_FIXPIPE_BF16 1
#endif
#if !PPFM_TILE_MMAD
#undef PPFM_T1_FIXPIPE_BF16
#define PPFM_T1_FIXPIPE_BF16 0
#endif
// 满 chunk（rows == BT）时 AIC 直接读输入张量里的 w/k，省掉 AIV 每 chunk 的
// w 载入(16KiB)+w 落盘(16KiB)+k 落盘(16KiB) 与对应事件对；尾块仍走 staging
// 零填充路径。两侧都由 rows 判定，天然一致。
// 注意： 实测**净负收益**（2026-09-29，两平台一致），故默认 0：
//   950  T=1024 211.6→217.7 µs(+2.9%)、T=4096 624.8→643.5 µs(+3.0%)、模型 case 3825→3891 µs(+1.7%)
//   910B T=4096 651.3→661.7 µs(+1.6%)、模型 case 4055→4164 µs(+2.7%)
//   猜测原因：AIV 的 staging 写在读侧把数据"预热"进了 L2（AIC 随后读的是热行），
//   改成读输入张量后 AIC 每次拿的是冷行；省下的 48 KiB MTE3 抵不过这次延迟变差。
#ifndef PPFM_AIC_DIRECT_INPUTS
#define PPFM_AIC_DIRECT_INPUTS 0
#endif
// （**默认 0，实测更慢**）：**只让 AIC 直读 `w`，保留 `k` 的 staging**。
// 动机：`w` 只有 AIC 用（`mm1/mm3` 的 A 操作数），AIV 侧只是"读了再写一遍"——
// 每 chunk 每子核白搬 32 KiB（读 16 KiB + 写 16 KiB）；`k` 则不能省（AIV 要拿它算 `left`）。
// 整体 `PPFM_AIC_DIRECT_INPUTS` 当年实测 +2~3%（归因：k/w 都变成 AIC 的冷行）⇒ 这里只把 w
// 拆出来，
// k 仍由 AIV staging 保持 L2 热行。仅对**满 chunk** 生效（尾块需要 staging 的零填充）。
// （已否决） 实测（950/247，TAG=r14）：T=1024 150.53→153.33（+1.9%）、T=4096
// 459.12→476.97（+3.9%）、
//    模型 case 2747.58→2805.84（+2.1%）⇒ **即使只去掉 w 的 staging 也变慢**，
//    说明"AIV 写出 staging"对 AIC 的读就是**预热**：省下的 32 KiB 搬运抵不过 AIC
// 侧冷行延迟。
//    ⇒ **staging 不要动**（这条边看起来"白搬"，实际是 L2 行为的一部分）。
#ifndef PPFM_AIC_DIRECT_W
#define PPFM_AIC_DIRECT_W 0
#endif
// 早期怀疑"C 的跨核可见性"时加的 4 处"过渡探读"（各读 8 个 fp32 并配一次 PIPE_ALL）。
// 可见性结论已明确（见 validation /：hBf_ 写坏、T1 双写等），这些探读是纯开销。
// **默认 0 = 删除**（依据 validation ，2026-09-29，950/247）：
//   位级 `BIT_IDENTICAL`、L2/L4/L3 全绿、**进程级竞态探针 0/20 CLEAN**（原来担心的 1/6
// 竞态未回归），
//   性能 T=1024 −3.7%、T=4096 −4.2%、模型 case −3.2%（3828.7 → 3707.6 µs）。
// 置 1 可改用"保留探读"的此前实现（若哪天出现跨核可见性症状，先开这个再查）。
#ifndef PPFM_LEGACY_PROBE_READS
// 默认 0 = 删除（依据 validation ；950 上实测 −3.2~4.2%）。
// 实验：把它们在 A2 上单独留回来试过——加上之后"A2 h 常驻 vs 基线"能到 5/5
// 位级一致，
// 但同一 kernel 连跑 6 次 dump 之间仍有 1~2/5 文件抖动 ⇒
// 探读只是**减小**那条窗口、没有关掉它。
// 因此仍保持 0（A2 h 常驻继续当前不启用）。
#define PPFM_LEGACY_PROBE_READS 0
#endif
#if !PPFM_TILE_MMAD
#undef PPFM_AIC_DIRECT_INPUTS
#define PPFM_AIC_DIRECT_INPUTS 0
#endif

// ---------------- AIV 侧跨流水同步：事件对----------------
// 热路径原来用 PipeBarrier<PIPE_ALL>
// 把所有流水排空；跨流水的依赖其实只需要"生产者→消费者"
// 的事件对。PPFM_AIV_EVENTS=0 时退化成与原来等价的 PIPE_ALL（用于 A/B 与快速回退）。
// 事件 ID 分工（每个 SET 都有同 ID 的 WAIT，成对消耗；AIC
// 侧用的是它自己的一套，互不影响）：
//   ID0 MTE2->V   ID1 V->MTE3   ID2 MTE3->MTE2   ID3 MTE3->V
//   ID4 V->MTE2   ID5 MTE2->MTE3   ID6 V->S      ID7 S->V
// 注意： 经验（见 docs/pipeline_parallel_plan.md §P1a）：**必须用事件对**，
// 用 PipeBarrier<PIPE_X> 代替 PIPE_ALL 会丢跨流水依赖（曾 6/6 全错）。
#ifndef PPFM_AIV_EVENTS
#define PPFM_AIV_EVENTS 1
#endif
#if PPFM_AIV_EVENTS
#define AIV_SET_MTE2_V()     SetFlag<HardEvent::MTE2_V>(EVENT_ID0)
#define AIV_WAIT_MTE2_V()    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0)
#define AIV_SET_V_MTE3()     SetFlag<HardEvent::V_MTE3>(EVENT_ID1)
#define AIV_WAIT_V_MTE3()    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1)
#define AIV_SET_MTE3_MTE2()  SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2)
#define AIV_WAIT_MTE3_MTE2() WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2)
#define AIV_SET_MTE3_V()     SetFlag<HardEvent::MTE3_V>(EVENT_ID3)
#define AIV_WAIT_MTE3_V()    WaitFlag<HardEvent::MTE3_V>(EVENT_ID3)
#define AIV_SET_V_MTE2()     SetFlag<HardEvent::V_MTE2>(EVENT_ID4)
#define AIV_WAIT_V_MTE2()    WaitFlag<HardEvent::V_MTE2>(EVENT_ID4)
#define AIV_SET_MTE2_MTE3()  SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID5)
#define AIV_WAIT_MTE2_MTE3() WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID5)
#define AIV_SET_V_S()        SetFlag<HardEvent::V_S>(EVENT_ID6)
#define AIV_WAIT_V_S()       WaitFlag<HardEvent::V_S>(EVENT_ID6)
#define AIV_SET_S_V()        SetFlag<HardEvent::S_V>(EVENT_ID7)
#define AIV_WAIT_S_V()       WaitFlag<HardEvent::S_V>(EVENT_ID7)
#else
// 回退：SET 侧放一次全栅栏，WAIT 侧空操作 —— 与改造前的语义一致
#define AIV_SET_MTE2_V()     do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_MTE2_V()    do { } while (0)
#define AIV_SET_V_MTE3()     do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_V_MTE3()    do { } while (0)
#define AIV_SET_MTE3_MTE2()  do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_MTE3_MTE2() do { } while (0)
#define AIV_SET_MTE3_V()     do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_MTE3_V()    do { } while (0)
#define AIV_SET_V_MTE2()     do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_V_MTE2()    do { } while (0)
#define AIV_SET_MTE2_MTE3()  do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_MTE2_MTE3() do { } while (0)
#define AIV_SET_V_S()        do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_V_S()       do { } while (0)
#define AIV_SET_S_V()        do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_S_V()       do { } while (0)
#endif

// ---- C9（A2 专用实验，默认 0）：状态更新循环里补 **V -> MTE2** 的 WAR 序 ----
// 现象（离线代码审计，2026-09-30）：_vec.h 的状态更新循环（h 相位 1126-1202、m 相位 1338-1395）
//   在每轮开头用 `AIV_SET_MTE3_MTE2/WAIT` 保护"上一轮 MTE3 读完 stateBlkBf_"，
//   但**上一轮对 stateBlkF_ / extBlkF_ 的 V 读**（`Muls` / `Add` / `Sub` / `Cast`）没有被排序 ——
//   下一轮的 `DataCopy(stateBlkF_, ...)` / `DataCopy(extBlkF_, ...)`（MTE2）可能覆盖仍在被读的 UB。
//   代码注释本身写的就是"上一块（或上一相位）对 extBlkF_ 的 **V 读**要先完成"
//   （见 _vec.h 的 H_UB 分支注释），但实际补的是 MTE3_MTE2 而不是 V_MTE2。
//   ⇒ 这是一条**核内 UB 的 WAR 窗口**，与项目此前查的"跨核 GM 可见性"是**不同的机制**，
//     所以 9 组跨核排除实验没有覆盖到它。
//   * RB=32（A2 默认）时每个子核每相位 2 轮 ⇒ 每个 chunk 有 2 个这样的窗口；
//   * RB=64（PPFM_A2_RB64=1，见 C7）时只有 1 轮 ⇒ 窗口自然消失；
//   * 950 是 RB=64 ⇒ 本来就没有这个窗口（也正因如此，这条只对 A2 有意义）。
// 本开关只**插入一个事件对**（不改任何算术、不改 UB 布局）⇒ 位级不变；
// 950 上展开为空 ⇒ 950 预处理输出与 kernel .o 逐字节不变。
// **A2 默认开**（2026-10-08 上机验证：默认开 = 已验证的 allsw 产物，.o md5 d5a5ee92）。
#ifndef PPFM_A2_WAR_FIX
#define PPFM_A2_WAR_FIX (!PPFM_ARCH_IS_950)
#endif
#if PPFM_A2_WAR_FIX && !PPFM_ARCH_IS_950
#define AIV_WAR_BEFORE_STATE_MTE2() do { AIV_SET_V_MTE2(); AIV_WAIT_V_MTE2(); } while (0)
#else
#define AIV_WAR_BEFORE_STATE_MTE2() do { } while (0)
#endif

// ---- C10（A2 专用实验，默认 0）：逐行缩放改用 Brcb 广播 + 反复式 Mul ----
// 依据（A2 指令级仿真）：A2 的最大单一成本是**标量 load**（同 shape `LD_XD_XN_IMM`
//   4.2 万条 / 1041.6 µs，是 950 的 4.9~11×），而 A2 又是纯 AIV-bound（96~97%）
//   ⇒ 砍标量指令直接兑现。这些标量读主要来自 `left` / `v_new` 与状态相位的逐行缩放：
//   每 chunk 每子核各 32 次 `GetValue` + 32 次 `Muls`（K1-a 只是把标量读提前，数量不变）。
//   950 用 RegBase/VF 一趟做完，A2 没有 VF。
// A2 可用、且仓内同类算子已在用的等价写法（见 chunk_bwd_dqkwg / chunk_bwd_dv_local 的
//   vector 实现："Brcb 处理数据个数需要 8 对齐"，随后用带 BinaryRepeatParams 的反复式 Mul）：
//     Brcb(fac8_, dgF_[off], SEG / 8, {1, 8});                    // [SEG] → [SEG,8]（每行 8 份）
//     Mul(dst, dst, fac8_, 64, SEG, {1,1,0, row/8, row/8, 1});     // 每 repeat 一行；行宽 >64 拆段
//   ⇒ 32 次标量读 + 32 次 Muls 变成 1 次 Brcb + 1~2 次 Mul。乘数仍是同一个 fp32 数，
//      Brcb 只搬数据 ⇒ **位级不变**。
// 只作用于 A2（950 上与宏无关 ⇒ 预处理输出/机器码逐字节不变）。**A2 默认开**。
#ifndef PPFM_A2_FAC_BROADCAST
#define PPFM_A2_FAC_BROADCAST (!PPFM_ARCH_IS_950)
#endif
#if PPFM_A2_FAC_BROADCAST && !PPFM_ARCH_IS_950
#define PPFM_FAC8_ON 1
#else
#define PPFM_FAC8_ON 0
#endif
// C10b：FAC8 打开时，状态相位的 K1-a 批量预取分支**整体让位**给 Brcb+Mul 版本
//   （K1-a 只是把标量读提前、数量不变；Brcb+Mul 是直接取代）
//   做法：把 PPFM_KDA_ROW_PREFETCH 置 0 ⇒ 代码走 `#else` 分支，而该分支在 FAC8 打开时
//   已在 _vec.h 的两处 **A2 活跃**站点被换成 Brcb+Mul（h 相位非 H_UB、m 相位 !T2_CV）。
//   只影响 A2（FAC8 需要 !PPFM_ARCH_IS_950）⇒ 950 预处理输出与机器码不变。
#if PPFM_FAC8_ON
#undef PPFM_KDA_ROW_PREFETCH
#define PPFM_KDA_ROW_PREFETCH 0
#endif

// ---------------- AIV 侧 UB 布局（字节）----------------
// 注意： 历史结论（**已修订**）：早期按"950 MIX 下 UB 由 AIC + 两个 AIV 子核共享"的
//   假设，把"两个子核都会写"的 scratch 按 subIdx_ 切成两份（每个常量 =
// 两份的总字节数）。
//   但实测反证了这个假设：`SPLIT_M` 的 fixpipe 把 C 的 M
//   两半写进两个子核**各自 bank 的同一偏移**，而两个子核从**同一个 UB
// 偏移**读到了**不同**
//   的数据（否则 不可能位级一致）⇒ **每个 AIV 子核有独立 UB bank**（253952 B/子核，
//   见 ），按 subIdx_ 切两份是纯冗余。`PPFM_UB_SHARE=1`（默认）改为单份布局，省 ~39 KiB；
//   置 0 可改用历史两份布局。
//   仍然"按段（off）分区"共享的：kBlk/wBlk/vBlk/scr（两个子核用不同 off，互不相交）。
constexpr int32_t PPFM_SUB = 2;      // AIV 子核数（UB 共享）
constexpr int32_t PPFM_SEG = 16;     // left / v_new 的段长（行）
constexpr int32_t PPFM_RB = 32;      // 状态更新的行块（行）16->32，
                                      // 每 chunk 状态相位搬运/栅栏减半（UB +24K）
// h/m 都常驻 UB 后，状态相位按 32 行分两块已经没有意义（两块都不再搬运状态本体）
// ⇒ 并成**一块 64 行**：每 chunk 的 V 运算与同步对从 8 次降到 4
// 次（位级不变，只是把两块拼起来）。
// A2 版（实验，默认关）：A2 没有 h/m 常驻，但"8 次 → 4 次"的收益与常驻无关 ——
//   状态更新本身仍是逐行 elementwise，分块的唯一作用是"每次搬运/同步的规模"。
//   依据（A2 指令级仿真，2026-09-30）：A2 是纯 AIV-bound（96~97%），其 MTE3/MTE2/SCALARLDST
//   全面是 950 的 3.3~11×，且**搬运被切得更碎**（MTE3 指令条数 4.2×）。
//   开关 PPFM_A2_RB64=1 时 A2 也用 64 行块 ⇒ 每 chunk 状态相位的搬运/事件对减半
//   （行分配由交错块自动变成连续半区，与 950 的 H_UB 情形同构；逐行数值不变）。
// **A2 默认开**（2026-10-08 上机：单独开这一条即拿到 −8.5~12%，是四个开关里的主力）。
#ifndef PPFM_A2_RB64
#define PPFM_A2_RB64 (!PPFM_ARCH_IS_950)
#endif
#if (PPFM_M_UB && PPFM_H_UB) || (!PPFM_ARCH_IS_950 && PPFM_A2_RB64)
constexpr int32_t PPFM_SBRB = CV_K / PPFM_SUB;   // 64
#else
constexpr int32_t PPFM_SBRB = PPFM_RB;           // 32（A2 与回退路径）
#endif
// m 常驻 UB 时，mUb_ 的本地偏移（lo = rb - subIdx_*RB）必须与 epilogue 的
// 连续半区一致 ⇒ 状态更新的行块必须正好是 CV_K/PPFM_SUB。
static_assert(!PPFM_M_UB || PPFM_SBRB == CV_K / PPFM_SUB,
              "PPFM_M_UB 要求 PPFM_SBRB == CV_K/PPFM_SUB（行块 = 一个半区）");
// A2 的 PPFM_UB_SHARE=0 是历史两份布局：stateBlkF_ 的每子核偏移仍按 PPFM_RB 算，
// 与 RB=64 的容量不一致 ⇒ 直接禁止这个组合（默认 UB_SHARE=1，不受影响）。
static_assert(!(PPFM_A2_RB64 && !PPFM_UB_SHARE),
              "PPFM_A2_RB64 需要 PPFM_UB_SHARE=1（历史两份布局的子核偏移按 32 行算）");

// ---------------- 诊断开关（定位概率性 h 错）----------------
// 打开后：每个工作项把前 N 个 chunk 的 "AIV 读到的 vTmpF_[0]"（AIV 侧）与
// "AIC 写出的 vTmpF_[0]"（AIC 侧）指纹写进 hm 的 m 半边第 0 行（覆盖该行，验收时排除）。
//   lane 0..3  = AIV 读到的 vTmpF_[0]（第 c 个 chunk）
//   lane 8..11 = AIC 写出的 vTmpF_[0]（第 c 个 chunk）
// 判读：两者不等 → AIV 读到别的代（跨核可见性/flag 提前）；相等但≠期望 → AIC 的 mm1
// 输入不对。
// 注意： 当前不启用：诊断收尾会把指纹写进 hm 的 m 半边【第 0 行】（见下），
//   验收/对拍脚本不会排除该行 → 默认开启时表现为"m 只有第一行几个元素错、
//   (0,0) 恒为 0"，曾被误判成算子精度缺陷（PPFM-31/33 的 m 坏点就是这么来的）。
//   只在定位跨核可见性时临时打开，并对拍时排除 hm[..., 0, V:]。
#ifndef PPFM_DIAG
#define PPFM_DIAG 0
#endif
constexpr int32_t PPFM_DIAG_CHUNKS = 4;
// 诊断区必须落在 workspace 的**空闲段**，且不能与任何活缓冲区重叠：
//   * 原值 626688 == `WS_K_BF_1`（k_c 第二槽）⇒ 一开 PPFM_DIAG 就写坏 k，
//     现象是 h 半边变成 1e37 量级（曾被误当成"算子精度缺陷"）；
//   * 一度改到 659456，又被 m 链 fp32 化的 `WS_KW_F32`（659456..724992，64 KiB）占用。
//   本核 workspace 共 768 KiB，`WS_KW_F32` 之后的空闲段是 724992..786432 ⇒ 诊断区放末尾。
constexpr int64_t WS_DIAG = 778240;               // 每核 4 KiB（AIC 写，AIV epilogue 搬到 hm）

// PPFM_NSLOT = 1（默认）时这些都是**单份**；= PPFM_SUB 时回退历史的两份布局。
constexpr int32_t UB_ROW0_BF = 0;                                  // [K] bf16
constexpr int32_t UB_ROW1_BF = UB_ROW0_BF + PPFM_NSLOT * CV_K * 2;
constexpr int32_t UB_ROW2_BF = UB_ROW1_BF + PPFM_NSLOT * CV_K * 2;
constexpr int32_t UB_ROW0_F32 = UB_ROW2_BF + PPFM_NSLOT * CV_K * 2;
constexpr int32_t UB_ROW1_F32 = UB_ROW0_F32 + PPFM_NSLOT * CV_K * 4;
constexpr int32_t UB_ROW2_F32 = UB_ROW1_F32 + PPFM_NSLOT * CV_K * 4;
constexpr int32_t UB_DG = UB_ROW2_F32 + PPFM_NSLOT * CV_K * 4;     // [BT] fp32
constexpr int32_t UB_DECAY = UB_DG + PPFM_NSLOT * CV_BT * 4;       // [K] fp32
constexpr int32_t UB_EXP = UB_DECAY + PPFM_NSLOT * CV_K * 4;       // [8] fp32
constexpr int32_t UB_DECAY_PREV = UB_EXP + PPFM_NSLOT * CV_LANES * 4;
constexpr int32_t UB_GBLK = UB_DECAY_PREV + PPFM_NSLOT * CV_K * 4;  // [BT] fp32
constexpr int32_t UB_STATE_F = UB_GBLK + PPFM_NSLOT * CV_BT * 4;   // [RB,K] fp32
// stateBlkF_ / extBlkF_ 的行容量（按用途取大者，保证 950 与 A2 默认布局逐字节不变）：
//   * 950（PPFM_H_UB=1）：状态相位本体在 hUb_/dHUb，这两块只做小暂存 ⇒ 仍是 32 行；
//   * A2（无 H_UB）：状态更新用 stateBlkF_ 载入状态本体、用 extBlkF_ 暂存 dH
//     ⇒ 容量必须 ≥ PPFM_SBRB 行（PPFM_A2_RB64=1 时为 64 行）。
// [FIX] 只要 **m 状态不在 UB**（PPFM_M_UB=0；A2/A3 就是这种），stateBlkF_/extBlkF_
//   就必须放得下状态更新的整块（PPFM_SBRB 行），与 H_UB 无关。
//   原式按 H_UB 判断 ⇒ A2 打开 H_UB 时只给 32 行、而 m 相位仍按 64 行写
//   ⇒ UB 越界 16 KiB（多头用例整块错）。改成按 M_UB 判断后：
//   A2 默认(H_UB=0,M_UB=0) 与 950(H_UB=1,M_UB=1) 的布局都与原式完全一致。
constexpr int32_t UB_STATE_ROWS =
    PPFM_M_UB ? PPFM_RB : ((PPFM_SBRB > PPFM_RB) ? PPFM_SBRB : PPFM_RB);
constexpr int32_t UB_EXT_ROWS =
    PPFM_DH_CV ? (2 * PPFM_SEG)
               : ((PPFM_SBRB > 2 * PPFM_SEG) ? PPFM_SBRB : (2 * PPFM_SEG));
constexpr int32_t UB_EXT_F = UB_STATE_F + PPFM_NSLOT * UB_STATE_ROWS * CV_V * 4;
constexpr int32_t UB_STATE_BF = UB_EXT_F + PPFM_NSLOT * UB_EXT_ROWS * CV_V * 4;
// extBlkF_ 的可用 fp32 元素数：A1 的整块宽度由它推导（见 _vec.h 的 rowsPerCopy_），
// 保证"按 UB 实际容量分块"而不是写死 2*SEG。
constexpr int32_t UB_EXT_F_ELEMS = PPFM_NSLOT * UB_EXT_ROWS * CV_V;
// 下面这些按"段"分区。起改成**子核本地段**寻址（`lo`，每个子核只有自己那 32 行）
// ⇒ 尺寸砍半（k/w/v/scr 合计省 48 KiB）。依据：每个 AIV 子核有独立 UB bank（/）。
constexpr int32_t PPFM_SEGROWS = CV_BT / PPFM_SUB;                             // 32
// stateBlkBf_ 的尺寸按 PPFM_SBRB（起 950 上是 64 行 ⇒ 16 KiB）
constexpr int32_t UB_KBLK_BF = UB_STATE_BF + PPFM_NSLOT * PPFM_SBRB * CV_V * 2;
// 容量自检：A2（无 H_UB）的状态相位必须放得下 PPFM_SBRB 行；950 走 hUb_/dHUb 不受此限。
static_assert(PPFM_H_UB || (UB_EXT_F - UB_STATE_F) / 4 >= PPFM_SBRB * CV_V,
              "stateBlkF_ 容量不足：A2 的状态更新需 >= PPFM_SBRB 行 fp32");
static_assert(PPFM_H_UB || (UB_STATE_BF - UB_EXT_F) / 4 >= PPFM_SBRB * CV_V,
              "extBlkF_ 容量不足：A2 的 dH 暂存需 >= PPFM_SBRB 行 fp32");
static_assert((UB_KBLK_BF - UB_STATE_BF) / 2 >= PPFM_SBRB * CV_V,
              "stateBlkBf_ 容量不足：需 >= PPFM_SBRB 行 bf16");
constexpr int32_t UB_WBLK_BF = UB_KBLK_BF + PPFM_SEGROWS * CV_K * 2;          // [SEGROWS,K] bf16
constexpr int32_t UB_VBLK_BF = UB_WBLK_BF + PPFM_SEGROWS * CV_K * 2;          // [SEGROWS,V] bf16
constexpr int32_t UB_SCR_F = UB_VBLK_BF + PPFM_SEGROWS * CV_V * 2;            // [SEGROWS,K] fp32
constexpr int32_t UB_SCR_BF = UB_SCR_F + PPFM_SEGROWS * CV_K * 4;             // [SEGROWS,K] bf16
constexpr int32_t UB_DBG = UB_SCR_BF + PPFM_SEGROWS * CV_K * 2;               // 诊断槽 ×2
// ---- dH 的 CV 落点（L0C->UB，2 槽 ping-pong，见 validation /）----
// 每个 AIV 子核有**自己的 UB bank**（253952 B/子核），SPLIT_M 只把 C 的 M 两半分别写进
// 两个 bank 的**同一偏移** ⇒ 单槽按「一个子核那一半的行数」算：CV_K/2 = 64 行 × CV_V ×
// 4B
// = 32768 B；2 槽 = 65536 B。188096 + 65536 = 253632 ≤ 253952（余 320 B）。
constexpr int32_t UB_DH_CV = UB_DBG + PPFM_SUB * 16 * 4;
constexpr int32_t DH_CV_ROWS = CV_K / PPFM_SUB;                              // 64
constexpr int32_t UB_DH_CV_SLOT = DH_CV_ROWS * CV_V * 4;                     // 32768 B
constexpr int32_t UB_DH_CV_ELEM = UB_DH_CV / 4;
constexpr int32_t UB_DH_CV_SLOT_ELEM = UB_DH_CV_SLOT / 4;
// ---- /dH 与 T2 的 CV 落点（各 **1 槽**）----
// 单槽就够，不需要 ping-pong：AIC 写 dH(c)/T2(c) 必然在 `AicWaitFromAiv(kFlagVNew(c))` 之后，
// 而 AIV 早在迭代 c 开头就把 c-1 那一代消费完了 ⇒ 天然串行。显式 credit 保留做双保险
// （计数：AIV set = 1(prime) + nt；AIC wait = nt + 1(尾 drain) ⇒ 逐链平衡）。
constexpr int32_t DH_CV_SLOTS = 1;
constexpr int32_t UB_T2_CV = UB_DH_CV + DH_CV_SLOTS * UB_DH_CV_SLOT;
constexpr int32_t UB_T2_CV_ELEM = UB_T2_CV / 4;
#if PPFM_DH_CV
constexpr int32_t UB_CV_END = UB_T2_CV + (PPFM_T2_CV ? UB_DH_CV_SLOT : 0);
#else
constexpr int32_t UB_CV_END = UB_DH_CV;
#endif
// ---- h 状态常驻 UB（每个子核只放自己那 64 行）----
// 注意： V 运算（Muls/Add/Cast/Duplicate）对 UB 偏移要求 32B 对齐（历史上的 error 340）⇒
// 显式对齐。
constexpr int32_t H_UB_ROWS = CV_K / PPFM_SUB;                        // 64
constexpr int32_t UB_H_UB = ((UB_CV_END + 31) / 32) * 32;
constexpr int32_t UB_H_UB_ELEM = UB_H_UB / 4;
#if PPFM_H_UB
#if PPFM_M_UB
// m 也常驻 UB（32 KiB），紧跟在 h 区之后
constexpr int32_t UB_M_UB = UB_H_UB + H_UB_ROWS * CV_V * 4;
constexpr int32_t UB_M_UB_ELEM = UB_M_UB / 4;
constexpr int32_t PPFM_VEC_UB_BYTES = UB_M_UB + H_UB_ROWS * CV_V * 4;  // h + m 各 32768 B
#else
constexpr int32_t UB_M_UB_ELEM = 0;
constexpr int32_t PPFM_VEC_UB_BYTES = UB_H_UB + H_UB_ROWS * CV_V * 4;  // 只有 h（A2 的 h 常驻）
#endif
#else
constexpr int32_t UB_M_UB_ELEM = 0;
constexpr int32_t PPFM_VEC_UB_BYTES = UB_CV_END;
#endif

// ---- 编译期自检（§7.7：常量集中 + 空间上限配 static_assert）----
// 950 的 AIV UB 上限 256 KiB；A2/910B 为 192 KiB。
static_assert(PPFM_VEC_UB_BYTES <= PPFM_UB_CAP_BYTES,
              "AIV UB 用量超上限：950=256KiB / A2=192KiB，请按 UB_* 布局重算");
// ---- C10 的 factor 广播暂存（[FAC8_ROWS, 8] fp32）----
// 行数取两种用法的较大者：`left`/`v_new` 用 PPFM_SEGROWS 行；状态相位的 decay 缩放用 PPFM_SBRB 行。
// 追加在当前用量的**末尾**并对齐 32B ⇒ 前面所有偏移一个都不变；PPFM_FAC8_ON=0（默认）时
// 该区不占空间、总用量与现状完全相同（两个平台的 PPFM_VEC_UB_BYTES 本来就 32B 对齐）。
constexpr int32_t FAC8_ROWS = (PPFM_SEGROWS > PPFM_SBRB) ? PPFM_SEGROWS : PPFM_SBRB;
constexpr int32_t UB_FAC8 = ((PPFM_VEC_UB_BYTES + 31) / 32) * 32;
constexpr int32_t UB_FAC8_ELEM = UB_FAC8 / 4;
constexpr int32_t FAC8_BYTES = PPFM_FAC8_ON ? (FAC8_ROWS * 8 * 4) : 0;
constexpr int32_t PPFM_VEC_UB_BYTES_TOTAL = UB_FAC8 + FAC8_BYTES;
static_assert(PPFM_VEC_UB_BYTES_TOTAL <= (PPFM_ARCH_IS_950 ? 256 * 1024 : 192 * 1024),
              "AIV UB 用量（含 C10 的 factor 暂存）超上限");
// 状态常驻区：每个 AIV 子核 64 行 x cb_（cb_ 最大 CV_V）fp32 x h/m 两份
static_assert(H_UB_ROWS * CV_V * 4 * 2 <= 96 * 1024,
              "h/m 常驻 UB 超过预留的 96 KiB");
// 段缓冲与行缓冲的 32B 对齐前提（V 运算对 UB 偏移要求 32B 对齐）
// NOTE: UB_M_UB is only defined when BOTH h and m are UB-resident
// (A2/A3 take the PPFM_H_UB==0 branch, where UB_M_UB does not exist).
static_assert((UB_H_UB % 32) == 0 && (UB_DH_CV % 32) == 0,
              "critical UB slot offsets must be 32B aligned");
#if PPFM_H_UB && PPFM_M_UB
static_assert((UB_M_UB % 32) == 0,
              "m-resident UB slot offset must be 32B aligned");
#endif
static_assert(PPFM_SUB == 1 || PPFM_SUB == 2, "AIV 子核数只支持 1 或 2");
static_assert(CV_BT % PPFM_SEGROWS == 0 && (CV_BT / PPFM_SEGROWS) % PPFM_SUB == 0,
              "staging 段划分必须整除：CV_BT 能被 SEGROWS*PPFM_SUB 整除");

// 上面都是**字节**偏移，取 Tensor 时要按元素大小换算（bf16 → /2，fp32 → /4）
constexpr int32_t UB_ROW0_BF_ELEM = UB_ROW0_BF / 2;
constexpr int32_t UB_ROW1_BF_ELEM = UB_ROW1_BF / 2;
constexpr int32_t UB_ROW2_BF_ELEM = UB_ROW2_BF / 2;
constexpr int32_t UB_ROW0_F_ELEM = UB_ROW0_F32 / 4;
constexpr int32_t UB_ROW1_F_ELEM = UB_ROW1_F32 / 4;
constexpr int32_t UB_ROW2_F_ELEM = UB_ROW2_F32 / 4;
constexpr int32_t UB_DG_ELEM = UB_DG / 4;
constexpr int32_t UB_DECAY_ELEM = UB_DECAY / 4;
constexpr int32_t UB_EXP_ELEM = UB_EXP / 4;
constexpr int32_t UB_DECAY_PREV_ELEM = UB_DECAY_PREV / 4;
constexpr int32_t UB_GBLK_ELEM = UB_GBLK / 4;
constexpr int32_t UB_STATE_F_ELEM = UB_STATE_F / 4;
constexpr int32_t UB_EXT_F_ELEM = UB_EXT_F / 4;
constexpr int32_t UB_STATE_BF_ELEM = UB_STATE_BF / 2;
constexpr int32_t UB_KBLK_BF_ELEM = UB_KBLK_BF / 2;
constexpr int32_t UB_WBLK_BF_ELEM = UB_WBLK_BF / 2;
constexpr int32_t UB_VBLK_BF_ELEM = UB_VBLK_BF / 2;
constexpr int32_t UB_SCR_F_ELEM = UB_SCR_F / 4;
constexpr int32_t UB_SCR_BF_ELEM = UB_SCR_BF / 2;

struct PpFwdCtx {
    GM_ADDR k = nullptr;
    GM_ADDR w = nullptr;
    GM_ADDR u = nullptr;
    GM_ADDR g = nullptr;
    GM_ADDR gk = nullptr;
    GM_ADDR v = nullptr;
    GM_ADDR cu = nullptr;
    GM_ADDR hm = nullptr;
    GM_ADDR ws = nullptr;
    const PreProcessFwdKernelMergedTilingData *tiling = nullptr;
};

// =====================================================================================
// AIV：elementwise
// =====================================================================================

// =====================================================================================
// 任务号 → (n, hv, 列片)。整宽链 pieces=1；余数链按 hybridS 切列。
// =====================================================================================
// 任务换算结果（§4.4 ⑤）：Stage/Process 只消费本结构，不再各自重算列窗
struct PpFwdChunkInfo {
    int64_t n;        // 序列号（varlen 打包里的第几条）
    int64_t hv;       // value head
    int32_t piece;    // 列片序号（混合调度）
    int32_t pieces;   // 列片总数（1 = 整宽）
    int32_t cb;       // 本任务的列宽 = CV_V / pieces
    int32_t colBase;  // 本任务列块起始列 = piece * cb
    int32_t colEnd;   // 本任务列块结束列（= colBase + cb，便于边界判断）
};

__aicore__ inline PpFwdChunkInfo GetChunkInfo(const PreProcessFwdKernelMergedTilingData *t,
                                               int32_t colSplit, int64_t task)
{
    PpFwdChunkInfo r{0, 0, 0, (colSplit > 0) ? colSplit : 1, 0, 0, 0};
    if (t->hybridS > 1 && task >= t->hybridBase) {
        const int64_t j = task - t->hybridBase;
        r.pieces = static_cast<int32_t>(t->hybridS);
        r.piece = static_cast<int32_t>(j % r.pieces);
        const int64_t chain = t->hybridBase + j / r.pieces;
        r.hv = chain % t->Hv;
        r.n = chain / t->Hv;
    } else {
        const int32_t s = (colSplit > 0) ? colSplit : 1;
        r.pieces = s;
        r.piece = static_cast<int32_t>(task % s);
        const int64_t chain = task / s;
        r.hv = chain % t->Hv;
        r.n = chain / t->Hv;
    }
    r.cb = static_cast<int32_t>(CV_V) / r.pieces;
    r.colBase = r.piece * r.cb;
    r.colEnd = r.colBase + r.cb;
    return r;
}

} // namespace GDN

#endif  // PREF_PROCESS_FWD_KERNEL_MERGED_COMMON_H
