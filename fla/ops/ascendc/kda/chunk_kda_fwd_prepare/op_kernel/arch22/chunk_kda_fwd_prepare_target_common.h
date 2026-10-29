/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */
#ifndef ARCH22_CHUNK_KDA_FWD_PREPARE_TARGET_COMMON_H
#define ARCH22_CHUNK_KDA_FWD_PREPARE_TARGET_COMMON_H

#include "../chunk_kda_fwd_prepare_utils.h"
#include "chunk_kda_fwd_prepare_target_config.h"

namespace KdaPrepare::Arch22Target {

// These resources belong only to the target variant. The upstream fallback
// keeps its FP32 RHS/Akk layout and its existing event ownership.

namespace Workspace {
// 两套架构共用前 105 KiB：33 KiB context + 72 KiB stage payload。
constexpr uint32_t kQHat = 0x00000;
constexpr uint32_t kKHat = 0x04000;
constexpr uint32_t kBetaEff = 0x08000;
constexpr uint32_t kPayload = 0x08400;
constexpr uint32_t kSlotStride = 0x1A400;
constexpr uint32_t kArch35SlotStride = kSlotStride;

// Arch22 的 Cube 结果先经 GM 交给 Vector。该区只由 AIC 写：
// C2 放四段连续 raw score，V3 消费后 C4 复用前 4 KiB 放 T。
constexpr uint32_t kArch22CubeRelay = 0x1A400;
constexpr uint32_t kArch22RawScoreBytes = 0x5000;
constexpr uint32_t kArch22TRelay = kArch22CubeRelay;
// Arch22：V6→C7 的 RHS（K_beta_g/V_beta）使用 slot 尾部独立 32 KiB，
// 不再与 V1 payload 复用地址，避免 RHS 与 score payload 的地址重叠写。
constexpr uint32_t kArch22RhsBase = kArch22CubeRelay + kArch22RawScoreBytes;
constexpr uint32_t kArch22RhsMatrixBytes = 0x4000; // [64,128] * BF16
constexpr uint32_t kArch22RhsKBetaG = kArch22RhsBase;
constexpr uint32_t kArch22RhsVBeta = kArch22RhsBase + kArch22RhsMatrixBytes;
constexpr uint32_t kArch22RhsBytes = 2 * kArch22RhsMatrixBytes;
// AIC 为下一 group 预算 q/k 行平方和，V0 只消费 Gram 矩阵的对角线。
constexpr uint32_t kArch22NormSum = kArch22RhsBase + kArch22RhsBytes;
// 每 head 16 KiB：[q 的 2 个 32x32 Gram 块][k 的 2 个]，行主序 FP32，
// 块内元素 (i,j) 在 i*32+j；对角线 i*33 就是第 i 行的 |x|^2。
constexpr uint32_t kArch22NormSumBytes = 0x4000;
// AIC 叶逆暂存：L00/L11 保留原 64 列 pitch，避免 UB 侧搬位；
// Y 和 I 使用紧凑 32x32 FP32。
constexpr uint32_t kArch22InvMatrixBytes = 0x1000;
constexpr uint32_t kArch22InvAux = kArch22NormSum + kArch22NormSumBytes;
constexpr uint32_t kArch22InvL00 = 0x0000; // [32,64] FP32
constexpr uint32_t kArch22InvL11 = 0x2000; // [32,64] FP32, 使用列 32..63
constexpr uint32_t kArch22InvY00 = 0x4000;
constexpr uint32_t kArch22InvY11 = 0x5000;
constexpr uint32_t kArch22InvIdentity = 0x6000;
constexpr uint32_t kArch22InvAuxBytes = 0x7000;
constexpr uint32_t kArch22BaseSlotStride = kArch22InvAux;
constexpr uint32_t kArch22TargetSlotStride = kArch22InvAux + kArch22InvAuxBytes;
// Compatibility name used by the original arch22 implementation.
constexpr uint32_t kArch22SlotStride = kArch22BaseSlotStride;

// payload 在不同 Stage 原址换义，不在 UB/L1 内搬位。
constexpr uint32_t kX0 = 0x0000;
constexpr uint32_t kNegX1 = 0x2000;
constexpr uint32_t kB = 0x3000;
constexpr uint32_t kAkk = 0x5800;
constexpr uint32_t kKBetaG = 0x7800;
constexpr uint32_t kVBeta = 0xB800;

constexpr uint32_t kArch22SlotCount = 4;
constexpr uint32_t kArch22WorkgroupStride = kArch22SlotCount * kArch22SlotStride;
constexpr uint32_t kArch35SlotCount = 4;
constexpr uint32_t kArch35WorkgroupStride = kArch35SlotCount * kSlotStride;
} // namespace Workspace

namespace L1 {
constexpr uint32_t kCapacity = 0x80000;
constexpr uint32_t kHeadLaneBytes = 0x12000;
constexpr uint32_t kHeadLane[Shape::kHeadsPerGroup] = {0x00000, 0x12000, 0x24000, 0x36000};
constexpr uint32_t kX0 = 0x48000;
constexpr uint32_t kNegX1 = 0x4C000;
constexpr uint32_t kT = 0x50000;
constexpr uint32_t kAkk = 0x54000;
constexpr uint32_t kQuadrantStride = 0x1000;
constexpr uint32_t kAkkStride = 0x2000;
// BF16 NZ 为 [N1,M1,M0,N0]；q10=(row 32,col 0) 的元素偏移。
constexpr uint32_t kAkkQ10Elements = 32 * 16;
constexpr uint32_t kPeak = 0x5C000;
} // namespace L1

namespace Arch22Ub {
constexpr uint32_t kHardwareBytes = 0x30000; // 192 KiB
// V1/V6 的 16 行 tile 需要两个独立 FP32 staging，覆盖全部 192 KiB UB。
constexpr uint32_t kUsableBytes = 0x30000;
constexpr uint32_t kPrivateBytes = 0x12000;
constexpr uint32_t kPrivateBase[2] = {0x00000, 0x1C000};
constexpr uint32_t kSharedBase = 0x12000;
constexpr uint32_t kSharedG = kSharedBase;
constexpr uint32_t kSharedScratch = kSharedBase + 0x8000;
// 第二个 FP32 staging 使用尾部 8 KiB，与 private/shared 活跃区互不重叠。
constexpr uint32_t kScratchTail = 0x2E000;

constexpr uint32_t kQ = 0x0000;
constexpr uint32_t kK = 0x4000;
constexpr uint32_t kGateOrKMinus = 0x8000;
constexpr uint32_t kKMinus[4] = {0x8000, 0x9000, 0xB000, 0xE000};
// A2/A3 先把 sequence-major beta 按 32 Byte 行距搬入，再 Gather 成
// 连续标量；head-major beta 直接写入 kBetaRaw。
constexpr uint32_t kBetaRawStrided = 0x10000;
constexpr uint32_t kBetaRaw = 0x10800;
constexpr uint32_t kBetaEff = 0x10A00;
constexpr uint32_t kDtBias = 0x10C00;
constexpr uint32_t kALog = 0x10E00;
constexpr uint32_t kQRstd = 0x10F00;
constexpr uint32_t kKRstd = 0x11000;
constexpr uint32_t kGLast = 0x11200;
constexpr uint32_t kBetaGatherOffsets = kSharedScratch + 0x1F00;
// sequence-major beta 的 64 个 gather 偏移（64 x uint32 = 256 B）放在
// 第一个 private 槽的尾部，释放共享 scratch 的 0x1F00 限制，
// 使 V1/V6 的 FP32 tile 可以从 7 行提升到 8 行。
constexpr uint32_t kBetaGatherOffsetsPrivate = 0x11300;
// band 之间的参考行比值 exp(g[m_s]-g[m_{s-1}])，3 个 128-vector（1.5 KiB）。
// 0x11400..0x12000 在 private 槽里未被其它状态使用。
constexpr uint32_t kBandDelta = 0x11400;

constexpr uint32_t kV3Aqk = 0x0000;
constexpr uint32_t kV3Lkk = 0x2000;
constexpr uint32_t kV3Leaf0 = 0x6000;
constexpr uint32_t kV3Leaf1 = 0x7000;
constexpr uint32_t kV3B = 0x8000;
constexpr uint32_t kV3X0 = 0x9000;
constexpr uint32_t kV3X1 = 0xA000;
constexpr uint32_t kV3NegX1 = 0xB000;
constexpr uint32_t kV3CompactRaw = 0xD000;
constexpr uint32_t kV3AkkPack = 0xE000;
// V3 读取 compact raw 时，betaEff 放在已经结束 G 生命周期的共享区。
// 不能复用 kBetaEff=0x10200，它位于 compact raw [0xD000,0x12000) 内。
constexpr uint32_t kV3BetaEff = kSharedBase;

constexpr uint32_t kV6Qg = 0x0000;
constexpr uint32_t kV6Kg = 0x4000;
constexpr uint32_t kV6VBeta = 0x8000;
constexpr uint32_t kV6KBetaG = 0xC000;
// V6 正序消费 FP32 G 后，将 BF16 qgScaled 压缩写入同一共享区低 16 KiB。
constexpr uint32_t kV6QgScaled = kSharedG;
} // namespace Arch22Ub

static_assert(Workspace::kArch22TargetSlotStride == Arch22TargetConfig::kSlotBytes);
static_assert(Workspace::kArch22SlotCount == Arch22TargetConfig::kSlotCount);
static_assert(L1::kPeak <= L1::kCapacity);
static_assert(Arch22Ub::kUsableBytes <= Arch22Ub::kHardwareBytes);

__aicore__ inline uint32_t TargetPartitionCount(const PrepareRuntimeTiling &tiling)
{
    return CeilDiv(tiling.valueHeadNum, tiling.headsPerPartition);
}

__aicore__ inline uint32_t TargetTotalWorkItems(const PrepareRuntimeTiling &tiling)
{
    return tiling.batch * tiling.totalChunks * TargetPartitionCount(tiling);
}

__aicore__ inline void TargetDecodeWorkItem(const PrepareRuntimeTiling &tiling, uint32_t workItem, uint32_t &chunk,
                                            uint32_t &headPartition)
{
    const uint32_t partitions = TargetPartitionCount(tiling);
    chunk = workItem / partitions;
    headPartition = workItem % partitions;
}

__aicore__ inline void TargetHeadRange(const PrepareRuntimeTiling &tiling, uint32_t headPartition, uint32_t &headBegin,
                                       uint32_t &headEnd)
{
    headBegin = headPartition * tiling.headsPerPartition;
    headEnd = headBegin + tiling.headsPerPartition;
    if (headEnd > tiling.valueHeadNum) {
        headEnd = tiling.valueHeadNum;
    }
}
} // namespace KdaPrepare::Arch22Target
#endif // ARCH22_CHUNK_KDA_FWD_PREPARE_TARGET_COMMON_H
