/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_POLICY_H
#define CHUNK_KDA_FWD_PREPARE_POLICY_H

#include <cstdint>

namespace KdaPrepare {

namespace Shape {
constexpr uint32_t kChunkRows = 64;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kValueDim = 128;
constexpr uint32_t kSubChunkRows = 16;
constexpr uint32_t kSubChunkCount = 4;
constexpr uint32_t kHeadsPerGroup = 4;
constexpr uint32_t kAivPerAic = 2;
constexpr uint32_t kBf16MatrixBytes = 0x4000;    // [64,128] * BF16
constexpr uint32_t kGateMatrixBytes = 0x8000;    // [64,128] * FP32
constexpr uint32_t kScoreMatrixBytes = 0x4000;   // [64,64] * FP32
constexpr uint32_t kQuadrantFp32Bytes = 0x1000;  // [32,32] * FP32
constexpr uint32_t kQuadrantBf16Bytes = 0x0800;  // [32,32] * BF16
constexpr uint32_t kRstdBytes = 0x0100;          // [64] * FP32
constexpr uint32_t kPrefixRows[kSubChunkCount] = {16, 32, 48, 64};
constexpr uint32_t kKMinusBytes[kSubChunkCount] = {
    0x1000, 0x2000, 0x3000, 0x4000};
constexpr uint32_t kScorePayloadBytes = 0x12000; // 16K Q+ + 16K K+ + 40K K-
} // namespace Shape

namespace Workspace {
// 每个 slot 的 context 固定在前 33 KiB，72 KiB stage payload 固定在末尾。
constexpr uint32_t kQHat = 0x00000;
constexpr uint32_t kKHat = 0x04000;
constexpr uint32_t kBetaEff = 0x08000;
constexpr uint32_t kPayload = 0x08400;
constexpr uint32_t kSlotStride = 0x1A400;

// payload 在不同 Stage 原址换义，不在 UB/L1 内搬位。
// C2 按 sub-chunk 顺序写 [rawAqk_s; rawAkk_s]，四段合计 20 KiB。
constexpr uint32_t kRawScore = 0x0000;
constexpr uint32_t kX0 = 0x0000;
constexpr uint32_t kNegX1 = 0x2000;
constexpr uint32_t kB = 0x3000;
// C4 在这里暂存 32x32 FP32 NZ T；Arch22/Arch35 都经 GM relay 回到 L1。
constexpr uint32_t kTRelay = 0x4000;
constexpr uint32_t kAkk = 0x5800;
constexpr uint32_t kKBetaG = 0x7800;
constexpr uint32_t kVBeta = 0xB800;

constexpr uint32_t kArch22SlotCount = 4;
constexpr uint32_t kArch22WorkgroupStride = kArch22SlotCount * kSlotStride;
constexpr uint32_t kArch35SlotCount = 4;
constexpr uint32_t kArch35WorkgroupStride = kArch35SlotCount * kSlotStride;
} // namespace Workspace

namespace ScorePayload {
constexpr uint32_t kQPlus = 0x0000;
constexpr uint32_t kKPlus = 0x4000;
constexpr uint32_t kKMinus[Shape::kSubChunkCount] = {
    0x8000, 0x9000, 0xB000, 0xE000};
} // namespace ScorePayload

namespace L1 {
constexpr uint32_t kCapacity = 0x80000;
constexpr uint32_t kHeadLaneBytes = 0x12000;
constexpr uint32_t kHeadLane[Shape::kHeadsPerGroup] = {
    0x00000, 0x12000, 0x24000, 0x36000};
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

namespace Arch35Ub {
constexpr uint32_t kCapacity = 0x3E000; // 248 KiB
constexpr uint32_t kComputeSlotBytes = 0x1C000;
constexpr uint32_t kStateBytes = 0x03000;
constexpr uint32_t kComputeSlotBase[2] = {0x00000, 0x1C000};
constexpr uint32_t kStateBase[2] = {0x38000, 0x3B000};

// V0/V1 的主计算区。
constexpr uint32_t kQ = 0x0000;
constexpr uint32_t kK = 0x4000;
constexpr uint32_t kG = 0x8000;
constexpr uint32_t kGateInput = 0x10000;
constexpr uint32_t kV0Work = 0x18000;
// V1 先生成四段 Kminus，再原位把 Q/K 改写为 Qplus/Kplus。
// 最终布局为 Qplus 16 KiB + Kplus 16 KiB + G 32 KiB + Kminus 40 KiB。
constexpr uint32_t kKMinus = 0x10000;

// V3 主计算区。
constexpr uint32_t kRawScore = 0x0000;
constexpr uint32_t kAqk = 0x5000;
constexpr uint32_t kLkk = 0x9000;
constexpr uint32_t kB = 0xD000;
constexpr uint32_t kX0 = 0xE000;
constexpr uint32_t kX1 = 0xF000;
constexpr uint32_t kNegX1 = 0x10000;
constexpr uint32_t kAkkPack = 0x11000;

// V6 主计算区。
constexpr uint32_t kQg = 0x0000;
constexpr uint32_t kKg = 0x4000;
constexpr uint32_t kVBeta = 0x8000;
constexpr uint32_t kGForPost = 0xC000;
constexpr uint32_t kKBetaG = 0x14000;
constexpr uint32_t kQgScaled = 0x18000;

// 每个 local head 的向量状态与临时区。
// sequence-major beta 按每行一个 32 Byte data block 搬入；head-major
// beta 仍从同一起点连续存放。后续状态避开完整的 2 KiB 暂存区。
constexpr uint32_t kBetaRaw = 0x0000;
constexpr uint32_t kBetaEff = 0x0800;
constexpr uint32_t kGRef[4] = {0x0A00, 0x0C00, 0x0E00, 0x1000};
constexpr uint32_t kQRstd = 0x1200;
constexpr uint32_t kKRstd = 0x1300;
constexpr uint32_t kGLast = 0x1400;
constexpr uint32_t kVfScratch = 0x1600;
} // namespace Arch35Ub

namespace Arch22Ub {
constexpr uint32_t kHardwareBytes = 0x30000; // 192 KiB
constexpr uint32_t kUsableBytes = 0x2E000;   // 末尾 8 KiB 保留
constexpr uint32_t kPrivateBytes = 0x12000;
constexpr uint32_t kPrivateBase[2] = {0x00000, 0x1C000};
constexpr uint32_t kSharedBase = 0x12000;
constexpr uint32_t kSharedG = kSharedBase;
constexpr uint32_t kSharedScratch = kSharedBase + 0x8000;

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

} // namespace KdaPrepare

#endif // CHUNK_KDA_FWD_PREPARE_POLICY_H
