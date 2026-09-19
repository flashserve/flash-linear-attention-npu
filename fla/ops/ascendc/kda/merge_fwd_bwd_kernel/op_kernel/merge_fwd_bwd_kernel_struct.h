/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_KERNEL_STRUCT_H
#define MERGE_FWD_BWD_KERNEL_STRUCT_H

#include <cstdint>

namespace MergeFwBwd {

struct MergeFwdBwdKernelTilingData {
    int64_t S;
    int64_t Hv;
    int64_t K;
    int64_t V;
    int64_t rank;
    int64_t N;
    int64_t forward;
    int64_t usedAic;
    int64_t sysWorkspaceSize;
};

static_assert(sizeof(MergeFwdBwdKernelTilingData) == 72, "TilingData is 9 int64 fields");

constexpr uint32_t kKDim = 128;
constexpr uint32_t kVDim = 128;
constexpr uint32_t kTileM = 64;
constexpr uint32_t kRowStride = kVDim + kKDim; // 256
constexpr uint32_t kKvElems = kKDim * kVDim;
constexpr uint32_t kKkElems = kKDim * kKDim;

constexpr uint32_t kHFp32Bytes = kKvElems * 4;
constexpr uint32_t kMFp32Bytes = kKkElems * 4;
constexpr uint32_t kSlotBytes = kHFp32Bytes + kMFp32Bytes; // 128 KiB: FP32 H | FP32 M
// One Mix core pipelines at most 4 heads (AIV0: 0/2, AIV1: 1/3), like ChunkFwdH.
constexpr uint32_t kMaxTaskPerAic = 4;
constexpr uint32_t kCoreStride = kMaxTaskPerAic * kSlotBytes;
// Cube C scratch on the Mix kernel workspace argument (3510
// GetUserWorkspace is not AIC/AIV shared). Past every per-core H/M slot.
constexpr uint32_t kCTileElems = kTileM * kVDim;
constexpr uint32_t kC0TileBytes = kCTileElems * 4U;
constexpr uint32_t kCTilesPerHead = 4;
// C scratch starts after usedAic * kCoreStride (see CScratchElemOff).
constexpr int64_t kCWarmupTile = 2;
constexpr int64_t kCWarmupTileL0C0 = 3;

constexpr uint8_t kCrossCoreModeIndep = 0x2;
// Mix Mode 2 (FFTS): AIC Wait aggregates both AIVs. Flag IDs 0..7.
// Shared by Ascend950 (arch35 wrappers) and 910b/910_93 (common.h A2 wrappers).
// One Ready/Free/C1 per rank-step (not per head). Idle AIV DummyHandshake. No +16.
constexpr uint32_t kAivHeadSlots = 2;
constexpr uint8_t kChunkReadyFlag = 2;
constexpr uint8_t kChunkFreeFlag = 4;
constexpr uint8_t kChunkC1Flag = 3;
constexpr uint8_t kChunkRoundFlag = 5;
constexpr uint8_t kSubBlockFlagOffset = 16;

} // namespace MergeFwBwd

#endif
