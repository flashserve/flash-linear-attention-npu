/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_STRUCT_H
#define MERGE_FWD_BWD_STRUCT_H

#include <cstdint>

namespace MergeFwBwd {

struct MergeFwdBwdTilingData {
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

static_assert(sizeof(MergeFwdBwdTilingData) == 72, "TilingData is 9 int64 fields");

constexpr uint32_t kKDim = 128;
constexpr uint32_t kVDim = 128;
constexpr uint32_t kTileM = 64;
constexpr uint32_t kRowStride = kVDim + kKDim; // 256
constexpr uint32_t kKvElems = kKDim * kVDim;
constexpr uint32_t kKkElems = kKDim * kKDim;

constexpr uint32_t kHFp32Bytes = kKvElems * 4;
constexpr uint32_t kMFp32Bytes = kKkElems * 4;
constexpr uint32_t kSlotBytes = kHFp32Bytes + kMFp32Bytes; // 128 KiB: FP32 H | FP32 M
constexpr uint32_t kCoreStride = 2 * kSlotBytes;
// Cube C scratch on the Mix kernel workspace argument (3510
// GetUserWorkspace is not AIC/AIV shared). Past every per-core H/M slot.
constexpr uint32_t kCTileElems = kTileM * kVDim;
constexpr uint32_t kC0TileBytes = kCTileElems * 4U;
constexpr uint32_t kCTilesPerHead = 4;
// C scratch starts after usedAic * kCoreStride (see CScratchElemOff).
constexpr int64_t kCWarmupTile = 2;
constexpr int64_t kCWarmupTileL0C0 = 3;

constexpr uint8_t kCrossCoreModeIndep = 0x4;
// 6/7 are KDA Mix leftover on the die; Mode 4 flag state persists across launches.
constexpr uint8_t kChunkReadyFlag = 10;
constexpr uint8_t kChunkFreeFlag = 11;
constexpr uint8_t kSubBlockFlagOffset = 16;

} // namespace MergeFwBwd

#endif
