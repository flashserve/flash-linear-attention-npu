/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_GDN_CORE_FWD_STRUCT_H
#define CHUNK_GDN_CORE_FWD_STRUCT_H

#include <cstdint>
#include "chunk_gdn_core_output_mask.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GDN {

// Internal compile-time experiment selector. The host encodes this selector in
// the tiling key; it must not be serialized into the public tiling ABI.
enum class GdnCoreSyncVariant : uint32_t {
    B0 = 0,
    B1 = 1,
    B2 = 2,
    B3 = 3,
    B4 = 4,
    B5 = 5,
    B6 = 6,
    B7 = 7,
    B8 = 8,
    B9 = 9,
    B10 = 10,
    B11 = 11,
    B12 = 12,
    B13 = 13,
    B14 = 14,
    B15 = 15,
    B16 = 16,
    B17 = 17,
    B18 = 18,
    B19 = 19,
    B20 = 20,
    B21 = 21,
};

struct ChunkGdnCoreCoefficientTiling {
    uint64_t B;
    uint64_t Hk;
    uint64_t Hv;
    uint64_t hvPerHk;
    uint64_t T;
    uint64_t K;
    uint64_t BT;
    uint64_t NT;
    uint64_t taskNum;
    uint64_t usedAicNum;
    uint64_t usedAivNum;
    uint64_t btAlign;
    uint64_t isVarlen;
    uint64_t scoreWorkspaceBytes;
    uint64_t aWorkspaceBytes;
    uint64_t solveWorkspacePerCoreBytes;
    int64_t totalTiles;
    int64_t matrixSize;
    int64_t numHeads;
    int64_t seqLen;
    int64_t batchSize;
    int64_t isLower;
    int64_t hasCuSeqlens;
    int64_t tilesPerCore;
    int64_t chunkSize;
    int64_t numChunks;
    int64_t lastChunkValidSize;
    int64_t totalChunks;
    int64_t layoutMode;
    int64_t dtypeMode;
    int64_t totalTokens;
    TCubeTiling cubeTilingData;
};

struct ChunkGdnCoreFwdTrailer {
    ChunkGdnCoreCoefficientTiling coefficient;
    uint64_t scoreWorkspaceOffset;
    uint64_t aWorkspaceOffset;
    uint64_t solveWorkspaceOffset;
    uint64_t gCumsumBhtOffset;
    uint64_t outputMask;
};

} // namespace GDN

#endif // CHUNK_GDN_CORE_FWD_STRUCT_H
