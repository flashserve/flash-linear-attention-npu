/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_GATED_DELTA_RULE_FWD_ARCH35_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_FWD_ARCH35_STRUCT_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace GDN {

// Internal compile-time policy; no change to the serialized tiling ABI.
enum class Arch35GdnSyncVariant : uint32_t { B0 = 0, B30 = 30 };

template <Arch35GdnSyncVariant Variant>
struct Arch35GdnSyncTraits {
    static constexpr bool kB30 = Variant == Arch35GdnSyncVariant::B30;
    static constexpr bool kHeadMajorSolve64 = kB30;
    static constexpr bool kKktToSolveGroup = kB30;
    static constexpr bool kSolveToWuGroup = kB30;
    static constexpr bool kAggregateQkMask = kB30;
    static constexpr bool kAggregateOutput = kB30;
    static_assert(!(kKktToSolveGroup || kSolveToWuGroup || kAggregateQkMask || kAggregateOutput) ||
                      (kHeadMajorSolve64 && kKktToSolveGroup && kSolveToWuGroup),
                  "B30 requires the paired head-major coefficient protocol.");
};

struct Arch35ChunkGatedDeltaRuleCoefficientTiling {
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

struct Arch35ChunkGatedDeltaRuleFwdTrailer {
    Arch35ChunkGatedDeltaRuleCoefficientTiling coefficient;
    uint64_t scoreWorkspaceOffset;
    uint64_t aWorkspaceOffset;
    uint64_t solveWorkspaceOffset;
    uint64_t gCumsumBhtOffset;
    uint64_t writeGCumsum;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_FWD_ARCH35_STRUCT_H
