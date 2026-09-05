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

// Internal compile-time implementation selector. The host encodes it in
// the tiling key; it must not be serialized into the public tiling ABI.
enum class GdnCoreSyncVariant : uint32_t {
    B0 = 0,
    B30 = 30,
};

// B30 carries the validated coefficient hand-offs and FwdO aggregation as one
// indivisible release specialization. Keep the traits centralized so a key
// cannot select a hand-off without the matching ownership policy.
template <GdnCoreSyncVariant kVariant>
struct GdnCoreSyncVariantTraits {
    static constexpr bool kHeadMajorSolve64Ownership =
        kVariant == GdnCoreSyncVariant::B30;
    static constexpr bool kKktToSolveGroupHandoff =
        kVariant == GdnCoreSyncVariant::B30;
    static constexpr bool kSolveToWuGroupHandoff =
        kVariant == GdnCoreSyncVariant::B30;
    static constexpr bool kUseImmediateMte2Mte1 = false;
    static constexpr bool kDeferMte2Mte1Wait = false;
    static constexpr bool kFwdOAggregateQkMaskBarrier =
        kVariant == GdnCoreSyncVariant::B30;
    static constexpr bool kFwdOAggregateOutputBarrier =
        kVariant == GdnCoreSyncVariant::B30;
    static constexpr bool kHasFwdOAggregation =
        kFwdOAggregateQkMaskBarrier || kFwdOAggregateOutputBarrier;

    static_assert(!(kKktToSolveGroupHandoff || kSolveToWuGroupHandoff) ||
                      kHeadMajorSolve64Ownership,
                  "A group-local coefficient hand-off requires head-major Solve64 ownership.");
    static_assert(!kDeferMte2Mte1Wait,
                  "The hardware-rejected deferred Solve64 wait must remain disabled.");
    static_assert(!kHasFwdOAggregation ||
                      (kHeadMajorSolve64Ownership && kKktToSolveGroupHandoff &&
                       kSolveToWuGroupHandoff),
                  "B30 FwdO aggregation requires the fully paired coefficient protocol.");
    static_assert(!kHasFwdOAggregation || !kUseImmediateMte2Mte1,
                  "B30 FwdO aggregation must not inherit the precision-risky Solve64 R path.");
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
