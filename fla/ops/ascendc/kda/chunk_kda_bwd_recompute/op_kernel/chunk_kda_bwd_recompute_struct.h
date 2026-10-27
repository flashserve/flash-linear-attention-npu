/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_STRUCT_H
#define CHUNK_KDA_BWD_RECOMPUTE_STRUCT_H

#include <cstdint>

namespace KDA {

// All int64_t so host and CCE layouts match (no 4-byte float padding).
// sizeof must stay a multiple of 16 to equal kernel opParaSize.
struct ChunkKdaBwdRecomputeTilingData {
    int64_t B;
    int64_t Hk;
    int64_t Hv;
    int64_t hvPerHk;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkNum;
    int64_t chunkSize;
    int64_t isVariable;
    int64_t useGateInKernel;
    int64_t useExp2;
    int64_t hasALog;
    int64_t hasDtBias;
    int64_t lowerBoundBits;
    // Vector row tile selected from the target UB budget.
    int64_t vecRow;
    // Logical AIC count selected by host tiling (used by the arch35 task mapping).
    int64_t usedAic;
    int64_t reserved;
};

static_assert(sizeof(ChunkKdaBwdRecomputeTilingData) == 144,
              "TilingData is 18 int64 fields");

} // namespace KDA

#endif // CHUNK_KDA_BWD_RECOMPUTE_STRUCT_H
