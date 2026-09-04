/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Kernel-side view of SolveTriTilingData, copied from origin/vcs_test.
 * Field order matches the host BEGIN_TILING_DATA_DEF.
 */

#ifndef SOLVE_TRI_TILING_H
#define SOLVE_TRI_TILING_H

struct SolveTriTilingData {
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
    int64_t isVarlen;
    int64_t totalChunks;
    int64_t layoutMode;
    int64_t totalTokens;   // NTD: total tokens
};

#endif
