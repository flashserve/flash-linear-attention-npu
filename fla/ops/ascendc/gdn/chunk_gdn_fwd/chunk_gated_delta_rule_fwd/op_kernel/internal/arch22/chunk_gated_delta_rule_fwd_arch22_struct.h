/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_GATED_DELTA_RULE_FWD_ARCH22_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_FWD_ARCH22_STRUCT_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace GDN {

constexpr uint64_t FP32_SOLVE_MERGE_BATCH_SIZE = 16;
constexpr uint64_t FP32_SOLVE_RESULT_BUFFER_COUNT = 2;
constexpr uint64_t FP32_SOLVE_SMALL_TEMP_SLOT_COUNT = 16;
constexpr uint64_t FP32_SOLVE_LARGE_TEMP_SLOT_COUNT = 2;

struct Arch22ChunkGatedDeltaRuleFwdAbcTiling {
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

struct Arch22ChunkGatedDeltaRuleFwdTrailer {
    Arch22ChunkGatedDeltaRuleFwdAbcTiling abc;
    uint64_t scoreWorkspaceOffset;
    uint64_t aWorkspaceOffset;
    uint64_t solveWorkspaceOffset;
    uint64_t gCumsumBhtOffset;
    // 仅 DAV_2201 的分层 FP32 Solve 使用；其余架构保持零值且不访问。
    uint64_t solveFp32InputOffset;
    uint64_t solveD16Offset;
    uint64_t solveD32Offset;
    uint64_t solveD64Offset;
    uint64_t solveSequenceCount;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_FWD_ARCH22_STRUCT_H
