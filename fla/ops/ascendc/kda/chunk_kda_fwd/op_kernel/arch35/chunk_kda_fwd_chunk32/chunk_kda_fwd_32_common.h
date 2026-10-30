#pragma once

#include <stdint.h>

// 与 op_host/chunk_kda_fwd_tiling.h 的 ChunkKdaFwd32TilingData 一致。
struct KdaFd32TilingData {
    uint64_t nChunks;
    uint64_t scaleBf16;
    uint64_t gateScaleBits;
    uint64_t headNum;
    uint64_t seqLen;
    uint64_t hasInitialState;
    uint64_t outputFinalState;
    uint64_t qkGroup;
    uint64_t taskCount;
    uint64_t hvNumReal;
    uint64_t blockNum;
    uint64_t varlenMode;
    uint64_t epsilonBits;
    uint64_t gateFlags;
    uint64_t cuData[1025];
};
