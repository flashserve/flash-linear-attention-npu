/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_COMMON_H
#define CHUNK_KDA_BWD_RECOMPUTE_COMMON_H

#include "kernel_operator.h"

constexpr uint64_t KDA_BWD_RECOMPUTE_SYNC_AIV_AIC_FLAG = 3;
constexpr uint64_t KDA_BWD_RECOMPUTE_SYNC_AIC_AIV_FLAG = 5;
constexpr uint64_t KDA_BWD_RECOMPUTE_ONE_BLOCK_32 = 32;
constexpr uint32_t KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64 = 64;
constexpr float KDA_BWD_RECOMPUTE_RCP_LN2 = 1.4426950408889634f;
constexpr float KDA_BWD_RECOMPUTE_LN2 = 0.69314718055994530942f;
constexpr uint32_t KDA_BWD_RECOMPUTE_BT = 64;
constexpr uint32_t KDA_BWD_RECOMPUTE_K = 128;
constexpr uint32_t KDA_BWD_RECOMPUTE_V = 128;

__aicore__ inline void KdaBwdRecomputeGetChunkOffset(
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, uint64_t B, uint64_t H, uint64_t T,
    uint64_t chunkSize, uint32_t loopIdx, uint32_t &bos, uint32_t &eos, int64_t isVariable)
{
    // gen_placeholder keeps optional GM_ADDR non-null. Use tiling.isVariable, not pointer.
    if (isVariable == 0 || cu_seqlens == nullptr || chunk_indices == nullptr) {
        uint32_t coreLoopsInB = (T + chunkSize - 1) / chunkSize;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        uint32_t bIdx = loopIdx / coreLoopsInB;
        bos = chunkIdx * chunkSize;
        eos = bos + chunkSize > T ? T : bos + chunkSize;
        bos += (bIdx * H * T);
        eos += (bIdx * H * T);
    } else {
        AscendC::GlobalTensor<uint64_t> cuSeqlensTensor;
        AscendC::GlobalTensor<uint64_t> chunkIndicesTensor;
        cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)cu_seqlens);
        chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)chunk_indices);
        uint32_t seqIdx = chunkIndicesTensor.GetValue(2 * loopIdx);
        uint32_t chunkIdx = chunkIndicesTensor.GetValue(2 * loopIdx + 1);
        uint32_t curSeqBegin = cuSeqlensTensor.GetValue(seqIdx);
        uint32_t curSeqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
        bos = curSeqBegin + chunkIdx * chunkSize;
        eos = bos + chunkSize > curSeqEnd ? curSeqEnd : bos + chunkSize;
    }
}

#endif // CHUNK_KDA_BWD_RECOMPUTE_COMMON_H
