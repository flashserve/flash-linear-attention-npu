/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CHUNK_BWD_DQKWG_COMMON_H
#define CHUNK_BWD_DQKWG_COMMON_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t CAL_NUM_FLOAT = 64;
constexpr int32_t NUM_PER_REP_FP32 = 64;
constexpr int32_t MOD_64_MASK = 0x3f;
constexpr int32_t LOG2_64 = 6;

constexpr uint64_t SYNC_AIC_AIV_FLAG_0 = 5;
constexpr uint64_t SYNC_AIV_AIC_FLAG_0 = 3;
constexpr uint64_t SYNC_AIV_AIC_FLAG_1 = 4;

__aicore__ inline uint64_t DqkwgBtxKElemOffset(uint32_t coreIdx, uint32_t h, uint64_t H, uint64_t BT, uint64_t K)
{
    return (coreIdx * H + (uint64_t)h) * (BT * K);
}

__aicore__ inline uint64_t DqkwgBtbElemOffset(uint32_t coreIdx, uint32_t h, uint64_t H, uint64_t BT)
{
    return (coreIdx * H + (uint64_t)h) * (BT * BT);
}

__aicore__ inline uint64_t DqkwgScalarElemOffset(uint32_t coreIdx, uint32_t h, uint64_t H)
{
    return coreIdx * H + (uint64_t)h;
}

__aicore__ inline void SetCubeReady()
{
    CrossCoreSetFlag<0x4, PIPE_FIX>(SYNC_AIC_AIV_FLAG_0);
}
__aicore__ inline void WaitCredit()
{

    CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_0);
}
__aicore__ inline void WaitClear()
{
    CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_1);
}

__aicore__ inline void WaitCubeReady()
{
    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
}
__aicore__ inline void SetVecCredit()
{
    CrossCoreSetFlag<0x4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
}
__aicore__ inline void SetVecClear()
{
    CrossCoreSetFlag<0x4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_1);
}

constexpr uint32_t FP32_PER_REPEAT = 64;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__aicore__ inline void GetChunkOffset(GlobalTensor<uint64_t> cuSeqlensTensor, GlobalTensor<uint64_t> chunkIndicesTensor, uint64_t B, uint64_t HV, uint64_t T,
                                      uint64_t chunkSize, uint32_t loopIdx, uint32_t &bos, uint32_t &eos, bool isVarLen)
{
    if (isVarLen) {
        uint32_t seqIdx = chunkIndicesTensor.GetValue(2 * loopIdx);
        uint32_t chunkIdx = chunkIndicesTensor.GetValue(2 * loopIdx + 1);
        uint32_t curSeqBegin = cuSeqlensTensor.GetValue(seqIdx);
        uint32_t curSeqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
        bos = curSeqBegin + chunkIdx * chunkSize;
        eos = bos + chunkSize > curSeqEnd ? curSeqEnd : bos + chunkSize;
    } else {
        uint32_t coreLoopsInB = CEIL_DIV(T, chunkSize);
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        uint32_t bIdx = loopIdx / coreLoopsInB;
        bos = chunkIdx * chunkSize;
        eos = bos + chunkSize > T ? T : bos + chunkSize;
        bos += (bIdx * HV * T);
        eos += (bIdx * HV * T);
    }
}

__aicore__ inline void ReduceSumCustom(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
                                       uint32_t count)
{
    uint32_t repeatTimes = count >> LOG2_64;
    uint32_t tailCount = count & MOD_64_MASK;

    BinaryRepeatParams repeatParams = {1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0};
    if (likely(repeatTimes > 1)) {
        Add(srcLocal, srcLocal[NUM_PER_REP_FP32], srcLocal, NUM_PER_REP_FP32, repeatTimes - 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount > 0)) {
        Add(srcLocal, srcLocal[repeatTimes << LOG2_64], srcLocal, tailCount, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    WholeReduceSum(dstLocal, srcLocal, repeatTimes > 0 ? NUM_PER_REP_FP32 : count, 1, 0, 1, 0);
}

#endif
