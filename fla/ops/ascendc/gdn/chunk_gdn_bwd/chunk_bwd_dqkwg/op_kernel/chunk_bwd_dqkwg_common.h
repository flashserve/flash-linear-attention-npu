/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_common.h
 * \brief ChunkBwdDqkwg 通用常量和定义
 */

#ifndef CHUNK_BWD_DQKWG_COMMON_H
#define CHUNK_BWD_DQKWG_COMMON_H

#include "kernel_operator.h"
#include "chunk_bwd_dqkwg_struct.h"

constexpr uint64_t CONST_B = 1;
constexpr uint64_t CONST_HV = 4;
constexpr uint64_t CONST_HK = 4;
constexpr uint64_t CONST_T = 2816;
constexpr uint64_t CONST_K = 128;
constexpr uint64_t CONST_V = 128;
constexpr uint64_t CONST_BT = 64;
constexpr uint64_t CONST_NUM_CHUNKS = 44;
constexpr int32_t CAL_NUM_FLOAT = 64;
constexpr uint64_t SYNC_PART1_AIC_AIV = 10;
constexpr uint64_t SYNC_PART1_AIV_AIC = 11;
constexpr uint64_t SYNC_PART2_AIC_AIV = 20;
constexpr uint64_t SYNC_PART2_AIV_AIC = 21;
constexpr uint64_t SYNC_PART3_AIC_AIV = 30;
constexpr uint64_t SYNC_PART3_AIV_AIC = 31;
constexpr uint64_t SYNC_PART4_AIC_AIV = 40;
constexpr uint64_t SYNC_PART4_AIV_AIC = 41;
constexpr uint64_t SYNC_PART5_AIC_AIV = 50;
constexpr uint64_t SYNC_PART5_AIV_AIC = 51;
constexpr uint64_t SYNC_PART6_AIC_AIV = 60;
constexpr uint64_t SYNC_PART6_AIV_AIC = 61;
constexpr uint64_t SYNC_PART7_AIC_AIV = 70;
constexpr uint64_t SYNC_PART7_AIV_AIC = 71;
constexpr uint64_t SYNC_PART8_AIC_AIV = 80;
constexpr uint64_t SYNC_PART8_AIV_AIC = 81;
constexpr uint64_t SYNC_AIV_AIC_FLAG_0 = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG_0 = 5;

constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t ONE_BLOCK_32 = 32;
constexpr uint32_t FP32_PER_REPEAT = 64;
constexpr uint32_t FP16_PER_BLOCK = 16;

constexpr uint32_t FP16_SIZE = 2;
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BF16_SIZE = 2;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ALIGN_UP(x, align) (((x) + (align) - 1) / (align) * (align))

template<typename T>
struct TypeTraits {
    using ComputeType = float;
    static constexpr bool needsCast = true;
};

template<>
struct TypeTraits<half> {
    using ComputeType = float;
    static constexpr bool needsCast = true;
};

namespace GDN {

__aicore__ inline int64_t CeilDiv(int64_t dividend, int64_t divisor)
{
    if (unlikely(divisor == 0)) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

struct IndexResult {
    int64_t curBatchId;
    int64_t curTokenId;
    int64_t chunkLen;

    __aicore__ inline IndexResult() : curBatchId(0), curTokenId(0), chunkLen(0)
    {
    }

    __aicore__ inline IndexResult(int64_t curBatchId_, int64_t curTokenId_, int64_t chunkLen_)
        : curBatchId(curBatchId_), curTokenId(curTokenId_), chunkLen(chunkLen_)
    {
    }
};

struct FixedLengthStrategy {
    int64_t chunkSize;
    int64_t lenT;
    int64_t chunkNumForT;
    int64_t chunkLenTail;
    int64_t HV;

    __aicore__ inline FixedLengthStrategy(int64_t chunkSize_, int64_t lenT_, int64_t chunkNumForT_, int64_t HV_ = 0)
        : chunkSize(chunkSize_), lenT(lenT_), chunkNumForT(chunkNumForT_), HV(HV_)
    {
        chunkLenTail = lenT - (chunkNumForT - 1) * chunkSize;
    }

    __aicore__ inline void calculate(int64_t loopIdx, IndexResult &result) const
    {
        int64_t curChunkId = loopIdx % chunkNumForT;
        int64_t bIdx = loopIdx / chunkNumForT;
        result.curTokenId = curChunkId * chunkSize;
        result.chunkLen = curChunkId == chunkNumForT - 1 ? chunkLenTail : chunkSize;
        result.curBatchId = bIdx;
    }
};

struct VariableLengthStrategy {
    int64_t chunkSize;
    int64_t lenT;
    int64_t chunkNumForT;
    AscendC::GlobalTensor<int64_t> cuSeqlensGm;
    AscendC::GlobalTensor<int64_t> chunkIndicesGm;

    __aicore__ inline VariableLengthStrategy(int64_t chunkSize_, int64_t lenT_, int64_t chunkNumForT_,
                                             GM_ADDR cuSeqlens_, GM_ADDR chunkIndices_)
    {
        chunkSize = chunkSize_;
        lenT = lenT_;
        chunkNumForT = chunkNumForT_;
        cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens_);
        chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunkIndices_);
    }

    __aicore__ inline void calculate(int64_t loopIdx, IndexResult &result) const
    {
        int64_t curSeqId = chunkIndicesGm.GetValue(loopIdx * 2);
        int64_t curSeqChunkId = chunkIndicesGm.GetValue(loopIdx * 2 + 1);
        int64_t bos = cuSeqlensGm.GetValue(curSeqId);
        int64_t eos = cuSeqlensGm.GetValue(curSeqId + 1);
        int64_t curSeqT = eos - bos;
        int64_t chunkStartToken = curSeqChunkId * chunkSize;
        int64_t chunkEndToken = chunkStartToken + chunkSize;
        chunkEndToken = chunkEndToken > curSeqT ? curSeqT : chunkEndToken;
        result.curBatchId = 0;
        result.curTokenId = bos + chunkStartToken;
        result.chunkLen = chunkEndToken - chunkStartToken;
    }
};

} // namespace GDN

__aicore__ void inline GetChunkOffset(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, uint64_t B, uint64_t HV, uint64_t T,
                                      uint64_t chunkSize, uint32_t loopIdx, uint32_t &bos, uint32_t &eos)
{
    if (cu_seqlens == nullptr) {
        uint32_t coreLoopsInB = CEIL_DIV(T, chunkSize);
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        uint32_t bIdx = loopIdx / coreLoopsInB;
        bos = chunkIdx * chunkSize;
        eos = bos + chunkSize > T ? T : bos + chunkSize;
        bos += (bIdx * HV * T);
        eos += (bIdx * HV * T);
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

    return;
}

#endif  // CHUNK_BWD_DQKWG_COMMON_H
