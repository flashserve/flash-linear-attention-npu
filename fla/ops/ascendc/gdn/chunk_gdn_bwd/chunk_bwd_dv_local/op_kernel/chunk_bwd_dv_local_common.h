/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dv_local_common.h
 * \brief
 */

#ifndef CHUNK_BWD_DV_LOCAL_COMMON_H
#define CHUNK_BWD_DV_LOCAL_COMMON_H

namespace GDN {
constexpr int32_t NUM_2 = 2;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t SIZE_FLOAT = 4;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t CHUNK_SIZE_64 = 64;
constexpr int32_t CAL_NUM_FLOAT = 64; // API一次能处理256B，能计算64个float元素
constexpr uint64_t SYNC_AIV_AIC_FLAG_2 = 2;
constexpr uint64_t SYNC_AIC_AIV_FLAG_3 = 3;


__aicore__ inline int64_t CeilDiv(int64_t dividend, int64_t divisor)
{
    if (unlikely(divisor == 0)) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}


__aicore__ inline int64_t GetCoreBufferSingleHWorkspaceOffset(int64_t coreId, int64_t headSlotId, int64_t rowId,
                                                              int64_t chunkSize,
                                                              int64_t workspaceHeadSlotNum)
{
    int64_t headSlotStride = chunkSize * chunkSize;
    int64_t coreStride = workspaceHeadSlotNum * headSlotStride;
    int64_t headSlot = headSlotId % workspaceHeadSlotNum;
    return coreId * coreStride + headSlot * headSlotStride + rowId * chunkSize;
}


struct ChunkTaskIndex {
    int64_t batchId;
    int64_t tokenStart;
    int64_t chunkLen;

    __aicore__ inline ChunkTaskIndex() : batchId(0), tokenStart(0), chunkLen(0) {}

    __aicore__ inline ChunkTaskIndex(int64_t batchId_, int64_t tokenStart_, int64_t chunkLen_)
        : batchId(batchId_), tokenStart(tokenStart_), chunkLen(chunkLen_)
    {
    }
};

struct FixedLengthStrategy {
    int64_t chunkSize;
    int64_t seqLen;
    int64_t chunkNumForT;
    int64_t tailChunkLen;
    __aicore__ inline FixedLengthStrategy(int64_t chunkSize_, int64_t seqLen_, int64_t chunkNumForT_)
        : chunkSize(chunkSize_), seqLen(seqLen_), chunkNumForT(chunkNumForT_)
    {
        tailChunkLen = seqLen - (chunkNumForT - 1) * chunkSize;
    }

    __aicore__ inline void ResolveTask(int64_t loopIdx, ChunkTaskIndex &result) const
    {
        int64_t curChunkId = loopIdx % chunkNumForT;
        result.tokenStart = curChunkId * chunkSize;
        result.chunkLen = curChunkId == chunkNumForT - 1 ? tailChunkLen : chunkSize;
        result.batchId = loopIdx / chunkNumForT;
    }
};

struct VariableLengthStrategy {
    int64_t chunkSize;
    int64_t seqLen;
    int64_t chunkNumForT;
    AscendC::GlobalTensor<int64_t> cuSeqlensGm;
    AscendC::GlobalTensor<int64_t> chunkIndicesGm;
    __aicore__ inline VariableLengthStrategy(int64_t chunkSize_, int64_t seqLen_, int64_t chunkNumForT_,
                                             GM_ADDR cuSeqlens_, GM_ADDR chunkIndices_)
    {
        chunkSize = chunkSize_;
        seqLen = seqLen_;
        chunkNumForT = chunkNumForT_;
        cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens_);
        chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunkIndices_);
    }

    __aicore__ inline void ResolveTask(int64_t loopIdx, ChunkTaskIndex &result) const
    {
        int64_t curSeqId = chunkIndicesGm.GetValue(loopIdx * 2);
        int64_t curSeqChunkId = chunkIndicesGm.GetValue(loopIdx * 2 + 1);
        int64_t bos = cuSeqlensGm.GetValue(curSeqId);
        int64_t eos = cuSeqlensGm.GetValue(curSeqId + 1);
        int64_t curSeqT = eos - bos;
        int64_t chunkStartToken = curSeqChunkId * chunkSize;
        int64_t chunkEndToken = chunkStartToken + chunkSize;
        chunkEndToken = chunkEndToken > curSeqT ? curSeqT : chunkEndToken;
        result.batchId = 0;
        result.tokenStart = bos + chunkStartToken;
        result.chunkLen = chunkEndToken - chunkStartToken;
    }
};


} // namespace GDN
#endif // CHUNK_BWD_DV_LOCAL_COMMON_H
