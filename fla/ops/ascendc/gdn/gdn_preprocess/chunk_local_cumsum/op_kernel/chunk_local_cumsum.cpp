/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_local_cumsum.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "chunk_local_cumsum_tiling_data.h"

using namespace AscendC;

namespace {
constexpr int64_t H_TILE_SIZE = 512;
constexpr int64_t UB_ALIGN_BYTES = 32;
constexpr int64_t FLOAT_ALIGN_ELEMS = UB_ALIGN_BYTES / static_cast<int64_t>(sizeof(float));
constexpr int64_t FAST_CHUNK_BUFFER_LIMIT = 160 * 1024;
constexpr int64_t FAST_CHUNK_SCAN_BUFFER_NUM = 2;
constexpr int64_t FP32_REPEAT_ELEMS = 64;
constexpr int64_t VECTOR_MAX_REPEAT_TIMES = 255;
constexpr int64_t VECTOR_MAX_CALC_ELEMS = FP32_REPEAT_ELEMS * VECTOR_MAX_REPEAT_TIMES;
constexpr int32_t BUFFER_NUM = 1;

__aicore__ inline int64_t MinInt64(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

__aicore__ inline int64_t CeilDivInt64(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline int64_t AlignUpInt64(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

__aicore__ inline int64_t AlignDownInt64(int64_t value, int64_t align)
{
    return (value / align) * align;
}

class ChunkLocalCumsumKernel {
public:
    __aicore__ inline ChunkLocalCumsumKernel() = default;

    __aicore__ inline void Init(GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR out,
                                const ChunkLocalCumsumTilingData *tiling)
    {
        tiling_ = tiling;
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g), tiling_->totalElements);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), tiling_->totalElements);
        if (tiling_->isVarlen != 0) {
            cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
            chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices), tiling_->numBlocks * 2);
        }
        int64_t maxFastHLen = FAST_CHUNK_BUFFER_LIMIT /
                              (FAST_CHUNK_SCAN_BUFFER_NUM * tiling_->chunkSize *
                               static_cast<int64_t>(sizeof(float)));
        fastHTileSize_ = AlignDownInt64(MinInt64(MinInt64(H_TILE_SIZE, tiling_->h), maxFastHLen),
                                        FLOAT_ALIGN_ELEMS);
        // The log-step scan needs two chunk buffers; shrink only the fast H tile, not the whole fast path.
        chunkFastPath_ = ((tiling_->h & (FLOAT_ALIGN_ELEMS - 1)) == 0) &&
                         (fastHTileSize_ >= FLOAT_ALIGN_ELEMS);
        if (chunkFastPath_) {
            int64_t chunkBufferBytes = AlignUpInt64(tiling_->chunkSize * fastHTileSize_ *
                                                    static_cast<int64_t>(sizeof(float)), UB_ALIGN_BYTES);
            pipe_.InitBuffer(chunkQueue_, BUFFER_NUM, chunkBufferBytes);
            pipe_.InitBuffer(scanBuf_, chunkBufferBytes);
        } else {
            int64_t rowBufferBytes = AlignUpInt64(H_TILE_SIZE * static_cast<int64_t>(sizeof(float)), UB_ALIGN_BYTES);
            pipe_.InitBuffer(rowQueue_, BUFFER_NUM, rowBufferBytes);
            pipe_.InitBuffer(outQueue_, BUFFER_NUM, rowBufferBytes);
            pipe_.InitBuffer(accBuf_, rowBufferBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (tiling_->isVarlen != 0) {
            ProcessVarlen();
        } else {
            ProcessFixed();
        }
    }

private:
    __aicore__ inline void WaitVToMte3()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void WaitMte3ToV()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline LocalTensor<float> LoadRowToUb(int64_t gmOffset, int64_t elementCount)
    {
        LocalTensor<float> rowLocal = rowQueue_.AllocTensor<float>();
        if ((elementCount & 7) == 0) {
            DataCopy(rowLocal, gGm_[gmOffset], static_cast<uint32_t>(elementCount));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * static_cast<int64_t>(sizeof(float))),
                                         0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            DataCopyPad(rowLocal, gGm_[gmOffset], copyParams, padParams);
        }
        rowQueue_.EnQue(rowLocal);
        return rowQueue_.DeQue<float>();
    }

    __aicore__ inline void CopyUbToGm(int64_t gmOffset, LocalTensor<float> srcLocal, int64_t elementCount)
    {
        if ((elementCount & 7) == 0) {
            DataCopy(outGm_[gmOffset], srcLocal, static_cast<uint32_t>(elementCount));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * static_cast<int64_t>(sizeof(float))),
                                         0, 0, 0};
            DataCopyPad(outGm_[gmOffset], srcLocal, copyParams);
        }
    }

    __aicore__ inline LocalTensor<float> LoadChunkToUb(int64_t gmOffset, int64_t chunkLen, int64_t hLen,
                                                       int64_t hLenAlign)
    {
        LocalTensor<float> chunkLocal = chunkQueue_.AllocTensor<float>();
        int64_t totalElems = chunkLen * hLen;
        if (hLen == tiling_->h && ((gmOffset & (FLOAT_ALIGN_ELEMS - 1)) == 0) &&
            ((totalElems & (FLOAT_ALIGN_ELEMS - 1)) == 0)) {
            DataCopy(chunkLocal, gGm_[gmOffset], static_cast<uint32_t>(totalElems));
        } else {
            DataCopyExtParams copyParams{static_cast<uint16_t>(chunkLen),
                                         static_cast<uint32_t>(hLen * static_cast<int64_t>(sizeof(float))),
                                         static_cast<uint32_t>((tiling_->h - hLen) *
                                                              static_cast<int64_t>(sizeof(float))),
                                         0,
                                         0};
            DataCopyPadExtParams<float> padParams{hLenAlign != hLen, 0,
                                                  static_cast<uint8_t>(hLenAlign - hLen), 0.0f};
            DataCopyPad(chunkLocal, gGm_[gmOffset], copyParams, padParams);
        }
        chunkQueue_.EnQue(chunkLocal);
        return chunkQueue_.DeQue<float>();
    }

    __aicore__ inline void CopyChunkUbToGm(int64_t gmOffset, LocalTensor<float> chunkLocal, int64_t chunkLen,
                                           int64_t hLen, int64_t hLenAlign)
    {
        int64_t totalElems = chunkLen * hLen;
        if (hLen == tiling_->h && ((gmOffset & (FLOAT_ALIGN_ELEMS - 1)) == 0) &&
            ((totalElems & (FLOAT_ALIGN_ELEMS - 1)) == 0)) {
            DataCopy(outGm_[gmOffset], chunkLocal, static_cast<uint32_t>(totalElems));
        } else {
            DataCopyExtParams copyParams{static_cast<uint16_t>(chunkLen),
                                         static_cast<uint32_t>(hLen * static_cast<int64_t>(sizeof(float))),
                                         static_cast<uint32_t>((hLenAlign - hLen) *
                                                              static_cast<int64_t>(sizeof(float)) / UB_ALIGN_BYTES),
                                         static_cast<uint32_t>((tiling_->h - hLen) *
                                                              static_cast<int64_t>(sizeof(float))),
                                         0};
            DataCopyPad(outGm_[gmOffset], chunkLocal, copyParams);
        }
    }

    __aicore__ inline void AddBatched(LocalTensor<float> dstLocal, LocalTensor<float> src0Local,
                                      LocalTensor<float> src1Local, int64_t elementCount)
    {
        for (int64_t offset = 0; offset < elementCount; offset += VECTOR_MAX_CALC_ELEMS) {
            int64_t curCount = MinInt64(VECTOR_MAX_CALC_ELEMS, elementCount - offset);
            Add(dstLocal[offset], src0Local[offset], src1Local[offset], static_cast<uint32_t>(curCount));
        }
    }

    __aicore__ inline void AddsBatched(LocalTensor<float> dstLocal, LocalTensor<float> srcLocal, float scalar,
                                       int64_t elementCount)
    {
        for (int64_t offset = 0; offset < elementCount; offset += VECTOR_MAX_CALC_ELEMS) {
            int64_t curCount = MinInt64(VECTOR_MAX_CALC_ELEMS, elementCount - offset);
            Adds(dstLocal[offset], srcLocal[offset], scalar, static_cast<uint32_t>(curCount));
        }
    }

    __aicore__ inline void MulsBatched(LocalTensor<float> dstLocal, LocalTensor<float> srcLocal, float scalar,
                                       int64_t elementCount)
    {
        for (int64_t offset = 0; offset < elementCount; offset += VECTOR_MAX_CALC_ELEMS) {
            int64_t curCount = MinInt64(VECTOR_MAX_CALC_ELEMS, elementCount - offset);
            Muls(dstLocal[offset], srcLocal[offset], scalar, static_cast<uint32_t>(curCount));
        }
    }

    __aicore__ inline void ComputeForwardScanStep(LocalTensor<float> dstLocal, LocalTensor<float> srcLocal,
                                                  int64_t chunkLen, int64_t hLenAlign, int64_t step)
    {
        AddsBatched(dstLocal, srcLocal, 0.0f, step * hLenAlign);
        AddBatched(dstLocal[step * hLenAlign], srcLocal[step * hLenAlign], srcLocal,
                   (chunkLen - step) * hLenAlign);
    }

    __aicore__ inline void ComputeReverseScanStep(LocalTensor<float> dstLocal, LocalTensor<float> srcLocal,
                                                  int64_t chunkLen, int64_t hLenAlign, int64_t step)
    {
        int64_t activeRows = chunkLen - step;
        AddBatched(dstLocal, srcLocal, srcLocal[step * hLenAlign], activeRows * hLenAlign);
        AddsBatched(dstLocal[activeRows * hLenAlign], srcLocal[activeRows * hLenAlign], 0.0f, step * hLenAlign);
    }

    // Ping-pong Hillis-Steele scan; each step reads from one UB buffer and writes the other.
    __aicore__ inline bool ComputeChunkPrefixInUb(LocalTensor<float> chunkLocal, int64_t chunkLen, int64_t hLenAlign)
    {
        LocalTensor<float> scanLocal = scanBuf_.Get<float>();
        bool nextSrcIsChunk = true;
        for (int64_t step = 1; step < chunkLen; step <<= 1) {
            LocalTensor<float> srcLocal = nextSrcIsChunk ? chunkLocal : scanLocal;
            LocalTensor<float> dstLocal = nextSrcIsChunk ? scanLocal : chunkLocal;
            if (tiling_->reverse != 0) {
                ComputeReverseScanStep(dstLocal, srcLocal, chunkLen, hLenAlign, step);
            } else {
                ComputeForwardScanStep(dstLocal, srcLocal, chunkLen, hLenAlign, step);
            }
            PipeBarrier<PIPE_V>();
            nextSrcIsChunk = !nextSrcIsChunk;
        }

        if (tiling_->scale != 1.0f) {
            LocalTensor<float> resultLocal = nextSrcIsChunk ? chunkLocal : scanLocal;
            MulsBatched(resultLocal, resultLocal, tiling_->scale, chunkLen * hLenAlign);
            PipeBarrier<PIPE_V>();
        }
        return nextSrcIsChunk;
    }

    __aicore__ inline void ProcessSequenceChunkFast(int64_t baseOffset, int64_t chunkStart, int64_t chunkEnd,
                                                    int64_t hStart, int64_t hLen)
    {
        int64_t chunkLen = chunkEnd - chunkStart;
        int64_t hLenAlign = AlignUpInt64(hLen, FLOAT_ALIGN_ELEMS);
        int64_t rowOffset = baseOffset + chunkStart * tiling_->h + hStart;
        LocalTensor<float> chunkLocal = LoadChunkToUb(rowOffset, chunkLen, hLen, hLenAlign);
        bool resultInChunk = ComputeChunkPrefixInUb(chunkLocal, chunkLen, hLenAlign);
        LocalTensor<float> outLocal = resultInChunk ? chunkLocal : scanBuf_.Get<float>();
        WaitVToMte3();
        CopyChunkUbToGm(rowOffset, outLocal, chunkLen, hLen, hLenAlign);
        WaitMte3ToV();
        chunkQueue_.FreeTensor(chunkLocal);
    }

    __aicore__ inline void StoreAccumVector(int64_t outOffset, LocalTensor<float> accLocal, int64_t hLen)
    {
        if (tiling_->scale == 1.0f) {
            WaitVToMte3();
            CopyUbToGm(outOffset, accLocal, hLen);
            WaitMte3ToV();
            return;
        }

        LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
        Muls(outLocal, accLocal, tiling_->scale, static_cast<uint32_t>(hLen));
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<float>();
        CopyUbToGm(outOffset, outLocal, hLen);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void ProcessSequenceChunk(int64_t baseOffset, int64_t chunkStart, int64_t chunkEnd,
                                                int64_t hStart, int64_t hLen)
    {
        LocalTensor<float> accLocal = accBuf_.Get<float>();
        if (tiling_->reverse != 0) {
            for (int64_t localT = chunkEnd - 1; localT >= chunkStart; --localT) {
                int64_t rowOffset = baseOffset + localT * tiling_->h + hStart;
                LocalTensor<float> rowLocal = LoadRowToUb(rowOffset, hLen);
                if (localT == chunkEnd - 1) {
                    Adds(accLocal, rowLocal, 0.0f, static_cast<uint32_t>(hLen));
                } else {
                    Add(accLocal, accLocal, rowLocal, static_cast<uint32_t>(hLen));
                }
                rowQueue_.FreeTensor(rowLocal);
                PipeBarrier<PIPE_V>();
                StoreAccumVector(rowOffset, accLocal, hLen);
            }
        } else {
            for (int64_t localT = chunkStart; localT < chunkEnd; ++localT) {
                int64_t rowOffset = baseOffset + localT * tiling_->h + hStart;
                LocalTensor<float> rowLocal = LoadRowToUb(rowOffset, hLen);
                if (localT == chunkStart) {
                    Adds(accLocal, rowLocal, 0.0f, static_cast<uint32_t>(hLen));
                } else {
                    Add(accLocal, accLocal, rowLocal, static_cast<uint32_t>(hLen));
                }
                rowQueue_.FreeTensor(rowLocal);
                PipeBarrier<PIPE_V>();
                StoreAccumVector(rowOffset, accLocal, hLen);
            }
        }
    }

    __aicore__ inline void ProcessFixed()
    {
        int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t chunkNum = CeilDivInt64(tiling_->t, tiling_->chunkSize);
        int64_t hTileSize = chunkFastPath_ ? fastHTileSize_ : H_TILE_SIZE;
        int64_t hTileNum = CeilDivInt64(tiling_->h, hTileSize);
        int64_t taskNum = tiling_->b * chunkNum * hTileNum;
        for (int64_t taskIdx = blockIdx; taskIdx < taskNum; taskIdx += blockNum) {
            int64_t hTileIdx = taskIdx % hTileNum;
            int64_t chunkLinear = taskIdx / hTileNum;
            int64_t chunkIdx = chunkLinear % chunkNum;
            int64_t bIdx = chunkLinear / chunkNum;
            int64_t hStart = hTileIdx * hTileSize;
            int64_t hLen = MinInt64(hTileSize, tiling_->h - hStart);
            int64_t chunkStart = chunkIdx * tiling_->chunkSize;
            int64_t chunkEnd = MinInt64(chunkStart + tiling_->chunkSize, tiling_->t);
            int64_t baseOffset = bIdx * tiling_->t * tiling_->h;
            if (chunkFastPath_) {
                ProcessSequenceChunkFast(baseOffset, chunkStart, chunkEnd, hStart, hLen);
            } else {
                ProcessSequenceChunk(baseOffset, chunkStart, chunkEnd, hStart, hLen);
            }
        }
    }

    __aicore__ inline void ProcessVarlen()
    {
        int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t hTileSize = chunkFastPath_ ? fastHTileSize_ : H_TILE_SIZE;
        int64_t hTileNum = CeilDivInt64(tiling_->h, hTileSize);
        int64_t taskNum = tiling_->b * tiling_->numBlocks * hTileNum;
        for (int64_t taskIdx = blockIdx; taskIdx < taskNum; taskIdx += blockNum) {
            int64_t hTileIdx = taskIdx % hTileNum;
            int64_t blockLinear = taskIdx / hTileNum;
            int64_t globalBlock = blockLinear % tiling_->numBlocks;
            int64_t outerIdx = blockLinear / tiling_->numBlocks;
            int64_t hStart = hTileIdx * hTileSize;
            int64_t hLen = MinInt64(hTileSize, tiling_->h - hStart);
            int64_t seqId = chunkIndicesGm_.GetValue(globalBlock * 2);
            int64_t localBlock = chunkIndicesGm_.GetValue(globalBlock * 2 + 1);
            int64_t bos = cuSeqlensGm_.GetValue(seqId);
            int64_t eos = cuSeqlensGm_.GetValue(seqId + 1);
            int64_t seqLen = eos - bos;
            int64_t tStart = localBlock * tiling_->blockT;
            int64_t tEnd = MinInt64(tStart + tiling_->blockT, seqLen);
            int64_t baseOffset = outerIdx * tiling_->t * tiling_->h + bos * tiling_->h;
            for (int64_t chunkStart = tStart; chunkStart < tEnd; chunkStart += tiling_->chunkSize) {
                int64_t chunkEnd = MinInt64(chunkStart + tiling_->chunkSize, tEnd);
                if (chunkFastPath_) {
                    ProcessSequenceChunkFast(baseOffset, chunkStart, chunkEnd, hStart, hLen);
                } else {
                    ProcessSequenceChunk(baseOffset, chunkStart, chunkEnd, hStart, hLen);
                }
            }
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> chunkQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<> scanBuf_;
    TBuf<> accBuf_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<int64_t> cuSeqlensGm_;
    GlobalTensor<int64_t> chunkIndicesGm_;
    const ChunkLocalCumsumTilingData *tiling_ = nullptr;
    int64_t fastHTileSize_ = H_TILE_SIZE;
    bool chunkFastPath_ = false;
};
} // namespace

extern "C" __global__ __aicore__ void chunk_local_cumsum(GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                                          GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ChunkLocalCumsumTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkLocalCumsumTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ChunkLocalCumsumKernel op;
    op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
    op.Process();
}
