/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_bwd.h
 * \brief Causal Conv1D backward kernel
 *
 * Algorithm (non-final-state):
 *  Given x[B,T,D], dy[B,T,D], weight[W,D]:
 *    For each timestep t, for each filter offset i_w ∈ [0,W):
 *      dx[t] += dy[t+i_w] * weight[W-1-i_w]
 *      dw[W-1-i_w] += dy[t+i_w] * x[t]
 *      db += dy[t+i_w]   (only when i_w==0)
 *
 *  SiLU backward:
 *    dy_grad = dy * sigmoid(y) * (1 + y * (1 - sigmoid(y)))
 *
 * Tiling: tasks = numChunks * numBlksD, each core processes chunkPerCore chunks.
 * Each task computes partial dx[BT,BD], dw[W,BD], db[BD]:
 *   dx written directly to output GM,
 *   dw/db written to per-core user workspace, then reduced to output GM after SyncAll.
 *
 * Sync rules:
 *  MTE2→V: MTE2_V sync AFTER DataCopy, BEFORE V ops
 *  V→MTE3: V_MTE3 sync AFTER V ops, BEFORE DataCopy
 *  MTE3→MTE2: MTE3_MTE2 sync BEFORE new MTE2
 *  Consecutive V: PipeBarrier<PIPE_V>
 */

#ifndef ASCENDC_CAUSAL_CONV1D_BWD_H_
#define ASCENDC_CAUSAL_CONV1D_BWD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_bwd_tiling_data.h"
#include "causal_conv1d_bwd_tiling_key.h"

using namespace AscendC;

constexpr uint32_t FP32_DTYPE_SIZE = 4;
constexpr uint32_t BLOCK_ALIGN_NUM = 8;
constexpr uint32_t ACTIVATION_NONE = 0;
constexpr uint32_t ACTIVATION_SILU = 1;
constexpr uint32_t ACTIVATION_SWISH = 2;
constexpr uint32_t INPUT_LAYOUT_BSND = 0;
constexpr uint32_t INPUT_LAYOUT_TND = 1;
constexpr uint32_t INPUT_LAYOUT_BNSD = 2;
constexpr uint32_t INPUT_LAYOUT_NTD = 3;

template <typename inputT, typename calT>
class CausalConv1dBwdKernel {
public:
    __aicore__ inline CausalConv1dBwdKernel() {}
    __aicore__ inline void InitTilingData(const CausalConv1dBwdTilingData *tilingData,
        GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR dy,
        GM_ADDR initial_state, GM_ADDR dht, GM_ADDR queryStartLoc,
        GM_ADDR dx, GM_ADDR dw, GM_ADDR db, GM_ADDR dh0, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void Process();

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        return (a + b - 1) / b;
    }

private:
    __aicore__ inline uint64_t GetInputOffset(uint64_t bos, uint32_t row, uint32_t channel) const;
    __aicore__ inline uint32_t GetInputRowStride(uint32_t channel) const;
    __aicore__ inline uint32_t GetInputSegmentLen(uint32_t channel, uint32_t remain) const;
    __aicore__ inline void CopyInInputTile(GlobalTensor<inputT> &srcGm, LocalTensor<float> dst,
                                           uint64_t bos, uint32_t startRow, uint32_t i_d,
                                           uint32_t totalRows, uint32_t seqLen, bool useGradLayout);
    __aicore__ inline void CopyInX(uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t seqLen);
    __aicore__ inline void CopyInWeight(uint32_t i_d);
    __aicore__ inline void CopyInDy(uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t i_w, uint32_t seqLen);
    __aicore__ inline void CopyInDyWindow(uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t seqLen);
    __aicore__ inline void CopyInYAndSiluBwd(uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t i_w, uint32_t seqLen);
    __aicore__ inline void CopyInInitialStateBlock(uint32_t i_b, uint32_t i_d);
    __aicore__ inline void CopyInDhtBlock(uint32_t i_b, uint32_t i_d);
    __aicore__ inline void ApplySiluBackward(LocalTensor<float> dyLocal);
    __aicore__ inline void AccumulateInitialStateDw(uint32_t i_t, uint32_t i_d, uint32_t i_b, uint32_t seqLen);
    __aicore__ inline void AccumulateDhtDx(uint32_t i_t, uint32_t i_d, uint32_t i_b, uint32_t seqLen);
    __aicore__ inline void ComputeWdyAndAcc(uint32_t i_w, uint32_t dyRowOffset);
    __aicore__ inline void ComputeDwPartial(uint32_t i_w, uint32_t dyRowOffset);
    __aicore__ inline void ComputeDwRowsAccum(uint32_t i_w, uint32_t dyRowOffset);
    __aicore__ inline void FinalizeDwRowsAccum();
    __aicore__ inline void ComputeDbPartial(uint32_t dyRowOffset);
    __aicore__ inline void ComputeDbRowsAccum(uint32_t dyRowOffset);
    __aicore__ inline void FinalizeDbRowsAccum();
    __aicore__ inline void ComputeDh0(uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t i_b, uint32_t seqLen);
    __aicore__ inline void ComputeDwDbForD(uint32_t i_d);
    __aicore__ inline void CopyOutDx(uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t seqLen);
    __aicore__ inline void CopyOutDwDb(uint32_t i_d);
    __aicore__ inline void CopyOutPartialDwDb(uint32_t coreIdx, uint32_t i_d);
    __aicore__ inline void ReducePartialDwDb(uint32_t i_d);
    __aicore__ inline void ReduceRowsInplace(LocalTensor<float> tensor, uint32_t rows, uint32_t cols);
    __aicore__ inline bool ResolveChunk(uint32_t chunkIdx, uint32_t &i_b, uint32_t &i_t,
                                        uint64_t &bos, uint32_t &seqLen);

    TPipe *pipe_;
    TBuf<TPosition::VECIN> xBuf_;
    TBuf<TPosition::VECIN> weightBuf_;
    TBuf<TPosition::VECIN> dyBuf_;
    TBuf<TPosition::VECIN> yBuf_;
    TBuf<TPosition::VECOUT> dxBuf_;
    TBuf<TPosition::VECOUT> dwBuf_;
    TBuf<TPosition::VECOUT> dbBuf_;
    TBuf<TPosition::VECOUT> castBuf_;
    TBuf<TPosition::VECCALC> wdyBuf_;
    TBuf<TPosition::VECCALC> tempBuf_;
    TBuf<TPosition::VECCALC> dwRowsBuf_;
    TBuf<TPosition::VECCALC> dbRowsBuf_;
    TBuf<TPosition::VECCALC> sigmoidBuf_;
    TBuf<TPosition::VECCALC> dh0Buf_;

    uint32_t B_ = 0, T_ = 0, D_ = 0, W_ = 0;
    uint32_t activation_ = ACTIVATION_NONE;
    uint32_t hasWeight_ = 1, hasBias_ = 0;
    uint32_t useInitialState_ = 0;
    uint32_t useFinalState_ = 0;
    uint32_t hasDh0_ = 0;
    uint32_t inputMode_ = 1, totalTokens_ = 0;
    uint32_t inputLayout_ = INPUT_LAYOUT_BSND, inputN_ = 1, inputHeadDim_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t BT_ = 0, BD_ = 0;
    uint32_t numBlksD_ = 0, numChunks_ = 0;
    uint32_t chunkPerCore_ = 0, tailChunk_ = 0;
    uint32_t btBdCount_ = 0, wBdCount_ = 0;
    uint32_t dyRows_ = 0, dyBdCount_ = 0;

    GlobalTensor<inputT> xGm_, yGm_, weightGm_, dyGm_, initialStateGm_, dhtGm_;
    GlobalTensor<inputT> dxGm_, dwGm_, dbGm_, dh0Gm_;
    GlobalTensor<int64_t> queryStartLocGm_;
    GlobalTensor<float> partialDwGm_, partialDbGm_;

    uint32_t srcStrideBytes_ = 0;
    uint32_t dstStrideBytes_ = 0;
    uint32_t blockBytesSrc_ = 0;
    uint32_t blockBytesDst_ = 0;
};

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::InitTilingData(
    const CausalConv1dBwdTilingData *tilingData,
    GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR dy,
    GM_ADDR initial_state, GM_ADDR dht, GM_ADDR queryStartLoc,
    GM_ADDR dx, GM_ADDR dw, GM_ADDR db, GM_ADDR dh0, GM_ADDR workspace)
{
    B_ = tilingData->B;       T_ = tilingData->T;
    D_ = tilingData->D;       W_ = tilingData->W;
    activation_ = tilingData->activation;
    hasWeight_ = tilingData->hasWeight;
    hasBias_ = tilingData->hasBias;
    useInitialState_ = tilingData->useInitialState;
    useFinalState_ = tilingData->useFinalState;
    hasDh0_ = (dh0 != nullptr && useInitialState_) ? 1 : 0;
    inputMode_ = tilingData->inputMode;
    inputLayout_ = tilingData->inputLayout;
    inputN_ = tilingData->inputN;
    inputHeadDim_ = tilingData->inputHeadDim;
    totalTokens_ = tilingData->totalTokens;
    blockNum_ = tilingData->blockNum;
    BT_ = tilingData->BT;     BD_ = tilingData->BD;
    numBlksD_ = tilingData->numBlksD;
    numChunks_ = tilingData->numChunks;
    chunkPerCore_ = tilingData->chunkPerCore;
    tailChunk_ = tilingData->tailChunk;
    btBdCount_ = BT_ * BD_;
    wBdCount_ = W_ * BD_;
    dyRows_ = BT_ + W_ - 1;
    dyBdCount_ = dyRows_ * BD_;

    srcStrideBytes_ = static_cast<uint32_t>(D_ * sizeof(inputT));
    dstStrideBytes_ = static_cast<uint32_t>(BD_ * sizeof(calT));
    blockBytesSrc_  = static_cast<uint32_t>(BD_ * sizeof(inputT));
    blockBytesDst_  = static_cast<uint32_t>(BD_ * sizeof(calT));

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(x));
    weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(weight));
    dyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(dy));
    if (initial_state != nullptr && useInitialState_)
        initialStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(initial_state));
    if (dht != nullptr && useFinalState_)
        dhtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(dht));
    dxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(dx));
    dwGm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(dw));
    dbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(db));
    uint64_t partialDwElems = static_cast<uint64_t>(blockNum_) * numBlksD_ * wBdCount_;
    auto *partialBase = reinterpret_cast<__gm__ float *>(workspace);
    partialDwGm_.SetGlobalBuffer(partialBase, partialDwElems);
    partialDbGm_.SetGlobalBuffer(partialBase + partialDwElems,
                                 static_cast<uint64_t>(blockNum_) * numBlksD_ * BD_);
    if (hasDh0_)
        dh0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(dh0));
    if (y != nullptr)
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(y));
    if (queryStartLoc != nullptr)
        queryStartLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(queryStartLoc));
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::InitBuffer(TPipe *inputPipe)
{
    pipe_ = inputPipe;
    pipe_->InitBuffer(xBuf_, btBdCount_ * FP32_DTYPE_SIZE);
    pipe_->InitBuffer(dyBuf_, dyBdCount_ * FP32_DTYPE_SIZE);
    pipe_->InitBuffer(dxBuf_, btBdCount_ * FP32_DTYPE_SIZE);
    uint32_t calcCount = (btBdCount_ > wBdCount_) ? btBdCount_ : wBdCount_;
    if constexpr (!IsSameType<inputT, float>::value) {
        pipe_->InitBuffer(castBuf_, calcCount * sizeof(inputT));
    }
    pipe_->InitBuffer(wdyBuf_, calcCount * FP32_DTYPE_SIZE);
    pipe_->InitBuffer(tempBuf_, calcCount * FP32_DTYPE_SIZE);
    if (hasWeight_) {
        pipe_->InitBuffer(weightBuf_, wBdCount_ * FP32_DTYPE_SIZE);
        pipe_->InitBuffer(dwBuf_, wBdCount_ * FP32_DTYPE_SIZE);
        if (activation_ == ACTIVATION_NONE) {
            pipe_->InitBuffer(dwRowsBuf_, W_ * btBdCount_ * FP32_DTYPE_SIZE);
        }
    }
    if (hasBias_) {
        pipe_->InitBuffer(dbBuf_, BD_ * FP32_DTYPE_SIZE);
        if (activation_ == ACTIVATION_NONE) {
            pipe_->InitBuffer(dbRowsBuf_, btBdCount_ * FP32_DTYPE_SIZE);
        }
    }
    if (activation_ == ACTIVATION_SILU || activation_ == ACTIVATION_SWISH) {
        pipe_->InitBuffer(yBuf_, btBdCount_ * FP32_DTYPE_SIZE);
        pipe_->InitBuffer(sigmoidBuf_, btBdCount_ * FP32_DTYPE_SIZE);
    }
    if (useInitialState_ || useFinalState_)
        pipe_->InitBuffer(dh0Buf_, BD_ * FP32_DTYPE_SIZE);
}

template <typename inputT, typename calT>
__aicore__ inline uint64_t CausalConv1dBwdKernel<inputT, calT>::GetInputOffset(
    uint64_t bos, uint32_t row, uint32_t channel) const
{
    if (inputLayout_ == INPUT_LAYOUT_BNSD) {
        uint64_t batch = bos / T_;
        uint32_t n = channel / inputHeadDim_;
        uint32_t d = channel - n * inputHeadDim_;
        return ((batch * inputN_ + n) * T_ + row) * inputHeadDim_ + d;
    }
    if (inputLayout_ == INPUT_LAYOUT_NTD) {
        uint32_t n = channel / inputHeadDim_;
        uint32_t d = channel - n * inputHeadDim_;
        return (static_cast<uint64_t>(n) * totalTokens_ + bos + row) * inputHeadDim_ + d;
    }
    return (bos + row) * D_ + channel;
}

template <typename inputT, typename calT>
__aicore__ inline uint32_t CausalConv1dBwdKernel<inputT, calT>::GetInputRowStride(uint32_t channel) const
{
    (void)channel;
    if (inputLayout_ == INPUT_LAYOUT_BNSD || inputLayout_ == INPUT_LAYOUT_NTD) {
        return inputHeadDim_;
    }
    return D_;
}

template <typename inputT, typename calT>
__aicore__ inline uint32_t CausalConv1dBwdKernel<inputT, calT>::GetInputSegmentLen(
    uint32_t channel, uint32_t remain) const
{
    if (inputLayout_ == INPUT_LAYOUT_BNSD || inputLayout_ == INPUT_LAYOUT_NTD) {
        uint32_t headRemain = inputHeadDim_ - (channel % inputHeadDim_);
        return (remain < headRemain) ? remain : headRemain;
    }
    return remain;
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyInInputTile(
    GlobalTensor<inputT> &srcGm, LocalTensor<float> dst, uint64_t bos, uint32_t startRow,
    uint32_t i_d, uint32_t totalRows, uint32_t seqLen, bool useGradLayout)
{
    uint32_t validRows = (startRow + totalRows <= seqLen) ? totalRows :
                         ((startRow < seqLen) ? (seqLen - startRow) : 0);
    bool fullTile = (validRows == totalRows);
    uint32_t totalCount = totalRows * BD_;
    if (!fullTile) {
        Duplicate(dst, float(0), totalCount);
        if constexpr (!IsSameType<inputT, float>::value) {
            LocalTensor<inputT> srcTemp = tempBuf_.Get<inputT>();
            Duplicate(srcTemp, static_cast<inputT>(0), totalCount);
        }
        PipeBarrier<PIPE_V>();
    }
    if (validRows == 0) {
        return;
    }

    event_t vMte2Ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(vMte2Ev);
    WaitFlag<HardEvent::V_MTE2>(vMte2Ev);

    DataCopyPadExtParams<inputT> padParams{false, 0, 0, static_cast<inputT>(0)};
    if (useGradLayout &&
        (inputLayout_ == INPUT_LAYOUT_BNSD || inputLayout_ == INPUT_LAYOUT_NTD)) {
        if constexpr (IsSameType<inputT, float>::value) {
            for (uint32_t row = 0; row < validRows; row++) {
                uint32_t channel = i_d * BD_;
                uint32_t remain = BD_;
                uint32_t localOffset = row * BD_;
                while (remain > 0) {
                    uint32_t segLen = GetInputSegmentLen(channel, remain);
                    DataCopyExtParams copyParams{
                        1,
                        static_cast<uint32_t>(segLen * sizeof(inputT)),
                        0,
                        0,
                        0};
                    uint64_t base = GetInputOffset(bos, startRow + row, channel);
                    DataCopyPad(dst[localOffset], srcGm[base], copyParams, padParams);
                    channel += segLen;
                    localOffset += segLen;
                    remain -= segLen;
                }
            }
            event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(ev);
            WaitFlag<HardEvent::MTE2_V>(ev);
        } else {
            LocalTensor<inputT> srcTemp = tempBuf_.Get<inputT>();
            for (uint32_t row = 0; row < validRows; row++) {
                uint32_t channel = i_d * BD_;
                uint32_t remain = BD_;
                uint32_t localOffset = row * BD_;
                while (remain > 0) {
                    uint32_t segLen = GetInputSegmentLen(channel, remain);
                    DataCopyExtParams copyParams{
                        1,
                        static_cast<uint32_t>(segLen * sizeof(inputT)),
                        0,
                        0,
                        0};
                    uint64_t base = GetInputOffset(bos, startRow + row, channel);
                    DataCopyPad(srcTemp[localOffset], srcGm[base], copyParams, padParams);
                    channel += segLen;
                    localOffset += segLen;
                    remain -= segLen;
                }
            }
            event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(ev);
            WaitFlag<HardEvent::MTE2_V>(ev);
            Cast(dst, srcTemp, RoundMode::CAST_NONE, totalCount);
            PipeBarrier<PIPE_V>();
        }
        return;
    }

    uint32_t channel = i_d * BD_;
    uint32_t remain = BD_;
    uint32_t localOffset = 0;
    if constexpr (IsSameType<inputT, float>::value) {
        while (remain > 0) {
            uint32_t segLen = useGradLayout ? GetInputSegmentLen(channel, remain) : remain;
            uint32_t rowStride = useGradLayout ? GetInputRowStride(channel) : D_;
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(validRows),
                static_cast<uint32_t>(segLen * sizeof(inputT)),
                static_cast<uint32_t>((rowStride - segLen) * sizeof(inputT)),
                static_cast<uint32_t>((BD_ - segLen) * sizeof(float)),
                0};
            uint64_t base = useGradLayout ? GetInputOffset(bos, startRow, channel)
                                          : (bos + startRow) * D_ + channel;
            DataCopyPad(dst[localOffset], srcGm[base], copyParams, padParams);
            channel += segLen;
            localOffset += segLen;
            remain -= segLen;
        }
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
    } else {
        LocalTensor<inputT> srcTemp = tempBuf_.Get<inputT>();
        while (remain > 0) {
            uint32_t segLen = useGradLayout ? GetInputSegmentLen(channel, remain) : remain;
            uint32_t rowStride = useGradLayout ? GetInputRowStride(channel) : D_;
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(validRows),
                static_cast<uint32_t>(segLen * sizeof(inputT)),
                static_cast<uint32_t>((rowStride - segLen) * sizeof(inputT)),
                static_cast<uint32_t>((BD_ - segLen) * sizeof(inputT)),
                0};
            uint64_t base = useGradLayout ? GetInputOffset(bos, startRow, channel)
                                          : (bos + startRow) * D_ + channel;
            DataCopyPad(srcTemp[localOffset], srcGm[base], copyParams, padParams);
            channel += segLen;
            localOffset += segLen;
            remain -= segLen;
        }
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
        Cast(dst, srcTemp, RoundMode::CAST_NONE, totalCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyInX(
    uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t seqLen)
{
    CopyInInputTile(xGm_, xBuf_.Get<float>(), bos, i_t * BT_, i_d, BT_, seqLen, false);
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyInWeight(uint32_t i_d)
{
    if (!hasWeight_) return;
    LocalTensor<float> wLocal = weightBuf_.Get<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(W_), blockBytesSrc_,
        static_cast<uint32_t>((D_ - BD_) * sizeof(inputT)), 0, 0};
    DataCopyPadExtParams<inputT> padParams{false, 0, 0, static_cast<inputT>(0)};
    uint64_t off = static_cast<uint64_t>(i_d) * BD_;
    event_t vMte2Ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(vMte2Ev);
    WaitFlag<HardEvent::V_MTE2>(vMte2Ev);
    if constexpr (IsSameType<inputT, float>::value) {
        DataCopyPad(wLocal, weightGm_[off], copyParams, padParams);
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
    } else {
        LocalTensor<inputT> wTemp = tempBuf_.Get<inputT>();
        DataCopyPad(wTemp, weightGm_[off], copyParams, padParams);
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
        Cast(wLocal, wTemp, RoundMode::CAST_NONE, wBdCount_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyInDy(
    uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t i_w, uint32_t seqLen)
{
    CopyInInputTile(dyGm_, dyBuf_.Get<float>(), bos, i_t * BT_ + i_w, i_d, BT_, seqLen, true);
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyInDyWindow(
    uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t seqLen)
{
    CopyInInputTile(dyGm_, dyBuf_.Get<float>(), bos, i_t * BT_, i_d, dyRows_, seqLen, true);
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyInYAndSiluBwd(
    uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t i_w, uint32_t seqLen)
{
    CopyInInputTile(yGm_, yBuf_.Get<float>(), bos, i_t * BT_ + i_w, i_d, BT_, seqLen, true);
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyInInitialStateBlock(
    uint32_t i_b, uint32_t i_d)
{
    LocalTensor<float> stateBlock = tempBuf_.Get<float>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(W_),
        static_cast<uint32_t>(BD_ * sizeof(inputT)),
        static_cast<uint32_t>((D_ - BD_) * sizeof(inputT)),
        0,
        0};
    DataCopyPadExtParams<inputT> padParams{false, 0, 0, static_cast<inputT>(0)};
    uint64_t off = static_cast<uint64_t>(i_b) * W_ * D_ +
                   static_cast<uint64_t>(i_d) * BD_;

    event_t vMte2Ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(vMte2Ev);
    WaitFlag<HardEvent::V_MTE2>(vMte2Ev);
    if constexpr (IsSameType<inputT, float>::value) {
        DataCopyPad(stateBlock, initialStateGm_[off], copyParams, padParams);
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
    } else {
        LocalTensor<inputT> stateIn = castBuf_.Get<inputT>();
        DataCopyPad(stateIn, initialStateGm_[off], copyParams, padParams);
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
        Cast(stateBlock, stateIn, RoundMode::CAST_NONE, BD_ * W_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyInDhtBlock(
    uint32_t i_b, uint32_t i_d)
{
    LocalTensor<float> stateBlock = tempBuf_.Get<float>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(W_),
        static_cast<uint32_t>(BD_ * sizeof(inputT)),
        static_cast<uint32_t>((D_ - BD_) * sizeof(inputT)),
        0,
        0};
    DataCopyPadExtParams<inputT> padParams{false, 0, 0, static_cast<inputT>(0)};
    uint64_t off = static_cast<uint64_t>(i_b) * W_ * D_ +
                   static_cast<uint64_t>(i_d) * BD_;

    event_t vMte2Ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(vMte2Ev);
    WaitFlag<HardEvent::V_MTE2>(vMte2Ev);
    if constexpr (IsSameType<inputT, float>::value) {
        DataCopyPad(stateBlock, dhtGm_[off], copyParams, padParams);
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
    } else {
        LocalTensor<inputT> stateIn = castBuf_.Get<inputT>();
        DataCopyPad(stateIn, dhtGm_[off], copyParams, padParams);
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
        Cast(stateBlock, stateIn, RoundMode::CAST_NONE, BD_ * W_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ApplySiluBackward(LocalTensor<float> dyLocal)
{
    LocalTensor<float> yLocal = yBuf_.Get<float>();
    LocalTensor<float> sigmoidLocal = sigmoidBuf_.Get<float>();
    LocalTensor<float> t = tempBuf_.Get<float>();

    Muls(sigmoidLocal, yLocal, float(-1.0), btBdCount_);
    PipeBarrier<PIPE_V>();
    Exp(sigmoidLocal, sigmoidLocal, btBdCount_);
    PipeBarrier<PIPE_V>();
    Adds(sigmoidLocal, sigmoidLocal, float(1.0), btBdCount_);
    PipeBarrier<PIPE_V>();
    Duplicate(t, float(1.0), btBdCount_);
    PipeBarrier<PIPE_V>();
    Div(sigmoidLocal, t, sigmoidLocal, btBdCount_);
    PipeBarrier<PIPE_V>();

    Sub(t, t, sigmoidLocal, btBdCount_);
    PipeBarrier<PIPE_V>();
    Mul(t, t, yLocal, btBdCount_);
    PipeBarrier<PIPE_V>();
    Adds(t, t, float(1.0), btBdCount_);
    PipeBarrier<PIPE_V>();
    Mul(dyLocal, dyLocal, sigmoidLocal, btBdCount_);
    PipeBarrier<PIPE_V>();
    Mul(dyLocal, dyLocal, t, btBdCount_);
    PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::AccumulateInitialStateDw(
    uint32_t i_t, uint32_t i_d, uint32_t i_b, uint32_t seqLen)
{
    if (!useInitialState_ || !hasWeight_ || W_ <= 1) return;
    uint32_t startRow = i_t * BT_;
    if (startRow >= W_ - 1 || startRow >= seqLen) return;

    uint32_t validRows = (startRow + BT_ <= seqLen) ? BT_ : (seqLen - startRow);
    CopyInInitialStateBlock(i_b, i_d);

    LocalTensor<float> dyLocal = dyBuf_.Get<float>();
    LocalTensor<float> stateTrans = tempBuf_.Get<float>();
    LocalTensor<float> prod = dh0Buf_.Get<float>();

    for (uint32_t i_w = 1; i_w < W_; i_w++) {
        uint32_t wIdx = W_ - i_w - 1;
        for (uint32_t r = 0; r < validRows; r++) {
            uint32_t absRow = startRow + r;
            if (absRow >= i_w) {
                break;
            }
            uint32_t slot = W_ - i_w + absRow;
            Mul(prod, dyLocal[r * BD_], stateTrans[slot * BD_], BD_);
            PipeBarrier<PIPE_V>();
            if (activation_ == ACTIVATION_NONE) {
                LocalTensor<float> dwRows = dwRowsBuf_.Get<float>();
                LocalTensor<float> dst = dwRows[wIdx * btBdCount_ + r * BD_];
                Add(dst, dst, prod, BD_);
            } else {
                LocalTensor<float> dwLocal = dwBuf_.Get<float>();
                LocalTensor<float> dst = dwLocal[wIdx * BD_];
                Add(dst, dst, prod, BD_);
            }
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::AccumulateDhtDx(
    uint32_t i_t, uint32_t i_d, uint32_t i_b, uint32_t seqLen)
{
    if (!useFinalState_ || W_ <= 1) return;
    uint32_t startRow = i_t * BT_;
    if (startRow >= seqLen) return;

    uint32_t validRows = (startRow + BT_ <= seqLen) ? BT_ : (seqLen - startRow);
    uint32_t tailStart = (seqLen > W_ - 1) ? (seqLen - (W_ - 1)) : 0;
    if (startRow + validRows <= tailStart) return;

    CopyInDhtBlock(i_b, i_d);

    LocalTensor<float> dxLocal = dxBuf_.Get<float>();
    LocalTensor<float> stateTrans = tempBuf_.Get<float>();
    for (uint32_t r = 0; r < validRows; r++) {
        uint32_t absRow = startRow + r;
        if (absRow < tailStart) {
            continue;
        }
        uint32_t slot = 1 + absRow - tailStart;
        if (slot >= W_) {
            continue;
        }
        LocalTensor<float> dst = dxLocal[r * BD_];
        Add(dst, dst, stateTrans[slot * BD_], BD_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ComputeWdyAndAcc(
    uint32_t i_w, uint32_t dyRowOffset)
{
    LocalTensor<float> dyLocal = dyBuf_.Get<float>()[dyRowOffset * BD_];
    LocalTensor<float> dxLocal = dxBuf_.Get<float>();
    LocalTensor<float> wdyLocal = wdyBuf_.Get<float>();

    if (activation_ == ACTIVATION_SILU || activation_ == ACTIVATION_SWISH) {
        ApplySiluBackward(dyLocal);
    }

    uint32_t wIdx = W_ - i_w - 1;
    if (hasWeight_) {
        LocalTensor<float> wLocal = weightBuf_.Get<float>();
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = static_cast<uint16_t>(BD_ / BLOCK_ALIGN_NUM);
        repeatParams.src0RepStride = static_cast<uint16_t>(BD_ / BLOCK_ALIGN_NUM);
        repeatParams.src1RepStride = 0;
        Mul(wdyLocal, dyLocal, wLocal[wIdx * BD_], static_cast<uint64_t>(BD_),
            static_cast<uint8_t>(BT_), repeatParams);
        PipeBarrier<PIPE_V>();
    } else {
        wdyLocal = dyLocal;
    }

    Add(dxLocal, dxLocal, wdyLocal, btBdCount_);
    PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ReduceRowsInplace(
    LocalTensor<float> tensor, uint32_t rows, uint32_t cols)
{
    uint32_t remainRows = rows;
    while (remainRows > 1) {
        uint32_t upperRows = (remainRows + 1) >> 1;
        uint32_t pairRows = remainRows >> 1;
        Add(tensor, tensor, tensor[upperRows * cols], pairRows * cols);
        PipeBarrier<PIPE_V>();
        remainRows = upperRows;
    }
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ComputeDwPartial(
    uint32_t i_w, uint32_t dyRowOffset)
{
    if (!hasWeight_) return;
    LocalTensor<float> dyLocal = dyBuf_.Get<float>()[dyRowOffset * BD_];
    LocalTensor<float> xLocal = xBuf_.Get<float>();
    LocalTensor<float> t = tempBuf_.Get<float>();
    LocalTensor<float> dwLocal = dwBuf_.Get<float>();

    Mul(t, dyLocal, xLocal, btBdCount_);
    PipeBarrier<PIPE_V>();
    ReduceRowsInplace(t, BT_, BD_);

    uint32_t wIdx = W_ - i_w - 1;
    Add(dwLocal[wIdx * BD_], dwLocal[wIdx * BD_], t, BD_);
    PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ComputeDwRowsAccum(
    uint32_t i_w, uint32_t dyRowOffset)
{
    if (!hasWeight_) return;
    LocalTensor<float> dyLocal = dyBuf_.Get<float>()[dyRowOffset * BD_];
    LocalTensor<float> xLocal = xBuf_.Get<float>();
    LocalTensor<float> t = tempBuf_.Get<float>();
    LocalTensor<float> rows = dwRowsBuf_.Get<float>();

    Mul(t, dyLocal, xLocal, btBdCount_);
    PipeBarrier<PIPE_V>();

    uint32_t wIdx = W_ - i_w - 1;
    Add(rows[wIdx * btBdCount_], rows[wIdx * btBdCount_], t, btBdCount_);
    PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::FinalizeDwRowsAccum()
{
    if (!hasWeight_) return;
    LocalTensor<float> rows = dwRowsBuf_.Get<float>();
    LocalTensor<float> dwLocal = dwBuf_.Get<float>();
    Duplicate(dwLocal, float(0), wBdCount_);
    PipeBarrier<PIPE_V>();

    for (uint32_t w = 0; w < W_; w++) {
        LocalTensor<float> rowBase = rows[w * btBdCount_];
        ReduceRowsInplace(rowBase, BT_, BD_);
        Add(dwLocal[w * BD_], dwLocal[w * BD_], rowBase, BD_);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ComputeDbPartial(uint32_t dyRowOffset)
{
    if (!hasBias_) return;
    LocalTensor<float> dyLocal = dyBuf_.Get<float>()[dyRowOffset * BD_];
    LocalTensor<float> dbLocal = dbBuf_.Get<float>();

    ReduceRowsInplace(dyLocal, BT_, BD_);
    Add(dbLocal, dbLocal, dyLocal, BD_);
    PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ComputeDbRowsAccum(uint32_t dyRowOffset)
{
    if (!hasBias_) return;
    LocalTensor<float> dyLocal = dyBuf_.Get<float>()[dyRowOffset * BD_];
    LocalTensor<float> dbRows = dbRowsBuf_.Get<float>();

    Add(dbRows, dbRows, dyLocal, btBdCount_);
    PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::FinalizeDbRowsAccum()
{
    if (!hasBias_) return;
    LocalTensor<float> dbRows = dbRowsBuf_.Get<float>();
    LocalTensor<float> dbLocal = dbBuf_.Get<float>();
    Duplicate(dbLocal, float(0), BD_);
    PipeBarrier<PIPE_V>();

    ReduceRowsInplace(dbRows, BT_, BD_);
    Add(dbLocal, dbLocal, dbRows, BD_);
    PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ComputeDh0(
    uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t i_b, uint32_t seqLen)
{
    if (!useInitialState_ || W_ <= 1) return;
    uint32_t startRow = i_t * BT_;
    if (startRow >= W_ - 1 || startRow >= seqLen) return;

    CopyInDy(bos, i_t, i_d, 0, seqLen);
    if (activation_ == ACTIVATION_SILU || activation_ == ACTIVATION_SWISH) {
        CopyInYAndSiluBwd(bos, i_t, i_d, 0, seqLen);
    }

    LocalTensor<float> dy0Local = dyBuf_.Get<float>();
    if (activation_ == ACTIVATION_SILU || activation_ == ACTIVATION_SWISH) {
        ApplySiluBackward(dy0Local);
    }

    AccumulateInitialStateDw(i_t, i_d, i_b, seqLen);
    if (!hasDh0_ || startRow != 0) return;

    LocalTensor<float> wLocal;
    if (hasWeight_) {
        wLocal = weightBuf_.Get<float>();
    }

    LocalTensor<float> dh0Src = wdyBuf_.Get<float>();
    LocalTensor<float> temp = tempBuf_.Get<float>();
    for (uint32_t i_w = 0; i_w < W_; i_w++) {
        LocalTensor<float> dh0Acc = dh0Src[i_w * BD_];
        Duplicate(dh0Acc, float(0), BD_);
        PipeBarrier<PIPE_V>();

        uint32_t validRows = (BT_ <= seqLen) ? BT_ : seqLen;
        uint32_t maxRow = (validRows < i_w) ? validRows : i_w;

        for (uint32_t r = 0; r < maxRow; r++) {
            uint32_t wIdx = i_w - 1 - r;
            if (wIdx < W_) {
                if (hasWeight_) {
                    Mul(temp, dy0Local[r * BD_], wLocal[wIdx * BD_], BD_);
                    PipeBarrier<PIPE_V>();
                    Add(dh0Acc, dh0Acc, temp, BD_);
                    PipeBarrier<PIPE_V>();
                } else {
                    Add(dh0Acc, dh0Acc, dy0Local[r * BD_], BD_);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }

    uint64_t dh0Base = static_cast<uint64_t>(i_b) * W_ * D_ +
                       static_cast<uint64_t>(i_d) * BD_;
    uint32_t stateRowStrideBytes = static_cast<uint32_t>((D_ - BD_) * sizeof(inputT));
    if constexpr (IsSameType<inputT, float>::value) {
        DataCopyExtParams outParams{
            static_cast<uint16_t>(W_),
            static_cast<uint32_t>(BD_ * FP32_DTYPE_SIZE),
            0,
            stateRowStrideBytes,
            0};
        event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        DataCopyPad(dh0Gm_[dh0Base], dh0Src, outParams);
    } else {
        LocalTensor<inputT> dh0Out = castBuf_.Get<inputT>();
        Cast(dh0Out, dh0Src, RoundMode::CAST_RINT, BD_ * W_);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams outParams{
            static_cast<uint16_t>(W_),
            static_cast<uint32_t>(BD_ * sizeof(inputT)),
            0,
            stateRowStrideBytes,
            0};
        event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        DataCopyPad(dh0Gm_[dh0Base], dh0Out, outParams);
    }
    event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyOutDx(
    uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t seqLen)
{
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(ev);
    WaitFlag<HardEvent::V_MTE3>(ev);

    LocalTensor<float> dl = dxBuf_.Get<float>();
    uint64_t off = bos * D_ + i_t * BT_ * D_ + i_d * BD_;
    uint64_t startRow = static_cast<uint64_t>(i_t) * BT_;
    uint32_t validRows = (startRow + BT_ <= seqLen) ? BT_ : ((startRow < seqLen) ? (seqLen - startRow) : 0);
    if (validRows == 0) {
        return;
    }
    uint32_t rowStrideBytes = static_cast<uint32_t>((D_ - BD_) * sizeof(inputT));
    if constexpr (IsSameType<inputT, float>::value) {
        DataCopyExtParams outParams{static_cast<uint16_t>(validRows), blockBytesDst_, 0, rowStrideBytes, 0};
        DataCopyPad(dxGm_[off], dl, outParams);
        event_t evMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(evMte3V);
        WaitFlag<HardEvent::MTE3_V>(evMte3V);
    } else {
        LocalTensor<inputT> dlIn = castBuf_.Get<inputT>();
        DataCopyExtParams outParams{
            static_cast<uint16_t>(validRows),
            static_cast<uint32_t>(BD_ * sizeof(inputT)),
            0,
            rowStrideBytes,
            0};
        Cast(dlIn, dl, RoundMode::CAST_RINT, validRows * BD_);
        PipeBarrier<PIPE_V>();
        event_t evMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evMte3);
        WaitFlag<HardEvent::V_MTE3>(evMte3);
        DataCopyPad(dxGm_[off], dlIn, outParams);
        event_t evMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(evMte3V);
        WaitFlag<HardEvent::MTE3_V>(evMte3V);
    }
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyOutDwDb(uint32_t i_d)
{
    if (!hasWeight_ && !hasBias_) return;

    if constexpr (IsSameType<inputT, float>::value) {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ev);
        WaitFlag<HardEvent::V_MTE3>(ev);

        if (hasWeight_) {
            LocalTensor<float> dwLocal = dwBuf_.Get<float>();
            DataCopyExtParams dwOutParams{
                static_cast<uint16_t>(W_),
                static_cast<uint32_t>(BD_ * sizeof(inputT)),
                0,
                static_cast<uint32_t>((D_ - BD_) * sizeof(inputT)),
                0};
            uint64_t dstOff = static_cast<uint64_t>(i_d) * BD_;
            DataCopyPad(dwGm_[dstOff], dwLocal, dwOutParams);
        }

        if (hasBias_) {
            LocalTensor<float> dbLocal = dbBuf_.Get<float>();
            uint64_t dbOff = static_cast<uint64_t>(i_d) * BD_;
            DataCopyExtParams dbParams{
                1, static_cast<uint32_t>(BD_ * sizeof(inputT)), 0, 0, 0};
            DataCopyPad(dbGm_[dbOff], dbLocal, dbParams);
        }

        event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(mte3ToV);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV);
    } else {
        LocalTensor<inputT> outputLocal = castBuf_.Get<inputT>();
        if (hasWeight_) {
            Cast(outputLocal, dwBuf_.Get<float>(), RoundMode::CAST_RINT, wBdCount_);
            PipeBarrier<PIPE_V>();
            event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ev);
            WaitFlag<HardEvent::V_MTE3>(ev);

            DataCopyExtParams dwOutParams{
                static_cast<uint16_t>(W_),
                static_cast<uint32_t>(BD_ * sizeof(inputT)),
                0,
                static_cast<uint32_t>((D_ - BD_) * sizeof(inputT)),
                0};
            uint64_t dstOff = static_cast<uint64_t>(i_d) * BD_;
            DataCopyPad(dwGm_[dstOff], outputLocal, dwOutParams);

            event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(mte3ToV);
            WaitFlag<HardEvent::MTE3_V>(mte3ToV);
        }

        if (hasBias_) {
            Cast(outputLocal, dbBuf_.Get<float>(), RoundMode::CAST_RINT, BD_);
            PipeBarrier<PIPE_V>();
            event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ev);
            WaitFlag<HardEvent::V_MTE3>(ev);

            uint64_t dbOff = static_cast<uint64_t>(i_d) * BD_;
            DataCopyExtParams dbParams{
                1, static_cast<uint32_t>(BD_ * sizeof(inputT)), 0, 0, 0};
            DataCopyPad(dbGm_[dbOff], outputLocal, dbParams);

            event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(mte3ToV);
            WaitFlag<HardEvent::MTE3_V>(mte3ToV);
        }
    }
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::CopyOutPartialDwDb(uint32_t coreIdx, uint32_t i_d)
{
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(ev);
    WaitFlag<HardEvent::V_MTE3>(ev);

    uint64_t partialIdx = (static_cast<uint64_t>(coreIdx) * numBlksD_ + i_d);
    if (hasWeight_) {
        LocalTensor<float> dwLocal = dwBuf_.Get<float>();
        DataCopy(partialDwGm_[partialIdx * wBdCount_], dwLocal, wBdCount_);
    }
    if (hasBias_) {
        LocalTensor<float> dbLocal = dbBuf_.Get<float>();
        DataCopy(partialDbGm_[partialIdx * BD_], dbLocal, BD_);
    }

    event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ReducePartialDwDb(uint32_t i_d)
{
    LocalTensor<float> tmp = tempBuf_.Get<float>();
    if (hasWeight_) {
        LocalTensor<float> dwLocal = dwBuf_.Get<float>();
        Duplicate(dwLocal, float(0), wBdCount_);
        PipeBarrier<PIPE_V>();
        for (uint32_t core = 0; core < blockNum_; core++) {
            uint64_t partialIdx = (static_cast<uint64_t>(core) * numBlksD_ + i_d) * wBdCount_;
            event_t vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            DataCopy(tmp, partialDwGm_[partialIdx], wBdCount_);
            event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            Add(dwLocal, dwLocal, tmp, wBdCount_);
            PipeBarrier<PIPE_V>();
        }
    }

    if (hasBias_) {
        LocalTensor<float> dbLocal = dbBuf_.Get<float>();
        Duplicate(dbLocal, float(0), BD_);
        PipeBarrier<PIPE_V>();
        for (uint32_t core = 0; core < blockNum_; core++) {
            uint64_t partialIdx = (static_cast<uint64_t>(core) * numBlksD_ + i_d) * BD_;
            event_t vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            DataCopy(tmp, partialDbGm_[partialIdx], BD_);
            event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            Add(dbLocal, dbLocal, tmp, BD_);
            PipeBarrier<PIPE_V>();
        }
    }

    CopyOutDwDb(i_d);
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::ComputeDwDbForD(uint32_t i_d)
{
    if (!hasWeight_ && !hasBias_) return;

    uint32_t ntPerSeq = CeilDiv(T_, BT_);
    if (hasWeight_) {
        CopyInWeight(i_d);
        Duplicate(dwBuf_.Get<float>(), float(0), wBdCount_);
        PipeBarrier<PIPE_V>();
    }
    if (hasBias_) {
        Duplicate(dbBuf_.Get<float>(), float(0), BD_);
        PipeBarrier<PIPE_V>();
    }
    if (activation_ == ACTIVATION_SILU || activation_ == ACTIVATION_SWISH) {
        Duplicate(dxBuf_.Get<float>(), float(0), btBdCount_);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t chunkIdx = 0; chunkIdx < numChunks_; chunkIdx++) {
        uint32_t i_b = chunkIdx / ntPerSeq;
        uint32_t i_t = chunkIdx % ntPerSeq;
        uint64_t bos = static_cast<uint64_t>(i_b) * T_;

        if (hasWeight_) {
            event_t vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            CopyInX(bos, i_t, i_d, T_);
        }

        for (uint32_t i_w = 0; i_w < W_; i_w++) {
            CopyInDy(bos, i_t, i_d, i_w, T_);
            if (activation_ == ACTIVATION_SILU || activation_ == ACTIVATION_SWISH) {
                CopyInYAndSiluBwd(bos, i_t, i_d, i_w, T_);
                event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(e);
                WaitFlag<HardEvent::MTE2_V>(e);
                ComputeWdyAndAcc(i_w, 0);
            }
            ComputeDwPartial(i_w, 0);
            if (hasBias_ && i_w == 0) ComputeDbPartial(0);
        }
    }

    CopyOutDwDb(i_d);
}

template <typename inputT, typename calT>
__aicore__ inline bool CausalConv1dBwdKernel<inputT, calT>::ResolveChunk(
    uint32_t chunkIdx, uint32_t &i_b, uint32_t &i_t, uint64_t &bos, uint32_t &seqLen)
{
    if (inputMode_ != 0) {
        uint32_t ntPerSeq = CeilDiv(T_, BT_);
        i_b = chunkIdx / ntPerSeq;
        i_t = chunkIdx % ntPerSeq;
        bos = static_cast<uint64_t>(i_b) * T_;
        seqLen = T_;
        return i_b < B_;
    }

    if (B_ == 1) {
        i_b = 0;
        i_t = chunkIdx;
        bos = 0;
        seqLen = totalTokens_;
        return chunkIdx < CeilDiv(seqLen, BT_);
    }

    uint32_t chunkBase = 0;
    for (uint32_t seq = 0; seq < B_; seq++) {
        int64_t start = queryStartLocGm_.GetValue(seq);
        int64_t end = queryStartLocGm_.GetValue(seq + 1);
        uint32_t curLen = static_cast<uint32_t>(end - start);
        uint32_t curChunks = CeilDiv(curLen, BT_);
        if (chunkIdx < chunkBase + curChunks) {
            i_b = seq;
            i_t = chunkIdx - chunkBase;
            bos = static_cast<uint64_t>(start);
            seqLen = curLen;
            return true;
        }
        chunkBase += curChunks;
    }
    return false;
}

template <typename inputT, typename calT>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT>::Process()
{
    uint32_t bIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();

    uint32_t loopC = (bIdx < tailChunk_) ? (chunkPerCore_ + 1) : chunkPerCore_;

    for (uint32_t i_d = 0; i_d < numBlksD_; i_d++) {
        if (hasWeight_) {
            CopyInWeight(i_d);
            if (activation_ == ACTIVATION_NONE) {
                Duplicate(dwRowsBuf_.Get<float>(), float(0), W_ * btBdCount_);
            } else {
                Duplicate(dwBuf_.Get<float>(), float(0), wBdCount_);
            }
            PipeBarrier<PIPE_V>();
        }
        if (hasBias_) {
            if (activation_ == ACTIVATION_NONE) {
                Duplicate(dbRowsBuf_.Get<float>(), float(0), btBdCount_);
            } else {
                Duplicate(dbBuf_.Get<float>(), float(0), BD_);
            }
            PipeBarrier<PIPE_V>();
        }

        for (uint32_t loop = 0; loop < loopC; loop++) {
            uint32_t chunkIdx = (bIdx < tailChunk_)
                ? (bIdx * (chunkPerCore_ + 1) + loop)
                : (bIdx * chunkPerCore_ + tailChunk_ + loop);

            if (chunkIdx >= numChunks_) {
                continue;
            }

            uint32_t i_b = 0;
            uint32_t i_t = 0;
            uint32_t seqLen = 0;
            uint64_t bos = 0;
            if (!ResolveChunk(chunkIdx, i_b, i_t, bos, seqLen)) {
                continue;
            }

            CopyInX(bos, i_t, i_d, seqLen);

            Duplicate(dxBuf_.Get<float>(), float(0), btBdCount_);
            PipeBarrier<PIPE_V>();

            if (activation_ == ACTIVATION_SILU || activation_ == ACTIVATION_SWISH) {
                for (uint32_t i_w = 0; i_w < W_; i_w++) {
                    CopyInDy(bos, i_t, i_d, i_w, seqLen);
                    CopyInYAndSiluBwd(bos, i_t, i_d, i_w, seqLen);
                    ComputeWdyAndAcc(i_w, 0);
                    ComputeDwPartial(i_w, 0);
                    if (hasBias_ && i_w == 0) ComputeDbPartial(0);
                }
            } else {
                CopyInDyWindow(bos, i_t, i_d, seqLen);
                for (uint32_t i_w = 0; i_w < W_; i_w++) {
                    ComputeWdyAndAcc(i_w, i_w);
                    ComputeDwRowsAccum(i_w, i_w);
                }
                if (hasBias_) ComputeDbRowsAccum(0);
            }

            ComputeDh0(bos, i_t, i_d, i_b, seqLen);
            AccumulateDhtDx(i_t, i_d, i_b, seqLen);
            CopyOutDx(bos, i_t, i_d, seqLen);
        }
        if (hasWeight_ && activation_ == ACTIVATION_NONE) {
            FinalizeDwRowsAccum();
        }
        if (hasBias_ && activation_ == ACTIVATION_NONE) {
            FinalizeDbRowsAccum();
        }
        CopyOutPartialDwDb(bIdx, i_d);
    }

    SyncAll();

    for (uint32_t i_d = bIdx; i_d < numBlksD_; i_d += blockNum) {
        ReducePartialDwDb(i_d);
    }
}

#endif  // ASCENDC_CAUSAL_CONV1D_BWD_H_
