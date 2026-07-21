/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_vector.h
 * \brief Vector side process for fused prepare_wy_repr_bwd A2/A3.
 */

#ifndef PREPARE_WY_REPR_BWD_VECTOR_H
#define PREPARE_WY_REPR_BWD_VECTOR_H

#include <type_traits>

#include "prepare_wy_repr_bwd_common.h"
#include "catlass/arch/cross_core_sync.hpp"

using namespace AscendC;

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
class PrepareWyReprBwdVectorProcess {
public:
    __aicore__ inline PrepareWyReprBwdVectorProcess(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g, GM_ADDR cuSeqlens,
                                                    GM_ADDR chunkIndices, GM_ADDR workspace, GM_ADDR debugKbg,
                                                    GM_ADDR debugVb, GM_ADDR debugKbeta);
    __aicore__ inline void Init(const GDN::PrepareWyReprBwdTilingData &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    template <typename copyType, bool USE_DATA_COPY_PAD>
    __aicore__ inline void CopyInRows(AscendC::GlobalTensor<copyType> &inputTensor,
                                      AscendC::LocalTensor<float32_t> &dstTensor, uint64_t inputOffset,
                                      uint32_t elements);
    template <typename copyType, bool USE_DATA_COPY_PAD>
    __aicore__ inline void CopyInBetaGRows(AscendC::GlobalTensor<copyType> &inputTensor,
                                           AscendC::LocalTensor<float32_t> &dstTensor, uint64_t inputOffset,
                                           uint32_t elements);
    __aicore__ inline void CopyOutRows(AscendC::GlobalTensor<kType> &outTensor, uint64_t outOffset);
    __aicore__ inline void ProcessVectorTask(const PrepareWyReprBwdTaskInfo &task, uint32_t taskIdx, uint64_t hv,
                                             uint64_t hk, GM_ADDR slotBase);

private:
    GDN::PrepareWyReprBwdTilingData tiling_{};
    uint32_t curSlot_ = 0;
    uint32_t curInputPingPong_ = 0;
    uint32_t curBetaGInputPingPong_ = 0;
    uint32_t curOutputPingPong_ = 0;
    Arch::CrossCoreFlagWithReverse<> vecToCubeFlag_{PREPARE_WY_REPR_BWD_VEC_TO_CUBE_FLAG_READY,
                                                    PREPARE_WY_REPR_BWD_VEC_TO_CUBE_FLAG_REVERSE};
    Arch::CrossCoreFlagWithReverse<> cubeToVecFlag_{PREPARE_WY_REPR_BWD_CUBE_TO_VEC_FLAG_READY,
                                                    PREPARE_WY_REPR_BWD_CUBE_TO_VEC_FLAG_REVERSE};

    GM_ADDR k_ = nullptr;
    GM_ADDR v_ = nullptr;
    GM_ADDR beta_ = nullptr;
    GM_ADDR g_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    GM_ADDR debugKbg_ = nullptr;
    GM_ADDR debugVb_ = nullptr;
    GM_ADDR debugKbeta_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;

    AscendC::GlobalTensor<kType> kTensor_;
    AscendC::GlobalTensor<kType> vTensor_;
    AscendC::GlobalTensor<gType> betaTensor_;
    AscendC::GlobalTensor<gType> gTensor_;
    AscendC::GlobalTensor<kType> debugKbgTensor_;
    AscendC::GlobalTensor<kType> debugVbTensor_;
    AscendC::GlobalTensor<kType> debugKbetaTensor_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> inputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaGInputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaGInputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> brcbFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputFp32_;

    AscendC::LocalTensor<kType> outputBuf_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT];
    AscendC::LocalTensor<float32_t> betaFp32Tensor_;
    AscendC::LocalTensor<float32_t> gFp32Tensor_;
    AscendC::LocalTensor<float32_t> scaleFp32Tensor_;
    AscendC::LocalTensor<float32_t> brcbFp32Tensor_;
    AscendC::LocalTensor<float32_t> inputFp32Tensor_;
    AscendC::LocalTensor<float32_t> outputFp32Tensor_;

    AscendC::GlobalTensor<kType> gmKbg_;
    AscendC::GlobalTensor<kType> gmVb_;
    AscendC::GlobalTensor<kType> gmKbeta_;

    event_t mte2ToVEvent_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t vToMte2Event_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t betaGMte2ToVEvent_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t betaGVToMte2Event_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t vToMte3Event_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t mte3ToVEvent_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    uint32_t subBlockNum_ = 1;
    uint32_t subBlockIdx_ = 0;
    uint64_t keyBase_ = 0;
    uint64_t valueBase_ = 0;
    uint64_t debugLineBase_ = 0;
    uint32_t curRow_ = 0;
    uint32_t curCol_ = 0;
    uint32_t rowTaskIdx_ = 0;
    uint32_t localRowTask_ = 0;
    uint32_t rowOffset_ = 0;
    uint32_t colOffset_ = 0;
    uint32_t inputIdx_ = 0;
    uint32_t betaGInputIdx_ = 0;
    uint32_t outputIdx_ = 0;
    uint32_t eventIdx_ = 0;
    uint8_t repeatStride_ = 0;
};

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::PrepareWyReprBwdVectorProcess(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
    GM_ADDR debugKbg, GM_ADDR debugVb, GM_ADDR debugKbeta)
    : k_(k), v_(v), beta_(beta), g_(g), cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices), workspace_(workspace),
      debugKbg_(debugKbg), debugVb_(debugVb), debugKbeta_(debugKbeta)
{
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::Init(const GDN::PrepareWyReprBwdTilingData &tiling,
                                                                     AscendC::TPipe *pipe)
{
    tiling_ = tiling;
    pipe_ = pipe;
    kTensor_.SetGlobalBuffer((__gm__ kType *)k_);
    vTensor_.SetGlobalBuffer((__gm__ kType *)v_);
    betaTensor_.SetGlobalBuffer((__gm__ gType *)beta_);
    gTensor_.SetGlobalBuffer((__gm__ gType *)g_);
    debugKbgTensor_.SetGlobalBuffer((__gm__ kType *)debugKbg_);
    debugVbTensor_.SetGlobalBuffer((__gm__ kType *)debugVb_);
    debugKbetaTensor_.SetGlobalBuffer((__gm__ kType *)debugKbeta_);

    uint32_t maxRow = static_cast<uint32_t>(tiling_.kVecRow > tiling_.vVecRow ? tiling_.kVecRow : tiling_.vVecRow);
    pipe_->InitBuffer(inputPing_, PREPARE_WY_REPR_BWD_UB_IO_BYTES);
    pipe_->InitBuffer(inputPong_, PREPARE_WY_REPR_BWD_UB_IO_BYTES);
    pipe_->InitBuffer(betaGInputPing_, PrepareWyReprBwdAlign32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(betaGInputPong_, PrepareWyReprBwdAlign32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(outputPing_, PREPARE_WY_REPR_BWD_UB_IO_BYTES);
    pipe_->InitBuffer(outputPong_, PREPARE_WY_REPR_BWD_UB_IO_BYTES);
    pipe_->InitBuffer(betaFp32_, PrepareWyReprBwdAlign32(maxRow * sizeof(float32_t)));
    pipe_->InitBuffer(gFp32_, PrepareWyReprBwdAlign32(maxRow * sizeof(float32_t)));
    pipe_->InitBuffer(scaleFp32_, PrepareWyReprBwdAlign32(maxRow * sizeof(float32_t)));
    pipe_->InitBuffer(brcbFp32_, PrepareWyReprBwdAlign32(maxRow * PREPARE_WY_REPR_BWD_ONE_BLOCK_BYTES));
    pipe_->InitBuffer(inputFp32_, 2 * PREPARE_WY_REPR_BWD_UB_IO_BYTES);
    pipe_->InitBuffer(outputFp32_, 2 * PREPARE_WY_REPR_BWD_UB_IO_BYTES);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
template <typename copyType, bool USE_DATA_COPY_PAD>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyInRows(AscendC::GlobalTensor<copyType> &inputTensor,
                                                                           AscendC::LocalTensor<float32_t> &dstTensor,
                                                                           uint64_t inputOffset, uint32_t elements)
{
    inputIdx_ = curInputPingPong_;
    AscendC::LocalTensor<copyType> inputBuf[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT] = {inputPing_.Get<copyType>(),
                                                                                       inputPong_.Get<copyType>()};
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[inputIdx_]);
    if constexpr (USE_DATA_COPY_PAD) {
        DataCopyPad(inputBuf[inputIdx_], inputTensor[inputOffset],
                    {1, elements * static_cast<uint32_t>(sizeof(copyType)), 0, 0, 0}, {false, 0, 0, 0});
    } else {
        DataCopy(inputBuf[inputIdx_], inputTensor[inputOffset], elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_[inputIdx_]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_[inputIdx_]);

    if constexpr (std::is_same<copyType, float32_t>::value) {
        Adds(dstTensor, inputBuf[inputIdx_], 0.0f, elements);
    } else {
        Cast(dstTensor, inputBuf[inputIdx_], RoundMode::CAST_NONE, elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[inputIdx_]);
    curInputPingPong_ ^= 1U;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
template <typename copyType, bool USE_DATA_COPY_PAD>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyInBetaGRows(
    AscendC::GlobalTensor<copyType> &inputTensor, AscendC::LocalTensor<float32_t> &dstTensor, uint64_t inputOffset,
    uint32_t elements)
{
    betaGInputIdx_ = curBetaGInputPingPong_;
    AscendC::LocalTensor<copyType> inputBuf[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT] = {
        betaGInputPing_.Get<copyType>(), betaGInputPong_.Get<copyType>()};
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[betaGInputIdx_]);
    if constexpr (USE_DATA_COPY_PAD) {
        DataCopyPad(inputBuf[betaGInputIdx_], inputTensor[inputOffset],
                    {1, elements * static_cast<uint32_t>(sizeof(copyType)), 0, 0, 0}, {false, 0, 0, 0});
    } else {
        DataCopy(inputBuf[betaGInputIdx_], inputTensor[inputOffset], elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(betaGMte2ToVEvent_[betaGInputIdx_]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(betaGMte2ToVEvent_[betaGInputIdx_]);

    if constexpr (std::is_same<copyType, float32_t>::value) {
        Adds(dstTensor, inputBuf[betaGInputIdx_], 0.0f, elements);
    } else {
        Cast(dstTensor, inputBuf[betaGInputIdx_], RoundMode::CAST_NONE, elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[betaGInputIdx_]);
    curBetaGInputPingPong_ ^= 1U;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyOutRows(AscendC::GlobalTensor<kType> &outTensor,
                                                                            uint64_t outOffset)
{
    outputIdx_ = curOutputPingPong_;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    if constexpr (std::is_same<kType, float32_t>::value) {
        Adds(outputBuf_[outputIdx_], outputFp32Tensor_, 0.0f, curRow_ * curCol_);
    } else {
        Cast(outputBuf_[outputIdx_], outputFp32Tensor_, RoundMode::CAST_RINT, curRow_ * curCol_);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    DataCopy(outTensor[outOffset], outputBuf_[outputIdx_], curRow_ * curCol_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    curOutputPingPong_ ^= 1U;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ProcessVectorTask(
    const PrepareWyReprBwdTaskInfo &task, uint32_t taskIdx, uint64_t hv, uint64_t hk, GM_ADDR slotBase)
{
    outputBuf_[0] = outputPing_.Get<kType>();
    outputBuf_[1] = outputPong_.Get<kType>();
    betaFp32Tensor_ = betaFp32_.Get<float32_t>();
    gFp32Tensor_ = gFp32_.Get<float32_t>();
    scaleFp32Tensor_ = scaleFp32_.Get<float32_t>();
    brcbFp32Tensor_ = brcbFp32_.Get<float32_t>();
    inputFp32Tensor_ = inputFp32_.Get<float32_t>();
    outputFp32Tensor_ = outputFp32_.Get<float32_t>();

    gmKbg_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.kbgOffset));
    gmVb_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.vbOffset));
    gmKbeta_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.kbetaOffset));

    for (eventIdx_ = 0; eventIdx_ < PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT; ++eventIdx_) {
        mte2ToVEvent_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        vToMte2Event_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        betaGMte2ToVEvent_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        betaGVToMte2Event_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        vToMte3Event_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>());
        mte3ToVEvent_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>());
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[eventIdx_]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[eventIdx_]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx_]);
    }

    subBlockNum_ = AscendC::GetSubBlockNum();
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    keyBase_ = hk * tiling_.T + task.keyBos;
    valueBase_ = hv * tiling_.T + task.valueBos;
    debugLineBase_ =
        (static_cast<uint64_t>(taskIdx) * static_cast<uint64_t>(tiling_.HV) + hv) * static_cast<uint64_t>(CHUNK_SIZE);
    rowTaskIdx_ = 0;

    for (rowOffset_ = 0; rowOffset_ < task.curChunkSize; rowOffset_ += static_cast<uint32_t>(tiling_.kVecRow)) {
        localRowTask_ = rowTaskIdx_++;
        if (localRowTask_ % subBlockNum_ != subBlockIdx_) {
            continue;
        }
        curRow_ = rowOffset_ + static_cast<uint32_t>(tiling_.kVecRow) > task.curChunkSize ?
                      task.curChunkSize - rowOffset_ :
                      static_cast<uint32_t>(tiling_.kVecRow);
        curCol_ = K_DIM;

        CopyInRows<kType, false>(kTensor_, inputFp32Tensor_, (keyBase_ + rowOffset_) * K_DIM, curRow_ * curCol_);
        CopyInBetaGRows<gType, true>(betaTensor_, betaFp32Tensor_, valueBase_ + rowOffset_, curRow_);
        CopyInBetaGRows<gType, true>(gTensor_, gFp32Tensor_, valueBase_ + rowOffset_, curRow_);
        PipeBarrier<PIPE_V>();
        Exp(gFp32Tensor_, gFp32Tensor_, curRow_);
        PipeBarrier<PIPE_V>();
        Mul(scaleFp32Tensor_, betaFp32Tensor_, gFp32Tensor_, curRow_);
        PipeBarrier<PIPE_V>();
        Brcb(brcbFp32Tensor_, scaleFp32Tensor_, static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)), {1, 8});
        PipeBarrier<PIPE_V>();
        repeatStride_ = curCol_ * sizeof(float32_t) / PREPARE_WY_REPR_BWD_ONE_BLOCK_BYTES;
        for (colOffset_ = 0; colOffset_ < curCol_; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(outputFp32Tensor_[colOffset_], inputFp32Tensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        CopyOutRows(gmKbg_, rowOffset_ * K_DIM);
        CopyOutRows(debugKbgTensor_, (debugLineBase_ + rowOffset_) * K_DIM);

        Brcb(brcbFp32Tensor_, betaFp32Tensor_, static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)), {1, 8});
        PipeBarrier<PIPE_V>();
        repeatStride_ = curCol_ * sizeof(float32_t) / PREPARE_WY_REPR_BWD_ONE_BLOCK_BYTES;
        for (colOffset_ = 0; colOffset_ < curCol_; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(outputFp32Tensor_[colOffset_], inputFp32Tensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        CopyOutRows(gmKbeta_, rowOffset_ * K_DIM);
        CopyOutRows(debugKbetaTensor_, (debugLineBase_ + rowOffset_) * K_DIM);
    }

    rowTaskIdx_ = 0;
    for (rowOffset_ = 0; rowOffset_ < task.curChunkSize; rowOffset_ += static_cast<uint32_t>(tiling_.vVecRow)) {
        localRowTask_ = rowTaskIdx_++;
        if (localRowTask_ % subBlockNum_ != subBlockIdx_) {
            continue;
        }
        curRow_ = rowOffset_ + static_cast<uint32_t>(tiling_.vVecRow) > task.curChunkSize ?
                      task.curChunkSize - rowOffset_ :
                      static_cast<uint32_t>(tiling_.vVecRow);
        curCol_ = V_DIM;

        CopyInRows<kType, false>(vTensor_, inputFp32Tensor_, (valueBase_ + rowOffset_) * V_DIM, curRow_ * curCol_);
        CopyInBetaGRows<gType, true>(betaTensor_, betaFp32Tensor_, valueBase_ + rowOffset_, curRow_);
        PipeBarrier<PIPE_V>();
        Brcb(brcbFp32Tensor_, betaFp32Tensor_, static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)), {1, 8});
        PipeBarrier<PIPE_V>();
        repeatStride_ = curCol_ * sizeof(float32_t) / PREPARE_WY_REPR_BWD_ONE_BLOCK_BYTES;
        for (colOffset_ = 0; colOffset_ < curCol_; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(outputFp32Tensor_[colOffset_], inputFp32Tensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        CopyOutRows(gmVb_, rowOffset_ * V_DIM);
        CopyOutRows(debugVbTensor_, (debugLineBase_ + rowOffset_) * V_DIM);
    }

    for (eventIdx_ = 0; eventIdx_ < PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT; ++eventIdx_) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[eventIdx_]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[eventIdx_]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vToMte2Event_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(betaGMte2ToVEvent_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3Event_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx_]);
    }
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::Process()
{
    uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    uint32_t coreNum = AscendC::GetBlockNum();
    uint64_t groupSize = PrepareWyReprBwdGetGroupSize(tiling_);

    for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(tiling_.chunkNum); taskIdx += coreNum) {
        PrepareWyReprBwdTaskInfo task;
        PrepareWyReprBwdGetTaskInfo(cuSeqlens_, chunkIndices_, tiling_, taskIdx, task);
        uint64_t hvTotal = static_cast<uint64_t>(tiling_.HV);
        for (uint64_t hvBase = 0; hvBase < hvTotal; hvBase += PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT) {
            uint32_t headCnt = hvBase + PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT <= hvTotal ?
                                   PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT :
                                   static_cast<uint32_t>(hvTotal - hvBase);
            uint32_t windowStartSlot = curSlot_;

            // Stage0 fills both workspace slots before later stages consume the first head.
            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                uint64_t hk = hv / groupSize;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                ProcessVectorTask(task, taskIdx, hv, hk, slotBase);

                curSlot_ ^= 1U;
            }
        }
    }
}

#endif // PREPARE_WY_REPR_BWD_VECTOR_H
