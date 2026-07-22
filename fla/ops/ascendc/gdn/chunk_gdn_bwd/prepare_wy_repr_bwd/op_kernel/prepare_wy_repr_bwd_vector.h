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
                                                    GM_ADDR chunkIndices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta,
                                                    GM_ADDR dg, GM_ADDR workspace, GM_ADDR debugKbg, GM_ADDR debugVb,
                                                    GM_ADDR debugKbeta, GM_ADDR debugDkb, GM_ADDR debugDK);
    __aicore__ inline void Init(const GDN::PrepareWyReprBwdTilingData &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    template <typename copyType, bool USE_DATA_COPY_PAD>
    __aicore__ inline void CopyInRows(AscendC::GlobalTensor<copyType> &inputTensor,
                                      AscendC::LocalTensor<copyType> dstTensor, uint64_t inputOffset,
                                      uint32_t elements);
    template <typename copyType>
    __aicore__ inline void CastInputRows(AscendC::LocalTensor<float32_t> dstTensor,
                                         AscendC::LocalTensor<copyType> srcTensor, uint32_t elements);
    template <typename copyType, bool USE_DATA_COPY_PAD>
    __aicore__ inline void CopyInBetaGRows(AscendC::GlobalTensor<copyType> &inputTensor,
                                           AscendC::LocalTensor<copyType> dstTensor, uint64_t inputOffset,
                                           uint32_t elements);
    template <typename copyType>
    __aicore__ inline void CastBetaGInputRows(AscendC::LocalTensor<float32_t> dstTensor,
                                              AscendC::LocalTensor<copyType> srcTensor, uint32_t elements);
    __aicore__ inline void InitVectorEvents();
    __aicore__ inline void ReleaseVectorEvents();
    __aicore__ inline void SetBetaGResidentTensors(uint32_t slot);
    __aicore__ inline void CastOutputRows(AscendC::LocalTensor<float32_t> srcTensor, uint32_t elements);
    __aicore__ inline void CopyOutRows(AscendC::GlobalTensor<kType> &outTensor,
                                       AscendC::LocalTensor<kType> srcTensor, uint64_t outOffset, uint32_t elements);
    __aicore__ inline void CopyOutBetaGRows(AscendC::GlobalTensor<gType> &outTensor,
                                            AscendC::LocalTensor<float32_t> srcTensor, uint64_t outOffset,
                                            uint32_t elements);
    __aicore__ inline void ProcessVectorTask(const PrepareWyReprBwdTaskInfo &task, uint32_t taskIdx, uint64_t hv,
                                             uint64_t hk, GM_ADDR slotBase);
    __aicore__ inline void ProcessDa4Task(const PrepareWyReprBwdTaskInfo &task, uint32_t taskIdx, uint64_t hv,
                                          GM_ADDR slotBase);
    __aicore__ inline void ProcessDTask(const PrepareWyReprBwdTaskInfo &task, uint32_t taskIdx, uint64_t hv,
                                        GM_ADDR slotBase);
    __aicore__ inline void ProcessOutputTask(const PrepareWyReprBwdTaskInfo &task, uint32_t taskIdx, uint64_t hv,
                                             uint64_t hk, uint64_t groupSize, GM_ADDR slotBase);

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
    GM_ADDR dk_ = nullptr;
    GM_ADDR dv_ = nullptr;
    GM_ADDR dbeta_ = nullptr;
    GM_ADDR dg_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    GM_ADDR debugKbg_ = nullptr;
    GM_ADDR debugVb_ = nullptr;
    GM_ADDR debugKbeta_ = nullptr;
    GM_ADDR debugDkb_ = nullptr;
    GM_ADDR debugDK_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;

    AscendC::GlobalTensor<kType> kTensor_;
    AscendC::GlobalTensor<kType> vTensor_;
    AscendC::GlobalTensor<kType> dkTensor_;
    AscendC::GlobalTensor<kType> dvTensor_;
    AscendC::GlobalTensor<gType> betaTensor_;
    AscendC::GlobalTensor<gType> gTensor_;
    AscendC::GlobalTensor<gType> dbetaTensor_;
    AscendC::GlobalTensor<gType> dgTensor_;
    AscendC::GlobalTensor<kType> debugKbgTensor_;
    AscendC::GlobalTensor<kType> debugVbTensor_;
    AscendC::GlobalTensor<kType> debugKbetaTensor_;
    AscendC::GlobalTensor<kType> debugDkbTensor_;
    AscendC::GlobalTensor<kType> debugDKTensor_;

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
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcFp32A_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcFp32B_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gRawAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gExpAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dbetaAccFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dgAccFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lowerTriMask_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> upperTriMask_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroFp32_;

    AscendC::LocalTensor<kType> outputBuf_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT];
    AscendC::LocalTensor<kType> matrixInputBuf_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT];
    AscendC::LocalTensor<gType> betaGInputBuf_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT];
    AscendC::LocalTensor<float32_t> betaFp32Tensor_;
    AscendC::LocalTensor<float32_t> gFp32Tensor_;
    AscendC::LocalTensor<float32_t> scaleFp32Tensor_;
    AscendC::LocalTensor<float32_t> brcbFp32Tensor_;
    AscendC::LocalTensor<float32_t> calcFp32ATensor_;
    AscendC::LocalTensor<float32_t> calcFp32BTensor_;
    AscendC::LocalTensor<float32_t> gRawAllFp32Tensor_;
    AscendC::LocalTensor<float32_t> gExpAllFp32Tensor_;
    AscendC::LocalTensor<float32_t> betaAllFp32Tensor_;
    AscendC::LocalTensor<float32_t> dbetaAccFp32Tensor_;
    AscendC::LocalTensor<float32_t> dgAccFp32Tensor_;
    AscendC::LocalTensor<uint8_t> lowerTriMaskTensor_;
    AscendC::LocalTensor<uint8_t> upperTriMaskTensor_;
    AscendC::LocalTensor<float32_t> zeroFp32Tensor_;

    AscendC::GlobalTensor<kType> gmKbg_;
    AscendC::GlobalTensor<kType> gmVb_;
    AscendC::GlobalTensor<kType> gmKbeta_;
    AscendC::GlobalTensor<kType> gmDkbg_;
    AscendC::GlobalTensor<kType> gmDvb_;
    AscendC::GlobalTensor<kType> gmKKT_;
    AscendC::GlobalTensor<kType> gmDA1_;
    AscendC::GlobalTensor<kType> gmDA2_;
    AscendC::GlobalTensor<kType> gmDA4_;
    AscendC::GlobalTensor<kType> gmDA6T_;
    AscendC::GlobalTensor<kType> gmD_;
    AscendC::GlobalTensor<kType> gmDkb_;
    AscendC::GlobalTensor<kType> gmDK_;

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
    uint32_t rowTaskIdx_ = 0;
    uint32_t localRowTask_ = 0;
    uint32_t rowOffset_ = 0;
    uint32_t colOffset_ = 0;
    uint32_t inputIdx_ = 0;
    uint32_t betaGInputIdx_ = 0;
    uint32_t outputIdx_ = 0;
    uint32_t eventIdx_ = 0;
    uint8_t repeatStride_ = 0;
    uint32_t nextKktSlot_ = 0;
    uint32_t cachedKktSlot_ = 0;
    uint64_t cachedKktHk_ = static_cast<uint64_t>(-1);
    uint32_t kktSlotForSlot_[BUFFER_COUNT_2] = {0, 0};
};

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::PrepareWyReprBwdVectorProcess(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR dbeta, GM_ADDR dg, GM_ADDR workspace, GM_ADDR debugKbg, GM_ADDR debugVb, GM_ADDR debugKbeta,
    GM_ADDR debugDkb, GM_ADDR debugDK)
    : k_(k), v_(v), beta_(beta), g_(g), cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices), dk_(dk), dv_(dv),
      dbeta_(dbeta), dg_(dg), workspace_(workspace), debugKbg_(debugKbg), debugVb_(debugVb), debugKbeta_(debugKbeta),
      debugDkb_(debugDkb), debugDK_(debugDK)
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
    dkTensor_.SetGlobalBuffer((__gm__ kType *)dk_);
    dvTensor_.SetGlobalBuffer((__gm__ kType *)dv_);
    betaTensor_.SetGlobalBuffer((__gm__ gType *)beta_);
    gTensor_.SetGlobalBuffer((__gm__ gType *)g_);
    dbetaTensor_.SetGlobalBuffer((__gm__ gType *)dbeta_);
    dgTensor_.SetGlobalBuffer((__gm__ gType *)dg_);
    debugKbgTensor_.SetGlobalBuffer((__gm__ kType *)debugKbg_);
    debugVbTensor_.SetGlobalBuffer((__gm__ kType *)debugVb_);
    debugKbetaTensor_.SetGlobalBuffer((__gm__ kType *)debugKbeta_);
    debugDkbTensor_.SetGlobalBuffer((__gm__ kType *)debugDkb_);
    debugDKTensor_.SetGlobalBuffer((__gm__ kType *)debugDK_);

    uint32_t maxRow = static_cast<uint32_t>(tiling_.kVecRow > tiling_.vVecRow ? tiling_.kVecRow : tiling_.vVecRow);
    maxRow = maxRow > static_cast<uint32_t>(tiling_.mVecRow) ? maxRow : static_cast<uint32_t>(tiling_.mVecRow);
    maxRow = maxRow > static_cast<uint32_t>(tiling_.kktVecRow) ? maxRow : static_cast<uint32_t>(tiling_.kktVecRow);
    pipe_->InitBuffer(inputPing_, UB_BYTES_16K);
    pipe_->InitBuffer(inputPong_, UB_BYTES_16K);
    pipe_->InitBuffer(betaGInputPing_, Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(betaGInputPong_, Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(outputPing_, UB_BYTES_16K);
    pipe_->InitBuffer(outputPong_, UB_BYTES_16K);
    pipe_->InitBuffer(betaFp32_, Align32(maxRow * sizeof(float32_t)));
    pipe_->InitBuffer(gFp32_, Align32(maxRow * sizeof(float32_t)));
    pipe_->InitBuffer(scaleFp32_, Align32(maxRow * sizeof(float32_t)));
    pipe_->InitBuffer(brcbFp32_, Align32(maxRow * PRONE_BLOCK_BYTES_32));
    pipe_->InitBuffer(calcFp32A_, 2 * UB_BYTES_16K);
    pipe_->InitBuffer(calcFp32B_, 2 * UB_BYTES_16K);
    pipe_->InitBuffer(gRawAllFp32_,
                      BUFFER_COUNT_2 *
                          Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(gExpAllFp32_,
                      BUFFER_COUNT_2 *
                          Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(betaAllFp32_,
                      BUFFER_COUNT_2 *
                          Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(dbetaAccFp32_, Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(dgAccFp32_, Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(lowerTriMask_, CHUNK_SIZE * CHUNK_SIZE / PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE);
    pipe_->InitBuffer(upperTriMask_, CHUNK_SIZE * CHUNK_SIZE / PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE);
    pipe_->InitBuffer(zeroFp32_, PRONE_BLOCK_BYTES_32);

    matrixInputBuf_[0] = inputPing_.Get<kType>();
    matrixInputBuf_[1] = inputPong_.Get<kType>();
    betaGInputBuf_[0] = betaGInputPing_.Get<gType>();
    betaGInputBuf_[1] = betaGInputPong_.Get<gType>();
    lowerTriMaskTensor_ = lowerTriMask_.Get<uint8_t>();
    upperTriMaskTensor_ = upperTriMask_.Get<uint8_t>();
    zeroFp32Tensor_ = zeroFp32_.Get<float32_t>();
    uint32_t maskBlocksPerRow = CHUNK_SIZE / PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE;
    for (uint32_t row = 0; row < CHUNK_SIZE; ++row) {
        for (uint32_t block = 0; block < maskBlocksPerRow; ++block) {
            uint32_t colStart = block * PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE;
            uint8_t maskVal = 0;
            for (uint32_t bit = 0; bit < PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE; ++bit) {
                uint32_t col = colStart + bit;
                if (col >= row) {
                    maskVal |= static_cast<uint8_t>(1U << bit);
                }
            }
            lowerTriMaskTensor_.SetValue(row * maskBlocksPerRow + block, maskVal);
            uint8_t upperMaskVal = 0;
            for (uint32_t bit = 0; bit < PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE; ++bit) {
                uint32_t col = colStart + bit;
                if (col <= row) {
                    upperMaskVal |= static_cast<uint8_t>(1U << bit);
                }
            }
            upperTriMaskTensor_.SetValue(row * maskBlocksPerRow + block, upperMaskVal);
        }
    }
    Duplicate(zeroFp32Tensor_, 0.0f, PRONE_BLOCK_BYTES_32 / sizeof(float32_t));
    PipeBarrier<PIPE_V>();
    InitVectorEvents();
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::InitVectorEvents()
{
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
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ReleaseVectorEvents()
{
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
template <typename copyType, bool USE_DATA_COPY_PAD>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyInRows(AscendC::GlobalTensor<copyType> &inputTensor,
                                                                           AscendC::LocalTensor<copyType> dstTensor,
                                                                           uint64_t inputOffset, uint32_t elements)
{
    inputIdx_ = curInputPingPong_;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[inputIdx_]);
    if constexpr (USE_DATA_COPY_PAD) {
        DataCopyPad(dstTensor, inputTensor[inputOffset],
                    {1, elements * static_cast<uint32_t>(sizeof(copyType)), 0, 0, 0}, {false, 0, 0, 0});
    } else {
        DataCopy(dstTensor, inputTensor[inputOffset], elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_[inputIdx_]);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
template <typename copyType>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CastInputRows(
    AscendC::LocalTensor<float32_t> dstTensor, AscendC::LocalTensor<copyType> srcTensor, uint32_t elements)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_[inputIdx_]);
    if constexpr (std::is_same<copyType, float32_t>::value) {
        Adds(dstTensor, srcTensor, 0.0f, elements);
    } else {
        Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[inputIdx_]);
    curInputPingPong_ ^= 1U;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
template <typename copyType, bool USE_DATA_COPY_PAD>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyInBetaGRows(
    AscendC::GlobalTensor<copyType> &inputTensor, AscendC::LocalTensor<copyType> dstTensor, uint64_t inputOffset,
    uint32_t elements)
{
    betaGInputIdx_ = curBetaGInputPingPong_;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[betaGInputIdx_]);
    if constexpr (USE_DATA_COPY_PAD) {
        DataCopyPad(dstTensor, inputTensor[inputOffset],
                    {1, elements * static_cast<uint32_t>(sizeof(copyType)), 0, 0, 0}, {false, 0, 0, 0});
    } else {
        DataCopy(dstTensor, inputTensor[inputOffset], elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(betaGMte2ToVEvent_[betaGInputIdx_]);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
template <typename copyType>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CastBetaGInputRows(
    AscendC::LocalTensor<float32_t> dstTensor, AscendC::LocalTensor<copyType> srcTensor, uint32_t elements)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(betaGMte2ToVEvent_[betaGInputIdx_]);
    if constexpr (std::is_same<copyType, float32_t>::value) {
        Adds(dstTensor, srcTensor, 0.0f, elements);
    } else {
        Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[betaGInputIdx_]);
    curBetaGInputPingPong_ ^= 1U;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::SetBetaGResidentTensors(uint32_t slot)
{
    uint32_t residentOffset = slot * CHUNK_SIZE;
    betaAllFp32Tensor_ = betaAllFp32_.Get<float32_t>()[residentOffset];
    gRawAllFp32Tensor_ = gRawAllFp32_.Get<float32_t>()[residentOffset];
    gExpAllFp32Tensor_ = gExpAllFp32_.Get<float32_t>()[residentOffset];
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CastOutputRows(
    AscendC::LocalTensor<float32_t> srcTensor, uint32_t elements)
{
    outputIdx_ = curOutputPingPong_;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    if constexpr (std::is_same<kType, float32_t>::value) {
        Adds(outputBuf_[outputIdx_], srcTensor, 0.0f, elements);
    } else {
        Cast(outputBuf_[outputIdx_], srcTensor, RoundMode::CAST_RINT, elements);
    }
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyOutRows(AscendC::GlobalTensor<kType> &outTensor,
                                                                            AscendC::LocalTensor<kType> srcTensor,
                                                                            uint64_t outOffset, uint32_t elements)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    DataCopy(outTensor[outOffset], srcTensor, elements);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    curOutputPingPong_ ^= 1U;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyOutBetaGRows(
    AscendC::GlobalTensor<gType> &outTensor, AscendC::LocalTensor<float32_t> srcTensor, uint64_t outOffset,
    uint32_t elements)
{
    outputIdx_ = curOutputPingPong_;
    AscendC::LocalTensor<gType> outputBuf[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT] = {outputPing_.Get<gType>(),
                                                                                     outputPong_.Get<gType>()};
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    if constexpr (std::is_same<gType, float32_t>::value) {
        Adds(outputBuf[outputIdx_], srcTensor, 0.0f, elements);
    } else {
        Cast(outputBuf[outputIdx_], srcTensor, RoundMode::CAST_RINT, elements);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    DataCopyPad(outTensor[outOffset], outputBuf[outputIdx_],
                {1, elements * static_cast<uint32_t>(sizeof(gType)), 0, 0, 0});
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
    calcFp32ATensor_ = calcFp32A_.Get<float32_t>();
    calcFp32BTensor_ = calcFp32B_.Get<float32_t>();
    SetBetaGResidentTensors(curSlot_);
    lowerTriMaskTensor_ = lowerTriMask_.Get<uint8_t>();
    zeroFp32Tensor_ = zeroFp32_.Get<float32_t>();

    gmKbg_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.kbgOffset));
    gmVb_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.vbOffset));
    gmKbeta_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.kbetaOffset));

    subBlockNum_ = AscendC::GetSubBlockNum();
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    keyBase_ = hk * tiling_.T + task.keyBos;
    valueBase_ = hv * tiling_.T + task.valueBos;
    debugLineBase_ =
        (static_cast<uint64_t>(taskIdx) * static_cast<uint64_t>(tiling_.HV) + hv) * static_cast<uint64_t>(CHUNK_SIZE);

    CopyInBetaGRows<gType, true>(betaTensor_, betaGInputBuf_[curBetaGInputPingPong_], valueBase_,
                                 task.curChunkSize);
    CastBetaGInputRows<gType>(betaAllFp32Tensor_, betaGInputBuf_[betaGInputIdx_], task.curChunkSize);
    CopyInBetaGRows<gType, true>(gTensor_, betaGInputBuf_[curBetaGInputPingPong_], valueBase_, task.curChunkSize);
    CastBetaGInputRows<gType>(gRawAllFp32Tensor_, betaGInputBuf_[betaGInputIdx_], task.curChunkSize);
    PipeBarrier<PIPE_V>();
    Adds(gExpAllFp32Tensor_, gRawAllFp32Tensor_, 0.0f, task.curChunkSize);
    PipeBarrier<PIPE_V>();
    Exp(gExpAllFp32Tensor_, gExpAllFp32Tensor_, task.curChunkSize);
    PipeBarrier<PIPE_V>();

    rowTaskIdx_ = 0;

    for (rowOffset_ = 0; rowOffset_ < task.curChunkSize; rowOffset_ += static_cast<uint32_t>(tiling_.kVecRow)) {
        localRowTask_ = rowTaskIdx_++;
        if (localRowTask_ % subBlockNum_ != subBlockIdx_) {
            continue;
        }
        curRow_ = rowOffset_ + static_cast<uint32_t>(tiling_.kVecRow) > task.curChunkSize ?
                      task.curChunkSize - rowOffset_ :
                      static_cast<uint32_t>(tiling_.kVecRow);

        CopyInRows<kType, false>(kTensor_, matrixInputBuf_[curInputPingPong_], (keyBase_ + rowOffset_) * K_DIM,
                                  curRow_ * K_DIM);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();
        Mul(scaleFp32Tensor_, betaAllFp32Tensor_[rowOffset_], gExpAllFp32Tensor_[rowOffset_], curRow_);
        PipeBarrier<PIPE_V>();
        Brcb(brcbFp32Tensor_, scaleFp32Tensor_, static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)), {1, 8});
        PipeBarrier<PIPE_V>();
        repeatStride_ = K_DIM * sizeof(float32_t) / PRONE_BLOCK_BYTES_32;
        for (colOffset_ = 0; colOffset_ < K_DIM; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(calcFp32BTensor_[colOffset_], calcFp32ATensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        CastOutputRows(calcFp32BTensor_, curRow_ * K_DIM);
        CopyOutRows(gmKbg_, outputBuf_[outputIdx_], rowOffset_ * K_DIM, curRow_ * K_DIM);

        Brcb(brcbFp32Tensor_, betaAllFp32Tensor_[rowOffset_],
             static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)), {1, 8});
        PipeBarrier<PIPE_V>();
        repeatStride_ = K_DIM * sizeof(float32_t) / PRONE_BLOCK_BYTES_32;
        for (colOffset_ = 0; colOffset_ < K_DIM; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(calcFp32BTensor_[colOffset_], calcFp32ATensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        CastOutputRows(calcFp32BTensor_, curRow_ * K_DIM);
        CopyOutRows(gmKbeta_, outputBuf_[outputIdx_], rowOffset_ * K_DIM, curRow_ * K_DIM);
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

        CopyInRows<kType, false>(vTensor_, matrixInputBuf_[curInputPingPong_], (valueBase_ + rowOffset_) * V_DIM,
                                  curRow_ * V_DIM);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * V_DIM);
        PipeBarrier<PIPE_V>();
        Brcb(brcbFp32Tensor_, betaAllFp32Tensor_[rowOffset_],
             static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)), {1, 8});
        PipeBarrier<PIPE_V>();
        repeatStride_ = V_DIM * sizeof(float32_t) / PRONE_BLOCK_BYTES_32;
        for (colOffset_ = 0; colOffset_ < V_DIM; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(calcFp32BTensor_[colOffset_], calcFp32ATensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        CastOutputRows(calcFp32BTensor_, curRow_ * V_DIM);
        CopyOutRows(gmVb_, outputBuf_[outputIdx_], rowOffset_ * V_DIM, curRow_ * V_DIM);
    }

}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ProcessDa4Task(
    const PrepareWyReprBwdTaskInfo &task, uint32_t taskIdx, uint64_t hv, GM_ADDR slotBase)
{
    outputBuf_[0] = outputPing_.Get<kType>();
    outputBuf_[1] = outputPong_.Get<kType>();
    calcFp32ATensor_ = calcFp32A_.Get<float32_t>();
    calcFp32BTensor_ = calcFp32B_.Get<float32_t>();

    gmDA1_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da1Offset));
    gmDA2_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da2Offset));
    gmDA4_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da4Offset));
    debugLineBase_ =
        (static_cast<uint64_t>(taskIdx) * static_cast<uint64_t>(tiling_.HV) + hv) * static_cast<uint64_t>(CHUNK_SIZE);

    subBlockNum_ = AscendC::GetSubBlockNum();
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    rowTaskIdx_ = 0;
    for (rowOffset_ = 0; rowOffset_ < task.curChunkSize; rowOffset_ += static_cast<uint32_t>(tiling_.mVecRow)) {
        localRowTask_ = rowTaskIdx_++;
        if (localRowTask_ % subBlockNum_ != subBlockIdx_) {
            continue;
        }
        curRow_ = rowOffset_ + static_cast<uint32_t>(tiling_.mVecRow) > task.curChunkSize ?
                      task.curChunkSize - rowOffset_ :
                      static_cast<uint32_t>(tiling_.mVecRow);

        CopyInRows<kType, false>(gmDA1_, matrixInputBuf_[curInputPingPong_], rowOffset_ * CHUNK_SIZE,
                                  curRow_ * CHUNK_SIZE);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * CHUNK_SIZE);
        CopyInRows<kType, false>(gmDA2_, matrixInputBuf_[curInputPingPong_], rowOffset_ * CHUNK_SIZE,
                                  curRow_ * CHUNK_SIZE);
        CastInputRows<kType>(calcFp32BTensor_, matrixInputBuf_[inputIdx_], curRow_ * CHUNK_SIZE);
        PipeBarrier<PIPE_V>();
        Add(calcFp32BTensor_, calcFp32ATensor_, calcFp32BTensor_, curRow_ * CHUNK_SIZE);
        PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
        Select(calcFp32BTensor_,
               lowerTriMaskTensor_[rowOffset_ * CHUNK_SIZE / PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE],
               zeroFp32Tensor_, calcFp32BTensor_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
               PREPARE_WY_REPR_BWD_FP32_PER_REPEAT,
               curRow_ * CHUNK_SIZE / PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, repeatParams);
        PipeBarrier<PIPE_V>();
        CastOutputRows(calcFp32BTensor_, curRow_ * CHUNK_SIZE);
        CopyOutRows(gmDA4_, outputBuf_[outputIdx_], rowOffset_ * CHUNK_SIZE, curRow_ * CHUNK_SIZE);
    }

}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ProcessDTask(
    const PrepareWyReprBwdTaskInfo &task, uint32_t taskIdx, uint64_t hv, GM_ADDR slotBase)
{
    outputBuf_[0] = outputPing_.Get<kType>();
    outputBuf_[1] = outputPong_.Get<kType>();
    gFp32Tensor_ = gFp32_.Get<float32_t>();
    brcbFp32Tensor_ = brcbFp32_.Get<float32_t>();
    calcFp32ATensor_ = calcFp32A_.Get<float32_t>();
    calcFp32BTensor_ = calcFp32B_.Get<float32_t>();
    SetBetaGResidentTensors(curSlot_);
    upperTriMaskTensor_ = upperTriMask_.Get<uint8_t>();
    zeroFp32Tensor_ = zeroFp32_.Get<float32_t>();

    gmDA6T_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da6Offset));
    gmD_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dOffset));
    valueBase_ = hv * tiling_.T + task.valueBos;
    debugLineBase_ =
        (static_cast<uint64_t>(taskIdx) * static_cast<uint64_t>(tiling_.HV) + hv) * static_cast<uint64_t>(CHUNK_SIZE);

    subBlockNum_ = AscendC::GetSubBlockNum();
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    rowTaskIdx_ = 0;
    for (rowOffset_ = 0; rowOffset_ < task.curChunkSize; rowOffset_ += static_cast<uint32_t>(tiling_.mVecRow)) {
        localRowTask_ = rowTaskIdx_++;
        if (localRowTask_ % subBlockNum_ != subBlockIdx_) {
            continue;
        }
        curRow_ = rowOffset_ + static_cast<uint32_t>(tiling_.mVecRow) > task.curChunkSize ?
                      task.curChunkSize - rowOffset_ :
                      static_cast<uint32_t>(tiling_.mVecRow);

        CopyInRows<kType, false>(gmDA6T_, matrixInputBuf_[curInputPingPong_], rowOffset_ * CHUNK_SIZE,
                                  curRow_ * CHUNK_SIZE);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * CHUNK_SIZE);
        PipeBarrier<PIPE_V>();

        repeatStride_ = CHUNK_SIZE * sizeof(float32_t) / PRONE_BLOCK_BYTES_32;
        for (colOffset_ = 0; colOffset_ < CHUNK_SIZE; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Copy(calcFp32BTensor_[colOffset_], gRawAllFp32Tensor_[colOffset_], PREPARE_WY_REPR_BWD_FP32_PER_REPEAT,
                 curRow_, {1, 1, repeatStride_, 0});
        }
        PipeBarrier<PIPE_V>();
        Brcb(brcbFp32Tensor_, gRawAllFp32Tensor_[rowOffset_],
             static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)), {1, 8});
        PipeBarrier<PIPE_V>();

        for (colOffset_ = 0; colOffset_ < CHUNK_SIZE; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Sub(calcFp32BTensor_[colOffset_], calcFp32BTensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        Mins(calcFp32BTensor_, calcFp32BTensor_, 0.0f, curRow_ * CHUNK_SIZE);
        PipeBarrier<PIPE_V>();
        Exp(calcFp32BTensor_, calcFp32BTensor_, curRow_ * CHUNK_SIZE);
        PipeBarrier<PIPE_V>();
        Muls(calcFp32ATensor_, calcFp32ATensor_, -1.0f, curRow_ * CHUNK_SIZE);
        PipeBarrier<PIPE_V>();
        Mul(calcFp32BTensor_, calcFp32ATensor_, calcFp32BTensor_, curRow_ * CHUNK_SIZE);
        PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
        Select(calcFp32BTensor_,
               upperTriMaskTensor_[rowOffset_ * CHUNK_SIZE / PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE],
               zeroFp32Tensor_, calcFp32BTensor_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
               PREPARE_WY_REPR_BWD_FP32_PER_REPEAT,
               curRow_ * CHUNK_SIZE / PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, repeatParams);
        PipeBarrier<PIPE_V>();
        CastOutputRows(calcFp32BTensor_, curRow_ * CHUNK_SIZE);
        CopyOutRows(gmD_, outputBuf_[outputIdx_], rowOffset_ * CHUNK_SIZE, curRow_ * CHUNK_SIZE);
    }

}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ProcessOutputTask(
    const PrepareWyReprBwdTaskInfo &task, uint32_t taskIdx, uint64_t hv, uint64_t hk, uint64_t groupSize,
    GM_ADDR slotBase)
{
    (void)taskIdx;
    outputBuf_[0] = outputPing_.Get<kType>();
    outputBuf_[1] = outputPong_.Get<kType>();
    betaFp32Tensor_ = betaFp32_.Get<float32_t>();
    gFp32Tensor_ = gFp32_.Get<float32_t>();
    scaleFp32Tensor_ = scaleFp32_.Get<float32_t>();
    brcbFp32Tensor_ = brcbFp32_.Get<float32_t>();
    calcFp32ATensor_ = calcFp32A_.Get<float32_t>();
    calcFp32BTensor_ = calcFp32B_.Get<float32_t>();
    SetBetaGResidentTensors(curSlot_);
    dbetaAccFp32Tensor_ = dbetaAccFp32_.Get<float32_t>();
    dgAccFp32Tensor_ = dgAccFp32_.Get<float32_t>();

    gmDkbg_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dkbgOffset));
    gmDvb_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dvbOffset));
    gmD_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dOffset));
    gmDkb_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dkbOffset));
    gmDK_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dkOffset));
    GM_ADDR kktBase = PrepareWyReprBwdGetKktBase(workspace_, AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(),
                                                 kktSlotForSlot_[curSlot_], tiling_);
    gmKKT_.SetGlobalBuffer((__gm__ kType *)kktBase);

    keyBase_ = hk * tiling_.T + task.keyBos;
    valueBase_ = hv * tiling_.T + task.valueBos;
    bool isFirstValueHeadInGroup = (hv % groupSize) == 0;
    uint32_t rowOwned = static_cast<uint32_t>(tiling_.kVecRow < tiling_.vVecRow ? tiling_.kVecRow : tiling_.vVecRow);
    rowOwned =
        rowOwned < static_cast<uint32_t>(tiling_.kktVecRow) ? rowOwned : static_cast<uint32_t>(tiling_.kktVecRow);
    uint32_t wholeReduceKCnt =
        static_cast<uint32_t>(PrepareWyReprBwdCeilDiv(K_DIM, PREPARE_WY_REPR_BWD_FP32_PER_REPEAT));
    uint32_t wholeReduceVCnt =
        static_cast<uint32_t>(PrepareWyReprBwdCeilDiv(V_DIM, PREPARE_WY_REPR_BWD_FP32_PER_REPEAT));
    uint32_t wholeReduceChunkCnt =
        static_cast<uint32_t>(PrepareWyReprBwdCeilDiv(task.curChunkSize, PREPARE_WY_REPR_BWD_FP32_PER_REPEAT));
    constexpr uint32_t fp32PerBlock = PRONE_BLOCK_BYTES_32 / sizeof(float32_t);

    Duplicate(dbetaAccFp32Tensor_, 0.0f, task.curChunkSize);
    Duplicate(dgAccFp32Tensor_, 0.0f, task.curChunkSize);
    PipeBarrier<PIPE_V>();

    subBlockNum_ = AscendC::GetSubBlockNum();
    subBlockIdx_ = AscendC::GetSubBlockIdx();

    for (rowOffset_ = subBlockIdx_ * rowOwned; rowOffset_ < task.curChunkSize; rowOffset_ += rowOwned * subBlockNum_) {
        curRow_ = rowOffset_ + rowOwned > task.curChunkSize ? task.curChunkSize - rowOffset_ : rowOwned;
        repeatStride_ = K_DIM * sizeof(float32_t) / PRONE_BLOCK_BYTES_32;

        CopyInRows<kType, false>(gmDkb_, matrixInputBuf_[curInputPingPong_], rowOffset_ * K_DIM, curRow_ * K_DIM);
        CastInputRows<kType>(calcFp32BTensor_, matrixInputBuf_[inputIdx_], curRow_ * K_DIM);
        CopyInRows<kType, false>(kTensor_, matrixInputBuf_[curInputPingPong_], (keyBase_ + rowOffset_) * K_DIM,
                                  curRow_ * K_DIM);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();
        Mul(calcFp32ATensor_, calcFp32BTensor_, calcFp32ATensor_, curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < curRow_; ++row) {
            WholeReduceSum(brcbFp32Tensor_[row * fp32PerBlock], calcFp32ATensor_[row * K_DIM],
                           PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, wholeReduceKCnt, 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(scaleFp32Tensor_, brcbFp32Tensor_, wholeReduceKCnt, curRow_, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Add(dbetaAccFp32Tensor_[rowOffset_], dbetaAccFp32Tensor_[rowOffset_], scaleFp32Tensor_, curRow_);

        CopyInRows<kType, false>(gmDkbg_, matrixInputBuf_[curInputPingPong_], rowOffset_ * K_DIM, curRow_ * K_DIM);
        CastInputRows<kType>(calcFp32BTensor_, matrixInputBuf_[inputIdx_], curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();
        Brcb(brcbFp32Tensor_, gExpAllFp32Tensor_[rowOffset_],
             static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)),
             {1, 8});
        PipeBarrier<PIPE_V>();
        for (colOffset_ = 0; colOffset_ < K_DIM; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(calcFp32BTensor_[colOffset_], calcFp32BTensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        CopyInRows<kType, false>(kTensor_, matrixInputBuf_[curInputPingPong_], (keyBase_ + rowOffset_) * K_DIM,
                                  curRow_ * K_DIM);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();
        Mul(calcFp32ATensor_, calcFp32BTensor_, calcFp32ATensor_, curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < curRow_; ++row) {
            WholeReduceSum(brcbFp32Tensor_[row * fp32PerBlock], calcFp32ATensor_[row * K_DIM],
                           PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, wholeReduceKCnt, 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(scaleFp32Tensor_, brcbFp32Tensor_, wholeReduceKCnt, curRow_, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Add(dbetaAccFp32Tensor_[rowOffset_], dbetaAccFp32Tensor_[rowOffset_], scaleFp32Tensor_, curRow_);
        Mul(scaleFp32Tensor_, scaleFp32Tensor_, betaAllFp32Tensor_[rowOffset_], curRow_);
        PipeBarrier<PIPE_V>();
        Add(dgAccFp32Tensor_[rowOffset_], dgAccFp32Tensor_[rowOffset_], scaleFp32Tensor_, curRow_);

        CopyInRows<kType, false>(gmDK_, matrixInputBuf_[curInputPingPong_], rowOffset_ * K_DIM, curRow_ * K_DIM);
        CastInputRows<kType>(calcFp32BTensor_, matrixInputBuf_[inputIdx_], curRow_ * K_DIM);
        CopyInRows<kType, false>(gmDkb_, matrixInputBuf_[curInputPingPong_], rowOffset_ * K_DIM, curRow_ * K_DIM);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();
        Brcb(brcbFp32Tensor_, betaAllFp32Tensor_[rowOffset_], static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)),
             {1, 8});
        PipeBarrier<PIPE_V>();
        for (colOffset_ = 0; colOffset_ < K_DIM; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(calcFp32ATensor_[colOffset_], calcFp32ATensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        Add(calcFp32BTensor_, calcFp32BTensor_, calcFp32ATensor_, curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();

        CopyInRows<kType, false>(gmDkbg_, matrixInputBuf_[curInputPingPong_], rowOffset_ * K_DIM, curRow_ * K_DIM);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();
        Mul(scaleFp32Tensor_, betaAllFp32Tensor_[rowOffset_], gExpAllFp32Tensor_[rowOffset_], curRow_);
        PipeBarrier<PIPE_V>();
        Brcb(brcbFp32Tensor_, scaleFp32Tensor_, static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)), {1, 8});
        PipeBarrier<PIPE_V>();
        for (colOffset_ = 0; colOffset_ < K_DIM; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(calcFp32ATensor_[colOffset_], calcFp32ATensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        Add(calcFp32BTensor_, calcFp32BTensor_, calcFp32ATensor_, curRow_ * K_DIM);
        PipeBarrier<PIPE_V>();
        if (!isFirstValueHeadInGroup) {
            CopyInRows<kType, false>(dkTensor_, matrixInputBuf_[curInputPingPong_],
                                      (keyBase_ + rowOffset_) * K_DIM, curRow_ * K_DIM);
            CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * K_DIM);
            PipeBarrier<PIPE_V>();
            Add(calcFp32BTensor_, calcFp32BTensor_, calcFp32ATensor_, curRow_ * K_DIM);
            PipeBarrier<PIPE_V>();
        }
        CastOutputRows(calcFp32BTensor_, curRow_ * K_DIM);
        CopyOutRows(dkTensor_, outputBuf_[outputIdx_], (keyBase_ + rowOffset_) * K_DIM, curRow_ * K_DIM);
    }

    for (rowOffset_ = subBlockIdx_ * rowOwned; rowOffset_ < task.curChunkSize; rowOffset_ += rowOwned * subBlockNum_) {
        curRow_ = rowOffset_ + rowOwned > task.curChunkSize ? task.curChunkSize - rowOffset_ : rowOwned;
        repeatStride_ = V_DIM * sizeof(float32_t) / PRONE_BLOCK_BYTES_32;

        CopyInRows<kType, false>(gmDvb_, matrixInputBuf_[curInputPingPong_], rowOffset_ * V_DIM, curRow_ * V_DIM);
        CastInputRows<kType>(calcFp32BTensor_, matrixInputBuf_[inputIdx_], curRow_ * V_DIM);
        CopyInRows<kType, false>(vTensor_, matrixInputBuf_[curInputPingPong_], (valueBase_ + rowOffset_) * V_DIM,
                                  curRow_ * V_DIM);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * V_DIM);
        PipeBarrier<PIPE_V>();
        Mul(calcFp32ATensor_, calcFp32BTensor_, calcFp32ATensor_, curRow_ * V_DIM);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < curRow_; ++row) {
            WholeReduceSum(brcbFp32Tensor_[row * fp32PerBlock], calcFp32ATensor_[row * V_DIM],
                           PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, wholeReduceVCnt, 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(scaleFp32Tensor_, brcbFp32Tensor_, wholeReduceVCnt, curRow_, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Add(dbetaAccFp32Tensor_[rowOffset_], dbetaAccFp32Tensor_[rowOffset_], scaleFp32Tensor_, curRow_);

        Brcb(brcbFp32Tensor_, betaAllFp32Tensor_[rowOffset_], static_cast<uint8_t>(PrepareWyReprBwdCeilDiv(curRow_, 8)),
             {1, 8});
        PipeBarrier<PIPE_V>();
        for (colOffset_ = 0; colOffset_ < V_DIM; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(calcFp32BTensor_[colOffset_], calcFp32BTensor_[colOffset_], brcbFp32Tensor_,
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 0, repeatStride_, repeatStride_, 1});
        }
        PipeBarrier<PIPE_V>();
        CastOutputRows(calcFp32BTensor_, curRow_ * V_DIM);
        CopyOutRows(dvTensor_, outputBuf_[outputIdx_], (valueBase_ + rowOffset_) * V_DIM, curRow_ * V_DIM);
    }

    for (rowOffset_ = subBlockIdx_ * rowOwned; rowOffset_ < task.curChunkSize; rowOffset_ += rowOwned * subBlockNum_) {
        curRow_ = rowOffset_ + rowOwned > task.curChunkSize ? task.curChunkSize - rowOffset_ : rowOwned;
        repeatStride_ = CHUNK_SIZE * sizeof(float32_t) / PRONE_BLOCK_BYTES_32;

        CopyInRows<kType, false>(gmD_, matrixInputBuf_[curInputPingPong_], rowOffset_ * CHUNK_SIZE,
                                  curRow_ * CHUNK_SIZE);
        CastInputRows<kType>(calcFp32BTensor_, matrixInputBuf_[inputIdx_], curRow_ * CHUNK_SIZE);
        CopyInRows<kType, false>(gmKKT_, matrixInputBuf_[curInputPingPong_], rowOffset_ * CHUNK_SIZE,
                                  curRow_ * CHUNK_SIZE);
        CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curRow_ * CHUNK_SIZE);
        PipeBarrier<PIPE_V>();
        Mul(calcFp32ATensor_, calcFp32BTensor_, calcFp32ATensor_, curRow_ * CHUNK_SIZE);
        PipeBarrier<PIPE_V>();
        for (colOffset_ = 0; colOffset_ < task.curChunkSize; colOffset_ += PREPARE_WY_REPR_BWD_FP32_PER_REPEAT) {
            Mul(calcFp32ATensor_[colOffset_], calcFp32ATensor_[colOffset_], betaAllFp32Tensor_[colOffset_],
                PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, curRow_, {1, 1, 1, repeatStride_, repeatStride_, 0});
        }
        PipeBarrier<PIPE_V>();
        uint32_t remainCnt = task.curChunkSize % PREPARE_WY_REPR_BWD_FP32_PER_REPEAT;
        if (remainCnt > 0) {
            uint32_t duplicateOffset =
                wholeReduceChunkCnt * PREPARE_WY_REPR_BWD_FP32_PER_REPEAT - PREPARE_WY_REPR_BWD_FP32_PER_REPEAT;
            uint64_t mask[1] = {0xffffffffffffffff};
            mask[0] <<= remainCnt;
            for (uint32_t row = 0; row < curRow_; ++row) {
                Duplicate(calcFp32ATensor_[row * CHUNK_SIZE + duplicateOffset], 0.0f, mask, 1, 1, 8);
            }
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t row = 0; row < curRow_; ++row) {
            WholeReduceSum(brcbFp32Tensor_[row * fp32PerBlock], calcFp32ATensor_[row * CHUNK_SIZE],
                           PREPARE_WY_REPR_BWD_FP32_PER_REPEAT, wholeReduceChunkCnt, 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(scaleFp32Tensor_, brcbFp32Tensor_, wholeReduceChunkCnt, curRow_, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Muls(scaleFp32Tensor_, scaleFp32Tensor_, -1.0f, curRow_);
        PipeBarrier<PIPE_V>();
        Add(dgAccFp32Tensor_[rowOffset_], dgAccFp32Tensor_[rowOffset_], scaleFp32Tensor_, curRow_);
        PipeBarrier<PIPE_V>();

        Duplicate(scaleFp32Tensor_, 0.0f, curRow_);
        PipeBarrier<PIPE_V>();
        for (uint32_t srcOffset = 0; srcOffset < task.curChunkSize; srcOffset += rowOwned) {
            uint32_t curSrcRow = srcOffset + rowOwned > task.curChunkSize ? task.curChunkSize - srcOffset : rowOwned;
            CopyInRows<kType, false>(gmD_, matrixInputBuf_[curInputPingPong_], srcOffset * CHUNK_SIZE,
                                      curSrcRow * CHUNK_SIZE);
            CastInputRows<kType>(calcFp32BTensor_, matrixInputBuf_[inputIdx_], curSrcRow * CHUNK_SIZE);
            CopyInRows<kType, false>(gmKKT_, matrixInputBuf_[curInputPingPong_], srcOffset * CHUNK_SIZE,
                                      curSrcRow * CHUNK_SIZE);
            CastInputRows<kType>(calcFp32ATensor_, matrixInputBuf_[inputIdx_], curSrcRow * CHUNK_SIZE);
            PipeBarrier<PIPE_V>();
            Mul(calcFp32ATensor_, calcFp32BTensor_, calcFp32ATensor_, curSrcRow * CHUNK_SIZE);
            PipeBarrier<PIPE_V>();
            uint32_t remainRow = curSrcRow;
            while (remainRow > 1) {
                uint32_t calcCnt = (remainRow / 2) * CHUNK_SIZE;
                remainRow = static_cast<uint32_t>(PrepareWyReprBwdCeilDiv(remainRow, 2));
                uint32_t offset = remainRow * CHUNK_SIZE;
                Add(calcFp32ATensor_, calcFp32ATensor_, calcFp32ATensor_[offset], calcCnt);
                PipeBarrier<PIPE_V>();
            }
            Mul(gFp32Tensor_, calcFp32ATensor_[rowOffset_], betaAllFp32Tensor_[rowOffset_], curRow_);
            PipeBarrier<PIPE_V>();
            Add(scaleFp32Tensor_, scaleFp32Tensor_, gFp32Tensor_, curRow_);
            PipeBarrier<PIPE_V>();
        }
        Add(dgAccFp32Tensor_[rowOffset_], dgAccFp32Tensor_[rowOffset_], scaleFp32Tensor_, curRow_);
        PipeBarrier<PIPE_V>();
    }

    for (rowOffset_ = subBlockIdx_ * rowOwned; rowOffset_ < task.curChunkSize; rowOffset_ += rowOwned * subBlockNum_) {
        curRow_ = rowOffset_ + rowOwned > task.curChunkSize ? task.curChunkSize - rowOffset_ : rowOwned;
        CopyOutBetaGRows(dbetaTensor_, dbetaAccFp32Tensor_[rowOffset_], valueBase_ + rowOffset_, curRow_);
        CopyOutBetaGRows(dgTensor_, dgAccFp32Tensor_[rowOffset_], valueBase_ + rowOffset_, curRow_);
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
        nextKktSlot_ = 0;
        cachedKktSlot_ = 0;
        cachedKktHk_ = static_cast<uint64_t>(-1);
        kktSlotForSlot_[0] = 0;
        kktSlotForSlot_[1] = 0;
        uint64_t hvTotal = static_cast<uint64_t>(tiling_.HV);
        for (uint64_t hvBase = 0; hvBase < hvTotal; hvBase += BUFFER_COUNT_2) {
            uint32_t headCnt = hvBase + BUFFER_COUNT_2 <= hvTotal ?
                                   BUFFER_COUNT_2 :
                                   static_cast<uint32_t>(hvTotal - hvBase);
            uint32_t windowStartSlot = curSlot_;

            // Stage0 fills both workspace slots before later stages consume the first head.
            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                uint64_t hk = hv / groupSize;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                if (cachedKktHk_ != hk) {
                    cachedKktHk_ = hk;
                    cachedKktSlot_ = nextKktSlot_;
                    nextKktSlot_ ^= 1U;
                }
                kktSlotForSlot_[curSlot_] = cachedKktSlot_;
                ProcessVectorTask(task, taskIdx, hv, hk, slotBase);
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(vecToCubeFlag_);

                curSlot_ ^= 1U;
            }

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(cubeToVecFlag_);
                ProcessDa4Task(task, taskIdx, hv, slotBase);
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(vecToCubeFlag_);
                curSlot_ ^= 1U;
            }

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(cubeToVecFlag_);
                ProcessDTask(task, taskIdx, hv, slotBase);
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(vecToCubeFlag_);
                curSlot_ ^= 1U;
            }

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                uint64_t hk = hv / groupSize;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(cubeToVecFlag_);
                ProcessOutputTask(task, taskIdx, hv, hk, groupSize, slotBase);
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(vecToCubeFlag_);
                curSlot_ ^= 1U;
            }
        }
    }
    ReleaseVectorEvents();
}

#endif // PREPARE_WY_REPR_BWD_VECTOR_H
