/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda.h
 * \brief Single-kernel fused recurrent KDA implementation.
 */

#ifndef __RECURRENT_KDA_KERNEL_H_
#define __RECURRENT_KDA_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "recurrent_kda_tiling_data.h"

namespace RecurrentKda {

using namespace matmul;
using namespace AscendC;
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t INPUT_BUFFER_NUM = 2;
constexpr uint32_t PACKED_INPUT_BUFFER_NUM = 2;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
// Input capacity per time window; logical sequences may contain more tokens.
constexpr uint64_t TOKEN_WINDOW = 8;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LENTH = 64; // 256 bytes for float.
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t ADD_FOLD_REDUCE_MIN_K = 128;
constexpr uint64_t INVALID_STATE_SLOT = static_cast<uint64_t>(-1);
constexpr uint32_t STATE_TRANSPOSE_BLOCK = 16;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr float QK_L2NORM_EPS = 1e-6f;

#ifndef RKDA_ENABLE_ADD_FOLD_REDUCE
#define RKDA_ENABLE_ADD_FOLD_REDUCE  1
#endif

struct RKDAInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR ssmStateIndices;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR numAcceptedTokens;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename inType, typename outType, typename stateType>
class RKDA {
public:
    __aicore__ inline explicit RKDA(const RecurrentKdaTilingData *tilingData)
    {
        B_ = tilingData->b;
        T_ = tilingData->t;
        seqLen_ = tilingData->seqLen;
        NK_ = tilingData->nk;
        realK_ = tilingData->dk;
        NV_ = tilingData->nv;
        realV_ = tilingData->dv;
        stateCapacity_ = tilingData->sBlockNum;
        queryTokenStride_ = tilingData->queryTokenStride;
        queryHeadStride_ = tilingData->queryHeadStride;
        keyTokenStride_ = tilingData->keyTokenStride;
        keyHeadStride_ = tilingData->keyHeadStride;
        valueTokenStride_ = tilingData->valueTokenStride;
        valueHeadStride_ = tilingData->valueHeadStride;
        ssmStateStride_ = tilingData->ssmStateStride;
        stateInStride0_ = tilingData->stateInStride0;
        stateInStride1_ = tilingData->stateInStride1;
        stateInStride2_ = tilingData->stateInStride2;
        stateInStride3_ = tilingData->stateInStride3;
        stateOutStride0_ = tilingData->stateOutStride0;
        stateOutStride1_ = tilingData->stateOutStride1;
        stateOutStride2_ = tilingData->stateOutStride2;
        stateOutStride3_ = tilingData->stateOutStride3;
        scale_ = tilingData->scale;
        lowerBound_ = tilingData->lowerBound;
        hasCuSeqlens_ = (tilingData->hasCuSeqlens == 1);
        hasSsmStateIndices_ = (tilingData->hasSsmStateIndices == 1);
        hasAcceptedTokens_ = (tilingData->hasAcceptedTokens == 1);
        hasALog_ = (tilingData->hasALog == 1);
        hasDtBias_ = (tilingData->hasDtBias == 1);
        useQkL2norm_ = (tilingData->useQkL2norm == 1);
        useGateInKernel_ = (tilingData->useGateInKernel == 1);
        useBetaSigmoid_ = (tilingData->useBetaSigmoid == 1);
        allowNegEigval_ = (tilingData->allowNegEigval == 1);
        safeGate_ = (tilingData->safeGate == 1);
        stateVFirst_ = (tilingData->stateVFirst == 1);
        shouldStoreState_ = (tilingData->inplaceFinalState == 1 || tilingData->outputFinalState == 1);
        gateDtype_ = tilingData->gateDtype;
        betaDtype_ = tilingData->betaDtype;
        cuSeqlensDtype_ = tilingData->cuSeqlensDtype;
        ssmStateIndicesDtype_ = tilingData->ssmStateIndicesDtype;
        acceptedTokensDtype_ = tilingData->acceptedTokensDtype;
        useAddFoldReduce_ = (RKDA_ENABLE_ADD_FOLD_REDUCE != 0);
        vStep_ = tilingData->vStep;
        stateOutBufferNum_ = (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        attnOutBufferNum_ = (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        restUbSize_ = tilingData->ubRestBytes;
        alignK_ = Ceil(tilingData->dk, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(tilingData->dv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        eventMte2ToVInitialized_ = false;
        eventVToMte2Initialized_ = false;
        eventVToSInitialized_ = false;
        eventVToMte3Initialized_ = false;
        eventMte3ToVInitialized_ = false;
    }

    __aicore__ inline void Init(const RKDAInitParams &initParams, TPipe *pipe)
    {
        uint64_t blockDim = GetBlockNum();
        blockIdx = GetBlockIdx();
        if (blockIdx >= blockDim) {
            return;
        }
        InitTaskRange(blockDim);
        pipe_ = pipe;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void SetGlobalTensors(const RKDAInitParams &initParams)
    {
        queryGm_.SetGlobalBuffer((__gm__ inType *)initParams.query, 0);
        keyGm_.SetGlobalBuffer((__gm__ inType *)initParams.key, 0);
        valueGm_.SetGlobalBuffer((__gm__ inType *)initParams.value, 0);
        gateFloatGm_.SetGlobalBuffer((__gm__ float *)initParams.gate, 0);
        gateBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)initParams.gate, 0);
        gateFp16Gm_.SetGlobalBuffer((__gm__ half *)initParams.gate, 0);
        betaFloatGm_.SetGlobalBuffer((__gm__ float *)initParams.beta, 0);
        betaBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)initParams.beta, 0);
        betaFp16Gm_.SetGlobalBuffer((__gm__ half *)initParams.beta, 0);
        initStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.initState, 0);
        cuSeqlensInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens, 0);
        cuSeqlensInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.cuSeqlens, 0);
        ssmStateIndicesInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices, 0);
        ssmStateIndicesInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.ssmStateIndices, 0);
        aLogGm_.SetGlobalBuffer((__gm__ float *)initParams.aLog, 0);
        dtBiasGm_.SetGlobalBuffer((__gm__ float *)initParams.dtBias, 0);
        numAcceptedTokensInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.numAcceptedTokens, 0);
        numAcceptedTokensInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.numAcceptedTokens, 0);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.finalState, 0);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut, 0);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t cubeSize = alignK_ * vStep_ * sizeof(float);
        uint32_t singleVSize = vStep_ * sizeof(float);
        uint32_t vSize = TOKEN_WINDOW * alignV_ * sizeof(float);
        uint32_t kSize = TOKEN_WINDOW * alignK_ * sizeof(float);
        uint32_t betaUbSize =
            Ceil(TOKEN_WINDOW * NV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK * sizeof(float);
        packedKOffset_ = TOKEN_WINDOW * alignK_;
        packedVOffset_ = 2 * packedKOffset_;
        packedGateOffset_ = packedVOffset_ + TOKEN_WINDOW * alignV_;
        uint32_t packedBytes =
            packedGateOffset_ * sizeof(inType) + TOKEN_WINDOW * alignK_ * sizeof(float);
        pipe_->InitBuffer(packedInQueue_, PACKED_INPUT_BUFFER_NUM, packedBytes);
        pipe_->InitBuffer(betaInQueue_, BUFFER_NUM, betaUbSize);
        pipe_->InitBuffer(stateInQueue_, INPUT_BUFFER_NUM, alignK_ * vStep_ * sizeof(stateType));
        pipe_->InitBuffer(stateOutQueue_, stateOutBufferNum_, alignK_ * vStep_ * sizeof(stateType));
        if (!stateVFirst_) {
            pipe_->InitBuffer(stateTransposeBuf_, alignK_ * vStep_ * sizeof(stateType));
        }
        pipe_->InitBuffer(attnOutQueue_, attnOutBufferNum_, vStep_ * sizeof(outType));
        pipe_->InitBuffer(tmpBuff, restUbSize_);
        pipe_->InitBuffer(scalarBuf_, 64);

        uint32_t buffOffset = 0;
        deltaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        attnInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        vUbByteOffset_ = buffOffset;
        vInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(TOKEN_WINDOW * alignV_), buffOffset);
        buffOffset += vSize;
        qInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(TOKEN_WINDOW * alignK_), buffOffset);
        buffOffset += kSize;
        kInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(TOKEN_WINDOW * alignK_), buffOffset);
        buffOffset += kSize;
        stateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;
        broadTmpInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;
        betaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaUbSize / sizeof(float)), buffOffset);
        buffOffset += betaUbSize;
        gateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(TOKEN_WINDOW * alignK_), buffOffset);

    }

    __aicore__ inline void SyncMte2ToV()
    {
        if (!eventMte2ToVInitialized_) {
            eventIdMte2ToV_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
            eventMte2ToVInitialized_ = true;
        }
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
    }

    __aicore__ inline void SyncVToMte2()
    {
        if (!eventVToMte2Initialized_) {
            eventIdVToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
            eventVToMte2Initialized_ = true;
        }
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
    }

    __aicore__ inline void SyncVToS()
    {
        if (!eventVToSInitialized_) {
            eventIdVToS_ = GetTPipePtr()->AllocEventID<HardEvent::V_S>();
            eventVToSInitialized_ = true;
        }
        SetFlag<HardEvent::V_S>(eventIdVToS_);
        WaitFlag<HardEvent::V_S>(eventIdVToS_);
    }

    __aicore__ inline void SyncVToMte3()
    {
        if (!eventVToMte3Initialized_) {
            eventIdVToMte3_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
            eventVToMte3Initialized_ = true;
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        if (!eventMte3ToVInitialized_) {
            eventIdMte3ToV_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
            eventMte3ToVInitialized_ = true;
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
    }

    __aicore__ inline void ReleaseEvents()
    {
        if (eventMte2ToVInitialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV_);
            eventMte2ToVInitialized_ = false;
        }
        if (eventVToMte2Initialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2_);
            eventVToMte2Initialized_ = false;
        }
        if (eventVToSInitialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(eventIdVToS_);
            eventVToSInitialized_ = false;
        }
        if (eventVToMte3Initialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3_);
            eventVToMte3Initialized_ = false;
        }
        if (eventMte3ToVInitialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV_);
            eventMte3ToVInitialized_ = false;
        }
    }

    __aicore__ inline void Process()
    {
        if (!ValidateCuSeqlens()) {
            ReleaseEvents();
            return;
        }
        for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
            int64_t seq0 = SequenceStart(batch_i);
            int64_t seq1 = SequenceEnd(batch_i);
            int64_t seqLen64 = seq1 - seq0;
            if (seqLen64 == 0) {
                continue;
            }
            int64_t seqLen = seqLen64;
            if (!ValidateStateSlots(batch_i, seq0, seqLen)) {
                ReleaseEvents();
                return;
            }

            uint64_t head_i = FindNextTaskHead(batch_i, 0);
            if (head_i == NV_) {
                continue;
            }
            uint64_t stateSlot = ResolveInitialStateSlot(batch_i, seq0, seqLen);
            if (stateSlot == INVALID_STATE_SLOT) {
                ReleaseEvents();
                return;
            }
            const int32_t firstWindowLen = static_cast<int32_t>(
                seqLen < static_cast<int64_t>(TOKEN_WINDOW) ? seqLen : TOKEN_WINDOW);
            if (seqLen <= static_cast<int64_t>(TOKEN_WINDOW)) {
                CopyInBeta(seq0, seq1);
            }
            PrefetchPackedInputs(seq0, firstWindowLen, head_i);
            // Select using the validated logical segment, including device cu_seqlens.
            if (seqLen == 1) {
                ProcessHeadTasks<1>(batch_i, seq0, seq1, head_i, stateSlot);
            } else if (seqLen <= static_cast<int64_t>(TOKEN_WINDOW)) {
                ProcessHeadTasks<-1>(batch_i, seq0, seq1, head_i, stateSlot);
            } else {
                ProcessHeadTasks<0>(batch_i, seq0, seq1, head_i, stateSlot);
            }
        }
        ReleaseEvents();
    }

private:
    __aicore__ inline bool ValidateCuSeqlens() const
    {
        if (!hasCuSeqlens_) {
            return true;
        }
        int64_t seq0 = LoadCuSeqlens(0);
        if (seq0 != 0) {
            return false;
        }
        for (uint64_t i = 0; i < B_; i++) {
            int64_t seq1 = LoadCuSeqlens(i + 1);
            int64_t length = seq1 - seq0;
            if (seq1 < seq0 || seq1 > static_cast<int64_t>(T_) ||
                (hasSsmStateIndices_ && ssmStateStride_ > 0 && length > ssmStateStride_)) {
                return false;
            }
            seq0 = seq1;
        }
        return seq0 <= static_cast<int64_t>(T_);
    }

    __aicore__ inline int64_t LoadCuSeqlens(uint64_t index) const
    {
        return cuSeqlensDtype_ == 0 ? static_cast<int64_t>(cuSeqlensInt32Gm_.GetValue(index)) :
                                     cuSeqlensInt64Gm_.GetValue(index);
    }

    __aicore__ inline int64_t SequenceStart(uint64_t batchIdx) const
    {
        return hasCuSeqlens_ ? LoadCuSeqlens(batchIdx) : static_cast<int64_t>(batchIdx * seqLen_);
    }

    __aicore__ inline int64_t SequenceEnd(uint64_t batchIdx) const
    {
        return hasCuSeqlens_ ? LoadCuSeqlens(batchIdx + 1) :
                               static_cast<int64_t>((batchIdx + 1) * seqLen_);
    }

    __aicore__ inline int64_t LoadSsmStateIndex(uint64_t index) const
    {
        return ssmStateIndicesDtype_ == 0 ? static_cast<int64_t>(ssmStateIndicesInt32Gm_.GetValue(index)) :
                                           ssmStateIndicesInt64Gm_.GetValue(index);
    }

    __aicore__ inline int64_t LoadAcceptedTokens(uint64_t index) const
    {
        return acceptedTokensDtype_ == 0 ? static_cast<int64_t>(numAcceptedTokensInt32Gm_.GetValue(index)) :
                                          numAcceptedTokensInt64Gm_.GetValue(index);
    }

    __aicore__ inline uint64_t StateMetadataOffset(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        if (ssmStateStride_ == 0) {
            return static_cast<uint64_t>(tokenIdx);
        }
        return batchIdx * ssmStateStride_ + static_cast<uint64_t>(tokenIdx - seq0);
    }

    __aicore__ inline uint64_t LoadStateSlot(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        int64_t stateSlot = LoadSsmStateIndex(StateMetadataOffset(batchIdx, seq0, tokenIdx));
        if (stateSlot < 0 || stateSlot >= static_cast<int64_t>(stateCapacity_)) {
            return INVALID_STATE_SLOT;
        }
        return static_cast<uint64_t>(stateSlot);
    }

    __aicore__ inline bool ValidateStateSlots(uint64_t batchIdx, int64_t seq0, int64_t seqLen) const
    {
        if (!hasSsmStateIndices_) {
            return batchIdx < stateCapacity_;
        }
        if (hasAcceptedTokens_) {
            int64_t acceptedTokenNum = LoadAcceptedTokens(batchIdx);
            if (acceptedTokenNum <= 0 || acceptedTokenNum > seqLen) {
                return false;
            }
        }
        for (int64_t step = 0; step < seqLen; ++step) {
            if (LoadStateSlot(batchIdx, seq0, seq0 + step) == INVALID_STATE_SLOT) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint64_t ResolveInitialStateSlot(uint64_t batchIdx, int64_t seq0, int64_t seqLen) const
    {
        if (!hasSsmStateIndices_) {
            return batchIdx;
        }
        int64_t tokenIdx = seq0;
        if (hasAcceptedTokens_) {
            tokenIdx = seq0 + LoadAcceptedTokens(batchIdx) - 1;
        }
        return LoadStateSlot(batchIdx, seq0, tokenIdx);
    }

    template <typename dataType>
    __aicore__ inline void CopyVectorIn(LocalTensor<dataType> &dst, GlobalTensor<dataType> &src, uint64_t offset,
                                        uint64_t count)
    {
        uint64_t rowBytes = count * sizeof(dataType);
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
        } else {
            DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyPad(dst, src[offset], params, padParams);
        }
    }

    __aicore__ inline float ReadExpFloat(GlobalTensor<float> &tensor, uint64_t offset)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(scalar, tensor[offset], params, padParams);
        SyncMte2ToV();
        Exp(scalar, scalar, 1);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline float ExpScalar(float x)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        Duplicate(scalar, x, 1);
        PipeBarrier<PIPE_V>();
        Exp(scalar, scalar, 1);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline float SigmoidScalar(float x)
    {
        float denom = 1.0f + ExpScalar(-x);
        return 1.0f / denom;
    }

    __aicore__ inline void NormalizeQKRows(int32_t seqLen)
    {
        const uint32_t rows = static_cast<uint32_t>(seqLen);
        const uint32_t elements = rows * alignK_;
        // L <= TOKEN_WINDOW <= vStep: the existing scratch and norm slices are sufficient.
        Mul(broadTmpInUb, qInUb, qInUb, elements);
        PipeBarrier<PIPE_V>();
        ReduceSumDispatch(deltaInUb, broadTmpInUb, rows);
        // The Q reduction must finish reading before K overwrites the square scratch.
        PipeBarrier<PIPE_V>();
        Mul(broadTmpInUb, kInUb, kInUb, elements);
        PipeBarrier<PIPE_V>();
        ReduceSumDispatch(attnInUb, broadTmpInUb, rows);
        PipeBarrier<PIPE_V>();
        Adds(deltaInUb, deltaInUb, QK_L2NORM_EPS, rows);
        Adds(attnInUb, attnInUb, QK_L2NORM_EPS, rows);
        PipeBarrier<PIPE_V>();
        Sqrt(deltaInUb, deltaInUb, rows);
        Sqrt(attnInUb, attnInUb, rows);
        PipeBarrier<PIPE_V>();
        SyncVToS();

        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t rowOffset = row * alignK_;
            const float qNorm = deltaInUb.GetValue(row);
            const float kNorm = attnInUb.GetValue(row);
            if (qNorm > 0.0f) {
                Muls(qInUb[rowOffset], qInUb[rowOffset], 1.0f / qNorm, alignK_);
            }
            if (kNorm > 0.0f) {
                Muls(kInUb[rowOffset], kInUb[rowOffset], 1.0f / kNorm, alignK_);
            }
        }
        // All row outputs must be ready for the independent Q scale and later compute.
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyGateInKernel(uint64_t head, int32_t seqLen)
    {
        uint32_t total = static_cast<uint32_t>(seqLen) * alignK_;
        float expA = hasALog_ ? ReadExpFloat(aLogGm_, head) : 1.0f;

        if (hasDtBias_) {
            CopyVectorIn(broadTmpInUb, dtBiasGm_, head * realK_, realK_);
            SyncMte2ToV();
            for (int32_t row = 0; row < seqLen; ++row) {
                Add(gateInUb[row * alignK_], gateInUb[row * alignK_], broadTmpInUb, alignK_);
                PipeBarrier<PIPE_V>();
            }
        }

        if (safeGate_) {
            Muls(gateInUb, gateInUb, expA, total);
            PipeBarrier<PIPE_V>();
            Muls(broadTmpInUb, gateInUb, -1.0f, total);
            PipeBarrier<PIPE_V>();
            Exp(broadTmpInUb, broadTmpInUb, total);
            PipeBarrier<PIPE_V>();
            Adds(broadTmpInUb, broadTmpInUb, 1.0f, total);
            PipeBarrier<PIPE_V>();
            Duplicate(gateInUb, 1.0f, total);
            PipeBarrier<PIPE_V>();
            Div(gateInUb, gateInUb, broadTmpInUb, total);
            PipeBarrier<PIPE_V>();
            Muls(gateInUb, gateInUb, lowerBound_, total);
            PipeBarrier<PIPE_V>();
        } else {
            Exp(gateInUb, gateInUb, total);
            PipeBarrier<PIPE_V>();
            Adds(gateInUb, gateInUb, 1.0f, total);
            PipeBarrier<PIPE_V>();
            Ln(gateInUb, gateInUb, total);
            PipeBarrier<PIPE_V>();
            Muls(gateInUb, gateInUb, -expA, total);
            PipeBarrier<PIPE_V>();
        }
    }

    template <typename gateType>
    __aicore__ inline void CopyPackedGate(LocalTensor<inType> &packed, uint64_t gateOffset, int32_t seqLen)
    {
        LocalTensor<gateType> gateLocal = packed[packedGateOffset_].template ReinterpretCast<gateType>();
        DataCopyExtParams params{
            static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(gateType)),
            static_cast<uint32_t>((NV_ - 1) * realK_ * sizeof(gateType)), 0, 0};
        DataCopyPadExtParams<gateType> pad{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        if constexpr (std::is_same<gateType, float32_t>()) {
            DataCopyPad(gateLocal, gateFloatGm_[gateOffset], params, pad);
        } else if constexpr (std::is_same<gateType, bfloat16_t>()) {
            DataCopyPad(gateLocal, gateBf16Gm_[gateOffset], params, pad);
        } else {
            DataCopyPad(gateLocal, gateFp16Gm_[gateOffset], params, pad);
        }
    }

    __aicore__ inline void PrefetchPackedInputs(int64_t seq0, int32_t seqLen, uint64_t head)
    {
        LocalTensor<inType> packed = packedInQueue_.AllocTensor<inType>();
        LocalTensor<inType> qLocal = packed;
        LocalTensor<inType> kLocal = packed[packedKOffset_];
        LocalTensor<inType> vLocal = packed[packedVOffset_];
        const uint64_t token = static_cast<uint64_t>(seq0);
        const uint64_t qkHead = head / (NV_ / NK_);
        DataCopyExtParams qParams{
            static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
            static_cast<uint32_t>((queryTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams kParams{
            static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
            static_cast<uint32_t>((keyTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams vParams{
            static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
            static_cast<uint32_t>((valueTokenStride_ - realV_) * sizeof(inType)), 0, 0};
        DataCopyPadExtParams<inType> qkPad{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPadExtParams<inType> vPad{true, 0, static_cast<uint8_t>(alignV_ - realV_), 0};
        DataCopyPad(qLocal, queryGm_[token * queryTokenStride_ + qkHead * queryHeadStride_], qParams, qkPad);
        DataCopyPad(kLocal, keyGm_[token * keyTokenStride_ + qkHead * keyHeadStride_], kParams, qkPad);
        DataCopyPad(vLocal, valueGm_[token * valueTokenStride_ + head * valueHeadStride_], vParams, vPad);
        uint64_t gateOffset = (token * NV_ + head) * realK_;
        if (gateDtype_ == 0) {
            CopyPackedGate<float>(packed, gateOffset, seqLen);
        } else if (gateDtype_ == 1) {
            CopyPackedGate<bfloat16_t>(packed, gateOffset, seqLen);
        } else {
            CopyPackedGate<half>(packed, gateOffset, seqLen);
        }
        packedInQueue_.EnQue<inType>(packed);
    }

    __aicore__ inline void PreprocessQKVGate(int32_t seqLen, uint64_t head)
    {
        if (useQkL2norm_) {
            NormalizeQKRows(seqLen);
        }
        Muls(qInUb, qInUb, scale_, seqLen * alignK_);
        AscendC::PipeBarrier<PIPE_V>();
        if (useGateInKernel_) {
            ApplyGateInKernel(head, seqLen);
        }
        Exp(gateInUb, gateInUb, alignK_ * seqLen);
        AscendC::PipeBarrier<PIPE_V>();

    }

    __aicore__ inline void CopyInQKVGate(int64_t seq0, int32_t seqLen, uint64_t head,
                                         bool hasNextHead, uint64_t nextHead)
    {
        LocalTensor<inType> packed = packedInQueue_.DeQue<inType>();
        LocalTensor<inType> qLocal = packed;
        LocalTensor<inType> kLocal = packed[packedKOffset_];
        LocalTensor<inType> vLocal = packed[packedVOffset_];
        if (gateDtype_ == 0) {
            LocalTensor<float> gateLocal = packed[packedGateOffset_].template ReinterpretCast<float>();
            Adds(gateInUb, gateLocal, 0.0f, alignK_ * seqLen);
        } else if (gateDtype_ == 1) {
            LocalTensor<bfloat16_t> gateLocal =
                packed[packedGateOffset_].template ReinterpretCast<bfloat16_t>();
            Cast(gateInUb, gateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        } else {
            LocalTensor<half> gateLocal = packed[packedGateOffset_].template ReinterpretCast<half>();
            Cast(gateInUb, gateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        }
        Cast(qInUb, qLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(kInUb, kLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(vInUb, vLocal, AscendC::RoundMode::CAST_NONE, alignV_ * seqLen);
        PipeBarrier<PIPE_V>();

        // Keep the current raw slot owned until the producer acquires the other slot.
        if (hasNextHead) {
            PrefetchPackedInputs(seq0, seqLen, nextHead);
        }
        packedInQueue_.FreeTensor(packed);
        PreprocessQKVGate(seqLen, head);
    }

    __aicore__ inline void CopyInWindow(int32_t seqLen, uint64_t head, bool hasNextInput,
                                       int64_t nextStart, int32_t nextLen, uint64_t nextHead)
    {
        LocalTensor<inType> packed = packedInQueue_.DeQue<inType>();
        LocalTensor<inType> qLocal = packed;
        LocalTensor<inType> kLocal = packed[packedKOffset_];
        LocalTensor<inType> vLocal = packed[packedVOffset_];
        if (gateDtype_ == 0) {
            LocalTensor<float> gateLocal = packed[packedGateOffset_].template ReinterpretCast<float>();
            Adds(gateInUb, gateLocal, 0.0f, alignK_ * seqLen);
        } else if (gateDtype_ == 1) {
            LocalTensor<bfloat16_t> gateLocal =
                packed[packedGateOffset_].template ReinterpretCast<bfloat16_t>();
            Cast(gateInUb, gateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        } else {
            LocalTensor<half> gateLocal = packed[packedGateOffset_].template ReinterpretCast<half>();
            Cast(gateInUb, gateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        }
        Cast(qInUb, qLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(kInUb, kLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(vInUb, vLocal, AscendC::RoundMode::CAST_NONE, alignV_ * seqLen);
        PipeBarrier<PIPE_V>();

        // Keep the current raw slot owned until the producer acquires the other slot.
        if (hasNextInput) {
            PrefetchPackedInputs(nextStart, nextLen, nextHead);
        }
        packedInQueue_.FreeTensor(packed);
        PreprocessQKVGate(seqLen, head);
    }

    template <typename transType>
    __aicore__ inline void TransposeStateMatrixTyped(const LocalTensor<transType> &dst,
                                                      const LocalTensor<transType> &src,
                                                      uint32_t rowCount, uint32_t columnCount,
                                                      uint32_t srcRowStride, uint32_t dstRowStride)
    {
        constexpr uint32_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(transType);
        uint64_t dstList[STATE_TRANSPOSE_BLOCK];
        uint64_t srcList[STATE_TRANSPOSE_BLOCK];
        uint64_t dstAddr = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
        uint64_t srcAddr = reinterpret_cast<uint64_t>(src.GetPhyAddr());
        uint16_t repeatTimes = static_cast<uint16_t>(columnCount / elementsPerBlock);
        TransDataTo5HDParams transposeParams{
            false, false, static_cast<uint8_t>(repeatTimes),
            static_cast<uint16_t>(repeatTimes > 1 ? dstRowStride : 0),
            static_cast<uint16_t>(repeatTimes > 1 ? 1 : 0)};
        for (uint32_t rowBlock = 0; rowBlock < rowCount; rowBlock += STATE_TRANSPOSE_BLOCK) {
            for (uint32_t i = 0; i < STATE_TRANSPOSE_BLOCK; ++i) {
                srcList[i] = srcAddr + (rowBlock + i) * srcRowStride * sizeof(transType);
                if constexpr (sizeof(transType) == sizeof(float)) {
                    dstList[i] = dstAddr +
                        (rowBlock + (i / 2) * dstRowStride + (i % 2) * elementsPerBlock) * sizeof(transType);
                } else {
                    dstList[i] = dstAddr + (rowBlock + i * dstRowStride) * sizeof(transType);
                }
            }
            TransDataTo5HD<transType>(dstList, srcList, transposeParams);
        }
    }

    __aicore__ inline void TransposeKFirstToVFirst(const LocalTensor<stateType> &dst,
                                                   const LocalTensor<stateType> &src,
                                                   uint32_t curSingleV)
    {
        if constexpr (std::is_same<stateType, float32_t>()) {
            TransposeStateMatrixTyped<float>(dst, src, alignK_, curSingleV, vStep_, alignK_);
        } else {
            TransposeStateMatrixTyped<uint16_t>(
                dst.template ReinterpretCast<uint16_t>(), src.template ReinterpretCast<uint16_t>(),
                alignK_, curSingleV, vStep_, alignK_);
        }
    }

    __aicore__ inline void TransposeVFirstToKFirst(const LocalTensor<stateType> &dst,
                                                   const LocalTensor<stateType> &src,
                                                   uint32_t curSingleV)
    {
        if constexpr (std::is_same<stateType, float32_t>()) {
            TransposeStateMatrixTyped<float>(dst, src, curSingleV, alignK_, alignK_, vStep_);
        } else {
            TransposeStateMatrixTyped<uint16_t>(
                dst.template ReinterpretCast<uint16_t>(), src.template ReinterpretCast<uint16_t>(),
                curSingleV, alignK_, alignK_, vStep_);
        }
    }

    __aicore__ inline void PrefetchState(uint64_t stateSlot, uint64_t head, uint64_t vOffset,
                                          uint32_t curSingleV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.AllocTensor<stateType>();
        if (stateVFirst_) {
            uint64_t stateOffset =
                stateInStride0_ * stateSlot + stateInStride1_ * head + stateInStride2_ * vOffset;
            DataCopyExtParams stateInParams{static_cast<uint16_t>(curSingleV),
                                            static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0, 0};
            DataCopyPadExtParams<stateType> padParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
            DataCopyPad(stateLocal, initStateGm_[stateOffset], stateInParams, padParams);
        } else {
            uint64_t stateOffset = stateInStride0_ * stateSlot + stateInStride1_ * head +
                                   stateInStride3_ * vOffset;
            uint32_t srcStride = static_cast<uint32_t>(
                (stateInStride2_ - curSingleV) * sizeof(stateType));
            uint32_t dstStride = static_cast<uint32_t>(
                (vStep_ - curSingleV) * sizeof(stateType) / DATA_BLOCK_BYTES);
            DataCopyExtParams stateInParams{static_cast<uint16_t>(realK_),
                                            static_cast<uint32_t>(curSingleV * sizeof(stateType)),
                                            srcStride, dstStride, 0};
            DataCopyPadExtParams<stateType> padParams{false, 0, 0, 0};
            DataCopyPad(stateLocal, initStateGm_[stateOffset], stateInParams, padParams);
        }
        stateInQueue_.EnQue<stateType>(stateLocal);
    }

    __aicore__ inline void LoadPrefetchedState(uint32_t curSingleV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.DeQue<stateType>();
        if (stateVFirst_) {
            if constexpr (std::is_same<stateType, float32_t>()) {
                DataCopy(stateInUb, stateLocal, alignK_ * curSingleV);
            } else {
                Cast(stateInUb, stateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * curSingleV);
            }
        } else {
            LocalTensor<stateType> stateTransposeLocal = stateTransposeBuf_.Get<stateType>();
            TransposeKFirstToVFirst(stateTransposeLocal, stateLocal, curSingleV);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<stateType, float32_t>()) {
                Adds(stateInUb, stateTransposeLocal, 0.0f, alignK_ * curSingleV);
            } else {
                Cast(stateInUb, stateTransposeLocal, AscendC::RoundMode::CAST_NONE, alignK_ * curSingleV);
            }
        }
        stateInQueue_.FreeTensor(stateLocal);
    }


    __aicore__ inline void MatVecMul(const LocalTensor<float> &cubeTensor, const LocalTensor<float> &vecTensor,
                                     LocalTensor<float> &dstTensor, uint32_t cols, bool isAdd)
    {
        uint8_t repeatStride = alignK_ / FP32_NUM_PER_BLOCK;
        for (uint32_t i = 0; i < alignK_; i += REPEAT_LENTH) {
            uint64_t mask = Std::min(REPEAT_LENTH, alignK_ - i);
            for (uint32_t j = 0; j < cols; j += MAX_REPEAT_TIME) {
                uint64_t repeatTime = Std::min(MAX_REPEAT_TIME, cols - j);
                if (isAdd) {
                    MulAddDst(dstTensor[j * alignK_ + i], cubeTensor[j * alignK_ + i], vecTensor[i], mask, repeatTime,
                              {1, 1, 1, repeatStride, repeatStride, 0});
                } else {
                    Mul(dstTensor[j * alignK_ + i], cubeTensor[j * alignK_ + i], vecTensor[i], mask, repeatTime,
                        {1, 1, 1, repeatStride, repeatStride, 0});
                }
            }
        }
    }

    __aicore__ inline void ReduceSumBaseline(LocalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor,
                                             uint32_t rows)
    {
        uint32_t stateShape[2] = {rows, alignK_};
        ReduceSum<float, Pattern::Reduce::AR, true>(dstTensor, srcTensor, stateShape, true);
    }

    __aicore__ inline bool CanUseK128AddFoldFastPath(uint32_t rows) const
    {
        if (alignK_ != ADD_FOLD_REDUCE_MIN_K) {
            return false;
        }
        if (rows == 0 || rows > MAX_REPEAT_TIME) {
            return false;
        }
        return true;
    }

    __aicore__ inline void ReduceSumAddFoldK128(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                                uint32_t rows)
    {
        const uint8_t repeatTime = static_cast<uint8_t>(rows);
        const uint8_t rowRepStride = static_cast<uint8_t>(alignK_ / FP32_NUM_PER_BLOCK);

        Add(srcTensor[REPEAT_LENTH], srcTensor, srcTensor[REPEAT_LENTH], REPEAT_LENTH, repeatTime,
            {1, 1, 1, rowRepStride, rowRepStride, rowRepStride});
        AscendC::PipeBarrier<PIPE_V>();
        WholeReduceSum(dstTensor, srcTensor[REPEAT_LENTH], REPEAT_LENTH, repeatTime, 1, 1, rowRepStride);
    }

    __aicore__ inline void ReduceSumAddFold(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                            uint32_t rows)
    {
        if (alignK_ < REPEAT_LENTH) {
            ReduceSumBaseline(dstTensor, srcTensor, rows);
            return;
        }

        if ((alignK_ & (alignK_ - 1)) != 0) {
            ReduceSumBaseline(dstTensor, srcTensor, rows);
            return;
        }

        if (CanUseK128AddFoldFastPath(rows)) {
            ReduceSumAddFoldK128(dstTensor, srcTensor, rows);
            return;
        }

        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t rowOffset = row * alignK_;
            uint32_t activeLen = alignK_;
            while (activeLen > REPEAT_LENTH) {
                uint32_t half = activeLen >> 1;
                Add(srcTensor[rowOffset], srcTensor[rowOffset], srcTensor[rowOffset + half], half);
                AscendC::PipeBarrier<PIPE_V>();
                activeLen = half;
            }

            WholeReduceSum(dstTensor[row], srcTensor[rowOffset], REPEAT_LENTH, 1, 1, 1, FP32_NUM_PER_BLOCK);
        }
    }

    __aicore__ inline void ReduceSumDispatch(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                             uint32_t rows)
    {
        if (useAddFoldReduce_ && alignK_ >= ADD_FOLD_REDUCE_MIN_K) {
            ReduceSumAddFold(dstTensor, srcTensor, rows);
            return;
        }
        ReduceSumBaseline(dstTensor, srcTensor, rows);
    }

    __aicore__ inline void Compute(uint32_t curSingleV, uint64_t curQKOffset, uint64_t curVOffset)
    {
        MatVecMul(stateInUb, gateInUb[curQKOffset], stateInUb, curSingleV, false);
        AscendC::PipeBarrier<PIPE_V>();
        MatVecMul(stateInUb, kInUb[curQKOffset], broadTmpInUb, curSingleV, false);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSumDispatch(deltaInUb, broadTmpInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> vActive = tmpBuff.GetWithOffset<float>(
            curSingleV, vUbByteOffset_ + curVOffset * sizeof(float));
        Sub(deltaInUb, vActive, deltaInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(deltaInUb, deltaInUb, beta_, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        // Supported V sizes and vStep are multiples of eight rows.
        // Keep one 32-byte block per residual instead of copying it across K.
        Brcb(broadTmpInUb, deltaInUb, curSingleV / FP32_NUM_PER_BLOCK, {1, FP32_NUM_PER_BLOCK});
        AscendC::PipeBarrier<PIPE_V>();
        const uint8_t rowStride = static_cast<uint8_t>(alignK_ / FP32_NUM_PER_BLOCK);
        for (uint32_t k = 0; k < alignK_; k += REPEAT_LENTH) {
            MulAddDst(stateInUb[k], broadTmpInUb, kInUb[curQKOffset + k],
                      REPEAT_LENTH, curSingleV, {1, 0, 1, rowStride, 1, 0});
        }
        AscendC::PipeBarrier<PIPE_V>();
        MatVecMul(stateInUb, qInUb[curQKOffset], broadTmpInUb, curSingleV, false);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSumDispatch(attnInUb, broadTmpInUb, curSingleV);
        LocalTensor<outType> attnOutLocal = attnOutQueue_.AllocTensor<outType>();
        if (shouldStoreState_) {
            LocalTensor<stateType> stateOutLocal = stateOutQueue_.AllocTensor<stateType>();
            if (stateVFirst_) {
                if constexpr (std::is_same<stateType, float32_t>()) {
                    DataCopy(stateOutLocal, stateInUb, alignK_ * curSingleV);
                } else {
                    Cast(stateOutLocal, stateInUb, AscendC::RoundMode::CAST_RINT, alignK_ * curSingleV);
                }
            } else {
                if constexpr (std::is_same<stateType, float32_t>()) {
                    TransposeVFirstToKFirst(stateOutLocal, stateInUb, curSingleV);
                } else {
                    LocalTensor<stateType> stateTransposeLocal = stateTransposeBuf_.Get<stateType>();
                    Cast(stateTransposeLocal, stateInUb, AscendC::RoundMode::CAST_RINT, alignK_ * curSingleV);
                    AscendC::PipeBarrier<PIPE_V>();
                    TransposeVFirstToKFirst(stateOutLocal, stateTransposeLocal, curSingleV);
                }
            }
            stateOutQueue_.EnQue<stateType>(stateOutLocal);
        }
        Cast(attnOutLocal, attnInUb, AscendC::RoundMode::CAST_RINT, curSingleV);
        attnOutQueue_.EnQue<outType>(attnOutLocal);
    }

    __aicore__ inline void CopyOutAttn(uint64_t attnOffset, uint32_t curSingleV)
    {
        LocalTensor<outType> attnLocal = attnOutQueue_.DeQue<outType>();
        DataCopyParams attnOutParams{1, static_cast<uint16_t>(curSingleV * sizeof(outType)), 0, 0};
        DataCopyPad(attnOutGm_[attnOffset], attnLocal, attnOutParams);
        attnOutQueue_.FreeTensor(attnLocal);
    }

    __aicore__ inline void CopyOutState(uint64_t stateSlot, uint64_t head, uint64_t vOffset,
                                         uint32_t curSingleV)
    {
        LocalTensor<stateType> stateOutLocal = stateOutQueue_.DeQue<stateType>();
        if (stateVFirst_) {
            uint64_t stateOffset =
                stateOutStride0_ * stateSlot + stateOutStride1_ * head + stateOutStride2_ * vOffset;
            DataCopyParams stateOutParams{static_cast<uint16_t>(curSingleV),
                                          static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0};
            DataCopyPad(finalStateGm_[stateOffset], stateOutLocal, stateOutParams);
        } else {
            uint64_t stateOffset = stateOutStride0_ * stateSlot + stateOutStride1_ * head +
                                   stateOutStride3_ * vOffset;
            uint32_t srcStride = static_cast<uint32_t>(
                (vStep_ - curSingleV) * sizeof(stateType) / DATA_BLOCK_BYTES);
            uint32_t dstStride = static_cast<uint32_t>(
                (stateOutStride2_ - curSingleV) * sizeof(stateType));
            DataCopyExtParams stateOutParams{static_cast<uint16_t>(realK_),
                                             static_cast<uint32_t>(curSingleV * sizeof(stateType)),
                                             srcStride, dstStride, 0};
            DataCopyPad(finalStateGm_[stateOffset], stateOutLocal, stateOutParams);
        }
        stateOutQueue_.FreeTensor(stateOutLocal);
    }

    template <typename betaType>
    __aicore__ inline void CopyInBetaTyped(int64_t seq0, int64_t seq1)
    {
        int64_t seqLen = seq1 - seq0;
        uint64_t betaCount = static_cast<uint64_t>(seqLen) * NV_;
        uint64_t betaBatchSize = Ceil(betaCount, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
        LocalTensor<betaType> betaLocal = betaInQueue_.AllocTensor<betaType>();
        if constexpr (std::is_same<betaType, float32_t>()) {
            CopyVectorIn(betaLocal, betaFloatGm_, static_cast<uint64_t>(seq0) * NV_, betaCount);
        } else if constexpr (std::is_same<betaType, bfloat16_t>()) {
            CopyVectorIn(betaLocal, betaBf16Gm_, static_cast<uint64_t>(seq0) * NV_, betaCount);
        } else {
            CopyVectorIn(betaLocal, betaFp16Gm_, static_cast<uint64_t>(seq0) * NV_, betaCount);
        }
        betaInQueue_.EnQue<betaType>(betaLocal);
        betaLocal = betaInQueue_.DeQue<betaType>();
        if constexpr (std::is_same<betaType, float32_t>()) {
            Adds(betaInUb, betaLocal, 0.0f, static_cast<uint32_t>(betaBatchSize));
        } else {
            Cast(betaInUb, betaLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(betaBatchSize));
        }
        betaInQueue_.FreeTensor(betaLocal);
        PipeBarrier<PIPE_V>();
        SyncVToS();
    }

    __aicore__ inline void CopyInBeta(int64_t seq0, int64_t seq1)
    {
        if (betaDtype_ == 0) {
            CopyInBetaTyped<float>(seq0, seq1);
        } else if (betaDtype_ == 1) {
            CopyInBetaTyped<bfloat16_t>(seq0, seq1);
        } else {
            CopyInBetaTyped<half>(seq0, seq1);
        }
    }

    __aicore__ inline uint64_t StateSlotForToken(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        if (hasSsmStateIndices_) {
            return LoadStateSlot(batchIdx, seq0, tokenIdx);
        }
        return batchIdx;
    }

    __aicore__ inline float LoadBeta(uint64_t gbOffset)
    {
        float beta = betaInUb.GetValue(gbOffset);
        if (useBetaSigmoid_) {
            beta = SigmoidScalar(beta);
            if (allowNegEigval_) {
                beta *= 2.0f;
            }
        }
        return beta;
    }

    template <int32_t ShortTokens>
    __aicore__ inline void ProcessHeadTasks(uint64_t batchIdx, int64_t seq0, int64_t seq1,
                                           uint64_t head, uint64_t stateSlot)
    {
        const uint64_t headEnd = TaskHeadEnd(batchIdx);
        bool statePrefetched = false;
        while (head < headEnd) {
            uint64_t nextHead = head + 1;
            statePrefetched = ProcessHead<ShortTokens>(batchIdx, seq0, seq1, head, stateSlot,
                                                      statePrefetched, nextHead < headEnd, stateSlot, nextHead);
            head = nextHead;
        }
    }

    template <int32_t ShortTokens>
    __aicore__ inline bool ProcessHead(uint64_t batchIdx, int64_t seq0, int64_t seq1,
                                      uint64_t head_i, uint64_t stateSlot, bool statePrefetched,
                                      bool hasNextHead, uint64_t nextStateSlot, uint64_t nextHead)
    {
        const int64_t sequenceLength = seq1 - seq0;
        const bool windowed = ShortTokens == 0 && sequenceLength > static_cast<int64_t>(TOKEN_WINDOW);
        if (!windowed) {
            // Keep the baseline short-path order and bound the Beta view's live range.
            const int32_t shortLength =
                ShortTokens > 0 ? ShortTokens : static_cast<int32_t>(sequenceLength);
            CopyInQKVGate(seq0, shortLength, head_i, hasNextHead, nextHead);
            if (realV_ == 0) {
                return false;
            }
        }
        LocalTensor<float> betaValues = scalarBuf_.GetWithOffset<float>(TOKEN_WINDOW, DATA_BLOCK_BYTES);
        if (!windowed) {
            const int32_t shortLength =
                ShortTokens > 0 ? ShortTokens : static_cast<int32_t>(sequenceLength);
            for (int32_t token = 0; token < shortLength; ++token) {
                betaValues.SetValue(token, LoadBeta(head_i + static_cast<uint64_t>(token) * NV_));
            }
        }
        uint64_t nextVOffset = statePrefetched ? vStep_ : 0;
        uint32_t queuedBufferNum = statePrefetched ? 1 : 0;
        for (uint32_t bufferIdx = queuedBufferNum;
             bufferIdx < INPUT_BUFFER_NUM && nextVOffset < realV_; ++bufferIdx) {
            uint32_t nextSingleV =
                nextVOffset + vStep_ > realV_ ? realV_ - nextVOffset : vStep_;
            PrefetchState(stateSlot, head_i, nextVOffset, nextSingleV);
            nextVOffset += vStep_;
        }
        bool nextStatePrefetched = false;
        for (uint64_t v_i = 0; v_i < realV_; v_i += vStep_) {
            uint32_t curSingleV = v_i + vStep_ > realV_ ? realV_ - v_i : vStep_;
            LoadPrefetchedState(curSingleV);
            if (nextVOffset < realV_) {
                uint32_t nextSingleV =
                    nextVOffset + vStep_ > realV_ ? realV_ - nextVOffset : vStep_;
                PrefetchState(stateSlot, head_i, nextVOffset, nextSingleV);
                nextVOffset += vStep_;
            } else if (hasNextHead && !nextStatePrefetched) {
                uint32_t nextSingleV = realV_ > vStep_ ? vStep_ : realV_;
                PrefetchState(nextStateSlot, nextHead, 0, nextSingleV);
                nextStatePrefetched = true;
            }
            uint64_t pendingAttnOffset = 0;
            uint64_t pendingStateSlot = 0;
            bool hasPendingAttn = false;
            bool hasPendingState = false;
            int64_t windowStart = seq0;
            do {
                const int64_t remaining = seq1 - windowStart;
                const int32_t windowLen = ShortTokens > 0 ? ShortTokens :
                    (ShortTokens < 0 ? static_cast<int32_t>(sequenceLength) : static_cast<int32_t>(
                        remaining < static_cast<int64_t>(TOKEN_WINDOW) ? remaining : TOKEN_WINDOW));
                const int64_t windowEnd = ShortTokens < 0 ? seq1 : windowStart + windowLen;
                if (windowed) {
                    // The previous window must release its FP32 inputs and shared scratch.
                    PipeBarrier<PIPE_V>();
                    CopyInBeta(windowStart, windowEnd);

                    int64_t nextStart = windowEnd;
                    uint64_t nextInputHead = head_i;
                    bool hasNextInput = true;
                    if (windowEnd == seq1) {
                        nextStart = seq0;
                        if (v_i + vStep_ >= realV_) {
                            hasNextInput = hasNextHead;
                            nextInputHead = nextHead;
                        }
                    }
                    const int64_t nextRemaining = seq1 - nextStart;
                    const int32_t nextLen = static_cast<int32_t>(
                        nextRemaining < static_cast<int64_t>(TOKEN_WINDOW) ? nextRemaining : TOKEN_WINDOW);
                    CopyInWindow(windowLen, head_i, hasNextInput, nextStart, nextLen, nextInputHead);
                    for (int32_t token = 0; token < windowLen; ++token) {
                        betaValues.SetValue(token, LoadBeta(head_i + static_cast<uint64_t>(token) * NV_));
                    }
                }
                for (int64_t seq_i = windowStart; seq_i < windowEnd; seq_i++) {
                    uint64_t curQKOffset = static_cast<uint64_t>(seq_i - windowStart) * alignK_;
                    uint64_t curVOffset = static_cast<uint64_t>(seq_i - windowStart) * alignV_ + v_i;
                    uint64_t attnOffset = (static_cast<uint64_t>(seq_i) * NV_ + head_i) * realV_ + v_i;
                    uint64_t curStateSlot = StateSlotForToken(batchIdx, seq0, seq_i);
                    uint64_t curStateOutSlot = curStateSlot;
                    beta_ = betaValues.GetValue(seq_i - windowStart);
                    Compute(curSingleV, curQKOffset, curVOffset);
                    if (attnOutBufferNum_ == BUFFER_NUM) {
                        CopyOutAttn(attnOffset, curSingleV);
                    } else {
                        if (hasPendingAttn) {
                            CopyOutAttn(pendingAttnOffset, curSingleV);
                        }
                        pendingAttnOffset = attnOffset;
                        hasPendingAttn = true;
                    }
                    if (shouldStoreState_) {
                        if (stateOutBufferNum_ == BUFFER_NUM) {
                            CopyOutState(curStateOutSlot, head_i, v_i, curSingleV);
                        } else {
                            if (hasPendingState) {
                                CopyOutState(pendingStateSlot, head_i, v_i, curSingleV);
                            }
                            pendingStateSlot = curStateOutSlot;
                            hasPendingState = true;
                        }
                    }
                }
                if constexpr (ShortTokens != 0) {
                    break; // Short instantiations have no outer window loop at runtime.
                }
                windowStart = windowEnd;
            } while (windowStart < seq1);
            if (hasPendingAttn) {
                CopyOutAttn(pendingAttnOffset, curSingleV);
            }
            if (hasPendingState) {
                // The final two snapshots can target the same GM slot without another
                // queue allocation to wait for the preceding MTE3 write.
                if (seq1 - seq0 > 1 &&
                    pendingStateSlot == StateSlotForToken(batchIdx, seq0, seq1 - 2)) {
                    int32_t eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>();
                    SetFlag<HardEvent::MTE3_S>(eventId);
                    WaitFlag<HardEvent::MTE3_S>(eventId);
                    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_S>(eventId);
                }
                CopyOutState(pendingStateSlot, head_i, v_i, curSingleV);
            }
        }
        return nextStatePrefetched;
    }

    __aicore__ inline void InitTaskRange(uint64_t coreNum)
    {
        const uint64_t totalTasks = static_cast<uint64_t>(B_) * NV_;
        const uint64_t baseTasks = totalTasks / coreNum;
        const uint64_t extraTasks = totalTasks % coreNum;
        coreTaskBegin_ = blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        coreTaskEnd_ = coreTaskBegin_ + baseTasks + (blockIdx < extraTasks ? 1 : 0);
    }

    __aicore__ inline uint64_t TaskHeadEnd(uint64_t batchIdx) const
    {
        const uint64_t sequenceBegin = batchIdx * NV_;
        if (coreTaskEnd_ <= sequenceBegin) {
            return 0;
        }
        const uint64_t headEnd = coreTaskEnd_ - sequenceBegin;
        return headEnd < NV_ ? headEnd : NV_;
    }

    __aicore__ inline uint64_t FindNextTaskHead(uint64_t batchIdx, uint64_t startHead) const
    {
        const uint64_t sequenceBegin = batchIdx * NV_;
        const uint64_t requestedBegin = sequenceBegin + startHead;
        if (coreTaskBegin_ >= sequenceBegin + NV_ || coreTaskEnd_ <= requestedBegin) {
            return NV_;
        }
        const uint64_t task = coreTaskBegin_ > requestedBegin ? coreTaskBegin_ : requestedBegin;
        return task - sequenceBegin;
    }

private:
    GlobalTensor<inType> queryGm_;
    GlobalTensor<inType> keyGm_;
    GlobalTensor<inType> valueGm_;
    GlobalTensor<float> gateFloatGm_;
    GlobalTensor<bfloat16_t> gateBf16Gm_;
    GlobalTensor<half> gateFp16Gm_;
    GlobalTensor<float> betaFloatGm_;
    GlobalTensor<bfloat16_t> betaBf16Gm_;
    GlobalTensor<half> betaFp16Gm_;
    GlobalTensor<stateType> initStateGm_;
    GlobalTensor<int32_t> cuSeqlensInt32Gm_;
    GlobalTensor<int64_t> cuSeqlensInt64Gm_;
    GlobalTensor<int32_t> ssmStateIndicesInt32Gm_;
    GlobalTensor<int64_t> ssmStateIndicesInt64Gm_;
    GlobalTensor<float> aLogGm_;
    GlobalTensor<float> dtBiasGm_;
    GlobalTensor<int32_t> numAcceptedTokensInt32Gm_;
    GlobalTensor<int64_t> numAcceptedTokensInt64Gm_;
    GlobalTensor<stateType> finalStateGm_;
    GlobalTensor<outType> attnOutGm_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, PACKED_INPUT_BUFFER_NUM> packedInQueue_;
    TQue<QuePosition::VECIN, 1> betaInQueue_;
    TQue<QuePosition::VECIN, INPUT_BUFFER_NUM> stateInQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> attnOutQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> stateOutQueue_;
    TBuf<TPosition::VECCALC> tmpBuff;
    TBuf<TPosition::VECCALC> stateTransposeBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    uint32_t packedKOffset_;
    uint32_t packedVOffset_;
    uint32_t packedGateOffset_;
    LocalTensor<float> qInUb;
    LocalTensor<float> kInUb;
    LocalTensor<float> vInUb;
    LocalTensor<float> gateInUb;
    LocalTensor<float> betaInUb;
    LocalTensor<float> deltaInUb;
    LocalTensor<float> broadTmpInUb;
    LocalTensor<float> attnInUb;
    LocalTensor<float> stateInUb;
    TEventID eventIdMte2ToV_;
    TEventID eventIdVToMte2_;
    TEventID eventIdVToS_;
    TEventID eventIdVToMte3_;
    TEventID eventIdMte3ToV_;
    bool eventMte2ToVInitialized_;
    bool eventVToMte2Initialized_;
    bool eventVToSInitialized_;
    bool eventVToMte3Initialized_;
    bool eventMte3ToVInitialized_;
    uint32_t B_;
    uint32_t T_;
    uint32_t seqLen_;
    uint32_t NK_;
    uint32_t alignK_;
    uint32_t realK_;
    uint32_t NV_;
    uint32_t alignV_;
    uint32_t realV_;
    uint32_t stateCapacity_;
    uint32_t ssmStateStride_;
    uint64_t queryTokenStride_;
    uint64_t queryHeadStride_;
    uint64_t keyTokenStride_;
    uint64_t keyHeadStride_;
    uint64_t valueTokenStride_;
    uint64_t valueHeadStride_;
    uint64_t stateInStride0_;
    uint64_t stateInStride1_;
    uint64_t stateInStride2_;
    uint64_t stateInStride3_;
    uint64_t stateOutStride0_;
    uint64_t stateOutStride1_;
    uint64_t stateOutStride2_;
    uint64_t stateOutStride3_;
    uint32_t vStep_;
    uint32_t vUbByteOffset_;
    uint32_t stateOutBufferNum_;
    uint32_t attnOutBufferNum_;
    uint32_t restUbSize_;
    uint32_t gateDtype_;
    uint32_t betaDtype_;
    uint32_t cuSeqlensDtype_;
    uint32_t ssmStateIndicesDtype_;
    uint32_t acceptedTokensDtype_;
    bool hasCuSeqlens_;
    bool hasSsmStateIndices_;
    bool hasAcceptedTokens_;
    bool hasALog_;
    bool hasDtBias_;
    bool useQkL2norm_;
    bool useGateInKernel_;
    bool useBetaSigmoid_;
    bool allowNegEigval_;
    bool safeGate_;
    bool stateVFirst_;
    bool shouldStoreState_;
    bool useAddFoldReduce_;
    float beta_;
    float scale_;
    float lowerBound_;
    uint64_t blockIdx;
    uint64_t coreTaskBegin_;
    uint64_t coreTaskEnd_;
};
} // namespace RecurrentKda
#endif
