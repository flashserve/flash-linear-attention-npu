/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule.h
 * \brief
 */

#ifndef __RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define __RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_utils/vector/regbase.hpp"
#include "../recurrent_gated_delta_rule_tiling_data.h"

namespace RecurrentGatedDeltaRule {

using namespace matmul;
using namespace AscendC;
using namespace AscendC::MicroAPI;
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint64_t MAX_MTP = 8;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

constexpr CastTrait castTraitB16ToB32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static CastTrait castTraitFp32ToB16ZeroRint = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, RoundMode::CAST_RINT};
constexpr static CastTrait castTraitFp32ToB16OneRint = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
struct RGDRInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gama;
    GM_ADDR gamaK;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR ssmStateIndices;
    GM_ADDR numAcceptedTokens;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename stateType, typename outType, bool hasGama, bool hasGamaK>
__simd_vf__ inline void ComputeRecurrentGatedDeltaRuleVF(
    __ubuf__ float *state, __ubuf__ float *key, __ubuf__ float *query, __ubuf__ float *value,
    __ubuf__ float *gamaK, __ubuf__ stateType *stateOut, __ubuf__ outType *attnOut,
    __ubuf__ float *attnTmp, uint16_t rows, uint16_t kLength, float gama, float beta)
{
    // The public interface fixes Dk to 128, i.e. two FP32 vector registers on A5.
    // Keeping the complete recurrence in one VF lets both state halves remain in registers
    // across gate, dot-product, rank-one update, output projection and output conversion.
    constexpr uint16_t FP32_PER_REG = VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullFp32Mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg fullB16Mask = CreateMask<bfloat16_t, MaskPattern::ALL>();

    RegTensor<float> key0;
    RegTensor<float> key1;
    RegTensor<float> query0;
    RegTensor<float> query1;
    if constexpr (std::is_same<stateType, float32_t>()) {
        DataCopy(key0, key);
        DataCopy(key1, key + FP32_PER_REG);
        DataCopy(query0, query);
        DataCopy(query1, query + FP32_PER_REG);
    } else {
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(key0, key1, key);
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(query0, query1, query);
    }

    RegTensor<float> gamaK0;
    RegTensor<float> gamaK1;
    if constexpr (hasGamaK) {
        if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(gamaK0, gamaK);
            DataCopy(gamaK1, gamaK + FP32_PER_REG);
        } else {
            DataCopy<float, LoadDist::DIST_DINTLV_B32>(gamaK0, gamaK1, gamaK);
        }
    }

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * kLength;
        RegTensor<float> state0;
        RegTensor<float> state1;
        if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(state0, state + rowOffset);
            DataCopy(state1, state + rowOffset + FP32_PER_REG);
        } else {
            DataCopy<float, LoadDist::DIST_DINTLV_B32>(state0, state1, state + rowOffset);
        }

        if constexpr (hasGama) {
            Muls(state0, state0, gama, fullFp32Mask);
            Muls(state1, state1, gama, fullFp32Mask);
        }
        if constexpr (hasGamaK) {
            Mul(state0, state0, gamaK0, fullFp32Mask);
            Mul(state1, state1, gamaK1, fullFp32Mask);
        }

        RegTensor<float> dot0;
        RegTensor<float> dot1;
        RegTensor<float> dotSum;
        Mul(dot0, state0, key0, fullFp32Mask);
        Mul(dot1, state1, key1, fullFp32Mask);
        Add(dot0, dot0, dot1, fullFp32Mask);
        ReduceSum(dotSum, dot0, fullFp32Mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(attnTmp + row, dotSum, fullFp32Mask);
        if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(state + rowOffset, state0, fullFp32Mask);
            DataCopy(state + rowOffset + FP32_PER_REG, state1, fullFp32Mask);
        } else {
            DataCopy<float, StoreDist::DIST_INTLV_B32>(state + rowOffset, state0, state1, fullFp32Mask);
        }
    }

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * kLength;

        RegTensor<float> attn;
        RegTensor<float> delta;
        DataCopy<float, LoadDist::DIST_BRC_B32>(attn, value + row);
        RegTensor<float> dotSum;
        DataCopy<float, LoadDist::DIST_BRC_B32>(dotSum, attnTmp + row);
        Sub(attn, attn, dotSum, fullFp32Mask);
        Muls(delta, attn, beta, fullFp32Mask);

        RegTensor<float> state0;
        RegTensor<float> state1;
        if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(state0, state + rowOffset);
            DataCopy(state1, state + rowOffset + FP32_PER_REG);
        } else {
            DataCopy<float, LoadDist::DIST_DINTLV_B32>(state0, state1, state + rowOffset);
        }
        RegTensor<float> update0;
        RegTensor<float> update1;
        Mul(update0, delta, key0, fullFp32Mask);
        Mul(update1, delta, key1, fullFp32Mask);
        Add(state0, state0, update0, fullFp32Mask);
        Add(state1, state1, update1, fullFp32Mask);
        if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(state + rowOffset, state0, fullFp32Mask);
            DataCopy(state + rowOffset + FP32_PER_REG, state1, fullFp32Mask);
        } else {
            DataCopy<float, StoreDist::DIST_INTLV_B32>(state + rowOffset, state0, state1, fullFp32Mask);
        }

        RegTensor<float> out0;
        RegTensor<float> out1;
        RegTensor<float> outSum;
        Mul(out0, state0, query0, fullFp32Mask);
        Mul(out1, state1, query1, fullFp32Mask);
        Add(out0, out0, out1, fullFp32Mask);
        ReduceSum(outSum, out0, fullFp32Mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(attnTmp + row, outSum, fullFp32Mask);

        if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(stateOut + rowOffset, state0, fullFp32Mask);
            DataCopy(stateOut + rowOffset + FP32_PER_REG, state1, fullFp32Mask);
        } else {
            RegTensor<stateType> stateOutReg;
            Cast<stateType, float, castTraitFp32ToB16OneRint>(
                stateOutReg, state1, fullFp32Mask);
            Cast<stateType, float, castTraitFp32ToB16ZeroRint>(
                stateOutReg, state0, fullFp32Mask);
            StoreAlign(stateOut + rowOffset, stateOutReg, fullB16Mask);
        }
    }

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    if constexpr (std::is_same<outType, float32_t>()) {
        uint32_t remaining = rows;
        for (uint16_t offset = 0; offset < rows; offset += FP32_PER_REG) {
            MaskReg mask = UpdateMask<float>(remaining);
            RegTensor<float> output;
            DataCopy(output, attnTmp + offset);
            DataCopy(attnOut + offset, output, mask);
        }
    } else {
        RegTensor<float> output0;
        RegTensor<float> output1;
        RegTensor<outType> output;
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(output0, output1, attnTmp);
        Cast<outType, float, castTraitFp32ToB16OneRint>(
            output, output1, fullFp32Mask);
        Cast<outType, float, castTraitFp32ToB16ZeroRint>(
            output, output0, fullFp32Mask);
        uint32_t remaining = rows;
        MaskReg outputMask = UpdateMask<outType>(remaining);
        StoreAlign(attnOut, output, outputMask);
    }
}

template <typename inType, typename outType, typename stateType>
class RGDR {
public:
    __aicore__ inline RGDR(const RecurrentGatedDeltaRuleTilingData *tilingData)
    {
        B_ = tilingData->b;
        T_ = tilingData->t;
        NK_ = tilingData->nk;
        realK_ = tilingData->dk;
        NV_ = tilingData->nv;
        realV_ = tilingData->dv;
        scale_ = tilingData->scale;
        stateStride0_ = tilingData->stateStride0;
        stateStride1_ = tilingData->stateStride1;
        stateStride2_ = tilingData->stateStride2;
        hasAcceptedTokens_ = (tilingData->hasAcceptedTokens == 1);
        hasGama_ = (tilingData->hasGama == 1);
        hasGamaK_ = (tilingData->hasGamaK == 1);
        vStep_ = tilingData->vStep;
        stateOutBufferNum_ = (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        attnOutBufferNum_ = (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        restUbSize_ = tilingData->ubRestBytes;
        alignK_ = Ceil(tilingData->dk, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(tilingData->dv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        load = 0;
        usedblk = 0;
    }

    __aicore__ inline void Init(const RGDRInitParams &initParams, TPipe *pipe)
    {
        uint64_t blockDim = GetBlockNum();
        blockIdx = GetBlockIdx();
        if (blockIdx >= blockDim) {
            return;
        }
        pipe_ = pipe;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void SetGlobalTensors(const RGDRInitParams &initParams)
    {
        queryGm_.SetGlobalBuffer((__gm__ inType *)initParams.query);
        keyGm_.SetGlobalBuffer((__gm__ inType *)initParams.key);
        valueGm_.SetGlobalBuffer((__gm__ inType *)initParams.value);
        gamaGm_.SetGlobalBuffer((__gm__ float *)initParams.gama);
        gamaKGm_.SetGlobalBuffer((__gm__ float *)initParams.gamaK);
        betaGm_.SetGlobalBuffer((__gm__ inType *)initParams.beta);
        initStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.initState);
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens);
        ssmStateIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
        numAcceptedTokensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.numAcceptedTokens);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.finalState);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t cubeSize = alignK_ * vStep_ * sizeof(float);
        uint32_t singleVSize = vStep_ * sizeof(float);
        uint32_t vSize = MAX_MTP * alignV_ * sizeof(float);
        uint32_t kSize = MAX_MTP * alignK_ * sizeof(float);
        uint32_t betaNumAlign = Ceil(MAX_MTP * NV_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        uint32_t betaUbSize = betaNumAlign * sizeof(float); //  8: 8 * 4 = 32B;
        pipe_->InitBuffer(qInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(kInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(vInQueue_, BUFFER_NUM, MAX_MTP * alignV_ * sizeof(inType));
        pipe_->InitBuffer(stateInQueue_, BUFFER_NUM, alignK_ * vStep_ * sizeof(stateType));
        if (hasGama_) {
            pipe_->InitBuffer(gamaInQueue_, BUFFER_NUM, MAX_MTP * NV_ * sizeof(float));
        }
        if (hasGamaK_) {
            pipe_->InitBuffer(gamaKInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(float));
        }
        pipe_->InitBuffer(betaInQueue_, BUFFER_NUM, MAX_MTP * NV_ * sizeof(inType));
        pipe_->InitBuffer(stateOutQueue_, stateOutBufferNum_, alignK_ * vStep_ * sizeof(stateType));
        pipe_->InitBuffer(attnOutQueue_, attnOutBufferNum_, vStep_ * sizeof(outType));
        pipe_->InitBuffer(tmpBuff, restUbSize_);
        uint32_t buffOffset = 0;
        attnInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        vInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignV_), buffOffset);
        buffOffset += vSize;
        qInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        kInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        stateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;
        betaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaNumAlign), buffOffset);
        buffOffset += betaUbSize;
        gamaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaNumAlign), buffOffset);
    }

    __aicore__ inline void ComputeAvgload()
    {
        uint64_t realT = 0;
        for (uint64_t batch_i = 1; batch_i < B_ + 1; batch_i++) {
            realT += cuSeqlensGm_.GetValue(batch_i);
        }
        avgload = Ceil(realT * NV_, GetBlockNum());
    }

    __aicore__ inline void Process()
    {
        ComputeAvgload();
        int32_t seq1 = cuSeqlensGm_.GetValue(0);
        for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
            int32_t seqLen = cuSeqlensGm_.GetValue(batch_i+1);
            if (seqLen <= 0) {
                continue;
            }
            if (seqLen > static_cast<int32_t>(MAX_MTP)) {
                return;
            }
            if (seq1 < 0 || seq1 > static_cast<int32_t>(T_) || (seq1 + seqLen) > static_cast<int32_t>(T_)) {
                return;
            }
            int32_t seq0 = seq1;
            seq1 += seqLen;
            uint32_t copyFlag = 0;
            uint64_t stateOffset;
            for (uint64_t head_i = 0; head_i < NV_; head_i++) {
                if (!IsCurrentBlock(seq1 - seq0)) {
                    continue;
                }
                copyFlag++;
                if (copyFlag == 1) {
                    int32_t stateTokenIdx = seq0;
                    if (hasAcceptedTokens_) {
                        int32_t acceptedTokenNum = numAcceptedTokensGm_.GetValue(batch_i);
                        if (acceptedTokenNum <= 0 || acceptedTokenNum > seqLen) {
                            return;
                        }
                        stateTokenIdx = seq0 + acceptedTokenNum - 1;
                    }
                    stateOffset = ssmStateIndicesGm_.GetValue(stateTokenIdx);
                    CopyInGamaBeta(seq0, seq1);
                }
                ProcessHead(seq0, seq1, head_i, stateOffset);
            }
        }
    }

private:
    __aicore__ inline void CopyInQKV(uint64_t vOffset, uint64_t qkOffset, int32_t seqLen)
    {
        LocalTensor<inType> qLocal = qInQueue_.AllocTensor<inType>();
        LocalTensor<inType> kLocal = kInQueue_.AllocTensor<inType>();
        LocalTensor<inType> vLocal = vInQueue_.AllocTensor<inType>();
        DataCopyExtParams qkInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                     static_cast<uint32_t>((NK_ - 1) * realK_ * sizeof(inType)), 0, 0};
        DataCopyExtParams vInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
                                    static_cast<uint32_t>((NV_ - 1) * realV_ * sizeof(inType)), 0, 0};
        DataCopyPadExtParams<inType> qkPadParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPadExtParams<inType> vPadParams{true, 0, static_cast<uint8_t>(alignV_ - realV_), 0};
        if (hasGamaK_) {
            uint32_t alignKGamma = Ceil(realK_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
            uint32_t stride = alignKGamma < alignK_ ? 1 : 0;
            DataCopyExtParams gkInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(float)),
                                     static_cast<uint32_t>((NV_ - 1) * realK_ * sizeof(float)), stride, 0};
            DataCopyPadExtParams<float> gkPadParams{true, 0, static_cast<uint8_t>(alignKGamma - realK_), 0};
            LocalTensor<float> gamaKLocal = gamaKInQueue_.AllocTensor<float>();
            Duplicate<float>(gamaKLocal, 0, alignK_ * seqLen);
            TEventID evevtIdVtoMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(evevtIdVtoMte2);
            WaitFlag<HardEvent::V_MTE2>(evevtIdVtoMte2);
            DataCopyPad(gamaKLocal, gamaKGm_[vOffset / realV_ * realK_], gkInParams, gkPadParams);
            gamaKInQueue_.EnQue<float>(gamaKLocal);
            gamaKInUb = gamaKInQueue_.DeQue<float>();
            ExpMasked(gamaKInUb, gamaKInUb, alignK_ * seqLen);
            AscendC::PipeBarrier<PIPE_V>();
        }
        DataCopyPad(qLocal, queryGm_[qkOffset], qkInParams, qkPadParams);
        DataCopyPad(kLocal, keyGm_[qkOffset], qkInParams, qkPadParams);
        DataCopyPad(vLocal, valueGm_[vOffset], vInParams, vPadParams);
        qInQueue_.EnQue<inType>(qLocal);
        kInQueue_.EnQue<inType>(kLocal);
        vInQueue_.EnQue<inType>(vLocal);
        qLocal = qInQueue_.DeQue<inType>();
        kLocal = kInQueue_.DeQue<inType>();
        vLocal = vInQueue_.DeQue<inType>();
        Cast(qInUb, qLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(kInUb, kLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(vInUb, vLocal, AscendC::RoundMode::CAST_NONE, alignV_ * seqLen);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(qInUb, qInUb, scale_, seqLen * alignK_);
        qInQueue_.FreeTensor(qLocal);
        kInQueue_.FreeTensor(kLocal);
        vInQueue_.FreeTensor(vLocal);
    }

    __aicore__ inline void PrefetchState(uint64_t stateOffest, uint32_t curSingleV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.AllocTensor<stateType>();
        DataCopyExtParams stateInParams{static_cast<uint16_t>(curSingleV),
                                        static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0, 0};
        DataCopyPadExtParams<stateType> padParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPad(stateLocal, initStateGm_[stateOffest], stateInParams, padParams);
        stateInQueue_.EnQue<stateType>(stateLocal);
    }

    __aicore__ inline void LoadPrefetchedState(uint32_t curSingleV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.DeQue<stateType>();
        if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(stateInUb, stateLocal, alignK_ * curSingleV);
        } else {
            Cast(stateInUb, stateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * curSingleV);
        }
        stateInQueue_.FreeTensor(stateLocal);
    }

    __aicore__ inline void ExpMasked(LocalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor,
                                    uint32_t count)
    {
        UnaryRepeatParams repeatParams{1, 1, FP32_NUM_PER_BLOCK, FP32_NUM_PER_BLOCK};
        uint8_t repeatTime = static_cast<uint8_t>(count / V_LENGTH);
        uint32_t tailCount = count % V_LENGTH;
        if (repeatTime > 0) {
            Exp(dstTensor, srcTensor, static_cast<uint64_t>(V_LENGTH), repeatTime, repeatParams);
        }
        if (tailCount > 0) {
            uint32_t tailOffset = count - tailCount;
            Exp(dstTensor[tailOffset], srcTensor[tailOffset], static_cast<uint64_t>(tailCount), 1, repeatParams);
        }
    }

    __aicore__ inline void Compute(uint32_t curSingleV, uint64_t curQKOffset, uint64_t curVOffset)
    {
        LocalTensor<stateType> stateOutLocal = stateOutQueue_.AllocTensor<stateType>();
        LocalTensor<outType> attnOutLocal = attnOutQueue_.AllocTensor<outType>();
        __ubuf__ float *stateAddr = reinterpret_cast<__ubuf__ float *>(stateInUb.GetPhyAddr());
        __ubuf__ float *keyAddr = reinterpret_cast<__ubuf__ float *>(kInUb[curQKOffset].GetPhyAddr());
        __ubuf__ float *queryAddr = reinterpret_cast<__ubuf__ float *>(qInUb[curQKOffset].GetPhyAddr());
        __ubuf__ float *valueAddr = reinterpret_cast<__ubuf__ float *>(vInUb[curVOffset].GetPhyAddr());
        __ubuf__ float *gamaKAddr = hasGamaK_
            ? reinterpret_cast<__ubuf__ float *>(gamaKInUb[curQKOffset].GetPhyAddr())
            : stateAddr;
        __ubuf__ stateType *stateOutAddr =
            reinterpret_cast<__ubuf__ stateType *>(stateOutLocal.GetPhyAddr());
        __ubuf__ outType *attnOutAddr = reinterpret_cast<__ubuf__ outType *>(attnOutLocal.GetPhyAddr());
        __ubuf__ float *attnTmpAddr = reinterpret_cast<__ubuf__ float *>(attnInUb.GetPhyAddr());
        if (hasGama_) {
            if (hasGamaK_) {
                ComputeRecurrentGatedDeltaRuleVF<stateType, outType, true, true>(
                    stateAddr, keyAddr, queryAddr, valueAddr, gamaKAddr, stateOutAddr, attnOutAddr,
                    attnTmpAddr, static_cast<uint16_t>(curSingleV), static_cast<uint16_t>(alignK_),
                    gama_, beta_);
            } else {
                ComputeRecurrentGatedDeltaRuleVF<stateType, outType, true, false>(
                    stateAddr, keyAddr, queryAddr, valueAddr, gamaKAddr, stateOutAddr, attnOutAddr,
                    attnTmpAddr, static_cast<uint16_t>(curSingleV), static_cast<uint16_t>(alignK_),
                    gama_, beta_);
            }
        } else if (hasGamaK_) {
            ComputeRecurrentGatedDeltaRuleVF<stateType, outType, false, true>(
                stateAddr, keyAddr, queryAddr, valueAddr, gamaKAddr, stateOutAddr, attnOutAddr,
                attnTmpAddr, static_cast<uint16_t>(curSingleV), static_cast<uint16_t>(alignK_),
                gama_, beta_);
        } else {
            ComputeRecurrentGatedDeltaRuleVF<stateType, outType, false, false>(
                stateAddr, keyAddr, queryAddr, valueAddr, gamaKAddr, stateOutAddr, attnOutAddr,
                attnTmpAddr, static_cast<uint16_t>(curSingleV), static_cast<uint16_t>(alignK_),
                gama_, beta_);
        }
        stateOutQueue_.EnQue<stateType>(stateOutLocal);
        attnOutQueue_.EnQue<outType>(attnOutLocal);
    }

    __aicore__ inline void CopyOutAttn(uint64_t attnOffset, uint32_t curSingleV)
    {
        LocalTensor<outType> attnLocal = attnOutQueue_.DeQue<outType>();
        DataCopyParams attnOutParams{1, static_cast<uint16_t>(curSingleV * sizeof(outType)), 0, 0};
        DataCopyPad(attnOutGm_[attnOffset], attnLocal, attnOutParams);
        attnOutQueue_.FreeTensor(attnLocal);
    }

    __aicore__ inline void CopyOutState(uint64_t stateOffset, uint32_t curSingleV)
    {
        LocalTensor<stateType> stateOutLocal = stateOutQueue_.DeQue<stateType>();
        DataCopyParams stateOutParams{static_cast<uint16_t>(curSingleV),
                                      static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0};
        DataCopyPad(finalStateGm_[stateOffset], stateOutLocal, stateOutParams);
        stateOutQueue_.FreeTensor(stateOutLocal);
    }

    __aicore__ inline void CopyInGamaBeta(int32_t seq0, int32_t seq1)
    {
        int32_t seqLen = seq1 - seq0;
        LocalTensor<inType> betaLocal = betaInQueue_.AllocTensor<inType>();
        DataCopyParams betaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(inType)), 0, 0};
        DataCopyPadParams padParams;
        DataCopyPad(betaLocal, betaGm_[seq0 * NV_], betaInParams, padParams);
        betaInQueue_.EnQue<inType>(betaLocal);
        betaLocal = betaInQueue_.DeQue<inType>();
        Cast(betaInUb, betaLocal, AscendC::RoundMode::CAST_NONE, seqLen * NV_);
        betaInQueue_.FreeTensor(betaLocal);
        if (hasGama_) {
            LocalTensor<float> gamaLocal = gamaInQueue_.AllocTensor<float>();
            DataCopyParams gamaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(float)), 0, 0};
            DataCopyPad(gamaLocal, gamaGm_[seq0 * NV_], gamaInParams, padParams);
            gamaInQueue_.EnQue<float>(gamaLocal);
            gamaLocal = gamaInQueue_.DeQue<float>();
            ExpMasked(gamaInUb, gamaLocal, seqLen * NV_);
            gamaInQueue_.FreeTensor(gamaLocal);
        }
    }

    __aicore__ inline void ProcessHead(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t stateOffset)
    {
        uint64_t vOffset = (seq0 * NV_ + head_i) * realV_;
        uint64_t qkOffset = (seq0 * NK_ + head_i / (NV_ / NK_)) * realK_;
        CopyInQKV(vOffset, qkOffset, seq1 - seq0);
        if (realV_ == 0) {
            if (hasGamaK_) {
                gamaKInQueue_.FreeTensor(gamaKInUb);
            }
            return;
        }
        uint64_t nextVOffset = 0;
        uint32_t nextSingleV = realV_ > vStep_ ? vStep_ : realV_;
        uint64_t nextStateOffset = stateStride0_ * stateOffset + stateStride1_ * head_i;
        PrefetchState(nextStateOffset, nextSingleV);
        for (uint64_t v_i = 0; v_i < realV_; v_i += vStep_) {
            uint32_t curSingleV = v_i + vStep_ > realV_ ? realV_ - v_i : vStep_;
            LoadPrefetchedState(curSingleV);
            nextVOffset = v_i + vStep_;
            if (nextVOffset < realV_) {
                nextSingleV = nextVOffset + vStep_ > realV_ ? realV_ - nextVOffset : vStep_;
                nextStateOffset = stateStride0_ * stateOffset + stateStride1_ * head_i + stateStride2_ * nextVOffset;
                PrefetchState(nextStateOffset, nextSingleV);
            }
            uint64_t pendingAttnOffset = 0;
            uint64_t pendingStateOffset = 0;
            bool hasPendingAttn = false;
            bool hasPendingState = false;
            for (uint64_t seq_i = seq0; seq_i < seq1; seq_i++) {
                uint64_t gbOffset = head_i + (seq_i - seq0) * NV_;
                uint64_t curQKOffset = (seq_i - seq0) * alignK_;
                uint64_t curVOffset = (seq_i - seq0) * alignV_ + v_i;
                uint64_t attnOffset = (seq_i * NV_ + head_i) * realV_ + v_i;
                uint64_t curStateOutOffset =
                    stateStride0_ * ssmStateIndicesGm_.GetValue(seq_i) +
                    stateStride1_ * head_i + stateStride2_ * v_i;
                gama_ = hasGama_ ? gamaInUb.GetValue(gbOffset) : 1;
                beta_ = betaInUb.GetValue(gbOffset);
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
                if (stateOutBufferNum_ == BUFFER_NUM) {
                    CopyOutState(curStateOutOffset, curSingleV);
                } else {
                    if (hasPendingState) {
                        CopyOutState(pendingStateOffset, curSingleV);
                    }
                    pendingStateOffset = curStateOutOffset;
                    hasPendingState = true;
                }
            }
            if (hasPendingAttn) {
                CopyOutAttn(pendingAttnOffset, curSingleV);
            }
            if (hasPendingState) {
                CopyOutState(pendingStateOffset, curSingleV);
            }
        }
        if (hasGamaK_) {
            gamaKInQueue_.FreeTensor(gamaKInUb);
        }
    }

    __aicore__ inline bool IsCurrentBlock(int32_t seqlen)
    {
        load += seqlen;
        bool ret = (blockIdx == usedblk && seqlen > 0);
        if (load >= avgload) {
            load = 0;
            usedblk++;
        }
        return ret;
    }

private:
    GlobalTensor<inType> queryGm_;
    GlobalTensor<inType> keyGm_;
    GlobalTensor<inType> valueGm_;
    GlobalTensor<inType> betaGm_;
    GlobalTensor<float> gamaGm_;
    GlobalTensor<float> gamaKGm_;
    GlobalTensor<stateType> initStateGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> ssmStateIndicesGm_;
    GlobalTensor<int32_t> numAcceptedTokensGm_;
    GlobalTensor<stateType> finalStateGm_;
    GlobalTensor<outType> attnOutGm_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> qInQueue_;
    TQue<QuePosition::VECIN, 1> kInQueue_;
    TQue<QuePosition::VECIN, 1> vInQueue_;
    TQue<QuePosition::VECIN, 1> gamaInQueue_;
    TQue<QuePosition::VECIN, 1> gamaKInQueue_;
    TQue<QuePosition::VECIN, 1> betaInQueue_;
    TQue<QuePosition::VECIN, 1> stateInQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> attnOutQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> stateOutQueue_;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<float> qInUb;
    LocalTensor<float> kInUb;
    LocalTensor<float> vInUb;
    LocalTensor<float> gamaInUb;
    LocalTensor<float> gamaKInUb;
    LocalTensor<float> betaInUb;
    LocalTensor<float> attnInUb;
    LocalTensor<float> stateInUb;
    uint32_t B_;
    uint32_t T_;
    uint32_t NK_;
    uint32_t alignK_;
    uint32_t realK_;
    uint32_t NV_;
    uint32_t alignV_;
    uint32_t realV_;
    uint32_t vStep_;
    uint32_t stateOutBufferNum_;
    uint32_t attnOutBufferNum_;
    uint32_t restUbSize_;
    uint32_t load;
    uint32_t usedblk;
    uint32_t avgload;
    bool hasAcceptedTokens_;
    bool hasGama_;
    bool hasGamaK_;
    float gama_;
    float beta_;
    float scale_;
    uint64_t blockIdx;
    uint32_t stateStride0_;
    uint32_t stateStride1_;
    uint32_t stateStride2_;
};
} // namespace RecurrentGatedDeltaRule
#endif
