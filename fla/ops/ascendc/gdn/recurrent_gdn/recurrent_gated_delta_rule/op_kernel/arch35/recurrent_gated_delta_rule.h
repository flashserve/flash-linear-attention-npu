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
 * \brief Ascend 950 RegBase implementation.
 */

#ifndef __RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define __RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../recurrent_gated_delta_rule_tiling_data.h"
#include "recurrent_gated_delta_rule_regbase.h"

namespace RecurrentGatedDeltaRule {

using namespace matmul;
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint32_t MAX_MTP = 8;
constexpr uint32_t BF16_NUM_PER_BLOCK = 16;

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
        stateOutBufferNum_ =
            (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        attnOutBufferNum_ =
            (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        restUbSize_ = tilingData->ubRestBytes;
        alignK_ = Ceil(tilingData->dk, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(tilingData->dv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        load_ = 0;
        usedBlock_ = 0;
    }

    __aicore__ inline void Init(const RGDRInitParams &initParams, TPipe *pipe)
    {
        const uint64_t blockDim = GetBlockNum();
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= blockDim) {
            return;
        }
        pipe_ = pipe;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void Process()
    {
        ComputeAvgLoad();
        int32_t seq1 = cuSeqlensGm_.GetValue(0);
        for (uint64_t batch = 0; batch < B_; ++batch) {
            const int32_t seqLen = cuSeqlensGm_.GetValue(batch + 1);
            if (seqLen <= 0) {
                continue;
            }
            if (seqLen > static_cast<int32_t>(MAX_MTP)) {
                return;
            }
            if (seq1 < 0 || seq1 > static_cast<int32_t>(T_) || (seq1 + seqLen) > static_cast<int32_t>(T_)) {
                return;
            }
            const int32_t seq0 = seq1;
            seq1 += seqLen;
            uint32_t localHeadCount = 0;
            uint64_t stateOffset = 0;
            for (uint64_t head = 0; head < NV_; ++head) {
                if (!IsCurrentBlock(seqLen)) {
                    continue;
                }
                ++localHeadCount;
                if (localHeadCount == 1) {
                    int32_t stateTokenIdx = seq0;
                    if (hasAcceptedTokens_) {
                        const int32_t acceptedTokenNum = numAcceptedTokensGm_.GetValue(batch);
                        if (acceptedTokenNum <= 0 || acceptedTokenNum > seqLen) {
                            return;
                        }
                        stateTokenIdx = seq0 + acceptedTokenNum - 1;
                    }
                    stateOffset = ssmStateIndicesGm_.GetValue(stateTokenIdx);
                    CopyInGamaBeta(seq0, seq1);
                }
                ProcessHead(seq0, seq1, head, stateOffset);
            }
            if (localHeadCount > 0) {
                ReleaseGamaBeta();
            }
        }
    }

private:
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
        const uint32_t betaElements = Ceil(MAX_MTP * NV_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        pipe_->InitBuffer(betaInQueue_, BUFFER_NUM, betaElements * sizeof(inType));
        pipe_->InitBuffer(stateOutQueue_, stateOutBufferNum_, alignK_ * vStep_ * sizeof(stateType));
        pipe_->InitBuffer(attnOutQueue_, attnOutBufferNum_, vStep_ * sizeof(outType));
        pipe_->InitBuffer(tmpBuffer_, restUbSize_);
        recurrentStateUb_ = tmpBuffer_.Get<float>();
    }

    __aicore__ inline void ComputeAvgLoad()
    {
        uint64_t realT = 0;
        for (uint64_t batch = 1; batch < B_ + 1; ++batch) {
            realT += cuSeqlensGm_.GetValue(batch);
        }
        avgLoad_ = Ceil(realT * NV_, GetBlockNum());
    }

    __aicore__ inline bool IsCurrentBlock(int32_t seqLen)
    {
        load_ += seqLen;
        const bool current = (blockIdx_ == usedBlock_ && seqLen > 0);
        if (load_ >= avgLoad_) {
            load_ = 0;
            ++usedBlock_;
        }
        return current;
    }

    __aicore__ inline void CopyInGamaBeta(int32_t seq0, int32_t seq1)
    {
        const int32_t seqLen = seq1 - seq0;
        DataCopyPadParams padParams;
        betaInUb_ = betaInQueue_.AllocTensor<inType>();
        DataCopyParams betaParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(inType)), 0, 0};
        DataCopyPad(betaInUb_, betaGm_[seq0 * NV_], betaParams, padParams);
        betaInQueue_.EnQue<inType>(betaInUb_);
        betaInUb_ = betaInQueue_.DeQue<inType>();

        if (hasGama_) {
            gamaInUb_ = gamaInQueue_.AllocTensor<float>();
            DataCopyParams gamaParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(float)), 0, 0};
            DataCopyPad(gamaInUb_, gamaGm_[seq0 * NV_], gamaParams, padParams);
            gamaInQueue_.EnQue<float>(gamaInUb_);
            gamaInUb_ = gamaInQueue_.DeQue<float>();
        }
    }

    __aicore__ inline void ReleaseGamaBeta()
    {
        betaInQueue_.FreeTensor(betaInUb_);
        if (hasGama_) {
            gamaInQueue_.FreeTensor(gamaInUb_);
        }
    }

    __aicore__ inline void CopyInQKV(uint64_t vOffset, uint64_t qkOffset, int32_t seqLen)
    {
        qInUb_ = qInQueue_.AllocTensor<inType>();
        kInUb_ = kInQueue_.AllocTensor<inType>();
        vInUb_ = vInQueue_.AllocTensor<inType>();
        DataCopyExtParams qkParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                   static_cast<uint32_t>((NK_ - 1) * realK_ * sizeof(inType)), 0, 0};
        DataCopyExtParams vParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
                                  static_cast<uint32_t>((NV_ - 1) * realV_ * sizeof(inType)), 0, 0};
        DataCopyPadExtParams<inType> qkPad{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPadExtParams<inType> vPad{true, 0, static_cast<uint8_t>(alignV_ - realV_), 0};
        DataCopyPad(qInUb_, queryGm_[qkOffset], qkParams, qkPad);
        DataCopyPad(kInUb_, keyGm_[qkOffset], qkParams, qkPad);
        DataCopyPad(vInUb_, valueGm_[vOffset], vParams, vPad);
        qInQueue_.EnQue<inType>(qInUb_);
        kInQueue_.EnQue<inType>(kInUb_);
        vInQueue_.EnQue<inType>(vInUb_);
        qInUb_ = qInQueue_.DeQue<inType>();
        kInUb_ = kInQueue_.DeQue<inType>();
        vInUb_ = vInQueue_.DeQue<inType>();

        if (hasGamaK_) {
            gamaKInUb_ = gamaKInQueue_.AllocTensor<float>();
            DataCopyExtParams gkParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(float)),
                                       static_cast<uint32_t>((NV_ - 1) * realK_ * sizeof(float)), 0, 0};
            DataCopyPadExtParams<float> gkPad{true, 0, 0, 0};
            DataCopyPad(gamaKInUb_, gamaKGm_[vOffset / realV_ * realK_], gkParams, gkPad);
            gamaKInQueue_.EnQue<float>(gamaKInUb_);
            gamaKInUb_ = gamaKInQueue_.DeQue<float>();
        }
    }

    __aicore__ inline void ReleaseQKV()
    {
        qInQueue_.FreeTensor(qInUb_);
        kInQueue_.FreeTensor(kInUb_);
        vInQueue_.FreeTensor(vInUb_);
        if (hasGamaK_) {
            gamaKInQueue_.FreeTensor(gamaKInUb_);
        }
    }

    __aicore__ inline void PrefetchState(uint64_t stateOffset, uint32_t rows)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.AllocTensor<stateType>();
        DataCopyExtParams stateParams{static_cast<uint16_t>(rows),
                                      static_cast<uint32_t>(realK_ * sizeof(stateType)), 0, 0, 0};
        DataCopyPadExtParams<stateType> statePad{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPad(stateLocal, initStateGm_[stateOffset], stateParams, statePad);
        stateInQueue_.EnQue<stateType>(stateLocal);
    }

    template <bool HAS_G, bool HAS_GK, bool READ_INITIAL, bool WRITE_RECURRENT>
    __aicore__ inline void CallTokenVf(
        uint32_t rows, uint64_t qkOffset, uint64_t vOffset, uint64_t gbOffset,
        LocalTensor<stateType> &initialState, LocalTensor<stateType> &stateOut, LocalTensor<outType> &attnOut)
    {
        __ubuf__ float *stateAddr = reinterpret_cast<__ubuf__ float *>(recurrentStateUb_.GetPhyAddr());
        __ubuf__ stateType *initialAddr;
        if constexpr (READ_INITIAL) {
            initialAddr = reinterpret_cast<__ubuf__ stateType *>(initialState.GetPhyAddr());
        } else {
            initialAddr = reinterpret_cast<__ubuf__ stateType *>(stateAddr);
        }
        __ubuf__ float *gamaAddr = stateAddr;
        __ubuf__ float *gamaKAddr = stateAddr;
        if constexpr (HAS_G) {
            gamaAddr = reinterpret_cast<__ubuf__ float *>(gamaInUb_.GetPhyAddr()) + gbOffset;
        }
        if constexpr (HAS_GK) {
            gamaKAddr = reinterpret_cast<__ubuf__ float *>(gamaKInUb_.GetPhyAddr()) + qkOffset;
        }
        AscendC::VF_CALL<RgdrRecurrentToken128Vf<stateType, outType, HAS_G, HAS_GK,
                                                  READ_INITIAL, WRITE_RECURRENT>>(
            stateAddr, initialAddr, reinterpret_cast<__ubuf__ stateType *>(stateOut.GetPhyAddr()),
            reinterpret_cast<__ubuf__ outType *>(attnOut.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(qInUb_.GetPhyAddr()) + qkOffset,
            reinterpret_cast<__ubuf__ bfloat16_t *>(kInUb_.GetPhyAddr()) + qkOffset,
            reinterpret_cast<__ubuf__ bfloat16_t *>(vInUb_.GetPhyAddr()) + vOffset,
            gamaAddr, gamaKAddr,
            reinterpret_cast<__ubuf__ bfloat16_t *>(betaInUb_.GetPhyAddr()) + gbOffset,
            scale_, static_cast<uint16_t>(rows));
    }

    template <bool READ_INITIAL, bool WRITE_RECURRENT>
    __aicore__ inline void Compute(
        uint32_t rows, uint64_t qkOffset, uint64_t vOffset, uint64_t gbOffset,
        LocalTensor<stateType> &initialState)
    {
        LocalTensor<stateType> stateOut = stateOutQueue_.AllocTensor<stateType>();
        LocalTensor<outType> attnOut = attnOutQueue_.AllocTensor<outType>();
        if (hasGama_) {
            if (hasGamaK_) {
                CallTokenVf<true, true, READ_INITIAL, WRITE_RECURRENT>(
                    rows, qkOffset, vOffset, gbOffset, initialState, stateOut, attnOut);
            } else {
                CallTokenVf<true, false, READ_INITIAL, WRITE_RECURRENT>(
                    rows, qkOffset, vOffset, gbOffset, initialState, stateOut, attnOut);
            }
        } else if (hasGamaK_) {
            CallTokenVf<false, true, READ_INITIAL, WRITE_RECURRENT>(
                rows, qkOffset, vOffset, gbOffset, initialState, stateOut, attnOut);
        } else {
            CallTokenVf<false, false, READ_INITIAL, WRITE_RECURRENT>(
                rows, qkOffset, vOffset, gbOffset, initialState, stateOut, attnOut);
        }
        if constexpr (WRITE_RECURRENT) {
            // The next token reloads this FP32 bank in a new VF call. Keep the sole recurrent store->load
            // dependency explicit; all intra-token intermediates stay in registers and need no PIPE_V barrier.
            AscendC::PipeBarrier<PIPE_V>();
        }
        stateOutQueue_.EnQue<stateType>(stateOut);
        attnOutQueue_.EnQue<outType>(attnOut);
    }

    __aicore__ inline void CopyOutAttn(uint64_t attnOffset, uint32_t rows)
    {
        LocalTensor<outType> attn = attnOutQueue_.DeQue<outType>();
        DataCopyParams params{1, static_cast<uint16_t>(rows * sizeof(outType)), 0, 0};
        DataCopyPad(attnOutGm_[attnOffset], attn, params);
        attnOutQueue_.FreeTensor(attn);
    }

    __aicore__ inline void CopyOutState(uint64_t stateOffset, uint32_t rows)
    {
        LocalTensor<stateType> state = stateOutQueue_.DeQue<stateType>();
        DataCopyParams params{static_cast<uint16_t>(rows),
                              static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0};
        DataCopyPad(finalStateGm_[stateOffset], state, params);
        stateOutQueue_.FreeTensor(state);
    }

    __aicore__ inline void ProcessHead(int32_t seq0, int32_t seq1, uint64_t head, uint64_t stateOffset)
    {
        const uint64_t vBase = (seq0 * NV_ + head) * realV_;
        const uint64_t qkBase = (seq0 * NK_ + head / (NV_ / NK_)) * realK_;
        CopyInQKV(vBase, qkBase, seq1 - seq0);
        if (realV_ == 0) {
            ReleaseQKV();
            return;
        }

        const uint32_t firstRows = realV_ > vStep_ ? vStep_ : realV_;
        const uint64_t firstStateOffset = stateStride0_ * stateOffset + stateStride1_ * head;
        PrefetchState(firstStateOffset, firstRows);
        for (uint64_t v = 0; v < realV_; v += vStep_) {
            const uint32_t rows = v + vStep_ > realV_ ? realV_ - v : vStep_;
            LocalTensor<stateType> initialState = stateInQueue_.DeQue<stateType>();
            const uint64_t nextV = v + vStep_;
            bool initialStateReleased = false;

            uint64_t pendingAttnOffset = 0;
            uint64_t pendingStateOffset = 0;
            bool hasPendingAttn = false;
            bool hasPendingState = false;
            for (int32_t seq = seq0; seq < seq1; ++seq) {
                const uint64_t localSeq = static_cast<uint64_t>(seq - seq0);
                const uint64_t gbOffset = head + localSeq * NV_;
                const uint64_t qkOffset = localSeq * alignK_;
                const uint64_t valueOffset = localSeq * alignV_ + v;
                const uint64_t attnOffset = (static_cast<uint64_t>(seq) * NV_ + head) * realV_ + v;
                const uint64_t stateOutOffset =
                    stateStride0_ * ssmStateIndicesGm_.GetValue(seq) +
                    stateStride1_ * head + stateStride2_ * v;
                const bool firstToken = (seq == seq0);
                const bool lastToken = (seq + 1 == seq1);

                if (firstToken) {
                    if (lastToken) {
                        Compute<true, false>(rows, qkOffset, valueOffset, gbOffset, initialState);
                    } else {
                        Compute<true, true>(rows, qkOffset, valueOffset, gbOffset, initialState);
                    }
                    stateInQueue_.FreeTensor(initialState);
                    initialStateReleased = true;
                    // Token 0 releases the MTE2 slot. The next V tile can overlap later recurrent tokens.
                    if (nextV < realV_) {
                        const uint32_t nextRows = nextV + vStep_ > realV_ ? realV_ - nextV : vStep_;
                        const uint64_t nextStateOffset =
                            stateStride0_ * stateOffset + stateStride1_ * head + stateStride2_ * nextV;
                        PrefetchState(nextStateOffset, nextRows);
                    }
                } else if (lastToken) {
                    Compute<false, false>(rows, qkOffset, valueOffset, gbOffset, initialState);
                } else {
                    Compute<false, true>(rows, qkOffset, valueOffset, gbOffset, initialState);
                }

                if (attnOutBufferNum_ == BUFFER_NUM) {
                    CopyOutAttn(attnOffset, rows);
                } else {
                    if (hasPendingAttn) {
                        CopyOutAttn(pendingAttnOffset, rows);
                    }
                    pendingAttnOffset = attnOffset;
                    hasPendingAttn = true;
                }
                if (stateOutBufferNum_ == BUFFER_NUM) {
                    CopyOutState(stateOutOffset, rows);
                } else {
                    if (hasPendingState) {
                        CopyOutState(pendingStateOffset, rows);
                    }
                    pendingStateOffset = stateOutOffset;
                    hasPendingState = true;
                }
            }
            if (!initialStateReleased) {
                stateInQueue_.FreeTensor(initialState);
            }
            if (hasPendingAttn) {
                CopyOutAttn(pendingAttnOffset, rows);
            }
            if (hasPendingState) {
                CopyOutState(pendingStateOffset, rows);
            }
        }
        ReleaseQKV();
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
    TBuf<TPosition::VECCALC> tmpBuffer_;
    LocalTensor<inType> qInUb_;
    LocalTensor<inType> kInUb_;
    LocalTensor<inType> vInUb_;
    LocalTensor<float> gamaInUb_;
    LocalTensor<float> gamaKInUb_;
    LocalTensor<inType> betaInUb_;
    LocalTensor<float> recurrentStateUb_;
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
    uint32_t load_;
    uint32_t usedBlock_;
    uint32_t avgLoad_;
    bool hasAcceptedTokens_;
    bool hasGama_;
    bool hasGamaK_;
    float scale_;
    uint64_t blockIdx_;
    uint32_t stateStride0_;
    uint32_t stateStride1_;
    uint32_t stateStride2_;
};

} // namespace RecurrentGatedDeltaRule

#endif // __RECURRENT_GATED_DELTA_RULE_KERNEL_H_
