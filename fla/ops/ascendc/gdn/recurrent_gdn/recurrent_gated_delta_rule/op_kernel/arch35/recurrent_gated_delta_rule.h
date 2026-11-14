/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef RECURRENT_GATED_DELTA_RULE_ARCH35_H
#define RECURRENT_GATED_DELTA_RULE_ARCH35_H

#include <type_traits>

#include "kernel_operator.h"
#include "../recurrent_gated_delta_rule_struct.h"

namespace RecurrentGatedDeltaRule {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t TOKEN_TILE_4 = 4;
constexpr uint32_t INPUT_BUFFERS_2 = 2;
constexpr uint32_t STATE_BANKS_3 = 3;
constexpr uint32_t DATA_BLOCK_BYTES_32 = 32;
constexpr uint32_t BF16_PER_BLOCK_16 = DATA_BLOCK_BYTES_32 / sizeof(bfloat16_t);
constexpr uint16_t FP32_PER_REG_64 = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint16_t REQUIRED_DIM_128 = 2 * FP32_PER_REG_64;

constexpr CastTrait FP32_TO_B16_ZERO = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, RoundMode::CAST_RINT};
constexpr CastTrait FP32_TO_B16_ONE = {RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait B16_TO_FP32_ZERO = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr CastTrait B16_TO_FP32_ONE = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

template <typename inType, typename outType, typename stateType, typename gType, typename betaType, bool NORMALIZE_QK,
          bool RAW_GATE, bool BETA_SIGMOID, bool ALLOW_NEGATIVE>
__simd_vf__ inline void
RecurrentGatedDeltaRuleVf(__ubuf__ stateType *stateOut, __ubuf__ stateType *stateIn, __ubuf__ inType *keyRaw,
                          __ubuf__ inType *queryRaw, __ubuf__ inType *valueRaw, __ubuf__ gType *gRaw,
                          __ubuf__ betaType *betaRaw, __ubuf__ outType *output, float scale, float aLog, float dtBias)
{
    MaskReg maskFp32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskB16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskFirst = CreateMask<float, MaskPattern::VL1>();

    RegTensor<inType> keyPacked;
    RegTensor<inType> queryPacked;
    RegTensor<inType> valuePacked;
    RegTensor<inType> valuePackedNext;
    RegTensor<gType> gPacked;
    RegTensor<betaType> betaPacked;
    RegTensor<stateType> statePacked;
    RegTensor<stateType> statePackedNext;
    RegTensor<outType> outputPacked;
    RegTensor<float> key0;
    RegTensor<float> key1;
    RegTensor<float> updateKey0;
    RegTensor<float> updateKey1;
    RegTensor<float> query0;
    RegTensor<float> query1;
    RegTensor<float> state0;
    RegTensor<float> state1;
    RegTensor<float> state2;
    RegTensor<float> state3;
    RegTensor<float> value;
    RegTensor<float> valueNext;
    RegTensor<float> scalarGate;
    RegTensor<float> beta;
    RegTensor<float> one;
    RegTensor<float> queryWork;
    RegTensor<float> keyWork;
    RegTensor<float> queryNorm;
    RegTensor<float> keyNorm;
    RegTensor<float> queryInvNorm;
    RegTensor<float> keyInvNorm;
    RegTensor<float> expA;
    RegTensor<float> dot0;
    RegTensor<float> dot1;
    RegTensor<float> dot;
    RegTensor<float> dotNext;
    RegTensor<float> delta;
    RegTensor<float> deltaNext;
    RegTensor<float> out0;
    RegTensor<float> out1;
    RegTensor<float> out2;
    RegTensor<float> outSum;
    RegTensor<float> outSumNext;
    UnalignRegForStore outputStore;
    __ubuf__ outType *outputAddr = output;
    __ubuf__ float *stateInFp32 = (__ubuf__ float *)stateIn;
    __ubuf__ float *stateOutFp32 = (__ubuf__ float *)stateOut;
    uint32_t rowOffset = 0;
    uint16_t row = 0;

    LoadAlign<inType, LoadDist::DIST_NORM>(keyPacked, keyRaw);
    LoadAlign<inType, LoadDist::DIST_NORM>(queryPacked, queryRaw);
    Cast<float, inType, B16_TO_FP32_ZERO>(key0, keyPacked, maskB16);
    Cast<float, inType, B16_TO_FP32_ZERO>(query0, queryPacked, maskB16);
    Cast<float, inType, B16_TO_FP32_ONE>(key1, keyPacked, maskB16);
    Cast<float, inType, B16_TO_FP32_ONE>(query1, queryPacked, maskB16);

    Duplicate(one, 1.0f, maskFp32);
    if constexpr (NORMALIZE_QK) {
        Mul(dot0, query0, query0, maskFp32);
        Mul(out0, key0, key0, maskFp32);
        Mul(dot1, query1, query1, maskFp32);
        Mul(out1, key1, key1, maskFp32);
        Add(queryWork, dot0, dot1, maskFp32);
        Add(keyWork, out0, out1, maskFp32);
        ReduceSum(queryNorm, queryWork, maskFp32);
        ReduceSum(keyNorm, keyWork, maskFp32);
        Adds(queryNorm, queryNorm, 1.0e-6f, maskFirst);
        Adds(keyNorm, keyNorm, 1.0e-6f, maskFirst);
        Sqrt(queryNorm, queryNorm, maskFirst);
        Sqrt(keyNorm, keyNorm, maskFirst);
        Div(queryInvNorm, one, queryNorm, maskFirst);
        Div(keyInvNorm, one, keyNorm, maskFirst);
        Duplicate(queryWork, queryInvNorm, maskFp32);
        Duplicate(keyWork, keyInvNorm, maskFp32);
        Mul(query0, query0, queryWork, maskFp32);
        Mul(key0, key0, keyWork, maskFp32);
        Mul(query1, query1, queryWork, maskFp32);
        Mul(key1, key1, keyWork, maskFp32);
    }
    Muls(query0, query0, scale, maskFp32);
    Muls(query1, query1, scale, maskFp32);

    if constexpr (std::is_same<gType, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scalarGate, gRaw);
    } else {
        LoadAlign<gType, LoadDist::DIST_BRC_B16>(gPacked, gRaw);
        Cast<float, gType, B16_TO_FP32_ZERO>(scalarGate, gPacked, maskB16);
    }

    if constexpr (std::is_same<betaType, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(beta, betaRaw);
    } else {
        LoadAlign<betaType, LoadDist::DIST_BRC_B16>(betaPacked, betaRaw);
        Cast<float, betaType, B16_TO_FP32_ZERO>(beta, betaPacked, maskB16);
    }

    if constexpr (RAW_GATE) {
        Adds(scalarGate, scalarGate, dtBias, maskFp32);
    }
    if constexpr (BETA_SIGMOID) {
        Muls(beta, beta, -1.0f, maskFp32);
    }
    if constexpr (RAW_GATE) {
        Duplicate(expA, aLog, maskFp32);
    }
    if constexpr (!RAW_GATE) {
        Exp(scalarGate, scalarGate, maskFp32);
    }
    if constexpr (RAW_GATE) {
        Exp(expA, expA, maskFp32);
    }
    if constexpr (BETA_SIGMOID) {
        Exp(beta, beta, maskFp32);
    }
    if constexpr (RAW_GATE) {
        Exp(queryWork, scalarGate, maskFp32);
    }
    if constexpr (BETA_SIGMOID) {
        Adds(beta, beta, 1.0f, maskFp32);
    }
    if constexpr (RAW_GATE) {
        Adds(queryWork, queryWork, 1.0f, maskFp32);
    }
    if constexpr (BETA_SIGMOID) {
        Div(beta, one, beta, maskFp32);
    }
    if constexpr (RAW_GATE) {
        Ln(queryWork, queryWork, maskFp32);
    }
    if constexpr (BETA_SIGMOID && ALLOW_NEGATIVE) {
        Muls(beta, beta, 2.0f, maskFp32);
    }
    if constexpr (RAW_GATE) {
        Mul(scalarGate, queryWork, expA, maskFp32);
        Muls(scalarGate, scalarGate, -1.0f, maskFp32);
        Exp(scalarGate, scalarGate, maskFp32);
    }
    Mul(updateKey0, key0, beta, maskFp32);
    Mul(updateKey1, key1, beta, maskFp32);

    for (row = 0; row < REQUIRED_DIM_128; row += 2) {
        rowOffset = static_cast<uint32_t>(row) * REQUIRED_DIM_128;
        if constexpr (!std::is_same<stateType, float>::value) {
            LoadAlign<stateType, LoadDist::DIST_NORM>(statePacked, stateIn + rowOffset);
            LoadAlign<stateType, LoadDist::DIST_NORM>(statePackedNext, stateIn + rowOffset + REQUIRED_DIM_128);
            Cast<float, stateType, B16_TO_FP32_ZERO>(state0, statePacked, maskB16);
            Cast<float, stateType, B16_TO_FP32_ONE>(state1, statePacked, maskB16);
            Cast<float, stateType, B16_TO_FP32_ZERO>(state2, statePackedNext, maskB16);
            Cast<float, stateType, B16_TO_FP32_ONE>(state3, statePackedNext, maskB16);
        } else {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(state0, state1, stateInFp32 + rowOffset);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(state2, state3, stateInFp32 + rowOffset + REQUIRED_DIM_128);
        }
        LoadAlign<inType, LoadDist::DIST_BRC_B16>(valuePacked, valueRaw + row);
        LoadAlign<inType, LoadDist::DIST_BRC_B16>(valuePackedNext, valueRaw + row + 1);

        Mul(state0, state0, scalarGate, maskFp32);
        Mul(state2, state2, scalarGate, maskFp32);
        Mul(state1, state1, scalarGate, maskFp32);
        Mul(state3, state3, scalarGate, maskFp32);

        Cast<float, inType, B16_TO_FP32_ZERO>(value, valuePacked, maskB16);
        Cast<float, inType, B16_TO_FP32_ZERO>(valueNext, valuePackedNext, maskB16);
        Mul(dot0, state0, key0, maskFp32);
        Mul(out0, state2, key0, maskFp32);
        MulAddDst(dot0, state1, key1, maskFp32);
        MulAddDst(out0, state3, key1, maskFp32);
        ReduceSum(dot, dot0, maskFp32);
        ReduceSum(dotNext, out0, maskFp32);
        Duplicate(dot, dot, maskFp32);
        Duplicate(dotNext, dotNext, maskFp32);

        Sub(delta, value, dot, maskFp32);
        Sub(deltaNext, valueNext, dotNext, maskFp32);
        MulAddDst(state0, delta, updateKey0, maskFp32);
        MulAddDst(state2, deltaNext, updateKey0, maskFp32);
        MulAddDst(state1, delta, updateKey1, maskFp32);
        MulAddDst(state3, deltaNext, updateKey1, maskFp32);

        Mul(out0, state0, query0, maskFp32);
        Mul(out2, state2, query0, maskFp32);
        MulAddDst(out0, state1, query1, maskFp32);
        MulAddDst(out2, state3, query1, maskFp32);
        ReduceSum(outSum, out0, maskFp32);
        ReduceSum(outSumNext, out2, maskFp32);

        if constexpr (std::is_same<stateType, float>::value) {
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(stateOutFp32 + rowOffset, state0, state1, maskFp32);
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(stateOutFp32 + rowOffset + REQUIRED_DIM_128, state2, state3,
                                                         maskFp32);
        } else {
            Cast<stateType, float, FP32_TO_B16_ONE>(statePacked, state1, maskFp32);
            Cast<stateType, float, FP32_TO_B16_ZERO>(statePacked, state0, maskFp32);
            StoreAlign(stateOut + rowOffset, statePacked, maskB16);
            Cast<stateType, float, FP32_TO_B16_ONE>(statePackedNext, state3, maskFp32);
            Cast<stateType, float, FP32_TO_B16_ZERO>(statePackedNext, state2, maskFp32);
            StoreAlign(stateOut + rowOffset + REQUIRED_DIM_128, statePackedNext, maskB16);
        }
        if constexpr (std::is_same<outType, float>::value) {
            StoreUnAlign(outputAddr, outSum, outputStore, 1);
            StoreUnAlign(outputAddr, outSumNext, outputStore, 1);
        } else {
            Cast<outType, float, FP32_TO_B16_ZERO>(outputPacked, outSum, maskFirst);
            StoreUnAlign(outputAddr, outputPacked, outputStore, 1);
            Cast<outType, float, FP32_TO_B16_ZERO>(outputPacked, outSumNext, maskFirst);
            StoreUnAlign(outputAddr, outputPacked, outputStore, 1);
        }
    }
    StoreUnAlignPost(outputAddr, outputStore, 0);
}

constexpr uint8_t INPUT_MUTEX[INPUT_BUFFERS_2] = {3, 4};
constexpr uint8_t OUTPUT_MUTEX[INPUT_BUFFERS_2] = {5, 6};
constexpr uint8_t STATE_MUTEX[STATE_BANKS_3] = {7, 8, 9};

template <typename inType, typename outType, typename stateType, typename gType, typename betaType, bool USE_QK_L2NORM,
          bool USE_GATE_IN_KERNEL, bool USE_BETA_SIGMOID, bool ALLOW_NEG_EIGVAL>
class RGDR {
public:
    __aicore__ inline explicit RGDR(const RecurrentGatedDeltaRuleTilingData *tilingData)
    {
        B_ = tilingData->b;
        T_ = tilingData->t;
        NK_ = tilingData->nk;
        NV_ = tilingData->nv;
        ssmStateStride_ = tilingData->stateIndicesWidth;
        queryTokenStride_ = tilingData->queryStrides[0];
        queryHeadStride_ = tilingData->queryStrides[1];
        keyTokenStride_ = tilingData->keyStrides[0];
        keyHeadStride_ = tilingData->keyStrides[1];
        valueTokenStride_ = tilingData->valueStrides[0];
        valueHeadStride_ = tilingData->valueStrides[1];
        stateInStride0_ = tilingData->stateStrides[0];
        stateInStride1_ = tilingData->stateStrides[1];
        stateInStride2_ = tilingData->stateStrides[2];
        stateOutStride0_ = tilingData->stateStrides[0];
        stateOutStride1_ = tilingData->stateStrides[1];
        stateOutStride2_ = tilingData->stateStrides[2];
        scale_ = tilingData->scale;
        hasAcceptedTokens_ = tilingData->hasAcceptedTokens != 0;
        hasDtBias_ = tilingData->hasDtBias != 0;
        hasGama_ = tilingData->hasGama != 0;
    }

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR g, GM_ADDR beta, GM_ADDR initState,
                                GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR aLog, GM_ADDR dtBias,
                                GM_ADDR numAcceptedTokens, GM_ADDR attnOut, GM_ADDR finalState)
    {
        blockIdx_ = GetBlockIdx();
        queryGm_.SetGlobalBuffer((__gm__ inType *)query);
        keyGm_.SetGlobalBuffer((__gm__ inType *)key);
        valueGm_.SetGlobalBuffer((__gm__ inType *)value);
        gGm_.SetGlobalBuffer((__gm__ gType *)g);
        betaGm_.SetGlobalBuffer((__gm__ betaType *)beta);
        initStateGm_.SetGlobalBuffer((__gm__ stateType *)initState);
        cuSeqlensInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
        stateIndicesInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)ssmStateIndices);
        aLogGm_.SetGlobalBuffer((__gm__ float *)aLog);
        dtBiasGm_.SetGlobalBuffer((__gm__ float *)dtBias);
        acceptedInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)numAcceptedTokens);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)attnOut);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType *)finalState);
        InitBuffers();
        if (!hasGama_) {
            for (uint32_t buffer = 0; buffer < INPUT_BUFFERS_2; ++buffer) {
                Duplicate(gTile_[buffer], static_cast<gType>(0), TOKEN_TILE_4 * DATA_BLOCK_BYTES_32 / sizeof(gType));
            }
            PipeBarrier<PIPE_ALL>();
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void Process()
    {
        const uint64_t taskNum = static_cast<uint64_t>(B_) * NV_;
        const uint64_t coreNum = GetBlockNum();
        uint32_t currentBank = 0;
        seqValid_ = false;
        uint32_t tileIndex = 0;
        for (uint64_t taskIdx = blockIdx_; taskIdx < taskNum; taskIdx += coreNum) {
            const uint64_t batch = taskIdx / NV_;
            const uint64_t head = taskIdx % NV_;
            UpdateSeq(batch);
            if (seqStart_ == seqEnd_) {
                continue;
            }
            const uint64_t stateSlot = ResolveInitialStateSlot(batch, seqStart_);
            const uint64_t qkHead = head / (NV_ / NK_);
            const uint64_t stateOffset = stateInStride0_ * stateSlot + stateInStride1_ * head;
            constexpr uint32_t stateElements = REQUIRED_DIM_128 * REQUIRED_DIM_128;
            const int64_t taskTokens = seqEnd_ - seqStart_;
            const bool hasNextTask = taskIdx + coreNum < taskNum;
            currentBank = (currentBank + 1U) % STATE_BANKS_3;
            LocalTensor<stateType> stateInput = stateBanks_[currentBank].template ReinterpretCast<stateType>();
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            Mutex::Lock<PIPE_MTE2>(STATE_MUTEX[currentBank]);
            const DataCopyExtParams stateInParams{
                REQUIRED_DIM_128, REQUIRED_DIM_128 * sizeof(stateType),
                static_cast<uint32_t>((stateInStride2_ - REQUIRED_DIM_128) * sizeof(stateType)), 0, 0};
            const DataCopyPadExtParams<stateType> stateInPad{false, 0, 0, static_cast<stateType>(0)};
            DataCopyPad(stateInput, initStateGm_[stateOffset], stateInParams, stateInPad);
            Mutex::Unlock<PIPE_MTE2>(STATE_MUTEX[currentBank]);
            if (taskTokens == 1 && hasNextTask) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }

            const float aLog = USE_GATE_IN_KERNEL ? aLogGm_.GetValue(head) : 0.0f;
            const float dtBias = USE_GATE_IN_KERNEL && hasDtBias_ ? dtBiasGm_.GetValue(head) : 0.0f;
            for (int64_t start = seqStart_; start < seqEnd_; start += TOKEN_TILE_4, ++tileIndex) {
                const int64_t end = start + TOKEN_TILE_4 < seqEnd_ ? start + TOKEN_TILE_4 : seqEnd_;
                const uint16_t rows = static_cast<uint16_t>(end - start);
                const uint32_t tileBuffer = tileIndex & 1U;
                CopyInputTile(tileBuffer, start, end, qkHead, head);

                Mutex::Lock<PIPE_V>(INPUT_MUTEX[tileBuffer]);
                Mutex::Lock<PIPE_V>(OUTPUT_MUTEX[tileBuffer]);
                for (uint16_t token = 0; token < rows; ++token) {
                    const int64_t seqIndex = start + token;
                    const uint64_t outputSlot = LoadStateSlot(batch, seqStart_, seqIndex);
                    const uint32_t nextBank = (currentBank + 1U) % STATE_BANKS_3;
                    Mutex::Lock<PIPE_V>(STATE_MUTEX[nextBank]);
                    ComputeOne<USE_GATE_IN_KERNEL>(tileBuffer, token, currentBank, nextBank, aLog, dtBias);
                    Mutex::Unlock<PIPE_V>(STATE_MUTEX[nextBank]);
                    if (taskTokens > 1 && hasNextTask && seqIndex - seqStart_ == taskTokens - 2) {
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    }

                    const uint64_t stateOutOffset = stateOutStride0_ * outputSlot + stateOutStride1_ * head;
                    LocalTensor<stateType> stateOut = stateBanks_[nextBank].template ReinterpretCast<stateType>();
                    Mutex::Lock<PIPE_MTE3>(STATE_MUTEX[nextBank]);
                    const DataCopyExtParams stateOutParams{
                        REQUIRED_DIM_128, REQUIRED_DIM_128 * sizeof(stateType), 0,
                        static_cast<uint32_t>((stateOutStride2_ - REQUIRED_DIM_128) * sizeof(stateType)), 0};
                    DataCopyPad(finalStateGm_[stateOutOffset], stateOut, stateOutParams);
                    Mutex::Unlock<PIPE_MTE3>(STATE_MUTEX[nextBank]);
                    currentBank = nextBank;
                }

                Mutex::Unlock<PIPE_V>(INPUT_MUTEX[tileBuffer]);
                Mutex::Unlock<PIPE_V>(OUTPUT_MUTEX[tileBuffer]);

                const uint64_t attnOffset = (static_cast<uint64_t>(start) * NV_ + head) * REQUIRED_DIM_128;
                DataCopyExtParams attnParams{rows, REQUIRED_DIM_128 * sizeof(outType), 0,
                                             static_cast<uint32_t>((NV_ - 1) * REQUIRED_DIM_128 * sizeof(outType)), 0};
                Mutex::Lock<PIPE_MTE3>(OUTPUT_MUTEX[tileBuffer]);
                DataCopyPad(attnOutGm_[attnOffset], attnTileOut_[tileBuffer], attnParams);
                Mutex::Unlock<PIPE_MTE3>(OUTPUT_MUTEX[tileBuffer]);
            }
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        uint32_t offset = 0;
        constexpr uint32_t stateElements = REQUIRED_DIM_128 * REQUIRED_DIM_128;
        constexpr uint32_t stateBytes = stateElements * sizeof(float);
        constexpr uint32_t qkvElements = TOKEN_TILE_4 * REQUIRED_DIM_128;
        constexpr uint32_t scalarBytes = TOKEN_TILE_4 * DATA_BLOCK_BYTES_32;
        constexpr uint32_t outputElements = TOKEN_TILE_4 * REQUIRED_DIM_128;
        for (uint32_t bank = 0; bank < STATE_BANKS_3; ++bank) {
            stateBanks_[bank] = LocalTensor<float>(TPosition::VECCALC, offset, stateElements);
            offset += stateBytes;
        }
        for (uint32_t buffer = 0; buffer < INPUT_BUFFERS_2; ++buffer) {
            qTile_[buffer] = LocalTensor<inType>(TPosition::VECCALC, offset, qkvElements);
            offset += qkvElements * sizeof(inType);
            kTile_[buffer] = LocalTensor<inType>(TPosition::VECCALC, offset, qkvElements);
            offset += qkvElements * sizeof(inType);
            vTile_[buffer] = LocalTensor<inType>(TPosition::VECCALC, offset, qkvElements);
            offset += qkvElements * sizeof(inType);
            gTile_[buffer] = LocalTensor<gType>(TPosition::VECCALC, offset, scalarBytes / sizeof(gType));
            offset += scalarBytes;
            betaTile_[buffer] = LocalTensor<betaType>(TPosition::VECCALC, offset, scalarBytes / sizeof(betaType));
            offset += scalarBytes;
            attnTileOut_[buffer] = LocalTensor<outType>(TPosition::VECCALC, offset, outputElements);
            offset += outputElements * sizeof(outType);
        }
    }

    __aicore__ inline void UpdateSeq(uint64_t batch)
    {
        if (seqValid_ && batch == seqBatch_) {
            return;
        }
        if (!seqValid_ || batch < seqBatch_) {
            seqBatch_ = 0;
            seqStart_ = static_cast<int64_t>(cuSeqlensInt32Gm_.GetValue(0));
            seqEnd_ = seqStart_ + static_cast<int64_t>(cuSeqlensInt32Gm_.GetValue(1));
            seqValid_ = true;
        }
        while (seqBatch_ < batch) {
            seqStart_ = seqEnd_;
            ++seqBatch_;
            seqEnd_ = seqStart_ + static_cast<int64_t>(cuSeqlensInt32Gm_.GetValue(seqBatch_ + 1));
        }
    }

    __aicore__ inline uint64_t LoadStateSlot(uint64_t batch, int64_t seq0, int64_t token) const
    {
        const uint64_t metadataOffset = ssmStateStride_ == 0 ?
                                            static_cast<uint64_t>(token) :
                                            batch * ssmStateStride_ + static_cast<uint64_t>(token - seq0);
        const int64_t slot = static_cast<int64_t>(stateIndicesInt32Gm_.GetValue(metadataOffset));
        return static_cast<uint64_t>(slot);
    }

    __aicore__ inline uint64_t ResolveInitialStateSlot(uint64_t batch, int64_t seq0) const
    {
        int64_t token = seq0;
        if (hasAcceptedTokens_) {
            const int64_t accepted = static_cast<int64_t>(acceptedInt32Gm_.GetValue(batch));
            token += accepted - 1;
        }
        const uint64_t metadataOffset = ssmStateStride_ == 0 ?
                                            static_cast<uint64_t>(token) :
                                            batch * ssmStateStride_ + static_cast<uint64_t>(token - seq0);
        const int64_t slot = static_cast<int64_t>(stateIndicesInt32Gm_.GetValue(metadataOffset));
        return static_cast<uint64_t>(slot);
    }

    __aicore__ inline void CopyInputTile(uint32_t buffer, int64_t start, int64_t end, uint64_t qkHead,
                                         uint64_t valueHead)
    {
        const uint16_t rows = static_cast<uint16_t>(end - start);
        const uint64_t qOffset = static_cast<uint64_t>(start) * queryTokenStride_ + qkHead * queryHeadStride_;
        const uint64_t kOffset = static_cast<uint64_t>(start) * keyTokenStride_ + qkHead * keyHeadStride_;
        const uint64_t vOffset = static_cast<uint64_t>(start) * valueTokenStride_ + valueHead * valueHeadStride_;
        const uint64_t scalarOffset = static_cast<uint64_t>(start) * NV_ + valueHead;
        constexpr uint32_t scalarRow = DATA_BLOCK_BYTES_32 / sizeof(gType);
        DataCopyExtParams params{rows, static_cast<uint32_t>(sizeof(gType)),
                                 static_cast<uint32_t>((NV_ - 1) * sizeof(gType)), 0, 0};
        DataCopyPadExtParams<gType> pad{false, 0, static_cast<uint8_t>(scalarRow - 1), static_cast<gType>(0)};
        constexpr uint32_t betaRow = DATA_BLOCK_BYTES_32 / sizeof(betaType);
        DataCopyExtParams betaParams{rows, static_cast<uint32_t>(sizeof(betaType)),
                                     static_cast<uint32_t>((NV_ - 1) * sizeof(betaType)), 0, 0};
        DataCopyPadExtParams<betaType> betaPad{false, 0, static_cast<uint8_t>(betaRow - 1), static_cast<betaType>(0)};
        const DataCopyExtParams queryCopyParams{
            rows, REQUIRED_DIM_128 * sizeof(inType),
            static_cast<uint32_t>((queryTokenStride_ - REQUIRED_DIM_128) * sizeof(inType)), 0, 0};
        const DataCopyExtParams keyCopyParams{
            rows, REQUIRED_DIM_128 * sizeof(inType),
            static_cast<uint32_t>((keyTokenStride_ - REQUIRED_DIM_128) * sizeof(inType)), 0, 0};
        const DataCopyExtParams valueCopyParams{
            rows, REQUIRED_DIM_128 * sizeof(inType),
            static_cast<uint32_t>((valueTokenStride_ - REQUIRED_DIM_128) * sizeof(inType)), 0, 0};
        const DataCopyPadExtParams<inType> qkvPad{false, 0, 0, static_cast<inType>(0)};
        Mutex::Lock<PIPE_MTE2>(INPUT_MUTEX[buffer]);
        DataCopyPad(qTile_[buffer], queryGm_[qOffset], queryCopyParams, qkvPad);
        DataCopyPad(kTile_[buffer], keyGm_[kOffset], keyCopyParams, qkvPad);
        DataCopyPad(vTile_[buffer], valueGm_[vOffset], valueCopyParams, qkvPad);
        if (hasGama_) { DataCopyPad(gTile_[buffer], gGm_[scalarOffset], params, pad); }
        DataCopyPad(betaTile_[buffer], betaGm_[scalarOffset], betaParams, betaPad);
        Mutex::Unlock<PIPE_MTE2>(INPUT_MUTEX[buffer]);
    }

    template <bool RAW_GATE>
    __aicore__ inline void ComputeOne(uint32_t buffer, uint16_t token, uint32_t srcBank, uint32_t dstBank, float aLog,
                                      float dtBias)
    {
        constexpr uint32_t gStride = DATA_BLOCK_BYTES_32 / sizeof(gType);
        constexpr uint32_t betaStride = DATA_BLOCK_BYTES_32 / sizeof(betaType);
        __ubuf__ stateType *stateOutputAddr = (__ubuf__ stateType *)stateBanks_[dstBank].GetPhyAddr();
        __ubuf__ stateType *stateInputAddr = (__ubuf__ stateType *)stateBanks_[srcBank].GetPhyAddr();
        RecurrentGatedDeltaRuleVf<inType, outType, stateType, gType, betaType, USE_QK_L2NORM, RAW_GATE,
                                  USE_BETA_SIGMOID, ALLOW_NEG_EIGVAL>(
            stateOutputAddr, stateInputAddr, (__ubuf__ inType *)kTile_[buffer][token * REQUIRED_DIM_128].GetPhyAddr(),
            (__ubuf__ inType *)qTile_[buffer][token * REQUIRED_DIM_128].GetPhyAddr(),
            (__ubuf__ inType *)vTile_[buffer][token * REQUIRED_DIM_128].GetPhyAddr(),
            (__ubuf__ gType *)gTile_[buffer][token * gStride].GetPhyAddr(),
            (__ubuf__ betaType *)betaTile_[buffer][token * betaStride].GetPhyAddr(),
            (__ubuf__ outType *)attnTileOut_[buffer][token * REQUIRED_DIM_128].GetPhyAddr(), scale_, aLog, dtBias);
    }

private:
    GlobalTensor<inType> queryGm_;
    GlobalTensor<inType> keyGm_;
    GlobalTensor<inType> valueGm_;
    GlobalTensor<gType> gGm_;
    GlobalTensor<betaType> betaGm_;
    GlobalTensor<stateType> initStateGm_;
    GlobalTensor<int32_t> cuSeqlensInt32Gm_;
    GlobalTensor<int32_t> stateIndicesInt32Gm_;
    GlobalTensor<int32_t> acceptedInt32Gm_;
    GlobalTensor<float> aLogGm_;
    GlobalTensor<float> dtBiasGm_;
    GlobalTensor<outType> attnOutGm_;
    GlobalTensor<stateType> finalStateGm_;

    LocalTensor<float> stateBanks_[STATE_BANKS_3];
    LocalTensor<inType> qTile_[INPUT_BUFFERS_2];
    LocalTensor<inType> kTile_[INPUT_BUFFERS_2];
    LocalTensor<inType> vTile_[INPUT_BUFFERS_2];
    LocalTensor<gType> gTile_[INPUT_BUFFERS_2];
    LocalTensor<betaType> betaTile_[INPUT_BUFFERS_2];
    LocalTensor<outType> attnTileOut_[INPUT_BUFFERS_2];

    uint32_t B_;
    uint32_t T_;
    uint32_t NK_;
    uint32_t NV_;
    uint32_t ssmStateStride_;
    uint64_t seqBatch_;
    int64_t seqStart_;
    int64_t seqEnd_;
    uint64_t queryTokenStride_;
    uint64_t queryHeadStride_;
    uint64_t keyTokenStride_;
    uint64_t keyHeadStride_;
    uint64_t valueTokenStride_;
    uint64_t valueHeadStride_;
    uint64_t stateInStride0_;
    uint64_t stateInStride1_;
    uint64_t stateInStride2_;
    uint64_t stateOutStride0_;
    uint64_t stateOutStride1_;
    uint64_t stateOutStride2_;
    bool hasAcceptedTokens_;
    bool hasDtBias_;
    bool hasGama_;
    bool seqValid_;
    float scale_;
    uint64_t blockIdx_;
};

} // namespace RecurrentGatedDeltaRule

#endif // RECURRENT_GATED_DELTA_RULE_ARCH35_H
