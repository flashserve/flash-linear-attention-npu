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

namespace RecurrentGatedDeltaRuleV2 {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t TOKEN_TILE = 4;
constexpr uint32_t INPUT_BUFFERS = 2;
constexpr uint32_t STATE_BANKS = 3;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t BF16_PER_BLOCK = DATA_BLOCK_BYTES / sizeof(bfloat16_t);
constexpr uint16_t FP32_PER_REG = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint16_t REQUIRED_DIM = 2 * FP32_PER_REG;
constexpr uint64_t INVALID_STATE_SLOT = static_cast<uint64_t>(-1);

constexpr CastTrait FP32_TO_B16_ZERO = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, RoundMode::CAST_RINT};
constexpr CastTrait FP32_TO_B16_ONE = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait B16_TO_FP32_ZERO = {
    RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr CastTrait B16_TO_FP32_ONE = {
    RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

enum class VfMode : uint8_t {
    COMPUTE,
    STORE_STATE,
};

template <VfMode MODE, typename inType, typename outType, typename stateType,
          typename gType, typename betaType,
          bool HAS_G, bool NORMALIZE_QK, bool RAW_GATE,
          bool BETA_SIGMOID, bool ALLOW_NEGATIVE, bool INITIAL_STATE>
__simd_vf__ inline void RecurrentGatedDeltaRuleVf(
    __ubuf__ float *stateOut, __ubuf__ float *stateIn,
    __ubuf__ stateType *stateIo,
    __ubuf__ inType *keyRaw, __ubuf__ inType *queryRaw, __ubuf__ inType *valueRaw,
    __ubuf__ gType *gRaw, __ubuf__ betaType *betaRaw,
    __ubuf__ outType *output, float scale, float aLog, float dtBias)
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
    uint32_t rowOffset = 0;
    uint16_t row = 0;

    if constexpr (MODE == VfMode::STORE_STATE) {
        for (row = 0; row < REQUIRED_DIM; ++row) {
            rowOffset = static_cast<uint32_t>(row) * REQUIRED_DIM;
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(state0, state1, stateIn + rowOffset);
            if constexpr (std::is_same<stateType, float>::value) {
                StoreAlign<float, StoreDist::DIST_INTLV_B32>(
                    stateIo + rowOffset, state0, state1, maskFp32);
            } else {
                Cast<stateType, float, FP32_TO_B16_ONE>(statePacked, state1, maskFp32);
                Cast<stateType, float, FP32_TO_B16_ZERO>(statePacked, state0, maskFp32);
                StoreAlign(stateIo + rowOffset, statePacked, maskB16);
            }
        }
    } else {
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

        if constexpr (HAS_G) {
            if constexpr (std::is_same<gType, float>::value) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(scalarGate, gRaw);
            } else {
                LoadAlign<gType, LoadDist::DIST_BRC_B16>(gPacked, gRaw);
                Cast<float, gType, B16_TO_FP32_ZERO>(scalarGate, gPacked, maskB16);
            }
        }

        if constexpr (std::is_same<betaType, float>::value) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(beta, betaRaw);
        } else {
            LoadAlign<betaType, LoadDist::DIST_BRC_B16>(betaPacked, betaRaw);
            Cast<float, betaType, B16_TO_FP32_ZERO>(beta, betaPacked, maskB16);
        }

        if constexpr (HAS_G && RAW_GATE) {
            Adds(scalarGate, scalarGate, dtBias, maskFp32);
        }
        if constexpr (BETA_SIGMOID) {
            Muls(beta, beta, -1.0f, maskFp32);
        }
        if constexpr (HAS_G && RAW_GATE) {
            Duplicate(expA, aLog, maskFp32);
        }
        if constexpr (HAS_G && !RAW_GATE) {
            Exp(scalarGate, scalarGate, maskFp32);
        }
        if constexpr (HAS_G && RAW_GATE) {
            Exp(expA, expA, maskFp32);
        }
        if constexpr (BETA_SIGMOID) {
            Exp(beta, beta, maskFp32);
        }
        if constexpr (HAS_G && RAW_GATE) {
            Exp(queryWork, scalarGate, maskFp32);
        }
        if constexpr (BETA_SIGMOID) {
            Adds(beta, beta, 1.0f, maskFp32);
        }
        if constexpr (HAS_G && RAW_GATE) {
            Adds(queryWork, queryWork, 1.0f, maskFp32);
        }
        if constexpr (BETA_SIGMOID) {
            Div(beta, one, beta, maskFp32);
        }
        if constexpr (HAS_G && RAW_GATE) {
            Ln(queryWork, queryWork, maskFp32);
        }
        if constexpr (BETA_SIGMOID && ALLOW_NEGATIVE) {
            Muls(beta, beta, 2.0f, maskFp32);
        }
        if constexpr (HAS_G && RAW_GATE) {
            Mul(scalarGate, queryWork, expA, maskFp32);
            Muls(scalarGate, scalarGate, -1.0f, maskFp32);
            Exp(scalarGate, scalarGate, maskFp32);
        }
        Mul(updateKey0, key0, beta, maskFp32);
        Mul(updateKey1, key1, beta, maskFp32);

        for (row = 0; row < REQUIRED_DIM; row += 2) {
            rowOffset = static_cast<uint32_t>(row) * REQUIRED_DIM;
            if constexpr (INITIAL_STATE && !std::is_same<stateType, float>::value) {
                LoadAlign<stateType, LoadDist::DIST_NORM>(statePacked, stateIo + rowOffset);
                LoadAlign<stateType, LoadDist::DIST_NORM>(
                    statePackedNext, stateIo + rowOffset + REQUIRED_DIM);
                Cast<float, stateType, B16_TO_FP32_ZERO>(state0, statePacked, maskB16);
                Cast<float, stateType, B16_TO_FP32_ONE>(state1, statePacked, maskB16);
                Cast<float, stateType, B16_TO_FP32_ZERO>(state2, statePackedNext, maskB16);
                Cast<float, stateType, B16_TO_FP32_ONE>(state3, statePackedNext, maskB16);
            } else {
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(state0, state1, stateIn + rowOffset);
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    state2, state3, stateIn + rowOffset + REQUIRED_DIM);
            }
            LoadAlign<inType, LoadDist::DIST_BRC_B16>(valuePacked, valueRaw + row);
            LoadAlign<inType, LoadDist::DIST_BRC_B16>(valuePackedNext, valueRaw + row + 1);

            if constexpr (HAS_G) {
                Mul(state0, state0, scalarGate, maskFp32);
                Mul(state2, state2, scalarGate, maskFp32);
                Mul(state1, state1, scalarGate, maskFp32);
                Mul(state3, state3, scalarGate, maskFp32);
            }

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

            StoreAlign<float, StoreDist::DIST_INTLV_B32>(
                stateOut + rowOffset, state0, state1, maskFp32);
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(
                stateOut + rowOffset + REQUIRED_DIM, state2, state3, maskFp32);
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
}

constexpr uint8_t MUTEX_MTE2_V = 0;
constexpr uint8_t MUTEX_V_MTE2 = 1;
constexpr uint8_t MUTEX_S_V = 2;
constexpr uint8_t INPUT_MUTEX[INPUT_BUFFERS] = {3, 4};
constexpr uint8_t OUTPUT_MUTEX[INPUT_BUFFERS] = {5, 6};
constexpr uint8_t STATE_MUTEX[STATE_BANKS] = {7, 8, 9};
constexpr uint8_t STATE_IO_MUTEX = 11;
struct RGDRV2InitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR g;
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

template <typename inType, typename outType, typename stateType, typename gType, typename betaType>
class RGDRV2 {
public:
    __aicore__ inline explicit RGDRV2(
        const RecurrentGatedDeltaRule::RecurrentGatedDeltaRuleTilingData *tilingData)
    {
        B_ = tilingData->b;
        T_ = tilingData->t;
        seqLen_ = 0;
        NK_ = tilingData->nk;
        NV_ = tilingData->nv;
        stateCapacity_ = tilingData->sBlockNum;
        ssmStateStride_ = tilingData->stateIndicesWidth;
        queryTokenStride_ = tilingData->queryStrides[0];
        queryHeadStride_ = tilingData->queryStrides[1];
        queryFeatureStride_ = tilingData->queryStrides[2];
        keyTokenStride_ = tilingData->keyStrides[0];
        keyHeadStride_ = tilingData->keyStrides[1];
        keyFeatureStride_ = tilingData->keyStrides[2];
        valueTokenStride_ = tilingData->valueStrides[0];
        valueHeadStride_ = tilingData->valueStrides[1];
        valueFeatureStride_ = tilingData->valueStrides[2];
        stateInStride0_ = tilingData->stateStrides[0];
        stateInStride1_ = tilingData->stateStrides[1];
        stateOutStride0_ = tilingData->stateStrides[0];
        stateOutStride1_ = tilingData->stateStrides[1];
        scale_ = tilingData->scale;
        hasCuSeqlens_ = true;
        cuSeqlensAreLengths_ = true;
        hasSsmStateIndices_ = true;
        hasAcceptedTokens_ = tilingData->hasAcceptedTokens != 0;
        hasG_ = tilingData->hasGama != 0;
        hasDtBias_ = tilingData->hasDtBias != 0;
        useQkL2norm_ = tilingData->useQkL2NormInKernel != 0;
        useGateInKernel_ = tilingData->useGateInKernel != 0;
        useBetaSigmoid_ = tilingData->useBetaSigmoidInKernel != 0;
        allowNegEigval_ = tilingData->allowNegEigval != 0;
        shouldStoreState_ = true;
        cuSeqlensDtype_ = 0;
        ssmStateIndicesDtype_ = 0;
        acceptedTokensDtype_ = 0;
    }

    __aicore__ inline void Init(const RGDRV2InitParams &params)
    {
        blockIdx_ = GetBlockIdx();
        queryGm_.SetGlobalBuffer((__gm__ inType *)params.query);
        keyGm_.SetGlobalBuffer((__gm__ inType *)params.key);
        valueGm_.SetGlobalBuffer((__gm__ inType *)params.value);
        gGm_.SetGlobalBuffer((__gm__ gType *)params.g);
        betaGm_.SetGlobalBuffer((__gm__ betaType *)params.beta);
        initStateGm_.SetGlobalBuffer((__gm__ stateType *)params.initState);
        cuSeqlensInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)params.cuSeqlens);
        cuSeqlensInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)params.cuSeqlens);
        stateIndicesInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)params.ssmStateIndices);
        stateIndicesInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)params.ssmStateIndices);
        aLogGm_.SetGlobalBuffer((__gm__ float *)params.aLog);
        dtBiasGm_.SetGlobalBuffer((__gm__ float *)params.dtBias);
        acceptedInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)params.numAcceptedTokens);
        acceptedInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)params.numAcceptedTokens);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)params.attnOut);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType *)params.finalState);
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        if (useGateInKernel_) {
            DispatchQk<true, true>();
        } else if (hasG_) {
            DispatchQk<true, false>();
        } else {
            DispatchQk<false, false>();
        }
    }

private:
    struct TaskInfo {
        uint64_t batch;
        uint64_t head;
        uint64_t stateSlot;
        int64_t seq0;
        int64_t seq1;
    };

    __aicore__ inline void InitBuffers()
    {
        uint32_t offset = 0;
        constexpr uint32_t stateElements = REQUIRED_DIM * REQUIRED_DIM;
        constexpr uint32_t stateBytes = stateElements * sizeof(float);
        constexpr uint32_t qkvElements = TOKEN_TILE * REQUIRED_DIM;
        constexpr uint32_t scalarBytes = TOKEN_TILE * DATA_BLOCK_BYTES;
        constexpr uint32_t outputElements = TOKEN_TILE * REQUIRED_DIM;
        for (uint32_t bank = 0; bank < STATE_BANKS; ++bank) {
            stateBanks_[bank] = LocalTensor<float>(TPosition::VECCALC, offset, stateElements);
            offset += stateBytes;
        }
        for (uint32_t buffer = 0; buffer < INPUT_BUFFERS; ++buffer) {
            qTile_[buffer] = LocalTensor<inType>(TPosition::VECCALC, offset, qkvElements);
            offset += qkvElements * sizeof(inType);
            kTile_[buffer] = LocalTensor<inType>(TPosition::VECCALC, offset, qkvElements);
            offset += qkvElements * sizeof(inType);
            vTile_[buffer] = LocalTensor<inType>(TPosition::VECCALC, offset, qkvElements);
            offset += qkvElements * sizeof(inType);
            gTile_[buffer] = LocalTensor<gType>(TPosition::VECCALC, offset,
                                               scalarBytes / sizeof(gType));
            offset += scalarBytes;
            betaTile_[buffer] = LocalTensor<betaType>(TPosition::VECCALC, offset,
                                                      scalarBytes / sizeof(betaType));
            offset += scalarBytes;
            attnTileOut_[buffer] = LocalTensor<outType>(TPosition::VECCALC, offset, outputElements);
            offset += outputElements * sizeof(outType);
        }
        if constexpr (!std::is_same<stateType, float>::value) {
            stateIo_ = LocalTensor<stateType>(TPosition::VECCALC, offset, stateElements);
            offset += stateElements * sizeof(stateType);
        }
        stridedStaging_ = LocalTensor<inType>(
            TPosition::VECCALC, offset, REQUIRED_DIM * DATA_BLOCK_BYTES / sizeof(inType));
        offset += REQUIRED_DIM * DATA_BLOCK_BYTES;
        stridedOffsets_ = LocalTensor<uint32_t>(TPosition::VECCALC, offset, REQUIRED_DIM);
        for (uint32_t feature = 0; feature < REQUIRED_DIM; ++feature) {
            stridedOffsets_.SetValue(feature, feature * DATA_BLOCK_BYTES);
        }
        SyncSToV();
    }

    __aicore__ inline void SyncMte2ToV()
    {
        Mutex::Lock<PIPE_MTE2>(MUTEX_MTE2_V);
        Mutex::Unlock<PIPE_MTE2>(MUTEX_MTE2_V);
        Mutex::Lock<PIPE_V>(MUTEX_MTE2_V);
        Mutex::Unlock<PIPE_V>(MUTEX_MTE2_V);
    }

    __aicore__ inline void SyncVToMte2()
    {
        Mutex::Lock<PIPE_V>(MUTEX_V_MTE2);
        Mutex::Unlock<PIPE_V>(MUTEX_V_MTE2);
        Mutex::Lock<PIPE_MTE2>(MUTEX_V_MTE2);
        Mutex::Unlock<PIPE_MTE2>(MUTEX_V_MTE2);
    }

    __aicore__ inline void SyncSToV()
    {
        Mutex::Lock<PIPE_S>(MUTEX_S_V);
        Mutex::Unlock<PIPE_S>(MUTEX_S_V);
        Mutex::Lock<PIPE_V>(MUTEX_S_V);
        Mutex::Unlock<PIPE_V>(MUTEX_S_V);
    }

    template <bool HAS_G, bool RAW_GATE>
    __aicore__ inline void DispatchQk()
    {
        if (useQkL2norm_) {
            DispatchBeta<HAS_G, true, RAW_GATE>();
        } else {
            DispatchBeta<HAS_G, false, RAW_GATE>();
        }
    }

    template <bool HAS_G, bool NORMALIZE_QK, bool RAW_GATE>
    __aicore__ inline void DispatchBeta()
    {
        if (useBetaSigmoid_) {
            if (allowNegEigval_) {
                ProcessImpl<HAS_G, NORMALIZE_QK, RAW_GATE, true, true>();
            } else {
                ProcessImpl<HAS_G, NORMALIZE_QK, RAW_GATE, true, false>();
            }
        } else {
            ProcessImpl<HAS_G, NORMALIZE_QK, RAW_GATE, false, false>();
        }
    }

    template <bool HAS_G, bool NORMALIZE_QK, bool RAW_GATE,
              bool BETA_SIGMOID, bool ALLOW_NEGATIVE>
    __aicore__ inline void ProcessImpl()
    {
        uint64_t taskIdx = blockIdx_;
        const uint64_t taskNum = static_cast<uint64_t>(B_) * NV_;
        const uint64_t taskStride = GetBlockNum();
        TaskInfo task{};
        TaskInfo nextTask{};
        bool taskValid = true;
        if (!PrepareNextTask(taskIdx, taskNum, taskStride, task, taskValid)) {
            return;
        }

        uint32_t initialBank = 0;
        uint32_t bufferPhase = 0;
        bool inputReady = false;
        CopyInitialState(task, initialBank);
        while (true) {
            bool nextTaskValid = true;
            const bool hasNextTask = PrepareNextTask(
                taskIdx, taskNum, taskStride, nextTask, nextTaskValid);
            const uint32_t nextInitialBank = hasNextTask ?
                NextInitialBank(initialBank, static_cast<uint32_t>(task.seq1 - task.seq0)) : 0;
            ProcessHead<HAS_G, NORMALIZE_QK, RAW_GATE,
                        BETA_SIGMOID, ALLOW_NEGATIVE>(
                task, initialBank, nextTask, nextInitialBank, hasNextTask,
                bufferPhase, inputReady);
            if (!hasNextTask) {
                return;
            }
            task = nextTask;
            initialBank = nextInitialBank;
        }
    }

    __aicore__ inline uint32_t NextInitialBank(uint32_t initialBank, uint32_t tokenCount) const
    {
        if (tokenCount == 1) {
            return (initialBank + STATE_BANKS - 1U) % STATE_BANKS;
        }
        return (initialBank + tokenCount - 2U) % STATE_BANKS;
    }

    __aicore__ inline bool PrepareNextTask(uint64_t &taskIdx, uint64_t taskNum, uint64_t taskStride,
                                           TaskInfo &task, bool &taskValid)
    {
        taskValid = true;
        while (taskIdx < taskNum) {
            const uint64_t current = taskIdx;
            taskIdx += taskStride;
            const uint64_t batch = current / NV_;
            const int64_t seq0 = SequenceStart(batch);
            const int64_t seq1 = SequenceEnd(batch);
            if (seq0 < 0 || seq1 < seq0 || seq1 > static_cast<int64_t>(T_) ||
                (ssmStateStride_ > 0 && seq1 - seq0 > static_cast<int64_t>(ssmStateStride_))) {
                taskValid = false;
                return false;
            }
            if (seq0 == seq1) {
                continue;
            }
            const uint64_t stateSlot = ResolveInitialStateSlot(batch, seq0,
                                                               static_cast<int32_t>(seq1 - seq0));
            if (stateSlot == INVALID_STATE_SLOT) {
                taskValid = false;
                return false;
            }
            task = {batch, current % NV_, stateSlot, seq0, seq1};
            return true;
        }
        return false;
    }

    __aicore__ inline int64_t LoadCuSeqlens(uint64_t index) const
    {
        return cuSeqlensDtype_ == 0 ? static_cast<int64_t>(cuSeqlensInt32Gm_.GetValue(index)) :
                                     cuSeqlensInt64Gm_.GetValue(index);
    }

    __aicore__ inline int64_t SequenceStart(uint64_t batch) const
    {
        if (cuSeqlensAreLengths_) {
            int64_t start = LoadCuSeqlens(0);
            for (uint64_t i = 0; i < batch; ++i) {
                start += LoadCuSeqlens(i + 1);
            }
            return start;
        }
        return hasCuSeqlens_ ? LoadCuSeqlens(batch) : static_cast<int64_t>(batch * seqLen_);
    }

    __aicore__ inline int64_t SequenceEnd(uint64_t batch) const
    {
        if (cuSeqlensAreLengths_) {
            return SequenceStart(batch) + LoadCuSeqlens(batch + 1);
        }
        return hasCuSeqlens_ ? LoadCuSeqlens(batch + 1) :
                               static_cast<int64_t>((batch + 1) * seqLen_);
    }

    __aicore__ inline int64_t LoadStateIndex(uint64_t index) const
    {
        return ssmStateIndicesDtype_ == 0 ? static_cast<int64_t>(stateIndicesInt32Gm_.GetValue(index)) :
                                           stateIndicesInt64Gm_.GetValue(index);
    }

    __aicore__ inline int64_t LoadAccepted(uint64_t index) const
    {
        return acceptedTokensDtype_ == 0 ? static_cast<int64_t>(acceptedInt32Gm_.GetValue(index)) :
                                          acceptedInt64Gm_.GetValue(index);
    }

    __aicore__ inline uint64_t StateMetadataOffset(uint64_t batch, int64_t seq0, int64_t token) const
    {
        return ssmStateStride_ == 0 ? static_cast<uint64_t>(token) :
               batch * ssmStateStride_ + static_cast<uint64_t>(token - seq0);
    }

    __aicore__ inline uint64_t LoadStateSlot(uint64_t batch, int64_t seq0, int64_t token) const
    {
        const int64_t slot = LoadStateIndex(StateMetadataOffset(batch, seq0, token));
        return slot >= 0 && slot < static_cast<int64_t>(stateCapacity_) ?
               static_cast<uint64_t>(slot) : INVALID_STATE_SLOT;
    }

    __aicore__ inline uint64_t ResolveInitialStateSlot(uint64_t batch, int64_t seq0, int32_t seqLen) const
    {
        if (!hasSsmStateIndices_) {
            return batch < stateCapacity_ ? batch : INVALID_STATE_SLOT;
        }
        int64_t token = seq0;
        if (hasAcceptedTokens_) {
            const int64_t accepted = LoadAccepted(batch);
            if (accepted <= 0 || accepted > seqLen) {
                return INVALID_STATE_SLOT;
            }
            token += accepted - 1;
        }
        return LoadStateSlot(batch, seq0, token);
    }

    __aicore__ inline uint64_t StateSlotForToken(uint64_t batch, int64_t seq0, int64_t token) const
    {
        return hasSsmStateIndices_ ? LoadStateSlot(batch, seq0, token) : batch;
    }

    __aicore__ inline void CopyInitialState(const TaskInfo &task, uint32_t bank)
    {
        const uint64_t stateOffset = stateInStride0_ * task.stateSlot + stateInStride1_ * task.head;
        DataCopyExtParams params{REQUIRED_DIM, REQUIRED_DIM * sizeof(stateType), 0, 0, 0};
        DataCopyPadExtParams<stateType> pad{true, 0, 0, static_cast<stateType>(0)};
        LocalTensor<stateType> stateInput = stateBanks_[bank].template ReinterpretCast<stateType>();
        Mutex::Lock<PIPE_MTE2>(STATE_MUTEX[bank]);
        DataCopyPad(stateInput, initStateGm_[stateOffset], params, pad);
        Mutex::Unlock<PIPE_MTE2>(STATE_MUTEX[bank]);
    }

    __aicore__ inline void PrefetchInitialState(const TaskInfo &task, uint32_t bank)
    {
        Mutex::Lock<PIPE_V>(STATE_MUTEX[bank]);
        Mutex::Unlock<PIPE_V>(STATE_MUTEX[bank]);
        CopyInitialState(task, bank);
    }

    __aicore__ inline void CopyStridedRows(LocalTensor<inType> &dst, GlobalTensor<inType> &src,
                                           uint64_t baseOffset, uint64_t tokenStride,
                                           uint64_t featureStride, uint16_t rows)
    {
        constexpr uint32_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(inType);
        DataCopyExtParams params{REQUIRED_DIM, static_cast<uint32_t>(sizeof(inType)),
                                 static_cast<uint32_t>((featureStride - 1) * sizeof(inType)), 0, 0};
        DataCopyPadExtParams<inType> pad{true, 0, static_cast<uint8_t>(elementsPerBlock - 1),
                                        static_cast<inType>(0)};
        for (uint16_t row = 0; row < rows; ++row) {
            const uint64_t srcOffset = baseOffset + static_cast<uint64_t>(row) * tokenStride;
            const uint32_t dstOffset = static_cast<uint32_t>(row) * REQUIRED_DIM;
            DataCopyPad(stridedStaging_, src[srcOffset], params, pad);
            SyncMte2ToV();
            Gather(dst[dstOffset], stridedStaging_, stridedOffsets_, 0, REQUIRED_DIM);
            SyncVToMte2();
        }
    }

    __aicore__ inline void CopyQkvRows(LocalTensor<inType> &dst, GlobalTensor<inType> &src,
                                       uint64_t baseOffset, uint64_t tokenStride,
                                       uint64_t featureStride, uint16_t rows)
    {
        if (featureStride == 1) {
            DataCopyExtParams params{rows, REQUIRED_DIM * sizeof(inType),
                                     static_cast<uint32_t>((tokenStride - REQUIRED_DIM) * sizeof(inType)),
                                     0, 0};
            DataCopyPadExtParams<inType> pad{true, 0, 0, static_cast<inType>(0)};
            DataCopyPad(dst, src[baseOffset], params, pad);
        } else {
            CopyStridedRows(dst, src, baseOffset, tokenStride, featureStride, rows);
        }
    }

    template <bool HAS_G>
    __aicore__ inline void CopyInputTile(uint32_t buffer, int64_t start, int64_t end,
                                        uint64_t qkHead, uint64_t valueHead)
    {
        const uint16_t rows = static_cast<uint16_t>(end - start);
        const uint64_t qOffset = static_cast<uint64_t>(start) * queryTokenStride_ +
                                 qkHead * queryHeadStride_;
        const uint64_t kOffset = static_cast<uint64_t>(start) * keyTokenStride_ +
                                 qkHead * keyHeadStride_;
        const uint64_t vOffset = static_cast<uint64_t>(start) * valueTokenStride_ +
                                 valueHead * valueHeadStride_;
        Mutex::Lock<PIPE_MTE2>(INPUT_MUTEX[buffer]);
        CopyQkvRows(qTile_[buffer], queryGm_, qOffset, queryTokenStride_, queryFeatureStride_, rows);
        CopyQkvRows(kTile_[buffer], keyGm_, kOffset, keyTokenStride_, keyFeatureStride_, rows);
        CopyQkvRows(vTile_[buffer], valueGm_, vOffset, valueTokenStride_, valueFeatureStride_, rows);

        const uint64_t scalarOffset = static_cast<uint64_t>(start) * NV_ + valueHead;
        if constexpr (HAS_G) {
            constexpr uint32_t scalarRow = DATA_BLOCK_BYTES / sizeof(gType);
            DataCopyExtParams params{rows, static_cast<uint32_t>(sizeof(gType)),
                                     static_cast<uint32_t>((NV_ - 1) * sizeof(gType)), 0, 0};
            DataCopyPadExtParams<gType> pad{true, 0, static_cast<uint8_t>(scalarRow - 1),
                                           static_cast<gType>(0)};
            DataCopyPad(gTile_[buffer], gGm_[scalarOffset], params, pad);
        }
        constexpr uint32_t betaRow = DATA_BLOCK_BYTES / sizeof(betaType);
        DataCopyExtParams betaParams{rows, static_cast<uint32_t>(sizeof(betaType)),
                                     static_cast<uint32_t>((NV_ - 1) * sizeof(betaType)), 0, 0};
        DataCopyPadExtParams<betaType> betaPad{true, 0, static_cast<uint8_t>(betaRow - 1),
                                              static_cast<betaType>(0)};
        DataCopyPad(betaTile_[buffer], betaGm_[scalarOffset], betaParams, betaPad);
        Mutex::Unlock<PIPE_MTE2>(INPUT_MUTEX[buffer]);
    }

    template <bool HAS_G>
    __aicore__ inline void CopyFirstInputTile(const TaskInfo &task, uint32_t buffer)
    {
        const uint64_t qkHead = task.head / (NV_ / NK_);
        const int64_t end = task.seq0 + TOKEN_TILE < task.seq1 ?
                            task.seq0 + TOKEN_TILE : task.seq1;
        CopyInputTile<HAS_G>(buffer, task.seq0, end, qkHead, task.head);
    }

    template <bool HAS_G, bool NORMALIZE_QK, bool RAW_GATE,
              bool BETA_SIGMOID, bool ALLOW_NEGATIVE, bool INITIAL_STATE>
    __aicore__ inline void ComputeOne(uint32_t inputBuffer, uint32_t outputBuffer,
                                     uint16_t token, uint32_t srcBank, uint32_t dstBank,
                                     float aLog, float dtBias)
    {
        constexpr uint32_t gStride = DATA_BLOCK_BYTES / sizeof(gType);
        constexpr uint32_t betaStride = DATA_BLOCK_BYTES / sizeof(betaType);
        __ubuf__ stateType *stateIoAddr;
        if constexpr (std::is_same<stateType, float>::value) {
            stateIoAddr = (__ubuf__ stateType *)stateBanks_[srcBank].GetPhyAddr();
        } else if constexpr (INITIAL_STATE) {
            stateIoAddr = (__ubuf__ stateType *)stateBanks_[srcBank].GetPhyAddr();
        } else {
            stateIoAddr = (__ubuf__ stateType *)stateIo_.GetPhyAddr();
        }
        RecurrentGatedDeltaRuleVf<VfMode::COMPUTE, inType, outType, stateType,
                                  gType, betaType, HAS_G, NORMALIZE_QK,
                                  RAW_GATE, BETA_SIGMOID, ALLOW_NEGATIVE, INITIAL_STATE>(
            (__ubuf__ float *)stateBanks_[dstBank].GetPhyAddr(),
            (__ubuf__ float *)stateBanks_[srcBank].GetPhyAddr(),
            stateIoAddr,
            (__ubuf__ inType *)kTile_[inputBuffer][token * REQUIRED_DIM].GetPhyAddr(),
            (__ubuf__ inType *)qTile_[inputBuffer][token * REQUIRED_DIM].GetPhyAddr(),
            (__ubuf__ inType *)vTile_[inputBuffer][token * REQUIRED_DIM].GetPhyAddr(),
            (__ubuf__ gType *)gTile_[inputBuffer][token * gStride].GetPhyAddr(),
            (__ubuf__ betaType *)betaTile_[inputBuffer][token * betaStride].GetPhyAddr(),
            (__ubuf__ outType *)attnTileOut_[outputBuffer][token * REQUIRED_DIM].GetPhyAddr(),
            scale_, aLog, dtBias);
    }

    __aicore__ inline void ConvertStateForStore(uint32_t bank)
    {
        __ubuf__ float *stateAddr = (__ubuf__ float *)stateBanks_[bank].GetPhyAddr();
        __ubuf__ stateType *stateIoAddr = (__ubuf__ stateType *)stateIo_.GetPhyAddr();
        RecurrentGatedDeltaRuleVf<VfMode::STORE_STATE, inType, outType, stateType,
                                  gType, betaType, false, false, false, false, false, false>(
            stateAddr, stateAddr, stateIoAddr,
            (__ubuf__ inType *)stateAddr, (__ubuf__ inType *)stateAddr,
            (__ubuf__ inType *)stateAddr, (__ubuf__ gType *)stateAddr,
            (__ubuf__ betaType *)stateAddr, (__ubuf__ outType *)stateIoAddr,
            0.0f, 0.0f, 0.0f);
    }

    __aicore__ inline void CopyStateOut(uint32_t bank, uint64_t stateSlot, uint64_t head)
    {
        const uint64_t offset = stateOutStride0_ * stateSlot + stateOutStride1_ * head;
        DataCopyExtParams params{REQUIRED_DIM, REQUIRED_DIM * sizeof(stateType), 0, 0, 0};
        if constexpr (std::is_same<stateType, float>::value) {
            DataCopyPad(finalStateGm_[offset], stateBanks_[bank], params);
        } else {
            DataCopyPad(finalStateGm_[offset], stateIo_, params);
        }
    }

    __aicore__ inline void CopyAttnOut(uint32_t buffer, int64_t start, uint16_t rows, uint64_t head)
    {
        const uint64_t offset = (static_cast<uint64_t>(start) * NV_ + head) * REQUIRED_DIM;
        DataCopyExtParams params{rows, REQUIRED_DIM * sizeof(outType), 0,
                                 static_cast<uint32_t>((NV_ - 1) * REQUIRED_DIM * sizeof(outType)), 0};
        DataCopyPad(attnOutGm_[offset], attnTileOut_[buffer], params);
    }

    template <bool HAS_G, bool NORMALIZE_QK, bool RAW_GATE,
              bool BETA_SIGMOID, bool ALLOW_NEGATIVE>
    __aicore__ inline void ProcessHead(const TaskInfo &task, uint32_t initialBank,
                                       const TaskInfo &nextTask, uint32_t nextInitialBank,
                                       bool hasNextTask, uint32_t &bufferPhase,
                                       bool &inputReady)
    {
        const uint64_t qkHead = task.head / (NV_ / NK_);
        if (!inputReady) {
            CopyFirstInputTile<HAS_G>(task, bufferPhase);
        }
        inputReady = false;
        Mutex::Lock<PIPE_V>(STATE_MUTEX[initialBank]);
        Mutex::Unlock<PIPE_V>(STATE_MUTEX[initialBank]);

        const int64_t tokenCount = task.seq1 - task.seq0;
        if (hasNextTask && tokenCount == 1) {
            PrefetchInitialState(nextTask, nextInitialBank);
            CopyFirstInputTile<HAS_G>(nextTask, bufferPhase ^ 1U);
            inputReady = true;
        }
        const float aLog = RAW_GATE ? aLogGm_.GetValue(task.head) : 0.0f;
        const float dtBias = RAW_GATE && hasDtBias_ ? dtBiasGm_.GetValue(task.head) : 0.0f;
        const int64_t prefetchToken = task.seq1 - (tokenCount > 1 ? 2 : 1);
        uint32_t currentBank = initialBank;
        uint32_t tileIndex = 0;
        for (int64_t start = task.seq0; start < task.seq1; start += TOKEN_TILE, ++tileIndex) {
            const int64_t end = start + TOKEN_TILE < task.seq1 ? start + TOKEN_TILE : task.seq1;
            const uint16_t rows = static_cast<uint16_t>(end - start);
            const uint32_t inputBuffer = (bufferPhase + tileIndex) & 1U;
            const uint32_t outputBuffer = (bufferPhase + tileIndex) & 1U;
            Mutex::Lock<PIPE_V>(INPUT_MUTEX[inputBuffer]);
            Mutex::Lock<PIPE_V>(OUTPUT_MUTEX[outputBuffer]);

            for (uint16_t token = 0; token < rows; ++token) {
                const int64_t seqIndex = start + token;
                const uint64_t outputSlot = StateSlotForToken(task.batch, task.seq0, seqIndex);
                if (outputSlot == INVALID_STATE_SLOT) {
                    return;
                }
                const uint32_t nextBank = (currentBank + 1U) % STATE_BANKS;
                Mutex::Lock<PIPE_V>(STATE_MUTEX[nextBank]);
                if constexpr (std::is_same<stateType, float>::value) {
                    ComputeOne<HAS_G, NORMALIZE_QK, RAW_GATE,
                               BETA_SIGMOID, ALLOW_NEGATIVE, false>(
                        inputBuffer, outputBuffer, token, currentBank, nextBank, aLog, dtBias);
                } else if (tileIndex == 0 && token == 0) {
                    Mutex::Lock<PIPE_V>(STATE_IO_MUTEX);
                    ComputeOne<HAS_G, NORMALIZE_QK, RAW_GATE,
                               BETA_SIGMOID, ALLOW_NEGATIVE, true>(
                        inputBuffer, outputBuffer, token, currentBank, nextBank, aLog, dtBias);
                    Mutex::Unlock<PIPE_V>(STATE_IO_MUTEX);
                } else {
                    ComputeOne<HAS_G, NORMALIZE_QK, RAW_GATE,
                               BETA_SIGMOID, ALLOW_NEGATIVE, false>(
                        inputBuffer, outputBuffer, token, currentBank, nextBank, aLog, dtBias);
                }
                Mutex::Unlock<PIPE_V>(STATE_MUTEX[nextBank]);
                if (hasNextTask && tokenCount > 1 && seqIndex == prefetchToken) {
                    PrefetchInitialState(nextTask, nextInitialBank);
                }
                if (token == 0 && end < task.seq1) {
                    const int64_t nextEnd = end + TOKEN_TILE < task.seq1 ?
                                            end + TOKEN_TILE : task.seq1;
                    CopyInputTile<HAS_G>(inputBuffer ^ 1U, end, nextEnd, qkHead, task.head);
                } else if (token == 0 && hasNextTask && !inputReady) {
                    CopyFirstInputTile<HAS_G>(nextTask, inputBuffer ^ 1U);
                    inputReady = true;
                }
                if (shouldStoreState_) {
                    if constexpr (!std::is_same<stateType, float>::value) {
                        Mutex::Lock<PIPE_V>(STATE_IO_MUTEX);
                        ConvertStateForStore(nextBank);
                        Mutex::Unlock<PIPE_V>(STATE_IO_MUTEX);
                        Mutex::Lock<PIPE_MTE3>(STATE_IO_MUTEX);
                    }
                    Mutex::Lock<PIPE_MTE3>(STATE_MUTEX[nextBank]);
                    CopyStateOut(nextBank, outputSlot, task.head);
                    Mutex::Unlock<PIPE_MTE3>(STATE_MUTEX[nextBank]);
                    if constexpr (!std::is_same<stateType, float>::value) {
                        Mutex::Unlock<PIPE_MTE3>(STATE_IO_MUTEX);
                    }
                }
                currentBank = nextBank;
            }

            Mutex::Unlock<PIPE_V>(INPUT_MUTEX[inputBuffer]);
            Mutex::Unlock<PIPE_V>(OUTPUT_MUTEX[outputBuffer]);
            Mutex::Lock<PIPE_MTE3>(OUTPUT_MUTEX[outputBuffer]);
            CopyAttnOut(outputBuffer, start, rows, task.head);
            Mutex::Unlock<PIPE_MTE3>(OUTPUT_MUTEX[outputBuffer]);
        }
        bufferPhase = (bufferPhase + tileIndex) & 1U;
    }

private:
    GlobalTensor<inType> queryGm_;
    GlobalTensor<inType> keyGm_;
    GlobalTensor<inType> valueGm_;
    GlobalTensor<gType> gGm_;
    GlobalTensor<betaType> betaGm_;
    GlobalTensor<stateType> initStateGm_;
    GlobalTensor<int32_t> cuSeqlensInt32Gm_;
    GlobalTensor<int64_t> cuSeqlensInt64Gm_;
    GlobalTensor<int32_t> stateIndicesInt32Gm_;
    GlobalTensor<int64_t> stateIndicesInt64Gm_;
    GlobalTensor<int32_t> acceptedInt32Gm_;
    GlobalTensor<int64_t> acceptedInt64Gm_;
    GlobalTensor<float> aLogGm_;
    GlobalTensor<float> dtBiasGm_;
    GlobalTensor<outType> attnOutGm_;
    GlobalTensor<stateType> finalStateGm_;

    LocalTensor<float> stateBanks_[STATE_BANKS];
    LocalTensor<stateType> stateIo_;
    LocalTensor<inType> qTile_[INPUT_BUFFERS];
    LocalTensor<inType> kTile_[INPUT_BUFFERS];
    LocalTensor<inType> vTile_[INPUT_BUFFERS];
    LocalTensor<gType> gTile_[INPUT_BUFFERS];
    LocalTensor<betaType> betaTile_[INPUT_BUFFERS];
    LocalTensor<outType> attnTileOut_[INPUT_BUFFERS];
    LocalTensor<inType> stridedStaging_;
    LocalTensor<uint32_t> stridedOffsets_;

    uint32_t B_;
    uint32_t T_;
    uint32_t seqLen_;
    uint32_t NK_;
    uint32_t NV_;
    uint32_t stateCapacity_;
    uint32_t ssmStateStride_;
    uint64_t queryTokenStride_;
    uint64_t queryHeadStride_;
    uint64_t queryFeatureStride_;
    uint64_t keyTokenStride_;
    uint64_t keyHeadStride_;
    uint64_t keyFeatureStride_;
    uint64_t valueTokenStride_;
    uint64_t valueHeadStride_;
    uint64_t valueFeatureStride_;
    uint64_t stateInStride0_;
    uint64_t stateInStride1_;
    uint64_t stateOutStride0_;
    uint64_t stateOutStride1_;
    uint32_t cuSeqlensDtype_;
    uint32_t ssmStateIndicesDtype_;
    uint32_t acceptedTokensDtype_;
    bool hasCuSeqlens_;
    bool cuSeqlensAreLengths_;
    bool hasSsmStateIndices_;
    bool hasAcceptedTokens_;
    bool hasG_;
    bool hasDtBias_;
    bool useQkL2norm_;
    bool useGateInKernel_;
    bool useBetaSigmoid_;
    bool allowNegEigval_;
    bool shouldStoreState_;
    float scale_;
    uint64_t blockIdx_;
};

} // namespace RecurrentGatedDeltaRuleV2

#endif // RECURRENT_GATED_DELTA_RULE_ARCH35_H
