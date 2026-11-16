/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <type_traits>
#include "kernel_operator.h"
#include "../recurrent_kda_struct.h"

// stand-pr: class-based engine following PR #1083 (efbe1866).
// This namespace owns the entire UB and its fixed Mutex IDs; it is never mixed
// with TPipe allocation or the x8-pr scheduler.
namespace RecurrentKdaStand {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using RecurrentKda::RecurrentKdaTilingData;
constexpr uint32_t DIM = 128;
constexpr uint32_t TOKEN_TILE = 4;
constexpr uint32_t STATE_BANKS = 3;
constexpr uint32_t INPUT_BUFFERS = 2;
constexpr uint32_t PARAMETER_BUFFERS = 2;
constexpr uint32_t STATE_BYTES = DIM * DIM * sizeof(float);
constexpr uint32_t TILE_VECTOR_BYTES = TOKEN_TILE * DIM * sizeof(float);
constexpr uint32_t TILE_SCALAR_BYTES = TOKEN_TILE * 32;
constexpr uint32_t PREDICTION_BYTES = DIM * sizeof(float);
constexpr uint64_t INVALID_STATE_SLOT = uint64_t(-1);
constexpr uint32_t UB_BYTES = 3 * STATE_BYTES +
    2 * (4 * TOKEN_TILE * DIM * sizeof(bfloat16_t) + TILE_VECTOR_BYTES + TILE_SCALAR_BYTES) +
    PARAMETER_BUFFERS * DIM * sizeof(float);
static_assert(UB_BYTES + PREDICTION_BYTES <= 256 * 1024, "stand-pr UB exceeds A5 capacity");
constexpr uint8_t INPUT_MUTEX[2] = {3, 4};
constexpr uint8_t OUTPUT_MUTEX[2] = {5, 6};
constexpr uint8_t STATE_MUTEX[3] = {7, 8, 9};
constexpr uint8_t PARAMETER_MUTEX[PARAMETER_BUFFERS] = {10, 11};
constexpr CastTrait TO_FLOAT_ZERO = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr CastTrait TO_FLOAT_ONE = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr CastTrait TO_B16_ZERO = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, RoundMode::CAST_RINT};
constexpr CastTrait TO_B16_ONE = {RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait TO_B16_WORD = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// even/odd K lanes match the reference's DIST_DINTLV_B32 FP32 State load.
template <typename T>
__simd_callee__ inline void LoadPair(RegTensor<float>& even, RegTensor<float>& odd,
    __ubuf__ T* source, MaskReg& mask16)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(even, odd, source);
    } else {
        RegTensor<T> packed;
        LoadAlign(packed, source);
        Cast<float, T, TO_FLOAT_ZERO>(even, packed, mask16);
        Cast<float, T, TO_FLOAT_ONE>(odd, packed, mask16);
    }
}

__simd_callee__ inline void NormalizePair(RegTensor<float>& even, RegTensor<float>& odd, MaskReg& mask)
{
    RegTensor<float> work, norm, one;
    MaskReg first = CreateMask<float, MaskPattern::VL1>();
    Mul(work, even, even, mask);
    MulAddDst(work, odd, odd, mask);
    ReduceSum(norm, work, mask);
    Adds(norm, norm, 1.0e-6f, first);
    Sqrt(norm, norm, first);
    Duplicate(one, 1.0f, mask);
    Div(norm, one, norm, first);
    Duplicate(norm, norm, mask);
    Mul(even, even, norm, mask);
    Mul(odd, odd, norm, mask);
}

// Stable softplus; exp() never receives a positive argument in this part.
__simd_callee__ inline void Decay(RegTensor<float>& gate, RegTensor<float>& expA,
    bool rawGate, bool safeGate, float lowerBound, MaskReg& mask)
{
    if (rawGate) {
        RegTensor<float> work, positive, one;
        if (safeGate) {
            Mul(gate, gate, expA, mask);
            Muls(gate, gate, -1.0f, mask);
            Exp(gate, gate, mask);
            Adds(gate, gate, 1.0f, mask);
            Duplicate(one, 1.0f, mask);
            Div(gate, one, gate, mask);
            Muls(gate, gate, lowerBound, mask);
        } else {
            Maxs(positive, gate, 0.0f, mask);
            Abs(work, gate, mask);
            Muls(work, work, -1.0f, mask);
            Exp(work, work, mask);
            Adds(work, work, 1.0f, mask);
            Ln(work, work, mask);
            Add(gate, positive, work, mask);
            Mul(gate, gate, expA, mask);
            Muls(gate, gate, -1.0f, mask);
        }
    }
    Exp(gate, gate, mask);
}

template <typename State>
__simd_callee__ inline void StoreVFirstRows(__ubuf__ State* target, uint16_t row,
    RegTensor<float>& s0, RegTensor<float>& s1, RegTensor<float>& s2, RegTensor<float>& s3,
    MaskReg& mask, MaskReg& mask16)
{
    if constexpr (std::is_same<State, float>::value) {
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(target + row * DIM, s0, s1, mask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(target + (row + 1) * DIM, s2, s3, mask);
    } else {
        RegTensor<State> packed;
        Cast<State, float, TO_B16_ONE>(packed, s1, mask);
        Cast<State, float, TO_B16_ZERO>(packed, s0, mask);
        StoreAlign(target + row * DIM, packed, mask16);
        Cast<State, float, TO_B16_ONE>(packed, s3, mask);
        Cast<State, float, TO_B16_ZERO>(packed, s2, mask);
        StoreAlign(target + (row + 1) * DIM, packed, mask16);
    }
}

// Keep the reference engine's two-row grain and streamed out collection.
// The first pass frees State registers after prediction; the second reloads
// the same source bank and consumes one residual per row.
template <typename State>
__simd_callee__ inline void RecurrentKdaVFirstUpdateRows(
    __ubuf__ State* target, __ubuf__ State* source, __ubuf__ bfloat16_t*& outputAddress,
    __ubuf__ float* residual,
    RegTensor<float>& queryEven, RegTensor<float>& queryOdd,
    RegTensor<float>& decayEven, RegTensor<float>& decayOdd,
    RegTensor<float>& updateKeyEven, RegTensor<float>& updateKeyOdd,
    MaskReg& mask, MaskReg& mask16, MaskReg& first,
    UnalignRegForStore& outputStore)
{
    #pragma unroll 1
    for (uint16_t row = 0; row < DIM; row += 2) {
        RegTensor<float> s0, s1, s2, s3, residual0, residual1, dot0, dot1, out0, out1;
        RegTensor<bfloat16_t> packedOut;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(residual0, residual + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(residual1, residual + row + 1);
        LoadPair(s0, s1, source + row * DIM, mask16);
        LoadPair(s2, s3, source + (row + 1) * DIM, mask16);
        Mul(s0, s0, decayEven, mask);
        Mul(s2, s2, decayEven, mask);
        Mul(s1, s1, decayOdd, mask);
        Mul(s3, s3, decayOdd, mask);
        MulAddDst(s0, residual0, updateKeyEven, mask);
        MulAddDst(s2, residual1, updateKeyEven, mask);
        MulAddDst(s1, residual0, updateKeyOdd, mask);
        MulAddDst(s3, residual1, updateKeyOdd, mask);
        Mul(dot0, s0, queryEven, mask);
        Mul(dot1, s2, queryEven, mask);
        MulAddDst(dot0, s1, queryOdd, mask);
        MulAddDst(dot1, s3, queryOdd, mask);
        ReduceSum(out0, dot0, mask);
        ReduceSum(out1, dot1, mask);
        StoreVFirstRows(target, row, s0, s1, s2, s3, mask, mask16);
        Cast<bfloat16_t, float, TO_B16_ZERO>(packedOut, out0, first);
        StoreUnAlign(outputAddress, packedOut, outputStore, 1);
        Cast<bfloat16_t, float, TO_B16_ZERO>(packedOut, out1, first);
        StoreUnAlign(outputAddress, packedOut, outputStore, 1);
    }
}

template <typename State>
__simd_callee__ inline void RecurrentKdaVFirstTwoPass(
    __ubuf__ State* target, __ubuf__ State* source, __ubuf__ float* scratch,
    __ubuf__ bfloat16_t* valueRaw, __ubuf__ bfloat16_t* output,
    RegTensor<float>& queryEven, RegTensor<float>& queryOdd,
    RegTensor<float>& keyEven, RegTensor<float>& keyOdd,
    RegTensor<float>& decayEven, RegTensor<float>& decayOdd,
    RegTensor<float>& updateKeyEven, RegTensor<float>& updateKeyOdd,
    MaskReg& mask, MaskReg& mask16, MaskReg& first)
{
    RegTensor<float> decayKeyEven, decayKeyOdd;
    Mul(decayKeyEven, decayEven, keyEven, mask);
    Mul(decayKeyOdd, decayOdd, keyOdd, mask);
    UnalignRegForStore predictionStore;
    __ubuf__ float* predictionAddress = scratch;
    #pragma unroll 1
    for (uint16_t row = 0; row < DIM; row += 2) {
        RegTensor<float> s0, s1, s2, s3, dot0, dot1, prediction0, prediction1;
        LoadPair(s0, s1, source + row * DIM, mask16);
        LoadPair(s2, s3, source + (row + 1) * DIM, mask16);
        Mul(dot0, s0, decayKeyEven, mask);
        Mul(dot1, s2, decayKeyEven, mask);
        MulAddDst(dot0, s1, decayKeyOdd, mask);
        MulAddDst(dot1, s3, decayKeyOdd, mask);
        ReduceSum(prediction0, dot0, mask);
        ReduceSum(prediction1, dot1, mask);
        StoreUnAlign(predictionAddress, prediction0, predictionStore, 1);
        StoreUnAlign(predictionAddress, prediction1, predictionStore, 1);
    }
    StoreUnAlignPost(predictionAddress, predictionStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    RegTensor<float> predictionEven, predictionOdd, residualEven, residualOdd;
    LoadPair(predictionEven, predictionOdd, scratch, mask16);
    LoadPair(residualEven, residualOdd, valueRaw, mask16);
    Sub(residualEven, residualEven, predictionEven, mask);
    Sub(residualOdd, residualOdd, predictionOdd, mask);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(scratch, residualEven, residualOdd, mask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    UnalignRegForStore outputStore;
    __ubuf__ bfloat16_t* outputAddress = output;
    RecurrentKdaVFirstUpdateRows<State>(target, source, outputAddress, scratch,
        queryEven, queryOdd, decayEven, decayOdd, updateKeyEven, updateKeyOdd,
        mask, mask16, first, outputStore);
    StoreUnAlignPost(outputAddress, outputStore, 0);
}

// K-first lanes represent consecutive V elements; coefficient registers hold
// consecutive K elements. State remains tightly packed [K,V] in all banks.
template <typename T>
__simd_callee__ inline void LoadContiguous(RegTensor<float>& values, __ubuf__ T* source, MaskReg& mask)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadAlign(values, source);
    } else {
        RegTensor<T> packed;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(packed, source);
        Cast<float, T, TO_FLOAT_ZERO>(values, packed, mask);
    }
}

template <typename T>
__simd_callee__ inline void StoreContiguous(__ubuf__ T* target, RegTensor<float>& values, MaskReg& mask)
{
    if constexpr (std::is_same<T, float>::value) {
        StoreAlign(target, values, mask);
    } else {
        RegTensor<T> packed;
        Cast<T, float, TO_B16_WORD>(packed, values, mask);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(target, packed, mask);
    }
}

template <typename State>
__simd_callee__ inline void PredictKFirstHalf(RegTensor<float>& predictionLow,
    RegTensor<float>& predictionHigh, __ubuf__ State* source, __ubuf__ float* decayKey,
    MaskReg& mask)
{
    #pragma unroll 1
    for (uint16_t keyRow = 0; keyRow < DIM / 2; ++keyRow) {
        RegTensor<float> coefficient, stateLow, stateHigh;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(coefficient, decayKey + keyRow);
        LoadContiguous(stateLow, source + keyRow * DIM, mask);
        LoadContiguous(stateHigh, source + keyRow * DIM + DIM / 2, mask);
        MulAddDst(predictionLow, stateLow, coefficient, mask);
        MulAddDst(predictionHigh, stateHigh, coefficient, mask);
    }
}

template <typename State>
__simd_callee__ inline void UpdateKFirstHalf(__ubuf__ State* target, __ubuf__ State* source,
    RegTensor<float>& deltaLow, RegTensor<float>& deltaHigh,
    RegTensor<float>& outputLow, RegTensor<float>& outputHigh,
    RegTensor<float>& query, RegTensor<float>& key, RegTensor<float>& decay, MaskReg& mask)
{
    RegTensor<uint32_t> coefficientLane;
    Duplicate(coefficientLane, uint32_t(0));
    #pragma unroll 1
    for (uint16_t keyRow = 0; keyRow < DIM / 2; ++keyRow) {
        RegTensor<float> queryValue, keyValue, decayValue, stateLow, stateHigh;
        Gather(decayValue, decay, coefficientLane);
        Gather(keyValue, key, coefficientLane);
        Gather(queryValue, query, coefficientLane);
        LoadContiguous(stateLow, source + keyRow * DIM, mask);
        LoadContiguous(stateHigh, source + keyRow * DIM + DIM / 2, mask);
        Mul(stateLow, stateLow, decayValue, mask);
        Mul(stateHigh, stateHigh, decayValue, mask);
        MulAddDst(stateLow, deltaLow, keyValue, mask);
        MulAddDst(stateHigh, deltaHigh, keyValue, mask);
        // Use the FP32 update for out, before rounding the next BF16 State.
        MulAddDst(outputLow, stateLow, queryValue, mask);
        MulAddDst(outputHigh, stateHigh, queryValue, mask);
        StoreContiguous(target + keyRow * DIM, stateLow, mask);
        StoreContiguous(target + keyRow * DIM + DIM / 2, stateHigh, mask);
        Adds(coefficientLane, coefficientLane, uint32_t(1), mask);
    }
}

template <typename State>
__simd_callee__ inline void RecurrentKdaKFirstTwoPass(__ubuf__ State* target, __ubuf__ State* source,
    __ubuf__ float* coefficientScratch, __ubuf__ bfloat16_t* value, __ubuf__ bfloat16_t* output,
    RegTensor<float>& queryLow, RegTensor<float>& queryHigh,
    RegTensor<float>& keyLow, RegTensor<float>& keyHigh,
    RegTensor<float>& decayLow, RegTensor<float>& decayHigh, RegTensor<float>& beta, MaskReg& mask)
{
    RegTensor<float> predictionLow, predictionHigh, decayKeyLow, decayKeyHigh;
    Duplicate(predictionLow, 0.0f);
    Duplicate(predictionHigh, 0.0f);
    Mul(decayKeyLow, decayLow, keyLow, mask);
    Mul(decayKeyHigh, decayHigh, keyHigh, mask);
    StoreAlign(coefficientScratch, decayKeyLow, mask);
    StoreAlign(coefficientScratch + DIM / 2, decayKeyHigh, mask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    PredictKFirstHalf(predictionLow, predictionHigh, source, coefficientScratch, mask);
    PredictKFirstHalf(predictionLow, predictionHigh, source + (DIM / 2) * DIM,
        coefficientScratch + DIM / 2, mask);

    RegTensor<float> valueLow, valueHigh;
    LoadContiguous(valueLow, value, mask);
    LoadContiguous(valueHigh, value + DIM / 2, mask);
    Sub(predictionLow, valueLow, predictionLow, mask);
    Sub(predictionHigh, valueHigh, predictionHigh, mask);
    Mul(predictionLow, predictionLow, beta, mask);
    Mul(predictionHigh, predictionHigh, beta, mask);

    RegTensor<float> outputLow, outputHigh;
    Duplicate(outputLow, 0.0f);
    Duplicate(outputHigh, 0.0f);
    UpdateKFirstHalf(target, source, predictionLow, predictionHigh, outputLow, outputHigh,
        queryLow, keyLow, decayLow, mask);
    UpdateKFirstHalf(target + (DIM / 2) * DIM, source + (DIM / 2) * DIM,
        predictionLow, predictionHigh, outputLow, outputHigh, queryHigh, keyHigh, decayHigh, mask);
    StoreContiguous(output, outputLow, mask);
    StoreContiguous(output + DIM / 2, outputHigh, mask);
}

template <typename State, typename Gate, typename Beta, bool V_FIRST, bool FUSED>
__simd_vf__ inline void RecurrentKdaTokenVf(__ubuf__ State* target, __ubuf__ State* source,
    __ubuf__ bfloat16_t* keyRaw, __ubuf__ bfloat16_t* queryRaw, __ubuf__ bfloat16_t* valueRaw,
    __ubuf__ Gate* gateRaw, __ubuf__ Beta* betaRaw, __ubuf__ float* biasRaw,
    __ubuf__ bfloat16_t* output, float scale, float aLog, uint32_t flags, float lowerBound,
    __ubuf__ float* predictionScratch)
{
    const bool normalize = FUSED || (flags & RecurrentKda::RKDA_VF_NORM);
    const bool rawGate = FUSED || (flags & RecurrentKda::RKDA_VF_RAW_GATE);
    const bool sigmoid = FUSED || (flags & RecurrentKda::RKDA_VF_SIGMOID);
    const bool bias = FUSED || (flags & RecurrentKda::RKDA_VF_BIAS);
    const bool negative = !FUSED && (flags & RecurrentKda::RKDA_VF_NEGATIVE);
    const bool safe = !FUSED && (flags & RecurrentKda::RKDA_VF_SAFE);
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg mask16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg first = CreateMask<float, MaskPattern::VL1>();
    RegTensor<float> q0, q1, k0, k1, decay0, decay1, beta, expA, work0, work1, one;
    RegTensor<float> updateKey0, updateKey1;
    if constexpr (V_FIRST) {
        LoadPair(k0, k1, keyRaw, mask16);
        LoadPair(q0, q1, queryRaw, mask16);
    } else {
        LoadContiguous(k0, keyRaw, mask);
        LoadContiguous(k1, keyRaw + DIM / 2, mask);
        LoadContiguous(q0, queryRaw, mask);
        LoadContiguous(q1, queryRaw + DIM / 2, mask);
    }
    if (normalize) { NormalizePair(q0, q1, mask); NormalizePair(k0, k1, mask); }
    Muls(q0, q0, scale, mask);
    Muls(q1, q1, scale, mask);
    if constexpr (V_FIRST) {
        LoadPair(decay0, decay1, gateRaw, mask16);
    } else {
        LoadContiguous(decay0, gateRaw, mask);
        LoadContiguous(decay1, gateRaw + DIM / 2, mask);
    }
    if constexpr (std::is_same<Beta, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(beta, betaRaw);
    } else {
        RegTensor<Beta> packed;
        LoadAlign<Beta, LoadDist::DIST_BRC_B16>(packed, betaRaw);
        Cast<float, Beta, TO_FLOAT_ZERO>(beta, packed, mask16);
    }
    if (sigmoid) {
        Duplicate(one, 1.0f, mask);
        Muls(beta, beta, -1.0f, mask);
        Exp(beta, beta, mask);
        Adds(beta, beta, 1.0f, mask);
        Div(beta, one, beta, mask);
        if (negative) Muls(beta, beta, 2.0f, mask);
    }
    if (rawGate) {
        Duplicate(expA, aLog, mask);
        Exp(expA, expA, mask);
        if (bias) {
            if constexpr (V_FIRST) {
                LoadPair(work0, work1, biasRaw, mask16);
            } else {
                LoadContiguous(work0, biasRaw, mask);
                LoadContiguous(work1, biasRaw + DIM / 2, mask);
            }
            Add(decay0, decay0, work0, mask);
            Add(decay1, decay1, work1, mask);
        }
    }
    Decay(decay0, expA, rawGate, safe, lowerBound, mask);
    Decay(decay1, expA, rawGate, safe, lowerBound, mask);
    if constexpr (V_FIRST && (std::is_same<State, float>::value ||
                              std::is_same<State, bfloat16_t>::value)) {
        Mul(updateKey0, k0, beta, mask);
        Mul(updateKey1, k1, beta, mask);
        RecurrentKdaVFirstTwoPass<State>(target, source, predictionScratch, valueRaw, output,
            q0, q1, k0, k1, decay0, decay1, updateKey0, updateKey1, mask, mask16, first);
    } else {
        RecurrentKdaKFirstTwoPass(target, source, predictionScratch, valueRaw, output,
            q0, q1, k0, k1, decay0, decay1, beta, mask);
    }
}

// The class below is the only task scheduler for the stand-pr A5 path.
template <typename State, typename Gate, typename Beta, bool V_FIRST, bool FUSED>
class RecurrentKdaEngine {
public:
    __aicore__ inline explicit RecurrentKdaEngine(const RecurrentKdaTilingData* tiling) : tiling_(tiling) {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gate, GM_ADDR beta,
        GM_ADDR state, GM_ADDR cuSeqlens, GM_ADDR indices, GM_ADDR aLog, GM_ADDR bias, GM_ADDR accepted,
        GM_ADDR output, GM_ADDR stateOutput)
    {
        queryGm_.SetGlobalBuffer((__gm__ bfloat16_t*)query);
        keyGm_.SetGlobalBuffer((__gm__ bfloat16_t*)key);
        valueGm_.SetGlobalBuffer((__gm__ bfloat16_t*)value);
        gateGm_.SetGlobalBuffer((__gm__ Gate*)gate);
        betaGm_.SetGlobalBuffer((__gm__ Beta*)beta);
        stateGm_.SetGlobalBuffer((__gm__ State*)state);
        stateOutGm_.SetGlobalBuffer((__gm__ State*)stateOutput);
        outputGm_.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        aLogGm_.SetGlobalBuffer((__gm__ float*)aLog);
        biasGm_.SetGlobalBuffer((__gm__ float*)bias);
        cuAddress_ = cuSeqlens;
        indicesAddress_ = indices;
        acceptedAddress_ = accepted;
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        if (!ValidateSequences()) return;
        const uint64_t taskCount = uint64_t(tiling_->b) * tiling_->nv;
        uint32_t currentBank = 0;
        uint32_t tileIndex = 0;
        uint32_t parameterBuffer = 0;
        const bool usesBias = tiling_->hasDtBias && tiling_->useGateInKernel;
        bool stateReadyEvent = true;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for (uint64_t task = GetBlockIdx(); task < taskCount; task += GetBlockNum()) {
            const uint64_t request = task / tiling_->nv;
            const uint64_t head = task % tiling_->nv;
            UpdateSequence(request);
            if (sequenceBegin_ == sequenceEnd_) continue;
            if (!ValidateSlots(request)) break;
            const uint64_t initialSlot = InitialSlot(request);
            const uint64_t keyHead = head / (tiling_->nv / tiling_->nk);
            const int64_t taskTokens = sequenceEnd_ - sequenceBegin_;
            const bool hasFollowingTask = task + GetBlockNum() < taskCount;
            float aLog = 0.0f;
            // Independent V slices share one task's metadata, parameters and bank protocol.
            for (uint32_t valueOffset = 0; valueOffset < tiling_->dv; valueOffset += DIM) {
                const bool hasFollowingWork = valueOffset + DIM < tiling_->dv || hasFollowingTask;
                currentBank = (currentBank + 1) % STATE_BANKS;
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                stateReadyEvent = false;
                Mutex::Lock<PIPE_MTE2>(STATE_MUTEX[currentBank]);
                CopyStateIn(currentBank, initialSlot, head, valueOffset);
                Mutex::Unlock<PIPE_MTE2>(STATE_MUTEX[currentBank]);
                // Alternate only for executed tasks, including across empty requests.
                // A second slot lets the next task's input DMA pass the current VF.
                if (usesBias && valueOffset == 0) {
                    Mutex::Lock<PIPE_MTE2>(PARAMETER_MUTEX[parameterBuffer]);
                    DataCopy(biasUb_[parameterBuffer], biasGm_[head * DIM], DIM);
                    Mutex::Unlock<PIPE_MTE2>(PARAMETER_MUTEX[parameterBuffer]);
                }
                if (valueOffset == 0) aLog = tiling_->useGateInKernel ? aLogGm_.GetValue(head) : 0.0f;
                if (hasFollowingWork && taskTokens == 1) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    stateReadyEvent = true;
                }
                if (usesBias && valueOffset == 0) Mutex::Lock<PIPE_V>(PARAMETER_MUTEX[parameterBuffer]);
                uint64_t pendingStateSlots[STATE_BANKS];
                if (tiling_->hasSsmStateIndices) {
                    for (uint32_t bank = 0; bank < STATE_BANKS; ++bank) {
                        pendingStateSlots[bank] = INVALID_STATE_SLOT;
                    }
                }
                for (int64_t start = sequenceBegin_; start < sequenceEnd_; start += TOKEN_TILE) {
                    const uint16_t rows = static_cast<uint16_t>((sequenceEnd_ - start < TOKEN_TILE ? sequenceEnd_ - start : TOKEN_TILE));
                    const uint32_t buffer = tileIndex++ % INPUT_BUFFERS;
                    CopyInputTile(buffer, start, rows, keyHead, head, valueOffset);
                    Mutex::Lock<PIPE_V>(INPUT_MUTEX[buffer]);
                    Mutex::Lock<PIPE_V>(OUTPUT_MUTEX[buffer]);
                    for (uint16_t token = 0; token < rows; ++token) {
                        const uint32_t nextBank = (currentBank + 1) % STATE_BANKS;
                        Mutex::Lock<PIPE_V>(STATE_MUTEX[nextBank]);
                        ComputeToken(buffer, token, currentBank, nextBank, aLog, parameterBuffer);
                        Mutex::Unlock<PIPE_V>(STATE_MUTEX[nextBank]);
                        const int64_t position = start + token;
                        // The next work item (V slice or task) may reuse this bank after its last read.
                        if (hasFollowingWork && position + 2 == sequenceEnd_) {
                            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                            stateReadyEvent = true;
                        }
                        if ((tiling_->inplaceFinalState || tiling_->outputFinalState) &&
                            (tiling_->hasSsmStateIndices || position + 1 == sequenceEnd_)) {
                            const uint64_t slot = StateSlot(request, position);
                            if (tiling_->hasSsmStateIndices) {
                                OrderStateSnapshotWrites(nextBank, slot, pendingStateSlots);
                            }
                            Mutex::Lock<PIPE_MTE3>(STATE_MUTEX[nextBank]);
                            CopyStateOut(nextBank, slot, head, valueOffset);
                            Mutex::Unlock<PIPE_MTE3>(STATE_MUTEX[nextBank]);
                        }
                        currentBank = nextBank;
                    }
                    Mutex::Unlock<PIPE_V>(INPUT_MUTEX[buffer]);
                    Mutex::Unlock<PIPE_V>(OUTPUT_MUTEX[buffer]);
                    const uint64_t outOffset = (uint64_t(start) * tiling_->nv + head) * tiling_->dv + valueOffset;
                    const DataCopyExtParams outputParams{rows, DIM * sizeof(bfloat16_t), 0,
                        uint32_t((uint64_t(tiling_->nv) * tiling_->dv - DIM) * sizeof(bfloat16_t)), 0};
                    Mutex::Lock<PIPE_MTE3>(OUTPUT_MUTEX[buffer]);
                    DataCopyPad(outputGm_[outOffset], outputTile_[buffer], outputParams);
                    Mutex::Unlock<PIPE_MTE3>(OUTPUT_MUTEX[buffer]);
                }
            }
            if (usesBias) Mutex::Unlock<PIPE_V>(PARAMETER_MUTEX[parameterBuffer]);
            parameterBuffer = (parameterBuffer + 1) % PARAMETER_BUFFERS;
        }
        // Close the V->MTE2 handshake even if trailing tasks were empty/invalid.
        if (stateReadyEvent) WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        // Drain both output slots on V: the final tile output follows all
        // preceding State DMA on MTE3. Unused slots are already unlocked.
        // Each input DMA was already consumed under INPUT_MUTEX before its VF.
        Mutex::Lock<PIPE_V>(OUTPUT_MUTEX[0]);
        Mutex::Unlock<PIPE_V>(OUTPUT_MUTEX[0]);
        Mutex::Lock<PIPE_V>(OUTPUT_MUTEX[1]);
        Mutex::Unlock<PIPE_V>(OUTPUT_MUTEX[1]);
    }

private:
    __aicore__ inline void InitBuffers()
    {
        uint32_t offset = 0;
        for (uint32_t bank = 0; bank < STATE_BANKS; ++bank) {
            stateBanks_[bank] = LocalTensor<State>(TPosition::VECCALC, offset, DIM * DIM);
            offset += STATE_BYTES;
        }
        for (uint32_t buffer = 0; buffer < INPUT_BUFFERS; ++buffer) {
            queryTile_[buffer] = LocalTensor<bfloat16_t>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TOKEN_TILE * DIM * sizeof(bfloat16_t);
            keyTile_[buffer] = LocalTensor<bfloat16_t>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TOKEN_TILE * DIM * sizeof(bfloat16_t);
            valueTile_[buffer] = LocalTensor<bfloat16_t>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TOKEN_TILE * DIM * sizeof(bfloat16_t);
            gateTile_[buffer] = LocalTensor<Gate>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TILE_VECTOR_BYTES;
            betaTile_[buffer] = LocalTensor<Beta>(TPosition::VECCALC, offset, TILE_SCALAR_BYTES / sizeof(Beta));
            offset += TILE_SCALAR_BYTES;
            outputTile_[buffer] = LocalTensor<bfloat16_t>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TOKEN_TILE * DIM * sizeof(bfloat16_t);
        }
        for (uint32_t buffer = 0; buffer < PARAMETER_BUFFERS; ++buffer) {
            biasUb_[buffer] = LocalTensor<float>(TPosition::VECCALC, offset, DIM);
            offset += DIM * sizeof(float);
        }
        predictionScratchUb_ = LocalTensor<float>(TPosition::VECCALC, offset, DIM);
    }

    __aicore__ inline int64_t Metadata(GM_ADDR address, uint32_t dtype, uint64_t element) const
    {
        if (dtype == 0) {
            GlobalTensor<int32_t> tensor;
            tensor.SetGlobalBuffer((__gm__ int32_t*)address);
            return tensor.GetValue(element);
        }
        GlobalTensor<int64_t> tensor;
        tensor.SetGlobalBuffer((__gm__ int64_t*)address);
        return tensor.GetValue(element);
    }

    __aicore__ inline bool ValidateSequences() const
    {
        if (!tiling_->hasCuSeqlens) return true;
        int64_t previous = Metadata(cuAddress_, tiling_->cuSeqlensDtype, 0);
        if (previous != 0) return false;
        for (uint32_t request = 0; request < tiling_->b; ++request) {
            const int64_t end = Metadata(cuAddress_, tiling_->cuSeqlensDtype, request + 1);
            if (end < previous || uint64_t(end) > tiling_->t ||
                (tiling_->hasSsmStateIndices && tiling_->ssmStateStride &&
                 uint64_t(end - previous) > tiling_->ssmStateStride)) return false;
            previous = end;
        }
        return true;
    }

    __aicore__ inline void UpdateSequence(uint64_t request)
    {
        if (cachedRequest_ == request) return;
        cachedRequest_ = request;
        sequenceBegin_ = tiling_->hasCuSeqlens ? Metadata(cuAddress_, tiling_->cuSeqlensDtype, request) :
            int64_t(request * tiling_->seqLen);
        sequenceEnd_ = tiling_->hasCuSeqlens ? Metadata(cuAddress_, tiling_->cuSeqlensDtype, request + 1) :
            int64_t((request + 1) * tiling_->seqLen);
    }

    __aicore__ inline uint64_t StateSlot(uint64_t request, int64_t token) const
    {
        if (!tiling_->hasSsmStateIndices) return request;
        const uint64_t offset = tiling_->ssmStateStride ? request * tiling_->ssmStateStride + token - sequenceBegin_ : token;
        return uint64_t(Metadata(indicesAddress_, tiling_->ssmStateIndicesDtype, offset));
    }

    __aicore__ inline bool ValidateSlots(uint64_t request) const
    {
        const int64_t sequenceLength = sequenceEnd_ - sequenceBegin_;
        if (tiling_->hasAcceptedTokens) {
            const int64_t accepted = Metadata(acceptedAddress_, tiling_->acceptedTokensDtype, request);
            // A two-dimensional speculative index table describes a state
            // window, which may be longer than the current query window.  The
            // accepted prefix selects the initial State from that window, so
            // it is bounded by max_step (ssmStateStride), not by the number of
            // tokens currently being queried.  A packed one-dimensional table
            // has no such extra window and remains bounded by sequenceLength.
            const uint64_t acceptedLimit = tiling_->ssmStateStride != 0 ?
                static_cast<uint64_t>(tiling_->ssmStateStride) : static_cast<uint64_t>(sequenceLength);
            if (accepted < 1 || static_cast<uint64_t>(accepted) > acceptedLimit) return false;

            // The accepted prefix can point beyond the current query window;
            // validate that InitialSlot itself is still inside the State pool.
            const int64_t initialToken = sequenceBegin_ + accepted - 1;
            if (StateSlot(request, initialToken) >= tiling_->sBlockNum) return false;
        }
        if (!tiling_->hasSsmStateIndices) return request < tiling_->sBlockNum;
        for (int64_t token = sequenceBegin_; token < sequenceEnd_; ++token)
            if (StateSlot(request, token) >= tiling_->sBlockNum) return false;
        return true;
    }

    __aicore__ inline uint64_t InitialSlot(uint64_t request) const
    {
        int64_t token = sequenceBegin_;
        if (tiling_->hasAcceptedTokens) token += Metadata(acceptedAddress_, tiling_->acceptedTokensDtype, request) - 1;
        return StateSlot(request, token);
    }

    __aicore__ inline void OrderStateSnapshotWrites(
        uint32_t sourceBank, uint64_t slot, uint64_t (&pendingStateSlots)[STATE_BANKS])
    {
        // The token's V lock already recycled this bank after its old snapshot.
        pendingStateSlots[sourceBank] = INVALID_STATE_SLOT;
        for (uint32_t bank = 0; bank < STATE_BANKS; ++bank) {
            if (pendingStateSlots[bank] == slot) {
                // Bank Mutexes protect UB reuse, not overlapping GM destinations.
                // A same-pipe Mutex pair cannot order two MTE3 writes either.
                PipeBarrier<PIPE_MTE3>();
                for (uint32_t completedBank = 0; completedBank < STATE_BANKS; ++completedBank) {
                    pendingStateSlots[completedBank] = INVALID_STATE_SLOT;
                }
                break;
            }
        }
        pendingStateSlots[sourceBank] = slot;
    }

    __aicore__ inline void CopyStateIn(uint32_t bank, uint64_t slot, uint64_t head, uint32_t valueOffset)
    {
        const uint64_t offset = slot * tiling_->stateInStride0 + head * tiling_->stateInStride1 +
            valueOffset * (V_FIRST ? tiling_->stateInStride2 : tiling_->stateInStride3);
        if (tiling_->stateInStride2 == DIM) {
            DataCopy(stateBanks_[bank], stateGm_[offset], DIM * DIM);
        } else {
            const DataCopyExtParams params{DIM, DIM * sizeof(State),
                uint32_t((tiling_->stateInStride2 - DIM) * sizeof(State)), 0, 0};
            DataCopyPad(stateBanks_[bank], stateGm_[offset], params, DataCopyPadExtParams<State>{false, 0, 0, State(0)});
        }
    }

    __aicore__ inline void CopyStateOut(uint32_t bank, uint64_t slot, uint64_t head, uint32_t valueOffset)
    {
        const uint64_t offset = slot * tiling_->stateOutStride0 + head * tiling_->stateOutStride1 +
            valueOffset * (V_FIRST ? tiling_->stateOutStride2 : tiling_->stateOutStride3);
        if (tiling_->stateOutStride2 == DIM) {
            DataCopy(stateOutGm_[offset], stateBanks_[bank], DIM * DIM);
        } else {
            const DataCopyExtParams params{DIM, DIM * sizeof(State), 0,
                uint32_t((tiling_->stateOutStride2 - DIM) * sizeof(State)), 0};
            DataCopyPad(stateOutGm_[offset], stateBanks_[bank], params);
        }
    }

    __aicore__ inline void CopyInputTile(uint32_t buffer, int64_t start, uint16_t rows, uint64_t keyHead, uint64_t head,
        uint32_t valueOffset)
    {
        const uint64_t queryOffset = uint64_t(start) * tiling_->queryTokenStride + keyHead * tiling_->queryHeadStride;
        const uint64_t keyOffset = uint64_t(start) * tiling_->keyTokenStride + keyHead * tiling_->keyHeadStride;
        const uint64_t valueGmOffset = uint64_t(start) * tiling_->valueTokenStride + head * tiling_->valueHeadStride + valueOffset;
        const uint64_t gateOffset = (uint64_t(start) * tiling_->nv + head) * DIM;
        const uint64_t betaOffset = uint64_t(start) * tiling_->nv + head;
        const DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, bfloat16_t(0)};
        const DataCopyExtParams queryParams{rows, DIM * sizeof(bfloat16_t), uint32_t((tiling_->queryTokenStride - DIM) * sizeof(bfloat16_t)), 0, 0};
        const DataCopyExtParams keyParams{rows, DIM * sizeof(bfloat16_t), uint32_t((tiling_->keyTokenStride - DIM) * sizeof(bfloat16_t)), 0, 0};
        const DataCopyExtParams valueParams{rows, DIM * sizeof(bfloat16_t), uint32_t((tiling_->valueTokenStride - DIM) * sizeof(bfloat16_t)), 0, 0};
        const DataCopyExtParams gateParams{rows, DIM * sizeof(Gate), uint32_t((tiling_->nv - 1) * DIM * sizeof(Gate)), 0, 0};
        const DataCopyExtParams betaParams{rows, sizeof(Beta), uint32_t((tiling_->nv - 1) * sizeof(Beta)), 0, 0};
        Mutex::Lock<PIPE_MTE2>(INPUT_MUTEX[buffer]);
        DataCopyPad(queryTile_[buffer], queryGm_[queryOffset], queryParams, pad);
        DataCopyPad(keyTile_[buffer], keyGm_[keyOffset], keyParams, pad);
        DataCopyPad(valueTile_[buffer], valueGm_[valueGmOffset], valueParams, pad);
        DataCopyPad(gateTile_[buffer], gateGm_[gateOffset], gateParams, DataCopyPadExtParams<Gate>{false, 0, 0, Gate(0)});
        DataCopyPad(betaTile_[buffer], betaGm_[betaOffset], betaParams, DataCopyPadExtParams<Beta>{false, 0, 0, Beta(0)});
        Mutex::Unlock<PIPE_MTE2>(INPUT_MUTEX[buffer]);
    }

    __aicore__ inline void ComputeToken(uint32_t buffer, uint16_t token, uint32_t source, uint32_t target,
        float aLog, uint32_t parameterBuffer)
    {
        __ubuf__ float* predictionScratch = (__ubuf__ float*)predictionScratchUb_.GetPhyAddr();
        RecurrentKdaTokenVf<State, Gate, Beta, V_FIRST, FUSED>(
            (__ubuf__ State*)stateBanks_[target].GetPhyAddr(), (__ubuf__ State*)stateBanks_[source].GetPhyAddr(),
            (__ubuf__ bfloat16_t*)keyTile_[buffer][token * DIM].GetPhyAddr(),
            (__ubuf__ bfloat16_t*)queryTile_[buffer][token * DIM].GetPhyAddr(),
            (__ubuf__ bfloat16_t*)valueTile_[buffer][token * DIM].GetPhyAddr(),
            (__ubuf__ Gate*)gateTile_[buffer][token * DIM].GetPhyAddr(),
            (__ubuf__ Beta*)betaTile_[buffer][token * (32 / sizeof(Beta))].GetPhyAddr(),
            (__ubuf__ float*)biasUb_[parameterBuffer].GetPhyAddr(),
            (__ubuf__ bfloat16_t*)outputTile_[buffer][token * DIM].GetPhyAddr(),
            tiling_->scale, aLog, tiling_->a5VfFlags, tiling_->lowerBound, predictionScratch);
    }

    const RecurrentKdaTilingData* tiling_;
    GlobalTensor<bfloat16_t> queryGm_, keyGm_, valueGm_, outputGm_;
    GlobalTensor<State> stateGm_, stateOutGm_;
    GlobalTensor<Gate> gateGm_;
    GlobalTensor<Beta> betaGm_;
    GlobalTensor<float> aLogGm_, biasGm_;
    GM_ADDR cuAddress_;
    GM_ADDR indicesAddress_;
    GM_ADDR acceptedAddress_;
    LocalTensor<State> stateBanks_[STATE_BANKS];
    LocalTensor<bfloat16_t> queryTile_[INPUT_BUFFERS], keyTile_[INPUT_BUFFERS], valueTile_[INPUT_BUFFERS], outputTile_[INPUT_BUFFERS];
    LocalTensor<Gate> gateTile_[INPUT_BUFFERS];
    LocalTensor<Beta> betaTile_[INPUT_BUFFERS];
    LocalTensor<float> biasUb_[PARAMETER_BUFFERS];
    LocalTensor<float> predictionScratchUb_;
    uint64_t cachedRequest_ = uint64_t(-1);
    int64_t sequenceBegin_ = 0, sequenceEnd_ = 0;
};
} // namespace RecurrentKdaStand
