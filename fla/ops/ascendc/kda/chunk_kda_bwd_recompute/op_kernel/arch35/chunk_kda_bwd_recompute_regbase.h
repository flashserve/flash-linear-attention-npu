/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH35_REGBASE_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH35_REGBASE_H

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310

#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif

#include "chunk_kda_bwd_recompute_common.h"

namespace KdaBwdRecomputeArch35 {

constexpr float kLn2 = 0.69314718055994530942f;
constexpr float kExpInputMax = 80.0f * kLn2;
constexpr float kExpInputMin = -80.0f * kLn2;
// fp16 Exp saturates beyond ~exp(±11). Pass-2 qg/kg write bf16 so this matches
// the output dtype; gk cumsum stays fp32 Exp.
constexpr float kHalfExpInputMax = 11.0f;
constexpr float kHalfExpInputMin = -11.0f;

template <bool HAS_BIAS, bool HAS_ALOG>
static __simd_vf__ inline void AccumulateSafeGateChunk128Regbase(
    __ubuf__ float *input, __ubuf__ float *bias, __ubuf__ float *acc,
    uint16_t rows, __ubuf__ float *aLog, float lowerBound)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t ROW_ELEMENTS = 2 * FLOAT_ELEMENTS_PER_REG;

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> accZeroReg;
    RegTensor<float> accOneReg;
    RegTensor<float> oneZeroReg;
    RegTensor<float> oneOneReg;
    RegTensor<float> biasZeroReg;
    RegTensor<float> biasOneReg;
    RegTensor<float> expAReg;
    LoadAlign<float, LoadDist::DIST_NORM>(accZeroReg, acc);
    LoadAlign<float, LoadDist::DIST_NORM>(accOneReg, acc + FLOAT_ELEMENTS_PER_REG);
    Duplicate(oneZeroReg, 1.0f, floatMask);
    Duplicate(oneOneReg, 1.0f, floatMask);
    if constexpr (HAS_BIAS) {
        LoadAlign<float, LoadDist::DIST_NORM>(biasZeroReg, bias);
        LoadAlign<float, LoadDist::DIST_NORM>(biasOneReg, bias + FLOAT_ELEMENTS_PER_REG);
    }
    if constexpr (HAS_ALOG) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(expAReg, aLog);
        Exp(expAReg, expAReg, floatMask);
        Muls(expAReg, expAReg, -1.0f, floatMask);
    } else {
        Duplicate(expAReg, -1.0f, floatMask);
    }

    RegTensor<float> gateZeroReg;
    RegTensor<float> gateOneReg;
    RegTensor<float> sigmoidZeroReg;
    RegTensor<float> sigmoidOneReg;
    const float gateScale = lowerBound * KDA_BWD_RECOMPUTE_RCP_LN2;
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
        LoadAlign<float, LoadDist::DIST_NORM>(gateZeroReg, input + rowOffset);
        LoadAlign<float, LoadDist::DIST_NORM>(gateOneReg, input + rowOffset + FLOAT_ELEMENTS_PER_REG);
        if constexpr (HAS_BIAS) {
            Add(gateZeroReg, gateZeroReg, biasZeroReg, floatMask);
            Add(gateOneReg, gateOneReg, biasOneReg, floatMask);
        }
        Mul(gateZeroReg, gateZeroReg, expAReg, floatMask);
        Mul(gateOneReg, gateOneReg, expAReg, floatMask);
        Exp(gateZeroReg, gateZeroReg, floatMask);
        Exp(gateOneReg, gateOneReg, floatMask);
        Adds(gateZeroReg, gateZeroReg, 1.0f, floatMask);
        Adds(gateOneReg, gateOneReg, 1.0f, floatMask);
        Div(sigmoidZeroReg, oneZeroReg, gateZeroReg, floatMask);
        Div(sigmoidOneReg, oneOneReg, gateOneReg, floatMask);
        Muls(sigmoidZeroReg, sigmoidZeroReg, gateScale, floatMask);
        Muls(sigmoidOneReg, sigmoidOneReg, gateScale, floatMask);
        Add(accZeroReg, accZeroReg, sigmoidZeroReg, floatMask);
        Add(accOneReg, accOneReg, sigmoidOneReg, floatMask);
        StoreAlign(input + rowOffset, accZeroReg, floatMask);
        StoreAlign(input + rowOffset + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
    }
    StoreAlign(acc, accZeroReg, floatMask);
    StoreAlign(acc + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
}

// K-contiguous pair: regs hold [0:63] and [64:127], matching pass-1 gk StoreAlign.
// DINTLV / CastHalf2Float ZERO+ONE deinterleave even/odd and scramble GM pairing.
template <typename InputT>
__simd_callee__ inline void LoadGateRegbasePair(
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    __ubuf__ InputT *src,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::RegTensor<InputT> &inputReg)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    if constexpr (std::is_same<InputT, float>()) {
        LoadAlign<float, LoadDist::DIST_NORM>(zeroReg, src);
        LoadAlign<float, LoadDist::DIST_NORM>(oneReg, src + FLOAT_ELEMENTS_PER_REG);
        (void)inputReg;
        (void)inputMask;
        (void)floatMask;
    } else {
        LoadAlign<InputT, LoadDist::DIST_UNPACK_B16>(inputReg, src);
        Cast<float, InputT, ctHalf2Fp32Zero>(zeroReg, inputReg, floatMask);
        LoadAlign<InputT, LoadDist::DIST_UNPACK_B16>(inputReg, src + FLOAT_ELEMENTS_PER_REG);
        Cast<float, InputT, ctHalf2Fp32Zero>(oneReg, inputReg, floatMask);
        (void)inputMask;
    }
}

template <typename OutputT>
__simd_callee__ inline void StoreGateRegbasePair(
    __ubuf__ OutputT *dst,
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::MaskReg &floatMask,
    AscendC::MicroAPI::RegTensor<OutputT> &outputReg)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    if constexpr (std::is_same<OutputT, float>()) {
        StoreAlign(dst, zeroReg, floatMask);
        StoreAlign(dst + FLOAT_ELEMENTS_PER_REG, oneReg, floatMask);
        (void)inputMask;
        (void)outputReg;
    } else {
        Cast<OutputT, float, ctFp322HalfZero>(outputReg, zeroReg, floatMask);
        StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(dst, outputReg, floatMask);
        Cast<OutputT, float, ctFp322HalfZero>(outputReg, oneReg, floatMask);
        StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(dst + FLOAT_ELEMENTS_PER_REG, outputReg, floatMask);
        (void)inputMask;
    }
}

__simd_callee__ inline void ExpPairViaHalf(
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::RegTensor<half> &halfReg,
    AscendC::MicroAPI::MaskReg &floatMask,
    AscendC::MicroAPI::MaskReg &halfMask)
{
    using namespace AscendC::MicroAPI;
    CastFloat2Half<half>(halfReg, zeroReg, oneReg, floatMask);
    Exp(halfReg, halfReg, halfMask);
    CastHalf2Float<half>(zeroReg, oneReg, halfReg, halfMask);
}

template <typename InputT, typename OutputT, typename GateT, typename BetaT, bool HAS_BIAS, bool HAS_ALOG,
          bool kFixed64 = false>
static __simd_vf__ inline void FusedRecomputeChunk128Regbase(
    __ubuf__ float *gk, __ubuf__ GateT *gIn, __ubuf__ float *bias, __ubuf__ float *aLog, __ubuf__ BetaT *betaRow,
    __ubuf__ InputT *q, __ubuf__ InputT *k, __ubuf__ InputT *v,
    __ubuf__ OutputT *qg, __ubuf__ OutputT *kbg, __ubuf__ OutputT *kg, __ubuf__ OutputT *vb,
    uint16_t rows, float lowerBound)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t ROW_ELEMENTS = 2 * FLOAT_ELEMENTS_PER_REG;
    const uint16_t nRows = kFixed64 ? static_cast<uint16_t>(64) : rows;

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg inputMask = CreateMask<InputT, MaskPattern::ALL>();
    MaskReg betaMask = CreateMask<BetaT, MaskPattern::ALL>();
    RegTensor<float> lastZeroReg;
    RegTensor<float> lastOneReg;
    RegTensor<float> gateZeroReg;
    RegTensor<float> gateOneReg;
    {
        RegTensor<float> accZeroReg;
        RegTensor<float> accOneReg;
        RegTensor<float> oneZeroReg;
        RegTensor<float> oneOneReg;
        RegTensor<float> biasZeroReg;
        RegTensor<float> biasOneReg;
        RegTensor<float> expAReg;
        RegTensor<float> sigmoidZeroReg;
        RegTensor<float> sigmoidOneReg;
        RegTensor<GateT> gateRawReg;
        const float gateScale = lowerBound * KDA_BWD_RECOMPUTE_RCP_LN2;
        Duplicate(accZeroReg, 0.0f, floatMask);
        Duplicate(accOneReg, 0.0f, floatMask);
        Duplicate(oneZeroReg, gateScale, floatMask);
        Duplicate(oneOneReg, gateScale, floatMask);
        if constexpr (HAS_BIAS) {
            LoadAlign<float, LoadDist::DIST_NORM>(biasZeroReg, bias);
            LoadAlign<float, LoadDist::DIST_NORM>(biasOneReg, bias + FLOAT_ELEMENTS_PER_REG);
        }
        if constexpr (HAS_ALOG) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(expAReg, aLog);
            Exp(expAReg, expAReg, floatMask);
            Muls(expAReg, expAReg, -1.0f, floatMask);
        } else {
            Duplicate(expAReg, -1.0f, floatMask);
        }

        for (uint16_t row = 0; row < nRows; ++row) {
            const uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
            if constexpr (std::is_same<GateT, float>::value) {
                LoadAlign<float, LoadDist::DIST_NORM>(gateZeroReg, gk + rowOffset);
                LoadAlign<float, LoadDist::DIST_NORM>(gateOneReg, gk + rowOffset + FLOAT_ELEMENTS_PER_REG);
            } else {
                // UNPACK+ZERO keeps K-contiguous fp32, matching UB Cast then DIST_NORM.
                // CastHalf2Float ZERO/ONE deinterleaves even/odd and scrambles gk GM.
                LoadAlign<GateT, LoadDist::DIST_UNPACK_B16>(gateRawReg, gIn + rowOffset);
                Cast<float, GateT, ctHalf2Fp32Zero>(gateZeroReg, gateRawReg, floatMask);
                LoadAlign<GateT, LoadDist::DIST_UNPACK_B16>(
                    gateRawReg, gIn + rowOffset + FLOAT_ELEMENTS_PER_REG);
                Cast<float, GateT, ctHalf2Fp32Zero>(gateOneReg, gateRawReg, floatMask);
            }
            if constexpr (HAS_BIAS) {
                Add(gateZeroReg, gateZeroReg, biasZeroReg, floatMask);
                Add(gateOneReg, gateOneReg, biasOneReg, floatMask);
            }
            Mul(gateZeroReg, gateZeroReg, expAReg, floatMask);
            Mul(gateOneReg, gateOneReg, expAReg, floatMask);
            Exp(gateZeroReg, gateZeroReg, floatMask);
            Exp(gateOneReg, gateOneReg, floatMask);
            Adds(gateZeroReg, gateZeroReg, 1.0f, floatMask);
            Adds(gateOneReg, gateOneReg, 1.0f, floatMask);
            Div(sigmoidZeroReg, oneZeroReg, gateZeroReg, floatMask);
            Div(sigmoidOneReg, oneOneReg, gateOneReg, floatMask);
            Add(accZeroReg, accZeroReg, sigmoidZeroReg, floatMask);
            Add(accOneReg, accOneReg, sigmoidOneReg, floatMask);
            StoreAlign(gk + rowOffset, accZeroReg, floatMask);
            StoreAlign(gk + rowOffset + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
        }
        // Last-row gk stays K-contiguous in acc{Zero,One} (NORM 0:63 | 64:127),
        // matching pass-2 DIST_NORM / UNPACK loads. DeInterleave would switch
        // last to even/odd and scramble kg. Reloading this row from UB can hoist
        // before the accumulate loop and produce Inf kg = exp2(stale_g - cumsum).
        Adds(lastZeroReg, accZeroReg, 0.0f, floatMask);
        Adds(lastOneReg, accOneReg, 0.0f, floatMask);
    }

    MaskReg halfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<half> expHalfReg;
    RegTensor<float> betaReg;
    RegTensor<BetaT> betaRawReg;
    RegTensor<float> expZeroReg;
    RegTensor<float> expOneReg;
    RegTensor<float> qZeroReg;
    RegTensor<float> qOneReg;
    RegTensor<float> kZeroReg;
    RegTensor<float> kOneReg;
    RegTensor<float> outZeroReg;
    RegTensor<float> outOneReg;
    RegTensor<float> deltaZeroReg;
    RegTensor<float> deltaOneReg;
    RegTensor<float> vZeroReg;
    RegTensor<float> vOneReg;
    RegTensor<InputT> inputReg;
    RegTensor<OutputT> outputReg;
    RegTensor<float> floatScratchReg;
    for (uint16_t row = 0; row < nRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * static_cast<uint32_t>(kK);
        if constexpr (std::is_same<BetaT, float>::value) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, betaRow + row);
        } else {
            LoadIn<BetaT, true>(betaRawReg, betaRow + row);
            HalfOrFloat2Float(betaReg, betaRawReg, betaMask, floatMask);
        }

        LoadGateRegbasePair<float>(gateZeroReg, gateOneReg, gk + rowOffset, inputMask, floatScratchReg);
        LoadGateRegbasePair<InputT>(qZeroReg, qOneReg, q + rowOffset, inputMask, inputReg);
        LoadGateRegbasePair<InputT>(kZeroReg, kOneReg, k + rowOffset, inputMask, inputReg);
        LoadGateRegbasePair<InputT>(vZeroReg, vOneReg, v + rowOffset, inputMask, inputReg);
        Muls(expZeroReg, gateZeroReg, kLn2, floatMask);
        Muls(expOneReg, gateOneReg, kLn2, floatMask);
        Mins(expZeroReg, expZeroReg, kHalfExpInputMax, floatMask);
        Mins(expOneReg, expOneReg, kHalfExpInputMax, floatMask);
        Maxs(expZeroReg, expZeroReg, kHalfExpInputMin, floatMask);
        Maxs(expOneReg, expOneReg, kHalfExpInputMin, floatMask);
        ExpPairViaHalf(expZeroReg, expOneReg, expHalfReg, floatMask, halfMask);

        Mul(outZeroReg, qZeroReg, expZeroReg, floatMask);
        Mul(outOneReg, qOneReg, expOneReg, floatMask);
        StoreGateRegbasePair<OutputT>(qg + rowOffset, outZeroReg, outOneReg, inputMask, floatMask, outputReg);

        Mul(outZeroReg, kZeroReg, expZeroReg, floatMask);
        Mul(outOneReg, kOneReg, expOneReg, floatMask);
        Mul(outZeroReg, outZeroReg, betaReg, floatMask);
        Mul(outOneReg, outOneReg, betaReg, floatMask);
        StoreGateRegbasePair<OutputT>(kbg + rowOffset, outZeroReg, outOneReg, inputMask, floatMask, outputReg);

        Sub(deltaZeroReg, lastZeroReg, gateZeroReg, floatMask);
        Sub(deltaOneReg, lastOneReg, gateOneReg, floatMask);
        Muls(deltaZeroReg, deltaZeroReg, kLn2, floatMask);
        Muls(deltaOneReg, deltaOneReg, kLn2, floatMask);
        Mins(deltaZeroReg, deltaZeroReg, kHalfExpInputMax, floatMask);
        Mins(deltaOneReg, deltaOneReg, kHalfExpInputMax, floatMask);
        Maxs(deltaZeroReg, deltaZeroReg, kHalfExpInputMin, floatMask);
        Maxs(deltaOneReg, deltaOneReg, kHalfExpInputMin, floatMask);
        ExpPairViaHalf(deltaZeroReg, deltaOneReg, expHalfReg, floatMask, halfMask);
        Mul(outZeroReg, kZeroReg, deltaZeroReg, floatMask);
        Mul(outOneReg, kOneReg, deltaOneReg, floatMask);
        StoreGateRegbasePair<OutputT>(kg + rowOffset, outZeroReg, outOneReg, inputMask, floatMask, outputReg);

        Mul(vZeroReg, vZeroReg, betaReg, floatMask);
        Mul(vOneReg, vOneReg, betaReg, floatMask);
        StoreGateRegbasePair<OutputT>(vb + rowOffset, vZeroReg, vOneReg, inputMask, floatMask, outputReg);
    }
}

template <typename InputT, typename OutputT>
static __simd_vf__ inline void ComputeQgKbgKgRegbase(
    __ubuf__ InputT *q, __ubuf__ InputT *k, __ubuf__ OutputT *qg, __ubuf__ OutputT *kbg,
    __ubuf__ OutputT *kg, __ubuf__ float *gk, __ubuf__ float *gkLast, __ubuf__ float *betaRow,
    uint16_t rows, uint16_t cols, uint16_t validRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(InputT);

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> lastZeroReg;
    RegTensor<float> lastOneReg;
    RegTensor<float> betaReg;
    RegTensor<float> gateZeroReg;
    RegTensor<float> gateOneReg;
    RegTensor<float> expZeroReg;
    RegTensor<float> expOneReg;
    RegTensor<float> qZeroReg;
    RegTensor<float> qOneReg;
    RegTensor<float> kZeroReg;
    RegTensor<float> kOneReg;
    RegTensor<float> outZeroReg;
    RegTensor<float> outOneReg;
    RegTensor<float> deltaZeroReg;
    RegTensor<float> deltaOneReg;
    RegTensor<float> floatScratchReg;
    RegTensor<InputT> inputReg;
    RegTensor<OutputT> outputReg;
    LoadAlign<float, LoadDist::DIST_NORM>(lastZeroReg, gkLast);
    LoadAlign<float, LoadDist::DIST_NORM>(
        lastOneReg, gkLast + (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, betaRow + row);
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<InputT>(activeCount);
            const uint32_t offset = rowOffset + col;

            LoadGateRegbasePair<float>(gateZeroReg, gateOneReg, gk + offset, inputMask, floatScratchReg);
            Muls(expZeroReg, gateZeroReg, kLn2, floatMask);
            Muls(expOneReg, gateOneReg, kLn2, floatMask);
            Mins(expZeroReg, expZeroReg, kExpInputMax, floatMask);
            Mins(expOneReg, expOneReg, kExpInputMax, floatMask);
            Maxs(expZeroReg, expZeroReg, kExpInputMin, floatMask);
            Maxs(expOneReg, expOneReg, kExpInputMin, floatMask);
            Exp(expZeroReg, expZeroReg, floatMask);
            Exp(expOneReg, expOneReg, floatMask);

            LoadGateRegbasePair<InputT>(qZeroReg, qOneReg, q + offset, inputMask, inputReg);
            Mul(outZeroReg, qZeroReg, expZeroReg, floatMask);
            Mul(outOneReg, qOneReg, expOneReg, floatMask);
            StoreGateRegbasePair<OutputT>(qg + offset, outZeroReg, outOneReg, inputMask, floatMask, outputReg);

            LoadGateRegbasePair<InputT>(kZeroReg, kOneReg, k + offset, inputMask, inputReg);
            Mul(outZeroReg, kZeroReg, expZeroReg, floatMask);
            Mul(outOneReg, kOneReg, expOneReg, floatMask);
            Mul(outZeroReg, outZeroReg, betaReg, floatMask);
            Mul(outOneReg, outOneReg, betaReg, floatMask);
            StoreGateRegbasePair<OutputT>(kbg + offset, outZeroReg, outOneReg, inputMask, floatMask, outputReg);

            Sub(deltaZeroReg, lastZeroReg, gateZeroReg, floatMask);
            Sub(deltaOneReg, lastOneReg, gateOneReg, floatMask);
            Muls(deltaZeroReg, deltaZeroReg, kLn2, floatMask);
            Muls(deltaOneReg, deltaOneReg, kLn2, floatMask);
            Exp(deltaZeroReg, deltaZeroReg, floatMask);
            Exp(deltaOneReg, deltaOneReg, floatMask);
            Mul(outZeroReg, kZeroReg, deltaZeroReg, floatMask);
            Mul(outOneReg, kOneReg, deltaOneReg, floatMask);
            if (row >= validRows) {
                Duplicate(outZeroReg, 0.0f, floatMask);
                Duplicate(outOneReg, 0.0f, floatMask);
            }
            StoreGateRegbasePair<OutputT>(kg + offset, outZeroReg, outOneReg, inputMask, floatMask, outputReg);
        }
    }
}

template <typename VType>
static __simd_vf__ inline void ComputeVbRegbase(
    __ubuf__ VType *vIn, __ubuf__ VType *vbOut, __ubuf__ float *betaRow,
    uint16_t rows, uint16_t cols, uint16_t validRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(VType);
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> betaReg;
    RegTensor<float> vZeroReg;
    RegTensor<float> vOneReg;
    RegTensor<VType> outReg;
    RegTensor<VType> inputReg;

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, betaRow + row);
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<VType>(activeCount);
            const uint32_t offset = rowOffset + col;
            LoadGateRegbasePair<VType>(vZeroReg, vOneReg, vIn + offset, inputMask, inputReg);
            Mul(vZeroReg, vZeroReg, betaReg, floatMask);
            Mul(vOneReg, vOneReg, betaReg, floatMask);
            if (row >= validRows) {
                Duplicate(vZeroReg, 0.0f, floatMask);
                Duplicate(vOneReg, 0.0f, floatMask);
            }
            CastFloat2Half<VType>(outReg, vZeroReg, vOneReg, floatMask);
            StoreAlign(vbOut + offset, outReg, inputMask);
        }
    }
}

} // namespace KdaBwdRecomputeArch35

#endif // __CCE_AICORE__ == 310

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_REGBASE_H
