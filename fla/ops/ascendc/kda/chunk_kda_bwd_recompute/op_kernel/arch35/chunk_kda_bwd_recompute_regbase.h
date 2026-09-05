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

template <bool HAS_BIAS>
static __simd_vf__ inline void AccumulateSafeGateChunk128Regbase(
    __ubuf__ float *input, __ubuf__ float *bias, __ubuf__ float *acc,
    uint16_t rows, float expA, float lowerBound)
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
    LoadAlign<float, LoadDist::DIST_NORM>(accZeroReg, acc);
    LoadAlign<float, LoadDist::DIST_NORM>(accOneReg, acc + FLOAT_ELEMENTS_PER_REG);
    Duplicate(oneZeroReg, 1.0f, floatMask);
    Duplicate(oneOneReg, 1.0f, floatMask);
    if constexpr (HAS_BIAS) {
        LoadAlign<float, LoadDist::DIST_NORM>(biasZeroReg, bias);
        LoadAlign<float, LoadDist::DIST_NORM>(biasOneReg, bias + FLOAT_ELEMENTS_PER_REG);
    }

    const float gateScale = lowerBound * KDA_BWD_RECOMPUTE_RCP_LN2;
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
        RegTensor<float> gateZeroReg;
        RegTensor<float> gateOneReg;
        RegTensor<float> sigmoidZeroReg;
        RegTensor<float> sigmoidOneReg;
        LoadAlign<float, LoadDist::DIST_NORM>(gateZeroReg, input + rowOffset);
        LoadAlign<float, LoadDist::DIST_NORM>(gateOneReg, input + rowOffset + FLOAT_ELEMENTS_PER_REG);
        if constexpr (HAS_BIAS) {
            Add(gateZeroReg, gateZeroReg, biasZeroReg, floatMask);
            Add(gateOneReg, gateOneReg, biasOneReg, floatMask);
        }
        Muls(gateZeroReg, gateZeroReg, -expA, floatMask);
        Muls(gateOneReg, gateOneReg, -expA, floatMask);
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

template <typename InputT>
__simd_callee__ inline void LoadGateRegbasePair(
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    __ubuf__ InputT *src,
    AscendC::MicroAPI::MaskReg &inputMask)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<InputT, float>()) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zeroReg, oneReg, src);
    } else {
        RegTensor<InputT> inputReg;
        LoadIn<InputT, false>(inputReg, src);
        CastHalf2Float<InputT>(zeroReg, oneReg, inputReg, inputMask);
    }
}

template <typename OutputT>
__simd_callee__ inline void StoreGateRegbasePair(
    __ubuf__ OutputT *dst,
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    RegTensor<OutputT> outputReg;
    CastFloat2Half<OutputT>(outputReg, zeroReg, oneReg, floatMask);
    StoreAlign(dst, outputReg, inputMask);
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
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        RegTensor<float> betaReg;
        LoadAlign<float, LoadDist::DIST_NORM>(betaReg, betaRow + row);
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            const uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<InputT>(activeCount);
            const uint32_t offset = rowOffset + col;

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
            RegTensor<float> lastZeroReg;
            RegTensor<float> lastOneReg;
            RegTensor<float> deltaZeroReg;
            RegTensor<float> deltaOneReg;

            LoadGateRegbasePair<float>(gateZeroReg, gateOneReg, gk + offset, inputMask);
            Muls(expZeroReg, gateZeroReg, kLn2, floatMask);
            Muls(expOneReg, gateOneReg, kLn2, floatMask);
            Mins(expZeroReg, expZeroReg, kExpInputMax, floatMask);
            Mins(expOneReg, expOneReg, kExpInputMax, floatMask);
            Maxs(expZeroReg, expZeroReg, kExpInputMin, floatMask);
            Maxs(expOneReg, expOneReg, kExpInputMin, floatMask);
            Exp(expZeroReg, expZeroReg, floatMask);
            Exp(expOneReg, expOneReg, floatMask);

            LoadGateRegbasePair<InputT>(qZeroReg, qOneReg, q + offset, inputMask);
            Mul(outZeroReg, qZeroReg, expZeroReg, floatMask);
            Mul(outOneReg, qOneReg, expOneReg, floatMask);
            StoreGateRegbasePair<OutputT>(qg + offset, outZeroReg, outOneReg, inputMask, floatMask);

            LoadGateRegbasePair<InputT>(kZeroReg, kOneReg, k + offset, inputMask);
            Mul(outZeroReg, kZeroReg, expZeroReg, floatMask);
            Mul(outOneReg, kOneReg, expOneReg, floatMask);
            Mul(outZeroReg, outZeroReg, betaReg, floatMask);
            Mul(outOneReg, outOneReg, betaReg, floatMask);
            StoreGateRegbasePair<OutputT>(kbg + offset, outZeroReg, outOneReg, inputMask, floatMask);

            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(lastZeroReg, lastOneReg, gkLast + col);
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
            StoreGateRegbasePair<OutputT>(kg + offset, outZeroReg, outOneReg, inputMask, floatMask);
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

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        RegTensor<float> betaReg;
        LoadAlign<float, LoadDist::DIST_NORM>(betaReg, betaRow + row);
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            const uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<VType>(activeCount);
            const uint32_t offset = rowOffset + col;
            RegTensor<float> vZeroReg;
            RegTensor<float> vOneReg;
            RegTensor<VType> outReg;
            LoadGateRegbasePair<VType>(vZeroReg, vOneReg, vIn + offset, inputMask);
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
