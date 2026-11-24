/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CHUNK_BWD_DQKWG_ARCH35_REGBASE_H
#define CHUNK_BWD_DQKWG_ARCH35_REGBASE_H

#include "../chunk_bwd_dqkwg_common.h"
#include "kernel_utils/vector/regbase.hpp"

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint16_t V_LENGTH_FP32 = VECTOR_REG_WIDTH / sizeof(float);

constexpr uint16_t V_LENGTH_HALF = VECTOR_REG_WIDTH / sizeof(half);

template <uint16_t BT_SIZE, uint16_t BT_SUB_START>
static __simd_vf__ inline void Mul1Half(
    __ubuf__ float *outFp32,
    __ubuf__ float *gFp32,
    __ubuf__ float *maskAddr,
    uint16_t realBt,
    float scale)
{
    RegTensor<float> regGLeft, regSum, regExp, regMask, regOut;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();

    if constexpr (BT_SIZE == 64) {
        RegTensor<float> regG, regNegG;
        LoadAlign<float>(regG, gFp32);
        Muls(regNegG, regG, -1.0f, maskAll);

        for (uint16_t i = 0; i < realBt; i++) {

            LoadAlign<float, LoadDist::DIST_BRC_B32>(regGLeft, gFp32 + i);
            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regMask, maskAddr, V_LENGTH_FP32);
            Add(regSum, regNegG, regGLeft, maskAll);
            Mins(regSum, regSum, 0.0f, maskAll);
            Exp(regExp, regSum, maskAll);

            Mul(regOut, regExp, regMask, maskAll);
            StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(outFp32, regOut, V_LENGTH_FP32, maskAll);
        }
    } else {

        RegTensor<float> regG0, regG1, regNegG0, regNegG1;
        RegTensor<float> regSum1, regExp1, regOut1;
        LoadAlign<float>(regG0, gFp32);
        LoadAlign<float>(regG1, gFp32 + V_LENGTH_FP32);
        Muls(regNegG0, regG0, -1.0f, maskAll);
        Muls(regNegG1, regG1, -1.0f, maskAll);

        for (uint16_t i = 0; i < realBt; i++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(regGLeft, gFp32 + BT_SUB_START + i);
            Add(regSum, regNegG0, regGLeft, maskAll);
            Add(regSum1, regNegG1, regGLeft, maskAll);
            Mins(regSum, regSum, 0.0f, maskAll);
            Mins(regSum1, regSum1, 0.0f, maskAll);
            Exp(regExp, regSum, maskAll);
            Exp(regExp1, regSum1, maskAll);

            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regMask, maskAddr, V_LENGTH_FP32);
            if constexpr (BT_SUB_START == 0) {
                Mul(regOut, regExp, regMask, maskAll);
                Muls(regOut1, regExp1, 0.0f, maskAll);
            } else {
                Muls(regOut, regExp, scale, maskAll);
                Mul(regOut1, regExp1, regMask, maskAll);
            }
            StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(outFp32, regOut, V_LENGTH_FP32, maskAll);
            StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(outFp32, regOut1, V_LENGTH_FP32, maskAll);
        }
    }
}

static __simd_vf__ inline void DqState(
    __ubuf__ float *dqStateFp32,
    __ubuf__ float *dqFp32,
    __ubuf__ float *gFp32,
    __ubuf__ float *factorScratch,
    uint32_t realBt,
    float scale)
{
    RegTensor<float> regG, regFactor, regDq1, regDq2, regOut1, regOut2;

    uint16_t gLoopTimes = static_cast<uint16_t>((realBt + V_LENGTH_FP32 - 1) / V_LENGTH_FP32);
    uint32_t gLength = realBt;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskG;
    for (uint16_t j = 0; j < gLoopTimes; j++) {
        maskG = UpdateMask<float>(gLength);
        LoadAlign<float>(regG, gFp32 + j * V_LENGTH_FP32);
        Exp(regFactor, regG, maskG);
        Muls(regFactor, regFactor, scale, maskG);
        StoreAlign<float>(factorScratch + j * V_LENGTH_FP32, regFactor, maskG);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < realBt; i++) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(regFactor, factorScratch + i);
        LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regDq1, dqFp32, V_LENGTH_FP32);
        LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regDq2, dqFp32, V_LENGTH_FP32);
        Mul(regOut1, regDq1, regFactor, maskAll);
        Mul(regOut2, regDq2, regFactor, maskAll);
        StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(dqStateFp32, regOut1, V_LENGTH_FP32, maskAll);
        StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(dqStateFp32, regOut2, V_LENGTH_FP32, maskAll);
    }
}

template <typename HalfT>
static __simd_vf__ inline void DwNegate(__ubuf__ HalfT *dwOutBuf, __ubuf__ HalfT *dwInBuf, uint32_t elemCount)
{

    uint16_t colLoopTimes = static_cast<uint16_t>(elemCount / V_LENGTH_HALF);
    RegTensor<HalfT> regH;
    RegTensor<float> regF0, regF1;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskHalfAll = CreateMask<HalfT, MaskPattern::ALL>();
    for (uint16_t j = 0; j < colLoopTimes; j++) {
        LoadAlign<HalfT, PostLiteral::POST_MODE_UPDATE>(regH, dwInBuf, V_LENGTH_HALF);
        CastHalf2Float<HalfT>(regF0, regF1, regH, maskHalfAll);
        Muls(regF0, regF0, -1.0f, maskAll);
        Muls(regF1, regF1, -1.0f, maskAll);
        Cast<HalfT, float, ctFp322HalfOne>(regH, regF1, maskAll);
        Cast<HalfT, float, ctFp322HalfZero>(regH, regF0, maskAll);
        StoreAlign<HalfT, PostLiteral::POST_MODE_UPDATE>(dwOutBuf, regH, V_LENGTH_HALF, maskHalfAll);
    }
}

template <typename HalfT, bool needAdd>
static __simd_vf__ inline void DgLastMulAccum(
    __ubuf__ float *sumFp32,
    __ubuf__ HalfT *hHalf,
    __ubuf__ HalfT *dhHalf,
    uint32_t elemCount)
{
    __ubuf__ float *srcAddr = sumFp32;
    uint16_t colLoopTimes = static_cast<uint16_t>(elemCount / V_LENGTH_FP32);
    RegTensor<HalfT> regHH, regDhH;
    RegTensor<float> regHF, regDhF, regProd;
    RegTensor<float> regS;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t j = 0; j < colLoopTimes; j++) {
        LoadAlign<HalfT, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK_B16>(regHH, hHalf, V_LENGTH_FP32);
        LoadAlign<HalfT, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK_B16>(regDhH, dhHalf, V_LENGTH_FP32);
        Cast<float, HalfT, ctHalf2Fp32Zero>(regHF, regHH, maskAll);
        Cast<float, HalfT, ctHalf2Fp32Zero>(regDhF, regDhH, maskAll);
        Mul(regProd, regHF, regDhF, maskAll);
        if constexpr (needAdd) {
            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regS, srcAddr, V_LENGTH_FP32);
            Add(regProd, regS, regProd, maskAll);
        }
        StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(sumFp32, regProd, V_LENGTH_FP32, maskAll);
    }
}

#endif
