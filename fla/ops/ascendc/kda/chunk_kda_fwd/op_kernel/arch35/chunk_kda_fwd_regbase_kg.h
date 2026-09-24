/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

// 状态累加消费的 gated K（kg[row] = k[row] * 2^(ref - gk[row])）的 regbase 实现。
// 原实现位于 chunk_kda_fwd_post_wu.h；这里抽到共享头，供 post-wu 与 prepare 收尾阶段复用，
// 避免 prepare 侧用逐行 Sub + 全局 Exp 的等价实现引入额外开销。数值语义与调用方约定完全一致：
// 输入 k 行（T）就地写回 kg（T），gate 为按行的门控（GK_T），ref 为参考行门控（float，逐列）。

#pragma once

#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif

namespace KdaRegbaseKg {

namespace {
constexpr float KDA_KG_LN2 = 0.69314718055994530942f;
constexpr float KDA_KG_EXP2_CLAMP = 80.0f;
constexpr float KDA_KG_EXP_INPUT_MAX = KDA_KG_EXP2_CLAMP * KDA_KG_LN2;
constexpr float KDA_KG_EXP_INPUT_MIN = -KDA_KG_EXP2_CLAMP * KDA_KG_LN2;
constexpr float KDA_KG_FP16_MAX = 65504.0f;
}  // namespace

template <typename InputT>
__simd_callee__ inline void LoadPostKdaGateRegbasePair(
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
__simd_callee__ inline void StorePostKdaGateRegbasePair(
    __ubuf__ OutputT *dst,
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<OutputT, half>()) {
        Mins(zeroReg, zeroReg, KDA_KG_FP16_MAX, floatMask);
        Mins(oneReg, oneReg, KDA_KG_FP16_MAX, floatMask);
        Maxs(zeroReg, zeroReg, -KDA_KG_FP16_MAX, floatMask);
        Maxs(oneReg, oneReg, -KDA_KG_FP16_MAX, floatMask);
    }
    RegTensor<OutputT> outputReg;
    CastFloat2Half<OutputT>(outputReg, zeroReg, oneReg, floatMask);
    StoreAlign(dst, outputReg, inputMask);
}

template <typename T, typename GK_T>
static __simd_vf__ inline void ComputePostKdaKgRegbase(
    __ubuf__ T *kAndKg, __ubuf__ GK_T *gate, __ubuf__ float *ref,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(T);
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<T>(activeCount);
            uint32_t offset = rowOffset + col;

            RegTensor<float> gateZeroReg;
            RegTensor<float> gateOneReg;
            RegTensor<float> refZeroReg;
            RegTensor<float> refOneReg;
            RegTensor<float> expZeroReg;
            RegTensor<float> expOneReg;
            RegTensor<float> inputZeroReg;
            RegTensor<float> inputOneReg;
            RegTensor<float> outputZeroReg;
            RegTensor<float> outputOneReg;

            LoadPostKdaGateRegbasePair<GK_T>(
                gateZeroReg, gateOneReg, gate + offset, inputMask);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                refZeroReg, refOneReg, ref + col);
            SubFloatTwoReg(expZeroReg, expOneReg, refZeroReg, refOneReg,
                           gateZeroReg, gateOneReg, floatMask);
            Muls(expZeroReg, expZeroReg, KDA_KG_LN2, floatMask);
            Muls(expOneReg, expOneReg, KDA_KG_LN2, floatMask);
            MinsFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg,
                            KDA_KG_EXP_INPUT_MAX, floatMask);
            Maxs(expZeroReg, expZeroReg, KDA_KG_EXP_INPUT_MIN, floatMask);
            Maxs(expOneReg, expOneReg, KDA_KG_EXP_INPUT_MIN, floatMask);
            ExpFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg, floatMask);

            LoadPostKdaGateRegbasePair<T>(
                inputZeroReg, inputOneReg, kAndKg + offset, inputMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           expZeroReg, expOneReg, floatMask);
            StorePostKdaGateRegbasePair<T>(
                kAndKg + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
        }
    }
}

}  // namespace KdaRegbaseKg
#endif
