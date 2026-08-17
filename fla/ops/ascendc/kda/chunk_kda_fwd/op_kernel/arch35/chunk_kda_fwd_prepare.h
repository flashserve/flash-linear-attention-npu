/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "../chunk_kda_fwd_varlen.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif
#endif
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "../chunk_kda_fwd_plan.h"
#include "chunk_kda_fwd_post_wu.h"

using namespace AscendC;

namespace KdaPrepare {
namespace {
using KdaInt64 = tla::Int<64>;
using KdaInt128 = tla::Int<128>;
constexpr float LN2 = 0.69314718055994530942f;
constexpr float RCP_LN2 = 1.44269504088896340736f;
constexpr float KDA_EXP2_CLAMP = 80.0f;
constexpr float KDA_EXP_INPUT_MAX = KDA_EXP2_CLAMP * LN2;
constexpr float KDA_EXP_INPUT_MIN = -KDA_EXP2_CLAMP * LN2;
constexpr float KDA_SCORE_EXP2_CLAMP = 120.0f;
constexpr float KDA_SCORE_EXP2_MIN_CLAMP = 126.0f;
constexpr float KDA_SCORE_EXP_INPUT_MAX = KDA_SCORE_EXP2_CLAMP * LN2;
constexpr float KDA_SCORE_EXP_INPUT_MIN = -KDA_SCORE_EXP2_MIN_CLAMP * LN2;
constexpr float KDA_FP16_MAX = 65504.0f;
constexpr uint32_t EXP2_UB_ELEMENTS = 256;
constexpr uint32_t EXP2_UB_BYTES = EXP2_UB_ELEMENTS * (sizeof(float) + sizeof(uint16_t));
constexpr uint32_t EXP2_EVENT_ID = 0;
constexpr uint32_t KDA_SOLVE_BT = 64;
constexpr uint32_t KDA_SOLVE_MATRIX_ELEMENTS = KDA_SOLVE_BT * KDA_SOLVE_BT;
constexpr uint32_t KDA_SOLVE_SCRATCH_X = 0;
constexpr uint32_t KDA_SOLVE_SCRATCH_Y0 = 1;
constexpr uint32_t KDA_SOLVE_SCRATCH_TMP = 2;
constexpr uint32_t KDA_SOLVE_SCRATCH_Y1 = 3;
constexpr uint32_t KDA_SOLVE_SCRATCH_IDENTITY = 4;
constexpr uint32_t KDA_SOLVE_SCRATCH_RAW_AKK = KDA_SOLVE_SCRATCH_Y1;
constexpr uint32_t KDA_SOLVE_SCRATCH_RAW_AQK = KDA_SOLVE_SCRATCH_IDENTITY;
constexpr uint32_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint32_t KDA_SOLVE_PIPELINE_DEPTH = 4;
constexpr uint32_t KDA_SOLVE_DIAG_BT = 16;
constexpr uint32_t KDA_SOLVE_DIAG_BLOCKS = KDA_SOLVE_BT / KDA_SOLVE_DIAG_BT;
constexpr uint32_t KDA_SOLVE_DIAG_MCH_ITERS = 3;
// 将 safe-gate 的局部指数跨度限制在 BF16 score 可表示范围内，同时减少
// 重复的 gate factor 计算和 AIV/AIC 握手。
constexpr uint32_t KDA_SCORE_REF_BC = 32;
constexpr uint32_t KDA_SAFE_SCORE_REF_BC = 32;
constexpr uint32_t KDA_VEC_ARENA_ELEMENTS = 32768;
constexpr uint32_t KDA_BITS_PER_MASK_BYTE = 8;
constexpr uint32_t KDA_SELECT_COL_BLOCKS = 2;
constexpr uint32_t KDA_SELECT_COL_MASK_BYTES = KDA_SOLVE_MATRIX_ELEMENTS / KDA_BITS_PER_MASK_BYTE;
constexpr uint32_t KDA_SELECT_MASK_BYTES = KDA_SELECT_COL_BLOCKS * KDA_SELECT_COL_MASK_BYTES;
constexpr uint32_t KDA_SELECT_AQK_MASK_BYTE_OFFSET = 120 * 1024;
constexpr uint32_t KDA_SELECT_AKK_MASK_BYTE_OFFSET = KDA_SELECT_AQK_MASK_BYTE_OFFSET + KDA_SELECT_MASK_BYTES;
constexpr uint32_t KDA_SELECT_ZERO_BYTE_OFFSET = KDA_SELECT_AKK_MASK_BYTE_OFFSET + KDA_SELECT_MASK_BYTES;
constexpr uint32_t KDA_SELECT_ZERO_FLOAT_OFFSET = KDA_SELECT_ZERO_BYTE_OFFSET / sizeof(float);
constexpr uint8_t KDA_SCORE_DONE_FLAG0 = 2;
constexpr uint8_t KDA_SCORE_DONE_FLAG1 = 3;
constexpr uint8_t KDA_SCORE_READY_FLAG0 = 4;
constexpr uint8_t KDA_SCORE_READY_FLAG1 = 5;
constexpr uint8_t KDA_SOLVE_DONE_FLAG = 6;
constexpr uint8_t KDA_SOLVE_READY_FLAG = 7;
constexpr uint8_t KDA_POST_READY_FLAG = 0;
constexpr uint8_t KDA_POST_FREE_FLAG = 1;
constexpr uint32_t KDA_SCORE_QUEUE_DEPTH = 2;
constexpr uint32_t KDA_SCORE_LANES = 2;
constexpr uint32_t KDA_POST_QUEUE_DEPTH = 4;
constexpr uint32_t KDA_SCORE_SCRATCH_SLOTS = KDA_SCORE_QUEUE_DEPTH * KDA_SCORE_LANES;
constexpr uint32_t KDA_SYNC_REVERSE_DEPTH = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint32_t KDA_SCORE_SCRATCH_QG = 0;
constexpr uint32_t KDA_SCORE_SCRATCH_W = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_KG = 2;
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;
constexpr uint32_t KDA_GATE_TILE_ROWS = 16;
constexpr uint32_t KDA_GATE_PIPELINE_DEPTH = 3;
constexpr uint32_t KDA_AIV_UB_BUDGET_BYTES = 192 * 1024;
constexpr uint32_t KDA_LOCAL_GK_FLOAT_OFFSET = 10 * 1024;
constexpr uint32_t KDA_SCALED_QG_FLOAT_OFFSET = 18 * 1024;
// 特殊双head路径删除后，96KB起始的UB区间用于保存一个AIV在
// C64/K128目标路径上负责的32行raw Q/K。两份输入共占16KB，结束于
// 112KB；它避开80KB起始的beta缩放typed scratch，也不触碰120KB
// 起始的因果mask区间。
constexpr uint32_t KDA_QK_RESIDENT_FLOAT_OFFSET = 24 * 1024;
constexpr uint32_t KDA_QK_RESIDENT_ROWS = 32;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <bool HAS_BIAS>
static __simd_vf__ inline void AccumulateRawSafeGateChunk128Regbase(
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

    const float gateScale = lowerBound * RCP_LN2;
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
__simd_callee__ inline void LoadKdaGateRegbasePair(
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
__simd_callee__ inline void ClampKdaGateRegbaseOutput(
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<OutputT, half>()) {
        Mins(zeroReg, zeroReg, KDA_FP16_MAX, floatMask);
        Mins(oneReg, oneReg, KDA_FP16_MAX, floatMask);
        Maxs(zeroReg, zeroReg, -KDA_FP16_MAX, floatMask);
        Maxs(oneReg, oneReg, -KDA_FP16_MAX, floatMask);
    }
}

template <typename OutputT, bool USE_REF, bool NEGATIVE>
__simd_callee__ inline void BuildKdaGateRegbaseExp(
    AscendC::MicroAPI::RegTensor<float> &expZeroReg,
    AscendC::MicroAPI::RegTensor<float> &expOneReg,
    AscendC::MicroAPI::RegTensor<float> &gateZeroReg,
    AscendC::MicroAPI::RegTensor<float> &gateOneReg,
    __ubuf__ float *ref,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    constexpr float expInputMax =
        std::is_same<OutputT, bfloat16_t>() ? KDA_SCORE_EXP_INPUT_MAX : KDA_EXP_INPUT_MAX;
    constexpr float expInputMin =
        std::is_same<OutputT, bfloat16_t>() ? KDA_SCORE_EXP_INPUT_MIN : KDA_EXP_INPUT_MIN;
    if constexpr (USE_REF) {
        RegTensor<float> refZeroReg;
        RegTensor<float> refOneReg;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(refZeroReg, refOneReg, ref);
        if constexpr (NEGATIVE) {
            SubFloatTwoReg(expZeroReg, expOneReg, refZeroReg, refOneReg,
                           gateZeroReg, gateOneReg, floatMask);
        } else {
            SubFloatTwoReg(expZeroReg, expOneReg, gateZeroReg, gateOneReg,
                           refZeroReg, refOneReg, floatMask);
        }
    } else if constexpr (NEGATIVE) {
        Muls(expZeroReg, gateZeroReg, -1.0f, floatMask);
        Muls(expOneReg, gateOneReg, -1.0f, floatMask);
    } else {
        Adds(expZeroReg, gateZeroReg, 0.0f, floatMask);
        Adds(expOneReg, gateOneReg, 0.0f, floatMask);
    }
    Muls(expZeroReg, expZeroReg, LN2, floatMask);
    Muls(expOneReg, expOneReg, LN2, floatMask);
    MinsFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg,
                    expInputMax, floatMask);
    Maxs(expZeroReg, expZeroReg, expInputMin, floatMask);
    Maxs(expOneReg, expOneReg, expInputMin, floatMask);
    ExpFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg, floatMask);
}

template <typename OutputT>
__simd_callee__ inline void StoreKdaGateRegbasePair(
    __ubuf__ OutputT *dst,
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    RegTensor<OutputT> outputReg;
    ClampKdaGateRegbaseOutput<OutputT>(zeroReg, oneReg, floatMask);
    CastFloat2Half<OutputT>(outputReg, zeroReg, oneReg, floatMask);
    StoreAlign(dst, outputReg, inputMask);
}

template <typename InputT, typename OutputT, typename GK_T, bool USE_REF>
static __simd_vf__ inline void PrepareKdaGateQwRegbase(
    __ubuf__ InputT *q, __ubuf__ InputT *k, __ubuf__ OutputT *qOut,
    __ubuf__ OutputT *kOut, __ubuf__ InputT *qDirect, __ubuf__ InputT *kDirect,
    __ubuf__ GK_T *gate, __ubuf__ float *ref, uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(InputT);

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<InputT>(activeCount);
            uint32_t offset = rowOffset + col;

            RegTensor<float> gateZeroReg;
            RegTensor<float> gateOneReg;
            RegTensor<float> expZeroReg;
            RegTensor<float> expOneReg;
            RegTensor<float> directZeroReg;
            RegTensor<float> directOneReg;
            RegTensor<float> inputZeroReg;
            RegTensor<float> inputOneReg;
            RegTensor<float> outputZeroReg;
            RegTensor<float> outputOneReg;

            LoadKdaGateRegbasePair<GK_T>(gateZeroReg, gateOneReg, gate + offset, inputMask);
            BuildKdaGateRegbaseExp<OutputT, USE_REF, false>(
                expZeroReg, expOneReg, gateZeroReg, gateOneReg, ref + col, floatMask);
            BuildKdaGateRegbaseExp<InputT, false, false>(
                directZeroReg, directOneReg, gateZeroReg, gateOneReg, ref + col, floatMask);

            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, q + offset, inputMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           directZeroReg, directOneReg, floatMask);
            StoreKdaGateRegbasePair<InputT>(
                qDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           expZeroReg, expOneReg, floatMask);
            StoreKdaGateRegbasePair<OutputT>(
                qOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);

            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, k + offset, inputMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           directZeroReg, directOneReg, floatMask);
            StoreKdaGateRegbasePair<InputT>(
                kDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           expZeroReg, expOneReg, floatMask);
            StoreKdaGateRegbasePair<OutputT>(
                kOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
        }
    }
}

template <typename InputT, typename OutputT, typename GK_T, bool USE_REF>
static __simd_vf__ inline void PrepareKdaGateKgRegbase(
    __ubuf__ OutputT *kg, __ubuf__ InputT *k, __ubuf__ GK_T *gate,
    __ubuf__ float *ref, uint16_t rows, uint16_t cols, uint16_t validRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(InputT);

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<InputT>(activeCount);
            uint32_t offset = rowOffset + col;

            RegTensor<float> gateZeroReg;
            RegTensor<float> gateOneReg;
            RegTensor<float> expZeroReg;
            RegTensor<float> expOneReg;
            RegTensor<float> inputZeroReg;
            RegTensor<float> inputOneReg;
            RegTensor<float> outputZeroReg;
            RegTensor<float> outputOneReg;

            LoadKdaGateRegbasePair<GK_T>(gateZeroReg, gateOneReg, gate + offset, inputMask);
            BuildKdaGateRegbaseExp<OutputT, USE_REF, true>(
                expZeroReg, expOneReg, gateZeroReg, gateOneReg, ref + col, floatMask);
            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, k + offset, inputMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           expZeroReg, expOneReg, floatMask);
            if constexpr (USE_REF) {
                if (row >= validRows) {
                    Duplicate(outputZeroReg, 0.0f, floatMask);
                    Duplicate(outputOneReg, 0.0f, floatMask);
                }
            }
            StoreKdaGateRegbasePair<OutputT>(
                kg + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
        }
    }
}

template <typename InputT, typename OutputT, typename GK_T, bool USE_REF, bool STORE_DIRECT,
          bool EXPORT_FINAL_KG, bool SCALE_SCORE_W = false>
static __simd_vf__ inline void PrepareKdaGateQwKgRegbase(
    __ubuf__ InputT *q, __ubuf__ InputT *k, __ubuf__ OutputT *qOut,
    __ubuf__ OutputT *wOut, __ubuf__ OutputT *kgOut, __ubuf__ InputT *qDirect,
    __ubuf__ InputT *wDirect, __ubuf__ InputT *v, __ubuf__ InputT *vDirect,
    __ubuf__ InputT *finalKgOut, __ubuf__ float *beta, __ubuf__ GK_T *gate,
    __ubuf__ float *ref, __ubuf__ float *finalRef,
    uint16_t rows, uint16_t cols, uint16_t validRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(InputT);
    constexpr float scoreExpInputMax =
        std::is_same<OutputT, bfloat16_t>() ? KDA_SCORE_EXP_INPUT_MAX : KDA_EXP_INPUT_MAX;
    constexpr float scoreExpInputMin =
        std::is_same<OutputT, bfloat16_t>() ? KDA_SCORE_EXP_INPUT_MIN : KDA_EXP_INPUT_MIN;

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<InputT>(activeCount);
            uint32_t offset = rowOffset + col;

            RegTensor<float> gateZeroReg;
            RegTensor<float> gateOneReg;
            RegTensor<float> posZeroReg;
            RegTensor<float> posOneReg;
            RegTensor<float> negZeroReg;
            RegTensor<float> negOneReg;
            RegTensor<float> directZeroReg;
            RegTensor<float> directOneReg;
            RegTensor<float> finalNegZeroReg;
            RegTensor<float> finalNegOneReg;
            RegTensor<float> inputZeroReg;
            RegTensor<float> inputOneReg;
            RegTensor<float> outputZeroReg;
            RegTensor<float> outputOneReg;
            RegTensor<float> betaReg;

            LoadKdaGateRegbasePair<GK_T>(gateZeroReg, gateOneReg, gate + offset, inputMask);
            if constexpr (USE_REF) {
                RegTensor<float> refZeroReg;
                RegTensor<float> refOneReg;
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(refZeroReg, refOneReg, ref + col);
                SubFloatTwoReg(posZeroReg, posOneReg, gateZeroReg, gateOneReg,
                               refZeroReg, refOneReg, floatMask);
                SubFloatTwoReg(negZeroReg, negOneReg, refZeroReg, refOneReg,
                               gateZeroReg, gateOneReg, floatMask);
            } else {
                Adds(posZeroReg, gateZeroReg, 0.0f, floatMask);
                Adds(posOneReg, gateOneReg, 0.0f, floatMask);
                Muls(negZeroReg, gateZeroReg, -1.0f, floatMask);
                Muls(negOneReg, gateOneReg, -1.0f, floatMask);
            }
            if constexpr (STORE_DIRECT) {
                Adds(directZeroReg, gateZeroReg, 0.0f, floatMask);
                Adds(directOneReg, gateOneReg, 0.0f, floatMask);
                Muls(directZeroReg, directZeroReg, LN2, floatMask);
                Muls(directOneReg, directOneReg, LN2, floatMask);
                MinsFloatTwoReg(directZeroReg, directOneReg, directZeroReg, directOneReg,
                                KDA_EXP_INPUT_MAX, floatMask);
                Maxs(directZeroReg, directZeroReg, KDA_EXP_INPUT_MIN, floatMask);
                Maxs(directOneReg, directOneReg, KDA_EXP_INPUT_MIN, floatMask);
                ExpFloatTwoReg(directZeroReg, directOneReg, directZeroReg, directOneReg, floatMask);
            }
            Muls(posZeroReg, posZeroReg, LN2, floatMask);
            Muls(posOneReg, posOneReg, LN2, floatMask);
            Muls(negZeroReg, negZeroReg, LN2, floatMask);
            Muls(negOneReg, negOneReg, LN2, floatMask);
            MinsFloatTwoReg(posZeroReg, posOneReg, posZeroReg, posOneReg,
                            scoreExpInputMax, floatMask);
            MinsFloatTwoReg(negZeroReg, negOneReg, negZeroReg, negOneReg,
                            scoreExpInputMax, floatMask);
            Maxs(posZeroReg, posZeroReg, scoreExpInputMin, floatMask);
            Maxs(posOneReg, posOneReg, scoreExpInputMin, floatMask);
            Maxs(negZeroReg, negZeroReg, scoreExpInputMin, floatMask);
            Maxs(negOneReg, negOneReg, scoreExpInputMin, floatMask);
            ExpFloatTwoReg(posZeroReg, posOneReg, posZeroReg, posOneReg, floatMask);
            ExpFloatTwoReg(negZeroReg, negOneReg, negZeroReg, negOneReg, floatMask);

            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, q + offset, inputMask);
            if constexpr (STORE_DIRECT) {
                MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                               directZeroReg, directOneReg, floatMask);
                StoreKdaGateRegbasePair<InputT>(
                    qDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            }
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           posZeroReg, posOneReg, floatMask);
            StoreKdaGateRegbasePair<OutputT>(
                qOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);

            if constexpr (STORE_DIRECT || SCALE_SCORE_W) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
            }
            if constexpr (STORE_DIRECT) {
                LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, v + offset, inputMask);
                Mul(outputZeroReg, inputZeroReg, betaReg, floatMask);
                Mul(outputOneReg, inputOneReg, betaReg, floatMask);
                StoreKdaGateRegbasePair<InputT>(
                    vDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            }

            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, k + offset, inputMask);
            if constexpr (STORE_DIRECT) {
                MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                               directZeroReg, directOneReg, floatMask);
                RegTensor<InputT> roundedReg;
                ClampKdaGateRegbaseOutput<InputT>(outputZeroReg, outputOneReg, floatMask);
                CastFloat2Half<InputT>(roundedReg, outputZeroReg, outputOneReg, floatMask);
                CastHalf2Float<InputT>(outputZeroReg, outputOneReg, roundedReg, inputMask);
                Mul(outputZeroReg, outputZeroReg, betaReg, floatMask);
                Mul(outputOneReg, outputOneReg, betaReg, floatMask);
                StoreKdaGateRegbasePair<InputT>(
                    wDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            }
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           posZeroReg, posOneReg, floatMask);
            if constexpr (SCALE_SCORE_W) {
                Mul(outputZeroReg, outputZeroReg, betaReg, floatMask);
                Mul(outputOneReg, outputOneReg, betaReg, floatMask);
            }
            StoreKdaGateRegbasePair<OutputT>(
                wOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           negZeroReg, negOneReg, floatMask);
            if constexpr (USE_REF) {
                if (row >= validRows) {
                    Duplicate(outputZeroReg, 0.0f, floatMask);
                    Duplicate(outputOneReg, 0.0f, floatMask);
                }
            }
            StoreKdaGateRegbasePair<OutputT>(
                kgOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);

            if constexpr (EXPORT_FINAL_KG) {
                RegTensor<float> finalRefZeroReg;
                RegTensor<float> finalRefOneReg;
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    finalRefZeroReg, finalRefOneReg, finalRef + col);
                SubFloatTwoReg(finalNegZeroReg, finalNegOneReg,
                               finalRefZeroReg, finalRefOneReg,
                               gateZeroReg, gateOneReg, floatMask);
                Muls(finalNegZeroReg, finalNegZeroReg, LN2, floatMask);
                Muls(finalNegOneReg, finalNegOneReg, LN2, floatMask);
                MinsFloatTwoReg(finalNegZeroReg, finalNegOneReg,
                                finalNegZeroReg, finalNegOneReg,
                                KDA_EXP_INPUT_MAX, floatMask);
                Maxs(finalNegZeroReg, finalNegZeroReg, KDA_EXP_INPUT_MIN, floatMask);
                Maxs(finalNegOneReg, finalNegOneReg, KDA_EXP_INPUT_MIN, floatMask);
                ExpFloatTwoReg(finalNegZeroReg, finalNegOneReg,
                               finalNegZeroReg, finalNegOneReg, floatMask);
                MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                               finalNegZeroReg, finalNegOneReg, floatMask);
                StoreKdaGateRegbasePair<InputT>(
                    finalKgOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            }
        }
    }
}

static __simd_vf__ inline void ForwardSubDiag16Regbase(__ubuf__ float *diag, uint16_t valid)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t DIAG_SIZE = KDA_SOLVE_DIAG_BT;
    uint32_t activeCount = DIAG_SIZE;
    MaskReg rowMask = UpdateMask<float>(activeCount);

    for (uint16_t row = 2; row < valid; ++row) {
        RegTensor<float> currentReg;
        RegTensor<float> scaleReg;
        RegTensor<float> matrixReg;
        RegTensor<float> productReg;
        RegTensor<float> sumReg;
        LoadAlign(currentReg, diag + static_cast<uint32_t>(row) * DIAG_SIZE);
        Duplicate(sumReg, 0.0f, rowMask);

        for (uint16_t sourceRow = 0; sourceRow < row; ++sourceRow) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(
                scaleReg, diag + static_cast<uint32_t>(row) * DIAG_SIZE + sourceRow);
            LoadAlign(matrixReg, diag + static_cast<uint32_t>(sourceRow) * DIAG_SIZE);
            Mul(productReg, matrixReg, scaleReg, rowMask);
            Add(sumReg, sumReg, productReg, rowMask);
        }
        Add(currentReg, currentReg, sumReg, rowMask);
        StoreAlign(diag + static_cast<uint32_t>(row) * DIAG_SIZE, currentReg, rowMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

static __simd_vf__ inline void SelectCausalRows64Regbase(
    __ubuf__ float *aqk, __ubuf__ float *akk, uint16_t rowBegin, uint16_t rowCount)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ROW_ELEMENTS = 64;

    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t localRow = 0; localRow < rowCount; ++localRow) {
        const uint16_t row = rowBegin + localRow;
        const uint32_t rowOffset = static_cast<uint32_t>(localRow) * ROW_ELEMENTS;
        RegTensor<float> zeroReg;
        RegTensor<float> aqkInputReg;
        RegTensor<float> akkInputReg;
        RegTensor<float> aqkReg;
        RegTensor<float> akkReg;
        uint32_t aqkCount = static_cast<uint32_t>(row) + 1;
        uint32_t akkCount = static_cast<uint32_t>(row);
        MaskReg aqkMask = UpdateMask<float>(aqkCount);
        MaskReg akkMask = UpdateMask<float>(akkCount);
        Duplicate(zeroReg, 0.0f, fullMask);
        LoadAlign(aqkInputReg, aqk + rowOffset);
        LoadAlign(akkInputReg, akk + rowOffset);
        Select(aqkReg, aqkInputReg, zeroReg, aqkMask);
        Select(akkReg, akkInputReg, zeroReg, akkMask);
        StoreAlign(aqk + rowOffset, aqkReg, fullMask);
        StoreAlign(akk + rowOffset, akkReg, fullMask);
    }
}

static __simd_vf__ inline void ForwardSubDiag16StridedRegbase(
    __ubuf__ float *matrix, uint16_t rowStride, uint16_t rowBegin, uint16_t colBegin,
    uint16_t valid)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t DIAG_SIZE = KDA_SOLVE_DIAG_BT;
    uint32_t activeCount = DIAG_SIZE;
    MaskReg rowMask = UpdateMask<float>(activeCount);

    for (uint16_t row = 2; row < valid; ++row) {
        uint32_t currentOffset =
            static_cast<uint32_t>(rowBegin + row) * rowStride + colBegin;
        RegTensor<float> currentReg;
        RegTensor<float> scaleReg;
        RegTensor<float> matrixReg;
        RegTensor<float> productReg;
        RegTensor<float> sumReg;
        LoadAlign(currentReg, matrix + currentOffset);
        Duplicate(sumReg, 0.0f, rowMask);

        for (uint16_t sourceRow = 0; sourceRow < row; ++sourceRow) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(
                scaleReg, matrix + currentOffset + sourceRow);
            uint32_t sourceOffset =
                static_cast<uint32_t>(rowBegin + sourceRow) * rowStride + colBegin;
            LoadAlign(matrixReg, matrix + sourceOffset);
            Mul(productReg, matrixReg, scaleReg, rowMask);
            Add(sumReg, sumReg, productReg, rowMask);
        }
        Add(currentReg, currentReg, sumReg, rowMask);
        StoreAlign(matrix + currentOffset, currentReg, rowMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

static __simd_vf__ inline void ForwardSubDiag16PairStridedRegbase(
    __ubuf__ float *matrix, uint16_t rowStride, uint16_t colBegin,
    uint16_t firstValid, uint16_t secondValid)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t DIAG_SIZE = KDA_SOLVE_DIAG_BT;
    constexpr uint16_t SECOND_LOCAL_ROW = DIAG_SIZE;
    uint32_t activeCount = DIAG_SIZE;
    MaskReg rowMask = UpdateMask<float>(activeCount);

    for (uint16_t row = 2; row < DIAG_SIZE; ++row) {
        uint32_t firstCurrentOffset =
            static_cast<uint32_t>(row) * rowStride + colBegin;
        uint32_t secondCurrentOffset =
            static_cast<uint32_t>(SECOND_LOCAL_ROW + row) * rowStride +
            colBegin + DIAG_SIZE;
        RegTensor<float> firstCurrentReg;
        RegTensor<float> secondCurrentReg;
        RegTensor<float> firstSumReg;
        RegTensor<float> secondSumReg;
        LoadAlign(firstCurrentReg, matrix + firstCurrentOffset);
        LoadAlign(secondCurrentReg, matrix + secondCurrentOffset);
        Duplicate(firstSumReg, 0.0f, rowMask);
        Duplicate(secondSumReg, 0.0f, rowMask);

        if (row < firstValid || row < secondValid) {
            for (uint16_t sourceRow = 0; sourceRow < row; ++sourceRow) {
                RegTensor<float> firstScaleReg;
                RegTensor<float> secondScaleReg;
                RegTensor<float> firstMatrixReg;
                RegTensor<float> secondMatrixReg;
                RegTensor<float> firstProductReg;
                RegTensor<float> secondProductReg;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    firstScaleReg, matrix + firstCurrentOffset + sourceRow);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    secondScaleReg, matrix + secondCurrentOffset + sourceRow);
                uint32_t firstSourceOffset =
                    static_cast<uint32_t>(sourceRow) * rowStride + colBegin;
                uint32_t secondSourceOffset =
                    static_cast<uint32_t>(SECOND_LOCAL_ROW + sourceRow) * rowStride +
                    colBegin + DIAG_SIZE;
                LoadAlign(firstMatrixReg, matrix + firstSourceOffset);
                LoadAlign(secondMatrixReg, matrix + secondSourceOffset);
                Mul(firstProductReg, firstMatrixReg, firstScaleReg, rowMask);
                Mul(secondProductReg, secondMatrixReg, secondScaleReg, rowMask);
                Add(firstSumReg, firstSumReg, firstProductReg, rowMask);
                Add(secondSumReg, secondSumReg, secondProductReg, rowMask);
            }
        }
        if (row < firstValid) {
            Add(firstCurrentReg, firstCurrentReg, firstSumReg, rowMask);
            StoreAlign(matrix + firstCurrentOffset, firstCurrentReg, rowMask);
        }
        if (row < secondValid) {
            Add(secondCurrentReg, secondCurrentReg, secondSumReg, rowMask);
            StoreAlign(matrix + secondCurrentOffset, secondCurrentReg, rowMask);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    RegTensor<int32_t> indexReg;
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(indexReg, 0);
    for (uint16_t row = 0; row < DIAG_SIZE; ++row) {
        MaskReg diagMask;
        CompareScalar<int32_t, CMPMODE::EQ>(
            diagMask, indexReg, static_cast<int32_t>(row), rowMask);
        uint32_t firstOffset = static_cast<uint32_t>(row) * rowStride + colBegin;
        uint32_t secondOffset =
            static_cast<uint32_t>(SECOND_LOCAL_ROW + row) * rowStride +
            colBegin + DIAG_SIZE;
        RegTensor<float> firstReg;
        RegTensor<float> secondReg;
        LoadAlign(firstReg, matrix + firstOffset);
        LoadAlign(secondReg, matrix + secondOffset);
        Adds(firstReg, firstReg, 1.0f, diagMask);
        Adds(secondReg, secondReg, 1.0f, diagMask);
        StoreAlign(matrix + firstOffset, firstReg, rowMask);
        StoreAlign(matrix + secondOffset, secondReg, rowMask);
    }
}

static __simd_vf__ inline void ApplyKdaRowScaleRegbase(
    __ubuf__ float *matrix, __ubuf__ float *rowScale, uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> matrixReg0;
    RegTensor<float> matrixReg1;
    RegTensor<float> scaleReg0;
    RegTensor<float> scaleReg1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg1, rowScale + row + 1);
        for (uint16_t col = 0; col < cols; col += FP32_PER_REG) {
            uint32_t activeCount0 = static_cast<uint32_t>(cols - col);
            uint32_t activeCount1 = activeCount0;
            MaskReg mask0 = UpdateMask<float>(activeCount0);
            MaskReg mask1 = UpdateMask<float>(activeCount1);
            uint32_t offset0 = static_cast<uint32_t>(row) * cols + col;
            uint32_t offset1 = static_cast<uint32_t>(row + 1) * cols + col;
            LoadAlign(matrixReg0, matrix + offset0);
            LoadAlign(matrixReg1, matrix + offset1);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask0);
            Mul(matrixReg1, matrixReg1, scaleReg1, mask1);
            StoreAlign(matrix + offset0, matrixReg0, mask0);
            StoreAlign(matrix + offset1, matrixReg1, mask1);
        }
    }
    if (row < rows) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + row);
        for (uint16_t col = 0; col < cols; col += FP32_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg mask = UpdateMask<float>(activeCount);
            uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadAlign(matrixReg0, matrix + offset);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask);
            StoreAlign(matrix + offset, matrixReg0, mask);
        }
    }
}
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaArchTag = Catlass::Arch::Ascend950;
#else
using KdaArchTag = Catlass::Arch::AtlasA2;
#endif
// Aqk and Akk share the right operand; the resident-address check reuses
// each L1 B tile across the two consecutive products.
using KdaScoreDispatchPolicy =
    Catlass::Gemm::MmadPingpongTlaMulti<KdaArchTag, false, false, 2, true, 2, 1, 2, 2>;
static_assert(KdaScoreDispatchPolicy::ENABLE_L1_RESIDENT,
              "KDA Aqk/Akk score MMAD must keep the shared right matrix resident in L1");
static_assert(KdaScoreDispatchPolicy::L1B_STAGES == 1,
              "KDA Aqk/Akk score MMAD reuses one resident L1 B stage");
// Solve 的每次 Cube 输出都会立即作为下一次 MTE2 的 GM 输入，RAW 依赖会
// 串行化相邻调用；这里明确使用单 L0C，避免把单次调用误标成双缓冲。
using KdaSolveDispatchPolicy = Common::MmadPingpong<KdaArchTag, false, false, 1>;
static_assert(!KdaSolveDispatchPolicy::USE_HF32_MODE, "KDA triangular solve must use IEEE FP32 Cube mode");
using KdaL1TileShape = tla::Shape<KdaInt64, KdaInt128, KdaInt128>;
using KdaL0TileShape = tla::Shape<KdaInt64, KdaInt128, KdaInt128>;
using KdaSolveL1TileShape = tla::Shape<KdaInt64, KdaInt64, KdaInt64>;
using KdaSolveL0TileShape = KdaSolveL1TileShape;

__aicore__ inline uint32_t FloatToBits(float value)
{
    union Bits {
        __aicore__ Bits() {}
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

__aicore__ inline float BitsToFloat(uint32_t value)
{
    union Bits {
        __aicore__ Bits() {}
        uint32_t u;
        float f;
    } bits;
    bits.u = value;
    return bits.f;
}

__aicore__ inline uint16_t Bf16ToBits(bfloat16_t value)
{
    union Bits {
        __aicore__ Bits() {}
        bfloat16_t f;
        uint16_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

__aicore__ inline bfloat16_t BitsToBf16(uint16_t value)
{
    union Bits {
        __aicore__ Bits() {}
        uint16_t u;
        bfloat16_t f;
    } bits;
    bits.u = value;
    return bits.f;
}

template <typename T>
__aicore__ inline T FloatToType(float value)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        uint32_t bits = FloatToBits(value);
        uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
        return BitsToBf16(static_cast<uint16_t>((bits + bias) >> 16));
    }
    return static_cast<T>(value);
}

template <bool SAFE_GATE, typename T,
          typename GK_T = float, typename BETA_T = float,
          typename A_LOG_T = float, typename DT_BIAS_T = float,
          uint32_t COMPILE_BT = 0, uint32_t COMPILE_K = 0, uint32_t COMPILE_V = 0>
class ChunkKdaFwdPrepareKernel {
public:
    using OUT_T = T;
    using AKK_T = float;
    using SCORE_T =
        std::conditional_t<SAFE_GATE && IsSameType<T, half>::value, bfloat16_t, T>;
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR rawG,
                                GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR beta, GM_ADDR initialState,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR compactPlan,
                                GM_ADDR preparedQG, GM_ADDR preparedAqk,
                                GM_ADDR propagatedVNew, GM_ADDR propagatedH, GM_ADDR o, GM_ADDR finalState, GM_ADDR aqk,
                                GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
                                GM_ADDR finalKg, GM_ADDR workspace, const TilingData &tiling, TPipe *pipe,
                                bool initVecBuffers = true, bool storeQG = true)
    {
        pipe_ = pipe;
        q_.SetGlobalBuffer((__gm__ T *)q);
        k_.SetGlobalBuffer((__gm__ T *)k);
        v_.SetGlobalBuffer((__gm__ T *)v);
        gk_.SetGlobalBuffer((__gm__ GK_T *)gk);
        rawG_.SetGlobalBuffer((__gm__ float *)rawG);
        aLog_.SetGlobalBuffer((__gm__ A_LOG_T *)aLog);
        dtBias_.SetGlobalBuffer((__gm__ DT_BIAS_T *)dtBias);
        beta_.SetGlobalBuffer((__gm__ BETA_T *)beta);
        if (initialState != nullptr) {
            initialState_.SetGlobalBuffer((__gm__ float *)initialState);
        }
        cuSeqlensAddr_ = reinterpret_cast<__gm__ int64_t *>(cuSeqlens);
        compactPlanAddr_ = compactPlan;
        if (preparedQG != nullptr) {
            preparedQG_.SetGlobalBuffer((__gm__ T *)preparedQG);
        }
        if (preparedAqk != nullptr) {
            preparedAqk_.SetGlobalBuffer((__gm__ T *)preparedAqk);
        }
        if (propagatedVNew != nullptr) {
            propagatedVNew_.SetGlobalBuffer((__gm__ T *)propagatedVNew);
        }
        if (propagatedH != nullptr) {
            propagatedH_.SetGlobalBuffer((__gm__ T *)propagatedH);
        }
        chunkIndicesAddr_ = reinterpret_cast<__gm__ int64_t *>(chunkIndices);
        o_.SetGlobalBuffer((__gm__ OUT_T *)o);
        finalState_.SetGlobalBuffer((__gm__ float *)finalState);
        aqk_.SetGlobalBuffer((__gm__ float *)aqk);
        akk_.SetGlobalBuffer((__gm__ AKK_T *)akk);
        w_.SetGlobalBuffer((__gm__ T *)w);
        u_.SetGlobalBuffer((__gm__ OUT_T *)u);
        qg_.SetGlobalBuffer((__gm__ T *)qg);
        kg_.SetGlobalBuffer((__gm__ T *)kg);
        vNew_.SetGlobalBuffer((__gm__ T *)vNew);
        h_.SetGlobalBuffer((__gm__ float *)h);
        finalKg_.SetGlobalBuffer((__gm__ T *)finalKg);
        solveWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);

        B_ = tiling.batch;
        N_ = tiling.seqNum;
        H_ = tiling.qHeadNum;
        HV_ = tiling.vHeadNum;
        T_ = tiling.seqlen;
        K_ = COMPILE_K == 0 ? tiling.kHeadDim : COMPILE_K;
        V_ = COMPILE_V == 0 ? tiling.vHeadDim : COMPILE_V;
        BT_ = COMPILE_BT == 0 ? tiling.chunkSize : COMPILE_BT;
        NT_ = tiling.totalChunks;
        scale_ = tiling.scale;
        hasInitial_ = tiling.hasInitialState;
        isVarLen_ = tiling.isVarLen;
        inputSequenceMajor_ = tiling.inputSequenceMajor;
        fusePostWu_ = tiling.fusePostWu;
        computeGateInPrepare_ = tiling.computeGateInPrepare;
        hasALog_ = tiling.hasALog;
        hasDtBias_ = tiling.hasDtBias;
        lowerBound_ = tiling.lowerBound;
        storeQG_ = storeQG;
        usedCoreNum_ = tiling.prepareUsedCoreNum;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // 只有确认仅访问一次GM的数据流才绕过L2。V、beta和K会被后续
        // Prepare模块复用，必须保留NORMAL策略。
        if (computeGateInPrepare_) {
            rawG_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if (H_ == HV_) {
            q_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
#endif
        constexpr uint64_t solvePipelineDepth = SAFE_GATE ? KDA_SOLVE_PIPELINE_DEPTH : 1;
        const uint64_t solveBytes =
            usedCoreNum_ * solvePipelineDepth * KDA_SOLVE_SCRATCH_SLOTS * BT_ * BT_ * sizeof(float);
        const uint64_t alignedSolveBytes =
            (solveBytes + KDA_WORKSPACE_ALIGN - 1) / KDA_WORKSPACE_ALIGN * KDA_WORKSPACE_ALIGN;
        scoreWorkspace_.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + alignedSolveBytes));
        if ASCEND_IS_AIV {
            uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            solveCoreIdx_ = subBlockNum == 0 ? 0 : static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        } else {
            solveCoreIdx_ = static_cast<uint64_t>(GetBlockIdx());
        }
        if (pipe_ != nullptr && initVecBuffers) {
            pipe_->InitBuffer(exp2Buf_, EXP2_UB_BYTES);
            pipe_->InitBuffer(vecBuf_, KDA_VEC_ARENA_ELEMENTS * sizeof(float));
            const uint64_t gateStageElems = GatePipelineRows() * K_;
            const uint64_t gateInputSlotBytes = GateInputSlotBytes();
            const uint64_t gatePipelineBytes =
                GateBufferDepth() * (gateInputSlotBytes + gateStageElems * sizeof(T));
            pipe_->InitBuffer(gateWritebackBuf_, static_cast<uint32_t>(gatePipelineBytes));
            AllocVectorEvents();
        }
    }
    __aicore__ inline void ProcessAivOnly()
    {
        isAivOnly_ = true;
        ProcessPreAiv();
        ReleaseVectorEvents();
    }

    __aicore__ inline void ProcessAiv()
    {
        ProcessPreAiv();
        ReleaseVectorEvents();
    }

    __aicore__ inline void ProcessAic()
    {
        AllocSolveCubeEvent();
        ProcessPreAic();
        ReleaseSolveCubeEvent();
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessAicFused(PostWuOp &postWu)
    {
        AllocSolveCubeEvent();
        ProcessPreAicHeadWindowsFused(postWu);
        ReleaseSolveCubeEvent();
    }

private:
    __aicore__ inline void AllocSolveCubeEvent()
    {
        // Solve 的 BlockMmad 会在同一次 AIC 任务内反复写回 GM，事件在任务级
        // 生命周期中只分配一次，避免每个 helper 重复占用有限的 EventID。
        solveFixToMte2Event_ = pipe_->AllocEventID<HardEvent::FIX_MTE2>();
    }

    __aicore__ inline void ReleaseSolveCubeEvent()
    {
        pipe_->ReleaseEventID<HardEvent::FIX_MTE2>(solveFixToMte2Event_);
    }

    __aicore__ inline void AllocVectorEvents()
    {
        mte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Event_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        mte2ToMte3Event_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        vToSEvent_ = pipe_->AllocEventID<HardEvent::V_S>();
        for (uint32_t slot = 0; slot < KDA_GATE_PIPELINE_DEPTH; ++slot) {
            mte3ToMte2Events_[slot] = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        }
        vectorEventsAllocated_ = true;
    }

    __aicore__ inline void ReleaseVectorEvents()
    {
        if (!vectorEventsAllocated_) {
            return;
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(vToMte2Event_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Event_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        pipe_->ReleaseEventID<HardEvent::V_S>(vToSEvent_);
        for (uint32_t slot = 0; slot < KDA_GATE_PIPELINE_DEPTH; ++slot) {
            pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
        }
        vectorEventsAllocated_ = false;
    }

    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t, uint64_t d) const
    {
        if (inputSequenceMajor_) {
            return ((b * T_ + t) * H_ + h) * K_ + d;
        }
        return ((b * H_ + h) * T_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t VInputOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d) const
    {
        if (inputSequenceMajor_) {
            return ((b * T_ + t) * HV_ + hv) * V_ + d;
        }
        return ((b * HV_ + hv) * T_ + t) * V_ + d;
    }

    __aicore__ inline uint64_t RawGateOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d) const
    {
        if (inputSequenceMajor_) {
            return ((b * T_ + t) * HV_ + hv) * K_ + d;
        }
        return ((b * HV_ + hv) * T_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t KVOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d, uint64_t dim) const
    {
        return ((b * HV_ + hv) * T_ + t) * dim + d;
    }

    __aicore__ inline uint64_t BetaOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return (b * HV_ + hv) * T_ + t;
    }

    __aicore__ inline uint64_t AOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t j) const
    {
        return ((b * HV_ + hv) * T_ + t) * BT_ + j;
    }

    __aicore__ inline uint64_t HOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t d, uint64_t r) const
    {
        return (((b * HV_ + hv) * NT_ + chunkIdx) * K_ + d) * V_ + r;
    }

    __aicore__ inline uint64_t WScratchOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t t, uint64_t d) const
    {
        return (((b * HV_ + hv) * NT_ + chunkIdx) * BT_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t SolveScratchOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                  uint64_t slot) const
    {
        (void)b;
        (void)hv;
        (void)chunkIdx;
        constexpr uint64_t solvePipelineDepth = SAFE_GATE ? KDA_SOLVE_PIPELINE_DEPTH : 1;
        uint64_t matrixElements = BT_ * BT_;
        return ((solveCoreIdx_ * solvePipelineDepth + activeSolveSlot_) * KDA_SOLVE_SCRATCH_SLOTS + slot) *
               matrixElements;
    }

    __aicore__ inline uint64_t ScoreScratchOffset(uint64_t slot, uint64_t plane, uint64_t t = 0,
                                                  uint64_t d = 0) const
    {
        return (((solveCoreIdx_ * KDA_SCORE_SCRATCH_SLOTS + slot) * KDA_SCORE_SCRATCH_PLANES + plane) * BT_ + t) *
                   K_ +
               d;
    }

    __aicore__ inline uint64_t ScoreRefBlockSize() const
    {
        if constexpr (SAFE_GATE) {
            return KDA_SAFE_SCORE_REF_BC;
        }
        return KDA_SCORE_REF_BC;
    }

    __aicore__ inline uint64_t ScoreRowBlockCount(uint64_t curT, uint64_t rowBegin) const
    {
        uint64_t blockSize = ScoreRefBlockSize();
        uint64_t rowCount = curT - rowBegin;
        if (rowCount > blockSize) {
            rowCount = blockSize;
        }
        return rowCount;
    }

    __aicore__ inline uint64_t ScoreRefToken(uint64_t start, uint64_t curT, uint64_t rowBegin,
                                             uint64_t rowCount) const
    {
        uint64_t ref = rowBegin + rowCount / 2;
        if (ref >= curT) {
            ref = curT - 1;
        }
        return start + ref;
    }

    __aicore__ inline void RunExp2(LocalTensor<float> &tensor, uint32_t count)
    {
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        ClampExpInput(tensor, count);
        Exp(tensor, tensor, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
    }

    __aicore__ inline void ClampExpInput(LocalTensor<float> &tensor, uint32_t count)
    {
        Mins(tensor, tensor, KDA_EXP_INPUT_MAX, count);
        PipeBarrier<PIPE_V>();
        Maxs(tensor, tensor, KDA_EXP_INPUT_MIN, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ClampScoreExpInput(LocalTensor<float> &tensor, uint32_t count)
    {
        constexpr float expInputMax =
            IsSameType<SCORE_T, bfloat16_t>::value ? KDA_SCORE_EXP_INPUT_MAX : KDA_EXP_INPUT_MAX;
        constexpr float expInputMin =
            IsSameType<SCORE_T, bfloat16_t>::value ? KDA_SCORE_EXP_INPUT_MIN : KDA_EXP_INPUT_MIN;
        Mins(tensor, tensor, expInputMax, count);
        PipeBarrier<PIPE_V>();
        Maxs(tensor, tensor, expInputMin, count);
        PipeBarrier<PIPE_V>();
    }

    template <typename OutputT>
    __aicore__ inline void ClampFp32ForCast(LocalTensor<float> &tensor, uint32_t count)
    {
        if constexpr (IsSameType<OutputT, half>::value) {
            Mins(tensor, tensor, KDA_FP16_MAX, count);
            PipeBarrier<PIPE_V>();
            Maxs(tensor, tensor, -KDA_FP16_MAX, count);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ClampFp32ToOutputType(LocalTensor<float> &tensor, uint32_t count)
    {
        ClampFp32ForCast<T>(tensor, count);
    }

    template <typename CopyT>
    __aicore__ inline void CopyVectorIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src, uint64_t offset,
                                        uint64_t count)
    {
        uint64_t rowBytes = count * static_cast<uint64_t>(sizeof(CopyT));
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
            return;
        }
        DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], params, padParams);
    }

    template <typename CopyT>
    __aicore__ inline void CopyRowsIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src,
                                      uint64_t offset, uint64_t rows, uint64_t cols,
                                      uint64_t rowStride)
    {
        if (rows == 0 || cols == 0) {
            return;
        }
        if (rowStride == cols) {
            CopyVectorIn(dst, src, offset, rows * cols);
            return;
        }
        DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(CopyT)),
            static_cast<uint32_t>((rowStride - cols) * sizeof(CopyT)),
            0,
            0};
        DataCopyPadExtParams<CopyT> padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], params, padParams);
    }

    template <typename CopyT>
    __aicore__ inline void CopyVectorOut(GlobalTensor<CopyT> &dst, uint64_t offset, LocalTensor<CopyT> &src,
                                         uint64_t count)
    {
        uint64_t rowBytes = count * static_cast<uint64_t>(sizeof(CopyT));
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst[offset], src, static_cast<uint32_t>(count));
            return;
        }
        DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
        DataCopyPad(dst[offset], src, params);
    }

    template <typename CopyT>
    __aicore__ inline void CopyRowIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src, uint64_t offset)
    {
        CopyVectorIn(dst, src, offset, K_);
    }

    template <typename CopyT>
    __aicore__ inline void CopyRowOut(GlobalTensor<CopyT> &dst, uint64_t offset, LocalTensor<CopyT> &src)
    {
        CopyVectorOut(dst, offset, src, K_);
    }

    __aicore__ inline LocalTensor<float> VecScratch(uint64_t slot)
    {
        return vecBuf_.Get<float>()[slot * EXP2_UB_ELEMENTS];
    }

    __aicore__ inline uint64_t GateStageElems() const
    {
        return GatePipelineRows() * K_;
    }

    __aicore__ inline uint64_t GatePipelineRows() const
    {
        constexpr uint64_t fixedBytes =
            static_cast<uint64_t>(KDA_VEC_ARENA_ELEMENTS) * sizeof(float) + EXP2_UB_BYTES;
        constexpr uint64_t availableBytes = KDA_AIV_UB_BUDGET_BYTES - fixedBytes;
        uint64_t bytesPerRow = K_ * KDA_GATE_PIPELINE_DEPTH * (3 * sizeof(T) + sizeof(GK_T));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            bytesPerRow = GateBufferDepth() *
                          (K_ * (4 * sizeof(T) + sizeof(GK_T)) + sizeof(BETA_T));
        }
#endif
        uint64_t rows = bytesPerRow == 0 ? 0 : availableBytes / bytesPerRow;
        return rows < KDA_GATE_TILE_ROWS ? rows : KDA_GATE_TILE_ROWS;
    }

    __aicore__ inline constexpr uint64_t GateBufferDepth() const
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            return 2;
        }
#endif
        return KDA_GATE_PIPELINE_DEPTH;
    }

    __aicore__ inline uint64_t GateInputSlotBytes() const
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            return GateStageElems() * (3 * sizeof(T) + sizeof(GK_T)) +
                   GatePipelineRows() * sizeof(BETA_T);
        }
#endif
        return GateStageElems() * (2 * sizeof(T) + sizeof(GK_T));
    }

    __aicore__ inline LocalTensor<T> GateQTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes();
        return gateWritebackBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<T> GateKTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes() + GateStageElems() * sizeof(T);
        return gateWritebackBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<GK_T> GateGTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes() + 2 * GateStageElems() * sizeof(T);
        return gateWritebackBuf_.Get<GK_T>()[byteOffset / sizeof(GK_T)];
    }

    __aicore__ inline LocalTensor<T> GateVTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes() +
                              GateStageElems() * (2 * sizeof(T) + sizeof(GK_T));
        return gateWritebackBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<BETA_T> GateBetaTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes() +
                              GateStageElems() * (3 * sizeof(T) + sizeof(GK_T));
        return gateWritebackBuf_.Get<BETA_T>()[byteOffset / sizeof(BETA_T)];
    }

    __aicore__ inline LocalTensor<T> GateKgTyped(uint64_t slot)
    {
        uint64_t byteOffset = GateBufferDepth() * GateInputSlotBytes() +
                              slot * GateStageElems() * sizeof(T);
        return gateWritebackBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<T> RawQResidentArch35()
    {
        constexpr uint64_t byteOffset =
            static_cast<uint64_t>(KDA_QK_RESIDENT_FLOAT_OFFSET) * sizeof(float);
        return vecBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<T> RawKResidentArch35()
    {
        return RawQResidentArch35()[KDA_QK_RESIDENT_ROWS * 128];
    }

    __aicore__ inline void BeginRawQkResidentGroupArch35(
        uint64_t b, uint64_t h, uint64_t start, uint64_t curT,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (COMPILE_BT == 64 && COMPILE_K == 128 &&
                      COMPILE_V == 128) {
            if (subBlockNum == 0 || subBlockIdx >= subBlockNum) {
                rawQkResidentEnabled_ = false;
                return;
            }
            static_assert(sizeof(T) == sizeof(uint16_t),
                          "arch35 raw Q/K resident requires fp16/bf16 inputs");
            static_assert(
                KDA_LOCAL_GK_FLOAT_OFFSET + 64 * 128 <=
                    KDA_SCALED_QG_FLOAT_OFFSET,
                "arch35 local gate overlaps scaled QG");
            static_assert(
                (KDA_GATE_PIPELINE_DEPTH * 3 * KDA_GATE_TILE_ROWS * 128 *
                     sizeof(uint16_t) +
                 KDA_GATE_PIPELINE_DEPTH * KDA_GATE_TILE_ROWS * sizeof(float)) <=
                    KDA_LOCAL_GK_FLOAT_OFFSET * sizeof(float),
                "arch35 direct gate outputs overlap local gate");
            static_assert(
                3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512 +
                        2 * KDA_SOLVE_BT <=
                    KDA_QK_RESIDENT_FLOAT_OFFSET,
                "arch35 solve scratch overlaps raw Q/K resident");
            static_assert(
                20480 * sizeof(float) +
                        KDA_GATE_TILE_ROWS * 128 * sizeof(uint16_t) <=
                    KDA_QK_RESIDENT_FLOAT_OFFSET * sizeof(float),
                "arch35 beta-scale typed scratch overlaps raw Q/K resident");
            static_assert(
                (2 * KDA_QK_RESIDENT_ROWS * KDA_SOLVE_BT +
                 KDA_QK_RESIDENT_ROWS * 128) * sizeof(float) +
                        (2 * KDA_QK_RESIDENT_ROWS * KDA_SOLVE_BT +
                         KDA_QK_RESIDENT_ROWS * 128) * sizeof(uint16_t) <=
                    KDA_QK_RESIDENT_FLOAT_OFFSET * sizeof(float),
                "arch35 finalize scratch overlaps raw Q/K resident");
            static_assert(
                KDA_SCALED_QG_FLOAT_OFFSET + KDA_GATE_TILE_ROWS * 128 <=
                    KDA_QK_RESIDENT_FLOAT_OFFSET,
                "arch35 scaled QG overlaps raw Q/K resident");
            static_assert(
                KDA_QK_RESIDENT_FLOAT_OFFSET * sizeof(float) +
                    2 * KDA_QK_RESIDENT_ROWS * 128 * sizeof(uint16_t) <=
                    KDA_SELECT_AQK_MASK_BYTE_OFFSET,
                "arch35 raw Q/K resident overlaps the causal-mask arena");
            // 下一组会从offset 0覆盖resident；先闭合上一组V读到MTE2写的
            // WAR依赖。每个qHead组只在这里从GM搬入一次所属行区间。
            if (rawQkResidentHasVectorReader_) {
                SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
                rawQkResidentHasVectorReader_ = false;
            }
            rawQkResidentEnabled_ = subBlockNum == KDA_SCORE_LANES &&
                                    subBlockIdx < subBlockNum && curT <= BT_;
            const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
            const uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
            rawQkResidentEnabled_ = rawQkResidentEnabled_ &&
                                    rowBegin < rowEnd &&
                                    rowEnd - rowBegin <= KDA_QK_RESIDENT_ROWS;
            rawQkResidentB_ = b;
            rawQkResidentH_ = h;
            rawQkResidentTokenBegin_ = start + rowBegin;
            rawQkResidentTokenEnd_ = start + rowEnd;
            if (rawQkResidentEnabled_) {
                const uint64_t rows = rowEnd - rowBegin;
                LocalTensor<T> qResident = RawQResidentArch35();
                LocalTensor<T> kResident = RawKResidentArch35();
                CopyRowsIn(qResident, q_,
                           QOffset(b, h, start + rowBegin, 0), rows, K_,
                           inputSequenceMajor_ ? H_ * K_ : K_);
                CopyRowsIn(kResident, k_,
                           QOffset(b, h, start + rowBegin, 0), rows, K_,
                           inputSequenceMajor_ ? H_ * K_ : K_);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            }
            return;
        }
#endif
        (void)b;
        (void)h;
        (void)start;
        (void)curT;
        (void)subBlockIdx;
        (void)subBlockNum;
        rawQkResidentEnabled_ = false;
    }

    __aicore__ inline bool RawQkResidentContainsArch35(
        uint64_t b, uint64_t h, uint64_t token, uint64_t rows) const
    {
        return rawQkResidentEnabled_ && b == rawQkResidentB_ &&
               h == rawQkResidentH_ && token >= rawQkResidentTokenBegin_ &&
               token + rows <= rawQkResidentTokenEnd_;
    }

    __aicore__ inline bool StageRawQFromResidentArch35(
        LocalTensor<T> qTyped, uint64_t b, uint64_t h,
        uint64_t token, uint64_t rows)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (COMPILE_BT == 64 && COMPILE_K == 128 &&
                      COMPILE_V == 128) {
            if (!RawQkResidentContainsArch35(b, h, token, rows)) {
                return false;
            }
            const uint64_t rowOffset = token - rawQkResidentTokenBegin_;
            const uint32_t elems = static_cast<uint32_t>(rows * K_);
            Adds(qTyped,
                 RawQResidentArch35()[rowOffset * K_],
                 0.0f, elems);
            PipeBarrier<PIPE_V>();
            rawQkResidentHasVectorReader_ = true;
            return true;
        }
#endif
        (void)qTyped;
        (void)b;
        (void)h;
        (void)token;
        (void)rows;
        return false;
    }

    __aicore__ inline bool StageRawKFromResidentArch35(
        LocalTensor<T> kTyped, uint64_t b, uint64_t h,
        uint64_t token, uint64_t rows)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (COMPILE_BT == 64 && COMPILE_K == 128 &&
                      COMPILE_V == 128) {
            if (!RawQkResidentContainsArch35(b, h, token, rows)) {
                return false;
            }
            const uint64_t rowOffset = token - rawQkResidentTokenBegin_;
            Adds(kTyped, RawKResidentArch35()[rowOffset * K_], 0.0f,
                 static_cast<uint32_t>(rows * K_));
            PipeBarrier<PIPE_V>();
            rawQkResidentHasVectorReader_ = true;
            return true;
        }
#endif
        (void)kTyped;
        (void)b;
        (void)h;
        (void)token;
        (void)rows;
        return false;
    }

    __aicore__ inline LocalTensor<GK_T> LocalGateChunk()
    {
        constexpr uint64_t byteOffset =
            static_cast<uint64_t>(KDA_LOCAL_GK_FLOAT_OFFSET) * sizeof(float);
        return vecBuf_.Get<GK_T>()[byteOffset / sizeof(GK_T)];
    }

    __aicore__ inline LocalTensor<GK_T> GateScoreTyped(uint64_t slot, uint64_t tileRow)
    {
        if constexpr (IsSameType<GK_T, float>::value) {
            if (computeGateInPrepare_) {
                return LocalGateChunk()[tileRow * K_];
            }
        }
        return GateGTyped(slot);
    }

    __aicore__ inline void LoadGateScoreRef(
        LocalTensor<float> dst, uint64_t b, uint64_t hv, uint64_t token)
    {
        if constexpr (IsSameType<GK_T, float>::value) {
            if (computeGateInPrepare_) {
                const uint64_t tileRow = token - activeGateChunkStart_;
                Adds(dst, LocalGateChunk()[tileRow * K_], 0.0f, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                return;
            }
        }
        LoadAsFloatRow(gk_, KVOffset(b, hv, token, 0, K_), dst, K_);
    }

    __aicore__ inline void PrefetchQKGate(uint64_t slot, uint64_t b, uint64_t h, uint64_t hv,
                                          uint64_t token, uint64_t elems)
    {
        const uint64_t rows = elems / K_;
        LocalTensor<T> qTyped = GateQTyped(slot);
        LocalTensor<T> kTyped = GateKTyped(slot);
        LocalTensor<GK_T> gateTyped = GateGTyped(slot);
        // 目标路径在一个runtime head窗口内按qHead缓存raw Q/K；gate、V和
        // beta仍逐HV搬入。容量或shape不满足时立即回退到原逐HV GM读取。
        const bool stagedQ = StageRawQFromResidentArch35(
            qTyped, b, h, token, rows);
        const bool stagedK = StageRawKFromResidentArch35(
            kTyped, b, h, token, rows);
        if (!stagedQ) {
            CopyRowsIn(qTyped, q_, QOffset(b, h, token, 0), rows, K_,
                       inputSequenceMajor_ ? H_ * K_ : K_);
        }
        if (!stagedK) {
            CopyRowsIn(kTyped, k_, QOffset(b, h, token, 0), rows, K_,
                       inputSequenceMajor_ ? H_ * K_ : K_);
        }
        if (!computeGateInPrepare_) {
            CopyVectorIn(gateTyped, gk_, KVOffset(b, hv, token, 0, K_), elems);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            LocalTensor<T> vTyped = GateVTyped(slot);
            LocalTensor<BETA_T> betaTyped = GateBetaTyped(slot);
            CopyRowsIn(vTyped, v_, VInputOffset(b, hv, token, 0), rows, V_,
                       inputSequenceMajor_ ? HV_ * V_ : V_);
            CopyVectorIn(betaTyped, beta_, BetaOffset(b, hv, token), rows);
        }
#endif
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline float LoadGateExpA(uint64_t hv)
    {
        if (!hasALog_) {
            return 1.0f;
        }
        LocalTensor<float> scalar = exp2Buf_.Get<float>();
        LoadAsFloatRow(aLog_, hv, scalar, 1);
        Exp(scalar, scalar, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(vToSEvent_);
        WaitFlag<HardEvent::V_S>(vToSEvent_);
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline void PrefetchRawGateTile(
        uint64_t b, uint64_t hv, uint64_t token, uint64_t rows)
    {
        const uint64_t tileRow = token - activeGateChunkStart_;
        LocalTensor<float> gate =
            LocalGateChunk().template ReinterpretCast<float>()[tileRow * K_];
        CopyRowsIn(gate, rawG_, RawGateOffset(b, hv, token, 0), rows, K_,
                   inputSequenceMajor_ ? HV_ * K_ : K_);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    template <bool WRITE_GATE_TO_GM>
    __aicore__ inline void MaterializeRawGateChunkArch35(
        uint64_t b, uint64_t hv, uint64_t start, uint64_t rows)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (!computeGateInPrepare_) {
            return;
        }
        if constexpr (!IsSameType<GK_T, float>::value) {
            return;
        } else {
            activeGateChunkStart_ = start;
            const float expA = LoadGateExpA(hv);
            LocalTensor<float> acc = exp2Buf_.Get<float>();
            LocalTensor<float> bias = exp2Buf_.Get<float>()[K_];
            if (hasDtBias_) {
                LoadAsFloatRow(dtBias_, hv * K_, bias, K_);
            }
            Duplicate(acc, 0.0f, static_cast<uint32_t>(K_));
            PipeBarrier<PIPE_V>();

            const uint64_t tileRows = GatePipelineRows();
            const uint64_t tileCount = (rows + tileRows - 1) / tileRows;
            uint64_t currentRows = rows < tileRows ? rows : tileRows;
            PrefetchRawGateTile(b, hv, start, currentRows);
            for (uint64_t tile = 0; tile < tileCount; ++tile) {
                const uint64_t tileRow = tile * tileRows;
                currentRows = rows - tileRow;
                if (currentRows > tileRows) {
                    currentRows = tileRows;
                }
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

                const uint64_t nextTile = tile + 1;
                if (nextTile < tileCount) {
                    uint64_t nextRows = rows - nextTile * tileRows;
                    if (nextRows > tileRows) {
                        nextRows = tileRows;
                    }
                    // raw gate 直接暂存在整块 chunk arena 的最终行位置。
                    // 这里是对互不重叠行的预取，不是 ping-pong；删除无效
                    // 的 slot 翻转，避免把同一地址包装成“假双缓冲”。
                    PrefetchRawGateTile(
                        b, hv, start + nextTile * tileRows, nextRows);
                }

                LocalTensor<float> gate =
                    LocalGateChunk().template ReinterpretCast<float>()[tileRow * K_];
                if (hasDtBias_) {
                    AccumulateRawSafeGateChunk128Regbase<true>(
                        (__ubuf__ float *)gate.GetPhyAddr(), (__ubuf__ float *)bias.GetPhyAddr(),
                        (__ubuf__ float *)acc.GetPhyAddr(), static_cast<uint16_t>(currentRows),
                        expA, lowerBound_);
                } else {
                    AccumulateRawSafeGateChunk128Regbase<false>(
                        (__ubuf__ float *)gate.GetPhyAddr(), (__ubuf__ float *)bias.GetPhyAddr(),
                        (__ubuf__ float *)acc.GetPhyAddr(), static_cast<uint16_t>(currentRows),
                        expA, lowerBound_);
                }
                PipeBarrier<PIPE_V>();
                if constexpr (WRITE_GATE_TO_GM) {
                    SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
                    WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
                    CopyVectorOut(gk_, KVOffset(b, hv, start + tileRow, 0, K_), gate,
                                  currentRows * K_);
                }
            }
            if constexpr (WRITE_GATE_TO_GM) {
                SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
                WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            }
        }
#else
        (void)b;
        (void)hv;
        (void)start;
        (void)rows;
#endif
    }

    __aicore__ inline LocalTensor<T> GateDirectQ(uint64_t slot)
    {
        return vecBuf_.Get<T>()[slot * 3 * GateStageElems()];
    }

    __aicore__ inline LocalTensor<T> GateDirectW(uint64_t slot)
    {
        return GateDirectQ(slot)[GateStageElems()];
    }

    __aicore__ inline LocalTensor<T> GateDirectV(uint64_t slot)
    {
        return GateDirectQ(slot)[2 * GateStageElems()];
    }

    __aicore__ inline LocalTensor<float> GateBetaFloat(uint64_t slot)
    {
        constexpr uint64_t directBytes =
            KDA_GATE_PIPELINE_DEPTH * 3 * KDA_GATE_TILE_ROWS * COMPILE_K * sizeof(T);
        return vecBuf_.Get<float>()[directBytes / sizeof(float) + slot * KDA_GATE_TILE_ROWS];
    }

    __aicore__ inline void StorePreparedQG(uint64_t b, uint64_t hv, uint64_t token,
                                           LocalTensor<T> directQ, uint64_t elems)
    {
        static_assert(KDA_SCALED_QG_FLOAT_OFFSET + KDA_GATE_TILE_ROWS * 128 <=
                      KDA_QK_RESIDENT_FLOAT_OFFSET);
        const uint64_t offset = KVOffset(b, hv, token, 0, K_);
        if (storeQG_) {
            CopyVectorOut(qg_, offset, directQ, elems);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
        LocalTensor<float> scaledQG = vecBuf_.Get<float>()[KDA_SCALED_QG_FLOAT_OFFSET];
        Cast(scaledQG, directQ, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        Muls(scaledQG, scaledQG, scale_, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        ClampFp32ToOutputType(scaledQG, static_cast<uint32_t>(elems));
        Cast(directQ, scaledQG, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        CopyVectorOut(kg_, offset, directQ, elems);
    }

    __aicore__ inline void PrefetchKGate(uint64_t slot, uint64_t b, uint64_t h, uint64_t hv,
                                         uint64_t token, uint64_t elems)
    {
        const uint64_t rows = elems / K_;
        LocalTensor<T> kTyped = GateQTyped(slot);
        LocalTensor<GK_T> gateTyped = GateGTyped(slot);
        if (!StageRawKFromResidentArch35(
                kTyped, b, h, token, rows)) {
            CopyRowsIn(kTyped, k_, QOffset(b, h, token, 0), rows, K_,
                       inputSequenceMajor_ ? H_ * K_ : K_);
        }
        if (!computeGateInPrepare_) {
            CopyVectorIn(gateTyped, gk_, KVOffset(b, hv, token, 0, K_), elems);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void WaitGateInputReady()
    {
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void WaitGateOutputForMte2(uint64_t slot = 0)
    {
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
    }

    __aicore__ inline void WaitGateOutputForVector()
    {
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void SignalGateOutputDone()
    {
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void SignalGateOutputDoneForMte2(uint64_t slot)
    {
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
    }

    template <typename CopyT>
    __aicore__ inline void LoadAsFloatRow(GlobalTensor<CopyT> &src, uint64_t srcOffset, LocalTensor<float> &dst,
                                          uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            CopyVectorIn(dst, src, srcOffset, count);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Adds(dst, dst, 0.0f, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        } else {
            constexpr uint32_t typedOffset = EXP2_UB_ELEMENTS * sizeof(float) / sizeof(CopyT);
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>()[typedOffset];
            CopyVectorIn(rowLocal, src, srcOffset, count);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(dst, rowLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename CopyT>
    __aicore__ inline void LoadAsFloatVector(GlobalTensor<CopyT> &src, uint64_t srcOffset,
                                              LocalTensor<float> &dst, LocalTensor<CopyT> &typedScratch,
                                              uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            CopyVectorIn(dst, src, srcOffset, count);
        } else {
            CopyVectorIn(typedScratch, src, srcOffset, count);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        if constexpr (!IsSameType<CopyT, float>::value) {
            Cast(dst, typedScratch, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
        }
    }

    template <typename CopyT>
    __aicore__ inline void StoreFloatRow(GlobalTensor<CopyT> &dst, uint64_t dstOffset, LocalTensor<float> &src,
                                         uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(dst, dstOffset, src, count);
        } else {
            constexpr uint32_t typedOffset = EXP2_UB_ELEMENTS * sizeof(float) / sizeof(CopyT);
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>()[typedOffset];
            Cast(rowLocal, src, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(dst, dstOffset, rowLocal, count);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }





    __aicore__ inline LocalTensor<float> Exp2NegG(uint64_t b, uint64_t hv, uint64_t t)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        LoadAsFloatRow(gk_, KVOffset(b, hv, t, 0, K_), exp2Local, K_);
        Muls(exp2Local, exp2Local, -LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }

    __aicore__ inline void PrepareScoreFactorsBulk(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                                    uint64_t subBlockIdx, uint64_t subBlockNum,
                                                    uint64_t refToken, uint64_t scoreRowBegin,
                                                    uint64_t scoreRowCount, uint64_t validColEnd,
                                                    uint64_t finalRefToken, uint64_t scoreSlot)
    {
        LocalTensor<float> refFp32 = exp2Buf_.Get<float>();
        LoadGateScoreRef(refFp32, b, hv, refToken);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        constexpr bool exportFinalKg =
            SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128;
        LocalTensor<float> finalRefFp32 = exp2Buf_.Get<float>()[K_];
        if constexpr (exportFinalKg) {
            LoadGateScoreRef(finalRefFp32, b, hv, finalRefToken);
        }
#endif

        uint64_t qwBegin =
            scoreRowBegin + (scoreRowCount * subBlockIdx) / subBlockNum;
        uint64_t qwEnd =
            scoreRowBegin + (scoreRowCount * (subBlockIdx + 1)) / subBlockNum;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (COMPILE_BT == 64 && COMPILE_K == 128 &&
                      COMPILE_V == 128) {
            if (subBlockNum == KDA_SCORE_LANES) {
                // target路径固定整块行所有权：AIV0负责前半，AIV1负责
                // 后半。score block与owner求交，保证两个AIV从GM读取的
                // raw Q/K行区间不重叠，并可在整个qHead组内驻留复用。
                const uint64_t chunkRows = finalRefToken - start + 1;
                const uint64_t ownerBegin =
                    (chunkRows * subBlockIdx) / subBlockNum;
                const uint64_t ownerEnd =
                    (chunkRows * (subBlockIdx + 1)) / subBlockNum;
                const uint64_t scoreRowEnd = scoreRowBegin + scoreRowCount;
                qwBegin = ownerBegin > scoreRowBegin ? ownerBegin : scoreRowBegin;
                qwEnd = ownerEnd < scoreRowEnd ? ownerEnd : scoreRowEnd;
                if (qwBegin > qwEnd) {
                    qwBegin = qwEnd;
                }
            }
        }
#endif
        uint64_t qwMaxRows = GatePipelineRows();
        bool qwOutputPending = false;
        uint64_t qwSlot = 0;
        if (qwBegin < qwEnd && qwMaxRows > 0) {
            uint64_t firstRows = qwEnd - qwBegin;
            if (firstRows > qwMaxRows) {
                firstRows = qwMaxRows;
            }
            PrefetchQKGate(qwSlot, b, h, hv, start + qwBegin, firstRows * K_);
        }
        for (uint64_t tileRow = qwBegin; tileRow < qwEnd && qwMaxRows > 0; tileRow += qwMaxRows) {
            uint64_t tileRows = qwEnd - tileRow;
            if (tileRows > qwMaxRows) {
                tileRows = qwMaxRows;
            }
            uint64_t elems = tileRows * K_;
            LocalTensor<T> qTyped = GateQTyped(qwSlot);
            LocalTensor<T> kTyped = GateKTyped(qwSlot);
            LocalTensor<SCORE_T> qScore = qTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<SCORE_T> kScore = kTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<GK_T> gateTyped = GateScoreTyped(qwSlot, tileRow);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            LocalTensor<SCORE_T> kgScore =
                GateKgTyped(qwSlot).template ReinterpretCast<SCORE_T>();
            LocalTensor<T> vTyped = GateVTyped(qwSlot);
            LocalTensor<BETA_T> betaTyped = GateBetaTyped(qwSlot);
            LocalTensor<T> directQ = GateDirectQ(qwSlot);
            LocalTensor<T> directW = GateDirectW(qwSlot);
            LocalTensor<T> directV = GateDirectV(qwSlot);
            LocalTensor<float> betaFp32 = GateBetaFloat(qwSlot);
#endif
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> qFp32 = arena;
            LocalTensor<float> kFp32 = arena[elems];
            LocalTensor<float> gFp32 = arena[2 * elems];
            LocalTensor<float> expFp32 = arena[3 * elems];
            LocalTensor<float> outFp32 = arena[4 * elems];
#endif

            WaitGateInputReady();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            Cast(qFp32, qTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            if constexpr (IsSameType<GK_T, float>::value) {
                gFp32 = gateTyped;
            } else {
                Cast(gFp32, gateTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            }
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (IsSameType<BETA_T, float>::value) {
                Adds(betaFp32, betaTyped, 0.0f, static_cast<uint32_t>(tileRows));
            } else {
                Cast(betaFp32, betaTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(tileRows));
            }
            PipeBarrier<PIPE_V>();
#endif
            if (qwOutputPending) {
                WaitGateOutputForMte2();
            }
            uint64_t nextTileRow = tileRow + qwMaxRows;
            if (nextTileRow < qwEnd) {
                uint64_t nextRows = qwEnd - nextTileRow;
                if (nextRows > qwMaxRows) {
                    nextRows = qwMaxRows;
                }
                PrefetchQKGate(qwSlot ^ 1, b, h, hv, start + nextTileRow, nextRows * K_);
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            bool fuseQwKg = SAFE_GATE && BT_ == 64 && K_ == 128 && V_ == 128 && subBlockNum == 1;
            if (fuseQwKg) {
                PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true, true, true>(
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(qScore.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kScore.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kgScore.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directQ.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directW.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(vTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directV.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(vTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(betaFp32.GetPhyAddr()),
                    (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(finalRefFp32.GetPhyAddr()),
                    static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_),
                    static_cast<uint16_t>(tileRows));
            } else {
                PrepareKdaGateQwRegbase<T, SCORE_T, GK_T, true>(
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(qScore.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kScore.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directQ.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directW.GetPhyAddr()),
                    (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                    static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_));
            }
#else
            PipeBarrier<PIPE_V>();
            for (uint64_t row = 0; row < tileRows; ++row) {
                Sub(expFp32[row * K_], gFp32[row * K_], refFp32, static_cast<uint32_t>(K_));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampScoreExpInput(expFp32, static_cast<uint32_t>(elems));
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(outFp32, qFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
            Cast(qScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
            Cast(kScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
#endif

            if (qwOutputPending) {
                WaitGateOutputForVector();
            }
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(
                scoreWorkspace_,
                ScoreScratchOffset(
                    scoreSlot, KDA_SCORE_SCRATCH_QG, tileRow),
                qScore, elems);
            CopyVectorOut(
                scoreWorkspace_,
                ScoreScratchOffset(
                    scoreSlot, KDA_SCORE_SCRATCH_W, tileRow),
                kScore, elems);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if (fuseQwKg) {
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG, tileRow),
                              kgScore, elems);
            }
                StorePreparedQG(b, hv, start + tileRow, directQ, elems);
                CopyVectorOut(w_, KVOffset(b, hv, start + tileRow, 0, K_), directW, elems);
                CopyVectorOut(vNew_, KVOffset(b, hv, start + tileRow, 0, V_), directV,
                              tileRows * V_);
                if constexpr (exportFinalKg) {
                    CopyVectorOut(finalKg_, KVOffset(b, hv, start + tileRow, 0, K_), vTyped, elems);
                }
#endif
            SignalGateOutputDone();
            qwOutputPending = true;
            qwSlot ^= 1;
        }
        if (qwOutputPending) {
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }

        bool fuseQwKg = false;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        fuseQwKg = SAFE_GATE && BT_ == 64 && K_ == 128 && V_ == 128 && subBlockNum == 1;
#endif
        uint64_t kgRows = fuseQwKg ? scoreRowBegin : validColEnd;
        uint64_t kgBegin = (kgRows * subBlockIdx) / subBlockNum;
        uint64_t kgEnd = (kgRows * (subBlockIdx + 1)) / subBlockNum;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (COMPILE_BT == 64 && COMPILE_K == 128 &&
                      COMPILE_V == 128) {
            if (subBlockNum == KDA_SCORE_LANES) {
                const uint64_t chunkRows = finalRefToken - start + 1;
                const uint64_t ownerBegin =
                    (chunkRows * subBlockIdx) / subBlockNum;
                const uint64_t ownerEnd =
                    (chunkRows * (subBlockIdx + 1)) / subBlockNum;
                kgBegin = ownerBegin < kgRows ? ownerBegin : kgRows;
                kgEnd = ownerEnd < kgRows ? ownerEnd : kgRows;
                if (kgBegin > kgEnd) {
                    kgBegin = kgEnd;
                }
            }
        }
#endif
        uint64_t kgMaxRows = GatePipelineRows();
        bool kgOutputPending = false;
        uint64_t kgSlot = 0;
        if (kgBegin < kgEnd && kgMaxRows > 0) {
            uint64_t firstRows = kgEnd - kgBegin;
            if (firstRows > kgMaxRows) {
                firstRows = kgMaxRows;
            }
            PrefetchKGate(kgSlot, b, h, hv, start + kgBegin, firstRows * K_);
        }
        for (uint64_t tileRow = kgBegin; tileRow < kgEnd && kgMaxRows > 0; tileRow += kgMaxRows) {
            uint64_t tileRows = kgEnd - tileRow;
            if (tileRows > kgMaxRows) {
                tileRows = kgMaxRows;
            }
            uint64_t elems = tileRows * K_;
            LocalTensor<T> kTyped = GateQTyped(kgSlot);
            LocalTensor<SCORE_T> kgScore = kTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<GK_T> gateTyped = GateScoreTyped(kgSlot, tileRow);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> kFp32 = arena;
            LocalTensor<float> gFp32 = arena[elems];
            LocalTensor<float> expFp32 = arena[2 * elems];
            LocalTensor<float> outFp32 = arena[3 * elems];
#endif

            WaitGateInputReady();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            if constexpr (IsSameType<GK_T, float>::value) {
                gFp32 = gateTyped;
            } else {
                Cast(gFp32, gateTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            }
#endif
            if (kgOutputPending) {
                WaitGateOutputForMte2();
            }
            uint64_t nextTileRow = tileRow + kgMaxRows;
            if (nextTileRow < kgEnd) {
                uint64_t nextRows = kgEnd - nextTileRow;
                if (nextRows > kgMaxRows) {
                    nextRows = kgMaxRows;
                }
                PrefetchKGate(kgSlot ^ 1, b, h, hv, start + nextTileRow, nextRows * K_);
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            PrepareKdaGateKgRegbase<T, SCORE_T, GK_T, true>(
                (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kgScore.GetPhyAddr()),
                (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_),
                static_cast<uint16_t>(tileRows));
#else
            PipeBarrier<PIPE_V>();
            for (uint64_t row = 0; row < tileRows; ++row) {
                Sub(expFp32[row * K_], refFp32, gFp32[row * K_], static_cast<uint32_t>(K_));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampScoreExpInput(expFp32, static_cast<uint32_t>(elems));
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
            Cast(kgScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
#endif

            if (kgOutputPending) {
                WaitGateOutputForVector();
            }
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG, tileRow),
                          kgScore, elems);
            SignalGateOutputDone();
            kgOutputPending = true;
            kgSlot ^= 1;
        }
        if (kgOutputPending) {
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }
    }

    __aicore__ inline void PrepareGateProductsBulk(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                                   uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum,
                                                   bool useRef, uint64_t refToken, uint64_t validColEnd,
                                                   bool writeScoreScratch, uint64_t scoreSlot)
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum || K_ == 0) {
            return;
        }
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        if (rowBegin >= rowEnd) {
            return;
        }

        uint64_t maxRows = GatePipelineRows();
        if (maxRows == 0) {
            return;
        }
        LocalTensor<float> refFp32 = exp2Buf_.Get<float>();
        if (useRef) {
            LoadGateScoreRef(refFp32, b, hv, refToken);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        const bool fuseScoreWriteback =
            writeScoreScratch && useRef && K_ * 2 <= EXP2_UB_ELEMENTS;
#endif

        bool outputPending = false;
        uint64_t gateSlot = 0;
        uint64_t firstRows = rowEnd - rowBegin;
        if (firstRows > maxRows) {
            firstRows = maxRows;
        }
        PrefetchQKGate(gateSlot, b, h, hv, start + rowBegin, firstRows * K_);
        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += maxRows) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > maxRows) {
                tileRows = maxRows;
            }
            uint64_t elems = tileRows * K_;
            LocalTensor<T> qTyped = GateQTyped(gateSlot);
            LocalTensor<T> kTyped = GateKTyped(gateSlot);
            LocalTensor<T> kgTyped = GateKgTyped(gateSlot);
            LocalTensor<SCORE_T> qScore = qTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<SCORE_T> wScore = kTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<SCORE_T> kgScore = kgTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<GK_T> gateTyped = GateScoreTyped(gateSlot, tileRow);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> qFp32 = arena;
            LocalTensor<float> kFp32 = arena[elems];
            LocalTensor<float> gFp32 = arena[2 * elems];
            LocalTensor<float> expFp32 = arena[3 * elems];
            LocalTensor<float> outFp32 = arena[4 * elems];
#else
            LocalTensor<T> vTyped = GateVTyped(gateSlot);
            LocalTensor<BETA_T> betaTyped = GateBetaTyped(gateSlot);
            LocalTensor<T> directQ = GateDirectQ(gateSlot);
            LocalTensor<T> directW = GateDirectW(gateSlot);
            LocalTensor<T> directV = GateDirectV(gateSlot);
            LocalTensor<float> betaFp32 = GateBetaFloat(gateSlot);
#endif

            uint64_t token = start + tileRow;
            WaitGateInputReady();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            Cast(qFp32, qTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            if constexpr (IsSameType<GK_T, float>::value) {
                gFp32 = gateTyped;
            } else {
                Cast(gFp32, gateTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            }
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (IsSameType<BETA_T, float>::value) {
                Adds(betaFp32, betaTyped, 0.0f, static_cast<uint32_t>(tileRows));
            } else {
                Cast(betaFp32, betaTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(tileRows));
            }
            PipeBarrier<PIPE_V>();
#endif
            uint64_t nextTileRow = tileRow + maxRows;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            uint64_t nextGateSlot = (gateSlot + 1) % KDA_GATE_PIPELINE_DEPTH;
            if (nextTileRow < rowEnd) {
                uint64_t tileIndex = (tileRow - rowBegin) / maxRows;
                if (tileIndex + 1 >= KDA_GATE_PIPELINE_DEPTH) {
                    WaitGateOutputForMte2(nextGateSlot);
                }
                uint64_t nextRows = rowEnd - nextTileRow;
                if (nextRows > maxRows) {
                    nextRows = maxRows;
                }
                PrefetchQKGate(nextGateSlot, b, h, hv, start + nextTileRow, nextRows * K_);
            }
#else
            if (outputPending) {
                WaitGateOutputForMte2();
            }
            if (nextTileRow < rowEnd) {
                uint64_t nextRows = rowEnd - nextTileRow;
                if (nextRows > maxRows) {
                    nextRows = maxRows;
                }
                PrefetchQKGate(gateSlot ^ 1, b, h, hv, start + nextTileRow, nextRows * K_);
            }
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            uint16_t validRows = static_cast<uint16_t>(tileRows);
            if (useRef && tileRow >= validColEnd) {
                validRows = 0;
            } else if (useRef && tileRow + tileRows > validColEnd) {
                validRows = static_cast<uint16_t>(validColEnd - tileRow);
            }
            if (writeScoreScratch) {
                if (fuseScoreWriteback) {
                    PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true, false, true>(
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                        (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(qScore.GetPhyAddr()),
                        (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(wScore.GetPhyAddr()),
                        (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kgScore.GetPhyAddr()),
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(directQ.GetPhyAddr()),
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(directW.GetPhyAddr()),
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(vTyped.GetPhyAddr()),
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(directV.GetPhyAddr()),
                        nullptr,
                        (__ubuf__ float *)reinterpret_cast<uint64_t>(betaFp32.GetPhyAddr()),
                        (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                        (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                        nullptr,
                        static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_), validRows);
                } else {
                    PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, false, false>(
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                        (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(qScore.GetPhyAddr()),
                        (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(wScore.GetPhyAddr()),
                        (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kgScore.GetPhyAddr()),
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                        (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                        nullptr,
                        static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_), validRows);
                }
            } else if (useRef) {
                PrepareKdaGateQwKgRegbase<T, T, GK_T, true, false, false>(
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kgTyped.GetPhyAddr()),
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                    nullptr,
                    static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_), validRows);
            } else {
                PrepareKdaGateQwKgRegbase<T, T, GK_T, false, false, false>(
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kgTyped.GetPhyAddr()),
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                    nullptr,
                    static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_), validRows);
            }
#else
            PipeBarrier<PIPE_V>();

            if (useRef) {
                for (uint64_t row = 0; row < tileRows; ++row) {
                    Sub(expFp32[row * K_], gFp32[row * K_], refFp32, static_cast<uint32_t>(K_));
                }
            } else {
                Adds(expFp32, gFp32, 0.0f, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (writeScoreScratch) {
                ClampScoreExpInput(expFp32, static_cast<uint32_t>(elems));
            } else {
                ClampExpInput(expFp32, static_cast<uint32_t>(elems));
            }
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(outFp32, qFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (writeScoreScratch) {
                ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
                Cast(qScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            } else {
                ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
                Cast(qTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();

            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (writeScoreScratch) {
                ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
                Cast(wScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            } else {
                ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
                Cast(kTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();

            if (useRef) {
                for (uint64_t row = 0; row < tileRows; ++row) {
                    Sub(expFp32[row * K_], refFp32, gFp32[row * K_], static_cast<uint32_t>(K_));
                }
            } else {
                Muls(expFp32, gFp32, -1.0f, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (writeScoreScratch) {
                ClampScoreExpInput(expFp32, static_cast<uint32_t>(elems));
            } else {
                ClampExpInput(expFp32, static_cast<uint32_t>(elems));
            }
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (useRef && tileRow + tileRows > validColEnd) {
                for (uint64_t row = 0; row < tileRows; ++row) {
                    if (tileRow + row >= validColEnd) {
                        Duplicate(outFp32[row * K_], 0.0f, static_cast<uint32_t>(K_));
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            if (writeScoreScratch) {
                ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
            } else {
                ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
            }
            if (outputPending) {
                WaitGateOutputForVector();
            }
            if (writeScoreScratch) {
                Cast(kgScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            } else {
                Cast(kgTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();
#endif

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            if (writeScoreScratch) {
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG, tileRow),
                              qScore, elems);
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W, tileRow),
                              wScore, elems);
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG, tileRow),
                              kgScore, elems);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if (fuseScoreWriteback) {
                    StorePreparedQG(b, hv, token, directQ, elems);
                    CopyVectorOut(w_, KVOffset(b, hv, token, 0, K_), directW, elems);
                    CopyVectorOut(vNew_, KVOffset(b, hv, token, 0, V_), directV, tileRows * V_);
                }
#endif
            } else {
                CopyVectorOut(qg_, KVOffset(b, hv, token, 0, K_), qTyped, elems);
                CopyVectorOut(w_, KVOffset(b, hv, token, 0, K_), kTyped, elems);
                CopyVectorOut(kg_, KVOffset(b, hv, token, 0, K_), kgTyped, elems);
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            SignalGateOutputDoneForMte2(gateSlot);
#else
            SignalGateOutputDone();
#endif
            outputPending = true;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            gateSlot = (gateSlot + 1) % KDA_GATE_PIPELINE_DEPTH;
#else
            gateSlot ^= 1;
#endif
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        uint64_t tileCount = (rowEnd - rowBegin + maxRows - 1) / maxRows;
        uint64_t firstPending =
            tileCount > KDA_GATE_PIPELINE_DEPTH ? tileCount - KDA_GATE_PIPELINE_DEPTH : 0;
        for (uint64_t tile = firstPending; tile < tileCount; ++tile) {
            WaitGateOutputForMte2(tile % KDA_GATE_PIPELINE_DEPTH);
        }
#else
        if (outputPending) {
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }
#endif
        return;
    }

    __aicore__ inline void ZeroScoreScratchRange(uint64_t scoreSlot, uint64_t planeBegin,
                                                 uint64_t planeEnd, uint64_t firstRow,
                                                 uint64_t rowEnd)
    {
        if (firstRow >= rowEnd) {
            return;
        }
        const uint64_t maxRows = GatePipelineRows();
        LocalTensor<SCORE_T> zeroLocal = GateQTyped(0).template ReinterpretCast<SCORE_T>();
        for (uint64_t row = firstRow; row < rowEnd; row += maxRows) {
            uint64_t rows = rowEnd - row;
            if (rows > maxRows) {
                rows = maxRows;
            }
            const uint64_t elems = rows * K_;
            Duplicate(zeroLocal, static_cast<SCORE_T>(0), static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            for (uint64_t plane = planeBegin; plane < planeEnd; ++plane) {
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, plane, row),
                              zeroLocal, elems);
            }
            SignalGateOutputDone();
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }
    }

    __aicore__ inline void ZeroScoreScratchPadding(uint64_t scoreSlot,
                                                   uint64_t scoreRowBegin,
                                                   uint64_t scoreRowCount,
                                                   uint64_t validColEnd,
                                                   uint64_t subBlockIdx,
                                                   uint64_t subBlockNum)
    {
        if (subBlockNum == 0 || subBlockIdx + 1 != subBlockNum) {
            return;
        }
        const uint64_t validRowEnd = scoreRowBegin + scoreRowCount;
        const uint64_t paddedRowEnd = (validRowEnd + 15) / 16 * 16;
        const uint64_t paddedColEnd = BT_;
        ZeroScoreScratchRange(scoreSlot, KDA_SCORE_SCRATCH_QG,
                              KDA_SCORE_SCRATCH_KG, validRowEnd, paddedRowEnd);
        ZeroScoreScratchRange(scoreSlot, KDA_SCORE_SCRATCH_KG,
                              KDA_SCORE_SCRATCH_PLANES, validColEnd, paddedColEnd);
    }

    __aicore__ inline void PrepareGateProducts(uint64_t b, uint64_t h, uint64_t hv, uint64_t start, uint64_t curT,
                                               uint64_t subBlockIdx, uint64_t subBlockNum, bool useRef = false,
                                               uint64_t refToken = 0, uint64_t validColEnd = 0,
                                               bool writeScoreScratch = false, uint64_t scoreSlot = 0,
                                               uint64_t scoreRowBegin = 0, uint64_t scoreRowCount = 0)
    {
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum) {
            return;
        }
        if (validColEnd == 0 || validColEnd > curT) {
            validColEnd = curT;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (writeScoreScratch && curT == BT_ && scoreRowBegin == 0 &&
            scoreRowCount == curT && validColEnd == curT) {
            PrepareGateProductsBulk(b, h, hv, start, curT, subBlockIdx, subBlockNum, useRef, refToken,
                                    validColEnd, writeScoreScratch, scoreSlot);
            ZeroScoreScratchPadding(scoreSlot, scoreRowBegin, scoreRowCount, validColEnd,
                                    subBlockIdx, subBlockNum);
            return;
        }
#endif
        if (writeScoreScratch) {
            PrepareScoreFactorsBulk(b, h, hv, start, subBlockIdx, subBlockNum, refToken, scoreRowBegin,
                                    scoreRowCount, validColEnd, start + curT - 1, scoreSlot);
            ZeroScoreScratchPadding(scoreSlot, scoreRowBegin, scoreRowCount, validColEnd,
                                    subBlockIdx, subBlockNum);
            return;
        }
        PrepareGateProductsBulk(b, h, hv, start, curT, subBlockIdx, subBlockNum, useRef, refToken,
                                validColEnd, writeScoreScratch, scoreSlot);
    }

    __aicore__ inline void ComputeRawAqkAkkCube(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                uint64_t start, uint64_t curT)
    {
        ComputeRawAqkAkkCubeBlock(b, hv, chunkIdx, start, curT, 0, curT);
    }


    __aicore__ inline void ComputeRawAqkAkkCubeBlock(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                     uint64_t start, uint64_t curT,
                                                     uint64_t rowBegin, uint64_t rowCount,
                                                     bool readScoreScratch = false, uint64_t scoreSlot = 0,
                                                     uint64_t colCount = 0)
    {
        if (colCount == 0 || colCount > curT) {
            colCount = curT;
        }
        using ElementA = SCORE_T;
        using ElementB = SCORE_T;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaScoreDispatchPolicy, KdaL1TileShape, KdaL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(K_, BT_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(BT_, BT_);
        const bool paddedTail = curT < BT_;
        const uint64_t mmRowCount = paddedTail ? (rowCount + 15) / 16 * 16 : rowCount;
        const uint64_t mmColCount = paddedTail ? BT_ : colCount;
        Catlass::GemmCoord shape{static_cast<uint32_t>(mmRowCount), static_cast<uint32_t>(mmColCount),
                                 static_cast<uint32_t>(K_)};

        (void)readScoreScratch;
        auto tensorQPos =
            tla::MakeTensor(scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG)],
                            layoutA, Catlass::Arch::PositionGM{});
        auto tensorKPos =
            tla::MakeTensor(scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W)],
                            layoutA, Catlass::Arch::PositionGM{});
        auto tensorKNeg =
            tla::MakeTensor(scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG)],
                            layoutB, Catlass::Arch::PositionGM{});
        auto aqkBase = paddedTail
            ? solveWorkspace_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_RAW_AQK)]
            : aqk_[AOffset(b, hv, start, 0)];
        auto akkBase = paddedTail
            ? solveWorkspace_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_RAW_AKK)]
            : akk_[AOffset(b, hv, start, 0)];
        auto tensorAqk = tla::MakeTensor(aqkBase, layoutC, Catlass::Arch::PositionGM{});
        auto tensorAkk = tla::MakeTensor(akkBase, layoutC, Catlass::Arch::PositionGM{});

        auto blockQPos = GetTile(tensorQPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKPos = GetTile(tensorKPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKNeg = GetTile(tensorKNeg, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.n()));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.n()));

        blockMmad.preSetFlags();
        blockMmad(blockQPos, blockKNeg, blockAqk, shape);
        blockMmad(blockKPos, blockKNeg, blockAkk, shape);
        blockMmad.finalWaitFlags();
    }

    __aicore__ inline bool UseAkkCubeSolve(uint64_t curT) const
    {
        return curT > 0 && curT <= BT_ && (BT_ == 64 || BT_ == 128) && K_ >= 16 && V_ >= 16 &&
               V_ <= 256 && K_ % 16 == 0 && V_ % 16 == 0;
    }

    __aicore__ inline bool UsePostWuCube(uint64_t curT) const
    {
        return curT > 0 && curT <= BT_ && (BT_ == 64 || BT_ == 128) && K_ >= 16 && V_ >= 16 &&
               V_ <= 256 && K_ % 16 == 0 && V_ % 16 == 0;
    }

    __aicore__ inline void CopyLocalFloat(LocalTensor<float> dst, LocalTensor<float> src, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        Adds(dst, src, 0.0f, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FillLocalFloat(LocalTensor<float> dst, float value, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        Duplicate(dst, value, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ForwardSubDiag16(LocalTensor<float> diag, LocalTensor<float> row,
                                             LocalTensor<float> prod, LocalTensor<float> rowBrcb,
                                             LocalTensor<float> reduced, uint64_t valid)
    {
        constexpr uint32_t brcbStride = 8;
        constexpr uint32_t diagSize = KDA_SOLVE_DIAG_BT;
        constexpr uint8_t rowBlk = diagSize * sizeof(float) / 32;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ForwardSubDiag16Regbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(diag.GetPhyAddr()),
            static_cast<uint16_t>(valid));
#else
        for (uint64_t i = 2; i < valid; ++i) {
            uint32_t rowOffset = static_cast<uint32_t>(i * diagSize);
            DataCopy(row, diag[rowOffset], diagSize);
            PipeBarrier<PIPE_V>();

            Brcb(rowBrcb, row, diagSize / brcbStride, {1, 8});
            PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < diagSize; col += brcbStride) {
                Mul(prod[col], diag[col], rowBrcb, brcbStride, static_cast<uint8_t>(diagSize),
                    {1, 1, 0, rowBlk, rowBlk, 1});
            }
            PipeBarrier<PIPE_V>();

            uint32_t remain = diagSize;
            while (remain > 1) {
                uint32_t calcCount = (remain / 2) * diagSize;
                remain = (remain + 1) / 2;
                Add(prod, prod, prod[remain * diagSize], calcCount);
                PipeBarrier<PIPE_V>();
            }
            DataCopy(reduced, prod, diagSize);
            PipeBarrier<PIPE_V>();
            Add(row, row, reduced, diagSize);
            PipeBarrier<PIPE_V>();
            DataCopy(diag[rowOffset], row, diagSize);
            PipeBarrier<PIPE_V>();
        }
#endif

        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        for (uint32_t i = 0; i < diagSize; ++i) {
            uint32_t diagOffset = i * diagSize + i;
            if (i < valid) {
                diag.SetValue(diagOffset, diag.GetValue(diagOffset) + 1.0f);
            } else {
                diag.SetValue(diagOffset, 1.0f);
            }
        }
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
    }

    __aicore__ inline void SolveDiagonalBlocksInRows(LocalTensor<float> akkMat, LocalTensor<float> xMat,
                                                      LocalTensor<float> arena, uint64_t scratchBase,
                                                      uint64_t curT, uint64_t rowBegin, uint64_t rowCount)
    {
        constexpr uint32_t diagSize = KDA_SOLVE_DIAG_BT;
        constexpr uint32_t diagElements = diagSize * diagSize;
        constexpr uint32_t brcbElements = diagSize * 8;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        (void)akkMat;
        (void)arena;
        (void)scratchBase;
        uint64_t rowEnd = rowBegin + rowCount;
        for (uint64_t blockBegin = 0; blockBegin < BT_; blockBegin += diagSize) {
            if (blockBegin < rowBegin || blockBegin + diagSize > rowEnd) {
                continue;
            }
            uint64_t localBlockRow = blockBegin - rowBegin;
            uint64_t valid = blockBegin < curT ? curT - blockBegin : 0;
            if (valid > diagSize) {
                valid = diagSize;
            }
            ForwardSubDiag16StridedRegbase(
                (__ubuf__ float *)reinterpret_cast<uint64_t>(xMat.GetPhyAddr()),
                static_cast<uint16_t>(BT_), static_cast<uint16_t>(localBlockRow),
                static_cast<uint16_t>(blockBegin), static_cast<uint16_t>(valid));
            SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
            WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
            for (uint32_t rowIdx = 0; rowIdx < diagSize; ++rowIdx) {
                uint32_t diagOffset =
                    static_cast<uint32_t>((localBlockRow + rowIdx) * BT_ + blockBegin + rowIdx);
                if (rowIdx < valid) {
                    xMat.SetValue(diagOffset, xMat.GetValue(diagOffset) + 1.0f);
                } else {
                    xMat.SetValue(diagOffset, 1.0f);
                }
            }
            SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
            WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        }
#else
        LocalTensor<float> diag = arena[scratchBase];
        LocalTensor<float> row = diag[diagElements];
        LocalTensor<float> prod = row[diagSize];
        LocalTensor<float> rowBrcb = prod[diagElements];
        LocalTensor<float> reduced = rowBrcb[brcbElements];

        uint64_t rowEnd = rowBegin + rowCount;
        for (uint64_t blockBegin = 0; blockBegin < BT_; blockBegin += diagSize) {
            if (blockBegin < rowBegin || blockBegin + diagSize > rowEnd) {
                continue;
            }
            Duplicate(diag, 0.0f, diagElements);
            PipeBarrier<PIPE_V>();

            uint64_t localBlockRow = blockBegin - rowBegin;
            uint64_t valid = blockBegin < curT ? curT - blockBegin : 0;
            if (valid > diagSize) {
                valid = diagSize;
            }
            for (uint32_t rowIdx = 0; rowIdx < diagSize; ++rowIdx) {
                uint64_t srcOffset = (localBlockRow + rowIdx) * BT_ + blockBegin;
                Muls(diag[rowIdx * diagSize], akkMat[srcOffset], -1.0f, diagSize);
            }
            PipeBarrier<PIPE_V>();

            ForwardSubDiag16(diag, row, prod, rowBrcb, reduced, valid);
            for (uint32_t rowIdx = 0; rowIdx < diagSize; ++rowIdx) {
                uint64_t dstOffset = (localBlockRow + rowIdx) * BT_ + blockBegin;
                Adds(xMat[dstOffset], diag[rowIdx * diagSize], 0.0f, diagSize);
            }
            PipeBarrier<PIPE_V>();
        }
#endif
    }

    __aicore__ inline void BuildPrefixMask(LocalTensor<float> dst, uint64_t prefix, uint64_t count)
    {
        if (prefix > count) {
            prefix = count;
        }
        Duplicate(dst, 0.0f, static_cast<uint32_t>(count));
        if (prefix > 0) {
            Duplicate(dst, 1.0f, static_cast<uint32_t>(prefix));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint64_t BuildCausalMask(uint64_t threshold, uint64_t colBegin) const
    {
        if (threshold <= colBegin) {
            return ~0ULL;
        }
        if (threshold >= colBegin + KDA_SOLVE_BT) {
            return 0ULL;
        }
        return ~0ULL << (threshold - colBegin);
    }

    __aicore__ inline void BuildCausalSelectMasks(LocalTensor<uint8_t> aqkMask, LocalTensor<uint8_t> akkMask,
                                                  uint64_t rowBegin, uint64_t rowCount, uint64_t colBegin)
    {
        __ubuf__ uint64_t *aqkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(aqkMask.GetPhyAddr());
        __ubuf__ uint64_t *akkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(akkMask.GetPhyAddr());
        for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
            uint32_t row = static_cast<uint32_t>(rowBegin + localRow);
            aqkMaskPtr[localRow] = BuildCausalMask(static_cast<uint64_t>(row) + 1, colBegin);
            akkMaskPtr[localRow] = BuildCausalMask(static_cast<uint64_t>(row), colBegin);
        }
    }

    __aicore__ inline void SelectCausalRows(LocalTensor<float> aqkMat, LocalTensor<float> akkMat,
                                            uint64_t rowBegin, uint64_t rowCount)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            SelectCausalRows64Regbase(
                (__ubuf__ float *)reinterpret_cast<uint64_t>(aqkMat.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(akkMat.GetPhyAddr()),
                static_cast<uint16_t>(rowBegin), static_cast<uint16_t>(rowCount));
            PipeBarrier<PIPE_V>();
            return;
        }
#endif
        LocalTensor<uint8_t> aqkMask = vecBuf_.Get<uint8_t>()[KDA_SELECT_AQK_MASK_BYTE_OFFSET];
        LocalTensor<uint8_t> akkMask = vecBuf_.Get<uint8_t>()[KDA_SELECT_AKK_MASK_BYTE_OFFSET];
        LocalTensor<float> zeroLocal = vecBuf_.Get<float>()[KDA_SELECT_ZERO_FLOAT_OFFSET];
        Duplicate(zeroLocal, 0.0f, 8);
        PipeBarrier<PIPE_V>();

        uint64_t colBlockCount = (BT_ + KDA_SOLVE_BT - 1) / KDA_SOLVE_BT;
        for (uint64_t colBlock = 0; colBlock < colBlockCount; ++colBlock) {
            uint64_t maskOffset = colBlock * KDA_SELECT_COL_MASK_BYTES;
            uint64_t colBegin = colBlock * KDA_SOLVE_BT;
            BuildCausalSelectMasks(aqkMask[maskOffset], akkMask[maskOffset], rowBegin, rowCount, colBegin);
        }
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);

        uint8_t rowStride = static_cast<uint8_t>(BT_ * sizeof(float) / 32);
        BinaryRepeatParams repeatParams = {1, 0, 1, rowStride, 0, rowStride};
        for (uint64_t colBlock = 0; colBlock < colBlockCount; ++colBlock) {
            uint64_t maskOffset = colBlock * KDA_SELECT_COL_MASK_BYTES;
            uint64_t colBegin = colBlock * KDA_SOLVE_BT;
            Select(aqkMat[colBegin], aqkMask[maskOffset], zeroLocal, aqkMat[colBegin],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, KDA_SOLVE_BT, static_cast<uint8_t>(rowCount), repeatParams);
            Select(akkMat[colBegin], akkMask[maskOffset], zeroLocal, akkMat[colBegin],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, KDA_SOLVE_BT, static_cast<uint8_t>(rowCount), repeatParams);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
    }

    __aicore__ inline void PrepareAqkAkkSolveInput64(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> xMat = arena[2 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaBrcb = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512];
        LocalTensor<float> oneHotLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512 + KDA_SOLVE_BT];

        LoadAsFloatRow(beta_, BetaOffset(b, hv, start), betaLocal, KDA_SOLVE_BT);
        Brcb(betaBrcb, betaLocal, 8, {1, 8});
        PipeBarrier<PIPE_V>();

        DataCopy(aqkMat, aqk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(akkMat, akk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, KDA_SOLVE_BT, {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        SelectCausalRows(aqkMat, akkMat, 0, KDA_SOLVE_BT);

        Muls(xMat, akkMat, -1.0f, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (uint64_t row = 0; row < KDA_SOLVE_BT; ++row) {
            BuildPrefixMask(maskLocal, row + 1, KDA_SOLVE_BT);
            BuildPrefixMask(oneHotLocal, row, KDA_SOLVE_BT);
            Sub(maskLocal, maskLocal, oneHotLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
            Add(xMat[row * KDA_SOLVE_BT], xMat[row * KDA_SOLVE_BT], maskLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(aqk_[AOffset(b, hv, start, 0)], aqkMat, KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(akk_[AOffset(b, hv, start, 0)], akkMat, KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], xMat,
                 KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void PrepareAqkAkkSolveInputTail(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                       uint64_t start, uint64_t curT)
    {
        uint64_t elemCount = curT * KDA_SOLVE_BT;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> xMat = arena[2 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaBrcb = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512];
        LocalTensor<float> oneHotLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512 + KDA_SOLVE_BT];

        FillLocalFloat(betaLocal, 0.0f, KDA_SOLVE_BT);
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        LoadAsFloatRow(beta_, BetaOffset(b, hv, start), betaLocal, curT);
        Brcb(betaBrcb, betaLocal, 8, {1, 8});
        PipeBarrier<PIPE_V>();

        DataCopy(aqkMat, aqk_[AOffset(b, hv, start, 0)], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        if (elemCount < KDA_SOLVE_MATRIX_ELEMENTS) {
            FillLocalFloat(aqkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
        }
        DataCopy(akkMat, akk_[AOffset(b, hv, start, 0)], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        if (elemCount < KDA_SOLVE_MATRIX_ELEMENTS) {
            FillLocalFloat(akkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
        }

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, KDA_SOLVE_BT, {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        SelectCausalRows(aqkMat, akkMat, 0, KDA_SOLVE_BT);

        Muls(xMat, akkMat, -1.0f, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (uint64_t row = 0; row < KDA_SOLVE_BT; ++row) {
            BuildPrefixMask(maskLocal, row + 1, KDA_SOLVE_BT);
            BuildPrefixMask(oneHotLocal, row, KDA_SOLVE_BT);
            Sub(maskLocal, maskLocal, oneHotLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
            Add(xMat[row * KDA_SOLVE_BT], xMat[row * KDA_SOLVE_BT], maskLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(aqk_[AOffset(b, hv, start, 0)], aqkMat, static_cast<uint32_t>(elemCount));
        DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], xMat,
                 KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0)], akkMat,
                 KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void GetSolveRowRange(uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum,
                                            uint64_t &rowBegin, uint64_t &rowEnd) const
    {
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum) {
            rowBegin = 0;
            rowEnd = 0;
            return;
        }
        rowBegin = (curT * subBlockIdx) / subBlockNum;
        rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
    }

    __aicore__ inline void PrepareAqkAkkSolveInputRows(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                       uint64_t start, uint64_t curT, uint64_t rowBegin,
                                                       uint64_t rowEnd, bool storeLToAkk,
                                                       bool storeLToScratch)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t validRowCount = rowBegin < curT ? curT - rowBegin : 0;
        if (validRowCount > rowCount) {
            validRowCount = rowCount;
        }
        uint64_t elemCount = rowCount * BT_;
        uint64_t validElemCount = validRowCount * BT_;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[elemCount];
        LocalTensor<float> xMat = arena[2 * elemCount];
        LocalTensor<float> betaLocal = arena[3 * elemCount];
        LocalTensor<float> betaBrcb = arena[3 * elemCount + BT_];
        LocalTensor<float> maskLocal = arena[3 * elemCount + BT_ + 512];
        LocalTensor<float> oneHotLocal = arena[3 * elemCount + BT_ + 512 + BT_];

        uint64_t token = start + rowBegin;

        if (validRowCount < rowCount) {
            FillLocalFloat(aqkMat, 0.0f, elemCount);
            FillLocalFloat(akkMat, 0.0f, elemCount);
            FillLocalFloat(betaLocal, 0.0f, rowCount);
        }
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        if (validRowCount > 0) {
            LoadAsFloatRow(beta_, BetaOffset(b, hv, token), betaLocal, validRowCount);
            if (curT < BT_) {
                DataCopy(aqkMat,
                         solveWorkspace_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_RAW_AQK) +
                                         rowBegin * BT_],
                         static_cast<uint32_t>(validElemCount));
                DataCopy(akkMat,
                         solveWorkspace_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_RAW_AKK) +
                                         rowBegin * BT_],
                         static_cast<uint32_t>(validElemCount));
            } else {
                DataCopy(aqkMat, aqk_[AOffset(b, hv, token, 0)], static_cast<uint32_t>(validElemCount));
                DataCopy(akkMat, akk_[AOffset(b, hv, token, 0)], static_cast<uint32_t>(validElemCount));
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ApplyKdaRowScaleRegbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(akkMat.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaLocal.GetPhyAddr()),
            static_cast<uint16_t>(rowCount), static_cast<uint16_t>(BT_));
        PipeBarrier<PIPE_V>();
#else
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        uint8_t rowStride = static_cast<uint8_t>(BT_ * sizeof(float) / 32);
        for (uint64_t col = 0; col < BT_; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, static_cast<uint8_t>(rowCount),
                {1, 1, 0, rowStride, rowStride, 1});
        }
        PipeBarrier<PIPE_V>();
#endif
        if (validRowCount > 0) {
            SelectCausalRows(aqkMat, akkMat, rowBegin, validRowCount);
        }

        Muls(xMat, akkMat, -1.0f, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();
        if constexpr (SAFE_GATE) {
            uint64_t scratchBase = 3 * elemCount + BT_ + 512 + 2 * BT_;
            SolveDiagonalBlocksInRows(akkMat, xMat, arena, scratchBase, curT, rowBegin, rowCount);
        } else if (curT < BT_) {
            uint64_t scratchBase = 3 * elemCount + BT_ + 512 + 2 * BT_;
            SolveDiagonalBlocksInRows(akkMat, xMat, arena, scratchBase, curT, rowBegin, rowCount);
        } else {
            for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
                uint64_t row = rowBegin + localRow;
                BuildPrefixMask(maskLocal, row + 1, BT_);
                BuildPrefixMask(oneHotLocal, row, BT_);
                Sub(maskLocal, maskLocal, oneHotLocal, static_cast<uint32_t>(BT_));
                PipeBarrier<PIPE_V>();
                Add(xMat[localRow * BT_], xMat[localRow * BT_], maskLocal, static_cast<uint32_t>(BT_));
                PipeBarrier<PIPE_V>();
            }
        }

        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        uint64_t lBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0) + rowBegin * BT_;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            LocalTensor<T> aqkTyped = GateQTyped(0);
            if (validElemCount > 0) {
                Muls(aqkMat, aqkMat, scale_, static_cast<uint32_t>(validElemCount));
                PipeBarrier<PIPE_V>();
                ClampFp32ToOutputType(aqkMat, static_cast<uint32_t>(validElemCount));
                Cast(aqkTyped, aqkMat, RoundMode::CAST_RINT, static_cast<uint32_t>(validElemCount));
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            if (validElemCount > 0) {
                CopyVectorOut(o_, AOffset(b, hv, token, 0), aqkTyped, validElemCount);
            }
            DataCopy(solveWorkspace_[xBase], xMat, static_cast<uint32_t>(elemCount));
            if (storeLToScratch) {
                DataCopy(solveWorkspace_[lBase], akkMat, static_cast<uint32_t>(elemCount));
            }
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            return;
        }
#endif
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        if (validRowCount > 0) {
            DataCopy(aqk_[AOffset(b, hv, token, 0)], aqkMat, static_cast<uint32_t>(validElemCount));
            if (storeLToAkk) {
                DataCopy(akk_[AOffset(b, hv, token, 0)], akkMat, static_cast<uint32_t>(validElemCount));
            }
        }
        DataCopy(solveWorkspace_[xBase], xMat, static_cast<uint32_t>(elemCount));
        if (storeLToScratch) {
            DataCopy(solveWorkspace_[lBase], akkMat, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void CubeGemmSolveSub(GlobalTensor<float> &tensorA, uint64_t baseA, uint64_t rowA, uint64_t colA,
                                            GlobalTensor<float> &tensorB, uint64_t baseB, uint64_t rowB, uint64_t colB,
                                            GlobalTensor<float> &tensorC, uint64_t baseC, uint64_t rowC, uint64_t colC,
                                            uint32_t m, uint32_t n, uint32_t k)
    {
        using ElementA = float;
        using ElementB = float;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Common::BlockMmadTla<KdaSolveDispatchPolicy, KdaSolveL1TileShape,
                                                              KdaSolveL0TileShape, ElementA, ElementB, ElementC,
                                                              void, TileCopy>;
        Catlass::Arch::Resource<KdaArchTag> resource;
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(BT_, BT_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(BT_, BT_);
        auto tensorLayoutA = tla::MakeTensor(tensorA[baseA], layoutA, Catlass::Arch::PositionGM{});
        auto tensorLayoutB = tla::MakeTensor(tensorB[baseB], layoutB, Catlass::Arch::PositionGM{});
        auto tensorLayoutC = tla::MakeTensor(tensorC[baseC], layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, n, k};
        auto blockA = GetTile(tensorLayoutA, tla::MakeCoord(rowA, colA), tla::MakeShape(shape.m(), shape.k()));
        auto blockB = GetTile(tensorLayoutB, tla::MakeCoord(rowB, colB), tla::MakeShape(shape.k(), shape.n()));
        auto blockC = GetTile(tensorLayoutC, tla::MakeCoord(rowC, colC), tla::MakeShape(shape.m(), shape.n()));
        BlockMmad blockMmad(resource);
        blockMmad(blockA, blockB, blockC, shape);
        // BlockMmad 析构时排空单个 L0C；下一步会从 GM 读取本次结果，
        // 因此还要闭合 Fixpipe 到 MTE2 的写后读依赖。
        SetFlag<HardEvent::FIX_MTE2>(solveFixToMte2Event_);
        WaitFlag<HardEvent::FIX_MTE2>(solveFixToMte2Event_);
    }

    __aicore__ inline void AddSolveTmpToX(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                          bool storeAkk)
    {
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        DataCopy(xLocal, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(tmpLocal, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        Add(xLocal, xLocal, tmpLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(h_[xBase], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        if (storeAkk) {
            DataCopy(akk_[AOffset(b, hv, start, 0)], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void AddSolveTmpToXTail(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t curT, bool storeAkk)
    {
        uint64_t elemCount = curT * KDA_SOLVE_BT;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        DataCopy(xLocal, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(tmpLocal, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        Add(xLocal, xLocal, tmpLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(h_[xBase], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        if (storeAkk) {
            DataCopy(akk_[AOffset(b, hv, start, 0)], xLocal, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void AddSolveTmpToXRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t curT, uint64_t rowBegin, uint64_t rowEnd, bool storeAkk)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t validRowCount = rowBegin < curT ? curT - rowBegin : 0;
        if (validRowCount > rowCount) {
            validRowCount = rowCount;
        }
        uint64_t elemCount = rowCount * BT_;
        uint64_t validElemCount = validRowCount * BT_;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[elemCount];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP) + rowBegin * BT_;
        uint64_t token = start + rowBegin;

        DataCopy(xLocal, solveWorkspace_[xBase], static_cast<uint32_t>(elemCount));
        DataCopy(tmpLocal, solveWorkspace_[tmpBase], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        Add(xLocal, xLocal, tmpLocal, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(solveWorkspace_[xBase], xLocal, static_cast<uint32_t>(elemCount));
        if (storeAkk && validRowCount > 0) {
            DataCopy(akk_[AOffset(b, hv, token, 0)], xLocal, static_cast<uint32_t>(validElemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void AddSolveTmpToXDiagRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                  uint64_t rowBegin, uint64_t rowEnd, bool storeAkk)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t elemCount = rowCount * BT_;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[elemCount];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP) + rowBegin * BT_;
        uint64_t token = start + rowBegin;

        DataCopy(xLocal, solveWorkspace_[xBase], static_cast<uint32_t>(elemCount));
        DataCopy(tmpLocal, solveWorkspace_[tmpBase], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
            uint64_t row = rowBegin + localRow;
            uint64_t col = (row / KDA_SOLVE_DIAG_BT) * KDA_SOLVE_DIAG_BT;
            uint64_t offset = localRow * BT_ + col;
            Add(xLocal[offset], xLocal[offset], tmpLocal[offset], KDA_SOLVE_DIAG_BT);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(solveWorkspace_[xBase], xLocal, static_cast<uint32_t>(elemCount));
        if (storeAkk) {
            DataCopy(akk_[AOffset(b, hv, token, 0)], xLocal, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void StoreSolveXRowsToAkk(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                uint64_t curT, uint64_t rowBegin, uint64_t rowEnd)
    {
        uint64_t validRowCount = rowBegin < curT ? curT - rowBegin : 0;
        uint64_t rowCount = rowEnd - rowBegin;
        if (validRowCount > rowCount) {
            validRowCount = rowCount;
        }
        if (validRowCount == 0) {
            return;
        }
        uint64_t elemCount = validRowCount * BT_;
        LocalTensor<float> xLocal = vecBuf_.Get<float>();
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;

        DataCopy(xLocal, solveWorkspace_[xBase], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        DataCopy(akk_[AOffset(b, hv, start + rowBegin, 0)], xLocal, static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
    }

    __aicore__ inline void ComputeAkkMergeCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        uint64_t aiBase = AOffset(b, hv, start, 0);
        uint64_t negABase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        for (uint32_t mergeSize = 2 * KDA_SOLVE_DIAG_BT; mergeSize <= BT_; mergeSize *= 2) {
            uint32_t half = mergeSize / 2;
            for (uint32_t block = 0; block < BT_; block += mergeSize) {
                uint32_t lower = block + half;
                CubeGemmSolveSub(akk_, aiBase, lower, lower, solveWorkspace_, negABase, lower, block,
                                 solveWorkspace_, tmpBase, 0, 0, half, half, half);
                CubeGemmSolveSub(solveWorkspace_, tmpBase, 0, 0, akk_, aiBase, block, block,
                                 akk_, aiBase, lower, block, half, half, half);
            }
        }
    }

    __aicore__ inline void ComputeAkkMergeCubeWorkspace(uint64_t b, uint64_t hv, uint64_t chunkIdx)
    {
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        for (uint32_t mergeSize = 2 * KDA_SOLVE_DIAG_BT; mergeSize <= BT_; mergeSize *= 2) {
            uint32_t half = mergeSize / 2;
            for (uint32_t block = 0; block < BT_; block += mergeSize) {
                uint32_t lower = block + half;
                CubeGemmSolveSub(solveWorkspace_, xBase, lower, lower, solveWorkspace_, xBase, lower, block,
                                 solveWorkspace_, tmpBase, 0, 0, half, half, half);
                CubeGemmSolveSub(solveWorkspace_, tmpBase, 0, 0, solveWorkspace_, xBase, block, block,
                                 solveWorkspace_, xBase, lower, block, half, half, half);
            }
        }
    }

    __aicore__ inline void ComputeAkkMergeCubeWorkspaceDispatch(uint64_t b, uint64_t hv, uint64_t chunkIdx)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            ComputeAkkMergeCubeWorkspace(b, hv, chunkIdx);
            return;
        }
#endif
        ComputeAkkMergeCubeWorkspace(b, hv, chunkIdx);
    }

    __aicore__ inline void ComputeAkkInverseMchFull(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        uint64_t aBase = AOffset(b, hv, start, 0);
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t yBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);
        uint64_t yNextBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y1);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        uint32_t diagBlocks = static_cast<uint32_t>(BT_ / KDA_SOLVE_DIAG_BT);
        for (uint32_t block = 0; block < diagBlocks; ++block) {
            uint32_t off = block * KDA_SOLVE_DIAG_BT;
            CubeGemmSolveSub(akk_, aBase, off, off, akk_, aBase, off, off, solveWorkspace_, yBase, off, off,
                             KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
        }
        for (uint32_t iter = 0; iter < KDA_SOLVE_DIAG_MCH_ITERS; ++iter) {
            for (uint32_t block = 0; block < diagBlocks; ++block) {
                uint32_t off = block * KDA_SOLVE_DIAG_BT;
                CubeGemmSolveSub(solveWorkspace_, xBase, off, off, solveWorkspace_, yBase, off, off,
                                 solveWorkspace_, tmpBase, off, off,
                                 KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(mchSyncDoneFlag_);
            if (iter + 1 < KDA_SOLVE_DIAG_MCH_ITERS) {
                for (uint32_t block = 0; block < diagBlocks; ++block) {
                    uint32_t off = block * KDA_SOLVE_DIAG_BT;
                    CubeGemmSolveSub(solveWorkspace_, yBase, off, off, solveWorkspace_, yBase, off, off,
                                     solveWorkspace_, yNextBase, off, off,
                                     KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
                }
            }
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(mchSyncReadyFlag_);
            if (iter + 1 < KDA_SOLVE_DIAG_MCH_ITERS) {
                uint64_t oldYBase = yBase;
                yBase = yNextBase;
                yNextBase = oldYBase;
            }
        }
        ComputeAkkMergeCube(b, hv, chunkIdx, start);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(mchSyncDoneFlag_);
    }

    __aicore__ inline void ScaleRowsByBeta(GlobalTensor<T> &src, GlobalTensor<T> &dst, uint64_t b, uint64_t hv,
                                           uint64_t start, uint64_t rowBegin, uint64_t rowCount, uint64_t dim,
                                           LocalTensor<float> &betaLocal, LocalTensor<float> &betaBrcb,
                                           LocalTensor<float> &matrixLocal, bool sourceSequenceMajor = false)
    {
        constexpr uint64_t vecElemsPerRepeat = 64;
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        uint64_t elemCount = rowCount * dim;
        uint64_t baseOffset = KVOffset(b, hv, start + rowBegin, 0, dim);
        uint64_t sourceOffset = sourceSequenceMajor
                                    ? VInputOffset(b, hv, start + rowBegin, 0)
                                    : baseOffset;
        uint64_t sourceStride = sourceSequenceMajor ? HV_ * dim : dim;

        if constexpr (IsSameType<T, float>::value) {
            CopyRowsIn(matrixLocal, src, sourceOffset, rowCount, dim, sourceStride);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            CopyRowsIn(matrixTyped, src, sourceOffset, rowCount, dim, sourceStride);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(matrixLocal, matrixTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
        }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ApplyKdaRowScaleRegbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(matrixLocal.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaLocal.GetPhyAddr()),
            static_cast<uint16_t>(rowCount), static_cast<uint16_t>(dim));
#else
        uint8_t repeatStride = static_cast<uint8_t>(dim * sizeof(float) / 32);
        for (uint64_t col = 0; col < dim; col += vecElemsPerRepeat) {
            uint64_t mask = dim - col;
            if (mask > vecElemsPerRepeat) {
                mask = vecElemsPerRepeat;
            }
            Mul(matrixLocal[col], matrixLocal[col], betaBrcb, mask, static_cast<uint8_t>(rowCount),
                {1, 1, 0, repeatStride, repeatStride, 1});
        }
        PipeBarrier<PIPE_V>();
#endif

        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(dst[baseOffset], matrixLocal, static_cast<uint32_t>(elemCount));
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            Cast(matrixTyped, matrixLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(dst[baseOffset], matrixTyped, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void PrepareWuCubeInputs(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT,
                                               uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t rowsPerSubBlock = (curT + subBlockNum - 1) / subBlockNum;
        uint64_t rowBegin = subBlockIdx * rowsPerSubBlock;
        if (rowBegin >= curT) {
            return;
        }
        uint64_t rowCount = curT - rowBegin;
        if (rowCount > rowsPerSubBlock) {
            rowCount = rowsPerSubBlock;
        }
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> betaLocal = arena;
        LocalTensor<float> betaBrcb = arena[KDA_SOLVE_BT];
        LocalTensor<float> matrixLocal = arena[KDA_SOLVE_BT + 512];
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // 将每个 arch35 Cast/regbase 面板限制在单条 UB 指令可覆盖的 8K 元素范围内。
        constexpr uint64_t maxScaleElements = 8192;
        if (rowCount * K_ > maxScaleElements || rowCount * V_ > maxScaleElements) {
            constexpr uint64_t tileRows = 16;
            for (uint64_t tileRow = 0; tileRow < rowCount; tileRow += tileRows) {
                uint64_t tileCount = rowCount - tileRow;
                if (tileCount > tileRows) {
                    tileCount = tileRows;
                }
                LoadAsFloatRow(beta_, BetaOffset(b, hv, start + rowBegin + tileRow), betaLocal, tileCount);
                ScaleRowsByBeta(w_, w_, b, hv, start, rowBegin + tileRow, tileCount, K_,
                                betaLocal, betaBrcb, matrixLocal);
                ScaleRowsByBeta(v_, vNew_, b, hv, start, rowBegin + tileRow, tileCount, V_,
                                betaLocal, betaBrcb, matrixLocal, inputSequenceMajor_);
            }
            return;
        }
#endif
        LoadAsFloatRow(beta_, BetaOffset(b, hv, start + rowBegin), betaLocal, rowCount);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
#endif
        ScaleRowsByBeta(w_, w_, b, hv, start, rowBegin, rowCount, K_, betaLocal, betaBrcb, matrixLocal);
        ScaleRowsByBeta(v_, vNew_, b, hv, start, rowBegin, rowCount, V_, betaLocal, betaBrcb,
                        matrixLocal, inputSequenceMajor_);
    }

    __aicore__ inline void FinalizePrepareIntermediates(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                        uint64_t start, uint64_t curT,
                                                        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        constexpr bool qgScaledAlreadyStored =
            SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128;
#else
        constexpr bool qgScaledAlreadyStored = false;
#endif
        constexpr uint64_t tileRows = 32;
        // 尾行必须留在负责其补齐 solve 行的同一个 AIV 上。若按 curT 重新切分，
        // AIV0 尚在写 solve 矩阵时，短尾导出会错误移到 AIV1。
        const uint64_t rowBegin = (BT_ * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (BT_ * (subBlockIdx + 1)) / subBlockNum;
        if (rowEnd > curT) {
            rowEnd = curT;
        }
        if (rowBegin >= rowEnd) {
            return;
        }
        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += tileRows) {
            const uint64_t rows = (rowEnd - tileRow) > tileRows ? tileRows : (rowEnd - tileRow);
            const uint64_t matrixElems = rows * BT_;
            const uint64_t qgElems = rows * K_;
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> aqkLocal = arena;
            LocalTensor<float> akkLocal = arena[matrixElems];
            LocalTensor<float> qgLocal = arena[2 * matrixElems];
            const uint64_t typedOffset =
                (2 * matrixElems + qgElems) * sizeof(float) / sizeof(T);
            LocalTensor<T> typedBase = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> aqkTyped = typedBase;
            LocalTensor<T> akkTyped = typedBase[matrixElems];
            LocalTensor<T> qgTyped = typedBase[2 * matrixElems];

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
                const uint64_t xBase =
                    SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + tileRow * BT_;
                CopyVectorIn(akkLocal, solveWorkspace_, xBase, matrixElems);
            } else {
                CopyVectorIn(aqkLocal, aqk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
                CopyVectorIn(akkLocal, akk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
            }
#else
            CopyVectorIn(aqkLocal, aqk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
            CopyVectorIn(akkLocal, akk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
#endif
            if constexpr (!qgScaledAlreadyStored) {
                CopyVectorIn(qgTyped, qg_, KVOffset(b, hv, start + tileRow, 0, K_), qgElems);
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                Muls(aqkLocal, aqkLocal, scale_, static_cast<uint32_t>(matrixElems));
            }
#else
            Muls(aqkLocal, aqkLocal, scale_, static_cast<uint32_t>(matrixElems));
#endif
            if constexpr (!qgScaledAlreadyStored) {
                Cast(qgLocal, qgTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(qgElems));
                PipeBarrier<PIPE_V>();
                Muls(qgLocal, qgLocal, scale_, static_cast<uint32_t>(qgElems));
                PipeBarrier<PIPE_V>();
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                ClampFp32ToOutputType(aqkLocal, static_cast<uint32_t>(matrixElems));
            }
#else
            ClampFp32ToOutputType(aqkLocal, static_cast<uint32_t>(matrixElems));
#endif
            ClampFp32ToOutputType(akkLocal, static_cast<uint32_t>(matrixElems));
            if constexpr (!qgScaledAlreadyStored) {
                ClampFp32ToOutputType(qgLocal, static_cast<uint32_t>(qgElems));
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                Cast(aqkTyped, aqkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
            }
#else
            Cast(aqkTyped, aqkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
#endif
            Cast(akkTyped, akkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
            if constexpr (!qgScaledAlreadyStored) {
                Cast(qgTyped, qgLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(qgElems));
            }
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                CopyVectorOut(o_, AOffset(b, hv, start + tileRow, 0), aqkTyped, matrixElems);
            }
#else
            CopyVectorOut(o_, AOffset(b, hv, start + tileRow, 0), aqkTyped, matrixElems);
#endif
            CopyVectorOut(u_, AOffset(b, hv, start + tileRow, 0), akkTyped, matrixElems);
            if constexpr (!qgScaledAlreadyStored) {
                CopyVectorOut(kg_, KVOffset(b, hv, start + tileRow, 0, K_), qgTyped, qgElems);
            }
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
    }

    __aicore__ inline bool ResolveFlatChunk(uint64_t task, uint64_t &seq, uint64_t &b, uint64_t &h, uint64_t &hv,
                                            uint64_t &chunkIdx, uint64_t &start, uint64_t &end)
    {
        hv = task % HV_;
        uint64_t flatChunk = task / HV_;
        if (!isVarLen_) {
            seq = flatChunk / NT_;
            b = seq;
            chunkIdx = flatChunk % NT_;
            start = chunkIdx * BT_;
            end = start + BT_;
            if (end > T_) {
                end = T_;
            }
        } else {
            if (!KdaVarlen::ResolveChunkRange(
                    cuSeqlensAddr_, chunkIndicesAddr_, N_, T_, BT_, flatChunk,
                    seq, start, end)) {
                return false;
            }
            b = 0;
            chunkIdx = flatChunk;
        }
        h = hv / (HV_ / H_);
        return start < end;
    }

    __aicore__ inline void ProcessChunkPreAiv(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                              uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                              uint64_t subBlockNum)
    {
        if constexpr (IsSameType<AKK_T, float>::value) {
            ProcessChunkPreAivFp32(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void JoinAivMte3()
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if (!isAivOnly_) {
                // 两个AIV始终先在组内汇合，再由mode2向AIC提交一份完成
                // token；headCnt只改变循环次数，不改变同步协议。
                Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                PipeBarrier<PIPE_MTE3>();
            }
#endif
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void RunAicAfterBothAivReady(uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
            (void)subBlockIdx;
            (void)subBlockNum;
            JoinAivMte3();
            if constexpr (SAFE_GATE) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncReadyFlag_);
                Catlass::Arch::CrossCoreWaitFlag(syncDoneFlag_);
            } else {
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(mchSyncReadyFlag_);
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(mchSyncDoneFlag_);
            }
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void SignalAicSolveReady()
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
            JoinAivMte3();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncReadyFlag_);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void WaitAicSolveDone()
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
            Catlass::Arch::CrossCoreWaitFlag(syncDoneFlag_);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void SignalPostWuReady()
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(postWuReadyFlag_);
        }
    }

    __attribute__((noinline)) __aicore__ void ProcessChunkPreAivFp32(
        uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
        uint64_t start, uint64_t end, uint64_t subBlockIdx,
        uint64_t subBlockNum, bool deferSafeSolve = false,
        bool waitPendingSafeSolve = false)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            return;
        }

        if (K_ < 16) {
            return;
        }
        if (subBlockIdx == 0) {
            MaterializeRawGateChunkArch35<true>(b, hv, start, curT);
        } else {
            MaterializeRawGateChunkArch35<false>(b, hv, start, curT);
        }
        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        uint64_t solveRowBegin = 0;
        uint64_t solveRowEnd = 0;
        GetSolveRowRange(BT_, subBlockIdx, subBlockNum, solveRowBegin, solveRowEnd);
        // Safe-gate score factors need the bounded 16-row reference span.
        // A single 64-row reference loses BF16 dynamic range for valid gates.
        const bool useFullChunkScore = false;
        uint64_t scoreBlockSize = useFullChunkScore ? curT : ScoreRefBlockSize();
        uint64_t scoreBlockCount = (curT + scoreBlockSize - 1) / scoreBlockSize;
        uint64_t pipelineBlockCount = useFullChunkScore
            ? scoreBlockCount
            : (scoreBlockCount + KDA_SCORE_QUEUE_DEPTH - 1) / KDA_SCORE_QUEUE_DEPTH *
                  KDA_SCORE_QUEUE_DEPTH;
        for (uint64_t block = 0; block < pipelineBlockCount; ++block) {
            if (block < scoreBlockCount) {
                uint64_t rowBegin = block * scoreBlockSize;
                uint64_t rowCount = useFullChunkScore
                    ? curT - rowBegin
                    : ScoreRowBlockCount(curT, rowBegin);
                uint64_t refToken = ScoreRefToken(start, curT, rowBegin, rowCount);
                uint64_t queueSlot = useFullChunkScore
                    ? activeSolveSlot_ / KDA_SCORE_LANES
                    : block % KDA_SCORE_QUEUE_DEPTH;
                uint64_t scoreSlot = queueSlot;
                PrepareGateProducts(b, h, hv, start, curT, subBlockIdx, subBlockNum, true, refToken,
                                    rowBegin + rowCount, true, scoreSlot,
                                    rowBegin, rowCount);
            }
            JoinAivMte3();
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
            if (block > 0) {
                if constexpr (SAFE_GATE) {
                    if (waitPendingSafeSolve) {
                        WaitAicSolveDone();
                        waitPendingSafeSolve = false;
                    }
                }
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
            }
        }
        bool fusedScoreWriteback = false;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        fusedScoreWriteback = SAFE_GATE && BT_ == 64 && K_ == 128 && V_ == 128;
#endif
        if (!fusedScoreWriteback) {
            // 最后一次 score MMAD 只读取 scoreWorkspace_；AIC 排空 MMAD/Fixpipe
            // 路径时，可并行执行独立的 gate 写回。
            PrepareGateProducts(b, h, hv, start, curT, subBlockIdx, subBlockNum);
        }
        if (pipelineBlockCount > 0) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
        }
        if constexpr (SAFE_GATE) {
            if (waitPendingSafeSolve) {
                WaitAicSolveDone();
                waitPendingSafeSolve = false;
            }
        }
        if (useAkkCubeSolve) {
            bool fullChunk = curT == BT_;
            if constexpr (SAFE_GATE) {
                PrepareAqkAkkSolveInputRows(
                    b, hv, chunkIdx, start, curT,
                    solveRowBegin, solveRowEnd, false, false);
                if (deferSafeSolve) {
                    SignalAicSolveReady();
                    return;
                }
                RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                    StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
                }
#else
                StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
#endif
            } else {
                PrepareAqkAkkSolveInputRows(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd,
                                            fullChunk, false);
                if (!fullChunk) {
                    RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
                    StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
                } else {
                    uint32_t solveIters = KDA_SOLVE_DIAG_MCH_ITERS;
                    RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
                    for (uint32_t iter = 0; iter < solveIters; ++iter) {
                        AddSolveTmpToXDiagRows(b, hv, chunkIdx, start, solveRowBegin, solveRowEnd,
                                               iter + 1 == solveIters);
                        RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
                    }
                }
            }
        }
        // Host 校验保证所有已接受 shape 都为该 Cube 路径预留了足够 workspace。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
        } else {
            PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
        }
#else
        PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
#endif
        FinalizePrepareIntermediates(b, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void FinishDeferredSafeChunk(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                   uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                                   uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        uint64_t solveRowBegin = 0;
        uint64_t solveRowEnd = 0;
        GetSolveRowRange(BT_, subBlockIdx, subBlockNum, solveRowBegin, solveRowEnd);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
            StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
        }
#else
        StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
        } else {
            PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
        }
#else
        PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
#endif
        FinalizePrepareIntermediates(b, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
    }

    __attribute__((noinline)) __aicore__ void ProcessChunkPreAic(
        uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
        uint64_t end)
    {
        if constexpr (IsSameType<AKK_T, float>::value) {
            ProcessChunkPreAicFp32(b, hv, chunkIdx, start, end);
        }
    }

    __aicore__ inline void ProcessChunkPreAicFp32(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                  uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0 || K_ < 16) {
            return;
        }
        uint64_t scoreBlockSize = ScoreRefBlockSize();
        uint64_t scoreBlockCount = (curT + scoreBlockSize - 1) / scoreBlockSize;
        uint64_t pipelineBlockCount =
            (scoreBlockCount + KDA_SCORE_QUEUE_DEPTH - 1) / KDA_SCORE_QUEUE_DEPTH * KDA_SCORE_QUEUE_DEPTH;
        for (uint64_t block = 0; block < pipelineBlockCount; ++block) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
            if (block < scoreBlockCount) {
                uint64_t rowBegin = block * scoreBlockSize;
                uint64_t rowCount = ScoreRowBlockCount(curT, rowBegin);
                ComputeRawAqkAkkCubeBlock(b, hv, chunkIdx, start, curT, rowBegin, rowCount, true,
                                          block % KDA_SCORE_QUEUE_DEPTH, rowBegin + rowCount);
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
        }
        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        if (useAkkCubeSolve) {
            if constexpr (SAFE_GATE) {
                Catlass::Arch::CrossCoreWaitFlag(syncReadyFlag_);
                ComputeAkkMergeCubeWorkspaceDispatch(b, hv, chunkIdx);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(syncDoneFlag_);
            } else {
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(mchSyncReadyFlag_);
                if (curT == BT_) {
                    ComputeAkkInverseMchFull(b, hv, chunkIdx, start);
                } else {
                    ComputeAkkMergeCubeWorkspace(b, hv, chunkIdx);
                    Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(mchSyncDoneFlag_);
                }
            }
        }
        (void)usePostWuCube;
        (void)chunkIdx;
    }

    struct OwnedChunkDesc {
        uint64_t seq = 0;
        uint64_t b = 0;
        uint64_t chunkIdx = 0;
        uint64_t start = 0;
        uint64_t end = 0;
    };

    struct DeferredSingleHeadState {
        bool valid = false;
        uint64_t b = 0;
        uint64_t hv = 0;
        uint64_t chunkIdx = 0;
        uint64_t start = 0;
        uint64_t end = 0;
        uint64_t slot = 0;
        uint64_t taskIdx = 0;
    };

    struct AivHeadWindowState {
        DeferredSingleHeadState singlePending{};
    };

    struct AicHeadWindowState {
        uint64_t localTaskIdx = 0;
    };

    struct PendingSinglePostWuState {
        bool valid = false;
        uint64_t b = 0;
        uint64_t hv = 0;
        uint64_t start = 0;
        uint64_t rows = 0;
    };

    struct FusedFullHeadWindowState {
        uint64_t localTaskIdx = 0;
        // 每个batch项严格对应一个实际HV；两个AIV通过同一mode2 token
        // 汇聚完成状态，队列深度不随headCnt改变。
        uint64_t batchB[KDA_POST_QUEUE_DEPTH]{};
        uint64_t batchHv[KDA_POST_QUEUE_DEPTH]{};
        uint64_t batchStart[KDA_POST_QUEUE_DEPTH]{};
        uint16_t batchCount = 0;
    };

    struct FusedTailHeadWindowState {
        uint64_t localTaskIdx = 0;
        PendingSinglePostWuState singlePending{};
    };

    struct FullChunkIterator {
        uint64_t sequence = 0;
        uint64_t localChunk = 0;
        uint64_t sequenceStart = 0;
        uint64_t fullChunkCount = 0;
        bool sequenceLoaded = false;
    };

    struct GroupedFullTaskIterator {
        FullChunkIterator chunks{};
        OwnedChunkDesc chunk{};
        uint64_t loadedChunkOrdinal = 0;
        bool chunkLoaded = false;
    };

    struct GroupedTailTaskIterator {
        OwnedChunkDesc chunk{};
        uint64_t loadedChunkOrdinal = 0;
        bool chunkLoaded = false;
    };

    __aicore__ inline bool ResolveFlatChunkForHv(
        uint64_t flatChunk, uint64_t hv, uint64_t &seq, uint64_t &b, uint64_t &h,
        uint64_t &chunkIdx, uint64_t &start, uint64_t &end)
    {
        if (!isVarLen_) {
            seq = flatChunk / NT_;
            b = seq;
            chunkIdx = flatChunk % NT_;
            start = chunkIdx * BT_;
            end = start + BT_;
            if (end > T_) {
                end = T_;
            }
        } else {
            if (!KdaVarlen::ResolveChunkRange(
                    cuSeqlensAddr_, chunkIndicesAddr_, N_, T_, BT_, flatChunk,
                    seq, start, end)) {
                return false;
            }
            b = 0;
            chunkIdx = flatChunk;
        }
        h = hv / (HV_ / H_);
        return start < end;
    }

    __aicore__ inline bool LoadOwnedFullChunk(
        const KdaForward::CompactSequencePlanView &plan,
        FullChunkIterator &iterator, OwnedChunkDesc &desc)
    {
        while (iterator.sequence < plan.SequenceCount()) {
            if (!iterator.sequenceLoaded) {
                uint64_t sequenceEnd = T_;
                iterator.sequenceStart = 0;
                if (isVarLen_) {
                    iterator.sequenceStart = static_cast<uint64_t>(
                        cuSeqlensAddr_[iterator.sequence]);
                    sequenceEnd = static_cast<uint64_t>(
                        cuSeqlensAddr_[iterator.sequence + 1]);
                }
                iterator.fullChunkCount =
                    (sequenceEnd - iterator.sequenceStart) / BT_;
                iterator.sequenceLoaded = true;
            }
            if (iterator.localChunk < iterator.fullChunkCount) {
                desc.seq = iterator.sequence;
                desc.b = isVarLen_ ? 0 : iterator.sequence;
                desc.chunkIdx = isVarLen_
                    ? plan.SequenceChunkOffset(
                          static_cast<uint32_t>(iterator.sequence)) +
                          iterator.localChunk
                    : iterator.localChunk;
                desc.start = iterator.sequenceStart + iterator.localChunk * BT_;
                desc.end = desc.start + BT_;
                ++iterator.localChunk;
                if (iterator.localChunk == iterator.fullChunkCount) {
                    ++iterator.sequence;
                    iterator.localChunk = 0;
                    iterator.sequenceLoaded = false;
                }
                return true;
            }
            ++iterator.sequence;
            iterator.localChunk = 0;
            iterator.sequenceLoaded = false;
        }
        return false;
    }

    __aicore__ inline bool LoadOwnedTailChunk(
        const KdaForward::CompactSequencePlanView &plan, uint64_t tailOrdinal,
        OwnedChunkDesc &desc)
    {
        const uint64_t sequence = plan.TailedSequenceId(
            static_cast<uint32_t>(tailOrdinal));
        if (sequence >= plan.SequenceCount()) {
            return false;
        }
        uint64_t sequenceStart = 0;
        uint64_t sequenceEnd = T_;
        if (isVarLen_) {
            sequenceStart = static_cast<uint64_t>(cuSeqlensAddr_[sequence]);
            sequenceEnd = static_cast<uint64_t>(cuSeqlensAddr_[sequence + 1]);
        }
        const uint64_t fullChunks = (sequenceEnd - sequenceStart) / BT_;
        desc.seq = sequence;
        desc.b = isVarLen_ ? 0 : sequence;
        desc.chunkIdx = isVarLen_
            ? plan.SequenceChunkOffset(static_cast<uint32_t>(sequence)) + fullChunks
            : fullChunks;
        desc.start = sequenceStart + fullChunks * BT_;
        desc.end = sequenceEnd;
        return desc.start < desc.end;
    }

    __aicore__ inline bool LoadGroupedFullTask(
        const KdaForward::CompactSequencePlanView &plan, uint64_t task,
        GroupedFullTaskIterator &iterator, OwnedChunkDesc &chunk,
        uint64_t &headBegin, uint64_t &headEnd)
    {
        uint32_t chunkOrdinal = 0;
        uint32_t begin = 0;
        uint32_t end = 0;
        if (!plan.DecodeChunkHeadGroupTask(
                static_cast<uint32_t>(task), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_),
                chunkOrdinal, begin, end)) {
            return false;
        }
        if (!iterator.chunkLoaded ||
            iterator.loadedChunkOrdinal != chunkOrdinal) {
            if (!LoadOwnedFullChunk(plan, iterator.chunks, iterator.chunk)) {
                return false;
            }
            iterator.loadedChunkOrdinal = chunkOrdinal;
            iterator.chunkLoaded = true;
        }
        chunk = iterator.chunk;
        headBegin = begin;
        headEnd = end;
        return true;
    }

    __aicore__ inline bool LoadGroupedTailTask(
        const KdaForward::CompactSequencePlanView &plan, uint64_t task,
        GroupedTailTaskIterator &iterator, OwnedChunkDesc &chunk,
        uint64_t &headBegin, uint64_t &headEnd)
    {
        uint32_t chunkOrdinal = 0;
        uint32_t begin = 0;
        uint32_t end = 0;
        if (!plan.DecodeChunkHeadGroupTask(
                static_cast<uint32_t>(task), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_),
                chunkOrdinal, begin, end)) {
            return false;
        }
        if (!iterator.chunkLoaded ||
            iterator.loadedChunkOrdinal != chunkOrdinal) {
            if (!LoadOwnedTailChunk(plan, chunkOrdinal, iterator.chunk)) {
                return false;
            }
            iterator.loadedChunkOrdinal = chunkOrdinal;
            iterator.chunkLoaded = true;
        }
        chunk = iterator.chunk;
        headBegin = begin;
        headEnd = end;
        return true;
    }

    __aicore__ inline uint64_t SolvePipelineSlot(uint64_t taskIdx) const
    {
        if constexpr (SAFE_GATE) {
            return taskIdx % KDA_SOLVE_PIPELINE_DEPTH;
        }
        // 非safe路径只有一份solve workspace，不能套用深度4的slot编号。
        (void)taskIdx;
        return 0;
    }

    __aicore__ inline void ProcessOwnedChunkAivQueryHeadGroup(
        const OwnedChunkDesc &chunk, uint64_t subBlockIdx,
        uint64_t subBlockNum, uint64_t headBegin, uint32_t headCnt,
        DeferredSingleHeadState &pending)
    {
        for (uint32_t headOffset = 0; headOffset < headCnt;
             ++headOffset, ++pending.taskIdx) {
            const uint64_t hv = headBegin + headOffset;
            const uint64_t h = hv / (HV_ / H_);
            const uint64_t currentSlot = SolvePipelineSlot(pending.taskIdx);
            activeSolveSlot_ = currentSlot;
            bool deferSolve = false;
            if constexpr (SAFE_GATE) {
                deferSolve = UseAkkCubeSolve(chunk.end - chunk.start);
            }
            if (!deferSolve && pending.valid) {
                WaitAicSolveDone();
                activeSolveSlot_ = pending.slot;
                FinishDeferredSafeChunk(
                    pending.b, pending.hv, pending.chunkIdx,
                    pending.start, pending.end, subBlockIdx, subBlockNum);
                if (fusePostWu_) {
                    SignalPostWuReady();
                }
                pending.valid = false;
                activeSolveSlot_ = currentSlot;
            }
            ProcessChunkPreAivFp32(
                chunk.b, h, hv, chunk.chunkIdx, chunk.start, chunk.end,
                subBlockIdx, subBlockNum, deferSolve, pending.valid);
            if (pending.valid) {
                activeSolveSlot_ = pending.slot;
                FinishDeferredSafeChunk(
                    pending.b, pending.hv, pending.chunkIdx,
                    pending.start, pending.end, subBlockIdx, subBlockNum);
                if (fusePostWu_) {
                    SignalPostWuReady();
                }
            }
            if (fusePostWu_ && !deferSolve) {
                SignalPostWuReady();
            }
            pending.valid = deferSolve;
            if (pending.valid) {
                pending.b = chunk.b;
                pending.hv = hv;
                pending.chunkIdx = chunk.chunkIdx;
                pending.start = chunk.start;
                pending.end = chunk.end;
                pending.slot = currentSlot;
            }
        }
    }

    __aicore__ inline void ProcessOwnedChunkAivHeadWindow(
        const OwnedChunkDesc &chunk, uint64_t subBlockIdx,
        uint64_t subBlockNum, uint64_t headBegin, uint32_t headCnt,
        DeferredSingleHeadState &pending)
    {
        // 一个runtime窗口最多4个HV。窗口内只在qHead变化时重置raw Q/K
        // resident；下一个窗口会重新进入本函数，因此ratio8的第二个窗口
        // 允许按设计重读同一个qHead。
        uint32_t groupOffset = 0;
        while (groupOffset < headCnt) {
            const uint64_t groupHv = headBegin + groupOffset;
            const uint64_t groupH = groupHv / (HV_ / H_);
            uint32_t groupCnt = 1;
            while (groupOffset + groupCnt < headCnt) {
                const uint64_t nextHv = headBegin + groupOffset + groupCnt;
                if (nextHv / (HV_ / H_) != groupH) {
                    break;
                }
                ++groupCnt;
            }
            BeginRawQkResidentGroupArch35(
                chunk.b, groupH, chunk.start, chunk.end - chunk.start,
                subBlockIdx, subBlockNum);
            ProcessOwnedChunkAivQueryHeadGroup(
                chunk, subBlockIdx, subBlockNum, groupHv, groupCnt, pending);
            groupOffset += groupCnt;
        }
    }

    __aicore__ inline void ProcessOwnedChunkAivHeads(
        const OwnedChunkDesc &chunk, uint64_t subBlockIdx,
        uint64_t subBlockNum, uint64_t headBegin, uint64_t headEnd,
        DeferredSingleHeadState &pending)
    {
        for (uint64_t head = headBegin; head < headEnd;) {
            uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(head), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - head) {
                headCnt = static_cast<uint32_t>(headEnd - head);
            }
            ProcessOwnedChunkAivHeadWindow(
                chunk, subBlockIdx, subBlockNum, head, headCnt, pending);
            head += headCnt;
        }
    }

    __aicore__ inline void FlushDeferredSingleHead(
        DeferredSingleHeadState &pending, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        if (!pending.valid) {
            return;
        }
        WaitAicSolveDone();
        activeSolveSlot_ = pending.slot;
        FinishDeferredSafeChunk(
            pending.b, pending.hv, pending.chunkIdx,
            pending.start, pending.end, subBlockIdx, subBlockNum);
        if (fusePostWu_) {
            SignalPostWuReady();
        }
        pending.valid = false;
    }

    __aicore__ inline void DrainAivHeadWindowState(
        AivHeadWindowState &state, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        FlushDeferredSingleHead(
            state.singlePending, subBlockIdx, subBlockNum);
        state.singlePending.valid = false;
        state.singlePending.taskIdx = 0;
    }

    __aicore__ inline void ProcessOwnedChunkAivHeadWindowTask(
        const OwnedChunkDesc &chunk, uint64_t subBlockIdx,
        uint64_t subBlockNum, uint64_t headBegin, uint64_t headEnd,
        AivHeadWindowState &state)
    {
        ProcessOwnedChunkAivHeads(
            chunk, subBlockIdx, subBlockNum, headBegin, headEnd,
            state.singlePending);
    }

    __aicore__ inline void ProcessOwnedChunkAicHeadWindow(
        const OwnedChunkDesc &chunk, uint64_t headBegin,
        uint32_t headCnt, uint64_t &localTaskIdx)
    {
        for (uint32_t headOffset = 0; headOffset < headCnt;
             ++headOffset, ++localTaskIdx) {
            const uint64_t hv = headBegin + headOffset;
            activeSolveSlot_ = SolvePipelineSlot(localTaskIdx);
            ProcessChunkPreAic(
                chunk.b, hv, chunk.chunkIdx, chunk.start, chunk.end);
        }
    }

    __aicore__ inline void ProcessOwnedChunkAicHeads(
        const OwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t headEnd, uint64_t &localTaskIdx)
    {
        for (uint64_t head = headBegin; head < headEnd;) {
            uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(head), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - head) {
                headCnt = static_cast<uint32_t>(headEnd - head);
            }
            ProcessOwnedChunkAicHeadWindow(
                chunk, head, headCnt, localTaskIdx);
            head += headCnt;
        }
    }

    __aicore__ inline void ProcessOwnedChunkAicHeadWindowTask(
        const OwnedChunkDesc &chunk, uint64_t headBegin, uint64_t headEnd,
        AicHeadWindowState &state)
    {
        ProcessOwnedChunkAicHeads(
            chunk, headBegin, headEnd, state.localTaskIdx);
    }

    __aicore__ inline void ProcessPreAivHeadWindows()
    {
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        const uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        const uint64_t coreIdx =
            static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        KdaForward::ChunkCoreCursor cursor{};
        if (!plan.LoadChunkCoreCursor(
                static_cast<uint32_t>(coreIdx), cursor)) {
            return;
        }

        if (plan.HeadGroupCount() == 1) {
            AivHeadWindowState fullState{};
            FullChunkIterator fullIterator{};
            fullIterator.sequence = cursor.fullStartSequence;
            fullIterator.localChunk = cursor.fullStartLocalChunk;
            for (uint64_t ordinal = cursor.fullBegin;
                 ordinal < cursor.fullEnd; ++ordinal) {
                OwnedChunkDesc chunk{};
                if (LoadOwnedFullChunk(plan, fullIterator, chunk)) {
                    ProcessOwnedChunkAivHeadWindowTask(
                        chunk, subBlockIdx, subBlockNum, 0, HV_, fullState);
                }
            }
            DrainAivHeadWindowState(
                fullState, subBlockIdx, subBlockNum);

            AivHeadWindowState tailState{};
            for (uint64_t ordinal = cursor.tailBegin;
                 ordinal < cursor.tailEnd; ++ordinal) {
                OwnedChunkDesc chunk{};
                if (LoadOwnedTailChunk(plan, ordinal, chunk)) {
                    ProcessOwnedChunkAivHeadWindowTask(
                        chunk, subBlockIdx, subBlockNum, 0, HV_, tailState);
                }
            }
            DrainAivHeadWindowState(
                tailState, subBlockIdx, subBlockNum);
            return;
        }

        AivHeadWindowState fullState{};
        GroupedFullTaskIterator fullIterator{};
        fullIterator.chunks.sequence = cursor.fullStartSequence;
        fullIterator.chunks.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t task = cursor.fullBegin;
             task < cursor.fullEnd; ++task) {
            OwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadGroupedFullTask(
                    plan, task, fullIterator, chunk, headBegin, headEnd)) {
                ProcessOwnedChunkAivHeadWindowTask(
                    chunk, subBlockIdx, subBlockNum,
                    headBegin, headEnd, fullState);
            }
        }
        DrainAivHeadWindowState(
            fullState, subBlockIdx, subBlockNum);

        AivHeadWindowState tailState{};
        GroupedTailTaskIterator tailIterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            OwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadGroupedTailTask(
                    plan, task, tailIterator, chunk, headBegin, headEnd)) {
                ProcessOwnedChunkAivHeadWindowTask(
                    chunk, subBlockIdx, subBlockNum,
                    headBegin, headEnd, tailState);
            }
        }
        DrainAivHeadWindowState(
            tailState, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void ProcessPreAicHeadWindows()
    {
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        KdaForward::ChunkCoreCursor cursor{};
        if (!plan.LoadChunkCoreCursor(
                static_cast<uint32_t>(GetBlockIdx()), cursor)) {
            return;
        }
        if (plan.HeadGroupCount() == 1) {
            AicHeadWindowState fullState{};
            FullChunkIterator fullIterator{};
            fullIterator.sequence = cursor.fullStartSequence;
            fullIterator.localChunk = cursor.fullStartLocalChunk;
            for (uint64_t ordinal = cursor.fullBegin;
                 ordinal < cursor.fullEnd; ++ordinal) {
                OwnedChunkDesc chunk{};
                if (LoadOwnedFullChunk(plan, fullIterator, chunk)) {
                    ProcessOwnedChunkAicHeadWindowTask(
                        chunk, 0, HV_, fullState);
                }
            }

            AicHeadWindowState tailState{};
            for (uint64_t ordinal = cursor.tailBegin;
                 ordinal < cursor.tailEnd; ++ordinal) {
                OwnedChunkDesc chunk{};
                if (LoadOwnedTailChunk(plan, ordinal, chunk)) {
                    ProcessOwnedChunkAicHeadWindowTask(
                        chunk, 0, HV_, tailState);
                }
            }
            return;
        }

        AicHeadWindowState fullState{};
        GroupedFullTaskIterator fullIterator{};
        fullIterator.chunks.sequence = cursor.fullStartSequence;
        fullIterator.chunks.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t task = cursor.fullBegin;
             task < cursor.fullEnd; ++task) {
            OwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadGroupedFullTask(
                    plan, task, fullIterator, chunk,
                    headBegin, headEnd)) {
                ProcessOwnedChunkAicHeadWindowTask(
                    chunk, headBegin, headEnd, fullState);
            }
        }

        AicHeadWindowState tailState{};
        GroupedTailTaskIterator tailIterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            OwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadGroupedTailTask(
                    plan, task, tailIterator, chunk,
                    headBegin, headEnd)) {
                ProcessOwnedChunkAicHeadWindowTask(
                    chunk, headBegin, headEnd, tailState);
            }
        }
    }

    template <typename PostWuOp>
    __aicore__ inline void DrainTailSinglePostWu(
        PostWuOp &postWu, PendingSinglePostWuState &pending)
    {
        if (!pending.valid) {
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(
            postWuReadyFlag_);
        postWu.ProcessPreparedTailSingleArch35(
            pending.b, pending.hv, pending.start, pending.rows);
        pending.valid = false;
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessOwnedChunkAicHeadWindowFusedTail(
        PostWuOp &postWu, const OwnedChunkDesc &chunk,
        uint64_t headBegin, uint32_t headCnt, uint64_t &localTaskIdx,
        PendingSinglePostWuState &pending)
    {
        for (uint32_t headOffset = 0; headOffset < headCnt;
             ++headOffset, ++localTaskIdx) {
            const uint64_t hv = headBegin + headOffset;
            activeSolveSlot_ = SolvePipelineSlot(localTaskIdx);
            // tail不参与full batch聚合；消费上一项后再生产当前项，
            // 避免单项流水在等待下一项时占满ready credit。
            DrainTailSinglePostWu(postWu, pending);
            ProcessChunkPreAic(
                chunk.b, hv, chunk.chunkIdx, chunk.start, chunk.end);
            pending.valid = true;
            pending.b = chunk.b;
            pending.hv = hv;
            pending.start = chunk.start;
            pending.rows = chunk.end - chunk.start;
        }
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessOwnedChunkAicHeadsFusedTail(
        PostWuOp &postWu, const OwnedChunkDesc &chunk,
        uint64_t headBegin, uint64_t headEnd, uint64_t &localTaskIdx,
        PendingSinglePostWuState &pending)
    {
        for (uint64_t head = headBegin; head < headEnd;) {
            uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(head), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - head) {
                headCnt = static_cast<uint32_t>(headEnd - head);
            }
            ProcessOwnedChunkAicHeadWindowFusedTail(
                postWu, chunk, head, headCnt, localTaskIdx, pending);
            head += headCnt;
        }
    }

    template <typename PostWuOp>
    __aicore__ inline void FlushFusedFullPostWuBatch(
        PostWuOp &postWu, uint64_t *batchB, uint64_t *batchHv,
        uint64_t *batchStart, uint16_t batchCount)
    {
        if (batchCount == 0) {
            return;
        }
        // 每个实际HV由两个AIV各提交一次，mode2汇聚成AIC消费的一份
        // ready token；无论headCnt为1还是4都复用同一深度4信号流。
        for (uint16_t i = 0; i < batchCount; ++i) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(
                postWuReadyFlag_);
        }
        postWu.ProcessPreparedFullHeadBatchArch35(
            batchB, batchHv, batchStart, batchCount);
    }

    template <typename PostWuOp>
    __aicore__ inline void EnqueueFusedFullPostWuTask(
        PostWuOp &postWu, uint64_t b, uint64_t hv, uint64_t start,
        FusedFullHeadWindowState &state)
    {
        state.batchB[state.batchCount] = b;
        state.batchHv[state.batchCount] = hv;
        state.batchStart[state.batchCount] = start;
        ++state.batchCount;
        if (state.batchCount != KDA_POST_QUEUE_DEPTH) {
            return;
        }
        FlushFusedFullPostWuBatch(
            postWu, state.batchB, state.batchHv, state.batchStart,
            KDA_POST_QUEUE_DEPTH);
        state.batchCount = 0;
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessOwnedChunkAicHeadWindowFusedFull(
        PostWuOp &postWu, const OwnedChunkDesc &chunk,
        uint64_t headBegin, uint32_t headCnt,
        FusedFullHeadWindowState &state)
    {
        for (uint32_t headOffset = 0; headOffset < headCnt;
             ++headOffset, ++state.localTaskIdx) {
            const uint64_t hv = headBegin + headOffset;
            activeSolveSlot_ = SolvePipelineSlot(state.localTaskIdx);
            ProcessChunkPreAic(
                chunk.b, hv, chunk.chunkIdx, chunk.start, chunk.end);
            EnqueueFusedFullPostWuTask(
                postWu, chunk.b, hv, chunk.start, state);
        }
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessOwnedChunkAicHeadsFusedFull(
        PostWuOp &postWu, const OwnedChunkDesc &chunk,
        uint64_t headBegin, uint64_t headEnd,
        FusedFullHeadWindowState &state)
    {
        for (uint64_t head = headBegin; head < headEnd;) {
            uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(head), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - head) {
                headCnt = static_cast<uint32_t>(headEnd - head);
            }
            ProcessOwnedChunkAicHeadWindowFusedFull(
                postWu, chunk, head, headCnt, state);
            head += headCnt;
        }
    }

    template <typename PostWuOp>
    __aicore__ inline void DrainFusedFullHeadWindowState(
        PostWuOp &postWu, FusedFullHeadWindowState &state)
    {
        FlushFusedFullPostWuBatch(
            postWu, state.batchB, state.batchHv,
            state.batchStart, state.batchCount);
        state.localTaskIdx = 0;
        state.batchCount = 0;
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessOwnedChunkAicHeadWindowTaskFusedFull(
        PostWuOp &postWu, const OwnedChunkDesc &chunk,
        uint64_t headBegin, uint64_t headEnd,
        FusedFullHeadWindowState &state)
    {
        ProcessOwnedChunkAicHeadsFusedFull(
            postWu, chunk, headBegin, headEnd, state);
    }

    template <typename PostWuOp>
    __aicore__ inline void DrainFusedTailHeadWindowState(
        PostWuOp &postWu, FusedTailHeadWindowState &state)
    {
        DrainTailSinglePostWu(postWu, state.singlePending);
        state.localTaskIdx = 0;
        state.singlePending.valid = false;
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessOwnedChunkAicHeadWindowTaskFusedTail(
        PostWuOp &postWu, const OwnedChunkDesc &chunk,
        uint64_t headBegin, uint64_t headEnd,
        FusedTailHeadWindowState &state)
    {
        ProcessOwnedChunkAicHeadsFusedTail(
            postWu, chunk, headBegin, headEnd,
            state.localTaskIdx, state.singlePending);
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessPreAicHeadWindowsFused(PostWuOp &postWu)
    {
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        KdaForward::ChunkCoreCursor cursor{};
        if (!plan.LoadChunkCoreCursor(static_cast<uint32_t>(coreIdx), cursor)) {
            return;
        }
        if (plan.HeadGroupCount() == 1) {
            FusedFullHeadWindowState fullState{};
            FullChunkIterator fullIterator{};
            fullIterator.sequence = cursor.fullStartSequence;
            fullIterator.localChunk = cursor.fullStartLocalChunk;
            for (uint64_t ordinal = cursor.fullBegin;
                 ordinal < cursor.fullEnd; ++ordinal) {
                OwnedChunkDesc chunk{};
                if (LoadOwnedFullChunk(plan, fullIterator, chunk)) {
                    ProcessOwnedChunkAicHeadWindowTaskFusedFull(
                        postWu, chunk, 0, HV_, fullState);
                }
            }
            DrainFusedFullHeadWindowState(postWu, fullState);

            FusedTailHeadWindowState tailState{};
            for (uint64_t ordinal = cursor.tailBegin;
                 ordinal < cursor.tailEnd; ++ordinal) {
                OwnedChunkDesc chunk{};
                if (LoadOwnedTailChunk(plan, ordinal, chunk)) {
                    ProcessOwnedChunkAicHeadWindowTaskFusedTail(
                        postWu, chunk, 0, HV_, tailState);
                }
            }
            DrainFusedTailHeadWindowState(postWu, tailState);
            return;
        }

        FusedFullHeadWindowState fullState{};
        GroupedFullTaskIterator fullIterator{};
        fullIterator.chunks.sequence = cursor.fullStartSequence;
        fullIterator.chunks.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t task = cursor.fullBegin;
             task < cursor.fullEnd; ++task) {
            OwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadGroupedFullTask(
                    plan, task, fullIterator, chunk,
                    headBegin, headEnd)) {
                ProcessOwnedChunkAicHeadWindowTaskFusedFull(
                    postWu, chunk, headBegin, headEnd, fullState);
            }
        }
        DrainFusedFullHeadWindowState(postWu, fullState);

        FusedTailHeadWindowState tailState{};
        GroupedTailTaskIterator tailIterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            OwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadGroupedTailTask(
                    plan, task, tailIterator, chunk,
                    headBegin, headEnd)) {
                ProcessOwnedChunkAicHeadWindowTaskFusedTail(
                    postWu, chunk, headBegin, headEnd, tailState);
            }
        }
        DrainFusedTailHeadWindowState(postWu, tailState);
    }
    __aicore__ inline void ProcessPreAiv()
    {
        if constexpr (IsSameType<T, float>::value) {
            isAivOnly_ = true;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (!IsSameType<T, float>::value) {
            constexpr bool useRuntimeHeadWindows =
                SAFE_GATE || (COMPILE_BT == 64 && COMPILE_K == 128 &&
                              COMPILE_V == 128);
            if constexpr (useRuntimeHeadWindows) {
                KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
                if (!isAivOnly_ && plan.IsValid()) {
                    ProcessPreAivHeadWindows();
                    return;
                }
            }
        }
#endif
        uint64_t subBlockNum = isAivOnly_ ? 1 : static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        uint64_t subBlockIdx = isAivOnly_ ? 0 : static_cast<uint64_t>(GetSubBlockIdx());
        uint64_t coreNum = isAivOnly_ ? static_cast<uint64_t>(GetBlockNum()) : usedCoreNum_;
        uint64_t coreIdx = isAivOnly_ ? static_cast<uint64_t>(GetBlockIdx()) :
                                        static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        if constexpr (SAFE_GATE && !IsSameType<T, float>::value) {
            bool pendingValid = false;
            uint64_t pendingB = 0;
            uint64_t pendingHv = 0;
            uint64_t pendingChunkIdx = 0;
            uint64_t pendingStart = 0;
            uint64_t pendingEnd = 0;
            uint64_t pendingSlot = 0;
            uint64_t localTaskIdx = 0;
            for (uint64_t task = coreIdx; task < taskNum; task += coreNum, ++localTaskIdx) {
                uint64_t seq = 0;
                uint64_t b = 0;
                uint64_t h = 0;
                uint64_t hv = 0;
                uint64_t chunkIdx = 0;
                uint64_t start = 0;
                uint64_t end = 0;
                if (!ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                    continue;
                }
                (void)seq;
                uint64_t currentSlot = localTaskIdx % KDA_SOLVE_PIPELINE_DEPTH;
                activeSolveSlot_ = currentSlot;
                bool deferSolve = UseAkkCubeSolve(end - start);
                if (!deferSolve && pendingValid) {
                    WaitAicSolveDone();
                    activeSolveSlot_ = pendingSlot;
                    FinishDeferredSafeChunk(pendingB, pendingHv, pendingChunkIdx, pendingStart, pendingEnd,
                                            subBlockIdx, subBlockNum);
                    pendingValid = false;
                    activeSolveSlot_ = currentSlot;
                }
                ProcessChunkPreAivFp32(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum,
                                      deferSolve, pendingValid);
                if (pendingValid) {
                    activeSolveSlot_ = pendingSlot;
                    FinishDeferredSafeChunk(pendingB, pendingHv, pendingChunkIdx, pendingStart, pendingEnd,
                                            subBlockIdx, subBlockNum);
                }
                pendingValid = deferSolve;
                if (pendingValid) {
                    pendingB = b;
                    pendingHv = hv;
                    pendingChunkIdx = chunkIdx;
                    pendingStart = start;
                    pendingEnd = end;
                    pendingSlot = currentSlot;
                }
            }
            if (pendingValid) {
                WaitAicSolveDone();
                activeSolveSlot_ = pendingSlot;
                FinishDeferredSafeChunk(pendingB, pendingHv, pendingChunkIdx, pendingStart, pendingEnd,
                                        subBlockIdx, subBlockNum);
            }
            return;
        }
        for (uint64_t task = coreIdx; task < taskNum; task += coreNum) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                (void)seq;
                ProcessChunkPreAiv(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessPreAic()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE ||
                      (COMPILE_BT == 64 && COMPILE_K == 128 &&
                       COMPILE_V == 128)) {
            KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
            if (plan.IsValid()) {
                ProcessPreAicHeadWindows();
                return;
            }
        }
#endif
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t localTaskIdx = 0;
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += coreNum, ++localTaskIdx) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                if constexpr (SAFE_GATE) {
                    activeSolveSlot_ = localTaskIdx % KDA_SOLVE_PIPELINE_DEPTH;
                }
                (void)seq;
                (void)h;
                ProcessChunkPreAic(b, hv, chunkIdx, start, end);
            }
        }
    }


private:
    GlobalTensor<T> q_;
    GlobalTensor<T> k_;
    GlobalTensor<T> v_;
    GlobalTensor<GK_T> gk_;
    GlobalTensor<float> rawG_;
    GlobalTensor<A_LOG_T> aLog_;
    GlobalTensor<DT_BIAS_T> dtBias_;
    GlobalTensor<BETA_T> beta_;
    GlobalTensor<float> initialState_;
    GlobalTensor<OUT_T> o_;
    GlobalTensor<float> finalState_;
    GlobalTensor<float> aqk_;
    GlobalTensor<AKK_T> akk_;
    GlobalTensor<T> w_;
    GlobalTensor<OUT_T> u_;
    GlobalTensor<T> qg_;
    GlobalTensor<T> kg_;
    GlobalTensor<T> vNew_;
    GlobalTensor<float> h_;
    GlobalTensor<T> finalKg_;
    GlobalTensor<T> preparedQG_;
    GlobalTensor<T> preparedAqk_;
    GlobalTensor<T> propagatedVNew_;
    GlobalTensor<T> propagatedH_;
    GlobalTensor<float> solveWorkspace_;
    GlobalTensor<SCORE_T> scoreWorkspace_;
    TPipe *pipe_ = nullptr;
    TEventID solveFixToMte2Event_ = 0;
    TBuf<TPosition::VECCALC> exp2Buf_;
    TBuf<TPosition::VECCALC> vecBuf_;
    TBuf<TPosition::VECCALC> gateWritebackBuf_;
    TEventID mte2ToVEvent_ = 0;
    TEventID vToMte2Event_ = 0;
    TEventID vToMte3Event_ = 0;
    TEventID mte3ToVEvent_ = 0;
    TEventID mte2ToMte3Event_ = 0;
    TEventID vToSEvent_ = 0;
    TEventID mte3ToMte2Events_[KDA_GATE_PIPELINE_DEPTH] = {0, 0, 0};
    bool vectorEventsAllocated_ = false;
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreReadyFlag_{KDA_SCORE_READY_FLAG0,
                                                                                  KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreDoneFlag_{KDA_SCORE_DONE_FLAG0,
                                                                                 KDA_SCORE_DONE_FLAG1};
    // 每个核的 Solve 最多只有一个在途任务，因此复用主 score ID 作为有序 token 流；
    // score credit 仍保留在反向 ID 上，不额外消耗硬件 flag ID。
    Catlass::Arch::CrossCoreFlag syncReadyFlag_{KDA_SOLVE_READY_FLAG};
    Catlass::Arch::CrossCoreFlag syncDoneFlag_{KDA_SOLVE_DONE_FLAG};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_POST_QUEUE_DEPTH> postWuReadyFlag_{
        KDA_POST_READY_FLAG, KDA_POST_FREE_FLAG};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SYNC_REVERSE_DEPTH> mchSyncReadyFlag_{
        KDA_SCORE_READY_FLAG0, KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SYNC_REVERSE_DEPTH> mchSyncDoneFlag_{
        KDA_SCORE_DONE_FLAG0, KDA_SCORE_DONE_FLAG1};
    uint64_t B_ = 0;
    uint64_t N_ = 0;
    uint64_t H_ = 0;
    uint64_t HV_ = 0;
    uint64_t T_ = 0;
    uint64_t K_ = 0;
    uint64_t V_ = 0;
    uint64_t BT_ = 0;
    uint64_t NT_ = 0;
    float scale_ = 1.0f;
    bool hasInitial_ = false;
    bool isVarLen_ = false;
    bool isAivOnly_ = false;
    bool inputSequenceMajor_ = false;
    bool fusePostWu_ = false;
    bool computeGateInPrepare_ = false;
    bool hasALog_ = false;
    bool hasDtBias_ = false;
    bool storeQG_ = true;
    bool rawQkResidentEnabled_ = false;
    bool rawQkResidentHasVectorReader_ = false;
    float lowerBound_ = -5.0f;
    uint64_t usedCoreNum_ = 1;
    uint64_t solveCoreIdx_ = 0;
    uint64_t activeSolveSlot_ = 0;
    uint64_t activeGateChunkStart_ = 0;
    uint64_t rawQkResidentB_ = 0;
    uint64_t rawQkResidentH_ = 0;
    uint64_t rawQkResidentTokenBegin_ = 0;
    uint64_t rawQkResidentTokenEnd_ = 0;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
    __gm__ int64_t *cuSeqlensAddr_ = nullptr;
    GM_ADDR compactPlanAddr_ = nullptr;
};
} // namespace

template <bool SAFE_GATE, typename T, typename GK_T,
          typename BETA_T, typename A_LOG_T, typename DT_BIAS_T,
          typename TilingData,
          uint32_t COMPILE_BT = 0, uint32_t COMPILE_K = 0, uint32_t COMPILE_V = 0>
__aicore__ inline void RunChunkKdaPrepare(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR rawG, GM_ADDR aLog,
    GM_ADDR dtBias, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR compactPlan,
    GM_ADDR aqk, GM_ADDR akk, GM_ADDR qg,
    GM_ADDR qgScaled, GM_ADDR wSeed, GM_ADDR uSeed, GM_ADDR finalKg, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe, bool storeQG = true)
{
    GM_ADDR aqkFp32 = userWorkspace + tiling.prepareAqkFp32Offset;
    GM_ADDR akkFp32 = userWorkspace + tiling.prepareAkkFp32Offset;
    GM_ADDR prepareScratch = userWorkspace + tiling.prepareScratchOffset;

    if ASCEND_IS_AIC {
        ChunkKdaFwdPrepareKernel<SAFE_GATE, T, GK_T,
            BETA_T, A_LOG_T, DT_BIAS_T, COMPILE_BT, COMPILE_K, COMPILE_V> op;
        op.Init(q, k, v, gk, rawG, aLog, dtBias, beta, initialState, cuSeqlens, chunkIndices,
                compactPlan,
                nullptr, nullptr, nullptr, nullptr, aqk, userWorkspace, aqkFp32, akkFp32,
                wSeed, akk, qg, qgScaled, uSeed, userWorkspace, finalKg,
                prepareScratch, tiling, &pipe, false, storeQG);
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 &&
                      COMPILE_V == 128 && IsSameType<T, bfloat16_t>::value) {
            if (tiling.fusePostWu) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                KdaPostWu::ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> postWu;
                postWu.Init(nullptr, k, nullptr, gk, beta, initialState, cuSeqlens, chunkIndices,
                            compactPlan,
                            wSeed, akk, uSeed, nullptr, userWorkspace, userWorkspace, userWorkspace,
                            akk, wSeed, uSeed, userWorkspace, finalKg, userWorkspace,
                            prepareScratch, prepareScratch, tiling, &pipe, false);
                op.ProcessAicFused(postWu);
#else
                op.ProcessAic();
#endif
            } else {
                op.ProcessAic();
            }
        } else {
            op.ProcessAic();
        }
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdPrepareKernel<SAFE_GATE, T, GK_T,
            BETA_T, A_LOG_T, DT_BIAS_T, COMPILE_BT, COMPILE_K, COMPILE_V> op;
        op.Init(q, k, v, gk, rawG, aLog, dtBias, beta, initialState, cuSeqlens, chunkIndices,
                compactPlan,
                nullptr, nullptr, nullptr, nullptr, aqk, userWorkspace, aqkFp32, akkFp32,
                wSeed, akk, qg, qgScaled, uSeed, userWorkspace, finalKg,
                prepareScratch, tiling, &pipe, true, storeQG);
        op.ProcessAiv();
    }
}

} // namespace KdaPrepare
