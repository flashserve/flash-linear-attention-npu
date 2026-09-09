/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_ARCH35_CHUNK_KDA_FWD_PREPARE_VEC_H
#define PSEUDOCODE_ARCH35_CHUNK_KDA_FWD_PREPARE_VEC_H

#include <type_traits>

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"

#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace KdaPrepare::Arch35 {

namespace Detail {

using namespace AscendC::MicroAPI;

constexpr static CastTrait kFp32ToBf16RintZero = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr static CastTrait kFp32ToBf16RintOne = {
    RegLayout::ONE,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__simd_callee__ inline void CastFp32ToBf16Rint(
    RegTensor<bfloat16_t> &dst, RegTensor<float> &low,
    RegTensor<float> &high,
    MaskReg &mask)
{
    Cast<bfloat16_t, float, kFp32ToBf16RintOne>(dst, high, mask);
    Cast<bfloat16_t, float, kFp32ToBf16RintZero>(dst, low, mask);
}

template <typename T>
__simd_callee__ inline void Load128AsFp32(
    RegTensor<float> &low, RegTensor<float> &high, __ubuf__ T *src)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_NORM>(low, src);
        LoadAlign<float, LoadDist::DIST_NORM>(high, src + 64);
    } else {
        RegTensor<T> packed;
        LoadIn<T, false>(packed, src);
        MaskReg packedMask = CreateMask<T, MaskPattern::ALL>();
        CastHalf2Float<T>(low, high, packed, packedMask);
    }
}

__simd_callee__ inline void Store128FromFp32(
    __ubuf__ bfloat16_t *dst, RegTensor<float> &low,
    RegTensor<float> &high)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<bfloat16_t> packed;
    CastFp32ToBf16Rint(packed, low, high, floatMask);
    MaskReg packedMask = CreateMask<bfloat16_t, MaskPattern::ALL>();
    StoreAlign(dst, packed, packedMask);
}

__simd_callee__ inline void Store64FromFp32(
    __ubuf__ bfloat16_t *dst, RegTensor<float> &value)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero;
    RegTensor<bfloat16_t> packed;
    Duplicate(zero, 0.0F, floatMask);
    CastFp32ToBf16Rint(packed, value, zero, floatMask);
    uint32_t active = 64;
    MaskReg outputMask = UpdateMask<bfloat16_t>(active);
    StoreAlign(dst, packed, outputMask);
}

__simd_callee__ inline void Store32FromFp32(
    __ubuf__ bfloat16_t *dst, RegTensor<float> &value)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero;
    RegTensor<bfloat16_t> packed;
    Duplicate(zero, 0.0F, floatMask);
    CastFp32ToBf16Rint(packed, value, zero, floatMask);
    uint32_t active = 32;
    MaskReg outputMask = UpdateMask<bfloat16_t>(active);
    StoreAlign(dst, packed, outputMask);
}

template <typename T>
__simd_callee__ inline void LoadScalarAsFp32(
    RegTensor<float> &dst, __ubuf__ T *src)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadIn<float, true>(dst, src);
    } else {
        RegTensor<T> raw;
        RegTensor<float> unused;
        MaskReg inputMask = CreateMask<T, MaskPattern::ALL>();
        LoadIn<T, true>(raw, src);
        CastHalf2Float<T>(dst, unused, raw, inputMask);
    }
}

template <bool USE_EXP2>
__simd_callee__ inline void ExpPair(
    RegTensor<float> &low, RegTensor<float> &high, float lower, float upper)
{
    using Domain = ExpDomainTraits<USE_EXP2>;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    Maxs(low, low, lower, mask);
    Maxs(high, high, lower, mask);
    Mins(low, low, upper, mask);
    Mins(high, high, upper, mask);
    if constexpr (Domain::useExp2) {
        Muls(low, low, Domain::expInputScale, mask);
        Muls(high, high, Domain::expInputScale, mask);
    }
    Exp(low, low, mask);
    Exp(high, high, mask);
}

// V0 的循环和指令均属于一次 VF。这里直接展示寄存器级数据流；L2 归约
// 的 ReduceSum 结果广播形式仍需用目标 CANN 9.1.0 头文件做最小编译确认。
template <typename GateT, typename BetaT, typename CompilePolicy>
__simd_vf__ inline void StageV0Vf(
    __ubuf__ bfloat16_t *q, __ubuf__ bfloat16_t *k,
    __ubuf__ GateT *rawGate,
    __ubuf__ BetaT *betaRaw, __ubuf__ float *dtBias, __ubuf__ float *aLog,
    __ubuf__ float *g, __ubuf__ float *gRef, __ubuf__ float *gLast,
    __ubuf__ float *betaEff, uint16_t validRows, float epsilon,
    float lowerBound, bool hasDtBias, bool hasALog)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    uint32_t scalarCount = 1;
    MaskReg scalarMask = UpdateMask<float>(scalarCount);
    RegTensor<float> carryLow;
    RegTensor<float> carryHigh;
    Duplicate(carryLow, 0.0F, mask);
    Duplicate(carryHigh, 0.0F, mask);

    for (uint16_t row = 0; row < Shape::kChunkRows; ++row) {
        RegTensor<float> qLow;
        RegTensor<float> qHigh;
        RegTensor<float> kLow;
        RegTensor<float> kHigh;
        if (row < validRows) {
            Load128AsFp32(qLow, qHigh, q + row * Shape::kHeadDim);
            Load128AsFp32(kLow, kHigh, k + row * Shape::kHeadDim);
            if constexpr (CompilePolicy::normMode == QkNormMode::L2) {
                RegTensor<float> qSquareLow;
                RegTensor<float> qSquareHigh;
                RegTensor<float> kSquareLow;
                RegTensor<float> kSquareHigh;
                RegTensor<float> qSumLow;
                RegTensor<float> qSumHigh;
                RegTensor<float> kSumLow;
                RegTensor<float> kSumHigh;
                Mul(qSquareLow, qLow, qLow, mask);
                Mul(qSquareHigh, qHigh, qHigh, mask);
                Mul(kSquareLow, kLow, kLow, mask);
                Mul(kSquareHigh, kHigh, kHigh, mask);
                ReduceSum(qSumLow, qSquareLow, mask);
                ReduceSum(qSumHigh, qSquareHigh, mask);
                ReduceSum(kSumLow, kSquareLow, mask);
                ReduceSum(kSumHigh, kSquareHigh, mask);
                // ReduceSum 只保证首 lane 有效。先按单 lane 合并两半，
                // 按冻结语义加 epsilon 后开方，再借 betaEff 的未使用标量区
                // 落 UB 并显式广播。
                Add(qSumLow, qSumLow, qSumHigh, scalarMask);
                Add(kSumLow, kSumLow, kSumHigh, scalarMask);
                Adds(qSumLow, qSumLow, epsilon, scalarMask);
                Adds(kSumLow, kSumLow, epsilon, scalarMask);
                Sqrt(qSumLow, qSumLow, scalarMask);
                Sqrt(kSumLow, kSumLow, scalarMask);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    betaEff + row, qSumLow, scalarMask);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    betaEff + Shape::kChunkRows + row, kSumLow,
                    scalarMask);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    qSumLow, betaEff + row);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    kSumLow, betaEff + Shape::kChunkRows + row);
                Div(qLow, qLow, qSumLow, mask);
                Div(qHigh, qHigh, qSumLow, mask);
                Div(kLow, kLow, kSumLow, mask);
                Div(kHigh, kHigh, kSumLow, mask);
            }
        } else {
            Duplicate(qLow, 0.0F, mask);
            Duplicate(qHigh, 0.0F, mask);
            Duplicate(kLow, 0.0F, mask);
            Duplicate(kHigh, 0.0F, mask);
        }
        Store128FromFp32(q + row * Shape::kHeadDim, qLow, qHigh);
        Store128FromFp32(k + row * Shape::kHeadDim, kLow, kHigh);

        if (row >= validRows) {
            RegTensor<float> zero;
            Duplicate(zero, 0.0F, mask);
            StoreAlign(g + row * Shape::kHeadDim, zero, mask);
            StoreAlign(g + row * Shape::kHeadDim + 64, zero, mask);
            continue;
        }
        RegTensor<float> gateLow;
        RegTensor<float> gateHigh;
        Load128AsFp32(gateLow, gateHigh,
                      rawGate + row * Shape::kHeadDim);
        if constexpr (CompilePolicy::gateMode != GateMode::PrecomputedStep) {
            RegTensor<float> biasLow;
            RegTensor<float> biasHigh;
            if (hasDtBias) {
                LoadAlign(biasLow, dtBias);
                LoadAlign(biasHigh, dtBias + 64);
            } else {
                Duplicate(biasLow, 0.0F, mask);
                Duplicate(biasHigh, 0.0F, mask);
            }
            Add(gateLow, gateLow, biasLow, mask);
            Add(gateHigh, gateHigh, biasHigh, mask);

            RegTensor<float> a;
            if (hasALog) {
                LoadScalarAsFp32(a, aLog);
                Exp(a, a, mask); // 计算 a_h=exp(A_log[h])。
            } else {
                Duplicate(a, 1.0F, mask);
            }
            if constexpr (CompilePolicy::safeGate ||
                          CompilePolicy::gateMode == GateMode::SafeSigmoid) {
                RegTensor<float> one;
                Duplicate(one, 1.0F, mask);
                Mul(gateLow, gateLow, a, mask);
                Mul(gateHigh, gateHigh, a, mask);
                Muls(gateLow, gateLow, -1.0F, mask);
                Muls(gateHigh, gateHigh, -1.0F, mask);
                Exp(gateLow, gateLow, mask);
                Exp(gateHigh, gateHigh, mask);
                Adds(gateLow, gateLow, 1.0F, mask);
                Adds(gateHigh, gateHigh, 1.0F, mask);
                Div(gateLow, one, gateLow, mask);
                Div(gateHigh, one, gateHigh, mask);
                Muls(gateLow, gateLow, lowerBound, mask);
                Muls(gateHigh, gateHigh, lowerBound, mask);
            } else {
                RegTensor<float> positiveLow;
                RegTensor<float> positiveHigh;
                RegTensor<float> softplusLow;
                RegTensor<float> softplusHigh;
                Maxs(positiveLow, gateLow, 0.0F, mask);
                Maxs(positiveHigh, gateHigh, 0.0F, mask);
                Abs(softplusLow, gateLow, mask);
                Abs(softplusHigh, gateHigh, mask);
                Muls(softplusLow, softplusLow, -1.0F, mask);
                Muls(softplusHigh, softplusHigh, -1.0F, mask);
                Exp(softplusLow, softplusLow, mask);
                Exp(softplusHigh, softplusHigh, mask);
                Adds(softplusLow, softplusLow, 1.0F, mask);
                Adds(softplusHigh, softplusHigh, 1.0F, mask);
                Ln(softplusLow, softplusLow, mask);
                Ln(softplusHigh, softplusHigh, mask);
                Add(gateLow, positiveLow, softplusLow, mask);
                Add(gateHigh, positiveHigh, softplusHigh, mask);
                Mul(gateLow, gateLow, a, mask);
                Mul(gateHigh, gateHigh, a, mask);
                Muls(gateLow, gateLow, -1.0F, mask);
                Muls(gateHigh, gateHigh, -1.0F, mask);
            }
        }
        // USE_EXP2=true 时 G 保存 log2 值，后续用 Exp(G*ln2)；
        // false 时 G 保存自然对数值，后续直接 Exp(G)。
        if constexpr (Domain::useExp2) {
            Muls(gateLow, gateLow, Domain::stepScale, mask);
            Muls(gateHigh, gateHigh, Domain::stepScale, mask);
        }
        Add(carryLow, carryLow, gateLow, mask);
        Add(carryHigh, carryHigh, gateHigh, mask);
        StoreAlign(g + row * Shape::kHeadDim, carryLow, mask);
        StoreAlign(g + row * Shape::kHeadDim + 64, carryHigh, mask);

        for (uint16_t s = 0; s < Shape::kSubChunkCount; ++s) {
            const uint16_t begin = s * Shape::kSubChunkRows;
            const uint16_t end = validRows < begin + Shape::kSubChunkRows
                                     ? validRows
                                     : begin + Shape::kSubChunkRows;
            const uint16_t refRow = (begin + end) / 2;
            if (begin < end && row == refRow) {
                StoreAlign(gRef + s * Shape::kHeadDim, carryLow, mask);
                StoreAlign(gRef + s * Shape::kHeadDim + 64, carryHigh, mask);
            }
        }
        if (row + 1 == validRows) {
            StoreAlign(gLast, carryLow, mask);
            StoreAlign(gLast + 64, carryHigh, mask);
        }
    }

    for (uint16_t row = 0; row < Shape::kChunkRows; ++row) {
        RegTensor<float> beta;
        if (row < validRows) {
            RegTensor<float> one;
            LoadScalarAsFp32(beta, betaRaw + row);
            Duplicate(one, 1.0F, mask);
            if constexpr (CompilePolicy::betaMode != BetaMode::Raw) {
                Muls(beta, beta, -1.0F, mask);
                Exp(beta, beta, mask);
                Adds(beta, beta, 1.0F, mask);
                Div(beta, one, beta, mask);
                if constexpr (CompilePolicy::betaMode == BetaMode::TwoSigmoid) {
                    Muls(beta, beta, 2.0F, mask);
                }
            }
        } else {
            Duplicate(beta, 0.0F, mask);
        }
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            betaEff + row, beta, mask);
    }
}

template <typename CompilePolicy>
__simd_vf__ inline void StageV1Vf(
    __ubuf__ bfloat16_t *qHat, __ubuf__ bfloat16_t *kHat,
    __ubuf__ float *g, __ubuf__ float *gRef,
    __ubuf__ bfloat16_t *qPlus, __ubuf__ bfloat16_t *kPlus,
    __ubuf__ bfloat16_t *kMinus, uint16_t validRows)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    constexpr uint32_t prefixBase[4] = {0, 16 * 128, 48 * 128, 96 * 128};

    // 先生成 Kminus，再原位覆盖 qHat/kHat 为 Qplus/Kplus，避免额外保存 Khat。
    for (uint16_t s = 0; s < Shape::kSubChunkCount; ++s) {
        const uint16_t prefixRows = Shape::kPrefixRows[s];
        const uint16_t bandBegin = s * Shape::kSubChunkRows;
        for (uint16_t row = 0; row < prefixRows; ++row) {
            RegTensor<float> outLow;
            RegTensor<float> outHigh;
            if (bandBegin < validRows && row < validRows) {
                RegTensor<float> gateLow;
                RegTensor<float> gateHigh;
                RegTensor<float> refLow;
                RegTensor<float> refHigh;
                RegTensor<float> kLow;
                RegTensor<float> kHigh;
                LoadAlign(gateLow, g + row * Shape::kHeadDim);
                LoadAlign(gateHigh, g + row * Shape::kHeadDim + 64);
                LoadAlign(refLow, gRef + s * Shape::kHeadDim);
                LoadAlign(refHigh, gRef + s * Shape::kHeadDim + 64);
                Sub(refLow, refLow, gateLow, mask);
                Sub(refHigh, refHigh, gateHigh, mask);
                constexpr float base2Lower =
                    ExpDomain::kV1Bf16LowerBase2;
                constexpr float base2Upper =
                    ExpDomain::kV1Bf16UpperBase2;
                constexpr float lower = Domain::StoredBound(base2Lower);
                constexpr float upper = Domain::StoredBound(base2Upper);
                ExpPair<CompilePolicy::useExp2>(
                    refLow, refHigh, lower, upper);
                Load128AsFp32(kLow, kHigh,
                              kHat + row * Shape::kHeadDim);
                Mul(outLow, kLow, refLow, mask);
                Mul(outHigh, kHigh, refHigh, mask);
            } else {
                Duplicate(outLow, 0.0F, mask);
                Duplicate(outHigh, 0.0F, mask);
            }
            Store128FromFp32(
                kMinus + prefixBase[s] + row * Shape::kHeadDim,
                outLow, outHigh);
        }
    }

    for (uint16_t row = 0; row < Shape::kChunkRows; ++row) {
        RegTensor<float> qLow;
        RegTensor<float> qHigh;
        RegTensor<float> kLow;
        RegTensor<float> kHigh;
        RegTensor<float> expLow;
        RegTensor<float> expHigh;
        if (row < validRows) {
            const uint16_t s = row / Shape::kSubChunkRows;
            LoadAlign(expLow, g + row * Shape::kHeadDim);
            LoadAlign(expHigh, g + row * Shape::kHeadDim + 64);
            RegTensor<float> refLow;
            RegTensor<float> refHigh;
            LoadAlign(refLow, gRef + s * Shape::kHeadDim);
            LoadAlign(refHigh, gRef + s * Shape::kHeadDim + 64);
            Sub(expLow, expLow, refLow, mask);
            Sub(expHigh, expHigh, refHigh, mask);
            constexpr float base2Lower =
                ExpDomain::kV1Bf16LowerBase2;
            constexpr float base2Upper =
                ExpDomain::kV1Bf16UpperBase2;
            constexpr float lower = Domain::StoredBound(base2Lower);
            constexpr float upper = Domain::StoredBound(base2Upper);
            ExpPair<CompilePolicy::useExp2>(
                expLow, expHigh, lower, upper);
            Load128AsFp32(qLow, qHigh, qHat + row * Shape::kHeadDim);
            Load128AsFp32(kLow, kHigh, kHat + row * Shape::kHeadDim);
            Mul(qLow, qLow, expLow, mask);
            Mul(qHigh, qHigh, expHigh, mask);
            Mul(kLow, kLow, expLow, mask);
            Mul(kHigh, kHigh, expHigh, mask);
        } else {
            Duplicate(qLow, 0.0F, mask);
            Duplicate(qHigh, 0.0F, mask);
            Duplicate(kLow, 0.0F, mask);
            Duplicate(kHigh, 0.0F, mask);
        }
        Store128FromFp32(qPlus + row * Shape::kHeadDim, qLow, qHigh);
        Store128FromFp32(kPlus + row * Shape::kHeadDim, kLow, kHigh);
    }
}

__simd_vf__ inline void StageV3Vf(
    __ubuf__ float *rawScore,
    __ubuf__ float *betaEff, __ubuf__ bfloat16_t *aqk,
    __ubuf__ float *lkk, __ubuf__ float *b, __ubuf__ float *x0,
    __ubuf__ float *x1, __ubuf__ float *negX1,
    __ubuf__ bfloat16_t *akk, uint16_t validRows, float scale)
{
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    uint32_t lowerColumnCount = 32;
    MaskReg lowerColumnMask = UpdateMask<float>(lowerColumnCount);
    RegTensor<int32_t> column;
    MaskReg upperColumnMask;
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(column, 0);
    CompareScalar<int32_t, CMPMODE::GE>(upperColumnMask, column, 32, full);
    constexpr uint32_t stackedBase[4] = {
        0, 32 * 16, 32 * 48, 32 * 96};

    // 从后往前展开 compact 行，避免 dense 目的区覆盖尚未读取的 compact 源。
    for (int32_t row = Shape::kChunkRows - 1; row >= 0; --row) {
        RegTensor<float> aqkRow;
        RegTensor<float> akkRow;
        RegTensor<float> zero;
        Duplicate(zero, 0.0F, full);
        if (row < validRows) {
            const uint16_t s = static_cast<uint16_t>(row) / Shape::kSubChunkRows;
            const uint16_t bandRow = static_cast<uint16_t>(row) % Shape::kSubChunkRows;
            const uint16_t columns = Shape::kPrefixRows[s];
            LoadAlign(aqkRow,
                      rawScore + stackedBase[s] + bandRow * columns);
            LoadAlign(akkRow,
                      rawScore + stackedBase[s] +
                          Shape::kSubChunkRows * columns +
                          bandRow * columns);
            uint32_t aqkCount = static_cast<uint32_t>(row) + 1;
            uint32_t akkCount = static_cast<uint32_t>(row);
            MaskReg aqkMask = UpdateMask<float>(aqkCount);
            MaskReg akkMask = UpdateMask<float>(akkCount);
            Select(aqkRow, aqkRow, zero, aqkMask);
            Select(akkRow, akkRow, zero, akkMask);
            Muls(aqkRow, aqkRow, scale, full);
            RegTensor<float> beta;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(
                beta, betaEff + row);
            Mul(akkRow, akkRow, beta, full);
        } else {
            Duplicate(aqkRow, 0.0F, full);
            Duplicate(akkRow, 0.0F, full);
        }
        // V1/C2 操作数和公开 Aqk 都固定按 BF16 RINT。
        Store64FromFp32(aqk + row * Shape::kChunkRows, aqkRow);
        if (row < 32) {
            StoreAlign(lkk + row * Shape::kChunkRows,
                       akkRow, lowerColumnMask);
        } else {
            // rawAkk 的低 32 列就是最终 B；高 32 列仅写入 L11。
            StoreAlign(b + (row - 32) * 32,
                       akkRow, lowerColumnMask);
            StoreAlign(lkk + row * Shape::kChunkRows,
                       akkRow, upperColumnMask);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // LoadAlign 每次读取 64 个 FP32 lane，先清零两个 32x32 结果区的全部
    // 物理空间，避免逐行递推读取相邻行尚未写入的高 32 lane。
    RegTensor<float> xZero;
    Duplicate(xZero, 0.0F, full);
    for (uint16_t offset = 0; offset < 32 * 32; offset += 64) {
        StoreAlign(x0 + offset, xZero, full);
        StoreAlign(x1 + offset, xZero, full);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // 两个 32x32 叶子独立执行单位下三角前代：X0=(I+L00)^-1，
    // X1=(I+L11)^-1；B=L10。下面的 VEC_STORE->VEC_LOAD 屏障是同一
    // VF 内逐行递推所必需的，不是额外 VF。
    for (uint16_t leaf = 0; leaf < 2; ++leaf) {
        const uint16_t rowBase = leaf * 32;
        uint32_t rowCount = 32;
        MaskReg rowMask = UpdateMask<float>(rowCount);
        __ubuf__ float *x = leaf == 0 ? x0 : x1;
        for (uint16_t row = 0; row < 32; ++row) {
            RegTensor<float> result;
            RegTensor<float> zero;
            RegTensor<float> one;
            RegTensor<int32_t> index;
            MaskReg diagonal;
            Duplicate(zero, 0.0F, rowMask);
            Duplicate(one, 1.0F, rowMask);
            Arange<int32_t, IndexOrder::INCREASE_ORDER>(index, 0);
            CompareScalar<int32_t, CMPMODE::EQ>(
                diagonal, index, static_cast<int32_t>(row), rowMask);
            Select(result, one, zero, diagonal);
            for (uint16_t source = 0; source < row; ++source) {
                RegTensor<float> factor;
                RegTensor<float> sourceRow;
                RegTensor<float> product;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    factor, lkk + (rowBase + row) * 64 + rowBase + source);
                LoadAlign(sourceRow, x + source * 32);
                Mul(product, sourceRow, factor, rowMask);
                Sub(result, result, product, rowMask);
            }
            StoreAlign(x + row * 32, result, rowMask);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }
    for (uint16_t row = 0; row < 32; ++row) {
        RegTensor<float> x1Row;
        uint32_t rowCount = 32;
        MaskReg rowMask = UpdateMask<float>(rowCount);
        LoadAlign(x1Row, x1 + row * 32);
        Muls(x1Row, x1Row, -1.0F, rowMask);
        StoreAlign(negX1 + row * 32, x1Row, rowMask);
    }
    // q00/q11 写入最终 64x64 行主序 Akk；q01/q10 先置零，C5 只补 q10。
    RegTensor<float> zero;
    Duplicate(zero, 0.0F, full);
    for (uint16_t row = 0; row < 32; ++row) {
        RegTensor<float> diagonalRow;
        LoadAlign(diagonalRow, x0 + row * 32);
        Store32FromFp32(akk + row * 64, diagonalRow);
        Store32FromFp32(akk + row * 64 + 32, zero);
    }
    for (uint16_t row = 0; row < 32; ++row) {
        RegTensor<float> diagonalRow;
        Store32FromFp32(akk + (row + 32) * 64, zero);
        LoadAlign(diagonalRow, x1 + row * 32);
        Store32FromFp32(akk + (row + 32) * 64 + 32, diagonalRow);
    }
}

template <typename CompilePolicy>
__simd_vf__ inline void StageV6Vf(
    __ubuf__ bfloat16_t *qHat, __ubuf__ bfloat16_t *kHat,
    __ubuf__ bfloat16_t *v, __ubuf__ float *g, __ubuf__ float *gLast,
    __ubuf__ float *betaEff, __ubuf__ bfloat16_t *qg,
    __ubuf__ bfloat16_t *kg, __ubuf__ bfloat16_t *qgScaled,
    __ubuf__ bfloat16_t *kBetaG, __ubuf__ bfloat16_t *vBeta,
    uint16_t validRows, float scale)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < Shape::kChunkRows; ++row) {
        RegTensor<float> qLow;
        RegTensor<float> qHigh;
        RegTensor<float> kLow;
        RegTensor<float> kHigh;
        RegTensor<float> vLow;
        RegTensor<float> vHigh;
        if (row < validRows) {
            RegTensor<float> gateLow;
            RegTensor<float> gateHigh;
            RegTensor<float> lastLow;
            RegTensor<float> lastHigh;
            RegTensor<float> beta;
            LoadAlign(gateLow, g + row * 128);
            LoadAlign(gateHigh, g + row * 128 + 64);
            LoadAlign(lastLow, gLast);
            LoadAlign(lastHigh, gLast + 64);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(beta, betaEff + row);
            Load128AsFp32(qLow, qHigh, qHat + row * 128);
            Load128AsFp32(kLow, kHigh, kHat + row * 128);
            Load128AsFp32(vLow, vHigh, v + row * 128);

            RegTensor<float> posLow;
            RegTensor<float> posHigh;
            RegTensor<float> kPosLow;
            RegTensor<float> kPosHigh;
            Adds(posLow, gateLow, 0.0F, mask);
            Adds(posHigh, gateHigh, 0.0F, mask);
            constexpr float lower =
                Domain::StoredBound(ExpDomain::kV6LowerBase2);
            constexpr float upper =
                Domain::StoredBound(ExpDomain::kV6UpperBase2);
            ExpPair<CompilePolicy::useExp2>(
                posLow, posHigh, lower, upper);
            Mul(qLow, qLow, posLow, mask);
            Mul(qHigh, qHigh, posHigh, mask);
            Mul(kPosLow, kLow, posLow, mask);
            Mul(kPosHigh, kHigh, posHigh, mask);
            // 第一次舍入先落到最终 kBetaG 物理区，随后立即回读 FP32。
            Store128FromFp32(kBetaG + row * 128, kPosLow, kPosHigh);

            Sub(lastLow, lastLow, gateLow, mask);
            Sub(lastHigh, lastHigh, gateHigh, mask);
            ExpPair<CompilePolicy::useExp2>(
                lastLow, lastHigh, lower, upper);
            Mul(kLow, kLow, lastLow, mask);
            Mul(kHigh, kHigh, lastHigh, mask);
            Store128FromFp32(kg + row * 128, kLow, kHigh);

            // K_beta_g 保留两次 BF16 舍入：先写 Khat*E(G)，再回读乘 beta。
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            Load128AsFp32(kLow, kHigh, kBetaG + row * 128);
            Mul(kLow, kLow, beta, mask);
            Mul(kHigh, kHigh, beta, mask);
            Mul(vLow, vLow, beta, mask);
            Mul(vHigh, vHigh, beta, mask);
        } else {
            Duplicate(qLow, 0.0F, mask);
            Duplicate(qHigh, 0.0F, mask);
            Duplicate(kLow, 0.0F, mask);
            Duplicate(kHigh, 0.0F, mask);
            Duplicate(vLow, 0.0F, mask);
            Duplicate(vHigh, 0.0F, mask);
        }
        Store128FromFp32(qg + row * 128, qLow, qHigh);
        if (row < validRows) {
            // qgScaled 必须从已舍入的公开 qg 回读，保留两次 BF16 舍入。
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            Load128AsFp32(qLow, qHigh, qg + row * 128);
            Muls(qLow, qLow, scale, mask);
            Muls(qHigh, qHigh, scale, mask);
            Store128FromFp32(qgScaled + row * 128, qLow, qHigh);
        }
        Store128FromFp32(kBetaG + row * 128, kLow, kHigh);
        Store128FromFp32(vBeta + row * 128, vLow, vHigh);
    }
}

} // namespace Detail

template <typename GateT, typename BetaT, typename CompilePolicy>
class ChunkKdaFwdPrepareVec {
public:
    __aicore__ inline void Init(const PrepareKernelArgs &args)
    {
        args_ = args;
        workgroup_ = WorkgroupId();
        aiv_ = AscendC::GetSubBlockIdx();
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.v));
        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(args.rawGate));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(args.beta));
        if (args.dtBias != nullptr) {
            dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.dtBias));
        }
        if (args.aLog != nullptr) {
            aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.aLog));
        }
        qgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.qg));
        qgScaledGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(args.qgScaled));
        kgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.kg));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.gk));
        aqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.aqk));
        akkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.akk));
    }

    __aicore__ inline void Process()
    {
        if (args_.tiling.usedCoreNum == 0 ||
            workgroup_ >= args_.tiling.usedCoreNum) {
            return;
        }
        // 每个 AIV 都在自己的本地 flag 空间使用同一组固定编号：
        // localSlot0/1 的 ready=0/1，free=4/5。AIV1 不能写 16/17/20/21。
        constexpr uint16_t kAivToAicPayloadReadyFlagId[2] = {0, 1};
        constexpr uint16_t kAicToAivSlotReusableFlagId[2] = {4, 5};
        bool usedLocalSlot[2] = {false, false};
        const uint32_t total = TotalWorkItems(args_.tiling);
        const uint32_t workBegin = WorkBegin(
            total, workgroup_, args_.tiling.usedCoreNum);
        const uint32_t workEnd = WorkEnd(
            total, workgroup_, args_.tiling.usedCoreNum);
        for (uint32_t work = workBegin; work < workEnd; ++work) {
            uint32_t globalChunk = 0;
            uint32_t headPartition = 0;
            DecodeWorkItem(args_.tiling, work, globalChunk, headPartition);
            ChunkRange chunk{};
            if (!ResolveChunk(args_, globalChunk, chunk)) {
                continue;
            }
            uint32_t headBegin = 0;
            uint32_t headEnd = 0;
            HeadRange(args_.tiling, headPartition, headBegin, headEnd);
            for (uint32_t groupBegin = headBegin; groupBegin < headEnd;
                 groupBegin += Shape::kHeadsPerGroup) {
                for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
                    const uint32_t localHead = aiv_ * 2 + localSlot;
                    const uint32_t valueHead = groupBegin + localHead;
                    if (valueHead >= headEnd) {
                        continue;
                    }
                    usedLocalSlot[localSlot] = true;
                    // 初始 free 或上一组 C7 free；V0 首个消费者是 MTE2。
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                        kAicToAivSlotReusableFlagId[localSlot]);
                    StageV0(chunk, valueHead, localHead, localSlot);
                    StageV1(chunk, localHead, localSlot);
                    // V1 的 72 KiB score payload 已经写入 workspace。
                    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                        kAivToAicPayloadReadyFlagId[localSlot]);
                }
                for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
                    const uint32_t localHead = aiv_ * 2 + localSlot;
                    const uint32_t valueHead = groupBegin + localHead;
                    if (valueHead >= headEnd) {
                        continue;
                    }
                    // C2 已写回 raw Aqk/Akk，且不再读取 V1 payload。
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                        kAicToAivSlotReusableFlagId[localSlot]);
                    StageV3(chunk, valueHead, localHead, localSlot);
                    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                        kAivToAicPayloadReadyFlagId[localSlot]);
                }
                for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
                    const uint32_t localHead = aiv_ * 2 + localSlot;
                    const uint32_t valueHead = groupBegin + localHead;
                    if (valueHead >= headEnd) {
                        continue;
                    }
                    // C4 已一次性读完 B/X0/negX1/Akk，V6 可以原址换义。
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                        kAicToAivSlotReusableFlagId[localSlot]);
                    StageV6(chunk, valueHead, localHead, localSlot);
                    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                        kAivToAicPayloadReadyFlagId[localSlot]);
                }
            }
        }
        // 消费最后一次 C7 free，保证每次 set 都有对应 wait。
        for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
            if (usedLocalSlot[localSlot]) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                    kAicToAivSlotReusableFlagId[localSlot]);
            }
        }
    }

private:
    __aicore__ inline void StageV0(const ChunkRange &chunk,
                                    uint32_t valueHead,
                                    uint32_t localHead,
                                    uint32_t localSlot)
    {
        const uint8_t mutex = static_cast<uint8_t>(localSlot); // slot0=0，slot1=1
        const uint32_t computeSlot = Arch35Ub::kComputeSlotBase[localSlot];
        const uint32_t state = Arch35Ub::kStateBase[localSlot];
        auto q = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kQ);
        auto k = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kK);
        auto gate = resource_.ubBuf.template GetBufferByByte<GateT>(
            computeSlot + Arch35Ub::kGateInput);
        auto g = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kG);
        auto beta = resource_.ubBuf.template GetBufferByByte<BetaT>(
            state + Arch35Ub::kBetaRaw);
        auto betaEff = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kBetaEff);
        auto gRef = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kGRef[0]);
        auto gLast = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kGLast);
        auto dtBias = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kV0Work);
        auto aLog = dtBias[Shape::kHeadDim];

        const uint32_t qkHead = QkHeadForValueHead(args_.tiling, valueHead);
        const uint64_t qkOffset = QkInputOffset(args_.tiling, chunk, qkHead);
        const uint64_t gateOffset =
            RawGateInputOffset(args_.tiling, chunk, valueHead);
        const uint64_t betaOffset =
            HeadScalarOffset(args_.tiling, chunk, valueHead);
        const uint32_t qkStride = args_.tiling.inputSequenceMajor
                                      ? (args_.tiling.qkHeadNum - 1) *
                                            Shape::kHeadDim * sizeof(bfloat16_t)
                                      : 0;
        const uint32_t gateStride = args_.tiling.inputSequenceMajor
                                        ? (args_.tiling.valueHeadNum - 1) *
                                              Shape::kHeadDim * sizeof(GateT)
                                        : 0;

        AscendC::Mutex::Lock<PIPE_MTE2>(mutex);
        AscendC::DataCopyPad(q, qGm_[qkOffset],
            {static_cast<uint16_t>(chunk.validRows),
             Shape::kHeadDim * sizeof(bfloat16_t), qkStride, 0, 0},
            {false, 0, 0, 0});
        AscendC::DataCopyPad(k, kGm_[qkOffset],
            {static_cast<uint16_t>(chunk.validRows),
             Shape::kHeadDim * sizeof(bfloat16_t), qkStride, 0, 0},
            {false, 0, 0, 0});
        AscendC::DataCopyPad(gate, gateGm_[gateOffset],
            {static_cast<uint16_t>(chunk.validRows),
             Shape::kHeadDim * sizeof(GateT), gateStride, 0, 0},
            {false, 0, 0, 0});
        AscendC::DataCopyPad(beta, betaGm_[betaOffset],
            {1, chunk.validRows * sizeof(BetaT), 0, 0, 0},
            {false, 0, 0, 0});
        if constexpr (CompilePolicy::gateMode != GateMode::PrecomputedStep) {
            if (args_.tiling.hasDtBias) {
                AscendC::DataCopy(dtBias,
                    dtBiasGm_[valueHead * Shape::kHeadDim],
                    Shape::kHeadDim);
            }
            if (args_.aLog != nullptr) {
                AscendC::DataCopyExtParams scalarCopy{
                    1, sizeof(float), 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> scalarPad{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(aLog, aLogGm_[valueHead], scalarCopy,
                                     scalarPad);
            }
        }
        AscendC::Mutex::Unlock<PIPE_MTE2>(mutex);

        AscendC::Mutex::Lock<PIPE_V>(mutex);
        AscendC::VF_CALL<Detail::StageV0Vf<GateT, BetaT, CompilePolicy>>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ GateT *>(gate.GetPhyAddr()),
            reinterpret_cast<__ubuf__ BetaT *>(beta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dtBias.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(aLog.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gRef.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gLast.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(betaEff.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows), args_.tiling.epsilon,
            args_.tiling.lowerBound, args_.tiling.hasDtBias,
            args_.aLog != nullptr);
        AscendC::Mutex::Unlock<PIPE_V>(mutex);

        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch35WorkgroupStride);
        AscendC::GlobalTensor<bfloat16_t> qContext;
        AscendC::GlobalTensor<bfloat16_t> kContext;
        qContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kQHat));
        kContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kKHat));
        AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
        AscendC::DataCopy(qContext, q,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kContext, k,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(gkGm_[HeadTensorOffset(
            args_.tiling, chunk, valueHead, Shape::kHeadDim)],
            g, chunk.validRows * Shape::kHeadDim);
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    }

    __aicore__ inline void StageV1(const ChunkRange &chunk,
                                    uint32_t localHead,
                                    uint32_t localSlot)
    {
        const uint8_t mutex = static_cast<uint8_t>(localSlot); // slot0=0，slot1=1
        const uint32_t computeSlot = Arch35Ub::kComputeSlotBase[localSlot];
        const uint32_t state = Arch35Ub::kStateBase[localSlot];
        auto qPlus = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kQ);
        auto kPlus = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kK);
        auto g = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kG);
        auto gRef = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kGRef[0]);
        auto kMinus = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kKMinus);

        AscendC::Mutex::Lock<PIPE_V>(mutex);
        AscendC::VF_CALL<Detail::StageV1Vf<CompilePolicy>>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(qPlus.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kPlus.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gRef.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(qPlus.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kPlus.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kMinus.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::Mutex::Unlock<PIPE_V>(mutex);

        AscendC::GlobalTensor<bfloat16_t> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + WorkspaceSlotBase(
                workgroup_, localHead, Workspace::kArch35WorkgroupStride) +
            Workspace::kPayload));
        AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
        AscendC::DataCopy(payload, qPlus,
            2 * Shape::kChunkRows * Shape::kHeadDim);
        AscendC::DataCopy(payload[2 * Shape::kChunkRows * Shape::kHeadDim],
            kMinus, 40 * 1024 / sizeof(bfloat16_t));
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    }

    __aicore__ inline void StageV3(const ChunkRange &chunk,
                                    uint32_t valueHead,
                                    uint32_t localHead,
                                    uint32_t localSlot)
    {
        const uint8_t mutex = static_cast<uint8_t>(localSlot); // slot0=0，slot1=1
        const uint32_t computeSlot = Arch35Ub::kComputeSlotBase[localSlot];
        const uint32_t state = Arch35Ub::kStateBase[localSlot];
        auto rawScore = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kRawScore);
        auto aqk = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kAqk);
        auto lkk = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kLkk);
        auto b = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kB);
        auto x0 = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kX0);
        auto x1 = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kX1);
        auto negX1 = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kNegX1);
        auto akk = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kAkkPack);
        auto betaEff = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kBetaEff);

        AscendC::Mutex::Lock<PIPE_V>(mutex);
        AscendC::VF_CALL<Detail::StageV3Vf>(
            reinterpret_cast<__ubuf__ float *>(rawScore.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(betaEff.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(aqk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(lkk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(b.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(x0.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(x1.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(negX1.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(akk.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows), args_.tiling.scale);
        AscendC::Mutex::Unlock<PIPE_V>(mutex);

        AscendC::GlobalTensor<float> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + WorkspaceSlotBase(
                workgroup_, localHead, Workspace::kArch35WorkgroupStride) +
            Workspace::kPayload));
        AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
        AscendC::DataCopy(aqkGm_[AOutputOffset(
            args_.tiling, chunk, valueHead)], aqk,
            chunk.validRows * Shape::kChunkRows);
        AscendC::DataCopy(payload[Workspace::kB / sizeof(float)], b, 1024);
        AscendC::DataCopy(payload[Workspace::kX0 / sizeof(float)], x0, 1024);
        AscendC::DataCopy(payload[Workspace::kNegX1 / sizeof(float)],
                          negX1, 1024);
        // C4 固定读取完整 64x64 矩阵，因此补零后的中转矩阵始终写入
        // 工作空间；公开 Akk 只有 T 行，尾 chunk 只能写有效行。
        AscendC::GlobalTensor<bfloat16_t> akkRelay;
        akkRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + WorkspaceSlotBase(
                workgroup_, localHead, Workspace::kArch35WorkgroupStride) +
            Workspace::kPayload + Workspace::kAkk));
        AscendC::DataCopy(akkRelay, akk,
            Shape::kChunkRows * Shape::kChunkRows);
        AscendC::DataCopy(akkGm_[AOutputOffset(
            args_.tiling, chunk, valueHead)], akk,
            chunk.validRows * Shape::kChunkRows);
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    }

    __aicore__ inline void StageV6(const ChunkRange &chunk,
                                    uint32_t valueHead,
                                    uint32_t localHead,
                                    uint32_t localSlot)
    {
        const uint8_t mutex = static_cast<uint8_t>(localSlot); // slot0=0，slot1=1
        const uint32_t computeSlot = Arch35Ub::kComputeSlotBase[localSlot];
        const uint32_t state = Arch35Ub::kStateBase[localSlot];
        auto qg = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kQg);
        auto kg = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kKg);
        auto vBeta = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kVBeta);
        auto g = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kGForPost);
        auto kBetaG = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kKBetaG);
        auto qgScaled = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kQgScaled);
        auto betaEff = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kBetaEff);
        auto gLast = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kGLast);
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch35WorkgroupStride);
        AscendC::GlobalTensor<bfloat16_t> qContext;
        AscendC::GlobalTensor<bfloat16_t> kContext;
        qContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kQHat));
        kContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kKHat));

        AscendC::Mutex::Lock<PIPE_MTE2>(mutex);
        AscendC::DataCopy(qg, qContext,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kg, kContext,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(g,
            gkGm_[HeadTensorOffset(
                args_.tiling, chunk, valueHead, Shape::kHeadDim)],
            chunk.validRows * Shape::kHeadDim);
        const uint64_t vOffset =
            ValueInputOffset(args_.tiling, chunk, valueHead);
        const uint32_t vStride = args_.tiling.inputSequenceMajor
                                     ? (args_.tiling.valueHeadNum - 1) *
                                           Shape::kValueDim * sizeof(bfloat16_t)
                                     : 0;
        AscendC::DataCopyPad(vBeta, vGm_[vOffset],
            {static_cast<uint16_t>(chunk.validRows),
             Shape::kValueDim * sizeof(bfloat16_t), vStride, 0, 0},
            {false, 0, 0, 0});
        AscendC::Mutex::Unlock<PIPE_MTE2>(mutex);

        AscendC::Mutex::Lock<PIPE_V>(mutex);
        AscendC::VF_CALL<Detail::StageV6Vf<CompilePolicy>>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(qg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(vBeta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gLast.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(betaEff.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(qg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(qgScaled.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kBetaG.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(vBeta.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows), args_.tiling.scale);
        AscendC::Mutex::Unlock<PIPE_V>(mutex);

        AscendC::GlobalTensor<bfloat16_t> kBetaRelay;
        AscendC::GlobalTensor<bfloat16_t> vBetaRelay;
        kBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload +
            Workspace::kKBetaG));
        vBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload +
            Workspace::kVBeta));
        const uint32_t rhsRows = chunk.validRows > 32 ? 64 : 32;
        AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
        const uint64_t out = HeadTensorOffset(
            args_.tiling, chunk, valueHead, Shape::kHeadDim);
        AscendC::DataCopy(qgGm_[out], qg,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(qgScaledGm_[out], qgScaled,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kgGm_[out], kg,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kBetaRelay, kBetaG,
            rhsRows * Shape::kHeadDim);
        AscendC::DataCopy(vBetaRelay, vBeta,
            rhsRows * Shape::kValueDim);
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    }

    PrepareKernelArgs args_{};
    uint32_t workgroup_ = 0;
    uint32_t aiv_ = 0;
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource_{};
    AscendC::GlobalTensor<bfloat16_t> qGm_{};
    AscendC::GlobalTensor<bfloat16_t> kGm_{};
    AscendC::GlobalTensor<bfloat16_t> vGm_{};
    AscendC::GlobalTensor<GateT> gateGm_{};
    AscendC::GlobalTensor<BetaT> betaGm_{};
    AscendC::GlobalTensor<float> dtBiasGm_{};
    AscendC::GlobalTensor<float> aLogGm_{};
    AscendC::GlobalTensor<bfloat16_t> qgGm_{};
    AscendC::GlobalTensor<bfloat16_t> qgScaledGm_{};
    AscendC::GlobalTensor<bfloat16_t> kgGm_{};
    AscendC::GlobalTensor<float> gkGm_{};
    AscendC::GlobalTensor<bfloat16_t> aqkGm_{};
    AscendC::GlobalTensor<bfloat16_t> akkGm_{};
};

} // namespace KdaPrepare::Arch35

#endif // PSEUDOCODE_ARCH35_CHUNK_KDA_FWD_PREPARE_VEC_H
