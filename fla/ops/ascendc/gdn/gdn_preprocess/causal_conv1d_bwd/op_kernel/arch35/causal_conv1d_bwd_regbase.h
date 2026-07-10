/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ASCENDC_CAUSAL_CONV1D_BWD_REGBASE_H_
#define ASCENDC_CAUSAL_CONV1D_BWD_REGBASE_H_

#include "kernel_utils/vector/regbase.hpp"

namespace NsCausalConv1dBwd {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t BWD_DIRECT_BD = 64;
constexpr uint32_t BWD_DIRECT_W_MAX = 4;

constexpr CastTrait B16_TO_F32_ZERO = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

template <typename T>
__simd_callee__ inline void LoadAsFloatPair(
    RegTensor<float> &zero, RegTensor<float> &one, __ubuf__ T *src, MaskReg &maskB16)
{
    if constexpr (std::is_same<T, float>()) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, reinterpret_cast<__ubuf__ float *>(src));
    } else if constexpr (std::is_same<T, half>()) {
        RegTensor<half> raw;
        LoadIn<T, false>(raw, src);
        CastHalf2Float<half>(zero, one, raw, maskB16);
    } else {
        static_assert(!std::is_same<T, T>::value, "LoadAsFloatPair supports float/half only");
    }
}

__simd_callee__ inline void LoadFloatPair(
    RegTensor<float> &zero, RegTensor<float> &one, __ubuf__ float *src)
{
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, src);
}

__simd_callee__ inline void StoreFloatPair(
    __ubuf__ float *dst, RegTensor<float> &zero, RegTensor<float> &one, MaskReg &maskF32)
{
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, zero, one, maskF32);
}

__simd_callee__ inline void AddPair(
    RegTensor<float> &dstZero, RegTensor<float> &dstOne,
    RegTensor<float> &srcZero, RegTensor<float> &srcOne, MaskReg &maskF32)
{
    Add(dstZero, dstZero, srcZero, maskF32);
    Add(dstOne, dstOne, srcOne, maskF32);
}

__simd_callee__ inline void MulPair(
    RegTensor<float> &dstZero, RegTensor<float> &dstOne,
    RegTensor<float> &lhsZero, RegTensor<float> &lhsOne,
    RegTensor<float> &rhsZero, RegTensor<float> &rhsOne, MaskReg &maskF32)
{
    Mul(dstZero, lhsZero, rhsZero, maskF32);
    Mul(dstOne, lhsOne, rhsOne, maskF32);
}

__simd_callee__ inline void ApplySiluBackwardOne(
    RegTensor<float> &dy, RegTensor<float> &y, MaskReg &maskF32)
{
    RegTensor<float> sigmoid;
    RegTensor<float> one;
    Muls(sigmoid, y, -1.0f, maskF32);
    Exp(sigmoid, sigmoid, maskF32);
    Adds(sigmoid, sigmoid, 1.0f, maskF32);
    Duplicate(one, 1.0f, maskF32);
    Div(sigmoid, one, sigmoid, maskF32);

    Sub(one, one, sigmoid, maskF32);
    Mul(one, one, y, maskF32);
    Adds(one, one, 1.0f, maskF32);
    Mul(dy, dy, sigmoid, maskF32);
    Mul(dy, dy, one, maskF32);
}

__simd_callee__ inline void ApplySiluBackwardPair(
    RegTensor<float> &dyZero, RegTensor<float> &dyOne,
    RegTensor<float> &yZero, RegTensor<float> &yOne, MaskReg &maskF32)
{
    ApplySiluBackwardOne(dyZero, yZero, maskF32);
    ApplySiluBackwardOne(dyOne, yOne, maskF32);
}

__simd_callee__ inline void StoreDwSum(
    __ubuf__ float *dwAddr, RegTensor<float> &sumZero, RegTensor<float> &sumOne,
    uint32_t wIdx, MaskReg &maskF32)
{
    RegTensor<float> oldZero;
    RegTensor<float> oldOne;
    LoadFloatPair(oldZero, oldOne, dwAddr + wIdx * BWD_DIRECT_BD);
    AddPair(oldZero, oldOne, sumZero, sumOne, maskF32);
    StoreFloatPair(dwAddr + wIdx * BWD_DIRECT_BD, oldZero, oldOne, maskF32);
}

template <typename T>
__simd_callee__ inline void LoadB16AsFloat(
    RegTensor<float> &dst, __ubuf__ T *src, MaskReg &maskF32)
{
    if constexpr (std::is_same<T, half>() || std::is_same<T, bfloat16_t>()) {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        Cast<float, T, B16_TO_F32_ZERO>(dst, raw, maskF32);
    } else {
        static_assert(!std::is_same<T, T>::value, "LoadB16AsFloat supports half/bfloat16_t only");
    }
}

__simd_callee__ inline void LoadFloatOne(RegTensor<float> &dst, __ubuf__ float *src)
{
    DataCopy<float, LoadDist::DIST_NORM>(dst, src);
}

__simd_callee__ inline void StoreFloatOne(
    __ubuf__ float *dst, RegTensor<float> &src, MaskReg &maskF32)
{
    DataCopy<float, StoreDist::DIST_NORM_B32>(dst, src, maskF32);
}

__simd_callee__ inline void StoreDwSumOne(
    __ubuf__ float *dwAddr, RegTensor<float> &sum, uint32_t wIdx, MaskReg &maskF32)
{
    RegTensor<float> old;
    LoadFloatOne(old, dwAddr + wIdx * BWD_DIRECT_BD);
    Add(old, old, sum, maskF32);
    StoreFloatOne(dwAddr + wIdx * BWD_DIRECT_BD, old, maskF32);
}

template <typename T>
static __simd_vf__ inline void ComputeActivatedDyB16Vf(
    __ubuf__ T *dyAddr, __ubuf__ T *yAddr, __ubuf__ float *dyActAddr, uint32_t rows)
{
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> dy;
    RegTensor<float> y;

    for (uint32_t row = 0; row < rows; row++) {
        LoadB16AsFloat<T>(dy, dyAddr + row * BWD_DIRECT_BD, maskF32);
        LoadB16AsFloat<T>(y, yAddr + row * BWD_DIRECT_BD, maskF32);
        ApplySiluBackwardOne(dy, y, maskF32);
        StoreFloatOne(dyActAddr + row * BWD_DIRECT_BD, dy, maskF32);
    }
}

template <typename T>
static __simd_vf__ inline void ComputeDxActivatedB16Vf(
    __ubuf__ float *dyActAddr, __ubuf__ T *weightAddr, __ubuf__ float *dxAddr,
    uint32_t rows, uint32_t width)
{
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> dy;
    RegTensor<float> weight;
    RegTensor<float> dx;
    RegTensor<float> prod;

    for (uint32_t row = 0; row < rows; row++) {
        Duplicate(dx, 0.0f, maskF32);
        for (uint32_t iW = 0; iW < width; iW++) {
            uint32_t wIdx = width - iW - 1;
            LoadFloatOne(dy, dyActAddr + (row + iW) * BWD_DIRECT_BD);
            LoadB16AsFloat<T>(weight, weightAddr + wIdx * BWD_DIRECT_BD, maskF32);
            Mul(prod, dy, weight, maskF32);
            Add(dx, dx, prod, maskF32);
        }
        StoreFloatOne(dxAddr + row * BWD_DIRECT_BD, dx, maskF32);
    }
}

template <typename T>
static __simd_vf__ inline void ComputeDwOneActivatedB16Vf(
    __ubuf__ T *xAddr, __ubuf__ float *dyActAddr, __ubuf__ float *dwAddr,
    uint32_t rows, uint32_t width, uint32_t wIdx)
{
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> sum;
    RegTensor<float> x;
    RegTensor<float> dy;
    RegTensor<float> prod;

    Duplicate(sum, 0.0f, maskF32);

    uint32_t iW = width - wIdx - 1U;
    for (uint32_t row = 0; row < rows; row++) {
        LoadB16AsFloat<T>(x, xAddr + row * BWD_DIRECT_BD, maskF32);
        LoadFloatOne(dy, dyActAddr + (row + iW) * BWD_DIRECT_BD);
        Mul(prod, dy, x, maskF32);
        Add(sum, sum, prod, maskF32);
    }

    StoreDwSumOne(dwAddr, sum, wIdx, maskF32);
}

static __simd_vf__ inline void ComputeDbActivatedB16Vf(
    __ubuf__ float *dyActAddr, __ubuf__ float *dbAddr, uint32_t rows)
{
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> sum;
    RegTensor<float> dy;
    RegTensor<float> old;

    Duplicate(sum, 0.0f, maskF32);
    for (uint32_t row = 0; row < rows; row++) {
        LoadFloatOne(dy, dyActAddr + row * BWD_DIRECT_BD);
        Add(sum, sum, dy, maskF32);
    }

    LoadFloatOne(old, dbAddr);
    Add(old, old, sum, maskF32);
    StoreFloatOne(dbAddr, old, maskF32);
}

template <typename T>
static __simd_vf__ inline void ComputeDxDirectVf(
    __ubuf__ T *dyAddr, __ubuf__ T *weightAddr, __ubuf__ float *dxAddr,
    uint32_t rows, uint32_t width)
{
    MaskReg maskB16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> dyZero;
    RegTensor<float> dyOne;
    RegTensor<float> weightZero;
    RegTensor<float> weightOne;
    RegTensor<float> dxZero;
    RegTensor<float> dxOne;
    RegTensor<float> prodZero;
    RegTensor<float> prodOne;

    for (uint32_t row = 0; row < rows; row++) {
        Duplicate(dxZero, 0.0f, maskF32);
        Duplicate(dxOne, 0.0f, maskF32);
        for (uint32_t iW = 0; iW < width; iW++) {
            uint32_t wIdx = width - iW - 1;
            LoadAsFloatPair<T>(dyZero, dyOne, dyAddr + (row + iW) * BWD_DIRECT_BD, maskB16);
            LoadAsFloatPair<T>(weightZero, weightOne, weightAddr + wIdx * BWD_DIRECT_BD, maskB16);
            MulPair(prodZero, prodOne, dyZero, dyOne, weightZero, weightOne, maskF32);
            AddPair(dxZero, dxOne, prodZero, prodOne, maskF32);
        }
        StoreFloatPair(dxAddr + row * BWD_DIRECT_BD, dxZero, dxOne, maskF32);
    }
}

template <typename T>
static __simd_vf__ inline void ComputeActivatedDyVf(
    __ubuf__ T *dyAddr, __ubuf__ T *yAddr, __ubuf__ float *dyActAddr, uint32_t rows)
{
    MaskReg maskB16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> dyZero;
    RegTensor<float> dyOne;
    RegTensor<float> yZero;
    RegTensor<float> yOne;

    for (uint32_t row = 0; row < rows; row++) {
        LoadAsFloatPair<T>(dyZero, dyOne, dyAddr + row * BWD_DIRECT_BD, maskB16);
        LoadAsFloatPair<T>(yZero, yOne, yAddr + row * BWD_DIRECT_BD, maskB16);
        ApplySiluBackwardPair(dyZero, dyOne, yZero, yOne, maskF32);
        StoreFloatPair(dyActAddr + row * BWD_DIRECT_BD, dyZero, dyOne, maskF32);
    }
}

template <typename T>
static __simd_vf__ inline void ComputeDxActivatedVf(
    __ubuf__ float *dyActAddr, __ubuf__ T *weightAddr, __ubuf__ float *dxAddr,
    uint32_t rows, uint32_t width)
{
    MaskReg maskB16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> dyZero;
    RegTensor<float> dyOne;
    RegTensor<float> weightZero;
    RegTensor<float> weightOne;
    RegTensor<float> dxZero;
    RegTensor<float> dxOne;
    RegTensor<float> prodZero;
    RegTensor<float> prodOne;

    for (uint32_t row = 0; row < rows; row++) {
        Duplicate(dxZero, 0.0f, maskF32);
        Duplicate(dxOne, 0.0f, maskF32);
        for (uint32_t iW = 0; iW < width; iW++) {
            uint32_t wIdx = width - iW - 1;
            LoadFloatPair(dyZero, dyOne, dyActAddr + (row + iW) * BWD_DIRECT_BD);
            LoadAsFloatPair<T>(weightZero, weightOne, weightAddr + wIdx * BWD_DIRECT_BD, maskB16);
            MulPair(prodZero, prodOne, dyZero, dyOne, weightZero, weightOne, maskF32);
            AddPair(dxZero, dxOne, prodZero, prodOne, maskF32);
        }
        StoreFloatPair(dxAddr + row * BWD_DIRECT_BD, dxZero, dxOne, maskF32);
    }
}

template <typename T>
static __simd_vf__ inline void ComputeDwOneDirectVf(
    __ubuf__ T *xAddr, __ubuf__ T *dyAddr, __ubuf__ float *dwAddr,
    uint32_t rows, uint32_t width, uint32_t wIdx)
{
    MaskReg maskB16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> sumZero;
    RegTensor<float> sumOne;
    RegTensor<float> xZero;
    RegTensor<float> xOne;
    RegTensor<float> dyZero;
    RegTensor<float> dyOne;
    RegTensor<float> prodZero;
    RegTensor<float> prodOne;

    Duplicate(sumZero, 0.0f, maskF32);
    Duplicate(sumOne, 0.0f, maskF32);

    uint32_t iW = width - wIdx - 1U;
    for (uint32_t row = 0; row < rows; row++) {
        LoadAsFloatPair<T>(xZero, xOne, xAddr + row * BWD_DIRECT_BD, maskB16);
        LoadAsFloatPair<T>(dyZero, dyOne, dyAddr + (row + iW) * BWD_DIRECT_BD, maskB16);
        MulPair(prodZero, prodOne, dyZero, dyOne, xZero, xOne, maskF32);
        AddPair(sumZero, sumOne, prodZero, prodOne, maskF32);
    }

    StoreDwSum(dwAddr, sumZero, sumOne, wIdx, maskF32);
}

template <typename T>
static __simd_vf__ inline void ComputeDwOneActivatedVf(
    __ubuf__ T *xAddr, __ubuf__ float *dyActAddr, __ubuf__ float *dwAddr,
    uint32_t rows, uint32_t width, uint32_t wIdx)
{
    MaskReg maskB16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> sumZero;
    RegTensor<float> sumOne;
    RegTensor<float> xZero;
    RegTensor<float> xOne;
    RegTensor<float> dyZero;
    RegTensor<float> dyOne;
    RegTensor<float> prodZero;
    RegTensor<float> prodOne;

    Duplicate(sumZero, 0.0f, maskF32);
    Duplicate(sumOne, 0.0f, maskF32);

    uint32_t iW = width - wIdx - 1U;
    for (uint32_t row = 0; row < rows; row++) {
        LoadAsFloatPair<T>(xZero, xOne, xAddr + row * BWD_DIRECT_BD, maskB16);
        LoadFloatPair(dyZero, dyOne, dyActAddr + (row + iW) * BWD_DIRECT_BD);
        MulPair(prodZero, prodOne, dyZero, dyOne, xZero, xOne, maskF32);
        AddPair(sumZero, sumOne, prodZero, prodOne, maskF32);
    }

    StoreDwSum(dwAddr, sumZero, sumOne, wIdx, maskF32);
}

template <typename T>
static __simd_vf__ inline void ComputeDbDirectVf(
    __ubuf__ T *dyAddr, __ubuf__ float *dbAddr, uint32_t rows)
{
    MaskReg maskB16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> sumZero;
    RegTensor<float> sumOne;
    RegTensor<float> dyZero;
    RegTensor<float> dyOne;
    RegTensor<float> oldZero;
    RegTensor<float> oldOne;

    Duplicate(sumZero, 0.0f, maskF32);
    Duplicate(sumOne, 0.0f, maskF32);
    for (uint32_t row = 0; row < rows; row++) {
        LoadAsFloatPair<T>(dyZero, dyOne, dyAddr + row * BWD_DIRECT_BD, maskB16);
        AddPair(sumZero, sumOne, dyZero, dyOne, maskF32);
    }

    LoadFloatPair(oldZero, oldOne, dbAddr);
    AddPair(oldZero, oldOne, sumZero, sumOne, maskF32);
    StoreFloatPair(dbAddr, oldZero, oldOne, maskF32);
}

static __simd_vf__ inline void ComputeDbActivatedVf(
    __ubuf__ float *dyActAddr, __ubuf__ float *dbAddr, uint32_t rows)
{
    MaskReg maskF32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> sumZero;
    RegTensor<float> sumOne;
    RegTensor<float> dyZero;
    RegTensor<float> dyOne;
    RegTensor<float> oldZero;
    RegTensor<float> oldOne;

    Duplicate(sumZero, 0.0f, maskF32);
    Duplicate(sumOne, 0.0f, maskF32);
    for (uint32_t row = 0; row < rows; row++) {
        LoadFloatPair(dyZero, dyOne, dyActAddr + row * BWD_DIRECT_BD);
        AddPair(sumZero, sumOne, dyZero, dyOne, maskF32);
    }

    LoadFloatPair(oldZero, oldOne, dbAddr);
    AddPair(oldZero, oldOne, sumZero, sumOne, maskF32);
    StoreFloatPair(dbAddr, oldZero, oldOne, maskF32);
}

template <typename T>
__aicore__ inline void ComputeTileDirectRegbase(
    LocalTensor<T> xRaw, LocalTensor<T> dyRaw, LocalTensor<T> yRaw, LocalTensor<T> weightRaw,
    LocalTensor<float> stateLocal, LocalTensor<float> dxLocal, LocalTensor<float> dwLocal,
    LocalTensor<float> dbLocal, LocalTensor<float> dh0Local,
    uint32_t rows, uint32_t width, uint32_t dyRows, uint32_t activation,
    uint32_t validRows, uint32_t useInitialState, uint32_t computeDh0, uint32_t hasBias)
{
    auto xAddr = reinterpret_cast<__ubuf__ T *>(xRaw.GetPhyAddr());
    auto dyAddr = reinterpret_cast<__ubuf__ T *>(dyRaw.GetPhyAddr());
    auto yAddr = reinterpret_cast<__ubuf__ T *>(yRaw.GetPhyAddr());
    auto weightAddr = reinterpret_cast<__ubuf__ T *>(weightRaw.GetPhyAddr());
    auto dyActAddr = reinterpret_cast<__ubuf__ float *>(stateLocal.GetPhyAddr());
    auto dxAddr = reinterpret_cast<__ubuf__ float *>(dxLocal.GetPhyAddr());
    auto dwAddr = reinterpret_cast<__ubuf__ float *>(dwLocal.GetPhyAddr());
    auto dbAddr = reinterpret_cast<__ubuf__ float *>(dbLocal.GetPhyAddr());
    (void)dh0Local;
    (void)validRows;
    (void)useInitialState;
    (void)computeDh0;
    if constexpr (std::is_same<T, half>() || std::is_same<T, bfloat16_t>()) {
        if (activation == 1U) {
            AscendC::VF_CALL<ComputeActivatedDyB16Vf<T>>(dyAddr, yAddr, dyActAddr, dyRows);
            AscendC::VF_CALL<ComputeDxActivatedB16Vf<T>>(dyActAddr, weightAddr, dxAddr, rows, width);
            if (width > 0U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 0U);
            }
            if (width > 1U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 1U);
            }
            if (width > 2U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 2U);
            }
            if (width > 3U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 3U);
            }
            if (hasBias != 0U) {
                AscendC::VF_CALL<ComputeDbActivatedB16Vf>(dyActAddr, dbAddr, rows);
            }
            return;
        }
        return;
    } else {
        if (activation == 1U) {
            AscendC::VF_CALL<ComputeActivatedDyVf<T>>(dyAddr, yAddr, dyActAddr, dyRows);
            AscendC::VF_CALL<ComputeDxActivatedVf<T>>(dyActAddr, weightAddr, dxAddr, rows, width);
            if (width > 0U) {
                AscendC::VF_CALL<ComputeDwOneActivatedVf<T>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 0U);
            }
            if (width > 1U) {
                AscendC::VF_CALL<ComputeDwOneActivatedVf<T>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 1U);
            }
            if (width > 2U) {
                AscendC::VF_CALL<ComputeDwOneActivatedVf<T>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 2U);
            }
            if (width > 3U) {
                AscendC::VF_CALL<ComputeDwOneActivatedVf<T>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 3U);
            }
            if (hasBias != 0U) {
                AscendC::VF_CALL<ComputeDbActivatedVf>(dyActAddr, dbAddr, rows);
            }
            return;
        }

        AscendC::VF_CALL<ComputeDxDirectVf<T>>(dyAddr, weightAddr, dxAddr, rows, width);
        if (width > 0U) {
            AscendC::VF_CALL<ComputeDwOneDirectVf<T>>(
                xAddr, dyAddr, dwAddr, rows, width, 0U);
        }
        if (width > 1U) {
            AscendC::VF_CALL<ComputeDwOneDirectVf<T>>(
                xAddr, dyAddr, dwAddr, rows, width, 1U);
        }
        if (width > 2U) {
            AscendC::VF_CALL<ComputeDwOneDirectVf<T>>(
                xAddr, dyAddr, dwAddr, rows, width, 2U);
        }
        if (width > 3U) {
            AscendC::VF_CALL<ComputeDwOneDirectVf<T>>(
                xAddr, dyAddr, dwAddr, rows, width, 3U);
        }
        if (hasBias != 0U) {
            AscendC::VF_CALL<ComputeDbDirectVf<T>>(dyAddr, dbAddr, rows);
        }
    }
}

} // namespace NsCausalConv1dBwd

#endif // ASCENDC_CAUSAL_CONV1D_BWD_REGBASE_H_
