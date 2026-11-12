/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ASCENDC_CAUSAL_CONV1D_BWD_ARCH92_REGBASE_H_
#define ASCENDC_CAUSAL_CONV1D_BWD_ARCH92_REGBASE_H_

#include "kernel_utils/vector/regbase.hpp"

namespace NsCausalConv1dBwd {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t BWD_DIRECT_BD = 64;
// 快路径支持的最大卷积宽度：dw 侧按 wIdx 展开成独立的 VF 调用（每个 wIdx 一趟行遍历），
// 支持到 8 无需额外寄存器（每个 VF 仍只用 4 个数据寄存器），UB 仅随 W*BD 线性增长。
constexpr uint32_t BWD_DIRECT_W_MAX = 8;

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
    // 等价式 silu'(y) = (d + y*e) / d^2，e=exp(-y)、d=1+e，仅 8 条向量指令
    //（原 10 条：省去 Duplicate 与一次 Mul）。仅重排运算顺序，数值语义不变。
    RegTensor<float> e;
    RegTensor<float> d;
    Muls(e, y, -1.0f, maskF32);
    Exp(e, e, maskF32);
    Adds(d, e, 1.0f, maskF32);
    Mul(e, e, y, maskF32);
    Add(e, d, e, maskF32);
    Mul(d, d, d, maskF32);
    Div(d, e, d, maskF32);
    Mul(dy, dy, d, maskF32);
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

// 快路径的行宽 BD 既可以是 64（一个 fp32 向量寄存器的全宽），也可以是 16（16 车道掩码）。
// 除 BD=64 的原有实例外，全用 16 车道掩码处理 BD=16；BD=64 模板实参与改动前完全一致。
template <uint32_t BD>
__simd_callee__ inline MaskReg MaskF32ForBd()
{
    if constexpr (BD == BWD_DIRECT_BD) {
        return CreateMask<float, MaskPattern::ALL>();
    } else {
        uint32_t bdCnt = BD;
        return UpdateMask<float>(bdCnt);
    }
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

template <uint32_t BD = BWD_DIRECT_BD>
__simd_callee__ inline void StoreDwSumOne(
    __ubuf__ float *dwAddr, RegTensor<float> &sum, uint32_t wIdx, MaskReg &maskF32)
{
    RegTensor<float> old;
    LoadFloatOne(old, dwAddr + wIdx * BD);
    Add(old, old, sum, maskF32);
    StoreFloatOne(dwAddr + wIdx * BD, old, maskF32);
}

template <typename T, uint32_t BD = BWD_DIRECT_BD>
static __simd_vf__ inline void ComputeActivatedDyB16Vf(
    __ubuf__ T *dyAddr, __ubuf__ T *yAddr, __ubuf__ float *dyActAddr, uint32_t rows)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> dy;
    RegTensor<float> y;

    for (uint32_t row = 0; row < rows; row++) {
        LoadB16AsFloat<T>(dy, dyAddr + row * BD, maskF32);
        LoadB16AsFloat<T>(y, yAddr + row * BD, maskF32);
        ApplySiluBackwardOne(dy, y, maskF32);
        StoreFloatOne(dyActAddr + row * BD, dy, maskF32);
    }
}

template <typename T, uint32_t BD = BWD_DIRECT_BD>
static __simd_vf__ inline void ComputeDxActivatedB16Vf(
    __ubuf__ float *dyActAddr, __ubuf__ T *weightAddr, __ubuf__ float *dxAddr,
    uint32_t rows, uint32_t width)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> dy;
    RegTensor<float> weight;
    RegTensor<float> dx;
    RegTensor<float> prod;

    for (uint32_t row = 0; row < rows; row++) {
        Duplicate(dx, 0.0f, maskF32);
        for (uint32_t iW = 0; iW < width; iW++) {
            uint32_t wIdx = width - iW - 1;
            LoadFloatOne(dy, dyActAddr + (row + iW) * BD);
            LoadB16AsFloat<T>(weight, weightAddr + wIdx * BD, maskF32);
            Mul(prod, dy, weight, maskF32);
            Add(dx, dx, prod, maskF32);
        }
        StoreFloatOne(dxAddr + row * BD, dx, maskF32);
    }
}

template <typename T, uint32_t BD = BWD_DIRECT_BD>
static __simd_vf__ inline void ComputeDwOneActivatedB16Vf(
    __ubuf__ T *xAddr, __ubuf__ float *dyActAddr, __ubuf__ float *dwAddr,
    uint32_t rows, uint32_t width, uint32_t wIdx)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> sum;
    RegTensor<float> x;
    RegTensor<float> dy;
    RegTensor<float> prod;

    Duplicate(sum, 0.0f, maskF32);

    uint32_t iW = width - wIdx - 1U;
    for (uint32_t row = 0; row < rows; row++) {
        LoadB16AsFloat<T>(x, xAddr + row * BD, maskF32);
        LoadFloatOne(dy, dyActAddr + (row + iW) * BD);
        Mul(prod, dy, x, maskF32);
        Add(sum, sum, prod, maskF32);
    }

    StoreDwSumOne<BD>(dwAddr, sum, wIdx, maskF32);
}

template <uint32_t BD = BWD_DIRECT_BD>
static __simd_vf__ inline void ComputeDbActivatedB16Vf(
    __ubuf__ float *dyActAddr, __ubuf__ float *dbAddr, uint32_t rows)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> sum;
    RegTensor<float> dy;
    RegTensor<float> old;

    Duplicate(sum, 0.0f, maskF32);
    for (uint32_t row = 0; row < rows; row++) {
        LoadFloatOne(dy, dyActAddr + row * BD);
        Add(sum, sum, dy, maskF32);
    }

    LoadFloatOne(old, dbAddr);
    Add(old, old, sum, maskF32);
    StoreFloatOne(dbAddr, old, maskF32);
}

// activation == NONE 时的 b16 直接路径：与 Activated 版本同构，仅把 dyAct(float) 换成 dy(b16) 直读。
// 保持 4 个活跃寄存器（dy/weight/dx/prod），与控制流结构一致，避免寄存器压力上升导致标量化退化。
template <typename T, uint32_t BD = BWD_DIRECT_BD>
static __simd_vf__ inline void ComputeDxDirectB16Vf(
    __ubuf__ T *dyAddr, __ubuf__ T *weightAddr, __ubuf__ float *dxAddr,
    uint32_t rows, uint32_t width)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> dy;
    RegTensor<float> weight;
    RegTensor<float> dx;
    RegTensor<float> prod;

    for (uint32_t row = 0; row < rows; row++) {
        Duplicate(dx, 0.0f, maskF32);
        for (uint32_t iW = 0; iW < width; iW++) {
            uint32_t wIdx = width - iW - 1;
            LoadB16AsFloat<T>(dy, dyAddr + (row + iW) * BD, maskF32);
            LoadB16AsFloat<T>(weight, weightAddr + wIdx * BD, maskF32);
            Mul(prod, dy, weight, maskF32);
            Add(dx, dx, prod, maskF32);
        }
        StoreFloatOne(dxAddr + row * BD, dx, maskF32);
    }
}

template <typename T, uint32_t BD = BWD_DIRECT_BD>
static __simd_vf__ inline void ComputeDwOneDirectB16Vf(
    __ubuf__ T *xAddr, __ubuf__ T *dyAddr, __ubuf__ float *dwAddr,
    uint32_t rows, uint32_t width, uint32_t wIdx)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> sum;
    RegTensor<float> x;
    RegTensor<float> dy;
    RegTensor<float> prod;

    Duplicate(sum, 0.0f, maskF32);

    uint32_t iW = width - wIdx - 1U;
    for (uint32_t row = 0; row < rows; row++) {
        LoadB16AsFloat<T>(x, xAddr + row * BD, maskF32);
        LoadB16AsFloat<T>(dy, dyAddr + (row + iW) * BD, maskF32);
        Mul(prod, dy, x, maskF32);
        Add(sum, sum, prod, maskF32);
    }

    StoreDwSumOne<BD>(dwAddr, sum, wIdx, maskF32);
}

template <typename T, uint32_t BD = BWD_DIRECT_BD>
static __simd_vf__ inline void ComputeDbDirectB16Vf(
    __ubuf__ T *dyAddr, __ubuf__ float *dbAddr, uint32_t rows)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> sum;
    RegTensor<float> dy;
    RegTensor<float> old;

    Duplicate(sum, 0.0f, maskF32);
    for (uint32_t row = 0; row < rows; row++) {
        LoadB16AsFloat<T>(dy, dyAddr + row * BD, maskF32);
        Add(sum, sum, dy, maskF32);
    }

    LoadFloatOne(old, dbAddr);
    Add(old, old, sum, maskF32);
    StoreFloatOne(dbAddr, old, maskF32);
}

// fp32 行宽 BD 模板化 VF：BD=64 时行步长恰为一个 fp32 寄存器全宽；BD=16 时用 16 车道掩码，
// 加载沿用无掩码 DIST_NORM（一次读满 64 元素，低 BD 车道即目标行），与 b16 BD=16 路径同构，仅省去 Cast。
template <uint32_t BD>
static __simd_vf__ inline void ComputeActivatedDyF32Vf(
    __ubuf__ float *dyAddr, __ubuf__ float *yAddr, __ubuf__ float *dyActAddr, uint32_t rows)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> dy;
    RegTensor<float> y;

    for (uint32_t row = 0; row < rows; row++) {
        LoadFloatOne(dy, dyAddr + row * BD);
        LoadFloatOne(y, yAddr + row * BD);
        ApplySiluBackwardOne(dy, y, maskF32);
        StoreFloatOne(dyActAddr + row * BD, dy, maskF32);
    }
}

template <uint32_t BD>
static __simd_vf__ inline void ComputeDxActivatedF32Vf(
    __ubuf__ float *dyActAddr, __ubuf__ float *weightAddr, __ubuf__ float *dxAddr,
    uint32_t rows, uint32_t width)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> dy;
    RegTensor<float> weight;
    RegTensor<float> dx;
    RegTensor<float> prod;

    for (uint32_t row = 0; row < rows; row++) {
        Duplicate(dx, 0.0f, maskF32);
        for (uint32_t iW = 0; iW < width; iW++) {
            uint32_t wIdx = width - iW - 1;
            LoadFloatOne(dy, dyActAddr + (row + iW) * BD);
            LoadFloatOne(weight, weightAddr + wIdx * BD);
            Mul(prod, dy, weight, maskF32);
            Add(dx, dx, prod, maskF32);
        }
        StoreFloatOne(dxAddr + row * BD, dx, maskF32);
    }
}

template <uint32_t BD>
static __simd_vf__ inline void ComputeDwOneActivatedF32Vf(
    __ubuf__ float *xAddr, __ubuf__ float *dyActAddr, __ubuf__ float *dwAddr,
    uint32_t rows, uint32_t width, uint32_t wIdx)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> sum;
    RegTensor<float> x;
    RegTensor<float> dy;
    RegTensor<float> prod;

    Duplicate(sum, 0.0f, maskF32);

    uint32_t iW = width - wIdx - 1U;
    for (uint32_t row = 0; row < rows; row++) {
        LoadFloatOne(x, xAddr + row * BD);
        LoadFloatOne(dy, dyActAddr + (row + iW) * BD);
        Mul(prod, dy, x, maskF32);
        Add(sum, sum, prod, maskF32);
    }

    StoreDwSumOne<BD>(dwAddr, sum, wIdx, maskF32);
}

template <uint32_t BD>
static __simd_vf__ inline void ComputeDbActivatedF32Vf(
    __ubuf__ float *dyActAddr, __ubuf__ float *dbAddr, uint32_t rows)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> sum;
    RegTensor<float> dy;
    RegTensor<float> old;

    Duplicate(sum, 0.0f, maskF32);
    for (uint32_t row = 0; row < rows; row++) {
        LoadFloatOne(dy, dyActAddr + row * BD);
        Add(sum, sum, dy, maskF32);
    }

    LoadFloatOne(old, dbAddr);
    Add(old, old, sum, maskF32);
    StoreFloatOne(dbAddr, old, maskF32);
}

// activation == NONE 时的 fp32 直接路径：与 Activated 版本同构，仅把 dyAct 换成 dy 直读。
template <uint32_t BD>
static __simd_vf__ inline void ComputeDxDirectF32Vf(
    __ubuf__ float *dyAddr, __ubuf__ float *weightAddr, __ubuf__ float *dxAddr,
    uint32_t rows, uint32_t width)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> dy;
    RegTensor<float> weight;
    RegTensor<float> dx;
    RegTensor<float> prod;

    for (uint32_t row = 0; row < rows; row++) {
        Duplicate(dx, 0.0f, maskF32);
        for (uint32_t iW = 0; iW < width; iW++) {
            uint32_t wIdx = width - iW - 1;
            LoadFloatOne(dy, dyAddr + (row + iW) * BD);
            LoadFloatOne(weight, weightAddr + wIdx * BD);
            Mul(prod, dy, weight, maskF32);
            Add(dx, dx, prod, maskF32);
        }
        StoreFloatOne(dxAddr + row * BD, dx, maskF32);
    }
}

template <uint32_t BD>
static __simd_vf__ inline void ComputeDwOneDirectF32Vf(
    __ubuf__ float *xAddr, __ubuf__ float *dyAddr, __ubuf__ float *dwAddr,
    uint32_t rows, uint32_t width, uint32_t wIdx)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> sum;
    RegTensor<float> x;
    RegTensor<float> dy;
    RegTensor<float> prod;

    Duplicate(sum, 0.0f, maskF32);

    uint32_t iW = width - wIdx - 1U;
    for (uint32_t row = 0; row < rows; row++) {
        LoadFloatOne(x, xAddr + row * BD);
        LoadFloatOne(dy, dyAddr + (row + iW) * BD);
        Mul(prod, dy, x, maskF32);
        Add(sum, sum, prod, maskF32);
    }

    StoreDwSumOne<BD>(dwAddr, sum, wIdx, maskF32);
}

template <uint32_t BD>
static __simd_vf__ inline void ComputeDbDirectF32Vf(
    __ubuf__ float *dyAddr, __ubuf__ float *dbAddr, uint32_t rows)
{
    MaskReg maskF32 = MaskF32ForBd<BD>();

    RegTensor<float> sum;
    RegTensor<float> dy;
    RegTensor<float> old;

    Duplicate(sum, 0.0f, maskF32);
    for (uint32_t row = 0; row < rows; row++) {
        LoadFloatOne(dy, dyAddr + row * BD);
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

// 行宽 BD 作为编译期模板参数：BD=64 时行步长恰为一个 fp32 向量寄存器全宽（原有实例，语义不变）；
// BD=16 时用 16 车道掩码 + 行步长 16，让 D 不是 64 整数倍的场景也走同一套 VF 快路径。
template <typename T, uint32_t BD>
__aicore__ inline void ComputeTileDirectRegbaseBd(
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
        if (activation == 1U || activation == 2U) {
            AscendC::VF_CALL<ComputeActivatedDyB16Vf<T, BD>>(dyAddr, yAddr, dyActAddr, dyRows);
            AscendC::VF_CALL<ComputeDxActivatedB16Vf<T, BD>>(dyActAddr, weightAddr, dxAddr, rows, width);
            if (width > 0U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T, BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 0U);
            }
            if (width > 1U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T, BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 1U);
            }
            if (width > 2U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T, BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 2U);
            }
            if (width > 3U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T, BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 3U);
            }
            if (width > 4U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T, BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 4U);
            }
            if (width > 5U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T, BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 5U);
            }
            if (width > 6U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T, BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 6U);
            }
            if (width > 7U) {
                AscendC::VF_CALL<ComputeDwOneActivatedB16Vf<T, BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 7U);
            }
            if (hasBias != 0U) {
                AscendC::VF_CALL<ComputeDbActivatedB16Vf<BD>>(dyActAddr, dbAddr, rows);
            }
            return;
        }
        if (activation != 0U) {
            return;
        }
        // activation == NONE：dy 直接参与计算，无需 y 与 dyAct 中间缓冲。
        AscendC::VF_CALL<ComputeDxDirectB16Vf<T, BD>>(dyAddr, weightAddr, dxAddr, rows, width);
        if (width > 0U) {
            AscendC::VF_CALL<ComputeDwOneDirectB16Vf<T, BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 0U);
        }
        if (width > 1U) {
            AscendC::VF_CALL<ComputeDwOneDirectB16Vf<T, BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 1U);
        }
        if (width > 2U) {
            AscendC::VF_CALL<ComputeDwOneDirectB16Vf<T, BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 2U);
        }
        if (width > 3U) {
            AscendC::VF_CALL<ComputeDwOneDirectB16Vf<T, BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 3U);
        }
        if (width > 4U) {
            AscendC::VF_CALL<ComputeDwOneDirectB16Vf<T, BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 4U);
        }
        if (width > 5U) {
            AscendC::VF_CALL<ComputeDwOneDirectB16Vf<T, BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 5U);
        }
        if (width > 6U) {
            AscendC::VF_CALL<ComputeDwOneDirectB16Vf<T, BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 6U);
        }
        if (width > 7U) {
            AscendC::VF_CALL<ComputeDwOneDirectB16Vf<T, BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 7U);
        }
        if (hasBias != 0U) {
            AscendC::VF_CALL<ComputeDbDirectB16Vf<T, BD>>(dyAddr, dbAddr, rows);
        }
        return;
    } else if constexpr (BD == 16U) {
        // fp32 BD=16（D 非 64 整数倍）走与 b16 BD=16 同构的 VF 快路径，仅省去 Cast。
        if (activation == 1U || activation == 2U) {
            AscendC::VF_CALL<ComputeActivatedDyF32Vf<BD>>(dyAddr, yAddr, dyActAddr, dyRows);
            AscendC::VF_CALL<ComputeDxActivatedF32Vf<BD>>(dyActAddr, weightAddr, dxAddr, rows, width);
            if (width > 0U) {
                AscendC::VF_CALL<ComputeDwOneActivatedF32Vf<BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 0U);
            }
            if (width > 1U) {
                AscendC::VF_CALL<ComputeDwOneActivatedF32Vf<BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 1U);
            }
            if (width > 2U) {
                AscendC::VF_CALL<ComputeDwOneActivatedF32Vf<BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 2U);
            }
            if (width > 3U) {
                AscendC::VF_CALL<ComputeDwOneActivatedF32Vf<BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 3U);
            }
            if (width > 4U) {
                AscendC::VF_CALL<ComputeDwOneActivatedF32Vf<BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 4U);
            }
            if (width > 5U) {
                AscendC::VF_CALL<ComputeDwOneActivatedF32Vf<BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 5U);
            }
            if (width > 6U) {
                AscendC::VF_CALL<ComputeDwOneActivatedF32Vf<BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 6U);
            }
            if (width > 7U) {
                AscendC::VF_CALL<ComputeDwOneActivatedF32Vf<BD>>(
                    xAddr, dyActAddr, dwAddr, rows, width, 7U);
            }
            if (hasBias != 0U) {
                AscendC::VF_CALL<ComputeDbActivatedF32Vf<BD>>(dyActAddr, dbAddr, rows);
            }
            return;
        }
        if (activation != 0U) {
            return;
        }
        // activation == NONE：dy 直接参与计算，无需 y 与 dyAct 中间缓冲。
        AscendC::VF_CALL<ComputeDxDirectF32Vf<BD>>(dyAddr, weightAddr, dxAddr, rows, width);
        if (width > 0U) {
            AscendC::VF_CALL<ComputeDwOneDirectF32Vf<BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 0U);
        }
        if (width > 1U) {
            AscendC::VF_CALL<ComputeDwOneDirectF32Vf<BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 1U);
        }
        if (width > 2U) {
            AscendC::VF_CALL<ComputeDwOneDirectF32Vf<BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 2U);
        }
        if (width > 3U) {
            AscendC::VF_CALL<ComputeDwOneDirectF32Vf<BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 3U);
        }
        if (width > 4U) {
            AscendC::VF_CALL<ComputeDwOneDirectF32Vf<BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 4U);
        }
        if (width > 5U) {
            AscendC::VF_CALL<ComputeDwOneDirectF32Vf<BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 5U);
        }
        if (width > 6U) {
            AscendC::VF_CALL<ComputeDwOneDirectF32Vf<BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 6U);
        }
        if (width > 7U) {
            AscendC::VF_CALL<ComputeDwOneDirectF32Vf<BD>>(
                xAddr, dyAddr, dwAddr, rows, width, 7U);
        }
        if (hasBias != 0U) {
            AscendC::VF_CALL<ComputeDbDirectF32Vf<BD>>(dyAddr, dbAddr, rows);
        }
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

// 运行期按行宽分发：BD=64（b16/多 64 通道）与 BD=16（D 非 64 整数倍）各实例化一份。
template <typename T>
__aicore__ inline void ComputeTileDirectRegbase(
    LocalTensor<T> xRaw, LocalTensor<T> dyRaw, LocalTensor<T> yRaw, LocalTensor<T> weightRaw,
    LocalTensor<float> stateLocal, LocalTensor<float> dxLocal, LocalTensor<float> dwLocal,
    LocalTensor<float> dbLocal, LocalTensor<float> dh0Local,
    uint32_t rows, uint32_t width, uint32_t dyRows, uint32_t activation,
    uint32_t validRows, uint32_t useInitialState, uint32_t computeDh0, uint32_t hasBias,
    uint32_t bd)
{
    if (bd == 16U) {
        ComputeTileDirectRegbaseBd<T, 16U>(xRaw, dyRaw, yRaw, weightRaw, stateLocal, dxLocal, dwLocal,
                                           dbLocal, dh0Local, rows, width, dyRows, activation,
                                           validRows, useInitialState, computeDh0, hasBias);
    } else {
        ComputeTileDirectRegbaseBd<T, BWD_DIRECT_BD>(xRaw, dyRaw, yRaw, weightRaw, stateLocal, dxLocal, dwLocal,
                                                     dbLocal, dh0Local, rows, width, dyRows, activation,
                                                     validRows, useInitialState, computeDh0, hasBias);
    }
}

} // namespace NsCausalConv1dBwd

#endif // ASCENDC_CAUSAL_CONV1D_BWD_ARCH92_REGBASE_H_
