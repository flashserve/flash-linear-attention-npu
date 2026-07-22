/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_vector.h
 * \brief Vector side process for fused prepare_wy_repr_bwd A5.
 */

#ifndef PREPARE_WY_REPR_BWD_VECTOR_H
#define PREPARE_WY_REPR_BWD_VECTOR_H

#include <type_traits>

#include "prepare_wy_repr_bwd_common.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_utils/vector/regbase.hpp"

using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename kType, bool USE_GEXP>
__simd_vf__ inline void PrepareWyReprBwdScaleRowsRegbase(__ubuf__ kType *out, __ubuf__ kType *in,
                                                         __ubuf__ float *beta, __ubuf__ float *gExp,
                                                         uint16_t rowCount, uint16_t colCount, uint16_t rowOffset)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t colLoop = (colCount + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<kType> inputReg;
    RegTensor<kType> outputReg;
    RegTensor<float> betaReg;
    RegTensor<float> gExpReg;
    RegTensor<float> scaleReg;
    RegTensor<float> inputZeroReg;
    RegTensor<float> inputOneReg;
    RegTensor<float> outputZeroReg;
    RegTensor<float> outputOneReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadIn<float, true>(betaReg, beta + rowOffset + row);
        if constexpr (USE_GEXP) {
            LoadIn<float, true>(gExpReg, gExp + rowOffset + row);
            Mul(scaleReg, betaReg, gExpReg, maskFull32);
        }
        for (uint32_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            uint32_t elemOffset = row * colCount + colIdx * eleNumPerVf;
            LoadIn<kType, false>(inputReg, in + elemOffset);
            CastHalf2Float<kType>(inputZeroReg, inputOneReg, inputReg, maskFull16);
            if constexpr (USE_GEXP) {
                MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg, scaleReg, scaleReg,
                               maskFull32);
            } else {
                MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg, betaReg, betaReg, maskFull32);
            }
            CastFloat2Half<kType>(outputReg, outputZeroReg, outputOneReg, maskFull32);
            StoreAlign(out + elemOffset, outputReg, maskFull16);
        }
    }
}

template <typename kType>
__simd_vf__ inline void PrepareWyReprBwdAddStrictLowerRegbase(__ubuf__ kType *out, __ubuf__ kType *lhs,
                                                              __ubuf__ kType *rhs, uint16_t rowCount,
                                                              uint16_t colCount, uint16_t rowOffset)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t colLoop = (colCount + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<kType> lhsReg;
    RegTensor<kType> rhsReg;
    RegTensor<kType> outputReg;
    RegTensor<float> lhsZeroReg;
    RegTensor<float> lhsOneReg;
    RegTensor<float> rhsZeroReg;
    RegTensor<float> rhsOneReg;
    RegTensor<float> resultZeroReg;
    RegTensor<float> resultOneReg;
    RegTensor<half> colIdxReg;
    RegTensor<float> colBaseZeroReg;
    RegTensor<float> colBaseOneReg;
    RegTensor<float> colIdxZeroReg;
    RegTensor<float> colIdxOneReg;
    RegTensor<float> rowIdxReg;
    RegTensor<float> zeroReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskZeroSelect;
    MaskReg maskOneSelect;
    MaskReg maskStore;

    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colBaseZeroReg, colBaseOneReg, colIdxReg, maskFull16);
    Duplicate(zeroReg, 0.0f, maskFull32);
    Duplicate(rowIdxReg, static_cast<float>(rowOffset), maskFull32);

    for (uint16_t row = 0; row < rowCount; ++row) {
        for (uint32_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            uint32_t elemOffset = row * colCount + colIdx * eleNumPerVf;
            uint32_t curEleNum = eleNumPerVf;
            if (colIdx * eleNumPerVf + curEleNum > colCount) {
                curEleNum = colCount - colIdx * eleNumPerVf;
            }
            maskStore = UpdateMask<kType>(curEleNum);
            LoadIn<kType, false>(lhsReg, lhs + elemOffset);
            LoadIn<kType, false>(rhsReg, rhs + elemOffset);
            CastHalf2Float<kType>(lhsZeroReg, lhsOneReg, lhsReg, maskStore);
            CastHalf2Float<kType>(rhsZeroReg, rhsOneReg, rhsReg, maskStore);
            Add(resultZeroReg, lhsZeroReg, rhsZeroReg, maskFull32);
            Add(resultOneReg, lhsOneReg, rhsOneReg, maskFull32);
            Adds(colIdxZeroReg, colBaseZeroReg, static_cast<float>(colIdx * eleNumPerVf), maskFull32);
            Adds(colIdxOneReg, colBaseOneReg, static_cast<float>(colIdx * eleNumPerVf), maskFull32);
            CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect, colIdxZeroReg, colIdxOneReg,
                                              rowIdxReg, rowIdxReg, maskFull32);
            SelectTwoReg(resultZeroReg, resultOneReg, zeroReg, zeroReg, resultZeroReg, resultOneReg, maskZeroSelect,
                         maskOneSelect);
            CastFloat2Half<kType>(outputReg, resultZeroReg, resultOneReg, maskFull32);
            StoreAlign(out + elemOffset, outputReg, maskStore);
        }
        Adds(rowIdxReg, rowIdxReg, 1.0f, maskFull32);
    }
}

template <typename kType>
__simd_vf__ inline void PrepareWyReprBwdBuildDStrictUpperRegbase(__ubuf__ kType *out, __ubuf__ kType *da6t,
                                                                 __ubuf__ float *gAll, uint16_t rowCount,
                                                                 uint16_t colCount, uint16_t rowOffset)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t colLoop = (colCount + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<kType> da6tReg;
    RegTensor<kType> outputReg;
    RegTensor<float> da6tZeroReg;
    RegTensor<float> da6tOneReg;
    RegTensor<float> gColZeroReg;
    RegTensor<float> gColOneReg;
    RegTensor<float> gRowReg;
    RegTensor<float> gFactorZeroReg;
    RegTensor<float> gFactorOneReg;
    RegTensor<float> resultZeroReg;
    RegTensor<float> resultOneReg;
    RegTensor<half> colIdxReg;
    RegTensor<float> colBaseZeroReg;
    RegTensor<float> colBaseOneReg;
    RegTensor<float> colIdxZeroReg;
    RegTensor<float> colIdxOneReg;
    RegTensor<float> rowIdxReg;
    RegTensor<float> zeroReg;
    RegTensor<float> negOneReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskZeroSelect;
    MaskReg maskOneSelect;
    MaskReg maskStore;

    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colBaseZeroReg, colBaseOneReg, colIdxReg, maskFull16);
    Duplicate(zeroReg, 0.0f, maskFull32);
    Duplicate(negOneReg, -1.0f, maskFull32);
    Duplicate(rowIdxReg, static_cast<float>(rowOffset), maskFull32);

    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadIn<float, true>(gRowReg, gAll + rowOffset + row);
        for (uint32_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            uint32_t elemOffset = row * colCount + colIdx * eleNumPerVf;
            uint32_t curEleNum = eleNumPerVf;
            if (colIdx * eleNumPerVf + curEleNum > colCount) {
                curEleNum = colCount - colIdx * eleNumPerVf;
            }
            maskStore = UpdateMask<kType>(curEleNum);

            LoadIn<kType, false>(da6tReg, da6t + elemOffset);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(gColZeroReg, gColOneReg, gAll + colIdx * eleNumPerVf);

            CastHalf2Float<kType>(da6tZeroReg, da6tOneReg, da6tReg, maskStore);
            SubFloatTwoReg(gFactorZeroReg, gFactorOneReg, gColZeroReg, gColOneReg, gRowReg, gRowReg, maskFull32);
            MinsFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, 0.0f, maskFull32);
            ExpFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, maskFull32);
            MulFloatTwoReg(da6tZeroReg, da6tOneReg, da6tZeroReg, da6tOneReg, negOneReg, negOneReg, maskFull32);
            MulFloatTwoReg(resultZeroReg, resultOneReg, da6tZeroReg, da6tOneReg, gFactorZeroReg, gFactorOneReg,
                           maskFull32);

            Adds(colIdxZeroReg, colBaseZeroReg, static_cast<float>(colIdx * eleNumPerVf), maskFull32);
            Adds(colIdxOneReg, colBaseOneReg, static_cast<float>(colIdx * eleNumPerVf), maskFull32);
            CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect, maskOneSelect, colIdxZeroReg, colIdxOneReg,
                                              rowIdxReg, rowIdxReg, maskFull32);
            SelectTwoReg(resultZeroReg, resultOneReg, zeroReg, zeroReg, resultZeroReg, resultOneReg, maskZeroSelect,
                         maskOneSelect);
            CastFloat2Half<kType>(outputReg, resultZeroReg, resultOneReg, maskFull32);
            StoreAlign(out + elemOffset, outputReg, maskStore);
        }
        Adds(rowIdxReg, rowIdxReg, 1.0f, maskFull32);
    }
}

template <typename kType>
__simd_vf__ inline void PrepareWyReprBwdDvbOutputRegbase(__ubuf__ kType *dvOut, __ubuf__ float *dbetaOut,
                                                         __ubuf__ kType *dvbIn, __ubuf__ kType *vIn,
                                                         __ubuf__ float *beta, __ubuf__ float *dbetaIn,
                                                         uint16_t rowCount, uint16_t colCount)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t colLoop = (colCount + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<kType> dvbReg;
    RegTensor<kType> vReg;
    RegTensor<kType> dvOutReg;
    RegTensor<float> betaReg;
    RegTensor<float> dbetaInReg;
    RegTensor<float> dbetaAddReg;
    RegTensor<float> dvbZeroReg;
    RegTensor<float> dvbOneReg;
    RegTensor<float> vZeroReg;
    RegTensor<float> vOneReg;
    RegTensor<float> prodZeroReg;
    RegTensor<float> prodOneReg;
    RegTensor<float> reduceVecReg;
    RegTensor<float> blockSumReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    UnalignRegForStore uStore;

    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadIn<float, true>(betaReg, beta + row);
        Duplicate(reduceVecReg, 0.0f, maskFull32);
        for (uint32_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            uint32_t elemOffset = row * colCount + colIdx * eleNumPerVf;
            LoadIn<kType, false>(dvbReg, dvbIn + elemOffset);
            LoadIn<kType, false>(vReg, vIn + elemOffset);
            CastHalf2Float<kType>(dvbZeroReg, dvbOneReg, dvbReg, maskFull16);
            CastHalf2Float<kType>(vZeroReg, vOneReg, vReg, maskFull16);

            MulFloatTwoReg(prodZeroReg, prodOneReg, vZeroReg, vOneReg, dvbZeroReg, dvbOneReg, maskFull32);
            Add(blockSumReg, prodZeroReg, prodOneReg, maskFull32);
            Add(reduceVecReg, reduceVecReg, blockSumReg, maskFull32);

            MulFloatTwoReg(dvbZeroReg, dvbOneReg, dvbZeroReg, dvbOneReg, betaReg, betaReg, maskFull32);
            CastFloat2Half<kType>(dvOutReg, dvbZeroReg, dvbOneReg, maskFull32);
            StoreAlign(dvOut + elemOffset, dvOutReg, maskFull16);
        }
        ReduceSum(dbetaAddReg, reduceVecReg, maskFull32);
        LoadIn<float, true>(dbetaInReg, dbetaIn + row);
        Add(dbetaAddReg, dbetaAddReg, dbetaInReg, maskFull32);
        StoreUnAlignOut<float>(dbetaOut + row, dbetaAddReg, maskFull32, uStore, 1);
    }
}

template <typename kType>
__simd_vf__ inline void PrepareWyReprBwdDkktRowDgRegbase(__ubuf__ float *dgOut, __ubuf__ kType *dIn,
                                                         __ubuf__ kType *kktIn, __ubuf__ float *beta,
                                                         __ubuf__ float *dgIn, uint16_t rowCount, uint16_t colCount,
                                                         uint16_t validColCount)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t colLoop = (colCount + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<kType> dReg;
    RegTensor<kType> kktReg;
    RegTensor<float> dZeroReg;
    RegTensor<float> dOneReg;
    RegTensor<float> kktZeroReg;
    RegTensor<float> kktOneReg;
    RegTensor<float> betaZeroReg;
    RegTensor<float> betaOneReg;
    RegTensor<float> prodZeroReg;
    RegTensor<float> prodOneReg;
    RegTensor<float> reduceVecReg;
    RegTensor<float> reduceReg;
    RegTensor<float> dgReg;
    RegTensor<float> zeroReg;
    RegTensor<float> negOneReg;
    RegTensor<half> colIdxReg;
    RegTensor<float> colBaseZeroReg;
    RegTensor<float> colBaseOneReg;
    RegTensor<float> colIdxZeroReg;
    RegTensor<float> colIdxOneReg;
    RegTensor<float> validColReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskStore;
    MaskReg maskInvalidZero;
    MaskReg maskInvalidOne;
    UnalignRegForStore uStore;

    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colBaseZeroReg, colBaseOneReg, colIdxReg, maskFull16);
    Duplicate(zeroReg, 0.0f, maskFull32);
    Duplicate(negOneReg, -1.0f, maskFull32);
    Duplicate(validColReg, static_cast<float>(validColCount), maskFull32);

    for (uint16_t row = 0; row < rowCount; ++row) {
        Duplicate(reduceVecReg, 0.0f, maskFull32);
        for (uint32_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            uint32_t elemOffset = row * colCount + colIdx * eleNumPerVf;
            uint32_t curEleNum = eleNumPerVf;
            if (colIdx * eleNumPerVf + curEleNum > colCount) {
                curEleNum = colCount - colIdx * eleNumPerVf;
            }
            maskStore = UpdateMask<kType>(curEleNum);
            LoadIn<kType, false>(dReg, dIn + elemOffset);
            LoadIn<kType, false>(kktReg, kktIn + elemOffset);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(betaZeroReg, betaOneReg, beta + colIdx * eleNumPerVf);
            CastHalf2Float<kType>(dZeroReg, dOneReg, dReg, maskStore);
            CastHalf2Float<kType>(kktZeroReg, kktOneReg, kktReg, maskStore);
            MulFloatTwoReg(prodZeroReg, prodOneReg, dZeroReg, dOneReg, kktZeroReg, kktOneReg, maskFull32);
            MulFloatTwoReg(prodZeroReg, prodOneReg, prodZeroReg, prodOneReg, betaZeroReg, betaOneReg, maskFull32);

            Adds(colIdxZeroReg, colBaseZeroReg, static_cast<float>(colIdx * eleNumPerVf), maskFull32);
            Adds(colIdxOneReg, colBaseOneReg, static_cast<float>(colIdx * eleNumPerVf), maskFull32);
            CompareTwoReg<float, CMPMODE::GE>(maskInvalidZero, maskInvalidOne, colIdxZeroReg, colIdxOneReg,
                                              validColReg, validColReg, maskFull32);
            SelectTwoReg(prodZeroReg, prodOneReg, zeroReg, zeroReg, prodZeroReg, prodOneReg, maskInvalidZero,
                         maskInvalidOne);
            Add(prodZeroReg, prodZeroReg, prodOneReg, maskFull32);
            Add(reduceVecReg, reduceVecReg, prodZeroReg, maskFull32);
        }
        ReduceSum(reduceReg, reduceVecReg, maskFull32);
        Mul(reduceReg, reduceReg, negOneReg, maskFull32);
        LoadIn<float, true>(dgReg, dgIn + row);
        Add(dgReg, dgReg, reduceReg, maskFull32);
        StoreUnAlignOut<float>(dgOut + row, dgReg, maskFull32, uStore, 1);
    }
}

template <typename kType, bool INIT_ACC>
__simd_vf__ inline void PrepareWyReprBwdDkktColDgRegbase(__ubuf__ float *accOut, __ubuf__ kType *dIn,
                                                         __ubuf__ kType *kktIn, __ubuf__ float *beta,
                                                         uint16_t srcRowCount, uint16_t colCount,
                                                         uint16_t targetColOffset, uint16_t targetColCount)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t colLoop = (targetColCount + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<kType> dReg;
    RegTensor<kType> kktReg;
    RegTensor<float> dRegFp32;
    RegTensor<float> kktRegFp32;
    RegTensor<float> prodReg;
    RegTensor<float> sumReg;
    RegTensor<float> betaReg;
    RegTensor<float> accReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<kType, MaskPattern::ALL>();
    MaskReg maskStore;
    MaskReg maskLoad;
    UnalignRegForStore uStore;

    for (uint32_t colIdx = 0; colIdx < colLoop; ++colIdx) {
        uint32_t targetElemOffset = colIdx * eleNumPerVf;
        uint32_t elemOffset = targetColOffset + targetElemOffset;
        uint32_t curEleNum = eleNumPerVf;
        if (targetElemOffset + curEleNum > targetColCount) {
            curEleNum = targetColCount - targetElemOffset;
        }
        maskStore = UpdateMask<float>(curEleNum);
        maskLoad = UpdateMask<kType>(curEleNum);
        if (elemOffset + eleNumPerVf <= colCount) {
            Duplicate(sumReg, 0.0f, maskFull32);
            for (uint16_t row = 0; row < srcRowCount; ++row) {
                uint32_t srcElemOffset = row * colCount + elemOffset;
                DataCopy<kType, LoadDist::DIST_UNPACK_B16>(dReg, dIn + srcElemOffset);
                DataCopy<kType, LoadDist::DIST_UNPACK_B16>(kktReg, kktIn + srcElemOffset);
                Cast<float, kType, ctHalf2Fp32Zero>(dRegFp32, dReg, maskLoad);
                Cast<float, kType, ctHalf2Fp32Zero>(kktRegFp32, kktReg, maskLoad);
                Mul(prodReg, dRegFp32, kktRegFp32, maskFull32);
                Add(sumReg, sumReg, prodReg, maskFull32);
            }
            DataCopy<float, LoadDist::DIST_NORM>(betaReg, beta + elemOffset);
            Mul(sumReg, sumReg, betaReg, maskFull32);
            if constexpr (!INIT_ACC) {
                DataCopy<float, LoadDist::DIST_NORM>(accReg, accOut + targetElemOffset);
                Add(sumReg, sumReg, accReg, maskFull32);
            }
            DataCopy<float, StoreDist::DIST_NORM_B32>(accOut + targetElemOffset, sumReg, maskStore);
        } else {
            for (uint32_t col = 0; col < curEleNum; ++col) {
                Duplicate(sumReg, 0.0f, maskFull32);
                for (uint16_t row = 0; row < srcRowCount; ++row) {
                    uint32_t srcElemOffset = row * colCount + elemOffset + col;
                    LoadIn<kType, true>(dReg, dIn + srcElemOffset);
                    LoadIn<kType, true>(kktReg, kktIn + srcElemOffset);
                    Cast<float, kType, ctHalf2Fp32Zero>(dRegFp32, dReg, maskFull16);
                    Cast<float, kType, ctHalf2Fp32Zero>(kktRegFp32, kktReg, maskFull16);
                    Mul(prodReg, dRegFp32, kktRegFp32, maskFull32);
                    Add(sumReg, sumReg, prodReg, maskFull32);
                }
                LoadIn<float, true>(betaReg, beta + elemOffset + col);
                Mul(sumReg, sumReg, betaReg, maskFull32);
                if constexpr (!INIT_ACC) {
                    LoadIn<float, true>(accReg, accOut + targetElemOffset + col);
                    Add(sumReg, sumReg, accReg, maskFull32);
                }
                StoreUnAlignOut<float>(accOut + targetElemOffset + col, sumReg, maskFull32, uStore, 1);
            }
        }
    }
}

__simd_vf__ inline void PrepareWyReprBwdAddFp32RowsRegbase(__ubuf__ float *out, __ubuf__ float *lhs,
                                                           __ubuf__ float *rhs, uint16_t colOffset,
                                                           uint16_t elements, uint16_t colCount)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t loopCnt = (elements + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<float> lhsReg;
    RegTensor<float> rhsReg;
    RegTensor<float> outReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskStore;
    UnalignRegForStore uStore;

    for (uint32_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
        uint32_t elemOffset = loopIdx * eleNumPerVf;
        uint32_t globalElemOffset = colOffset + elemOffset;
        uint32_t curEleNum = eleNumPerVf;
        if (elemOffset + curEleNum > elements) {
            curEleNum = elements - elemOffset;
        }
        maskStore = UpdateMask<float>(curEleNum);
        if (globalElemOffset + eleNumPerVf <= colCount) {
            DataCopy<float, LoadDist::DIST_NORM>(lhsReg, lhs + globalElemOffset);
            DataCopy<float, LoadDist::DIST_NORM>(rhsReg, rhs + elemOffset);
            Add(outReg, lhsReg, rhsReg, maskFull32);
            DataCopy<float, StoreDist::DIST_NORM_B32>(out + globalElemOffset, outReg, maskStore);
        } else {
            for (uint32_t col = 0; col < curEleNum; ++col) {
                LoadIn<float, true>(lhsReg, lhs + globalElemOffset + col);
                LoadIn<float, true>(rhsReg, rhs + elemOffset + col);
                Add(outReg, lhsReg, rhsReg, maskFull32);
                StoreUnAlignOut<float>(out + globalElemOffset + col, outReg, maskFull32, uStore, 1);
            }
        }
    }
}

template <typename gType>
__simd_vf__ inline void PrepareWyReprBwdCastBetaGRegbase(__ubuf__ float *dst, __ubuf__ gType *src, uint16_t elements)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(gType);
    uint32_t loopCnt = (elements + eleNumPerVf - 1) / eleNumPerVf;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    if constexpr (std::is_same<gType, float>::value) {
        RegTensor<float> srcReg;
        for (uint32_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
            uint32_t elemOffset = loopIdx * eleNumPerVf;
            LoadAlign(srcReg, src + elemOffset);
            StoreAlign(dst + elemOffset, srcReg, maskFull32);
        }
    } else {
        RegTensor<gType> srcReg;
        RegTensor<float> srcZeroReg;
        RegTensor<float> srcOneReg;
        for (uint32_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
            uint32_t elemOffset = loopIdx * eleNumPerVf;
            LoadIn<gType, false>(srcReg, src + elemOffset);
            CastHalf2Float<gType>(srcZeroReg, srcOneReg, srcReg, maskFull16);
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst + elemOffset, srcZeroReg, srcOneReg, maskFull32);
        }
    }
}

__simd_vf__ inline void PrepareWyReprBwdExpFloatRegbase(__ubuf__ float *dst, __ubuf__ float *src, uint16_t elements)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t loopCnt = (elements + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<float> srcReg;
    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();

    for (uint32_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
        uint32_t elemOffset = loopIdx * eleNumPerVf;
        LoadAlign(srcReg, src + elemOffset);
        Exp(srcReg, srcReg, maskFull32);
        StoreAlign(dst + elemOffset, srcReg, maskFull32);
    }
}

template <typename kType>
__simd_vf__ inline void PrepareWyReprBwdDkbFirstRegbase(__ubuf__ float *kFp32Out, __ubuf__ float *dkbBetaFp32Out,
                                                        __ubuf__ float *dbetaOut, __ubuf__ kType *kIn,
                                                        __ubuf__ kType *dkbIn, __ubuf__ float *beta,
                                                        uint16_t rowCount, uint16_t colCount)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t colLoop = (colCount + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<kType> kReg;
    RegTensor<kType> dkbReg;
    RegTensor<float> betaReg;
    RegTensor<float> kZeroReg;
    RegTensor<float> kOneReg;
    RegTensor<float> dkbZeroReg;
    RegTensor<float> dkbOneReg;
    RegTensor<float> prodZeroReg;
    RegTensor<float> prodOneReg;
    RegTensor<float> dkbBetaZeroReg;
    RegTensor<float> dkbBetaOneReg;
    RegTensor<float> reduceVecReg;
    RegTensor<float> blockSumReg;
    RegTensor<float> dbetaReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    UnalignRegForStore uStore;

    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadIn<float, true>(betaReg, beta + row);
        Duplicate(reduceVecReg, 0.0f, maskFull32);
        for (uint32_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            uint32_t elemOffset = row * colCount + colIdx * eleNumPerVf;
            LoadIn<kType, false>(kReg, kIn + elemOffset);
            LoadIn<kType, false>(dkbReg, dkbIn + elemOffset);
            CastHalf2Float<kType>(kZeroReg, kOneReg, kReg, maskFull16);
            CastHalf2Float<kType>(dkbZeroReg, dkbOneReg, dkbReg, maskFull16);
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(kFp32Out + elemOffset, kZeroReg, kOneReg, maskFull32);

            MulFloatTwoReg(prodZeroReg, prodOneReg, kZeroReg, kOneReg, dkbZeroReg, dkbOneReg, maskFull32);
            Add(blockSumReg, prodZeroReg, prodOneReg, maskFull32);
            Add(reduceVecReg, reduceVecReg, blockSumReg, maskFull32);

            MulFloatTwoReg(dkbBetaZeroReg, dkbBetaOneReg, dkbZeroReg, dkbOneReg, betaReg, betaReg, maskFull32);
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(dkbBetaFp32Out + elemOffset, dkbBetaZeroReg, dkbBetaOneReg,
                                                         maskFull32);
        }
        ReduceSum(dbetaReg, reduceVecReg, maskFull32);
        StoreUnAlignOut<float>(dbetaOut + row, dbetaReg, maskFull32, uStore, 1);
    }
}

template <typename kType>
__simd_vf__ inline void PrepareWyReprBwdDkbgSecondRegbase(__ubuf__ float *dkbgGExpFp32Out, __ubuf__ float *dbetaOut,
                                                          __ubuf__ float *dgOut, __ubuf__ kType *dkbgIn,
                                                          __ubuf__ float *kFp32In, __ubuf__ float *beta,
                                                          __ubuf__ float *gExp, __ubuf__ float *dbetaIn,
                                                          uint16_t rowCount, uint16_t colCount)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t colLoop = (colCount + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<kType> dkbgReg;
    RegTensor<float> betaReg;
    RegTensor<float> gExpReg;
    RegTensor<float> dbetaReg;
    RegTensor<float> dgReg;
    RegTensor<float> dkbgZeroReg;
    RegTensor<float> dkbgOneReg;
    RegTensor<float> dkbgGExpZeroReg;
    RegTensor<float> dkbgGExpOneReg;
    RegTensor<float> kZeroReg;
    RegTensor<float> kOneReg;
    RegTensor<float> prodZeroReg;
    RegTensor<float> prodOneReg;
    RegTensor<float> reduceVecReg;
    RegTensor<float> blockSumReg;
    RegTensor<float> scaleReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    UnalignRegForStore uStoreDBeta;
    UnalignRegForStore uStoreDg;

    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadIn<float, true>(betaReg, beta + row);
        LoadIn<float, true>(gExpReg, gExp + row);
        Duplicate(reduceVecReg, 0.0f, maskFull32);
        for (uint32_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            uint32_t elemOffset = row * colCount + colIdx * eleNumPerVf;
            LoadIn<kType, false>(dkbgReg, dkbgIn + elemOffset);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(kZeroReg, kOneReg, kFp32In + elemOffset);
            CastHalf2Float<kType>(dkbgZeroReg, dkbgOneReg, dkbgReg, maskFull16);
            MulFloatTwoReg(dkbgGExpZeroReg, dkbgGExpOneReg, dkbgZeroReg, dkbgOneReg, gExpReg, gExpReg, maskFull32);
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(dkbgGExpFp32Out + elemOffset, dkbgGExpZeroReg,
                                                         dkbgGExpOneReg, maskFull32);

            MulFloatTwoReg(prodZeroReg, prodOneReg, kZeroReg, kOneReg, dkbgGExpZeroReg, dkbgGExpOneReg, maskFull32);
            Add(blockSumReg, prodZeroReg, prodOneReg, maskFull32);
            Add(reduceVecReg, reduceVecReg, blockSumReg, maskFull32);
        }
        ReduceSum(scaleReg, reduceVecReg, maskFull32);
        LoadIn<float, true>(dbetaReg, dbetaIn + row);
        Add(dbetaReg, dbetaReg, scaleReg, maskFull32);
        StoreUnAlignOut<float>(dbetaOut + row, dbetaReg, maskFull32, uStoreDBeta, 1);
        Mul(dgReg, scaleReg, betaReg, maskFull32);
        StoreUnAlignOut<float>(dgOut + row, dgReg, maskFull32, uStoreDg, 1);
    }
}

template <typename kType, bool HAS_OLD_DK>
__simd_vf__ inline void PrepareWyReprBwdDkOutputRegbase(__ubuf__ kType *dkOut, __ubuf__ kType *dkIn,
                                                        __ubuf__ kType *oldDkIn, __ubuf__ float *dkbBetaFp32In,
                                                        __ubuf__ float *dkbgGExpFp32In, __ubuf__ float *beta,
                                                        uint16_t rowCount, uint16_t colCount)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t colLoop = (colCount + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<kType> dkReg;
    RegTensor<kType> oldDkReg;
    RegTensor<kType> outReg;
    RegTensor<float> betaReg;
    RegTensor<float> dkZeroReg;
    RegTensor<float> dkOneReg;
    RegTensor<float> oldDkZeroReg;
    RegTensor<float> oldDkOneReg;
    RegTensor<float> dkbBetaZeroReg;
    RegTensor<float> dkbBetaOneReg;
    RegTensor<float> dkbgGExpZeroReg;
    RegTensor<float> dkbgGExpOneReg;
    RegTensor<float> resultZeroReg;
    RegTensor<float> resultOneReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadIn<float, true>(betaReg, beta + row);
        for (uint32_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            uint32_t elemOffset = row * colCount + colIdx * eleNumPerVf;
            LoadIn<kType, false>(dkReg, dkIn + elemOffset);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(dkbBetaZeroReg, dkbBetaOneReg, dkbBetaFp32In + elemOffset);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(dkbgGExpZeroReg, dkbgGExpOneReg,
                                                        dkbgGExpFp32In + elemOffset);

            CastHalf2Float<kType>(dkZeroReg, dkOneReg, dkReg, maskFull16);
            MulFloatTwoReg(dkbgGExpZeroReg, dkbgGExpOneReg, dkbgGExpZeroReg, dkbgGExpOneReg, betaReg, betaReg,
                           maskFull32);
            AddFloatTwoReg(resultZeroReg, resultOneReg, dkZeroReg, dkOneReg, dkbBetaZeroReg, dkbBetaOneReg,
                           maskFull32);
            AddFloatTwoReg(resultZeroReg, resultOneReg, resultZeroReg, resultOneReg, dkbgGExpZeroReg, dkbgGExpOneReg,
                           maskFull32);

            if constexpr (HAS_OLD_DK) {
                LoadIn<kType, false>(oldDkReg, oldDkIn + elemOffset);
                CastHalf2Float<kType>(oldDkZeroReg, oldDkOneReg, oldDkReg, maskFull16);
                AddFloatTwoReg(resultZeroReg, resultOneReg, resultZeroReg, resultOneReg, oldDkZeroReg, oldDkOneReg,
                               maskFull32);
            }

            CastFloat2Half<kType>(outReg, resultZeroReg, resultOneReg, maskFull32);
            StoreAlign(dkOut + elemOffset, outReg, maskFull16);
        }
    }
}

template <typename gType>
__simd_vf__ inline void PrepareWyReprBwdCastBetaGOutputRegbase(__ubuf__ gType *dst, __ubuf__ float *src,
                                                               uint16_t elements)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(gType);
    uint32_t loopCnt = (elements + eleNumPerVf - 1) / eleNumPerVf;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    if constexpr (std::is_same<gType, float>::value) {
        RegTensor<float> srcReg;
        for (uint32_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
            uint32_t elemOffset = loopIdx * eleNumPerVf;
            LoadAlign(srcReg, src + elemOffset);
            StoreAlign(dst + elemOffset, srcReg, maskFull32);
        }
    } else {
        RegTensor<float> srcZeroReg;
        RegTensor<float> srcOneReg;
        RegTensor<gType> dstReg;
        for (uint32_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
            uint32_t elemOffset = loopIdx * eleNumPerVf;
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(srcZeroReg, srcOneReg, src + elemOffset);
            CastFloat2Half<gType>(dstReg, srcZeroReg, srcOneReg, maskFull32);
            StoreAlign(dst + elemOffset, dstReg, maskFull16);
        }
    }
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
class PrepareWyReprBwdVectorProcess {
public:
    __aicore__ inline PrepareWyReprBwdVectorProcess(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g, GM_ADDR cuSeqlens,
                                                    GM_ADDR chunkIndices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta,
                                                    GM_ADDR dg, GM_ADDR workspace);
    __aicore__ inline void Init(const GDN::PrepareWyReprBwdTilingData &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    template <typename copyType>
    __aicore__ inline uint32_t CopyInRows(AscendC::GlobalTensor<copyType> &inputTensor,
                                          AscendC::LocalTensor<copyType> dstTensor, uint64_t inputOffset,
                                          uint32_t elements);
    template <typename copyType>
    __aicore__ inline uint32_t CopyInBetaGRows(AscendC::GlobalTensor<copyType> &inputTensor,
                                               AscendC::LocalTensor<copyType> dstTensor, uint64_t inputOffset,
                                               uint32_t elements);
    __aicore__ inline void CastInputRows(AscendC::LocalTensor<float32_t> dstTensor,
                                         AscendC::LocalTensor<kType> srcTensor, uint32_t elements, uint32_t inputIdx);
    __aicore__ inline void WaitInputRows(uint32_t inputIdx);
    __aicore__ inline void ReleaseInputRows(uint32_t inputIdx);
    template <typename copyType>
    __aicore__ inline void CastBetaGInputRows(AscendC::LocalTensor<float32_t> dstTensor,
                                              AscendC::LocalTensor<copyType> srcTensor, uint32_t elements,
                                              uint32_t betaGInputIdx);
    __aicore__ inline void InitVectorEvents();
    __aicore__ inline void ReleaseVectorEvents();
    __aicore__ inline void SetBetaGResidentTensors(uint32_t slot);
    __aicore__ inline void PrepareOutputRows();
    __aicore__ inline void CastOutputRows(AscendC::LocalTensor<float32_t> srcTensor, uint32_t elements);
    __aicore__ inline void CopyOutRows(AscendC::GlobalTensor<kType> &outTensor,
                                       AscendC::LocalTensor<kType> srcTensor, uint64_t outOffset, uint32_t elements);
#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR
    __aicore__ inline void CopyOutRowsStrided(AscendC::GlobalTensor<kType> &outTensor,
                                              AscendC::LocalTensor<kType> srcTensor, uint64_t outOffset,
                                              uint32_t outStride, uint32_t colCount, uint32_t rowCount);
#endif
    __aicore__ inline void CopyOutBetaGRows(AscendC::GlobalTensor<gType> &outTensor,
                                            AscendC::LocalTensor<float32_t> srcTensor, uint64_t outOffset,
                                            uint32_t elements);
    __aicore__ inline void ProcessVectorTask(const PrepareWyReprBwdTaskInfo &task, uint64_t hv, uint64_t hk,
                                             GM_ADDR slotBase);
    __aicore__ inline void ProcessDa4Task(const PrepareWyReprBwdTaskInfo &task, GM_ADDR slotBase);
    __aicore__ inline void ProcessDTask(const PrepareWyReprBwdTaskInfo &task, uint64_t hv, GM_ADDR slotBase);
    __aicore__ inline void ProcessOutputTask(const PrepareWyReprBwdTaskInfo &task, uint64_t hv, uint64_t hk,
                                             uint64_t groupSize, GM_ADDR slotBase);

private:
    GDN::PrepareWyReprBwdTilingData tiling_{};
    uint32_t curSlot_ = 0;
    uint32_t curInputPingPong_ = 0;
    uint32_t curBetaGInputPingPong_ = 0;
    uint32_t curOutputPingPong_ = 0;
    Arch::CrossCoreFlag vecToCubeFlag_{PREPARE_WY_REPR_BWD_VEC_TO_CUBE_FLAG_READY};
    Arch::CrossCoreFlag cubeToVecFlag_{PREPARE_WY_REPR_BWD_CUBE_TO_VEC_FLAG_READY};

    GM_ADDR k_ = nullptr;
    GM_ADDR v_ = nullptr;
    GM_ADDR beta_ = nullptr;
    GM_ADDR g_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR dk_ = nullptr;
    GM_ADDR dv_ = nullptr;
    GM_ADDR dbeta_ = nullptr;
    GM_ADDR dg_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;

    AscendC::GlobalTensor<kType> kTensor_;
    AscendC::GlobalTensor<kType> vTensor_;
    AscendC::GlobalTensor<kType> dkTensor_;
    AscendC::GlobalTensor<kType> dvTensor_;
    AscendC::GlobalTensor<gType> betaTensor_;
    AscendC::GlobalTensor<gType> gTensor_;
    AscendC::GlobalTensor<gType> dbetaTensor_;
    AscendC::GlobalTensor<gType> dgTensor_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> inputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaGInputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaGInputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> brcbFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcFp32A_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcFp32B_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcFp32C_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gRawAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gExpAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dbetaAccFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dgAccFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lowerTriMask_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> upperTriMask_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroFp32_;

    AscendC::LocalTensor<kType> outputBuf_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT];
    AscendC::LocalTensor<kType> matrixInputBuf_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT];
    AscendC::LocalTensor<gType> betaGInputBuf_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT];
    AscendC::LocalTensor<gType> betaGOutputBuf_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT];
    AscendC::LocalTensor<float32_t> betaFp32Tensor_;
    AscendC::LocalTensor<float32_t> gFp32Tensor_;
    AscendC::LocalTensor<float32_t> scaleFp32Tensor_;
    AscendC::LocalTensor<float32_t> brcbFp32Tensor_;
    AscendC::LocalTensor<float32_t> calcFp32ATensor_;
    AscendC::LocalTensor<float32_t> calcFp32BTensor_;
    AscendC::LocalTensor<float32_t> calcFp32CTensor_;
    AscendC::LocalTensor<float32_t> gRawAllFp32Tensor_;
    AscendC::LocalTensor<float32_t> gExpAllFp32Tensor_;
    AscendC::LocalTensor<float32_t> betaAllFp32Tensor_;
    AscendC::LocalTensor<float32_t> dbetaAccFp32Tensor_;
    AscendC::LocalTensor<float32_t> dgAccFp32Tensor_;
    AscendC::LocalTensor<uint8_t> lowerTriMaskTensor_;
    AscendC::LocalTensor<uint8_t> upperTriMaskTensor_;
    AscendC::LocalTensor<float32_t> zeroFp32Tensor_;

    AscendC::GlobalTensor<kType> gmKbg_;
    AscendC::GlobalTensor<kType> gmVb_;
    AscendC::GlobalTensor<kType> gmKbeta_;
    AscendC::GlobalTensor<kType> gmDkbg_;
    AscendC::GlobalTensor<kType> gmDvb_;
    AscendC::GlobalTensor<kType> gmKKT_;
    AscendC::GlobalTensor<kType> gmDA1_;
    AscendC::GlobalTensor<kType> gmDA2_;
    AscendC::GlobalTensor<kType> gmDA4_;
    AscendC::GlobalTensor<kType> gmDA6T_;
    AscendC::GlobalTensor<kType> gmD_;
    AscendC::GlobalTensor<kType> gmDkb_;
    AscendC::GlobalTensor<kType> gmDK_;

    event_t mte2ToVEvent_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t vToMte2Event_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t betaGMte2ToVEvent_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t betaGVToMte2Event_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t vToMte3Event_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    event_t mte3ToVEvent_[PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT]{};
    uint32_t subBlockNum_ = 1;
    uint32_t subBlockIdx_ = 0;
    uint64_t keyBase_ = 0;
    uint64_t valueBase_ = 0;
    uint32_t curRow_ = 0;
    uint32_t rowTaskIdx_ = 0;
    uint32_t localRowTask_ = 0;
    uint32_t rowOffset_ = 0;
    uint32_t inputIdx_ = 0;
    uint32_t inputIdxA_ = 0;
    uint32_t inputIdxB_ = 0;
    uint32_t inputIdxC_ = 0;
    uint32_t inputIdxD_ = 0;
    uint32_t betaGInputIdx_ = 0;
    uint32_t betaGInputIdxA_ = 0;
    uint32_t betaGInputIdxB_ = 0;
    uint32_t outputIdx_ = 0;
    uint32_t eventIdx_ = 0;
    uint32_t nextKktSlot_ = 0;
    uint32_t cachedKktSlot_ = 0;
    uint64_t cachedKktHk_ = static_cast<uint64_t>(-1);
    uint32_t kktSlotForSlot_[BUFFER_COUNT_4] = {0, 0, 0, 0};
};

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::PrepareWyReprBwdVectorProcess(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR dbeta, GM_ADDR dg, GM_ADDR workspace)
    : k_(k), v_(v), beta_(beta), g_(g), cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices), dk_(dk), dv_(dv),
      dbeta_(dbeta), dg_(dg), workspace_(workspace)
{
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::Init(const GDN::PrepareWyReprBwdTilingData &tiling,
                                                                     AscendC::TPipe *pipe)
{
    tiling_ = tiling;
    pipe_ = pipe;
    kTensor_.SetGlobalBuffer((__gm__ kType *)k_);
    vTensor_.SetGlobalBuffer((__gm__ kType *)v_);
    dkTensor_.SetGlobalBuffer((__gm__ kType *)dk_);
    dvTensor_.SetGlobalBuffer((__gm__ kType *)dv_);
    betaTensor_.SetGlobalBuffer((__gm__ gType *)beta_);
    gTensor_.SetGlobalBuffer((__gm__ gType *)g_);
    dbetaTensor_.SetGlobalBuffer((__gm__ gType *)dbeta_);
    dgTensor_.SetGlobalBuffer((__gm__ gType *)dg_);

    uint32_t maxRow = static_cast<uint32_t>(tiling_.kVecRow > tiling_.vVecRow ? tiling_.kVecRow : tiling_.vVecRow);
    maxRow = maxRow > static_cast<uint32_t>(tiling_.mVecRow) ? maxRow : static_cast<uint32_t>(tiling_.mVecRow);
    maxRow = maxRow > static_cast<uint32_t>(tiling_.kktVecRow) ? maxRow : static_cast<uint32_t>(tiling_.kktVecRow);
    pipe_->InitBuffer(inputPing_, UB_BYTES_16K);
    pipe_->InitBuffer(inputPong_, UB_BYTES_16K);
    pipe_->InitBuffer(betaGInputPing_, Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(betaGInputPong_, Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(outputPing_, UB_BYTES_16K);
    pipe_->InitBuffer(outputPong_, UB_BYTES_16K);
    pipe_->InitBuffer(betaFp32_, Align32(maxRow * sizeof(float32_t)));
    pipe_->InitBuffer(gFp32_, Align32(maxRow * sizeof(float32_t)));
    pipe_->InitBuffer(scaleFp32_, Align32(maxRow * sizeof(float32_t)));
    pipe_->InitBuffer(brcbFp32_, Align32(maxRow * PRONE_BLOCK_BYTES_32));
    pipe_->InitBuffer(calcFp32A_, 2 * UB_BYTES_16K);
    pipe_->InitBuffer(calcFp32B_, 2 * UB_BYTES_16K);
    pipe_->InitBuffer(calcFp32C_, 2 * UB_BYTES_16K);
    pipe_->InitBuffer(gRawAllFp32_,
                      BUFFER_COUNT_2 *
                          Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(gExpAllFp32_,
                      BUFFER_COUNT_2 *
                          Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(betaAllFp32_,
                      BUFFER_COUNT_2 *
                          Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(dbetaAccFp32_, Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(dgAccFp32_, Align32(CHUNK_SIZE * sizeof(float32_t)));
    pipe_->InitBuffer(lowerTriMask_, CHUNK_SIZE * CHUNK_SIZE / PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE);
    pipe_->InitBuffer(upperTriMask_, CHUNK_SIZE * CHUNK_SIZE / PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE);
    pipe_->InitBuffer(zeroFp32_, PRONE_BLOCK_BYTES_32);

    matrixInputBuf_[0] = inputPing_.Get<kType>();
    matrixInputBuf_[1] = inputPong_.Get<kType>();
    betaGInputBuf_[0] = betaGInputPing_.Get<gType>();
    betaGInputBuf_[1] = betaGInputPong_.Get<gType>();
    outputBuf_[0] = outputPing_.Get<kType>();
    outputBuf_[1] = outputPong_.Get<kType>();
    betaGOutputBuf_[0] = outputPing_.Get<gType>();
    betaGOutputBuf_[1] = outputPong_.Get<gType>();
    betaFp32Tensor_ = betaFp32_.Get<float32_t>();
    gFp32Tensor_ = gFp32_.Get<float32_t>();
    scaleFp32Tensor_ = scaleFp32_.Get<float32_t>();
    brcbFp32Tensor_ = brcbFp32_.Get<float32_t>();
    calcFp32ATensor_ = calcFp32A_.Get<float32_t>();
    calcFp32BTensor_ = calcFp32B_.Get<float32_t>();
    calcFp32CTensor_ = calcFp32C_.Get<float32_t>();
    dbetaAccFp32Tensor_ = dbetaAccFp32_.Get<float32_t>();
    dgAccFp32Tensor_ = dgAccFp32_.Get<float32_t>();
    lowerTriMaskTensor_ = lowerTriMask_.Get<uint8_t>();
    upperTriMaskTensor_ = upperTriMask_.Get<uint8_t>();
    zeroFp32Tensor_ = zeroFp32_.Get<float32_t>();
    uint32_t maskBlocksPerRow = CHUNK_SIZE / PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE;
    for (uint32_t row = 0; row < CHUNK_SIZE; ++row) {
        for (uint32_t block = 0; block < maskBlocksPerRow; ++block) {
            uint32_t colStart = block * PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE;
            uint8_t lowerMaskVal = 0;
            uint8_t upperMaskVal = 0;
            for (uint32_t bit = 0; bit < PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE; ++bit) {
                uint32_t col = colStart + bit;
                if (col >= row) {
                    lowerMaskVal |= static_cast<uint8_t>(1U << bit);
                }
                if (col <= row) {
                    upperMaskVal |= static_cast<uint8_t>(1U << bit);
                }
            }
            lowerTriMaskTensor_.SetValue(row * maskBlocksPerRow + block, lowerMaskVal);
            upperTriMaskTensor_.SetValue(row * maskBlocksPerRow + block, upperMaskVal);
        }
    }
    Duplicate(zeroFp32Tensor_, 0.0f, PRONE_BLOCK_BYTES_32 / sizeof(float32_t));
    PipeBarrier<PIPE_V>();
    InitVectorEvents();
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::InitVectorEvents()
{
    for (eventIdx_ = 0; eventIdx_ < PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT; ++eventIdx_) {
        mte2ToVEvent_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        vToMte2Event_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        betaGMte2ToVEvent_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        betaGVToMte2Event_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        vToMte3Event_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>());
        mte3ToVEvent_[eventIdx_] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>());
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[eventIdx_]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[eventIdx_]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx_]);
    }
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ReleaseVectorEvents()
{
    for (eventIdx_ = 0; eventIdx_ < PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT; ++eventIdx_) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[eventIdx_]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[eventIdx_]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vToMte2Event_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(betaGMte2ToVEvent_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3Event_[eventIdx_]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx_]);
    }
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
template <typename copyType>
__aicore__ inline uint32_t
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyInRows(AscendC::GlobalTensor<copyType> &inputTensor,
                                                                           AscendC::LocalTensor<copyType> dstTensor,
                                                                           uint64_t inputOffset, uint32_t elements)
{
    inputIdx_ = curInputPingPong_;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[inputIdx_]);
    DataCopy(dstTensor, inputTensor[inputOffset], elements);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_[inputIdx_]);
    curInputPingPong_ ^= 1U;
    return inputIdx_;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CastInputRows(
    AscendC::LocalTensor<float32_t> dstTensor, AscendC::LocalTensor<kType> srcTensor, uint32_t elements,
    uint32_t inputIdx)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_[inputIdx]);
    Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, elements);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[inputIdx]);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::WaitInputRows(uint32_t inputIdx)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_[inputIdx]);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ReleaseInputRows(uint32_t inputIdx)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_[inputIdx]);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
template <typename copyType>
__aicore__ inline uint32_t PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyInBetaGRows(
    AscendC::GlobalTensor<copyType> &inputTensor, AscendC::LocalTensor<copyType> dstTensor, uint64_t inputOffset,
    uint32_t elements)
{
    betaGInputIdx_ = curBetaGInputPingPong_;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[betaGInputIdx_]);
    DataCopyPad(dstTensor, inputTensor[inputOffset],
                {1, elements * static_cast<uint32_t>(sizeof(copyType)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(betaGMte2ToVEvent_[betaGInputIdx_]);
    curBetaGInputPingPong_ ^= 1U;
    return betaGInputIdx_;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
template <typename copyType>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CastBetaGInputRows(
    AscendC::LocalTensor<float32_t> dstTensor, AscendC::LocalTensor<copyType> srcTensor, uint32_t elements,
    uint32_t betaGInputIdx)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(betaGMte2ToVEvent_[betaGInputIdx]);
    constexpr uint32_t copyEleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(copyType);
    if (elements == CHUNK_SIZE && (std::is_same<copyType, float32_t>::value || CHUNK_SIZE >= copyEleNumPerVf)) {
        PrepareWyReprBwdCastBetaGRegbase<copyType>(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dstTensor.GetPhyAddr()),
            (__ubuf__ copyType *)reinterpret_cast<uint64_t>(srcTensor.GetPhyAddr()), static_cast<uint16_t>(elements));
    } else {
        if constexpr (std::is_same<copyType, float32_t>::value) {
            Adds(dstTensor, srcTensor, 0.0f, elements);
        } else {
            Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, elements);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(betaGVToMte2Event_[betaGInputIdx]);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::SetBetaGResidentTensors(uint32_t slot)
{
    uint32_t residentOffset = slot * CHUNK_SIZE;
    betaAllFp32Tensor_ = betaAllFp32_.Get<float32_t>()[residentOffset];
    gRawAllFp32Tensor_ = gRawAllFp32_.Get<float32_t>()[residentOffset];
    gExpAllFp32Tensor_ = gExpAllFp32_.Get<float32_t>()[residentOffset];
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::PrepareOutputRows()
{
    outputIdx_ = curOutputPingPong_;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CastOutputRows(
    AscendC::LocalTensor<float32_t> srcTensor, uint32_t elements)
{
    outputIdx_ = curOutputPingPong_;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    Cast(outputBuf_[outputIdx_], srcTensor, RoundMode::CAST_RINT, elements);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyOutRows(AscendC::GlobalTensor<kType> &outTensor,
                                                                            AscendC::LocalTensor<kType> srcTensor,
                                                                            uint64_t outOffset, uint32_t elements)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    DataCopy(outTensor[outOffset], srcTensor, elements);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    curOutputPingPong_ ^= 1U;
}

#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR
template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyOutRowsStrided(
    AscendC::GlobalTensor<kType> &outTensor, AscendC::LocalTensor<kType> srcTensor, uint64_t outOffset,
    uint32_t outStride, uint32_t colCount, uint32_t rowCount)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    for (uint32_t row = 0; row < rowCount; ++row) {
        DataCopy(outTensor[outOffset + row * outStride], srcTensor[row * colCount], colCount);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    curOutputPingPong_ ^= 1U;
}
#endif

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::CopyOutBetaGRows(
    AscendC::GlobalTensor<gType> &outTensor, AscendC::LocalTensor<float32_t> srcTensor, uint64_t outOffset,
    uint32_t elements)
{
    outputIdx_ = curOutputPingPong_;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    constexpr uint32_t outputEleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(gType);
    if (elements == CHUNK_SIZE && (std::is_same<gType, float32_t>::value || CHUNK_SIZE >= outputEleNumPerVf)) {
        PrepareWyReprBwdCastBetaGOutputRegbase<gType>(
            (__ubuf__ gType *)reinterpret_cast<uint64_t>(betaGOutputBuf_[outputIdx_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(srcTensor.GetPhyAddr()), static_cast<uint16_t>(elements));
    } else {
        if constexpr (std::is_same<gType, float32_t>::value) {
            Adds(betaGOutputBuf_[outputIdx_], srcTensor, 0.0f, elements);
        } else {
            Cast(betaGOutputBuf_[outputIdx_], srcTensor, RoundMode::CAST_RINT, elements);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx_]);
    DataCopyPad(outTensor[outOffset], betaGOutputBuf_[outputIdx_],
                {1, elements * static_cast<uint32_t>(sizeof(gType)), 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx_]);
    curOutputPingPong_ ^= 1U;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ProcessVectorTask(
    const PrepareWyReprBwdTaskInfo &task, uint64_t hv, uint64_t hk, GM_ADDR slotBase)
{
    SetBetaGResidentTensors(curSlot_ & 1U);

    gmKbg_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.kbgOffset));
    gmVb_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.vbOffset));
    gmKbeta_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.kbetaOffset));

    subBlockNum_ = AscendC::GetSubBlockNum();
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    keyBase_ = hk * tiling_.T + task.keyBos;
    valueBase_ = hv * tiling_.T + task.valueBos;

    betaGInputIdxA_ =
        CopyInBetaGRows<gType>(betaTensor_, betaGInputBuf_[curBetaGInputPingPong_], valueBase_,
                               task.curChunkSize);
    betaGInputIdxB_ =
        CopyInBetaGRows<gType>(gTensor_, betaGInputBuf_[curBetaGInputPingPong_], valueBase_, task.curChunkSize);
    CastBetaGInputRows<gType>(betaAllFp32Tensor_, betaGInputBuf_[betaGInputIdxA_], task.curChunkSize,
                              betaGInputIdxA_);
    CastBetaGInputRows<gType>(gRawAllFp32Tensor_, betaGInputBuf_[betaGInputIdxB_], task.curChunkSize,
                              betaGInputIdxB_);
    PipeBarrier<PIPE_V>();
    if (task.curChunkSize == CHUNK_SIZE) {
        PrepareWyReprBwdExpFloatRegbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(gExpAllFp32Tensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(gRawAllFp32Tensor_.GetPhyAddr()),
            static_cast<uint16_t>(task.curChunkSize));
    } else {
        Exp(gExpAllFp32Tensor_, gRawAllFp32Tensor_, task.curChunkSize);
    }
    PipeBarrier<PIPE_V>();

    rowTaskIdx_ = 0;

    for (rowOffset_ = 0; rowOffset_ < task.curChunkSize; rowOffset_ += static_cast<uint32_t>(tiling_.kVecRow)) {
        localRowTask_ = rowTaskIdx_++;
        if (localRowTask_ % subBlockNum_ != subBlockIdx_) {
            continue;
        }
        curRow_ = rowOffset_ + static_cast<uint32_t>(tiling_.kVecRow) > task.curChunkSize ?
                      task.curChunkSize - rowOffset_ :
                      static_cast<uint32_t>(tiling_.kVecRow);

        inputIdx_ = CopyInRows<kType>(kTensor_, matrixInputBuf_[curInputPingPong_],
                                             (keyBase_ + rowOffset_) * K_DIM, curRow_ * K_DIM);
        WaitInputRows(inputIdx_);
        PrepareOutputRows();
        PrepareWyReprBwdScaleRowsRegbase<kType, true>(
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdx_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(gExpAllFp32Tensor_.GetPhyAddr()),
            static_cast<uint16_t>(curRow_), static_cast<uint16_t>(K_DIM), static_cast<uint16_t>(rowOffset_));
#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR == 1
        CopyOutRowsStrided(dvTensor_, outputBuf_[outputIdx_], (valueBase_ + rowOffset_) * V_DIM, V_DIM, K_DIM,
                           curRow_);
#else
        CopyOutRows(gmKbg_, outputBuf_[outputIdx_], rowOffset_ * K_DIM, curRow_ * K_DIM);
#endif

        PrepareOutputRows();
        PrepareWyReprBwdScaleRowsRegbase<kType, false>(
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdx_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(gExpAllFp32Tensor_.GetPhyAddr()),
            static_cast<uint16_t>(curRow_), static_cast<uint16_t>(K_DIM), static_cast<uint16_t>(rowOffset_));
        ReleaseInputRows(inputIdx_);
#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR == 2
        CopyOutRowsStrided(dvTensor_, outputBuf_[outputIdx_], (valueBase_ + rowOffset_) * V_DIM, V_DIM, K_DIM,
                           curRow_);
#else
        CopyOutRows(gmKbeta_, outputBuf_[outputIdx_], rowOffset_ * K_DIM, curRow_ * K_DIM);
#endif
    }

    rowTaskIdx_ = 0;
    for (rowOffset_ = 0; rowOffset_ < task.curChunkSize; rowOffset_ += static_cast<uint32_t>(tiling_.vVecRow)) {
        localRowTask_ = rowTaskIdx_++;
        if (localRowTask_ % subBlockNum_ != subBlockIdx_) {
            continue;
        }
        curRow_ = rowOffset_ + static_cast<uint32_t>(tiling_.vVecRow) > task.curChunkSize ?
                      task.curChunkSize - rowOffset_ :
                      static_cast<uint32_t>(tiling_.vVecRow);

        inputIdx_ = CopyInRows<kType>(vTensor_, matrixInputBuf_[curInputPingPong_],
                                             (valueBase_ + rowOffset_) * V_DIM, curRow_ * V_DIM);
        WaitInputRows(inputIdx_);
        PrepareOutputRows();
        PrepareWyReprBwdScaleRowsRegbase<kType, false>(
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdx_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(gExpAllFp32Tensor_.GetPhyAddr()),
            static_cast<uint16_t>(curRow_), static_cast<uint16_t>(V_DIM), static_cast<uint16_t>(rowOffset_));
        ReleaseInputRows(inputIdx_);
#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR == 3
        CopyOutRows(dvTensor_, outputBuf_[outputIdx_], (valueBase_ + rowOffset_) * V_DIM, curRow_ * V_DIM);
#else
        CopyOutRows(gmVb_, outputBuf_[outputIdx_], rowOffset_ * V_DIM, curRow_ * V_DIM);
#endif
    }

#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR
    return;
#endif
    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ProcessDa4Task(
    const PrepareWyReprBwdTaskInfo &task, GM_ADDR slotBase)
{
    gmDA1_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da1Offset));
    gmDA2_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da2Offset));
    gmDA4_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da4Offset));

    subBlockNum_ = AscendC::GetSubBlockNum();
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    rowTaskIdx_ = 0;
    Arch::CrossCoreWaitFlag(cubeToVecFlag_);
    for (rowOffset_ = 0; rowOffset_ < task.curChunkSize; rowOffset_ += static_cast<uint32_t>(tiling_.mVecRow)) {
        localRowTask_ = rowTaskIdx_++;
        if (localRowTask_ % subBlockNum_ != subBlockIdx_) {
            continue;
        }
        curRow_ = rowOffset_ + static_cast<uint32_t>(tiling_.mVecRow) > task.curChunkSize ?
                      task.curChunkSize - rowOffset_ :
                      static_cast<uint32_t>(tiling_.mVecRow);

        inputIdxA_ = CopyInRows<kType>(gmDA1_, matrixInputBuf_[curInputPingPong_],
                                              rowOffset_ * CHUNK_SIZE, curRow_ * CHUNK_SIZE);
        inputIdxB_ = CopyInRows<kType>(gmDA2_, matrixInputBuf_[curInputPingPong_],
                                              rowOffset_ * CHUNK_SIZE, curRow_ * CHUNK_SIZE);
        WaitInputRows(inputIdxA_);
        WaitInputRows(inputIdxB_);
        PrepareOutputRows();
        PrepareWyReprBwdAddStrictLowerRegbase<kType>(
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxA_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxB_].GetPhyAddr()),
            static_cast<uint16_t>(curRow_), static_cast<uint16_t>(CHUNK_SIZE), static_cast<uint16_t>(rowOffset_));
        ReleaseInputRows(inputIdxA_);
        ReleaseInputRows(inputIdxB_);
        CopyOutRows(gmDA4_, outputBuf_[outputIdx_], rowOffset_ * CHUNK_SIZE, curRow_ * CHUNK_SIZE);
    }

    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ProcessDTask(
    const PrepareWyReprBwdTaskInfo &task, uint64_t hv, GM_ADDR slotBase)
{
    SetBetaGResidentTensors(curSlot_ & 1U);

    gmDA6T_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da6Offset));
    gmD_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dOffset));
    valueBase_ = hv * tiling_.T + task.valueBos;

    subBlockNum_ = AscendC::GetSubBlockNum();
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    rowTaskIdx_ = 0;
    Arch::CrossCoreWaitFlag(cubeToVecFlag_);
    for (rowOffset_ = 0; rowOffset_ < task.curChunkSize; rowOffset_ += static_cast<uint32_t>(tiling_.mVecRow)) {
        localRowTask_ = rowTaskIdx_++;
        if (localRowTask_ % subBlockNum_ != subBlockIdx_) {
            continue;
        }
        curRow_ = rowOffset_ + static_cast<uint32_t>(tiling_.mVecRow) > task.curChunkSize ?
                      task.curChunkSize - rowOffset_ :
                      static_cast<uint32_t>(tiling_.mVecRow);

        inputIdx_ = CopyInRows<kType>(gmDA6T_, matrixInputBuf_[curInputPingPong_],
                                             rowOffset_ * CHUNK_SIZE, curRow_ * CHUNK_SIZE);
        WaitInputRows(inputIdx_);
        PrepareOutputRows();
        PrepareWyReprBwdBuildDStrictUpperRegbase<kType>(
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdx_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(gRawAllFp32Tensor_.GetPhyAddr()),
            static_cast<uint16_t>(curRow_), static_cast<uint16_t>(CHUNK_SIZE), static_cast<uint16_t>(rowOffset_));
        ReleaseInputRows(inputIdx_);
        CopyOutRows(gmD_, outputBuf_[outputIdx_], rowOffset_ * CHUNK_SIZE, curRow_ * CHUNK_SIZE);
    }

    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::ProcessOutputTask(
    const PrepareWyReprBwdTaskInfo &task, uint64_t hv, uint64_t hk, uint64_t groupSize, GM_ADDR slotBase)
{
    SetBetaGResidentTensors(curSlot_ & 1U);

    gmDkbg_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dkbgOffset));
    gmDvb_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dvbOffset));
    gmD_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dOffset));
    gmDkb_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dkbOffset));
    gmDK_.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dkOffset));
    GM_ADDR kktBase = PrepareWyReprBwdGetKktBase(workspace_, AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(),
                                                 kktSlotForSlot_[curSlot_], tiling_);
    gmKKT_.SetGlobalBuffer((__gm__ kType *)kktBase);

    keyBase_ = hk * tiling_.T + task.keyBos;
    valueBase_ = hv * tiling_.T + task.valueBos;
    bool isFirstValueHeadInGroup = (hv % groupSize) == 0;
    uint32_t rowOwned = static_cast<uint32_t>(tiling_.kVecRow < tiling_.vVecRow ? tiling_.kVecRow : tiling_.vVecRow);
    rowOwned =
        rowOwned < static_cast<uint32_t>(tiling_.kktVecRow) ? rowOwned : static_cast<uint32_t>(tiling_.kktVecRow);

    subBlockNum_ = AscendC::GetSubBlockNum();
    subBlockIdx_ = AscendC::GetSubBlockIdx();

    Arch::CrossCoreWaitFlag(cubeToVecFlag_);
    for (rowOffset_ = subBlockIdx_ * rowOwned; rowOffset_ < task.curChunkSize; rowOffset_ += rowOwned * subBlockNum_) {
        curRow_ = rowOffset_ + rowOwned > task.curChunkSize ? task.curChunkSize - rowOffset_ : rowOwned;

        inputIdxA_ = CopyInRows<kType>(kTensor_, matrixInputBuf_[curInputPingPong_],
                                              (keyBase_ + rowOffset_) * K_DIM, curRow_ * K_DIM);
        inputIdxB_ =
            CopyInRows<kType>(gmDkb_, matrixInputBuf_[curInputPingPong_], rowOffset_ * K_DIM, curRow_ * K_DIM);
        WaitInputRows(inputIdxA_);
        WaitInputRows(inputIdxB_);
        PrepareWyReprBwdDkbFirstRegbase<kType>(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(calcFp32ATensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(calcFp32BTensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dbetaAccFp32Tensor_[rowOffset_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxA_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxB_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_[rowOffset_].GetPhyAddr()),
            static_cast<uint16_t>(curRow_), static_cast<uint16_t>(K_DIM));
        ReleaseInputRows(inputIdxA_);
        ReleaseInputRows(inputIdxB_);
        for (uint32_t srcOffset = 0; srcOffset < task.curChunkSize; srcOffset += rowOwned) {
            uint32_t curSrcRow = srcOffset + rowOwned > task.curChunkSize ? task.curChunkSize - srcOffset : rowOwned;
            inputIdxA_ = CopyInRows<kType>(gmD_, matrixInputBuf_[curInputPingPong_], srcOffset * CHUNK_SIZE,
                                                  curSrcRow * CHUNK_SIZE);
            inputIdxB_ = CopyInRows<kType>(gmKKT_, matrixInputBuf_[curInputPingPong_], srcOffset * CHUNK_SIZE,
                                                  curSrcRow * CHUNK_SIZE);
            WaitInputRows(inputIdxA_);
            WaitInputRows(inputIdxB_);
            if (srcOffset == 0) {
                PrepareWyReprBwdDkktColDgRegbase<kType, true>(
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(scaleFp32Tensor_.GetPhyAddr()),
                    (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxA_].GetPhyAddr()),
                    (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxB_].GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_.GetPhyAddr()),
                    static_cast<uint16_t>(curSrcRow), static_cast<uint16_t>(CHUNK_SIZE),
                    static_cast<uint16_t>(rowOffset_), static_cast<uint16_t>(curRow_));
            } else {
                PrepareWyReprBwdDkktColDgRegbase<kType, false>(
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(scaleFp32Tensor_.GetPhyAddr()),
                    (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxA_].GetPhyAddr()),
                    (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxB_].GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_.GetPhyAddr()),
                    static_cast<uint16_t>(curSrcRow), static_cast<uint16_t>(CHUNK_SIZE),
                    static_cast<uint16_t>(rowOffset_), static_cast<uint16_t>(curRow_));
            }
            ReleaseInputRows(inputIdxA_);
            ReleaseInputRows(inputIdxB_);
        }
        PrepareWyReprBwdAddFp32RowsRegbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dgAccFp32Tensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dgAccFp32Tensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(scaleFp32Tensor_.GetPhyAddr()),
            static_cast<uint16_t>(rowOffset_), static_cast<uint16_t>(curRow_), static_cast<uint16_t>(CHUNK_SIZE));

        inputIdxC_ =
            CopyInRows<kType>(gmDkbg_, matrixInputBuf_[curInputPingPong_], rowOffset_ * K_DIM, curRow_ * K_DIM);

        WaitInputRows(inputIdxC_);
        PrepareWyReprBwdDkbgSecondRegbase<kType>(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(calcFp32CTensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dbetaAccFp32Tensor_[rowOffset_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dgAccFp32Tensor_[rowOffset_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxC_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(calcFp32ATensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_[rowOffset_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(gExpAllFp32Tensor_[rowOffset_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dbetaAccFp32Tensor_[rowOffset_].GetPhyAddr()),
            static_cast<uint16_t>(curRow_), static_cast<uint16_t>(K_DIM));
        ReleaseInputRows(inputIdxC_);
        PipeBarrier<PIPE_V>();
        inputIdxC_ =
            CopyInRows<kType>(gmDK_, matrixInputBuf_[curInputPingPong_], rowOffset_ * K_DIM, curRow_ * K_DIM);
        if (!isFirstValueHeadInGroup) {
            inputIdxD_ = CopyInRows<kType>(dkTensor_, matrixInputBuf_[curInputPingPong_],
                                                  (keyBase_ + rowOffset_) * K_DIM, curRow_ * K_DIM);
        }

        WaitInputRows(inputIdxC_);
        if (!isFirstValueHeadInGroup) {
            WaitInputRows(inputIdxD_);
        }
        PrepareOutputRows();
        if (!isFirstValueHeadInGroup) {
            PrepareWyReprBwdDkOutputRegbase<kType, true>(
                (__ubuf__ kType *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx_].GetPhyAddr()),
                (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxC_].GetPhyAddr()),
                (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxD_].GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(calcFp32BTensor_.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(calcFp32CTensor_.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_[rowOffset_].GetPhyAddr()),
                static_cast<uint16_t>(curRow_), static_cast<uint16_t>(K_DIM));
            ReleaseInputRows(inputIdxD_);
        } else {
            PrepareWyReprBwdDkOutputRegbase<kType, false>(
                (__ubuf__ kType *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx_].GetPhyAddr()),
                (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxC_].GetPhyAddr()),
                (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxC_].GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(calcFp32BTensor_.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(calcFp32CTensor_.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_[rowOffset_].GetPhyAddr()),
                static_cast<uint16_t>(curRow_), static_cast<uint16_t>(K_DIM));
        }
        ReleaseInputRows(inputIdxC_);
        CopyOutRows(dkTensor_, outputBuf_[outputIdx_], (keyBase_ + rowOffset_) * K_DIM, curRow_ * K_DIM);
    }

    for (rowOffset_ = subBlockIdx_ * rowOwned; rowOffset_ < task.curChunkSize; rowOffset_ += rowOwned * subBlockNum_) {
        curRow_ = rowOffset_ + rowOwned > task.curChunkSize ? task.curChunkSize - rowOffset_ : rowOwned;

        inputIdxA_ =
            CopyInRows<kType>(gmDvb_, matrixInputBuf_[curInputPingPong_], rowOffset_ * V_DIM, curRow_ * V_DIM);
        inputIdxB_ = CopyInRows<kType>(vTensor_, matrixInputBuf_[curInputPingPong_],
                                              (valueBase_ + rowOffset_) * V_DIM, curRow_ * V_DIM);
        WaitInputRows(inputIdxA_);
        WaitInputRows(inputIdxB_);
        PrepareOutputRows();
        PrepareWyReprBwdDvbOutputRegbase<kType>(
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dbetaAccFp32Tensor_[rowOffset_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxA_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxB_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_[rowOffset_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dbetaAccFp32Tensor_[rowOffset_].GetPhyAddr()),
            static_cast<uint16_t>(curRow_), static_cast<uint16_t>(V_DIM));
        ReleaseInputRows(inputIdxA_);
        ReleaseInputRows(inputIdxB_);
        CopyOutRows(dvTensor_, outputBuf_[outputIdx_], (valueBase_ + rowOffset_) * V_DIM, curRow_ * V_DIM);
    }

    for (rowOffset_ = subBlockIdx_ * rowOwned; rowOffset_ < task.curChunkSize; rowOffset_ += rowOwned * subBlockNum_) {
        curRow_ = rowOffset_ + rowOwned > task.curChunkSize ? task.curChunkSize - rowOffset_ : rowOwned;

        inputIdxA_ = CopyInRows<kType>(gmD_, matrixInputBuf_[curInputPingPong_], rowOffset_ * CHUNK_SIZE,
                                              curRow_ * CHUNK_SIZE);
        inputIdxB_ = CopyInRows<kType>(gmKKT_, matrixInputBuf_[curInputPingPong_], rowOffset_ * CHUNK_SIZE,
                                              curRow_ * CHUNK_SIZE);
        WaitInputRows(inputIdxA_);
        WaitInputRows(inputIdxB_);
        PrepareWyReprBwdDkktRowDgRegbase<kType>(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dgAccFp32Tensor_[rowOffset_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxA_].GetPhyAddr()),
            (__ubuf__ kType *)reinterpret_cast<uint64_t>(matrixInputBuf_[inputIdxB_].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaAllFp32Tensor_.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(dgAccFp32Tensor_[rowOffset_].GetPhyAddr()),
            static_cast<uint16_t>(curRow_), static_cast<uint16_t>(CHUNK_SIZE),
            static_cast<uint16_t>(task.curChunkSize));
        ReleaseInputRows(inputIdxA_);
        ReleaseInputRows(inputIdxB_);
        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
    }

    for (rowOffset_ = subBlockIdx_ * rowOwned; rowOffset_ < task.curChunkSize; rowOffset_ += rowOwned * subBlockNum_) {
        curRow_ = rowOffset_ + rowOwned > task.curChunkSize ? task.curChunkSize - rowOffset_ : rowOwned;
        CopyOutBetaGRows(dbetaTensor_, dbetaAccFp32Tensor_[rowOffset_], valueBase_ + rowOffset_, curRow_);
        CopyOutBetaGRows(dgTensor_, dgAccFp32Tensor_[rowOffset_], valueBase_ + rowOffset_, curRow_);
    }
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdVectorProcess<kType, gType, V_DIM, CHUNK_SIZE>::Process()
{
    uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    uint32_t coreNum = AscendC::GetBlockNum();
    uint64_t groupSize = PrepareWyReprBwdGetGroupSize(tiling_);
    uint64_t windowIdx = 0;

    for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(tiling_.chunkNum); taskIdx += coreNum) {
        PrepareWyReprBwdTaskInfo task;
        PrepareWyReprBwdGetTaskInfo(cuSeqlens_, chunkIndices_, tiling_, taskIdx, task);
        for (uint32_t slot = 0; slot < BUFFER_COUNT_4; ++slot) {
            kktSlotForSlot_[slot] = 0;
        }
        uint64_t hvTotal = static_cast<uint64_t>(tiling_.HV);
        for (uint64_t hvBase = 0; hvBase < hvTotal; hvBase += BUFFER_COUNT_2) {
            uint32_t headCnt = hvBase + BUFFER_COUNT_2 <= hvTotal ?
                                   BUFFER_COUNT_2 :
                                   static_cast<uint32_t>(hvTotal - hvBase);
            uint32_t windowStartSlot = static_cast<uint32_t>((windowIdx & 1U) * BUFFER_COUNT_2);
            nextKktSlot_ = windowStartSlot;
            cachedKktSlot_ = windowStartSlot;
            cachedKktHk_ = static_cast<uint64_t>(-1);

            // Stage0 fills both workspace slots before later stages consume the first head.
            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                uint64_t hk = hv / groupSize;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                if (cachedKktHk_ != hk) {
                    cachedKktHk_ = hk;
                    cachedKktSlot_ = nextKktSlot_;
                    ++nextKktSlot_;
                }
                kktSlotForSlot_[curSlot_] = cachedKktSlot_;
                ProcessVectorTask(task, hv, hk, slotBase);
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE
                Arch::CrossCoreWaitFlag(cubeToVecFlag_);
#endif

                curSlot_ ^= 1U;
            }

#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR || PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE
            ++windowIdx;
            continue;
#endif

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                ProcessDa4Task(task, slotBase);
                curSlot_ ^= 1U;
            }

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                ProcessDTask(task, hv, slotBase);
                curSlot_ ^= 1U;
            }

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                uint64_t hk = hv / groupSize;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                ProcessOutputTask(task, hv, hk, groupSize, slotBase);
                curSlot_ ^= 1U;
            }
            ++windowIdx;
        }
    }
    ReleaseVectorEvents();
}

#endif // PREPARE_WY_REPR_BWD_VECTOR_H
