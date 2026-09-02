/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_C_ARCH35_INTRA_REGBASE_H
#define CHUNK_KDA_BWD_C_ARCH35_INTRA_REGBASE_H
#include "kernel_utils/vector/regbase.hpp"

namespace KDA {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t kKdaRegbaseFp32Elements =
    AscendC::VECTOR_REG_WIDTH / sizeof(float);
constexpr CastTrait kKdaRegbaseBf16ToFp32 = {
    RegLayout::ZERO,
    SatMode::SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

static __simd_vf__ inline void KdaRegbaseCopy(
    __ubuf__ float *dst, __ubuf__ float *src, uint16_t count)
{
    RegTensor<float> value;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += kKdaRegbaseFp32Elements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(value, src + offset);
        DataCopy(dst + offset, value, mask);
    }
}

static __simd_vf__ inline void KdaRegbaseFill(
    __ubuf__ float *dst, float value, uint16_t count)
{
    RegTensor<float> valueReg;
    uint32_t remaining = count;
    Duplicate(valueReg, value);
    for (uint32_t offset = 0; offset < count; offset += kKdaRegbaseFp32Elements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(dst + offset, valueReg, mask);
    }
}

static __simd_vf__ inline void KdaRegbaseCastBf16ToFp32(
    __ubuf__ float *dst, __ubuf__ bfloat16_t *src, uint16_t count)
{
    RegTensor<bfloat16_t> srcReg;
    RegTensor<float> dstReg;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += kKdaRegbaseFp32Elements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcReg, src + offset);
        Cast<float, bfloat16_t, kKdaRegbaseBf16ToFp32>(
            dstReg, srcReg, mask);
        DataCopy(dst + offset, dstReg, mask);
    }
}

static __simd_vf__ inline void KdaRegbaseExp2(
    __ubuf__ float *dst, __ubuf__ float *src, uint16_t count)
{
    RegTensor<float> value;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += kKdaRegbaseFp32Elements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(value, src + offset);
        Muls(value, value, kLn2, mask);
        Exp(value, value, mask);
        DataCopy(dst + offset, value, mask);
    }
}

static __simd_vf__ inline void KdaRegbaseGatherScalars(
    __ubuf__ float *dst, __ubuf__ float *src, uint16_t rows,
    uint16_t srcRowElements)
{
    RegTensor<float> value;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(
            value, src + row * srcRowElements);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dst + row, value, fullMask);
    }
}

static __simd_vf__ inline void KdaRegbaseScatterScalars(
    __ubuf__ float *dst, __ubuf__ float *src, uint16_t rows,
    uint16_t dstRowElements)
{
    RegTensor<float> value;
    RegTensor<float> zero;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zero, 0.0f);
    for (uint32_t row = 0; row < rows; ++row) {
        uint32_t rowRemaining = dstRowElements;
        MaskReg rowMask = UpdateMask<float>(rowRemaining);
        DataCopy(dst + row * dstRowElements, zero, rowMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(value, src + row);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dst + row * dstRowElements, value, fullMask);
    }
}

static __simd_vf__ inline void KdaRegbaseMaskLowerA(
    __ubuf__ float *dst, __ubuf__ float *src, uint16_t validRows,
    uint16_t rowStart, uint16_t prefix, uint16_t rowBlock,
    uint16_t includeDiagonal)
{
    RegTensor<float> value;
    RegTensor<float> zero;
    Duplicate(zero, 0.0f);
    for (uint32_t row = 0; row < rowBlock; ++row) {
        const uint32_t validCols =
            row < validRows ? rowStart + row + includeDiagonal : 0;
        uint32_t remaining = prefix;
        for (uint32_t col = 0; col < prefix; col += kKdaRegbaseFp32Elements) {
            MaskReg storeMask = UpdateMask<float>(remaining);
            StoreAlign(dst + row * prefix + col, zero, storeMask);
        }
        uint32_t copyRemaining = validCols;
        for (uint32_t col = 0; col < validCols; col += kKdaRegbaseFp32Elements) {
            MaskReg copyMask = UpdateMask<float>(copyRemaining);
            LoadAlign(value, src + row * prefix + col);
            StoreAlign(dst + row * prefix + col, value, copyMask);
        }
    }
}

static __simd_vf__ inline void KdaRegbaseMaskUpperA(
    __ubuf__ float *dst, __ubuf__ float *src, uint16_t future,
    uint16_t validRows, uint16_t rowBlock, uint16_t includeDiagonal)
{
    RegTensor<float> value;
    RegTensor<float> zero;
    Duplicate(zero, 0.0f);
    for (uint32_t row = 0; row < future; ++row) {
        const uint32_t validCols =
            row < validRows ? row + includeDiagonal : rowBlock;
        uint32_t fullRowRemaining = rowBlock;
        MaskReg fullRowMask = UpdateMask<float>(fullRowRemaining);
        StoreAlign(dst + row * rowBlock, zero, fullRowMask);
        uint32_t copyRemaining = validCols;
        for (uint32_t col = 0; col < validCols; col += kKdaRegbaseFp32Elements) {
            MaskReg copyMask = UpdateMask<float>(copyRemaining);
            LoadAlign(value, src + row * rowBlock + col);
            StoreAlign(dst + row * rowBlock + col, value, copyMask);
        }
    }
}

template <bool ANCHOR_MINUS_GATE, bool APPLY_BETA>
static __simd_vf__ inline void KdaRegbaseGateScale(
    __ubuf__ float *data, __ubuf__ float *gate, __ubuf__ float *anchor,
    __ubuf__ float *beta, uint16_t rows, uint16_t cols)
{
    RegTensor<float> dataReg;
    RegTensor<float> gateReg;
    RegTensor<float> anchorReg;
    RegTensor<float> scaleReg;
    RegTensor<float> betaReg;
    for (uint32_t row = 0; row < rows; ++row) {
        if constexpr (APPLY_BETA) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        }
        uint32_t remaining = cols;
        for (uint32_t col = 0; col < cols; col += kKdaRegbaseFp32Elements) {
            MaskReg mask = UpdateMask<float>(remaining);
            DataCopy(dataReg, data + row * cols + col);
            DataCopy(gateReg, gate + row * cols + col);
            // The anchor is one logical row shared by every row in this
            // tile.  Read it directly instead of materializing 8/16 copies
            // in UB.  Besides removing redundant VEC stores, this avoids
            // depending on a replicated row at the end of an arena plane.
            DataCopy(anchorReg, anchor + col);
            if constexpr (ANCHOR_MINUS_GATE) {
                Sub(scaleReg, anchorReg, gateReg, mask);
            } else {
                Sub(scaleReg, gateReg, anchorReg, mask);
            }
            Muls(scaleReg, scaleReg, kLn2, mask);
            Exp(scaleReg, scaleReg, mask);
            Mul(dataReg, dataReg, scaleReg, mask);
            if constexpr (APPLY_BETA) {
                Mul(dataReg, dataReg, betaReg, mask);
            }
            DataCopy(data + row * cols + col, dataReg, mask);
        }
    }
}

static __simd_vf__ inline void KdaRegbaseGateScalePair(
    __ubuf__ float *qData, __ubuf__ float *kData,
    __ubuf__ float *gate, __ubuf__ float *anchor,
    __ubuf__ float *beta, uint16_t rows, uint16_t cols)
{
    RegTensor<float> qReg;
    RegTensor<float> kReg;
    RegTensor<float> gateReg;
    RegTensor<float> anchorReg;
    RegTensor<float> scaleReg;
    RegTensor<float> betaReg;
    for (uint32_t row = 0; row < rows; ++row) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        uint32_t remaining = cols;
        for (uint32_t col = 0; col < cols;
             col += kKdaRegbaseFp32Elements) {
            MaskReg mask = UpdateMask<float>(remaining);
            DataCopy(qReg, qData + row * cols + col);
            DataCopy(kReg, kData + row * cols + col);
            DataCopy(gateReg, gate + row * cols + col);
            DataCopy(anchorReg, anchor + col);
            Sub(scaleReg, gateReg, anchorReg, mask);
            Muls(scaleReg, scaleReg, kLn2, mask);
            Exp(scaleReg, scaleReg, mask);
            Mul(qReg, qReg, scaleReg, mask);
            Mul(kReg, kReg, scaleReg, mask);
            Mul(kReg, kReg, betaReg, mask);
            DataCopy(qData + row * cols + col, qReg, mask);
            DataCopy(kData + row * cols + col, kReg, mask);
        }
    }
}

template <bool DUAL_ANCHOR = false>
static __simd_vf__ inline void KdaRegbaseGateScaleLowerPair(
    __ubuf__ float *qData, __ubuf__ float *kData,
    __ubuf__ float *lowerData, __ubuf__ float *gate,
    __ubuf__ float *lowerAnchor, __ubuf__ float *upperAnchor,
    __ubuf__ float *beta,
    uint16_t rows, uint16_t cols)
{
    RegTensor<float> qReg;
    RegTensor<float> kReg;
    RegTensor<float> lowerReg;
    RegTensor<float> gateReg;
    RegTensor<float> lowerAnchorReg;
    RegTensor<float> upperAnchorReg;
    RegTensor<float> lowerScaleReg;
    RegTensor<float> upperScaleReg;
    RegTensor<float> betaReg;
    for (uint32_t row = 0; row < rows; ++row) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        uint32_t remaining = cols;
        for (uint32_t col = 0; col < cols;
             col += kKdaRegbaseFp32Elements) {
            MaskReg mask = UpdateMask<float>(remaining);
            DataCopy(qReg, qData + row * cols + col);
            DataCopy(kReg, kData + row * cols + col);
            DataCopy(gateReg, gate + row * cols + col);
            DataCopy(lowerAnchorReg, lowerAnchor + col);
            if constexpr (DUAL_ANCHOR) {
                DataCopy(upperAnchorReg, upperAnchor + col);
            }

            Sub(lowerScaleReg, lowerAnchorReg, gateReg, mask);
            Muls(lowerScaleReg, lowerScaleReg, kLn2, mask);
            Exp(lowerScaleReg, lowerScaleReg, mask);
            Mul(lowerReg, kReg, lowerScaleReg, mask);

            if constexpr (DUAL_ANCHOR) {
                Sub(upperScaleReg, gateReg, upperAnchorReg, mask);
            } else {
                Sub(upperScaleReg, gateReg, lowerAnchorReg, mask);
            }
            Muls(upperScaleReg, upperScaleReg, kLn2, mask);
            Exp(upperScaleReg, upperScaleReg, mask);
            Mul(qReg, qReg, upperScaleReg, mask);
            Mul(kReg, kReg, upperScaleReg, mask);
            Mul(kReg, kReg, betaReg, mask);

            DataCopy(lowerData + row * cols + col, lowerReg, mask);
            DataCopy(qData + row * cols + col, qReg, mask);
            DataCopy(kData + row * cols + col, kReg, mask);
        }
    }
}

template <bool FUSE_DQ_BASE, bool DUAL_ANCHOR = false>
static __simd_vf__ inline void KdaRegbaseFinishScale(
    __ubuf__ float *rawDq, __ubuf__ float *rawDkLower,
    __ubuf__ float *rawDkUpper, __ubuf__ float *k,
    __ubuf__ float *gate, __ubuf__ float *lowerAnchor,
    __ubuf__ float *upperAnchor, __ubuf__ float *beta,
    __ubuf__ float *dbAcc, __ubuf__ float *dqBaseRaw,
    __ubuf__ float *anchorExp, __ubuf__ float *dqFinal, float scale,
    uint16_t rows, uint16_t cols)
{
    RegTensor<float> dqReg;
    RegTensor<float> dkLowerReg;
    RegTensor<float> dkUpperReg;
    RegTensor<float> kReg;
    RegTensor<float> gateReg;
    RegTensor<float> lowerAnchorReg;
    RegTensor<float> upperAnchorReg;
    RegTensor<float> posReg;
    RegTensor<float> negReg;
    RegTensor<float> productReg;
    RegTensor<float> productAccReg;
    RegTensor<float> blockSumReg;
    RegTensor<float> betaReg;
    RegTensor<float> dbReg;
    RegTensor<float> dqBaseReg;
    RegTensor<float> anchorExpReg;
    RegTensor<float> dqFinalReg;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();

    for (uint32_t row = 0; row < rows; ++row) {
        Duplicate(productAccReg, 0.0f);
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        uint32_t remaining = cols;
        for (uint32_t col = 0; col < cols; col += kKdaRegbaseFp32Elements) {
            MaskReg mask = UpdateMask<float>(remaining);
            DataCopy(dqReg, rawDq + row * cols + col);
            DataCopy(dkLowerReg, rawDkLower + row * cols + col);
            DataCopy(dkUpperReg, rawDkUpper + row * cols + col);
            DataCopy(kReg, k + row * cols + col);
            DataCopy(gateReg, gate + row * cols + col);
            DataCopy(lowerAnchorReg, lowerAnchor + col);
            if constexpr (DUAL_ANCHOR) {
                DataCopy(upperAnchorReg, upperAnchor + col);
            }

            Sub(posReg, gateReg, lowerAnchorReg, mask);
            if constexpr (DUAL_ANCHOR) {
                Sub(negReg, upperAnchorReg, gateReg, mask);
            } else {
                Sub(negReg, lowerAnchorReg, gateReg, mask);
            }
            Muls(posReg, posReg, kLn2, mask);
            Muls(negReg, negReg, kLn2, mask);
            Exp(posReg, posReg, mask);
            Exp(negReg, negReg, mask);
            Mul(dqReg, dqReg, posReg, mask);
            if constexpr (FUSE_DQ_BASE) {
                DataCopy(dqBaseReg, dqBaseRaw + row * cols + col);
                DataCopy(anchorExpReg, anchorExp + col);
                Mul(dqBaseReg, dqBaseReg, posReg, mask);
                Mul(dqBaseReg, dqBaseReg, anchorExpReg, mask);
                Muls(dqBaseReg, dqBaseReg, scale, mask);
                Add(dqFinalReg, dqReg, dqBaseReg, mask);
                DataCopy(dqFinal + row * cols + col, dqFinalReg, mask);
            }
            Mul(dkLowerReg, dkLowerReg, posReg, mask);
            Mul(dkUpperReg, dkUpperReg, negReg, mask);
            Mul(productReg, dkLowerReg, kReg, mask);
            Add<float, MaskMergeMode::MERGING>(
                productAccReg, productAccReg, productReg, mask);
            Mul(dkLowerReg, dkLowerReg, betaReg, mask);

            DataCopy(rawDq + row * cols + col, dqReg, mask);
            DataCopy(rawDkLower + row * cols + col, dkLowerReg, mask);
            DataCopy(rawDkUpper + row * cols + col, dkUpperReg, mask);
        }
        ReduceSum(blockSumReg, productAccReg, fullMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(dbReg, dbAcc + row);
        Add(dbReg, dbReg, blockSumReg, fullMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dbAcc + row, dbReg, fullMask);
    }
}

static __simd_vf__ inline void KdaRegbaseAdd2(
    __ubuf__ float *dst, __ubuf__ float *lhs, __ubuf__ float *rhs,
    uint16_t count)
{
    RegTensor<float> lhsReg;
    RegTensor<float> rhsReg;
    RegTensor<float> outReg;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += kKdaRegbaseFp32Elements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(lhsReg, lhs + offset);
        DataCopy(rhsReg, rhs + offset);
        Add(outReg, lhsReg, rhsReg, mask);
        DataCopy(dst + offset, outReg, mask);
    }
}

static __simd_vf__ inline void KdaRegbaseAdd3(
    __ubuf__ float *dst, __ubuf__ float *first, __ubuf__ float *second,
    __ubuf__ float *third, uint16_t count)
{
    RegTensor<float> firstReg;
    RegTensor<float> secondReg;
    RegTensor<float> thirdReg;
    RegTensor<float> outReg;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += kKdaRegbaseFp32Elements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(firstReg, first + offset);
        DataCopy(secondReg, second + offset);
        DataCopy(thirdReg, third + offset);
        Add(outReg, secondReg, thirdReg, mask);
        Add(outReg, firstReg, outReg, mask);
        DataCopy(dst + offset, outReg, mask);
    }
}

static __simd_vf__ inline void KdaRegbaseDg(
    __ubuf__ float *dst, __ubuf__ float *inputGrad, __ubuf__ float *q,
    __ubuf__ float *rawDq, __ubuf__ float *k, __ubuf__ float *rawDkLower,
    __ubuf__ float *rawDkUpper, uint16_t count)
{
    RegTensor<float> inputReg;
    RegTensor<float> qReg;
    RegTensor<float> dqReg;
    RegTensor<float> kReg;
    RegTensor<float> lowerReg;
    RegTensor<float> upperReg;
    RegTensor<float> tempReg;
    RegTensor<float> outReg;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += kKdaRegbaseFp32Elements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(inputReg, inputGrad + offset);
        DataCopy(qReg, q + offset);
        DataCopy(dqReg, rawDq + offset);
        DataCopy(kReg, k + offset);
        DataCopy(lowerReg, rawDkLower + offset);
        DataCopy(upperReg, rawDkUpper + offset);
        Mul(outReg, qReg, dqReg, mask);
        Add(outReg, inputReg, outReg, mask);
        Sub(tempReg, lowerReg, upperReg, mask);
        Mul(tempReg, kReg, tempReg, mask);
        Add(outReg, outReg, tempReg, mask);
        DataCopy(dst + offset, outReg, mask);
    }
}

static __simd_vf__ inline void KdaRegbaseDkDg(
    __ubuf__ float *dkDst, __ubuf__ float *dgDst,
    __ubuf__ float *dkInput, __ubuf__ float *dgInput,
    __ubuf__ float *q, __ubuf__ float *rawDq, __ubuf__ float *k,
    __ubuf__ float *rawDkLower, __ubuf__ float *rawDkUpper,
    uint16_t count)
{
    RegTensor<float> dkInputReg;
    RegTensor<float> dgInputReg;
    RegTensor<float> qReg;
    RegTensor<float> dqReg;
    RegTensor<float> kReg;
    RegTensor<float> lowerReg;
    RegTensor<float> upperReg;
    RegTensor<float> dkOutReg;
    RegTensor<float> dgOutReg;
    RegTensor<float> tempReg;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count;
         offset += kKdaRegbaseFp32Elements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(dkInputReg, dkInput + offset);
        DataCopy(dgInputReg, dgInput + offset);
        DataCopy(qReg, q + offset);
        DataCopy(dqReg, rawDq + offset);
        DataCopy(kReg, k + offset);
        DataCopy(lowerReg, rawDkLower + offset);
        DataCopy(upperReg, rawDkUpper + offset);

        Add(dkOutReg, lowerReg, upperReg, mask);
        Add(dkOutReg, dkInputReg, dkOutReg, mask);

        Mul(dgOutReg, qReg, dqReg, mask);
        Add(dgOutReg, dgInputReg, dgOutReg, mask);
        Sub(tempReg, lowerReg, upperReg, mask);
        Mul(tempReg, kReg, tempReg, mask);
        Add(dgOutReg, dgOutReg, tempReg, mask);

        DataCopy(dkDst + offset, dkOutReg, mask);
        DataCopy(dgDst + offset, dgOutReg, mask);
    }
}

// Reverse inclusive scan for one contiguous row segment.  Carry contains the
// suffix accumulated by later segments and is updated with this segment's
// total.  Calling segments in descending token order is bitwise equivalent to
// scanning the complete chunk in one pass while allowing an earlier row block
// to remain resident in UB across the Intra four-slot pipeline.
static __simd_vf__ inline void KdaRegbaseReverseScanCarry(
    __ubuf__ float *data, __ubuf__ float *carry, uint16_t rows)
{
    constexpr uint32_t kCols = 128;
    RegTensor<float> valueReg;
    RegTensor<float> carryReg;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < kCols;
         col += kKdaRegbaseFp32Elements) {
        DataCopy(carryReg, carry + col);
        for (uint32_t row = rows; row > 0; --row) {
            const uint32_t offset = (row - 1U) * kCols + col;
            DataCopy(valueReg, data + offset);
            Add(carryReg, carryReg, valueReg, mask);
            DataCopy(data + offset, carryReg, mask);
        }
        DataCopy(carry + col, carryReg, mask);
    }
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_ARCH35_INTRA_REGBASE_H
