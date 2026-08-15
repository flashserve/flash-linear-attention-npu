#ifndef CHUNK_SCALED_DOT_KKT_ARCH35_VECTOR_H
#define CHUNK_SCALED_DOT_KKT_ARCH35_VECTOR_H

#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"

namespace NsChunkScaledDotKkt {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint16_t KKT_A5_BT64 = 64;

__simd_callee__ inline void LoadKktFloatPair(RegTensor<float> &zero, RegTensor<float> &one, __ubuf__ float *src)
{
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, src);
}

__simd_callee__ inline void StoreKktFloatPair(__ubuf__ float *dst,
                                              RegTensor<float> &zero,
                                              RegTensor<float> &one,
                                              MaskReg &maskF32)
{
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, zero, one, maskF32);
}

__simd_callee__ inline void MaxsKktFloatPair(RegTensor<float> &dstZero,
                                             RegTensor<float> &dstOne,
                                             RegTensor<float> &srcZero,
                                             RegTensor<float> &srcOne,
                                             float scalar,
                                             MaskReg &maskF32)
{
    Maxs(dstZero, srcZero, scalar, maskF32);
    Maxs(dstOne, srcOne, scalar, maskF32);
}

// A5 RegBase epilogue for the profiled BT=64 CATLASS score path:
// out[row, col] = score[row, col] * beta[row] * exp(clip(g[row] - g[col], -50, 50)),
// with columns col >= row or col >= validCols zeroed.
__simd_vf__ inline void ProcessKktEpilogue64VF(__ubuf__ float *out,
                                               __ubuf__ float *score,
                                               __ubuf__ float *g,
                                               __ubuf__ float *beta,
                                               uint16_t rowBegin,
                                               uint16_t rowCount,
                                               uint16_t validCols,
                                               uint16_t subBlockIdx,
                                               uint16_t subBlockNum,
                                               uint16_t rowGroupRows)
{
    RegTensor<float> gZeroReg;
    RegTensor<float> gOneReg;
    RegTensor<float> scoreZeroReg;
    RegTensor<float> scoreOneReg;
    RegTensor<float> gateZeroReg;
    RegTensor<float> gateOneReg;
    RegTensor<float> resultZeroReg;
    RegTensor<float> resultOneReg;
    RegTensor<float> gRowReg;
    RegTensor<float> betaRowReg;
    RegTensor<float> zeroReg;
    RegTensor<float> rowIdxReg;
    RegTensor<float> validColReg;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxZeroReg;
    RegTensor<float> colIdxOneReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskUpperZero;
    MaskReg maskUpperOne;
    MaskReg maskInvalidZero;
    MaskReg maskInvalidOne;

    Duplicate(zeroReg, 0.0f, maskFull32);
    Duplicate(rowIdxReg, 0.0f, maskFull32);
    Duplicate(validColReg, static_cast<float>(validCols), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxZeroReg, colIdxOneReg, colIdxReg, maskFull16);
    LoadKktFloatPair(gZeroReg, gOneReg, g);

    for (uint16_t row = 0; row < KKT_A5_BT64; ++row) {
        LoadKktFloatPair(scoreZeroReg, scoreOneReg, score + row * KKT_A5_BT64);
        LoadIn<float, true>(gRowReg, g + row);
        LoadIn<float, true>(betaRowReg, beta + row);

        SubFloatTwoReg(gateZeroReg, gateOneReg, gRowReg, gRowReg, gZeroReg, gOneReg, maskFull32);
        MaxsKktFloatPair(gateZeroReg, gateOneReg, gateZeroReg, gateOneReg, -50.0f, maskFull32);
        MinsFloatTwoReg(gateZeroReg, gateOneReg, gateZeroReg, gateOneReg, 50.0f, maskFull32);
        ExpFloatTwoReg(gateZeroReg, gateOneReg, gateZeroReg, gateOneReg, maskFull32);
        Mul(gateZeroReg, gateZeroReg, betaRowReg, maskFull32);
        Mul(gateOneReg, gateOneReg, betaRowReg, maskFull32);

        MulFloatTwoReg(resultZeroReg, resultOneReg, scoreZeroReg, scoreOneReg, gateZeroReg, gateOneReg, maskFull32);
        CompareTwoReg<float, CMPMODE::GE>(maskUpperZero, maskUpperOne, colIdxZeroReg, colIdxOneReg, rowIdxReg,
                                          rowIdxReg, maskFull32);
        SelectTwoReg(resultZeroReg, resultOneReg, zeroReg, zeroReg, resultZeroReg, resultOneReg, maskUpperZero,
                     maskUpperOne);
        CompareTwoReg<float, CMPMODE::GE>(maskInvalidZero, maskInvalidOne, colIdxZeroReg, colIdxOneReg, validColReg,
                                          validColReg, maskFull32);
        SelectTwoReg(resultZeroReg, resultOneReg, zeroReg, zeroReg, resultZeroReg, resultOneReg, maskInvalidZero,
                     maskInvalidOne);
        StoreKktFloatPair(out + row * KKT_A5_BT64, resultZeroReg, resultOneReg, maskFull32);
        Adds(rowIdxReg, rowIdxReg, 1.0f, maskFull32);
    }
}

} // namespace NsChunkScaledDotKkt

#endif // CHUNK_SCALED_DOT_KKT_ARCH35_VECTOR_H
