/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
// A6 register formulas, included inside namespace KDA::Arch92.
// All UB pointers are explicit. Load/store helpers bound the active lanes;
// VF loops use constant bounds with predicates for valid rows. Stage5 emits
// FP32 NZ operands; state/WY stages retain BF16 RNE high/low correction.
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH92_REGBASE_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH92_REGBASE_H
using namespace AscendC::MicroAPI;

// Constant VF loop bounds; each caller predicates rows outside the valid tile.
__simd_callee__ inline uint16_t FinalizeLoopBound(uint16_t rows, uint16_t maximum)
{
    (void)rows;
    return maximum;
}

template <typename T, uint32_t ELEMENTS>
__simd_callee__ inline MaskReg FinalizeMask()
{
    uint32_t count = ELEMENTS;
    return UpdateMask<T>(count);
}

template <typename T, bool BROADCAST = false>
__simd_callee__ inline void FinalizeLoadIn(RegTensor<T> &dst, __ubuf__ T *src)
{
    if constexpr (!BROADCAST) {
        uint32_t count = sizeof(T) == sizeof(float) ? 64 : 128;
        MaskReg mask = UpdateMask<T>(count);
        LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY>(dst, src, 1, mask);
    } else {
        LoadIn<T, true>(dst, src);
    }
}

__simd_callee__ inline void FinalizeLoad64(RegTensor<float> &dst, __ubuf__ float *src)
{
    uint32_t count = 64;
    MaskReg mask = UpdateMask<float>(count);
    LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(dst, src, 1, mask);
}

__simd_callee__ inline void FinalizeLoadPair(RegTensor<float> &even, RegTensor<float> &odd, __ubuf__ float *src)
{
    if constexpr (AscendC::VECTOR_REG_WIDTH / sizeof(float) >= 128) {
        RegTensor<float> row;
        uint32_t count = 128;
        MaskReg mask = UpdateMask<float>(count);
        LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(row, src, 1, mask);
        DeInterleave(even, odd, row, row);
    } else {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(even, odd, src);
    }
}

__simd_callee__ inline void FinalizeStorePair(__ubuf__ float *dst, RegTensor<float> &even, RegTensor<float> &odd,
                                              MaskReg &mask)
{
    if constexpr (AscendC::VECTOR_REG_WIDTH / sizeof(float) >= 128) {
        RegTensor<float> row, unused;
        Interleave(row, unused, even, odd);
        uint32_t count = 128;
        MaskReg rowMask = UpdateMask<float>(count);
        StoreAlign(dst, row, rowMask);
    } else {
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, even, odd, mask);
    }
}


// Retain small intra-chunk gradients through Cube accumulation. Undo the
// power-of-two scale only after restoring the per-band exponent factor.
constexpr float KDA_FINALIZE_LOCAL_SCALE = 65536.0f;
constexpr float KDA_FINALIZE_LN2 = 0.6931471805599453f;
constexpr CastTrait KDA_FINALIZE_FP32_TO_BF16_RNE = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING,
                                                     AscendC::RoundMode::CAST_RINT};
constexpr CastTrait KDA_FINALIZE_FP32_TO_BF16_RNE_ONE = {RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};

__simd_callee__ inline void FinalizeCastBf16(RegTensor<bfloat16_t> &dst, RegTensor<float> &even, RegTensor<float> &odd,
                                             MaskReg &mask)
{
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(dst, odd, mask);
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(dst, even, mask);
}

// Cast ZERO places one BF16 value in each 32-bit lane. Pack the 64
// values in registers, then scatter them into a 64-row NZ plane.
__simd_callee__ inline void FinalizeStoreNz64(__ubuf__ bfloat16_t *dst, RegTensor<bfloat16_t> &packed,
                                              RegTensor<uint16_t> &nzIndex)
{
    RegTensor<bfloat16_t> values, unused;
    DeInterleave(values, unused, packed, packed);
    uint32_t count = 64;
    MaskReg mask = UpdateMask<bfloat16_t>(count);
    Scatter(dst, values, nzIndex, mask);
}

__simd_callee__ inline void FinalizeZeroNzTail(__ubuf__ bfloat16_t *dst, uint16_t validRows, uint16_t cols)
{
    uint32_t count = cols;
    MaskReg mask = UpdateMask<bfloat16_t>(count);
    RegTensor<bfloat16_t> zero;
    Duplicate(zero, static_cast<bfloat16_t>(0), mask);
    RegTensor<uint16_t> column, block, nzIndex;
    MaskReg indexMask = FinalizeMask<half, 128>();
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), indexMask);
    Muls(block, block, uint16_t(1008), indexMask);
    Add(nzIndex, column, block, indexMask);
    for (uint16_t row = validRows; row < KDA_FINALIZE_CHUNK; ++row) {
        Scatter(dst + row * 16, zero, nzIndex, mask);
    }
}

__simd_callee__ inline void FinalizeLocalNzIndex(RegTensor<uint32_t> &index, MaskReg &mask)
{
    RegTensor<uint32_t> column, block;
    Arange(reinterpret_cast<RegTensor<int32_t> &>(column), int32_t(0));
    ShiftRights(block, column, int16_t(3), mask);
    Muls(block, block, uint32_t(504), mask);
    Add(index, column, block, mask);
}

__simd_callee__ inline void FinalizeStoreLocalPair(__ubuf__ float *dst, RegTensor<float> &even, RegTensor<float> &odd,
                                                   MaskReg &mask, RegTensor<uint32_t> &index)
{
    // Only 64 lanes of even/odd are active. Scatter channels 2*i and 2*i+1
    // directly: the wider A6 Interleave would put both halves in one register.
    (void)index;
    RegTensor<uint32_t> column, block, pairIndex;
    Arange(reinterpret_cast<RegTensor<int32_t> &>(column), int32_t(0));
    Muls(column, column, uint32_t(2), mask);
    ShiftRights(block, column, int16_t(3), mask);
    Muls(block, block, uint32_t(504), mask);
    Add(pairIndex, column, block, mask);
    Scatter(dst, even, pairIndex, mask);
    Adds(pairIndex, pairIndex, uint32_t(1), mask);
    Scatter(dst, odd, pairIndex, mask);
}

__simd_vf__ inline void FinalizeStage5TailVF(__ubuf__ float *daq, __ubuf__ float *dak, __ubuf__ float *kn,
                                             __ubuf__ float *qp, __ubuf__ float *bp, uint16_t validRows)
{
    MaskReg mask = FinalizeMask<float, 64>();
    RegTensor<uint32_t> index;
    FinalizeLocalNzIndex(index, mask);
    RegTensor<float> zero;
    Duplicate(zero, 0.0f, mask);
    const uint16_t paddedRows = (validRows + 7U) / 8U * 8U;
    for (uint16_t tail = 0; tail < FinalizeLoopBound(paddedRows - validRows, 7); ++tail) {
        const uint16_t row = validRows + tail;
        if (row >= paddedRows)
            continue;
        Scatter(daq + row * 8, zero, index, mask);
        Scatter(dak + row * 8, zero, index, mask);
        FinalizeStoreLocalPair(kn + row * 8, zero, zero, mask, index);
        FinalizeStoreLocalPair(qp + row * 8, zero, zero, mask, index);
        FinalizeStoreLocalPair(bp + row * 8, zero, zero, mask, index);
    }
}

// Stage0 emits paired NZ kE for contiguous UB-to-L1 copies. The raw log gates
// stay resident in the g frame; consumers recompute Exp(g*LN2) inline.
__simd_vf__ inline void FinalizeStage0VF(__ubuf__ bfloat16_t *kENd, __ubuf__ bfloat16_t *lowNd, __ubuf__ float *rH,
                                         __ubuf__ bfloat16_t *k, __ubuf__ float *gk, __ubuf__ bfloat16_t *h,
                                         __ubuf__ bfloat16_t *dh, uint16_t validRows)
{
    MaskReg fpMask = FinalizeMask<float, 64>();
    MaskReg bfMask = FinalizeMask<half, 128>();
    RegTensor<float> g0;
    RegTensor<float> g1;
    RegTensor<float> e0;
    RegTensor<float> e1;
    RegTensor<float> k0;
    RegTensor<float> k1;
    RegTensor<bfloat16_t> kb;
    RegTensor<bfloat16_t> out;

    RegTensor<uint16_t> column, block, nzIndex;
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), bfMask);
    Muls(block, block, uint16_t(1008), bfMask);
    Add(nzIndex, column, block, bfMask);

    // The kE plane rows and the rH reduction rows touch disjoint UB ranges.
    // Fuse both loops so the rH loads and multiplies fill the kE exp/scatter
    // latency; two rH rows per iteration cover the full K dimension.
    for (uint16_t row = 0; row < FinalizeLoopBound(validRows, 64); ++row) {
        if (row >= validRows)
            continue;
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        FinalizeLoadPair(g0, g1, gk + rowOffset);
        Muls(e0, g0, KDA_FINALIZE_LN2, fpMask);
        Muls(e1, g1, KDA_FINALIZE_LN2, fpMask);
        Exp(e0, e0, fpMask);
        Exp(e1, e1, fpMask);

        FinalizeLoadIn<bfloat16_t, false>(kb, k + rowOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        Mul(k0, k0, e0, fpMask);
        Mul(k1, k1, e1, fpMask);
        FinalizeCastBf16(out, k0, k1, fpMask);
        Scatter(kENd + row * 16, out, nzIndex, bfMask);
        RegTensor<float> high0, high1;
        CastHalf2Float<bfloat16_t>(high0, high1, out, bfMask);
        Sub(high0, k0, high0, fpMask);
        Sub(high1, k1, high1, fpMask);
        FinalizeCastBf16(out, high0, high1, fpMask);
        Scatter(lowNd + row * 16, out, nzIndex, bfMask);

        // r_h[k] = sum_v h[k,v] * dh[k,v].
        const uint32_t rhOffset = 2 * rowOffset;
        RegTensor<bfloat16_t> hb;
        RegTensor<bfloat16_t> dhb;
        RegTensor<float> h0;
        RegTensor<float> h1;
        RegTensor<float> dh0;
        RegTensor<float> dh1;
        RegTensor<float> product;
        RegTensor<float> sum;
        FinalizeLoadIn<bfloat16_t, false>(hb, h + rhOffset);
        FinalizeLoadIn<bfloat16_t, false>(dhb, dh + rhOffset);
        CastHalf2Float<bfloat16_t>(h0, h1, hb, bfMask);
        CastHalf2Float<bfloat16_t>(dh0, dh1, dhb, bfMask);
        Mul(h0, h0, dh0, fpMask);
        Mul(h1, h1, dh1, fpMask);
        Add(product, h0, h1, fpMask);
        ReduceSum(sum, product, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rH + 2 * row, sum, fpMask);
        FinalizeLoadIn<bfloat16_t, false>(hb, h + rhOffset + KDA_FINALIZE_DIM);
        FinalizeLoadIn<bfloat16_t, false>(dhb, dh + rhOffset + KDA_FINALIZE_DIM);
        CastHalf2Float<bfloat16_t>(h0, h1, hb, bfMask);
        CastHalf2Float<bfloat16_t>(dh0, dh1, dhb, bfMask);
        Mul(h0, h0, dh0, fpMask);
        Mul(h1, h1, dh1, fpMask);
        Add(product, h0, h1, fpMask);
        ReduceSum(sum, product, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rH + 2 * row + 1, sum, fpMask);
    }

    // Short chunks leave the rH rows beyond 2 * validRows to this loop.
    for (uint16_t row = static_cast<uint16_t>(2 * validRows); row < KDA_FINALIZE_DIM; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<bfloat16_t> hb;
        RegTensor<bfloat16_t> dhb;
        RegTensor<float> h0;
        RegTensor<float> h1;
        RegTensor<float> dh0;
        RegTensor<float> dh1;
        RegTensor<float> product;
        RegTensor<float> sum;
        FinalizeLoadIn<bfloat16_t, false>(hb, h + rowOffset);
        FinalizeLoadIn<bfloat16_t, false>(dhb, dh + rowOffset);
        CastHalf2Float<bfloat16_t>(h0, h1, hb, bfMask);
        CastHalf2Float<bfloat16_t>(dh0, dh1, dhb, bfMask);
        Mul(h0, h0, dh0, fpMask);
        Mul(h1, h1, dh1, fpMask);
        Add(product, h0, h1, fpMask);
        ReduceSum(sum, product, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rH + row, sum, fpMask);
    }

    if (validRows < KDA_FINALIZE_CHUNK) {
        Duplicate(out, static_cast<bfloat16_t>(0), bfMask);
        for (uint16_t row = validRows; row < KDA_FINALIZE_CHUNK; ++row) {
            Scatter(kENd + row * 16, out, nzIndex, bfMask);
            Scatter(lowNd + row * 16, out, nzIndex, bfMask);
        }
    }
}

// Stage2 BuildZ.  zV/zW are direct FP32 FixPipe results in row-major UB.
// Paired Zb is produced in NZ order before the contiguous L1 publication.
template <typename BetaT>
__simd_vf__ inline void FinalizeStage2VF(__ubuf__ bfloat16_t *zbNd, __ubuf__ float *zV, __ubuf__ float *zW,
                                         __ubuf__ BetaT *beta, uint16_t validRows)
{
    MaskReg fpMask = FinalizeMask<float, 64>();
    MaskReg bfMask = FinalizeMask<half, 128>();
    RegTensor<bfloat16_t> betaBf;
    RegTensor<float> betaFp;
    if constexpr (AscendC::IsSameType<BetaT, float>::value) {
        FinalizeLoad64(betaFp, beta);
    } else {
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(betaBf, beta);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, fpMask);
    }

    RegTensor<float> zero;
    Duplicate(zero, 0.0f, fpMask);
    uint32_t betaCount = validRows;
    MaskReg betaMask = UpdateMask<float>(betaCount);
    Select(betaFp, betaFp, zero, betaMask);
    RegTensor<uint16_t> column, block, nzIndex;
    MaskReg indexMask = FinalizeMask<half, 128>();
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), indexMask);
    Muls(block, block, uint16_t(1008), indexMask);
    Add(nzIndex, column, block, indexMask);
    // Interleave independent operations from two rows; handle an odd tail separately.
    uint16_t row = 0;
    for (uint16_t pair = 0; pair < FinalizeLoopBound(validRows / 2, 32); ++pair) {
        if (2U * pair + 1U >= validRows)
            continue;
        row = 2U * pair;
        const uint32_t offsetA = static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        const uint32_t offsetB = static_cast<uint32_t>(row + 1U) * KDA_FINALIZE_CHUNK;
        RegTensor<float> zvA, zwA, resultA;
        RegTensor<float> zvB, zwB, resultB;
        RegTensor<bfloat16_t> packedA, packedB;
        FinalizeLoad64(zvA, zV + offsetA);
        FinalizeLoad64(zvB, zV + offsetB);
        FinalizeLoad64(zwA, zW + offsetA);
        FinalizeLoad64(zwB, zW + offsetB);
        Sub(resultA, zvA, zwA, fpMask);
        Sub(resultB, zvB, zwB, fpMask);
        Mul(resultA, resultA, betaFp, fpMask);
        Mul(resultB, resultB, betaFp, fpMask);
        uint32_t lowerCountA = row;
        uint32_t lowerCountB = row + 1U;
        MaskReg lowerA = UpdateMask<float>(lowerCountA);
        MaskReg lowerB = UpdateMask<float>(lowerCountB);
        Select(resultA, resultA, zero, lowerA);
        Select(resultB, resultB, zero, lowerB);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedA, resultA, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, resultB, fpMask);
        FinalizeStoreNz64(zbNd + row * 16, packedA, nzIndex);
        FinalizeStoreNz64(zbNd + (row + 1U) * 16, packedB, nzIndex);
        RegTensor<float> highA, highB;
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(highA, packedA, fpMask);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(highB, packedB, fpMask);
        Sub(highA, resultA, highA, fpMask);
        Sub(highB, resultB, highB, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedA, highA, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, highB, fpMask);
        FinalizeStoreNz64(zbNd + 4096 + row * 16, packedA, nzIndex);
        FinalizeStoreNz64(zbNd + 4096 + (row + 1U) * 16, packedB, nzIndex);
        row += 2U;
    }
    if (row < validRows) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        RegTensor<float> zv;
        RegTensor<float> zw;
        RegTensor<float> result;
        RegTensor<bfloat16_t> packed;
        FinalizeLoad64(zv, zV + rowOffset);
        FinalizeLoad64(zw, zW + rowOffset);
        Sub(result, zv, zw, fpMask);
        Mul(result, result, betaFp, fpMask);
        uint32_t lowerCount = row;
        MaskReg lower = UpdateMask<float>(lowerCount);
        Select(result, result, zero, lower);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, result, fpMask);
        FinalizeStoreNz64(zbNd + row * 16, packed, nzIndex);
        RegTensor<float> high;
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(high, packed, fpMask);
        Sub(high, result, high, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, high, fpMask);
        FinalizeStoreNz64(zbNd + 4096 + row * 16, packed, nzIndex);
    }
    if (validRows < KDA_FINALIZE_CHUNK) {
        FinalizeZeroNzTail(zbNd, validRows, KDA_FINALIZE_CHUNK);
        FinalizeZeroNzTail(zbNd + 4096, validRows, KDA_FINALIZE_CHUNK);
    }
}

// Stage3 StatePre.  One VF invocation consumes one 32-row band slice of a
// head/chunk.  The two 64-lane FP32 register halves cover K/V=128 without a
// second pass.  gk is the resident raw log-gate frame; exp2(g) factors are
// recomputed inline with the same Muls(LN2)/Exp sequence Stage0 used to
// materialize.  gateState accumulates across bands through its persistent
// slot in the original row order; the rH term is added by the last band only.
template <typename BetaT>
__simd_vf__ inline void FinalizeStage3VF(__ubuf__ float *dkState, __ubuf__ float *dvb, __ubuf__ bfloat16_t *dv,
                                         __ubuf__ float *gateState, __ubuf__ float *dbV, __ubuf__ float *dqRaw,
                                         __ubuf__ float *gk, __ubuf__ bfloat16_t *k, __ubuf__ bfloat16_t *v,
                                         __ubuf__ BetaT *beta, __ubuf__ float *gkLast, __ubuf__ float *rH, float scale,
                                         uint16_t validRows, uint16_t rowBegin, uint16_t rows)
{
    MaskReg fpMask = FinalizeMask<float, 64>();
    MaskReg bfMask = FinalizeMask<half, 128>();
    RegTensor<float> last0;
    RegTensor<float> last1;
    RegTensor<float> gate0;
    RegTensor<float> gate1;
    FinalizeLoadPair(last0, last1, gkLast);
    if (rowBegin == 0) {
        Duplicate(gate0, 0.0f, fpMask);
        Duplicate(gate1, 0.0f, fpMask);
    } else {
        FinalizeLoadPair(gate0, gate1, gateState);
    }

    for (uint16_t row = 0; row < FinalizeLoopBound(rows, 32); ++row) {
        if (row >= rows)
            continue;
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> g0;
        RegTensor<float> g1;
        RegTensor<float> decay0;
        RegTensor<float> decay1;
        RegTensor<float> state0;
        RegTensor<float> state1;
        FinalizeLoadPair(g0, g1, gk + rowOffset);
        Sub(decay0, last0, g0, fpMask);
        Sub(decay1, last1, g1, fpMask);
        Muls(decay0, decay0, KDA_FINALIZE_LN2, fpMask);
        Muls(decay1, decay1, KDA_FINALIZE_LN2, fpMask);
        Exp(decay0, decay0, fpMask);
        Exp(decay1, decay1, fpMask);
        FinalizeLoadPair(state0, state1, dkState + rowOffset);
        Mul(state0, state0, decay0, fpMask);
        Mul(state1, state1, decay1, fpMask);
        FinalizeStorePair(dkState + rowOffset, state0, state1, fpMask);

        RegTensor<bfloat16_t> kb;
        RegTensor<float> k0;
        RegTensor<float> k1;
        FinalizeLoadIn<bfloat16_t, false>(kb, k + rowOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        Mul(k0, k0, state0, fpMask);
        Mul(k1, k1, state1, fpMask);
        Add(gate0, gate0, k0, fpMask);
        Add(gate1, gate1, k1, fpMask);

        RegTensor<float> dvb0;
        RegTensor<float> dvb1;
        FinalizeLoadPair(dvb0, dvb1, dvb + rowOffset);
        RegTensor<bfloat16_t> betaBf;
        RegTensor<float> betaFp;
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            FinalizeLoadIn<float, true>(betaFp, beta + row);
        } else {
            FinalizeLoadIn<bfloat16_t, true>(betaBf, beta + row);
            Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, bfMask);
        }
        RegTensor<float> dv0;
        RegTensor<float> dv1;
        Mul(dv0, dvb0, betaFp, fpMask);
        Mul(dv1, dvb1, betaFp, fpMask);
        RegTensor<bfloat16_t> dvBf;
        FinalizeCastBf16(dvBf, dv0, dv1, fpMask);
        StoreAlign(dv + rowOffset, dvBf, bfMask);

        RegTensor<bfloat16_t> vb;
        RegTensor<float> v0;
        RegTensor<float> v1;
        RegTensor<float> product;
        RegTensor<float> sum;
        FinalizeLoadIn<bfloat16_t, false>(vb, v + rowOffset);
        CastHalf2Float<bfloat16_t>(v0, v1, vb, bfMask);
        Mul(v0, v0, dvb0, fpMask);
        Mul(v1, v1, dvb1, fpMask);
        Add(product, v0, v1, fpMask);
        ReduceSum(sum, product, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbV + row, sum, fpMask);

        RegTensor<float> dq0;
        RegTensor<float> dq1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        Muls(exp0, g0, KDA_FINALIZE_LN2, fpMask);
        Muls(exp1, g1, KDA_FINALIZE_LN2, fpMask);
        Exp(exp0, exp0, fpMask);
        Exp(exp1, exp1, fpMask);
        FinalizeLoadPair(dq0, dq1, dqRaw + rowOffset);
        Mul(dq0, dq0, exp0, fpMask);
        Mul(dq1, dq1, exp1, fpMask);
        Muls(dq0, dq0, scale, fpMask);
        Muls(dq1, dq1, scale, fpMask);
        // dqRaw's row is already consumed into registers; dqBase takes its
        // place in the caller-provided shared band buffer.
        FinalizeStorePair(dqRaw + rowOffset, dq0, dq1, fpMask);
    }

    // The rH term enters the gate accumulation exactly once, after the last
    // band's rows, keeping the original FP add order across bands.
    if (rowBegin + rows == validRows) {
        RegTensor<float> lastExp0;
        RegTensor<float> lastExp1;
        RegTensor<float> rh0;
        RegTensor<float> rh1;
        Muls(lastExp0, last0, KDA_FINALIZE_LN2, fpMask);
        Muls(lastExp1, last1, KDA_FINALIZE_LN2, fpMask);
        Exp(lastExp0, lastExp0, fpMask);
        Exp(lastExp1, lastExp1, fpMask);
        FinalizeLoadPair(rh0, rh1, rH);
        Mul(rh0, rh0, lastExp0, fpMask);
        Mul(rh1, rh1, lastExp1, fpMask);
        Add(gate0, gate0, rh0, fpMask);
        Add(gate1, gate1, rh1, fpMask);
    }
    FinalizeStorePair(gateState, gate0, gate1, fpMask);
}

// Stage4 BaseFinalize, one 32-row band slice per invocation.  dkState is
// updated in place to dkBase and dbV to dbBase; dg is written straight to
// its per-head resident frame (band offset applied by the caller).  dqBase
// and the resident raw g frame stay live for later stages; the exp2(g)
// factor is recomputed inline from g.  gateState joins dg only on the
// chunk's global last row.
template <typename BetaT>
__simd_vf__ inline void FinalizeStage4VF(__ubuf__ float *dkState, __ubuf__ float *dqBase, __ubuf__ float *dKgbRaw,
                                         __ubuf__ float *dg, __ubuf__ float *g, __ubuf__ bfloat16_t *q,
                                         __ubuf__ bfloat16_t *k, __ubuf__ BetaT *beta, __ubuf__ float *gateState,
                                         __ubuf__ float *dbV, uint16_t validRows, uint16_t rowBegin, uint16_t rows)
{
    MaskReg fpMask = FinalizeMask<float, 64>();
    MaskReg bfMask = FinalizeMask<half, 128>();
    RegTensor<float> gate0;
    RegTensor<float> gate1;
    FinalizeLoadPair(gate0, gate1, gateState);

    for (uint16_t row = 0; row < FinalizeLoopBound(rows, 32); ++row) {
        if (row >= rows)
            continue;
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> state0;
        RegTensor<float> state1;
        RegTensor<float> dq0;
        RegTensor<float> dq1;
        RegTensor<float> dkg0;
        RegTensor<float> dkg1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        FinalizeLoadPair(state0, state1, dkState + rowOffset);
        FinalizeLoadPair(dq0, dq1, dqBase + rowOffset);
        FinalizeLoadPair(dkg0, dkg1, dKgbRaw + rowOffset);
        FinalizeLoadPair(exp0, exp1, g + rowOffset);
        Muls(exp0, exp0, KDA_FINALIZE_LN2, fpMask);
        Muls(exp1, exp1, KDA_FINALIZE_LN2, fpMask);
        Exp(exp0, exp0, fpMask);
        Exp(exp1, exp1, fpMask);

        RegTensor<bfloat16_t> betaBf;
        RegTensor<float> betaFp;
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            FinalizeLoadIn<float, true>(betaFp, beta + row);
        } else {
            FinalizeLoadIn<bfloat16_t, true>(betaBf, beta + row);
            Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, bfMask);
        }

        RegTensor<float> dkTerm0;
        RegTensor<float> dkTerm1;
        RegTensor<float> dkgExp0;
        RegTensor<float> dkgExp1;
        Mul(dkgExp0, exp0, dkg0, fpMask);
        Mul(dkgExp1, exp1, dkg1, fpMask);
        Mul(dkTerm0, dkgExp0, betaFp, fpMask);
        Mul(dkTerm1, dkgExp1, betaFp, fpMask);
        Sub(dkTerm0, state0, dkTerm0, fpMask);
        Sub(dkTerm1, state1, dkTerm1, fpMask);
        FinalizeStorePair(dkState + rowOffset, dkTerm0, dkTerm1, fpMask);

        RegTensor<bfloat16_t> qBf;
        RegTensor<bfloat16_t> kBf;
        RegTensor<float> q0;
        RegTensor<float> q1;
        RegTensor<float> k0;
        RegTensor<float> k1;
        FinalizeLoadIn<bfloat16_t, false>(qBf, q + rowOffset);
        FinalizeLoadIn<bfloat16_t, false>(kBf, k + rowOffset);
        CastHalf2Float<bfloat16_t>(q0, q1, qBf, bfMask);
        CastHalf2Float<bfloat16_t>(k0, k1, kBf, bfMask);

        RegTensor<float> dg0;
        RegTensor<float> dg1;
        RegTensor<float> stateK0;
        RegTensor<float> stateK1;
        RegTensor<float> dkgKe0;
        RegTensor<float> dkgKe1;
        Mul(dg0, q0, dq0, fpMask);
        Mul(dg1, q1, dq1, fpMask);
        Mul(stateK0, k0, state0, fpMask);
        Mul(stateK1, k1, state1, fpMask);
        Sub(dg0, dg0, stateK0, fpMask);
        Sub(dg1, dg1, stateK1, fpMask);
        // Vector gradients must not consume Cube's rounded BF16 kE.
        // Reuse exp*dKgb from dkBase instead of recomputing or storing kE.
        Mul(dkgKe0, dkgExp0, k0, fpMask);
        Mul(dkgKe1, dkgExp1, k1, fpMask);

        RegTensor<float> reduceInput;
        RegTensor<float> reduceSum;
        RegTensor<float> dbBase;
        Add(reduceInput, dkgKe0, dkgKe1, fpMask);
        ReduceSum(reduceSum, reduceInput, fpMask);
        FinalizeLoadIn<float, true>(dbBase, dbV + row);
        Sub(dbBase, dbBase, reduceSum, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbV + row, dbBase, fpMask);

        Mul(dkgKe0, dkgKe0, betaFp, fpMask);
        Mul(dkgKe1, dkgKe1, betaFp, fpMask);
        Sub(dg0, dg0, dkgKe0, fpMask);
        Sub(dg1, dg1, dkgKe1, fpMask);
        if (rowBegin + row + 1U == validRows) {
            Add(dg0, dg0, gate0, fpMask);
            Add(dg1, dg1, gate1, fpMask);
        }
        FinalizeStorePair(dg + rowOffset, dg0, dg1, fpMask);
    }
}

// Stage5 rebases one output band. The FP32 operands are
// produced directly in NZ order for contiguous UB-to-L1 copies.
// The supported safe-gate contract supplies a chunk cumsum of nonpositive
// gate steps. Each feature is nonincreasing, so its endpoints are exactly
// the max/min previously obtained by scanning every row of this band.
__simd_vf__ inline void FinalizeIntraCenterVF(__ubuf__ float *g, __ubuf__ float *center, uint16_t rows)
{
    MaskReg mask = FinalizeMask<float, 64>();
    for (uint16_t half = 0; half < 2; ++half) {
        RegTensor<float> first, last, value;
        FinalizeLoad64(first, g + half * 64);
        FinalizeLoad64(last, g + (rows - 1U) * 128 + half * 64);
        Add(value, first, last, mask);
        Muls(value, value, 0.5f, mask);
        StoreAlign(center + half * 64, value, mask);
    }
}



template <typename BetaT>
__simd_vf__ inline void FinalizeStage5VF(__ubuf__ FinalizeLocalType *dAkkNd, __ubuf__ FinalizeLocalType *kNegNd,
                                         __ubuf__ FinalizeLocalType *qPosNd, __ubuf__ FinalizeLocalType *bkPosNd,
                                         __ubuf__ float *dAqkOut, __ubuf__ float *dAqk, __ubuf__ float *diagonal,
                                         __ubuf__ float *dAkkRaw, __ubuf__ bfloat16_t *q, __ubuf__ bfloat16_t *k,
                                         __ubuf__ float *g, __ubuf__ BetaT *beta, __ubuf__ float *center,
                                         uint16_t validRows, uint16_t rowBegin, uint16_t rowEnd)
{
    MaskReg fpMask = FinalizeMask<float, 64>();
    MaskReg bfMask = FinalizeMask<half, 128>();
    RegTensor<float> zero;
    Duplicate(zero, 0.0f, fpMask);

    RegTensor<float> center0, center1;
    FinalizeLoadPair(center0, center1, center);
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    RegTensor<uint32_t> nzIndex;
    FinalizeLocalNzIndex(nzIndex, fpMask);
    for (uint16_t row = 0; row < FinalizeLoopBound(validRows, 64); ++row) {
        if (row >= validRows)
            continue;
        const uint32_t matrixOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        if (rowBegin == 0) {
            RegTensor<float> raw0;
            FinalizeLoad64(raw0, dAkkRaw + matrixOffset);
            Muls(raw0, raw0, -KDA_FINALIZE_LOCAL_SCALE, fpMask);
            uint32_t lowerCount = row;
            MaskReg lower = UpdateMask<float>(lowerCount);
            Select(raw0, raw0, zero, lower);
            Scatter(dAkkNd + row * 8, raw0, nzIndex, fpMask);
            // The diagonal contributes to dq/dk but cancels in the gate derivative.
            RegTensor<float> diag, value;
            DataCopy<float, LoadDist::DIST_BRC_B32>(diag, dAqk + matrixOffset + row);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(diagonal + row, diag, one);
            FinalizeLoad64(value, dAqk + matrixOffset);
            Select(value, value, zero, lower);
            Muls(value, value, KDA_FINALIZE_LOCAL_SCALE, fpMask);
            Scatter(dAqkOut + row * 8, value, nzIndex, fpMask);
        }

        const uint32_t vectorOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> e0;
        RegTensor<float> e1;
        FinalizeLoadPair(e0, e1, g + vectorOffset);

        RegTensor<bfloat16_t> kBf;
        RegTensor<float> k0;
        RegTensor<float> k1;
        RegTensor<float> kNeg0;
        RegTensor<float> kNeg1;
        FinalizeLoadIn<bfloat16_t, false>(kBf, k + vectorOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kBf, bfMask);
        // Only pack the causal K range read by the two band GEMMs.
        if (row < rowEnd) {
            Sub(kNeg0, center0, e0, fpMask);
            Sub(kNeg1, center1, e1, fpMask);
            Muls(kNeg0, kNeg0, KDA_FINALIZE_LN2, fpMask);
            Muls(kNeg1, kNeg1, KDA_FINALIZE_LN2, fpMask);
            Exp(kNeg0, kNeg0, fpMask);
            Exp(kNeg1, kNeg1, fpMask);
            Mul(kNeg0, kNeg0, k0, fpMask);
            Mul(kNeg1, kNeg1, k1, fpMask);
            FinalizeStoreLocalPair(kNegNd + row * 8, kNeg0, kNeg1, fpMask, nzIndex);
        }
        if (row >= rowBegin) {
            Sub(e0, e0, center0, fpMask);
            Sub(e1, e1, center1, fpMask);
            Muls(e0, e0, KDA_FINALIZE_LN2, fpMask);
            Muls(e1, e1, KDA_FINALIZE_LN2, fpMask);
            Exp(e0, e0, fpMask);
            Exp(e1, e1, fpMask);
            RegTensor<bfloat16_t> qBf;
            RegTensor<float> q0, q1;
            FinalizeLoadIn<bfloat16_t, false>(qBf, q + vectorOffset);
            CastHalf2Float<bfloat16_t>(q0, q1, qBf, bfMask);
            Mul(q0, q0, e0, fpMask);
            Mul(q1, q1, e1, fpMask);
            FinalizeStoreLocalPair(qPosNd + row * 8, q0, q1, fpMask, nzIndex);
            RegTensor<bfloat16_t> betaBf;
            RegTensor<float> betaFp;
            if constexpr (AscendC::IsSameType<BetaT, float>::value) {
                FinalizeLoadIn<float, true>(betaFp, beta + row);
            } else {
                FinalizeLoadIn<bfloat16_t, true>(betaBf, beta + row);
                Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, bfMask);
            }
            Mul(k0, k0, e0, fpMask);
            Mul(k1, k1, e1, fpMask);
            Mul(k0, k0, betaFp, fpMask);
            Mul(k1, k1, betaFp, fpMask);
            FinalizeStoreLocalPair(bkPosNd + row * 8, k0, k1, fpMask, nzIndex);
        }
    }
}

// Stage7 consumes the FP32 Cube result, writes the final BF16 dq, and updates
// dg_base in FP32 for the later K and gate stages.  The band factor is
// computed once and retained in the band-local scratch for Stage9.
__simd_vf__ inline void FinalizeStage7VF(__ubuf__ bfloat16_t *dqOut, __ubuf__ float *dgBase, __ubuf__ float *dqLocalRaw,
                                         __ubuf__ float *dqBase, __ubuf__ float *g, __ubuf__ float *center,
                                         __ubuf__ bfloat16_t *q, __ubuf__ float *qRstd, uint32_t hasQkL2Norm,
                                         uint16_t validRows, __ubuf__ float *diagonal, __ubuf__ bfloat16_t *k,
                                         __ubuf__ float *bandExp)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg fpMask = FinalizeMask<float, 64>();
    MaskReg bfMask = FinalizeMask<half, 128>();
    RegTensor<float> center0;
    RegTensor<float> center1;
    FinalizeLoadPair(center0, center1, center);
    for (uint16_t row = 0; row < FinalizeLoopBound(validRows, 64); ++row) {
        if (row >= validRows)
            continue;
        const uint32_t offset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> raw0;
        RegTensor<float> raw1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        RegTensor<float> dq0;
        RegTensor<float> dq1;
        FinalizeLoadPair(raw0, raw1, dqLocalRaw + offset);
        // Restore the band factor in registers, before the original gradient
        // arithmetic. Keep the two scaling operations and their rounding order.
        FinalizeLoadPair(exp0, exp1, g + offset);
        Sub(exp0, exp0, center0, fpMask);
        Sub(exp1, exp1, center1, fpMask);
        Muls(exp0, exp0, KDA_FINALIZE_LN2, fpMask);
        Muls(exp1, exp1, KDA_FINALIZE_LN2, fpMask);
        Exp(exp0, exp0, fpMask);
        Exp(exp1, exp1, fpMask);
        FinalizeStorePair(bandExp + offset, exp0, exp1, fpMask);
        Mul(raw0, raw0, exp0, fpMask);
        Mul(raw1, raw1, exp1, fpMask);
        Muls(raw0, raw0, 1.0f / KDA_FINALIZE_LOCAL_SCALE, fpMask);
        Muls(raw1, raw1, 1.0f / KDA_FINALIZE_LOCAL_SCALE, fpMask);
        FinalizeLoadPair(dq0, dq1, dqBase + offset);
        Add(dq0, dq0, raw0, fpMask);
        Add(dq1, dq1, raw1, fpMask);

        RegTensor<bfloat16_t> qBf;
        RegTensor<float> q0;
        RegTensor<float> q1;
        FinalizeLoadIn<bfloat16_t, false>(qBf, q + offset);
        CastHalf2Float<bfloat16_t>(q0, q1, qBf, bfMask);

        RegTensor<float> dg0;
        RegTensor<float> dg1;
        RegTensor<float> product0;
        RegTensor<float> product1;
        FinalizeLoadPair(dg0, dg1, dgBase + offset);
        Mul(product0, q0, raw0, fpMask);
        Mul(product1, q1, raw1, fpMask);
        Add(dg0, dg0, product0, fpMask);
        Add(dg1, dg1, product1, fpMask);
        FinalizeStorePair(dgBase + offset, dg0, dg1, fpMask);

        // exp2(g_i-g_i) is exactly one and its gate derivative is zero.
        // Keep this diagonal outside the scaled GEMMs, avoiding subtraction
        // of two rounded, dominant diagonal terms in the small gate gradient.
        RegTensor<bfloat16_t> kb;
        RegTensor<float> k0, k1, diag;
        FinalizeLoadIn<bfloat16_t, false>(kb, k + offset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(diag, diagonal + row);
        Mul(k0, k0, diag, fpMask);
        Mul(k1, k1, diag, fpMask);
        Add(dq0, dq0, k0, fpMask);
        Add(dq1, dq1, k1, fpMask);

        if (hasQkL2Norm != 0U) {
            RegTensor<float> dot;
            RegTensor<float> rstd;
            Mul(product0, dq0, q0, fpMask);
            Mul(product1, dq1, q1, fpMask);
            Add(product0, product0, product1, fpMask);
            ReduceSum(dot, product0, fpMask);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstd, qRstd + row);
            // ReduceSum only guarantees lane 0; broadcast it in-register
            // instead of a UB scratch round trip.
            RegTensor<float> dotBrc;
            Duplicate(dotBrc, dot, fpMask);
            Mul(product0, q0, dotBrc, fpMask);
            Mul(product1, q1, dotBrc, fpMask);
            Sub(dq0, dq0, product0, fpMask);
            Sub(dq1, dq1, product1, fpMask);
            Mul(dq0, dq0, rstd, fpMask);
            Mul(dq1, dq1, rstd, fpMask);
        }

        RegTensor<bfloat16_t> packed;
        FinalizeCastBf16(packed, dq0, dq1, fpMask);
        StoreAlign(dqOut + offset, packed, bfMask);
    }
}

template <typename BetaT>
__simd_vf__ inline void FinalizeStage9VF(__ubuf__ float *left, __ubuf__ float *right, __ubuf__ float *dbDelta,
                                         __ubuf__ float *dg, __ubuf__ float *bandExp, __ubuf__ bfloat16_t *k,
                                         __ubuf__ BetaT *beta, uint16_t rows, __ubuf__ float *diagonal,
                                         __ubuf__ bfloat16_t *q)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = FinalizeMask<float, 64>();
    MaskReg bfMask = FinalizeMask<half, 128>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    for (uint16_t row = 0; row < FinalizeLoopBound(rows, 32); ++row) {
        if (row >= rows)
            continue;
        uint32_t offset = row * KDA_FINALIZE_DIM;
        RegTensor<float> l0, l1, r0, r1, e0, e1, k0, k1, g0, g1, b;
        RegTensor<bfloat16_t> kb, bb;
        FinalizeLoadPair(l0, l1, left + offset);
        FinalizeLoadPair(r0, r1, right + offset);
        FinalizeLoadPair(e0, e1, bandExp + offset);
        Mul(l0, l0, e0, mask);
        Mul(l1, l1, e1, mask);
        Muls(l0, l0, 1.0f / KDA_FINALIZE_LOCAL_SCALE, mask);
        Muls(l1, l1, 1.0f / KDA_FINALIZE_LOCAL_SCALE, mask);
        Div(r0, r0, e0, mask);
        Div(r1, r1, e1, mask);
        Muls(r0, r0, 1.0f / KDA_FINALIZE_LOCAL_SCALE, mask);
        Muls(r1, r1, 1.0f / KDA_FINALIZE_LOCAL_SCALE, mask);
        FinalizeLoadIn<bfloat16_t, false>(kb, k + offset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(b, beta + row);
        } else {
            DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(bb, beta + row);
            Cast<float, bfloat16_t, ctHalf2Fp32Zero>(b, bb, mask);
        }
        RegTensor<float> p0, p1, sum;
        Mul(p0, l0, k0, mask);
        Mul(p1, l1, k1, mask);
        Add(p0, p0, p1, mask);
        ReduceSum(sum, p0, mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbDelta + row, sum, one);
        Mul(l0, l0, b, mask);
        Mul(l1, l1, b, mask);
        Sub(p0, l0, r0, mask);
        Sub(p1, l1, r1, mask);
        Mul(p0, p0, k0, mask);
        Mul(p1, p1, k1, mask);
        FinalizeLoadPair(g0, g1, dg + offset);
        Add(g0, g0, p0, mask);
        Add(g1, g1, p1, mask);
        FinalizeStorePair(dg + offset, g0, g1, mask);
        Add(l0, l0, r0, mask);
        Add(l1, l1, r1, mask);
        RegTensor<bfloat16_t> qb;
        RegTensor<float> q0, q1, diag;
        FinalizeLoadIn<bfloat16_t, false>(qb, q + offset);
        CastHalf2Float<bfloat16_t>(q0, q1, qb, bfMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(diag, diagonal + row);
        Mul(q0, q0, diag, mask);
        Mul(q1, q1, diag, mask);
        Add(l0, l0, q0, mask);
        Add(l1, l1, q1, mask);
        FinalizeStorePair(left + offset, l0, l1, mask);
    }
}

template <typename BetaT>
__simd_vf__ inline void FinalizeStage10VF(__ubuf__ bfloat16_t *out, __ubuf__ BetaT *dbOut, __ubuf__ float *delta,
                                          __ubuf__ float *base, __ubuf__ float *dbDelta, __ubuf__ float *dbBase,
                                          __ubuf__ bfloat16_t *k, __ubuf__ float *rstd, uint32_t hasNorm, uint16_t rows)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = FinalizeMask<float, 64>();
    MaskReg bfMask = FinalizeMask<half, 128>();
    // Interleave independent operations from two rows; handle an odd tail separately.
    uint16_t row = 0;
    for (uint16_t pair = 0; pair < FinalizeLoopBound(rows / 2, 16); ++pair) {
        if (2U * pair + 1U >= rows)
            continue;
        row = 2U * pair;
        uint32_t offsetA = row * KDA_FINALIZE_DIM;
        uint32_t offsetB = (row + 1U) * KDA_FINALIZE_DIM;
        RegTensor<float> dA0, dA1, bA0, bA1;
        RegTensor<float> dB0, dB1, bB0, bB1;
        FinalizeLoadPair(dA0, dA1, delta + offsetA);
        FinalizeLoadPair(dB0, dB1, delta + offsetB);
        FinalizeLoadPair(bA0, bA1, base + offsetA);
        FinalizeLoadPair(bB0, bB1, base + offsetB);
        Add(dA0, dA0, bA0, mask);
        Add(dB0, dB0, bB0, mask);
        Add(dA1, dA1, bA1, mask);
        Add(dB1, dB1, bB1, mask);
        if (hasNorm != 0U) {
            RegTensor<bfloat16_t> kbA, kbB;
            RegTensor<float> kA0, kA1, pA0, pA1, dotA, rsA;
            RegTensor<float> kB0, kB1, pB0, pB1, dotB, rsB;
            FinalizeLoadIn<bfloat16_t, false>(kbA, k + offsetA);
            FinalizeLoadIn<bfloat16_t, false>(kbB, k + offsetB);
            CastHalf2Float<bfloat16_t>(kA0, kA1, kbA, bfMask);
            CastHalf2Float<bfloat16_t>(kB0, kB1, kbB, bfMask);
            Mul(pA0, dA0, kA0, mask);
            Mul(pB0, dB0, kB0, mask);
            Mul(pA1, dA1, kA1, mask);
            Mul(pB1, dB1, kB1, mask);
            Add(pA0, pA0, pA1, mask);
            Add(pB0, pB0, pB1, mask);
            ReduceSum(dotA, pA0, mask);
            ReduceSum(dotB, pB0, mask);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rsA, rstd + row);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rsB, rstd + row + 1U);
            // ReduceSum only guarantees lane 0; broadcast it in-register
            // instead of a UB scratch round trip.
            RegTensor<float> dotBrcA, dotBrcB;
            Duplicate(dotBrcA, dotA, mask);
            Duplicate(dotBrcB, dotB, mask);
            Mul(pA0, kA0, dotBrcA, mask);
            Mul(pB0, kB0, dotBrcB, mask);
            Mul(pA1, kA1, dotBrcA, mask);
            Mul(pB1, kB1, dotBrcB, mask);
            Sub(dA0, dA0, pA0, mask);
            Sub(dB0, dB0, pB0, mask);
            Sub(dA1, dA1, pA1, mask);
            Sub(dB1, dB1, pB1, mask);
            Mul(dA0, dA0, rsA, mask);
            Mul(dB0, dB0, rsB, mask);
            Mul(dA1, dA1, rsA, mask);
            Mul(dB1, dB1, rsB, mask);
        }
        RegTensor<bfloat16_t> packedA, packedB;
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packedA, dA1, mask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packedB, dB1, mask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedA, dA0, mask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, dB0, mask);
        StoreAlign(out + offsetA, packedA, bfMask);
        StoreAlign(out + offsetB, packedB, bfMask);
        row += 2U;
    }
    if (row < rows) {
        uint32_t offset = row * KDA_FINALIZE_DIM;
        RegTensor<float> d0, d1, b0, b1;
        FinalizeLoadPair(d0, d1, delta + offset);
        FinalizeLoadPair(b0, b1, base + offset);
        Add(d0, d0, b0, mask);
        Add(d1, d1, b1, mask);
        if (hasNorm != 0U) {
            RegTensor<bfloat16_t> kb;
            RegTensor<float> k0, k1, p0, p1, dot, rs;
            FinalizeLoadIn<bfloat16_t, false>(kb, k + offset);
            CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
            Mul(p0, d0, k0, mask);
            Mul(p1, d1, k1, mask);
            Add(p0, p0, p1, mask);
            ReduceSum(dot, p0, mask);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rs, rstd + row);
            // ReduceSum only guarantees lane 0; broadcast it in-register
            // instead of a UB scratch round trip.
            RegTensor<float> dotBrc;
            Duplicate(dotBrc, dot, mask);
            Mul(p0, k0, dotBrc, mask);
            Mul(p1, k1, dotBrc, mask);
            Sub(d0, d0, p0, mask);
            Sub(d1, d1, p1, mask);
            Mul(d0, d0, rs, mask);
            Mul(d1, d1, rs, mask);
        }
        RegTensor<bfloat16_t> packed;
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packed, d1, mask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, d0, mask);
        StoreAlign(out + offset, packed, bfMask);
    }
    // One 32-byte UB row per scalar enables a single strided token-major DMA.
    uint32_t count = 1;
    MaskReg scalarMask = UpdateMask<float>(count);
    for (uint16_t row = 0; row < FinalizeLoopBound(rows, 32); ++row) {
        if (row >= rows)
            continue;
        RegTensor<float> db, deltaB;
        DataCopy<float, LoadDist::DIST_BRC_B32>(db, dbBase + row);
        DataCopy<float, LoadDist::DIST_BRC_B32>(deltaB, dbDelta + row);
        Add(db, db, deltaB, scalarMask);
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            StoreAlign(dbOut + row * 8, db, scalarMask);
        } else {
            RegTensor<bfloat16_t> packedB;
            Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, db, scalarMask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dbOut + row * 16, packedB, scalarMask);
        }
    }
}

// The dW residual has 128 columns. Produce one NZ row per iteration.
__simd_vf__ inline void FinalizeDwResidualVF(__ubuf__ bfloat16_t *dst, __ubuf__ float *src)
{
    MaskReg fpMask = FinalizeMask<float, 64>();
    MaskReg bfMask = FinalizeMask<half, 128>();
    RegTensor<uint16_t> column, block, nzIndex;
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), bfMask);
    Muls(block, block, uint16_t(1008), bfMask);
    Add(nzIndex, column, block, bfMask);
    static_assert(KDA_FINALIZE_CHUNK % 2U == 0U, "unrolled pairs need an even trip");
    // Interleave independent operations from two rows; keep the tail outside the loop.
    uint16_t row = 0;
    for (; row + 1U < KDA_FINALIZE_CHUNK; row += 2U) {
        RegTensor<float> valueA0, valueA1, highA0, highA1;
        RegTensor<float> valueB0, valueB1, highB0, highB1;
        RegTensor<bfloat16_t> packedA, packedB;
        FinalizeLoadPair(valueA0, valueA1, src + row * 128);
        FinalizeLoadPair(valueB0, valueB1, src + (row + 1U) * 128);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packedA, valueA1, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packedB, valueB1, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedA, valueA0, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, valueB0, fpMask);
        CastHalf2Float<bfloat16_t>(highA0, highA1, packedA, bfMask);
        CastHalf2Float<bfloat16_t>(highB0, highB1, packedB, bfMask);
        Sub(valueA0, valueA0, highA0, fpMask);
        Sub(valueB0, valueB0, highB0, fpMask);
        Sub(valueA1, valueA1, highA1, fpMask);
        Sub(valueB1, valueB1, highB1, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packedA, valueA1, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packedB, valueB1, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedA, valueA0, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, valueB0, fpMask);
        Scatter(dst + row * 16, packedA, nzIndex, bfMask);
        Scatter(dst + (row + 1U) * 16, packedB, nzIndex, bfMask);
    }
    if (row < KDA_FINALIZE_CHUNK) {
        RegTensor<float> value0, value1, high0, high1;
        RegTensor<bfloat16_t> packed;
        FinalizeLoadPair(value0, value1, src + row * 128);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packed, value1, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, value0, fpMask);
        CastHalf2Float<bfloat16_t>(high0, high1, packed, bfMask);
        Sub(value0, value0, high0, fpMask);
        Sub(value1, value1, high1, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packed, value1, fpMask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, value0, fpMask);
        Scatter(dst + row * 16, packed, nzIndex, bfMask);
    }
}

__simd_vf__ inline void FinalizeResidualVF(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint16_t blocks)
{
    MaskReg mask = FinalizeMask<float, 64>();
    RegTensor<uint16_t> column, block, nzIndex;
    MaskReg indexMask = FinalizeMask<half, 128>();
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), indexMask);
    Muls(block, block, uint16_t(1008), indexMask);
    Add(nzIndex, column, block, indexMask);
    // Interleave independent operations from two blocks; handle an odd tail separately.
    // The loop variable is named blk to avoid redefining the function-scope RegTensor named block.
    uint16_t blk = 0;
    for (uint16_t pair = 0; pair < FinalizeLoopBound(blocks / 2, 32); ++pair) {
        if (2U * pair + 1U >= blocks)
            continue;
        blk = 2U * pair;
        RegTensor<float> valueA, highA;
        RegTensor<float> valueB, highB;
        RegTensor<bfloat16_t> packedA, packedB;
        FinalizeLoad64(valueA, src + blk * 64U);
        FinalizeLoad64(valueB, src + (blk + 1U) * 64U);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedA, valueA, mask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, valueB, mask);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(highA, packedA, mask);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(highB, packedB, mask);
        Sub(valueA, valueA, highA, mask);
        Sub(valueB, valueB, highB, mask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedA, valueA, mask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, valueB, mask);
        FinalizeStoreNz64(dst + blk * 16U, packedA, nzIndex);
        FinalizeStoreNz64(dst + (blk + 1U) * 16U, packedB, nzIndex);
        blk += 2U;
    }
    if (blk < blocks) {
        RegTensor<float> value, high;
        RegTensor<bfloat16_t> packed;
        FinalizeLoad64(value, src + blk * 64U);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, value, mask);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(high, packed, mask);
        Sub(value, value, high, mask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, value, mask);
        FinalizeStoreNz64(dst + blk * 16U, packed, nzIndex);
    }
}


#endif
