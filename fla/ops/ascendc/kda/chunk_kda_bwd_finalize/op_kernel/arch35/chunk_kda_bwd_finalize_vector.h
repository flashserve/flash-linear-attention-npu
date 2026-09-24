/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H

#include "chunk_kda_bwd_finalize_common.h"
#include "kernel_utils/vector/regbase.hpp"
#include "chunk_kda_bwd_finalize_gate.h"

namespace KDA {

using namespace AscendC::MicroAPI;

constexpr float KDA_FINALIZE_LN2 = 0.6931471805599453f;
constexpr CastTrait KDA_FINALIZE_FP32_TO_BF16_RNE = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_RINT};
constexpr CastTrait KDA_FINALIZE_FP32_TO_BF16_RNE_ONE = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__simd_callee__ inline void FinalizeCastBf16(
    RegTensor<bfloat16_t> &dst, RegTensor<float> &even,
    RegTensor<float> &odd, MaskReg &mask)
{
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(dst, odd, mask);
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(dst, even, mask);
}

// Cast ZERO places one BF16 value in each 32-bit lane. Pack the 64
// values in registers, then scatter them into a 64-row NZ plane.
__simd_callee__ inline void FinalizeStoreNz64(
    __ubuf__ bfloat16_t *dst, RegTensor<bfloat16_t> &packed,
    RegTensor<uint16_t> &nzIndex)
{
    RegTensor<bfloat16_t> values, unused;
    DeInterleave(values, unused, packed, packed);
    uint32_t count = 64;
    MaskReg mask = UpdateMask<bfloat16_t>(count);
    Scatter(dst, values, nzIndex, mask);
}

__simd_callee__ inline void FinalizeZeroNzTail(
    __ubuf__ bfloat16_t *dst, uint16_t validRows, uint16_t cols)
{
    uint32_t count = cols;
    MaskReg mask = UpdateMask<bfloat16_t>(count);
    RegTensor<bfloat16_t> zero;
    Duplicate(zero, static_cast<bfloat16_t>(0), mask);
    RegTensor<uint16_t> column, block, nzIndex;
    MaskReg indexMask = CreateMask<half, MaskPattern::ALL>();
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), indexMask);
    Muls(block, block, uint16_t(1008), indexMask);
    Add(nzIndex, column, block, indexMask);
    for (uint16_t row = validRows; row < KDA_FINALIZE_CHUNK; ++row) {
        Scatter(dst + row * 16, zero, nzIndex, mask);
    }
}

__simd_vf__ inline void FinalizeStage5TailVF(
    __ubuf__ FinalizeLocalType *daq, __ubuf__ FinalizeLocalType *dak,
    __ubuf__ FinalizeLocalType *kn, __ubuf__ FinalizeLocalType *qp,
    __ubuf__ FinalizeLocalType *bp, uint16_t validRows)
{
    uint32_t count = 64;
    MaskReg mask = UpdateMask<FinalizeLocalType>(count);
    RegTensor<FinalizeLocalType> zero;
    Duplicate(zero, static_cast<FinalizeLocalType>(0), mask);
    MaskReg vectorMask = CreateMask<FinalizeLocalType, MaskPattern::ALL>();
    RegTensor<uint16_t> column, block, nzIndex;
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), vectorMask);
    Muls(block, block, uint16_t(1008), vectorMask);
    Add(nzIndex, column, block, vectorMask);
    Duplicate(zero, static_cast<FinalizeLocalType>(0), vectorMask);
    for (uint16_t row = validRows; row < KDA_FINALIZE_CHUNK; ++row) {
        Scatter(daq + row * 16, zero, nzIndex, mask);
        Scatter(dak + row * 16, zero, nzIndex, mask);
        Scatter(daq + KDA_FINALIZE_MATRIX_ELEMS + row * 16, zero, nzIndex, mask);
        Scatter(dak + KDA_FINALIZE_MATRIX_ELEMS + row * 16, zero, nzIndex, mask);
        Scatter(kn + row * 16, zero, nzIndex, vectorMask);
        Scatter(qp + row * 16, zero, nzIndex, vectorMask);
        Scatter(bp + row * 16, zero, nzIndex, vectorMask);
        Scatter(kn + KDA_FINALIZE_VECTOR_ELEMS + row * 16, zero, nzIndex, vectorMask);
        Scatter(qp + KDA_FINALIZE_VECTOR_ELEMS + row * 16, zero, nzIndex, vectorMask);
        Scatter(bp + KDA_FINALIZE_VECTOR_ELEMS + row * 16, zero, nzIndex, vectorMask);
    }
}

__simd_callee__ inline void FinalizeStoreLocalPair(
    __ubuf__ FinalizeLocalType *dst, RegTensor<float> &even,
    RegTensor<float> &odd, MaskReg &mask, RegTensor<uint16_t> &nzIndex)
{
    RegTensor<FinalizeLocalType> packed;
    Cast<FinalizeLocalType, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(packed, odd, mask);
    Cast<FinalizeLocalType, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, even, mask);
    MaskReg all = CreateMask<FinalizeLocalType, MaskPattern::ALL>();
    Scatter(dst, packed, nzIndex, all);
    RegTensor<float> hi0, hi1, low0, low1;
    CastHalf2Float<FinalizeLocalType>(hi0, hi1, packed, all);
    Sub(low0, even, hi0, mask);
    Sub(low1, odd, hi1, mask);
    FinalizeCastBf16(packed, low0, low1, mask);
    Scatter(dst + KDA_FINALIZE_VECTOR_ELEMS, packed, nzIndex, all);
}

__simd_callee__ inline void FinalizeStoreLocalRow(
    __ubuf__ bfloat16_t *dst, RegTensor<float> &value, MaskReg &mask,
    RegTensor<uint16_t> &nzIndex)
{
    RegTensor<bfloat16_t> packed;
    RegTensor<float> high, residual;
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, value, mask);
    FinalizeStoreNz64(dst, packed, nzIndex);
    Cast<float, bfloat16_t, ctHalf2Fp32Zero>(high, packed, mask);
    Sub(residual, value, high, mask);
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, residual, mask);
    FinalizeStoreNz64(dst + KDA_FINALIZE_MATRIX_ELEMS, packed, nzIndex);
}

// Stage0 emits paired NZ kE for contiguous UB-to-L1 copies. The raw log gates
// stay resident in the g frame; consumers recompute Exp(g*LN2) inline.
// FULL_TILE fixes every chunk row count at KDA_FINALIZE_CHUNK (see N1).
template <bool FULL_TILE>
__simd_vf__ inline void FinalizeStage0VF(
    __ubuf__ bfloat16_t *kENd, __ubuf__ bfloat16_t *lowNd, __ubuf__ float *rH,
    __ubuf__ bfloat16_t *k, __ubuf__ float *gk,
    __ubuf__ bfloat16_t *h, __ubuf__ bfloat16_t *dh,
    uint16_t validRows)
{
    const uint16_t nRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_CHUNK) : validRows;
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
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
    for (uint16_t row = 0; row < nRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, gk + rowOffset);
        Muls(e0, g0, KDA_FINALIZE_LN2, fpMask);
        Muls(e1, g1, KDA_FINALIZE_LN2, fpMask);
        Exp(e0, e0, fpMask);
        Exp(e1, e1, fpMask);

        LoadIn<bfloat16_t, false>(kb, k + rowOffset);
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
        LoadIn<bfloat16_t, false>(hb, h + rhOffset);
        LoadIn<bfloat16_t, false>(dhb, dh + rhOffset);
        CastHalf2Float<bfloat16_t>(h0, h1, hb, bfMask);
        CastHalf2Float<bfloat16_t>(dh0, dh1, dhb, bfMask);
        Mul(h0, h0, dh0, fpMask);
        Mul(h1, h1, dh1, fpMask);
        Add(product, h0, h1, fpMask);
        ReduceSum(sum, product, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rH + 2 * row, sum, fpMask);
        LoadIn<bfloat16_t, false>(hb, h + rhOffset + KDA_FINALIZE_DIM);
        LoadIn<bfloat16_t, false>(dhb, dh + rhOffset + KDA_FINALIZE_DIM);
        CastHalf2Float<bfloat16_t>(h0, h1, hb, bfMask);
        CastHalf2Float<bfloat16_t>(dh0, dh1, dhb, bfMask);
        Mul(h0, h0, dh0, fpMask);
        Mul(h1, h1, dh1, fpMask);
        Add(product, h0, h1, fpMask);
        ReduceSum(sum, product, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rH + 2 * row + 1, sum, fpMask);
    }

    // Short chunks leave the rH rows beyond 2 * validRows to this loop.
    for (uint16_t row = static_cast<uint16_t>(2 * nRows); row < KDA_FINALIZE_DIM; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<bfloat16_t> hb;
        RegTensor<bfloat16_t> dhb;
        RegTensor<float> h0;
        RegTensor<float> h1;
        RegTensor<float> dh0;
        RegTensor<float> dh1;
        RegTensor<float> product;
        RegTensor<float> sum;
        LoadIn<bfloat16_t, false>(hb, h + rowOffset);
        LoadIn<bfloat16_t, false>(dhb, dh + rowOffset);
        CastHalf2Float<bfloat16_t>(h0, h1, hb, bfMask);
        CastHalf2Float<bfloat16_t>(dh0, dh1, dhb, bfMask);
        Mul(h0, h0, dh0, fpMask);
        Mul(h1, h1, dh1, fpMask);
        Add(product, h0, h1, fpMask);
        ReduceSum(sum, product, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rH + row, sum, fpMask);
    }

    if (nRows < KDA_FINALIZE_CHUNK) {
        Duplicate(out, static_cast<bfloat16_t>(0), bfMask);
        for (uint16_t row = nRows; row < KDA_FINALIZE_CHUNK; ++row) {
            Scatter(kENd + row * 16, out, nzIndex, bfMask);
            Scatter(lowNd + row * 16, out, nzIndex, bfMask);
        }
    }
}

// Stage2 BuildZ.  zV/zW are direct FP32 FixPipe results in row-major UB.
// Paired Zb is produced in NZ order before the contiguous L1 publication.
template <typename BetaT, bool FULL_TILE>
__simd_vf__ inline void FinalizeStage2VF(
    __ubuf__ bfloat16_t *zbNd,
    __ubuf__ float *zV, __ubuf__ float *zW,
    __ubuf__ BetaT *beta, uint16_t validRows)
{
    const uint16_t nRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_CHUNK) : validRows;
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<bfloat16_t> betaBf;
    RegTensor<float> betaFp;
    if constexpr (AscendC::IsSameType<BetaT, float>::value) {
        LoadAlign(betaFp, beta);
    } else {
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(betaBf, beta);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, fpMask);
    }

    RegTensor<float> zero;
    Duplicate(zero, 0.0f, fpMask);
    if constexpr (!FULL_TILE) {
        // Only a partial chunk can hold inactive beta lanes.
        uint32_t betaCount = nRows;
        MaskReg betaMask = UpdateMask<float>(betaCount);
        Select(betaFp, betaFp, zero, betaMask);
    }
    RegTensor<uint16_t> column, block, nzIndex;
    MaskReg indexMask = CreateMask<half, MaskPattern::ALL>();
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), indexMask);
    Muls(block, block, uint16_t(1008), indexMask);
    Add(nzIndex, column, block, indexMask);
    for (uint16_t row = 0; row < nRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        RegTensor<float> zv;
        RegTensor<float> zw;
        RegTensor<float> result;
        RegTensor<bfloat16_t> packed;
        LoadAlign(zv, zV + rowOffset);
        LoadAlign(zw, zW + rowOffset);
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
    if (nRows < KDA_FINALIZE_CHUNK) {
        FinalizeZeroNzTail(zbNd, nRows, KDA_FINALIZE_CHUNK);
        FinalizeZeroNzTail(zbNd + 4096, nRows, KDA_FINALIZE_CHUNK);
    }
}

// Stage3 StatePre.  One VF invocation consumes one 32-row band slice of a
// head/chunk.  The two 64-lane FP32 register halves cover K/V=128 without a
// second pass.  gk is the resident raw log-gate frame; exp2(g) factors are
// recomputed inline with the same Muls(LN2)/Exp sequence Stage0 used to
// materialize.  gateState accumulates across bands through its persistent
// slot in the original row order; the rH term is added by the last band only.
template <typename BetaT, bool FULL_TILE>
__simd_vf__ inline void FinalizeStage3VF(
    __ubuf__ float *dkState, __ubuf__ float *dvb,
    __ubuf__ bfloat16_t *dv, __ubuf__ float *gateState, __ubuf__ float *dbV,
    __ubuf__ float *dqRaw, __ubuf__ float *gk,
    __ubuf__ bfloat16_t *k, __ubuf__ bfloat16_t *v,
    __ubuf__ BetaT *beta, __ubuf__ float *gkLast, __ubuf__ float *rH,
    float scale, uint16_t validRows, uint16_t rowBegin, uint16_t rows)
{
    const uint16_t bandRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_INTRA_ROWS) : rows;
    const uint16_t nValid = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_CHUNK) : validRows;
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> last0;
    RegTensor<float> last1;
    RegTensor<float> gate0;
    RegTensor<float> gate1;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(last0, last1, gkLast);
    if (rowBegin == 0) {
        Duplicate(gate0, 0.0f, fpMask);
        Duplicate(gate1, 0.0f, fpMask);
    } else {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(gate0, gate1, gateState);
    }

    for (uint16_t row = 0; row < bandRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> g0;
        RegTensor<float> g1;
        RegTensor<float> decay0;
        RegTensor<float> decay1;
        RegTensor<float> state0;
        RegTensor<float> state1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, gk + rowOffset);
        Sub(decay0, last0, g0, fpMask);
        Sub(decay1, last1, g1, fpMask);
        Muls(decay0, decay0, KDA_FINALIZE_LN2, fpMask);
        Muls(decay1, decay1, KDA_FINALIZE_LN2, fpMask);
        Exp(decay0, decay0, fpMask);
        Exp(decay1, decay1, fpMask);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            state0, state1, dkState + rowOffset);
        Mul(state0, state0, decay0, fpMask);
        Mul(state1, state1, decay1, fpMask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dkState + rowOffset, state0, state1, fpMask);

        RegTensor<bfloat16_t> kb;
        RegTensor<float> k0;
        RegTensor<float> k1;
        LoadIn<bfloat16_t, false>(kb, k + rowOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        Mul(k0, k0, state0, fpMask);
        Mul(k1, k1, state1, fpMask);
        Add(gate0, gate0, k0, fpMask);
        Add(gate1, gate1, k1, fpMask);

        RegTensor<float> dvb0;
        RegTensor<float> dvb1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(dvb0, dvb1, dvb + rowOffset);
        RegTensor<bfloat16_t> betaBf;
        RegTensor<float> betaFp;
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            LoadIn<float, true>(betaFp, beta + row);
        } else {
            LoadIn<bfloat16_t, true>(betaBf, beta + row);
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
        LoadIn<bfloat16_t, false>(vb, v + rowOffset);
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
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(dq0, dq1, dqRaw + rowOffset);
        Mul(dq0, dq0, exp0, fpMask);
        Mul(dq1, dq1, exp1, fpMask);
        Muls(dq0, dq0, scale, fpMask);
        Muls(dq1, dq1, scale, fpMask);
        // dqRaw's row is already consumed into registers; dqBase takes its
        // place so the band buffer at 128 KiB matches the Stage7 load address.
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dqRaw + rowOffset, dq0, dq1, fpMask);
    }

    // The rH term enters the gate accumulation exactly once, after the last
    // band's rows, keeping the original FP add order across bands.
    if (rowBegin + bandRows == nValid) {
        RegTensor<float> lastExp0;
        RegTensor<float> lastExp1;
        RegTensor<float> rh0;
        RegTensor<float> rh1;
        Muls(lastExp0, last0, KDA_FINALIZE_LN2, fpMask);
        Muls(lastExp1, last1, KDA_FINALIZE_LN2, fpMask);
        Exp(lastExp0, lastExp0, fpMask);
        Exp(lastExp1, lastExp1, fpMask);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(rh0, rh1, rH);
        Mul(rh0, rh0, lastExp0, fpMask);
        Mul(rh1, rh1, lastExp1, fpMask);
        Add(gate0, gate0, rh0, fpMask);
        Add(gate1, gate1, rh1, fpMask);
    }
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(gateState, gate0, gate1, fpMask);
}

// Stage4 BaseFinalize, one 32-row band slice per invocation.  dkState is
// updated in place to dkBase and dbV to dbBase; dg is written straight to
// its resident frame at 216 KiB (band offset applied by the caller).  dqBase
// and the resident raw g frame stay live for later stages; the exp2(g)
// factor is recomputed inline from g.  gateState joins dg only on the
// chunk's global last row.
template <typename BetaT, bool FULL_TILE>
__simd_vf__ inline void FinalizeStage4VF(
    __ubuf__ float *dkState, __ubuf__ float *dqBase,
    __ubuf__ float *dKgbRaw, __ubuf__ float *dg, __ubuf__ float *g,
    __ubuf__ bfloat16_t *q,
    __ubuf__ bfloat16_t *k, __ubuf__ BetaT *beta,
    __ubuf__ float *gateState, __ubuf__ float *dbV,
    uint16_t validRows, uint16_t rowBegin, uint16_t rows)
{
    const uint16_t bandRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_INTRA_ROWS) : rows;
    const uint16_t nValid = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_CHUNK) : validRows;
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> gate0;
    RegTensor<float> gate1;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(gate0, gate1, gateState);

    for (uint16_t row = 0; row < bandRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> state0;
        RegTensor<float> state1;
        RegTensor<float> dq0;
        RegTensor<float> dq1;
        RegTensor<float> dkg0;
        RegTensor<float> dkg1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            state0, state1, dkState + rowOffset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            dq0, dq1, dqBase + rowOffset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            dkg0, dkg1, dKgbRaw + rowOffset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            exp0, exp1, g + rowOffset);
        Muls(exp0, exp0, KDA_FINALIZE_LN2, fpMask);
        Muls(exp1, exp1, KDA_FINALIZE_LN2, fpMask);
        Exp(exp0, exp0, fpMask);
        Exp(exp1, exp1, fpMask);

        RegTensor<bfloat16_t> betaBf;
        RegTensor<float> betaFp;
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            LoadIn<float, true>(betaFp, beta + row);
        } else {
            LoadIn<bfloat16_t, true>(betaBf, beta + row);
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
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dkState + rowOffset, dkTerm0, dkTerm1, fpMask);

        RegTensor<bfloat16_t> qBf;
        RegTensor<bfloat16_t> kBf;
        RegTensor<float> q0;
        RegTensor<float> q1;
        RegTensor<float> k0;
        RegTensor<float> k1;
        LoadIn<bfloat16_t, false>(qBf, q + rowOffset);
        LoadIn<bfloat16_t, false>(kBf, k + rowOffset);
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
        LoadIn<float, true>(dbBase, dbV + row);
        Sub(dbBase, dbBase, reduceSum, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbV + row, dbBase, fpMask);

        Mul(dkgKe0, dkgKe0, betaFp, fpMask);
        Mul(dkgKe1, dkgKe1, betaFp, fpMask);
        Sub(dg0, dg0, dkgKe0, fpMask);
        Sub(dg1, dg1, dkgKe1, fpMask);
        if (rowBegin + row + 1U == nValid) {
            Add(dg0, dg0, gate0, fpMask);
            Add(dg1, dg1, gate1, fpMask);
        }
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dg + rowOffset, dg0, dg1, fpMask);
    }
}

// Stage5 rebases one output band. The BF16 high/residual operands are
// produced directly in NZ order for contiguous UB-to-L1 copies.
#include "chunk_kda_bwd_finalize_intra.h"

template <typename BetaT, bool FULL_TILE>
__simd_vf__ inline void FinalizeStage5VF(
    __ubuf__ FinalizeLocalType *dAkkNd, __ubuf__ FinalizeLocalType *kNegNd,
    __ubuf__ FinalizeLocalType *qPosNd, __ubuf__ FinalizeLocalType *bkPosNd,
    __ubuf__ float *dAkkRaw, __ubuf__ bfloat16_t *q,
    __ubuf__ bfloat16_t *k, __ubuf__ float *g,
    __ubuf__ BetaT *beta, __ubuf__ float *center, uint16_t validRows,
    uint16_t rowBegin, uint16_t rowEnd)
{
    const uint16_t nRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_CHUNK) : validRows;
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> zero;
    RegTensor<float> one;
    Duplicate(zero, 0.0f, fpMask);
    Duplicate(one, 1.0f, fpMask);

    RegTensor<bfloat16_t> zeroBf;
    Duplicate(zeroBf, static_cast<bfloat16_t>(0), bfMask);
    RegTensor<float> center0, center1;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(center0, center1, center);
    // NZ [column/16, row, column%16], in BF16 element offsets. Produce the
    // physical L1 layout here so MTE3 can copy each complete pair of planes.
    RegTensor<uint16_t> column, block, nzIndex;
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), bfMask);
    Muls(block, block, uint16_t(1008), bfMask);
    Add(nzIndex, column, block, bfMask);
    for (uint16_t row = 0; row < nRows; ++row) {
        const uint32_t matrixOffset =
            static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        if (rowBegin == 0) {
            RegTensor<float> raw0;
            LoadAlign(raw0, dAkkRaw + matrixOffset);
            Muls(raw0, raw0, -65536.0f, fpMask);
            uint32_t lowerCount = row;
            MaskReg lower = UpdateMask<float>(lowerCount);
            Select(raw0, raw0, zero, lower);
            FinalizeStoreLocalRow(dAkkNd + row * 16, raw0, fpMask, nzIndex);
        }

        const uint32_t vectorOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> e0;
        RegTensor<float> e1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            e0, e1, g + vectorOffset);

        RegTensor<bfloat16_t> kBf;
        RegTensor<float> k0;
        RegTensor<float> k1;
        RegTensor<float> kNeg0;
        RegTensor<float> kNeg1;
        RegTensor<bfloat16_t> packed;
        LoadIn<bfloat16_t, false>(kBf, k + vectorOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kBf, bfMask);
        // Noncausal rows are exactly zero. Publish zero planes directly,
        // rather than loading/scaling/splitting operands which cannot contribute.
        if (row < rowEnd) {
            Sub(kNeg0, center0, e0, fpMask);
            Sub(kNeg1, center1, e1, fpMask);
            Muls(kNeg0, kNeg0, KDA_FINALIZE_LN2, fpMask);
            Muls(kNeg1, kNeg1, KDA_FINALIZE_LN2, fpMask);
            Exp(kNeg0, kNeg0, fpMask);
            Exp(kNeg1, kNeg1, fpMask);
            Mul(kNeg0, kNeg0, k0, fpMask);
            Mul(kNeg1, kNeg1, k1, fpMask);
            FinalizeStoreLocalPair(kNegNd + row * 16, kNeg0, kNeg1, fpMask, nzIndex);
        } else {
            Scatter(kNegNd + row * 16, zeroBf, nzIndex, bfMask);
            Scatter(kNegNd + KDA_FINALIZE_VECTOR_ELEMS + row * 16, zeroBf, nzIndex, bfMask);
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
            LoadIn<bfloat16_t, false>(qBf, q + vectorOffset);
            CastHalf2Float<bfloat16_t>(q0, q1, qBf, bfMask);
            Mul(q0, q0, e0, fpMask);
            Mul(q1, q1, e1, fpMask);
            FinalizeStoreLocalPair(qPosNd + row * 16, q0, q1, fpMask, nzIndex);
            RegTensor<bfloat16_t> betaBf;
            RegTensor<float> betaFp;
            if constexpr (AscendC::IsSameType<BetaT, float>::value) {
                LoadIn<float, true>(betaFp, beta + row);
            } else {
                LoadIn<bfloat16_t, true>(betaBf, beta + row);
                Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, bfMask);
            }
            Mul(k0, k0, e0, fpMask);
            Mul(k1, k1, e1, fpMask);
            Mul(k0, k0, betaFp, fpMask);
            Mul(k1, k1, betaFp, fpMask);
            FinalizeStoreLocalPair(bkPosNd + row * 16, k0, k1, fpMask, nzIndex);
        } else {
            Scatter(qPosNd + row * 16, zeroBf, nzIndex, bfMask);
            Scatter(bkPosNd + row * 16, zeroBf, nzIndex, bfMask);
            Scatter(qPosNd + KDA_FINALIZE_VECTOR_ELEMS + row * 16, zeroBf, nzIndex, bfMask);
            Scatter(bkPosNd + KDA_FINALIZE_VECTOR_ELEMS + row * 16, zeroBf, nzIndex, bfMask);
        }
    }
}

// Retain the FP32 dAqk input as BF16 high/residual planes.
template <bool FULL_TILE>
__simd_vf__ inline void FinalizeStage5DaqkVF(
    __ubuf__ FinalizeLocalType *dAqkOut, __ubuf__ float *dAqk,
    __ubuf__ float *diagonal, uint16_t validRows)
{
    const uint16_t nRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_CHUNK) : validRows;
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    RegTensor<float> zero;
    Duplicate(zero, 0.0f, fpMask);
    RegTensor<uint16_t> column, block, nzIndex;
    MaskReg indexMask = CreateMask<half, MaskPattern::ALL>();
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), indexMask);
    Muls(block, block, uint16_t(1008), indexMask);
    Add(nzIndex, column, block, indexMask);
    for (uint16_t row = 0; row < nRows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        RegTensor<float> value;
        RegTensor<float> diag;
        DataCopy<float, LoadDist::DIST_BRC_B32>(diag, dAqk + offset + row);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(diagonal + row, diag, one);
        LoadAlign(value, dAqk + offset);
        uint32_t count = row;
        MaskReg strict = UpdateMask<float>(count);
        Select(value, value, zero, strict);
        Muls(value, value, 65536.0f, fpMask);
        FinalizeStoreLocalRow(dAqkOut + row * 16, value, fpMask, nzIndex);
    }
}

// Stage7 consumes the FP32 Cube result, writes the final BF16 dq, and updates
// dg_base in FP32 for the later K and gate stages.  The band factor is
// recomputed inline from the resident raw gates and the band center.
template <bool FULL_TILE>
__simd_vf__ inline void FinalizeStage7VF(
    __ubuf__ bfloat16_t *dqOut, __ubuf__ float *dgBase,
    __ubuf__ float *dqLocalRaw, __ubuf__ float *dqBase,
    __ubuf__ float *g, __ubuf__ float *center, __ubuf__ bfloat16_t *q,
    __ubuf__ float *qRstd, uint32_t hasQkL2Norm, uint16_t validRows,
    __ubuf__ float *diagonal, __ubuf__ bfloat16_t *k)
{
    const uint16_t bandRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_INTRA_ROWS) : validRows;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> center0;
    RegTensor<float> center1;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(center0, center1, center);
    for (uint16_t row = 0; row < bandRows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> raw0;
        RegTensor<float> raw1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        RegTensor<float> dq0;
        RegTensor<float> dq1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            raw0, raw1, dqLocalRaw + offset);
        // Restore the band factor in registers, before the original gradient
        // arithmetic. Keep the two scaling operations and their rounding order.
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(exp0, exp1, g + offset);
        Sub(exp0, exp0, center0, fpMask);
        Sub(exp1, exp1, center1, fpMask);
        Muls(exp0, exp0, KDA_FINALIZE_LN2, fpMask);
        Muls(exp1, exp1, KDA_FINALIZE_LN2, fpMask);
        Exp(exp0, exp0, fpMask);
        Exp(exp1, exp1, fpMask);
        Mul(raw0, raw0, exp0, fpMask);
        Mul(raw1, raw1, exp1, fpMask);
        Muls(raw0, raw0, 1.0f / 65536.0f, fpMask);
        Muls(raw1, raw1, 1.0f / 65536.0f, fpMask);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            dq0, dq1, dqBase + offset);
        Add(dq0, dq0, raw0, fpMask);
        Add(dq1, dq1, raw1, fpMask);

        RegTensor<bfloat16_t> qBf;
        RegTensor<float> q0;
        RegTensor<float> q1;
        LoadIn<bfloat16_t, false>(qBf, q + offset);
        CastHalf2Float<bfloat16_t>(q0, q1, qBf, bfMask);

        RegTensor<float> dg0;
        RegTensor<float> dg1;
        RegTensor<float> product0;
        RegTensor<float> product1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            dg0, dg1, dgBase + offset);
        Mul(product0, q0, raw0, fpMask);
        Mul(product1, q1, raw1, fpMask);
        Add(dg0, dg0, product0, fpMask);
        Add(dg1, dg1, product1, fpMask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dgBase + offset, dg0, dg1, fpMask);

        // exp2(g_i-g_i) is exactly one and its gate derivative is zero.
        // Keep this diagonal outside the scaled GEMMs, avoiding subtraction
        // of two rounded, dominant diagonal terms in the small gate gradient.
        RegTensor<bfloat16_t> kb;
        RegTensor<float> k0, k1, diag;
        LoadIn<bfloat16_t, false>(kb, k + offset);
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

template <typename BetaT, bool FULL_TILE>
__simd_vf__ inline void FinalizeStage9VF(
    __ubuf__ float *left, __ubuf__ float *right, __ubuf__ float *dbDelta,
    __ubuf__ float *dg, __ubuf__ float *g, __ubuf__ float *center,
    __ubuf__ bfloat16_t *k, __ubuf__ BetaT *beta, uint16_t rows,
    __ubuf__ float *diagonal, __ubuf__ bfloat16_t *q)
{
    const uint16_t bandRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_INTRA_ROWS) : rows;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    RegTensor<float> center0;
    RegTensor<float> center1;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(center0, center1, center);
    for (uint16_t row = 0; row < bandRows; ++row) {
        uint32_t offset = row * KDA_FINALIZE_DIM;
        RegTensor<float> l0, l1, r0, r1, e0, e1, k0, k1, g0, g1, b;
        RegTensor<bfloat16_t> kb, bb;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(l0, l1, left + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(r0, r1, right + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(e0, e1, g + offset);
        Sub(e0, e0, center0, mask);
        Sub(e1, e1, center1, mask);
        Muls(e0, e0, KDA_FINALIZE_LN2, mask);
        Muls(e1, e1, KDA_FINALIZE_LN2, mask);
        Exp(e0, e0, mask);
        Exp(e1, e1, mask);
        Mul(l0, l0, e0, mask);
        Mul(l1, l1, e1, mask);
        Muls(l0, l0, 1.0f / 65536.0f, mask);
        Muls(l1, l1, 1.0f / 65536.0f, mask);
        Div(r0, r0, e0, mask);
        Div(r1, r1, e1, mask);
        Muls(r0, r0, 1.0f / 65536.0f, mask);
        Muls(r1, r1, 1.0f / 65536.0f, mask);
        LoadIn<bfloat16_t, false>(kb, k + offset);
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
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, dg + offset);
        Add(g0, g0, p0, mask);
        Add(g1, g1, p1, mask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(dg + offset, g0, g1, mask);
        Add(l0, l0, r0, mask);
        Add(l1, l1, r1, mask);
        RegTensor<bfloat16_t> qb;
        RegTensor<float> q0, q1, diag;
        LoadIn<bfloat16_t, false>(qb, q + offset);
        CastHalf2Float<bfloat16_t>(q0, q1, qb, bfMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(diag, diagonal + row);
        Mul(q0, q0, diag, mask);
        Mul(q1, q1, diag, mask);
        Add(l0, l0, q0, mask);
        Add(l1, l1, q1, mask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(left + offset, l0, l1, mask);
    }
}

template <typename BetaT, bool FULL_TILE>
__simd_vf__ inline void FinalizeStage10VF(
    __ubuf__ bfloat16_t *out, __ubuf__ BetaT *dbOut,
    __ubuf__ float *delta, __ubuf__ float *base,
    __ubuf__ float *dbDelta, __ubuf__ float *dbBase,
    __ubuf__ bfloat16_t *k, __ubuf__ float *rstd,
    uint32_t hasNorm, uint16_t rows)
{
    const uint16_t bandRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_INTRA_ROWS) : rows;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    for (uint16_t row = 0; row < bandRows; ++row) {
        uint32_t offset = row * KDA_FINALIZE_DIM;
        RegTensor<float> d0, d1, b0, b1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(d0, d1, delta + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(b0, b1, base + offset);
        Add(d0, d0, b0, mask);
        Add(d1, d1, b1, mask);
        if (hasNorm != 0U) {
            RegTensor<bfloat16_t> kb;
            RegTensor<float> k0, k1, p0, p1, dot, rs;
            LoadIn<bfloat16_t, false>(kb, k + offset);
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
        FinalizeCastBf16(packed, d0, d1, mask);
        StoreAlign(out + offset, packed, bfMask);
    }
    uint32_t count = bandRows;
    MaskReg rowMask = UpdateMask<float>(count);
    RegTensor<float> db, deltaB;
    RegTensor<bfloat16_t> packedB;
    LoadAlign(db, dbBase);
    LoadAlign(deltaB, dbDelta);
    Add(db, db, deltaB, rowMask);
    if constexpr (AscendC::IsSameType<BetaT, float>::value) {
        StoreAlign(dbOut, db, rowMask);
    } else {
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, db, rowMask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dbOut, packedB, rowMask);
    }
}

// The dW residual has 128 columns. Produce one NZ row per iteration.
__simd_vf__ inline void FinalizeDwResidualVF(
    __ubuf__ bfloat16_t *dst, __ubuf__ float *src)
{
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<uint16_t> column, block, nzIndex;
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), bfMask);
    Muls(block, block, uint16_t(1008), bfMask);
    Add(nzIndex, column, block, bfMask);
    for (uint16_t row = 0; row < KDA_FINALIZE_CHUNK; ++row) {
        RegTensor<float> value0, value1, high0, high1;
        RegTensor<bfloat16_t> packed;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(value0, value1, src + row * 128);
        FinalizeCastBf16(packed, value0, value1, fpMask);
        CastHalf2Float<bfloat16_t>(high0, high1, packed, bfMask);
        Sub(value0, value0, high0, fpMask);
        Sub(value1, value1, high1, fpMask);
        FinalizeCastBf16(packed, value0, value1, fpMask);
        Scatter(dst + row * 16, packed, nzIndex, bfMask);
    }
}

__simd_vf__ inline void FinalizeResidualVF(
    __ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint16_t blocks)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<uint16_t> column, block, nzIndex;
    MaskReg indexMask = CreateMask<half, MaskPattern::ALL>();
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), indexMask);
    Muls(block, block, uint16_t(1008), indexMask);
    Add(nzIndex, column, block, indexMask);
    for (uint16_t block = 0; block < blocks; ++block) {
        RegTensor<float> value, high;
        RegTensor<bfloat16_t> packed;
        LoadAlign(value, src + block * 64U);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, value, mask);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(high, packed, mask);
        Sub(value, value, high, mask);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, value, mask);
        FinalizeStoreNz64(dst + block * 16U, packed, nzIndex);
    }
}

template <bool FULL_TILE>
class ChunkKdaBwdFinalizeVectorStage12 {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
        GM_ADDR h, GM_ADDR dh,
        GM_ADDR dAqk, GM_ADDR dqRaw, GM_ADDR qRstd, GM_ADDR dq, GM_ADDR dv,
        GM_ADDR kRstd, GM_ADDR dk, GM_ADDR dBeta,
        GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR dG,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
        const ChunkKdaBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        rawG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rawG));
        aLog_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_A_LOG *>(aLog));
        dtBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBias));
        dG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dG));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_BETA *>(beta));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
        if (tiling->hasQkL2Norm != 0U) {
            qRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
        }
        dq_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dq));
        dk_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dk));
        dBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_BETA *>(dBeta));
        if (tiling->hasQkL2Norm != 0U) {
            kRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
        }
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv));
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(workspace));
        // These final outputs are streaming writes. Preserve L2 capacity for
        // the input tiles and the workspace reused by later phases.
        dq_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dk_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dv_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dG_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dBeta_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // Each tile of these inputs is read once by this kernel.
        rawG_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dAqk_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dqRaw_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // q/k/beta are loaded once and retained in UB through all bands.
        q_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        k_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        beta_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        pipe_ = pipe;
        subBlockNum_ = AscendC::GetSubBlockNum();
        if (subBlockNum_ == 0U) {
            subBlockNum_ = KDA_FINALIZE_AIV_COUNT;
        }
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        pipe_->InitBuffer(ubBuf_, KDA_FINALIZE_UB_BYTES);
        ub_ = ubBuf_.Get<uint8_t>();
        for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
            mte2ToV_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToMte2_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            // The first Stage0 has no preceding zB MTE3 reader.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
            zBPublishCount_[slot] = 0;
        }
        stage3Mte3ToV_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
        stage3Mte3ToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        stage0Mte3ToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        stage5VToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        stage7Mte3ToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        gateVToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        stateVToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
    }

    __aicore__ inline void Process()
    {
        const int64_t logicalCore = AscendC::GetBlockIdx() / subBlockNum_;
        const int64_t coreNum = AscendC::GetBlockNum();
        uint64_t generation = 0;
        uint64_t groupGeneration = 0;
        for (int64_t workTask = logicalCore; workTask < tiling_->workTaskNum;
             workTask += coreNum, ++groupGeneration) {
            const int64_t headWindow = workTask / tiling_->chunkTaskNum;
            const int64_t chunkTask = workTask - headWindow * tiling_->chunkTaskNum;
            const int64_t headBegin = headWindow * KDA_FINALIZE_HEADS_PER_WINDOW;
            const int64_t headEnd = FinalizeMin(
                headBegin + KDA_FINALIZE_HEADS_PER_WINDOW, tiling_->NV);
            FinalizeChunkInfo chunk;
            ResolveFinalizeChunk<FULL_TILE>(chunkTask, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }

            // Even an idle AIV must consume this task's credit. Otherwise a
            // one-head window can run ahead and overwrite the preceding
            // task's qPos/bkPos in L1 while Stage8 still reads them.
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                KDA_FINALIZE_TASK_L1_FREE);

            // The task's later phases reinterpret nearly the whole UB.  The
            // next work task starts from slot0, while the previous task
            // finishes on slot1, so a per-slot credit alone cannot protect
            // the shared range.  Drain the preceding task's final MTE3 before
            // any Stage0 MTE2 reinterprets UB, then seed the same event for
            // this task's TzaResidual entry.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);

            // Stage0 AIV is independent of Stage0 AIC.  Each AIV owns alternate
            // heads and alternates its two physical UB slots.
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunStage0(chunk, chunkTask, head, owner, slot);
            }
            // Stage0 uses one shared UB working set for all local heads.  Its
            // final workspace egress must finish before Stage2 reinterprets
            // the same physical UB range.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);

            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_) continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                    KDA_FINALIZE_ZV_FREE_BASE + slot);
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                    KDA_FINALIZE_ZW_FREE_BASE + slot);
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                    KDA_FINALIZE_LOCAL_READY_BASE + slot);
            }

            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_) continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunDwResidual(static_cast<uint32_t>(head - headBegin), slot);
            }

            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunStage2(chunk, head, owner, slot);
            }

            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_) continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunTzaResidual(static_cast<uint32_t>(head - headBegin), slot);
            }
            // Drain the Tza residual L1 publication before Stage5 reuses
            // [16,144) KiB.  Retained q/k/g/beta above that range remain live.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);

            const uint32_t chunkRows =
                FULL_TILE ? KDA_FINALIZE_CHUNK : static_cast<uint32_t>(chunk.validRows);
            for (uint32_t rowBegin = 0; rowBegin < chunkRows;
                 rowBegin += KDA_FINALIZE_INTRA_ROWS) {
                generation -= static_cast<uint64_t>(headEnd - headBegin);
                for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                    if ((generation & 1U) != subBlockIdx_) continue;
                    const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                    const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                    RunStage5(chunk, head, owner, slot, rowBegin);
                    RunStateBaseSlice(chunk, head, owner, slot, logicalCore, groupGeneration, rowBegin);
                    RunIntraBandResults(chunk, head, slot, rowBegin);
                }
            }
            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_) continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
                // Resolve the next work task on this core; when this AIV owns
                // a head there, Stage11 prefetches that head's Stage0 inputs
                // into the tail's dead UB.  The generation at the next task's
                // start is this loop's value plus its remaining increments.
                int64_t nextHead = -1;
                int64_t nextChunkTask = -1;
                FinalizeChunkInfo nextChunk;
                const int64_t nextWorkTask = workTask + coreNum;
                if (nextWorkTask < tiling_->workTaskNum) {
                    const int64_t nextHeadWindow = nextWorkTask / tiling_->chunkTaskNum;
                    nextChunkTask = nextWorkTask - nextHeadWindow * tiling_->chunkTaskNum;
                    ResolveFinalizeChunk<FULL_TILE>(nextChunkTask, cuSeqlens_, chunkIndices_, *tiling_, nextChunk);
                    if (nextChunk.valid) {
                        const int64_t nextHeadBegin = nextHeadWindow * KDA_FINALIZE_HEADS_PER_WINDOW;
                        const int64_t nextHeadEnd = FinalizeMin(
                            nextHeadBegin + KDA_FINALIZE_HEADS_PER_WINDOW, tiling_->NV);
                        uint64_t nextGeneration = generation + static_cast<uint64_t>(headEnd - head);
                        for (int64_t nh = nextHeadBegin; nh < nextHeadEnd; ++nh, ++nextGeneration) {
                            if ((nextGeneration & 1U) == subBlockIdx_) {
                                nextHead = nh;
                                break;
                            }
                        }
                    }
                }
                RunStage11(chunk, chunkTask, head, nextChunk, nextChunkTask, nextHead);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
        }
        for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
    }

private:
    __aicore__ inline void RunStage11(
        const FinalizeChunkInfo &chunk, int64_t chunkTask, int64_t head,
        const FinalizeChunkInfo &nextChunk, int64_t nextChunkTask, int64_t nextHead)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
        // Both bands leave the complete dg frame resident at 216 KiB; scan it
        // in place for the unchanged full-chunk reverse scan.  Stage11 scratch
        // sits in dead regions: rawG in the dead kNeg plane, the gate scalars
        // in the dead beta strip, keeping [80,144) and [160,208) KiB free for
        // the Stage0 prefetch below.
        auto dg = UbBytes(KDA_FINALIZE_UB_DG).template ReinterpretCast<float>();
        auto raw = UbBytes(48 * 1024).template ReinterpretCast<float>();
        auto bias = UbBytes(208 * 1024).template ReinterpretCast<float>();
        auto log = UbBytes(208 * 1024 + 512).template ReinterpretCast<DTYPE_A_LOG>();
        auto da = UbBytes(209 * 1024).template ReinterpretCast<float>();
        auto db = UbBytes(209 * 1024 + 512).template ReinterpretCast<float>();
        const int64_t token = FinalizeTokenOffset<FULL_TILE>(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint32_t elems =
            (FULL_TILE ? KDA_FINALIZE_CHUNK : chunk.validRows) * KDA_FINALIZE_DIM;
        AscendC::DataCopy(raw, rawG_[token], elems);
        AscendC::DataCopy(bias, dtBias_[head * KDA_FINALIZE_DIM], KDA_FINALIZE_DIM);
        AscendC::DataCopyPad(log, aLog_[head],
            AscendC::DataCopyExtParams{1, sizeof(DTYPE_A_LOG), 0, 0, 0},
            AscendC::DataCopyPadExtParams<DTYPE_A_LOG>{false, 0, 0, static_cast<DTYPE_A_LOG>(0)});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
        // Prefetch the next task's Stage0 inputs into this Stage11's dead UB.
        // They land at the same fixed addresses RunStage0 uses, so the next
        // Stage0 skips its own copies.  Issued after the MTE2_V pair so the
        // scan VF waits only for its own three loads; the next Stage0's
        // MTE2_V pair covers these copies in pipe order.
        if (nextHead >= 0) {
            auto kUb = UbBytes(KDA_FINALIZE_UB_K).template ReinterpretCast<bfloat16_t>();
            auto gkUb = UbBytes(KDA_FINALIZE_UB_EXP2_GK).template ReinterpretCast<float>();
            auto hUb = UbBytes(80 * 1024).template ReinterpretCast<bfloat16_t>();
            auto dhUb = UbBytes(112 * 1024).template ReinterpretCast<bfloat16_t>();
            const int64_t nextToken =
                FinalizeTokenOffset<FULL_TILE>(*tiling_, nextChunk, nextHead, KDA_FINALIZE_DIM);
            const uint32_t nextElems =
                (FULL_TILE ? KDA_FINALIZE_CHUNK : nextChunk.validRows) * KDA_FINALIZE_DIM;
            AscendC::DataCopy(kUb, k_[nextToken], nextElems);
            AscendC::DataCopy(gkUb, gk_[nextToken], nextElems);
            AscendC::DataCopy(hUb, h_[FinalizeHOffset<FULL_TILE>(*tiling_, nextChunk, nextHead)],
                KDA_FINALIZE_STATE_ELEMS);
            AscendC::DataCopy(dhUb, dh_[FinalizeDhOffset<FULL_TILE>(*tiling_, nextChunk, nextHead)],
                KDA_FINALIZE_STATE_ELEMS);
            prefetchedHead_ = nextHead;
            prefetchedChunkTask_ = nextChunkTask;
        }
        FinalizeStage11VF<DTYPE_A_LOG, FULL_TILE>(
            reinterpret_cast<__ubuf__ float *>(dg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(bias.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_A_LOG *>(log.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(da.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(db.GetPhyAddr()), chunk.validRows, tiling_->lowerBound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        const int64_t partial = head * tiling_->totalChunkNum + chunkTask;
        AscendC::DataCopy(dG_[token], dg, elems);
        AscendC::DataCopyPad(workspace_[tiling_->gatePartialOffset].template ReinterpretCast<float>()[partial * 8], da,
            AscendC::DataCopyExtParams{1, sizeof(float), 0, 0, 0});
        AscendC::DataCopy(workspace_[tiling_->dtBiasPartialOffset].template ReinterpretCast<float>()[partial * 128], db, 128);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> UbBytes(uint32_t offset)
    {
        return ub_[offset];
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L1Bf16(uint32_t offset)
    {
        AscendC::LocalTensor<uint8_t> l1(AscendC::TPosition::A1, 0, 512 * 1024);
        return l1[offset].template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline void RunTzaResidual(uint32_t owner, uint32_t slot)
    {
        // StatePre egress and the following BaseFinalize loads share this UB.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_ZV_FREE_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZW_READY_BASE + slot);
        auto raw = UbBytes(0).template ReinterpretCast<float>();
        auto low = UbBytes(16 * 1024).template ReinterpretCast<bfloat16_t>();
        FinalizeResidualVF(reinterpret_cast<__ubuf__ bfloat16_t *>(low.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()), 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopy(L1Bf16(416 * 1024 + (owner * 2 + 1) * KDA_FINALIZE_MATRIX_BF16_BYTES),
            low, KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_KE_READY_BASE + slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
    }

    __aicore__ inline void RunDwResidual(uint32_t owner, uint32_t slot)
    {
        // Hold the next phase's MTE2 credit until the overlapping raw UB is consumed.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZB_READY_BASE + slot);
        auto raw = UbBytes(64 * 1024).template ReinterpretCast<float>();
        auto low = UbBytes(96 * 1024).template ReinterpretCast<bfloat16_t>();
        FinalizeDwResidualVF(reinterpret_cast<__ubuf__ bfloat16_t *>(low.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopy(L1Bf16(64 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES), low, 8192);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_KE_READY_BASE + slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
    }

    __aicore__ inline void RunStage0(
        const FinalizeChunkInfo &chunk, int64_t chunkTask, int64_t head, uint32_t owner,
        uint32_t slot)
    {
        // Stage0 reinterprets the same physical UB range used by the previous
        // work task's zB source.  Do not let MTE2 overwrite it until MTE3 has
        // finished the UB->L1 handoff.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        // All heads on this AIV share the Stage0 UB layout.  Do not let this
        // head's MTE2/V overwrite the preceding head while its MTE3
        // L1 egress is still reading that layout.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        auto kUb = UbBytes(KDA_FINALIZE_UB_K).template ReinterpretCast<bfloat16_t>();
        auto gkUb = UbBytes(KDA_FINALIZE_UB_EXP2_GK).template ReinterpretCast<float>();
        auto hUb = UbBytes(80 * 1024).template ReinterpretCast<bfloat16_t>();
        auto dhUb = UbBytes(112 * 1024).template ReinterpretCast<bfloat16_t>();
        auto kENd = UbBytes(144 * 1024).template ReinterpretCast<bfloat16_t>();
        auto rH = UbBytes(209 * 1024).template ReinterpretCast<float>();
        auto lowNd = UbBytes(224 * 1024).template ReinterpretCast<bfloat16_t>();

        const int64_t token = FinalizeTokenOffset<FULL_TILE>(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t hState = FinalizeHOffset<FULL_TILE>(*tiling_, chunk, head);
        const int64_t state = FinalizeDhOffset<FULL_TILE>(*tiling_, chunk, head);
        // The previous task's Stage11 prefetched these four inputs into the
        // same fixed UB addresses when it resolved this (chunkTask, head).
        // Consume the marker unconditionally so a stale one cannot linger.
        const bool prefetched = head == prefetchedHead_ && chunkTask == prefetchedChunkTask_;
        prefetchedHead_ = -1;
        prefetchedChunkTask_ = -1;
        if (!prefetched) {
            const uint32_t loadElems =
                (FULL_TILE ? KDA_FINALIZE_CHUNK : chunk.validRows) * KDA_FINALIZE_DIM;
            AscendC::DataCopy(kUb, k_[token], loadElems);
            AscendC::DataCopy(gkUb, gk_[token], loadElems);
            AscendC::DataCopy(hUb, h_[hState], KDA_FINALIZE_STATE_ELEMS);
            AscendC::DataCopy(dhUb, dh_[state], KDA_FINALIZE_STATE_ELEMS);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage0VF<FULL_TILE>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(kENd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(lowNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(rH.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gkUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(hUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(dhUb.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);

        // Direct UB->L1 egress for Stage1.  The fixed destination is the
        // corresponding owner in the two-head kE resident window.
        auto kEL1 = L1Bf16(96 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        AscendC::DataCopy(kEL1, kENd, 8192);
        AscendC::DataCopy(L1Bf16(128 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES), lowNd, 8192);
        // Only after all Stage0 uses of these physical ranges are drained may
        // AIC overwrite them with zV/zW.
    }

    __aicore__ inline void RunStage2(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner, uint32_t slot)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZV_READY_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZW_READY_BASE + slot);
        auto zV = UbBytes(KDA_FINALIZE_UB_ZV + slot * KDA_FINALIZE_MATRIX_FP32_BYTES)
                      .template ReinterpretCast<float>();
        auto zW = UbBytes(KDA_FINALIZE_UB_ZW + slot * KDA_FINALIZE_MATRIX_FP32_BYTES)
                      .template ReinterpretCast<float>();
        auto betaUb = UbBytes(KDA_FINALIZE_UB_BETA).template ReinterpretCast<DTYPE_BETA>();
        auto zBNd = UbBytes(KDA_FINALIZE_UB_WORK).template ReinterpretCast<bfloat16_t>();
        // q is retained at 144 KiB from here through the band loop (Stage5
        // qPos and Stage7/9).  Stage0's kENd L1 egress above is already
        // drained by this function's entry credit.
        auto qUb = UbBytes(KDA_FINALIZE_UB_Q).template ReinterpretCast<bfloat16_t>();
        const int64_t betaOffset = FinalizeTokenOffset<FULL_TILE>(*tiling_, chunk, head, 1);
        const int64_t token = FinalizeTokenOffset<FULL_TILE>(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        if constexpr (FULL_TILE) {
            // 64 beta values are 128B/256B aligned; no tail padding exists.
            AscendC::DataCopy(betaUb, beta_[betaOffset], KDA_FINALIZE_CHUNK);
        } else {
            // beta is a scalar per token.  The last chunk is not necessarily
            // 32-byte aligned. DMA pads only to a block boundary; Stage2 masks
            // inactive beta lanes. Padding all the way to 64 can exceed the
            // instruction's per-side padding limit on a short tail.
            AscendC::DataCopyExtParams betaCopy{
                1, static_cast<uint32_t>(chunk.validRows * sizeof(DTYPE_BETA)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DTYPE_BETA> betaPad{
                true, 0,
                0,
                static_cast<DTYPE_BETA>(0)};
            AscendC::DataCopyPad(betaUb, beta_[betaOffset], betaCopy, betaPad);
        }
        AscendC::DataCopy(qUb, q_[token],
            (FULL_TILE ? KDA_FINALIZE_CHUNK : chunk.validRows) * KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage2VF<DTYPE_BETA, FULL_TILE>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(zBNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(zV.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(zW.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(betaUb.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        auto zBL1 = L1Bf16(160 * 1024 + owner * 2 * KDA_FINALIZE_MATRIX_BF16_BYTES);
        if (zBPublishCount_[slot] != 0U) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                KDA_FINALIZE_ZB_FREE_BASE + slot);
        }
        AscendC::DataCopy(zBL1, zBNd, 2 * KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_ZB_READY_BASE + slot);
        ++zBPublishCount_[slot];
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
    }

    __aicore__ inline void RunStage5(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, uint32_t rowBegin)
    {
        if (rowBegin == 0) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
                KDA_FINALIZE_DAKK_READY_BASE + slot);
        }
        // Protect this ping/pong egress from its previous MTE3 reader while
        // allowing the other slot's MTE3 to overlap the current VF.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        // Retained q/k/g/beta stay disjoint from Stage5 outputs.
        // Keep the phase credit protecting reuse in the next head window.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);

        auto dAkkRaw = UbBytes(KDA_FINALIZE_UB_DAKK_RAW).template ReinterpretCast<float>();
        // Two-head windows give each AIV one head. The MTE3 handoff drains
        // before this phase is reused, so one FP32 egress region suffices.
        auto dAqkBf16 = UbBytes(16 * 1024).template ReinterpretCast<FinalizeLocalType>();
        auto dAkkNd = UbBytes(32 * 1024).template ReinterpretCast<FinalizeLocalType>();
        auto kNegNd = UbBytes(48 * 1024).template ReinterpretCast<FinalizeLocalType>();
        auto qPosNd = UbBytes(80 * 1024).template ReinterpretCast<FinalizeLocalType>();
        auto bkPosNd = UbBytes(112 * 1024).template ReinterpretCast<FinalizeLocalType>();
        auto q = UbBytes(KDA_FINALIZE_UB_Q).template ReinterpretCast<bfloat16_t>();
        auto k = UbBytes(KDA_FINALIZE_UB_K).template ReinterpretCast<bfloat16_t>();
        auto g = UbBytes(KDA_FINALIZE_UB_EXP2_GK).template ReinterpretCast<float>();
        auto beta = UbBytes(KDA_FINALIZE_UB_BETA).template ReinterpretCast<DTYPE_BETA>();
        auto dAqkFp32 = UbBytes(216 * 1024).template ReinterpretCast<float>();

        const int64_t matrixToken =
            FinalizeTokenOffset<FULL_TILE>(*tiling_, chunk, head, KDA_FINALIZE_CHUNK);
        const uint32_t matrixElems =
            (FULL_TILE ? KDA_FINALIZE_CHUNK : static_cast<uint32_t>(chunk.validRows)) *
            KDA_FINALIZE_CHUNK;
        if (rowBegin == 0) {
            AscendC::DataCopy(dAqkFp32, dAqk_[matrixToken], matrixElems);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        // Persistent center slot: Stage5(b) -> Stage7/9(b), disjoint from rH
        // [209,209.5) KiB and diagonal [210,210.5) KiB.
        auto center = UbBytes(209 * 1024 + 512).template ReinterpretCast<float>();
        const uint32_t rowEnd = FinalizeMin(
            rowBegin + KDA_FINALIZE_INTRA_ROWS,
            FULL_TILE ? KDA_FINALIZE_CHUNK : chunk.validRows);
        FinalizeIntraCenterVF<FULL_TILE>(
            reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()) + rowBegin * 128,
            reinterpret_cast<__ubuf__ float *>(center.GetPhyAddr()),
            rowEnd - rowBegin);
        FinalizeStage5VF<DTYPE_BETA, FULL_TILE>(
            reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAkkNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ FinalizeLocalType *>(kNegNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ FinalizeLocalType *>(qPosNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ FinalizeLocalType *>(bkPosNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dAkkRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(center.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows), rowBegin, rowEnd);
        if (rowBegin == 0) {
        FinalizeStage5DaqkVF<FULL_TILE>(
            reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAqkBf16.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dAqkFp32.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(UbBytes(210 * 1024).GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        }
        if constexpr (!FULL_TILE) {
        if (chunk.validRows < KDA_FINALIZE_CHUNK) {
            FinalizeStage5TailVF(
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAqkBf16.GetPhyAddr()),
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAkkNd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(kNegNd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(qPosNd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(bkPosNd.GetPhyAddr()), chunk.validRows);
        }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        auto local = L1Bf16(
            KDA_FINALIZE_LOCAL_BASE + owner * KDA_FINALIZE_LOCAL_BYTES).template ReinterpretCast<FinalizeLocalType>();
        if (rowBegin == 0) {
        AscendC::DataCopy(
            local[KDA_FINALIZE_LOCAL_DAQK / sizeof(FinalizeLocalType)], dAqkBf16,
            2 * KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::DataCopy(
            local[KDA_FINALIZE_LOCAL_DAKK / sizeof(FinalizeLocalType)], dAkkNd,
            2 * KDA_FINALIZE_MATRIX_ELEMS);
        }
        AscendC::DataCopy(local[KDA_FINALIZE_LOCAL_K_NEG / sizeof(FinalizeLocalType)],
                          kNegNd, 2 * KDA_FINALIZE_VECTOR_ELEMS);
        AscendC::DataCopy(local[KDA_FINALIZE_LOCAL_Q_POS / sizeof(FinalizeLocalType)],
                          qPosNd, 2 * KDA_FINALIZE_VECTOR_ELEMS);
        AscendC::DataCopy(local[KDA_FINALIZE_LOCAL_BK_POS / sizeof(FinalizeLocalType)],
                          bkPosNd, 2 * KDA_FINALIZE_VECTOR_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_LOCAL_READY_BASE + slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
    }

    // StatePre and BaseFinalize for one 32-row band, fused between Stage5 and
    // the intra-band GEMM consumers.  dkBase/dqBase land at the same
    // [112,128)/[128,144) KiB addresses the band loop already used, dg is
    // written straight to its 216 KiB resident frame, and the band scalars
    // use band-local row indexing; the band consumer reads all of them
    // directly from UB on the same vector pipe.
    __aicore__ inline void RunStateBaseSlice(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner, uint32_t slot,
        int64_t coreIdx, uint64_t groupGeneration, uint32_t rowBegin)
    {
        const uint32_t rows =
            FULL_TILE ? KDA_FINALIZE_INTRA_ROWS
                      : FinalizeMin(KDA_FINALIZE_INTRA_ROWS, chunk.validRows - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t offset = rowBegin * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset<FULL_TILE>(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);

        auto dkState = UbBytes(112 * 1024).template ReinterpretCast<float>();
        auto dqRaw = UbBytes(128 * 1024).template ReinterpretCast<float>();
        // X sits in the other slot's result half, which this task's cube
        // Stage6/8 never writes.  X overlaps five pre-band transients
        // (zV/zW/dAqkBf16/dAkkNd/TzaResidual low), all drained by Stage5's
        // closing MTE3_V barrier and program order.
        auto dvb = UbBytes((2 - slot) * 16 * 1024).template ReinterpretCast<float>();
        auto v = UbBytes(64 * 1024 + slot * 32 * 1024).template ReinterpretCast<bfloat16_t>();
        auto dv = UbBytes(72 * 1024 + slot * 32 * 1024).template ReinterpretCast<bfloat16_t>();
        auto q = UbBytes(KDA_FINALIZE_UB_Q).template ReinterpretCast<bfloat16_t>();
        auto k = UbBytes(KDA_FINALIZE_UB_K).template ReinterpretCast<bfloat16_t>();
        auto g = UbBytes(KDA_FINALIZE_UB_EXP2_GK).template ReinterpretCast<float>();
        auto gkLast = g[(FULL_TILE ? KDA_FINALIZE_CHUNK - 1U
                                   : static_cast<uint32_t>(chunk.validRows - 1)) * KDA_FINALIZE_DIM];
        auto beta = UbBytes(KDA_FINALIZE_UB_BETA).template ReinterpretCast<DTYPE_BETA>();
        auto rH = UbBytes(209 * 1024).template ReinterpretCast<float>();
        auto gateState = UbBytes(210 * 1024 + 512).template ReinterpretCast<float>();
        auto dbV = UbBytes(211 * 1024).template ReinterpretCast<float>();
        auto dg = UbBytes(KDA_FINALIZE_UB_DG).template ReinterpretCast<float>();

        auto wsDkState = workspace_[ws + KDA_FINALIZE_WS_DK_STATE_RAW].template ReinterpretCast<float>();
        auto wsDvb = workspace_[ws + KDA_FINALIZE_WS_DVB].template ReinterpretCast<float>();
        auto wsDkg = workspace_[ws + KDA_FINALIZE_WS_DKGB_RAW].template ReinterpretCast<float>();

        // Drain Stage5's in-flight L1 publication before MTE2 reuses
        // [16,144) KiB and X/right: the closing MTE3_V barrier in RunStage5
        // only orders the vector pipe, so the MTE3->MTE2 wait is explicit
        // here, matching every other UB reinterpretation site.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        AscendC::DataCopy(dkState, wsDkState[offset], elems);
        AscendC::DataCopy(dvb, wsDvb[offset], elems);
        AscendC::DataCopy(dqRaw, dqRaw_[token + offset], elems);
        AscendC::DataCopy(v, v_[token + offset], elems);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage3VF<DTYPE_BETA, FULL_TILE>(
            reinterpret_cast<__ubuf__ float *>(dkState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dvb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(dv.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbV.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dqRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(v.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta[rowBegin].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gkLast.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(rH.GetPhyAddr()),
            tiling_->scale, static_cast<uint16_t>(chunk.validRows),
            static_cast<uint16_t>(rowBegin), static_cast<uint16_t>(rows));
        // The dv egress must drain before Stage8 may rewrite the right area.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopy(dv_[token + offset], dv, elems);
        // V->MTE2 protects the last Stage3 read of dVb in X before the
        // dKgb_raw DMA overwrites it.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_);
        AscendC::DataCopy(dvb, wsDkg[offset], elems);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage4VF<DTYPE_BETA, FULL_TILE>(
            reinterpret_cast<__ubuf__ float *>(dkState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dqRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dvb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dg[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta[rowBegin].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbV.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows),
            static_cast<uint16_t>(rowBegin), static_cast<uint16_t>(rows));
        // The band consumer reads the on-chip dqBase/dkBase/dbBase directly;
        // only the dv egress needs covering before Stage8's right rewrite.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
    }

    // Complete each band while its Cube results are resident. dg stays on
    // chip from the slice's band0 write through the Stage11 full-chunk
    // reverse scan.
    __aicore__ inline void RunIntraBandResults(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t slot, uint32_t rowBegin)
    {
        const uint32_t rows =
            FULL_TILE ? KDA_FINALIZE_INTRA_ROWS
                      : FinalizeMin(KDA_FINALIZE_INTRA_ROWS, chunk.validRows - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t offset = rowBegin * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset<FULL_TILE>(*tiling_, chunk, head, KDA_FINALIZE_DIM) + offset;
        auto result = UbBytes(slot * 32 * 1024).template ReinterpretCast<float>();
        auto left = result[KDA_FINALIZE_INTRA_ROWS * KDA_FINALIZE_DIM];
        auto right = UbBytes(64 * 1024 + slot * 32 * 1024).template ReinterpretCast<float>();
        // dg is written straight to its 216 KiB resident frame by the slice's
        // Stage4VF, band by band; it stays resident through Stage11.  The
        // slice runs before this stage on the same V pipe, so no load remains.
        auto dg = UbBytes(KDA_FINALIZE_UB_DG).template ReinterpretCast<float>();
        // dqBase/dkBase/dbBase are read straight from the slice's on-chip
        // products: same addresses and same bits the workspace mirror held.
        auto dqBase = UbBytes(128 * 1024).template ReinterpretCast<float>();
        auto dkBase = UbBytes(112 * 1024).template ReinterpretCast<float>();
        // dqOut reuses this slot's result area and dkOut the right area; both
        // cube handoffs are dead once Stage7/Stage9 have read them. The
        // DQ_LOCAL_FREE / KE_READY / mte3ToMte2_ credits drain the dq/dk
        // egress reads before the next band's cube writes these ranges.
        auto dqOut = UbBytes(slot * 32 * 1024).template ReinterpretCast<bfloat16_t>();
        auto dkOut = UbBytes(64 * 1024 + slot * 32 * 1024).template ReinterpretCast<bfloat16_t>();
        // The resident g frame holds the raw log gates; Stage7/9 recompute the
        // band factor inline with this band's persistent center slot.
        auto g = UbBytes(KDA_FINALIZE_UB_EXP2_GK).template ReinterpretCast<float>()[offset];
        auto center = UbBytes(209 * 1024 + 512).template ReinterpretCast<float>();
        auto q = UbBytes(KDA_FINALIZE_UB_Q).template ReinterpretCast<bfloat16_t>()[offset];
        auto k = UbBytes(KDA_FINALIZE_UB_K).template ReinterpretCast<bfloat16_t>()[offset];
        auto beta = UbBytes(KDA_FINALIZE_UB_BETA).template ReinterpretCast<DTYPE_BETA>()[rowBegin];
        auto diagonal = UbBytes(210 * 1024).template ReinterpretCast<float>()[rowBegin];
        auto dbBase = UbBytes(211 * 1024).template ReinterpretCast<float>();
        auto dbDelta = UbBytes(212 * 1024).template ReinterpretCast<float>();
        auto qRstd = UbBytes(213 * 1024).template ReinterpretCast<float>();
        auto kRstd = UbBytes(214 * 1024).template ReinterpretCast<float>();
        auto dbOut = UbBytes(215 * 1024).template ReinterpretCast<DTYPE_BETA>();
        if (tiling_->hasQkL2Norm != 0U) {
            if constexpr (FULL_TILE) {
                // 32 rstd scalars are 128B aligned; no tail padding exists.
                AscendC::DataCopy(qRstd, qRstd_[token / KDA_FINALIZE_DIM], KDA_FINALIZE_INTRA_ROWS);
                AscendC::DataCopy(kRstd, kRstd_[token / KDA_FINALIZE_DIM], KDA_FINALIZE_INTRA_ROWS);
            } else {
                const AscendC::DataCopyExtParams scalarCopy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
                const AscendC::DataCopyPadExtParams<float> scalarPad{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(qRstd, qRstd_[token / KDA_FINALIZE_DIM], scalarCopy, scalarPad);
                AscendC::DataCopyPad(kRstd, kRstd_[token / KDA_FINALIZE_DIM], scalarCopy, scalarPad);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        // Issue the cube handshake before draining MTE2 so the load latency
        // overlaps the wait for the cube result.
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_DQ_LOCAL_FREE_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_DQ_LOCAL_READY_BASE + slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage7VF<FULL_TILE>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(dqOut.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dg[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(result.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dqBase.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(center.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(qRstd.GetPhyAddr()), tiling_->hasQkL2Norm, rows,
            reinterpret_cast<__ubuf__ float *>(diagonal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()));
        // Drain the slice's dv egress out of the right area before KE_READY
        // lets cube Stage8 rewrite it.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_KE_READY_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZW_READY_BASE + slot);
        FinalizeStage9VF<DTYPE_BETA, FULL_TILE>(
            reinterpret_cast<__ubuf__ float *>(left.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(right.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbDelta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dg[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(center.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta.GetPhyAddr()), rows,
            reinterpret_cast<__ubuf__ float *>(diagonal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()));
        FinalizeStage10VF<DTYPE_BETA, FULL_TILE>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(dkOut.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(dbOut.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(left.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dkBase.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbDelta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbBase.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(kRstd.GetPhyAddr()), tiling_->hasQkL2Norm, rows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopy(dq_[token], dqOut, elems);
        AscendC::DataCopy(dk_[token], dkOut, elems);
        if constexpr (FULL_TILE) {
            // 32 dBeta scalars are 64B/128B aligned; no tail padding exists.
            AscendC::DataCopy(dBeta_[token / KDA_FINALIZE_DIM], dbOut, KDA_FINALIZE_INTRA_ROWS);
        } else {
            AscendC::DataCopyPad(dBeta_[token / KDA_FINALIZE_DIM], dbOut,
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(rows * sizeof(DTYPE_BETA)), 0, 0, 0});
        }
        // The next Stage5 uses these UB ranges. Preserve all output reads.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
    }


    AscendC::GlobalTensor<float> kRstd_;
    AscendC::GlobalTensor<float> rawG_, dtBias_, dG_;
    AscendC::GlobalTensor<DTYPE_A_LOG> aLog_;
    AscendC::GlobalTensor<bfloat16_t> dk_;
    AscendC::GlobalTensor<DTYPE_BETA> dBeta_;
    AscendC::GlobalTensor<bfloat16_t> q_;
    AscendC::GlobalTensor<bfloat16_t> k_;
    AscendC::GlobalTensor<bfloat16_t> v_;
    AscendC::GlobalTensor<float> gk_;
    AscendC::GlobalTensor<DTYPE_BETA> beta_;
    AscendC::GlobalTensor<bfloat16_t> h_;
    AscendC::GlobalTensor<bfloat16_t> dh_;
    AscendC::GlobalTensor<float> dAqk_;
    AscendC::GlobalTensor<float> dqRaw_;
    AscendC::GlobalTensor<float> qRstd_;
    AscendC::GlobalTensor<bfloat16_t> dq_;
    AscendC::GlobalTensor<bfloat16_t> dv_;
    AscendC::GlobalTensor<uint8_t> workspace_;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::TEventID mte2ToV_[KDA_FINALIZE_AIV_SLOTS];
    AscendC::TEventID vToMte3_[KDA_FINALIZE_AIV_SLOTS];
    AscendC::TEventID mte3ToMte2_[KDA_FINALIZE_AIV_SLOTS];
    AscendC::TEventID stage3Mte3ToV_;
    AscendC::TEventID stage3Mte3ToMte2_;
    AscendC::TEventID stage0Mte3ToMte2_;
    AscendC::TEventID stage5VToMte2_;
    AscendC::TEventID stage7Mte3ToMte2_;
    AscendC::TEventID gateVToMte2_;
    AscendC::TEventID stateVToMte2_;
    uint32_t zBPublishCount_[KDA_FINALIZE_AIV_SLOTS];
    uint32_t subBlockNum_ = KDA_FINALIZE_AIV_COUNT;
    uint32_t subBlockIdx_ = 0;
    // Stage0 inputs prefetched by the previous task's Stage11, identified by
    // (chunkTask, head); -1 when the next Stage0 must load its own inputs.
    int64_t prefetchedHead_ = -1;
    int64_t prefetchedChunkTask_ = -1;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H
