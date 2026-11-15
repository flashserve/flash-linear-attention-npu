/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH92_REGBASE_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH92_REGBASE_H

#include <type_traits>
#include "chunk_kda_bwd_recompute_common.h"

namespace KdaBwdRecomputeArch92 {
using namespace AscendC::MicroAPI;

// Same K-contiguous 0:63 / 64:127 register pairing as arch35. Do not use
// even/odd deinterleave for the gk last row.
constexpr CastTrait kCastToFp32 = {
    RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
constexpr CastTrait kCastFromFp32 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, AscendC::RoundMode::CAST_ROUND};
constexpr CastTrait kCastFromFp32One = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};
constexpr CastTrait kCastToFp32One = {
    RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};

template <typename T>
__simd_callee__ inline void LoadFp32(RegTensor<float> &out, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadAlign(out, src);
    } else {
        RegTensor<T> raw;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        Cast<float, T, kCastToFp32>(out, raw, mask);
    }
}

template <typename T>
__simd_callee__ inline void StoreBf16(__ubuf__ T *dst, RegTensor<float> &src, MaskReg &mask)
{
    RegTensor<T> out;
    Cast<T, float, kCastFromFp32>(out, src, mask);
    StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, out, mask);
}

__simd_callee__ inline void ExpPair(RegTensor<float> &lo, RegTensor<float> &hi, MaskReg &mask)
{
    // BF16 outputs keep the full fp32 exponent range, so the half-precision
    // Exp approximation (which saturates below exp(-11) and cannot represent
    // values under ~6e-8) is not usable here: it turns every deeply decayed
    // qg/kg element into a clamped constant and blows up the small-value
    // absolute-error count. Use the fp32 Exp for both lanes.
    Exp(lo, lo, mask);
    Exp(hi, hi, mask);
}

template <typename GateT, bool USE_GATE, bool USE_EXP2, uint16_t FIXED_ROWS>
__simd_callee__ inline void GateCumsum(
    __ubuf__ float *gk, __ubuf__ GateT *g, __ubuf__ float *bias, __ubuf__ float *aLog,
    uint16_t rows, float lowerBound, bool hasBias, bool hasALog,
    RegTensor<float> &lastLo, RegTensor<float> &lastHi)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> accLo, accHi, biasLo, biasHi, negExpA, numerator;
    Duplicate(accLo, 0.0f, mask);
    Duplicate(accHi, 0.0f, mask);
    constexpr float scale = USE_EXP2 ? KDA_BWD_RECOMPUTE_RCP_LN2 : 1.0f;
    if constexpr (USE_GATE) {
        Duplicate(numerator, lowerBound * scale, mask);
        if (hasBias) {
            LoadFp32(biasLo, bias, mask);
            LoadFp32(biasHi, bias + 64, mask);
        }
        if (hasALog) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(negExpA, aLog);
            Exp(negExpA, negExpA, mask);
            Muls(negExpA, negExpA, -1.0f, mask);
        } else {
            Duplicate(negExpA, -1.0f, mask);
        }
    }
    const uint16_t count = FIXED_ROWS == 0 ? rows : FIXED_ROWS;
    // Static trip count: a runtime-bounded VF loop makes the A6 compiler emit
    // an instruction the simulator cannot decode (Undef ISA 0x80000007).
    for (uint16_t row = 0; row < kRows; ++row) {
        if (row >= count) {
            continue;
        }
        const uint32_t offset = static_cast<uint32_t>(row) * kChannels;
        RegTensor<float> lo, hi;
        // In particular FP32 g is read from IN, never the uninitialized OUT.
        LoadFp32(lo, g + offset, mask);
        LoadFp32(hi, g + offset + 64, mask);
        if constexpr (USE_GATE) {
            if (hasBias) {
                Add(lo, lo, biasLo, mask);
                Add(hi, hi, biasHi, mask);
            }
            Mul(lo, lo, negExpA, mask);
            Mul(hi, hi, negExpA, mask);
            Exp(lo, lo, mask);
            Exp(hi, hi, mask);
            Adds(lo, lo, 1.0f, mask);
            Adds(hi, hi, 1.0f, mask);
            Div(lo, numerator, lo, mask);
            Div(hi, numerator, hi, mask);
        } else if constexpr (USE_EXP2) {
            Muls(lo, lo, scale, mask);
            Muls(hi, hi, scale, mask);
        }
        Add(accLo, accLo, lo, mask);
        Add(accHi, accHi, hi, mask);
        StoreAlign(gk + offset, accLo, mask);
        StoreAlign(gk + offset + 64, accHi, mask);
    }
    Adds(lastLo, accLo, 0.0f, mask);
    Adds(lastHi, accHi, 0.0f, mask);
}

template <typename T, typename BetaT, bool USE_GATE, bool USE_EXP2, uint16_t FIXED_ROWS>
__simd_callee__ inline void ProduceResults(
    __ubuf__ float *gk, __ubuf__ BetaT *beta, __ubuf__ T *q, __ubuf__ T *k, __ubuf__ T *v,
    __ubuf__ T *qg, __ubuf__ T *kbg, __ubuf__ T *kg, __ubuf__ T *vb,
    uint16_t rows, RegTensor<float> &lastLo, RegTensor<float> &lastHi)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    const uint16_t count = FIXED_ROWS == 0 ? rows : FIXED_ROWS;
    constexpr float expScale = USE_EXP2 ? KDA_BWD_RECOMPUTE_LN2 : 1.0f;
    for (uint16_t row = 0; row < kRows; ++row) {
        if (row >= count) {
            continue;
        }
        const uint32_t offset = static_cast<uint32_t>(row) * kChannels;
        RegTensor<float> gateLo, gateHi, expLo, expHi, deltaLo, deltaHi;
        RegTensor<float> qLo, qHi, kLo, kHi, vLo, vHi, betaReg, outLo, outHi;
        if constexpr (std::is_same<BetaT, float>::value) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        } else {
            RegTensor<BetaT> raw;
            LoadAlign<BetaT, LoadDist::DIST_BRC_B16>(raw, beta + row);
            Cast<float, BetaT, kCastToFp32>(betaReg, raw, mask);
        }
        LoadFp32(gateLo, gk + offset, mask);
        LoadFp32(gateHi, gk + offset + 64, mask);
        LoadFp32(qLo, q + offset, mask);
        LoadFp32(qHi, q + offset + 64, mask);
        LoadFp32(kLo, k + offset, mask);
        LoadFp32(kHi, k + offset + 64, mask);
        LoadFp32(vLo, v + offset, mask);
        LoadFp32(vHi, v + offset + 64, mask);
        Muls(expLo, gateLo, expScale, mask);
        Muls(expHi, gateHi, expScale, mask);
        ExpPair(expLo, expHi, mask);
        Mul(outLo, qLo, expLo, mask);
        Mul(outHi, qHi, expHi, mask);
        StoreBf16(qg + offset, outLo, mask);
        StoreBf16(qg + offset + 64, outHi, mask);
        Mul(outLo, kLo, expLo, mask);
        Mul(outHi, kHi, expHi, mask);
        Mul(outLo, outLo, betaReg, mask);
        Mul(outHi, outHi, betaReg, mask);
        StoreBf16(kbg + offset, outLo, mask);
        StoreBf16(kbg + offset + 64, outHi, mask);
        Sub(deltaLo, lastLo, gateLo, mask);
        Sub(deltaHi, lastHi, gateHi, mask);
        Muls(deltaLo, deltaLo, expScale, mask);
        Muls(deltaHi, deltaHi, expScale, mask);
        ExpPair(deltaLo, deltaHi, mask);
        Mul(outLo, kLo, deltaLo, mask);
        Mul(outHi, kHi, deltaHi, mask);
        StoreBf16(kg + offset, outLo, mask);
        StoreBf16(kg + offset + 64, outHi, mask);
        Mul(outLo, vLo, betaReg, mask);
        Mul(outHi, vHi, betaReg, mask);
        StoreBf16(vb + offset, outLo, mask);
        StoreBf16(vb + offset + 64, outHi, mask);
    }
    // Padding is assigned directly, before MMAD, without evaluating exp on
    // uninitialized gate rows. Every L1 RHS byte loaded by C1 is defined.
    RegTensor<float> zero;
    Duplicate(zero, 0.0f, mask);
    for (uint16_t row = 0; row < kRows; ++row) {
        if (row < count) {
            continue;
        }
        const uint32_t offset = static_cast<uint32_t>(row) * kChannels;
        StoreBf16(kbg + offset, zero, mask);
        StoreBf16(kbg + offset + 64, zero, mask);
        StoreBf16(vb + offset, zero, mask);
        StoreBf16(vb + offset + 64, zero, mask);
    }
}

template <typename T, typename GateT, typename BetaT, bool USE_GATE, bool USE_EXP2>
static __simd_vf__ inline void RecomputeFullVf(
    __ubuf__ float *gk, __ubuf__ GateT *g, __ubuf__ float *bias, __ubuf__ float *aLog,
    __ubuf__ BetaT *beta, __ubuf__ T *q, __ubuf__ T *k, __ubuf__ T *v,
    __ubuf__ T *qg, __ubuf__ T *kbg, __ubuf__ T *kg, __ubuf__ T *vb,
    float lowerBound, bool hasBias, bool hasALog)
{
    RegTensor<float> lastLo, lastHi;
    GateCumsum<GateT, USE_GATE, USE_EXP2, 64>(
        gk, g, bias, aLog, 64, lowerBound, hasBias, hasALog, lastLo, lastHi);
    ProduceResults<T, BetaT, USE_GATE, USE_EXP2, 64>(
        gk, beta, q, k, v, qg, kbg, kg, vb, 64, lastLo, lastHi);
}

template <typename GateT, bool USE_GATE, bool USE_EXP2>
static __simd_vf__ inline void GateTailVf(
    __ubuf__ float *gk, __ubuf__ GateT *g, __ubuf__ float *bias, __ubuf__ float *aLog,
    uint16_t rows, float lowerBound, bool hasBias, bool hasALog)
{
    RegTensor<float> lastLo, lastHi;
    GateCumsum<GateT, USE_GATE, USE_EXP2, 0>(
        gk, g, bias, aLog, rows, lowerBound, hasBias, hasALog, lastLo, lastHi);
}

template <typename T, typename BetaT, bool USE_GATE, bool USE_EXP2>
static __simd_vf__ inline void ResultsTailVf(
    __ubuf__ float *gk, __ubuf__ BetaT *beta, __ubuf__ T *q, __ubuf__ T *k, __ubuf__ T *v,
    __ubuf__ T *qg, __ubuf__ T *kbg, __ubuf__ T *kg, __ubuf__ T *vb, uint16_t rows)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> lastLo, lastHi;
    const uint32_t offset = (rows - 1) * kChannels;
    LoadFp32(lastLo, gk + offset, mask);
    LoadFp32(lastHi, gk + offset + 64, mask);
    ProduceResults<T, BetaT, USE_GATE, USE_EXP2, 0>(
        gk, beta, q, k, v, qg, kbg, kg, vb, rows, lastLo, lastHi);
}

} // namespace KdaBwdRecomputeArch92

#endif
