/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Vector VF used by fused chunk_gdn_fwd_prepare.
 */

#ifndef CHUNK_GDN_FWD_PREPARE_VF_H
#define CHUNK_GDN_FWD_PREPARE_VF_H

#include "kernel_operator.h"
#include "common.h"

namespace GdnStage {
using namespace AscendC;
using namespace AscendC::MicroAPI;

namespace {
constexpr int32_t VL = 64;

constexpr CastTrait kCastB162B32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN,
};
constexpr CastTrait kCastB322B16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT,
};
} // namespace

template <typename T>
__aicore__ inline void LoadCastB16(__ubuf__ T *src, RegTensor<float> &dst, MaskReg &mask)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src);
    } else {
        RegTensor<T> xB16;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xB16, src);
        Cast<float, T, kCastB162B32>(dst, xB16, mask);
    }
}

template <typename T>
__aicore__ inline float ScalarToFp32(T x)
{
    if constexpr (IsSameType<T, float>::value) {
        return x;
    } else if constexpr (IsSameType<T, half>::value) {
        return static_cast<float>(x);
    } else {
        return ToFloat(x);
    }
}

template <typename T>
__aicore__ inline void StoreCastB16(__ubuf__ T *dst, RegTensor<float> &src, MaskReg &mask)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign(dst, src, mask);
    } else {
        RegTensor<T> yB16;
        Cast<T, float, kCastB322B16>(yB16, src, mask);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, yB16, mask);
    }
}

// Inclusive prefix sum of n fp32 values, then * scale (RCP_LN2).
// Loads a 32B group (8 floats) first so stores cannot alias the next load;
// no per-element LocalMemBar. Duplicate(scale) is hoisted. Unroll-8 dual-issues
// independent DIST_BRC loads. Tail uses for(nTail) instead of if.
__aicore__ inline void ChunkCumsumScaleVF(LocalTensor<float> &x, float scale, uint32_t n)
{
    __ubuf__ float *addr = (__ubuf__ float *)x.GetPhyAddr();
    const uint16_t n16 = static_cast<uint16_t>(n);
    const uint16_t n8 = n16 >> 3;
    const uint16_t nTail = n16 & 7;
    __VEC_SCOPE__
    {
        MaskReg preg1 = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> acc, s, v0, v1, v2, v3, v4, v5, v6, v7;
        Duplicate(acc, 0.0f, preg1);
        Duplicate(s, scale, preg1);
        for (uint16_t b = 0; b < n8; b++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v0, addr + b * 8);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v1, addr + b * 8 + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v2, addr + b * 8 + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v3, addr + b * 8 + 3);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v4, addr + b * 8 + 4);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v5, addr + b * 8 + 5);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v6, addr + b * 8 + 6);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v7, addr + b * 8 + 7);
            Add(acc, acc, v0, preg1);
            Mul(v0, acc, s, preg1);
            Add(acc, acc, v1, preg1);
            Mul(v1, acc, s, preg1);
            Add(acc, acc, v2, preg1);
            Mul(v2, acc, s, preg1);
            Add(acc, acc, v3, preg1);
            Mul(v3, acc, s, preg1);
            Add(acc, acc, v4, preg1);
            Mul(v4, acc, s, preg1);
            Add(acc, acc, v5, preg1);
            Mul(v5, acc, s, preg1);
            Add(acc, acc, v6, preg1);
            Mul(v6, acc, s, preg1);
            Add(acc, acc, v7, preg1);
            Mul(v7, acc, s, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + b * 8, v0, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + b * 8 + 1, v1, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + b * 8 + 2, v2, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + b * 8 + 3, v3, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + b * 8 + 4, v4, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + b * 8 + 5, v5, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + b * 8 + 6, v6, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + b * 8 + 7, v7, preg1);
        }
        for (uint16_t t = 0; t < nTail; t++) {
            const uint32_t off = static_cast<uint32_t>(n8) * 8 + t;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v0, addr + off);
            Add(acc, acc, v0, preg1);
            Mul(v0, acc, s, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + off, v0, preg1);
        }
    }
}

// fused gate: g_raw = -exp(a_log) * softplus(g + dt_bias)
// Stable softplus: relu(x) + ln(1 + exp(-|x|)). a_log is broadcast per head.
template <typename T>
__aicore__ inline void GateSoftplusVF(LocalTensor<T> &gIn, LocalTensor<float> &gRaw,
                                      float aLog, float dtBias, uint32_t n)
{
    __ubuf__ T *src = (__ubuf__ T *)gIn.GetPhyAddr();
    __ubuf__ float *dst = (__ubuf__ float *)gRaw.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t sreg = n;
        MaskReg preg = UpdateMask<float>(sreg);
        RegTensor<float> x, ax, nx, e, sp, reluX, negA, out, zero;
        LoadCastB16<T>(src, x, preg);
        Adds(x, x, dtBias, preg);
        Abs(ax, x, preg);
        Muls(nx, ax, -1.0f, preg);
        Exp(e, nx, preg);
        Adds(e, e, 1.0f, preg);
        Ln(sp, e, preg);
        Duplicate(zero, 0.0f, preg);
        Max(reluX, x, zero, preg);
        Add(sp, sp, reluX, preg);
        Duplicate(negA, aLog, preg);
        Exp(negA, negA, preg);
        Muls(negA, negA, -1.0f, preg);
        Mul(out, sp, negA, preg);
        StoreAlign(dst, out, preg);
    }
}

template <typename T>
__aicore__ inline void CopyToFp32VF(LocalTensor<T> &src, LocalTensor<float> &dst, uint32_t n)
{
    __ubuf__ T *s = (__ubuf__ T *)src.GetPhyAddr();
    __ubuf__ float *d = (__ubuf__ float *)dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t sreg = n;
        MaskReg preg = UpdateMask<float>(sreg);
        RegTensor<float> x;
        LoadCastB16<T>(s, x, preg);
        StoreAlign(d, x, preg);
    }
}

// sigmoid or 2*sigmoid
template <typename T>
__aicore__ inline void BetaSigmoidVF(LocalTensor<T> &betaIn, LocalTensor<float> &betaOut,
                                     float scale, uint32_t n)
{
    __ubuf__ T *src = (__ubuf__ T *)betaIn.GetPhyAddr();
    __ubuf__ float *dst = (__ubuf__ float *)betaOut.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t sreg = n;
        MaskReg preg = UpdateMask<float>(sreg);
        RegTensor<float> x, e, one, den, s;
        LoadCastB16<T>(src, x, preg);
        Muls(x, x, -1.0f, preg);
        Exp(e, x, preg);
        Duplicate(one, 1.0f, preg);
        Add(den, one, e, preg);
        Div(s, one, den, preg);
        Muls(s, s, scale, preg);
        StoreAlign(dst, s, preg);
    }
}

// Writes -L: -mask * beta * kkt * exp2(clip(g[i]-g[j], -50, 50)).
// mask is 64x64 fp32 strict-lower (1 if i>j else 0). Pack and L1 both consume -L,
// so the two later Muls(-1) are not needed.
// VF handbook: Duplicate/gJ hoisted; unroll 2 so independent row chains dual-issue.
// Rows are 64 fp32 / 256 B aligned — no unaligned Load path.
__aicore__ inline void ConstructLowerLExp2VF(LocalTensor<float> &kkt, LocalTensor<float> &g,
                                             LocalTensor<float> &beta, LocalTensor<float> &mask,
                                             LocalTensor<float> &L, uint32_t n)
{
    constexpr float kLn2 = 0.6931471825f;
    __ubuf__ float *kktAddr = (__ubuf__ float *)kkt.GetPhyAddr();
    __ubuf__ float *gAddr = (__ubuf__ float *)g.GetPhyAddr();
    __ubuf__ float *betaAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *maskAddr = (__ubuf__ float *)mask.GetPhyAddr();
    __ubuf__ float *lAddr = (__ubuf__ float *)L.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> gJ, gI0, gI1, kkt0, kkt1, d0, d1, gate0, gate1;
        RegTensor<float> b0, b1, out0, out1, m0, m1, lo, hi;
        Duplicate(lo, -kGdnGateClip, pregAll);
        Duplicate(hi, kGdnGateClip, pregAll);
        LoadAlign(gJ, gAddr);
        for (uint16_t i = 0; i < 32; i++) {
            const uint32_t r0 = static_cast<uint32_t>(i) * 128;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(gI0, gAddr + static_cast<uint32_t>(i) * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(gI1, gAddr + static_cast<uint32_t>(i) * 2 + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(b0, betaAddr + static_cast<uint32_t>(i) * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(b1, betaAddr + static_cast<uint32_t>(i) * 2 + 1);
            LoadAlign(kkt0, kktAddr + r0);
            LoadAlign(kkt1, kktAddr + r0 + 64);
            LoadAlign(m0, maskAddr + r0);
            LoadAlign(m1, maskAddr + r0 + 64);
            Sub(d0, gI0, gJ, pregAll);
            Sub(d1, gI1, gJ, pregAll);
            Max(d0, d0, lo, pregAll);
            Max(d1, d1, lo, pregAll);
            Min(d0, d0, hi, pregAll);
            Min(d1, d1, hi, pregAll);
            Muls(d0, d0, kLn2, pregAll);
            Muls(d1, d1, kLn2, pregAll);
            Exp(gate0, d0, pregAll);
            Exp(gate1, d1, pregAll);
            Mul(out0, kkt0, gate0, pregAll);
            Mul(out1, kkt1, gate1, pregAll);
            Mul(out0, out0, b0, pregAll);
            Mul(out1, out1, b1, pregAll);
            Mul(out0, out0, m0, pregAll);
            Mul(out1, out1, m1, pregAll);
            Muls(out0, out0, -1.0f, pregAll);
            Muls(out1, out1, -1.0f, pregAll);
            StoreAlign(lAddr + r0, out0, pregAll);
            StoreAlign(lAddr + r0 + 64, out1, pregAll);
        }
        (void)n;
    }
}

// y[t, :] = x[t, :] * scale[t]   last dim 128 (2 VL).
// VF handbook:
//   - Hardware Loop: uint16_t i, start 0, step 1; tail is for(nTail), not if
//   - hoist mask/regs (loop instruction dist)
//   - unroll 2 rows so independent Load/Mul/Store dual-issue
//   - addr = base + i * const for AddrReg; no pointer +=
//   - 256 B aligned rows — LoadAlign only, no LoadUnAlign
template <typename T>
__aicore__ inline void ScaleRowsK128VF(LocalTensor<T> &x, LocalTensor<T> &y,
                                       LocalTensor<float> &scale, uint32_t rows)
{
    constexpr uint32_t kRow = 2 * VL;
    constexpr uint32_t kPairElems = 2 * kRow;
    __ubuf__ T *xAddr = (__ubuf__ T *)x.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)y.GetPhyAddr();
    __ubuf__ float *sAddr = (__ubuf__ float *)scale.GetPhyAddr();
    const uint16_t nPair = static_cast<uint16_t>(rows >> 1);
    const uint16_t nTail = static_cast<uint16_t>(rows & 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> s0, s1, a0, b0, a1, b1, ya0, yb0, ya1, yb1;
        for (uint16_t p = 0; p < nPair; p++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s0, sAddr + p * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s1, sAddr + p * 2 + 1);
            LoadCastB16<T>(xAddr + p * kPairElems, a0, pregAll);
            LoadCastB16<T>(xAddr + p * kPairElems + VL, b0, pregAll);
            LoadCastB16<T>(xAddr + p * kPairElems + kRow, a1, pregAll);
            LoadCastB16<T>(xAddr + p * kPairElems + kRow + VL, b1, pregAll);
            Mul(ya0, a0, s0, pregAll);
            Mul(yb0, b0, s0, pregAll);
            Mul(ya1, a1, s1, pregAll);
            Mul(yb1, b1, s1, pregAll);
            StoreCastB16<T>(yAddr + p * kPairElems, ya0, pregAll);
            StoreCastB16<T>(yAddr + p * kPairElems + VL, yb0, pregAll);
            StoreCastB16<T>(yAddr + p * kPairElems + kRow, ya1, pregAll);
            StoreCastB16<T>(yAddr + p * kPairElems + kRow + VL, yb1, pregAll);
        }
        for (uint16_t t = 0; t < nTail; t++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s0, sAddr + nPair * 2);
            LoadCastB16<T>(xAddr + nPair * kPairElems, a0, pregAll);
            LoadCastB16<T>(xAddr + nPair * kPairElems + VL, b0, pregAll);
            Mul(ya0, a0, s0, pregAll);
            Mul(yb0, b0, s0, pregAll);
            StoreCastB16<T>(yAddr + nPair * kPairElems, ya0, pregAll);
            StoreCastB16<T>(yAddr + nPair * kPairElems + VL, yb0, pregAll);
        }
    }
}

// y[t, :] = x[t, :] * scale[t]   last dim 256 (4 VL).
// Same handbook rules as K128. Inner t is 2 VL pairs: addr p*const1 + t*const2.
template <typename T>
__aicore__ inline void ScaleRowsK256VF(LocalTensor<T> &x, LocalTensor<T> &y,
                                       LocalTensor<float> &scale, uint32_t rows)
{
    constexpr uint32_t kDim = 4 * VL;
    constexpr uint32_t kPairElems = 2 * kDim;
    constexpr uint32_t kVlPair = 2 * VL;
    __ubuf__ T *xAddr = (__ubuf__ T *)x.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)y.GetPhyAddr();
    __ubuf__ float *sAddr = (__ubuf__ float *)scale.GetPhyAddr();
    const uint16_t nPair = static_cast<uint16_t>(rows >> 1);
    const uint16_t nTail = static_cast<uint16_t>(rows & 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> s0, s1, a0, a1, b0, b1, ya0, ya1, yb0, yb1;
        for (uint16_t p = 0; p < nPair; p++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s0, sAddr + p * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s1, sAddr + p * 2 + 1);
            for (uint16_t t = 0; t < 2; t++) {
                LoadCastB16<T>(xAddr + p * kPairElems + t * kVlPair, a0, pregAll);
                LoadCastB16<T>(xAddr + p * kPairElems + t * kVlPair + VL, a1, pregAll);
                LoadCastB16<T>(xAddr + p * kPairElems + kDim + t * kVlPair, b0, pregAll);
                LoadCastB16<T>(xAddr + p * kPairElems + kDim + t * kVlPair + VL, b1, pregAll);
                Mul(ya0, a0, s0, pregAll);
                Mul(ya1, a1, s0, pregAll);
                Mul(yb0, b0, s1, pregAll);
                Mul(yb1, b1, s1, pregAll);
                StoreCastB16<T>(yAddr + p * kPairElems + t * kVlPair, ya0, pregAll);
                StoreCastB16<T>(yAddr + p * kPairElems + t * kVlPair + VL, ya1, pregAll);
                StoreCastB16<T>(yAddr + p * kPairElems + kDim + t * kVlPair, yb0, pregAll);
                StoreCastB16<T>(yAddr + p * kPairElems + kDim + t * kVlPair + VL, yb1, pregAll);
            }
        }
        for (uint16_t u = 0; u < nTail; u++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s0, sAddr + nPair * 2);
            for (uint16_t t = 0; t < 2; t++) {
                LoadCastB16<T>(xAddr + nPair * kPairElems + t * kVlPair, a0, pregAll);
                LoadCastB16<T>(xAddr + nPair * kPairElems + t * kVlPair + VL, a1, pregAll);
                Mul(ya0, a0, s0, pregAll);
                Mul(ya1, a1, s0, pregAll);
                StoreCastB16<T>(yAddr + nPair * kPairElems + t * kVlPair, ya0, pregAll);
                StoreCastB16<T>(yAddr + nPair * kPairElems + t * kVlPair + VL, ya1, pregAll);
            }
        }
    }
}

// kbg: y[t,:] = x[t,:] * beta[t] * exp2(g[t]), last dim 128.
// Fuse scale into the row VF: no ubGateTmp round-trip; independent row
// Exp chains dual-issue with the K loads.
template <typename T>
__aicore__ inline void ScaleRowsK128BetaExp2gVF(LocalTensor<T> &x, LocalTensor<T> &y,
                                                LocalTensor<float> &beta, LocalTensor<float> &g,
                                                uint32_t rows)
{
    constexpr float kLn2 = 0.6931471825f;
    constexpr uint32_t kRow = 2 * VL;
    constexpr uint32_t kPairElems = 2 * kRow;
    __ubuf__ T *xAddr = (__ubuf__ T *)x.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)y.GetPhyAddr();
    __ubuf__ float *bAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *gAddr = (__ubuf__ float *)g.GetPhyAddr();
    const uint16_t nPair = static_cast<uint16_t>(rows >> 1);
    const uint16_t nTail = static_cast<uint16_t>(rows & 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> g0, g1, b0, b1, k00, k01, k10, k11, y00, y01, y10, y11;
        for (uint16_t p = 0; p < nPair; p++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(g0, gAddr + p * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(g1, gAddr + p * 2 + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(b0, bAddr + p * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(b1, bAddr + p * 2 + 1);
            LoadCastB16<T>(xAddr + p * kPairElems, k00, pregAll);
            LoadCastB16<T>(xAddr + p * kPairElems + VL, k01, pregAll);
            LoadCastB16<T>(xAddr + p * kPairElems + kRow, k10, pregAll);
            LoadCastB16<T>(xAddr + p * kPairElems + kRow + VL, k11, pregAll);
            Muls(g0, g0, kLn2, pregAll);
            Muls(g1, g1, kLn2, pregAll);
            Exp(g0, g0, pregAll);
            Exp(g1, g1, pregAll);
            Mul(g0, g0, b0, pregAll);
            Mul(g1, g1, b1, pregAll);
            Mul(y00, k00, g0, pregAll);
            Mul(y01, k01, g0, pregAll);
            Mul(y10, k10, g1, pregAll);
            Mul(y11, k11, g1, pregAll);
            StoreCastB16<T>(yAddr + p * kPairElems, y00, pregAll);
            StoreCastB16<T>(yAddr + p * kPairElems + VL, y01, pregAll);
            StoreCastB16<T>(yAddr + p * kPairElems + kRow, y10, pregAll);
            StoreCastB16<T>(yAddr + p * kPairElems + kRow + VL, y11, pregAll);
        }
        for (uint16_t t = 0; t < nTail; t++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(g0, gAddr + nPair * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(b0, bAddr + nPair * 2);
            LoadCastB16<T>(xAddr + nPair * kPairElems, k00, pregAll);
            LoadCastB16<T>(xAddr + nPair * kPairElems + VL, k01, pregAll);
            Muls(g0, g0, kLn2, pregAll);
            Exp(g0, g0, pregAll);
            Mul(g0, g0, b0, pregAll);
            Mul(y00, k00, g0, pregAll);
            Mul(y01, k01, g0, pregAll);
            StoreCastB16<T>(yAddr + nPair * kPairElems, y00, pregAll);
            StoreCastB16<T>(yAddr + nPair * kPairElems + VL, y01, pregAll);
        }
    }
}

// y[t, :] = x[t, :] * scale[t]   last dim 256 (4 VL) or 128.
// Dispatch outside VEC_SCOPE so each path keeps constexpr strides.
template <typename T>
__aicore__ inline void ScaleRowsAlignedVF(LocalTensor<T> &x, LocalTensor<T> &y,
                                          LocalTensor<float> &scale, uint32_t rows, uint32_t dim)
{
    if (dim == 4 * VL) {
        ScaleRowsK256VF<T>(x, y, scale, rows);
        return;
    }
    ScaleRowsK128VF<T>(x, y, scale, rows);
}

// scale[t] = beta[t] * exp2(g[t])
__aicore__ inline void BetaExp2gVF(LocalTensor<float> &beta, LocalTensor<float> &g,
                                   LocalTensor<float> &out, uint32_t n)
{
    constexpr float kLn2 = 0.6931471825f;
    __ubuf__ float *bAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *gAddr = (__ubuf__ float *)g.GetPhyAddr();
    __ubuf__ float *oAddr = (__ubuf__ float *)out.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t sreg = n;
        MaskReg preg = UpdateMask<float>(sreg);
        RegTensor<float> gv, bv, e;
        LoadAlign(gv, gAddr);
        LoadAlign(bv, bAddr);
        Muls(gv, gv, kLn2, preg);
        Exp(e, gv, preg);
        Mul(e, e, bv, preg);
        StoreAlign(oAddr, e, preg);
    }
}

namespace Prepare {

// Strict-lower 64x64 fp32 mask: 1 if col < row, else 0.
// One VL is 64 fp32 on 950, so each row is one RegTensor.
// Unroll 2: independent Compare/Select/Store can dual-issue. Duplicate/Arange
// stay outside the loop. Address is ``dst + i * 128``.
__aicore__ inline void Stage0_ExpandBitMaskToFp32(AscendC::LocalTensor<float> ubMaskFp32)
{
    using namespace AscendC::MicroAPI;
    __ubuf__ float *dst = (__ubuf__ float *)ubMaskFp32.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> col, ones, zeros, row0, row1, idx0, idx1;
        Arange(col, 0.0f);
        Duplicate(ones, 1.0f, pregAll);
        Duplicate(zeros, 0.0f, pregAll);
        Duplicate(idx0, 0.0f, pregAll);
        Duplicate(idx1, 1.0f, pregAll);
        for (uint16_t i = 0; i < 32; ++i) {
            MaskReg m0, m1;
            Compare<float, AscendC::CMPMODE::LT>(m0, col, idx0, pregAll);
            Compare<float, AscendC::CMPMODE::LT>(m1, col, idx1, pregAll);
            Select(row0, ones, zeros, m0);
            Select(row1, ones, zeros, m1);
            StoreAlign(dst + static_cast<uint32_t>(i) * 128, row0, pregAll);
            StoreAlign(dst + static_cast<uint32_t>(i) * 128 + 64, row1, pregAll);
            Adds(idx0, idx0, 2.0f, pregAll);
            Adds(idx1, idx1, 2.0f, pregAll);
        }
    }
}

// 64x64 identity ND. Replaces scalar SetValue + V_S/S_V around Stage0 L1 I.
__aicore__ inline void Stage0_PaintIdentity64(AscendC::LocalTensor<float> ubI)
{
    using namespace AscendC::MicroAPI;
    __ubuf__ float *dst = (__ubuf__ float *)ubI.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> col, ones, zeros, row0, row1, idx0, idx1;
        Arange(col, 0.0f);
        Duplicate(ones, 1.0f, pregAll);
        Duplicate(zeros, 0.0f, pregAll);
        Duplicate(idx0, 0.0f, pregAll);
        Duplicate(idx1, 1.0f, pregAll);
        for (uint16_t i = 0; i < 32; ++i) {
            MaskReg m0, m1;
            Compare<float, AscendC::CMPMODE::EQ>(m0, col, idx0, pregAll);
            Compare<float, AscendC::CMPMODE::EQ>(m1, col, idx1, pregAll);
            Select(row0, ones, zeros, m0);
            Select(row1, ones, zeros, m1);
            StoreAlign(dst + static_cast<uint32_t>(i) * 128, row0, pregAll);
            StoreAlign(dst + static_cast<uint32_t>(i) * 128 + 64, row1, pregAll);
            Adds(idx0, idx0, 2.0f, pregAll);
            Adds(idx1, idx1, 2.0f, pregAll);
        }
    }
}

// I_vcs: 32x64, ones at (i,i) and (i, i+32).
__aicore__ inline void Stage0_PaintVcsIdentity(AscendC::LocalTensor<float> ubVcsI)
{
    using namespace AscendC::MicroAPI;
    __ubuf__ float *dst = (__ubuf__ float *)ubVcsI.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> col, ones, zeros, rowL, row, idx, idxR;
        Arange(col, 0.0f);
        Duplicate(ones, 1.0f, pregAll);
        Duplicate(zeros, 0.0f, pregAll);
        Duplicate(idx, 0.0f, pregAll);
        Duplicate(idxR, 32.0f, pregAll);
        for (uint16_t i = 0; i < static_cast<uint16_t>(kVcs32); ++i) {
            MaskReg mL, mR;
            Compare<float, AscendC::CMPMODE::EQ>(mL, col, idx, pregAll);
            Compare<float, AscendC::CMPMODE::EQ>(mR, col, idxR, pregAll);
            Select(rowL, ones, zeros, mL);
            Select(row, ones, rowL, mR);
            StoreAlign(dst + static_cast<uint32_t>(i) * kVcsPack32, row, pregAll);
            Adds(idx, idx, 1.0f, pregAll);
            Adds(idxR, idxR, 1.0f, pregAll);
        }
    }
}

} // namespace Prepare
} // namespace GdnStage

using namespace AscendC;

// 32×32 叶子 VF。两叶打包 32×64：oneRepeatSize=64，scatterCount=2。
// VF handbook (same math as solve_tri):
//   - hoist idx {0,32} and src0 row iterIdx out of the inner loop
//   - unroll inner by 2 so independent (i, i+1) Mul/Reduce/Scatter dual-issue
//   - tail is for(nTail), not if; LocalMemBar stays once per outer iter
//   - packed 32×64 rows are 256 B aligned, LoadAlign only
template <typename T, typename U>
__simd_vf__ inline void MulReduceScatterVF32(__ubuf__ T *dstAddr, __ubuf__ T *src0Addr, __ubuf__ T *src1Addr,
                                             __ubuf__ U *idxAddr, uint32_t scatterCount, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1a;
    AscendC::Reg::RegTensor<T> srcReg1b;
    AscendC::Reg::RegTensor<T> mulA;
    AscendC::Reg::RegTensor<T> mulB;
    AscendC::Reg::RegTensor<T> blkA;
    AscendC::Reg::RegTensor<T> blkB;
    AscendC::Reg::RegTensor<T> pairA;
    AscendC::Reg::RegTensor<T> pairB;
    AscendC::Reg::RegTensor<T> pair2A;
    AscendC::Reg::RegTensor<T> pair2B;
    AscendC::Reg::RegTensor<U> idxBase;
    AscendC::Reg::RegTensor<U> scatterIdxReg;

    uint32_t maskCount = oneRepeatSize;
    uint32_t pairMaskCount = scatterCount * 2;
    AscendC::Reg::MaskReg inputMask;
    AscendC::Reg::MaskReg pairMask;
    AscendC::Reg::MaskReg scatterMask;
    inputMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(maskCount);
    pairMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(pairMaskCount);
    scatterMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(scatterCount);
    AscendC::Reg::LoadAlign(idxBase, idxAddr);
    for (uint16_t iterIdx = 1; iterIdx < 32; iterIdx++) {
        AscendC::Reg::Adds(scatterIdxReg, idxBase, (uint32_t)iterIdx, scatterMask);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr + iterIdx * oneRepeatSize);
        const uint16_t nPair = static_cast<uint16_t>(iterIdx >> 1);
        const uint16_t nTail = static_cast<uint16_t>(iterIdx & 1);
        for (uint16_t p = 0; p < nPair; p++) {
            const uint32_t i0 = static_cast<uint32_t>(p) * 2;
            const uint32_t i1 = i0 + 1;
            AscendC::Reg::LoadAlign(srcReg1a, src1Addr + i0 * oneRepeatSize);
            AscendC::Reg::LoadAlign(srcReg1b, src1Addr + i1 * oneRepeatSize);
            AscendC::Reg::Mul(mulA, srcReg0, srcReg1a, inputMask);
            AscendC::Reg::Mul(mulB, srcReg0, srcReg1b, inputMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(blkA, mulA, inputMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(blkB, mulB, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pairA, blkA, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pairB, blkB, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pair2A, pairA, pairMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pair2B, pairB, pairMask);
            AscendC::Reg::Scatter(dstAddr + i0 * oneRepeatSize, pair2A, scatterIdxReg, scatterMask);
            AscendC::Reg::Scatter(dstAddr + i1 * oneRepeatSize, pair2B, scatterIdxReg, scatterMask);
        }
        for (uint16_t t = 0; t < nTail; t++) {
            const uint32_t i = static_cast<uint32_t>(nPair) * 2;
            AscendC::Reg::LoadAlign(srcReg1a, src1Addr + i * oneRepeatSize);
            AscendC::Reg::Mul(mulA, srcReg0, srcReg1a, inputMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(blkA, mulA, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pairA, blkA, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pair2A, pairA, pairMask);
            AscendC::Reg::Scatter(dstAddr + i * oneRepeatSize, pair2A, scatterIdxReg, scatterMask);
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
}

// LocalTensor entry: GetPhyAddr lives here, not at the Stage3 call site.
// dst and src1 are the same buffer (I_vcs overwritten with packed inverses).
__aicore__ inline void MulReduceScatterVF32(LocalTensor<float> &dst, LocalTensor<float> &src0,
                                            LocalTensor<float> &src1, LocalTensor<uint32_t> &idx)
{
    __ubuf__ float *dstAddr = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *src0Addr = (__ubuf__ float *)src0.GetPhyAddr();
    __ubuf__ float *src1Addr = (__ubuf__ float *)src1.GetPhyAddr();
    __ubuf__ uint32_t *idxAddr = (__ubuf__ uint32_t *)idx.GetPhyAddr();
    MulReduceScatterVF32(dstAddr, src0Addr, src1Addr, idxAddr,
                         GdnStage::kLeavesPerVec32, GdnStage::kVcsPack32);
}

#endif
