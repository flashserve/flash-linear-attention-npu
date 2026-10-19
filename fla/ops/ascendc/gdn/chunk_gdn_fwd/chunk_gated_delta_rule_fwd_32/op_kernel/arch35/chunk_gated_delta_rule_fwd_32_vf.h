/**
 * Vector VF. L2 / sigmoid / cumsum / row-scale reuse the prepare kernel's
 * MicroAPI implementations. Gate, state update and the 16x16 solve stay in
 * this file because their formulas are the fused forward's, not prepare's.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_VF_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_VF_H

#include "chunk_gated_delta_rule_fwd_32_common.h"

namespace GdnFwd32 {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr CastTrait kCastB162B32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait kCastB322B16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

__aicore__ inline void LoadCastBf16(__ubuf__ bfloat16_t *src, RegTensor<float> &dst, MaskReg &mask)
{
    RegTensor<bfloat16_t> xB16;
    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xB16, src);
    Cast<float, bfloat16_t, kCastB162B32>(dst, xB16, mask);
}

__aicore__ inline void StoreCastBf16(__ubuf__ bfloat16_t *dst, RegTensor<float> &src, MaskReg &mask)
{
    RegTensor<bfloat16_t> yB16;
    Cast<bfloat16_t, float, kCastB322B16>(yB16, src, mask);
    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dst, yB16, mask);
}

__aicore__ inline void LoadTwoVL(__ubuf__ bfloat16_t *addr, uint32_t offset0, uint32_t offset1, RegTensor<float> &a,
                                 RegTensor<float> &b, MaskReg &preg)
{
    LoadCastBf16(addr + offset0, a, preg);
    LoadCastBf16(addr + offset1, b, preg);
}

// In-place L2 of one [32,128] bf16 matrix. Two rows per iteration, same
// reduction as prepare's L2NormK128VF. rstd is not stored.
#define GDN32_L2_ROWS(xAddr)                                                                                           \
    for (uint16_t i = 0; i < nEven; i++) {                                                                             \
        RegTensor<float> x0a, x0b, x1a, x1b, s0a, s0b, s1a, s1b, mean0, mean1, r0, r1, brc0, brc1;                    \
        RegTensor<float> y0a, y0b, y1a, y1b;                                                                           \
        LoadTwoVL(xAddr, i * kPair, i * kPair + kVl, x0a, x0b, pregAll);                                               \
        LoadTwoVL(xAddr, i * kPair + kRow, i * kPair + kRow + kVl, x1a, x1b, pregAll);                                 \
        Mul(s0a, x0a, x0a, pregAll);                                                                                   \
        Mul(s0b, x0b, x0b, pregAll);                                                                                   \
        Mul(s1a, x1a, x1a, pregAll);                                                                                   \
        Mul(s1b, x1b, x1b, pregAll);                                                                                   \
        Add(s0a, s0a, s0b, pregAll);                                                                                   \
        Add(s1a, s1a, s1b, pregAll);                                                                                   \
        Reduce<AscendC::Reg::ReduceType::SUM>(mean0, s0a, pregAll);                                                   \
        Reduce<AscendC::Reg::ReduceType::SUM>(mean1, s1a, pregAll);                                                   \
        Adds(mean0, mean0, kEps, pregMerge);                                                                          \
        Adds(mean1, mean1, kEps, pregMerge);                                                                          \
        Sqrt(mean0, mean0, pregMerge);                                                                                \
        Sqrt(mean1, mean1, pregMerge);                                                                                \
        Div(r0, one, mean0, pregMerge);                                                                               \
        Div(r1, one, mean1, pregMerge);                                                                               \
        Duplicate(brc0, r0, pregAll);                                                                                 \
        Duplicate(brc1, r1, pregAll);                                                                                 \
        Mul(y0a, x0a, brc0, pregAll);                                                                                 \
        Mul(y0b, x0b, brc0, pregAll);                                                                                 \
        Mul(y1a, x1a, brc1, pregAll);                                                                                 \
        Mul(y1b, x1b, brc1, pregAll);                                                                                 \
        StoreCastBf16(xAddr + i * kPair, y0a, pregAll);                                                               \
        StoreCastBf16(xAddr + i * kPair + kVl, y0b, pregAll);                                                         \
        StoreCastBf16(xAddr + i * kPair + kRow, y1a, pregAll);                                                        \
        StoreCastBf16(xAddr + i * kPair + kRow + kVl, y1b, pregAll);                                                  \
    }

// q and k share one VEC_SCOPE. kda_32_32 also normalizes both in one pass;
// this keeps the single-pass rstd (no fp32 spill) and only drops the second
// scope entry.
__aicore__ inline void L2InplaceQKVF(LocalTensor<bfloat16_t> &q, LocalTensor<bfloat16_t> &k)
{
    constexpr uint32_t kRow = 2 * kVl;
    constexpr uint32_t kPair = 2 * kRow;
    __ubuf__ bfloat16_t *qAddr = (__ubuf__ bfloat16_t *)q.GetPhyAddr();
    __ubuf__ bfloat16_t *kAddr = (__ubuf__ bfloat16_t *)k.GetPhyAddr();
    const uint16_t nEven = static_cast<uint16_t>(kBt >> 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> one;
        Duplicate(one, 1.0f, pregMerge);
        GDN32_L2_ROWS(qAddr)
        GDN32_L2_ROWS(kAddr)
    }
}
#undef GDN32_L2_ROWS

__aicore__ inline void BetaSigmoidVF(LocalTensor<bfloat16_t> &beta, LocalTensor<float> &betaOut)
{
    __ubuf__ bfloat16_t *src = (__ubuf__ bfloat16_t *)beta.GetPhyAddr();
    __ubuf__ float *dst = (__ubuf__ float *)betaOut.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t n = static_cast<uint32_t>(kBt);
        MaskReg preg = UpdateMask<float>(n);
        RegTensor<float> x, e, one, den;
        LoadCastBf16(src, x, preg);
        Muls(x, x, -1.0f, preg);
        Exp(e, x, preg);
        Duplicate(one, 1.0f, preg);
        Add(den, one, e, preg);
        Div(x, one, den, preg);
        StoreAlign(dst, x, preg);
    }
}

// g' = cumsum(g) / ln(2), eg = exp2(g'). Register Sklansky, same tree as
// prepare CumsumScaleVF. n=32: lanes 32..63 are cleared so the 64-wide
// deinterleave does not mix garbage into the prefix. Scalar GetValue scan
// is the slow path kda avoids by keeping the reduction in the vector pipe.
__aicore__ inline void CumsumExp2VF(LocalTensor<float> &g, LocalTensor<float> &gp, LocalTensor<float> &eg)
{
    __ubuf__ float *gAddr = (__ubuf__ float *)g.GetPhyAddr();
    __ubuf__ float *gpAddr = (__ubuf__ float *)gp.GetPhyAddr();
    __ubuf__ float *egAddr = (__ubuf__ float *)eg.GetPhyAddr();
    constexpr uint16_t doS1 = 1;
    constexpr uint16_t doS2 = 1;
    constexpr uint16_t doS4 = 1;
    constexpr uint16_t doS8 = 1;
    constexpr uint16_t n16TailUpd = 1;
    constexpr uint16_t barAfter8 = 1;
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg32 = CreateMask<float, MaskPattern::VL32>();
        RegTensor<float> v, left, even, odd, eLo, eHi, oLo, oHi, unused, zero, dump;
        RegTensor<float> a0, a1, a2, a3, a4, a5, a6, a7;
        RegTensor<float> b0, b1, b2, b3, b4, b5, b6, b7;
        Duplicate(zero, 0.0f, pregAll);
        LoadAlign(v, gAddr);
        uint32_t n = static_cast<uint32_t>(kBt);
        MaskReg pregN = UpdateMask<float>(n);
        Select(v, v, zero, pregN);

        for (uint16_t t = 0; t < doS1; ++t) {
            DeInterleave(even, odd, v, zero);
            Add(odd, odd, even, preg32);
            Interleave(v, unused, even, odd);
        }
        for (uint16_t t = 0; t < doS2; ++t) {
            DeInterleave(even, odd, v, zero);
            DeInterleave(eLo, eHi, even, zero);
            DeInterleave(oLo, oHi, odd, zero);
            Add(eHi, eHi, oLo, preg32);
            Add(oHi, oHi, oLo, preg32);
            Interleave(even, unused, eLo, eHi);
            Interleave(odd, unused, oLo, oHi);
            Interleave(v, unused, even, odd);
        }
        for (uint16_t t = 0; t < doS4; ++t) {
            DeInterleave(even, odd, v, zero);
            DeInterleave(eLo, eHi, even, zero);
            DeInterleave(oLo, oHi, odd, zero);
            DeInterleave(a0, a4, eLo, zero);
            DeInterleave(a2, a6, eHi, zero);
            DeInterleave(a1, a5, oLo, zero);
            DeInterleave(a3, a7, oHi, zero);
            Add(a4, a4, a3, preg32);
            Add(a5, a5, a3, preg32);
            Add(a6, a6, a3, preg32);
            Add(a7, a7, a3, preg32);
            Interleave(eLo, unused, a0, a4);
            Interleave(eHi, unused, a2, a6);
            Interleave(oLo, unused, a1, a5);
            Interleave(oHi, unused, a3, a7);
            Interleave(even, unused, eLo, eHi);
            Interleave(odd, unused, oLo, oHi);
            Interleave(v, unused, even, odd);
        }
        for (uint16_t t = 0; t < doS8; ++t) {
            DeInterleave(even, odd, v, zero);
            DeInterleave(eLo, eHi, even, zero);
            DeInterleave(oLo, oHi, odd, zero);
            DeInterleave(a0, a4, eLo, zero);
            DeInterleave(a2, a6, eHi, zero);
            DeInterleave(a1, a5, oLo, zero);
            DeInterleave(a3, a7, oHi, zero);
            DeInterleave(b0, even, a0, zero);
            DeInterleave(b1, odd, a1, zero);
            DeInterleave(b2, eLo, a2, zero);
            DeInterleave(b3, eHi, a3, zero);
            DeInterleave(b4, oLo, a4, zero);
            DeInterleave(b5, oHi, a5, zero);
            DeInterleave(b6, unused, a6, zero);
            DeInterleave(b7, left, a7, zero);
            Add(even, even, b7, preg32);
            Add(odd, odd, b7, preg32);
            Add(eLo, eLo, b7, preg32);
            Add(eHi, eHi, b7, preg32);
            Add(oLo, oLo, b7, preg32);
            Add(oHi, oHi, b7, preg32);
            Add(unused, unused, b7, preg32);
            Add(left, left, b7, preg32);
            Interleave(a0, dump, b0, even);
            Interleave(a1, dump, b1, odd);
            Interleave(a2, dump, b2, eLo);
            Interleave(a3, dump, b3, eHi);
            Interleave(a4, dump, b4, oLo);
            Interleave(a5, dump, b5, oHi);
            Interleave(a6, dump, b6, unused);
            Interleave(a7, dump, b7, left);
            Interleave(eLo, dump, a0, a4);
            Interleave(eHi, dump, a2, a6);
            Interleave(oLo, dump, a1, a5);
            Interleave(oHi, dump, a3, a7);
            Interleave(even, dump, eLo, eHi);
            Interleave(odd, dump, oLo, oHi);
            Interleave(v, dump, even, odd);
        }
        for (uint16_t t = 0; t < barAfter8; ++t) {
            StoreAlign(gpAddr, v, pregAll);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        for (uint16_t t = 0; t < n16TailUpd; ++t) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(left, gpAddr + 15);
            uint32_t cLo = 16;
            uint32_t cHi = 32;
            MaskReg pregLo = UpdateMask<float>(cLo);
            MaskReg pregHi = UpdateMask<float>(cHi);
            MaskReg pregRange;
            Xor(pregRange, pregHi, pregLo, pregAll);
            Add<float, MaskMergeMode::MERGING>(v, v, left, pregRange);
        }
        Muls(v, v, kRcpLn2, pregN);
        StoreAlign(gpAddr, v, pregN);
        Muls(v, v, kLn2, pregN);
        Exp(v, v, pregN);
        StoreAlign(egAddr, v, pregN);
    }
}

// y[r,:] = x[r,:] * scale[r]. Two rows per iteration, matching kda's normalize
// loop and prepare ScaleRowsK128VF (16 trips instead of 32).
__aicore__ inline void ScaleRowsVF(LocalTensor<bfloat16_t> &x, LocalTensor<bfloat16_t> &y, LocalTensor<float> &scale)
{
    constexpr uint32_t kRow = 2 * kVl;
    constexpr uint32_t kPairElems = 2 * kRow;
    __ubuf__ bfloat16_t *xAddr = (__ubuf__ bfloat16_t *)x.GetPhyAddr();
    __ubuf__ bfloat16_t *yAddr = (__ubuf__ bfloat16_t *)y.GetPhyAddr();
    __ubuf__ float *sAddr = (__ubuf__ float *)scale.GetPhyAddr();
    const uint16_t nPair = static_cast<uint16_t>(kBt >> 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> s0, s1, a0, b0, a1, b1;
        for (uint16_t p = 0; p < nPair; ++p) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s0, sAddr + p * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s1, sAddr + p * 2 + 1);
            LoadCastBf16(xAddr + p * kPairElems, a0, pregAll);
            LoadCastBf16(xAddr + p * kPairElems + kVl, b0, pregAll);
            LoadCastBf16(xAddr + p * kPairElems + kRow, a1, pregAll);
            LoadCastBf16(xAddr + p * kPairElems + kRow + kVl, b1, pregAll);
            Mul(a0, a0, s0, pregAll);
            Mul(b0, b0, s0, pregAll);
            Mul(a1, a1, s1, pregAll);
            Mul(b1, b1, s1, pregAll);
            StoreCastBf16(yAddr + p * kPairElems, a0, pregAll);
            StoreCastBf16(yAddr + p * kPairElems + kVl, b0, pregAll);
            StoreCastBf16(yAddr + p * kPairElems + kRow, a1, pregAll);
            StoreCastBf16(yAddr + p * kPairElems + kRow + kVl, b1, pregAll);
        }
    }
}

// y[r,:] = x[r,:] * beta[r] * eg[r]. beta*eg stays in the row broadcast,
// so the 32-wide product is not spilled through UB the way MulBetaExp did.
__aicore__ inline void ScaleRowsBetaExpVF(LocalTensor<bfloat16_t> &x, LocalTensor<bfloat16_t> &y,
                                          LocalTensor<float> &beta, LocalTensor<float> &eg)
{
    constexpr uint32_t kRow = 2 * kVl;
    constexpr uint32_t kPairElems = 2 * kRow;
    __ubuf__ bfloat16_t *xAddr = (__ubuf__ bfloat16_t *)x.GetPhyAddr();
    __ubuf__ bfloat16_t *yAddr = (__ubuf__ bfloat16_t *)y.GetPhyAddr();
    __ubuf__ float *bAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *eAddr = (__ubuf__ float *)eg.GetPhyAddr();
    const uint16_t nPair = static_cast<uint16_t>(kBt >> 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> b0, b1, e0, e1, s0, s1, a0, c0, a1, c1;
        for (uint16_t p = 0; p < nPair; ++p) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(b0, bAddr + p * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(b1, bAddr + p * 2 + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(e0, eAddr + p * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(e1, eAddr + p * 2 + 1);
            Mul(s0, b0, e0, pregAll);
            Mul(s1, b1, e1, pregAll);
            LoadCastBf16(xAddr + p * kPairElems, a0, pregAll);
            LoadCastBf16(xAddr + p * kPairElems + kVl, c0, pregAll);
            LoadCastBf16(xAddr + p * kPairElems + kRow, a1, pregAll);
            LoadCastBf16(xAddr + p * kPairElems + kRow + kVl, c1, pregAll);
            Mul(a0, a0, s0, pregAll);
            Mul(c0, c0, s0, pregAll);
            Mul(a1, a1, s1, pregAll);
            Mul(c1, c1, s1, pregAll);
            StoreCastBf16(yAddr + p * kPairElems, a0, pregAll);
            StoreCastBf16(yAddr + p * kPairElems + kVl, c0, pregAll);
            StoreCastBf16(yAddr + p * kPairElems + kRow, a1, pregAll);
            StoreCastBf16(yAddr + p * kPairElems + kRow + kVl, c1, pregAll);
        }
    }
}

// L[i,j] = (i>j) ? beta[i] * exp2(g'[i]-g'[j]) * kkt : 0
// attn[i,j] = (i>=j) ? exp2(g'[i]-g'[j]) * qk : 0, stored as bf16.
__aicore__ inline void GateAttnVF(LocalTensor<float> &gp, LocalTensor<float> &beta, LocalTensor<float> &kkt,
                                  LocalTensor<float> &qk, LocalTensor<float> &lower, LocalTensor<bfloat16_t> &attn)
{
    __ubuf__ float *gAddr = (__ubuf__ float *)gp.GetPhyAddr();
    __ubuf__ float *bAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *kktAddr = (__ubuf__ float *)kkt.GetPhyAddr();
    __ubuf__ float *qkAddr = (__ubuf__ float *)qk.GetPhyAddr();
    __ubuf__ float *lAddr = (__ubuf__ float *)lower.GetPhyAddr();
    __ubuf__ bfloat16_t *aAddr = (__ubuf__ bfloat16_t *)attn.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg preg32 = CreateMask<float, MaskPattern::VL32>();
        RegTensor<float> gJ, gI, bI, diff, gate, kk, qq, outL, outA, zero;
        Duplicate(zero, 0.0f, preg32);
        LoadAlign(gJ, gAddr);
        for (uint16_t i = 0; i < static_cast<uint16_t>(kBt); ++i) {
            uint32_t lowerCount = static_cast<uint32_t>(i);
            uint32_t attnCount = static_cast<uint32_t>(i) + 1;
            MaskReg lowerMask = UpdateMask<float>(lowerCount);
            MaskReg attnMask = UpdateMask<float>(attnCount);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(gI, gAddr + static_cast<uint32_t>(i));
            LoadAlign<float, LoadDist::DIST_BRC_B32>(bI, bAddr + static_cast<uint32_t>(i));
            Sub(diff, gI, gJ, preg32);
            Muls(diff, diff, kLn2, preg32);
            Exp(gate, diff, preg32);
            LoadAlign(kk, kktAddr + static_cast<uint32_t>(i) * kBt);
            LoadAlign(qq, qkAddr + static_cast<uint32_t>(i) * kBt);
            Mul(outL, gate, bI, preg32);
            Mul(outL, outL, kk, preg32);
            Select(outL, outL, zero, lowerMask);
            StoreAlign(lAddr + static_cast<uint32_t>(i) * kBt, outL, preg32);
            Mul(outA, gate, qq, preg32);
            Select(outA, outA, zero, attnMask);
            StoreCastBf16(aAddr + static_cast<uint32_t>(i) * kBt, outA, preg32);
        }
    }
}

// y[r,:] = cast_rint(src[r,:] * scale[r]), src is fp32 [32,128].
__aicore__ inline void ScaleFp32RowsVF(LocalTensor<float> &src, LocalTensor<float> &scale, LocalTensor<bfloat16_t> &dst)
{
    __ubuf__ float *sAddr = (__ubuf__ float *)src.GetPhyAddr();
    __ubuf__ float *cAddr = (__ubuf__ float *)scale.GetPhyAddr();
    __ubuf__ bfloat16_t *dAddr = (__ubuf__ bfloat16_t *)dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> sc, a, b;
        for (uint16_t r = 0; r < static_cast<uint16_t>(kBt); ++r) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sc, cAddr + r);
            LoadAlign(a, sAddr + static_cast<uint32_t>(r) * kDim);
            LoadAlign(b, sAddr + static_cast<uint32_t>(r) * kDim + kVl);
            Mul(a, a, sc, pregAll);
            Mul(b, b, sc, pregAll);
            StoreCastBf16(dAddr + static_cast<uint32_t>(r) * kDim, a, pregAll);
            StoreCastBf16(dAddr + static_cast<uint32_t>(r) * kDim + kVl, b, pregAll);
        }
    }
}

// v_new = u - P, kept as bf16. v_new_g = cast_rint(v_new * exp2(g_last - g'[i])).
// One row is 128 fp32, two registers, same shape as ScaleRows.
__aicore__ inline void VnewVF(LocalTensor<float> &u, LocalTensor<float> &p, LocalTensor<float> &gp,
                              LocalTensor<bfloat16_t> &vNew, LocalTensor<bfloat16_t> &vNewG, float gLast)
{
    __ubuf__ float *uAddr = (__ubuf__ float *)u.GetPhyAddr();
    __ubuf__ float *pAddr = (__ubuf__ float *)p.GetPhyAddr();
    __ubuf__ float *gAddr = (__ubuf__ float *)gp.GetPhyAddr();
    __ubuf__ bfloat16_t *nAddr = (__ubuf__ bfloat16_t *)vNew.GetPhyAddr();
    __ubuf__ bfloat16_t *ngAddr = (__ubuf__ bfloat16_t *)vNewG.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> last;
        Duplicate(last, gLast, pregAll);
        for (uint16_t r = 0; r < static_cast<uint16_t>(kBt); ++r) {
            RegTensor<float> a, b, pa, pb, gi, gate;
            LoadAlign(a, uAddr + static_cast<uint32_t>(r) * kDim);
            LoadAlign(b, uAddr + static_cast<uint32_t>(r) * kDim + kVl);
            LoadAlign(pa, pAddr + static_cast<uint32_t>(r) * kDim);
            LoadAlign(pb, pAddr + static_cast<uint32_t>(r) * kDim + kVl);
            Sub(a, a, pa, pregAll);
            Sub(b, b, pb, pregAll);
            StoreCastBf16(nAddr + static_cast<uint32_t>(r) * kDim, a, pregAll);
            StoreCastBf16(nAddr + static_cast<uint32_t>(r) * kDim + kVl, b, pregAll);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(gi, gAddr + r);
            Sub(gate, last, gi, pregAll);
            Muls(gate, gate, kLn2, pregAll);
            Exp(gate, gate, pregAll);
            Mul(a, a, gate, pregAll);
            Mul(b, b, gate, pregAll);
            StoreCastBf16(ngAddr + static_cast<uint32_t>(r) * kDim, a, pregAll);
            StoreCastBf16(ngAddr + static_cast<uint32_t>(r) * kDim + kVl, b, pregAll);
        }
    }
}

// o = cast_rint(o_acc * scale). scale is the kernel scalar, one multiply.
__aicore__ inline void ScaleOVf(LocalTensor<float> &src, LocalTensor<bfloat16_t> &dst, float scale)
{
    __ubuf__ float *sAddr = (__ubuf__ float *)src.GetPhyAddr();
    __ubuf__ bfloat16_t *dAddr = (__ubuf__ bfloat16_t *)dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> sc;
        Duplicate(sc, scale, pregAll);
        for (uint16_t r = 0; r < static_cast<uint16_t>(kBt); ++r) {
            RegTensor<float> a, b;
            LoadAlign(a, sAddr + static_cast<uint32_t>(r) * kDim);
            LoadAlign(b, sAddr + static_cast<uint32_t>(r) * kDim + kVl);
            Mul(a, a, sc, pregAll);
            Mul(b, b, sc, pregAll);
            StoreCastBf16(dAddr + static_cast<uint32_t>(r) * kDim, a, pregAll);
            StoreCastBf16(dAddr + static_cast<uint32_t>(r) * kDim + kVl, b, pregAll);
        }
    }
}

// state[r,:] = cast_rint(decay * state[r,:] + delta[r,:]).
__aicore__ inline void ApplyStateVF(LocalTensor<bfloat16_t> &state, LocalTensor<float> &delta, float decay)
{
    __ubuf__ bfloat16_t *sAddr = (__ubuf__ bfloat16_t *)state.GetPhyAddr();
    __ubuf__ float *dAddr = (__ubuf__ float *)delta.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> dec;
        Duplicate(dec, decay, pregAll);
        for (uint16_t r = 0; r < static_cast<uint16_t>(kDim); ++r) {
            RegTensor<float> a, b, da, db;
            LoadCastBf16(sAddr + static_cast<uint32_t>(r) * kDim, a, pregAll);
            LoadCastBf16(sAddr + static_cast<uint32_t>(r) * kDim + kVl, b, pregAll);
            LoadAlign(da, dAddr + static_cast<uint32_t>(r) * kDim);
            LoadAlign(db, dAddr + static_cast<uint32_t>(r) * kDim + kVl);
            Mul(a, a, dec, pregAll);
            Mul(b, b, dec, pregAll);
            Add(a, a, da, pregAll);
            Add(b, b, db, pregAll);
            StoreCastBf16(sAddr + static_cast<uint32_t>(r) * kDim, a, pregAll);
            StoreCastBf16(sAddr + static_cast<uint32_t>(r) * kDim + kVl, b, pregAll);
        }
    }
}

// K[32,128] -> K^T[128,32], row-major. Scatter has no dense VF form here.
__aicore__ inline void TransposeKVf(LocalTensor<bfloat16_t> &kHat, LocalTensor<bfloat16_t> &kT)
{
    for (int row = 0; row < kBt; ++row) {
        for (int col = 0; col < kDim; ++col) {
            kT.SetValue(col * kBt + row, kHat.GetValue(row * kDim + col));
        }
    }
}

// (I + Lii) X = I. Lii is strict lower, so row r is e_r - sum_{t<r} L[r,t] X[t].
// Each row is its own VEC_SCOPE: the next row reloads X from UB, and a store in the
// same scope is not visible to a later load. Diagonal 1 is a scalar store after V_S.
__aicore__ inline void Solve16VF(__ubuf__ float *lBase, __ubuf__ float *xAddr, uint32_t row0, uint32_t col0)
{
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> z;
        Duplicate(z, 0.f, pregAll);
        StoreAlign(xAddr, z, pregAll);
        StoreAlign(xAddr + kVl, z, pregAll);
        StoreAlign(xAddr + 2 * kVl, z, pregAll);
        StoreAlign(xAddr + 3 * kVl, z, pregAll);
    }
    PipeBarrier<PIPE_V>();
    for (int row = 0; row < 16; ++row) {
        __ubuf__ float *rowL = lBase + (row0 + static_cast<uint32_t>(row)) * kBt + col0;
        __VEC_SCOPE__
        {
            uint32_t n16 = 16;
            MaskReg preg = UpdateMask<float>(n16);
            RegTensor<float> acc, xt, coeff;
            Duplicate(acc, 0.f, preg);
            for (uint16_t t = 0; t < 16; ++t) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(coeff, rowL + t);
                LoadAlign(xt, xAddr + static_cast<uint32_t>(t) * 16);
                Mul(xt, xt, coeff, preg);
                Sub(acc, acc, xt, preg);
            }
            StoreAlign(xAddr + static_cast<uint32_t>(row) * 16, acc, preg);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        xAddr[row * 16 + row] = 1.f;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    }
}

// C[row,:] = sum_k left[row,k] * right[k,:], 16x16.
__aicore__ inline void Gemm16VF(__ubuf__ float *left, uint32_t leftStride, __ubuf__ float *right, __ubuf__ float *dst,
                                float scale)
{
    __VEC_SCOPE__
    {
        uint32_t n16 = 16;
        MaskReg preg = UpdateMask<float>(n16);
        for (uint16_t row = 0; row < 16; ++row) {
            RegTensor<float> acc, rv, coeff;
            Duplicate(acc, 0.f, preg);
            for (uint16_t k = 0; k < 16; ++k) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(coeff, left + row * leftStride + k);
                LoadAlign(rv, right + static_cast<uint32_t>(k) * 16);
                Mul(rv, rv, coeff, preg);
                Add(acc, acc, rv, preg);
            }
            Muls(acc, acc, scale, preg);
            StoreAlign(dst + static_cast<uint32_t>(row) * 16, acc, preg);
        }
    }
}

__aicore__ inline void InvertAVF(LocalTensor<float> &lower, LocalTensor<float> &x00, LocalTensor<float> &x11,
                                 LocalTensor<float> &temp, LocalTensor<float> &a, LocalTensor<bfloat16_t> &aBf)
{
    __ubuf__ float *lAddr = (__ubuf__ float *)lower.GetPhyAddr();
    __ubuf__ float *x00Addr = (__ubuf__ float *)x00.GetPhyAddr();
    __ubuf__ float *x11Addr = (__ubuf__ float *)x11.GetPhyAddr();
    __ubuf__ float *tmpAddr = (__ubuf__ float *)temp.GetPhyAddr();
    __ubuf__ float *aAddr = (__ubuf__ float *)a.GetPhyAddr();
    Solve16VF(lAddr, x00Addr, 0, 0);
    Solve16VF(lAddr, x11Addr, 16, 16);
    PipeBarrier<PIPE_ALL>();
    Gemm16VF(lAddr + 16 * kBt, kBt, x00Addr, tmpAddr, 1.f);
    Duplicate(a, 0.f, kBt * kBt);
    PipeBarrier<PIPE_V>();
    Gemm16VF(x11Addr, 16, tmpAddr, tmpAddr + 256, -1.f);
    PipeBarrier<PIPE_V>();
    __VEC_SCOPE__
    {
        uint32_t n16 = 16;
        MaskReg preg = UpdateMask<float>(n16);
        for (uint16_t row = 0; row < 16; ++row) {
            RegTensor<float> v;
            LoadAlign(v, x00Addr + static_cast<uint32_t>(row) * 16);
            StoreAlign(aAddr + static_cast<uint32_t>(row) * kBt, v, preg);
            LoadAlign(v, x11Addr + static_cast<uint32_t>(row) * 16);
            StoreAlign(aAddr + (16 + row) * kBt + 16, v, preg);
            LoadAlign(v, tmpAddr + 256 + static_cast<uint32_t>(row) * 16);
            StoreAlign(aAddr + (16 + row) * kBt, v, preg);
        }
    }
    Cast(aBf, a, RoundMode::CAST_RINT, kBt * kBt);
}

}  // namespace GdnFwd32

#endif
