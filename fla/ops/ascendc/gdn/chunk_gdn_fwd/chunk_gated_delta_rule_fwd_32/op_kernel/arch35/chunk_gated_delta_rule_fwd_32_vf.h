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
constexpr CastTrait kCastB162B32Hi = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait kCastB322B16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait kCastB322B16Hi = {RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// even/odd float halves -> one bf16 row -> tight NZ. Same round trip as KDA:
// layout ZERO and ONE split a row, Or puts it back, one block-copy stores it.
// KDA casts with the uint16 mask, in place, then Or. A float mask leaves the
// odd lanes zero, which is what a split mask produced.
__aicore__ inline void StoreNzEvenOdd(__ubuf__ bfloat16_t *nz, uint32_t row, RegTensor<float> &even,
                                      RegTensor<float> &odd, MaskReg &m16, MaskReg &m128)
{
    RegTensor<bfloat16_t> lo, hi;
    Cast<bfloat16_t, float, kCastB322B16>(lo, even, m16);
    Cast<bfloat16_t, float, kCastB322B16Hi>(hi, odd, m16);
    Or((RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)hi, m16);
    StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(nz + row * 16, lo, static_cast<uint32_t>(kBt), m128);
}

__aicore__ inline void LoadEvenOdd(__ubuf__ bfloat16_t *src, RegTensor<float> &even, RegTensor<float> &odd,
                                   MaskReg &m16)
{
    RegTensor<bfloat16_t> raw;
    LoadAlign(raw, src);
    Cast<float, bfloat16_t, kCastB162B32>(even, raw, m16);
    Cast<float, bfloat16_t, kCastB162B32Hi>(odd, raw, m16);
}

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

// One 64-wide half of a row, packed into the low 64 bf16 lanes, then written as
// four [rows, 16] panels. panel is 0 for elements [0:64) and 4 for [64:128).
__aicore__ inline void StoreNzHalf(__ubuf__ bfloat16_t *nz, uint32_t row, uint32_t panel, RegTensor<float> &src,
                                   MaskReg &fmask, MaskReg &m64, uint32_t stride = kBt)
{
    RegTensor<bfloat16_t> casted;
    Cast<bfloat16_t, float, kCastB322B16>(casted, src, fmask);
    RegTensor<uint16_t> packed;
    Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packed, (RegTensor<uint32_t> &)casted);
    StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
        nz + row * 16 + panel * stride * 16, (RegTensor<bfloat16_t> &)packed, stride, m64);
}

// ND stays the existing packed store. NZ is the same tight panel image the
// second-pass pack used to build, emitted from the values already in registers.
__aicore__ inline void StoreRow128(__ubuf__ bfloat16_t *nd, __ubuf__ bfloat16_t *nz, uint32_t row, RegTensor<float> &lo,
                                   RegTensor<float> &hi, MaskReg &fmask, MaskReg &m64)
{
    StoreCastBf16(nd + row * kDim, lo, fmask);
    StoreCastBf16(nd + row * kDim + kVl, hi, fmask);
    StoreNzHalf(nz, row, 0, lo, fmask, m64);
    StoreNzHalf(nz, row, 4, hi, fmask, m64);
}

__aicore__ inline void LoadTwoVL(__ubuf__ bfloat16_t *addr, uint32_t offset0, uint32_t offset1, RegTensor<float> &a,
                                 RegTensor<float> &b, MaskReg &preg)
{
    LoadCastBf16(addr + offset0, a, preg);
    LoadCastBf16(addr + offset1, b, preg);
}

// Sum of squares for rows [2*base, 2*base+2). Means land in m0/m1 with eps
// already added. The next pair's loads sit between Sqrt and Div so the sqrt
// latency is covered by useful work instead of a stall per row.
#define GDN32_L2_REDUCE(addr, base, ee0, oo0, ee1, oo1, m0, m1)                                                        \
    LoadEvenOdd((addr) + static_cast<uint32_t>(base) * kPair, ee0, oo0, m16);                                         \
    LoadEvenOdd((addr) + static_cast<uint32_t>(base) * kPair + kRow, ee1, oo1, m16);                                  \
    Mul(s0, ee0, ee0, pregAll);                                                                                        \
    Mul(s1, oo0, oo0, pregAll);                                                                                        \
    Add(s0, s0, s1, pregAll);                                                                                          \
    Mul(s1, ee1, ee1, pregAll);                                                                                        \
    Mul(brc0, oo1, oo1, pregAll);                                                                                      \
    Add(s1, s1, brc0, pregAll);                                                                                        \
    Reduce<AscendC::Reg::ReduceType::SUM>(m0, s0, pregAll);                                                           \
    Reduce<AscendC::Reg::ReduceType::SUM>(m1, s1, pregAll);                                                           \
    Adds(m0, m0, kEps, pregMerge);                                                                                    \
    Adds(m1, m1, kEps, pregMerge);

#define GDN32_L2_APPLY(ee0, oo0, ee1, oo1, m0, m1)                                                                     \
    Div(r0, one, m0, pregMerge);                                                                                       \
    Div(r1, one, m1, pregMerge);                                                                                       \
    Duplicate(brc0, r0, pregAll);                                                                                      \
    Duplicate(brc1, r1, pregAll);                                                                                      \
    Mul(ee0, ee0, brc0, pregAll);                                                                                      \
    Mul(oo0, oo0, brc0, pregAll);                                                                                      \
    Mul(ee1, ee1, brc1, pregAll);                                                                                      \
    Mul(oo1, oo1, brc1, pregAll);

#define GDN32_L2_ROWS(xAddr, nzAddr)                                                                                   \
    {                                                                                                                  \
        RegTensor<float> e0, o0, e1, o1, e2, o2, e3, o3, s0, s1, mean0, mean1, mean2, mean3, r0, r1, brc0, brc1;      \
        GDN32_L2_REDUCE(xAddr, 0, e0, o0, e1, o1, mean0, mean1)                                                        \
        for (uint16_t i = 0; i < nEven - 1; ++i) {                                                                     \
            Sqrt(mean0, mean0, pregMerge);                                                                             \
            Sqrt(mean1, mean1, pregMerge);                                                                             \
            GDN32_L2_REDUCE(xAddr, i + 1, e2, o2, e3, o3, mean2, mean3)                                                \
            GDN32_L2_APPLY(e0, o0, e1, o1, mean0, mean1)                                                               \
            StoreNzEvenOdd(nzAddr, static_cast<uint32_t>(i) * 2, e0, o0, m16, m128);                                  \
            StoreNzEvenOdd(nzAddr, static_cast<uint32_t>(i) * 2 + 1, e1, o1, m16, m128);                              \
            GDN32_L2_TAIL                                                                                              \
            e0 = e2;                                                                                                   \
            o0 = o2;                                                                                                   \
            e1 = e3;                                                                                                   \
            o1 = o3;                                                                                                   \
            mean0 = mean2;                                                                                             \
            mean1 = mean3;                                                                                             \
        }                                                                                                              \
        {                                                                                                              \
            uint16_t i = static_cast<uint16_t>(nEven - 1);                                                             \
            Sqrt(mean0, mean0, pregMerge);                                                                             \
            Sqrt(mean1, mean1, pregMerge);                                                                             \
            GDN32_L2_APPLY(e0, o0, e1, o1, mean0, mean1)                                                               \
            StoreNzEvenOdd(nzAddr, static_cast<uint32_t>(i) * 2, e0, o0, m16, m128);                                  \
            StoreNzEvenOdd(nzAddr, static_cast<uint32_t>(i) * 2 + 1, e1, o1, m16, m128);                              \
            GDN32_L2_TAIL                                                                                              \
        }                                                                                                              \
    }

// Which: 0 both, 1 q only, 2 k only. The prologue copies K and starts K@K
// before q's L2, so the two halves are separate VEC_SCOPEs.
template <int Which = 0>
__aicore__ inline void L2InplaceQKVF(LocalTensor<bfloat16_t> &q, LocalTensor<bfloat16_t> &k,
                                     LocalTensor<bfloat16_t> &kbgNz, LocalTensor<bfloat16_t> &qexpNz,
                                     LocalTensor<float> &beta, LocalTensor<float> &eg,
                                     LocalTensor<bfloat16_t> &qNz, LocalTensor<bfloat16_t> &kNz)
{
    constexpr uint32_t kRow = 2 * kVl;
    constexpr uint32_t kPair = 2 * kRow;
    __ubuf__ bfloat16_t *qAddr = (__ubuf__ bfloat16_t *)q.GetPhyAddr();
    __ubuf__ bfloat16_t *kAddr = (__ubuf__ bfloat16_t *)k.GetPhyAddr();
    __ubuf__ bfloat16_t *kbgAddr = (__ubuf__ bfloat16_t *)kbgNz.GetPhyAddr();
    __ubuf__ bfloat16_t *qeAddr = (__ubuf__ bfloat16_t *)qexpNz.GetPhyAddr();
    __ubuf__ bfloat16_t *qNzAddr = (__ubuf__ bfloat16_t *)qNz.GetPhyAddr();
    __ubuf__ bfloat16_t *kNzAddr = (__ubuf__ bfloat16_t *)kNz.GetPhyAddr();
    __ubuf__ float *bAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *eAddr = (__ubuf__ float *)eg.GetPhyAddr();
    const uint16_t nEven = static_cast<uint16_t>(kBt >> 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> one;
        Duplicate(one, 1.0f, pregMerge);
        if constexpr (Which != 2) {
#define GDN32_L2_TAIL                                                                                                  \
    {                                                                                                                  \
        RegTensor<float> g0, g1;                                                                                      \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g0, eAddr + i * 2);                                                  \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g1, eAddr + i * 2 + 1);                                              \
        Mul(e0, e0, g0, pregAll);                                                                                     \
        Mul(o0, o0, g0, pregAll);                                                                                     \
        Mul(e1, e1, g1, pregAll);                                                                                     \
        Mul(o1, o1, g1, pregAll);                                                                                     \
        StoreNzEvenOdd(qeAddr, static_cast<uint32_t>(i) * 2, e0, o0, m16, m128);                                      \
        StoreNzEvenOdd(qeAddr, static_cast<uint32_t>(i) * 2 + 1, e1, o1, m16, m128);                                  \
    }
            GDN32_L2_ROWS(qAddr, qNzAddr)
#undef GDN32_L2_TAIL
        }
        if constexpr (Which != 1) {
#define GDN32_L2_TAIL                                                                                                  \
    {                                                                                                                  \
        RegTensor<float> g0, g1, be0, be1, sc0, sc1;                                                                  \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g0, eAddr + i * 2);                                                  \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g1, eAddr + i * 2 + 1);                                              \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(be0, bAddr + i * 2);                                                 \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(be1, bAddr + i * 2 + 1);                                             \
        Mul(sc0, be0, g0, pregAll);                                                                                   \
        Mul(sc1, be1, g1, pregAll);                                                                                   \
        Muls(sc0, sc0, -1.0f, pregAll);                                                                               \
        Muls(sc1, sc1, -1.0f, pregAll);                                                                               \
        Mul(e0, e0, sc0, pregAll);                                                                                    \
        Mul(o0, o0, sc0, pregAll);                                                                                    \
        Mul(e1, e1, sc1, pregAll);                                                                                    \
        Mul(o1, o1, sc1, pregAll);                                                                                    \
        StoreNzEvenOdd(kbgAddr, static_cast<uint32_t>(i) * 2, e0, o0, m16, m128);                                     \
        StoreNzEvenOdd(kbgAddr, static_cast<uint32_t>(i) * 2 + 1, e1, o1, m16, m128);                                 \
    }
            GDN32_L2_ROWS(kAddr, kNzAddr)
#undef GDN32_L2_TAIL
        }
    }
}
#undef GDN32_L2_ROWS
#undef GDN32_L2_APPLY
#undef GDN32_L2_REDUCE

// k_beta_g and q_exp from the normalized ND that L2 just stored. NZ only.
// This runs after the q/k L1 copy has been issued, so the cube can start
// Stage2 while these scales are still in the vector pipe.
__aicore__ inline void ScaleKbgQexpVf(LocalTensor<bfloat16_t> &q, LocalTensor<bfloat16_t> &k,
                                      LocalTensor<bfloat16_t> &kbgNz, LocalTensor<bfloat16_t> &qexpNz,
                                      LocalTensor<float> &beta, LocalTensor<float> &eg)
{
    constexpr uint32_t kRow = 2 * kVl;
    constexpr uint32_t kPair = 2 * kRow;
    __ubuf__ bfloat16_t *qAddr = (__ubuf__ bfloat16_t *)q.GetPhyAddr();
    __ubuf__ bfloat16_t *kAddr = (__ubuf__ bfloat16_t *)k.GetPhyAddr();
    __ubuf__ bfloat16_t *kbgAddr = (__ubuf__ bfloat16_t *)kbgNz.GetPhyAddr();
    __ubuf__ bfloat16_t *qeAddr = (__ubuf__ bfloat16_t *)qexpNz.GetPhyAddr();
    __ubuf__ float *bAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *eAddr = (__ubuf__ float *)eg.GetPhyAddr();
    const uint16_t nEven = static_cast<uint16_t>(kBt >> 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        uint32_t n64 = 64;
        MaskReg m64 = UpdateMask<bfloat16_t>(n64);
        for (uint16_t i = 0; i < nEven; i++) {
            RegTensor<float> a0, b0, a1, b1, e0, e1, be0, be1, sc0, sc1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(e0, eAddr + i * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(e1, eAddr + i * 2 + 1);
            LoadTwoVL(qAddr, i * kPair, i * kPair + kVl, a0, b0, pregAll);
            LoadTwoVL(qAddr, i * kPair + kRow, i * kPair + kRow + kVl, a1, b1, pregAll);
            Mul(a0, a0, e0, pregAll);
            Mul(b0, b0, e0, pregAll);
            Mul(a1, a1, e1, pregAll);
            Mul(b1, b1, e1, pregAll);
            StoreNzHalf(qeAddr, static_cast<uint32_t>(i) * 2, 0, a0, pregAll, m64);
            StoreNzHalf(qeAddr, static_cast<uint32_t>(i) * 2, 4, b0, pregAll, m64);
            StoreNzHalf(qeAddr, static_cast<uint32_t>(i) * 2 + 1, 0, a1, pregAll, m64);
            StoreNzHalf(qeAddr, static_cast<uint32_t>(i) * 2 + 1, 4, b1, pregAll, m64);

            LoadAlign<float, LoadDist::DIST_BRC_B32>(be0, bAddr + i * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(be1, bAddr + i * 2 + 1);
            Mul(sc0, be0, e0, pregAll);
            Mul(sc1, be1, e1, pregAll);
            LoadTwoVL(kAddr, i * kPair, i * kPair + kVl, a0, b0, pregAll);
            LoadTwoVL(kAddr, i * kPair + kRow, i * kPair + kRow + kVl, a1, b1, pregAll);
            Mul(a0, a0, sc0, pregAll);
            Mul(b0, b0, sc0, pregAll);
            Mul(a1, a1, sc1, pregAll);
            Mul(b1, b1, sc1, pregAll);
            StoreNzHalf(kbgAddr, static_cast<uint32_t>(i) * 2, 0, a0, pregAll, m64);
            StoreNzHalf(kbgAddr, static_cast<uint32_t>(i) * 2, 4, b0, pregAll, m64);
            StoreNzHalf(kbgAddr, static_cast<uint32_t>(i) * 2 + 1, 0, a1, pregAll, m64);
            StoreNzHalf(kbgAddr, static_cast<uint32_t>(i) * 2 + 1, 4, b1, pregAll, m64);
        }
    }
}

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

// Shift index and the 32x32 identity. One VEC_SCOPE replaces the scalar
// SetValue loops that held the first MTE2 off the pipe for more than 1 µs.
// shiftIdx[s, i] = (i >= 2^s && i < 32) ? i - 2^s : 0, matching Gather's
// element index. Eye diagonal is 1, everything else 0.
__aicore__ inline void BuildTablesVF(LocalTensor<uint32_t> &shiftIdx, LocalTensor<float> &eye)
{
    __ubuf__ uint32_t *shAddr = (__ubuf__ uint32_t *)shiftIdx.GetPhyAddr();
    __ubuf__ float *eyeAddr = (__ubuf__ float *)eye.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<uint32_t, MaskPattern::ALL>();
        uint32_t n32 = 32;
        MaskReg m32 = UpdateMask<float>(n32);
        RegTensor<int32_t> idx, src, zero, distV;
        Arange<int32_t, IndexOrder::INCREASE_ORDER>(idx, 0);
        Duplicate(zero, 0, pregAll);
        for (uint16_t s = 0; s < 5; ++s) {
            int32_t dist = 1 << s;
            Duplicate(distV, dist, pregAll);
            Sub(src, idx, distV, pregAll);
            MaskReg ge, lt, keep;
            CompareScalar<int32_t, CMPMODE::GE>(ge, idx, dist, pregAll);
            CompareScalar<int32_t, CMPMODE::LT>(lt, idx, 32, pregAll);
            And(keep, ge, lt, pregAll);
            Select(src, src, zero, keep);
            StoreAlign(shAddr + static_cast<uint32_t>(s) * 64, (RegTensor<uint32_t> &)src, pregAll);
        }
        for (uint16_t r = 0; r < 32; ++r) {
            RegTensor<float> row;
            Duplicate(row, 0.f, m32);
            MaskReg diag;
            CompareScalar<int32_t, CMPMODE::EQ>(diag, idx, static_cast<int32_t>(r), m32);
            Adds(row, row, 1.0f, diag);
            StoreAlign(eyeAddr + static_cast<uint32_t>(r) * kBt, row, m32);
        }
    }
}

// g_cumsum stays in natural units (useExp2=false). eg = exp(g_cumsum).
// Five gather-adds over the 32 live lanes. shiftIdx[s] gathers lane i from
// i-2^s (0 when i < 2^s). StoreRcp writes 1/eg into the next 32 lanes so
// Gate and v_new are multiplies.
template <bool StoreRcp = false>
__aicore__ inline void CumsumExp2VF(LocalTensor<float> &g, LocalTensor<float> &gp, LocalTensor<float> &eg,
                                    LocalTensor<uint32_t> &shiftIdx)
{
    __ubuf__ float *gAddr = (__ubuf__ float *)g.GetPhyAddr();
    __ubuf__ float *gpAddr = (__ubuf__ float *)gp.GetPhyAddr();
    __ubuf__ float *egAddr = (__ubuf__ float *)eg.GetPhyAddr();
    __ubuf__ uint32_t *shAddr = (__ubuf__ uint32_t *)shiftIdx.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> v, zero, shifted;
        RegTensor<uint32_t> ix;
        Duplicate(zero, 0.0f, pregAll);
        LoadAlign(v, gAddr);
        uint32_t n = static_cast<uint32_t>(kBt);
        MaskReg pregN = UpdateMask<float>(n);
        Select(v, v, zero, pregN);
        for (uint16_t s = 0; s < 5; ++s) {
            LoadAlign(ix, shAddr + static_cast<uint32_t>(s) * 64);
            Gather(shifted, v, ix);
            uint32_t cLo = 1U << s;
            uint32_t cHi = static_cast<uint32_t>(kBt);
            MaskReg pregLo = UpdateMask<float>(cLo);
            MaskReg pregHi = UpdateMask<float>(cHi);
            MaskReg pregRange;
            Xor(pregRange, pregHi, pregLo, pregAll);
            Add<float, MaskMergeMode::MERGING>(v, v, shifted, pregRange);
        }
        StoreAlign(gpAddr, v, pregN);
        Exp(v, v, pregN);
        StoreAlign(egAddr, v, pregN);
        if constexpr (StoreRcp) {
            RegTensor<float> oneR, rcp;
            Duplicate(oneR, 1.0f, pregN);
            Div(rcp, oneR, v, pregN);
            StoreAlign(egAddr + 32, rcp, pregN);
        }
    }
}

__aicore__ inline void CopyF32VF(LocalTensor<float> dst, LocalTensor<float> src, uint32_t n)
{
    __ubuf__ float *dAddr = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *sAddr = (__ubuf__ float *)src.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg preg = UpdateMask<float>(n);
        RegTensor<float> v;
        LoadAlign(v, sAddr);
        StoreAlign(dAddr, v, preg);
    }
}

// gp[0] = sum(g) / ln(2). The other half only needs exp of that sum.
__aicore__ inline void SumLastVF(LocalTensor<float> &g, LocalTensor<float> &gp)
{
    __ubuf__ float *gAddr = (__ubuf__ float *)g.GetPhyAddr();
    __ubuf__ float *gpAddr = (__ubuf__ float *)gp.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg1 = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> v, zero, sum;
        Duplicate(zero, 0.0f, pregAll);
        LoadAlign(v, gAddr);
        uint32_t n = static_cast<uint32_t>(kBt);
        MaskReg pregN = UpdateMask<float>(n);
        Select(v, v, zero, pregN);
        Reduce<AscendC::Reg::ReduceType::SUM>(sum, v, pregN);
        Muls(sum, sum, kRcpLn2, preg1);
        StoreAlign(gpAddr, sum, preg1);
    }
}

// y[r,:] = x[r,:] * scale[r]. Two rows per iteration, matching kda's normalize
// loop and prepare ScaleRowsK128VF (16 trips instead of 32).
// EmitNz also writes the tight NZ image into nz (the idle slot copied to L1).
template <bool EmitNz = false>
__aicore__ inline void ScaleRowsVF(LocalTensor<bfloat16_t> &x, LocalTensor<bfloat16_t> &y, LocalTensor<float> &scale,
                                   __ubuf__ bfloat16_t *nz = nullptr)
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
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> s0, s1, a0, b0, a1, b1;
        for (uint16_t p = 0; p < nPair; ++p) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s0, sAddr + p * 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s1, sAddr + p * 2 + 1);
            if constexpr (EmitNz) {
                LoadEvenOdd(xAddr + p * kPairElems, a0, b0, m16);
                LoadEvenOdd(xAddr + p * kPairElems + kRow, a1, b1, m16);
                Mul(a0, a0, s0, pregAll);
                Mul(b0, b0, s0, pregAll);
                Mul(a1, a1, s1, pregAll);
                Mul(b1, b1, s1, pregAll);
                StoreNzEvenOdd(nz, static_cast<uint32_t>(p) * 2, a0, b0, m16, m128);
                StoreNzEvenOdd(nz, static_cast<uint32_t>(p) * 2 + 1, a1, b1, m16, m128);
            } else {
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
}

// y[r,:] = x[r,:] * beta[r] * eg[r]. beta*eg stays in the row broadcast,
// so the 32-wide product is not spilled through UB the way MulBetaExp did.
__aicore__ inline void ScaleRowsBetaExpVF(LocalTensor<bfloat16_t> &x, LocalTensor<bfloat16_t> &y,
                                          LocalTensor<float> &beta, LocalTensor<float> &eg,
                                          __ubuf__ bfloat16_t *nz)
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
        uint32_t n64 = 64;
        MaskReg m64 = UpdateMask<bfloat16_t>(n64);
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
            StoreRow128(yAddr, nz, static_cast<uint32_t>(p) * 2, a0, c0, pregAll, m64);
            StoreRow128(yAddr, nz, static_cast<uint32_t>(p) * 2 + 1, a1, c1, pregAll, m64);
        }
    }
}

// L[i,j] = (i>j) ? beta[i] * eg[i] / eg[j] * kkt : 0
// attn[i,j] = (i>=j) ? eg[i] / eg[j] * qk : 0, stored as bf16.
// eg = exp(g_cumsum), so the ratio is exp(g[i]-g[j]) without a per-row Exp.
// Which: 0 both, 1 L only (kkt is ready, qk is not), 2 attn only.
template <int Which = 0>
__aicore__ inline void GateAttnVF(LocalTensor<float> &eg, LocalTensor<float> &beta, LocalTensor<float> &kkt,
                                  LocalTensor<float> &qk, LocalTensor<float> &lower, LocalTensor<bfloat16_t> &attn)
{
    __ubuf__ float *eAddr = (__ubuf__ float *)eg.GetPhyAddr();
    __ubuf__ float *bAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *kktAddr = (__ubuf__ float *)kkt.GetPhyAddr();
    __ubuf__ float *qkAddr = (__ubuf__ float *)qk.GetPhyAddr();
    __ubuf__ float *lAddr = (__ubuf__ float *)lower.GetPhyAddr();
    __ubuf__ bfloat16_t *aAddr = (__ubuf__ bfloat16_t *)attn.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg preg32 = CreateMask<float, MaskPattern::VL32>();
        RegTensor<float> invJ, eI, bI, gate, kk, qq, outL, outA, zero;
        Duplicate(zero, 0.0f, preg32);
        LoadAlign(invJ, eAddr + 32);
        for (uint16_t i = 0; i < static_cast<uint16_t>(kBt); ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(eI, eAddr + static_cast<uint32_t>(i));
            Mul(gate, eI, invJ, preg32);
            if constexpr (Which != 2) {
                uint32_t lowerCount = static_cast<uint32_t>(i);
                MaskReg lowerMask = UpdateMask<float>(lowerCount);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(bI, bAddr + static_cast<uint32_t>(i));
                LoadAlign(kk, kktAddr + static_cast<uint32_t>(i) * kBt);
                Mul(outL, gate, bI, preg32);
                Mul(outL, outL, kk, preg32);
                Select(outL, outL, zero, lowerMask);
                // Which=1 feeds Solve32, whose update is y += x*L. Negate while
                // the row is still in a register; a later pass over the 32x32
                // block would read and write the whole matrix.
                if constexpr (Which == 1) {
                    Muls(outL, outL, -1.0f, preg32);
                }
                StoreAlign(lAddr + static_cast<uint32_t>(i) * kBt, outL, preg32);
            }
            if constexpr (Which != 1) {
                uint32_t attnCount = static_cast<uint32_t>(i) + 1;
                MaskReg attnMask = UpdateMask<float>(attnCount);
                LoadAlign(qq, qkAddr + static_cast<uint32_t>(i) * kBt);
                Mul(outA, gate, qq, preg32);
                Select(outA, outA, zero, attnMask);
                StoreCastBf16(aAddr + static_cast<uint32_t>(i) * kBt, outA, preg32);
            }
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

// v_new = u - P, v_new_g = v_new * eg[last] / eg[i]. u and P are bf16, one
// register per row. The gate is broadcast in fp32 and split into the same
// even/odd bf16 register the row subtract already uses.
__aicore__ inline void VnewVF(LocalTensor<bfloat16_t> &u, LocalTensor<bfloat16_t> &p, LocalTensor<float> &eg,
                              LocalTensor<bfloat16_t> &vNew, LocalTensor<bfloat16_t> &vNewG,
                              LocalTensor<bfloat16_t> &gateScratch)
{
    (void)gateScratch;
    __ubuf__ bfloat16_t *uAddr = (__ubuf__ bfloat16_t *)u.GetPhyAddr();
    __ubuf__ bfloat16_t *pAddr = (__ubuf__ bfloat16_t *)p.GetPhyAddr();
    __ubuf__ float *eAddr = (__ubuf__ float *)eg.GetPhyAddr();
    __ubuf__ bfloat16_t *nAddr = (__ubuf__ bfloat16_t *)vNew.GetPhyAddr();
    __ubuf__ bfloat16_t *gAddr = (__ubuf__ bfloat16_t *)vNewG.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> last;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(last, eAddr + (kBt - 1));
        for (uint16_t r = 0; r < static_cast<uint16_t>(kBt); ++r) {
            RegTensor<bfloat16_t> a, b, glo, ghi;
            RegTensor<float> ri, gate;
            LoadAlign(a, uAddr + static_cast<uint32_t>(r) * kDim);
            LoadAlign(b, pAddr + static_cast<uint32_t>(r) * kDim);
            Sub(a, a, b, m128);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(nAddr + static_cast<uint32_t>(r) * 16, a,
                                                                  static_cast<uint32_t>(kBt), m128);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ri, eAddr + 32 + r);
            Mul(gate, last, ri, pregAll);
            Cast<bfloat16_t, float, kCastB322B16>(glo, gate, m16);
            Cast<bfloat16_t, float, kCastB322B16Hi>(ghi, gate, m16);
            Or((RegTensor<uint16_t> &)glo, (RegTensor<uint16_t> &)glo, (RegTensor<uint16_t> &)ghi, m16);
            Mul(a, a, glo, m128);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(gAddr + static_cast<uint32_t>(r) * 16, a,
                                                                  static_cast<uint32_t>(kBt), m128);
        }
    }
}

// o = cast_rint(o_acc * scale). o_acc is the bf16 fixpipe. scale stays fp32.
__aicore__ inline void ScaleOVf(LocalTensor<bfloat16_t> &src, LocalTensor<bfloat16_t> &dst, float scale)
{
    __ubuf__ bfloat16_t *sAddr = (__ubuf__ bfloat16_t *)src.GetPhyAddr();
    __ubuf__ bfloat16_t *dAddr = (__ubuf__ bfloat16_t *)dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> sc;
        Duplicate(sc, scale, pregAll);
        for (uint16_t r = 0; r < static_cast<uint16_t>(kBt); ++r) {
            RegTensor<float> even, odd;
            LoadEvenOdd(sAddr + static_cast<uint32_t>(r) * kDim, even, odd, m16);
            Mul(even, even, sc, pregAll);
            Mul(odd, odd, sc, pregAll);
            RegTensor<bfloat16_t> lo, hi;
            Cast<bfloat16_t, float, kCastB322B16>(lo, even, m16);
            Cast<bfloat16_t, float, kCastB322B16Hi>(hi, odd, m16);
            Or((RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)hi, m16);
            StoreAlign(dAddr + static_cast<uint32_t>(r) * kDim, lo, m128);
        }
    }
}

// state = decay1 * (decay0 * state + d0) + d1, one pass over the 64 rows.
// TwoDeltas false is the trailing half: state = decay0 * state + d0.
// State stays ND. UploadH's MTE copies spread it into NZ panels.
template <bool TwoDeltas>
__aicore__ inline void ApplyStateVF(LocalTensor<bfloat16_t> state, LocalTensor<bfloat16_t> d0,
                                    LocalTensor<bfloat16_t> d1, float decay0, float decay1, uint16_t rows)
{
    __ubuf__ bfloat16_t *sAddr = (__ubuf__ bfloat16_t *)state.GetPhyAddr();
    __ubuf__ bfloat16_t *d0Addr = (__ubuf__ bfloat16_t *)d0.GetPhyAddr();
    __ubuf__ bfloat16_t *d1Addr = TwoDeltas ? (__ubuf__ bfloat16_t *)d1.GetPhyAddr() : d0Addr;
    for (uint16_t tile = 0; tile < 2; ++tile) {
        const uint32_t rowBase = static_cast<uint32_t>(tile) * kBt;
        if (rowBase >= rows) {
            break;
        }
        __VEC_SCOPE__
        {
            MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> dec0, dec1;
            Duplicate(dec0, decay0, pregAll);
            if constexpr (TwoDeltas) {
                Duplicate(dec1, decay1, pregAll);
            }
            for (uint16_t lr = 0; lr < static_cast<uint16_t>(kBt); ++lr) {
                const uint32_t r = rowBase + lr;
                RegTensor<float> a, b, da, db;
                LoadCastBf16(sAddr + r * kDim, a, pregAll);
                LoadCastBf16(sAddr + r * kDim + kVl, b, pregAll);
                LoadCastBf16(d0Addr + r * kDim, da, pregAll);
                LoadCastBf16(d0Addr + r * kDim + kVl, db, pregAll);
                // da = d0 + decay0 * state
                MulAddDst(da, a, dec0, pregAll);
                MulAddDst(db, b, dec0, pregAll);
                if constexpr (TwoDeltas) {
                    LoadCastBf16(d1Addr + r * kDim, a, pregAll);
                    LoadCastBf16(d1Addr + r * kDim + kVl, b, pregAll);
                    // a = d1 + decay1 * (d0 + decay0 * state)
                    MulAddDst(a, da, dec1, pregAll);
                    MulAddDst(b, db, dec1, pregAll);
                    StoreCastBf16(sAddr + r * kDim, a, pregAll);
                    StoreCastBf16(sAddr + r * kDim + kVl, b, pregAll);
                } else {
                    StoreCastBf16(sAddr + r * kDim, da, pregAll);
                    StoreCastBf16(sAddr + r * kDim + kVl, db, pregAll);
                }
            }
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

// In-place scale by -1. n64 is the number of 64-wide chunks.
__aicore__ inline void NegateRowsVF(__ubuf__ float *addr, uint16_t n64)
{
    __VEC_SCOPE__
    {
        MaskReg preg = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t i = 0; i < n64; ++i) {
            RegTensor<float> v;
            LoadAlign(v, addr + static_cast<uint32_t>(i) * 64);
            Muls(v, v, -1.0f, preg);
            StoreAlign(addr + static_cast<uint32_t>(i) * 64, v, preg);
        }
    }
}

// (I + Lii) X = I. Eight rows stay live. Earlier rows are reloaded once from UB.
// Sixteen live rows spilled and corrupted the inverse, so the block stays at 8.
// L is negated, so vmula (dst += src * coeff) is the triangular subtraction.
#define GDN32_SOLVE_XT(dst, rr)                                                                                        \
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coeff,                                                                   \
                                             lBase + (row0 + b + (rr)) * static_cast<uint32_t>(kBt) + col0 + t);     \
    MulAddDst(dst, xt, coeff, preg);
#define GDN32_SOLVE_LOC(dst, src, rr, cc)                                                                              \
    LoadAlign<float, LoadDist::DIST_BRC_B32>(                                                                         \
        coeff, lBase + (row0 + b + (rr)) * static_cast<uint32_t>(kBt) + col0 + b + (cc));                            \
    MulAddDst(dst, src, coeff, preg);

__aicore__ inline void Solve16VF(__ubuf__ float *lBase, __ubuf__ float *xAddr, __ubuf__ float *eye, uint32_t row0,
                                 uint32_t col0)
{
    for (int group = 0; group < 2; ++group) {
        const uint32_t b = static_cast<uint32_t>(group) * 8;
        __VEC_SCOPE__
        {
            uint32_t n16 = 16;
            MaskReg preg = UpdateMask<float>(n16);
            RegTensor<float> y0, y1, y2, y3, y4, y5, y6, y7, coeff, xt;
            LoadAlign(y0, eye + (b + 0) * 16);
            LoadAlign(y1, eye + (b + 1) * 16);
            LoadAlign(y2, eye + (b + 2) * 16);
            LoadAlign(y3, eye + (b + 3) * 16);
            LoadAlign(y4, eye + (b + 4) * 16);
            LoadAlign(y5, eye + (b + 5) * 16);
            LoadAlign(y6, eye + (b + 6) * 16);
            LoadAlign(y7, eye + (b + 7) * 16);
            for (uint16_t t = 0; t < static_cast<uint16_t>(b); ++t) {
                LoadAlign(xt, xAddr + static_cast<uint32_t>(t) * 16);
                GDN32_SOLVE_XT(y0, 0)
                GDN32_SOLVE_XT(y1, 1)
                GDN32_SOLVE_XT(y2, 2)
                GDN32_SOLVE_XT(y3, 3)
                GDN32_SOLVE_XT(y4, 4)
                GDN32_SOLVE_XT(y5, 5)
                GDN32_SOLVE_XT(y6, 6)
                GDN32_SOLVE_XT(y7, 7)
            }
            GDN32_SOLVE_LOC(y1, y0, 1, 0)
            GDN32_SOLVE_LOC(y2, y0, 2, 0)
            GDN32_SOLVE_LOC(y2, y1, 2, 1)
            GDN32_SOLVE_LOC(y3, y0, 3, 0)
            GDN32_SOLVE_LOC(y3, y1, 3, 1)
            GDN32_SOLVE_LOC(y3, y2, 3, 2)
            GDN32_SOLVE_LOC(y4, y0, 4, 0)
            GDN32_SOLVE_LOC(y4, y1, 4, 1)
            GDN32_SOLVE_LOC(y4, y2, 4, 2)
            GDN32_SOLVE_LOC(y4, y3, 4, 3)
            GDN32_SOLVE_LOC(y5, y0, 5, 0)
            GDN32_SOLVE_LOC(y5, y1, 5, 1)
            GDN32_SOLVE_LOC(y5, y2, 5, 2)
            GDN32_SOLVE_LOC(y5, y3, 5, 3)
            GDN32_SOLVE_LOC(y5, y4, 5, 4)
            GDN32_SOLVE_LOC(y6, y0, 6, 0)
            GDN32_SOLVE_LOC(y6, y1, 6, 1)
            GDN32_SOLVE_LOC(y6, y2, 6, 2)
            GDN32_SOLVE_LOC(y6, y3, 6, 3)
            GDN32_SOLVE_LOC(y6, y4, 6, 4)
            GDN32_SOLVE_LOC(y6, y5, 6, 5)
            GDN32_SOLVE_LOC(y7, y0, 7, 0)
            GDN32_SOLVE_LOC(y7, y1, 7, 1)
            GDN32_SOLVE_LOC(y7, y2, 7, 2)
            GDN32_SOLVE_LOC(y7, y3, 7, 3)
            GDN32_SOLVE_LOC(y7, y4, 7, 4)
            GDN32_SOLVE_LOC(y7, y5, 7, 5)
            GDN32_SOLVE_LOC(y7, y6, 7, 6)
            StoreAlign(xAddr + (b + 0) * 16, y0, preg);
            StoreAlign(xAddr + (b + 1) * 16, y1, preg);
            StoreAlign(xAddr + (b + 2) * 16, y2, preg);
            StoreAlign(xAddr + (b + 3) * 16, y3, preg);
            StoreAlign(xAddr + (b + 4) * 16, y4, preg);
            StoreAlign(xAddr + (b + 5) * 16, y5, preg);
            StoreAlign(xAddr + (b + 6) * 16, y6, preg);
            StoreAlign(xAddr + (b + 7) * 16, y7, preg);
        }
        PipeBarrier<PIPE_V>();
    }
}
#undef GDN32_SOLVE_XT
#undef GDN32_SOLVE_LOC

// C[row,:] = scale * sum_k left[row,k] * right[k,:], 16x16.
// Eight accumulators stay live for the whole K loop, so the partial is never
// stored and reloaded. A right row is loaded once and applied to all eight.
#define GDN32_GEMM_MAC(acc, row)                                                                                       \
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coeff, left + (base + (row)) * leftStride + kk);                         \
    MulAddDst(acc, rv, coeff, preg);

__aicore__ inline void Gemm16VF(__ubuf__ float *left, uint32_t leftStride, __ubuf__ float *right, __ubuf__ float *dst,
                                float scale)
{
    for (uint16_t group = 0; group < 2; ++group) {
        const uint32_t base = static_cast<uint32_t>(group) * 8;
        __VEC_SCOPE__
        {
            uint32_t n16 = 16;
            MaskReg preg = UpdateMask<float>(n16);
            RegTensor<float> a0, a1, a2, a3, a4, a5, a6, a7, rv, coeff;
            Duplicate(a0, 0.f, preg);
            Duplicate(a1, 0.f, preg);
            Duplicate(a2, 0.f, preg);
            Duplicate(a3, 0.f, preg);
            Duplicate(a4, 0.f, preg);
            Duplicate(a5, 0.f, preg);
            Duplicate(a6, 0.f, preg);
            Duplicate(a7, 0.f, preg);
            for (uint16_t kk = 0; kk < 16; ++kk) {
                LoadAlign(rv, right + static_cast<uint32_t>(kk) * 16);
                GDN32_GEMM_MAC(a0, 0);
                GDN32_GEMM_MAC(a1, 1);
                GDN32_GEMM_MAC(a2, 2);
                GDN32_GEMM_MAC(a3, 3);
                GDN32_GEMM_MAC(a4, 4);
                GDN32_GEMM_MAC(a5, 5);
                GDN32_GEMM_MAC(a6, 6);
                GDN32_GEMM_MAC(a7, 7);
            }
            Muls(a0, a0, scale, preg);
            Muls(a1, a1, scale, preg);
            Muls(a2, a2, scale, preg);
            Muls(a3, a3, scale, preg);
            Muls(a4, a4, scale, preg);
            Muls(a5, a5, scale, preg);
            Muls(a6, a6, scale, preg);
            Muls(a7, a7, scale, preg);
            StoreAlign(dst + (base + 0) * 16, a0, preg);
            StoreAlign(dst + (base + 1) * 16, a1, preg);
            StoreAlign(dst + (base + 2) * 16, a2, preg);
            StoreAlign(dst + (base + 3) * 16, a3, preg);
            StoreAlign(dst + (base + 4) * 16, a4, preg);
            StoreAlign(dst + (base + 5) * 16, a5, preg);
            StoreAlign(dst + (base + 6) * 16, a6, preg);
            StoreAlign(dst + (base + 7) * 16, a7, preg);
        }
    }
}
#undef GDN32_GEMM_MAC

// (I + L) X = I, X written in place at stride kBt. Eight rows live. Earlier
// rows are reloaded. One 32-wide instruction replaces the 16-wide block gemm.
#define GDN32_SOLVE32_XT(dst, rr)                                                                                      \
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coeff, lBase + (b + (rr)) * static_cast<uint32_t>(kBt) + t);             \
    MulAddDst(dst, xt, coeff, preg);
#define GDN32_SOLVE32_LOC(dst, src, rr, cc)                                                                            \
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coeff,                                                                   \
                                             lBase + (b + (rr)) * static_cast<uint32_t>(kBt) + b + (cc));            \
    MulAddDst(dst, src, coeff, preg);

__aicore__ inline void Solve32VF(__ubuf__ float *lBase, __ubuf__ float *xAddr, __ubuf__ float *eye)
{
    for (int group = 0; group < 4; ++group) {
        const uint32_t b = static_cast<uint32_t>(group) * 8;
        __VEC_SCOPE__
        {
            uint32_t n32 = 32;
            MaskReg preg = UpdateMask<float>(n32);
            RegTensor<float> y0, y1, y2, y3, y4, y5, y6, y7, coeff, xt;
            LoadAlign(y0, eye + (b + 0) * kBt);
            LoadAlign(y1, eye + (b + 1) * kBt);
            LoadAlign(y2, eye + (b + 2) * kBt);
            LoadAlign(y3, eye + (b + 3) * kBt);
            LoadAlign(y4, eye + (b + 4) * kBt);
            LoadAlign(y5, eye + (b + 5) * kBt);
            LoadAlign(y6, eye + (b + 6) * kBt);
            LoadAlign(y7, eye + (b + 7) * kBt);
            for (uint16_t t = 0; t < static_cast<uint16_t>(b); ++t) {
                LoadAlign(xt, xAddr + static_cast<uint32_t>(t) * kBt);
                GDN32_SOLVE32_XT(y0, 0)
                GDN32_SOLVE32_XT(y1, 1)
                GDN32_SOLVE32_XT(y2, 2)
                GDN32_SOLVE32_XT(y3, 3)
                GDN32_SOLVE32_XT(y4, 4)
                GDN32_SOLVE32_XT(y5, 5)
                GDN32_SOLVE32_XT(y6, 6)
                GDN32_SOLVE32_XT(y7, 7)
            }
            GDN32_SOLVE32_LOC(y1, y0, 1, 0)
            GDN32_SOLVE32_LOC(y2, y0, 2, 0)
            GDN32_SOLVE32_LOC(y2, y1, 2, 1)
            GDN32_SOLVE32_LOC(y3, y0, 3, 0)
            GDN32_SOLVE32_LOC(y3, y1, 3, 1)
            GDN32_SOLVE32_LOC(y3, y2, 3, 2)
            GDN32_SOLVE32_LOC(y4, y0, 4, 0)
            GDN32_SOLVE32_LOC(y4, y1, 4, 1)
            GDN32_SOLVE32_LOC(y4, y2, 4, 2)
            GDN32_SOLVE32_LOC(y4, y3, 4, 3)
            GDN32_SOLVE32_LOC(y5, y0, 5, 0)
            GDN32_SOLVE32_LOC(y5, y1, 5, 1)
            GDN32_SOLVE32_LOC(y5, y2, 5, 2)
            GDN32_SOLVE32_LOC(y5, y3, 5, 3)
            GDN32_SOLVE32_LOC(y5, y4, 5, 4)
            GDN32_SOLVE32_LOC(y6, y0, 6, 0)
            GDN32_SOLVE32_LOC(y6, y1, 6, 1)
            GDN32_SOLVE32_LOC(y6, y2, 6, 2)
            GDN32_SOLVE32_LOC(y6, y3, 6, 3)
            GDN32_SOLVE32_LOC(y6, y4, 6, 4)
            GDN32_SOLVE32_LOC(y6, y5, 6, 5)
            GDN32_SOLVE32_LOC(y7, y0, 7, 0)
            GDN32_SOLVE32_LOC(y7, y1, 7, 1)
            GDN32_SOLVE32_LOC(y7, y2, 7, 2)
            GDN32_SOLVE32_LOC(y7, y3, 7, 3)
            GDN32_SOLVE32_LOC(y7, y4, 7, 4)
            GDN32_SOLVE32_LOC(y7, y5, 7, 5)
            GDN32_SOLVE32_LOC(y7, y6, 7, 6)
            StoreAlign(xAddr + (b + 0) * kBt, y0, preg);
            StoreAlign(xAddr + (b + 1) * kBt, y1, preg);
            StoreAlign(xAddr + (b + 2) * kBt, y2, preg);
            StoreAlign(xAddr + (b + 3) * kBt, y3, preg);
            StoreAlign(xAddr + (b + 4) * kBt, y4, preg);
            StoreAlign(xAddr + (b + 5) * kBt, y5, preg);
            StoreAlign(xAddr + (b + 6) * kBt, y6, preg);
            StoreAlign(xAddr + (b + 7) * kBt, y7, preg);
        }
        PipeBarrier<PIPE_V>();
    }
}
#undef GDN32_SOLVE32_XT
#undef GDN32_SOLVE32_LOC

// 16 packed rows into a 32-wide row, starting at column col0. Mask is 16, so the
// other half of the row stays whatever the caller wrote (zero, for the off-block).
__aicore__ inline void Scatter16VF(__ubuf__ float *dst, __ubuf__ float *src, uint32_t row0, uint32_t col0)
{
    __VEC_SCOPE__
    {
        uint32_t n16 = 16;
        MaskReg preg = UpdateMask<float>(n16);
        for (uint16_t r = 0; r < 16; ++r) {
            RegTensor<float> v;
            LoadAlign(v, src + static_cast<uint32_t>(r) * 16);
            StoreAlign(dst + (row0 + r) * static_cast<uint32_t>(kBt) + col0, v, preg);
        }
    }
}


// (I+L)^{-1}. GateAttn Which=1 already stored -L, so the solve's add is the
// triangular subtraction. L is left negated: the next GateAttn overwrites it,
// and the attn pass does not read it.
__aicore__ inline void InvertAVF(LocalTensor<float> lower, LocalTensor<float> a, LocalTensor<bfloat16_t> aBf,
                                 LocalTensor<float> eye)
{
    __ubuf__ float *lAddr = (__ubuf__ float *)lower.GetPhyAddr();
    __ubuf__ float *aAddr = (__ubuf__ float *)a.GetPhyAddr();
    __ubuf__ float *eyeAddr = (__ubuf__ float *)eye.GetPhyAddr();
    // Solve16Live matches the digits but the 64-wide vlds made the vector pipe
    // slower (vec 6.3 -> 11). Solve32 stays the fast path.
    Solve32VF(lAddr, aAddr, eyeAddr);
    Cast(aBf, a, RoundMode::CAST_RINT, kBt * kBt);
}

}  // namespace GdnFwd32

#endif
