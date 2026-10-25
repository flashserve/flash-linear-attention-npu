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

// k_hat = k / sqrt(sum(k^2) + eps), stored NZ. kbg = -k_hat * beta * eg, stored NZ.
// The negation makes the later A@kbg equal -W, so w@H accumulates -P into u.
__aicore__ inline void L2NormKAndKbgVF(LocalTensor<bfloat16_t> &k, LocalTensor<bfloat16_t> &kNz,
                                       LocalTensor<bfloat16_t> &kbgNz, LocalTensor<float> &beta, LocalTensor<float> &eg)
{
    constexpr uint32_t kRow = 2 * kVl;
    constexpr uint32_t kPair = 2 * kRow;
    __ubuf__ bfloat16_t *kAddr = (__ubuf__ bfloat16_t *)k.GetPhyAddr();
    __ubuf__ bfloat16_t *kbgAddr = (__ubuf__ bfloat16_t *)kbgNz.GetPhyAddr();
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

// q_hat = q / sqrt(sum(q^2) + eps), stored NZ. q_exp = q_hat * eg, stored NZ.
// q@H consumes q_exp; the normalized q is the Q@K operand.
__aicore__ inline void L2NormQAndQexpVF(LocalTensor<bfloat16_t> &q, LocalTensor<bfloat16_t> &qNz,
                                        LocalTensor<bfloat16_t> &qexpNz, LocalTensor<float> &eg)
{
    constexpr uint32_t kRow = 2 * kVl;
    constexpr uint32_t kPair = 2 * kRow;
    __ubuf__ bfloat16_t *qAddr = (__ubuf__ bfloat16_t *)q.GetPhyAddr();
    __ubuf__ bfloat16_t *qeAddr = (__ubuf__ bfloat16_t *)qexpNz.GetPhyAddr();
    __ubuf__ bfloat16_t *qNzAddr = (__ubuf__ bfloat16_t *)qNz.GetPhyAddr();
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
}
#undef GDN32_L2_ROWS
#undef GDN32_L2_APPLY
#undef GDN32_L2_REDUCE


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

// 辅助矩阵生成。shiftIdx 是 cumsum 的 Sklansky 下标，eye 是 32×32 单位阵。
// 一个 VEC_SCOPE 做完，避免标量 SetValue 把第一条 MTE2 挡在 vector 之后。
// shiftIdx[s, i] = (i >= 2^s && i < 32) ? i - 2^s : 0, matching Gather's
// element index. Eye diagonal is 1, everything else 0.
__aicore__ inline void GenAuxMatrixVF(LocalTensor<uint32_t> &shiftIdx, LocalTensor<float> &eye)
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
__aicore__ inline void CumsumExp2VF(LocalTensor<float> &g, LocalTensor<float> &eg,
                                    LocalTensor<uint32_t> &shiftIdx)
{
    __ubuf__ float *gAddr = (__ubuf__ float *)g.GetPhyAddr();
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


// y[r,:] = x[r,:] * scale[r], written as tight NZ. Two rows per iteration.
__aicore__ inline void ScaleRowsVF(LocalTensor<bfloat16_t> &x, LocalTensor<float> &scale, __ubuf__ bfloat16_t *nz)
{
    constexpr uint32_t kRow = 2 * kVl;
    constexpr uint32_t kPairElems = 2 * kRow;
    __ubuf__ bfloat16_t *xAddr = (__ubuf__ bfloat16_t *)x.GetPhyAddr();
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
            LoadEvenOdd(xAddr + p * kPairElems, a0, b0, m16);
            LoadEvenOdd(xAddr + p * kPairElems + kRow, a1, b1, m16);
            Mul(a0, a0, s0, pregAll);
            Mul(b0, b0, s0, pregAll);
            Mul(a1, a1, s1, pregAll);
            Mul(b1, b1, s1, pregAll);
            StoreNzEvenOdd(nz, static_cast<uint32_t>(p) * 2, a0, b0, m16, m128);
            StoreNzEvenOdd(nz, static_cast<uint32_t>(p) * 2 + 1, a1, b1, m16, m128);
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

// 16 ND rows of u-p, the half this AIV received from the fixpipe.
// global row = rowBase + r, so the gate reads that lane of eg.
__aicore__ inline void VnewHalfVF(LocalTensor<bfloat16_t> u, LocalTensor<float> eg, LocalTensor<bfloat16_t> vNew,
                                  LocalTensor<bfloat16_t> vNewG, uint32_t rowBase, uint16_t r0, uint16_t nRows)
{
    __ubuf__ bfloat16_t *uAddr = (__ubuf__ bfloat16_t *)u.GetPhyAddr();
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
        for (uint16_t k = 0; k < nRows; ++k) {
            const uint16_t r = static_cast<uint16_t>(r0 + k);
            RegTensor<bfloat16_t> a, glo, ghi;
            RegTensor<float> ri, gate;
            RegTensor<float> even, odd;
            LoadAlign(a, uAddr + static_cast<uint32_t>(r) * kDim);
            // Tight NZ, panel height 16. Each column panel is 16 rows.
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(nAddr + static_cast<uint32_t>(r) * 16, a, 16,
                                                                  m128);
            // Gate stays fp32 through the multiply. One CAST_RINT writes v_new_g.
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ri, eAddr + 32 + rowBase + r);
            Mul(gate, last, ri, pregAll);
            Cast<float, bfloat16_t, kCastB162B32>(even, a, m16);
            Cast<float, bfloat16_t, kCastB162B32Hi>(odd, a, m16);
            Mul(even, even, gate, pregAll);
            Mul(odd, odd, gate, pregAll);
            Cast<bfloat16_t, float, kCastB322B16>(glo, even, m16);
            Cast<bfloat16_t, float, kCastB322B16Hi>(ghi, odd, m16);
            Or((RegTensor<uint16_t> &)glo, (RegTensor<uint16_t> &)glo, (RegTensor<uint16_t> &)ghi, m16);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(gAddr + static_cast<uint32_t>(r) * 16, glo, 16,
                                                                  m128);
        }
    }
}

// Dense 128 bf16. Even/odd is the same per-element saxpy; Or writes the row back.
__aicore__ inline void StoreFlat128(__ubuf__ bfloat16_t *dst, RegTensor<float> &even, RegTensor<float> &odd,
                                    MaskReg &m16, MaskReg &m128)
{
    RegTensor<bfloat16_t> lo, hi;
    Cast<bfloat16_t, float, kCastB322B16>(lo, even, m16);
    Cast<bfloat16_t, float, kCastB322B16Hi>(hi, odd, m16);
    Or((RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)hi, m16);
    StoreAlign(dst, lo, m128);
}

// state = decay * state + d over one AIV's 64x128 tile. Both buffers are the
// NZ panel image (8 contiguous [64, 16] panels), so this is a flat saxpy.
// Four rows stay live. Their casts overlap, and each D load sits under the
// previous row's store. Mul then Add keeps the one-row rounding.
// i0/nIter select a run of the 16 flat groups (512 elements). Groups 0-7 are
// panels 0-3, so that half can be copied to L1 while groups 8-15 run.
__aicore__ inline void SaxpyStateVF(LocalTensor<bfloat16_t> state, LocalTensor<bfloat16_t> d, float decay,
                                    uint16_t i0 = 0, uint16_t nIter = 16)
{
    __ubuf__ bfloat16_t *sAddr = (__ubuf__ bfloat16_t *)state.GetPhyAddr();
    __ubuf__ bfloat16_t *dAddr = (__ubuf__ bfloat16_t *)d.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg preg = CreateMask<float, MaskPattern::ALL>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> dec;
        Duplicate(dec, decay, preg);
        for (uint16_t i = 0; i < nIter; ++i) {
            const uint32_t off = (static_cast<uint32_t>(i0) + i) * 512;
            RegTensor<float> e0, o0, e1, o1, e2, o2, e3, o3, f, g;
            LoadEvenOdd(sAddr + off, e0, o0, m16);
            LoadEvenOdd(sAddr + off + 128, e1, o1, m16);
            LoadEvenOdd(sAddr + off + 256, e2, o2, m16);
            LoadEvenOdd(sAddr + off + 384, e3, o3, m16);
            Mul(e0, e0, dec, preg);
            Mul(o0, o0, dec, preg);
            Mul(e1, e1, dec, preg);
            Mul(o1, o1, dec, preg);
            Mul(e2, e2, dec, preg);
            Mul(o2, o2, dec, preg);
            Mul(e3, e3, dec, preg);
            Mul(o3, o3, dec, preg);
            LoadEvenOdd(dAddr + off, f, g, m16);
            Add(e0, e0, f, preg);
            Add(o0, o0, g, preg);
            LoadEvenOdd(dAddr + off + 128, f, g, m16);
            StoreFlat128(sAddr + off, e0, o0, m16, m128);
            Add(e1, e1, f, preg);
            Add(o1, o1, g, preg);
            LoadEvenOdd(dAddr + off + 256, f, g, m16);
            StoreFlat128(sAddr + off + 128, e1, o1, m16, m128);
            Add(e2, e2, f, preg);
            Add(o2, o2, g, preg);
            LoadEvenOdd(dAddr + off + 384, f, g, m16);
            StoreFlat128(sAddr + off + 256, e2, o2, m16, m128);
            Add(e3, e3, f, preg);
            Add(o3, o3, g, preg);
            StoreFlat128(sAddr + off + 384, e3, o3, m16, m128);
        }
    }
}

// [64, 128] ND <-> 8 contiguous [64, 16] panels. Used at the head and the tail
// only; the per-chunk state stays in panels.
__aicore__ inline void NdToPanels(LocalTensor<bfloat16_t> nz, LocalTensor<bfloat16_t> nd)
{
    __ubuf__ bfloat16_t *nAddr = (__ubuf__ bfloat16_t *)nd.GetPhyAddr();
    __ubuf__ bfloat16_t *zAddr = (__ubuf__ bfloat16_t *)nz.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        for (uint16_t r = 0; r < 64; ++r) {
            RegTensor<bfloat16_t> row;
            LoadAlign(row, nAddr + static_cast<uint32_t>(r) * kDim);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(zAddr + static_cast<uint32_t>(r) * 16, row, 64,
                                                                  m128);
        }
    }
}

__aicore__ inline void PanelsToNd(LocalTensor<bfloat16_t> nd, LocalTensor<bfloat16_t> nz)
{
    __ubuf__ bfloat16_t *nAddr = (__ubuf__ bfloat16_t *)nd.GetPhyAddr();
    __ubuf__ bfloat16_t *zAddr = (__ubuf__ bfloat16_t *)nz.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        for (uint16_t r = 0; r < 64; ++r) {
            RegTensor<bfloat16_t> row;
            LoadAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(row, zAddr + static_cast<uint32_t>(r) * 16, 64, m128);
            StoreAlign(nAddr + static_cast<uint32_t>(r) * kDim, row, m128);
        }
    }
}

// One vector register. Used to keep this chunk's eg while the next chunk fills ubEg_.
__aicore__ inline void Copy64(LocalTensor<float> dst, LocalTensor<float> src)
{
    __ubuf__ float *sAddr = (__ubuf__ float *)src.GetPhyAddr();
    __ubuf__ float *dAddr = (__ubuf__ float *)dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg preg = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> v;
        LoadAlign(v, sAddr);
        StoreAlign(dAddr, v, preg);
    }
}

}  // namespace GdnFwd32

#endif
