/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Vector VF for chunk_gated_delta_rule_fwd_prepare.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_PREPARE_VF_H
#define CHUNK_GATED_DELTA_RULE_FWD_PREPARE_VF_H

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_fwd_prepare_common.h"

namespace ChunkGatedDeltaRuleFwdPrepare {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr int32_t VL = 64;

constexpr CastTrait kCastB162B32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr CastTrait kCastB322B16 = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

/**
 * function: 从 UB 取 1 个 VL，cast 成 fp32（bf16/fp16 走 unpack）。
 * input:  src (__ubuf__ T*, T=bf16/fp16/fp32), mask
 * output: dst (RegTensor<fp32>)
 */
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

/**
 * function: 把标量 gate 参数 cast 成 fp32（a_log / dt_bias）。
 * input:  x (T=bf16/fp16/fp32)
 * output: fp32 标量
 */
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

/**
 * function: 把 1 个 VL 的 fp32 cast 回 T 并写 UB（bf16/fp16 走 pack）。
 * input:  src (RegTensor<fp32>), mask
 * output: dst (__ubuf__ T*, T=bf16/fp16/fp32)
 */
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

/**
 * function: 连续 load 两个 VL（64+64）并 cast 成 fp32，结果留在寄存器给 L2Norm 复用。
 * input:  addr (__ubuf__ T*, T=bf16/fp16/fp32), offset0/offset1（元素偏移）
 * output: a, b (RegTensor<fp32>)
 */
template <typename T>
__aicore__ inline void LoadTwoVLFp32(__ubuf__ T *addr, uint32_t offset0, uint32_t offset1, RegTensor<float> &a,
                                     RegTensor<float> &b, MaskReg &pregLoop)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> aB16, bB16;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(aB16, addr + offset0);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(bB16, addr + offset1);
        Cast<float, half, kCastB162B32>(a, aB16, pregLoop);
        Cast<float, half, kCastB162B32>(b, bB16, pregLoop);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> aB16, bB16;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(aB16, addr + offset0);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(bB16, addr + offset1);
        Cast<float, bfloat16_t, kCastB162B32>(a, aB16, pregLoop);
        Cast<float, bfloat16_t, kCastB162B32>(b, bB16, pregLoop);
    } else {
        LoadAlign(a, addr + offset0);
        LoadAlign(b, addr + offset1);
    }
}

/**
 * function: 行方向 L2Norm，K=128。rstd = 1/sqrt(sum_k x^2 + eps)，y = x * rstd。
 *           偶数行两两处理，奇数尾行单独处理。
 * input:  x [rows,128] T(bf16/fp16/fp32), rows, eps(fp32, default 1e-6)
 * output: y [rows,128] T, rstd [rows] fp32
 */
template <typename T>
__aicore__ inline void L2NormK128VF(LocalTensor<T> &xLocal, LocalTensor<T> &yLocal, LocalTensor<float> &rstdLocal,
                                    uint32_t curRows, float eps = 1e-6f)
{
    constexpr uint32_t K = 2 * VL;
    constexpr uint32_t kPair = 2 * K;
    __ubuf__ T *xAddr = (__ubuf__ T *)xLocal.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yLocal.GetPhyAddr();
    __ubuf__ float *rstdAddr = (__ubuf__ float *)rstdLocal.GetPhyAddr();
    const uint16_t nRows = static_cast<uint16_t>(curRows);
    const uint16_t nEven = nRows >> 1;
    const uint16_t nTail = nRows & 1;

    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> one;
        Duplicate(one, 1.0f, pregMerge);

        for (uint16_t i = 0; i < nEven; i++) {
            RegTensor<float> x0a, x0b, x1a, x1b, s0a, s0b, s1a, s1b;
            RegTensor<float> mean0, mean1, r0, r1, brc0, brc1, y0a, y0b, y1a, y1b;

            LoadTwoVLFp32<T>(xAddr, i * kPair, i * kPair + VL, x0a, x0b, pregAll);
            LoadTwoVLFp32<T>(xAddr, i * kPair + K, i * kPair + K + VL, x1a, x1b, pregAll);

            Mul(s0a, x0a, x0a, pregAll);
            Mul(s0b, x0b, x0b, pregAll);
            Mul(s1a, x1a, x1a, pregAll);
            Mul(s1b, x1b, x1b, pregAll);
            Add(s0a, s0a, s0b, pregAll);
            Add(s1a, s1a, s1b, pregAll);
            Reduce<AscendC::Reg::ReduceType::SUM>(mean0, s0a, pregAll);
            Reduce<AscendC::Reg::ReduceType::SUM>(mean1, s1a, pregAll);
            Adds(mean0, mean0, eps, pregMerge);
            Adds(mean1, mean1, eps, pregMerge);
            Sqrt(mean0, mean0, pregMerge);
            Sqrt(mean1, mean1, pregMerge);
            Div(r0, one, mean0, pregMerge);
            Div(r1, one, mean1, pregMerge);
            Duplicate(brc0, r0, pregAll);
            Duplicate(brc1, r1, pregAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + i * 2, r0, pregMerge);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + i * 2 + 1, r1, pregMerge);

            Mul(y0a, x0a, brc0, pregAll);
            Mul(y0b, x0b, brc0, pregAll);
            Mul(y1a, x1a, brc1, pregAll);
            Mul(y1b, x1b, brc1, pregAll);
            StoreCastB16<T>(yAddr + i * kPair, y0a, pregAll);
            StoreCastB16<T>(yAddr + i * kPair + VL, y0b, pregAll);
            StoreCastB16<T>(yAddr + i * kPair + K, y1a, pregAll);
            StoreCastB16<T>(yAddr + i * kPair + K + VL, y1b, pregAll);
        }
        for (uint16_t t = 0; t < nTail; t++) {
            const uint32_t row = static_cast<uint32_t>(nEven) * 2 + t;
            RegTensor<float> xa, xb, sa, sb, vMean, rstdReg, brc, y0, y1;
            LoadTwoVLFp32<T>(xAddr, row * K, row * K + VL, xa, xb, pregAll);
            Mul(sa, xa, xa, pregAll);
            Mul(sb, xb, xb, pregAll);
            Add(sa, sa, sb, pregAll);
            Reduce<AscendC::Reg::ReduceType::SUM>(vMean, sa, pregAll);
            Adds(vMean, vMean, eps, pregMerge);
            Sqrt(vMean, vMean, pregMerge);
            Div(rstdReg, one, vMean, pregMerge);
            Duplicate(brc, rstdReg, pregAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + row, rstdReg, pregMerge);
            Mul(y0, xa, brc, pregAll);
            Mul(y1, xb, brc, pregAll);
            StoreCastB16<T>(yAddr + row * K, y0, pregAll);
            StoreCastB16<T>(yAddr + row * K + VL, y1, pregAll);
        }
    }
}

/**
 * function: 原地 inclusive prefix-sum 再乘 scale。x[t] = scale * sum_{i<=t} x[i]。
 *           use_exp2 时 scale=RCP_LN2，否则 1。
 * input:  x [n] fp32, scale(fp32), n
 * output: x [n] fp32（原地）
 */
__aicore__ inline void CumsumScaleVF(LocalTensor<float> &x, float scale, uint32_t n)
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

/**
 * function: 融合 gate。g_raw = -exp(a_log) * softplus(g + dt_bias)，
 *           softplus(x) = relu(x) + ln(1 + exp(-|x|))。
 * input:  gIn [n] T(bf16/fp32), aLog(fp32), dtBias(fp32), n
 * output: gRaw [n] fp32
 */
template <typename T>
__aicore__ inline void GateSoftplusVF(LocalTensor<T> &gIn, LocalTensor<float> &gRaw, float aLog, float dtBias,
                                      uint32_t n)
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

/**
 * function: 逐元素 T -> fp32，长度 n。
 * input:  src [n] T(bf16/fp16)
 * output: dst [n] fp32
 */
template <typename T>
__aicore__ inline void CastToFp32VF(LocalTensor<T> &src, LocalTensor<float> &dst, uint32_t n)
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

/**
 * function: beta_out = scale * sigmoid(beta)。allow_neg_eigval 时 scale=2，否则 1。
 * input:  betaIn [n] T(bf16/fp32), scale(fp32), n
 * output: betaOut [n] fp32
 */
template <typename T>
__aicore__ inline void BetaSigmoidVF(LocalTensor<T> &betaIn, LocalTensor<float> &betaOut, float scale, uint32_t n)
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

/**
 * function: 构造 MBH 用的 -L。整块写 64x64。
 *           -L[i,j] = -mask[i,j] * beta[i] * kkt[i,j] * exp2(clip(g[i]-g[j], -50, 50))。
 *           mask 是严格下三角（i>j 为 1）。
 * input:  kkt [64,64] fp32, g [64] fp32, beta [64] fp32, mask [64,64] fp32
 * output: L [64,64] fp32（即 -L）
 */
__aicore__ inline void NegLowerLVF(LocalTensor<float> &kkt, LocalTensor<float> &g, LocalTensor<float> &beta,
                                   LocalTensor<float> &mask, LocalTensor<float> &L)
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
    }
}

/**
 * function: 按行乘 scale，最后一维 128。y[t,:] = x[t,:] * scale[t]。
 * input:  x [rows,128] T(bf16), scale [rows] fp32, rows
 * output: y [rows,128] T
 */
template <typename T>
__aicore__ inline void ScaleRowsK128VF(LocalTensor<T> &x, LocalTensor<T> &y, LocalTensor<float> &scale, uint32_t rows)
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

/**
 * function: 按行乘 scale，最后一维 256。y[t,:] = x[t,:] * scale[t]。
 * input:  x [rows,256] T(bf16), scale [rows] fp32, rows
 * output: y [rows,256] T
 */
template <typename T>
__aicore__ inline void ScaleRowsK256VF(LocalTensor<T> &x, LocalTensor<T> &y, LocalTensor<float> &scale, uint32_t rows)
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

/**
 * function: kbg 缩放，最后一维 128。y[t,:] = x[t,:] * beta[t] * exp2(g[t])。
 * input:  x [rows,128] T(bf16), beta [rows] fp32, g [rows] fp32, rows
 * output: y [rows,128] T
 */
template <typename T>
__aicore__ inline void ScaleRowsBetaExp2gVF(LocalTensor<T> &x, LocalTensor<T> &y, LocalTensor<float> &beta,
                                            LocalTensor<float> &g, uint32_t rows)
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

/**
 * function: 按行乘 scale，最后一维 128 或 256。y[t,:] = x[t,:] * scale[t]。
 * input:  x [rows,dim] T(bf16), scale [rows] fp32, rows, dim∈{128,256}
 * output: y [rows,dim] T
 */
template <typename T>
__aicore__ inline void ScaleRowsVF(LocalTensor<T> &x, LocalTensor<T> &y, LocalTensor<float> &scale, uint32_t rows,
                                   uint32_t dim)
{
    if (dim == 4 * VL) {
        ScaleRowsK256VF<T>(x, y, scale, rows);
        return;
    }
    ScaleRowsK128VF<T>(x, y, scale, rows);
}

/**
 * function: 画 64x64 严格下三角 mask：col < row 为 1，否则 0。
 * input:  无
 * output: ubMask [64,64] fp32
 */
__aicore__ inline void LowerTriMaskVF(AscendC::LocalTensor<float> ubMaskFp32)
{
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

/**
 * function: 画 64x64 单位阵（对角为 1）。
 * input:  无
 * output: ubI [64,64] fp32
 */
__aicore__ inline void Identity64VF(AscendC::LocalTensor<float> ubI)
{
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

/**
 * function: 画 32x64 VCS 单位阵，(i,i) 和 (i, i+32) 为 1。
 * input:  无
 * output: ubVcsI [32,64] fp32
 */
__aicore__ inline void VcsIdentityVF(AscendC::LocalTensor<float> ubVcsI)
{
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

/**
 * function: 打包 32x32 叶子的 VCS 求逆。dst[i, idx] = sum_k src0[j,k] * src1[i,k]，idx∈{0,32}。
 *           dst 与 src1 可 alias，原地覆盖 I_vcs。scatterCount=2，oneRepeatSize=64。
 * input:  src0 L_packed [32,64] fp32, src1 I_vcs [32,64] fp32, idx [2] uint32 {0,32}
 * output: dst packed (I+Lii)^{-1} [32,64] fp32
 */
__aicore__ inline void MulReduceScatterVF32(LocalTensor<float> &dst, LocalTensor<float> &src0, LocalTensor<float> &src1,
                                            LocalTensor<uint32_t> &idx)
{
    __ubuf__ float *dstAddr = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *src0Addr = (__ubuf__ float *)src0.GetPhyAddr();
    __ubuf__ float *src1Addr = (__ubuf__ float *)src1.GetPhyAddr();
    __ubuf__ uint32_t *idxAddr = (__ubuf__ uint32_t *)idx.GetPhyAddr();
    uint32_t scatterCount = kLeavesPerVec32;
    uint32_t oneRepeatSize = kVcsPack32;
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg0;
        RegTensor<float> srcReg1a;
        RegTensor<float> srcReg1b;
        RegTensor<float> mulA;
        RegTensor<float> mulB;
        RegTensor<float> blkA;
        RegTensor<float> blkB;
        RegTensor<float> pairA;
        RegTensor<float> pairB;
        RegTensor<float> pair2A;
        RegTensor<float> pair2B;
        RegTensor<uint32_t> idxBase;
        RegTensor<uint32_t> scatterIdxReg;

        uint32_t maskCount = oneRepeatSize;
        uint32_t pairMaskCount = scatterCount * 2;
        MaskReg inputMask = UpdateMask<float, RegTraitNumOne>(maskCount);
        MaskReg pairMask = UpdateMask<float, RegTraitNumOne>(pairMaskCount);
        MaskReg scatterMask = UpdateMask<float, RegTraitNumOne>(scatterCount);
        LoadAlign(idxBase, idxAddr);
        for (uint16_t iterIdx = 1; iterIdx < 32; iterIdx++) {
            Adds(scatterIdxReg, idxBase, (uint32_t)iterIdx, scatterMask);
            LoadAlign(srcReg0, src0Addr + iterIdx * oneRepeatSize);
            const uint16_t nPair = static_cast<uint16_t>(iterIdx >> 1);
            const uint16_t nTail = static_cast<uint16_t>(iterIdx & 1);
            for (uint16_t p = 0; p < nPair; p++) {
                const uint32_t i0 = static_cast<uint32_t>(p) * 2;
                const uint32_t i1 = i0 + 1;
                LoadAlign(srcReg1a, src1Addr + i0 * oneRepeatSize);
                LoadAlign(srcReg1b, src1Addr + i1 * oneRepeatSize);
                Mul(mulA, srcReg0, srcReg1a, inputMask);
                Mul(mulB, srcReg0, srcReg1b, inputMask);
                ReduceDataBlock<ReduceType::SUM>(blkA, mulA, inputMask);
                ReduceDataBlock<ReduceType::SUM>(blkB, mulB, inputMask);
                PairReduceElem<PairReduce::SUM>(pairA, blkA, inputMask);
                PairReduceElem<PairReduce::SUM>(pairB, blkB, inputMask);
                PairReduceElem<PairReduce::SUM>(pair2A, pairA, pairMask);
                PairReduceElem<PairReduce::SUM>(pair2B, pairB, pairMask);
                Scatter(dstAddr + i0 * oneRepeatSize, pair2A, scatterIdxReg, scatterMask);
                Scatter(dstAddr + i1 * oneRepeatSize, pair2B, scatterIdxReg, scatterMask);
            }
            for (uint16_t t = 0; t < nTail; t++) {
                const uint32_t i = static_cast<uint32_t>(nPair) * 2;
                LoadAlign(srcReg1a, src1Addr + i * oneRepeatSize);
                Mul(mulA, srcReg0, srcReg1a, inputMask);
                ReduceDataBlock<ReduceType::SUM>(blkA, mulA, inputMask);
                PairReduceElem<PairReduce::SUM>(pairA, blkA, inputMask);
                PairReduceElem<PairReduce::SUM>(pair2A, pairA, pairMask);
                Scatter(dstAddr + i * oneRepeatSize, pair2A, scatterIdxReg, scatterMask);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }
}

} // namespace ChunkGatedDeltaRuleFwdPrepare

#endif
