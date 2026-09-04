/*!
 * \file l2norm_regbase_vf.h
 * \brief Stage1 L2Norm VF (regbase). Square keeps x in regs; rstd via Duplicate.
 *
 * Ascend 950 Regbase VF for FLA ``l2norm_fwd``.
 *
 * Formula (Triton / CPU golden, **not** RMSNorm / lp_norm_v2)::
 *
 *     rstd = 1 / sqrt(sum_k x_k^2 + eps)     # fp32, one scalar per row
 *     y    = x * rstd
 *
 * Differences vs ``rms_norm_regbase_common.h``:
 *   - no ``Muls(..., 1/K)``
 *   - no ``gamma``
 *   - ``eps`` is added **inside** the square root (FLA default ``1e-6``)
 *
 * Instruction pattern is copied from that file:
 *   LoadAlign + Cast + Mul(square) + Reduce SUM +
 *   Adds(eps) + Sqrt + Duplicate(1) + Div
 * (library uses Sqrt+Div instead of Rsqrt for fp32 accuracy).
 *
 * Preconditions:
 *   - last dim already in UB; pad unused rows/cols with 0
 *   - GDN ``K=128`` is exactly 2 VL on 950 (``V_LENGTH=64`` fp32)
 */
#ifndef GDN_L2NORM_REGBASE_VF_H
#define GDN_L2NORM_REGBASE_VF_H

#include "kernel_operator.h"

namespace GdnL2Norm {
using namespace AscendC;
using namespace AscendC::MicroAPI;

namespace {
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

constexpr int32_t VL_SIZE = GetVRegSize();
constexpr int32_t V_LENGTH = VL_SIZE / static_cast<int32_t>(sizeof(float)); // 64 on 950

constexpr CastTrait castTraitB162B32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr CastTrait castTraitB322B16 = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
} // namespace

template <typename T>
__aicore__ inline void LoadCastX(RegTensor<float>& dst, __ubuf__ T* src, MaskReg& mask)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src);
    } else {
        RegTensor<T> xB16;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xB16, src);
        Cast<float, T, castTraitB162B32>(dst, xB16, mask);
    }
}

template <typename T>
__aicore__ inline void StoreY(__ubuf__ T* dst, RegTensor<float>& src, MaskReg& mask)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign(dst, src, mask);
    } else {
        RegTensor<T> yB16;
        Cast<T, float, castTraitB322B16>(yB16, src, mask);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, yB16, mask);
    }
}

/*!
 * Load two VL as fp32. Dual-issue: two Loads then two Casts, originals kept
 * so the scale pass does not reload UB.
 */
template <typename T>
__aicore__ inline void LoadTwoVLFp32(
    __ubuf__ T* addr, uint32_t offset0, uint32_t offset1,
    RegTensor<float>& a, RegTensor<float>& b, MaskReg& pregLoop)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> aB16, bB16;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(aB16, addr + offset0);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(bB16, addr + offset1);
        Cast<float, half, castTraitB162B32>(a, aB16, pregLoop);
        Cast<float, half, castTraitB162B32>(b, bB16, pregLoop);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> aB16, bB16;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(aB16, addr + offset0);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(bB16, addr + offset1);
        Cast<float, bfloat16_t, castTraitB162B32>(a, aB16, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(b, bB16, pregLoop);
    } else {
        LoadAlign(a, addr + offset0);
        LoadAlign(b, addr + offset1);
    }
}

/*!
 * GDN ``K=128`` (exactly 2 VL).
 *
 * VF handbook:
 *   - hoist masks / Duplicate(1) out of the row loop (loop instruction dist)
 *   - keep x in registers (VF fusion, no UB reload after square)
 *   - Duplicate(LOWEST) broadcasts rstd, skip DIST_FIRST store + MemBar + DIST_BRC
 *   - unroll 2 rows so independent Loads/Muls can dual-issue
 *   - address as ``base + i * const`` for AddrReg
 */
template <typename T>
__aicore__ inline void L2NormFwdK128VF(
    LocalTensor<T>& xLocal,
    LocalTensor<T>& yLocal,
    LocalTensor<float>& rstdLocal,
    uint32_t curRows,
    float eps = 1e-6f)
{
    constexpr uint32_t K = 2 * V_LENGTH; // 128
    constexpr uint32_t kPair = 2 * K;    // two rows
    __ubuf__ T* xAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ T* yAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
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

            LoadTwoVLFp32<T>(xAddr, i * kPair, i * kPair + V_LENGTH, x0a, x0b, pregAll);
            LoadTwoVLFp32<T>(xAddr, i * kPair + K, i * kPair + K + V_LENGTH, x1a, x1b, pregAll);

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
            StoreY<T>(yAddr + i * kPair, y0a, pregAll);
            StoreY<T>(yAddr + i * kPair + V_LENGTH, y0b, pregAll);
            StoreY<T>(yAddr + i * kPair + K, y1a, pregAll);
            StoreY<T>(yAddr + i * kPair + K + V_LENGTH, y1b, pregAll);
        }
        for (uint16_t t = 0; t < nTail; t++) {
            const uint32_t row = static_cast<uint32_t>(nEven) * 2 + t;
            RegTensor<float> xa, xb, sa, sb, vMean, rstdReg, brc, y0, y1;
            LoadTwoVLFp32<T>(xAddr, row * K, row * K + V_LENGTH, xa, xb, pregAll);
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
            StoreY<T>(yAddr + row * K, y0, pregAll);
            StoreY<T>(yAddr + row * K + V_LENGTH, y1, pregAll);
        }
    }
}

} // namespace GdnL2Norm

#endif // GDN_L2NORM_REGBASE_VF_H
