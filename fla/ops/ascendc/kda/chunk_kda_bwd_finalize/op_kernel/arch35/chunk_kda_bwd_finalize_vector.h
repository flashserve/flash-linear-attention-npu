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

// 将两组交错 FP32 lane 按 RNE 舍入打包为完整 BF16 行。
__simd_callee__ inline void FinalizeCastBf16(
    RegTensor<bfloat16_t> &dst, RegTensor<float> &even,
    RegTensor<float> &odd, MaskReg &mask)
{
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(dst, odd, mask);
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(dst, even, mask);
}

// ZERO 布局的 Cast 在每个 32bit lane 放一个 BF16 值；
// 先在寄存器内打包 64 个值，再散写到 64 行 NZ 平面。
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

// 将有效行之后的 NZ padding 清零，供固定物理尺寸的 Cube tile 安全读取。
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

// Stage5 尾块：同时清零五项 NZ 操作数的高低位无效行。
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

// 128 列向量拆成 BF16 高位与舍入残差，分别写两个 NZ 平面。
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

// 64 列矩阵行拆成 BF16 高位与舍入残差，分别写两个 NZ 平面。
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

// Stage0 / Vector：计算 E=exp2(gk)、kE 高低位、g_last 和 r_h。
// kE 直接生成为 NZ 平面，供 MTE3 连续写入 L1；E 保留 FP32。
__simd_vf__ inline void FinalizeStage0VF(
    __ubuf__ bfloat16_t *kENd, __ubuf__ bfloat16_t *lowNd,
    __ubuf__ float *exp2Gk, __ubuf__ float *gkLast, __ubuf__ float *rH,
    __ubuf__ bfloat16_t *k, __ubuf__ float *gk,
    __ubuf__ bfloat16_t *h, __ubuf__ bfloat16_t *dh,
    uint16_t validRows)
{
    // 1. 准备 FP32/BF16 mask 与 NZ 列地址；逐有效行计算 E 和 kE 高低位。
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

    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, gk + rowOffset);
        Muls(e0, g0, KDA_FINALIZE_LN2, fpMask);
        Muls(e1, g1, KDA_FINALIZE_LN2, fpMask);
        Exp(e0, e0, fpMask);
        Exp(e1, e1, fpMask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(exp2Gk + rowOffset, e0, e1, fpMask);

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
    }

    // 2. 无效行的 kE 高低位都置零，保持 Cube 的固定 64 行物理输入。
    if (validRows < KDA_FINALIZE_CHUNK) {
        Duplicate(out, static_cast<bfloat16_t>(0), bfMask);
        for (uint16_t row = validRows; row < KDA_FINALIZE_CHUNK; ++row) {
            Scatter(kENd + row * 16, out, nzIndex, bfMask);
            Scatter(lowNd + row * 16, out, nzIndex, bfMask);
        }
    }

    // 3. 提取最后一个有效 token 的完整 K 维 gate，不能用物理第 63 行替代尾行。
    const uint32_t lastOffset = static_cast<uint32_t>(validRows - 1) * KDA_FINALIZE_DIM;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, gk + lastOffset);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(gkLast, g0, g1, fpMask);

    // 4. 沿 V 维归约 r_h[k] = sum_v(h[k,v] * dh[k,v])，保留 FP32。
    for (uint16_t row = 0; row < KDA_FINALIZE_DIM; ++row) {
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
}

// Stage2 / Vector：Zb = tril(zV - zW, -1) * beta[None, :]。
// zV/zW 来自 Fixpipe 的 FP32 UB 交接；Zb 高低位直接按 NZ 布局生成。
template <typename BetaT>
__simd_vf__ inline void FinalizeStage2VF(
    __ubuf__ bfloat16_t *zbNd,
    __ubuf__ float *zV, __ubuf__ float *zW,
    __ubuf__ BetaT *beta, uint16_t validRows)
{
    // 1. 将整行 beta 读成 FP32，无效列置零；beta 沿矩阵列广播。
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
    uint32_t betaCount = validRows;
    MaskReg betaMask = UpdateMask<float>(betaCount);
    Select(betaFp, betaFp, zero, betaMask);
    RegTensor<uint16_t> column, block, nzIndex;
    MaskReg indexMask = CreateMask<half, MaskPattern::ALL>();
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), indexMask);
    Muls(block, block, uint16_t(1008), indexMask);
    Add(nzIndex, column, block, indexMask);

    // 2. 逐行计算 (zV-zW)*beta，仅保留严格下三角，再生成高低位 NZ。
    for (uint16_t row = 0; row < validRows; ++row) {
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

    // 3. 清零高低位尾行，保持后续 64×64 GEMM 的 padding 为零。
    if (validRows < KDA_FINALIZE_CHUNK) {
        FinalizeZeroNzTail(zbNd, validRows, KDA_FINALIZE_CHUNK);
        FinalizeZeroNzTail(zbNd + 4096, validRows, KDA_FINALIZE_CHUNK);
    }
}

// Stage3 / Vector：整 chunk 计算 dk_state、dv、dq_base、db_v 与 gate_state。
// 两组 FP32 寄存器覆盖 128 个特征；消费后原位复用 DVb 区保存 dq_base。
template <typename BetaT>
__simd_vf__ inline void FinalizeStage3VF(
    __ubuf__ float *dkState, __ubuf__ float *dvb,
    __ubuf__ bfloat16_t *dv, __ubuf__ float *gateState, __ubuf__ float *dbV,
    __ubuf__ float *dqRaw, __ubuf__ float *exp2Gk, __ubuf__ float *gk,
    __ubuf__ bfloat16_t *k, __ubuf__ bfloat16_t *v,
    __ubuf__ BetaT *beta, __ubuf__ float *gkLast, __ubuf__ float *rH,
    float scale, uint16_t validRows)
{
    // 1. 载入 g_last，初始化逐特征 gate 累加器。
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> last0;
    RegTensor<float> last1;
    RegTensor<float> gate0;
    RegTensor<float> gate1;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(last0, last1, gkLast);
    Duplicate(gate0, 0.0f, fpMask);
    Duplicate(gate1, 0.0f, fpMask);

    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;

        // 2. dk_state = dk_state_raw * exp2(g_last-gk)，原位写回。
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

        // 3. 累加 k*dk_state，形成 gate_state 的 token 贡献。
        RegTensor<bfloat16_t> kb;
        RegTensor<float> k0;
        RegTensor<float> k1;
        LoadIn<bfloat16_t, false>(kb, k + rowOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        Mul(k0, k0, state0, fpMask);
        Mul(k1, k1, state1, fpMask);
        Add(gate0, gate0, k0, fpMask);
        Add(gate1, gate1, k1, fpMask);

        // 4. dv = beta*DVb，转换为 BF16；原始 FP32 DVb 仍用于 db_v。
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

        // 5. db_v = sum_V(v*DVb)，每个 token 归约为一个 FP32 标量。
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

        // 6. dq_base = dq_raw*E*scale，复用已消费完的 DVb 区。
        RegTensor<float> dq0;
        RegTensor<float> dq1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(dq0, dq1, dqRaw + rowOffset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(exp0, exp1, exp2Gk + rowOffset);
        Mul(dq0, dq0, exp0, fpMask);
        Mul(dq1, dq1, exp1, fpMask);
        Muls(dq0, dq0, scale, fpMask);
        Muls(dq1, dq1, scale, fpMask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dvb + rowOffset, dq0, dq1, fpMask);
    }

    // 7. gate_state 再加 r_h*exp2(g_last)，交给 Stage4 的末行修正。
    RegTensor<float> lastExp0;
    RegTensor<float> lastExp1;
    RegTensor<float> rh0;
    RegTensor<float> rh1;
    const uint32_t lastOffset = static_cast<uint32_t>(validRows - 1) * KDA_FINALIZE_DIM;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
        lastExp0, lastExp1, exp2Gk + lastOffset);
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(rh0, rh1, rH);
    Mul(rh0, rh0, lastExp0, fpMask);
    Mul(rh1, rh1, lastExp1, fpMask);
    Add(gate0, gate0, rh0, fpMask);
    Add(gate1, gate1, rh1, fpMask);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(gateState, gate0, gate1, fpMask);
}

// Stage4 / Vector：原位完成 dk_base、db_base、dg_base。
// dkState→dk_base，dKgbRaw→dg_base，dbV→db_base；dq_base 与 E 继续存活。
template <typename BetaT>
__simd_vf__ inline void FinalizeStage4VF(
    __ubuf__ float *dkState, __ubuf__ float *dqBase,
    __ubuf__ float *dKgbRaw, __ubuf__ float *exp2Gk,
    __ubuf__ bfloat16_t *q,
    __ubuf__ bfloat16_t *k, __ubuf__ BetaT *beta,
    __ubuf__ float *gateState, __ubuf__ float *dbV,
    uint16_t validRows)
{
    // 1. 载入 gate_state；逐行读取状态结果、dq_base、dKgb_raw 与 E。
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> gate0;
    RegTensor<float> gate1;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(gate0, gate1, gateState);

    for (uint16_t row = 0; row < validRows; ++row) {
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
            exp0, exp1, exp2Gk + rowOffset);

        RegTensor<bfloat16_t> betaBf;
        RegTensor<float> betaFp;
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            LoadIn<float, true>(betaFp, beta + row);
        } else {
            LoadIn<bfloat16_t, true>(betaBf, beta + row);
            Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, bfMask);
        }

        // 2. dk_base = dk_state-beta*E*dKgb_raw，原位覆盖 dkState。
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

        // 3. dg_base 先取 q*dq_base-k*dk_state，再计算共用项 k*E*dKgb_raw。
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
        // 向量梯度使用 FP32 的 E*dKgb*k，不读取 Cube 使用的 BF16 kE。
        // 复用刚计算的 E*dKgb，保持现有乘法顺序。
        Mul(dkgKe0, dkgExp0, k0, fpMask);
        Mul(dkgKe1, dkgExp1, k1, fpMask);

        // 4. db_base = db_v-sum_K(k*E*dKgb_raw)，原位覆盖 dbV。
        RegTensor<float> reduceInput;
        RegTensor<float> reduceSum;
        RegTensor<float> dbBase;
        Add(reduceInput, dkgKe0, dkgKe1, fpMask);
        ReduceSum(reduceSum, reduceInput, fpMask);
        LoadIn<float, true>(dbBase, dbV + row);
        Sub(dbBase, dbBase, reduceSum, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbV + row, dbBase, fpMask);

        // 5. dg_base 再减 beta*k*E*dKgb_raw，仅有效末行加 gate_state。
        Mul(dkgKe0, dkgKe0, betaFp, fpMask);
        Mul(dkgKe1, dkgKe1, betaFp, fpMask);
        Sub(dg0, dg0, dkgKe0, fpMask);
        Sub(dg1, dg1, dkgKe1, fpMask);
        if (row + 1U == validRows) {
            Add(dg0, dg0, gate0, fpMask);
            Add(dg1, dg1, gate1, fpMask);
        }
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dKgbRaw + rowOffset, dg0, dg1, fpMask);
    }
}

// Stage5 / Vector：按 32 行输出带选择 gate 中心，生成后续 Cube 所需的高低位 NZ 操作数。
#include "chunk_kda_bwd_finalize_intra.h"

// Stage5 数学变换：本函数的 exp2Gk 入参起初是 gk，当前带被原位改写为平移指数。
template <typename BetaT>
__simd_vf__ inline void FinalizeStage5VF(
    __ubuf__ FinalizeLocalType *dAkkNd, __ubuf__ FinalizeLocalType *kNegNd,
    __ubuf__ FinalizeLocalType *qPosNd, __ubuf__ FinalizeLocalType *bkPosNd,
    __ubuf__ float *dAkkRaw, __ubuf__ bfloat16_t *q,
    __ubuf__ bfloat16_t *k, __ubuf__ float *exp2Gk,
    __ubuf__ BetaT *beta, __ubuf__ float *center, uint16_t validRows,
    uint16_t rowBegin, uint16_t rowEnd)
{
    // 1. 读取带中心与 NZ 地址映射；后续平移在两侧 GEMM 操作数中抵消。
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
    // NZ 地址为 [column/16, row, column%16]，偏移单位是 BF16 元素；
    // VF 直接组织 L1 物理布局，MTE3 只需连续复制高低位平面。
    RegTensor<uint16_t> column, block, nzIndex;
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), bfMask);
    Muls(block, block, uint16_t(1008), bfMask);
    Add(nzIndex, column, block, bfMask);
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t matrixOffset =
            static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;

        // 2. 第一带生成 dAkk：严格下三角取负并乘 2^16，再拆高低位。
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
            e0, e1, exp2Gk + vectorOffset);

        RegTensor<bfloat16_t> kBf;
        RegTensor<float> k0;
        RegTensor<float> k1;
        RegTensor<float> kNeg0;
        RegTensor<float> kNeg1;
        RegTensor<bfloat16_t> packed;
        LoadIn<bfloat16_t, false>(kBf, k + vectorOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kBf, bfMask);
        // 当前带不可能消费的因果范围外行直接写零，避免无效指数计算及高低位转换。
        // 3. kNeg = k*exp2(center-gk)，只计算当前带可能消费的前缀行。
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

        // 4. qPos=q*exp2(gk-center)，bkPos=beta*k*exp2(gk-center)。
        // 当前带保存平移指数供 Stage7/9 还原；其余不参与的行写零。
        if (row >= rowBegin) {
            Sub(e0, e0, center0, fpMask);
            Sub(e1, e1, center1, fpMask);
            Muls(e0, e0, KDA_FINALIZE_LN2, fpMask);
            Muls(e1, e1, KDA_FINALIZE_LN2, fpMask);
            Exp(e0, e0, fpMask);
            Exp(e1, e1, fpMask);
            if (row < rowEnd) {
                StoreAlign<float, StoreDist::DIST_INTLV_B32>(exp2Gk + vectorOffset, e0, e1, fpMask);
            }
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

// Stage5：单独保留 dAqk 对角项，将严格下三角乘 2^16 后拆成 BF16 高低位。
__simd_vf__ inline void FinalizeStage5DaqkVF(
    __ubuf__ FinalizeLocalType *dAqkOut, __ubuf__ float *dAqk,
    __ubuf__ float *diagonal, uint16_t validRows)
{
    // 1. 准备 FP32 掩码与 NZ 索引，独立保留原始对角项。
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
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        RegTensor<float> value;
        RegTensor<float> diag;

        // 2. 对角项原样保存；严格下三角放大 2^16 后拆高低位，交给分带 GEMM。
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

// Stage7 / Vector：合并 dq_base 与 Cube 局部结果，累加 FP32 dg，
// 另加对角项，可选执行 Q 归一化反向，最后生成 BF16 dq。
__simd_vf__ inline void FinalizeStage7VF(
    __ubuf__ bfloat16_t *dqOut, __ubuf__ float *dgBase,
    __ubuf__ float *dqLocalRaw, __ubuf__ float *dqBase,
    __ubuf__ float *exp2Gk, __ubuf__ bfloat16_t *q,
    __ubuf__ float *qRstd, uint32_t hasQkL2Norm, uint16_t validRows,
    __ubuf__ float *diagonal, __ubuf__ bfloat16_t *k)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> raw0;
        RegTensor<float> raw1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        RegTensor<float> dq0;
        RegTensor<float> dq1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            raw0, raw1, dqLocalRaw + offset);

        // 1. 先乘当前带的平移指数，再乘 2^-16 还原局部梯度；保持两次缩放的舍入顺序。
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(exp0, exp1, exp2Gk + offset);
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

        // 2. dg 累加 q*dq_local；此时尚未加入无 gate 导数的对角项。
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

        // 3. 对角指数 exp2(g_i-g_i)=1，gate 导数为零；仅补入 dq。
        // 对角项不进缩放 GEMM，避免较大舍入项相减污染较小的 gate 梯度。
        RegTensor<bfloat16_t> kb;
        RegTensor<float> k0, k1, diag;
        LoadIn<bfloat16_t, false>(kb, k + offset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(diag, diagonal + row);
        Mul(k0, k0, diag, fpMask);
        Mul(k1, k1, diag, fpMask);
        Add(dq0, dq0, k0, fpMask);
        Add(dq1, dq1, k1, fpMask);

        // 4. 可选 Q 归一化反向：rstd*(dq-q*sum(dq*q))。
        if (hasQkL2Norm != 0U) {
            RegTensor<float> dot;
            RegTensor<float> rstd;
            Mul(product0, dq0, q0, fpMask);
            Mul(product1, dq1, q1, fpMask);
            Add(product0, product0, product1, fpMask);
            ReduceSum(dot, product0, fpMask);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstd, qRstd + row);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                qRstd + row, dot, scalarMask);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(dot, qRstd + row);
            Mul(product0, q0, dot, fpMask);
            Mul(product1, q1, dot, fpMask);
            Sub(dq0, dq0, product0, fpMask);
            Sub(dq1, dq1, product1, fpMask);
            Mul(dq0, dq0, rstd, fpMask);
            Mul(dq1, dq1, rstd, fpMask);
        }

        // 5. 按原舍入方式生成 BF16 dq，q 输入仍保留供后续 K 路径使用。
        RegTensor<bfloat16_t> packed;
        FinalizeCastBf16(packed, dq0, dq1, fpMask);
        StoreAlign(dqOut + offset, packed, bfMask);
    }
}

// Stage9 / Vector：还原 left/right，生成 dk 局部增量、db 增量，并更新 dg。
template <typename BetaT>
__simd_vf__ inline void FinalizeStage9VF(
    __ubuf__ float *left, __ubuf__ float *right, __ubuf__ float *dbDelta,
    __ubuf__ float *dg, __ubuf__ float *expG,
    __ubuf__ bfloat16_t *k, __ubuf__ BetaT *beta, uint16_t rows,
    __ubuf__ float *diagonal, __ubuf__ bfloat16_t *q)
{
    // 1. 保证前序 VF 写入可见；逐行恢复平移指数和 2^-16 缩放。
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t offset = row * KDA_FINALIZE_DIM;
        RegTensor<float> l0, l1, r0, r1, e0, e1, k0, k1, g0, g1, b;
        RegTensor<bfloat16_t> kb, bb;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(l0, l1, left + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(r0, r1, right + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(e0, e1, expG + offset);
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

        // 2. db_delta = sum_K(left*k)，在 left 乘 beta 前归约。
        RegTensor<float> p0, p1, sum;
        Mul(p0, l0, k0, mask);
        Mul(p1, l1, k1, mask);
        Add(p0, p0, p1, mask);
        ReduceSum(sum, p0, mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbDelta + row, sum, one);

        // 3. dg 累加 k*(beta*left-right)，保留到完整 chunk 的 gate 扫描。
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

        // 4. dk_delta = beta*left+right，再补 dAqk 对角项*q；原位存回 left。
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

// Stage10 / Vector：合并 dk/db 的 base 与增量，执行可选 K 归一化反向。
template <typename BetaT>
__simd_vf__ inline void FinalizeStage10VF(
    __ubuf__ bfloat16_t *out, __ubuf__ BetaT *dbOut,
    __ubuf__ float *delta, __ubuf__ float *base,
    __ubuf__ float *dbDelta, __ubuf__ float *dbBase,
    __ubuf__ bfloat16_t *k, __ubuf__ float *rstd,
    uint32_t hasNorm, uint16_t rows)
{
    // 1. 读取 Stage9 的增量与 base，逐行合并 dk。
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t offset = row * KDA_FINALIZE_DIM;
        RegTensor<float> d0, d1, b0, b1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(d0, d1, delta + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(b0, b1, base + offset);
        Add(d0, d0, b0, mask);
        Add(d1, d1, b1, mask);

        // 2. 可选 K 归一化反向：rstd*(dk-k*sum(dk*k))。
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
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstd + row, dot, one);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(dot, rstd + row);
            Mul(p0, k0, dot, mask);
            Mul(p1, k1, dot, mask);
            Sub(d0, d0, p0, mask);
            Sub(d1, d1, p1, mask);
            Mul(d0, d0, rs, mask);
            Mul(d1, d1, rs, mask);
        }

        // 3. 生成 BF16 dk 输出。
        RegTensor<bfloat16_t> packed;
        FinalizeCastBf16(packed, d0, d1, mask);
        StoreAlign(out + offset, packed, bfMask);
    }

    // 4. 合并 db_base+db_delta，仅存有效行，并保持 beta 的接口类型。
    uint32_t count = rows;
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

// dW 残差转换：每行 128 列，low = BF16(raw - FP32(BF16(raw)))，直接写 NZ。
__simd_vf__ inline void FinalizeDwResidualVF(
    __ubuf__ bfloat16_t *dst, __ubuf__ float *src)
{
    // 1. 准备 NZ 地址，FP32 原值按 RNE 转成与 Fixpipe 高位一致的 BF16。
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<uint16_t> column, block, nzIndex;
    Arange(reinterpret_cast<RegTensor<int16_t> &>(column), int16_t(0));
    ShiftRights(block, column, int16_t(4), bfMask);
    Muls(block, block, uint16_t(1008), bfMask);
    Add(nzIndex, column, block, bfMask);

    // 2. 逐行减去高位的 FP32 值，将差值转 BF16 写入低位平面。
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

// Tza 残差转换：每行 64 列，从 FP32 原值减去 BF16 高位，生成低位 NZ。
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

class ChunkKdaBwdFinalizeVectorStage12 {
public:
    // 绑定 AIV 输入输出、缓存策略、UB 及本地事件；不在此提交阶段计算。
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
        GM_ADDR h, GM_ADDR dh,
        GM_ADDR dAqk, GM_ADDR dqRaw, GM_ADDR qRstd, GM_ADDR dq, GM_ADDR dv,
        GM_ADDR kRstd, GM_ADDR dk, GM_ADDR dBeta,
        GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR dG,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
        const ChunkKdaBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
    {
        // 1. 绑定张量及可选 Q/K 归一化数据。
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
        // 最终输出只流式写出，关闭 L2 缓存，为输入与循环 workspace 留出容量。
        dq_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dk_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dv_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dG_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dBeta_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // 这些输入每 tile 只读一次，采用流式读取。
        rawG_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dAqk_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dqRaw_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // q/k/beta 只载入一次，保留在 UB 跨所有分带使用。
        q_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        k_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        beta_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        // 2. 记录任务元数据与 1C2V 映射，申请 248 KiB UB。
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

        // 3. 为轮转交接分配事件，预置首轮 FREE；共享工作区另用阶段事件保护。
        for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
            mte2ToV_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToMte2_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            // 首个 Stage0 没有前序 Zb 搬出读者，预置可写通知。
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

    // Vector 调度：先完成基础结果，再按带处理局部梯度，最后执行整 chunk gate 扫描。
    __aicore__ inline void Process()
    {
        // 1. 将物理 AIV 编号归一化为逻辑 AIC 编号，按相同 workTask 划分任务。
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
            ResolveFinalizeChunk(chunkTask, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }

            // 2. 即使本 AIV 没有有效 head，也要消费当前任务的 L1 可写通知。
            // 否则单头尾窗可能提前覆盖上一任务 Stage8 仍读取的 qPos/bkPos。
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                KDA_FINALIZE_TASK_L1_FREE);

            // StateAndBase 使用跨 slot 的共享 UB；本次 Stage0 覆写前必须等前序写回结束。
            // 消费后重新发布阶段通知，供本次 StateAndBase 使用。
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);

            // 3. Stage0：AIV 独立于 AIC 计算 E/kE/r_h。
            // generation 的低位选择 AIV，次低位选择交接 slot；双头窗口每 AIV 一头。
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunStage0(chunk, head, owner, slot, logicalCore, groupGeneration);
            }
            // Stage0 的共享 UB 写回完成后，才能让 Cube 在重叠区写 dW/zV/zW。
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);

            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_)
                    continue;
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
                if ((generation & 1U) != subBlockIdx_)
                    continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);

                // 4. dW 残差交接：读取 Cube 原值、生成低位并通知 Stage1 可消费。
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

                // 5. Stage2：等 zV/zW，生成 Zb 高低位并交给 Cube Stage3。
                RunStage2(chunk, head, owner, slot);
            }

            // 6. StateAndBase 将复用 BuildZ 的 UB；先等两份 Zb 的 UB→L1 读取结束。
            for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
            }
            generation -= static_cast<uint64_t>(headEnd - headBegin);
            // Stage3/4 向量路径：本 AIV 完成 BuildZ 后即可处理状态与 base，
            // 与 Cube 的 Tza/dAkk 矩阵乘按各自依赖推进。
            uint32_t stage3ActiveMask = 0;
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                stage3ActiveMask |= 1U << slot;
                RunStateAndBase(chunk, head, owner, slot, logicalCore, groupGeneration);
            }
            // 尾窗口可能有未使用的 slot；显式补回已消费的 FREE，保持下一任务事件配对。
            for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
                if ((stage3ActiveMask & (1U << slot)) == 0U) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
                }
            }

            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_)
                    continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);

                // 7. Tza 残差交接：补齐低位，通知 Cube Stage4 可生成 dAkk。
                RunTzaResidual(static_cast<uint32_t>(head - headBegin), slot);
            }
            // Stage5 复用 [16,144) KiB 前排空 StateAndBase 写回；
            // 高地址的 q/k/E/beta 仍保留给后续计算。
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);

            // 8. 按 32 行推进 Stage5→Stage7/9/10；各带处理完再复用同一操作数工作区。
            for (uint32_t rowBegin = 0; rowBegin < chunk.validRows;
                 rowBegin += KDA_FINALIZE_INTRA_ROWS) {
                generation -= static_cast<uint64_t>(headEnd - headBegin);
                for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                    if ((generation & 1U) != subBlockIdx_)
                        continue;
                    const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                    const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                    RunStage5(chunk, head, owner, slot, logicalCore, groupGeneration, rowBegin);
                    RunIntraBandResults(chunk, head, owner, slot, logicalCore, groupGeneration, rowBegin);
                }
            }
            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_)
                    continue;
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);

                // 9. 所有带的 dg 完成后执行 Stage11，输出 d_g 和参数梯度部分和。
                RunStage11(chunk, chunkTask, head,
                    FinalizeWorkspaceSlotBase(logicalCore, groupGeneration, owner));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
        }

        // 10. 排空末次本地事件；跨核参数归约在 kernel 入口的 SyncAll 之后执行。
        for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
    }

private:
    // Stage11：收齐本 chunk 的 dg，完成 gate 反向与参数部分和。
    __aicore__ inline void RunStage11(
        const FinalizeChunkInfo &chunk, int64_t chunkTask, int64_t head, uint64_t ws)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);

        // 1. 所有有效分带都已更新 workspace 的 dg；此处收齐完整 chunk，执行逆序扫描。
        auto dg = UbBytes(KDA_FINALIZE_UB_DG).ReinterpretCast<float>();
        auto raw = UbBytes(176 * 1024).ReinterpretCast<float>();
        auto bias = UbBytes(160 * 1024).ReinterpretCast<float>();
        auto log = UbBytes(161 * 1024).ReinterpretCast<DTYPE_A_LOG>();
        auto da = UbBytes(162 * 1024).ReinterpretCast<float>();
        auto db = UbBytes(163 * 1024).ReinterpretCast<float>();
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint32_t elems = chunk.validRows * KDA_FINALIZE_DIM;
        AscendC::DataCopy(dg, workspace_[ws + KDA_FINALIZE_WS_DG_BASE].ReinterpretCast<float>(), elems);
        AscendC::DataCopy(raw, rawG_[token], elems);
        AscendC::DataCopy(bias, dtBias_[head * KDA_FINALIZE_DIM], KDA_FINALIZE_DIM);
        AscendC::DataCopyPad(log, aLog_[head],
            AscendC::DataCopyExtParams{1, sizeof(DTYPE_A_LOG), 0, 0, 0},
            AscendC::DataCopyPadExtParams<DTYPE_A_LOG>{false, 0, 0, static_cast<DTYPE_A_LOG>(0)});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);

        // 2. VF 逆序扫描 dg，计算 raw gate 梯度及 a_log/dt_bias 的部分和。
        FinalizeStage11VF<DTYPE_A_LOG>(
            reinterpret_cast<__ubuf__ float *>(dg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(bias.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_A_LOG *>(log.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(da.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(db.GetPhyAddr()), chunk.validRows, tiling_->lowerBound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);

        // 3. d_g 写最终输出；部分和按 [head,chunk] 独立写入，不使用原子累加。
        const int64_t partial = head * tiling_->totalChunkNum + chunkTask;
        AscendC::DataCopy(dG_[token], dg, elems);
        AscendC::DataCopyPad(workspace_[tiling_->gatePartialOffset].ReinterpretCast<float>()[partial * 8], da,
            AscendC::DataCopyExtParams{1, sizeof(float), 0, 0, 0});
        AscendC::DataCopy(workspace_[tiling_->dtBiasPartialOffset].ReinterpretCast<float>()[partial * 128], db, 128);

        // 4. 写回后归还共享 UB，供后续任务复用。
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
    }

    // 按字节偏移取得 UB 视图；实际占用范围由调用阶段的生命周期约束。
    __aicore__ inline AscendC::LocalTensor<uint8_t> UbBytes(uint32_t offset)
    {
        return ub_[offset];
    }

    // 按字节偏移取得 L1 的 BF16 视图；只有取得对应 FREE 后才允许写入。
    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L1Bf16(uint32_t offset)
    {
        AscendC::LocalTensor<uint8_t> l1(AscendC::TPosition::A1, 0, 512 * 1024);
        return l1[offset].ReinterpretCast<bfloat16_t>();
    }

    // Stage3→4 交接：Tza 高位由 Cube 写 L1，AIV 从 FP32 原值补出低位。
    __aicore__ inline void RunTzaResidual(uint32_t owner, uint32_t slot)
    {
        // 1. 等共享 UB 的 StateAndBase 写回完成，再允许 Cube 写入 Tza 原值。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_ZV_FREE_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZW_READY_BASE + slot);
        auto raw = UbBytes(0).ReinterpretCast<float>();
        auto low = UbBytes(16 * 1024).ReinterpretCast<bfloat16_t>();

        // 2. 原值到达后计算 BF16 舍入残差，输出 NZ 低位。
        FinalizeResidualVF(reinterpret_cast<__ubuf__ bfloat16_t *>(low.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()), 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        // 3. 将低位送入本 owner 的 Tza L1 槽；搬出完成后允许覆盖 UB。
        AscendC::DataCopy(L1Bf16(416 * 1024 + (owner * 2 + 1) * KDA_FINALIZE_MATRIX_BF16_BYTES),
            low, KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);

        // 4. 同一通知同时表示 Tza 高低位可读、dAkk 目标 UB 可写。
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_KE_READY_BASE + slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
    }

    // Stage0→1 交接：补齐 dW 低位，再发布 dW/kE 联合 READY。
    __aicore__ inline void RunDwResidual(uint32_t owner, uint32_t slot)
    {
        // 1. 占用共享 UB，等 Cube 的 FP32 dW 到达，期间禁止下一阶段搬入覆盖。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZB_READY_BASE + slot);
        auto raw = UbBytes(64 * 1024).ReinterpretCast<float>();
        auto low = UbBytes(96 * 1024).ReinterpretCast<bfloat16_t>();

        // 2. 用 Cube 传来的 FP32 dW 计算低位，保持与 L1 高位相同的舍入规则。
        FinalizeDwResidualVF(reinterpret_cast<__ubuf__ bfloat16_t *>(low.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        // 3. 低位写入 owner 的 L1 槽，等 MTE3 读完共享 UB。
        AscendC::DataCopy(L1Bf16(64 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES), low, 8192);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);

        // 4. 发布高低位 READY，并归还共享 UB 的本地 FREE。
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_KE_READY_BASE + slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
    }

    // Stage0 / Vector：载入状态与 gate，生成片上 E/kE 和状态归约。
    __aicore__ inline void RunStage0(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration)
    {
        // 1. 等前序 Zb 的 UB→L1 读取结束，再复用相同地址搬入本 head 的输入。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        // 共享 UB 还受阶段级通知保护；上一任务的 workspace 写回结束后才能覆写。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);

        // 2. 绑定共享 UB；分别按 h 的 chunk-major 与 dh 的 head-major 载入。
        auto kUb = UbBytes(KDA_FINALIZE_UB_K).ReinterpretCast<bfloat16_t>();
        auto gkUb = UbBytes(48 * 1024).ReinterpretCast<float>();
        auto hUb = UbBytes(80 * 1024).ReinterpretCast<bfloat16_t>();
        auto dhUb = UbBytes(112 * 1024).ReinterpretCast<bfloat16_t>();
        auto expUb = UbBytes(KDA_FINALIZE_UB_EXP2_GK).ReinterpretCast<float>();
        auto kENd = UbBytes(144 * 1024).ReinterpretCast<bfloat16_t>();
        auto gkLast = UbBytes(208 * 1024).ReinterpretCast<float>();
        auto rH = UbBytes(209 * 1024).ReinterpretCast<float>();
        auto lowNd = UbBytes(224 * 1024).ReinterpretCast<bfloat16_t>();

        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t hState = FinalizeHOffset(*tiling_, chunk, head);
        const int64_t state = FinalizeDhOffset(*tiling_, chunk, head);
        AscendC::DataCopy(kUb, k_[token], chunk.validRows * KDA_FINALIZE_DIM);
        AscendC::DataCopy(gkUb, gk_[token], chunk.validRows * KDA_FINALIZE_DIM);
        AscendC::DataCopy(hUb, h_[hState], KDA_FINALIZE_STATE_ELEMS);
        AscendC::DataCopy(dhUb, dh_[state], KDA_FINALIZE_STATE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);

        // 3. VF 生成 E、kE 高低位、g_last、r_h；仅后两项写入 workspace。
        FinalizeStage0VF(
            reinterpret_cast<__ubuf__ bfloat16_t *>(kENd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(lowNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(expUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gkLast.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(rH.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gkUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(hUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(dhUb.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto wsLast = workspace_[ws + KDA_FINALIZE_WS_GK_LAST].ReinterpretCast<float>();
        auto wsRh = workspace_[ws + KDA_FINALIZE_WS_RH].ReinterpretCast<float>();
        AscendC::DataCopy(wsLast, gkLast, KDA_FINALIZE_DIM);
        AscendC::DataCopy(wsRh, rH, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);

        // 4. 将 kE 高低位直接写入本 owner 的 L1 槽，供 Cube Stage1 消费。
        auto kEL1 = L1Bf16(96 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        AscendC::DataCopy(kEL1, kENd, 8192);
        AscendC::DataCopy(L1Bf16(128 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES), lowNd, 8192);
        // 当前 UB→L1 搬出结束后，Process 才发布 FREE，允许 Cube 覆写重叠区。
    }

    // Stage2 / Vector：消费 FP32 zV/zW，在 UB 构造 Zb 并直接送 L1。
    __aicore__ inline void RunStage2(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner, uint32_t slot)
    {
        // 1. 等共享区可用与 Cube 的 zV/zW READY，再绑定源、目标地址。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZV_READY_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZW_READY_BASE + slot);
        auto zV = UbBytes(KDA_FINALIZE_UB_ZV + slot * KDA_FINALIZE_MATRIX_FP32_BYTES)
                      .ReinterpretCast<float>();
        auto zW = UbBytes(KDA_FINALIZE_UB_ZW + slot * KDA_FINALIZE_MATRIX_FP32_BYTES)
                      .ReinterpretCast<float>();
        auto betaUb = UbBytes(KDA_FINALIZE_UB_BETA).ReinterpretCast<DTYPE_BETA>();
        auto zBNd = UbBytes(KDA_FINALIZE_UB_WORK).ReinterpretCast<bfloat16_t>();
        const int64_t betaOffset = FinalizeTokenOffset(*tiling_, chunk, head, 1);

        // 2. beta 每 token 一个标量，仅载入有效行；短尾按 DMA 块对齐，VF 屏蔽无效 lane。
        // 不额外补满 64 行，以免超过单侧 padding 限制。
        AscendC::DataCopyExtParams betaCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(DTYPE_BETA)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DTYPE_BETA> betaPad{
            true, 0,
            0,
            static_cast<DTYPE_BETA>(0)};
        AscendC::DataCopyPad(betaUb, beta_[betaOffset], betaCopy, betaPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);

        // 3. 计算严格下三角 Zb，并生成高低位 NZ 平面。
        FinalizeStage2VF(
            reinterpret_cast<__ubuf__ bfloat16_t *>(zBNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(zV.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(zW.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(betaUb.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        // 4. 除首次发布外，先消费旧 Zb 的 FREE；再写 L1 并通知 Cube Stage3。
        auto zBL1 = L1Bf16(160 * 1024 + owner * 2 * KDA_FINALIZE_MATRIX_BF16_BYTES);
        if (zBPublishCount_[slot] != 0U) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                KDA_FINALIZE_ZB_FREE_BASE + slot);
        }
        AscendC::DataCopy(zBL1, zBNd, 2 * KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_ZB_READY_BASE + slot);
        ++zBPublishCount_[slot];

        // 5. 等 UB→L1 的读取结束，归还本地共享区供 StateAndBase 使用。
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
    }

    // Stage3/4 / Vector：整 chunk 原位计算状态/base；两阶段间不把临时结果往返 GM。
    __aicore__ inline void RunStateAndBase(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration)
    {
        // 1. 取得共享 UB 所有权；阶段通知同时保护后续 MTE2 与 VF 两类写者。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);

        // 2. 绑定 UB，载入 dk_state_raw、DVb、dq_raw、gk/v 和状态归约。
        auto dkState = UbBytes(0).ReinterpretCast<float>();
        auto dvb = UbBytes(32 * 1024).ReinterpretCast<float>();
        auto dqRaw = UbBytes(64 * 1024).ReinterpretCast<float>();
        auto exp2Gk = UbBytes(KDA_FINALIZE_UB_EXP2_GK).ReinterpretCast<float>();
        auto gk = UbBytes(96 * 1024).ReinterpretCast<float>();
        auto k = UbBytes(KDA_FINALIZE_UB_K).ReinterpretCast<bfloat16_t>();
        auto v = UbBytes(144 * 1024).ReinterpretCast<bfloat16_t>();
        auto dv = UbBytes(128 * 1024).ReinterpretCast<bfloat16_t>();
        auto beta = UbBytes(KDA_FINALIZE_UB_BETA).ReinterpretCast<DTYPE_BETA>();
        auto gkLast = UbBytes(209 * 1024).ReinterpretCast<float>();
        auto rH = UbBytes(210 * 1024).ReinterpretCast<float>();
        auto gateState = UbBytes(211 * 1024).ReinterpretCast<float>();
        auto dbV = UbBytes(212 * 1024).ReinterpretCast<float>();

        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto wsDkState = workspace_[ws + KDA_FINALIZE_WS_DK_STATE_RAW].ReinterpretCast<float>();
        auto wsDvb = workspace_[ws + KDA_FINALIZE_WS_DVB].ReinterpretCast<float>();
        auto wsLast = workspace_[ws + KDA_FINALIZE_WS_GK_LAST].ReinterpretCast<float>();
        auto wsRh = workspace_[ws + KDA_FINALIZE_WS_RH].ReinterpretCast<float>();

        const uint32_t vectorElems =
            static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_DIM;
        AscendC::DataCopy(dkState, wsDkState, vectorElems);
        AscendC::DataCopy(dvb, wsDvb, vectorElems);
        AscendC::DataCopy(dqRaw, dqRaw_[token], vectorElems);
        AscendC::DataCopy(gk, gk_[token], vectorElems);
        AscendC::DataCopy(v, v_[token], vectorElems);
        AscendC::DataCopy(gkLast, wsLast, KDA_FINALIZE_DIM);
        AscendC::DataCopy(rH, wsRh, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);

        // 3. StatePre 生成 dk_state、dv、dq_base、db_v、gate_state。
        FinalizeStage3VF(
            reinterpret_cast<__ubuf__ float *>(dkState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dvb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(dv.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbV.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dqRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(exp2Gk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(v.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gkLast.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(rH.GetPhyAddr()),
            tiling_->scale, static_cast<uint16_t>(chunk.validRows));

        // 4. Stage4 直接消费 UB 中的 FP32 状态结果；gk/v 已读完，改作 dKgb_raw/q。
        // V→MTE2 保护旧输入的最后一次读取，之后才提交覆盖搬运。
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_);
        auto dKgbRaw = gk;
        auto q = v;
        auto wsDkg = workspace_[ws + KDA_FINALIZE_WS_DKGB_RAW].ReinterpretCast<float>();
        AscendC::DataCopy(dKgbRaw, wsDkg, vectorElems);
        AscendC::DataCopy(q, q_[token], vectorElems);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);

        // 5. BaseFinalize 合并 dKgb；三块 FP32 结果写回循环 workspace，dv 写最终输出。
        FinalizeStage4VF(
            reinterpret_cast<__ubuf__ float *>(dkState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dvb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dKgbRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(exp2Gk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbV.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        auto wsDbV = workspace_[ws + KDA_FINALIZE_WS_DB_V].ReinterpretCast<float>();
        AscendC::DataCopy(wsDkState, dkState, vectorElems);
        AscendC::DataCopy(wsDvb, dvb, vectorElems);
        AscendC::DataCopy(wsDkg, dKgbRaw, vectorElems);
        AscendC::DataCopyExtParams dbCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(wsDbV, dbV, dbCopy);
        AscendC::DataCopy(dv_[token], dv, vectorElems);

        // 6. 此阶段使用一套共享工作区；写回完全结束后才允许下一使用者覆写，
        // 不能把 slot 事件误认为两套独立的 StateAndBase 缓冲区。
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
    }

    // Stage5 / Vector：当前带准备五项 L1 操作数，跨带保留 q/k/beta 与 dA 对角项。
    __aicore__ inline void RunStage5(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration, uint32_t rowBegin)
    {
        // 1. 首带等待 dAkk_raw；其余带复用已有 dA 高低位，不重复等待该 READY。
        if (rowBegin == 0) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
                KDA_FINALIZE_DAKK_READY_BASE + slot);
        }
        // 先等当前 slot 的旧 MTE3 读者完成，确保本次搬入不会覆盖尚在搬出的数据。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        // q/k/E/beta 与本阶段输出分离；阶段通知保护下次窗口的共享区复用。
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);

        auto dAkkRaw = UbBytes(KDA_FINALIZE_UB_DAKK_RAW).ReinterpretCast<float>();
        // 双头窗口每 AIV 一头；共享区在前序 MTE3 读完后复用，
        // 下面是同一工作区的不同视图，不是两套独立物理缓冲。
        auto dAqkBf16 = UbBytes(16 * 1024).ReinterpretCast<FinalizeLocalType>();
        auto dAkkNd = UbBytes(32 * 1024).ReinterpretCast<FinalizeLocalType>();
        auto kNegNd = UbBytes(48 * 1024).ReinterpretCast<FinalizeLocalType>();
        auto qPosNd = UbBytes(80 * 1024).ReinterpretCast<FinalizeLocalType>();
        auto bkPosNd = UbBytes(112 * 1024).ReinterpretCast<FinalizeLocalType>();
        auto q = UbBytes(KDA_FINALIZE_UB_Q).ReinterpretCast<bfloat16_t>();
        auto k = UbBytes(KDA_FINALIZE_UB_K).ReinterpretCast<bfloat16_t>();
        auto exp2Gk = UbBytes(KDA_FINALIZE_UB_EXP2_GK).ReinterpretCast<float>();
        auto beta = UbBytes(KDA_FINALIZE_UB_BETA).ReinterpretCast<DTYPE_BETA>();
        auto dAqkFp32 = UbBytes(216 * 1024).ReinterpretCast<float>();

        const int64_t matrixToken =
            FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK);
        const uint32_t matrixElems =
            static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_CHUNK;
        if (rowBegin == 0) {
            AscendC::DataCopy(dAqkFp32, dAqk_[matrixToken], matrixElems);
        }
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint32_t vectorElems = chunk.validRows * KDA_FINALIZE_DIM;
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);

        // 2. 重新载入原始 gk；已下溢的 exp2(gk) 无法通过平移恢复。
        // exp2Gk 变量此时表示原始 gk，VF 才将当前带改写为 exp2(gk-center)。
        AscendC::DataCopy(exp2Gk, gk_[token], vectorElems);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        auto center = UbBytes(48 * 1024).ReinterpretCast<float>();
        const uint32_t rowEnd = FinalizeMin(rowBegin + KDA_FINALIZE_INTRA_ROWS, chunk.validRows);

        // 3. 取当前带首末 gate 的均值作为中心，生成因果范围内的平移操作数。
        FinalizeIntraCenterVF(
            reinterpret_cast<__ubuf__ float *>(exp2Gk.GetPhyAddr()) + rowBegin * 128,
            reinterpret_cast<__ubuf__ float *>(center.GetPhyAddr()),
            rowEnd - rowBegin);
        FinalizeStage5VF(
            reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAkkNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ FinalizeLocalType *>(kNegNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ FinalizeLocalType *>(qPosNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ FinalizeLocalType *>(bkPosNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dAkkRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(exp2Gk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(center.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows), rowBegin, rowEnd);

        // 4. 首带提取 dAqk 对角并生成严格下三角高低位；短尾清零所有操作数 padding。
        if (rowBegin == 0) {
            FinalizeStage5DaqkVF(
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAqkBf16.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(dAqkFp32.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(UbBytes(210 * 1024).GetPhyAddr()),
                static_cast<uint16_t>(chunk.validRows));
        }
        if (chunk.validRows < KDA_FINALIZE_CHUNK) {
            FinalizeStage5TailVF(
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAqkBf16.GetPhyAddr()),
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAkkNd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(kNegNd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(qPosNd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ FinalizeLocalType *>(bkPosNd.GetPhyAddr()), chunk.validRows);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        // 5. 首带发布 dAqk/dAkk，每带更新 kNeg/qPos/bkPos；五项全部直接写 L1。
        auto local = L1Bf16(
            KDA_FINALIZE_LOCAL_BASE + owner * KDA_FINALIZE_LOCAL_BYTES).ReinterpretCast<FinalizeLocalType>();
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

        // 6. 搬出完成后发布 LOCAL_READY，允许 Cube 开始本带 Stage6/8。
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_LOCAL_READY_BASE + slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
    }

    // Stage7/9/10 / Vector：趁 Cube 结果仍在 UB 中完成当前带 dq/dk/db。
    // 更新后的 dg 写回 workspace，供 Stage11 收齐完整 chunk 后扫描。
    __aicore__ inline void RunIntraBandResults(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner, uint32_t slot,
        int64_t coreIdx, uint64_t groupGeneration, uint32_t rowBegin)
    {
        // 1. 等本带操作数搬出结束，绑定结果区并载入对应 base 与可选 rstd。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        const uint32_t rows = FinalizeMin(KDA_FINALIZE_INTRA_ROWS, chunk.validRows - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t offset = rowBegin * KDA_FINALIZE_DIM;
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM) + offset;
        auto result = UbBytes(slot * 32 * 1024).ReinterpretCast<float>();
        auto left = result[KDA_FINALIZE_INTRA_ROWS * KDA_FINALIZE_DIM];
        auto right = UbBytes(64 * 1024 + slot * 32 * 1024).ReinterpretCast<float>();
        // Stage5 搬出结束后，[112,144) 与 [216,248) KiB 可复用。
        // q/k 保留给下一带，输出使用独立临时区，不原位覆盖 q/k。
        auto dg = UbBytes(112 * 1024).ReinterpretCast<float>();
        auto dqBase = UbBytes(128 * 1024).ReinterpretCast<float>();
        auto dkBase = UbBytes(216 * 1024).ReinterpretCast<float>();
        auto dkOut = UbBytes(232 * 1024).ReinterpretCast<bfloat16_t>();
        auto dqOut = UbBytes(240 * 1024).ReinterpretCast<bfloat16_t>();
        auto e = UbBytes(KDA_FINALIZE_UB_EXP2_GK).ReinterpretCast<float>()[offset];
        auto q = UbBytes(KDA_FINALIZE_UB_Q).ReinterpretCast<bfloat16_t>()[offset];
        auto k = UbBytes(KDA_FINALIZE_UB_K).ReinterpretCast<bfloat16_t>()[offset];
        auto beta = UbBytes(KDA_FINALIZE_UB_BETA).ReinterpretCast<DTYPE_BETA>()[rowBegin];
        auto diagonal = UbBytes(210 * 1024).ReinterpretCast<float>()[rowBegin];
        auto dbBase = UbBytes(211 * 1024).ReinterpretCast<float>();
        auto dbDelta = UbBytes(212 * 1024).ReinterpretCast<float>();
        auto qRstd = UbBytes(213 * 1024).ReinterpretCast<float>();
        auto kRstd = UbBytes(214 * 1024).ReinterpretCast<float>();
        auto dbOut = UbBytes(215 * 1024).ReinterpretCast<DTYPE_BETA>();
        AscendC::DataCopy(dg, workspace_[ws + KDA_FINALIZE_WS_DG_BASE].ReinterpretCast<float>()[offset], elems);
        AscendC::DataCopy(dqBase, workspace_[ws + KDA_FINALIZE_WS_DQ_BASE].ReinterpretCast<float>()[offset], elems);
        AscendC::DataCopy(dkBase, workspace_[ws + KDA_FINALIZE_WS_DK_BASE].ReinterpretCast<float>()[offset], elems);
        const AscendC::DataCopyExtParams scalarCopy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<float> scalarPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(dbBase,
            workspace_[ws + KDA_FINALIZE_WS_DB_BASE].ReinterpretCast<float>()[rowBegin], scalarCopy, scalarPad);
        if (tiling_->hasQkL2Norm != 0U) {
            AscendC::DataCopyPad(qRstd, qRstd_[token / KDA_FINALIZE_DIM], scalarCopy, scalarPad);
            AscendC::DataCopyPad(kRstd, kRstd_[token / KDA_FINALIZE_DIM], scalarCopy, scalarPad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);

        // 2. 发布 Stage6 目标 UB 的 FREE，收到 [dq_local; left] 后执行 Stage7。
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_DQ_LOCAL_FREE_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_DQ_LOCAL_READY_BASE + slot);
        FinalizeStage7VF(
            reinterpret_cast<__ubuf__ bfloat16_t *>(dqOut.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(result.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dqBase.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(e.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(qRstd.GetPhyAddr()), tiling_->hasQkL2Norm, rows,
            reinterpret_cast<__ubuf__ float *>(diagonal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()));

        // 3. 允许 Cube 写 right；收到 Stage8 READY 后执行 Stage9 更新 dk/db 增量与 dg。
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_KE_READY_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZW_READY_BASE + slot);
        FinalizeStage9VF(
            reinterpret_cast<__ubuf__ float *>(left.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(right.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbDelta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(e.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta.GetPhyAddr()), rows,
            reinterpret_cast<__ubuf__ float *>(diagonal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()));

        // 4. Stage10 合并 dk/db base，执行可选 K 归一化反向。
        FinalizeStage10VF(
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

        // 5. 写最终 dq/dk/db，将本带 dg 写回 workspace 供完整 chunk 扫描。
        AscendC::DataCopy(dq_[token], dqOut, elems);
        AscendC::DataCopy(dk_[token], dkOut, elems);
        AscendC::DataCopyPad(dBeta_[token / KDA_FINALIZE_DIM], dbOut,
            AscendC::DataCopyExtParams{1, static_cast<uint32_t>(rows * sizeof(DTYPE_BETA)), 0, 0, 0});
        AscendC::DataCopy(workspace_[ws + KDA_FINALIZE_WS_DG_BASE].ReinterpretCast<float>()[offset], dg, elems);

        // 6. 排空所有输出的 MTE3 读取，再允许下一带 Stage5 覆写共享 UB。
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
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H
