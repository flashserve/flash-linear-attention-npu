/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_regbase.h
 * \brief A5 (arch35 / __CCE_AICORE__==310, "Ascend950"/"350") RegBase SIMD vector helpers.
 *
 * 本文件在命名空间内内联一组 __simd_callee__ 寄存器助手
 * (CastHalf2Float / CastFloat2Half / *FloatTwoReg / LoadIn / HalfOrFloat2Float / ...)
 * (而非放到 kernel_utils 公共路径), 以便:
 *   1) 统一 RegBase 写法、CastTrait 和取整模式 (fp32->half 用 CAST_ROUND);
 *   2) 保持实现自包含, 不向公共路径引入算子专用依赖。
 *
 * 本文件只提供"纯按元素"的融合 VF 入口 (NegCastVF / AddCastVF) —— 这是最安全、收益最高的转换点。
 * 跨行/多寄存器归约 (列求和、WholeReduceSum 树、Brcb 广播) 与 B stage 就地复用缓冲的
 * ds*mul1*mm5 暂仍走原 LocalTensor 路径 (见 chunk_bwd_dqkwg_vector.h 说明), 留作后续。
 *
 * 调用约定 (与 arch35/ 既有算子一致): VF 入口由 __aicore__ 端在 TQue DeQue 与 EnQue 之间直接调用,
 * 入参为 LocalTensor::GetPhyAddr() 得到的 __ubuf__ 裸指针; TQue 的 MTE2->V / V->MTE3 同步与 VF
 * (跑在 V 流水) 自然组合, 无需额外手工同步。
 *
 * 整个文件体被 arch35 宏保护: 910B (非 310) 编译时为空, 不影响既有路径。
 */

#ifndef CHUNK_BWD_DQKWG_REGBASE_H
#define CHUNK_BWD_DQKWG_REGBASE_H

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310

#include "kernel_operator.h"

namespace ChunkBwdDqkwgRegBase {

using namespace AscendC::MicroAPI;

// ===== CastTrait =====
// fp16/bf16 -> fp32: 偶 lane (ZERO) / 奇 lane (ONE)。ZEROING => 掩码外 lane 置 0, fp32 运算可用满掩码。
constexpr static CastTrait ctHalf2Fp32Zero = {
    RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE,
};
constexpr static CastTrait ctHalf2Fp32One = {
    RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE,
};
// fp32 -> fp16/bf16: 先写奇 lane (ONE, ZEROING 清寄存器), 再写偶 lane (ZERO, MERGING 保留奇 lane)。
// 取整用 CAST_ROUND (RegBase 路径的 golden 以此为准)。
constexpr static CastTrait ctFp322HalfZero = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, AscendC::RoundMode::CAST_ROUND,
};
constexpr static CastTrait ctFp322HalfOne = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND,
};

constexpr uint16_t PRELOAD_NUM = 2;

// ===== 寄存器助手 =====

// 连续读 (DIST_NORM) 或广播读 (DIST_BRC_B16/B32)。
template <typename TType, bool BroadCast = false>
__simd_callee__ inline void LoadIn(RegTensor<TType>& dstReg, __ubuf__ TType* srcUb)
{
    if constexpr (BroadCast) {
        if constexpr (!std::is_same<TType, float>()) {
            LoadAlign<TType, LoadDist::DIST_BRC_B16>(dstReg, srcUb);
        } else {
            LoadAlign<TType, LoadDist::DIST_BRC_B32>(dstReg, srcUb);
        }
    } else {
        LoadAlign<TType, LoadDist::DIST_NORM>(dstReg, srcUb);
    }
}

// fp16/bf16 -> fp32 偶(ZERO)/奇(ONE) 两寄存器。
template <typename TType>
__simd_callee__ inline void CastHalf2Float(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<TType>& srcReg, MaskReg& mask)
{
    Cast<float, TType, ctHalf2Fp32Zero>(dstZeroReg, srcReg, mask);
    Cast<float, TType, ctHalf2Fp32One>(dstOneReg, srcReg, mask);
}

// fp32 偶/奇 两寄存器 -> fp16/bf16 (先奇后偶, 重新交织)。
template <typename TType>
__simd_callee__ inline void CastFloat2Half(RegTensor<TType>& dstReg, RegTensor<float>& srcZeroReg,
                                           RegTensor<float>& srcOneReg, MaskReg& mask)
{
    Cast<TType, float, ctFp322HalfOne>(dstReg, srcOneReg, mask);
    Cast<TType, float, ctFp322HalfZero>(dstReg, srcZeroReg, mask);
}

// 源是 fp16/bf16 则 cast 到 fp32 (偶 lane); 源本就是 fp32 则按 lane 广播复制。
template <typename TType>
__simd_callee__ inline void HalfOrFloat2Float(RegTensor<float>& dstReg, RegTensor<TType>& srcReg,
                                              MaskReg& maskHalf, MaskReg& maskFloat)
{
    if constexpr (!std::is_same<TType, float>()) {
        Cast<float, TType, ctHalf2Fp32Zero>(dstReg, srcReg, maskHalf);
    } else {
        Duplicate(dstReg, srcReg, maskFloat);
    }
}

__simd_callee__ inline void MulFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<float>& src1ZeroReg, RegTensor<float>& src1OneReg,
                                           RegTensor<float>& src2ZeroReg, RegTensor<float>& src2OneReg,
                                           MaskReg& maskFloat)
{
    Mul(dstZeroReg, src1ZeroReg, src2ZeroReg, maskFloat);
    Mul(dstOneReg, src1OneReg, src2OneReg, maskFloat);
}

__simd_callee__ inline void MinsFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                            RegTensor<float>& srcZeroReg, RegTensor<float>& srcOneReg,
                                            float scalarValue, MaskReg& maskFloat)
{
    Mins(dstZeroReg, srcZeroReg, scalarValue, maskFloat);
    Mins(dstOneReg, srcOneReg, scalarValue, maskFloat);
}

__simd_callee__ inline void ExpFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<float>& srcZeroReg, RegTensor<float>& srcOneReg,
                                           MaskReg& maskFloat)
{
    Exp(dstZeroReg, srcZeroReg, maskFloat);
    Exp(dstOneReg, srcOneReg, maskFloat);
}

__simd_callee__ inline void SubFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<float>& src1ZeroReg, RegTensor<float>& src1OneReg,
                                           RegTensor<float>& src2ZeroReg, RegTensor<float>& src2OneReg,
                                           MaskReg& maskFloat)
{
    Sub(dstZeroReg, src1ZeroReg, src2ZeroReg, maskFloat);
    Sub(dstOneReg, src1OneReg, src2OneReg, maskFloat);
}

__simd_callee__ inline void AddFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<float>& src1ZeroReg, RegTensor<float>& src1OneReg,
                                           RegTensor<float>& src2ZeroReg, RegTensor<float>& src2OneReg,
                                           MaskReg& maskFloat)
{
    Add(dstZeroReg, src1ZeroReg, src2ZeroReg, maskFloat);
    Add(dstOneReg, src1OneReg, src2OneReg, maskFloat);
}

// fp32 标量乘 (取负等)。上游 regbase.hpp 未提供, 这里按 Mins/Muls 同族补一个 (Muls 不可用时可换 Mul + Duplicate(-1))。
__simd_callee__ inline void MulsFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                            RegTensor<float>& srcZeroReg, RegTensor<float>& srcOneReg,
                                            float scalarValue, MaskReg& maskFloat)
{
    Muls(dstZeroReg, srcZeroReg, scalarValue, maskFloat);
    Muls(dstOneReg, srcOneReg, scalarValue, maskFloat);
}

// ===== VF 入口 (本算子用) =====

/**
 * out_fp16 = (T)( -in_fp32 )   (按元素)
 * 用于 A_vector 的 dw = -dw 收尾段 (RefineSmallDw 之后)。in 与 out 必须是不同 UB 缓冲。
 * count = 元素个数 (BT_sub * K)。in 缓冲需按最大 tile (BT*K) 分配, 以容忍尾块整寄存器读取;
 * 输出用 fp16 尾掩码 m16 写, 不会越界写。
 */
template <typename T>
__simd_vf__ inline void NegCastVF(__ubuf__ T* out, __ubuf__ float* in, uint32_t count)
{
    constexpr uint32_t lanes16 = AscendC::VECTOR_REG_WIDTH / sizeof(T);  // 每寄存器 fp16 元素数 = VL/2
    MaskReg m32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> z, o;
    RegTensor<T> outReg;

    uint32_t cnt = count;
    uint16_t loops = static_cast<uint16_t>((count + lanes16 - 1) / lanes16);
    for (uint16_t i = 0; i < loops; ++i) {
        MaskReg m16 = UpdateMask<T>(cnt);  // fp16 域尾掩码 (递减 cnt); 无尾时退化为满掩码
        uint32_t off = static_cast<uint32_t>(i) * lanes16;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(z, o, in + off);  // 连续 fp32 -> 偶/奇寄存器对
        MulsFloatTwoReg(z, o, z, o, static_cast<float>(-1.0f), m32);
        CastFloat2Half<T>(outReg, z, o, m32);
        StoreAlign(out + off, outReg, m16);
    }
}

/**
 * out_fp16 = (T)( accFp32 + (float)addHalf )   (按元素)
 * 用于 C_vector 的 dq += mm6 与 D_vector 的 dk += mm7 收尾段。
 *   accFp32  : dq_state / dk_state (UB 中连续 fp32, 计算结果仍驻留)
 *   addHalf  : mm6 / mm7 (刚 DataCopy 进来的 fp16)
 *   out      : dq / dk 输出 (fp16, 独立 outQue 缓冲)
 * 三个缓冲互不重叠。count = real_BT * K。acc 缓冲按最大 tile (BT*K) 分配。
 */
template <typename T>
__simd_vf__ inline void AddCastVF(__ubuf__ T* out, __ubuf__ float* accFp32,
                                  __ubuf__ T* addHalf, uint32_t count)
{
    constexpr uint32_t lanes16 = AscendC::VECTOR_REG_WIDTH / sizeof(T);
    MaskReg m32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> aZ, aO, bZ, bO;
    RegTensor<T> addReg, outReg;

    uint32_t cnt = count;
    uint16_t loops = static_cast<uint16_t>((count + lanes16 - 1) / lanes16);
    for (uint16_t i = 0; i < loops; ++i) {
        MaskReg m16 = UpdateMask<T>(cnt);
        uint32_t off = static_cast<uint32_t>(i) * lanes16;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(aZ, aO, accFp32 + off);  // acc fp32 偶/奇
        LoadIn<T, false>(addReg, addHalf + off);                            // add fp16 (DIST_NORM)
        CastHalf2Float<T>(bZ, bO, addReg, m16);                            // add -> fp32 偶/奇
        AddFloatTwoReg(aZ, aO, aZ, aO, bZ, bO, m32);
        CastFloat2Half<T>(outReg, aZ, aO, m32);
        StoreAlign(out + off, outReg, m16);
    }
}

/**
 * B_vector 第 1 段 (全 RegBase): ds_temp = ds*mul1; ds2 = ds_temp*mm5;
 *                                 dgcScratch[i] = rowsum_i(ds2); colScratch[j] = colsum_j(ds2)。
 * 使用 ReduceSum 计算行和, 并通过跨行累加计算列和。
 *
 * 形状: ds/mul1/mm5/dsTempOut 均为 [real_BT 行, BT 列] fp16 (行 stride = BT)。
 * 关键: 一行 <= BT <= 128 列 <= 1 个 fp16 寄存器 (A5 VL=256B => 128 fp16 lane), 故按行循环、每行一寄存器。
 *   - ds_temp 用满 BT 掩码 (mBT) 算 + 存 (保留全部 BT 列, 与原 LocalTensor 路径一致, 含 col>=real_BT 的真实乘积);
 *   - ds2 把 mm5 用 real_BT 掩码 (mReal, ZEROING) 转换 => col>=real_BT 置 0, 行/列归约只统计有效 real_BT 列;
 *   - 行和: 偶/奇两寄存器相加后 ReduceSum -> 标量, 逐行写 dgcScratch[i];
 *   - 列和: 偶/奇寄存器跨行累加 (colZ/colO), 循环后 INTLV 交织写回 colScratch (连续列序)。
 * scratch 需 >= 2*(VL/4) 个 fp32 (列和 INTLV 会写满一寄存器对)。dgcScratch / colScratch 与本段输出由
 * 调用方在两段 VF 之间用 PipeBarrier<PIPE_V> 排空 (store -> load 跨段别名)。
 */
template <typename DataType>
__simd_vf__ inline void BdsTempRowColVF(
    __ubuf__ DataType* dsTempOut, __ubuf__ float* dgcScratch, __ubuf__ float* colScratch,
    __ubuf__ DataType* ds, __ubuf__ DataType* mul1, __ubuf__ DataType* mm5,
    uint32_t real_BT, uint32_t BT)
{
    MaskReg m32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<DataType> dsReg, mul1Reg, mm5Reg, dsTempOutReg;
    RegTensor<float> dsZ, dsO, mul1Z, mul1O, mm5Z, mm5O, tZ, tO, d2Z, d2O, rowReg, rsReg;
    RegTensor<float> colZ, colO;
    UnalignRegForStore uStoreDgc;

    Duplicate(colZ, static_cast<float>(0.0f), m32);
    Duplicate(colO, static_cast<float>(0.0f), m32);

    for (uint32_t i = 0; i < real_BT; ++i) {
        uint32_t cntBT = BT;          MaskReg mBT = UpdateMask<DataType>(cntBT);     // 满 BT 列
        uint32_t cntReal = real_BT;   MaskReg mReal = UpdateMask<DataType>(cntReal); // 有效 real_BT 列
        uint32_t off = i * BT;
        LoadIn<DataType, false>(dsReg, ds + off);
        LoadIn<DataType, false>(mul1Reg, mul1 + off);
        LoadIn<DataType, false>(mm5Reg, mm5 + off);

        // ds_temp = ds * mul1 (保留全部 BT 列)
        CastHalf2Float<DataType>(dsZ, dsO, dsReg, mBT);
        CastHalf2Float<DataType>(mul1Z, mul1O, mul1Reg, mBT);
        MulFloatTwoReg(tZ, tO, dsZ, dsO, mul1Z, mul1O, m32);
        CastFloat2Half<DataType>(dsTempOutReg, tZ, tO, m32);
        StoreAlign(dsTempOut + off, dsTempOutReg, mBT);

        // ds2 = ds_temp * mm5 (mm5 用 real_BT 掩码 => col>=real_BT 置 0)
        CastHalf2Float<DataType>(mm5Z, mm5O, mm5Reg, mReal);
        MulFloatTwoReg(d2Z, d2O, tZ, tO, mm5Z, mm5O, m32);

        // 行和 -> dgcScratch[i]
        Add(rowReg, d2Z, d2O, m32);
        ReduceSum(rsReg, rowReg, m32);
        auto dgcOut = dgcScratch + i;
        StoreUnAlign(dgcOut, rsReg, uStoreDgc, 1);
        StoreUnAlignPost(dgcOut, uStoreDgc, 0);

        // 列和累加 (偶/奇)
        AddFloatTwoReg(colZ, colO, colZ, colO, d2Z, d2O, m32);
    }
    // 列和交织写回 (连续列序)
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(colScratch, colZ, colO, m32);
}

/**
 * B_vector 第 2 段 (全 RegBase): dg[k] = dgcScratch[k] - colScratch[k], k in [0,real_BT), 按 GType 写出。
 * 调用方需在第 1 段后插 PipeBarrier<PIPE_V> (dgc/col scratch 是上一段 store、本段 load 的同址别名)。
 */
template <typename GType>
__simd_vf__ inline void BdgFinalizeVF(
    __ubuf__ GType* dgOut, __ubuf__ float* dgcScratch, __ubuf__ float* colScratch, uint32_t real_BT)
{
    MaskReg m32 = CreateMask<float, MaskPattern::ALL>();
    if constexpr (!std::is_same<GType, float>()) {
        // dgc/col 是连续 fp32。反交织读成偶/奇两寄存器后分别相减，
        // 再用两次 Cast 合成连续 fp16/bf16；单独 ZERO cast 会使奇 lane 未定义。
        RegTensor<float> rcZ, rcO, ccZ, ccO, dgZ, dgO;
        RegTensor<GType> dgOutReg;
        UnalignRegForStore uStoreDg;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(rcZ, rcO, dgcScratch);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(ccZ, ccO, colScratch);
        Sub(dgZ, rcZ, ccZ, m32);
        Sub(dgO, rcO, ccO, m32);
        CastFloat2Half<GType>(dgOutReg, dgZ, dgO, m32);
        auto actDgOut = dgOut;
        StoreUnAlign(actDgOut, dgOutReg, uStoreDg, real_BT);
        StoreUnAlignPost(actDgOut, uStoreDg, 0);
    } else {
        constexpr uint32_t flanes = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint16_t floops = static_cast<uint16_t>((real_BT + flanes - 1) / flanes);
        for (uint16_t b = 0; b < floops; ++b) {
            uint32_t foff = static_cast<uint32_t>(b) * flanes;
            uint32_t copyNum = min(flanes, real_BT - foff);
            RegTensor<float> rcReg, ccReg, dgReg;
            UnalignRegForStore uStoreDg;
            LoadAlign<float, LoadDist::DIST_NORM>(rcReg, dgcScratch + foff);
            LoadAlign<float, LoadDist::DIST_NORM>(ccReg, colScratch + foff);
            Sub(dgReg, rcReg, ccReg, m32);
            auto actDgOut = dgOut + foff;
            StoreUnAlign(actDgOut, dgReg, uStoreDg, copyNum);
            StoreUnAlignPost(actDgOut, uStoreDg, 0);
        }
    }
}

}  // namespace ChunkBwdDqkwgRegBase

#endif  // __CCE_AICORE__ == 310

#endif  // CHUNK_BWD_DQKWG_REGBASE_H
