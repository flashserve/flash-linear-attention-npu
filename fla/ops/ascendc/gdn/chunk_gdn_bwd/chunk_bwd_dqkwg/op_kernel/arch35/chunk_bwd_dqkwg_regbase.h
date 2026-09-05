/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_regbase.h
 * \brief A5 (dav-3510) RegBase __simd_vf__ 热链骨架 for chunk_bwd_dqkwg vector 端。
 */

#ifndef CHUNK_BWD_DQKWG_ARCH35_REGBASE_H
#define CHUNK_BWD_DQKWG_ARCH35_REGBASE_H

#include "../chunk_bwd_dqkwg_common.h"
#include "kernel_utils/vector/regbase.hpp"

using namespace AscendC;
using namespace AscendC::MicroAPI;

// 一个 fp32 寄存器可容纳的元素数 (dav-3510 = 64)
constexpr uint16_t V_LENGTH_FP32 = VECTOR_REG_WIDTH / sizeof(float);
// 一个 half 寄存器可容纳的元素数 (dav-3510 = 128; bfloat16_t 同为 2 字节, 共用此常量)
constexpr uint16_t V_LENGTH_HALF = VECTOR_REG_WIDTH / sizeof(half);


// ============================================================================
// P0-1: Mul1Half
//   out[i][j] = mask[i][j] * exp(min(0, g[BT_SUB_START+i] - g[j]))
//
// 调用方需先完成 g 的 GType->fp32 cast (保留 MemBase, 1 条指令), 将 fp32 g 指针
// 传入; Muls(-1)/Brcb/strided-Add/Mins/Exp/Mul(mask)/ 全部下沉到本 VF。
//
// mask 布局 (与 vector.h gBuf 64x64 下三角一致):
//   BT==64 : out[i][0..63], mask row = BT_SUB_START + i  (causal col j <= BT_SUB_START+i)
//   BT==128: 分前后两个 64 元素半段
//            BT_SUB_START==0  : cols[0..63]  *= maskA[i]; cols[64..127] = 0
//            BT_SUB_START==64 : cols[0..63]  保留 (mask=1); cols[64..127] *= maskA[i]
//
// BT_SUB_START 作为模板参数 (编译期常量), BT==128 的分流用 if constexpr 实现,
// 不破坏 Hardware Loop / #pragma unroll 约束 (循环体内无 runtime 分支)。
//
// 注意: gFp32 buffer 需 >= BT_SIZE + V_LENGTH_FP32 元素以避免末行 broadcast-load
// 越界 (calcBuf2 分配 >=128 对 BT==64 安全; BT==128 需 caller 确认 padding)。
// ============================================================================
template <uint16_t BT_SIZE, uint16_t BT_SUB_START>
static __simd_vf__ inline void Mul1Half(
    __ubuf__ float *outFp32,       // [realBt, BT_SIZE] row-major 输出 (fp32)
    __ubuf__ float *gFp32,         // [BT_SIZE] gate 已升精度 (caller cast 后)
    __ubuf__ float *maskAddr,      // [64, 64] 下三角 mask, 行 r = maskAddr + r*64
    uint16_t realBt,               // 有效输出行数
    float scale)
{
    RegTensor<float> regGLeft, regSum, regExp, regMask, regOut;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();

    // 一次性载入 g[0..BT-1] 并取负 (-g 在所有行共享)
    if constexpr (BT_SIZE == 64) {
        RegTensor<float> regG, regNegG;
        LoadAlign<float>(regG, gFp32);
        Muls(regNegG, regG, -1.0f, maskAll);

        for (uint16_t i = 0; i < realBt; i++) {
            // gLeft = g[i], 广播到全 lane (DIST_BRC_B32)
            LoadAlign<float, LoadDist::DIST_BRC_B32>(regGLeft, gFp32 + i);
            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regMask, maskAddr, V_LENGTH_FP32);
            Add(regSum, regNegG, regGLeft, maskAll);     // gLeft - g
            Mins(regSum, regSum, 0.0f, maskAll);
            Exp(regExp, regSum, maskAll);
            // mask row = i
            Mul(regOut, regExp, regMask, maskAll);
            StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(outFp32, regOut, V_LENGTH_FP32, maskAll);
        }
    } else {
        // BT_SIZE == 128: g 分两个 64 元素半段
        RegTensor<float> regG0, regG1, regNegG0, regNegG1;
        RegTensor<float> regSum1, regExp1, regOut1;
        LoadAlign<float>(regG0, gFp32);
        LoadAlign<float>(regG1, gFp32 + V_LENGTH_FP32);
        Muls(regNegG0, regG0, -1.0f, maskAll);
        Muls(regNegG1, regG1, -1.0f, maskAll);

        for (uint16_t i = 0; i < realBt; i++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(regGLeft, gFp32 + BT_SUB_START + i);
            Add(regSum, regNegG0, regGLeft, maskAll);
            Add(regSum1, regNegG1, regGLeft, maskAll);
            Mins(regSum, regSum, 0.0f, maskAll);
            Mins(regSum1, regSum1, 0.0f, maskAll);
            Exp(regExp, regSum, maskAll);
            Exp(regExp1, regSum1, maskAll);
            // mask (row = i): BT_SUB_START==0 -> 前半 mask / 后半置 0;
            //                 BT_SUB_START==64 -> 前半 * scale / 后半 mask
            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regMask, maskAddr, V_LENGTH_FP32);
            if constexpr (BT_SUB_START == 0) {
                Mul(regOut, regExp, regMask, maskAll);
                Muls(regOut1, regExp1, 0.0f, maskAll);
            } else {
                Muls(regOut, regExp, scale, maskAll);
                Mul(regOut1, regExp1, regMask, maskAll);
            }
            StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(outFp32, regOut, V_LENGTH_FP32, maskAll);
            StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(outFp32, regOut1, V_LENGTH_FP32, maskAll);
        }
    }
}

// ============================================================================
// P1-4: DqState  (dq_state = dq_inner * exp(g) * scale, factor 按行广播到 K)
//   MemBase: ProcessCVector L979-988 (Exp -> Muls(scale) -> Brcb -> Mul strided x2)
//   与 DkStateMUL2 同构 (factor = scale*exp(g[i]) 而非 exp(gLast-g[i])), 阶段 2 一致。
// ============================================================================
static __simd_vf__ inline void DqState(
    __ubuf__ float *dqStateFp32,   // [realBt, kDim = 128] out (fp32)
    __ubuf__ float *dqFp32,        // [realBt, kDim = 128] in (fp32)
    __ubuf__ float *gFp32,         // [realBt] gate (fp32, 原值只读)
    __ubuf__ float *factorScratch, // [realBt] factor (fp32, 临时存储)
    uint32_t realBt,
    float scale)
{
    RegTensor<float> regG, regFactor, regDq1, regDq2, regOut1, regOut2;

    // 阶段 1: factor[i] = scale * exp(g[i]) -> factorScratch
    uint16_t gLoopTimes = static_cast<uint16_t>((realBt + V_LENGTH_FP32 - 1) / V_LENGTH_FP32);
    uint32_t gLength = realBt;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskG;
    for (uint16_t j = 0; j < gLoopTimes; j++) {
        maskG = UpdateMask<float>(gLength);
        LoadAlign<float>(regG, gFp32 + j * V_LENGTH_FP32);
        Exp(regFactor, regG, maskG);
        Muls(regFactor, regFactor, scale, maskG);
        StoreAlign<float>(factorScratch + j * V_LENGTH_FP32, regFactor, maskG);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // 阶段 2: dq[i][:] *= factor[i]  (与 DkStateMUL2 阶段 2 一致)
    for (uint16_t i = 0; i < realBt; i++) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(regFactor, factorScratch + i);
        LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regDq1, dqFp32, V_LENGTH_FP32);
        LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regDq2, dqFp32, V_LENGTH_FP32);
        Mul(regOut1, regDq1, regFactor, maskAll);
        Mul(regOut2, regDq2, regFactor, maskAll);
        StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(dqStateFp32, regOut1, V_LENGTH_FP32, maskAll);
        StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(dqStateFp32, regOut2, V_LENGTH_FP32, maskAll);
    }
}

// ============================================================================
// DwNegate  (dw = -dw, half in-place)
//   (Cast half->fp32 -> Muls(-1) -> Cast fp32->half, 3 PipeBarrier)
//   极短极热 (每 head 每 chunk 一次)。half 寄存器 128 元素, CastHalf2Float 拆两个 64 float reg。
// ============================================================================
template <typename HalfT>
static __simd_vf__ inline void DwNegate(__ubuf__ HalfT *dwOutBuf, __ubuf__ HalfT *dwInBuf, uint32_t elemCount)
{
    // elemCount must be multiple of V_LENGTH_HALF
    uint16_t colLoopTimes = static_cast<uint16_t>(elemCount / V_LENGTH_HALF);
    RegTensor<HalfT> regH;
    RegTensor<float> regF0, regF1;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskHalfAll = CreateMask<HalfT, MaskPattern::ALL>();
    for (uint16_t j = 0; j < colLoopTimes; j++) {
        LoadAlign<HalfT, PostLiteral::POST_MODE_UPDATE>(regH, dwInBuf, V_LENGTH_HALF);
        CastHalf2Float<HalfT>(regF0, regF1, regH, maskHalfAll);
        Muls(regF0, regF0, -1.0f, maskAll);
        Muls(regF1, regF1, -1.0f, maskAll);
        Cast<HalfT, float, ctFp322HalfOne>(regH, regF1, maskAll);
        Cast<HalfT, float, ctFp322HalfZero>(regH, regF0, maskAll);
        StoreAlign<HalfT, PostLiteral::POST_MODE_UPDATE>(dwOutBuf, regH, V_LENGTH_HALF, maskHalfAll);
    }
}

// ============================================================================
// P3-6: DgLastMulAccum  (sum = h * dh 或 sum += h * dh, 预归约段)
//   MemBase: ProcessAVector dg_last L510-534 (Cast h/dh -> Mul -> Add 累加, 跨 K 行 tile)
//   模板参数 needAdd:
//     - true  : 累加模式, sum = sum + h*dh (读已有 sum 并相加, 用于第 2+ 个 K-row tile)
//     - false : 覆盖模式, sum = h*dh       (首轮 tile 初始化, 等价 MemBase row==0 的 Mul 初始化)
//   caller 跨 K-row tile 多次调用: 首 tile 用 needAdd=false, 后续 tile 用 needAdd=true。
//   归约 (Add 折半 + WholeReduceSum) 仍走 MemBase。
// ============================================================================
template <typename HalfT, bool needAdd>
static __simd_vf__ inline void DgLastMulAccum(
    __ubuf__ float *sumFp32,       // [elemCount] in/out running sum (fp32); needAdd=false 时被覆盖, needAdd=true 时读加写
    __ubuf__ HalfT *hHalf,         // [elemCount] h tile (half / bfloat16_t)
    __ubuf__ HalfT *dhHalf,        // [elemCount] dh tile (half / bfloat16_t)
    uint32_t elemCount)            // elemCount must be multiple of V_LENGTH_FP32
{
    __ubuf__ float *srcAddr = sumFp32;
    uint16_t colLoopTimes = static_cast<uint16_t>(elemCount / V_LENGTH_FP32);
    RegTensor<HalfT> regHH, regDhH;
    RegTensor<float> regHF, regDhF, regProd;
    RegTensor<float> regS;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t j = 0; j < colLoopTimes; j++) {
        LoadAlign<HalfT, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK_B16>(regHH, hHalf, V_LENGTH_FP32);
        LoadAlign<HalfT, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK_B16>(regDhH, dhHalf, V_LENGTH_FP32);
        Cast<float, HalfT, ctHalf2Fp32Zero>(regHF, regHH, maskAll);
        Cast<float, HalfT, ctHalf2Fp32Zero>(regDhF, regDhH, maskAll);
        Mul(regProd, regHF, regDhF, maskAll);
        if constexpr (needAdd) {
            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regS, srcAddr, V_LENGTH_FP32);
            Add(regProd, regS, regProd, maskAll);
        }
        StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(sumFp32, regProd, V_LENGTH_FP32, maskAll);
    }
}

#endif // CHUNK_BWD_DQKWG_ARCH35_REGBASE_H
