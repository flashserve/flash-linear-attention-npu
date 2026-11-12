/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda.h
 * \brief A6（DAV_9201）状态递推引擎：静态 UB、State bank 轮换、资源 Mutex 调度。
 *
 * 与主线 A5 的 RecurrentKdaEngine 保持同一套算法与同步模型，差异来自 A6 的硬件预算
 * 与指令发射特性：
 *   - 向量 UB 为 384 KiB，每个 tile 的 token 数 4 -> 32：摊薄 tile 级的 DMA 与 Mutex 开销；
 *   - bank 跨度按 State dtype 计算（BF16 32 KiB / FP32 64 KiB），两种 dtype 都用满 4 份；
 *   - A6 向量单元支持一长两短的三指令发射，V-first 热循环按四行展开，把独立的
 *     乘法/归约/搬出短指令交错到归约长指令的发射槽里。
 * V=256 不再单独实现：State 的 V 维按 DIM 切片，每片与 V=128 走同一套 bank、
 * VF 与同步路径。逐 token 的 VF 数学与 A5 完全一致（同样的 FMA 与归约顺序），
 * 因此精度口径不变。
 */
#pragma once

#include <type_traits>
#include "kernel_operator.h"
#include "../recurrent_kda_struct.h"

namespace RecurrentKdaA6 {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using RecurrentKda::RecurrentKdaTilingData;

// 布局常量与 host tiling 共用 recurrent_kda_struct.h 中的定义，单点修改。
constexpr uint32_t DIM = static_cast<uint32_t>(RecurrentKda::RKDA_A6_DIM);
// A6 每个 tile 的 token 上界：取 32 是为了在 384 KiB UB 内摊薄 tile 级开销，
// 同时给 4 份 State bank 与双槽输入留出余量（FP32 布局下仍余约 28 KiB）。
constexpr uint32_t TOKEN_TILE = static_cast<uint32_t>(RecurrentKda::RKDA_A6_TOKEN_TILE);
constexpr uint32_t LAYOUT_BSND = 0;
// 每个 bank 放一片 [DIM, DIM] State：FP32 64 KiB、BF16 32 KiB。A6 的 384 KiB 向量 UB
// 让两种 dtype 都能留 4 份 bank（FP32 布局共 364 544 B），把初态预取深度拉满。
template <typename State>
constexpr uint32_t StateBanks()
{
    return std::is_same<State, float>::value ?
        static_cast<uint32_t>(RecurrentKda::RKDA_A6_FP32_STATE_BANKS) :
        static_cast<uint32_t>(RecurrentKda::RKDA_A6_BF16_STATE_BANKS);
}
constexpr uint32_t MAX_STATE_BANKS = 4;
constexpr uint32_t INPUT_BUFFERS = static_cast<uint32_t>(RecurrentKda::RKDA_A6_INPUT_BUFFERS);
constexpr uint32_t PARAMETER_BUFFERS = static_cast<uint32_t>(RecurrentKda::RKDA_A6_PARAMETER_BUFFERS);
constexpr uint32_t TILE_VECTOR_BYTES = TOKEN_TILE * DIM * sizeof(float);
constexpr uint32_t TILE_SCALAR_BYTES = TOKEN_TILE * 32;
constexpr uint32_t PREDICTION_BYTES = DIM * sizeof(float);
// V-first 的逐行归约结果先以 FP32 落到 UB，再由一次向量转换写回 bf16 输出，
// 避免 128 条单元素 Cast。
constexpr uint32_t OUTPUT_SCRATCH_BYTES = DIM * sizeof(float);
constexpr uint32_t PARAMETER_BYTES = DIM * sizeof(float);

// 逐 token VF 的功能位：由 tiling 属性在 Init 中拼装，避免为每种
// 布尔组合单独生成实例。
enum A6VfFlags : uint32_t {
    RKDA_VF_NORM = 1U << 0,
    RKDA_VF_RAW_GATE = 1U << 1,
    RKDA_VF_BIAS = 1U << 2,
    RKDA_VF_SIGMOID = 1U << 3,
    RKDA_VF_NEGATIVE = 1U << 4,
    RKDA_VF_SAFE = 1U << 5,
};

// 静态 UB 布局（字节）：
//   [0, banks*BANK)                       State bank
//   [.., +2*SLOT)                         q/k/v/gate/beta/out 双槽
//   [.., +2*PARAMETER)                    dt_bias 双槽
//   [.., +PREDICTION)                     VF scratch
//   [.., +OUTPUT_SCRATCH)                 V-first out staging
template <typename State>
constexpr uint32_t UbBytes()
{
    return StateBanks<State>() * DIM * DIM * static_cast<uint32_t>(sizeof(State)) +
           INPUT_BUFFERS * (4 * TOKEN_TILE * DIM * sizeof(bfloat16_t) + TILE_VECTOR_BYTES + TILE_SCALAR_BYTES) +
           PARAMETER_BUFFERS * PARAMETER_BYTES + PREDICTION_BYTES + OUTPUT_SCRATCH_BYTES;
}

// host 与 kernel 的账本必须逐字节一致：RkdaA6UbBytes 是 host 侧使用的同一份公式，
// 常量或公式任一侧改动都会在这里编译失败。
static_assert(UbBytes<float>() ==
                  static_cast<uint32_t>(RecurrentKda::RkdaA6UbBytes(RecurrentKda::RKDA_A6_FP32_STATE_BANKS,
                                                                   RecurrentKda::RKDA_A6_TOKEN_TILE, sizeof(float))),
              "A6 FP32 State UB ledger diverges from the host tiling formula");
static_assert(UbBytes<bfloat16_t>() == static_cast<uint32_t>(RecurrentKda::RkdaA6UbBytes(
                                          RecurrentKda::RKDA_A6_BF16_STATE_BANKS,
                                          RecurrentKda::RKDA_A6_TOKEN_TILE, sizeof(bfloat16_t))),
              "A6 BF16 State UB ledger diverges from the host tiling formula");

// 编译期只做保守校验；A6 实际 UB 由 host 读取平台信息后复核。
// FP32 State（4 bank）占 364 544 B，BF16 State（4 bank）占 233 472 B。
constexpr uint32_t UB_LIMIT_BYTES = 384 * 1024;
static_assert(UbBytes<float>() <= UB_LIMIT_BYTES, "A6 FP32 State UB exceeds the 384 KiB vector UB");
static_assert(UbBytes<bfloat16_t>() <= UB_LIMIT_BYTES, "A6 BF16 State UB exceeds the 384 KiB vector UB");

// Mutex 编号由本引擎独占：input/out 各两份、State 至多四份、参数两份。
constexpr uint8_t INPUT_MUTEX[INPUT_BUFFERS] = {3, 4};
constexpr uint8_t OUTPUT_MUTEX[INPUT_BUFFERS] = {5, 6};
constexpr uint8_t STATE_MUTEX[MAX_STATE_BANKS] = {7, 8, 9, 10};
constexpr uint8_t PARAMETER_MUTEX[PARAMETER_BUFFERS] = {11, 12};
// 未记录有效快照槽位的标记，与 A5 的 pendingStateSlots 语义一致。
constexpr uint64_t INVALID_STATE_SLOT = uint64_t(-1);

constexpr CastTrait TO_FLOAT_ZERO = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr CastTrait TO_FLOAT_ONE = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr CastTrait TO_B16_ZERO = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, RoundMode::CAST_RINT};
constexpr CastTrait TO_B16_ONE = {RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait TO_B16_WORD = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// 连续 64 lane 的 bf16/fp32 载入与落盘：K-first 的 V 方向与 V-first 的 out 转换共用。
template <typename T>
__simd_callee__ inline void LoadContiguous(RegTensor<float> &values, __ubuf__ T *source, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadAlign(values, source);
    } else {
        RegTensor<T> packed;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(packed, source);
        Cast<float, T, TO_FLOAT_ZERO>(values, packed, mask);
    }
}

template <typename T>
__simd_callee__ inline void StoreContiguous(__ubuf__ T *target, RegTensor<float> &values, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        StoreAlign(target, values, mask);
    } else {
        RegTensor<T> packed;
        Cast<T, float, TO_B16_WORD>(packed, values, mask);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(target, packed, mask);
    }
}

// V-first 的两行粒度：把偶/奇 K lane 分别载入两个寄存器。
template <typename T>
__simd_callee__ inline void LoadPair(RegTensor<float> &even, RegTensor<float> &odd,
                                     __ubuf__ T *source, MaskReg &mask16)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(even, odd, source);
    } else {
        RegTensor<T> packed;
        LoadAlign(packed, source);
        Cast<float, T, TO_FLOAT_ZERO>(even, packed, mask16);
        Cast<float, T, TO_FLOAT_ONE>(odd, packed, mask16);
    }
}

__simd_callee__ inline void NormalizePair(RegTensor<float> &even, RegTensor<float> &odd, MaskReg &mask)
{
    RegTensor<float> work, norm, one;
    MaskReg first = CreateMask<float, MaskPattern::VL1>();
    Mul(work, even, even, mask);
    MulAddDst(work, odd, odd, mask);
    ReduceSum(norm, work, mask);
    Adds(norm, norm, 1.0e-6f, first);
    Sqrt(norm, norm, first);
    Duplicate(one, 1.0f, mask);
    Div(norm, one, norm, first);
    Duplicate(norm, norm, mask);
    Mul(even, even, norm, mask);
    Mul(odd, odd, norm, mask);
}

// alpha = exp(g)；raw gate 下 g = -exp(A_log)*softplus(g+bias)，
// safe gate 下 g = lower_bound*sigmoid(exp(A_log)*(g+bias))。
// softplus 用 max(x,0)+log(1+exp(-|x|)) 避免中间指数溢出。
__simd_callee__ inline void Decay(RegTensor<float> &gate, RegTensor<float> &expA,
                                  bool rawGate, bool safeGate, float lowerBound, MaskReg &mask)
{
    if (rawGate) {
        RegTensor<float> work, positive, one;
        if (safeGate) {
            Mul(gate, gate, expA, mask);
            Muls(gate, gate, -1.0f, mask);
            Exp(gate, gate, mask);
            Adds(gate, gate, 1.0f, mask);
            Duplicate(one, 1.0f, mask);
            Div(gate, one, gate, mask);
            Muls(gate, gate, lowerBound, mask);
        } else {
            Maxs(positive, gate, 0.0f, mask);
            Abs(work, gate, mask);
            Muls(work, work, -1.0f, mask);
            Exp(work, work, mask);
            Adds(work, work, 1.0f, mask);
            Ln(work, work, mask);
            Add(gate, positive, work, mask);
            Mul(gate, gate, expA, mask);
            Muls(gate, gate, -1.0f, mask);
        }
    }
    Exp(gate, gate, mask);
}

template <typename State>
__simd_callee__ inline void StoreVFirstRows(__ubuf__ State *target, uint16_t row,
                                            RegTensor<float> &s0, RegTensor<float> &s1,
                                            RegTensor<float> &s2, RegTensor<float> &s3,
                                            MaskReg &mask, MaskReg &mask16)
{
    if constexpr (std::is_same<State, float>::value) {
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(target + row * DIM, s0, s1, mask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(target + (row + 1) * DIM, s2, s3, mask);
    } else {
        RegTensor<State> packed;
        Cast<State, float, TO_B16_ONE>(packed, s1, mask);
        Cast<State, float, TO_B16_ZERO>(packed, s0, mask);
        StoreAlign(target + row * DIM, packed, mask16);
        Cast<State, float, TO_B16_ONE>(packed, s3, mask);
        Cast<State, float, TO_B16_ZERO>(packed, s2, mask);
        StoreAlign(target + (row + 1) * DIM, packed, mask16);
    }
}

// 第二遍：逐四行重读源 bank，广播 residual，更新 State 并把 out 收集到 UB。
// 四行一组是为了配合 A6 的一长两短三指令发射：两组两行之间没有依赖，
// 归约长指令发射时另一组的乘法/搬出短指令可以填满其余发射槽。
// 归约结果以 FP32 连续落到 staging，避免每行一条单元素 Cast。
template <typename State>
__simd_callee__ inline void RecurrentKdaVFirstUpdateRows(
    __ubuf__ State *target, __ubuf__ State *source, __ubuf__ float *outputScalar,
    __ubuf__ float *residual,
    RegTensor<float> &queryEven, RegTensor<float> &queryOdd,
    RegTensor<float> &decayEven, RegTensor<float> &decayOdd,
    RegTensor<float> &updateKeyEven, RegTensor<float> &updateKeyOdd,
    MaskReg &mask, MaskReg &mask16, UnalignRegForStore &outputStore)
{
    #pragma unroll 1
    for (uint16_t row = 0; row < DIM; row += 4) {
        RegTensor<float> s0, s1, s2, s3, s4, s5, s6, s7;
        RegTensor<float> residual0, residual1, residual2, residual3;
        RegTensor<float> dot0, dot1, dot2, dot3, out0, out1, out2, out3;
        // 四行 residual 与 State 先全部载入，后面的计算只依赖寄存器。
        LoadAlign<float, LoadDist::DIST_BRC_B32>(residual0, residual + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(residual1, residual + row + 1);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(residual2, residual + row + 2);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(residual3, residual + row + 3);
        LoadPair(s0, s1, source + row * DIM, mask16);
        LoadPair(s2, s3, source + (row + 1) * DIM, mask16);
        LoadPair(s4, s5, source + (row + 2) * DIM, mask16);
        LoadPair(s6, s7, source + (row + 3) * DIM, mask16);
        // 衰减与更新：八条互相独立的短指令。
        Mul(s0, s0, decayEven, mask);
        Mul(s1, s1, decayOdd, mask);
        Mul(s2, s2, decayEven, mask);
        Mul(s3, s3, decayOdd, mask);
        Mul(s4, s4, decayEven, mask);
        Mul(s5, s5, decayOdd, mask);
        Mul(s6, s6, decayEven, mask);
        Mul(s7, s7, decayOdd, mask);
        MulAddDst(s0, residual0, updateKeyEven, mask);
        MulAddDst(s1, residual0, updateKeyOdd, mask);
        MulAddDst(s2, residual1, updateKeyEven, mask);
        MulAddDst(s3, residual1, updateKeyOdd, mask);
        MulAddDst(s4, residual2, updateKeyEven, mask);
        MulAddDst(s5, residual2, updateKeyOdd, mask);
        MulAddDst(s6, residual3, updateKeyEven, mask);
        MulAddDst(s7, residual3, updateKeyOdd, mask);
        // out = <state, q>：四行各两条短指令。
        Mul(dot0, s0, queryEven, mask);
        Mul(dot1, s2, queryEven, mask);
        Mul(dot2, s4, queryEven, mask);
        Mul(dot3, s6, queryEven, mask);
        MulAddDst(dot0, s1, queryOdd, mask);
        MulAddDst(dot1, s3, queryOdd, mask);
        MulAddDst(dot2, s5, queryOdd, mask);
        MulAddDst(dot3, s7, queryOdd, mask);
        // 四条归约长指令连续发射，前后的短指令可与之并行。
        ReduceSum(out0, dot0, mask);
        ReduceSum(out1, dot1, mask);
        ReduceSum(out2, dot2, mask);
        ReduceSum(out3, dot3, mask);
        StoreVFirstRows(target, row, s0, s1, s2, s3, mask, mask16);
        StoreVFirstRows(target, row + 2, s4, s5, s6, s7, mask, mask16);
        StoreUnAlign(outputScalar, out0, outputStore, 1);
        StoreUnAlign(outputScalar, out1, outputStore, 1);
        StoreUnAlign(outputScalar, out2, outputStore, 1);
        StoreUnAlign(outputScalar, out3, outputStore, 1);
    }
}

template <typename State>
__simd_callee__ inline void RecurrentKdaVFirstTwoPass(
    __ubuf__ State *target, __ubuf__ State *source, __ubuf__ float *scratch,
    __ubuf__ float *outputScalar, __ubuf__ bfloat16_t *valueRaw, __ubuf__ bfloat16_t *output,
    RegTensor<float> &queryEven, RegTensor<float> &queryOdd,
    RegTensor<float> &keyEven, RegTensor<float> &keyOdd,
    RegTensor<float> &decayEven, RegTensor<float> &decayOdd,
    RegTensor<float> &updateKeyEven, RegTensor<float> &updateKeyOdd,
    MaskReg &mask, MaskReg &mask16)
{
    RegTensor<float> decayKeyEven, decayKeyOdd;
    Mul(decayKeyEven, decayEven, keyEven, mask);
    Mul(decayKeyOdd, decayOdd, keyOdd, mask);
    UnalignRegForStore predictionStore;
    __ubuf__ float *predictionAddress = scratch;
    // 第一遍只归约 prediction，State 寄存器随即释放；同样按四行展开以填满发射槽。
    #pragma unroll 1
    for (uint16_t row = 0; row < DIM; row += 4) {
        RegTensor<float> s0, s1, s2, s3, s4, s5, s6, s7;
        RegTensor<float> dot0, dot1, dot2, dot3, prediction0, prediction1, prediction2, prediction3;
        LoadPair(s0, s1, source + row * DIM, mask16);
        LoadPair(s2, s3, source + (row + 1) * DIM, mask16);
        LoadPair(s4, s5, source + (row + 2) * DIM, mask16);
        LoadPair(s6, s7, source + (row + 3) * DIM, mask16);
        Mul(dot0, s0, decayKeyEven, mask);
        Mul(dot1, s2, decayKeyEven, mask);
        Mul(dot2, s4, decayKeyEven, mask);
        Mul(dot3, s6, decayKeyEven, mask);
        MulAddDst(dot0, s1, decayKeyOdd, mask);
        MulAddDst(dot1, s3, decayKeyOdd, mask);
        MulAddDst(dot2, s5, decayKeyOdd, mask);
        MulAddDst(dot3, s7, decayKeyOdd, mask);
        ReduceSum(prediction0, dot0, mask);
        ReduceSum(prediction1, dot1, mask);
        ReduceSum(prediction2, dot2, mask);
        ReduceSum(prediction3, dot3, mask);
        StoreUnAlign(predictionAddress, prediction0, predictionStore, 1);
        StoreUnAlign(predictionAddress, prediction1, predictionStore, 1);
        StoreUnAlign(predictionAddress, prediction2, predictionStore, 1);
        StoreUnAlign(predictionAddress, prediction3, predictionStore, 1);
    }
    StoreUnAlignPost(predictionAddress, predictionStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    RegTensor<float> predictionEven, predictionOdd, residualEven, residualOdd;
    LoadPair(predictionEven, predictionOdd, scratch, mask16);
    LoadPair(residualEven, residualOdd, valueRaw, mask16);
    Sub(residualEven, residualEven, predictionEven, mask);
    Sub(residualOdd, residualOdd, predictionOdd, mask);
    // residual 覆盖同一块 512 B scratch，供第二遍逐行广播。
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(scratch, residualEven, residualOdd, mask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    UnalignRegForStore outputStore;
    RecurrentKdaVFirstUpdateRows<State>(target, source, outputScalar, scratch,
        queryEven, queryOdd, decayEven, decayOdd, updateKeyEven, updateKeyOdd,
        mask, mask16, outputStore);
    StoreUnAlignPost(outputScalar, outputStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // 一次把 128 个 FP32 归约结果转换成 bf16 输出：与逐元素 CAST_RINT 等价，
    // 但把每 token 128 条单元素 Cast 降成两条向量 Cast。
    RegTensor<float> outputLow, outputHigh;
    LoadAlign<float>(outputLow, outputScalar);
    LoadAlign<float>(outputHigh, outputScalar + DIM / 2);
    StoreContiguous<bfloat16_t>(output, outputLow, mask);
    StoreContiguous<bfloat16_t>(output + DIM / 2, outputHigh, mask);
}

// K-first：lane 表示连续的 V，系数寄存器保存连续的 K，State 保持紧凑 [K,V]。
template <typename State>
__simd_callee__ inline void PredictKFirstHalf(RegTensor<float> &predictionLow,
                                              RegTensor<float> &predictionHigh,
                                              __ubuf__ State *source, __ubuf__ float *decayKey,
                                              MaskReg &mask)
{
    #pragma unroll 1
    for (uint16_t keyRow = 0; keyRow < DIM / 2; ++keyRow) {
        RegTensor<float> coefficient, stateLow, stateHigh;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(coefficient, decayKey + keyRow);
        LoadContiguous(stateLow, source + keyRow * DIM, mask);
        LoadContiguous(stateHigh, source + keyRow * DIM + DIM / 2, mask);
        MulAddDst(predictionLow, stateLow, coefficient, mask);
        MulAddDst(predictionHigh, stateHigh, coefficient, mask);
    }
}

template <typename State>
__simd_callee__ inline void UpdateKFirstHalf(__ubuf__ State *target, __ubuf__ State *source,
                                             RegTensor<float> &deltaLow, RegTensor<float> &deltaHigh,
                                             RegTensor<float> &outputLow, RegTensor<float> &outputHigh,
                                             RegTensor<float> &query, RegTensor<float> &key,
                                             RegTensor<float> &decay, MaskReg &mask)
{
    RegTensor<uint32_t> coefficientLane;
    Duplicate(coefficientLane, uint32_t(0));
    #pragma unroll 1
    for (uint16_t keyRow = 0; keyRow < DIM / 2; ++keyRow) {
        RegTensor<float> queryValue, keyValue, decayValue, stateLow, stateHigh;
        Gather(decayValue, decay, coefficientLane);
        Gather(keyValue, key, coefficientLane);
        Gather(queryValue, query, coefficientLane);
        LoadContiguous(stateLow, source + keyRow * DIM, mask);
        LoadContiguous(stateHigh, source + keyRow * DIM + DIM / 2, mask);
        Mul(stateLow, stateLow, decayValue, mask);
        Mul(stateHigh, stateHigh, decayValue, mask);
        MulAddDst(stateLow, deltaLow, keyValue, mask);
        MulAddDst(stateHigh, deltaHigh, keyValue, mask);
        // out 使用 BF16 舍入前的 FP32 更新值。
        MulAddDst(outputLow, stateLow, queryValue, mask);
        MulAddDst(outputHigh, stateHigh, queryValue, mask);
        StoreContiguous(target + keyRow * DIM, stateLow, mask);
        StoreContiguous(target + keyRow * DIM + DIM / 2, stateHigh, mask);
        Adds(coefficientLane, coefficientLane, uint32_t(1), mask);
    }
}

template <typename State>
__simd_callee__ inline void RecurrentKdaKFirstTwoPass(__ubuf__ State *target, __ubuf__ State *source,
                                                      __ubuf__ float *coefficientScratch,
                                                      __ubuf__ bfloat16_t *value,
                                                      __ubuf__ bfloat16_t *output,
                                                      RegTensor<float> &queryLow, RegTensor<float> &queryHigh,
                                                      RegTensor<float> &keyLow, RegTensor<float> &keyHigh,
                                                      RegTensor<float> &decayLow, RegTensor<float> &decayHigh,
                                                      RegTensor<float> &beta, MaskReg &mask)
{
    RegTensor<float> predictionLow, predictionHigh, decayKeyLow, decayKeyHigh;
    Duplicate(predictionLow, 0.0f);
    Duplicate(predictionHigh, 0.0f);
    Mul(decayKeyLow, decayLow, keyLow, mask);
    Mul(decayKeyHigh, decayHigh, keyHigh, mask);
    StoreAlign(coefficientScratch, decayKeyLow, mask);
    StoreAlign(coefficientScratch + DIM / 2, decayKeyHigh, mask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    PredictKFirstHalf(predictionLow, predictionHigh, source, coefficientScratch, mask);
    PredictKFirstHalf(predictionLow, predictionHigh, source + (DIM / 2) * DIM,
        coefficientScratch + DIM / 2, mask);

    RegTensor<float> valueLow, valueHigh;
    LoadContiguous(valueLow, value, mask);
    LoadContiguous(valueHigh, value + DIM / 2, mask);
    Sub(predictionLow, valueLow, predictionLow, mask);
    Sub(predictionHigh, valueHigh, predictionHigh, mask);
    Mul(predictionLow, predictionLow, beta, mask);
    Mul(predictionHigh, predictionHigh, beta, mask);

    RegTensor<float> outputLow, outputHigh;
    Duplicate(outputLow, 0.0f);
    Duplicate(outputHigh, 0.0f);
    UpdateKFirstHalf(target, source, predictionLow, predictionHigh, outputLow, outputHigh,
        queryLow, keyLow, decayLow, mask);
    UpdateKFirstHalf(target + (DIM / 2) * DIM, source + (DIM / 2) * DIM,
        predictionLow, predictionHigh, outputLow, outputHigh, queryHigh, keyHigh, decayHigh, mask);
    StoreContiguous(output, outputLow, mask);
    StoreContiguous(output + DIM / 2, outputHigh, mask);
}

// 单 token VF：q/k 归一化、gate 变换、prediction、State 更新与 out 收集。
template <typename State, typename Gate, typename Beta, bool V_FIRST, bool FUSED>
__simd_vf__ inline void RecurrentKdaTokenVf(__ubuf__ State *target, __ubuf__ State *source,
                                            __ubuf__ bfloat16_t *keyRaw, __ubuf__ bfloat16_t *queryRaw,
                                            __ubuf__ bfloat16_t *valueRaw, __ubuf__ Gate *gateRaw,
                                            __ubuf__ Beta *betaRaw, __ubuf__ float *biasRaw,
                                            __ubuf__ bfloat16_t *output, float scale, float aLog,
                                            uint32_t flags, float lowerBound, __ubuf__ float *predictionScratch,
                                            __ubuf__ float *outputScratch)
{
    const bool normalize = FUSED || (flags & RKDA_VF_NORM);
    const bool rawGate = FUSED || (flags & RKDA_VF_RAW_GATE);
    const bool sigmoid = FUSED || (flags & RKDA_VF_SIGMOID);
    const bool bias = FUSED || (flags & RKDA_VF_BIAS);
    const bool negative = !FUSED && (flags & RKDA_VF_NEGATIVE);
    const bool safe = !FUSED && (flags & RKDA_VF_SAFE);
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg mask16 = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> q0, q1, k0, k1, decay0, decay1, beta, expA, work0, work1, one;
    RegTensor<float> updateKey0, updateKey1;
    if constexpr (V_FIRST) {
        LoadPair(k0, k1, keyRaw, mask16);
        LoadPair(q0, q1, queryRaw, mask16);
    } else {
        LoadContiguous(k0, keyRaw, mask);
        LoadContiguous(k1, keyRaw + DIM / 2, mask);
        LoadContiguous(q0, queryRaw, mask);
        LoadContiguous(q1, queryRaw + DIM / 2, mask);
    }
    if (normalize) {
        NormalizePair(q0, q1, mask);
        NormalizePair(k0, k1, mask);
    }
    Muls(q0, q0, scale, mask);
    Muls(q1, q1, scale, mask);
    if constexpr (V_FIRST) {
        LoadPair(decay0, decay1, gateRaw, mask16);
    } else {
        LoadContiguous(decay0, gateRaw, mask);
        LoadContiguous(decay1, gateRaw + DIM / 2, mask);
    }
    if constexpr (std::is_same<Beta, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(beta, betaRaw);
    } else {
        RegTensor<Beta> packed;
        LoadAlign<Beta, LoadDist::DIST_BRC_B16>(packed, betaRaw);
        Cast<float, Beta, TO_FLOAT_ZERO>(beta, packed, mask16);
    }
    if (sigmoid) {
        Duplicate(one, 1.0f, mask);
        Muls(beta, beta, -1.0f, mask);
        Exp(beta, beta, mask);
        Adds(beta, beta, 1.0f, mask);
        Div(beta, one, beta, mask);
        if (negative) {
            Muls(beta, beta, 2.0f, mask);
        }
    }
    if (rawGate) {
        Duplicate(expA, aLog, mask);
        Exp(expA, expA, mask);
        if (bias) {
            if constexpr (V_FIRST) {
                LoadPair(work0, work1, biasRaw, mask16);
            } else {
                LoadContiguous(work0, biasRaw, mask);
                LoadContiguous(work1, biasRaw + DIM / 2, mask);
            }
            Add(decay0, decay0, work0, mask);
            Add(decay1, decay1, work1, mask);
        }
    }
    Decay(decay0, expA, rawGate, safe, lowerBound, mask);
    Decay(decay1, expA, rawGate, safe, lowerBound, mask);
    if constexpr (V_FIRST) {
        Mul(updateKey0, k0, beta, mask);
        Mul(updateKey1, k1, beta, mask);
        RecurrentKdaVFirstTwoPass<State>(target, source, predictionScratch, outputScratch, valueRaw, output,
            q0, q1, k0, k1, decay0, decay1, updateKey0, updateKey1, mask, mask16);
    } else {
        RecurrentKdaKFirstTwoPass(target, source, predictionScratch, valueRaw, output,
            q0, q1, k0, k1, decay0, decay1, beta, mask);
    }
}

// A6 V=128 的唯一任务调度器：独占 UB 与 Mutex 编号，不经过 TPipe。
template <typename State, typename Gate, typename Beta, bool V_FIRST, bool FUSED>
class RecurrentKdaA6Engine {
public:
    // bank 数由 State dtype 决定，编译期常量，不影响循环展开。
    static constexpr uint32_t BANKS = StateBanks<State>();

    __aicore__ inline explicit RecurrentKdaA6Engine(const RecurrentKdaTilingData *tiling) : tiling_(tiling) {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gate, GM_ADDR beta,
        GM_ADDR state, GM_ADDR cuSeqlens, GM_ADDR indices, GM_ADDR aLog, GM_ADDR bias, GM_ADDR accepted,
        GM_ADDR output, GM_ADDR stateOutput)
    {
        queryGm_.SetGlobalBuffer((__gm__ bfloat16_t *)query);
        keyGm_.SetGlobalBuffer((__gm__ bfloat16_t *)key);
        valueGm_.SetGlobalBuffer((__gm__ bfloat16_t *)value);
        gateGm_.SetGlobalBuffer((__gm__ Gate *)gate);
        betaGm_.SetGlobalBuffer((__gm__ Beta *)beta);
        stateGm_.SetGlobalBuffer((__gm__ State *)state);
        stateOutGm_.SetGlobalBuffer((__gm__ State *)stateOutput);
        outputGm_.SetGlobalBuffer((__gm__ bfloat16_t *)output);
        aLogGm_.SetGlobalBuffer((__gm__ float *)aLog);
        biasGm_.SetGlobalBuffer((__gm__ float *)bias);
        cuAddress_ = cuSeqlens;
        indicesAddress_ = indices;
        acceptedAddress_ = accepted;
        vfFlags_ = (tiling_->useQkL2norm ? RKDA_VF_NORM : 0U) |
            (tiling_->useGateInKernel ? RKDA_VF_RAW_GATE : 0U) |
            (tiling_->hasDtBias ? RKDA_VF_BIAS : 0U) |
            (tiling_->useBetaSigmoid ? RKDA_VF_SIGMOID : 0U) |
            (tiling_->allowNegEigval ? RKDA_VF_NEGATIVE : 0U) |
            (tiling_->safeGate ? RKDA_VF_SAFE : 0U);
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        if (!ValidateSequences()) {
            return;
        }
        const uint64_t taskCount = uint64_t(tiling_->b) * tiling_->nv;
        // V=256 的 State 按 DIM 切片：每片独占一段 V 区间，bank 布局、逐 token VF
        // 与同步模型都与 V=128 相同，因此不需要单独的兼容实现。
        const uint32_t valueSlices = tiling_->dv / DIM;
        uint32_t currentBank = 0;
        uint32_t tileIndex = 0;
        uint32_t parameterBuffer = 0;
        const bool usesBias = tiling_->hasDtBias && tiling_->useGateInKernel;
        bool stateReadyEvent = true;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for (uint64_t task = GetBlockIdx(); task < taskCount; task += GetBlockNum()) {
            const uint64_t request = task / tiling_->nv;
            const uint64_t head = task % tiling_->nv;
            UpdateSequence(request);
            // 空请求不产生输出，跳过即可；元数据越界（slot 超出 state 池）说明入参
            // 非法，本核后续任务同属一份元数据，直接结束整个循环。
            if (sequenceBegin_ == sequenceEnd_) {
                continue;
            }
            if (!ValidateSlots(request)) {
                break;
            }
            const uint64_t initialSlot = InitialSlot(request);
            const uint64_t keyHead = head / (tiling_->nv / tiling_->nk);
            const int64_t taskTokens = sequenceEnd_ - sequenceBegin_;
            const bool hasFollowingTask = task + GetBlockNum() < taskCount;
            const float aLog = tiling_->useGateInKernel ? aLogGm_.GetValue(head) : 0.0f;
            // 参数槽只对真正执行的任务轮换，空请求不推进；dt_bias 是 head 级参数，
            // 所有 V 片共用，因此搬入与锁都放在切片循环之外。
            if (usesBias) {
                Mutex::Lock<PIPE_MTE2>(PARAMETER_MUTEX[parameterBuffer]);
                DataCopy(biasUb_[parameterBuffer], biasGm_[head * DIM], DIM);
                Mutex::Unlock<PIPE_MTE2>(PARAMETER_MUTEX[parameterBuffer]);
                Mutex::Lock<PIPE_V>(PARAMETER_MUTEX[parameterBuffer]);
            }
            for (uint32_t slice = 0; slice < valueSlices; ++slice) {
                const uint32_t valueOffset = slice * DIM;
                const bool hasFollowingWork = slice + 1 < valueSlices || hasFollowingTask;
                // 不同 V 片写同一槽位的不同行/列区间，因此同槽写序记录按 V 片独立维护。
                uint64_t pendingStateSlots[MAX_STATE_BANKS];
                for (uint32_t bank = 0; bank < BANKS; ++bank) {
                    pendingStateSlots[bank] = INVALID_STATE_SLOT;
                }
                currentBank = (currentBank + 1) % BANKS;
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                stateReadyEvent = false;
                Mutex::Lock<PIPE_MTE2>(STATE_MUTEX[currentBank]);
                CopyStateIn(currentBank, initialSlot, head, valueOffset);
                // 初态搬入（MTE2）与首个 token 的 VF 读之间没有共享 Mutex：
                // V 侧只锁下一个 bank，因此必须显式建立 MTE2 -> V 的 RAW 依赖。
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                Mutex::Unlock<PIPE_MTE2>(STATE_MUTEX[currentBank]);
                // 工作量不足 BANKS-1 个 token 时，下一段（下一个 V 片或下一个任务）
                // 的 bank 从未被本段读取，可以在进入 token 循环前就发布握手；
                // 否则在最后一次读取之后发布。
                const bool earlyPrefetch = hasFollowingWork && taskTokens < static_cast<int64_t>(BANKS - 1);
                if (earlyPrefetch) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    stateReadyEvent = true;
                }
                // 等待本段初态搬入完成后再启动第一个 token 的 VF。
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                for (int64_t start = sequenceBegin_; start < sequenceEnd_; start += TOKEN_TILE) {
                    const uint16_t rows = static_cast<uint16_t>(
                        (sequenceEnd_ - start < TOKEN_TILE) ? (sequenceEnd_ - start) : TOKEN_TILE);
                    const uint32_t buffer = tileIndex++ % INPUT_BUFFERS;
                    CopyInputTile(buffer, start, rows, keyHead, head, valueOffset);
                    Mutex::Lock<PIPE_V>(INPUT_MUTEX[buffer]);
                    Mutex::Lock<PIPE_V>(OUTPUT_MUTEX[buffer]);
                    for (uint16_t token = 0; token < rows; ++token) {
                        const uint32_t nextBank = (currentBank + 1) % BANKS;
                        Mutex::Lock<PIPE_V>(STATE_MUTEX[nextBank]);
                        ComputeToken(buffer, token, currentBank, nextBank, aLog, parameterBuffer);
                        Mutex::Unlock<PIPE_V>(STATE_MUTEX[nextBank]);
                        const int64_t position = start + token;
                        if (hasFollowingWork && position + static_cast<int64_t>(BANKS - 1) == sequenceEnd_) {
                            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                            stateReadyEvent = true;
                        }
                        if ((tiling_->inplaceFinalState || tiling_->outputFinalState) &&
                            (tiling_->hasSsmStateIndices || position + 1 == sequenceEnd_)) {
                            const uint64_t slot = StateSlot(request, position);
                            if (tiling_->hasSsmStateIndices) {
                                OrderStateSnapshotWrites(nextBank, slot, pendingStateSlots);
                            }
                            Mutex::Lock<PIPE_MTE3>(STATE_MUTEX[nextBank]);
                            CopyStateOut(nextBank, slot, head, valueOffset);
                            Mutex::Unlock<PIPE_MTE3>(STATE_MUTEX[nextBank]);
                        }
                        currentBank = nextBank;
                    }
                    Mutex::Unlock<PIPE_V>(INPUT_MUTEX[buffer]);
                    Mutex::Unlock<PIPE_V>(OUTPUT_MUTEX[buffer]);
                    const uint64_t outOffset =
                        (uint64_t(start) * tiling_->nv + head) * tiling_->dv + valueOffset;
                    // out 的 token 跨度恒为 nv*dv，本片只写其中 DIM 列：行间隔是
                    // nv*dv - DIM，而不是 (nv-1)*dv。只有 dv == DIM 时两者相等，
                    // V=256 下少算 (dv-DIM) 会让每一行都左移，越界部分不写。
                    const DataCopyExtParams outputParams{rows, DIM * sizeof(bfloat16_t), 0,
                        uint32_t((uint64_t(tiling_->nv) * tiling_->dv - DIM) * sizeof(bfloat16_t)), 0};
                    Mutex::Lock<PIPE_MTE3>(OUTPUT_MUTEX[buffer]);
                    DataCopyPad(outputGm_[outOffset], outputTile_[buffer], outputParams);
                    Mutex::Unlock<PIPE_MTE3>(OUTPUT_MUTEX[buffer]);
                }
            }
            if (usesBias) Mutex::Unlock<PIPE_V>(PARAMETER_MUTEX[parameterBuffer]);
            parameterBuffer = (parameterBuffer + 1) % PARAMETER_BUFFERS;
        }
        // 结尾关闭未消费的握手，保证最后一次搬出完成。
        if (stateReadyEvent) WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        // 与 A5 一致：在 V 上依次排空两个 out 槽。最后一笔 out 搬出排在此前所有
        // State 快照之后（同在 MTE3 上按序发射），因此等待 out 槽即等价于等待全部
        // 收尾 DMA 完成；未使用的槽位本来就是解锁状态，锁一次再解锁无副作用。
        Mutex::Lock<PIPE_V>(OUTPUT_MUTEX[0]);
        Mutex::Unlock<PIPE_V>(OUTPUT_MUTEX[0]);
        Mutex::Lock<PIPE_V>(OUTPUT_MUTEX[1]);
        Mutex::Unlock<PIPE_V>(OUTPUT_MUTEX[1]);
    }

private:
    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t bankBytes = DIM * DIM * static_cast<uint32_t>(sizeof(State));
        uint32_t offset = 0;
        for (uint32_t bank = 0; bank < BANKS; ++bank) {
            stateBanks_[bank] = LocalTensor<State>(TPosition::VECCALC, offset, DIM * DIM);
            offset += bankBytes;
        }
        for (uint32_t buffer = 0; buffer < INPUT_BUFFERS; ++buffer) {
            queryTile_[buffer] = LocalTensor<bfloat16_t>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TOKEN_TILE * DIM * sizeof(bfloat16_t);
            keyTile_[buffer] = LocalTensor<bfloat16_t>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TOKEN_TILE * DIM * sizeof(bfloat16_t);
            valueTile_[buffer] = LocalTensor<bfloat16_t>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TOKEN_TILE * DIM * sizeof(bfloat16_t);
            gateTile_[buffer] = LocalTensor<Gate>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TILE_VECTOR_BYTES;
            betaTile_[buffer] = LocalTensor<Beta>(TPosition::VECCALC, offset, TILE_SCALAR_BYTES / sizeof(Beta));
            offset += TILE_SCALAR_BYTES;
            outputTile_[buffer] = LocalTensor<bfloat16_t>(TPosition::VECCALC, offset, TOKEN_TILE * DIM);
            offset += TOKEN_TILE * DIM * sizeof(bfloat16_t);
        }
        for (uint32_t buffer = 0; buffer < PARAMETER_BUFFERS; ++buffer) {
            biasUb_[buffer] = LocalTensor<float>(TPosition::VECCALC, offset, DIM);
            offset += PARAMETER_BYTES;
        }
        predictionScratchUb_ = LocalTensor<float>(TPosition::VECCALC, offset, DIM);
        offset += PREDICTION_BYTES;
        outputScratchUb_ = LocalTensor<float>(TPosition::VECCALC, offset, DIM);
        offset += OUTPUT_SCRATCH_BYTES;
    }

    __aicore__ inline int64_t Metadata(GM_ADDR address, uint32_t dtype, uint64_t element) const
    {
        if (dtype == 0) {
            GlobalTensor<int32_t> tensor;
            tensor.SetGlobalBuffer((__gm__ int32_t *)address);
            return tensor.GetValue(element);
        }
        GlobalTensor<int64_t> tensor;
        tensor.SetGlobalBuffer((__gm__ int64_t *)address);
        return tensor.GetValue(element);
    }

    __aicore__ inline bool ValidateSequences() const
    {
        if (!tiling_->hasCuSeqlens) {
            return true;
        }
        int64_t previous = Metadata(cuAddress_, tiling_->cuSeqlensDtype, 0);
        if (previous != 0) {
            return false;
        }
        for (uint32_t request = 0; request < tiling_->b; ++request) {
            const int64_t end = Metadata(cuAddress_, tiling_->cuSeqlensDtype, request + 1);
            if (end < previous || uint64_t(end) > tiling_->t ||
                (tiling_->hasSsmStateIndices && tiling_->ssmStateStride &&
                 uint64_t(end - previous) > tiling_->ssmStateStride)) {
                return false;
            }
            previous = end;
        }
        return true;
    }

    __aicore__ inline void UpdateSequence(uint64_t request)
    {
        if (cachedRequest_ == request) {
            return;
        }
        cachedRequest_ = request;
        sequenceBegin_ = tiling_->hasCuSeqlens ?
            Metadata(cuAddress_, tiling_->cuSeqlensDtype, request) :
            int64_t(request * tiling_->seqLen);
        sequenceEnd_ = tiling_->hasCuSeqlens ?
            Metadata(cuAddress_, tiling_->cuSeqlensDtype, request + 1) :
            int64_t((request + 1) * tiling_->seqLen);
    }

    __aicore__ inline uint64_t StateSlot(uint64_t request, int64_t token) const
    {
        if (!tiling_->hasSsmStateIndices) {
            return request;
        }
        const uint64_t offset = tiling_->ssmStateStride ?
            request * tiling_->ssmStateStride + token - sequenceBegin_ : token;
        return uint64_t(Metadata(indicesAddress_, tiling_->ssmStateIndicesDtype, offset));
    }

    __aicore__ inline bool ValidateSlots(uint64_t request) const
    {
        const int64_t sequenceLength = sequenceEnd_ - sequenceBegin_;
        if (tiling_->hasAcceptedTokens) {
            const int64_t accepted = Metadata(acceptedAddress_, tiling_->acceptedTokensDtype, request);
            // 与 A5 保持一致：二维 speculative 索引表描述的是一个状态窗口，可以比本次
            // 查询窗口更长，accepted 因此以 max_step（ssmStateStride）为上界；packed
            // 一维表没有额外窗口，仍以本次请求长度为上界。选中的初态槽位另行校验。
            const uint64_t acceptedLimit = tiling_->ssmStateStride != 0 ?
                static_cast<uint64_t>(tiling_->ssmStateStride) : static_cast<uint64_t>(sequenceLength);
            if (accepted < 1 || static_cast<uint64_t>(accepted) > acceptedLimit) {
                return false;
            }
            const int64_t initialToken = sequenceBegin_ + accepted - 1;
            if (StateSlot(request, initialToken) >= tiling_->sBlockNum) {
                return false;
            }
        }
        if (!tiling_->hasSsmStateIndices) {
            return request < tiling_->sBlockNum;
        }
        for (int64_t token = sequenceBegin_; token < sequenceEnd_; ++token) {
            if (StateSlot(request, token) >= tiling_->sBlockNum) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint64_t InitialSlot(uint64_t request) const
    {
        int64_t token = sequenceBegin_;
        if (tiling_->hasAcceptedTokens) {
            token += Metadata(acceptedAddress_, tiling_->acceptedTokensDtype, request) - 1;
        }
        return StateSlot(request, token);
    }

    // 逐 token 的 State 快照会写回 GM，同一 request 内多个 token 可能命中同一槽位。
    // bank Mutex 只保护 UB 复用，同流水上的 Lock/Unlock 也不能约束两笔 MTE3 搬出的
    // 完成顺序，因此记录每个 bank 最近一次快照的目标槽位；命中相同槽位时插入 MTE3
    // 屏障，保证同址覆盖按 token 顺序完成。
    __aicore__ inline void OrderStateSnapshotWrites(
        uint32_t sourceBank, uint64_t slot, uint64_t (&pendingStateSlots)[MAX_STATE_BANKS])
    {
        // 该 token 的 V 锁已经把源 bank 的上一笔快照回收完。
        pendingStateSlots[sourceBank] = INVALID_STATE_SLOT;
        for (uint32_t bank = 0; bank < BANKS; ++bank) {
            if (pendingStateSlots[bank] == slot) {
                PipeBarrier<PIPE_MTE3>();
                for (uint32_t completedBank = 0; completedBank < BANKS; ++completedBank) {
                    pendingStateSlots[completedBank] = INVALID_STATE_SLOT;
                }
                break;
            }
        }
        pendingStateSlots[sourceBank] = slot;
    }

    // State 的 GM 地址与 stride 由调用方给出，host 只校验形状与不重叠。
    // 末两维稠密（stride2 == DIM）且基址满足 32 B 对齐时用单块 DMA，否则按 DIM 行
    // DataCopyPad 搬运；与 A5 的选择一致，A6 额外复核基址对齐以满足单块搬运约束。
    // V-first 的 V 片是行区间，K-first 的 V 片是列区间；两者都落在最后两维的
    // 稠密矩阵内，列偏移在 K-first 下等于 valueOffset * stride3。
    __aicore__ inline void CopyStateIn(uint32_t bank, uint64_t slot, uint64_t head, uint32_t valueOffset)
    {
        const uint64_t offset = slot * tiling_->stateInStride0 + head * tiling_->stateInStride1 +
            valueOffset * (V_FIRST ? tiling_->stateInStride2 : tiling_->stateInStride3);
        if (tiling_->stateInStride2 == DIM && (offset * sizeof(State)) % 32 == 0) {
            DataCopy(stateBanks_[bank], stateGm_[offset], DIM * DIM);
            return;
        }
        const DataCopyExtParams params{DIM, DIM * sizeof(State),
            uint32_t((tiling_->stateInStride2 - DIM) * sizeof(State)), 0, 0};
        DataCopyPad(stateBanks_[bank], stateGm_[offset], params,
            DataCopyPadExtParams<State>{false, 0, 0, State(0)});
    }

    __aicore__ inline void CopyStateOut(uint32_t bank, uint64_t slot, uint64_t head, uint32_t valueOffset)
    {
        const uint64_t offset = slot * tiling_->stateOutStride0 + head * tiling_->stateOutStride1 +
            valueOffset * (V_FIRST ? tiling_->stateOutStride2 : tiling_->stateOutStride3);
        const DataCopyExtParams params{DIM, DIM * sizeof(State), 0,
            uint32_t((tiling_->stateOutStride2 - DIM) * sizeof(State)), 0};
        DataCopyPad(stateOutGm_[offset], stateBanks_[bank], params);
    }

    // 所有平台统一按 token-major 稠密跨度寻址：BSND 的 batch 跨度必须等于
    // seqLen * tokenStride，因此逻辑 token 在输入里线性排列，与 TND 同构。
    __aicore__ inline uint64_t InputOffset(int64_t token, uint64_t head,
        uint64_t tokenStride, uint64_t headStride) const
    {
        return uint64_t(token) * tokenStride + head * headStride;
    }

    // value 按 V 片搬入：列偏移 valueOffset，行内只取 DIM 个元素，
    // 其余位置由 valueTokenStride 的行间隔跳过。q/k/gate/beta 与 V 片无关，
    // 但切片在任务内串行，tile 缓冲会被后续 tile 覆盖，因此每片重新搬入。
    // 整 tile 一条 DataCopyPad：host 已保证 batch 跨度等于 T*tokenStride，
    // 一个 request 的 token 在输入里连续，tile 不会跨到别的 batch。
    __aicore__ inline void CopyInputTile(uint32_t buffer, int64_t start, uint16_t rows,
                                        uint64_t keyHead, uint64_t head, uint32_t valueOffset)
    {
        const uint64_t queryOffset = InputOffset(start, keyHead,
            tiling_->queryTokenStride, tiling_->queryHeadStride);
        const uint64_t keyOffset = InputOffset(start, keyHead,
            tiling_->keyTokenStride, tiling_->keyHeadStride);
        const uint64_t valueGmOffset = InputOffset(start, head,
            tiling_->valueTokenStride, tiling_->valueHeadStride) + valueOffset;
        const uint64_t gateOffset = (uint64_t(start) * tiling_->nv + head) * DIM;
        const uint64_t betaOffset = uint64_t(start) * tiling_->nv + head;
        const DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, bfloat16_t(0)};
        const DataCopyExtParams queryParams{rows, DIM * sizeof(bfloat16_t),
            uint32_t((tiling_->queryTokenStride - DIM) * sizeof(bfloat16_t)), 0, 0};
        const DataCopyExtParams keyParams{rows, DIM * sizeof(bfloat16_t),
            uint32_t((tiling_->keyTokenStride - DIM) * sizeof(bfloat16_t)), 0, 0};
        const DataCopyExtParams valueParams{rows, DIM * sizeof(bfloat16_t),
            uint32_t((tiling_->valueTokenStride - DIM) * sizeof(bfloat16_t)), 0, 0};
        const DataCopyExtParams gateParams{rows, DIM * sizeof(Gate),
            uint32_t((tiling_->nv - 1) * DIM * sizeof(Gate)), 0, 0};
        const DataCopyExtParams betaParams{rows, sizeof(Beta),
            uint32_t((tiling_->nv - 1) * sizeof(Beta)), 0, 0};
        Mutex::Lock<PIPE_MTE2>(INPUT_MUTEX[buffer]);
        DataCopyPad(queryTile_[buffer], queryGm_[queryOffset], queryParams, pad);
        DataCopyPad(keyTile_[buffer], keyGm_[keyOffset], keyParams, pad);
        DataCopyPad(valueTile_[buffer], valueGm_[valueGmOffset], valueParams, pad);
        DataCopyPad(gateTile_[buffer], gateGm_[gateOffset], gateParams,
            DataCopyPadExtParams<Gate>{false, 0, 0, Gate(0)});
        DataCopyPad(betaTile_[buffer], betaGm_[betaOffset], betaParams,
            DataCopyPadExtParams<Beta>{false, 0, 0, Beta(0)});
        Mutex::Unlock<PIPE_MTE2>(INPUT_MUTEX[buffer]);
    }

    __aicore__ inline void ComputeToken(uint32_t buffer, uint16_t token, uint32_t source,
                                        uint32_t target, float aLog, uint32_t parameterBuffer)
    {
        __ubuf__ float *predictionScratch = (__ubuf__ float *)predictionScratchUb_.GetPhyAddr();
        __ubuf__ float *outputScratch = (__ubuf__ float *)outputScratchUb_.GetPhyAddr();
        RecurrentKdaTokenVf<State, Gate, Beta, V_FIRST, FUSED>(
            (__ubuf__ State *)stateBanks_[target].GetPhyAddr(),
            (__ubuf__ State *)stateBanks_[source].GetPhyAddr(),
            (__ubuf__ bfloat16_t *)keyTile_[buffer][token * DIM].GetPhyAddr(),
            (__ubuf__ bfloat16_t *)queryTile_[buffer][token * DIM].GetPhyAddr(),
            (__ubuf__ bfloat16_t *)valueTile_[buffer][token * DIM].GetPhyAddr(),
            (__ubuf__ Gate *)gateTile_[buffer][token * DIM].GetPhyAddr(),
            (__ubuf__ Beta *)betaTile_[buffer][token * (32 / sizeof(Beta))].GetPhyAddr(),
            (__ubuf__ float *)biasUb_[parameterBuffer].GetPhyAddr(),
            (__ubuf__ bfloat16_t *)outputTile_[buffer][token * DIM].GetPhyAddr(),
            tiling_->scale, aLog, vfFlags_, tiling_->lowerBound, predictionScratch, outputScratch);
    }

    const RecurrentKdaTilingData *tiling_;
    GlobalTensor<bfloat16_t> queryGm_;
    GlobalTensor<bfloat16_t> keyGm_;
    GlobalTensor<bfloat16_t> valueGm_;
    GlobalTensor<bfloat16_t> outputGm_;
    GlobalTensor<State> stateGm_;
    GlobalTensor<State> stateOutGm_;
    GlobalTensor<Gate> gateGm_;
    GlobalTensor<Beta> betaGm_;
    GlobalTensor<float> aLogGm_;
    GlobalTensor<float> biasGm_;
    GM_ADDR cuAddress_;
    GM_ADDR indicesAddress_;
    GM_ADDR acceptedAddress_;
    LocalTensor<State> stateBanks_[MAX_STATE_BANKS];
    LocalTensor<bfloat16_t> queryTile_[INPUT_BUFFERS];
    LocalTensor<bfloat16_t> keyTile_[INPUT_BUFFERS];
    LocalTensor<bfloat16_t> valueTile_[INPUT_BUFFERS];
    LocalTensor<bfloat16_t> outputTile_[INPUT_BUFFERS];
    LocalTensor<Gate> gateTile_[INPUT_BUFFERS];
    LocalTensor<Beta> betaTile_[INPUT_BUFFERS];
    LocalTensor<float> biasUb_[PARAMETER_BUFFERS];
    LocalTensor<float> predictionScratchUb_;
    LocalTensor<float> outputScratchUb_;
    uint32_t vfFlags_ = 0;
    uint64_t cachedRequest_ = uint64_t(-1);
    int64_t sequenceBegin_ = 0;
    int64_t sequenceEnd_ = 0;
};
} // namespace RecurrentKdaA6
