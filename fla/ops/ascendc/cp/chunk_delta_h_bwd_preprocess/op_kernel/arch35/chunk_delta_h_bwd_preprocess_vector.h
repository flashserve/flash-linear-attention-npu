/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess_vector.h
 * \brief arch35（Ascend950 / dav-3510）Vector 侧实现。
 *
 * 与 arch22 的差别：把三段热点（V0 门控、V2 dV̂'、V4 状态更新 / P_c 对角注入）下沉到 RegBase
 * `__simd_vf__`（VECTOR_REG_WIDTH = 256B = 64 lane / fp32 寄存器）一趟算完，消除 arch22 版本里
 * "逐行 Muls + 逐行 ExpScalar(V→S 同步) + 逐元素 SetValue/GetValue"的标量往返；Cube 侧与
 * arch22 保持一致（见 arch35/chunk_delta_h_bwd_preprocess_cube.h）。
 *
 * VF 与 Stage 的对应关系：
 *   CDHP_V0GateScaleVF     V0：Q̄s / K̄ 门控行缩放（qFactor = exp(g·ln2)·scale，dvFactor = exp((g_last-g)·ln2)）
 *   CDHP_V0LastDecayVF     V0：decayK[:] = exp(g_last·ln2) 按 K 广播
 *   CDHP_V2DvHatVF         V2：dV̂' = -(dV_pre + dv_local)
 *   CDHP_V4StateUpdateVF   V4 路 A：dH_new = decayK ⊙ dH_old + (qterm + wterm)
 *   CDHP_V4PcDiagVF        V4 路 B：P_c = diag(decayK) - T1（对角用寄存器 mask 注入）
 *   CDHP_InitIdentityVF    InitState：P 初值 = I（对角用寄存器 mask 注入）
 *
 * 所有 VF 只在 FP32 UB 平面上工作（dtype 转换仍走既有的 Cast + DataCopyPad 通路），
 * 因此内部统一用默认 LoadDist / StoreDist，不涉及 b16 的 unpack / pack 布局。
 *
 * 六 Stage 的物理切分与 flag 协议见 chunk_delta_h_bwd_preprocess_policy.h；
 * 同 chunk 内 AIV/AIC 严格交替握手。
 */

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_ARCH35_VECTOR_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_ARCH35_VECTOR_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"
#include "../chunk_delta_h_bwd_preprocess_common.h"
#include "../chunk_delta_h_bwd_preprocess_policy.h"

using namespace AscendC;
using namespace AscendC::MicroAPI;

namespace CP {

constexpr float CDHP_LN2 = 0.6931471805599453f;
// 向量侧行分块：16 行 × 256 列 = 4096 元素（FP32 16 KiB）；K/V <= 256 时 UB 占用固定
// 每个向量 tile 的行数：取满一个 chunk（64 行）→ 每 chunk 每段只有 1 轮搬运，
// 单笔搬运 64x128（模型 dtype 16 KB / FP32 32 KB），替代原来 16 行 x 4 轮的小包。
constexpr uint32_t CDHP_VEC_TILE = 32;
// 单块 tile 的最大元素数：行数 x max(K, V)
constexpr uint32_t CDHP_SCRATCH_ELEMS = CDHP_VEC_TILE * 128;


// 「先集中下发、再统一等」用的事件 id（不同 HardEvent 对可复用同一数值）：
//   LOAD0..3  : MTE2→V，标记第 i 个平面搬运完成
//   STORE0..4 : V→MTE3，标记第 i 个平面搬出完成（下一次复用这块 UB 前要等它）
constexpr uint32_t CDHP_EV_LOAD0 = 3;
constexpr uint32_t CDHP_EV_LOAD1 = 4;
constexpr uint32_t CDHP_EV_LOAD2 = 5;
constexpr uint32_t CDHP_EV_LOAD3 = 6;
constexpr uint32_t CDHP_EV_STORE0 = 2;
constexpr uint32_t CDHP_EV_STORE1 = 3;
constexpr uint32_t CDHP_EV_STORE2 = 4;
constexpr uint32_t CDHP_EV_STORE3 = 6;
constexpr uint32_t CDHP_EV_STORE4 = 7;
// 模型 dtype 暂存区双槽（ping-pong）：搬入用 s0DT_、搬出用 s1DT_，各自 per-slot credit 平衡
constexpr uint32_t CDHP_UB_SLOT_NUM = 2;

// dav-3510：一个 fp32 向量寄存器 = 256B = 64 lane
constexpr uint16_t CDHP_VF_LANES = static_cast<uint16_t>(AscendC::VECTOR_REG_WIDTH / sizeof(float));
constexpr uint32_t CDHP_EV_V_S = 0;
constexpr uint32_t CDHP_EV_S_V = 1;
constexpr uint32_t CDHP_EV_MTE2_V = 2;
constexpr uint32_t CDHP_EV_V_MTE2 = 3;
constexpr uint32_t CDHP_EV_V_MTE3 = 4;
constexpr uint32_t CDHP_EV_MTE3_V = 5;
constexpr uint32_t CDHP_EV_MTE3_MTE2 = 6;
// 复用 s0DT_ 暂存区前的保护事件（与上面的 id 数值可以相同：不同 HardEvent 对是不同的硬件事件）
constexpr uint32_t CDHP_EV_S_MTE2 = 0;
// 复用 UB 前等 MTE3 读完的 MTE3→S 事件（与上面的 id 数值可以相同：不同 HardEvent 对是不同硬件事件）
constexpr uint32_t CDHP_EV_MTE3_S = 1;
// v13：W / do 的"同 dtype 纯拷贝"专用暂存槽（只走 MTE2/MTE3，不碰 V pipe），单槽自平衡
constexpr uint32_t CDHP_EV_COPY_MTE3_MTE2 = 7;  // 本槽上一笔搬出已完成 → 允许下一次搬入覆盖
constexpr uint32_t CDHP_EV_COPY_MTE2_MTE3 = 0;  // 本槽搬入已完成 → 允许搬出读取

// 编译期判断两个类型是否相同（避免依赖 <type_traits> 在 kernel 侧的可用性）
template <typename A, typename B>
struct ChunkDeltaHBwdPreprocessSameType {
    static constexpr bool value = false;
};
template <typename A>
struct ChunkDeltaHBwdPreprocessSameType<A, A> {
    static constexpr bool value = true;
};

__aicore__ inline uint32_t MinV(uint32_t a, uint32_t b)
{
    return (a < b) ? a : b;
}

// ===========================================================================
// RegBase VF 段（dav-3510）
//
// 约定：
//   * 行列 tile 的行距（K/V）是 CDHP_VF_LANES 的整数倍，列循环天然落在寄存器对齐边界上；
//     尾块用 UpdateMask 收窄写回范围，未被 mask 覆盖的 lane 保持原值，因此调用方仍需在
//     调用前把无效行清零（与 arch22 的零填充语义一致）。
//   * VF 是 V pipe 上的指令流，与本文件里其它 AscendC::Cast/Muls/Exp 等向量 API 同 pipe 顺序执行，
//     不需要额外的跨 pipe 事件。
// ===========================================================================

// V0 门控：Q̄s[r,:] *= exp(g[r]·ln2)·scale；K̄[r,:] *= exp((g_last - g[r])·ln2)
// gateTile 指向本 chunk 的第 r0 行（g 的行指针），gateLast 指向本 chunk 最后一行 g[rows-1]。
__simd_vf__ inline void CDHP_V0GateScaleVF(__ubuf__ float *qTile, __ubuf__ float *kTile,
                                           __ubuf__ float *gateTile, __ubuf__ float *gateLast,
                                           uint16_t validRows, uint16_t kDim, float scale)
{
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    const uint16_t colLoop = static_cast<uint16_t>((kDim + CDHP_VF_LANES - 1) / CDHP_VF_LANES);

    RegTensor<float> lastReg;
    LoadIn<float, true>(lastReg, gateLast);
    for (uint16_t r = 0; r < validRows; ++r) {
        RegTensor<float> gRowReg;
        RegTensor<float> qFacReg;
        RegTensor<float> dvFacReg;
        LoadIn<float, true>(gRowReg, gateTile + r);
        // 与 arch22 的逐行 ExpScalar 同序：exp(g·ln2) 先乘 scale，再乘到整行
        Muls(qFacReg, gRowReg, CDHP_LN2, maskAll);
        Exp(qFacReg, qFacReg, maskAll);
        Muls(qFacReg, qFacReg, scale, maskAll);
        Sub(dvFacReg, lastReg, gRowReg, maskAll);
        Muls(dvFacReg, dvFacReg, CDHP_LN2, maskAll);
        Exp(dvFacReg, dvFacReg, maskAll);
        for (uint16_t c = 0; c < colLoop; ++c) {
            const uint32_t colOff = static_cast<uint32_t>(c) * CDHP_VF_LANES;
            const uint32_t off = static_cast<uint32_t>(r) * kDim + colOff;
            uint32_t left = kDim - colOff;
            uint32_t lanes = (left > CDHP_VF_LANES) ? CDHP_VF_LANES : left;
            MaskReg maskCol = UpdateMask<float>(lanes);
            RegTensor<float> srcReg;
            RegTensor<float> dstReg;
            LoadAlign(srcReg, qTile + off);
            Mul(dstReg, srcReg, qFacReg, maskCol);
            StoreAlign(qTile + off, dstReg, maskCol);
            LoadAlign(srcReg, kTile + off);
            Mul(dstReg, srcReg, dvFacReg, maskCol);
            StoreAlign(kTile + off, dstReg, maskCol);
        }
    }
}

// V0 衰减：decayK[:] = exp(g_last·ln2)
__simd_vf__ inline void CDHP_V0LastDecayVF(__ubuf__ float *decayOut, __ubuf__ float *gateLast, uint16_t kDim)
{
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> lastReg;
    LoadIn<float, true>(lastReg, gateLast);
    Muls(lastReg, lastReg, CDHP_LN2, maskAll);
    Exp(lastReg, lastReg, maskAll);
    const uint16_t colLoop = static_cast<uint16_t>((kDim + CDHP_VF_LANES - 1) / CDHP_VF_LANES);
    for (uint16_t c = 0; c < colLoop; ++c) {
        const uint32_t colOff = static_cast<uint32_t>(c) * CDHP_VF_LANES;
        uint32_t left = kDim - colOff;
        uint32_t lanes = (left > CDHP_VF_LANES) ? CDHP_VF_LANES : left;
        MaskReg maskCol = UpdateMask<float>(lanes);
        StoreAlign(decayOut + colOff, lastReg, maskCol);
    }
}

// V2：dV̂' = -(dV_pre + dv_local)，一趟完成加与取负（out 允许与 pre 指向同一块 UB）
__simd_vf__ inline void CDHP_V2DvHatVF(__ubuf__ float *out, __ubuf__ float *pre, __ubuf__ float *dvLocal,
                                       uint16_t elems)
{
    const uint16_t loopCnt = static_cast<uint16_t>((elems + CDHP_VF_LANES - 1) / CDHP_VF_LANES);
    for (uint16_t c = 0; c < loopCnt; ++c) {
        const uint32_t off = static_cast<uint32_t>(c) * CDHP_VF_LANES;
        uint32_t left = elems - off;
        uint32_t lanes = (left > CDHP_VF_LANES) ? CDHP_VF_LANES : left;
        MaskReg mask = UpdateMask<float>(lanes);
        RegTensor<float> preReg;
        RegTensor<float> dvReg;
        RegTensor<float> sumReg;
        LoadAlign(preReg, pre + off);
        LoadAlign(dvReg, dvLocal + off);
        Add(sumReg, preReg, dvReg, mask);
        Muls(sumReg, sumReg, -1.0f, mask);
        StoreAlign(out + off, sumReg, mask);
    }
}

// V4 路 A：dH_new = decayK ⊙ dH_old + (qterm + wterm)
// decay 由行号索引（rowBase + r 是 dH 的 K 维行号）
__simd_vf__ inline void CDHP_V4StateUpdateVF(__ubuf__ float *state, __ubuf__ float *termQ, __ubuf__ float *termW,
                                             __ubuf__ float *decay, uint16_t rowBase, uint16_t rowCount,
                                             uint16_t colCount)
{
    const uint16_t colLoop = static_cast<uint16_t>((colCount + CDHP_VF_LANES - 1) / CDHP_VF_LANES);
    for (uint16_t r = 0; r < rowCount; ++r) {
        RegTensor<float> facReg;
        LoadIn<float, true>(facReg, decay + rowBase + r);
        for (uint16_t c = 0; c < colLoop; ++c) {
            const uint32_t colOff = static_cast<uint32_t>(c) * CDHP_VF_LANES;
            const uint32_t off = static_cast<uint32_t>(r) * colCount + colOff;
            uint32_t left = colCount - colOff;
            uint32_t lanes = (left > CDHP_VF_LANES) ? CDHP_VF_LANES : left;
            MaskReg mask = UpdateMask<float>(lanes);
            RegTensor<float> stateReg;
            RegTensor<float> qReg;
            RegTensor<float> wReg;
            RegTensor<float> addReg;
            LoadAlign(stateReg, state + off);
            LoadAlign(qReg, termQ + off);
            LoadAlign(wReg, termW + off);
            Add(addReg, qReg, wReg, mask);
            Mul(stateReg, stateReg, facReg, mask);
            Add(stateReg, stateReg, addReg, mask);
            StoreAlign(state + off, stateReg, mask);
        }
    }
}

// v3 状态链（单项版）：state = decay ⊙ state + term
// 用于 P 链：P_new = decayK ⊙ P_prev + ZP（ZP = (-T1)@P_prev 已由 Cube 算好）
__simd_vf__ inline void CDHP_V3AccumVF(__ubuf__ float *state, __ubuf__ float *term, __ubuf__ float *decay,
                                       uint16_t rowBase, uint16_t rowCount, uint16_t colCount)
{
    const uint16_t colLoop = static_cast<uint16_t>((colCount + CDHP_VF_LANES - 1) / CDHP_VF_LANES);
    for (uint16_t r = 0; r < rowCount; ++r) {
        RegTensor<float> facReg;
        LoadIn<float, true>(facReg, decay + rowBase + r);
        for (uint16_t c = 0; c < colLoop; ++c) {
            const uint32_t colOff = static_cast<uint32_t>(c) * CDHP_VF_LANES;
            const uint32_t off = static_cast<uint32_t>(r) * colCount + colOff;
            uint32_t left = colCount - colOff;
            uint32_t lanes = (left > CDHP_VF_LANES) ? CDHP_VF_LANES : left;
            MaskReg mask = UpdateMask<float>(lanes);
            RegTensor<float> stateReg;
            RegTensor<float> termReg;
            LoadAlign(stateReg, state + off);
            LoadAlign(termReg, term + off);
            Mul(stateReg, stateReg, facReg, mask);
            Add(stateReg, stateReg, termReg, mask);
            StoreAlign(state + off, stateReg, mask);
        }
    }
}

// V4 路 B：P_c[r,c] = -T1[r,c]（c == rowBase + r 时再叠上 decayK[rowBase + r]）
// 对角注入不再用标量 SetValue/GetValue：列号由 Arange 生成，与行号 Compares 出对角 mask，
// 先把 decay 落到只有对角 lane 有效的寄存器再整体相加（未命中的 lane 恒为 0，与合并模式无关）。
__simd_vf__ inline void CDHP_V4PcDiagVF(__ubuf__ float *out, __ubuf__ float *t1, __ubuf__ float *decay,
                                        uint16_t rowBase, uint16_t rowCount, uint16_t colCount)
{
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    const uint16_t colLoop = static_cast<uint16_t>((colCount + CDHP_VF_LANES - 1) / CDHP_VF_LANES);
    for (uint16_t r = 0; r < rowCount; ++r) {
        const int32_t rowIdx = static_cast<int32_t>(rowBase + r);
        RegTensor<float> facReg;
        LoadIn<float, true>(facReg, decay + rowBase + r);
        for (uint16_t c = 0; c < colLoop; ++c) {
            const uint32_t colOff = static_cast<uint32_t>(c) * CDHP_VF_LANES;
            const uint32_t off = static_cast<uint32_t>(r) * colCount + colOff;
            uint32_t left = colCount - colOff;
            uint32_t lanes = (left > CDHP_VF_LANES) ? CDHP_VF_LANES : left;
            MaskReg mask = UpdateMask<float>(lanes);
            MaskReg diagMask;
            RegTensor<int32_t> colIdxReg;
            RegTensor<float> t1Reg;
            RegTensor<float> negReg;
            RegTensor<float> diagReg;
            Arange<int32_t>(colIdxReg, static_cast<int32_t>(colOff));
            Compares<int32_t, CMPMODE::EQ>(diagMask, colIdxReg, rowIdx, mask);
            Duplicate(diagReg, 0.0f, maskAll);
            Add(diagReg, facReg, diagReg, diagMask);
            LoadAlign(t1Reg, t1 + off);
            Muls(negReg, t1Reg, -1.0f, mask);
            Add(negReg, negReg, diagReg, mask);
            StoreAlign(out + off, negReg, mask);
        }
    }
}

// InitState：P 初值 = I（[rowBase, rowBase+rowCount) 行 × colCount 列的对角块）
__simd_vf__ inline void CDHP_InitIdentityVF(__ubuf__ float *dst, uint16_t rowBase, uint16_t rowCount,
                                            uint16_t colCount)
{
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> oneReg;
    Duplicate(oneReg, 1.0f, maskAll);
    const uint16_t colLoop = static_cast<uint16_t>((colCount + CDHP_VF_LANES - 1) / CDHP_VF_LANES);
    for (uint16_t r = 0; r < rowCount; ++r) {
        const int32_t rowIdx = static_cast<int32_t>(rowBase + r);
        for (uint16_t c = 0; c < colLoop; ++c) {
            const uint32_t colOff = static_cast<uint32_t>(c) * CDHP_VF_LANES;
            const uint32_t off = static_cast<uint32_t>(r) * colCount + colOff;
            uint32_t left = colCount - colOff;
            uint32_t lanes = (left > CDHP_VF_LANES) ? CDHP_VF_LANES : left;
            MaskReg mask = UpdateMask<float>(lanes);
            MaskReg diagMask;
            RegTensor<int32_t> colIdxReg;
            RegTensor<float> diagReg;
            Arange<int32_t>(colIdxReg, static_cast<int32_t>(colOff));
            Compares<int32_t, CMPMODE::EQ>(diagMask, colIdxReg, rowIdx, mask);
            // 必须先把 diagReg 整行置 0 再做带 mask 的 Add：带 mask 的 Add 是 merge 语义，
            // 未被 mask 命中的 lane 保留的是**寄存器里**的旧值而不是内存里的旧值，
            // 少了这一步就会把上一轮残留的脏数据当成 I 的非对角元素（历史上表现为"只有 P 面错、
            // E 面正常，且随负载时好时坏"）。参照 CDHP_V4PcDiagVF 的写法。
            Duplicate(diagReg, 0.0f, maskAll);
            Add(diagReg, oneReg, diagReg, diagMask);
            StoreAlign(dst + off, diagReg, mask);
        }
    }
}

template <typename DT, typename GT>
class ChunkDeltaHBwdPreprocessVector : public ChunkDeltaHBwdPreprocessBase<DT, GT> {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk,
                                GM_ADDR cu_seqlens, GM_ADDR dhm, GM_ADDR userWs,
                                const ChunkDeltaHBwdPreprocessTilingData &tiling)
    {
        this->InitTilingData(tiling);
        this->SetUserWorkspace(userWs);
        this->ResolveSegment(cu_seqlens);
        qAddr_ = q;
        kAddr_ = k;
        wAddr_ = w;
        doAddr_ = d_o;
        dvAddr_ = dv;
        gAddr_ = g;
        gkAddr_ = gk;
        dhmGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dhm));
        scale_ = (tiling.isScale != 0) ? tiling.scale : 1.0f;
    }

    __aicore__ inline void InitBuffer(TPipe *pipe)
    {
        pipe_ = pipe;
        pipe_->InitBuffer(s0F32_, CDHP_SCRATCH_ELEMS * sizeof(float));
        pipe_->InitBuffer(s1F32_, CDHP_SCRATCH_ELEMS * sizeof(float));
        pipe_->InitBuffer(s2F32_, CDHP_SCRATCH_ELEMS * sizeof(float));
        pipe_->InitBuffer(s3F32_, CDHP_SCRATCH_ELEMS * sizeof(float));
        pipe_->InitBuffer(s4F32_, CDHP_SCRATCH_ELEMS * sizeof(float));
        // v10：搬入/搬出各用一块双槽 model dtype 暂存区（per-slot credit，见 LoadTileF32/StoreTileModel）
        pipe_->InitBuffer(s0DT_, CDHP_UB_SLOT_NUM * CDHP_SCRATCH_ELEMS * sizeof(DT));
        pipe_->InitBuffer(s1DT_, CDHP_UB_SLOT_NUM * CDHP_SCRATCH_ELEMS * sizeof(DT));
    // v13：W / do 的同 dtype 纯拷贝专用单槽暂存（8 KiB）
    pipe_->InitBuffer(s2DT_, CDHP_SCRATCH_ELEMS * sizeof(DT));
        // 门控单独一块 model dtype 暂存（g 的一行 / gk 的一行），不再借用搬运暂存槽
        pipe_->InitBuffer(gDT_, 256 * sizeof(DT));
        pipe_->InitBuffer(gF32_, 256 * sizeof(float));
        pipe_->InitBuffer(decayF32_, 256 * sizeof(float));
        // v5（参考 ChunkFwdH）：dH 的 rolling state 常驻 UB（K=V=128 → 64 KiB），
        // 每 chunk 只把模型 dtype 操作数写回 GM，不再每 chunk 经 GM workspace 读写 fp32 状态。
        pipe_->InitBuffer(dhStateF32_, 128u * 128u * sizeof(float));
    // v11：P 的 fp32 状态同样常驻 UB（K=K=128 → 64 KiB），不再走 PAt GM 平面往返
    pipe_->InitBuffer(pStateF32_, 128u * 128u * sizeof(float));
        for (uint32_t s = 0; s < CDHP_UB_SLOT_NUM; ++s) {
            SetFlag<HardEvent::V_MTE2>(s);    // 搬入槽 credit 预置
            SetFlag<HardEvent::MTE3_V>(s);    // 搬出槽 credit 预置
        }
    // v13：纯拷贝槽的初始 credit（MTE3→MTE2 方向）
    SetFlag<HardEvent::MTE3_MTE2>(CDHP_EV_COPY_MTE3_MTE2);
    }

    __aicore__ inline uint32_t NextUbSlot()
    {
        ubSlot_ ^= 1u;
        return ubSlot_;
    }

    // v13：W / do 这两个平面在 Vector 侧**完全没有实数运算**（既不门控也不取负），只需要把原始输入
    // 按模型 dtype 搬到 slot 的同一 dtype 平面上。原来走"载入 → Cast 到 FP32 → Cast 回模型 dtype → 落盘"，
    // 每 tile 多出 4 次 Cast（两个平面 × 载入/落出）。这里用一块专用暂存槽做同 dtype 中转：
    // 只保留 MTE2→MTE3 的顺序（一对 per-slot credit），完全不进 V pipe。
    // 契约：只对**满 tile** 调用——尾块的无效行必须落 0，仍走原来的 FP32 通路（先 Duplicate 清零）。
    __aicore__ inline void CopyTileModel(GM_ADDR src, GM_ADDR dst, uint32_t rows, uint32_t cols,
                                         uint32_t rowStrideIn, uint32_t rowStrideOut)
    {
        AscendC::GlobalTensor<DT> gSrc;
        AscendC::GlobalTensor<DT> gDst;
        gSrc.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(src));
        gDst.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dst));
        LocalTensor<DT> tmp = s2DT_.Get<DT>();
        // 本槽上一笔搬出已完成（首轮由 InitBuffer 预置）
        WaitFlag<HardEvent::MTE3_MTE2>(CDHP_EV_COPY_MTE3_MTE2);
        AscendC::DataCopyExtParams inParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(DT)),
                                            static_cast<uint32_t>((rowStrideIn - cols) * sizeof(DT)), 0, 0};
        AscendC::DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(tmp, gSrc, inParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(CDHP_EV_COPY_MTE2_MTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(CDHP_EV_COPY_MTE2_MTE3);
        AscendC::DataCopyExtParams outParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(DT)), 0,
                                             static_cast<uint32_t>((rowStrideOut - cols) * sizeof(DT)), 0};
        AscendC::DataCopyPad(gDst, tmp, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(CDHP_EV_COPY_MTE3_MTE2);
    }

    __aicore__ inline void Process()
    {
        const uint32_t chunkNum = this->ChunkNum(this->Bos(), this->Eos());
        const ChunkDeltaHBwdPreprocessTaskRange range = this->ResolveTaskRange();
        // 1:2 核型：本 AIV 只跑自己那一半 head 的整条逆序链（与 Cube 的 rounds 一一对应；
        // head 数已按 aivPerBlock 补齐，两个 AIV 的轮次、flag 收发严格对齐）
        const uint32_t headCount = this->AivHeadCount(range);
        for (uint32_t i = 0; i < headCount; ++i) {
            const uint32_t hv = this->AivHeadAt(range, i);
            InitState(chunkNum);
            // v3 流水：AIV 只做链外准备（slot）与链上的向量累加，Cube 承担全部 MMAD。
            // 前两个 chunk 的操作数先备好（各占一个 window），之后每轮追加 c+2。
            if (chunkNum > 0) {
                const uint32_t firstIdx = chunkNum - 1;
                StageV0(hv, firstIdx, firstIdx & 1u);
                CDHP_AIV_SET(CdhpFlag(CDHP_FLAG_V0_READY, firstIdx & 1u));
            }
            if (chunkNum > 1) {
                const uint32_t secondIdx = chunkNum - 2;
                StageV0(hv, secondIdx, secondIdx & 1u);
                CDHP_AIV_SET(CdhpFlag(CDHP_FLAG_V0_READY, secondIdx & 1u));
            }
            for (uint32_t c = 0; c < chunkNum; ++c) {
                const uint32_t chunkIdx = chunkNum - 1 - c;
                const uint32_t win = chunkIdx & 1u;
                // 链上准备：把"本 chunk 之前的"状态（首 chunk 即初值 0 / I）写成模型 dtype 操作数，
                // 并通知 Cube 可以开始 Z/ZP（STATE_READY 现在按 chunk 置位，取代原来的初值一次性置位）
                StageStateStore(win, c == 0);
                // 链上：等 Cube 的 Z/ZP，做 dH/P 的向量累加
                CDHP_AIV_WAIT(CdhpFlag(CDHP_FLAG_Z_READY, win));
                StageState(chunkIdx, win, c == 0);
                // 链外：继续备 c+2 的 slot（同 window 的上一位消费者已让出）
                if (c + 2 < chunkNum) {
                    const uint32_t nextIdx = chunkIdx - 2;
                    StageV0(hv, nextIdx, nextIdx & 1u);
                    CDHP_AIV_SET(CdhpFlag(CDHP_FLAG_V0_READY, nextIdx & 1u));
                }
            }
            WriteOutput(hv);
        }
    }

private:
    template <typename T>
    __aicore__ inline static __ubuf__ T *UbPtr(const LocalTensor<T> &tensor)
    {
        return (__ubuf__ T *)tensor.GetPhyAddr();
    }

    // 复用 s0DT_ 暂存区（以及门控用的 gF）前的保护：该缓冲可能刚被上一轮的
    //   - V（Cast 写 s0DT_ / Cast 写 gF）
    //   - MTE3（StoreTileModel 从 s0DT_ 搬出）
    // 使用过。关闭自动同步后必须显式等待，否则新的一轮 MTE2 载入会在上一轮还没读完时覆盖内容，
    // 表现为上一轮写出的平面出现整块错值（实测：InitState 落盘的 dHBf 被 gk 载入竞争覆盖，
    // 导致首个 chunk 的 dV_pre 用到被污染的 dH 操作数，误差随 chunk 数放大）。
    __aicore__ inline void GuardScratchRewrite()
    {
        SetFlag<HardEvent::S_MTE2>(CDHP_EV_S_MTE2);
        WaitFlag<HardEvent::S_MTE2>(CDHP_EV_S_MTE2);
        SetFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        // 门控（g/gk）单独用 gDT_ 暂存，不再借用搬入暂存区，因此这里只需排空 V/MTE3 到 MTE2 的信用
        SetFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
    }

    // v6：V0 的单槽搬入（GM → 暂存槽）。槽号即 credit id：
    //   Wait MTE3_MTE2(slot)：等该槽上一轮搬出读完（首轮由 InitBuffer 预置）；
    //   Set   MTE2_V(evId) ：本轮搬入完成，供随后的 Cast 等待。
    // v10：搬入侧回到"双槽 + per-slot credit"（与 dH 常驻、P 常驻都验证过的那套一致）。
    // 注意 A5（dav-3510）上 **不能用 PipeBarrier / 紧邻 set+wait 的自排空** 来复用暂存区：
    // v6 的 5 槽批量写法与 arch22 那套"全排空"写法在 A5 上都会让 **P 面**出错
    // （短链 T=128 可复现，E 面正常——因为只有 K̄ 经 T1 只喂 P 链）。这里用 credit 版本。
    __aicore__ inline void LoadTileF32(const LocalTensor<float> &dst, GM_ADDR src, uint32_t rows, uint32_t cols,
                                       uint32_t rowStride)
    {
        AscendC::GlobalTensor<DT> gSrc;
        gSrc.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(src));
        const uint32_t slot = NextUbSlot();
        LocalTensor<DT> tmp = s0DT_.Get<DT>()[static_cast<uint32_t>(slot) * CDHP_SCRATCH_ELEMS];
        // 只等本槽上一轮 V 读取结束（搬入方向自平衡；首轮由 InitBuffer 预置）
        WaitFlag<HardEvent::V_MTE2>(slot);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(DT)),
                                          static_cast<uint32_t>((rowStride - cols) * sizeof(DT)), 0, 0};
        AscendC::DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(tmp, gSrc, params, padParams);
        SetFlag<HardEvent::MTE2_V>(slot);
        WaitFlag<HardEvent::MTE2_V>(slot);
        AscendC::Cast(dst, tmp, AscendC::RoundMode::CAST_NONE, rows * cols);
        SetFlag<HardEvent::V_MTE2>(slot);
    }

    // FP32 UB → GM [rows, cols]（模型 dtype，行距 rowStride）
    __aicore__ inline void StoreTileModel(const LocalTensor<float> &src, GM_ADDR dst, uint32_t rows, uint32_t cols,
                                          uint32_t rowStride)
    {
        AscendC::GlobalTensor<DT> gDst;
        gDst.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dst));
        const uint32_t slot = NextUbSlot();
        LocalTensor<DT> tmp = s1DT_.Get<DT>()[static_cast<uint32_t>(slot) * CDHP_SCRATCH_ELEMS];
        // 搬出方向自平衡：只等本槽上一笔 MTE3 读完（首轮由 InitBuffer 预置）
        WaitFlag<HardEvent::MTE3_V>(slot);
        AscendC::Cast(tmp, src, AscendC::RoundMode::CAST_RINT, rows * cols);
        SetFlag<HardEvent::V_MTE3>(slot);
        WaitFlag<HardEvent::V_MTE3>(slot);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(DT)), 0,
                                          static_cast<uint32_t>((rowStride - cols) * sizeof(DT)), 0};
        AscendC::DataCopyPad(gDst, tmp, params);
        SetFlag<HardEvent::MTE3_V>(slot);
    }

    __aicore__ inline void StoreModelNoPad(const LocalTensor<float> &src, GM_ADDR dst, uint32_t count)
    {
        AscendC::GlobalTensor<DT> gDst;
        gDst.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dst));
        const uint32_t slot = NextUbSlot();
        LocalTensor<DT> tmp = s1DT_.Get<DT>()[static_cast<uint32_t>(slot) * CDHP_SCRATCH_ELEMS];
        WaitFlag<HardEvent::MTE3_V>(slot);
        AscendC::Cast(tmp, src, AscendC::RoundMode::CAST_RINT, count);
        SetFlag<HardEvent::V_MTE3>(slot);
        WaitFlag<HardEvent::V_MTE3>(slot);
        // 统一用 DataCopyPad（与 slot 落盘同一条通路；DataCopy 在 UB→GM 上实测会写错内容）
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(DT)), 0, 0, 0};
        AscendC::DataCopyPad(gDst, tmp, params);
        SetFlag<HardEvent::MTE3_V>(slot);
    }

    // FP32 平面 [r0, r0+rows) 行 → UB（连续行距）
    __aicore__ inline void LoadPlaneF32(const LocalTensor<float> &dst, GM_ADDR src, uint32_t r0, uint32_t rows,
                                        uint32_t cols)
    {
        AscendC::GlobalTensor<float> gSrc;
        gSrc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src));
        SetFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(float)), 0,
                                          0, 0};
        AscendC::DataCopyPad(dst, gSrc[static_cast<uint64_t>(r0) * cols], params,
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
    }

    // 只下发、不立刻等 MTE2 完成：多个平面的搬运可以先背靠背排进 MTE2 队列，
    // 之后再统一 WaitPlaneLoad(evId)，避免"每笔搬运都把标量线程卡住"。
    __aicore__ inline void IssueLoadPlaneF32(const LocalTensor<float> &dst, GM_ADDR src, uint32_t r0, uint32_t rows,
                                             uint32_t cols, uint32_t evId)
    {
        AscendC::GlobalTensor<float> gSrc;
        gSrc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src));
        SetFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(float)), 0,
                                          0, 0};
        AscendC::DataCopyPad(dst, gSrc[static_cast<uint64_t>(r0) * cols], params,
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(evId);
    }

    __aicore__ inline void WaitPlaneLoad(uint32_t evId)
    {
        WaitFlag<HardEvent::MTE2_V>(evId);
    }

    // 只下发搬出（不 PipeBarrier 等搬出完成）；搬出完成由 SetFlag<MTE3_V>(evId) 通知，
    // 下一次复用这块 UB 前再用 WaitPlaneStore(evId) 等它。
    __aicore__ inline void IssueStorePlaneF32(const LocalTensor<float> &src, GM_ADDR dst, uint32_t r0, uint32_t rows,
                                              uint32_t cols, uint32_t evId)
    {
        AscendC::GlobalTensor<float> gDst;
        gDst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
        SetFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(float)), 0,
                                          0, 0};
        AscendC::DataCopyPad(gDst[static_cast<uint64_t>(r0) * cols], src, params);
        SetFlag<HardEvent::MTE3_V>(evId);
    }

    __aicore__ inline void WaitPlaneStore(uint32_t evId)
    {
        WaitFlag<HardEvent::MTE3_V>(evId);
    }

    __aicore__ inline void StorePlaneF32(const LocalTensor<float> &src, GM_ADDR dst, uint32_t r0, uint32_t rows,
                                         uint32_t cols)
    {
        AscendC::GlobalTensor<float> gDst;
        gDst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
        // 源数据可能来自向量操作或标量写（InitState），两条路径都要与 MTE3 建立顺序
        SetFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        SetFlag<HardEvent::S_MTE3>(CDHP_EV_S_V + 2);
        WaitFlag<HardEvent::S_MTE3>(CDHP_EV_S_V + 2);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(float)), 0,
                                          0, 0};
        AscendC::DataCopyPad(gDst[static_cast<uint64_t>(r0) * cols], src, params);
        PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void StoreScalarF32(const LocalTensor<float> &src, GM_ADDR dst, uint32_t count)
    {
        AscendC::GlobalTensor<float> gDst;
        gDst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
        SetFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(gDst, src, params);
        PipeBarrier<PIPE_MTE3>();
    }

     // V0：门控/衰减准备 + slot 落盘（Q̄s / W / -K̄ / do / -dv / decayK）
     // 契约：入口只消费本 chunk 的 q/k/w/do/dv 原始输入与 g/gk；出口保证 slot 内五个平面的
     //       无效行（[rows, chunkSize)）为 0，且 decayK[K] 已按 chunk 末行门控算好。
     // v3：衰减折进 K̄ 并按负号落盘（Cube 侧 T1 = Wᵀ@(-K̄) 即为 -T1），dv 落盘取负使 AB 全为正累加。
     // 注意：衰减不能折进 W——AB 的 Wᵀ@(-dv) 项用的是未衰减的 W，两者不能共用一份平面。
    __aicore__ inline void StageV0(uint32_t hv, uint32_t chunkIdx, uint32_t window)
    {
        const uint32_t rows = this->ChunkRows(chunkIdx);
        const uint32_t t0 = this->ChunkStart(chunkIdx);
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t mDim = static_cast<uint32_t>(this->tiling_.chunkSize);
        const uint32_t hk = this->HkOfHv(hv);
        // slot 按工作组区分（同一 chunk 下不同 head 各用各的 slot）
        // 1:2 下一个 block 的两个 AIV 各有自己的 slot 份，按 sliceIdx（blockIdx*aivPerBlock + subBlockIdx）选
        const uint32_t slot = this->SliceOfWindow(window);
        const uint32_t useG = static_cast<uint32_t>(this->tiling_.useGateG);
        const uint32_t useGk = static_cast<uint32_t>(this->tiling_.useGateGk);
        // 布局：Q̄s | W | -K̄ | do | -dv | decay（Q̄s/W 相邻、do/-dv 相邻，便于合并 GEMM）
        GM_ADDR slotQ = this->SlotAt(slot, 0);
        GM_ADDR slotW = this->SlotAt(slot, this->SlotQBytes());
        GM_ADDR slotK = this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes());
        GM_ADDR slotDo = this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes() + this->SlotKBytes());
        GM_ADDR slotNegDv = this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes() + this->SlotKBytes() +
                                                   this->SlotDoBytes());
        GM_ADDR slotDecay = this->SlotAt(slot, this->SlotDecayOffset());

        LocalTensor<float> gF = gF32_.Get<float>();
        LocalTensor<float> decayF = decayF32_.Get<float>();
        __ubuf__ float *gateFirst = nullptr;
        if (useG != 0) {
            GuardScratchRewrite();
            if constexpr (ChunkDeltaHBwdPreprocessSameType<GT, float>::value) {
                // GDN 允许 g 直接是 FP32：此时直接搬进 FP32 缓冲，不能再做 Cast（同为 FP32 的 Cast 不是
                // 有效的向量转换指令，会得到错误数值）。
                AscendC::GlobalTensor<float> gSrcF;
                gSrcF.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gAddr_));
                AscendC::DataCopyExtParams gParamsF{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(gF, gSrcF[hv * this->tiling_.T + t0], gParamsF,
                                     AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
            } else {
                AscendC::GlobalTensor<GT> gSrc;
                gSrc.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(gAddr_));
                LocalTensor<GT> gIn = gDT_.Get<GT>();
                AscendC::DataCopyExtParams gParams{1, static_cast<uint32_t>(rows * sizeof(GT)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<GT> gPad{false, 0, 0, 0};
                AscendC::DataCopyPad(gIn, gSrc[hv * this->tiling_.T + t0], gParams, gPad);
                SetFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
                WaitFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
                AscendC::Cast(gF, gIn, AscendC::RoundMode::CAST_NONE, rows);
            }
            SetFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
            PipeBarrier<PIPE_V>();
            // 门控行因子与衰减都下沉到 VF：g[rows-1] 的广播值只算一次
            gateFirst = UbPtr(gF);
            CDHP_V0LastDecayVF(UbPtr(decayF), gateFirst + (rows - 1), static_cast<uint16_t>(kDim));
        } else if (useGk != 0) {
            GuardScratchRewrite();
            AscendC::GlobalTensor<DT> gkSrc;
            gkSrc.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(gkAddr_));
            LocalTensor<DT> gkLast = gDT_.Get<DT>();
            AscendC::DataCopyExtParams gkParams{1, static_cast<uint32_t>(kDim * sizeof(DT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT> gkPad{false, 0, 0, 0};
            AscendC::DataCopyPad(gkLast, gkSrc[(hv * this->tiling_.T + t0 + rows - 1) * kDim], gkParams, gkPad);
            SetFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
            AscendC::Cast(decayF, gkLast, AscendC::RoundMode::CAST_NONE, kDim);
            PipeBarrier<PIPE_V>();
            AscendC::Muls(decayF, decayF, CDHP_LN2, kDim);
            PipeBarrier<PIPE_V>();
            AscendC::Exp(decayF, decayF, kDim);
            PipeBarrier<PIPE_V>();
        } else {
            Duplicate(decayF, 1.0f, kDim);
            PipeBarrier<PIPE_V>();
        }

        for (uint32_t r0 = 0; r0 < mDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, mDim - r0);
            const uint32_t validRows = (r0 < rows) ? MinV(tileRows, rows - r0) : 0;
            LocalTensor<float> qF = s0F32_.Get<float>();
            LocalTensor<float> kF = s1F32_.Get<float>();
            LocalTensor<float> wF = s2F32_.Get<float>();
            LocalTensor<float> doF = s3F32_.Get<float>();
            LocalTensor<float> dvF = s4F32_.Get<float>();
            // v13：满 tile 时 W / do 走同 dtype 纯拷贝（不做 FP32 往返）；尾块仍走 FP32 通路以便把无效行清 0
            const bool wdNeedF32 = (validRows < tileRows);
            Duplicate(qF, 0.0f, tileRows * kDim);
            Duplicate(kF, 0.0f, tileRows * kDim);
            if (wdNeedF32) {
                Duplicate(wF, 0.0f, tileRows * kDim);
                Duplicate(doF, 0.0f, tileRows * vDim);
            }
            Duplicate(dvF, 0.0f, tileRows * vDim);
            if (validRows > 0) {
                LoadTileF32(qF, qAddr_ + static_cast<uint64_t>(hk * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                            validRows, kDim, kDim);
                LoadTileF32(kF, kAddr_ + static_cast<uint64_t>(hk * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                            validRows, kDim, kDim);
                if (wdNeedF32) {
                    LoadTileF32(wF,
                                wAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                                validRows, kDim, kDim);
                    LoadTileF32(
                        doF, doAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * vDim * sizeof(DT),
                        validRows, vDim, vDim);
                } else {
                    CopyTileModel(wAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                                  slotW + static_cast<uint64_t>(r0) * kDim * sizeof(DT), validRows, kDim, kDim, kDim);
                    CopyTileModel(
                        doAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * vDim * sizeof(DT),
                        slotDo + static_cast<uint64_t>(r0) * vDim * sizeof(DT), validRows, vDim, vDim, vDim);
                }
                LoadTileF32(dvF, dvAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * vDim * sizeof(DT),
                            validRows, vDim, vDim);
                if (useG != 0) {
                    CDHP_V0GateScaleVF(UbPtr(qF), UbPtr(kF), gateFirst + r0, gateFirst + (rows - 1),
                                       static_cast<uint16_t>(validRows), static_cast<uint16_t>(kDim), scale_);
                } else {
                    AscendC::Muls(qF, qF, scale_, validRows * kDim);
                }
            }
            // v3：K̄ 落盘取负（Cube 侧 T1 = Wᵀ@(-K̄) 即为 -T1），dv 落盘取负（AB 直接正累加）
            AscendC::Muls(kF, kF, -1.0f, tileRows * kDim);
            AscendC::Muls(dvF, dvF, -1.0f, tileRows * vDim);
            StoreTileModel(qF, slotQ + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim, kDim);
            StoreTileModel(kF, slotK + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim, kDim);
            if (wdNeedF32) {
                StoreTileModel(wF, slotW + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim, kDim);
                StoreTileModel(doF, slotDo + static_cast<uint64_t>(r0) * vDim * sizeof(DT), tileRows, vDim, vDim);
            }
            StoreTileModel(dvF, slotNegDv + static_cast<uint64_t>(r0) * vDim * sizeof(DT), tileRows, vDim, vDim);
        }
        StoreScalarF32(decayF, slotDecay, kDim);
    }

    // 初始状态：dH = 0、P = I 落到对应 parity 的 FP32 平面
    __aicore__ inline void InitState(uint32_t chunkNum)
    {
        // 跨 task 复用 UB 的 WAR：上一个 task 的最后一次搬出（MTE3，可能是 WriteOutput 的输出写回）
        // 可能仍在读 s0F32_/s1F32_。`PipeBarrier<PIPE_MTE3>` 只约束 MTE3 自己，拦不住随后的 V，
        // 若不等这一步，本轮 Duplicate 会把「还在搬出中的那块 UB」改写，
        // 输出的最后一个 tile 会写成当前 UB 里的内容（实测表现为中间 task 的 P 面尾部变成单位阵）。
        SetFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        SetFlag<HardEvent::MTE3_S>(CDHP_EV_MTE3_S);
        WaitFlag<HardEvent::MTE3_S>(CDHP_EV_MTE3_S);
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        // v5：dH 的状态常驻 UB（dhStateF32_），这里只把它清零（初值 0）；
        // v5：dH 的状态常驻 UB（dhStateF32_），这里只把它清零（初值 0）；
        // P 的初值（单位阵）在首 chunk 的 StageStateStore 里内联生成并落 GM 平面。
        LocalTensor<float> dhState = dhStateF32_.Get<float>();
        Duplicate(dhState, 0.0f, kDim * vDim);
        PipeBarrier<PIPE_V>();
    }

    // v5 链上准备：把"本 chunk 之前的"状态写成 Cube 需要的模型 dtype 操作数，并放行 Cube 的 Z/ZP。
    //   dH：直接由 UB 常驻状态转 bf16 落盘（首 chunk 即全零初值）
    //   P ：首 chunk 用内联单位阵；之后从 GM 读上一轮 P 再转 bf16（P 仍走 GM scratch）
    __aicore__ inline void StageStateStore(uint32_t window, bool isFirstChunk)
    {
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t prevWin = (window + 1u) & 1u;
        LocalTensor<float> dhState = dhStateF32_.Get<float>();
        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            // dH 操作数
            StoreTileModel(dhState[r0 * vDim],
                           this->DhBfAt(prevWin) + static_cast<uint64_t>(r0) * vDim * sizeof(DT), tileRows, vDim,
                           vDim);
            if (isFirstChunk) {
                // 初值 P = I：本 chunk 的 Cube 要读的 P 操作数就是这个单位阵。
                // 注意这里只写 PBf（模型 dtype）；fp32 的 P_prev 由 StageState 首 chunk 内联生成，
                // 因此 PAt 的 fp32 平面只在 StageState 里写、且只被"下一轮 StageState"读一次 ⇒
                // 每个 tile 恰好一 set 一 wait，可以安全地用单比特 event 做 credit。
                LocalTensor<float> pF = s1F32_.Get<float>();
                CDHP_InitIdentityVF(UbPtr(pF), static_cast<uint16_t>(r0), static_cast<uint16_t>(tileRows),
                                    static_cast<uint16_t>(kDim));
                StoreTileModel(pF, this->PBfAt(prevWin) + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows,
                               kDim, kDim);
            }
        }
        // 操作数已落盘：通知 Cube 可以开始本 chunk 的 Z/ZP
        CDHP_AIV_SET(CdhpFlag(CDHP_FLAG_STATE_READY, prevWin));
    }

    // v3 状态链（向量侧）：一次做完两条独立链
    //   dH_new = decayK ⊙ dH_prev + AB + Z      （Z  = (-T1)@dH_prev）
    //   P_new  = decayK ⊙ P_prev  + ZP          （ZP = (-T1)@P_prev）
    // AB/Z/ZP 都由 Cube 产出；本 Stage 只做逐元素累加与 dtype 副本落盘。
    __aicore__ inline void StageState(uint32_t chunkIdx, uint32_t window, bool isFirstChunk)
    {
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        LocalTensor<float> decayF = decayF32_.Get<float>();
        // V0 领先，UB 里的 decayF32_ 存活期不覆盖本 Stage → 从本 window 的 slot 重新载入 decayK
        LoadPlaneF32(decayF, this->SlotAtWindow(window, this->SlotDecayOffset()), 0, 1, kDim);
        SetFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        PipeBarrier<PIPE_V>();
        // v12：E 链与 P 链**按 tile 合并成一趟**，并把三条链上平面（AB / Z / ZP）一次性排进 MTE2：
        //   * 改造前：先把 AB/Z 两个平面按 32 行 tile 成对下发、等到齐后做 dH 的 VF；P 链的 ZP 是
        //     另一个循环，它的搬运完全排在 dH 的 VF 之后 ⇒ P 链的 ZP 载入与 dH 的 VF 串行、无法重叠；
        //   * 改造后：每个 tile 先下发 3 笔（AB/Z/ZP）再统一等，两个 VF 共用同一份已到齐的操作数，
        //     MTE2 队列深度从 2 提到 3，且 P 链不再等前一个循环收尾。
        // 第三个 fp32 缓冲复用 V0 的 do 平面（s3F32_）：StageState 与 StageV0 在同一轮里**串行**，
        // 且 StageV0 排在 StageState 之后，因此这块 UB 在 StageState 期间是空闲的，不需要新增 UB。
        // 三条链的常驻状态（dH / P fp32）都在 UB 里，模型 dtype 操作数在 StageStateStore / 本 Stage
        // 统一落盘；首 chunk 的 P_prev = I 由 CDHP_InitIdentityVF 内联生成。
        LocalTensor<float> dhState = dhStateF32_.Get<float>();
        LocalTensor<float> pState = pStateF32_.Get<float>();
        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            LocalTensor<float> abF = s0F32_.Get<float>();
            LocalTensor<float> zF = s1F32_.Get<float>();
            LocalTensor<float> zpF = s3F32_.Get<float>();
            IssueLoadPlaneF32(abF, this->AbAt(window), r0, tileRows, vDim, CDHP_EV_LOAD1);
            IssueLoadPlaneF32(zF, this->ZAt(window), r0, tileRows, vDim, CDHP_EV_LOAD2);
            IssueLoadPlaneF32(zpF, this->ZpAt(window), r0, tileRows, kDim, CDHP_EV_LOAD3);
            WaitPlaneLoad(CDHP_EV_LOAD1);
            WaitPlaneLoad(CDHP_EV_LOAD2);
            WaitPlaneLoad(CDHP_EV_LOAD3);
            // dH_new = decayK ⊙ dH_old + AB + Z：一份寄存器流一趟算完。
            CDHP_V4StateUpdateVF(UbPtr(dhState) + static_cast<uint64_t>(r0) * vDim, UbPtr(abF), UbPtr(zF),
                                 UbPtr(decayF), static_cast<uint16_t>(r0), static_cast<uint16_t>(tileRows),
                                 static_cast<uint16_t>(vDim));
            // P_new = decayK ⊙ P_old + ZP；首 chunk 的 P_old = I 内联生成
            if (isFirstChunk) {
                CDHP_InitIdentityVF(UbPtr(pState) + static_cast<uint64_t>(r0) * kDim,
                                    static_cast<uint16_t>(r0), static_cast<uint16_t>(tileRows),
                                    static_cast<uint16_t>(kDim));
            }
            CDHP_V3AccumVF(UbPtr(pState) + static_cast<uint64_t>(r0) * kDim, UbPtr(zpF), UbPtr(decayF),
                           static_cast<uint16_t>(r0), static_cast<uint16_t>(tileRows),
                           static_cast<uint16_t>(kDim));
            StoreTileModel(pState[r0 * kDim],
                           this->PBfAt(window) + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim,
                           kDim);
        }
    }

    // 末 chunk 之后把 dH 与 P 写进 dhm[hv, K, V+K]
    __aicore__ inline void WriteOutput(uint32_t hv)
    {
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t rowStride = vDim + kDim;
        // E 面与 P 面：末值都在 UB 常驻状态（dhStateF32_ / pStateF32_）里，不必再经 GM workspace 往返。
        // 源由 V（VF 累加）写，搬出前建立 V→MTE3 顺序。
        LocalTensor<float> dhState = dhStateF32_.Get<float>();
        LocalTensor<float> pState = pStateF32_.Get<float>();
        SetFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            LocalTensor<float> f = dhState[static_cast<uint32_t>(r0) * vDim];
            AscendC::DataCopyExtParams dhParams{static_cast<uint16_t>(tileRows),
                                                static_cast<uint32_t>(vDim * sizeof(float)), 0,
                                                static_cast<uint32_t>((rowStride - vDim) * sizeof(float)), 0};
            AscendC::DataCopyPad(dhmGm_[static_cast<uint64_t>(hv * kDim + r0) * rowStride], f, dhParams);
            PipeBarrier<PIPE_MTE3>();
            LocalTensor<float> fp = pState[static_cast<uint32_t>(r0) * kDim];
            AscendC::DataCopyExtParams pParams{static_cast<uint16_t>(tileRows),
                                               static_cast<uint32_t>(kDim * sizeof(float)), 0,
                                               static_cast<uint32_t>((rowStride - kDim) * sizeof(float)), 0};
            AscendC::DataCopyPad(dhmGm_[static_cast<uint64_t>(hv * kDim + r0) * rowStride + vDim], fp, pParams);
            PipeBarrier<PIPE_MTE3>();
        }
    }

    AscendC::TPipe *pipe_ = nullptr;
    uint32_t ubSlot_ = 0;   // model dtype 暂存区双槽轮转（槽号同时用作该槽的事件 id）
    AscendC::TBuf<AscendC::TPosition::VECCALC> s0F32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s1F32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s2F32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s3F32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s4F32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s0DT_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s1DT_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s2DT_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gDT_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decayF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dhStateF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pStateF32_;
    AscendC::GlobalTensor<float> dhmGm_;
    GM_ADDR qAddr_ = nullptr;
    GM_ADDR kAddr_ = nullptr;
    GM_ADDR wAddr_ = nullptr;
    GM_ADDR doAddr_ = nullptr;
    GM_ADDR dvAddr_ = nullptr;
    GM_ADDR gAddr_ = nullptr;
    GM_ADDR gkAddr_ = nullptr;
    float scale_ = 1.0f;
};

} // namespace CP

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_ARCH35_VECTOR_H
