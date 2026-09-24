/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess_vec.h
 * \brief Vector 侧 Stage：V0（门控与衰减准备 / 零填充）、V2（dV̂'）、V4（状态更新 + 对角注入）。
 *
 * V0 每个 chunk 产出 slot：Q̄s、K̄、W（零填充）、do（零填充）、decayK。
 *   [M, BT) 的无效 token 行必须在 Q̄s / K̄ / W / do 上写零：Cube 侧矩阵乘按完整 chunkSize 规约，
 *   这也复现了上游 tl.load(boundary_check) 的零填充语义。
 * V2  dV̂' = -(dV_pre + dv_local)，落模型 dtype 供 C3 当右操作数（无效行写零）。
 * V4  路 A：dH_new = decayK ⊙ dH_old + (qterm + wterm)，并落模型 dtype 拷贝供下一轮 C1；
 *     路 B：P_c = diag(decayK) - T1（模型 dtype），并把上一轮 P（FP32）转模型 dtype 供 C5。
 *
 * 同 chunk 内 AIV/AIC 严格交替握手，flag id 见 chunk_delta_h_bwd_preprocess_policy.h。
 */

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_VEC_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_VEC_H

#include "kernel_operator.h"
#include "chunk_delta_h_bwd_preprocess_base.h"
#include "chunk_delta_h_bwd_preprocess_policy.h"

using namespace AscendC;

namespace CP {

constexpr float CDHP_LN2 = 0.6931471805599453f;
// 向量侧行分块：16 行 × 256 列 = 4096 元素（FP32 16 KiB）；K/V <= 256 时 UB 占用固定
constexpr uint32_t CDHP_VEC_TILE = 16;
constexpr uint32_t CDHP_SCRATCH_ELEMS = CDHP_VEC_TILE * 256;
constexpr uint32_t CDHP_EV_V_S = 0;
constexpr uint32_t CDHP_EV_S_V = 1;
constexpr uint32_t CDHP_EV_MTE2_V = 2;
constexpr uint32_t CDHP_EV_V_MTE2 = 3;
constexpr uint32_t CDHP_EV_V_MTE3 = 4;
constexpr uint32_t CDHP_EV_MTE3_V = 5;
constexpr uint32_t CDHP_EV_MTE3_MTE2 = 6;
// GM→UB 的载入（MTE2）与 UB→GM 的搬出（MTE3）之间的顺序：搬出的源就是刚载入的同一块 UB，
// 关闭自动同步后必须显式建立 MTE2→MTE3，否则搬出会读到上一次留在 UB 里的旧内容。
constexpr uint32_t CDHP_EV_MTE2_MTE3 = 7;
// 复用 s0DT_ 暂存区前的保护事件（与上面的 id 数值可以相同：不同 HardEvent 对是不同的硬件事件）
constexpr uint32_t CDHP_EV_S_MTE2 = 0;
// 复用 UB 前等 MTE3 读完的 MTE3→S 事件（与上面的 id 数值可以相同：不同 HardEvent 对是不同硬件事件）
constexpr uint32_t CDHP_EV_MTE3_S = 1;

// 编译期判断两个类型是否相同（避免依赖 <type_traits> 在 kernel 侧的可用性）
template <typename A, typename B>
struct ChunkDeltaHBwdPreSameType {
    static constexpr bool value = false;
};
template <typename A>
struct ChunkDeltaHBwdPreSameType<A, A> {
    static constexpr bool value = true;
};

__aicore__ inline uint32_t MinV(uint32_t a, uint32_t b)
{
    return (a < b) ? a : b;
}

template <typename DT, typename GT>
class ChunkDeltaHBwdPreVec : public ChunkDeltaHBwdPreBase<DT, GT> {
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
        pipe_->InitBuffer(s0DT_, CDHP_SCRATCH_ELEMS * sizeof(DT));
        pipe_->InitBuffer(gF32_, 256 * sizeof(float));
        pipe_->InitBuffer(decayF32_, 256 * sizeof(float));
        pipe_->InitBuffer(scalarF32_, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t chunkNum = this->ChunkNum(this->Bos(), this->Eos());
        const ChunkDeltaHBwdPreTaskRange range = this->ResolveTaskRange();
        for (uint32_t i = 0; i < range.taskCount; ++i) {
            const uint32_t hv = range.taskBegin + i;
            InitState(chunkNum);
            for (uint32_t c = 0; c < chunkNum; ++c) {
                const uint32_t chunkIdx = chunkNum - 1 - c;
                StageV0(hv, chunkIdx);
                CDHP_AIV_SET(CDHP_FLAG_V0_DONE);
                CDHP_AIV_WAIT(CDHP_FLAG_C1_DONE);
                StageV2(hv, chunkIdx);
                CDHP_AIV_SET(CDHP_FLAG_V2_DONE);
                CDHP_AIV_WAIT(CDHP_FLAG_C3_DONE);
                StageV4(chunkIdx);
                CDHP_AIV_SET(CDHP_FLAG_V4_DONE);
                CDHP_AIV_WAIT(CDHP_FLAG_C5_DONE);
            }
            WriteOutput(hv);
        }
    }

private:
    // 复用 s0DT_ 暂存区（以及门控用的 gF）前的保护：该缓冲可能刚被上一轮的
    //   - V（Cast 写 s0DT_ / Cast 写 gF）
    //   - MTE3（StoreTileModel 从 s0DT_ 搬出）
    //   - S（标量读 gF/decayF）
    // 使用过。关闭自动同步后必须显式等待，否则新的一轮 MTE2 载入会在上一轮还没读完时覆盖内容，
    // 表现为上一轮写出的平面出现整块错值（实测：InitState 落盘的 dHBf 被 gk 载入竞争覆盖，
    // 导致首个 chunk 的 dV_pre 用到被污染的 dH 操作数，误差随 chunk 数放大）。
    __aicore__ inline void GuardScratchRewrite()
    {
        SetFlag<HardEvent::S_MTE2>(CDHP_EV_S_MTE2);
        WaitFlag<HardEvent::S_MTE2>(CDHP_EV_S_MTE2);
        SetFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
    }

    __aicore__ inline float ExpScalar(float x)
    {
        LocalTensor<float> scalar = scalarF32_.Get<float>();
        Duplicate(scalar, x, 1);
        PipeBarrier<PIPE_V>();
        Exp(scalar, scalar, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(CDHP_EV_V_S);
        WaitFlag<HardEvent::V_S>(CDHP_EV_V_S);
        return scalar.GetValue(0);
    }

    // GM [rows, cols]（行距 rowStride）→ FP32 UB
    __aicore__ inline void LoadTileF32(const LocalTensor<float> &dst, GM_ADDR src, uint32_t rows, uint32_t cols,
                                       uint32_t rowStride)
    {
        AscendC::GlobalTensor<DT> gSrc;
        gSrc.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(src));
        LocalTensor<DT> tmp = s0DT_.Get<DT>();
        // 复用同一 UB 暂存区前，先等上一轮读取它的向量操作与 MTE3 搬出结束（本仓编译关闭了自动同步）
        SetFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        // 自屏障：先等本核先前 MTE3 搬出排空，再复用暂存区
        SetFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(DT)),
                                          static_cast<uint32_t>((rowStride - cols) * sizeof(DT)), 0, 0};
        AscendC::DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(tmp, gSrc, params, padParams);
        SetFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        AscendC::Cast(dst, tmp, AscendC::RoundMode::CAST_NONE, rows * cols);
        PipeBarrier<PIPE_V>();
    }

    // FP32 UB → GM [rows, cols]（模型 dtype，行距 rowStride）
    __aicore__ inline void StoreTileModel(const LocalTensor<float> &src, GM_ADDR dst, uint32_t rows, uint32_t cols,
                                          uint32_t rowStride)
    {
        AscendC::GlobalTensor<DT> gDst;
        gDst.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dst));
        LocalTensor<DT> tmp = s0DT_.Get<DT>();
        // 自屏障：先等本核先前 MTE3 搬出排空，本次 V(Cast) 才能写暂存区
        SetFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        AscendC::Cast(tmp, src, AscendC::RoundMode::CAST_RINT, rows * cols);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(DT)), 0,
                                          static_cast<uint32_t>((rowStride - cols) * sizeof(DT)), 0};
        AscendC::DataCopyPad(gDst, tmp, params);
        PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void StoreModelNoPad(const LocalTensor<float> &src, GM_ADDR dst, uint32_t count)
    {
        AscendC::GlobalTensor<DT> gDst;
        gDst.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dst));
        LocalTensor<DT> tmp = s0DT_.Get<DT>();
        SetFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        AscendC::Cast(tmp, src, AscendC::RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        // 统一用 DataCopyPad（与 slot 落盘同一条通路；DataCopy 在 UB→GM 上实测会写错内容）
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(DT)), 0, 0, 0};
        AscendC::DataCopyPad(gDst, tmp, params);
        PipeBarrier<PIPE_MTE3>();
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

    // V0：门控/衰减准备 + slot 落盘（Q̄s / K̄ / W / do / decayK）
    __aicore__ inline void StageV0(uint32_t hv, uint32_t chunkIdx)
    {
        const uint32_t rows = this->ChunkRows(chunkIdx);
        const uint32_t t0 = this->ChunkStart(chunkIdx);
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t mDim = static_cast<uint32_t>(this->tiling_.chunkSize);
        const uint32_t hk = this->HkOfHv(hv);
        // slot 按工作组区分（同一 chunk 下不同 head 各用各的 slot）
        const uint32_t slot = this->BlockIdx();
        const uint32_t useG = static_cast<uint32_t>(this->tiling_.useGateG);
        const uint32_t useGk = static_cast<uint32_t>(this->tiling_.useGateGk);
        GM_ADDR slotQ = this->SlotAt(slot, 0);
        GM_ADDR slotK = this->SlotAt(slot, this->SlotQBytes());
        GM_ADDR slotW = this->SlotAt(slot, this->SlotQBytes() + this->SlotKBytes());
        GM_ADDR slotDo = this->SlotAt(slot, this->SlotQBytes() + this->SlotKBytes() + this->SlotWBytes());
        GM_ADDR slotDecay = this->SlotAt(slot, this->SlotDecayOffset());

        LocalTensor<float> gF = gF32_.Get<float>();
        LocalTensor<float> decayF = decayF32_.Get<float>();
        if (useG != 0) {
            GuardScratchRewrite();
            if constexpr (ChunkDeltaHBwdPreSameType<GT, float>::value) {
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
                LocalTensor<GT> gIn = s0DT_.Get<GT>();
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
            SetFlag<HardEvent::V_S>(CDHP_EV_V_S);
            WaitFlag<HardEvent::V_S>(CDHP_EV_V_S);
            const float eLast = ExpScalar(gF.GetValue(rows - 1) * CDHP_LN2);
            Duplicate(decayF, eLast, kDim);
            PipeBarrier<PIPE_V>();
        } else if (useGk != 0) {
            GuardScratchRewrite();
            AscendC::GlobalTensor<DT> gkSrc;
            gkSrc.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(gkAddr_));
            LocalTensor<DT> gkLast = s0DT_.Get<DT>();
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
            SetFlag<HardEvent::V_S>(CDHP_EV_V_S);
            WaitFlag<HardEvent::V_S>(CDHP_EV_V_S);
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
            Duplicate(qF, 0.0f, tileRows * kDim);
            Duplicate(kF, 0.0f, tileRows * kDim);
            Duplicate(wF, 0.0f, tileRows * kDim);
            Duplicate(doF, 0.0f, tileRows * vDim);
            PipeBarrier<PIPE_V>();
            if (validRows > 0) {
                LoadTileF32(qF, qAddr_ + static_cast<uint64_t>(hk * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                            validRows, kDim, kDim);
                LoadTileF32(kF, kAddr_ + static_cast<uint64_t>(hk * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                            validRows, kDim, kDim);
                LoadTileF32(wF, wAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                            validRows, kDim, kDim);
                LoadTileF32(doF, doAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * vDim * sizeof(DT),
                            validRows, vDim, vDim);
                if (useG != 0) {
                    const float gLast = gF.GetValue(rows - 1);
                    for (uint32_t r = 0; r < validRows; ++r) {
                        const float ev = ExpScalar(gF.GetValue(r0 + r) * CDHP_LN2);
                        AscendC::Muls(qF[r * kDim], qF[r * kDim], ev * scale_, kDim);
                        const float dv = ExpScalar((gLast - gF.GetValue(r0 + r)) * CDHP_LN2);
                        AscendC::Muls(kF[r * kDim], kF[r * kDim], dv, kDim);
                    }
                    PipeBarrier<PIPE_V>();
                } else {
                    AscendC::Muls(qF, qF, scale_, validRows * kDim);
                    PipeBarrier<PIPE_V>();
                }
            }
            StoreTileModel(qF, slotQ + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim, kDim);
            StoreTileModel(kF, slotK + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim, kDim);
            StoreTileModel(wF, slotW + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim, kDim);
            StoreTileModel(doF, slotDo + static_cast<uint64_t>(r0) * vDim * sizeof(DT), tileRows, vDim, vDim);
        }
        StoreScalarF32(decayF, slotDecay, kDim);
    }

    // V2：dV̂' = -(dV_pre + dv_local)（无效行写零）
    __aicore__ inline void StageV2(uint32_t hv, uint32_t chunkIdx)
    {
        const uint32_t rows = this->ChunkRows(chunkIdx);
        const uint32_t t0 = this->ChunkStart(chunkIdx);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t mDim = static_cast<uint32_t>(this->tiling_.chunkSize);
        for (uint32_t r0 = 0; r0 < mDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, mDim - r0);
            const uint32_t validRows = (r0 < rows) ? MinV(tileRows, rows - r0) : 0;
            LocalTensor<float> preF = s0F32_.Get<float>();
            LocalTensor<float> dvF = s1F32_.Get<float>();
            Duplicate(preF, 0.0f, tileRows * vDim);
            Duplicate(dvF, 0.0f, tileRows * vDim);
            PipeBarrier<PIPE_V>();
            if (validRows > 0) {
                LoadPlaneF32(preF, this->DvPreAt(), r0, validRows, vDim);
                // 走统一的 LoadTileF32（内部含 MTE2→V 与暂存区复用同步）
                LoadTileF32(dvF, dvAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * vDim * sizeof(DT),
                            validRows, vDim, vDim);
                AscendC::Add(preF, preF, dvF, validRows * vDim);
                PipeBarrier<PIPE_V>();
                AscendC::Muls(preF, preF, -1.0f, validRows * vDim);
                PipeBarrier<PIPE_V>();
            }
            StoreTileModel(preF, this->DvHatAt() + static_cast<uint64_t>(r0) * vDim * sizeof(DT), tileRows, vDim,
                           vDim);
        }
    }

    // 初始状态：dH = 0、P = I 落到对应 parity 的 FP32 平面
    __aicore__ inline void InitState(uint32_t chunkNum)
    {
        // 跨 task 复用 UB 的 WAR：上一个 task 的最后一次搬出（MTE3，可能是 WriteOutput 的输出写回）
        // 可能仍在读 s0F32_/s1F32_。`PipeBarrier<PIPE_MTE3>` 只约束 MTE3 自己，拦不住随后的 V/S，
        // 若不等这一步，本轮 Duplicate/SetValue 会把「还在搬出中的那块 UB」改写，
        // 输出的最后一个 tile 会写成当前 UB 里的内容（实测表现为中间 task 的 P 面尾部变成单位阵）。
        SetFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        SetFlag<HardEvent::MTE3_S>(CDHP_EV_MTE3_S);
        WaitFlag<HardEvent::MTE3_S>(CDHP_EV_MTE3_S);
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t parity = chunkNum % 2;
        LocalTensor<float> dhF = s0F32_.Get<float>();
        LocalTensor<float> pF = s1F32_.Get<float>();
        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            Duplicate(dhF, 0.0f, tileRows * vDim);
            Duplicate(pF, 0.0f, tileRows * kDim);
            PipeBarrier<PIPE_V>();
            // V 先写（Duplicate 清零）→ S 再写（对角注入）：关闭自动同步后必须先等 V 落盘，
            // 否则标量写的 1.0 会被随后的向量清零覆盖（表现为 P 初值全零）。
            SetFlag<HardEvent::V_S>(CDHP_EV_V_S);
            WaitFlag<HardEvent::V_S>(CDHP_EV_V_S);
            for (uint32_t r = 0; r < tileRows; ++r) {
                pF.SetValue(r * kDim + (r0 + r), 1.0f);
            }
            SetFlag<HardEvent::S_V>(CDHP_EV_S_V);
            WaitFlag<HardEvent::S_V>(CDHP_EV_S_V);
            StorePlaneF32(dhF, this->DhAt(parity), r0, tileRows, vDim);
            StorePlaneF32(pF, this->PAt(parity), r0, tileRows, kDim);
            // C1 真正消费的是模型 dtype 的 dH 操作数平面（首轮必须为零，否则读到未初始化的 workspace）
            StoreTileModel(dhF, this->DhBfAt() + static_cast<uint64_t>(r0) * vDim * sizeof(DT),
                           tileRows, vDim, vDim);
        }
    }

    // V4：路 A（dH 更新 + dHBf）+ 路 B（P_c + 上一轮 P 的 dtype 拷贝）
    __aicore__ inline void StageV4(uint32_t chunkIdx)
    {
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t dhParity = chunkIdx % 2;
        const uint32_t dhPrev = (chunkIdx + 1) % 2;
        LocalTensor<float> decayF = decayF32_.Get<float>();
        SetFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        SetFlag<HardEvent::V_S>(CDHP_EV_V_S);
        WaitFlag<HardEvent::V_S>(CDHP_EV_V_S);

        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            LocalTensor<float> dhF = s0F32_.Get<float>();
            LocalTensor<float> qF = s1F32_.Get<float>();
            LocalTensor<float> wF = s2F32_.Get<float>();
            LoadPlaneF32(dhF, this->DhAt(dhPrev), r0, tileRows, vDim);
            LoadPlaneF32(qF, this->QtermAt(), r0, tileRows, vDim);
            LoadPlaneF32(wF, this->WtermAt(), r0, tileRows, vDim);
            AscendC::Add(qF, qF, wF, tileRows * vDim);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < tileRows; ++r) {
                AscendC::Muls(dhF[r * vDim], dhF[r * vDim], decayF.GetValue(r0 + r), vDim);
            }
            PipeBarrier<PIPE_V>();
            AscendC::Add(dhF, dhF, qF, tileRows * vDim);
            PipeBarrier<PIPE_V>();
            StorePlaneF32(dhF, this->DhAt(dhParity), r0, tileRows, vDim);
            StoreTileModel(dhF, this->DhBfAt() + static_cast<uint64_t>(r0) * vDim * sizeof(DT), tileRows, vDim,
                           vDim);
        }

        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            LocalTensor<float> t1F = s0F32_.Get<float>();
            LoadPlaneF32(t1F, this->T1At(), r0, tileRows, kDim);
            AscendC::Muls(t1F, t1F, -1.0f, tileRows * kDim);
            PipeBarrier<PIPE_V>();
            // 对角注入必须用标量读写：UB 上的向量指令要求 32B 对齐，而 [i,i] 的偏移不满足该规律
            // （此前用 Adds(..., 1) 会触发 "The address for VEC to access UB is not aligned"）。
            SetFlag<HardEvent::V_S>(CDHP_EV_V_S);
            WaitFlag<HardEvent::V_S>(CDHP_EV_V_S);
            for (uint32_t r = 0; r < tileRows; ++r) {
                const uint32_t idx = r * kDim + (r0 + r);
                t1F.SetValue(idx, t1F.GetValue(idx) + decayF.GetValue(r0 + r));
            }
            SetFlag<HardEvent::S_V>(CDHP_EV_S_V);
            WaitFlag<HardEvent::S_V>(CDHP_EV_S_V);
            StoreTileModel(t1F, this->PcAt() + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim, kDim);
        }

        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            LocalTensor<float> pF = s0F32_.Get<float>();
            // 读上一轮 C5 产出的 P（parity = (chunkIdx + 1) % 2），不是本轮写入的那个面
            LoadPlaneF32(pF, this->PAt(dhPrev), r0, tileRows, kDim);
            StoreTileModel(pF, this->PBfAt() + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim, kDim);
        }
    }

    // 末 chunk 之后把 dH 与 P 写进 dhm[hv, K, V+K]
    __aicore__ inline void WriteOutput(uint32_t hv)
    {
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t rowStride = vDim + kDim;
        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            LocalTensor<float> f = s0F32_.Get<float>();
            LoadPlaneF32(f, this->DhAt(0), r0, tileRows, vDim);
            // 源 UB 刚由 MTE2 写入：搬出前必须等 MTE2 完成
            SetFlag<HardEvent::MTE2_MTE3>(CDHP_EV_MTE2_MTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(CDHP_EV_MTE2_MTE3);
            AscendC::DataCopyExtParams dhParams{static_cast<uint16_t>(tileRows),
                                                static_cast<uint32_t>(vDim * sizeof(float)), 0,
                                                static_cast<uint32_t>((rowStride - vDim) * sizeof(float)), 0};
            AscendC::DataCopyPad(dhmGm_[static_cast<uint64_t>(hv * kDim + r0) * rowStride], f, dhParams);
            PipeBarrier<PIPE_MTE3>();
            // P 面与 E 面分别用独立 UB 缓冲：连续两次 MTE2 落到同一块 UB 时，实测第二块
            // （P 面最后一个 tile）会被搬出旧值/零值，改用独立缓冲后稳定正确。
            LocalTensor<float> fp = s1F32_.Get<float>();
            LoadPlaneF32(fp, this->PAt(0), r0, tileRows, kDim);
            SetFlag<HardEvent::MTE2_MTE3>(CDHP_EV_MTE2_MTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(CDHP_EV_MTE2_MTE3);
            AscendC::DataCopyExtParams pParams{static_cast<uint16_t>(tileRows),
                                               static_cast<uint32_t>(kDim * sizeof(float)), 0,
                                               static_cast<uint32_t>((rowStride - kDim) * sizeof(float)), 0};
            AscendC::DataCopyPad(dhmGm_[static_cast<uint64_t>(hv * kDim + r0) * rowStride + vDim], fp, pParams);
            PipeBarrier<PIPE_MTE3>();
        }
    }

    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s0F32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s1F32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s2F32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s3F32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> s0DT_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decayF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarF32_;
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

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_VEC_H
