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

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_ARCH22_VECTOR_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_ARCH22_VECTOR_H

#include "kernel_operator.h"
#include "../chunk_delta_h_bwd_preprocess_common.h"
#include "../chunk_delta_h_bwd_preprocess_policy.h"

using namespace AscendC;

namespace CP {

constexpr float CDHP_LN2 = 0.6931471805599453f;
// 向量侧行分块：v21 起 32 行 × 128 列 = 4096 元素（FP32 16 KiB）。
// 行数由 16 提到 32 的理由：K=V=128、chunk_size=64 下，一个 chunk（64 行）与一条 dH/P 状态
// （128 行）的 tile 数各减半 ⇒ 每 chunk 的向量 API 调用次数与配套 SetFlag/WaitFlag 次数减半。
// A2 的 AIV 一直是"调用次数受限"（见 design.md §24：scalar 33%、mte2/mte3 只有 ~25 GB/s，
// 即每笔搬运/每条指令都在同步等待），因此这是 A2 侧最直接的杠杆。
// 元素总数保持 4096 不变（UB 占用不变）：本算子只支持 K=V=128，单 tile 最大就是 32×128。
constexpr uint32_t CDHP_VEC_TILE = 32;
constexpr uint32_t CDHP_SCRATCH_ELEMS = CDHP_VEC_TILE * 128;
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
        pipe_->InitBuffer(s0DT_, CDHP_SCRATCH_ELEMS * sizeof(DT));
        pipe_->InitBuffer(gF32_, 256 * sizeof(float));
        pipe_->InitBuffer(decayF32_, 256 * sizeof(float));
        pipe_->InitBuffer(scalarF32_, 8 * sizeof(float));
        // 门控行因子（qFac = exp2(g)·scale，kFac = -exp2(g_last-g)），每 chunk 算一次
        pipe_->InitBuffer(qFacF32_, 256 * sizeof(float));
        pipe_->InitBuffer(kFacF32_, 256 * sizeof(float));
        // v22：逐行因子的 Brcb 展开块（每行一个 32B block = 8 个 float），见 ScaleRowsByFactor
        pipe_->InitBuffer(facBlkF32_, 256 * sizeof(float));
        // v5（参考 ChunkFwdH 的 rolling state）：dH 的状态常驻 UB（K=V=128 → 64 KiB），
        // 每 chunk 只把模型 dtype 操作数写回 GM，不再每 chunk 经 GM workspace 读写 FP32 状态。
        pipe_->InitBuffer(dhStateF32_, 128u * 128u * sizeof(float));
        // v15：P 的 FP32 状态同样常驻 UB（K=K=128 → 64 KiB），不再走 PAt 平面往返
        pipe_->InitBuffer(pStateF32_, 128u * 128u * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t chunkNum = this->ChunkNum(this->Bos(), this->Eos());
        const ChunkDeltaHBwdPreprocessTaskRange range = this->ResolveTaskRange();
        for (uint32_t i = 0; i < range.taskCount; ++i) {
            const uint32_t hv = range.taskBegin + i;
            InitState(chunkNum);
            // v3 流水：AIV 只做链外准备（slot）与链上的向量累加，Cube 承担全部 MMAD
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
                // 链外：Cube 算完本 chunk 的 T1 后转成模型 dtype
                // （arch22 上不做"AIC 直接落模型 dtype + 核内 FIX→MTE2"：该自排空实测会让 A2 挂死）
                CDHP_AIV_WAIT(CdhpFlag(CDHP_FLAG_T1_READY, win));
                StageT1Convert(win);
                CDHP_AIV_SET(CdhpFlag(CDHP_FLAG_T1BF_READY, win));
                // 链上准备：把"本 chunk 之前的"状态（首 chunk 即初值 0 / I）写成 Cube 需要的
                // 模型 dtype 操作数，并放行 Cube 的 Z/ZP
                StageStateStore(win, c == 0);
                // 链上：等 Cube 的 Z/ZP，做 dH/P 的向量累加
                CDHP_AIV_WAIT(CdhpFlag(CDHP_FLAG_Z_READY, win));
                StageState(chunkIdx, win);
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

    // v22：逐行因子（decay / 门控行因子）的"块广播"写法。
    //
    // 改造前：对 tile 的每一行调一次 `Muls(row, row, factor.GetValue(row), cols)`——每 chunk 在
    // V0 门控（q/k 两遍）+ E 链 + P 链上共 ~380 次 "标量取因子 + 一次向量指令"，是 A2 上 scalar
    // 与 vec 两个 pipe 的最大单一来源（见 design.md §30.2 的剖面）。
    //
    // 改造后：先用 `Brcb` 把 rows 个因子各展开成一个 32B block（8 个 float），再用一次带
    // repeat 参数的 `Mul` 覆盖整块 tile：repeat 内 src1 广播同一个 block（src1BlkStride = 0），
    // 每 repeat 前进一个 block（src1RepStride = 1），src0/dst 每 repeat 前进一整行
    // （RepStride = rowStride / 8，单位是 32B block）。
    // 语义与参数取值照仓内已发行的 chunk_bwd_dqkwg（同 arch22、同 A2 硬件）的同名写法：
    //   Brcb(dstBlk, src, CEIL_DIV(rows, 8), {1, 8});
    //   Mul(tile, tile, dstBlk, 64, rows, {1, 1, 0, K/8, K/8, 1});
    // 一列 64 个 float 一次（K=V=128 时每次 tile 调 2 次）。
    __aicore__ inline void ScaleRowsByFactor(const LocalTensor<float> &tile, const LocalTensor<float> &factor,
                                             uint32_t rowStart, uint32_t rows, uint32_t rowStride)
    {
        LocalTensor<float> blk = facBlkF32_.Get<float>();
        AscendC::Brcb(blk, factor[rowStart], static_cast<uint8_t>((rows + 7U) / 8U),
                      AscendC::BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();
        const uint8_t repStride = static_cast<uint8_t>(rowStride / 8U);
        const AscendC::BinaryRepeatParams params{1, 1, 0, repStride, repStride, 1};
        for (uint32_t c = 0; c < rowStride; c += 64U) {
            AscendC::Mul(tile[c], tile[c], blk, 64, static_cast<uint8_t>(rows), params);
        }
    }

    // GM [rows, cols]（行距 rowStride）→ FP32 UB
    // v25：GM 侧行距与列数相同（行首尾相接）时退化成**单块**搬运。A2 剖面显示每 chunk 有 ~480 笔
    // "每行 256B"的小包（mte3 stall 占 AIV 时间 29%、mte2 stall 占 14%），单块化后 DMA 引擎的
    // 逐包开销直接消失。UB 侧永远是紧凑排布，所以只有 GM 侧的行距要判断。
    __aicore__ inline AscendC::DataCopyExtParams MakeInParams(uint32_t rows, uint32_t cols, uint32_t rowStride,
                                                              uint32_t elemBytes)
    {
        if (rowStride == cols) {
            return AscendC::DataCopyExtParams{1, rows * cols * elemBytes, 0, 0, 0};
        }
        return AscendC::DataCopyExtParams{static_cast<uint16_t>(rows), cols * elemBytes,
                                          (rowStride - cols) * elemBytes, 0, 0};
    }

    __aicore__ inline AscendC::DataCopyExtParams MakeOutParams(uint32_t rows, uint32_t cols, uint32_t rowStride,
                                                               uint32_t elemBytes)
    {
        if (rowStride == cols) {
            return AscendC::DataCopyExtParams{1, rows * cols * elemBytes, 0, 0, 0};
        }
        return AscendC::DataCopyExtParams{static_cast<uint16_t>(rows), cols * elemBytes, 0,
                                          (rowStride - cols) * elemBytes, 0};
    }

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
        AscendC::DataCopyExtParams params = MakeInParams(rows, cols, rowStride, sizeof(DT));
        AscendC::DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(tmp, gSrc, params, padParams);
        SetFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        AscendC::Cast(dst, tmp, AscendC::RoundMode::CAST_NONE, rows * cols);
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
        SetFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        AscendC::DataCopyExtParams params = MakeOutParams(rows, cols, rowStride, sizeof(DT));
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
        SetFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(CDHP_EV_V_MTE3);
        // 统一用 DataCopyPad（与 slot 落盘同一条通路；DataCopy 在 UB→GM 上实测会写错内容）
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(DT)), 0, 0, 0};
        AscendC::DataCopyPad(gDst, tmp, params);
        PipeBarrier<PIPE_MTE3>();
    }

    // v13/v14：`do` 与 `dv` 两个平面在 Vector 侧没有任何实数运算（取负已挪到 `W` 上，见 StageV0），
    // 只需要把原始输入按模型 dtype 搬到 slot 的同一 dtype 平面上。原来走"载入 → Cast 到 FP32 →
    // Cast 回模型 dtype → 落盘"，每 tile 多 4 次 Cast。这里复用搬入暂存区做同 dtype 中转：
    // 起手把 V 读取与 MTE3 搬出都排空（本文件一律用整管自屏障），加载完成后再搬出——完全不进 V pipe。
    __aicore__ inline void CopyTileModel(GM_ADDR src, GM_ADDR dst, uint32_t rows, uint32_t cols,
                                         uint32_t rowStrideIn, uint32_t rowStrideOut)
    {
        AscendC::GlobalTensor<DT> gSrc;
        AscendC::GlobalTensor<DT> gDst;
        gSrc.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(src));
        gDst.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dst));
        LocalTensor<DT> tmp = s0DT_.Get<DT>();
        // 先等本核先前 V（Cast 读取该暂存区）与 MTE3（搬出该暂存区）排空
        SetFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(CDHP_EV_V_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(CDHP_EV_MTE3_MTE2);
        AscendC::DataCopyExtParams inParams = MakeInParams(rows, cols, rowStrideIn, sizeof(DT));
        AscendC::DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(tmp, gSrc, inParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(CDHP_EV_MTE2_MTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(CDHP_EV_MTE2_MTE3);
        AscendC::DataCopyExtParams outParams = MakeOutParams(rows, cols, rowStrideOut, sizeof(DT));
        AscendC::DataCopyPad(gDst, tmp, outParams);
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
        // v25：FP32 平面的行在 GM 侧连续（行距 == 列数），整体一笔搬完
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(rows * cols * sizeof(float)), 0, 0, 0};
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
    // 契约：入口只消费本 chunk 的 q/k/w/do 原始输入与 g/gk；出口保证 slot 内四个平面的
    //       无效行（[rows, chunkSize)）为 0，且 decayK[K] 已按 chunk 末行门控算好。
    __aicore__ inline void StageV0(uint32_t hv, uint32_t chunkIdx, uint32_t window)
    {
        const uint32_t rows = this->ChunkRows(chunkIdx);
        const uint32_t t0 = this->ChunkStart(chunkIdx);
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t mDim = static_cast<uint32_t>(this->tiling_.chunkSize);
        const uint32_t hk = this->HkOfHv(hv);
        // slot 按 (工作组, window) 区分：v3 起每个工作组有 CDHP_WINDOW_COUNT 份 slot，
        // window = chunkIdx & 1，必须与 Cube 侧寻址一致（否则读到未初始化 slot）
        const uint32_t slot = static_cast<uint32_t>(this->BlockIdx()) * CDHP_WINDOW_COUNT + (window & 1u);
        const uint32_t useG = static_cast<uint32_t>(this->tiling_.useGateG);
        const uint32_t useGk = static_cast<uint32_t>(this->tiling_.useGateGk);
        // v3 布局：Q̄s | W | -K̄ | do | -dv | decay（Q̄s/W 相邻、do/-dv 相邻，便于 Cube 合并 GEMM）
        GM_ADDR slotQ = this->SlotAt(slot, 0);
        GM_ADDR slotW = this->SlotAt(slot, this->SlotQBytes());
        GM_ADDR slotK = this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes());
        GM_ADDR slotDo = this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes() + this->SlotKBytes());
        GM_ADDR slotNegDv = this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes() + this->SlotKBytes() +
                                                   this->SlotDoBytes());
        GM_ADDR slotDecay = this->SlotAt(slot, this->SlotDecayOffset());

        LocalTensor<float> gF = gF32_.Get<float>();
        LocalTensor<float> decayF = decayF32_.Get<float>();
        // 门控行因子（仅 gate=g 路径使用）：函数级声明，tile 循环里按行取用
        LocalTensor<float> qFac = qFacF32_.Get<float>();
        LocalTensor<float> kFac = kFacF32_.Get<float>();
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
            // 行因子一次性算好：qFac_r = exp2(g_r)·scale，kFac_r = -exp2(g_last-g_r)。
            // 原来在 tile 循环里每行调 2 次 ExpScalar（每 chunk/head 128 次 V↔S 往返），
            // 是 A2 scalar pipe 的最大来源；这里改成 6 条向量指令 + 1 次 V→S。
            const float gLastVal = gF.GetValue(rows - 1);
            AscendC::Muls(qFac, gF, CDHP_LN2, rows);
            AscendC::Adds(kFac, gF, -gLastVal, rows);
            PipeBarrier<PIPE_V>();
            AscendC::Exp(qFac, qFac, rows);
            AscendC::Muls(kFac, kFac, -CDHP_LN2, rows);
            PipeBarrier<PIPE_V>();
            AscendC::Muls(qFac, qFac, scale_, rows);
            AscendC::Exp(kFac, kFac, rows);
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
            if (validRows == 0) {
                // v14：无效行由 InitState 预置的整片 0 保持，这里不必再逐 tile 补零
                continue;
            }
            LocalTensor<float> qF = s0F32_.Get<float>();
            LocalTensor<float> kF = s1F32_.Get<float>();
            LocalTensor<float> wF = s2F32_.Get<float>();
            LoadTileF32(qF, qAddr_ + static_cast<uint64_t>(hk * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                        validRows, kDim, kDim);
            LoadTileF32(kF, kAddr_ + static_cast<uint64_t>(hk * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                        validRows, kDim, kDim);
            LoadTileF32(wF, wAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * kDim * sizeof(DT),
                        validRows, kDim, kDim);
            // v14：do / dv 只需同 dtype 搬运（取负已挪到 W 上）
            CopyTileModel(doAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * vDim * sizeof(DT),
                          slotDo + static_cast<uint64_t>(r0) * vDim * sizeof(DT), validRows, vDim, vDim, vDim);
            CopyTileModel(dvAddr_ + static_cast<uint64_t>(hv * this->tiling_.T + t0 + r0) * vDim * sizeof(DT),
                          slotNegDv + static_cast<uint64_t>(r0) * vDim * sizeof(DT), validRows, vDim, vDim, vDim);
            if (useG != 0) {
                // v22：因子已按 chunk 一次性算好（kFac 不含负号），这里改成 Brcb + 带广播的 Mul：
                // 原来逐行 `Muls + GetValue` 每 chunk 要 128 次（q/k 各 64 行），现在每 tile 2 次。
                ScaleRowsByFactor(qF, qFac, r0, validRows, kDim);
                PipeBarrier<PIPE_V>();
                ScaleRowsByFactor(kF, kFac, r0, validRows, kDim);
                PipeBarrier<PIPE_V>();
            } else {
                AscendC::Muls(qF, qF, scale_, validRows * kDim);
                PipeBarrier<PIPE_V>();
            }
            // v14：取负从 K̄/dv 挪到 W 上：T1 = (-W)ᵀ@K̄ = Wᵀ@(-K̄)，AB 的 Wᵀ@(-dv) 项 = (-W)ᵀ@dv，
            // Cube 侧的 A/B 操作数与形状都没变，因此一行都不用改。
            AscendC::Muls(wF, wF, -1.0f, validRows * kDim);
            PipeBarrier<PIPE_V>();
            StoreTileModel(qF, slotQ + static_cast<uint64_t>(r0) * kDim * sizeof(DT), validRows, kDim, kDim);
            StoreTileModel(kF, slotK + static_cast<uint64_t>(r0) * kDim * sizeof(DT), validRows, kDim, kDim);
            StoreTileModel(wF, slotW + static_cast<uint64_t>(r0) * kDim * sizeof(DT), validRows, kDim, kDim);
        }
        StoreScalarF32(decayF, slotDecay, kDim);
    }

    // v3 链外：把 Cube 算出的 T1(FP32 [K,K]) 转成模型 dtype，供 Cube 的 Z/ZP 当矩阵操作数
    __aicore__ inline void StageT1Convert(uint32_t window)
    {
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            // v26：T1 已经是模型 dtype（Cube 的 fixpipe 直接落盘），这里只把它搬到**另一块平面**——
            // cube 必须读一块自己没写过的平面，才能避开 arch22 上"核内 FIX→MTE2 自排空挂死"；
            // 同 dtype 搬运走 CopyTileModel（不进 V pipe，也不再有 Cast）。
            CopyTileModel(this->T1At(window) + static_cast<uint64_t>(r0) * kDim * sizeof(DT),
                          this->T1BfAt(window) + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim,
                          kDim, kDim);
        }
    }

    // InitState：只把常驻 UB 的 dH 状态清零（初值 0）。
    // 契约：每个 task（hv）进入 chunk 逆序循环前调用一次；首个 chunk 的模型 dtype 操作数
    //       （dH = 0、P = I）由 StageStateStore 在放行 Cube 之前写盘。
    __aicore__ inline void InitState(uint32_t chunkNum)
    {
        (void)chunkNum;
        // 跨 task 复用 UB 的 WAR：上一个 task 的最后一次搬出（MTE3，可能是 WriteOutput 的输出写回）
        // 可能仍在读 dhStateF32_/s1F32_。`PipeBarrier<PIPE_MTE3>` 只约束 MTE3 自己，拦不住随后的 V/S，
        // 若不等这一步，本轮 Duplicate 会把「还在搬出中的那块 UB」改写，
        // 输出的最后一个 tile 会写成当前 UB 里的内容（实测表现为中间 task 的 P 面尾部变成单位阵）。
        SetFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(CDHP_EV_MTE3_V);
        SetFlag<HardEvent::MTE3_S>(CDHP_EV_MTE3_S);
        WaitFlag<HardEvent::MTE3_S>(CDHP_EV_MTE3_S);
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        LocalTensor<float> dhState = dhStateF32_.Get<float>();
        Duplicate(dhState, 0.0f, kDim * vDim);
        PipeBarrier<PIPE_V>();
        // v14：slot 的 5 个平面在**每个 head 开始时**整体清 0（一个 head 只做一轮，代价可忽略）。
        // 之后 StageV0 只写有效行：一个 head 里第一个被处理的 chunk 是索引最大的那个（也就是唯一可能
        // 不满一个 chunk 的尾块），它读到的无效行正是这里预置的 0；后续满 chunk 会把整片重写，因此
        // 尾块的无效行永远保持 0，不必再为尾块保留一条 FP32 通路。
        LocalTensor<float> zeroF = s2F32_.Get<float>();
        Duplicate(zeroF, 0.0f, CDHP_SCRATCH_ELEMS);
        PipeBarrier<PIPE_V>();
        for (uint32_t window = 0; window < CP::CDHP_WINDOW_COUNT; ++window) {
            const uint32_t slot = this->SliceOfWindow(window);
            ZeroSlotPlane(this->SlotAt(slot, 0), kDim, zeroF);
            ZeroSlotPlane(this->SlotAt(slot, this->SlotQBytes()), kDim, zeroF);
            ZeroSlotPlane(this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes()), kDim, zeroF);
            ZeroSlotPlane(this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes() + this->SlotKBytes()), vDim,
                          zeroF);
            ZeroSlotPlane(this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes() + this->SlotKBytes() +
                                                 this->SlotDoBytes()),
                          vDim, zeroF);
        }
    }

    // v14：把一块 [chunkSize, cols] 的模型 dtype 平面整体写 0（逐 tile 落盘；每个 head 只调用一轮）
    __aicore__ inline void ZeroSlotPlane(GM_ADDR plane, uint32_t cols, const LocalTensor<float> &zeroF)
    {
        const uint32_t mDim = static_cast<uint32_t>(this->tiling_.chunkSize);
        for (uint32_t r0 = 0; r0 < mDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, mDim - r0);
            StoreTileModel(zeroF, plane + static_cast<uint64_t>(r0) * cols * sizeof(DT), tileRows, cols, cols);
        }
    }

    // v5 链上准备（参考 ChunkFwdH 的 rolling state）：
    // 把"本 chunk 之前的"状态写成 Cube 需要的模型 dtype 操作数，再放行 Cube 的 Z/ZP。
    //   dH：直接由 UB 常驻状态转模型 dtype 落盘（首 chunk 即全零初值）
    //   P ：v15 起与 dH 一样常驻 UB（pStateF32_）：首 chunk 内联单位阵写进常驻状态，之后每 chunk
    //       就地累加；这里只负责把常驻状态转成模型 dtype 操作数（PBf），不再有 PAt 平面往返。
    __aicore__ inline void StageStateStore(uint32_t window, bool isFirstChunk)
    {
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t prevWin = (window + 1u) & 1u;
        LocalTensor<float> dhState = dhStateF32_.Get<float>();
        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            // dH 操作数
            StoreTileModel(dhState[static_cast<uint32_t>(r0) * vDim],
                           this->DhBfAt(prevWin) + static_cast<uint64_t>(r0) * vDim * sizeof(DT), tileRows, vDim,
                           vDim);
            if (isFirstChunk) {
                InlineIdentityIntoPState(r0, tileRows);
            }
            // P 操作数：直接由常驻状态落盘
            LocalTensor<float> pState = pStateF32_.Get<float>();
            StoreTileModel(pState[static_cast<uint32_t>(r0) * kDim],
                           this->PBfAt(prevWin) + static_cast<uint64_t>(r0) * kDim * sizeof(DT), tileRows, kDim,
                           kDim);
        }
        // 操作数已落盘：通知 Cube 可以开始本 chunk 的 Z/ZP
        CDHP_AIV_SET(CdhpFlag(CDHP_FLAG_STATE_READY, prevWin));
    }

    // v15：把常驻 P 状态的 [r0, r0+tileRows) 行写成单位阵块（首 chunk 的初值）
    __aicore__ inline void InlineIdentityIntoPState(uint32_t r0, uint32_t tileRows)
    {
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        LocalTensor<float> pState = pStateF32_.Get<float>();
        Duplicate(pState[r0 * kDim], 0.0f, tileRows * kDim);
        PipeBarrier<PIPE_V>();
        // V 先写（Duplicate 清零）→ S 再写（对角注入）：关闭自动同步后必须先等 V 落盘，
        // 否则标量写的 1.0 会被随后的向量清零覆盖（表现为 P 初值全零）。
        SetFlag<HardEvent::V_S>(CDHP_EV_V_S);
        WaitFlag<HardEvent::V_S>(CDHP_EV_V_S);
        for (uint32_t r = 0; r < tileRows; ++r) {
            pState.SetValue(static_cast<uint32_t>(r0 + r) * kDim + (r0 + r), 1.0f);
        }
        SetFlag<HardEvent::S_V>(CDHP_EV_S_V);
        WaitFlag<HardEvent::S_V>(CDHP_EV_S_V);
    }

    // V4：路 A（dH 更新 + dHBf）+ 路 B（P_c + 上一轮 P 的 dtype 拷贝）
    // 契约：入口消费 C3 的 qterm/wterm、C1 的 T1 与 V0 的 decayK；出口 dH_new（FP32 落盘 + 模型 dtype
    //       操作数）、P_c（模型 dtype）与本轮 P 的模型 dtype 拷贝，供 C5 使用、并供下一轮 C1/V4 复用。
    // v3 状态链（向量侧）：dH_new = decay⊙dH_prev + AB + Z；P_new = decay⊙P_prev + ZP
    __aicore__ inline void StageState(uint32_t chunkIdx, uint32_t window)
    {
        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        LocalTensor<float> decayF = decayF32_.Get<float>();
        // V0 领先，UB 里的 decayF32_ 存活期不覆盖本 Stage → 从本 window 的 slot 重新载入 decayK
        LoadPlaneF32(decayF, this->SlotAtWindow(window, this->SlotDecayOffset()), 0, 1, kDim);
        SetFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CDHP_EV_MTE2_V);
        SetFlag<HardEvent::V_S>(CDHP_EV_V_S);
        WaitFlag<HardEvent::V_S>(CDHP_EV_V_S);

        LocalTensor<float> dhState = dhStateF32_.Get<float>();
        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            LocalTensor<float> abF = s0F32_.Get<float>();
            // v24：Z 已由 Cube 用 fixpipe 原子加进 AB 平面，这里只搬一块（原来 AB/Z 各 64 KiB + 一次 Add）
            LoadPlaneF32(abF, this->AbAt(window), r0, tileRows, vDim);
            // v5（参考 ChunkFwdH 的 rolling state）：dH_old 直接取 UB 常驻状态并就地更新，
            // 不再每 chunk 经 GM workspace 读写 FP32 状态。
            // v22：行衰减改成 Brcb + 带广播的 Mul（原来逐行 Muls，每 chunk 128 次）
            ScaleRowsByFactor(dhState[r0 * vDim], decayF, r0, tileRows, vDim);
            PipeBarrier<PIPE_V>();
            AscendC::Add(dhState[r0 * vDim], dhState[r0 * vDim], abF, tileRows * vDim);
            PipeBarrier<PIPE_V>();
        }

        for (uint32_t r0 = 0; r0 < kDim; r0 += CDHP_VEC_TILE) {
            const uint32_t tileRows = MinV(CDHP_VEC_TILE, kDim - r0);
            LocalTensor<float> zpF = s1F32_.Get<float>();
            LoadPlaneF32(zpF, this->ZpAt(window), r0, tileRows, kDim);
            // v15：P 的状态常驻 UB，就地累加 P_new = decay ⊙ P_old + ZP（首 chunk 的 P_old = I
            // 已由 StageStateStore 写进常驻状态），不再经 PAt 平面往返
            LocalTensor<float> pState = pStateF32_.Get<float>();
            ScaleRowsByFactor(pState[r0 * kDim], decayF, r0, tileRows, kDim);
            PipeBarrier<PIPE_V>();
            AscendC::Add(pState[r0 * kDim], pState[r0 * kDim], zpF, tileRows * kDim);
            PipeBarrier<PIPE_V>();
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
        // E 面：dH 的末值就在 UB 常驻状态（dhStateF32_）里，不必再经 GM workspace 往返。
        // 源由 V（状态累加）写，搬出前建立 V→MTE3 顺序。
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
            // v15：P 的末值就在常驻状态里，直接搬出（与 E 面同一条路径，不再经 PAt 回读）
            LocalTensor<float> fp = pState[static_cast<uint32_t>(r0) * kDim];
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> s0DT_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decayF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qFacF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kFacF32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> facBlkF32_;  // v22：逐行因子的 Brcb 展开块
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

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_ARCH22_VECTOR_H
