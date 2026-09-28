/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess_cube.h
 * \brief Cube 侧 Stage：C1（dV_pre 与 T1）、C3（qterm 与 wterm）、C5（P 链）。
 *
 * 数学（chunk 逆序）：
 *   C1  dV_pre[M,V] = K̄[M,K] @ dH_bf[K,V]        （A=K̄ RowMajor，B=dH_bf RowMajor）
 *       T1[K,K]     = W[M,K]^T @ K̄[M,K]          （A=W ColumnMajor，B=K̄ RowMajor）
 *   C3  qterm[K,V]  = Q̄s[M,K]^T @ do[M,V]        （A=Q̄s ColumnMajor，B=do RowMajor）
 *       wterm[K,V]  = W[M,K]^T @ dV̂'[M,V]         （A=W ColumnMajor，B=dV̂' RowMajor）
 *   C5  P_new[K,K]  = P_c[K,K] @ P_bf[K,K]        （A=P_c RowMajor，B=P_bf RowMajor）
 *
 * 说明：C3 的两个乘积分两个 FP32 平面写回，由向量侧 V4 求和（不依赖 L0C 跨调用累加语义）；链式累加
 * 仍在 FP32 平面完成，只有矩阵乘的操作数使用模型 dtype（与上游 tl.dot 前 cast 到模型 dtype 一致）。
 *
 * 同步：与 AIV 按同一 chunk 顺序严格交替，flag 见 chunk_delta_h_bwd_preprocess_policy.h。
 */

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_ARCH35_CUBE_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_ARCH35_CUBE_H

// 说明：不要引用仓内 common/kernel_utils/block/block_mmad_pingpong_tla.hpp —— 它经由
// kernel_utils/tile/copy_l0c_to_ub.hpp 无条件包含 ascend950 专用头，A2 构建会把 950 实现拉进来
// （ScaleGranularity / CopyL0CToGm 重定义）。这里与已在 A2/A5 双平台发行的
// chunk_bwd_dv_local_cube.h 保持一致：catlass 原生 BlockMmadTla + CATLASS_ARCH 选架构实现。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
using _128 = tla::Int<128>;
#else
#define CATLASS_ARCH 2201
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
using _128 = tla::Int<128>;
#endif

#include "kernel_operator.h"
#include "../chunk_delta_h_bwd_preprocess_common.h"
#include "../chunk_delta_h_bwd_preprocess_policy.h"

using namespace AscendC;

namespace CP {

// L1/L0 分块：与上游 L1/L0 tile 选择一致；实际 m/n 超过 128 时由外层循环切分，k 由 BlockMmad 内部累加
constexpr uint32_t CDHP_CUBE_TILE_M = 128;
constexpr uint32_t CDHP_CUBE_TILE_N = 128;

// BlockMmadTla 每次构造都会重新 SetFlag(MTE1_MTE2 / M_MTE1 / FIX_M) 做事件初始化，
// 相当于默认假定"上一轮各 pipe 已排空"。这里的 EventID 取 4/5/6，避开 BlockMmadTla 内部使用的
// 0..(stages-1) 区间（默认 L1A/L1B/L0A/L0B 各 2 级、L0C 1 级，即 0..3）。
constexpr uint32_t CDHP_AIC_EV_FIX_M = 4;      // Fixpipe（L0C→GM）已读完 L0C
constexpr uint32_t CDHP_AIC_EV_M_MTE1 = 5;     // MMAD 已读完 L0A/L0B
constexpr uint32_t CDHP_AIC_EV_MTE1_MTE2 = 6;  // MTE1（L1→L0）已读完 L1
// v4：T1 直接以模型 dtype 落地（fixpipe 完成 FP32 累加→模型 dtype 的转换），随后仍由本核在
// 下一 chunk 步用 MTE2 把它读回来当 Z/ZP 的矩阵操作数。这是**本核** FIX→MTE2 的 RAW 依赖，
// 用核内事件表达（每个 window 一个 id，set/wait 严格交替），不再占用核间 flag。
constexpr uint32_t CDHP_AIC_EV_FIX_MTE2 = 0;

template <typename DT>
class ChunkDeltaHBwdPreprocessCube : public ChunkDeltaHBwdPreprocessBase<DT, DT> {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR cu_seqlens,
                                GM_ADDR dhm, GM_ADDR userWs, const ChunkDeltaHBwdPreprocessTilingData &tiling)
    {
        this->InitTilingData(tiling);
        this->SetUserWorkspace(userWs);
        this->ResolveSegment(cu_seqlens);
    }

    __aicore__ inline void Process()
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ArchTag = Catlass::Arch::Ascend950;
#else
        using ArchTag = Catlass::Arch::AtlasA2;
#endif
        // A/B 实验：第二个模板参数是 ENABLE_UNIT_FLAG。置 true 时 Catlass 内部 L0C 只开 1 个
        // stage（M/FIX 同步由 MMAD 指令的 unit flag 承担），置 false 时 L0C 走 L0C_STAGES=2 的
        // 显式 FIX_M 事件。用同一份用例对比 Task Duration 与 fixpipe 占用率。
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, false>;
        using L1TileShape = tla::Shape<_128, _128, _128>;
        using L0TileShape = tla::Shape<_128, _128, _128>;
        using LayoutRow = Catlass::layout::RowMajor;
        using LayoutCol = Catlass::layout::ColumnMajor;

        // 模型 dtype × 模型 dtype → FP32 输出
        using TileCopyRowRow = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutRow, DT, LayoutRow, float,
                                                                      LayoutRow>;
        using TileCopyColRow = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutCol, DT, LayoutRow, float,
                                                                      LayoutRow>;
        // C 目标类型取模型 dtype：让 fixpipe 直接把 L0C 的 FP32 结果转成 bf16 落盘，
        // 省掉 Vector 侧"读 FP32 T1 → 转模型 dtype"这一级
        using TileCopyColRowBf16 = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutCol, DT, LayoutRow, DT,
                                                                          LayoutRow>;
        using BlockRowRow =
            Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DT, DT, float, void,
                                               TileCopyRowRow>;
        using BlockColRow =
            Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DT, DT, float, void,
                                               TileCopyColRow>;
        using BlockColRowBf16 =
            Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DT, DT, DT, void,
                                               TileCopyColRowBf16>;

        Catlass::Arch::Resource<ArchTag> resource;

        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t mDim = static_cast<uint32_t>(this->tiling_.chunkSize);
        const uint32_t twoM = 2U * mDim;
        const auto tagQ = LayoutRow::MakeLayout<DT>(mDim, kDim);
        const auto tagK = LayoutRow::MakeLayout<DT>(mDim, kDim);
        const auto tagQT = LayoutCol::MakeLayout<DT>(kDim, mDim);
        const auto tagWT = LayoutCol::MakeLayout<DT>(kDim, mDim);
        const auto tagDo = LayoutRow::MakeLayout<DT>(mDim, vDim);
        const auto tagDhBf = LayoutRow::MakeLayout<DT>(kDim, vDim);
        const auto tagDvHat = LayoutRow::MakeLayout<DT>(mDim, vDim);
        const auto tagKK = LayoutRow::MakeLayout<DT>(kDim, kDim);
        // v3 合并 GEMM：A = [Q̄s|W]ᵀ（K x 2M，列主序 = 存储 [Q̄s;W] 的 2M x K 行主序）
        const auto tagCol2M = LayoutCol::MakeLayout<DT>(kDim, 2U * mDim);
        const auto tagDo2M = LayoutRow::MakeLayout<DT>(2U * mDim, vDim);
        const auto tagMzV = LayoutRow::MakeLayout<float>(mDim, vDim);
        const auto tagKzV = LayoutRow::MakeLayout<float>(kDim, vDim);
        const auto tagKzK = LayoutRow::MakeLayout<float>(kDim, kDim);
        const auto layoutQ = tla::MakeLayoutFromTag(tagQ);
        const auto layoutK = tla::MakeLayoutFromTag(tagK);
        const auto layoutQT = tla::MakeLayoutFromTag(tagQT);
        const auto layoutWT = tla::MakeLayoutFromTag(tagWT);
        const auto layoutDo = tla::MakeLayoutFromTag(tagDo);
        const auto layoutDhBf = tla::MakeLayoutFromTag(tagDhBf);
        const auto layoutDvHat = tla::MakeLayoutFromTag(tagDvHat);
        const auto layoutKK = tla::MakeLayoutFromTag(tagKK);
        const auto layoutCol2M = tla::MakeLayoutFromTag(tagCol2M);
        const auto layoutDo2M = tla::MakeLayoutFromTag(tagDo2M);
        const auto layoutMzV = tla::MakeLayoutFromTag(tagMzV);
        const auto layoutKzV = tla::MakeLayoutFromTag(tagKzV);
        const auto layoutKzK = tla::MakeLayoutFromTag(tagKzK);

        const uint32_t chunkNum = this->ChunkNum(this->Bos(), this->Eos());
        const ChunkDeltaHBwdPreprocessTaskRange range = this->ResolveTaskRange();
        // A5（1 AIC : 2 AIV）：本 block 的两个 AIV 各承包一个 head 的整条逆序链，二者**并行推进**，
        // 因此 AIC 必须按 chunk 步把本轮参与的 head 交错服务：外层 chunk、内层 head。
        // 若把一个 head 的整条链跑完再跑下一个，另一个 AIV 会全程空等（等于没有并行）。
        // 每轮参与的是 h ∈ [r*aivs, r*aivs + aivs)，sub = h - r*aivs 就是配对 AIV 的序号；
        // A2/A3（aivs == 1）下 rounds == activeHeads、每轮只有一个 head，行为与改造前完全一致。
        const uint32_t activeHeads = this->AivSlotCount(range);
        const uint32_t aivs = this->AivPerBlock();
        const uint32_t rounds = (activeHeads + aivs - 1U) / aivs;
        for (uint32_t r = 0; r < rounds; ++r) {
            const uint32_t hBegin = r * aivs;
            const uint32_t hEnd = (hBegin + aivs < activeHeads) ? (hBegin + aivs) : activeHeads;
            // v5（参考 ChunkFwdH 的跨 chunk lookahead）：链外工作（T1/AB）**提前一个 chunk** 计算。
            // 这样下一 chunk 的 T1 dtype 转换落在当前 chunk 的 Z/ZP MMA 窗口内，AIC 不再在
            // T1BF_READY 上空等（改造前该等待占 AIC 时间的 88%）。
            if (chunkNum > 0) {
                const uint32_t headWin = (chunkNum - 1U) & 1u;
                for (uint32_t h = hBegin; h < hEnd; ++h) {
                    const uint32_t sub = h - hBegin;
                    const uint32_t slot = this->SlotOfWindowAiv(sub, headWin);
                    CDHP_AIC_WAIT(sub, CdhpFlag(CDHP_FLAG_V0_READY, headWin));
                    // T1 = Wᵀ@(-K̄) → **模型 dtype** 平面：fixpipe 直接完成 FP32→模型 dtype 的转换，
                    // 省掉 Vector 侧"读 FP32 T1 再转模型 dtype"的一整段搬运与一个 Stage
                    RunGemm<BlockColRowBf16, DT>(resource, layoutWT, layoutK, layoutKK,
                                                this->SlotAt(slot, this->SlotQBytes()),
                                                this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes()),
                                                this->T1BfAtAiv(sub, headWin), kDim, kDim, mDim);
                    // AB = Q̄sᵀ@do + Wᵀ@(-dv)（一次合并 GEMM：A=[Q̄s|W]ᵀ [K,2M]，B=[do;-dv] [2M,V]）
                    RunGemm<BlockColRow, float>(resource, layoutCol2M, layoutDo2M, layoutKzV,
                                                this->SlotAt(slot, 0),
                                                this->SlotAt(slot, this->SlotQBytes() + this->SlotWBytes() +
                                                                   this->SlotKBytes()),
                                                this->AbAtAiv(sub, headWin), kDim, vDim, twoM);
                }
            }
            if (chunkNum > 0) {
                // 本 window 全部 head 的 T1 都已 fixpipe 落盘：放行下一轮 MTE2 读回（每 window 一次）
                SetFlag<HardEvent::FIX_MTE2>(CDHP_AIC_EV_FIX_MTE2 + ((chunkNum - 1U) & 1u));
            }
            for (uint32_t c = 0; c < chunkNum; ++c) {
                const uint32_t chunkIdx = chunkNum - 1U - c;
                // v2.1：window = chunkIdx & 1；同一 chunk 的操作数/中间量都落在自己的 window 上，
                // AIC 因此可以与 AIV 提前准备的下一 window 并行，不再严格交替。
                const uint32_t win = chunkIdx & 1u;
                // 本 window 的 T1（模型 dtype）是上一轮 lookahead（或循环前）由本核 fixpipe 写的，
                // MTE2 读回前先等本核 FIX 完成。放在 head 循环外：每 window 恰好一次 set / 一次 wait，
                // 满足核内单比特事件的"不许连续 set 同一 id"约束（1:2 下两个 head 共用同一 id）。
                WaitFlag<HardEvent::FIX_MTE2>(CDHP_AIC_EV_FIX_MTE2 + win);
                for (uint32_t h = hBegin; h < hEnd; ++h) {
                    const uint32_t sub = h - hBegin;
                    // 链外：把下一个 chunk 的 T1/AB 补上（尾块没有下一个 chunk）
                    if (c + 1U < chunkNum) {
                        const uint32_t nextWin = (chunkIdx - 1U) & 1u;
                        const uint32_t nextSlot = this->SlotOfWindowAiv(sub, nextWin);
                        CDHP_AIC_WAIT(sub, CdhpFlag(CDHP_FLAG_V0_READY, nextWin));
                        RunGemm<BlockColRowBf16, DT>(resource, layoutWT, layoutK, layoutKK,
                                                    this->SlotAt(nextSlot, this->SlotQBytes()),
                                                    this->SlotAt(nextSlot, this->SlotQBytes() + this->SlotWBytes()),
                                                    this->T1BfAtAiv(sub, nextWin), kDim, kDim, mDim);
                        RunGemm<BlockColRow, float>(resource, layoutCol2M, layoutDo2M, layoutKzV,
                                                    this->SlotAt(nextSlot, 0),
                                                    this->SlotAt(nextSlot, this->SlotQBytes() + this->SlotWBytes() +
                                                                              this->SlotKBytes()),
                                                    this->AbAtAiv(sub, nextWin), kDim, vDim, twoM);
                    }
                    // 链上：Z = (-T1)@dH_bf(prev)、ZP = (-T1)@P_bf(prev)
                    CDHP_AIC_WAIT(sub, CdhpFlag(CDHP_FLAG_STATE_READY, win ^ 1u));
                    RunGemm<BlockRowRow, float>(resource, layoutKK, layoutDhBf, layoutKzV,
                                                this->T1BfAtAiv(sub, win), this->DhBfAtAiv(sub, win ^ 1u),
                                                this->ZAtAiv(sub, win), kDim, vDim, kDim);
                    RunGemm<BlockRowRow, float>(resource, layoutKK, layoutKK, layoutKzK,
                                                this->T1BfAtAiv(sub, win), this->PBfAtAiv(sub, win ^ 1u),
                                                this->ZpAtAiv(sub, win), kDim, kDim, kDim);
                    CDHP_AIC_SET(sub, CdhpFlag(CDHP_FLAG_Z_READY, win));
                }
                if (c + 1U < chunkNum) {
                    // 本轮两个 head 的 T1 都已落盘：放行下一轮 MTE2 读回同一 window
                    SetFlag<HardEvent::FIX_MTE2>(CDHP_AIC_EV_FIX_MTE2 + ((chunkIdx - 1U) & 1u));
                }
            }
        }
    }

private:
    // 与仓内 chunk_bwd_dv_local_cube.h 一致：BlockMmad 在每次调用时构造（复用同一 resource）
    // CElem：输出平面的元素类型（float = FP32 中间平面，DT = 直接落模型 dtype）
    template <typename Block, typename CElem, typename Resource, typename LayoutA, typename LayoutB,
              typename LayoutC>
    __aicore__ inline void RunGemm(Resource &resource, LayoutA layoutA, LayoutB layoutB, LayoutC layoutC, GM_ADDR a,
                                   GM_ADDR b, GM_ADDR c, uint32_t m, uint32_t n, uint32_t k)
    {
        // 构造新的 BlockMmadTla 之前，先把上一轮真正排空：否则新一轮的 MMAD 会覆盖上一轮 Fixpipe
        // 仍在读的 L0C，或覆盖上一轮 MMAD 仍在读的 L0A/L0B、上一轮 MTE1 仍在读的 L1，
        // 表现为上一轮写出的 GM 平面出现整块旧值（实测 wterm 部分 16 行 tile 被污染）。
        SetFlag<HardEvent::FIX_M>(CDHP_AIC_EV_FIX_M);
        WaitFlag<HardEvent::FIX_M>(CDHP_AIC_EV_FIX_M);
        SetFlag<HardEvent::M_MTE1>(CDHP_AIC_EV_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(CDHP_AIC_EV_M_MTE1);
        SetFlag<HardEvent::MTE1_MTE2>(CDHP_AIC_EV_MTE1_MTE2);
        WaitFlag<HardEvent::MTE1_MTE2>(CDHP_AIC_EV_MTE1_MTE2);
        Block block(resource);
        AscendC::GlobalTensor<DT> gA;
        AscendC::GlobalTensor<DT> gB;
        AscendC::GlobalTensor<CElem> gC;
        gA.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(a));
        gB.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(b));
        gC.SetGlobalBuffer(reinterpret_cast<__gm__ CElem *>(c));
        for (uint32_t m0 = 0; m0 < m; m0 += CDHP_CUBE_TILE_M) {
            const uint32_t mTile = (m - m0 < CDHP_CUBE_TILE_M) ? (m - m0) : CDHP_CUBE_TILE_M;
            for (uint32_t n0 = 0; n0 < n; n0 += CDHP_CUBE_TILE_N) {
                const uint32_t nTile = (n - n0 < CDHP_CUBE_TILE_N) ? (n - n0) : CDHP_CUBE_TILE_N;
                auto tensorA = tla::MakeTensor(gA, layoutA, Catlass::Arch::PositionGM{});
                auto tensorB = tla::MakeTensor(gB, layoutB, Catlass::Arch::PositionGM{});
                auto tensorC = tla::MakeTensor(gC, layoutC, Catlass::Arch::PositionGM{});
                auto blockA = GetTile(tensorA, tla::MakeCoord(m0, 0), tla::MakeShape(mTile, k));
                auto blockB = GetTile(tensorB, tla::MakeCoord(0, n0), tla::MakeShape(k, nTile));
                auto blockC = GetTile(tensorC, tla::MakeCoord(m0, n0), tla::MakeShape(mTile, nTile));
                Catlass::GemmCoord shape{mTile, nTile, k};
                block(blockA, blockB, blockC, shape);
            }
        }
    }
};

} // namespace CP

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_ARCH35_CUBE_H
