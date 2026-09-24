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

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_CUBE_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_CUBE_H

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
#include "chunk_delta_h_bwd_preprocess_base.h"
#include "chunk_delta_h_bwd_preprocess_policy.h"

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

template <typename DT>
class ChunkDeltaHBwdPreCube : public ChunkDeltaHBwdPreBase<DT, DT> {
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
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true>;
        using L1TileShape = tla::Shape<_128, _128, _128>;
        using L0TileShape = tla::Shape<_128, _128, _128>;
        using LayoutRow = Catlass::layout::RowMajor;
        using LayoutCol = Catlass::layout::ColumnMajor;

        // 模型 dtype × 模型 dtype → FP32 输出
        using TileCopyRowRow = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutRow, DT, LayoutRow, float,
                                                                      LayoutRow>;
        using TileCopyColRow = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutCol, DT, LayoutRow, float,
                                                                      LayoutRow>;
        using BlockRowRow =
            Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DT, DT, float, void,
                                               TileCopyRowRow>;
        using BlockColRow =
            Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DT, DT, float, void,
                                               TileCopyColRow>;

        Catlass::Arch::Resource<ArchTag> resource;

        const uint32_t kDim = static_cast<uint32_t>(this->tiling_.K);
        const uint32_t vDim = static_cast<uint32_t>(this->tiling_.V);
        const uint32_t mDim = static_cast<uint32_t>(this->tiling_.chunkSize);
        const auto tagQ = LayoutRow::MakeLayout<DT>(mDim, kDim);
        const auto tagK = LayoutRow::MakeLayout<DT>(mDim, kDim);
        const auto tagQT = LayoutCol::MakeLayout<DT>(kDim, mDim);
        const auto tagWT = LayoutCol::MakeLayout<DT>(kDim, mDim);
        const auto tagDo = LayoutRow::MakeLayout<DT>(mDim, vDim);
        const auto tagDhBf = LayoutRow::MakeLayout<DT>(kDim, vDim);
        const auto tagDvHat = LayoutRow::MakeLayout<DT>(mDim, vDim);
        const auto tagKK = LayoutRow::MakeLayout<DT>(kDim, kDim);
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
        const auto layoutMzV = tla::MakeLayoutFromTag(tagMzV);
        const auto layoutKzV = tla::MakeLayoutFromTag(tagKzV);
        const auto layoutKzK = tla::MakeLayoutFromTag(tagKzK);

        const uint32_t chunkNum = this->ChunkNum(this->Bos(), this->Eos());
        const ChunkDeltaHBwdPreTaskRange range = this->ResolveTaskRange();
        for (uint32_t i = 0; i < range.taskCount; ++i) {
            (void)range.taskBegin;
            for (uint32_t c = 0; c < chunkNum; ++c) {
                const uint32_t chunkIdx = chunkNum - 1 - c;
                const uint32_t slot = this->BlockIdx();
                const uint32_t parity = chunkIdx % 2;
                GM_ADDR slotQ = this->SlotAt(slot, 0);
                GM_ADDR slotK = this->SlotAt(slot, this->SlotQBytes());
                GM_ADDR slotW = this->SlotAt(slot, this->SlotQBytes() + this->SlotKBytes());
                GM_ADDR slotDo = this->SlotAt(slot, this->SlotQBytes() + this->SlotKBytes() + this->SlotWBytes());

                CDHP_AIC_WAIT(CDHP_FLAG_V0_DONE);
                // C1-① dV_pre = K̄ @ dH_bf
                RunGemm<BlockRowRow>(resource, layoutK, layoutDhBf, layoutMzV, slotK, this->DhBfAt(),
                                     this->DvPreAt(), mDim, vDim, kDim);
                // C1-② T1 = W^T @ K̄
                RunGemm<BlockColRow>(resource, layoutWT, layoutK, layoutKzK, slotW, slotK, this->T1At(), kDim, kDim,
                                     mDim);
                CDHP_AIC_SET(CDHP_FLAG_C1_DONE);

                CDHP_AIC_WAIT(CDHP_FLAG_V2_DONE);
                // C3-③ qterm = Q̄s^T @ do
                RunGemm<BlockColRow>(resource, layoutQT, layoutDo, layoutKzV, slotQ, slotDo, this->QtermAt(), kDim,
                                     vDim, mDim);
                // C3-④ wterm = W^T @ dV̂'
                RunGemm<BlockColRow>(resource, layoutWT, layoutDvHat, layoutKzV, slotW, this->DvHatAt(),
                                     this->WtermAt(), kDim, vDim, mDim);
                CDHP_AIC_SET(CDHP_FLAG_C3_DONE);

                CDHP_AIC_WAIT(CDHP_FLAG_V4_DONE);
                // C5 P_new = P_c @ P_bf（链式累加在 FP32 平面）
                RunGemm<BlockRowRow>(resource, layoutKK, layoutKK, layoutKzK, this->PcAt(), this->PBfAt(),
                                     this->PAt(parity), kDim, kDim, kDim);
                CDHP_AIC_SET(CDHP_FLAG_C5_DONE);
            }
        }
    }

private:
    // 与仓内 chunk_bwd_dv_local_cube.h 一致：BlockMmad 在每次调用时构造（复用同一 resource）
    template <typename Block, typename Resource, typename LayoutA, typename LayoutB, typename LayoutC>
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
        AscendC::GlobalTensor<float> gC;
        gA.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(a));
        gB.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(b));
        gC.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c));
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

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_CUBE_H
