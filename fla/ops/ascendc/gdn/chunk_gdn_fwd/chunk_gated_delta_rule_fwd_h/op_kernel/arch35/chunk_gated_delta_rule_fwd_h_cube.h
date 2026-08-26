/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_ARCH35_CUBE_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_ARCH35_CUBE_H

#define CATLASS_ARCH 3510

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "../chunk_gated_delta_rule_fwd_h_policy.h"
#include "../chunk_gated_delta_rule_fwd_h_scheduler.h"
#include "kernel_operator.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GDN::FwdHStandalone {

struct TileShapes128 {
    using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    using L0TileShape = L1TileShape;
};

template <class InputT, class WorkspaceT, class TileShapes, uint32_t GateMode>
class ChunkGatedDeltaRuleFwdHCube {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using CubeScheduler = Catlass::Gemm::Block::BlockSchedulerGdnFwdHCube;
    using DispatchPolicy = Catlass::Gemm::MmadPingpongTlaMulti<
        ArchTag, false, false, 2>;
    using TailDispatchPolicy = Catlass::Gemm::MmadPingpongTlaMulti<
        ArchTag, true, false, 1>;
    using L1TileShape = typename TileShapes::L1TileShape;
    using L0TileShape = typename TileShapes::L0TileShape;

    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, InputT, Catlass::layout::RowMajor,
        InputT, Catlass::layout::RowMajor,
        WorkspaceT, Catlass::layout::RowMajor>;
    using BlockMmadWH = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape,
        InputT, InputT, WorkspaceT, void, TileCopyWH>;
    using BlockMmadWHTail = Catlass::Gemm::Block::BlockMmadTla<
        TailDispatchPolicy, L1TileShape, L0TileShape,
        InputT, InputT, WorkspaceT, void, TileCopyWH>;

    using VUpdateType = Catlass::Gemm::GemmType<InputT, Catlass::layout::zN>;
    using VUpdateLayout = typename VUpdateType::Layout;
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, InputT, Catlass::layout::ColumnMajor,
        InputT, Catlass::layout::zN,
        WorkspaceT, Catlass::layout::RowMajor>;
    using BlockMmadKV = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape,
        InputT, InputT, WorkspaceT, void, TileCopyKV>;
    using BlockMmadKVTail = Catlass::Gemm::Block::BlockMmadTla<
        TailDispatchPolicy, L1TileShape, L0TileShape,
        InputT, InputT, WorkspaceT, void, TileCopyKV>;

    using Offsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    __aicore__ inline void Init(
        GM_ADDR k, GM_ADDR w, GM_ADDR h, GM_ADDR cuSeqlens,
        GM_ADDR chunkIndices, GM_ADDR user, GM_ADDR tiling)
    {
        auto tilingData = reinterpret_cast<
            __gm__ ChunkGatedDeltaRuleFwdHTilingData*>(tiling);
        seqlen_ = tilingData->seqlen;
        kNumHead_ = tilingData->kNumHead;
        vNumHead_ = tilingData->vNumHead;
        kHeadDim_ = tilingData->kHeadDim;
        vHeadDim_ = tilingData->vHeadDim;
        chunkSize_ = tilingData->chunkSize;
        isVariedLen_ = tilingData->isVariedLen;
        shapeBatch_ = tilingData->shapeBatch;

        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(k));
        gmW_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(w));
        gmH_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(h));
        gmVWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ WorkspaceT*>(user + tilingData->vWorkspaceOffset));
        gmVUpdateWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ InputT*>(user + tilingData->vUpdateWorkspaceOffset));
        gmHWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ WorkspaceT*>(user + tilingData->hWorkspaceOffset));

        scheduler_.Init(cuSeqlens, chunkIndices, tiling, user);
    }

    __aicore__ inline void Process()
    {
        uint32_t workspaceBytes = chunkSize_ * scheduler_.vBlockSize *
            sizeof(InputT) * LOCAL_PING_PONG_STAGES;
        BlockMmadWH stage0Mmad(resource_, workspaceBytes);
        BlockMmadWHTail stage0TailMmad(resource_, workspaceBytes);
        BlockMmadKV stage2Mmad(resource_, workspaceBytes);
        BlockMmadKVTail stage2TailMmad(resource_, workspaceBytes);
        bool useBoundedMmad = isVariedLen_ || (seqlen_ % chunkSize_ != 0);

        auto wLayout = tla::MakeLayout<InputT, Catlass::layout::RowMajor>(
            shapeBatch_ * kNumHead_ * scheduler_.totalTokens, kHeadDim_);
        auto hLayout = tla::MakeLayout<InputT, Catlass::layout::RowMajor>(
            shapeBatch_ * vNumHead_ * scheduler_.totalChunks * kHeadDim_, vHeadDim_);
        auto kLayout = tla::MakeLayout<InputT, Catlass::layout::ColumnMajor>(
            kHeadDim_, shapeBatch_ * kNumHead_ * scheduler_.totalTokens);
        auto hWorkLayout = tla::MakeLayout<WorkspaceT, Catlass::layout::RowMajor>(
            kHeadDim_, scheduler_.vBlockSize);

        while (scheduler_.isRunning) {
            scheduler_.InitTasks();
            if (!scheduler_.isRunning) {
                break;
            }

            uint32_t windowId = scheduler_.GetWindowId();
            Catlass::Arch::CrossCoreWaitFlag(scheduler_.vec2Done[windowId]);

            const auto& firstHead = scheduler_.GetHeadTask(0);
            bool stage0UsesTail = useBoundedMmad &&
                firstHead.offset.blockTokens < chunkSize_;
            if (stage0UsesTail) {
                stage0TailMmad.preSetFlags();
            } else {
                stage0Mmad.preSetFlags();
            }
            for (uint32_t head = 0; head < scheduler_.GetHeadsInRound(); ++head) {
                const auto& headTask = scheduler_.GetHeadTask(head);
                if (scheduler_.HeadTaskIsDone(headTask)) {
                    continue;
                }
                bool cubeProduced = Stage0(
                    headTask, stage0Mmad, stage0TailMmad,
                    wLayout, hLayout, useBoundedMmad);
                if (cubeProduced) {
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        scheduler_.cube1Done[windowId]);
                } else {
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                        scheduler_.cube1Done[windowId]);
                }
            }
            if (stage0UsesTail) {
                stage0TailMmad.finalWaitFlags();
            } else {
                stage0Mmad.finalWaitFlags();
            }

            bool runStage2 = !scheduler_.HeadTaskIsDone(firstHead) &&
                scheduler_.NeedProcessStage2(firstHead);
            bool stage2UsesTail = runStage2 && useBoundedMmad &&
                firstHead.offset.blockTokens < chunkSize_;
            if (runStage2) {
                if (stage2UsesTail) {
                    stage2TailMmad.preSetFlags();
                } else {
                    stage2Mmad.preSetFlags();
                }
            }
            for (uint32_t head = 0; head < scheduler_.GetHeadsInRound(); ++head) {
                const auto& headTask = scheduler_.GetHeadTask(head);
                if (scheduler_.HeadTaskIsDone(headTask)) {
                    continue;
                }
                Catlass::Arch::CrossCoreWaitFlag(scheduler_.vec1Done[windowId]);
                bool cubeProduced = Stage2(
                    headTask, stage2Mmad, stage2TailMmad,
                    kLayout, hWorkLayout, useBoundedMmad);
                if (cubeProduced) {
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        scheduler_.cube2Done[windowId]);
                } else {
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                        scheduler_.cube2Done[windowId]);
                }
            }
            if (runStage2) {
                if (stage2UsesTail) {
                    stage2TailMmad.finalWaitFlags();
                } else {
                    stage2Mmad.finalWaitFlags();
                }
            }
        }

        Catlass::Arch::CrossCoreWaitFlag(scheduler_.vec2Done[0]);
        Catlass::Arch::CrossCoreWaitFlag(scheduler_.vec2Done[1]);
    }

private:
    template <class HeadTask, class WLayout, class HLayout>
    __aicore__ inline bool Stage0(
        const HeadTask& headTask, BlockMmadWH& mmad,
        BlockMmadWHTail& tailMmad, const WLayout& wLayout,
        const HLayout& hLayout, bool useBoundedMmad)
    {
        const Offsets& offsets = scheduler_.GetCurTaskOffsets(headTask);
        if (!useBoundedMmad && offsets.blockTokens < 16) {
            return false;
        }

        auto vLayout = tla::MakeLayout<WorkspaceT, Catlass::layout::RowMajor>(
            offsets.blockTokens, offsets.vBlockDim);
        auto tensorW = tla::MakeTensor(
            gmW_[offsets.wOffset], wLayout, Catlass::Arch::PositionGM{});
        auto tensorH = tla::MakeTensor(
            gmH_[offsets.hSrcOffset], hLayout, Catlass::Arch::PositionGM{});
        auto tensorV = tla::MakeTensor(
            gmVWorkspace_[offsets.vWorkOffset], vLayout,
            Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{
            offsets.blockTokens, offsets.vBlockDim, kHeadDim_};
        auto blockW = tla::GetTile(
            tensorW, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockH = tla::GetTile(
            tensorH, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockV = tla::GetTile(
            tensorV, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        if (useBoundedMmad && offsets.blockTokens < chunkSize_) {
            tailMmad(blockW, blockH, blockV, shape, Catlass::EmptyClass{}, true);
        } else {
            mmad(blockW, blockH, blockV, shape);
        }
        return true;
    }

    template <class HeadTask, class KLayout, class HLayout>
    __aicore__ inline bool Stage2(
        const HeadTask& headTask, BlockMmadKV& mmad,
        BlockMmadKVTail& tailMmad, const KLayout& kLayout,
        const HLayout& hLayout, bool useBoundedMmad)
    {
        if (!scheduler_.NeedProcessStage2(headTask)) {
            return true;
        }
        const Offsets& offsets = scheduler_.GetCurTaskOffsets(headTask);
        if (!useBoundedMmad && offsets.blockTokens < 16) {
            return false;
        }

        // The formal k input is raw k for GDN v1 and pre-scaled kg for KDA/GDN2.
        auto tensorK = tla::MakeTensor(
            gmK_[offsets.wkOffset], kLayout, Catlass::Arch::PositionGM{});
        auto vLayout = tla::MakeLayout<InputT, VUpdateLayout>(
            offsets.blockTokens, offsets.vBlockDim);
        auto tensorV = tla::MakeTensor(
            gmVUpdateWorkspace_[offsets.vWorkOffset], vLayout,
            Catlass::Arch::PositionGM{});
        auto tensorH = tla::MakeTensor(
            gmHWorkspace_[offsets.hWorkOffset], hLayout,
            Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{
            kHeadDim_, offsets.vBlockDim, offsets.blockTokens};
        auto blockK = tla::GetTile(
            tensorK, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockV = tla::GetTile(
            tensorV, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockH = tla::GetTile(
            tensorH, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        if (useBoundedMmad && offsets.blockTokens < chunkSize_) {
            tailMmad(blockK, blockV, blockH, shape, Catlass::EmptyClass{}, true);
        } else {
            mmad(blockK, blockV, blockH, shape);
        }
        return true;
    }

    uint32_t seqlen_;
    uint32_t kNumHead_;
    uint32_t vNumHead_;
    uint32_t kHeadDim_;
    uint32_t vHeadDim_;
    uint32_t chunkSize_;
    uint32_t isVariedLen_;
    uint32_t shapeBatch_;

    AscendC::GlobalTensor<InputT> gmK_;
    AscendC::GlobalTensor<InputT> gmW_;
    AscendC::GlobalTensor<InputT> gmH_;
    AscendC::GlobalTensor<WorkspaceT> gmVWorkspace_;
    AscendC::GlobalTensor<InputT> gmVUpdateWorkspace_;
    AscendC::GlobalTensor<WorkspaceT> gmHWorkspace_;

    CubeScheduler scheduler_;
    Catlass::Arch::Resource<ArchTag> resource_;
};

} // namespace GDN::FwdHStandalone

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_ARCH35_CUBE_H
