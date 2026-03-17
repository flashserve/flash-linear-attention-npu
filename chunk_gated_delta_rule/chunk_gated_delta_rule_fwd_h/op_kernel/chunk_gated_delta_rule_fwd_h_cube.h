/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h_cube.h
 * \brief Cube (AIC) operations for chunk_gated_delta_rule_fwd_h
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_CUBE_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_CUBE_H

#include "chunk_gated_delta_rule_fwd_h_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

using namespace Catlass;

template <typename ElementType>
class ChunkGatedDeltaRuleFwdHCubeProcess {
public:
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutRowMajor = Catlass::layout::RowMajor;
    using LayoutColMajor = Catlass::layout::ColumnMajor;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true>;
    using L1TileShapeTla = Shape<_128, _128, _128>;
    using L0TileShapeTla = L1TileShapeTla;

    // Cube1: W @ H → V_work (RowMajor × RowMajor)
    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementType, LayoutRowMajor, ElementType, LayoutRowMajor, half, LayoutRowMajor>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, ElementType, ElementType, half, void, TileCopyWH>;

    // Cube2: K.T @ V_work → H_work (ColumnMajor × RowMajor)
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementType, LayoutColMajor, ElementType, LayoutRowMajor, half, LayoutRowMajor>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, ElementType, ElementType, half, void, TileCopyKV>;

    using ElementVWork = half;
    using ElementHWork = half;

    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    uint32_t vWorkspaceOffset;
    uint32_t hWorkspaceOffset;

    AscendC::GlobalTensor<ElementType> gmK;
    AscendC::GlobalTensor<ElementType> gmW;
    AscendC::GlobalTensor<ElementType> gmH;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspaceHalf;
    AscendC::GlobalTensor<ElementType> gmVWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspaceHalf;

    BlockSchedulerGdnFwdHCube cubeBlockScheduler;

    __aicore__ inline ChunkGatedDeltaRuleFwdHCubeProcess() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR inital_state,
        GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
        GM_ADDR tiling, GM_ADDR user) {

        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict tilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = tilingData->batch;
        seqlen = tilingData->seqlen;
        kNumHead = tilingData->kNumHead;
        vNumHead = tilingData->vNumHead;
        kHeadDim = tilingData->kHeadDim;
        vHeadDim = tilingData->vHeadDim;
        chunkSize = tilingData->chunkSize;
        useInitialState = tilingData->useInitialState;
        storeFinalState = tilingData->storeFinalState;
        isVariedLen = tilingData->isVariedLen;
        shapeBatch = tilingData->shapeBatch;
        tokenBatch = tilingData->tokenBatch;
        vWorkspaceOffset = tilingData->vWorkspaceOffset;
        hWorkspaceOffset = tilingData->hWorkspaceOffset;

        gmK.SetGlobalBuffer((__gm__ ElementType *)k);
        gmW.SetGlobalBuffer((__gm__ ElementType *)w);
        gmH.SetGlobalBuffer((__gm__ ElementType *)h);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementType *)(user + vWorkspaceOffset));
        gmVWorkspaceHalf.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmHWorkspaceHalf.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));

        cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling);
    }

    __aicore__ inline void Process() {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        Arch::Resource<ArchTag> resource;
        BlockMmadWH blockMmadWH(resource);
        BlockMmadKV blockMmadKV(resource);

        auto wLayout = tla::MakeLayout<ElementType, LayoutRowMajor>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
        auto hLayout = tla::MakeLayout<ElementType, LayoutRowMajor>(shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, vHeadDim);
        auto vLayout = tla::MakeLayout<ElementVWork, LayoutRowMajor>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);

        auto kLayout = tla::MakeLayout<ElementType, LayoutColMajor>(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens);
        auto vworkLayout = tla::MakeLayout<ElementVWork, LayoutRowMajor>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
        auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutRowMajor>(coreNum * kHeadDim * PING_PONG_STAGES, vHeadDim);

        bool needRun = false;

        while (cubeBlockScheduler.isRunning) {
            cubeBlockScheduler.InitTask();
            // step 1: v_work = w @ h[i]
            GDNFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
            if (cube1Offsets.chunkIdx != 0) {
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);
            } else {
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
            }
            if (!cube1Offsets.isDummyHead) {
                int64_t cube1OffsetW = cube1Offsets.wOffset;
                int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                int64_t cube1OffsetVwork = cube1Offsets.vWorkOffset;
                auto tensorW = tla::MakeTensor(gmW[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                auto tensorH = tla::MakeTensor(gmH[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                auto tensorV = tla::MakeTensor(gmVWorkspaceHalf[cube1OffsetVwork], vLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube1Shape {cube1Offsets.blockTokens, vHeadDim, kHeadDim};
                auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                blockMmadWH.preSetFlags();
                blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);
                blockMmadWH.finalWaitFlags();
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done);

            GDNFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetCube2Offsets();
            if (!cube2Offsets.isFinalState && needRun) {
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
                // step 3: h[i+1] = k.T @ v_work
                if (!cube2Offsets.isDummyHead) {
                    int64_t cube2OffsetK = cube2Offsets.wkOffset;
                    int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                    int64_t cube2OffsetH = cube2Offsets.hWorkOffset;
                    auto tensorK = tla::MakeTensor(gmK[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{});
                    auto tensorVwork = tla::MakeTensor(gmVWorkspace[cube2OffsetVwork], vworkLayout, Catlass::Arch::PositionGM{});
                    auto tensorHwork = tla::MakeTensor(gmHWorkspaceHalf[cube2OffsetH], hworkLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube2Shape{kHeadDim, vHeadDim, cube2Offsets.blockTokens};
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                    auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                    auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                    blockMmadKV.preSetFlags();
                    blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape);
                    blockMmadKV.finalWaitFlags();
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done);
            }
            needRun = true;
        }
        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
    }
};

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_CUBE_H
