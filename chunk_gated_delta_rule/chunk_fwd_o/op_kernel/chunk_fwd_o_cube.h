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
 * \file chunk_fwd_o_cube.h
 * \brief Cube (AIC) operations for chunk_fwd_o
 */

#ifndef CHUNK_FWD_O_CUBE_H
#define CHUNK_FWD_O_CUBE_H

#include "chunk_fwd_o_common.h"
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
class ChunkFwdOCubeProcess {
public:
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutRowMajor = Catlass::layout::RowMajor;
    using LayoutColMajor = Catlass::layout::ColumnMajor;

    // GEMM type definitions
    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true>;
    using L1TileShapeTla = Shape<_128, _128, _128>;
    using L0TileShapeTla = L1TileShapeTla;

    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementType, LayoutRowMajor, ElementType, LayoutColMajor, half, LayoutRowMajor>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, ElementType, ElementType, half, void, TileCopyQK>;

    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementType, LayoutRowMajor, ElementType, LayoutRowMajor, half, LayoutRowMajor>;
    using BlockMmadQH = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, ElementType, ElementType, half, void, TileCopyQH>;

    using TileCopyAttenVNEW = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementType, LayoutRowMajor, ElementType, LayoutRowMajor, half, LayoutRowMajor>;
    using BlockMmadAttenVNEW = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, ElementType, ElementType, half, void, TileCopyAttenVNEW>;

    uint32_t shapeBatch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    float scale;
    uint32_t vWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    uint32_t attnWorkspaceOffset;
    uint32_t aftermaskWorkspaceOffset;
    uint32_t maskWorkspaceOffset;

    AscendC::GlobalTensor<ElementType> gmQ;
    AscendC::GlobalTensor<ElementType> gmK;
    AscendC::GlobalTensor<ElementType> gmV;
    AscendC::GlobalTensor<ElementType> gmH;
    AscendC::GlobalTensor<float> gmG;
    AscendC::GlobalTensor<ElementType> gmO;
    AscendC::GlobalTensor<half> gmVWorkspace;
    AscendC::GlobalTensor<half> gmHWorkspace;
    AscendC::GlobalTensor<half> gmAttnWorkspace;
    AscendC::GlobalTensor<ElementType> gmAftermaskWorkspace;
    AscendC::GlobalTensor<bool> gmMask;

    BlockSchedulerGdnFwdOCube cubeBlockScheduler;

    __aicore__ inline ChunkFwdOCubeProcess() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
        GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, GM_ADDR tiling, GM_ADDR user) {

        __gm__ ChunkFwdOTilingData *__restrict tilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);

        shapeBatch = tilingData->shapeBatch;
        seqlen = tilingData->seqlen;
        kNumHead = tilingData->kNumHead;
        vNumHead = tilingData->vNumHead;
        kHeadDim = tilingData->kHeadDim;
        vHeadDim = tilingData->vHeadDim;
        scale = tilingData->scale;
        chunkSize = tilingData->chunkSize;
        vWorkspaceOffset = tilingData->vWorkspaceOffset;
        hWorkspaceOffset = tilingData->hWorkspaceOffset;
        attnWorkspaceOffset = tilingData->attnWorkspaceOffset;
        aftermaskWorkspaceOffset = tilingData->aftermaskWorkspaceOffset;
        maskWorkspaceOffset = tilingData->maskWorkspaceOffset;

        gmQ.SetGlobalBuffer((__gm__ ElementType *)q);
        gmK.SetGlobalBuffer((__gm__ ElementType *)k);
        gmV.SetGlobalBuffer((__gm__ ElementType *)v);
        gmH.SetGlobalBuffer((__gm__ ElementType *)h);
        gmG.SetGlobalBuffer((__gm__ float *)g);
        gmO.SetGlobalBuffer((__gm__ ElementType *)o);
        gmVWorkspace.SetGlobalBuffer((__gm__ half *)(user + vWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ half *)(user + hWorkspaceOffset));
        gmAttnWorkspace.SetGlobalBuffer((__gm__ half *)(user + attnWorkspaceOffset));
        gmAftermaskWorkspace.SetGlobalBuffer((__gm__ ElementType *)(user + aftermaskWorkspaceOffset));
        gmMask.SetGlobalBuffer((__gm__ bool *)(user + maskWorkspaceOffset));

        cubeBlockScheduler.Init(cu_seqlens, chunk_offsets, tiling);
    }

    __aicore__ inline void Process() {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        Arch::Resource<ArchTag> resource;
        BlockMmadQK blockMmadQK(resource);
        BlockMmadQH blockMmadQH(resource);
        BlockMmadAttenVNEW blockMmadAttenVNEW(resource);

        auto qLayout = tla::MakeLayout<ElementType, LayoutRowMajor>(shapeBatch * kNumHead * seqlen, kHeadDim);
        auto kLayout = tla::MakeLayout<ElementType, LayoutColMajor>(kHeadDim, shapeBatch * kNumHead * seqlen);
        auto hLayout = tla::MakeLayout<ElementType, LayoutRowMajor>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
        auto ointerLayout = tla::MakeLayout<half, LayoutRowMajor>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
        auto vnewLayout = tla::MakeLayout<ElementType, LayoutRowMajor>(shapeBatch * vNumHead * seqlen, vHeadDim);

        bool needRun = false;
        bool isFirstC3 = true;

        while (cubeBlockScheduler.isRunning) {
            cubeBlockScheduler.InitTask();

            // Cube1: Q × K → Attn
            if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);

                GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                int64_t cube1OffsetQ = cube1Offsets.qkOffset;
                int64_t cube1OffsetK = cube1Offsets.qkOffset;
                int64_t cube1OffsetAttn = cube1Offsets.attnWorkOffset;
                auto attenLayout = tla::MakeLayout<half, LayoutRowMajor>(coreNum * chunkSize * PING_PONG_STAGES, cube1Offsets.blockTokens);
                auto tensorQ = tla::MakeTensor(gmQ[cube1OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK[cube1OffsetK], kLayout, Catlass::Arch::PositionGM{});
                auto tensorAttn = tla::MakeTensor(gmAttnWorkspace[cube1OffsetAttn], attenLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
                auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                auto tensorBlockAttn = GetTile(tensorAttn, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                blockMmadQK.preSetFlags();
                blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                blockMmadQK.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done);
            }

            // Cube2: Q × H → O_inter
            if (needRun && coreIdx < coreNum) {
                if(!cubeBlockScheduler.isRunning) Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);

                GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                int64_t cube2OffsetQ = cube2Offsets.qkOffset;
                int64_t cube2OffsetH = cube2Offsets.hOffset;
                int64_t cube2OffsetHWork = cube2Offsets.hvWorkOffset;
                auto tensorQ = tla::MakeTensor(gmQ[cube2OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                auto tensorH = tla::MakeTensor(gmH[cube2OffsetH], hLayout, Catlass::Arch::PositionGM{});
                auto tensorHWork = tla::MakeTensor(gmHWorkspace[cube2OffsetHWork], ointerLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube2Shape{cube2Offsets.blockTokens, vHeadDim, kHeadDim};
                auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                auto tensorBlockHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                blockMmadQH.preSetFlags();
                blockMmadQH(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);
                blockMmadQH.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done);
            }

            // Cube3: AttnMasked × V → V_work
            if (needRun && coreIdx < coreNum) {
                GDNFwdOOffsets& cube3Offsets = cubeBlockScheduler.GetCube23Offsets();

                if(isFirstC3) Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);

                int64_t cube3OffsetAttnMask = cube3Offsets.attnWorkOffset;
                int64_t cube3OffsetV = cube3Offsets.ovOffset;
                int64_t cube3OffsetVWork = cube3Offsets.hvWorkOffset;
                auto attenLayout = tla::MakeLayout<half, LayoutRowMajor>(coreNum * chunkSize * PING_PONG_STAGES, cube3Offsets.blockTokens);
                auto tensorAttnMask = tla::MakeTensor(gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout, Catlass::Arch::PositionGM{});
                auto tensorV = tla::MakeTensor(gmV[cube3OffsetV], vnewLayout, Catlass::Arch::PositionGM{});
                auto tensorVWork = tla::MakeTensor(gmVWorkspace[cube3OffsetVWork], ointerLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube3Shape{cube3Offsets.blockTokens, vHeadDim, cube3Offsets.blockTokens};
                auto tensorBlockAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                auto tensorBlockVWork = GetTile(tensorVWork, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));
                blockMmadAttenVNEW.preSetFlags();
                blockMmadAttenVNEW(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
                blockMmadAttenVNEW.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube3Done);
                isFirstC3 = false;
            }
            needRun = true;
        }
        if (coreIdx < coreNum) {
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);
        }
    }
};

#endif // CHUNK_FWD_O_CUBE_H
