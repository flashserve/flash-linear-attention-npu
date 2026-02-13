/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_bwd_dv_local_cube_fix.h
 * \brief
 */

#ifndef CHUNK_BWD_DV_LOCAL_CUBE_FIX_H
#define CHUNK_BWD_DV_LOCAL_CUBE_FIX_H

#include "kernel_operator.h"

#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "chunk_bwd_dv_local_common.h"
namespace GDN {

template <typename QKVT, typename GT, typename Strategy>
class ChunkBwdDvLocalCube {
private:
    Strategy strategy;

public:
    __aicore__ inline ChunkBwdDvLocalCube(const Strategy &s) : strategy(s)
    {
    }
    __aicore__ inline void Process();

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                GM_ADDR d_v, GM_ADDR workspace, const ChunkBwdDvLocalTilingData *__restrict tilingData);
    AscendC::GlobalTensor<QKVT> qGm;
    AscendC::GlobalTensor<QKVT> kGm;
    AscendC::GlobalTensor<QKVT> dOGm;
    AscendC::GlobalTensor<QKVT> dVGm;
    AscendC::GlobalTensor<QKVT> workspaceGm;

    int64_t H;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t coreLoops;
    int64_t blockNum;
    int64_t coreIdx;
};

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void
ChunkBwdDvLocalCube<QKVT, GT, Strategy>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR cu_seqlens,
                                              GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace,
                                              const ChunkBwdDvLocalTilingData *__restrict tilingData)
{
    qGm.SetGlobalBuffer((__gm__ QKVT *)q);
    kGm.SetGlobalBuffer((__gm__ QKVT *)k);
    dOGm.SetGlobalBuffer((__gm__ QKVT *)d_o);
    dVGm.SetGlobalBuffer((__gm__ QKVT *)d_v);
    workspaceGm.SetGlobalBuffer((__gm__ QKVT *)workspace);

    H = tilingData->h;
    T = tilingData->t;
    K = tilingData->k;
    V = tilingData->v;
    coreLoops = tilingData->b * strategy.chunkNumForT;
    blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
    coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalCube<QKVT, GT, Strategy>::Process()
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = Shape<_128, _128, _128>;
    using L0TileShape = Shape<_128, _128, _128>;
    using ElementA = QKVT;
    using ElementB = QKVT;
    using ElementC = QKVT;
    Catlass::Arch::Resource<ArchTag> resource;
    // k @ q^T
    {
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
                                                                ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA,
                                                             ElementB, ElementC, void, TileCopy>;

        BlockMmad blockMmad(resource);
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(strategy.chunkSize, K);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(K, strategy.chunkSize);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(strategy.chunkSize, strategy.chunkSize);
        for (int64_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            int64_t curBatchId = static_cast<int64_t>(loopIdx) / strategy.chunkNumForT;
            IndexResult indexResult = strategy.calculate(loopIdx);
            Catlass::GemmCoord actualBlockShape{static_cast<uint32_t>(indexResult.chunkLen),
                                                static_cast<uint32_t>(indexResult.chunkLen), static_cast<uint32_t>(K)};
            for (int hIndex = 0; hIndex < H; hIndex++) {
                auto tensorA =
                    tla::MakeTensor(kGm[curBatchId * H * T * K + hIndex * T * K + indexResult.curTokenId * K], layoutA,
                                    Catlass::Arch::PositionGM{});
                auto tensorB =
                    tla::MakeTensor(qGm[curBatchId * H * T * K + hIndex * T * K + indexResult.curTokenId * K], layoutB,
                                    Catlass::Arch::PositionGM{});
                auto tensorC = tla::MakeTensor(
                    workspaceGm[curBatchId * H * T * strategy.chunkSize + hIndex * T * strategy.chunkSize +
                                indexResult.curTokenId * strategy.chunkSize],
                    layoutC, Catlass::Arch::PositionGM{});
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_1);
                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB =
                    GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockC =
                    GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_3);
            }
        }
        AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_1);
        AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_1);
        AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_1);
        AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_1);
    }
    AscendC::SyncAll<false>();
    // v @ d_o
    {
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
                                                                ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA,
                                                             ElementB, ElementC, void, TileCopy>;

        BlockMmad blockMmad(resource);

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(strategy.chunkSize, strategy.chunkSize);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(strategy.chunkSize, V);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(strategy.chunkSize, V);
        for (int64_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            int64_t curBatchId = static_cast<int64_t>(loopIdx) / strategy.chunkNumForT;
            IndexResult indexResult = strategy.calculate(loopIdx);
            Catlass::GemmCoord actualBlockShape{static_cast<uint32_t>(indexResult.chunkLen), static_cast<uint32_t>(V),
                                                static_cast<uint32_t>(indexResult.chunkLen)};
            for (int hIndex = 0; hIndex < H; hIndex++) {
                auto tensorA = tla::MakeTensor(
                    workspaceGm[curBatchId * H * T * strategy.chunkSize + hIndex * T * strategy.chunkSize +
                                indexResult.curTokenId * strategy.chunkSize],
                    layoutA, Catlass::Arch::PositionGM{});
                auto tensorB =
                    tla::MakeTensor(dOGm[curBatchId * H * T * V + hIndex * T * V + indexResult.curTokenId * V], layoutB,
                                    Catlass::Arch::PositionGM{});
                auto tensorC =
                    tla::MakeTensor(dVGm[curBatchId * H * T * V + hIndex * T * V + indexResult.curTokenId * V], layoutC,
                                    Catlass::Arch::PositionGM{});
                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB =
                    GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockC =
                    GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
            }
        }
    }
}

} // namespace GDN
#endif // CHUNK_BWD_DV_LOCAL_CUBE_FIX_H