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

template <typename QKVT, typename GT>
class ChunkBwdDvLocalCube {
public:
    __aicore__ inline ChunkBwdDvLocalCube(){};
    __aicore__ inline void Process();

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                GM_ADDR d_v, GM_ADDR workspace, const ChunkBwdDvLocalTilingData *__restrict tilingData);
    AscendC::GlobalTensor<QKVT> qGm;
    AscendC::GlobalTensor<QKVT> kGm;
    AscendC::GlobalTensor<QKVT> dOGm;
    AscendC::GlobalTensor<QKVT> dVGm;
    AscendC::GlobalTensor<QKVT> workspaceGm;

    int64_t B;
    int64_t H;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkSize;
    int64_t chunkNumForT;
    int64_t chunkLenTail;
    int64_t coreLoops;
    int64_t blockNum;
    int64_t coreIdx;
    // int64_t maxChunkIndexForT;
    // int64_t chunkNumPreCore;
    // int64_t chunkNumTailCore;
    // int64_t preCoreNum;
    // int64_t tailCoreNum;
    // int64_t totalCoreNum;
    // int64_t chunkSize;
    // float scale;
    // bool isVariable;
    // int64_t batchCount;
    // int64_t chunkNumCurCore;
    // int64_t coreId;
    // int64_t strideQK;
    // int64_t strideDoDv;
    // int64_t strideOut;

    // AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    // AscendC::DataCopyPadExtParams<QKVT> qkvPadParams{false, 0, 0, 0};
    // AscendC::DataCopyPadExtParams<GT> gPadParams{false, 0, 0, 0};
};

template <typename QKVT, typename GT>
__aicore__ inline void ChunkBwdDvLocalCube<QKVT, GT>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR cu_seqlens,
                                                           GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace,
                                                           const ChunkBwdDvLocalTilingData *__restrict tilingData)
{
    qGm.SetGlobalBuffer((__gm__ QKVT *)q);
    kGm.SetGlobalBuffer((__gm__ QKVT *)k);
    dOGm.SetGlobalBuffer((__gm__ QKVT *)d_o);
    dVGm.SetGlobalBuffer((__gm__ QKVT *)d_v);
    workspaceGm.SetGlobalBuffer((__gm__ QKVT *)workspace);

    B = tilingData->b;
    H = tilingData->h;
    T = tilingData->t;
    K = tilingData->k;
    V = tilingData->v;
    chunkSize = tilingData->chunkSize;
    chunkNumForT = tilingData->chunkNumForT;
    chunkLenTail = T - (chunkNumForT - 1) * chunkSize;
    coreLoops = B * chunkNumForT;
    blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
    coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    // AscendC::printf("[参数打印] cube coreLoops = %d  \n", coreLoops);
    // AscendC::printf("[参数打印] cube blockNum = %d  \n", blockNum);
    // AscendC::printf("[参数打印] cube coreIdx = %d  \n", coreIdx);
}

template <typename QKVT, typename GT>
__aicore__ inline void ChunkBwdDvLocalCube<QKVT, GT>::Process()
{
    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
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
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(chunkSize, K);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(K, chunkSize);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(chunkSize, chunkSize);
        for (int64_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            int64_t curBatchId = static_cast<int64_t>(loopIdx) / chunkNumForT;
            int64_t curChunkId = (static_cast<int64_t>(loopIdx) % chunkNumForT);
            int64_t curTokenId = curChunkId * chunkSize;
            int64_t chunkLen = curChunkId == chunkNumForT - 1 ? chunkLenTail : chunkSize;
            // AscendC::printf("[参数打印] cube curBatchId = %d  \n", curBatchId);
            // AscendC::printf("[参数打印] cube curChunkId = %d  \n", curChunkId);
            // AscendC::printf("[参数打印] cube curTokenId = %d  \n", curTokenId);
            // AscendC::printf("[参数打印] cube chunkLen = %d  \n", chunkLen);

            Catlass::GemmCoord actualBlockShape{static_cast<uint32_t>(chunkLen), static_cast<uint32_t>(chunkLen),
                                                static_cast<uint32_t>(K)};
            for (int hIndex = 0; hIndex < H; hIndex++) {
                auto tensorA = tla::MakeTensor(kGm[curBatchId * H * T * K + hIndex * T * K + curTokenId * K], layoutA,
                                               Catlass::Arch::PositionGM{});
                auto tensorB = tla::MakeTensor(qGm[curBatchId * H * T * K + hIndex * T * K + curTokenId * K], layoutB,
                                               Catlass::Arch::PositionGM{});
                auto tensorC = tla::MakeTensor(
                    workspaceGm[curBatchId * H * T * chunkSize + hIndex * T * chunkSize + curTokenId * chunkSize],
                    layoutC, Catlass::Arch::PositionGM{});
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_1);
                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB =
                    GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockC =
                    GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                // AscendC::printf("[tensor 打印]  tensorBlockC \n");
                // AscendC::DumpTensor(tensorBlockC.data(), 5, 128);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_3);
            }
        }
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

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(chunkSize, chunkSize);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(chunkSize, V);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(chunkSize, V);
        for (int64_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            int64_t curBatchId = static_cast<int64_t>(loopIdx) / chunkNumForT;
            int64_t curChunkId = (static_cast<int64_t>(loopIdx) % chunkNumForT);
            int64_t curTokenId = curChunkId * chunkSize;
            int64_t chunkLen = curChunkId == chunkNumForT - 1 ? chunkLenTail : chunkSize;
            Catlass::GemmCoord actualBlockShape{static_cast<uint32_t>(chunkLen), static_cast<uint32_t>(V),
                                                static_cast<uint32_t>(chunkLen)};
            for (int hIndex = 0; hIndex < H; hIndex++) {
                auto tensorA = tla::MakeTensor(
                    workspaceGm[curBatchId * H * T * chunkSize + hIndex * T * chunkSize + curTokenId * chunkSize],
                    layoutA, Catlass::Arch::PositionGM{});
                auto tensorB = tla::MakeTensor(dOGm[curBatchId * H * T * V + hIndex * T * V + curTokenId * V], layoutB,
                                               Catlass::Arch::PositionGM{});
                auto tensorC = tla::MakeTensor(dVGm[curBatchId * H * T * V + hIndex * T * V + curTokenId * V], layoutC,
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