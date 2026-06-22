/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dv_local_cube_fix.h
 * \brief
 */

#ifndef CHUNK_BWD_DV_LOCAL_CUBE_FIX_H
#define CHUNK_BWD_DV_LOCAL_CUBE_FIX_H

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
#endif

#include <type_traits>
#include "chunk_bwd_dv_local_struct.h"
#include "chunk_bwd_dv_local_common.h"
using namespace tla;
namespace GDN {

template <typename QKVT, typename GT, typename Strategy, int V>
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

    int64_t H_qk;
    int64_t H_do;
    int64_t hRatio;
    int64_t headBufNum;
    int64_t T;
    int64_t K;
    int64_t coreLoops;
    int64_t blockNum;
    int64_t coreIdx;
};

template <typename QKVT, typename GT, typename Strategy, int V>
__aicore__ inline void
ChunkBwdDvLocalCube<QKVT, GT, Strategy, V>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR cu_seqlens,
                                              GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace,
                                              const ChunkBwdDvLocalTilingData *__restrict tilingData)
{
    qGm.SetGlobalBuffer((__gm__ QKVT *)q);
    kGm.SetGlobalBuffer((__gm__ QKVT *)k);
    dOGm.SetGlobalBuffer((__gm__ QKVT *)d_o);
    dVGm.SetGlobalBuffer((__gm__ QKVT *)d_v);
    workspaceGm.SetGlobalBuffer((__gm__ QKVT *)workspace);

    H_qk = tilingData->hQk;
    H_do = tilingData->hDo;
    hRatio = tilingData->hRatio;
    headBufNum = tilingData->headBufNum;
    T = tilingData->t;
    K = tilingData->k;
    coreLoops = tilingData->b * strategy.chunkNumForT;
    blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
    coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
}

template <typename QKVT, typename GT, typename Strategy, int V>
__aicore__ inline void ChunkBwdDvLocalCube<QKVT, GT, Strategy, V>::Process()
{
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ArchTag = Catlass::Arch::Ascend950;
    #else
        using ArchTag = Catlass::Arch::AtlasA2;
    #endif    
    using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false>;
    using L1TileShapeQK = Shape<_128, _128, _128>;
    using L0TileShapeQK = Shape<_128, _128, _128>;
    using ElementA = QKVT;
    using ElementB = QKVT;
    using ElementC = QKVT;
    Catlass::Arch::Resource<ArchTag> resource;
    using L1TileShapeV = typename std::conditional<V == 128, Shape<_128, _128, _128>, Shape<_128, _256, _64>>::type;
    using L0TileShapeV = typename std::conditional<V == 128, Shape<_128, _128, _128>, Shape<_128, _256, _64>>::type;
    using LayoutTagA = Catlass::layout::RowMajor;
    using LayoutTagB = Catlass::layout::ColumnMajor;
    using LayoutTagC = Catlass::layout::RowMajor;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
                                                            ElementC, LayoutTagC>;
    using BlockMmadQK = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShapeQK, L0TileShapeQK, ElementA,
                                                         ElementB, ElementC, void, TileCopy>;

    using LayoutTagA_V = Catlass::layout::RowMajor;
    using LayoutTagB_V = Catlass::layout::RowMajor;
    using LayoutTagC_V = Catlass::layout::RowMajor;
    using TileCopyV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA_V, ElementB, LayoutTagB_V,
                                                             ElementC, LayoutTagC_V>;
    using BlockMmadV = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShapeV, L0TileShapeV, ElementA,
                                                           ElementB, ElementC, void, TileCopyV>;

    auto layoutA_QK = tla::MakeLayout<ElementA, Catlass::layout::RowMajor>(strategy.chunkSize, K);
    auto layoutB_QK = tla::MakeLayout<ElementB, Catlass::layout::ColumnMajor>(K, strategy.chunkSize);
    auto layoutC_QK = tla::MakeLayout<ElementC, Catlass::layout::RowMajor>(strategy.chunkSize, strategy.chunkSize);
    auto layoutA_V = tla::MakeLayout<ElementA, LayoutTagA_V>(strategy.chunkSize, strategy.chunkSize);
    auto layoutB_V = tla::MakeLayout<ElementB, LayoutTagB_V>(strategy.chunkSize, V);
    auto layoutC_V = tla::MakeLayout<ElementC, LayoutTagC_V>(strategy.chunkSize, V);

    IndexResult indexResult;
    int64_t coreBaseOffset = coreIdx * headBufNum * strategy.chunkSize * strategy.chunkSize;
    for (int64_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        int64_t curBatchId = static_cast<int64_t>(loopIdx) / strategy.chunkNumForT;
        strategy.calculate(loopIdx, indexResult);
        Catlass::GemmCoord actualBlockShapeQK{static_cast<uint32_t>(indexResult.chunkLen),
                                              static_cast<uint32_t>(indexResult.chunkLen),
                                              static_cast<uint32_t>(K)};
        Catlass::GemmCoord actualBlockShapeV{static_cast<uint32_t>(indexResult.chunkLen),
                                             static_cast<uint32_t>(V),
                                             static_cast<uint32_t>(indexResult.chunkLen)};
        for (int64_t qkHead = 0; qkHead < H_qk; qkHead++) {
            int64_t p1Slot = qkHead % 2;
            {
                BlockMmadQK blockMmadQK(resource);
                auto tensorA =
                    tla::MakeTensor(kGm[curBatchId * H_qk * T * K + qkHead * T * K + indexResult.curTokenId * K], layoutA_QK,
                                    Catlass::Arch::PositionGM{});
                auto tensorB =
                    tla::MakeTensor(qGm[curBatchId * H_qk * T * K + qkHead * T * K + indexResult.curTokenId * K], layoutB_QK,
                                    Catlass::Arch::PositionGM{});
                auto tensorC = tla::MakeTensor(
                    workspaceGm[coreBaseOffset + p1Slot * strategy.chunkSize * strategy.chunkSize],
                    layoutC_QK, Catlass::Arch::PositionGM{});
                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShapeQK.m(), actualBlockShapeQK.k()));
                auto tensorBlockB =
                    GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShapeQK.k(), actualBlockShapeQK.n()));
                auto tensorBlockC =
                    GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShapeQK.m(), actualBlockShapeQK.n()));
                blockMmadQK(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShapeQK);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_3);
            for (int64_t doGroup = 0; doGroup < hRatio; doGroup++) {
                AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_1);
            }
            {
                BlockMmadV blockMmadV(resource);
                for (int64_t doGroup = 0; doGroup < hRatio; doGroup++) {
                    int64_t doHead = qkHead * hRatio + doGroup;
                    int64_t gatedSlot = 2 + (qkHead % 2) * hRatio + doGroup;
                    auto tensorA_V = tla::MakeTensor(
                        workspaceGm[coreBaseOffset + gatedSlot * strategy.chunkSize * strategy.chunkSize],
                        layoutA_V, Catlass::Arch::PositionGM{});
                    auto tensorB_V =
                        tla::MakeTensor(dOGm[curBatchId * H_do * T * V + doHead * T * V + indexResult.curTokenId * V],
                                        layoutB_V, Catlass::Arch::PositionGM{});
                    auto tensorC_V =
                        tla::MakeTensor(dVGm[curBatchId * H_do * T * V + doHead * T * V + indexResult.curTokenId * V],
                                        layoutC_V, Catlass::Arch::PositionGM{});
                    auto tensorBlockA_V =
                        GetTile(tensorA_V, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShapeV.m(), actualBlockShapeV.k()));
                    auto tensorBlockB_V =
                        GetTile(tensorB_V, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShapeV.k(), actualBlockShapeV.n()));
                    auto tensorBlockC_V =
                        GetTile(tensorC_V, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShapeV.m(), actualBlockShapeV.n()));
                    blockMmadV(tensorBlockA_V, tensorBlockB_V, tensorBlockC_V, actualBlockShapeV);
                }
            }
        }
    }
}

} // namespace GDN
#endif // CHUNK_BWD_DV_LOCAL_CUBE_FIX_H
