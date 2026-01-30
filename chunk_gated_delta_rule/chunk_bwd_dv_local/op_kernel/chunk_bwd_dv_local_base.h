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
 * \file chunk_bwd_dv_local_base.h
 * \brief
 */

#ifndef CHUNK_BWD_DV_LOCAL_BASE_H
#define CHUNK_BWD_DV_LOCAL_BASE_H

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
namespace GDN {
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t SIZE_FLOAT = 4;

__aicore__ inline int64_t CeilDiv(int64_t dividend, int64_t divisor)
{
    if (unlikely(divisor == 0)) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

__aicore__ inline void MTE2ToVSync()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
}
template <typename QKVT, typename GT>
class ChunkBwdDvLocalBase {
public:
    __aicore__ inline ChunkBwdDvLocalBase(){};
    __aicore__ inline void Process();
    __aicore__ inline void ProcessForBatch(int64_t curStartChunkIdForBatch, int64_t curEndChunkIdForBatch,
                                           int64_t curBatchId);
    __aicore__ inline void cubeProcess(int64_t chunkLen, int64_t curChunkId, int64_t curBatchId);
    __aicore__ inline void vectorProcess(int64_t chunkLen, int64_t curChunkId, int64_t curBatchId);
    __aicore__ inline void ParamsInit(const ChunkBwdDvLocalTilingData *__restrict tilingData);
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g, GM_ADDR upper_tri_matrix,
                                GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace,
                                const ChunkBwdDvLocalTilingData *__restrict tilingData, AscendC::TPipe *pipe);

    AscendC::TPipe *pipe_;
    AscendC::GlobalTensor<QKVT> qGm;
    AscendC::GlobalTensor<QKVT> kGm;
    AscendC::GlobalTensor<QKVT> dOGm;
    AscendC::GlobalTensor<GT> gGm;
    AscendC::GlobalTensor<uint8_t> triMatrixGm;
    AscendC::GlobalTensor<int64_t> cuSeqlensGm;
    AscendC::GlobalTensor<int64_t> chunkIndicesGm;
    AscendC::GlobalTensor<QKVT> dVGm;
    AscendC::GlobalTensor<QKVT> workspaceGm;
    AscendC::GlobalTensor<QKVT> workspace2Gm;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gTQueIn;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> kqTQueIn;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFp32TBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFactorTBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kqFp32TBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> kqTQueOut;

    int64_t b;
    int64_t h;
    int64_t t;
    int64_t k;
    int64_t v;
    int64_t chunkNumForT;
    int64_t maxChunkIndexForT;
    int64_t chunkNumPreCore;
    int64_t chunkNumTailCore;
    int64_t preCoreNum;
    int64_t tailCoreNum;
    int64_t totalCoreNum;
    int64_t chunkSize;
    float scale;
    bool isVariable;
    int64_t batchCount;
    int64_t chunkNumCurCore;
    int64_t coreId;
    int64_t strideQK;
    int64_t strideDoDv;
    int64_t strideOut;
};

template <typename QKVT, typename GT>
__aicore__ inline void ChunkBwdDvLocalBase<QKVT, GT>::ParamsInit(const ChunkBwdDvLocalTilingData *__restrict tilingData)
{
    b = tilingData->b;
    h = tilingData->h;
    t = tilingData->t;
    k = tilingData->k;
    v = tilingData->v;
    chunkNumForT = tilingData->chunkNumForT;
    maxChunkIndexForT = chunkNumForT - 1;
    chunkNumPreCore = tilingData->chunkNumPreCore;
    chunkNumTailCore = tilingData->chunkNumTailCore;
    preCoreNum = tilingData->preCoreNum;
    tailCoreNum = tilingData->tailCoreNum;
    totalCoreNum = tilingData->totalCoreNum;
    chunkSize = tilingData->chunkSize;
    scale = tilingData->scale;
    isVariable = tilingData->isVariable;
    batchCount = b * h;

    coreId = AscendC::GetBlockIdx();
    chunkNumCurCore = coreId >= preCoreNum ? chunkNumTailCore : chunkNumPreCore;
    strideQK = t * k;
    strideDoDv = t * v;
    strideOut = chunkSize * t;
}

template <typename QKVT, typename GT>
__aicore__ inline void
ChunkBwdDvLocalBase<QKVT, GT>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g, GM_ADDR upper_tri_matrix,
                                    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace,
                                    const ChunkBwdDvLocalTilingData *__restrict tilingData, AscendC::TPipe *pipe)
{
    ParamsInit(tilingData);

    pipe_ = pipe;
    qGm.SetGlobalBuffer((__gm__ QKVT *)q);
    kGm.SetGlobalBuffer((__gm__ QKVT *)k);
    dOGm.SetGlobalBuffer((__gm__ QKVT *)d_o);
    gGm.SetGlobalBuffer((__gm__ GT *)g);
    triMatrixGm.SetGlobalBuffer((__gm__ uint8_t *)upper_tri_matrix);
    cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
    chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunk_indices);
    dVGm.SetGlobalBuffer((__gm__ QKVT *)d_v);
    workspaceGm.SetGlobalBuffer((__gm__ QKVT *)workspace);
    workspace2Gm.SetGlobalBuffer((__gm__ QKVT *)workspace + b * h * t * chunkSize);

    pipe_->InitBuffer(gTQueIn, BUFFER_NUM, chunkSize * sizeof(GT));
    pipe_->InitBuffer(kqTQueIn, BUFFER_NUM, chunkSize * sizeof(QKVT));
    pipe_->InitBuffer(kqTQueOut, BUFFER_NUM, chunkSize * sizeof(QKVT));
    pipe_->InitBuffer(kqFp32TBuf, chunkSize * SIZE_FLOAT);
    pipe_->InitBuffer(gFp32TBuf, chunkSize * SIZE_FLOAT);
    pipe_->InitBuffer(gFactorTBuf, chunkSize * SIZE_FLOAT);
}

template <typename QKVT, typename GT>
__aicore__ inline void ChunkBwdDvLocalBase<QKVT, GT>::Process()
{
    if (chunkNumCurCore == 0) {
        return;
    }
    AscendC::printf(" ====================  Process =============================== \n");
    int64_t startChunkId = coreId >= preCoreNum ?
                               chunkNumPreCore * preCoreNum + (coreId - preCoreNum) * chunkNumTailCore :
                               chunkNumPreCore * coreId;
    int64_t endChunkId = startChunkId + chunkNumCurCore - 1;
    int64_t startBatchId = startChunkId / chunkNumForT;
    int64_t endBatchId = endChunkId / chunkNumForT;
    int64_t startChunkIdForBatch = startChunkId % chunkNumForT;
    int64_t endChunkIdForBatch = endChunkId % chunkNumForT;
    int64_t curStartChunkIdForBatch;
    int64_t curEndChunkIdForBatch;
    for (int64_t curBatchId = startBatchId; curBatchId <= endBatchId; curBatchId++) {
        curStartChunkIdForBatch = 0;
        curEndChunkIdForBatch = maxChunkIndexForT;
        if (curBatchId == startBatchId) {
            curStartChunkIdForBatch =
                curStartChunkIdForBatch < startChunkIdForBatch ? startChunkIdForBatch : curStartChunkIdForBatch;
        }
        if (curBatchId == endBatchId) {
            curEndChunkIdForBatch =
                curEndChunkIdForBatch > endChunkIdForBatch ? endChunkIdForBatch : curEndChunkIdForBatch;
        }
        ProcessForBatch(curStartChunkIdForBatch, curEndChunkIdForBatch, curBatchId);
    }
}


template <typename QKVT, typename GT>
__aicore__ inline void ChunkBwdDvLocalBase<QKVT, GT>::vectorProcess(int64_t chunkLen, int64_t curChunkId,
                                                                    int64_t curBatchId)
{
    AscendC::printf(" ====================  vectorProcess =============================== \n");
    for (int hIndex = 0; hIndex < h; hIndex++) {
        AscendC::LocalTensor<GT> gLocalTensor = gTQueIn.template AllocTensor<GT>();
        AscendC::DataCopy(gLocalTensor, gGm[curBatchId * h * t + hIndex * t + curChunkId * chunkSize], chunkSize);
        MTE2ToVSync();
        AscendC::LocalTensor<float> gFp32LocalTensor = gFp32TBuf.template Get<float>();
        AscendC::LocalTensor<float> gFactorLocalTensor = gFactorTBuf.template Get<float>();
        AscendC::Duplicate<float>(gFactorLocalTensor, float(0.0), chunkSize); // 清零
        AscendC::Cast(gFp32LocalTensor, gLocalTensor, AscendC::RoundMode::CAST_NONE, chunkLen);
        AscendC::LocalTensor<float> kqFp32LocalTensor = kqFp32TBuf.template Get<float>();
        AscendC::LocalTensor<QKVT> kqLocalTensor = kqTQueIn.template AllocTensor<QKVT>();
        AscendC::LocalTensor<QKVT> kqOutLocalTensor = kqTQueOut.template AllocTensor<QKVT>();
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(0x1);
        for (int row = 0; row < chunkLen; row++) {
            AscendC::Adds(gFactorLocalTensor, gFp32LocalTensor, -1 * gFp32LocalTensor.GetValue(row), chunkLen);
            AscendC::Exp(gFactorLocalTensor, gFactorLocalTensor, chunkLen);
            AscendC::Duplicate<float>(gFactorLocalTensor, float(0.0), row);
            AscendC::Muls(gFactorLocalTensor, gFactorLocalTensor, scale, chunkLen);
            // 搬运 k * q^T 一行
            AscendC::DataCopy(kqLocalTensor,
                              workspaceGm[curBatchId * h * t * chunkSize + hIndex * t * chunkSize +
                                          curChunkId * chunkSize * chunkSize + row * chunkSize],
                              chunkLen);
            MTE2ToVSync();
            AscendC::Cast(kqFp32LocalTensor, kqLocalTensor, AscendC::RoundMode::CAST_NONE, chunkLen);
            AscendC::printf("[tensor 打印]  kqFp32LocalTensor \n");
            AscendC::DumpTensor(kqFp32LocalTensor, 5, 64);
            AscendC::Mul(gFp32LocalTensor, kqFp32LocalTensor, gFactorLocalTensor, chunkLen);
            AscendC::Cast(kqOutLocalTensor, gFp32LocalTensor, AscendC::RoundMode::CAST_NONE, chunkSize);
            AscendC::printf("[tensor 打印]  kqOutLocalTensor \n");
            AscendC::DumpTensor(kqOutLocalTensor, 5, 64);
            // 搬出到workspace
            AscendC::DataCopy(workspace2Gm[curBatchId * h * t * chunkSize + hIndex * t * chunkSize +
                                           curChunkId * chunkSize * chunkSize + row * chunkSize],
                              kqOutLocalTensor, chunkSize);
        }
        gTQueIn.FreeTensor(gLocalTensor);
        kqTQueIn.FreeTensor(kqLocalTensor);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x2);
    }
}

template <typename QKVT, typename GT>
__aicore__ inline void ChunkBwdDvLocalBase<QKVT, GT>::cubeProcess(int64_t chunkLen, int64_t curChunkId,
                                                                  int64_t curBatchId)
{
    {
        // AscendC::printf("============ q^T @ k ====================== \n");
        using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using ArchTag = Catlass::Arch::AtlasA2;
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true>;
        using L1TileShape = Shape<_64, _64, _128>;
        using L0TileShape = Shape<_64, _64, _64>;
        // todo 需要提取函数根据chunkSize 区分，注意using用法是编译期的
        // using L1TileShape = Shape<_128, _128,_128>;
        // using L0TileShape = Shape<_128, _128,_64>;
        using ElementA = QKVT;
        using ElementB = QKVT;
        using ElementC = QKVT;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
                                                                ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA,
                                                             ElementB, ElementC, void, TileCopy>;
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(chunkSize, k);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, chunkSize);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(chunkSize, chunkSize);

        Catlass::GemmCoord actualBlockShape{static_cast<uint32_t>(chunkLen), static_cast<uint32_t>(chunkLen),
                                            static_cast<uint32_t>(k)};
        for (int hIndex = 0; hIndex < h; hIndex++) {
            // Represent the full tensors
            auto tensorA = tla::MakeTensor(kGm[curBatchId * h * t * k + hIndex * t * k + curChunkId * chunkSize * k],
                                           layoutA, Catlass::Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(qGm[curBatchId * h * t * k + hIndex * t * k + curChunkId * chunkSize * k],
                                           layoutB, Catlass::Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(workspaceGm[curBatchId * h * t * chunkSize + hIndex * t * chunkSize +
                                                       curChunkId * chunkSize * chunkSize],
                                           layoutC, Catlass::Arch::PositionGM{});
            // Make tiled views
            auto tensorBlockA =
                GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB =
                GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC =
                GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
            // AscendC::printf("[参数打印] tensorC.data().GetPhyAddr()  = %d  \n", tensorC.data().GetPhyAddr());
            // // Compute block-scoped matrix multiply-add
            // AscendC::printf("[参数打印] hIndex = %d  \n", hIndex);
            blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
            // AscendC::printf("[tensor 打印]  tensorBlockC \n");
            // AscendC::DumpTensor(tensorBlockC.data(), 5, 32);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x1);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }


    {
        using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using ArchTag = Catlass::Arch::AtlasA2;
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true>;
        using L1TileShape = Shape<_64, _128, _64>;
        using L0TileShape = Shape<_64, _128, _64>;
        // todo 需要提取函数根据chunkSize 区分，注意using用法是编译期的
        // using L1TileShape = Shape<_128, _128,_128>;
        // using L0TileShape = Shape<_128, _64,_128>;
        using ElementA = QKVT;
        using ElementB = QKVT;
        using ElementC = QKVT;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
                                                                ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA,
                                                             ElementB, ElementC, void, TileCopy>;
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(chunkSize, chunkSize);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(chunkSize, v);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(chunkSize, v);

        Catlass::GemmCoord actualBlockShape{static_cast<uint32_t>(chunkLen), static_cast<uint32_t>(v),
                                            static_cast<uint32_t>(chunkLen)};
        for (int hIndex = 0; hIndex < h; hIndex++) {
            // Represent the full tensors
            auto tensorA = tla::MakeTensor(workspace2Gm[curBatchId * h * t * chunkSize + hIndex * t * chunkSize +
                                                        curChunkId * chunkSize * chunkSize],
                                           layoutA, Catlass::Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(
                dOGm[curBatchId * h * t * chunkSize + hIndex * t * chunkSize + curChunkId * chunkSize * chunkSize],
                layoutB, Catlass::Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(
                dVGm[curBatchId * h * t * chunkSize + hIndex * t * chunkSize + curChunkId * chunkSize * chunkSize],
                layoutC, Catlass::Arch::PositionGM{});
            // Make tiled views
            auto tensorBlockA =
                GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB =
                GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC =
                GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            AscendC::CrossCoreWaitFlag<0x2>(0x2);
            blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}


template <typename QKVT, typename GT>
__aicore__ inline void ChunkBwdDvLocalBase<QKVT, GT>::ProcessForBatch(int64_t curStartChunkIdForBatch,
                                                                      int64_t curEndChunkIdForBatch, int64_t curBatchId)
{
    int64_t startSeqId = 0;
    int64_t endSeqId = 0;
    if (isVariable) {
        startSeqId = chunkIndicesGm.GetValue(curStartChunkIdForBatch * 2);
        endSeqId = chunkIndicesGm.GetValue(curEndChunkIdForBatch * 2);
    }

    int64_t seqChunkStartId = curStartChunkIdForBatch;
    for (int64_t curSeqId = startSeqId; curSeqId <= endSeqId; curSeqId++) {
        int64_t curSeqT = t;
        if (isVariable) {
            int64_t bos = cuSeqlensGm.GetValue(curSeqId);
            int64_t eos = cuSeqlensGm.GetValue(curSeqId + 1);
            curSeqT = eos - bos;
        }
        int64_t curSeqChunkNum = CeilDiv(curSeqT, chunkSize);
        int64_t seqChunkEndId = seqChunkStartId + curSeqChunkNum - 1;
        seqChunkEndId = seqChunkEndId > curEndChunkIdForBatch ? curEndChunkIdForBatch : seqChunkEndId;
        for (int64_t curChunkId = seqChunkStartId; curChunkId <= seqChunkEndId; curChunkId++) {
            int64_t curSeqChunkId = curChunkId;
            if (isVariable) {
                curSeqChunkId = chunkIndicesGm.GetValue(curChunkId * 2 + 1);
            }
            int64_t chunkStartToken = curSeqChunkId * chunkSize;
            int64_t chunkEndToken = chunkStartToken + chunkSize;
            chunkEndToken = chunkEndToken > curSeqT ? curSeqT : chunkEndToken;
            int64_t chunkLen = chunkEndToken - chunkStartToken;
            AscendC::printf("[参数打印] curChunkId = %d  \n", curChunkId);
            AscendC::printf("[参数打印] chunkLen = %d  \n", chunkLen);
            if (chunkLen <= 0) {
                continue;
            }
            if ASCEND_IS_AIC {
                cubeProcess(chunkLen, curChunkId, curBatchId);
            }

            if ASCEND_IS_AIV {
                vectorProcess(chunkLen, curChunkId, curBatchId);
            }
        }
        seqChunkStartId += curSeqChunkNum;
    }
}

} // namespace GDN
#endif // CHUNK_BWD_DV_LOCAL_BASE_H