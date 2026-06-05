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
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
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
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#endif

#include "chunk_bwd_dv_local_common.h"
using namespace tla;

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

    int64_t headNum;
    int64_t seqLen;
    int64_t keyDim;
    int64_t valueDim;
    int64_t totalChunkTasks;
    int64_t aicCoreNum;
    int64_t coreIdx;
    int64_t workspaceHeadSlotNum;  // 每个 core 可同时保留的 head workspace 槽数
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

    headNum = tilingData->h;
    seqLen = tilingData->t;
    keyDim = tilingData->k;
    valueDim = tilingData->v;
    totalChunkTasks = tilingData->b * strategy.chunkNumForT;
    aicCoreNum = static_cast<int64_t>(AscendC::GetBlockNum());
    coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    workspaceHeadSlotNum = tilingData->headBufNum;
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalCube<QKVT, GT, Strategy>::Process()
{
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ArchTag = Catlass::Arch::Ascend950;
    #else
        using ArchTag = Catlass::Arch::AtlasA2;
    #endif    
    using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false>;
    using L1TileShape = Shape<_128, _128, _128>;
    using L0TileShape = Shape<_128, _128, _128>;
    using ElementA = QKVT;
    using ElementB = QKVT;
    using ElementC = QKVT;
    Catlass::Arch::Resource<ArchTag> resource;

    // P1 阶段负责生成原始注意力块 Ws，A 矩阵取 K，B 矩阵取 Q^T，结果先落到 workspace 供 AIV 做 gating。
    // kqMatrixGemm: K @ Q^T -> workspace（RowMajor × ColumnMajor -> RowMajor）
    using P1_LayoutA = Catlass::layout::RowMajor;
    using P1_LayoutB = Catlass::layout::ColumnMajor;
    using P1_LayoutC = Catlass::layout::RowMajor;
    using P1_TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, P1_LayoutA, ElementB, P1_LayoutB,
                                                                ElementC, P1_LayoutC>;
    using P1_CopyL1ToL0A = typename P1_TileCopy::CopyL1ToL0A;
    using P1_CopyL1ToL0B = typename P1_TileCopy::CopyL1ToL0B;
    using P1_LayoutTagL1A = typename P1_TileCopy::LayoutTagL1A;
    using P1_LayoutTagL1B = typename P1_TileCopy::LayoutTagL1B;
    using P1_LayoutTagL0A = typename P1_TileCopy::LayoutTagL0A;
    using P1_LayoutTagL0B = typename P1_TileCopy::LayoutTagL0B;
    using P1_TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, ElementA, P1_LayoutTagL1A>;
    using ElementAccumulator = typename P1_TileCopy::ElementAccumulator;

    // P2 阶段消费 AIV 写回的 gated Ws，并与 dO 做第二次 MMAD，直接写出 dV。
    // dvGemm: gated kqMatrix @ dO -> dV（RowMajor × RowMajor -> RowMajor）
    using P2_LayoutA = Catlass::layout::RowMajor;
    using P2_LayoutB = Catlass::layout::RowMajor;
    using P2_LayoutC = Catlass::layout::RowMajor;
    using P2_TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, P2_LayoutA, ElementB, P2_LayoutB,
                                                                ElementC, P2_LayoutC>;
    using P2_BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA,
                                                             ElementB, ElementC, void, P2_TileCopy>;
    using P2_CopyL1ToL0A = typename P2_TileCopy::CopyL1ToL0A;
    using P2_CopyL1ToL0B = typename P2_TileCopy::CopyL1ToL0B;
    using P2_LayoutTagL1A = typename P2_TileCopy::LayoutTagL1A;
    using P2_LayoutTagL1B = typename P2_TileCopy::LayoutTagL1B;
    using P2_LayoutTagL0A = typename P2_TileCopy::LayoutTagL0A;
    using P2_LayoutTagL0B = typename P2_TileCopy::LayoutTagL0B;
    using P2_AType = Catlass::Gemm::GemmType<ElementA, typename P2_TileCopy::LayoutA>;
    using P2_TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, P2_AType, P2_LayoutTagL1A>;

    auto p1LayoutA = tla::MakeLayout<ElementA, P1_LayoutA>(strategy.chunkSize, keyDim);
    auto p1LayoutB = tla::MakeLayout<ElementB, P1_LayoutB>(keyDim, strategy.chunkSize);
    auto p1LayoutC = tla::MakeLayout<ElementC, P1_LayoutC>(strategy.chunkSize, strategy.chunkSize);
    auto p1L1ALayout = tla::MakeLayout<ElementA, P1_LayoutTagL1A>(_128{}, _128{});
    auto p1L1BLayout = tla::MakeLayout<ElementB, P1_LayoutTagL1B>(_128{}, _128{});
    auto p2LayoutA = tla::MakeLayout<ElementA, P2_LayoutA>(strategy.chunkSize, strategy.chunkSize);
    auto p2LayoutB = tla::MakeLayout<ElementB, P2_LayoutB>(strategy.chunkSize, valueDim);
    auto p2LayoutC = tla::MakeLayout<ElementC, P2_LayoutC>(strategy.chunkSize, valueDim);
    auto p2L1ALayout = tla::MakeLayout<ElementA, P2_LayoutTagL1A>(_128{}, _128{});
    auto p2L1BLayout = tla::MakeLayout<ElementB, P2_LayoutTagL1B>(_128{}, _128{});

    constexpr uint32_t P1_STAGE_NUM = 2;
    constexpr uint32_t P1_L1A_TILE_BYTES = 128 * 128 * sizeof(ElementA);
    constexpr uint32_t P1_L1B_TILE_BYTES = 128 * 128 * sizeof(ElementB);
    constexpr uint32_t P1_L0A_TILE_BYTES = 128 * 128 * sizeof(ElementA);
    constexpr uint32_t P1_L0B_TILE_BYTES = 128 * 128 * sizeof(ElementB);
    constexpr int32_t P1_EVENT_A0 = 0;
    constexpr int32_t P1_EVENT_A1 = 1;
    constexpr int32_t P1_EVENT_B0 = 2;
    constexpr int32_t P1_EVENT_B1 = 3;
    constexpr int32_t P1_EVENT_C = 4;
    constexpr int32_t P1_EVENT_L0C = 0;

    // P1 使用双缓冲搬运 K/Q，尽量覆盖 GM->L1、L1->L0 和 MMAD 的等待时间。
    auto p1L1ATensorRaw0 = resource.l1Buf.template GetBufferByByte<ElementA>(0);
    auto p1L1ATensorRaw1 = resource.l1Buf.template GetBufferByByte<ElementA>(P1_L1A_TILE_BYTES);
    auto p1L1BTensorRaw0 = resource.l1Buf.template GetBufferByByte<ElementB>(P1_L1A_TILE_BYTES * P1_STAGE_NUM);
    auto p1L1BTensorRaw1 =
        resource.l1Buf.template GetBufferByByte<ElementB>(P1_L1A_TILE_BYTES * P1_STAGE_NUM + P1_L1B_TILE_BYTES);
    auto p1L0ATensorRaw0 = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
    auto p1L0ATensorRaw1 = resource.l0ABuf.template GetBufferByByte<ElementA>(P1_L0A_TILE_BYTES);
    auto p1L0BTensorRaw0 = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
    auto p1L0BTensorRaw1 = resource.l0BBuf.template GetBufferByByte<ElementB>(P1_L0B_TILE_BYTES);
    auto p1L0CTensorRaw0 = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    P1_CopyL1ToL0A p1CopyL1ToL0A;
    P1_CopyL1ToL0B p1CopyL1ToL0B;
    P1_TileMmad p1TileMmad;

    constexpr uint32_t P2_STAGE_NUM = 3;
    constexpr uint32_t P2_VALUE_TILE_N = 128;
    constexpr uint32_t P2_L1_TILE_BYTES = 128 * 128 * sizeof(ElementA);
    constexpr uint32_t P2_L0_TILE_BYTES = 128 * 128 * sizeof(ElementA);
    constexpr uint32_t P2_L0C_TILE_BYTES = 128 * 128 * sizeof(float);
    constexpr int32_t P2_EVENT_A = 7;
    constexpr int32_t P2_EVENT_A0 = 4;
    constexpr int32_t P2_EVENT_A1 = 5;
    constexpr int32_t P2_EVENT_B0 = 6;
    constexpr int32_t P2_EVENT_B1 = 7;
    constexpr int32_t P2_EVENT_B2 = 3;
    constexpr int32_t P2_EVENT_C0 = 0;
    constexpr int32_t P2_EVENT_C1 = 1;
    constexpr int32_t P2_PREFETCH_EVENT_B0 = 4;
    constexpr int32_t P2_PREFETCH_EVENT_B1 = 5;
    constexpr int32_t P2_PREFETCH_EVENT_B2 = 6;

    // P2 的 dO 使用 3 个 L1B stage 做 ring prefetch，workspace 使用 L1A 预取槽衔接 AIV 写回。
    auto p2L1BTensorRaw0 = resource.l1Buf.template GetBufferByByte<ElementB>(
        P1_L1A_TILE_BYTES * P1_STAGE_NUM + P1_L1B_TILE_BYTES * P1_STAGE_NUM);
    auto p2L1BTensorRaw1 = resource.l1Buf.template GetBufferByByte<ElementB>(
        P1_L1A_TILE_BYTES * P1_STAGE_NUM + P1_L1B_TILE_BYTES * P1_STAGE_NUM + P2_L1_TILE_BYTES);
    auto p2L1BTensorRaw2 = resource.l1Buf.template GetBufferByByte<ElementB>(
        P1_L1A_TILE_BYTES * P1_STAGE_NUM + P1_L1B_TILE_BYTES * P1_STAGE_NUM + P2_L1_TILE_BYTES * 2);
    auto p2L1ATensorRaw0 = resource.l1Buf.template GetBufferByByte<ElementA>(
        P1_L1A_TILE_BYTES * P1_STAGE_NUM + P1_L1B_TILE_BYTES * P1_STAGE_NUM + P2_L1_TILE_BYTES * 3);
    auto p2L1ATensorRaw1 = resource.l1Buf.template GetBufferByByte<ElementA>(
        P1_L1A_TILE_BYTES * P1_STAGE_NUM + P1_L1B_TILE_BYTES * P1_STAGE_NUM + P2_L1_TILE_BYTES * 4);
    auto p2L0ATensorRaw0 = resource.l0ABuf.template GetBufferByByte<ElementA>(P1_L0A_TILE_BYTES);
    auto p2L0ATensorRaw1 = p2L0ATensorRaw0;
    auto p2L0BTensorRaw0 = resource.l0BBuf.template GetBufferByByte<ElementB>(P1_L0B_TILE_BYTES);
    auto p2L0BTensorRaw1 = p2L0BTensorRaw0;
    auto p2L0CTensorRaw0 = resource.l0CBuf.template GetBufferByByte<float>(0);
    auto p2L0CTensorRaw1 = resource.l0CBuf.template GetBufferByByte<float>(P2_L0C_TILE_BYTES);
    P2_CopyL1ToL0A p2CopyL1ToL0A;
    P2_CopyL1ToL0B p2CopyL1ToL0B;
    P2_TileMmad p2TileMmad;

    ChunkTaskIndex chunkTask;

    // 每个 AIC core 按 chunk 维度轮询任务；chunk 内部再按 head 做 P1/AIV/P2 的流水。
    // NOTE: P1/P2 share Catlass resource, so the current implementation keeps one
    //       hand-written head-level rolling pipeline inside each chunk.
    for (int64_t loopIdx = coreIdx; loopIdx < totalChunkTasks; loopIdx += aicCoreNum) {
        strategy.ResolveTask(loopIdx, chunkTask);

        Catlass::GemmCoord kqGemmShape{static_cast<uint32_t>(chunkTask.chunkLen),
                                   static_cast<uint32_t>(chunkTask.chunkLen), static_cast<uint32_t>(keyDim)};
        Catlass::GemmCoord dvGemmShape{static_cast<uint32_t>(chunkTask.chunkLen), static_cast<uint32_t>(valueDim),
                                   static_cast<uint32_t>(chunkTask.chunkLen)};

        // 以下状态变量用于描述 head 级流水线中已经预取、等待写回或等待消费的 head。
        // loadedKqHead: K @ Q^T input prefetched to L1 but not written to workspace yet.
        // pendingDvHead: dV result in L0C waiting for GM store.
        // prefetchedKqMatrixHead: gated kqMatrix workspace prefetched for the next dV head.
        {
            AscendC::SetMMLayoutTransform(true);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P1_EVENT_A0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P1_EVENT_A1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P1_EVENT_B0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P1_EVENT_B1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(P1_EVENT_A0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(P1_EVENT_A1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(P1_EVENT_B0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(P1_EVENT_B1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(P1_EVENT_L0C);
            uint32_t p1L1AStage = 0;
            uint32_t p1L1BStage = 0;
            uint32_t p1L0AStage = 0;
            uint32_t p1L0BStage = 0;

// 执行一次完整的 P1 tile：K/Q 从 GM 搬到 L1/L0，发射 K @ Q^T，并把 L0C 写回 workspace。
// PREFETCH_DO 用来在 P1 MMAD 和 FIX 搬出窗口中插入 dO 预取，隐藏后续 P2 的 GM 访存。
#define RUN_KQ_TILE_MMAD(TENSOR_BLOCK_A, TENSOR_BLOCK_B, TENSOR_BLOCK_C, P1_ACTUAL_SHAPE, PREFETCH_DO)            \
            do {                                                                                                  \
                using P1_CopyGmToL1A = typename P1_TileCopy::template CopyGmToL1A<decltype(TENSOR_BLOCK_A)>;      \
                using P1_CopyGmToL1B = typename P1_TileCopy::template CopyGmToL1B<decltype(TENSOR_BLOCK_B)>;      \
                using P1_CopyL0CToGm = typename P1_TileCopy::template CopyL0CToGm<decltype(TENSOR_BLOCK_C)>;      \
                P1_CopyGmToL1A p1CopyGmToL1A;                                                                    \
                P1_CopyGmToL1B p1CopyGmToL1B;                                                                    \
                P1_CopyL0CToGm p1CopyL0CToGm;                                                                    \
                int32_t eventL1A = (p1L1AStage == 0) ? P1_EVENT_A0 : P1_EVENT_A1;                                \
                int32_t eventL1B = (p1L1BStage == 0) ? P1_EVENT_B0 : P1_EVENT_B1;                                \
                int32_t eventL0A = (p1L0AStage == 0) ? P1_EVENT_A0 : P1_EVENT_A1;                                \
                int32_t eventL0B = (p1L0BStage == 0) ? P1_EVENT_B0 : P1_EVENT_B1;                                \
                auto p1L1ATensorRaw = (p1L1AStage == 0) ? p1L1ATensorRaw0 : p1L1ATensorRaw1;                    \
                auto p1L1BTensorRaw = (p1L1BStage == 0) ? p1L1BTensorRaw0 : p1L1BTensorRaw1;                    \
                auto p1L0ATensorRaw = (p1L0AStage == 0) ? p1L0ATensorRaw0 : p1L0ATensorRaw1;                    \
                auto p1L0BTensorRaw = (p1L0BStage == 0) ? p1L0BTensorRaw0 : p1L0BTensorRaw1;                    \
                uint32_t p1MActual = (P1_ACTUAL_SHAPE).m();                                                      \
                if constexpr (std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) {                                 \
                    if (p1MActual == 1) {                                                                        \
                        p1MActual = 16;                                                                          \
                    }                                                                                            \
                }                                                                                                \
                auto tensorP1L1A = tla::MakeTensor(p1L1ATensorRaw, p1L1ALayout, Catlass::Arch::PositionL1{});    \
                auto tensorP1L1B = tla::MakeTensor(p1L1BTensorRaw, p1L1BLayout, Catlass::Arch::PositionL1{});    \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);                                      \
                p1CopyGmToL1A(tensorP1L1A, TENSOR_BLOCK_A);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);                                       \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);                                      \
                p1CopyGmToL1B(tensorP1L1B, TENSOR_BLOCK_B);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);                                       \
                auto p1LayoutAInL0 = tla::MakeLayout<ElementA, P1_LayoutTagL0A>(p1MActual,                       \
                                                                                (P1_ACTUAL_SHAPE).k());          \
                auto p1LayoutBInL0 = tla::MakeLayout<ElementB, P1_LayoutTagL0B>((P1_ACTUAL_SHAPE).k(),           \
                                                                                (P1_ACTUAL_SHAPE).n());          \
                auto tensorP1L0A = tla::MakeTensor(p1L0ATensorRaw, p1LayoutAInL0,                                \
                                                   Catlass::Arch::PositionL0A{});                                \
                auto tensorP1L0B = tla::MakeTensor(p1L0BTensorRaw, p1LayoutBInL0,                                \
                                                   Catlass::Arch::PositionL0B{});                                \
                auto tensorP1TileL1A = GetTile(tensorP1L1A, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(p1MActual, (P1_ACTUAL_SHAPE).k()));                \
                auto tensorP1TileL1B = GetTile(tensorP1L1B, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape((P1_ACTUAL_SHAPE).k(), (P1_ACTUAL_SHAPE).n()));    \
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventL0A);                                         \
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);                                      \
                p1CopyL1ToL0A(tensorP1L0A, tensorP1TileL1A);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);                                       \
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventL0B);                                         \
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);                                      \
                p1CopyL1ToL0B(tensorP1L0B, tensorP1TileL1B);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);                                       \
                auto p1LayoutInL0C = tla::MakeLayoutL0C(p1MActual, (P1_ACTUAL_SHAPE).n());                       \
                auto tensorP1L0C = tla::MakeTensor(p1L0CTensorRaw0, p1LayoutInL0C,                               \
                                                   Catlass::Arch::PositionL0C{});                                \
                auto tensorP1TileL0C = GetTile(tensorP1L0C, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(p1MActual, (P1_ACTUAL_SHAPE).n()));                \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(P1_EVENT_C);                                        \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(P1_EVENT_C);                                       \
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(P1_EVENT_L0C);                                      \
                p1TileMmad(tensorP1TileL0C, tensorP1L0A, tensorP1L0B, p1MActual, (P1_ACTUAL_SHAPE).n(),          \
                           (P1_ACTUAL_SHAPE).k(), true, 0b11);                                                   \
                PREFETCH_DO;                                                                                     \
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventL0B);                                          \
                p1L0BStage = p1L0BStage ^ 1;                                                                     \
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventL0A);                                          \
                p1L0AStage = p1L0AStage ^ 1;                                                                     \
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(P1_EVENT_L0C);                                       \
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(P1_EVENT_L0C);                                      \
                p1CopyL0CToGm(TENSOR_BLOCK_C, tensorP1L0C, 0b11);                                                \
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(P1_EVENT_L0C);                                       \
                p1L1AStage = p1L1AStage ^ 1;                                                                     \
                p1L1BStage = p1L1BStage ^ 1;                                                                     \
            } while (0)

            int64_t loadedKqHead = -1;
            uint32_t loadedKqL1AStage = 0;
            uint32_t loadedKqL1BStage = 0;

// 只把指定 head 的 K/Q 预取到 P1 L1，不立即发射 MMAD；用于和 P2 当前 head 计算交错。
#define PREFETCH_KQ_HEAD_TO_L1(P1_HEAD_INDEX)                                                                       \
            do {                                                                                                  \
                int64_t kqHeadIndex = (P1_HEAD_INDEX);                                                           \
                int64_t qkOffset = chunkTask.batchId * headNum * seqLen * keyDim + kqHeadIndex * seqLen * keyDim +                    \
                                   chunkTask.tokenStart * keyDim;                                                    \
                auto tensorA = tla::MakeTensor(kGm[qkOffset], p1LayoutA, Catlass::Arch::PositionGM{});           \
                auto tensorB = tla::MakeTensor(qGm[qkOffset], p1LayoutB, Catlass::Arch::PositionGM{});           \
                auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0, 0),                                       \
                                            tla::MakeShape(kqGemmShape.m(), kqGemmShape.k()));                           \
                auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(0, 0),                                       \
                                            tla::MakeShape(kqGemmShape.k(), kqGemmShape.n()));                           \
                using P1_CopyGmToL1A = typename P1_TileCopy::template CopyGmToL1A<decltype(tensorBlockA)>;        \
                using P1_CopyGmToL1B = typename P1_TileCopy::template CopyGmToL1B<decltype(tensorBlockB)>;        \
                P1_CopyGmToL1A p1CopyGmToL1A;                                                                    \
                P1_CopyGmToL1B p1CopyGmToL1B;                                                                    \
                int32_t eventL1A = (p1L1AStage == 0) ? P1_EVENT_A0 : P1_EVENT_A1;                                \
                int32_t eventL1B = (p1L1BStage == 0) ? P1_EVENT_B0 : P1_EVENT_B1;                                \
                auto p1L1ATensorRaw = (p1L1AStage == 0) ? p1L1ATensorRaw0 : p1L1ATensorRaw1;                    \
                auto p1L1BTensorRaw = (p1L1BStage == 0) ? p1L1BTensorRaw0 : p1L1BTensorRaw1;                    \
                auto tensorP1L1A = tla::MakeTensor(p1L1ATensorRaw, p1L1ALayout, Catlass::Arch::PositionL1{});    \
                auto tensorP1L1B = tla::MakeTensor(p1L1BTensorRaw, p1L1BLayout, Catlass::Arch::PositionL1{});    \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);                                      \
                p1CopyGmToL1A(tensorP1L1A, tensorBlockA);                                                       \
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);                                       \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);                                      \
                p1CopyGmToL1B(tensorP1L1B, tensorBlockB);                                                       \
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);                                       \
                loadedKqHead = kqHeadIndex;                                                                      \
                loadedKqL1AStage = p1L1AStage;                                                                   \
                loadedKqL1BStage = p1L1BStage;                                                                   \
                p1L1AStage = p1L1AStage ^ 1;                                                                     \
                p1L1BStage = p1L1BStage ^ 1;                                                                     \
            } while (0)

// 消费已经在 L1 中的 K/Q，完成 K @ Q^T 并写入 workspace，随后由 AIC->AIV flag 通知 vector。
#define COMPUTE_LOADED_KQ_HEAD(PREFETCH_DO)                                                                       \
            do {                                                                                                  \
                int64_t kqHeadIndex = loadedKqHead;                                                              \
                p1L0AStage = 0;                                                                                   \
                p1L0BStage = 0;                                                                                   \
                int64_t wsOffset = GetCoreBufferSingleHWorkspaceOffset(coreIdx, kqHeadIndex, 0,                 \
                                                                       strategy.chunkSize, workspaceHeadSlotNum);         \
                auto tensorC = tla::MakeTensor(workspaceGm[wsOffset], p1LayoutC, Catlass::Arch::PositionGM{});   \
                auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(0, 0),                                       \
                                            tla::MakeShape(kqGemmShape.m(), kqGemmShape.n()));                           \
                using P1_CopyL0CToGm = typename P1_TileCopy::template CopyL0CToGm<decltype(tensorBlockC)>;       \
                P1_CopyL0CToGm p1CopyL0CToGm;                                                                    \
                int32_t eventL1A = (loadedKqL1AStage == 0) ? P1_EVENT_A0 : P1_EVENT_A1;                          \
                int32_t eventL1B = (loadedKqL1BStage == 0) ? P1_EVENT_B0 : P1_EVENT_B1;                          \
                int32_t eventL0A = (p1L0AStage == 0) ? P1_EVENT_A0 : P1_EVENT_A1;                                \
                int32_t eventL0B = (p1L0BStage == 0) ? P1_EVENT_B0 : P1_EVENT_B1;                                \
                auto p1L1ATensorRaw = (loadedKqL1AStage == 0) ? p1L1ATensorRaw0 : p1L1ATensorRaw1;              \
                auto p1L1BTensorRaw = (loadedKqL1BStage == 0) ? p1L1BTensorRaw0 : p1L1BTensorRaw1;              \
                auto p1L0ATensorRaw = (p1L0AStage == 0) ? p1L0ATensorRaw0 : p1L0ATensorRaw1;                    \
                auto p1L0BTensorRaw = (p1L0BStage == 0) ? p1L0BTensorRaw0 : p1L0BTensorRaw1;                    \
                uint32_t p1MActual = kqGemmShape.m();                                                                \
                if constexpr (std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) {                                 \
                    if (p1MActual == 1) {                                                                        \
                        p1MActual = 16;                                                                          \
                    }                                                                                            \
                }                                                                                                \
                auto tensorP1L1A = tla::MakeTensor(p1L1ATensorRaw, p1L1ALayout, Catlass::Arch::PositionL1{});    \
                auto tensorP1L1B = tla::MakeTensor(p1L1BTensorRaw, p1L1BLayout, Catlass::Arch::PositionL1{});    \
                auto p1LayoutAInL0 = tla::MakeLayout<ElementA, P1_LayoutTagL0A>(p1MActual, kqGemmShape.k());         \
                auto p1LayoutBInL0 = tla::MakeLayout<ElementB, P1_LayoutTagL0B>(kqGemmShape.k(), kqGemmShape.n());       \
                auto tensorP1L0A = tla::MakeTensor(p1L0ATensorRaw, p1LayoutAInL0,                                \
                                                   Catlass::Arch::PositionL0A{});                                \
                auto tensorP1L0B = tla::MakeTensor(p1L0BTensorRaw, p1LayoutBInL0,                                \
                                                   Catlass::Arch::PositionL0B{});                                \
                auto tensorP1TileL1A = GetTile(tensorP1L1A, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(p1MActual, kqGemmShape.k()));                          \
                auto tensorP1TileL1B = GetTile(tensorP1L1B, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(kqGemmShape.k(), kqGemmShape.n()));                        \
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventL0A);                                         \
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);                                      \
                p1CopyL1ToL0A(tensorP1L0A, tensorP1TileL1A);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);                                       \
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventL0B);                                         \
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);                                      \
                p1CopyL1ToL0B(tensorP1L0B, tensorP1TileL1B);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);                                       \
                auto p1LayoutInL0C = tla::MakeLayoutL0C(p1MActual, kqGemmShape.n());                                 \
                auto tensorP1L0C = tla::MakeTensor(p1L0CTensorRaw0, p1LayoutInL0C,                               \
                                                   Catlass::Arch::PositionL0C{});                                \
                auto tensorP1TileL0C = GetTile(tensorP1L0C, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(p1MActual, kqGemmShape.n()));                          \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(P1_EVENT_C);                                        \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(P1_EVENT_C);                                       \
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(P1_EVENT_L0C);                                      \
                p1TileMmad(tensorP1TileL0C, tensorP1L0A, tensorP1L0B, p1MActual, kqGemmShape.n(),                    \
                           kqGemmShape.k(), true, 0b11);                                                             \
                PREFETCH_DO;                                                                                     \
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventL0B);                                          \
                p1L0BStage = p1L0BStage ^ 1;                                                                     \
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventL0A);                                          \
                p1L0AStage = p1L0AStage ^ 1;                                                                     \
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(P1_EVENT_L0C);                                       \
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(P1_EVENT_L0C);                                      \
                p1CopyL0CToGm(tensorBlockC, tensorP1L0C, 0b11);                                                  \
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(P1_EVENT_L0C);                                       \
                loadedKqHead = -1;                                                                               \
            } while (0)

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P2_PREFETCH_EVENT_B0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P2_PREFETCH_EVENT_B1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P2_PREFETCH_EVENT_B2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P2_EVENT_A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(P2_EVENT_A0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(P2_EVENT_B0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(P2_EVENT_C1);
            int64_t nextDoPrefetchHead = 0;
            int64_t pendingDvHead = -1;
            int64_t prefetchedKqMatrixHead = -1;

// 将指定 head 的 dO 预取到 P2 L1B ring，P2 MMAD 时只需要等待对应 stage ready。
#define PREFETCH_DO_HEAD_TO_L1(DO_HEAD_INDEX)                                                                       \
            do {                                                                                                  \
                int64_t doPrefetchHead = (DO_HEAD_INDEX);                                                           \
                uint32_t doStage = static_cast<uint32_t>(doPrefetchHead % P2_STAGE_NUM);                            \
                int32_t doPrefetchEvent = (doStage == 0) ? P2_PREFETCH_EVENT_B0 :                                \
                                         ((doStage == 1) ? P2_PREFETCH_EVENT_B1 : P2_PREFETCH_EVENT_B2);         \
                int64_t doOffset =                                                                               \
                    chunkTask.batchId * headNum * seqLen * valueDim + doPrefetchHead * seqLen * valueDim + chunkTask.tokenStart * valueDim;       \
                auto doTensorB = tla::MakeTensor(dOGm[doOffset], p2LayoutB, Catlass::Arch::PositionGM{});        \
                auto doTensorBlockB = GetTile(doTensorB, tla::MakeCoord(0, 0),                                   \
                                              tla::MakeShape(dvGemmShape.k(), dvGemmShape.n()));                         \
                using P2_CopyDoGmToL1B = typename P2_TileCopy::template CopyGmToL1B<decltype(doTensorBlockB)>;   \
                P2_CopyDoGmToL1B p2CopyDoGmToL1B;                                                                \
                auto doP2L1BTensorRaw = (doStage == 0) ? p2L1BTensorRaw0 :                                       \
                                          ((doStage == 1) ? p2L1BTensorRaw1 : p2L1BTensorRaw2);                  \
                auto doTensorP2L1B =                                                                             \
                    tla::MakeTensor(doP2L1BTensorRaw, p2L1BLayout, Catlass::Arch::PositionL1{});                 \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(doPrefetchEvent);                               \
                p2CopyDoGmToL1B(doTensorP2L1B, doTensorBlockB);                                                  \
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(doPrefetchEvent);                                \
            } while (0)

#define PREFETCH_NEXT_DO_IF_ANY()                                                                                \
            do {                                                                                                  \
                if (nextDoPrefetchHead < headNum) {                                                                         \
                    PREFETCH_DO_HEAD_TO_L1(nextDoPrefetchHead);                                                          \
                    ++nextDoPrefetchHead;                                                                             \
                }                                                                                                 \
            } while (0)

// 保证当前要计算的 dV head 已有 dO 在 L1B 中，避免 P2 发射前再同步等待长访存。
#define ENSURE_DO_READY(DO_HEAD_INDEX)                                                                            \
            do {                                                                                                  \
                int64_t requiredDoHead = (DO_HEAD_INDEX);                                                           \
                while (nextDoPrefetchHead <= requiredDoHead && nextDoPrefetchHead < headNum) {                                    \
                    PREFETCH_DO_HEAD_TO_L1(nextDoPrefetchHead);                                                          \
                    ++nextDoPrefetchHead;                                                                             \
                }                                                                                                 \
            } while (0)

// 提前等待 AIV 完成下一 head 的 gating，并把 gated Ws 从 workspace 预取到 P2 L1A。
#define PREFETCH_KQ_MATRIX_TO_L1A(P2_HEAD_INDEX)                                                                      \
            do {                                                                                                  \
                int64_t kqMatrixPrefetchHead = (P2_HEAD_INDEX);                                                        \
                if (kqMatrixPrefetchHead < headNum && prefetchedKqMatrixHead < 0) {                                               \
                    int64_t wsOffset = GetCoreBufferSingleHWorkspaceOffset(coreIdx, kqMatrixPrefetchHead, 0,           \
                                                                           strategy.chunkSize, workspaceHeadSlotNum);      \
                    auto p2TensorA =                                                                              \
                        tla::MakeTensor(workspaceGm[wsOffset], p2LayoutA, Catlass::Arch::PositionGM{});          \
                    auto p2TensorBlockA = GetTile(p2TensorA, tla::MakeCoord(0, 0),                               \
                                                  tla::MakeShape(dvGemmShape.m(), dvGemmShape.k()));                     \
                    using P2_CopyGmToL1A =                                                                        \
                        typename P2_TileCopy::template CopyGmToL1A<decltype(p2TensorBlockA)>;                    \
                    P2_CopyGmToL1A p2CopyGmToL1A;                                                                \
                    auto tensorP2L1A =                                                                            \
                        tla::MakeTensor(p2L1ATensorRaw1, p2L1ALayout, Catlass::Arch::PositionL1{});              \
                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_2);                                             \
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P2_EVENT_A);                                \
                    p2CopyGmToL1A(tensorP2L1A, p2TensorBlockA);                                                  \
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(P2_EVENT_A);                                 \
                    prefetchedKqMatrixHead = kqMatrixPrefetchHead;                                                         \
                }                                                                                                 \
            } while (0)

// 准备 P2 输入：gated Ws 进入 L0A，预取好的 dO 进入 L0B，为 gated Ws @ dO MMAD 铺好数据。
#define PREPARE_DV_HEAD_TO_L0(P2_HEAD_INDEX)                                                                      \
            do {                                                                                                  \
                int64_t dvHeadIndex = (P2_HEAD_INDEX);                                                           \
                uint32_t l0Stage = 0;                                                                             \
                uint32_t doStage = static_cast<uint32_t>(dvHeadIndex % P2_STAGE_NUM);                            \
                int32_t eventA = (l0Stage == 0) ? P2_EVENT_A0 : P2_EVENT_A1;                                     \
                int32_t eventB = (l0Stage == 0) ? P2_EVENT_B0 : P2_EVENT_B1;                                     \
                int32_t prefetchEventB = (doStage == 0) ? P2_PREFETCH_EVENT_B0 :                                 \
                                         ((doStage == 1) ? P2_PREFETCH_EVENT_B1 : P2_PREFETCH_EVENT_B2);         \
                int64_t wsOffset = GetCoreBufferSingleHWorkspaceOffset(coreIdx, dvHeadIndex, 0,                 \
                                                                       strategy.chunkSize, workspaceHeadSlotNum);         \
                auto p2TensorA = tla::MakeTensor(workspaceGm[wsOffset], p2LayoutA, Catlass::Arch::PositionGM{}); \
                auto p2TensorBlockA = GetTile(p2TensorA, tla::MakeCoord(0, 0),                                   \
                                              tla::MakeShape(dvGemmShape.m(), dvGemmShape.k()));                         \
                using P2_CopyGmToL1A = typename P2_TileCopy::template CopyGmToL1A<decltype(p2TensorBlockA)>;     \
                P2_CopyGmToL1A p2CopyGmToL1A;                                                                    \
                bool p2WsPrefetched = (prefetchedKqMatrixHead == dvHeadIndex);                                       \
                auto p2L1ATensorRaw = p2WsPrefetched ? p2L1ATensorRaw1 : p2L1ATensorRaw0;                       \
                auto tensorP2L1A = tla::MakeTensor(p2L1ATensorRaw, p2L1ALayout, Catlass::Arch::PositionL1{});    \
                auto p2L1BTensorRaw = (doStage == 0) ? p2L1BTensorRaw0 :                                         \
                                          ((doStage == 1) ? p2L1BTensorRaw1 : p2L1BTensorRaw2);                  \
                auto tensorP2L1B = tla::MakeTensor(p2L1BTensorRaw, p2L1BLayout, Catlass::Arch::PositionL1{});    \
                if (!p2WsPrefetched) {                                                                            \
                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_2);                                             \
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P2_EVENT_A);                                \
                    p2CopyGmToL1A(tensorP2L1A, p2TensorBlockA);                                                  \
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(P2_EVENT_A);                                 \
                }                                                                                                 \
                auto p2LayoutAInL0 = tla::MakeLayout<ElementA, P2_LayoutTagL0A>(dvGemmShape.m(), dvGemmShape.k());       \
                auto p2LayoutBInL0 = tla::MakeLayout<ElementB, P2_LayoutTagL0B>(dvGemmShape.k(), dvGemmShape.n());       \
                auto p2L0ATensorRaw = (l0Stage == 0) ? p2L0ATensorRaw0 : p2L0ATensorRaw1;                       \
                auto p2L0BTensorRaw = (l0Stage == 0) ? p2L0BTensorRaw0 : p2L0BTensorRaw1;                       \
                auto tensorP2L0A = tla::MakeTensor(p2L0ATensorRaw, p2LayoutAInL0, Catlass::Arch::PositionL0A{}); \
                auto tensorP2L0B = tla::MakeTensor(p2L0BTensorRaw, p2LayoutBInL0, Catlass::Arch::PositionL0B{}); \
                auto tensorP2TileL1A = GetTile(tensorP2L1A, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(dvGemmShape.m(), dvGemmShape.k()));                        \
                auto tensorP2TileL1B = GetTile(tensorP2L1B, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(dvGemmShape.k(), dvGemmShape.n()));                        \
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(P2_EVENT_A);                                    \
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(prefetchEventB);                                \
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventA);                                           \
                p2CopyL1ToL0A(tensorP2L0A, tensorP2TileL1A);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventA);                                            \
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventB);                                           \
                p2CopyL1ToL0B(tensorP2L0B, tensorP2TileL1B);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventB);                                            \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P2_EVENT_A);                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(prefetchEventB);                                 \
                if (p2WsPrefetched) {                                                                             \
                    prefetchedKqMatrixHead = -1;                                                                      \
                }                                                                                                 \
            } while (0)

// 发射 P2 MMAD，结果留在 L0C；写回动作延后到下一段流水中执行，用来覆盖后续预取。
#define ISSUE_DV_MMAD(P2_HEAD_INDEX)                                                                    \
            do {                                                                                                  \
                int64_t dvHeadIndex = (P2_HEAD_INDEX);                                                           \
                uint32_t l0Stage = 0;                                                                             \
                uint32_t cStage = static_cast<uint32_t>(dvHeadIndex + 1) & 1;                                    \
                int32_t eventA = (l0Stage == 0) ? P2_EVENT_A0 : P2_EVENT_A1;                                     \
                int32_t eventB = (l0Stage == 0) ? P2_EVENT_B0 : P2_EVENT_B1;                                     \
                int32_t eventC = (cStage == 0) ? P2_EVENT_C0 : P2_EVENT_C1;                                      \
                auto p2LayoutAInL0 = tla::MakeLayout<ElementA, P2_LayoutTagL0A>(dvGemmShape.m(), dvGemmShape.k());       \
                auto p2LayoutBInL0 = tla::MakeLayout<ElementB, P2_LayoutTagL0B>(dvGemmShape.k(), dvGemmShape.n());       \
                auto p2L0ATensorRaw = (l0Stage == 0) ? p2L0ATensorRaw0 : p2L0ATensorRaw1;                       \
                auto p2L0BTensorRaw = (l0Stage == 0) ? p2L0BTensorRaw0 : p2L0BTensorRaw1;                       \
                auto tensorP2L0A = tla::MakeTensor(p2L0ATensorRaw, p2LayoutAInL0, Catlass::Arch::PositionL0A{}); \
                auto tensorP2L0B = tla::MakeTensor(p2L0BTensorRaw, p2LayoutBInL0, Catlass::Arch::PositionL0B{}); \
                auto p2LayoutInL0C = tla::MakeLayoutL0C(dvGemmShape.m(), dvGemmShape.n());                              \
                auto p2L0CTensorRaw = (cStage == 0) ? p2L0CTensorRaw0 : p2L0CTensorRaw1;                        \
                auto tensorP2L0C = tla::MakeTensor(p2L0CTensorRaw, p2LayoutInL0C, Catlass::Arch::PositionL0C{}); \
                auto tensorP2TileL0C = GetTile(tensorP2L0C, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(dvGemmShape.m(), dvGemmShape.n()));                        \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventA);                                           \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventB);                                           \
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventC);                                            \
                p2TileMmad(tensorP2TileL0C, tensorP2L0A, tensorP2L0B, true, 0);                                  \
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventA);                                            \
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventB);                                            \
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(eventC);                                             \
                pendingDvHead = dvHeadIndex;                                                                     \
            } while (0)

// 将 P2 L0C 中的 dV 结果写回 GM，完成当前 head 的最终输出。
#define STORE_DV_HEAD(P2_HEAD_INDEX)                                                                             \
            do {                                                                                                  \
                int64_t dvHeadIndex = (P2_HEAD_INDEX);                                                           \
                uint32_t cStage = static_cast<uint32_t>(dvHeadIndex + 1) & 1;                                    \
                int32_t eventC = (cStage == 0) ? P2_EVENT_C0 : P2_EVENT_C1;                                      \
                int64_t dvOffset =                                                                               \
                    chunkTask.batchId * headNum * seqLen * valueDim + dvHeadIndex * seqLen * valueDim + chunkTask.tokenStart * valueDim;       \
                auto p2TensorC = tla::MakeTensor(dVGm[dvOffset], p2LayoutC, Catlass::Arch::PositionGM{});        \
                auto p2TensorBlockC = GetTile(p2TensorC, tla::MakeCoord(0, 0),                                   \
                                              tla::MakeShape(dvGemmShape.m(), dvGemmShape.n()));                         \
                using P2_CopyL0CToGm = typename P2_TileCopy::template CopyL0CToGm<decltype(p2TensorBlockC)>;     \
                P2_CopyL0CToGm p2CopyL0CToGm;                                                                    \
                auto p2LayoutInL0C = tla::MakeLayoutL0C(dvGemmShape.m(), dvGemmShape.n());                              \
                auto p2L0CTensorRaw = (cStage == 0) ? p2L0CTensorRaw0 : p2L0CTensorRaw1;                        \
                auto tensorP2L0C = tla::MakeTensor(p2L0CTensorRaw, p2LayoutInL0C, Catlass::Arch::PositionL0C{}); \
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(eventC);                                            \
                p2CopyL0CToGm(p2TensorBlockC, tensorP2L0C);                                                      \
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventC);                                             \
            } while (0)

// P1 head 的快捷封装：完成 K @ Q^T 后立即通知 AIV 可以读取 workspace 做 gating。
#define RUN_KQ_HEAD(P1_HEAD_INDEX, PREFETCH_DO)                                                                  \
            do {                                                                                                  \
                PREFETCH_KQ_HEAD_TO_L1(P1_HEAD_INDEX);                                                            \
                COMPUTE_LOADED_KQ_HEAD(PREFETCH_DO);                                                             \
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_3);                                   \
            } while (0)

// V=256 时 P2 按 N 维切成 128 列 tile。AIV 仍然每个 head 只通知一次，
// AIC 在收到 gated Ws ready 后串行完成该 head 的所有 V tile。
#define RUN_DV_HEAD_VALUE_TILE(P2_HEAD_INDEX, P2_VALUE_OFFSET, P2_TILE_N)                                          \
            do {                                                                                                  \
                int64_t dvHeadIndex = (P2_HEAD_INDEX);                                                           \
                uint32_t p2ValueOffset = static_cast<uint32_t>(P2_VALUE_OFFSET);                                  \
                uint32_t p2TileN = static_cast<uint32_t>(P2_TILE_N);                                              \
                Catlass::GemmCoord dvTileShape{static_cast<uint32_t>(dvGemmShape.m()), p2TileN,                  \
                                               static_cast<uint32_t>(dvGemmShape.k())};                          \
                int64_t wsOffset = GetCoreBufferSingleHWorkspaceOffset(coreIdx, dvHeadIndex, 0,                  \
                                                                       strategy.chunkSize, workspaceHeadSlotNum); \
                int64_t doOffset =                                                                               \
                    chunkTask.batchId * headNum * seqLen * valueDim +                                             \
                    dvHeadIndex * seqLen * valueDim + chunkTask.tokenStart * valueDim + p2ValueOffset;           \
                int64_t dvOffset =                                                                               \
                    chunkTask.batchId * headNum * seqLen * valueDim +                                             \
                    dvHeadIndex * seqLen * valueDim + chunkTask.tokenStart * valueDim + p2ValueOffset;           \
                auto p2TensorA = tla::MakeTensor(workspaceGm[wsOffset], p2LayoutA, Catlass::Arch::PositionGM{}); \
                auto p2TensorBlockA = GetTile(p2TensorA, tla::MakeCoord(0, 0),                                   \
                                              tla::MakeShape(dvTileShape.m(), dvTileShape.k()));                 \
                auto doTensorB = tla::MakeTensor(dOGm[doOffset], p2LayoutB, Catlass::Arch::PositionGM{});        \
                auto doTensorBlockB = GetTile(doTensorB, tla::MakeCoord(0, 0),                                   \
                                              tla::MakeShape(dvTileShape.k(), dvTileShape.n()));                 \
                auto p2TensorC = tla::MakeTensor(dVGm[dvOffset], p2LayoutC, Catlass::Arch::PositionGM{});        \
                auto p2TensorBlockC = GetTile(p2TensorC, tla::MakeCoord(0, 0),                                   \
                                              tla::MakeShape(dvTileShape.m(), dvTileShape.n()));                 \
                using P2_CopyGmToL1A = typename P2_TileCopy::template CopyGmToL1A<decltype(p2TensorBlockA)>;     \
                using P2_CopyDoGmToL1B = typename P2_TileCopy::template CopyGmToL1B<decltype(doTensorBlockB)>;   \
                using P2_CopyL0CToGm = typename P2_TileCopy::template CopyL0CToGm<decltype(p2TensorBlockC)>;     \
                P2_CopyGmToL1A p2CopyGmToL1A;                                                                    \
                P2_CopyDoGmToL1B p2CopyDoGmToL1B;                                                                \
                P2_CopyL0CToGm p2CopyL0CToGm;                                                                    \
                auto tensorP2L1A = tla::MakeTensor(p2L1ATensorRaw0, p2L1ALayout, Catlass::Arch::PositionL1{});   \
                auto tensorP2L1B = tla::MakeTensor(p2L1BTensorRaw0, p2L1BLayout, Catlass::Arch::PositionL1{});   \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P2_EVENT_A);                                    \
                p2CopyGmToL1A(tensorP2L1A, p2TensorBlockA);                                                      \
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(P2_EVENT_A);                                     \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P2_PREFETCH_EVENT_B0);                          \
                p2CopyDoGmToL1B(tensorP2L1B, doTensorBlockB);                                                    \
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(P2_PREFETCH_EVENT_B0);                           \
                auto p2LayoutAInL0 = tla::MakeLayout<ElementA, P2_LayoutTagL0A>(dvTileShape.m(), dvTileShape.k()); \
                auto p2LayoutBInL0 = tla::MakeLayout<ElementB, P2_LayoutTagL0B>(dvTileShape.k(), dvTileShape.n()); \
                auto tensorP2L0A = tla::MakeTensor(p2L0ATensorRaw0, p2LayoutAInL0, Catlass::Arch::PositionL0A{}); \
                auto tensorP2L0B = tla::MakeTensor(p2L0BTensorRaw0, p2LayoutBInL0, Catlass::Arch::PositionL0B{}); \
                auto tensorP2TileL1A = GetTile(tensorP2L1A, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(dvTileShape.m(), dvTileShape.k()));                \
                auto tensorP2TileL1B = GetTile(tensorP2L1B, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(dvTileShape.k(), dvTileShape.n()));                \
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(P2_EVENT_A0);                                      \
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(P2_EVENT_A);                                    \
                p2CopyL1ToL0A(tensorP2L0A, tensorP2TileL1A);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(P2_EVENT_A0);                                       \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P2_EVENT_A);                                     \
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(P2_EVENT_B0);                                      \
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(P2_PREFETCH_EVENT_B0);                          \
                p2CopyL1ToL0B(tensorP2L0B, tensorP2TileL1B);                                                     \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(P2_EVENT_B0);                                       \
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(P2_PREFETCH_EVENT_B0);                           \
                auto p2LayoutInL0C = tla::MakeLayoutL0C(dvTileShape.m(), dvTileShape.n());                       \
                auto tensorP2L0C = tla::MakeTensor(p2L0CTensorRaw1, p2LayoutInL0C, Catlass::Arch::PositionL0C{}); \
                auto tensorP2TileL0C = GetTile(tensorP2L0C, tla::MakeCoord(0, 0),                                \
                                               tla::MakeShape(dvTileShape.m(), dvTileShape.n()));                \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(P2_EVENT_A0);                                      \
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(P2_EVENT_B0);                                      \
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(P2_EVENT_C1);                                       \
                p2TileMmad(tensorP2TileL0C, tensorP2L0A, tensorP2L0B, true, 0);                                  \
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(P2_EVENT_A0);                                       \
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(P2_EVENT_B0);                                       \
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(P2_EVENT_C1);                                        \
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(P2_EVENT_C1);                                       \
                p2CopyL0CToGm(p2TensorBlockC, tensorP2L0C);                                                      \
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(P2_EVENT_C1);                                        \
            } while (0)

            if (valueDim > P2_VALUE_TILE_N) {
                for (int64_t hIndex = 0; hIndex < headNum; ++hIndex) {
                    RUN_KQ_HEAD(hIndex, do {} while (0));
                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_2);
                    for (uint32_t vOffset = 0; vOffset < static_cast<uint32_t>(valueDim);
                         vOffset += P2_VALUE_TILE_N) {
                        uint32_t tileN = static_cast<uint32_t>(valueDim) - vOffset;
                        if (tileN > P2_VALUE_TILE_N) {
                            tileN = P2_VALUE_TILE_N;
                        }
                        RUN_DV_HEAD_VALUE_TILE(hIndex, vOffset, tileN);
                    }
                }
            } else {

            // Prologue：先生成前几个 head 的 Ws，让 AIV 尽早开始 gating，同时顺手预取前两个 dO。
            RUN_KQ_HEAD(0, do {} while (0));
            if (headNum > 1) {
                RUN_KQ_HEAD(1, PREFETCH_NEXT_DO_IF_ANY());
            }
            if (headNum > 2) {
                RUN_KQ_HEAD(2, PREFETCH_NEXT_DO_IF_ANY());
            }
            int64_t nextKqHead = 3;
            // Steady：当前 head 做 P2，下一 head 预取/生成 P1，形成 KQ、gating、dV 三段流水。
            for (int64_t hIndex = 0; hIndex < headNum; ++hIndex) {
                ENSURE_DO_READY(hIndex);
                PREPARE_DV_HEAD_TO_L0(hIndex);
                ISSUE_DV_MMAD(hIndex);
                if (nextKqHead < headNum && loadedKqHead < 0) {
                    PREFETCH_KQ_HEAD_TO_L1(nextKqHead);
                    ++nextKqHead;
                }
                PREFETCH_KQ_MATRIX_TO_L1A(hIndex + 1);
                if (pendingDvHead >= 0) {
                    STORE_DV_HEAD(pendingDvHead);
                    pendingDvHead = -1;
                }
                if (loadedKqHead >= 0) {
                    COMPUTE_LOADED_KQ_HEAD(PREFETCH_NEXT_DO_IF_ANY());
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_3);
                }
            }
            }
            // Epilogue：离开 chunk 前回收所有事件，确保下一轮复用同一片片上资源时状态干净。
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P2_PREFETCH_EVENT_B0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P2_PREFETCH_EVENT_B1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P2_PREFETCH_EVENT_B2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P2_EVENT_A);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(P2_EVENT_A0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(P2_EVENT_B0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(P2_EVENT_C0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(P2_EVENT_C1);
#undef STORE_DV_HEAD
#undef ISSUE_DV_MMAD
#undef PREPARE_DV_HEAD_TO_L0
#undef RUN_DV_HEAD_VALUE_TILE
#undef PREFETCH_KQ_MATRIX_TO_L1A
#undef ENSURE_DO_READY
#undef PREFETCH_NEXT_DO_IF_ANY
#undef RUN_KQ_HEAD
#undef COMPUTE_LOADED_KQ_HEAD
#undef PREFETCH_KQ_HEAD_TO_L1
#undef PREFETCH_DO_HEAD_TO_L1
#undef RUN_KQ_TILE_MMAD
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P1_EVENT_A0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P1_EVENT_A1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P1_EVENT_B0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(P1_EVENT_B1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(P1_EVENT_A0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(P1_EVENT_A1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(P1_EVENT_B0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(P1_EVENT_B1);
            AscendC::SetMMLayoutTransform(false);
        } // ~head-level rolling pipeline
    }

}

} // namespace GDN
#endif // CHUNK_BWD_DV_LOCAL_CUBE_FIX_H
