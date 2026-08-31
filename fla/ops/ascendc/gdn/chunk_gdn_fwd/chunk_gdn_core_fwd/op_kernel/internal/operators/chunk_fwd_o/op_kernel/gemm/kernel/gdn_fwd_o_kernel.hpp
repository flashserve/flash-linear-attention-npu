/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../arch35/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp"
#include "../../arch35/epilogue/block/block_epilogue_gdn_fwdo_output.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "../../arch35/gemm/block/block_mmad_pingpong_tla_gdn_fwd_o.hpp"
#include "../../arch35/gemm/block/block_mmad_pingpong_tla_pipelined.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "../block/block_scheduler_gdn_fwd_o.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using _0 = tla::Int<0>;
using _1 = tla::Int<1>;
using _2 = tla::Int<2>;
using _4 = tla::Int<4>;
using _8 = tla::Int<8>;
using _16 = tla::Int<16>;
using _32 = tla::Int<32>;
using _64 = tla::Int<64>;
using _128 = tla::Int<128>;
using _256 = tla::Int<256>;
using _512 = tla::Int<512>;
using _1024 = tla::Int<1024>;
using _2048 = tla::Int<2048>;
using _4096 = tla::Int<4096>;
using _8192 = tla::Int<8192>;
using _16384 = tla::Int<16384>;
using _32768 = tla::Int<32768>;
using _65536 = tla::Int<65536>;

#else
#define CATLASS_ARCH 2201

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_output.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "../block/block_scheduler_gdn_fwd_o.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#endif

#include "kernel_operator.h"
#include "../../chunk_fwd_o_struct.h"
using namespace Catlass;
using namespace tla;

// template <>
namespace Catlass::Gemm::Kernel {

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename WORKSPACE_TYPE,
    bool kChunkPipeline = false
>
class GDNFwdOKernel {
public:

    static constexpr uint32_t HO_PIPELINE_CHUNK_SIZE = 64;
    static constexpr uint32_t HO_PIPELINE_VALUE_HEADS = 8;
    static constexpr uint32_t HO_PIPELINE_VALUE_DIM = 128;
    static constexpr uint32_t HO_PIPELINE_CUBE_CORES = 24;
    static constexpr uint32_t HO_PIPELINE_EVENT_COUNT = 2;
    static constexpr uint32_t HO_PIPELINE_SYNC_UB_OFFSET = 188 * 1024;
    static constexpr uint64_t HO_PIPELINE_WORKSPACE_ALIGNMENT = 512;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Arch::Ascend950;
#else
    using ArchTag = Arch::AtlasA2;
#endif
    using GDNFwdOOffsets = Catlass::Gemm::Block::GDNFwdOOffsets;

    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOVec;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using DispatchPolicyTla = Gemm::MmadPingpongTlaGdnFwdO<ArchTag, true, false>;
#else
    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
#endif
    using L1TileShapeQKTla = Shape<_128, _128, _128>;
    using L0TileShapeQKTla = L1TileShapeQKTla;
    using L1TileShapeV128Tla = Shape<_128, _128, _128>;
    using L0TileShapeV128Tla = L1TileShapeV128Tla;
    using L1TileShapeV256Tla = Shape<_128, _256, _128>;
    using L0TileShapeV256Tla = Shape<_128, _256, _64>;
    using QType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using AttenType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using AttenMaskedType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using OinterType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VNEWType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;

    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using OType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using MaskType = Gemm::GemmType<bool, layout::RowMajor>;

    // cube 1
    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::ColumnMajor, WORKSPACE_TYPE, layout::RowMajor>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using BlockMmadQK = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTla, L1TileShapeQKTla, L0TileShapeQKTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;
#else
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeQKTla, L0TileShapeQKTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;
#endif

    // cube 2
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using DispatchPolicyTlaShareL1A = Gemm::MmadPingpongTlaGdnFwdO<ArchTag, true, false, 1, false, true>;
    using BlockMmadQH128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaShareL1A, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadQH256 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaShareL1A, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
#else
    using BlockMmadQH128 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadQH256 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
#endif

    // cube 3
    using TileCopyAttenVNEW = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using BlockMmadAttenVNEW128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTla, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
    using BlockMmadAttenVNEW256 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTla, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
#else
    using BlockMmadAttenVNEW128 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
    using BlockMmadAttenVNEW256 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
#endif

    // vec 1
    using DispatchPolicyGDNFwdOQkmask = Epilogue::EpilogueAtlasGDNFwdOQkmask;
    using EpilogueGDNFwdOQkmask = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOQkmask, AttenMaskedType, GType, AttenType, MaskType>;

    // vec 2
    using DispatchPolicyGDNFwdOOutput = Epilogue::EpilogueAtlasGDNFwdOOutput;
    using EpilogueGDNFwdOOutput = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOOutput, OType, GType, OinterType, OinterType>;

    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = Catlass::layout::RowMajor;

    using ElementK =  typename BlockMmadQK::ElementB;
    using LayoutK = Catlass::layout::ColumnMajor;

    using ElementAtten = typename BlockMmadQK::ElementC;
    using LayoutAtten = Catlass::layout::RowMajor;

    using ElementAttenMasked = typename BlockMmadQH128::ElementA;
    using LayoutAttenMasked = Catlass::layout::RowMajor;

    using ElementH = typename BlockMmadQH128::ElementB;
    using LayoutH = Catlass::layout::RowMajor;

    using ElementOinter = typename BlockMmadQH128::ElementC;
    using LayoutOinter = Catlass::layout::RowMajor;


    using ElementVNEW = typename BlockMmadAttenVNEW128::ElementB;
    using LayoutVNEW = Catlass::layout::RowMajor;


    using ElementG = G_TYPE;
    using ElementMask = bool;

    using L1TileShape = typename BlockMmadQK::L1TileShape;

    uint32_t shapeBatch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    float scale;
    uint32_t numChunks;
    uint32_t isVariedLen;
    uint32_t tokenBatch;
    int64_t vWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t attnWorkspaceOffset;
    int64_t aftermaskWorkspaceOffset;
    int64_t maskWorkspaceOffset;

    AscendC::GlobalTensor<ElementQ> gmQ;
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementVNEW> gmV;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementVNEW> gmO;
    AscendC::GlobalTensor<ElementOinter> gmVWorkspace;
    AscendC::GlobalTensor<ElementOinter> gmHWorkspace;
    AscendC::GlobalTensor<ElementAtten> gmAttnWorkspace;
    AscendC::GlobalTensor<ElementAttenMasked> gmAftermaskWorkspace;
    AscendC::GlobalTensor<ElementMask> gmMask;
    AscendC::GlobalTensor<int32_t> gmPipelineSync;

    bool chunkPipelineEnabled{false};
    bool taskAffinityEnabled{false};

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;

    __aicore__ inline uint64_t PipelineVNewBytes() const
    {
        const uint64_t bytes = static_cast<uint64_t>(shapeBatch) * vNumHead * seqlen * vHeadDim *
                               sizeof(ElementVNEW);
        return (bytes + HO_PIPELINE_WORKSPACE_ALIGNMENT - 1) / HO_PIPELINE_WORKSPACE_ALIGNMENT *
               HO_PIPELINE_WORKSPACE_ALIGNMENT;
    }

    __aicore__ inline bool CanRunChunkPipeline() const
    {
        if constexpr (!kChunkPipeline) {
            return false;
        }
        return isVariedLen == 0 && shapeBatch == 1 && vNumHead == HO_PIPELINE_VALUE_HEADS &&
               vHeadDim == HO_PIPELINE_VALUE_DIM &&
               (chunkSize == HO_PIPELINE_CHUNK_SIZE || chunkSize == 2 * HO_PIPELINE_CHUNK_SIZE) &&
               AscendC::GetBlockNum() == HO_PIPELINE_CUBE_CORES;
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> GetPipelineSyncLocal()
    {
        return resource.ubBuf.template GetBufferByByte<int32_t>(HO_PIPELINE_SYNC_UB_OFFSET);
    }

    __aicore__ inline void WaitChunkReady(const GDNFwdOOffsets &offsets)
    {
        if (!chunkPipelineEnabled) {
            return;
        }
        const uint32_t producerAivIdx = offsets.headIdx * AscendC::GetSubBlockNum() +
                                        AscendC::GetSubBlockIdx();
        const uint32_t eventId = offsets.chunkIdx % HO_PIPELINE_EVENT_COUNT;
        AscendC::IBWait<false>(gmPipelineSync, GetPipelineSyncLocal(), producerAivIdx, eventId);
    }

    __aicore__ inline GDNFwdOKernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
        GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, const GDN::ChunkFwdOTilingData *tilingData, GM_ADDR user) {

        shapeBatch = tilingData->shapeBatch;
        seqlen = tilingData->seqlen;
        kNumHead = tilingData->kNumHead;
        vNumHead = tilingData->vNumHead;
        kHeadDim = tilingData->kHeadDim;
        vHeadDim = tilingData->vHeadDim;
        scale = tilingData->scale;
        chunkSize = tilingData->chunkSize;
        isVariedLen = tilingData->isVariedLen;
        tokenBatch = tilingData->tokenBatch;
        vWorkspaceOffset = tilingData->vWorkspaceOffset;
        hWorkspaceOffset = tilingData->hWorkspaceOffset;
        attnWorkspaceOffset = tilingData->attnWorkspaceOffset;
        aftermaskWorkspaceOffset = tilingData->aftermaskWorkspaceOffset;
        maskWorkspaceOffset = tilingData->maskWorkspaceOffset;

        gmQ.SetGlobalBuffer((__gm__ ElementQ *)q);
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmV.SetGlobalBuffer((__gm__ ElementVNEW *)v);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmO.SetGlobalBuffer((__gm__ ElementVNEW *)o);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + vWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + hWorkspaceOffset));
        gmAttnWorkspace.SetGlobalBuffer((__gm__ ElementAtten *)(user + attnWorkspaceOffset));
        gmAftermaskWorkspace.SetGlobalBuffer((__gm__ ElementAttenMasked *)(user + aftermaskWorkspaceOffset));
        gmMask.SetGlobalBuffer((__gm__ ElementMask *)(user + maskWorkspaceOffset));

        chunkPipelineEnabled = CanRunChunkPipeline();
        taskAffinityEnabled = kChunkPipeline && !chunkPipelineEnabled && (isVariedLen == 0);
        if (chunkPipelineEnabled) {
            gmPipelineSync.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(v + PipelineVNewBytes()));
        }

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_offsets, tilingData,
                                    chunkPipelineEnabled, taskAffinityEnabled);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_offsets, tilingData,
                                   chunkPipelineEnabled, taskAffinityEnabled);
        }
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadQK blockMmadQK(resource);
            BlockMmadQH128 blockMmadQH128(resource);
            BlockMmadQH256 blockMmadQH256(resource);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // A5 local-resource lifetime:
            //   Cube1 L1A/Q [0,64K), L1B/K [64K,128K), events A=0/1 B=2/3.
            //   Cube2 shares L1A/Q and reuses B [64K,128K|192K), events 0..3.
            //   Cube3 owns A [192K,256K), B [256K,320K|384K), events 4..7.
            // Shared Q/K-H slots use the event pair for producer/consumer
            // RAW+WAR ordering; Cube3 has no physical/event WAW alias. L0A/B/C
            // stay shared and each compute window is drained before the next.
            // Cross-core flags remain scheduler-owned (cube1/vec1/cube3/vec2 =
            // 0..7); their ping-pong init and final vec2 drain are unchanged.
            constexpr uint32_t cube3L1Offset = 192 * 1024;
            constexpr uint32_t cube3L1AEventId = 4;
            constexpr uint32_t cube3L1BEventId = 6;
            BlockMmadAttenVNEW128 blockMmadAttenVNEW128(
                resource, cube3L1Offset, cube3L1AEventId, cube3L1BEventId);
            BlockMmadAttenVNEW256 blockMmadAttenVNEW256(
                resource, cube3L1Offset, cube3L1AEventId, cube3L1BEventId);
#else
            BlockMmadAttenVNEW128 blockMmadAttenVNEW128(resource);
            BlockMmadAttenVNEW256 blockMmadAttenVNEW256(resource);
#endif

            auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
            auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(
                coreNum * chunkSize * GDN_FWD_O_PING_PONG_STAGES, cubeBlockScheduler.vBlockSize);
            auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

            bool needRun = false;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();

                // Phase 1a: launch Cube1, then release only the L1 producer
                // events. Its MMAD/FIX pipeline remains in flight.
                if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetCurStageId();
                    GDNFwdOOffsets &cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(
                        coreNum * chunkSize * GDN_FWD_O_PING_PONG_STAGES,
                        cube1Offsets.blockTokens);
                    auto tensorQ = tla::MakeTensor(
                        gmQ[cube1Offsets.qkOffset], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(
                        gmK[cube1Offsets.qkOffset], kLayout, Catlass::Arch::PositionGM{});
                    auto tensorAttn = tla::MakeTensor(
                        gmAttnWorkspace[cube1Offsets.attnWorkOffset], attenLayout,
                        Catlass::Arch::PositionGM{});
                    GemmCoord cube1Shape{
                        cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
                    auto tensorBlockQ = GetTile(
                        tensorQ, tla::MakeCoord(0, 0),
                        tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                    auto tensorBlockK = GetTile(
                        tensorK, tla::MakeCoord(0, 0),
                        tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                    auto tensorBlockAttn = GetTile(
                        tensorAttn, tla::MakeCoord(0, 0),
                        tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                    blockMmadQK.preSetFlags();
                    blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                    blockMmadQK.waitL1Drained();
                    (void)streamId;
                }

                // Phase 1b: Cube2 reuses Cube1's Q in L1 and preloads H while
                // Cube1 drains its MMAD pipeline. Shared L1 events enforce the
                // RAW/WAR ordering for the overlapping physical buffers.
                if (needRun && coreIdx < coreNum) {
                    GDNFwdOOffsets &cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                    auto tensorQ = tla::MakeTensor(
                        gmQ[cube2Offsets.qkOffset], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(
                        gmH[cube2Offsets.hOffset], hLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube2Shape{
                        cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                    auto tensorBlockQ = GetTile(
                        tensorQ, tla::MakeCoord(0, 0),
                        tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                    auto tensorBlockH = GetTile(
                        tensorH, tla::MakeCoord(0, 0),
                        tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                    if (cube2Offsets.vBlockDim <= 128) {
                        blockMmadQH128.preSetL1Flags();
                        blockMmadQH128.copyGmToL1(
                            tensorBlockQ, tensorBlockH, cube2Shape);
                    } else {
                        blockMmadQH256.preSetL1Flags();
                        blockMmadQH256.copyGmToL1(
                            tensorBlockQ, tensorBlockH, cube2Shape);
                    }
                }

                // Phase 1c: Cube1 no longer owns L0A/L0B. PIPE_FIX ordering
                // makes the workspace visible before Vec1 consumes it.
                if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetCurStageId();
                    blockMmadQK.waitL0Drained();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        cubeBlockScheduler.cube1Done[streamId]);
                }

                if (needRun && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetPrevStageId();
                    GDNFwdOOffsets &cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                    GDNFwdOOffsets &cube3Offsets = cubeBlockScheduler.GetCube23Offsets();

                    // H/V workspaces are ping-pong slots. Delay this wait until
                    // the first operation that can overwrite the previous slot.
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);

                    auto tensorHWork = tla::MakeTensor(
                        gmHWorkspace[cube2Offsets.hvWorkOffset], ointerLayout,
                        Catlass::Arch::PositionGM{});
                    GemmCoord cube2Shape{
                        cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                    auto tensorBlockHWork = GetTile(
                        tensorHWork, tla::MakeCoord(0, 0),
                        tla::MakeShape(cube2Shape.m(), cube2Shape.n()));

                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(
                        coreNum * chunkSize * GDN_FWD_O_PING_PONG_STAGES,
                        cube3Offsets.blockTokens);
                    auto tensorAttnMask = tla::MakeTensor(
                        gmAftermaskWorkspace[cube3Offsets.attnWorkOffset], attenLayout,
                        Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(
                        gmV[cube3Offsets.ovOffset], vnewLayout,
                        Catlass::Arch::PositionGM{});
                    GemmCoord cube3Shape{
                        cube3Offsets.blockTokens, cube3Offsets.vBlockDim,
                        cube3Offsets.blockTokens};
                    auto tensorBlockAttnMask = GetTile(
                        tensorAttnMask, tla::MakeCoord(0, 0),
                        tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                    auto tensorBlockV = GetTile(
                        tensorV, tla::MakeCoord(0, 0),
                        tla::MakeShape(cube3Shape.k(), cube3Shape.n()));

                    // V has no dependency on Vec1, so preload Cube3 L1B before
                    // Cube2's compute. Cube3 owns L1 events 6/7.
                    if (cube3Offsets.vBlockDim <= 128) {
                        blockMmadAttenVNEW128.preSetL1Flags();
                        blockMmadAttenVNEW128.copyGmToL1BOnly(
                            tensorBlockV, cube3Shape);
                    } else {
                        blockMmadAttenVNEW256.preSetL1Flags();
                        blockMmadAttenVNEW256.copyGmToL1BOnly(
                            tensorBlockV, cube3Shape);
                    }

                    // Cube2 compute overlaps Cube3's MTE2 preload. L0 resources
                    // remain serialized and are drained before Cube3 starts.
                    if (cube2Offsets.vBlockDim <= 128) {
                        blockMmadQH128.preSetL0Flags();
                        blockMmadQH128.executeCompute(
                            tensorBlockHWork, cube2Shape);
                    } else {
                        blockMmadQH256.preSetL0Flags();
                        blockMmadQH256.executeCompute(
                            tensorBlockHWork, cube2Shape);
                    }

                    // AttnMask is a Vec1 product; load it only after Vec1's
                    // MTE3 release. Cube3 L1A owns events 4/5.
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
                    if (cube3Offsets.vBlockDim <= 128) {
                        blockMmadAttenVNEW128.copyGmToL1AOnly(
                            tensorBlockAttnMask, cube3Shape);
                    } else {
                        blockMmadAttenVNEW256.copyGmToL1AOnly(
                            tensorBlockAttnMask, cube3Shape);
                    }

                    if (cube2Offsets.vBlockDim <= 128) {
                        blockMmadQH128.finalWaitFlags();
                    } else {
                        blockMmadQH256.finalWaitFlags();
                    }

                    auto tensorVWork = tla::MakeTensor(
                        gmVWorkspace[cube3Offsets.hvWorkOffset], ointerLayout,
                        Catlass::Arch::PositionGM{});
                    auto tensorBlockVWork = GetTile(
                        tensorVWork, tla::MakeCoord(0, 0),
                        tla::MakeShape(cube3Shape.m(), cube3Shape.n()));
                    if (cube3Offsets.vBlockDim <= 128) {
                        blockMmadAttenVNEW128.preSetL0Flags();
                        blockMmadAttenVNEW128.executeCompute(
                            tensorBlockVWork, cube3Shape);
                        blockMmadAttenVNEW128.finalWaitFlags();
                    } else {
                        blockMmadAttenVNEW256.preSetL0Flags();
                        blockMmadAttenVNEW256.executeCompute(
                            tensorBlockVWork, cube3Shape);
                        blockMmadAttenVNEW256.finalWaitFlags();
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        cubeBlockScheduler.cube3Done[streamId]);
                }
                needRun = true;
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
#else
            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();

                if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetCurStageId();

                    GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                    int64_t cube1OffsetQ = cube1Offsets.qkOffset;
                    int64_t cube1OffsetK = cube1Offsets.qkOffset;
                    int64_t cube1OffsetAttn = cube1Offsets.attnWorkOffset;
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(
                        coreNum * chunkSize * GDN_FWD_O_PING_PONG_STAGES, cube1Offsets.blockTokens);
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
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);

                }
                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetPrevStageId();
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
                    // vec2Done protects the H/V workspace consumed by Cube2/3; Cube1 uses a separate slot.
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                    GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube2OffsetQ = cube2Offsets.qkOffset;
                    int64_t cube2OffsetH = cube2Offsets.hOffset;
                    int64_t cube2OffsetHWork = cube2Offsets.hvWorkOffset;
                    auto tensorQ = tla::MakeTensor(gmQ[cube2OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[cube2OffsetH], hLayout, Catlass::Arch::PositionGM{});
                    auto tensorHWork = tla::MakeTensor(gmHWorkspace[cube2OffsetHWork], ointerLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube2Shape{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                    auto tensorBlockHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                    if (cube2Offsets.vBlockDim <= 128) {
                        blockMmadQH128.preSetFlags();
                        blockMmadQH128(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);
                        blockMmadQH128.finalWaitFlags();
                    } else {
                        blockMmadQH256.preSetFlags();
                        blockMmadQH256(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);
                        blockMmadQH256.finalWaitFlags();
                    }
                }

                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::PipeBarrier<PIPE_FIX>();

                if (needRun && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetPrevStageId();
                    GDNFwdOOffsets& cube3Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube3OffsetAttnMask = cube3Offsets.attnWorkOffset;
                    int64_t cube3OffsetV = cube3Offsets.ovOffset;
                    int64_t cube3OffsetVWork = cube3Offsets.hvWorkOffset;
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(
                        coreNum * chunkSize * GDN_FWD_O_PING_PONG_STAGES, cube3Offsets.blockTokens);
                    auto tensorAttnMask = tla::MakeTensor(gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout, Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(gmV[cube3OffsetV], vnewLayout, Catlass::Arch::PositionGM{});
                    auto tensorVWork = tla::MakeTensor(gmVWorkspace[cube3OffsetVWork], ointerLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube3Shape{cube3Offsets.blockTokens, cube3Offsets.vBlockDim, cube3Offsets.blockTokens};
                    auto tensorBlockAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                    auto tensorBlockVWork = GetTile(tensorVWork, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));
                    if (cube3Offsets.vBlockDim <= 128) {
                        blockMmadAttenVNEW128.preSetFlags();
                        blockMmadAttenVNEW128(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
                        blockMmadAttenVNEW128.finalWaitFlags();
                    } else {
                        blockMmadAttenVNEW256.preSetFlags();
                        blockMmadAttenVNEW256(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
                        blockMmadAttenVNEW256.finalWaitFlags();
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube3Done[streamId]);
                }
                needRun = true;
                // AscendC::PipeBarrier<PIPE_ALL>();
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
#endif
        }

        if ASCEND_IS_AIV {

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);

            AscendC::LocalTensor<float> maskUbTensor = resource.ubBuf.template GetBufferByByte<float>(0);
            AscendC::Duplicate<float>(maskUbTensor, (float)0.0, 64*64);
            AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t i = 0; i < 64; ++ i) AscendC::Duplicate<float>(maskUbTensor[i * 64], (float)1.0, i + 1);
            AscendC::PipeBarrier<PIPE_V>();

            bool needRun = false;
            uint32_t pingpongFlag = 0;

            while (vecBlockScheduler.isRunning) {
                vecBlockScheduler.InitTask();

                if (vecBlockScheduler.isRunning && coreIdx < coreNum * subBlockNum) {
                    uint32_t streamId = vecBlockScheduler.GetCurStageId();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done[streamId]);
#endif
                    GDNFwdOOffsets& vec1Offsets = vecBlockScheduler.GetVec1Offsets();
                    WaitChunkReady(vec1Offsets);
                    int64_t vec1OffsetAttnMask = vec1Offsets.attnWorkOffset;
                    int64_t vec1OffsetG = vec1Offsets.gOffset;
                    int64_t vec1OffsetAttn = vec1Offsets.attnWorkOffset;
                    EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource);
                    epilogueGDNFwdOQkmask(
                        gmAftermaskWorkspace[vec1OffsetAttnMask],
                        gmG[vec1OffsetG], gmAttnWorkspace[vec1OffsetAttn], gmMask,
                        chunkSize, vec1Offsets.blockTokens, kHeadDim, vHeadDim, pingpongFlag,
                        vec1Offsets.batchIdx, vec1Offsets.headIdx, vec1Offsets.chunkIdx
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                        , &vecBlockScheduler.cube1Done[streamId]
#endif
                    );
                    if (isVariedLen != 0) {
                        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done[streamId]);
                }

                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum * subBlockNum) {
                    uint32_t streamId = vecBlockScheduler.GetPrevStageId();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube3Done[streamId]);
#endif
                    GDNFwdOOffsets& vec2Offsets = vecBlockScheduler.GetVec2Offsets();
                    int64_t vec2OffsetO = vec2Offsets.ovOffset;
                    int64_t vec2OffsetG = vec2Offsets.gOffset;
                    int64_t vec2OffsetVWork = vec2Offsets.hvWorkOffset;
                    int64_t vec2OffsetHWork = vec2Offsets.hvWorkOffset;
                    EpilogueGDNFwdOOutput epilogueGDNFwdOOutput(resource);
                    epilogueGDNFwdOOutput(
                        gmO[vec2OffsetO],
                        gmG[vec2OffsetG], gmVWorkspace[vec2OffsetVWork], gmHWorkspace[vec2OffsetHWork],
                        scale, vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim,
                        vHeadDim, pingpongFlag, vec2Offsets.batchIdx, vec2Offsets.headIdx,
                        vec2Offsets.chunkIdx
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                        , &vecBlockScheduler.cube3Done[streamId]
                        , isVariedLen == 0 ? &vecBlockScheduler.vec2Done[streamId] : nullptr
#endif
                    );
                    if (isVariedLen != 0) {
                        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    // Varlen must join both AIV generations before publishing
                    // the reusable workspace slot. Fixed-length paths publish
                    // from the epilogue immediately after the final MTE2 read.
                    if (isVariedLen != 0) {
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                            vecBlockScheduler.vec2Done[streamId]);
                    }
#else
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);
#endif
                }
                needRun = true;
            }
        }
    }

};

}
