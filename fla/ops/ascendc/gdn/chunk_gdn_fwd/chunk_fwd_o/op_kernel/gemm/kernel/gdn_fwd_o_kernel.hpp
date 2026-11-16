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
#include "kernel_utils/block/block_mmad_pingpong_tla_pipelined.hpp"
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
    typename WORKSPACE_TYPE
>
class GDNFwdOKernel {
public:

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Arch::Ascend950;
#else
    using ArchTag = Arch::AtlasA2;
#endif
    using GDNFwdOOffsets = Catlass::Gemm::Block::GDNFwdOOffsets;

    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOVec;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
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
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeQKTla, L0TileShapeQKTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;

    // cube 2
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQH128 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadQH256 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;

    // cube 3
    using TileCopyAttenVNEW = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadAttenVNEW128 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
    using BlockMmadAttenVNEW256 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    // varlen 重叠路径类型（仅 A2/arch22）：每个 Cube 实例独占一段 L1 和一对 event id，
    // 装载可以和上一个 Cube 的计算重叠；L0C 分槽见 ProcessOverlapVarlen。V256 不走本路径。
    using DispatchPolicyTlaOvl = Gemm::MmadPingpongTlaGdnFwdO<
        ArchTag,
        /* ENABLE_UNIT_FLAG */ true,
        /* USE_HF32_MODE */ false,
        /* L0C_STAGES */ 1,
        /* ENABLE_L1_RESIDENT */ false,
        /* SHARE_L1A */ false,
        /* L1A_STAGES */ 1,
        /* L1B_STAGES */ 1,
        /* L0A_STAGES */ 2,
        /* L0B_STAGES */ 2>;
    using L1TileShapeOvlTla = Shape<_64, _128, _128>;
    using L0TileShapeOvlTla = L1TileShapeOvlTla;
    using BlockMmadQKOvl = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvlTla, L0TileShapeOvlTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;
    using BlockMmadQHOvl = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvlTla, L0TileShapeOvlTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadAttenVNEWOvl = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvlTla, L0TileShapeOvlTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
    using L1TileShapeOvl128Tla = Shape<_128, _128, _128>;
    using L0TileShapeOvl128Tla = L1TileShapeOvl128Tla;
    using BlockMmadQKOvl128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvl128Tla, L0TileShapeOvl128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;
    using BlockMmadQHOvl128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvl128Tla, L0TileShapeOvl128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadAttenVNEWOvl128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvl128Tla, L0TileShapeOvl128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
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

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;

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

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_offsets, tilingData);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_offsets, tilingData);
        }
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    /// varlen 专属重叠路径：Cube1 处理当前 task，Cube2/Cube3 处理上一个 task，
    /// 三组装载与计算交错重叠（相位见下方 Phase 标注），数值路径与串行版本一致。
    template <typename BlockQKT, typename BlockQHT, typename BlockAVT, uint32_t PART_BYTES,
              uint32_t L0C_QH_OFFSET, uint32_t L0C_AV_OFFSET>
    __aicore__ inline void ProcessOverlapVarlen(uint32_t coreIdx, uint32_t coreNum) {
        // 分区预算：每实例 L1A tile + L1B tile 必须装进 PART_BYTES
        static_assert(BlockQHT::L1A_TILE_SIZE + BlockQHT::L1B_TILE_SIZE <= PART_BYTES,
            "R11/R13: per-instance L1 partition (L1A + L1B tiles) exceeds PART_BYTES");
        // L0C 槽位约束：QK 固定用槽 0，QH/AV 互不相交；槽不相交后相邻 Cube 的 FIX 与 Mmad 可以并发，不需要屏障
        static_assert(BlockQHT::L0C_TILE_SIZE + L0C_QH_OFFSET <= ArchTag::L0C_SIZE,
            "R14: QH L0C slot exceeds L0C_SIZE");
        static_assert(BlockAVT::L0C_TILE_SIZE + L0C_AV_OFFSET <= ArchTag::L0C_SIZE,
            "R14: AV L0C slot exceeds L0C_SIZE");
        static_assert(L0C_QH_OFFSET + BlockQHT::L0C_TILE_SIZE <= L0C_AV_OFFSET ||
                          L0C_AV_OFFSET + BlockAVT::L0C_TILE_SIZE <= L0C_QH_OFFSET,
            "R14: QH/AV L0C slots must be disjoint");

        // 三实例 event id 基：QK 用构造默认 0/1，QH/AV 从 2 起错开，消除跨实例串扰
        constexpr uint32_t qhL1AEventId = 2;
        constexpr uint32_t qhL1BEventId = 3;
        constexpr uint32_t avL1AEventId = 4;
        constexpr uint32_t avL1BEventId = 5;
        BlockQKT blockMmadQKOvl(resource);
        BlockQHT blockMmadQHOvl(resource, PART_BYTES, qhL1AEventId, qhL1BEventId, 0, L0C_QH_OFFSET);
        BlockAVT blockMmadAttenVNEWOvl(resource, PART_BYTES * 2, avL1AEventId, avL1BEventId, 0, L0C_AV_OFFSET);

        auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
        auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
        auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
        auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(coreNum * chunkSize * PING_PONG_STAGES, cubeBlockScheduler.vBlockSize);
        auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

        bool needRun = false;

        while (cubeBlockScheduler.isRunning) {
            cubeBlockScheduler.InitTask();

            // === Phase 1a: Cube1 当前 task（装载+计算+Fixpipe 写出完整流）===
            if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                uint32_t streamId = cubeBlockScheduler.GetCurStageId();
                GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                int64_t cube1OffsetQ = cube1Offsets.qkOffset;
                int64_t cube1OffsetK = cube1Offsets.qkOffset;
                int64_t cube1OffsetAttn = cube1Offsets.attnWorkOffset;
                auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube1Offsets.blockTokens);
                auto tensorQ = tla::MakeTensor(gmQ[cube1OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK[cube1OffsetK], kLayout, Catlass::Arch::PositionGM{});
                auto tensorAttn = tla::MakeTensor(gmAttnWorkspace[cube1OffsetAttn], attenLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
                auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                auto tensorBlockAttn = GetTile(tensorAttn, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                blockMmadQKOvl.preSetFlags();
                blockMmadQKOvl(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                blockMmadQKOvl.waitL1Drained();
            }

            // === Phase 1b: Cube2 GM→L1（Q/H 纯 GM 输入），与 Cube1 在飞的 L0 Mmad 重叠 ===
            if (needRun && coreIdx < coreNum) {
                GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                int64_t cube2OffsetQ = cube2Offsets.qkOffset;
                int64_t cube2OffsetH = cube2Offsets.hOffset;
                auto tensorQ2 = tla::MakeTensor(gmQ[cube2OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                auto tensorH = tla::MakeTensor(gmH[cube2OffsetH], hLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube2Shape{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                auto tensorBlockQ2 = GetTile(tensorQ2, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                blockMmadQHOvl.preSetL1Flags();
                blockMmadQHOvl.copyGmToL1(tensorBlockQ2, tensorBlockH, cube2Shape);
            }

            // === Phase 1c: Cube1 L0 排空 + cube1Done（FIX 管线序）===
            if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                uint32_t streamId = cubeBlockScheduler.GetCurStageId();
                blockMmadQKOvl.waitL0Drained();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);
            }

            // === Phase 2: 上一 task 的 Cube2/Cube3（重叠编排）===
            if (needRun && coreIdx < coreNum) {
                uint32_t streamId = cubeBlockScheduler.GetPrevStageId();
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);

                GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                GDNFwdOOffsets& cube3Offsets = cubeBlockScheduler.GetCube23Offsets();
                int64_t cube2OffsetHWork = cube2Offsets.hvWorkOffset;
                int64_t cube3OffsetAttnMask = cube3Offsets.attnWorkOffset;
                int64_t cube3OffsetV = cube3Offsets.ovOffset;
                int64_t cube3OffsetVWork = cube3Offsets.hvWorkOffset;
                GemmCoord cube2Shape{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                GemmCoord cube3Shape{cube3Offsets.blockTokens, cube3Offsets.vBlockDim, cube3Offsets.blockTokens};
                auto tensorHWork = tla::MakeTensor(gmHWorkspace[cube2OffsetHWork], ointerLayout, Catlass::Arch::PositionGM{});
                auto tensorBlockHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube3Offsets.blockTokens);
                auto tensorAttnMask = tla::MakeTensor(gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout, Catlass::Arch::PositionGM{});
                auto tensorBlockAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                auto tensorV = tla::MakeTensor(gmV[cube3OffsetV], vnewLayout, Catlass::Arch::PositionGM{});
                auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                auto tensorVWork = tla::MakeTensor(gmVWorkspace[cube3OffsetVWork], ointerLayout, Catlass::Arch::PositionGM{});
                auto tensorBlockVWork = GetTile(tensorVWork, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));

                // 2b: Cube3 V 预载（纯 GM 输入，与 Cube2 计算重叠）
                blockMmadAttenVNEWOvl.preSetL1Flags();
                blockMmadAttenVNEWOvl.copyGmToL1BOnly(tensorBlockV, cube3Shape);

                // 2c: Cube2 计算（写 HWork；vec2Done 已等）
                blockMmadQHOvl.preSetL0Flags();
                blockMmadQHOvl.executeCompute(tensorBlockHWork, cube2Shape);
                blockMmadQHOvl.finalWaitFlags();

                // 此处不需要屏障：QH/AV 的 L0C 槽不相交，Cube2 的 FIX 不会与 Cube3 的 Mmad 撞地址；L0A/L0B 仍由 finalWaitFlags 保护

                // 2e: Cube3 AttnMask 装载（依赖 Vec1 的 aftermask 写出）
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
                blockMmadAttenVNEWOvl.copyGmToL1AOnly(tensorBlockAttnMask, cube3Shape);

                // 2f: Cube3 计算（写 VWork）
                blockMmadAttenVNEWOvl.preSetL0Flags();
                blockMmadAttenVNEWOvl.executeCompute(tensorBlockVWork, cube3Shape);
                blockMmadAttenVNEWOvl.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube3Done[streamId]);
            }
            needRun = true;
        }
        // 收尾消耗 AIV 启动时预置的 vec2Done 双槽 credit
        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
    }
#endif

    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            // 重叠路径守卫：仅 chunkSize 64/128、vHeadDim/kHeadDim<=128、序列数达标的
            // varlen 启用；定长和单序列 varlen 无重叠收益，走原串行路径
            constexpr uint32_t OVERLAP_MIN_VARLEN_SEQS = 4;
            const bool useOverlapPath = (chunkSize == 64 || chunkSize == 128) && (vHeadDim <= 128) &&
                                         (kHeadDim <= 128) && (isVariedLen != 0) &&
                                         (tokenBatch >= OVERLAP_MIN_VARLEN_SEQS);
            if (useOverlapPath) {
                if (chunkSize == 64) {
                    // chunk64：L1 分区与 L0C 槽均由 tile 形状推导（L0C 三槽不相交）
                    ProcessOverlapVarlen<BlockMmadQKOvl, BlockMmadQHOvl, BlockMmadAttenVNEWOvl,
                        BlockMmadQKOvl::L1A_TILE_SIZE + BlockMmadQKOvl::L1B_TILE_SIZE,
                        BlockMmadQKOvl::L0C_TILE_SIZE, BlockMmadQKOvl::L0C_TILE_SIZE * 2>(coreIdx, coreNum);
                } else {
                    // chunk128：L1 分区由 tile 形状推导；L0C 双槽——QK/QH 共槽 0（该
                    // 交接有跨核 vec2Done 等待提供间距），AV 独占槽 1
                    ProcessOverlapVarlen<BlockMmadQKOvl128, BlockMmadQHOvl128, BlockMmadAttenVNEWOvl128,
                        BlockMmadQKOvl128::L1A_TILE_SIZE + BlockMmadQKOvl128::L1B_TILE_SIZE,
                        0, BlockMmadQKOvl128::L0C_TILE_SIZE>(coreIdx, coreNum);
                }
                return;
            }
#endif

            BlockMmadQK blockMmadQK(resource);
            BlockMmadQH128 blockMmadQH128(resource);
            BlockMmadQH256 blockMmadQH256(resource);
            BlockMmadAttenVNEW128 blockMmadAttenVNEW128(resource);
            BlockMmadAttenVNEW256 blockMmadAttenVNEW256(resource);

            auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
            auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(coreNum * chunkSize * PING_PONG_STAGES, cubeBlockScheduler.vBlockSize);
            auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

            bool needRun = false;

            // Q 复用：流水线滞后一拍，Cube2 处理的正是 Cube1 上一迭代装载的 task，
            // 其 Q tile 还在 L1A 槽里；源相同（qkOffset/行数一致）即跳过重复装载，
            // 数值路径不变
            int64_t cube1QOffsetSlot[PING_PONG_STAGES] = {-1, -1};
            uint32_t cube1QRowsSlot[PING_PONG_STAGES] = {0, 0};
            // Q 复用收益取决于 Q 的空间局部性，用 HV×B 衡量；低头数长序列形态无
            // 复作可言，达密度阈值才开启。SkipNextADataCopy 依赖 L1A 跨迭代驻留，
            // 仅在 DAV_2201 上验证过，其他架构一律关闭。
            // FORM_DENSITY_MIN_HVxB 与 qkmask epilogue 的同名常量是同一阈值，需同步修改。
            constexpr uint32_t FORM_DENSITY_MIN_HVxB = 16;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            const bool enableQReuse = (vNumHead * shapeBatch >= FORM_DENSITY_MIN_HVxB);
#else
            const bool enableQReuse = false;
#endif

            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();

                if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetCurStageId();

                    GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                    int64_t cube1OffsetQ = cube1Offsets.qkOffset;
                    int64_t cube1OffsetK = cube1Offsets.qkOffset;
                    int64_t cube1OffsetAttn = cube1Offsets.attnWorkOffset;
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube1Offsets.blockTokens);
                    auto tensorQ = tla::MakeTensor(gmQ[cube1OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK[cube1OffsetK], kLayout, Catlass::Arch::PositionGM{});
                    auto tensorAttn = tla::MakeTensor(gmAttnWorkspace[cube1OffsetAttn], attenLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                    auto tensorBlockAttn = GetTile(tensorAttn, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                    // 记录本槽 Q tile 源，供下一迭代 Cube2 复用判定
                    cube1QOffsetSlot[streamId] = cube1OffsetQ;
                    cube1QRowsSlot[streamId] = cube1Offsets.blockTokens;
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
                    // Q 复用仅支持单 k-block（kHeadDim<=128）；更大 K 时多 k-block
                    // 的槽推进不再同步，必须正常装载
                    if (enableQReuse && kHeadDim <= 128 && cube2Offsets.qkOffset == cube1QOffsetSlot[streamId] &&
                        cube2Offsets.blockTokens == cube1QRowsSlot[streamId]) {
                        if (cube2Offsets.vBlockDim <= 128) {
                            blockMmadQH128.SkipNextADataCopy();
                        } else {
                            blockMmadQH256.SkipNextADataCopy();
                        }
                    }
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
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube3Offsets.blockTokens);
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
            // 掩码守卫变量 HV×B（varlen 时取 tokenBatch）：epilogue 按其是否 >=16 选择
            // 掩码实现。tile 化掩码仅在 DAV_2201 验证过，其他架构传 0 回退旧带状。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            const uint32_t hvTimesBatch = static_cast<uint32_t>(
                vNumHead * ((isVariedLen != 0) ? tokenBatch : shapeBatch));
#else
            const uint32_t hvTimesBatch = 0;
#endif

            while (vecBlockScheduler.isRunning) {
                vecBlockScheduler.InitTask();

                if (vecBlockScheduler.isRunning && coreIdx < coreNum * subBlockNum) {
                    uint32_t streamId = vecBlockScheduler.GetCurStageId();
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done[streamId]);
                    GDNFwdOOffsets& vec1Offsets = vecBlockScheduler.GetVec1Offsets();
                    int64_t vec1OffsetAttnMask = vec1Offsets.attnWorkOffset;
                    int64_t vec1OffsetG = vec1Offsets.gOffset;
                    int64_t vec1OffsetAttn = vec1Offsets.attnWorkOffset;
                    EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource);
                    epilogueGDNFwdOQkmask(
                        gmAftermaskWorkspace[vec1OffsetAttnMask],
                        gmG[vec1OffsetG], gmAttnWorkspace[vec1OffsetAttn], gmMask,
                        chunkSize, vec1Offsets.blockTokens, kHeadDim, vHeadDim, pingpongFlag, vec1Offsets.batchIdx, vec1Offsets.headIdx, vec1Offsets.chunkIdx, hvTimesBatch
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done[streamId]);
                }

                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum * subBlockNum) {
                    uint32_t streamId = vecBlockScheduler.GetPrevStageId();
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube3Done[streamId]);
                    GDNFwdOOffsets& vec2Offsets = vecBlockScheduler.GetVec2Offsets();
                    int64_t vec2OffsetO = vec2Offsets.ovOffset;
                    int64_t vec2OffsetG = vec2Offsets.gOffset;
                    int64_t vec2OffsetVWork = vec2Offsets.hvWorkOffset;
                    int64_t vec2OffsetHWork = vec2Offsets.hvWorkOffset;
                    EpilogueGDNFwdOOutput epilogueGDNFwdOOutput(resource);
                    epilogueGDNFwdOOutput(
                        gmO[vec2OffsetO],
                        gmG[vec2OffsetG], gmVWorkspace[vec2OffsetVWork], gmHWorkspace[vec2OffsetHWork],
                        scale, vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim, vHeadDim, pingpongFlag, vec2Offsets.batchIdx, vec2Offsets.headIdx, vec2Offsets.chunkIdx
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);
                }
                needRun = true;
            }
        }
    }

};

}
