/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

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



#include "kernel_operator.h"
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

    using ArchTag = Arch::Ascend950;
    using GDNFwdOOffsets = Catlass::Gemm::Block::GDNFwdOOffsets;

    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOVec;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
    using L1TileShapeTla = Shape<_128, _128, _128>;
    using L0TileShapeTla = L1TileShapeTla;
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

    static constexpr bool ENABLE_REUSE_Q_L1_FOR_QH = true;
    static constexpr bool ENABLE_VEC1_UB_TO_L1_FOR_CUBE3 = true;
    static constexpr uint32_t VEC1_L1_STAGES = PING_PONG_STAGES;
    static constexpr uint32_t VEC1_L1_TILE_SIZE = 128 * 128 * sizeof(INPUT_TYPE);

    // cube 1
    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::ColumnMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;

    // cube 2
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor, void, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadQH = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;

    // cube 3
    using TileCopyAttenVNEW = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor, void, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadAttenVNEW = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;

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

    using ElementAttenMasked = typename BlockMmadQH::ElementA;
    using LayoutAttenMasked = Catlass::layout::RowMajor;

    using ElementH = typename BlockMmadQH::ElementB;
    using LayoutH = Catlass::layout::RowMajor;

    using ElementOinter = typename BlockMmadQH::ElementC;
    using LayoutOinter = Catlass::layout::RowMajor;


    using ElementVNEW = typename BlockMmadAttenVNEW::ElementB;
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
    uint32_t vWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    uint32_t attnWorkspaceOffset;
    uint32_t aftermaskWorkspaceOffset;
    uint32_t maskWorkspaceOffset;

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
    AscendC::LocalTensor<ElementAttenMasked> l1AttenMaskWorkspace[VEC1_L1_STAGES];

    AscendC::LocalTensor<ElementAtten> ubAttenPing;
    AscendC::LocalTensor<ElementAtten> ubAttenPong;
    AscendC::LocalTensor<ElementOinter> ubHWorkPing;
    AscendC::LocalTensor<ElementOinter> ubHWorkPong;
    AscendC::LocalTensor<ElementOinter> ubVWorkPing;
    AscendC::LocalTensor<ElementOinter> ubVWorkPong;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;

    __aicore__ inline GDNFwdOKernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
        GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, GM_ADDR tiling, GM_ADDR user) {

        __gm__ ChunkFwdOTilingData *__restrict gdnFwdOTilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);

        shapeBatch = gdnFwdOTilingData->shapeBatch;
        seqlen = gdnFwdOTilingData->seqlen;
        kNumHead = gdnFwdOTilingData->kNumHead;
        vNumHead = gdnFwdOTilingData->vNumHead;
        kHeadDim = gdnFwdOTilingData->kHeadDim;
        vHeadDim = gdnFwdOTilingData->vHeadDim;
        scale = gdnFwdOTilingData->scale;
        chunkSize = gdnFwdOTilingData->chunkSize;
        isVariedLen = gdnFwdOTilingData->isVariedLen;
        tokenBatch = gdnFwdOTilingData->tokenBatch;
        vWorkspaceOffset = gdnFwdOTilingData->vWorkspaceOffset;
        hWorkspaceOffset = gdnFwdOTilingData->hWorkspaceOffset;
        attnWorkspaceOffset = gdnFwdOTilingData->attnWorkspaceOffset;
        aftermaskWorkspaceOffset = gdnFwdOTilingData->aftermaskWorkspaceOffset;
        maskWorkspaceOffset = gdnFwdOTilingData->maskWorkspaceOffset;

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

        constexpr uint32_t UB_WORK_SLOT_BYTES = 64 * 128 * sizeof(ElementOinter);
        constexpr uint32_t UB_WORK_BASE = 71 * 1024;
        constexpr uint32_t UB_V_WORK_PING_OFFSET = UB_WORK_BASE;
        constexpr uint32_t UB_H_WORK_PING_OFFSET = UB_V_WORK_PING_OFFSET + UB_WORK_SLOT_BYTES;
        constexpr uint32_t UB_V_WORK_PONG_OFFSET = UB_H_WORK_PING_OFFSET + UB_WORK_SLOT_BYTES;
        constexpr uint32_t UB_H_WORK_PONG_OFFSET = UB_V_WORK_PONG_OFFSET + UB_WORK_SLOT_BYTES;

        ubAttenPing = resource.ubBuf.template GetBufferByByte<ElementAtten>(UB_V_WORK_PING_OFFSET);
        ubAttenPong = resource.ubBuf.template GetBufferByByte<ElementAtten>(UB_V_WORK_PONG_OFFSET);
        ubHWorkPing = resource.ubBuf.template GetBufferByByte<ElementOinter>(UB_H_WORK_PING_OFFSET);
        ubHWorkPong = resource.ubBuf.template GetBufferByByte<ElementOinter>(UB_H_WORK_PONG_OFFSET);
        ubVWorkPing = resource.ubBuf.template GetBufferByByte<ElementOinter>(UB_V_WORK_PING_OFFSET);
        ubVWorkPong = resource.ubBuf.template GetBufferByByte<ElementOinter>(UB_V_WORK_PONG_OFFSET);

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_offsets, tiling);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_offsets, tiling);
        }
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();
            uint32_t l1BufAddrStart = ENABLE_VEC1_UB_TO_L1_FOR_CUBE3 ?
                VEC1_L1_TILE_SIZE * VEC1_L1_STAGES : 0;

            if constexpr (ENABLE_VEC1_UB_TO_L1_FOR_CUBE3) {
                for (uint32_t i = 0; i < VEC1_L1_STAGES; ++i) {
                    l1AttenMaskWorkspace[i] = resource.l1Buf.template GetBufferByByte<ElementAttenMasked>(
                        i * VEC1_L1_TILE_SIZE);
                }
            }

            BlockMmadQK blockMmadQK(resource, l1BufAddrStart);
            BlockMmadQH blockMmadQH(resource, l1BufAddrStart);
            BlockMmadAttenVNEW blockMmadAttenVNEW(resource, l1BufAddrStart);

            auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
            auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

            bool needRun = false;
            uint32_t cube2pingpongFlag = 0;
            uint32_t cube3pingpongFlag = 0;
            uint32_t l0c2ubProduceCount = 0;
            uint32_t l0c2ubProducedStages[PING_PONG_STAGES] = {0};

            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();

                if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {

                    GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                    uint32_t cube1Stage = cubeBlockScheduler.GetCube1Stage();
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
                    blockMmadQK.preSetFlags();
                    blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                    blockMmadQK.finalWaitFlags();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[cube1Stage]);

                }
                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum) {
                    uint32_t cube23Stage = cubeBlockScheduler.GetCube23Stage();
                    if (l0c2ubProduceCount >= PING_PONG_STAGES) {
                        for (uint32_t i = 0; i < subBlockNum; ++i) {
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[cube23Stage]);
                        }
                    }
                    for (uint32_t i = 0; i < subBlockNum; ++i) {
                        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[cube23Stage]);
                    }
                    GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube2OffsetQ = cube2Offsets.qkOffset;
                    int64_t cube2OffsetH = cube2Offsets.hOffset;
                    auto cube2OinterLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(cube2Offsets.blockTokens, vHeadDim);
                    auto tensorQ = tla::MakeTensor(gmQ[cube2OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[cube2OffsetH], hLayout, Catlass::Arch::PositionGM{});
                    AscendC::LocalTensor<ElementOinter> ubHWork = (cube2pingpongFlag == 0) ? ubHWorkPing : ubHWorkPong;
                    auto tensorHWork = tla::MakeTensor(ubHWork, cube2OinterLayout, Catlass::Arch::PositionUB{});
                    GemmCoord cube2Shape{cube2Offsets.blockTokens, vHeadDim, kHeadDim};
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                    blockMmadQH.preSetFlags();
                    if constexpr (ENABLE_REUSE_Q_L1_FOR_QH) {
                        blockMmadQH.SkipNextADataCopy();
                    }
                    blockMmadQH(tensorBlockQ, tensorBlockH, tensorHWork, cube2Shape);
                    blockMmadQH.finalWaitFlags();
                    cube2pingpongFlag = 1 - cube2pingpongFlag;
                }

                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::PipeBarrier<PIPE_FIX>();

                if (needRun && coreIdx < coreNum) {
                    GDNFwdOOffsets& cube3Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube3OffsetAttnMask = cube3Offsets.attnWorkOffset;
                    int64_t cube3OffsetV = cube3Offsets.ovOffset;
                    auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(cube3Offsets.blockTokens, vHeadDim);
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube3Offsets.blockTokens);
                    auto tensorAttnMask = tla::MakeTensor(gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout, Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(gmV[cube3OffsetV], vnewLayout, Catlass::Arch::PositionGM{});
                    AscendC::LocalTensor<ElementOinter> ubVWork = (cube3pingpongFlag == 0) ? ubVWorkPing : ubVWorkPong;
                    auto tensorVWork = tla::MakeTensor(ubVWork, ointerLayout, Catlass::Arch::PositionUB{});
                    GemmCoord cube3Shape{cube3Offsets.blockTokens, vHeadDim, cube3Offsets.blockTokens};
                    auto tensorBlockAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                    blockMmadAttenVNEW.preSetFlags();
                    if constexpr (ENABLE_VEC1_UB_TO_L1_FOR_CUBE3) {
                        uint32_t l1Stage = cubeBlockScheduler.GetCube23Stage();
                        blockMmadAttenVNEW.SkipNextADataCopy();
                        blockMmadAttenVNEW.UseExternalL1ATensor(l1AttenMaskWorkspace[l1Stage]);
                    }
                    blockMmadAttenVNEW(tensorBlockAttnMask, tensorBlockV, tensorVWork, cube3Shape);
                    blockMmadAttenVNEW.finalWaitFlags();
                    cube3pingpongFlag = 1 - cube3pingpongFlag;
                    l0c2ubProducedStages[l0c2ubProduceCount % PING_PONG_STAGES] = cubeBlockScheduler.GetCube23Stage();
                    ++l0c2ubProduceCount;
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube3Done[cubeBlockScheduler.GetCube23Stage()]);
                }
                needRun = true;
                // AscendC::PipeBarrier<PIPE_ALL>();
            }

            uint32_t l0c2ubDrainCount = (l0c2ubProduceCount < PING_PONG_STAGES) ? l0c2ubProduceCount : PING_PONG_STAGES;
            uint32_t l0c2ubDrainStart = l0c2ubProduceCount - l0c2ubDrainCount;
            for (uint32_t drainIdx = 0; drainIdx < l0c2ubDrainCount; ++drainIdx) {
                uint32_t drainStage = l0c2ubProducedStages[(l0c2ubDrainStart + drainIdx) % PING_PONG_STAGES];
                for (uint32_t subIdx = 0; subIdx < subBlockNum; ++subIdx) {
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[drainStage]);
                }
            }
        }

        if ASCEND_IS_AIV {

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();

            if constexpr (ENABLE_VEC1_UB_TO_L1_FOR_CUBE3) {
                for (uint32_t i = 0; i < VEC1_L1_STAGES; ++i) {
                    l1AttenMaskWorkspace[i] = resource.l1Buf.template GetBufferByByte<ElementAttenMasked>(
                        i * VEC1_L1_TILE_SIZE);
                }
            }

            AscendC::LocalTensor<float> maskUbTensor = resource.ubBuf.template GetBufferByByte<float>(0);
            AscendC::Duplicate<float>(maskUbTensor, (float)0.0, 64*64);
            AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t i = 0; i < 64; ++ i) AscendC::Duplicate<float>(maskUbTensor[i * 64], (float)1.0, i + 1);
            AscendC::PipeBarrier<PIPE_V>();

            bool needRun = false;
            uint32_t vec1pingpongFlag = 0;
            uint32_t vec2pingpongFlag = 0;
            bool vec2Mte3Pending = false;
            uint32_t vec2Mte3PendingEvent = 0;

            while (vecBlockScheduler.isRunning) {
                vecBlockScheduler.InitTask();

                if (vecBlockScheduler.isRunning && coreIdx < coreNum * subBlockNum) {
                    if (vec2Mte3Pending) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(vec2Mte3PendingEvent);
                        vec2Mte3Pending = false;
                    }
                    uint32_t vec1Stage = vecBlockScheduler.GetVec1Stage();
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done[vec1Stage]);
                    GDNFwdOOffsets& vec1Offsets = vecBlockScheduler.GetVec1Offsets();
                    int64_t vec1OffsetAttnMask = vec1Offsets.attnWorkOffset;
                    int64_t vec1OffsetG = vec1Offsets.gOffset;
                    int64_t vec1OffsetAttn = vec1Offsets.attnWorkOffset;
                    EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource);
                    if constexpr (ENABLE_VEC1_UB_TO_L1_FOR_CUBE3) {
                        epilogueGDNFwdOQkmask.EnableL1Output(l1AttenMaskWorkspace[vec1Stage]);
                    }
                    epilogueGDNFwdOQkmask(
                        gmAftermaskWorkspace[vec1OffsetAttnMask],
                        gmG[vec1OffsetG], gmAttnWorkspace[vec1OffsetAttn], gmMask,
                        chunkSize, vec1Offsets.blockTokens, kHeadDim, vHeadDim, vec1pingpongFlag, vec1Offsets.batchIdx, vec1Offsets.headIdx, vec1Offsets.chunkIdx
                    );
                    vec1pingpongFlag = 1 - vec1pingpongFlag;
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done[vec1Stage]);
                }

                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum * subBlockNum) {
                    uint32_t vec2Stage = vecBlockScheduler.GetVec2Stage();
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube3Done[vec2Stage]);
                    GDNFwdOOffsets& vec2Offsets = vecBlockScheduler.GetVec2Offsets();
                    int64_t vec2OffsetO = vec2Offsets.ovOffset;
                    int64_t vec2OffsetG = vec2Offsets.gOffset;
                    AscendC::LocalTensor<ElementOinter> ubVWork = (vec2pingpongFlag == 0) ? ubVWorkPing : ubVWorkPong;
                    AscendC::LocalTensor<ElementOinter> ubHWork = (vec2pingpongFlag == 0) ? ubHWorkPing : ubHWorkPong;
                    EpilogueGDNFwdOOutput epilogueGDNFwdOOutput(resource);
                    epilogueGDNFwdOOutput(
                        gmO[vec2OffsetO],
                        gmG[vec2OffsetG], ubVWork, ubHWork,
                        scale, vec2Offsets.blockTokens, kHeadDim, vHeadDim, vec2pingpongFlag,
                        vec2Mte3Pending, vec2Mte3PendingEvent,
                        vec2Offsets.batchIdx, vec2Offsets.headIdx, vec2Offsets.chunkIdx
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[vec2Stage]);
                    vec2pingpongFlag = 1 - vec2pingpongFlag;
                }
                needRun = true;
            }
            if (vec2Mte3Pending) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(vec2Mte3PendingEvent);
            }
        }
    }

};

}
