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
#include "kernel_utils/block/block_mmad_pingpong_tla_pipelined.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
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
#include "../../../chunk_fwd_o_struct.h"
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

    using DispatchPolicyTla = Gemm::MmadPingpongTlaGdnFwdO<ArchTag, true, false>;
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

    // === l0c2ub / ub2l1 data-path switches (docs/agents/chunk_fwd_o_l0c2ub_ub2l1_design.md) ===
    // ENABLE_L0C2UB: Cube2/Cube3 V128 output goes L0C -> UB slot directly (skip GM workspace).
    static constexpr bool ENABLE_L0C2UB = true;
    // ENABLE_UB2L1: Vec1 aftermask goes UB -> L1 directly (skip GM aftermaskWorkspace).
    static constexpr bool ENABLE_UB2L1 = false; // accurate but slower (1.02x vs 1.09x l0c2ub-only): sync overhead exceeds the GM-roundtrip saving; kept for future rework, see design doc R-7
    // UB work-slot layout (4 x 32KB: V/H ping + V/H pong), shared contract with both epilogues.
    static constexpr uint32_t UB_WORK_SLOT_BYTES = 64 * 128 * sizeof(float);
    static constexpr uint32_t UB_WORK_BASE = 71 * 1024;
    static constexpr uint32_t UB_V_WORK_PING_OFFSET = UB_WORK_BASE;
    static constexpr uint32_t UB_H_WORK_PING_OFFSET = UB_V_WORK_PING_OFFSET + UB_WORK_SLOT_BYTES;
    static constexpr uint32_t UB_V_WORK_PONG_OFFSET = UB_H_WORK_PING_OFFSET + UB_WORK_SLOT_BYTES;
    static constexpr uint32_t UB_H_WORK_PONG_OFFSET = UB_V_WORK_PONG_OFFSET + UB_WORK_SLOT_BYTES;
    // L1 head reservation for ub2l1: 2 stages x (128x128 INPUT_TYPE, zN layout).
    static constexpr uint32_t VEC1_L1_STAGES = PING_PONG_STAGES;
    static constexpr uint32_t VEC1_L1_TILE_BYTES = 128 * 128 * sizeof(INPUT_TYPE);

    // cube 1
    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::ColumnMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQK = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTla, L1TileShapeQKTla, L0TileShapeQKTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;

    // cube 2 (V128 direct-UB variant: L0C -> UB slot via Fixpipe SPLIT_M)
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using TileCopyQHToUB = Common::Tile::PackedTileCopyTlaToUB<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor, void, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using DispatchPolicyTlaShareL1A = Gemm::MmadPingpongTlaGdnFwdO<ArchTag, true, false, 1, false, true>;
    using BlockMmadQH128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaShareL1A, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadQH256 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaShareL1A, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadQH128ToUB = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaShareL1A, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQHToUB>;

    // cube 3 — BlockMmadTlaPipelined for overlap with Cube2 (V128 direct-UB variant included)
    using TileCopyAttenVNEW = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using TileCopyAttenVNEWToUB = Common::Tile::PackedTileCopyTlaToUB<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor, void, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadAttenVNEW128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTla, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
    using BlockMmadAttenVNEW256 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTla, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
    using BlockMmadAttenVNEW128ToUB = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTla, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEWToUB>;

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

    // ub2l1: L1 head slots holding Vec1's aftermask output (zN layout), indexed by stage.
    AscendC::LocalTensor<ElementAttenMasked> l1AttenMaskWorkspace[VEC1_L1_STAGES];

    // l0c2ub: number of Fixpipe writes this AIC has issued to the UB work slots.
    // The first PING_PONG_STAGES writes (first fill per slot stage) target slots no
    // one has read yet, so their vec2Done wait is skipped; from then on every wait
    // maps 1:1 to a real Vec2 consumption. A core with fewer than PING_PONG_STAGES
    // writes never overwrote a slot, so its end-of-kernel drain waits are skipped too.
    uint32_t l0c2ubProduceCount = 0;

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

    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadQK blockMmadQK(resource);
            BlockMmadQH128 blockMmadQH128(resource);
            BlockMmadQH128ToUB blockMmadQH128ToUB(resource);
            BlockMmadQH256 blockMmadQH256(resource);
            // Cube3: independent L1 buffer and eventID (offset 4).
            // V128 path: Cube1 L1A[0,64KB) + L1B[64KB,128KB) = 128KB total.
            //   Cube3 starts at 128KB → no overlap.
            // V256 path: Cube1 L1A[0,64KB) + Cube2 L1B[64KB,192KB) = 192KB total.
            //   Cube3 must start at 192KB to avoid overlapping Cube2's L1B.
            // Use 192KB to cover both paths safely.
            constexpr uint32_t cube3L1Offset = 192 * 1024;
            constexpr uint32_t cube3L1AEventId = 4;
            constexpr uint32_t cube3L1BEventId = 6;
            BlockMmadAttenVNEW128 blockMmadAttenVNEW128(resource, cube3L1Offset, cube3L1AEventId, cube3L1BEventId);
            BlockMmadAttenVNEW128ToUB blockMmadAttenVNEW128ToUB(resource, cube3L1Offset, cube3L1AEventId, cube3L1BEventId);
            BlockMmadAttenVNEW256 blockMmadAttenVNEW256(resource, cube3L1Offset, cube3L1AEventId, cube3L1BEventId);
            // ub2l1: Vec1 aftermask slots live at the L1 tail [448KB, 512KB).
            // The head cannot be used: SHARE_L1A pins Cube1's L1A at absolute address 0.
            // Cube region tops out at 384KB (V256: 192KB + 128KB L1B + 64KB L1A),
            // so the tail reservation is physically disjoint from all Cube tiles.
            static_assert(VEC1_L1_STAGES * VEC1_L1_TILE_BYTES <= 64 * 1024, "L1 tail reservation exceeds 64KB");
            constexpr uint32_t VEC1_L1_BASE = 512 * 1024 - VEC1_L1_STAGES * VEC1_L1_TILE_BYTES;
            if constexpr (ENABLE_UB2L1) {
                for (uint32_t i = 0; i < VEC1_L1_STAGES; ++i) {
                    l1AttenMaskWorkspace[i] = resource.l1Buf.template GetBufferByByte<ElementAttenMasked>(
                        VEC1_L1_BASE + i * VEC1_L1_TILE_BYTES);
                }
                // Pre-arm the reverse-sync flags so the AIV's first write to each
                // slot does not block (mode-2 counter semantics: each preset lets
                // exactly one wait pass).
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.l1SlotConsumed[0]);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.l1SlotConsumed[1]);
            }

            auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
            auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(coreNum * chunkSize * PING_PONG_STAGES, cubeBlockScheduler.vBlockSize);
            auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

            bool needRun = false;

            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();

                // === Phase 1a: Cube1 operator() + waitL1Drained ===
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
                    blockMmadQK.preSetFlags();
                    blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                    blockMmadQK.waitL1Drained();
                }

                // === Phase 1b: Cube2 copyGmToL1 (overlaps with Cube1's Mmad) ===
                if (needRun && coreIdx < coreNum) {
                    GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube2OffsetQ = cube2Offsets.qkOffset;
                    int64_t cube2OffsetH = cube2Offsets.hOffset;
                    auto tensorQ2 = tla::MakeTensor(gmQ[cube2OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[cube2OffsetH], hLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube2Shape{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                    auto tensorBlockQ2 = GetTile(tensorQ2, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                    // The same instance must be used for copyGmToL1 here and executeCompute in
                    // Phase 2: per-instance l1ListId/event bookkeeping must stay in lockstep.
                    if constexpr (ENABLE_L0C2UB) {
                        if (cube2Offsets.vBlockDim <= 128) {
                            blockMmadQH128ToUB.preSetL1Flags();
                            blockMmadQH128ToUB.copyGmToL1(tensorBlockQ2, tensorBlockH, cube2Shape);
                        } else {
                            blockMmadQH256.preSetL1Flags();
                            blockMmadQH256.copyGmToL1(tensorBlockQ2, tensorBlockH, cube2Shape);
                        }
                    } else {
                        if (cube2Offsets.vBlockDim <= 128) {
                            blockMmadQH128.preSetL1Flags();
                            blockMmadQH128.copyGmToL1(tensorBlockQ2, tensorBlockH, cube2Shape);
                        } else {
                            blockMmadQH256.preSetL1Flags();
                            blockMmadQH256.copyGmToL1(tensorBlockQ2, tensorBlockH, cube2Shape);
                        }
                    }
                }

                // === Phase 1c: Cube1 waitL0Drained + cube1Done ===
                if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetCurStageId();
                    blockMmadQK.waitL0Drained();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);
                }

                // === Phase 2: Cube2 executeCompute + Cube3 copyGmToL1 (overlapped) ===
                if (needRun && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetPrevStageId();
                    // First write per slot stage has no previous data to protect, so
                    // skip the vec2Done wait — this keeps flag credits strictly paired
                    // with real Vec2 consumptions. With the AIV startup preset removed
                    // (see the AIV entry below), a preset credit can no longer be
                    // spent by an early Fixpipe while the slower subBlock is still
                    // reading the slot — that race was the intermittent garbage rows.
                    if (l0c2ubProduceCount >= PING_PONG_STAGES) {
                        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                    }
                    ++l0c2ubProduceCount;
                    GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube2OffsetHWork = cube2Offsets.hvWorkOffset;
                    auto tensorHWork = tla::MakeTensor(gmHWorkspace[cube2OffsetHWork], ointerLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube2Shape{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                    auto tensorBlockHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                    // l0c2ub: Cube2 V128 output goes to its UB work slot (indexed by the
                    // stage the Vec2 consumer will read); V256 keeps the GM workspace path.
                    auto ubHWorkSlot = resource.ubBuf.template GetBufferByByte<ElementOinter>(
                        (streamId == 0) ? UB_H_WORK_PING_OFFSET : UB_H_WORK_PONG_OFFSET);
                    auto tensorHWorkUb = tla::MakeTensor(ubHWorkSlot,
                        tla::MakeLayout<ElementOinter, LayoutOinter>(cube2Shape.m(), cube2Shape.n()),
                        Catlass::Arch::PositionUB{});
                    auto tensorBlockHWorkUb = GetTile(tensorHWorkUb, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                    // Cube3 offsets/shapes (computed early for V preload)
                    GDNFwdOOffsets& cube3Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube3OffsetAttnMask = cube3Offsets.attnWorkOffset;
                    int64_t cube3OffsetV = cube3Offsets.ovOffset;
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube3Offsets.blockTokens);
                    auto tensorAttnMask = tla::MakeTensor(gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout, Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(gmV[cube3OffsetV], vnewLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube3Shape{cube3Offsets.blockTokens, cube3Offsets.vBlockDim, cube3Offsets.blockTokens};
                    auto tensorBlockAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                    auto ubVWorkSlot = resource.ubBuf.template GetBufferByByte<ElementOinter>(
                        (streamId == 0) ? UB_V_WORK_PING_OFFSET : UB_V_WORK_PONG_OFFSET);
                    auto tensorVWorkUb = tla::MakeTensor(ubVWorkSlot,
                        tla::MakeLayout<ElementOinter, LayoutOinter>(cube3Shape.m(), cube3Shape.n()),
                        Catlass::Arch::PositionUB{});
                    int64_t cube3OffsetVWork = cube3Offsets.hvWorkOffset;
                    auto tensorVWork = tla::MakeTensor(gmVWorkspace[cube3OffsetVWork], ointerLayout, Catlass::Arch::PositionGM{});
                    auto tensorBlockVWork = GetTile(tensorVWork, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));
                    auto tensorBlockVWorkUb = GetTile(tensorVWorkUb, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));
                    constexpr bool cubeToUb128 = ENABLE_L0C2UB;  // V128-only by the vBlockDim guards below
                    // Cube3: preload V (L1B) before Cube2 executeCompute — V has no cross-core dependency
                    if constexpr (cubeToUb128) {
                        if (cube3Offsets.vBlockDim <= 128) {
                            blockMmadAttenVNEW128ToUB.preSetL1Flags();
                            blockMmadAttenVNEW128ToUB.copyGmToL1BOnly(tensorBlockV, cube3Shape);
                        } else {
                            blockMmadAttenVNEW256.preSetL1Flags();
                            blockMmadAttenVNEW256.copyGmToL1BOnly(tensorBlockV, cube3Shape);
                        }
                    } else {
                        if (cube3Offsets.vBlockDim <= 128) {
                            blockMmadAttenVNEW128.preSetL1Flags();
                            blockMmadAttenVNEW128.copyGmToL1BOnly(tensorBlockV, cube3Shape);
                        } else {
                            blockMmadAttenVNEW256.preSetL1Flags();
                            blockMmadAttenVNEW256.copyGmToL1BOnly(tensorBlockV, cube3Shape);
                        }
                    }
                    // Cube2: executeCompute (L0 Mmad overlaps with Cube3's V GM->L1B above)
                    if constexpr (cubeToUb128) {
                        if (cube2Offsets.vBlockDim <= 128) {
                            blockMmadQH128ToUB.preSetL0Flags();
                            blockMmadQH128ToUB.executeCompute(tensorBlockHWorkUb, cube2Shape);
                        } else {
                            blockMmadQH256.preSetL0Flags();
                            blockMmadQH256.executeCompute(tensorBlockHWork, cube2Shape);
                        }
                    } else {
                        if (cube2Offsets.vBlockDim <= 128) {
                            blockMmadQH128.preSetL0Flags();
                            blockMmadQH128.executeCompute(tensorBlockHWork, cube2Shape);
                        } else {
                            blockMmadQH256.preSetL0Flags();
                            blockMmadQH256.executeCompute(tensorBlockHWork, cube2Shape);
                        }
                    }
                    // Cube3: acquire AttnMasked (L1A) — depends on vec1Done.
                    // ub2l1: Vec1 already wrote the aftermask into the L1 tail slot for
                    // this stage; consume it directly and skip the GM->L1A copy.
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
                    if constexpr (ENABLE_UB2L1) {
                        // SSBuffer visibility barrier (PR #245 pattern + MTE1): drain
                        // inbound L1 writes (MTE2), FIX and the MTE1 reader queue itself
                        // so the L1A reads below are ordered after all landed data.
                        AscendC::PipeBarrier<PIPE_MTE2>();
                        AscendC::PipeBarrier<PIPE_FIX>();
                        AscendC::PipeBarrier<PIPE_MTE1>();
                        if (cube3Offsets.vBlockDim <= 128) {
                            if constexpr (cubeToUb128) {
                                blockMmadAttenVNEW128ToUB.UseExternalL1ATensor(l1AttenMaskWorkspace[streamId]);
                            } else {
                                blockMmadAttenVNEW128.UseExternalL1ATensor(l1AttenMaskWorkspace[streamId]);
                            }
                        } else {
                            blockMmadAttenVNEW256.UseExternalL1ATensor(l1AttenMaskWorkspace[streamId]);
                        }
                    } else if (cube3Offsets.vBlockDim <= 128) {
                        if constexpr (cubeToUb128) {
                            blockMmadAttenVNEW128ToUB.copyGmToL1AOnly(tensorBlockAttnMask, cube3Shape);
                        } else {
                            blockMmadAttenVNEW128.copyGmToL1AOnly(tensorBlockAttnMask, cube3Shape);
                        }
                    } else {
                        blockMmadAttenVNEW256.copyGmToL1AOnly(tensorBlockAttnMask, cube3Shape);
                    }
                    if constexpr (cubeToUb128) {
                        if (cube2Offsets.vBlockDim <= 128) { blockMmadQH128ToUB.finalWaitFlags(); }
                        else { blockMmadQH256.finalWaitFlags(); }
                    } else {
                        if (cube2Offsets.vBlockDim <= 128) { blockMmadQH128.finalWaitFlags(); }
                        else { blockMmadQH256.finalWaitFlags(); }
                    }
                    if constexpr (cubeToUb128) {
                        if (cube3Offsets.vBlockDim <= 128) {
                            blockMmadAttenVNEW128ToUB.preSetL0Flags();
                            blockMmadAttenVNEW128ToUB.executeCompute(tensorBlockVWorkUb, cube3Shape);
                            blockMmadAttenVNEW128ToUB.finalWaitFlags();
                        } else {
                            blockMmadAttenVNEW256.preSetL0Flags();
                            blockMmadAttenVNEW256.executeCompute(tensorBlockVWork, cube3Shape);
                            blockMmadAttenVNEW256.finalWaitFlags();
                        }
                    } else {
                        if (cube3Offsets.vBlockDim <= 128) {
                            blockMmadAttenVNEW128.preSetL0Flags();
                            blockMmadAttenVNEW128.executeCompute(tensorBlockVWork, cube3Shape);
                            blockMmadAttenVNEW128.finalWaitFlags();
                        } else {
                            blockMmadAttenVNEW256.preSetL0Flags();
                            blockMmadAttenVNEW256.executeCompute(tensorBlockVWork, cube3Shape);
                            blockMmadAttenVNEW256.finalWaitFlags();
                        }
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube3Done[streamId]);
                    if constexpr (ENABLE_UB2L1) {
                        // ub2l1 reverse sync: drain Cube3's MTE1 queue (the actual L1A
                        // reader) before releasing the slot — finalWaitFlags covers the
                        // L0 pipeline but the MTE1 drain makes the release strictly
                        // after the last L1A read has retired.
                        AscendC::PipeBarrier<PIPE_MTE1>();
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.l1SlotConsumed[streamId]);
                    }
                }
                needRun = true;
            }
            // End-of-kernel drain: only a core that overwrote a slot (>= PING_PONG_STAGES
            // Fixpipe writes) has an in-flight Vec2 consumption left to wait for. Cores
            // with 0/1 writes never raced a reader, and with the startup preset removed
            // there is no credit for their wait to consume — waiting would deadlock
            // (task counts below coreNum leave such cores idle-but-waiting).
            if (l0c2ubProduceCount >= PING_PONG_STAGES) {
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
            }
        }

        if ASCEND_IS_AIV {

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();

            // NOTE: no startup preset of vec2Done. The AIC skips the vec2Done wait
            // for the first write of each slot stage, so the flags start un-credited
            // and every later AIC wait maps 1:1 to a real Vec2 consumption. The old
            // preset over-credited the flags, and an early Fixpipe could spend that
            // credit before the slower subBlock finished reading the slot — the
            // intermittent garbage rows seen only under cross-device parallel load.

            // ub2l1: mirror the AIC-side L1 tail slots (same physical L1 per block).
            static_assert(VEC1_L1_STAGES * VEC1_L1_TILE_BYTES <= 64 * 1024, "L1 tail reservation exceeds 64KB");
            constexpr uint32_t VEC1_L1_BASE = 512 * 1024 - VEC1_L1_STAGES * VEC1_L1_TILE_BYTES;
            if constexpr (ENABLE_UB2L1) {
                for (uint32_t i = 0; i < VEC1_L1_STAGES; ++i) {
                    l1AttenMaskWorkspace[i] = resource.l1Buf.template GetBufferByByte<ElementAttenMasked>(
                        VEC1_L1_BASE + i * VEC1_L1_TILE_BYTES);
                }
            }

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
                    GDNFwdOOffsets& vec1Offsets = vecBlockScheduler.GetVec1Offsets();
                    int64_t vec1OffsetAttnMask = vec1Offsets.attnWorkOffset;
                    int64_t vec1OffsetG = vec1Offsets.gOffset;
                    int64_t vec1OffsetAttn = vec1Offsets.attnWorkOffset;
                    EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource);
                    if constexpr (ENABLE_UB2L1) {
                        // ub2l1: Vec1 writes its aftermask into the L1 tail slot of this
                        // stage instead of the GM aftermask workspace. Wait for the
                        // reverse-sync flag first: Cube3's previous MTE1 read of this
                        // slot must have completed before the SSBuffer write overwrites it.
                        Arch::CrossCoreWaitFlag(vecBlockScheduler.l1SlotConsumed[streamId]);
                        epilogueGDNFwdOQkmask.EnableL1Output(l1AttenMaskWorkspace[streamId]);
                    }
                    epilogueGDNFwdOQkmask(
                        gmAftermaskWorkspace[vec1OffsetAttnMask],
                        gmG[vec1OffsetG], gmAttnWorkspace[vec1OffsetAttn], gmMask,
                        chunkSize, vec1Offsets.blockTokens, kHeadDim, vHeadDim, pingpongFlag, vec1Offsets.batchIdx, vec1Offsets.headIdx, vec1Offsets.chunkIdx,
                        &vecBlockScheduler.cube1Done[streamId]
                    );
                    if constexpr (ENABLE_UB2L1) {
                        // ub2l1: drain this AIV's MTE3 queue first so the SSBuffer write
                        // (CopyOutputToL1) has fully left the Vector core before the
                        // completion flag becomes visible to the AIC.
                        AscendC::PipeBarrier<PIPE_MTE3>();
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done[streamId]);
                }

                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum * subBlockNum) {
                    uint32_t streamId = vecBlockScheduler.GetPrevStageId();
                    GDNFwdOOffsets& vec2Offsets = vecBlockScheduler.GetVec2Offsets();
                    int64_t vec2OffsetO = vec2Offsets.ovOffset;
                    int64_t vec2OffsetG = vec2Offsets.gOffset;
                    int64_t vec2OffsetVWork = vec2Offsets.hvWorkOffset;
                    int64_t vec2OffsetHWork = vec2Offsets.hvWorkOffset;
                    EpilogueGDNFwdOOutput epilogueGDNFwdOOutput(resource);
                    if constexpr (ENABLE_L0C2UB) {
                        if (vec2Offsets.vBlockDim <= 128) {
                            // l0c2ub: read Cube2/Cube3 results straight from the UB work
                            // slots of this stage (Fixpipe wrote them there directly).
                            auto ubVWork = resource.ubBuf.template GetBufferByByte<ElementOinter>(
                                (streamId == 0) ? UB_V_WORK_PING_OFFSET : UB_V_WORK_PONG_OFFSET);
                            auto ubHWork = resource.ubBuf.template GetBufferByByte<ElementOinter>(
                                (streamId == 0) ? UB_H_WORK_PING_OFFSET : UB_H_WORK_PONG_OFFSET);
                            epilogueGDNFwdOOutput(
                                gmO[vec2OffsetO],
                                gmG[vec2OffsetG], ubVWork, ubHWork,
                                scale, vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim, vHeadDim, pingpongFlag, vec2Offsets.batchIdx, vec2Offsets.headIdx, vec2Offsets.chunkIdx,
                                &vecBlockScheduler.cube3Done[streamId],
                                &vecBlockScheduler.vec2Done[streamId]
                            );
                        } else {
                            epilogueGDNFwdOOutput(
                                gmO[vec2OffsetO],
                                gmG[vec2OffsetG], gmVWorkspace[vec2OffsetVWork], gmHWorkspace[vec2OffsetHWork],
                                scale, vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim, vHeadDim, pingpongFlag, vec2Offsets.batchIdx, vec2Offsets.headIdx, vec2Offsets.chunkIdx,
                                &vecBlockScheduler.cube3Done[streamId],
                                &vecBlockScheduler.vec2Done[streamId]
                            );
                        }
                    } else {
                        epilogueGDNFwdOOutput(
                            gmO[vec2OffsetO],
                            gmG[vec2OffsetG], gmVWorkspace[vec2OffsetVWork], gmHWorkspace[vec2OffsetHWork],
                            scale, vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim, vHeadDim, pingpongFlag, vec2Offsets.batchIdx, vec2Offsets.headIdx, vec2Offsets.chunkIdx,
                            &vecBlockScheduler.cube3Done[streamId],
                            &vecBlockScheduler.vec2Done[streamId]
                        );
                    }
                }
                needRun = true;
            }
        }
    }

};

}
