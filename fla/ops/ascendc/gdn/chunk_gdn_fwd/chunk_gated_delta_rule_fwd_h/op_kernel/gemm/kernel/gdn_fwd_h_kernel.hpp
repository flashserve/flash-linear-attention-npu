/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#define CATLASS_ARCH 2201

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "../block/block_scheduler_gdn_fwd_h.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "../../chunk_gated_delta_rule_fwd_h_policy.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"



#include "kernel_operator.h"
using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm::Kernel {

struct GDNFwdHTileShapes128 {
    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
};

struct GDNFwdHTileShapes256 {
    using L1TileShape = tla::Shape<_128, _256, _128>;
    using L0TileShape = tla::Shape<_128, _256, _64>;
};

template <uint32_t GateMode, uint32_t ExpMode>
struct GDNFwdHGateTag {
    static_assert(GateMode == GDN_FWD_H_GATE_G || GateMode == GDN_FWD_H_GATE_GK,
                  "unsupported FwdH gate mode");
    static_assert(ExpMode == GDN_FWD_H_EXP_E || ExpMode == GDN_FWD_H_EXP_2,
                  "unsupported FwdH exponent mode");
    static constexpr bool value = GateMode == GDN_FWD_H_GATE_GK;
    static constexpr bool scalarGated = GateMode == GDN_FWD_H_GATE_G;
    static constexpr bool useExp2 = ExpMode == GDN_FWD_H_EXP_2;
};

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE,
    typename WORKSPACE_TYPE,
    typename TileShapes,
    uint32_t gateMode,
    uint32_t expMode
>
class GDNFwdHKernel {
public:

    using ArchTag = Arch::AtlasA2;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHVec;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
    using L1TileShapeVTla = typename TileShapes::L1TileShape;
    using L0TileShapeVTla = typename TileShapes::L0TileShape;

    using WType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;

    // cube 1
    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWH>;

    // cube 2
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;

    // vec 1
    using DispatchPolicyGDNFwdHVnew = Epilogue::EpilogueAtlasGDNFwdHVnew;
    using GateTag = GDNFwdHGateTag<gateMode, expMode>;
    static constexpr bool kGated = GateTag::value;
    static constexpr bool scalarGated = GateTag::scalarGated;
    using EpilogueGDNFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHVnew, VType, GType, UType, VworkType, FinalStateType, GateTag>;

    // vec 2
    using DispatchPolicyGDNFwdHUpdate = Epilogue::EpilogueAtlasGDNFwdHUpdate;
    using EpilogueGDNFwdHUpdate = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHUpdate, HType, GType, HType, HworkType, FinalStateType, GateTag>;

    using GDNFwdHOffsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;

    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;


    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    uint32_t vWorkspaceOffset;
    uint32_t vUpdateWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    uint32_t numSeqWorkspaceOffset;
    uint32_t numChunksWorkspaceOffset;

    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;

    AscendC::GlobalTensor<ElementG> gmGk;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;


    __aicore__ inline GDNFwdHKernel() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
        GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user) {

        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        vWorkspaceOffset = gdnFwdHTilingData->vWorkspaceOffset;
        vUpdateWorkspaceOffset = gdnFwdHTilingData->vUpdateWorkspaceOffset;
        hWorkspaceOffset = gdnFwdHTilingData->hWorkspaceOffset;
        numSeqWorkspaceOffset = gdnFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = gdnFwdHTilingData->numChunksWorkspaceOffset;

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)(scalarGated ? g : gk));
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmGk.SetGlobalBuffer((__gm__ ElementG *)(kGated ? gk : g));

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }
    }

    template <typename TilingData>
    __aicore__ inline void InitFromData(
        GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state,
        GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
        GM_ADDR final_state, const TilingData& tilingData, GM_ADDR user) {
        batch = tilingData.batch;
        seqlen = tilingData.seqlen;
        kNumHead = tilingData.kNumHead;
        vNumHead = tilingData.vNumHead;
        kHeadDim = tilingData.kHeadDim;
        vHeadDim = tilingData.vHeadDim;
        chunkSize = tilingData.chunkSize;
        useInitialState = tilingData.useInitialState;
        storeFinalState = tilingData.storeFinalState;
        isVariedLen = tilingData.isVariedLen;
        shapeBatch = tilingData.shapeBatch;
        tokenBatch = tilingData.tokenBatch;
        vWorkspaceOffset = tilingData.vWorkspaceOffset;
        vUpdateWorkspaceOffset = tilingData.vUpdateWorkspaceOffset;
        hWorkspaceOffset = tilingData.hWorkspaceOffset;
        numSeqWorkspaceOffset = tilingData.numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = tilingData.numChunksWorkspaceOffset;

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)(scalarGated ? g : gk));
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmGk.SetGlobalBuffer((__gm__ ElementG *)(kGated ? gk : g));
        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
        }
        if ASCEND_IS_AIV {
            vecBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
        }
    }

    template <typename Element>
    __aicore__ inline float LoadScalarAsFloat(
        AscendC::GlobalTensor<Element> tensor, uint32_t offset) const
    {
        Element value = tensor.GetValue(offset);
        if constexpr (std::is_same<Element, bfloat16_t>::value) {
            return AscendC::ToFloat(value);
        }
        return static_cast<float>(value);
    }

    // Tail Stage0 borrows the headTask's V_MTE2 free token and restores it
    // before returning to the regular V1 epilogue.
    __aicore__ inline void ComputeTailVWorkspace(
        const GDNFwdHOffsets& offsets, uint32_t tailEventId, bool processWholeHead)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        if (processWholeHead) {
            subBlockIdx = 0;
            subBlockNum = 1;
        }
        uint32_t rowsPerSubBlock = CeilDiv(offsets.blockTokens, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, offsets.blockTokens);
        if (rowBegin >= rowEnd) {
            return;
        }
        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);

        constexpr uint32_t TAIL_INPUT_OFFSET = 166 * 1024;
        constexpr uint32_t TAIL_FLOAT_OFFSET = 167 * 1024;
        constexpr uint32_t TAIL_ACCUM_OFFSET = 168 * 1024;
        AscendC::LocalTensor<ElementH> inputUb =
            resource.ubBuf.template GetBufferByByte<ElementH>(TAIL_INPUT_OFFSET);
        AscendC::LocalTensor<float> floatUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_FLOAT_OFFSET);
        AscendC::LocalTensor<float> accumUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_ACCUM_OFFSET);

        for (uint32_t tokenRow = rowBegin; tokenRow < rowEnd; ++tokenRow) {
            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                AscendC::DataCopy(
                    inputUb, gmH[offsets.hSrcOffset + kIdx * vHeadDim],
                    offsets.vBlockDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::Cast(
                    floatUb, inputUb, AscendC::RoundMode::CAST_NONE,
                    offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                float weight = LoadScalarAsFloat(
                    gmW, offsets.wOffset + tokenRow * kHeadDim + kIdx);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::DataCopy(
                gmVWorkspace[offsets.vWorkOffset + tokenRow * offsets.vBlockDim],
                accumUb, offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(tailEventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
    }

    template <typename TensorW, typename TensorH, typename TensorV>
    __aicore__ inline void RunStage0Mmad(
        BlockMmadWH& blockMmadWH, BlockMmadWH& blockMmadWHTail,
        TensorW& tensorW, TensorH& tensorH, TensorV& tensorV,
        const GDNFwdHOffsets& offsets, bool manageFlags = true)
    {
        GemmCoord shape{offsets.blockTokens, offsets.vBlockDim, kHeadDim};
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        if (offsets.blockTokens < chunkSize) {
            if (manageFlags) {
                blockMmadWHTail.preSetFlags();
            }
            blockMmadWHTail(blockW, blockH, blockV, shape, EmptyClass{}, true);
            if (manageFlags) {
                blockMmadWHTail.finalWaitFlags();
            }
        } else {
            if (manageFlags) {
                blockMmadWH.preSetFlags();
            }
            blockMmadWH(blockW, blockH, blockV, shape);
            if (manageFlags) {
                blockMmadWH.finalWaitFlags();
            }
        }
    }

    template <typename WLayout, typename HLayout, typename VLayout>
    __aicore__ inline void ProcessCubeStage0(
        BlockMmadWH& blockMmadWH, BlockMmadWH& blockMmadWHTail,
        const WLayout& wLayout, const HLayout& hLayout, const VLayout& vLayout)
    {
        cubeBlockScheduler.InitTasks();
        if (!cubeBlockScheduler.isRunning) {
            return;
        }
        uint32_t windowId = cubeBlockScheduler.GetWindowId();
        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[windowId]);
        const auto& firstHeadTask = cubeBlockScheduler.GetHeadTask(0);
        bool useTailMmad = firstHeadTask.offset.blockTokens < chunkSize;
        bool useCube = firstHeadTask.offset.blockTokens >= 16;
        if (useCube) {
            if (useTailMmad) {
                blockMmadWHTail.preSetFlags();
            } else {
                blockMmadWH.preSetFlags();
            }
        }
        for (uint32_t i = 0; i < HEADS_PER_TASK; ++i) {
            const auto& headTask = cubeBlockScheduler.GetHeadTask(i);
            if (cubeBlockScheduler.HeadTaskIsDone(headTask)) {
                continue;
            }
            const GDNFwdHOffsets& offsets = cubeBlockScheduler.GetCurTaskOffsets(headTask);
            if (offsets.blockTokens < 16) {
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                    cubeBlockScheduler.cube1Done[windowId]);
                continue;
            }
            auto tensorW = tla::MakeTensor(gmW[offsets.wOffset], wLayout, Catlass::Arch::PositionGM{});
            auto tensorH = tla::MakeTensor(gmH[offsets.hSrcOffset], hLayout, Catlass::Arch::PositionGM{});
            auto tensorV = tla::MakeTensor(
                gmVWorkspace[offsets.vWorkOffset], vLayout, Catlass::Arch::PositionGM{});
            RunStage0Mmad(blockMmadWH, blockMmadWHTail, tensorW, tensorH, tensorV, offsets, false);
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                cubeBlockScheduler.cube1Done[windowId]);
        }
        if (useCube) {
            if (useTailMmad) {
                blockMmadWHTail.finalWaitFlags();
            } else {
                blockMmadWH.finalWaitFlags();
            }
        }
    }

    // Test-only entry for comparing the FP32 Stage0 result against the
    // stage-wise reference. It is not exposed by the production operator ABI.
    __aicore__ inline void ProcessStage0Only(GM_ADDR stage0Output)
    {
        if (isVariedLen) {
            AscendC::SyncAll<false>();
        }
        if ASCEND_IS_AIC {
            AscendC::GlobalTensor<ElementVWork> gmStage0Output;
            gmStage0Output.SetGlobalBuffer((__gm__ ElementVWork*)stage0Output);

            auto wLayout = tla::MakeLayout<ElementW, LayoutW>(
                shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(
                shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, vHeadDim);
            auto outputLayout = tla::MakeLayout<ElementVWork, LayoutV>(
                shapeBatch * vNumHead * cubeBlockScheduler.totalTokens, vHeadDim);

            BlockMmadWH blockMmadWH(resource);
            BlockMmadWH blockMmadWHTail(resource);
            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTasks();
                for (uint32_t i = 0; i < HEADS_PER_TASK; ++i) {
                    const auto& headTask = cubeBlockScheduler.GetHeadTask(i);
                    if (cubeBlockScheduler.HeadTaskIsDone(headTask)) {
                        continue;
                    }
                    const GDNFwdHOffsets& offsets = cubeBlockScheduler.GetCurTaskOffsets(headTask);
                    auto tensorW = tla::MakeTensor(
                        gmW[offsets.wOffset], wLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(
                        gmH[offsets.hSrcOffset], hLayout, Catlass::Arch::PositionGM{});
                    auto tensorOutput = tla::MakeTensor(
                        gmStage0Output[offsets.uvOffset], outputLayout, Catlass::Arch::PositionGM{});
                    RunStage0Mmad(
                        blockMmadWH, blockMmadWHTail, tensorW, tensorH, tensorOutput, offsets);
                }
            }
        }
    }

    __aicore__ inline void Process() {
        if (isVariedLen) {
            AscendC::SyncAll<false>();
        }

        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadWH blockMmadWH(resource);
            BlockMmadWH blockMmadWHTail(resource);
            BlockMmadKV blockMmadKV(resource);

            auto wLayout = tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, vHeadDim);
            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(
                coreNum * chunkSize * WORKSPACE_BUFFER_COUNT, cubeBlockScheduler.vBlockSize);

            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens);
            auto vworkLayout = tla::MakeLayout<ElementV, LayoutV>(
                coreNum * chunkSize * WORKSPACE_BUFFER_COUNT, cubeBlockScheduler.vBlockSize);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(
                coreNum * kHeadDim * WORKSPACE_BUFFER_COUNT, cubeBlockScheduler.vBlockSize);
            AscendC::SyncAll<false>();
            uint32_t currStage = 0; // 0: C1, 1: C2
            while (cubeBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* C1: v_work = w @ h[i] */
                    ProcessCubeStage0(blockMmadWH, blockMmadWHTail, wLayout, hLayout, vLayout);
                } else {
                    /* Stage2:
                     * GDN v1:   delta_h = raw_k.T @ v_new_decay
                     * KDA/GDN2: delta_h = kg.T @ v_new
                     */
                    uint32_t windowId = cubeBlockScheduler.GetWindowId();
                    const auto& firstHeadTask = cubeBlockScheduler.GetHeadTask(0);
                    bool runStage2 = !cubeBlockScheduler.HeadTaskIsDone(firstHeadTask) &&
                                     cubeBlockScheduler.NeedProcessStage2(firstHeadTask);
                    if (runStage2) {
                        blockMmadKV.preSetFlags();
                    }
                    for (uint32_t i = 0; i < HEADS_PER_TASK; ++i) {
                        const auto& headTask = cubeBlockScheduler.GetHeadTask(i);
                        if (cubeBlockScheduler.HeadTaskIsDone(headTask)) {
                            continue;
                        }
                        const GDNFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(headTask);
                        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[windowId]);

                        if (cubeBlockScheduler.NeedProcessStage2(headTask)) {
                            // The formal k input is raw k for GDN and pre-scaled kg for KDA/GDN2.
                            int64_t cube2OffsetKwork = cube2Offsets.wkOffset;
                            int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                            int64_t cube2OffsetH = cube2Offsets.hWorkOffset;
                            auto tensorK = tla::MakeTensor(
                                gmK[cube2OffsetKwork], kLayout,
                                Catlass::Arch::PositionGM{});
                            auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vworkLayout, Catlass::Arch::PositionGM{});
                            auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2OffsetH], hworkLayout, Catlass::Arch::PositionGM{});
                            GemmCoord cube2Shape{kHeadDim, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
                            auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                            auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                            auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                            blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                            cubeBlockScheduler.cube2Done[windowId]);
                    }
                    if (runStage2) {
                        blockMmadKV.finalWaitFlags();
                    }
                }
                currStage ^= 0x01;
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);

        }

        if ASCEND_IS_AIV {
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();
            uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t sequenceCount =
                isVariedLen ? vecBlockScheduler.tokenBatch : shapeBatch;
            uint32_t headWindowNum = (vNumHead + HEADS_PER_TASK - 1) / HEADS_PER_TASK;
            uint32_t taskCount = sequenceCount * headWindowNum;
            uint32_t hRowsPerTile = (32 * 1024) / (vHeadDim * sizeof(ElementH));
            uint32_t stateRowsPerTile =
                (64 * 1024) / (vHeadDim * sizeof(ElementInitialState));
            uint32_t rowsPerTile = Min(hRowsPerTile, stateRowsPerTile);
            uint32_t totalChunks =
                isVariedLen ? vecBlockScheduler.totalChunks : ((seqlen + chunkSize - 1) / chunkSize);
            uint32_t stateBlockSize = kHeadDim * vHeadDim;
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPing =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(0);
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(96 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPing =
                resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t taskIdx = coreIdx; taskIdx < taskCount; taskIdx += coreNum) {
                uint32_t batchIdx = taskIdx / headWindowNum;
                uint32_t headWindowIdx = taskIdx % headWindowNum;
                uint32_t headBase = headWindowIdx * HEADS_PER_TASK;
                for (uint32_t headOffset = 0; headOffset < HEADS_PER_TASK; ++headOffset) {
                    uint32_t vHeadIdx = headBase + headOffset;
                    if (vHeadIdx >= vNumHead || headOffset % subBlockNum != subBlockIdx) {
                        continue;
                    }
                    uint32_t pingpongFlag = headOffset < subBlockNum ? 1 : 0;
                    uint32_t chunkOffset =
                        isVariedLen ? vecBlockScheduler.GetVarlenChunkOffset(batchIdx) : 0;
                    uint32_t shapeBatchIdx = isVariedLen ? 0 : batchIdx;
                    uint32_t hBaseOffset =
                        (shapeBatchIdx * vNumHead * totalChunks + vHeadIdx * totalChunks + chunkOffset) *
                        stateBlockSize;
                    uint32_t initialStateBaseOffset =
                        (batchIdx * vNumHead + vHeadIdx) * stateBlockSize;
                    for (uint32_t rowOffset = 0; rowOffset < kHeadDim; rowOffset += rowsPerTile) {
                        uint32_t rowsThisTile = Min(rowsPerTile, kHeadDim - rowOffset);
                        uint32_t stateTileElems = rowsThisTile * vHeadDim;
                        uint32_t hOffset = hBaseOffset + rowOffset * vHeadDim;
                        AscendC::LocalTensor<ElementInitialState> stateUbTensor =
                            pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                        AscendC::LocalTensor<ElementH> hUbTensor =
                            pingpongFlag ? hUbTensorPing : hUbTensorPong;
                        auto eventId = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                        if (useInitialState) {
                            uint32_t initialStateOffset =
                                initialStateBaseOffset + rowOffset * vHeadDim;
                            if constexpr (!std::is_same<ElementInitialState, ElementH>::value) {
                                if constexpr (std::is_same<ElementInitialState, bfloat16_t>::value &&
                                              std::is_same<ElementH, half>::value) {
                                    // DAV 2201 has no direct BF16 -> FP16 Cast. Load BF16 into the
                                    // low-precision slot, widen in the 64 KiB state slot, then narrow.
                                    auto stateLoadTensor =
                                        hUbTensor.template ReinterpretCast<ElementInitialState>();
                                    auto stateFp32Tensor =
                                        stateUbTensor.template ReinterpretCast<float>();
                                    AscendC::DataCopy(
                                        stateLoadTensor, gmInitialState[initialStateOffset], stateTileElems);
                                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                    AscendC::Cast(
                                        stateFp32Tensor, stateLoadTensor, AscendC::RoundMode::CAST_NONE,
                                        stateTileElems);
                                    AscendC::PipeBarrier<PIPE_V>();
                                    AscendC::Cast(
                                        hUbTensor, stateFp32Tensor, AscendC::RoundMode::CAST_RINT,
                                        stateTileElems);
                                } else {
                                    AscendC::DataCopy(
                                        stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                    AscendC::Cast(
                                        hUbTensor, stateUbTensor, AscendC::RoundMode::CAST_RINT,
                                        stateTileElems);
                                }
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                AscendC::DataCopy(gmH[hOffset], hUbTensor, stateTileElems);
                            } else {
                                AscendC::DataCopy(
                                    stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                                AscendC::DataCopy(gmH[hOffset], stateUbTensor, stateTileElems);
                            }
                        } else {
                            AscendC::Duplicate(hUbTensor, static_cast<ElementH>(0), stateTileElems);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::DataCopy(gmH[hOffset], hUbTensor, stateTileElems);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                        pingpongFlag = 1 - pingpongFlag;
                    }
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

            AscendC::SyncAll<false>();

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);

            EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);
            EpilogueGDNFwdHUpdate epilogueGDNFwdHUpdate(resource);
            uint32_t pongBaseEvent = 4;

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset v
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset v
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); // preset h_update
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2); // preset h
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pongBaseEvent);
            uint32_t currStage = 0; // 0: V1, 1: V2
            bool waitStageFence = false;
            bool event0FromMte3[LOCAL_PING_PONG_STAGES] = {false, false};
            bool event2FromMte3[LOCAL_PING_PONG_STAGES] = {!(storeFinalState && std::is_same<ElementFinalState, float>::value),
                                                      !(storeFinalState && std::is_same<ElementFinalState, float>::value)};
            while (vecBlockScheduler.isRunning) {
                if (waitStageFence) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                }
                if (currStage == 0) {
                    /* Stage1:
                     * v_new = u - prediction
                     * GDN v1:   stage2_v = gate(g_last - g) * v_new
                     * KDA/GDN2: stage2_v = v_new; stage2_k = input kg
                    */
                    vecBlockScheduler.InitTasks();
                    uint32_t windowId = vecBlockScheduler.GetWindowId();
                    for (uint32_t i = 0; i < HEADS_PER_TASK; ++i) {
                        const auto& headTask = vecBlockScheduler.GetHeadTask(i);
                        if (vecBlockScheduler.HeadTaskIsDone(headTask)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec1Offsets = vecBlockScheduler.GetCurTaskOffsets(headTask);
                        bool ownsHead = i % subBlockNum == subBlockIdx;
                        uint32_t localSlot = i / subBlockNum;
                        bool isPing = localSlot == 0;
                        if (!ownsHead) {
                            Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done[windowId]);
                            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                                vecBlockScheduler.vec1Done[windowId]);
                            continue;
                        }
                        bool tailVectorPath = vec1Offsets.blockTokens < 16;
                        if (tailVectorPath) {
                            Arch::CrossCoreWaitFlag(
                                vecBlockScheduler.cube1Done[windowId]);
                            ComputeTailVWorkspace(
                                vec1Offsets, EVENT_ID3 + (isPing ? 0 : pongBaseEvent), true);
                        }
                        bool waitWsFromMte3 = storeFinalState && std::is_same<ElementFinalState, float>::value &&
                                              event0FromMte3[localSlot];
                        epilogueGDNFwdHVnew(
                            gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vec1Offsets.vWorkOffset],
                            gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset],
                            gmVWorkspace[vec1Offsets.vWorkOffset],
                            vec1Offsets.blockTokens, vec1Offsets.vBlockDim, vHeadDim,
                            vecBlockScheduler.cube1Done[windowId],
                            vecBlockScheduler.vec1Done[windowId],
                            vec1Offsets.isInitialState, vec1Offsets.isFinalState, storeFinalState,
                            waitWsFromMte3, isPing, tailVectorPath, true
                        );
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                            event0FromMte3[localSlot] = false;
                        }
                    }
                } else {
                    /* Stage3: decay h_prev, add delta_h, and publish h_next/final_state. */
                    uint32_t windowId = vecBlockScheduler.GetWindowId();
                    for (uint32_t i = 0; i < HEADS_PER_TASK; ++i) {
                        const auto& headTask = vecBlockScheduler.GetHeadTask(i);
                        if (vecBlockScheduler.HeadTaskIsDone(headTask)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec2Offsets = vecBlockScheduler.GetCurTaskOffsets(headTask);
                        bool ownsHead = i % subBlockNum == subBlockIdx;
                        uint32_t localSlot = i / subBlockNum;
                        bool isPing = localSlot == 0;
                        if (vecBlockScheduler.NeedProcessStage2(headTask)) {
                            if (!ownsHead) {
                                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[windowId]);
                            } else {
                                if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                                    event0FromMte3[localSlot] = true;
                                    event2FromMte3[localSlot] = !vec2Offsets.isFinalState;
                                }
                                // step 4:  h[i+1] += h_work if i < num_chunks - 1 else None
                                epilogueGDNFwdHUpdate(
                                    gmH[vec2Offsets.hDstOffset], gmFinalState[vec2Offsets.finalStateOffset],
                                    gmG[vec2Offsets.gOffset],
                                    gmH[vec2Offsets.hSrcOffset],
                                    gmHWorkspace[vec2Offsets.hWorkOffset],
                                    gmGk[vec2Offsets.gkOffset],
                                    gmInitialState[vec2Offsets.initialStateOffset],
                                    vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim, vHeadDim,
                                    vecBlockScheduler.cube2Done[windowId],
                                    vec2Offsets.isInitialState, vec2Offsets.isFinalState, storeFinalState,
                                    useInitialState, isPing, true
                                );
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                            }
                        } else {
                            Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[windowId]);
                        }
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        vecBlockScheduler.vec2Done[windowId]);
                }
                waitStageFence = vecBlockScheduler.isRunning;
                if (waitStageFence) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                }
                currStage ^= 0x01;
            }

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                if (event0FromMte3[0]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                }
                if (event0FromMte3[1]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pongBaseEvent);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                }
                if (event2FromMte3[0]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                }
                if (event2FromMte3[1]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
                }
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset v
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); // drain h_update
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2); // drain h
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pongBaseEvent);

        }
    }

};

}
