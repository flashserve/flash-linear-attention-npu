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
#include "../block/block_scheduler_kda_fwd_h.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_kda_fwdh_update.hpp"
#include "../../epilogue/block/block_epilogue_kda_fwdh_vnew.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "../../../../kernel_utils/block/block_mmad_pingpong_tla.hpp"
#include "../../../../kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "../../../../kernel_utils/block/block_mmad_pingpong_tla_preloadA_l1B.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
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

namespace Catlass::Gemm::Kernel {

struct KDAFwdHTileShapes128 {
    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
};

struct KDAFwdHTileShapes256 {
    using L1TileShape = tla::Shape<_128, _256, _128>;
    using L0TileShape = tla::Shape<_128, _256, _64>;
};

template <bool KGated, bool ScalarGated, bool UseExp2, bool HiLoC2 = false>
struct KDAFwdHGateTag {
    static constexpr bool value = KGated;
    static constexpr bool scalarGated = ScalarGated;
    static constexpr bool useExp2 = UseExp2;
    static constexpr bool hiLoC2 = HiLoC2;
};

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE,
    typename WORKSPACE_TYPE,
    typename TileShapes = KDAFwdHTileShapes128,
    bool kGated = false,
    bool scalarGated = true,
    bool useExp2 = false,
    bool HI_LO_C2 = false
>
class KDAFwdHKernel {
public:

    using ArchTag = Arch::Ascend950;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerKdaFwdHCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerKdaFwdHVec;

    static constexpr bool FP32_C2 =
        std::is_same<INPUT_TYPE, half>::value &&
        std::is_same<WORKSPACE_TYPE, float>::value &&
        kGated &&
        !HI_LO_C2;
    static constexpr bool FP32_H = FP32_C2;
    static constexpr bool WIDE_C2 =
        std::is_same<TileShapes, KDAFwdHTileShapes256>::value;
    static constexpr uint32_t C2_L0B_STAGES =
        FP32_C2 && WIDE_C2 ? 1 : 2;
    static constexpr uint32_t C1_L0B_STAGES =
        FP32_H && WIDE_C2 ? 1 : 2;

    using DispatchPolicyTlaMulti = Gemm::MmadPingpongTlaMulti<ArchTag, false, false, 2>;
    using DispatchPolicyTlaTail = Gemm::MmadPingpongTlaMulti<ArchTag, true, false, 1>;
    using DispatchPolicyDirectUb = Common::MmadPingpong<ArchTag, false, false, 2>;
    using DispatchPolicyTlaMultiC2 = Gemm::MmadPingpongTlaMulti<
        ArchTag, false, false, 2, false, 2, 2, 2, C2_L0B_STAGES>;
    using DispatchPolicyTlaTailC2 = Gemm::MmadPingpongTlaMulti<
        ArchTag, true, false, 1, false, 2, 2, 2, C2_L0B_STAGES>;
    using DispatchPolicyTlaMultiC1Fp32 = Gemm::MmadPingpongTlaMulti<
        ArchTag, false, false, 2, false, 2, 2, 2, C1_L0B_STAGES>;
    using DispatchPolicyTlaTailC1Fp32 = Gemm::MmadPingpongTlaMulti<
        ArchTag, true, false, 1, false, 2, 2, 2, C1_L0B_STAGES>;
    using DispatchPolicyTlaMultiC1 = std::conditional_t<
        FP32_H, DispatchPolicyTlaMultiC1Fp32, DispatchPolicyTlaMulti>;
    using DispatchPolicyTlaTailC1 = std::conditional_t<
        FP32_H, DispatchPolicyTlaTailC1Fp32, DispatchPolicyTlaTail>;
    using DispatchPolicyTlaPreloadAL1B = Gemm::MmadPingpongTlaPreloadAL1B<ArchTag, true>;
    using L1TileShapeVTla = typename TileShapes::L1TileShape;
    using L0TileShapeVTla = typename TileShapes::L0TileShape;
    using L0TileShapeC2 = typename std::conditional<
        FP32_C2 && !WIDE_C2,
        tla::Shape<_128, _128, _64>, L0TileShapeVTla>::type;
    using L0TileShapeC1 = typename std::conditional<
        FP32_H && !WIDE_C2,
        tla::Shape<_128, _128, _64>, L0TileShapeVTla>::type;

    using ElementC1 = std::conditional_t<FP32_H, float, INPUT_TYPE>;
    using WType = Gemm::GemmType<ElementC1, layout::RowMajor>;
    using HType = Gemm::GemmType<ElementC1, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using ElementUInternal = std::conditional_t<FP32_C2, float, INPUT_TYPE>;
    using UType = Gemm::GemmType<ElementUInternal, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;
    using ElementC2 = std::conditional_t<FP32_C2, float, INPUT_TYPE>;
    using VUpdateLayout = std::conditional_t<FP32_C2, layout::RowMajor, layout::zN>;
    using VUpdateType = Gemm::GemmType<ElementC2, VUpdateLayout>;

    // cube 1
    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementC1, layout::RowMajor, ElementC1, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using TileCopyWHDirectUb = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor,
        WORKSPACE_TYPE, layout::RowMajor, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTlaMultiC1, L1TileShapeVTla, L0TileShapeC1, ElementC1, ElementC1, WORKSPACE_TYPE, void, TileCopyWH>;
    using BlockMmadWHTail = Gemm::Block::BlockMmadTla<DispatchPolicyTlaTailC1, L1TileShapeVTla, L0TileShapeC1, ElementC1, ElementC1, WORKSPACE_TYPE, void, TileCopyWH>;
    using BlockMmadWHDirectUb = Common::BlockMmadTla<
        DispatchPolicyDirectUb, L1TileShapeVTla, L0TileShapeVTla,
        INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWHDirectUb>;

    // cube 2
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementC2, layout::ColumnMajor, ElementC2, VUpdateLayout, WORKSPACE_TYPE, layout::RowMajor>;
    using TileCopyKVDirectUb = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE, layout::zN,
        WORKSPACE_TYPE, layout::RowMajor, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using TileMmadKV = Gemm::Tile::TileMmadTla<ArchTag, ElementC2, typename TileCopyKV::LayoutTagL1A>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<DispatchPolicyTlaMultiC2, L1TileShapeVTla, L0TileShapeC2, ElementC2, ElementC2, WORKSPACE_TYPE, void, TileCopyKV>;
    using BlockMmadKVTail = Gemm::Block::BlockMmadTla<DispatchPolicyTlaTailC2, L1TileShapeVTla, L0TileShapeC2, ElementC2, ElementC2, WORKSPACE_TYPE, void, TileCopyKV>;
    using BlockMmadKVDirectUb = Common::BlockMmadTla<
        DispatchPolicyDirectUb, L1TileShapeVTla, L0TileShapeVTla,
        INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKVDirectUb>;

    // C2 uses one L0C tile per pipeline slot. Split the K dimension into
    // rows that fit that tile before issuing the matrix multiply.
    static constexpr uint32_t CUBE2_ROW_TILE_M = tla::get<0>(L0TileShapeC2{});

    template <bool CLEAR_L1_PADDING, typename BlockMmad, typename TensorK,
              typename TensorVwork, typename TensorHwork>
    __aicore__ inline void ComputeCube2RowTiles(
        BlockMmad &blockMmad, TensorK &tensorK, TensorVwork &tensorVwork,
        TensorHwork &tensorHwork, uint32_t vBlockDim, uint32_t blockTokens)
    {
        auto tensorBlockVwork = GetTile(
            tensorVwork, tla::MakeCoord(0, 0),
            tla::MakeShape(blockTokens, vBlockDim));
        if (kHeadDim <= CUBE2_ROW_TILE_M) {
            GemmCoord cube2Shape{kHeadDim, vBlockDim, blockTokens};
            auto tensorBlockK = GetTile(
                tensorK, tla::MakeCoord(0, 0),
                tla::MakeShape(kHeadDim, blockTokens));
            auto tensorBlockHwork = GetTile(
                tensorHwork, tla::MakeCoord(0, 0),
                tla::MakeShape(kHeadDim, vBlockDim));
            if constexpr (CLEAR_L1_PADDING) {
                blockMmad(
                    tensorBlockK, tensorBlockVwork, tensorBlockHwork,
                    cube2Shape, EmptyClass{}, true);
            } else {
                blockMmad(
                    tensorBlockK, tensorBlockVwork, tensorBlockHwork,
                    cube2Shape);
            }
            return;
        }
        for (uint32_t rowOffset = 0; rowOffset < kHeadDim;
             rowOffset += CUBE2_ROW_TILE_M) {
            uint32_t rowCount = Min(CUBE2_ROW_TILE_M, kHeadDim - rowOffset);
            GemmCoord cube2Shape{rowCount, vBlockDim, blockTokens};
            auto tensorBlockK = GetTile(
                tensorK, tla::MakeCoord(rowOffset, 0),
                tla::MakeShape(rowCount, blockTokens));
            auto tensorBlockHwork = GetTile(
                tensorHwork, tla::MakeCoord(rowOffset, 0),
                tla::MakeShape(rowCount, vBlockDim));
            blockMmad(
                tensorBlockK, tensorBlockVwork, tensorBlockHwork,
                cube2Shape, EmptyClass{}, CLEAR_L1_PADDING);
        }
    }

    template <typename TensorK, typename TensorVwork, typename TensorHwork>
    __aicore__ inline void ComputeBoundedCube2(
        BlockMmadKV &blockMmadKV, BlockMmadKVTail &blockMmadKVTail,
        TensorK &tensorK, TensorVwork &tensorVwork, TensorHwork &tensorHwork,
        uint32_t vBlockDim, uint32_t blockTokens)
    {
        if (blockTokens < chunkSize) {
            if (kHeadDim <= CUBE2_ROW_TILE_M) {
                blockMmadKVTail.preSetFlags();
                ComputeCube2RowTiles<true>(
                    blockMmadKVTail, tensorK, tensorVwork, tensorHwork,
                    vBlockDim, blockTokens);
                blockMmadKVTail.finalWaitFlags();
            } else {
                // K=256 needs two row tiles. The single-stage tail policy
                // cannot keep both row tiles live safely.
                blockMmadKV.preSetFlags();
                ComputeCube2RowTiles<true>(
                    blockMmadKV, tensorK, tensorVwork, tensorHwork,
                    vBlockDim, blockTokens);
                blockMmadKV.finalWaitFlags();
            }
        } else {
            blockMmadKV.preSetFlags();
            ComputeCube2RowTiles<false>(
                blockMmadKV, tensorK, tensorVwork, tensorHwork,
                vBlockDim, blockTokens);
            blockMmadKV.finalWaitFlags();
        }
    }

    // vec 1
    using DispatchPolicyKDAFwdHVnew = Epilogue::EpilogueAtlasKDAFwdHVnew;
    using GateTag = KDAFwdHGateTag<kGated, scalarGated, useExp2, HI_LO_C2>;
    using EpilogueKDAFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyKDAFwdHVnew, VType, GType, UType, VworkType, VUpdateType, FinalStateType, GateTag>;

    // vec 2
    using DispatchPolicyKDAFwdHUpdate = Epilogue::EpilogueAtlasKDAFwdHUpdate;
    using EpilogueKDAFwdHUpdate = Epilogue::Block::BlockEpilogue<DispatchPolicyKDAFwdHUpdate, HType, GType, HType, HworkType, FinalStateType, GateTag>;

    using KDAFwdHOffsets = Catlass::Gemm::Block::KDAFwdHOffsets;
    using KDAFwdHStream = Catlass::Gemm::Block::KDAFwdHStream;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementWInternal = ElementC1;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementHInternal = ElementC1;
    using ElementV = INPUT_TYPE;
    using ElementVUpdate = ElementC2;
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;

    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutVUpdate = typename VUpdateType::Layout;

    static_assert(
        !HI_LO_C2 ||
            (kGated && !scalarGated &&
             std::is_same<INPUT_TYPE, bfloat16_t>::value &&
             std::is_same<WORKSPACE_TYPE, float>::value),
        "HI_LO_C2 is reserved for the BF16 KDA recurrence path");

    static constexpr uint64_t DIRECT_UB_FREE_FLAG_BEGIN = 1;
    static constexpr uint64_t DIRECT_UB_READY_FLAG_BEGIN = 6;
    static constexpr uint64_t DIRECT_UB_FLAG_STRIDE = 16;
    static constexpr uint32_t DIRECT_UB_STAGES = 2;
    static constexpr uint32_t DIRECT_VEC_NUM = 2;

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
    uint32_t kDecayWorkspaceOffset;
    int64_t stateOperandFp32Offset;
    bool storeH;
    bool useDirectFp32Ub;

    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementUInternal> gmUInternal;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementG> gmGk;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementWInternal> gmWInternal;
    AscendC::GlobalTensor<ElementHInternal> gmHInternal;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementK> gmKResidual;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementVUpdate> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;
    AscendC::GlobalTensor<ElementVUpdate> gmKDecayWorkspace;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    AscendC::LocalTensor<ElementHWork> ubHUpdatePing;
    AscendC::LocalTensor<ElementHWork> ubHUpdatePong;
    AscendC::LocalTensor<ElementVWork> ubVWorkPing;
    AscendC::LocalTensor<ElementVWork> ubVWorkPong;

    AscendC::LocalTensor<ElementV> l1VUpdatePing;
    AscendC::LocalTensor<ElementV> l1VUpdatePong;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;


    __aicore__ inline KDAFwdHKernel() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
        GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user) {

        static_assert(!FP32_H,
            "FP32 h recurrence requires InitFromData with hidden FP32 buffers");

        __gm__ ChunkKdaFwdHTilingData *__restrict kdaFwdHTilingData = reinterpret_cast<__gm__ ChunkKdaFwdHTilingData *__restrict>(tiling);

        batch = kdaFwdHTilingData->batch;
        seqlen = kdaFwdHTilingData->seqlen;
        kNumHead = kdaFwdHTilingData->kNumHead;
        vNumHead = kdaFwdHTilingData->vNumHead;
        kHeadDim = kdaFwdHTilingData->kHeadDim;
        vHeadDim = kdaFwdHTilingData->vHeadDim;
        chunkSize = kdaFwdHTilingData->chunkSize;
        useInitialState = kdaFwdHTilingData->useInitialState;
        storeFinalState = kdaFwdHTilingData->storeFinalState;
        isVariedLen = kdaFwdHTilingData->isVariedLen;
        shapeBatch = kdaFwdHTilingData->shapeBatch;
        tokenBatch = kdaFwdHTilingData->tokenBatch;
        vWorkspaceOffset = kdaFwdHTilingData->vWorkspaceOffset;
        vUpdateWorkspaceOffset = kdaFwdHTilingData->vUpdateWorkspaceOffset;
        hWorkspaceOffset = kdaFwdHTilingData->hWorkspaceOffset;
        numSeqWorkspaceOffset = kdaFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = kdaFwdHTilingData->numChunksWorkspaceOffset;
        kDecayWorkspaceOffset = kdaFwdHTilingData->kDecayWorkspaceOffset;
        stateOperandFp32Offset = 0;
        storeH = true;
        uint64_t denseTaskCount = static_cast<uint64_t>(shapeBatch) * vNumHead;
        useDirectFp32Ub = !FP32_C2 && !FP32_H &&
                          std::is_same<ElementVWork, float>::value &&
                          !HI_LO_C2 &&
                          !isVariedLen && chunkSize <= 64 &&
                          seqlen % chunkSize == 0 &&
                          kHeadDim == 128 && vHeadDim == 128 &&
                          denseTaskCount >= AscendC::GetBlockNum();

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmKResidual.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmUInternal.SetGlobalBuffer((__gm__ ElementUInternal *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)(scalarGated ? g : gk));
        gmGk.SetGlobalBuffer((__gm__ ElementG *)(kGated ? gk : g));
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmWInternal.SetGlobalBuffer((__gm__ ElementWInternal *)w);
        gmHInternal.SetGlobalBuffer((__gm__ ElementHInternal *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementVUpdate *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmKDecayWorkspace.SetGlobalBuffer((__gm__ ElementVUpdate *)(user + kDecayWorkspaceOffset));

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        ubHUpdatePing = resource.ubBuf.template GetBufferByByte<ElementHWork>(32 * 1024);
        ubHUpdatePong = resource.ubBuf.template GetBufferByByte<ElementHWork>(96 * 1024);
        ubVWorkPing = resource.ubBuf.template GetBufferByByte<ElementVWork>(32 * 1024);
        ubVWorkPong = resource.ubBuf.template GetBufferByByte<ElementVWork>(96 * 1024);

        l1VUpdatePing = resource.l1Buf.template GetBufferByByte<ElementV>(0);
        l1VUpdatePong = resource.l1Buf.template GetBufferByByte<ElementV>(chunkSize * vHeadDim * sizeof(ElementV));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }
    }

    template <typename TilingData>
    __aicore__ inline void InitFromData(
        GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR w_fp32, GM_ADDR u_fp32,
        GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state,
        GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
        GM_ADDR v_new_fp32, GM_ADDR h_fp32, GM_ADDR final_state, const TilingData& tilingData,
        GM_ADDR user, GM_ADDR k_residual) {
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
        kDecayWorkspaceOffset = tilingData.kDecayWorkspaceOffset;
        stateOperandFp32Offset = tilingData.stateOperandFp32Offset;
        storeH = tilingData.storeH;
        uint64_t denseTaskCount = static_cast<uint64_t>(shapeBatch) * vNumHead;
        useDirectFp32Ub = !FP32_C2 && !FP32_H &&
                          std::is_same<ElementVWork, float>::value &&
                          !HI_LO_C2 &&
                          !isVariedLen && chunkSize <= 64 &&
                          seqlen % chunkSize == 0 &&
                          kHeadDim == 128 && vHeadDim == 128 &&
                          denseTaskCount >= AscendC::GetBlockNum();

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)(scalarGated ? g : gk));
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        if constexpr (FP32_H) {
            gmWInternal.SetGlobalBuffer((__gm__ ElementWInternal *)w_fp32);
            gmUInternal.SetGlobalBuffer((__gm__ ElementUInternal *)u_fp32);
            gmHInternal.SetGlobalBuffer((__gm__ ElementHInternal *)h_fp32);
        } else {
            gmWInternal.SetGlobalBuffer((__gm__ ElementWInternal *)w);
            gmUInternal.SetGlobalBuffer((__gm__ ElementUInternal *)u);
            gmHInternal.SetGlobalBuffer((__gm__ ElementHInternal *)h);
        }
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        if constexpr (HI_LO_C2) {
            gmKResidual.SetGlobalBuffer((__gm__ ElementK *)k_residual);
        } else {
            gmKResidual.SetGlobalBuffer((__gm__ ElementK *)k);
        }
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        if constexpr (FP32_C2) {
            gmVUpdateWorkspace.SetGlobalBuffer(
                (__gm__ ElementVUpdate *)v_new_fp32);
        } else {
            gmVUpdateWorkspace.SetGlobalBuffer(
                (__gm__ ElementVUpdate *)(user + vUpdateWorkspaceOffset));
        }
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmGk.SetGlobalBuffer((__gm__ ElementG *)(kGated ? gk : g));
        gmKDecayWorkspace.SetGlobalBuffer((__gm__ ElementVUpdate *)(user + kDecayWorkspaceOffset));
        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        ubHUpdatePing = resource.ubBuf.template GetBufferByByte<ElementHWork>(32 * 1024);
        ubHUpdatePong = resource.ubBuf.template GetBufferByByte<ElementHWork>(96 * 1024);
        ubVWorkPing = resource.ubBuf.template GetBufferByByte<ElementVWork>(32 * 1024);
        ubVWorkPong = resource.ubBuf.template GetBufferByByte<ElementVWork>(96 * 1024);

        l1VUpdatePing = resource.l1Buf.template GetBufferByByte<ElementV>(0);
        l1VUpdatePong = resource.l1Buf.template GetBufferByByte<ElementV>(chunkSize * vHeadDim * sizeof(ElementV));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
        }
        if ASCEND_IS_AIV {
            vecBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
        }
    }

    __aicore__ inline uint64_t VUpdateBase(
        const KDAFwdHOffsets& offsets) const
    {
        if constexpr (FP32_C2) {
            return offsets.uvOffset;
        }
        if constexpr (HI_LO_C2) {
            return 2ULL * offsets.vWorkOffset;
        }
        return offsets.vWorkOffset;
    }

    __aicore__ inline void ExportPublicH(
        uint32_t coreIdx, uint32_t coreNum,
        uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        if constexpr (FP32_H) {
            if (!storeH || subBlockNum == 0) {
                return;
            }
            constexpr float FP16_MAX = 65504.0f;
            constexpr uint32_t FP32_UB_OFFSET = 0;
            constexpr uint32_t OUTPUT_UB_OFFSET = 32 * 1024;
            constexpr uint32_t FP32_UB_BYTES = 32 * 1024;
            AscendC::LocalTensor<float> source =
                resource.ubBuf.template GetBufferByByte<float>(FP32_UB_OFFSET);
            AscendC::LocalTensor<ElementH> output =
                resource.ubBuf.template GetBufferByByte<ElementH>(OUTPUT_UB_OFFSET);
            uint32_t rowsPerTile = FP32_UB_BYTES /
                (vHeadDim * static_cast<uint32_t>(sizeof(float)));
            uint32_t taskCount =
                (isVariedLen ? vecBlockScheduler.tokenBatch : shapeBatch) * vNumHead;
            uint32_t totalChunks = vecBlockScheduler.totalChunks;
            uint64_t stateBlockSize =
                static_cast<uint64_t>(kHeadDim) * vHeadDim;
            for (uint32_t taskIdx = coreIdx; taskIdx < taskCount; taskIdx += coreNum) {
                uint32_t batchIdx = taskIdx / vNumHead;
                uint32_t vHeadIdx = taskIdx % vNumHead;
                uint32_t shapeBatchIdx = batchIdx;
                uint32_t chunkOffset = 0;
                uint32_t batchChunks = totalChunks;
                if (isVariedLen) {
                    KDAFwdHStream stream{};
                    vecBlockScheduler.ResolveVarlenSequence(batchIdx, stream);
                    shapeBatchIdx = 0;
                    chunkOffset = stream.chunkOffset;
                    batchChunks = stream.batchChunks;
                }
                uint32_t rowBegin =
                    (kHeadDim * subBlockIdx) / subBlockNum;
                uint32_t rowEnd =
                    (kHeadDim * (subBlockIdx + 1)) / subBlockNum;
                for (uint32_t chunk = 0; chunk < batchChunks; ++chunk) {
                    uint64_t hBase =
                        ((static_cast<uint64_t>(shapeBatchIdx) * vNumHead + vHeadIdx) *
                             totalChunks +
                         chunkOffset + chunk) *
                        stateBlockSize;
                    for (uint32_t row = rowBegin; row < rowEnd; row += rowsPerTile) {
                        uint32_t rowsThisTile = Min(rowsPerTile, rowEnd - row);
                        uint32_t count = rowsThisTile * vHeadDim;
                        uint64_t offset = hBase + static_cast<uint64_t>(row) * vHeadDim;
                        AscendC::DataCopy(source, gmHInternal[offset], count);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                        AscendC::Mins(source, source, FP16_MAX, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Maxs(source, source, -FP16_MAX, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Cast(
                            output, source, AscendC::RoundMode::CAST_RINT, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::DataCopy(gmH[offset], output, count);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    }
                }
            }
        }
    }

    // Tail helpers borrow the stream's V_MTE2 free token and restore it before returning.
    __aicore__ inline void ComputeTailVWorkspace(
        const KDAFwdHOffsets& offsets, uint32_t tailEventId)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
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
        constexpr uint32_t TAIL_WEIGHT_INPUT_OFFSET = 169 * 1024;
        constexpr uint32_t TAIL_WEIGHT_FLOAT_OFFSET = 170 * 1024;
        AscendC::LocalTensor<ElementH> inputUb =
            resource.ubBuf.template GetBufferByByte<ElementH>(TAIL_INPUT_OFFSET);
        AscendC::LocalTensor<float> floatUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_FLOAT_OFFSET);
        AscendC::LocalTensor<float> accumUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_ACCUM_OFFSET);
        AscendC::LocalTensor<ElementW> weightInputUb =
            resource.ubBuf.template GetBufferByByte<ElementW>(TAIL_WEIGHT_INPUT_OFFSET);
        AscendC::LocalTensor<float> weightFloatUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_WEIGHT_FLOAT_OFFSET);

        for (uint32_t tokenRow = rowBegin; tokenRow < rowEnd; ++tokenRow) {
            if constexpr (FP32_H) {
                AscendC::DataCopy(
                    weightFloatUb,
                    gmWInternal[offsets.wOffset + tokenRow * kHeadDim],
                    kHeadDim);
            } else {
                AscendC::DataCopy(
                    weightInputUb,
                    gmWInternal[offsets.wOffset + tokenRow * kHeadDim],
                    kHeadDim);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
            if constexpr (!FP32_H) {
                AscendC::Cast(
                    weightFloatUb, weightInputUb, AscendC::RoundMode::CAST_NONE,
                    kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(tailEventId);

            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                if constexpr (FP32_H) {
                    AscendC::DataCopy(
                        floatUb,
                        gmHInternal[offsets.hSrcOffset + kIdx * vHeadDim],
                        offsets.vBlockDim);
                } else {
                    AscendC::DataCopy(
                        inputUb,
                        gmHInternal[offsets.hSrcOffset + kIdx * vHeadDim],
                        offsets.vBlockDim);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                if constexpr (!FP32_H) {
                    AscendC::Cast(
                        floatUb, inputUb, AscendC::RoundMode::CAST_NONE,
                        offsets.vBlockDim);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                float weight = weightFloatUb.GetValue(kIdx);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
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

    __aicore__ inline void ComputeTailHWorkspace(
        const KDAFwdHOffsets& offsets, uint32_t tailEventId)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(kHeadDim, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);

        constexpr uint32_t TAIL_INPUT_OFFSET = 166 * 1024;
        constexpr uint32_t TAIL_FLOAT_OFFSET = 167 * 1024;
        constexpr uint32_t TAIL_ACCUM_OFFSET = 168 * 1024;
        constexpr uint32_t TAIL_WEIGHT_INPUT_OFFSET = 169 * 1024;
        constexpr uint32_t TAIL_WEIGHT_FLOAT_OFFSET = 170 * 1024;
        AscendC::LocalTensor<ElementV> inputUb =
            resource.ubBuf.template GetBufferByByte<ElementV>(TAIL_INPUT_OFFSET);
        AscendC::LocalTensor<float> floatUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_FLOAT_OFFSET);
        AscendC::LocalTensor<float> accumUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_ACCUM_OFFSET);
        AscendC::LocalTensor<ElementK> weightInputUb =
            resource.ubBuf.template GetBufferByByte<ElementK>(TAIL_WEIGHT_INPUT_OFFSET);
        AscendC::LocalTensor<float> weightFloatUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_WEIGHT_FLOAT_OFFSET);

        for (uint32_t kRow = rowBegin; kRow < rowEnd; ++kRow) {
            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t reductionRows =
                HI_LO_C2 ? 2 * offsets.blockTokens : offsets.blockTokens;
            for (uint32_t tokenRow = 0; tokenRow < reductionRows; ++tokenRow) {
                if constexpr (FP32_C2) {
                    AscendC::DataCopy(
                        weightFloatUb,
                        gmKDecayWorkspace[
                            offsets.kDecayWorkOffset + tokenRow * kHeadDim],
                        kHeadDim);
                } else if constexpr (kGated) {
                    uint64_t kDecayBase = HI_LO_C2
                        ? 2ULL * offsets.kDecayWorkOffset
                        : offsets.kDecayWorkOffset;
                    AscendC::DataCopy(
                        weightInputUb,
                        gmKDecayWorkspace[
                            kDecayBase + tokenRow * kHeadDim],
                        kHeadDim);
                } else {
                    AscendC::DataCopy(
                        weightInputUb,
                        gmK[offsets.wkOffset + tokenRow * kHeadDim],
                        kHeadDim);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                if constexpr (!FP32_C2) {
                    AscendC::Cast(
                        weightFloatUb, weightInputUb, AscendC::RoundMode::CAST_NONE,
                        kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_S>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(tailEventId);
                uint64_t vUpdateBase = VUpdateBase(offsets);
                if constexpr (FP32_C2) {
                    AscendC::DataCopy(
                        floatUb,
                        gmVUpdateWorkspace[
                            vUpdateBase + tokenRow * offsets.vBlockDim],
                        offsets.vBlockDim);
                } else {
                    // Low-precision V1 stores the Cube operand in zN layout.
                    // Gather one logical token row before the vector fallback.
                    constexpr uint32_t C0_SIZE = 16;
                    uint32_t paddedTokens =
                        CeilDiv(reductionRows, C0_SIZE) * C0_SIZE;
                    for (uint32_t vOffset = 0; vOffset < offsets.vBlockDim;
                         vOffset += C0_SIZE) {
                        uint64_t srcOffset =
                            vUpdateBase +
                            (vOffset / C0_SIZE) * paddedTokens * C0_SIZE +
                            tokenRow * C0_SIZE;
                        AscendC::DataCopy(
                            inputUb[vOffset], gmVUpdateWorkspace[srcOffset],
                            C0_SIZE);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                if constexpr (!FP32_C2) {
                    AscendC::Cast(
                        floatUb, inputUb, AscendC::RoundMode::CAST_NONE,
                        offsets.vBlockDim);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                float weight = weightFloatUb.GetValue(kRow);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::DataCopy(
                gmHWorkspace[offsets.hWorkOffset + kRow * offsets.vBlockDim],
                accumUb, offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(tailEventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
    }

    __aicore__ inline void PresetEmptyVarlenFinalState() {
        if (!isVariedLen || !storeFinalState ||
            vecBlockScheduler.inputTokenBatch == vecBlockScheduler.tokenBatch) {
            return;
        }

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t rowsPerSubBlock = (kHeadDim + subBlockNum - 1) / subBlockNum;
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
        uint32_t rowsPerTile = (64 * 1024) / (vHeadDim * sizeof(ElementFinalState));
        uint64_t stateBlockSize = static_cast<uint64_t>(kHeadDim) * vHeadDim;
        uint32_t stateTaskCount = vecBlockScheduler.inputTokenBatch * vNumHead;
        uint32_t pingpongFlag = 1;
        AscendC::LocalTensor<ElementFinalState> stateUbTensorPing =
            resource.ubBuf.template GetBufferByByte<ElementFinalState>(0);
        AscendC::LocalTensor<ElementFinalState> stateUbTensorPong =
            resource.ubBuf.template GetBufferByByte<ElementFinalState>(96 * 1024);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t taskIdx = coreIdx; taskIdx < stateTaskCount; taskIdx += coreNum) {
            uint32_t batchIdx = taskIdx / vNumHead;
            int64_t seqStart = vecBlockScheduler.gmSeqlen.GetValue(batchIdx);
            int64_t seqEnd = vecBlockScheduler.gmSeqlen.GetValue(batchIdx + 1);
            if (seqStart != seqEnd) {
                continue;
            }
            uint64_t stateBaseOffset = static_cast<uint64_t>(taskIdx) * stateBlockSize;
            for (uint32_t rowOffset = rowBegin; rowOffset < rowEnd; rowOffset += rowsPerTile) {
                uint32_t rowsThisTile = Min(rowsPerTile, rowEnd - rowOffset);
                uint32_t stateTileElems = rowsThisTile * vHeadDim;
                uint64_t stateOffset =
                    stateBaseOffset + static_cast<uint64_t>(rowOffset) * vHeadDim;
                AscendC::LocalTensor<ElementFinalState> stateUbTensor =
                    pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                auto eventId = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                if (useInitialState) {
                    AscendC::DataCopy(stateUbTensor, gmInitialState[stateOffset], stateTileElems);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                } else {
                    AscendC::Duplicate(
                        stateUbTensor, static_cast<ElementFinalState>(0), stateTileElems);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                }
                AscendC::DataCopy(gmFinalState[stateOffset], stateUbTensor, stateTileElems);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                pingpongFlag = 1 - pingpongFlag;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void Process() {
        // FwdH can run after another stage in a megakernel. Start its AIC/AIV
        // handshake only after every core has retired the preceding stage.
        AscendC::SyncAll<false>();

        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = vecBlockScheduler.cubeCoreNum;

            BlockMmadWH blockMmadWH(resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
            BlockMmadKV blockMmadKV(resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
            BlockMmadWHTail blockMmadWHTail(resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
            BlockMmadKVTail blockMmadKVTail(resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
            bool useBoundedMmad = isVariedLen || (seqlen % chunkSize != 0);

            uint64_t shapeKTokens =
                static_cast<uint64_t>(shapeBatch) * kNumHead * cubeBlockScheduler.totalTokens;
            uint64_t shapeHRows = static_cast<uint64_t>(shapeBatch) * vNumHead *
                                  cubeBlockScheduler.totalChunks * kHeadDim;
            auto wLayout = tla::MakeLayout<ElementWInternal, LayoutW>(shapeKTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementHInternal, LayoutH>(shapeHRows, vHeadDim);

            auto kLayout = tla::MakeLayout<ElementC2, LayoutK>(kHeadDim, shapeKTokens);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(kHeadDim, cubeBlockScheduler.vBlockSize);

            AscendC::SyncAll<false>();
            uint32_t currStage = 0; // 0: C1, 1: C2
            while (cubeBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* C1: v_work = w @ h[i] */
                    cubeBlockScheduler.InitTasks();
                    if (useDirectFp32Ub) {
                        if constexpr (!FP32_H) {
                            BlockMmadWHDirectUb blockMmadWHDirectUb(
                                resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
                            for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                                uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                                const auto& stream = cubeBlockScheduler.GetStream(i);
                                if (cubeBlockScheduler.StreamIsDone(stream)) {
                                    continue;
                                }

                                const KDAFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                                if (cube1Offsets.blockTokens < 16) {
                                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                                        cubeBlockScheduler.cube1Done[streamId]);
                                    continue;
                                }
                                int64_t cube1OffsetW = cube1Offsets.wOffset;
                                int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                                auto tensorW = tla::MakeTensor(gmWInternal[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                                auto tensorH = tla::MakeTensor(gmHInternal[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                                GemmCoord cube1Shape {cube1Offsets.blockTokens, cube1Offsets.vBlockDim, kHeadDim};
                                auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                                auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));

                                auto ubLayout = tla::MakeLayout<ElementVWork, LayoutV>(cube1Shape.m(), cube1Shape.n());
                                auto tensorUbPing = tla::MakeTensor(ubVWorkPing, ubLayout, Catlass::Arch::PositionUB{});
                                auto tensorUbPong = tla::MakeTensor(ubVWorkPong, ubLayout, Catlass::Arch::PositionUB{});
                                using UbTensor = decltype(tensorUbPing);
                                UbTensor tensorUbList[BlockMmadWHDirectUb::MAX_CUBE_VEC_SYNC_NUM];
                                for (uint32_t ubIdx = 0; ubIdx < BlockMmadWHDirectUb::MAX_CUBE_VEC_SYNC_NUM; ++ubIdx) {
                                    tensorUbList[ubIdx] = (ubIdx & 1U) ? tensorUbPong : tensorUbPing;
                                }
                                uint32_t ubListId = streamId;
                                uint32_t rowsPerSubBlock = CeilDiv(cube1Shape.m(), DIRECT_VEC_NUM);
                                blockMmadWHDirectUb(
                                    tensorBlockW, tensorBlockH, tensorUbList, cube1Shape, rowsPerSubBlock, 0,
                                    DIRECT_UB_FREE_FLAG_BEGIN, DIRECT_UB_READY_FLAG_BEGIN, ubListId,
                                    DIRECT_VEC_NUM, DIRECT_UB_STAGES);
                            }
                        }
                    } else if (useBoundedMmad) {
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto& stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }

                            const KDAFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                            if (cube1Offsets.blockTokens < 16) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                                    cubeBlockScheduler.cube1Done[streamId]);
                                continue;
                            }
                            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(
                                cube1Offsets.blockTokens, cube1Offsets.vBlockDim);
                            auto tensorW = tla::MakeTensor(
                                gmWInternal[cube1Offsets.wOffset], wLayout, Catlass::Arch::PositionGM{});
                            auto tensorH = tla::MakeTensor(
                                gmHInternal[cube1Offsets.hSrcOffset], hLayout, Catlass::Arch::PositionGM{});
                            auto tensorV = tla::MakeTensor(
                                gmVWorkspace[cube1Offsets.vWorkOffset], vLayout,
                                Catlass::Arch::PositionGM{});
                            GemmCoord cube1Shape{
                                cube1Offsets.blockTokens, cube1Offsets.vBlockDim, kHeadDim};
                            auto tensorBlockW = GetTile(
                                tensorW, tla::MakeCoord(0, 0),
                                tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                            auto tensorBlockH = GetTile(
                                tensorH, tla::MakeCoord(0, 0),
                                tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                            auto tensorBlockV = GetTile(
                                tensorV, tla::MakeCoord(0, 0),
                                tla::MakeShape(cube1Shape.m(), cube1Shape.n()));

                            if (cube1Offsets.blockTokens < chunkSize) {
                                blockMmadWHTail.preSetFlags();
                                blockMmadWHTail(
                                    tensorBlockW, tensorBlockH, tensorBlockV,
                                    cube1Shape, EmptyClass{}, true);
                                blockMmadWHTail.finalWaitFlags();
                            } else {
                                blockMmadWH.preSetFlags();
                                blockMmadWH(
                                    tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);
                                blockMmadWH.finalWaitFlags();
                            }
                            AscendC::PipeBarrier<PIPE_ALL>();
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                                cubeBlockScheduler.cube1Done[streamId]);
                        }
                    } else {
                        blockMmadWH.preSetFlags();
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto& stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }

                            const KDAFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                            if (cube1Offsets.blockTokens < 16) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                                    cubeBlockScheduler.cube1Done[streamId]);
                                continue;
                            }
                            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(cube1Offsets.blockTokens, cube1Offsets.vBlockDim);
                            int64_t cube1OffsetW = cube1Offsets.wOffset;
                            int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                            int64_t cube1OffsetVwork = cube1Offsets.vWorkOffset;
                            auto tensorW = tla::MakeTensor(gmWInternal[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                            auto tensorH = tla::MakeTensor(gmHInternal[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                            auto tensorV = tla::MakeTensor(gmVWorkspace[cube1OffsetVwork], vLayout, Catlass::Arch::PositionGM{});
                            GemmCoord cube1Shape {cube1Offsets.blockTokens, cube1Offsets.vBlockDim, kHeadDim};
                            auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                            auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                            auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));

                            blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);
                        }
                        blockMmadWH.finalWaitFlags();
                    }
                } else {
                    /* C2: h[i+1] = k.T @ v_work */
                    if constexpr (HI_LO_C2) {
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto& stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }
                            const KDAFwdHOffsets& cube2Offsets =
                                cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(
                                cubeBlockScheduler.vec1Done[streamId]);

                            if (!cubeBlockScheduler.NeedProcessStage2(stream)) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                                    cubeBlockScheduler.cube2Done[streamId]);
                                continue;
                            }
                            if (cube2Offsets.blockTokens < 16) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                                    cubeBlockScheduler.cube2Done[streamId]);
                                continue;
                            }

                            uint32_t reductionRows = 2 * cube2Offsets.blockTokens;
                            uint64_t cube2OffsetK =
                                2ULL * cube2Offsets.kDecayWorkOffset;
                            uint64_t cube2OffsetVwork =
                                2ULL * cube2Offsets.vWorkOffset;
                            auto kResidualLayout =
                                tla::MakeLayout<ElementK, LayoutK>(
                                    kHeadDim, reductionRows);
                            auto vUpdateResidualLayout =
                                tla::MakeLayout<ElementVUpdate, LayoutVUpdate>(
                                    reductionRows, cube2Offsets.vBlockDim);
                            auto tensorK = tla::MakeTensor(
                                gmKDecayWorkspace[cube2OffsetK], kResidualLayout,
                                Catlass::Arch::PositionGM{});
                            auto tensorVwork = tla::MakeTensor(
                                gmVUpdateWorkspace[cube2OffsetVwork],
                                vUpdateResidualLayout,
                                Catlass::Arch::PositionGM{});
                            auto tensorHwork = tla::MakeTensor(
                                gmHWorkspace[cube2Offsets.hWorkOffset], hworkLayout,
                                Catlass::Arch::PositionGM{});

                            blockMmadKV.preSetFlags();
                            if (cube2Offsets.blockTokens < chunkSize) {
                                ComputeCube2RowTiles<true>(
                                    blockMmadKV, tensorK, tensorVwork, tensorHwork,
                                    cube2Offsets.vBlockDim, reductionRows);
                            } else {
                                ComputeCube2RowTiles<false>(
                                    blockMmadKV, tensorK, tensorVwork, tensorHwork,
                                    cube2Offsets.vBlockDim, reductionRows);
                            }
                            blockMmadKV.finalWaitFlags();
                            AscendC::PipeBarrier<PIPE_ALL>();
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                                cubeBlockScheduler.cube2Done[streamId]);
                        }
                    } else if (useDirectFp32Ub) {
                        BlockMmadKVDirectUb blockMmadKVDirectUb(
                            resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto& stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }
                            const KDAFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);

                            if (cubeBlockScheduler.NeedProcessStage2(stream)) {
                                if (cube2Offsets.blockTokens < 16) {
                                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                                        cubeBlockScheduler.cube2Done[streamId]);
                                    continue;
                                }
                                int64_t cube2OffsetK = kGated ? cube2Offsets.kDecayWorkOffset : cube2Offsets.wkOffset;
                                int64_t cube2OffsetVwork = VUpdateBase(cube2Offsets);
                                auto vUpdateLayout = tla::MakeLayout<ElementVUpdate, LayoutVUpdate>(cube2Offsets.blockTokens, cube2Offsets.vBlockDim);
                                auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vUpdateLayout, Catlass::Arch::PositionGM{});
                                GemmCoord cube2Shape{kHeadDim, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
                                auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));

                                auto ubLayout = tla::MakeLayout<ElementHWork, LayoutH>(cube2Shape.m(), cube2Shape.n());
                                auto tensorUbPing = tla::MakeTensor(ubHUpdatePing, ubLayout, Catlass::Arch::PositionUB{});
                                auto tensorUbPong = tla::MakeTensor(ubHUpdatePong, ubLayout, Catlass::Arch::PositionUB{});
                                using UbTensor = decltype(tensorUbPing);
                                UbTensor tensorUbList[BlockMmadKVDirectUb::MAX_CUBE_VEC_SYNC_NUM];
                                for (uint32_t ubIdx = 0; ubIdx < BlockMmadKVDirectUb::MAX_CUBE_VEC_SYNC_NUM; ++ubIdx) {
                                    tensorUbList[ubIdx] = (ubIdx & 1U) ? tensorUbPong : tensorUbPing;
                                }
                                uint32_t ubListId = streamId;
                                uint32_t rowsPerSubBlock = CeilDiv(cube2Shape.m(), DIRECT_VEC_NUM);
                                if constexpr (!FP32_C2) {
                                    auto tensorK = kGated
                                        ? tla::MakeTensor(
                                              gmKDecayWorkspace[cube2OffsetK], kLayout,
                                              Catlass::Arch::PositionGM{})
                                        : tla::MakeTensor(
                                              gmK[cube2OffsetK], kLayout,
                                              Catlass::Arch::PositionGM{});
                                    auto tensorBlockK = GetTile(
                                        tensorK, tla::MakeCoord(0, 0),
                                        tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                                    blockMmadKVDirectUb(
                                        tensorBlockK, tensorBlockVwork, tensorUbList,
                                        cube2Shape, rowsPerSubBlock, 0,
                                        DIRECT_UB_FREE_FLAG_BEGIN, DIRECT_UB_READY_FLAG_BEGIN,
                                        ubListId, DIRECT_VEC_NUM, DIRECT_UB_STAGES);
                                }
                            }
                        }
                    } else if (useBoundedMmad) {
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto& stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }
                            const KDAFwdHOffsets& cube2Offsets =
                                cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);

                            bool useTailVector = cube2Offsets.blockTokens < 16;
                            if (cubeBlockScheduler.NeedProcessStage2(stream) && !useTailVector) {
                                int64_t cube2OffsetK = kGated
                                    ? cube2Offsets.kDecayWorkOffset
                                    : cube2Offsets.wkOffset;
                                auto vUpdateLayout = tla::MakeLayout<ElementVUpdate, LayoutVUpdate>(
                                    cube2Offsets.blockTokens, cube2Offsets.vBlockDim);
                                auto tensorVwork = tla::MakeTensor(
                                    gmVUpdateWorkspace[VUpdateBase(cube2Offsets)], vUpdateLayout,
                                    Catlass::Arch::PositionGM{});
                                auto tensorHwork = tla::MakeTensor(
                                    gmHWorkspace[cube2Offsets.hWorkOffset], hworkLayout,
                                    Catlass::Arch::PositionGM{});

                                if constexpr (FP32_C2) {
                                    auto tensorK = tla::MakeTensor(
                                        gmKDecayWorkspace[cube2Offsets.kDecayWorkOffset],
                                        kLayout, Catlass::Arch::PositionGM{});
                                    ComputeBoundedCube2(
                                        blockMmadKV, blockMmadKVTail, tensorK,
                                        tensorVwork, tensorHwork,
                                        cube2Offsets.vBlockDim,
                                        cube2Offsets.blockTokens);
                                } else {
                                    auto tensorK = kGated
                                        ? tla::MakeTensor(
                                              gmKDecayWorkspace[cube2OffsetK], kLayout,
                                              Catlass::Arch::PositionGM{})
                                        : tla::MakeTensor(
                                              gmK[cube2OffsetK], kLayout,
                                              Catlass::Arch::PositionGM{});
                                    ComputeBoundedCube2(
                                        blockMmadKV, blockMmadKVTail, tensorK,
                                        tensorVwork, tensorHwork,
                                        cube2Offsets.vBlockDim,
                                        cube2Offsets.blockTokens);
                                }
                                AscendC::PipeBarrier<PIPE_ALL>();
                            }
                            if (useTailVector && cubeBlockScheduler.NeedProcessStage2(stream)) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                                    cubeBlockScheduler.cube2Done[streamId]);
                            } else {
                                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                                    cubeBlockScheduler.cube2Done[streamId]);
                            }
                        }
                    } else {
                        blockMmadKV.preSetFlags();
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto& stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }
                            const KDAFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);

                            if (cubeBlockScheduler.NeedProcessStage2(stream)) {
                                if (cube2Offsets.blockTokens < 16) {
                                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                                        cubeBlockScheduler.cube2Done[streamId]);
                                    continue;
                                }
                                // step 3: h[i+1] = k.T @ v_work
                                int64_t cube2OffsetK = kGated ? cube2Offsets.kDecayWorkOffset : cube2Offsets.wkOffset;
                                int64_t cube2OffsetVwork = VUpdateBase(cube2Offsets);
                                auto vUpdateLayout = tla::MakeLayout<ElementVUpdate, LayoutVUpdate>(cube2Offsets.blockTokens, cube2Offsets.vBlockDim);
                                auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vUpdateLayout, Catlass::Arch::PositionGM{});
                                auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2Offsets.hWorkOffset], hworkLayout, Catlass::Arch::PositionGM{});
                                if constexpr (FP32_C2) {
                                    auto tensorK = tla::MakeTensor(
                                        gmKDecayWorkspace[cube2Offsets.kDecayWorkOffset],
                                        kLayout, Catlass::Arch::PositionGM{});
                                    ComputeCube2RowTiles<false>(
                                        blockMmadKV, tensorK, tensorVwork, tensorHwork,
                                        cube2Offsets.vBlockDim, cube2Offsets.blockTokens);
                                } else {
                                    auto tensorK = kGated
                                        ? tla::MakeTensor(
                                              gmKDecayWorkspace[cube2OffsetK], kLayout,
                                              Catlass::Arch::PositionGM{})
                                        : tla::MakeTensor(
                                              gmK[cube2OffsetK], kLayout,
                                              Catlass::Arch::PositionGM{});
                                    ComputeCube2RowTiles<false>(
                                        blockMmadKV, tensorK, tensorVwork, tensorHwork,
                                        cube2Offsets.vBlockDim, cube2Offsets.blockTokens);
                                }
                            }
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done[streamId]);
                        }
                        blockMmadKV.finalWaitFlags();
                    }
                }
                currStage ^= 0x01;
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
            if (useDirectFp32Ub) {
                for (uint32_t slot = 0; slot < DIRECT_UB_STAGES; ++slot) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(DIRECT_UB_FREE_FLAG_BEGIN + slot);
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(
                        DIRECT_UB_FREE_FLAG_BEGIN + DIRECT_UB_FLAG_STRIDE + slot);
                }
            }

        }

        if ASCEND_IS_AIV {
            PresetEmptyVarlenFinalState();
            bool useBoundedMmad = isVariedLen || (seqlen % chunkSize != 0);
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();
            uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t taskCount =
                (isVariedLen ? vecBlockScheduler.tokenBatch : shapeBatch) * vNumHead;
            uint32_t tasksPerCore = taskCount > coreNum ? PING_PONG_STAGES : 1;
            uint32_t taskStride = coreNum * tasksPerCore;
            uint32_t rowsPerSubBlock = (kHeadDim + subBlockNum - 1) / subBlockNum;
            uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
            uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
            uint32_t hRowsPerTile =
                (32 * 1024) / (vHeadDim * sizeof(ElementHInternal));
            uint32_t stateRowsPerTile =
                (64 * 1024) / (vHeadDim * sizeof(ElementInitialState));
            uint32_t rowsPerTile = Min(hRowsPerTile, stateRowsPerTile);
            uint32_t totalChunks =
                isVariedLen ? vecBlockScheduler.totalChunks : ((seqlen + chunkSize - 1) / chunkSize);
            uint64_t stateBlockSize = static_cast<uint64_t>(kHeadDim) * vHeadDim;
            uint32_t pingpongFlag = 1;
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPing =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(0);
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(96 * 1024);
            AscendC::LocalTensor<ElementHInternal> hUbTensorPing =
                resource.ubBuf.template GetBufferByByte<ElementHInternal>(64 * 1024);
            AscendC::LocalTensor<ElementHInternal> hUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementHInternal>(160 * 1024);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t slot = 0; slot < tasksPerCore; ++slot) {
                for (uint32_t taskIdx = coreIdx * tasksPerCore + slot;
                     taskIdx < taskCount; taskIdx += taskStride) {
                    uint32_t batchIdx = taskIdx / vNumHead;
                    uint32_t vHeadIdx = taskIdx % vNumHead;
                    uint32_t chunkOffset = 0;
                    uint32_t stateBatchIdx = batchIdx;
                    if (isVariedLen) {
                        KDAFwdHStream resolvedStream{};
                        vecBlockScheduler.ResolveVarlenSequence(batchIdx, resolvedStream);
                        chunkOffset = resolvedStream.chunkOffset;
                        stateBatchIdx = resolvedStream.batchIdx;
                    }
                    uint32_t shapeBatchIdx = isVariedLen ? 0 : batchIdx;
                    uint64_t hBaseOffset =
                        ((static_cast<uint64_t>(shapeBatchIdx) * vNumHead + vHeadIdx) *
                             totalChunks +
                         chunkOffset) *
                        stateBlockSize;
                    uint64_t initialStateBaseOffset =
                        (static_cast<uint64_t>(stateBatchIdx) * vNumHead + vHeadIdx) *
                        stateBlockSize;
                    for (uint32_t rowOffset = rowBegin; rowOffset < rowEnd; rowOffset += rowsPerTile) {
                        uint32_t rowsThisTile = Min(rowsPerTile, rowEnd - rowOffset);
                        uint32_t stateTileElems = rowsThisTile * vHeadDim;
                        uint64_t hOffset =
                            hBaseOffset + static_cast<uint64_t>(rowOffset) * vHeadDim;
                        AscendC::LocalTensor<ElementInitialState> stateUbTensor =
                            pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                        AscendC::LocalTensor<ElementHInternal> hUbTensor =
                            pingpongFlag ? hUbTensorPing : hUbTensorPong;
                        auto eventId = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                        if (useInitialState) {
                            uint64_t initialStateOffset =
                                initialStateBaseOffset +
                                static_cast<uint64_t>(rowOffset) * vHeadDim;
                            if constexpr (!std::is_same<ElementInitialState, ElementHInternal>::value) {
                                AscendC::DataCopy(
                                    stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                AscendC::Cast(
                                    hUbTensor, stateUbTensor, AscendC::RoundMode::CAST_RINT,
                                    stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                AscendC::DataCopy(
                                    gmHInternal[hOffset], hUbTensor, stateTileElems);
                            } else {
                                AscendC::DataCopy(
                                    stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                                AscendC::DataCopy(
                                    gmHInternal[hOffset], stateUbTensor, stateTileElems);
                            }
                        } else {
                            AscendC::Duplicate(
                                hUbTensor, static_cast<ElementHInternal>(0), stateTileElems);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::DataCopy(
                                gmHInternal[hOffset], hUbTensor, stateTileElems);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                        pingpongFlag = 1 - pingpongFlag;
                    }
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

            AscendC::SyncAll<false>();

            if (useDirectFp32Ub) {
                for (uint32_t slot = 0; slot < DIRECT_UB_STAGES; ++slot) {
                    AscendC::CrossCoreSetFlag<0x4, PIPE_V>(DIRECT_UB_FREE_FLAG_BEGIN + slot);
                }
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);

            EpilogueKDAFwdHVnew epilogueKDAFwdHVnew(resource);
            EpilogueKDAFwdHUpdate epilogueKDAFwdHUpdate(resource);
            uint32_t pongBaseEvent = 4;

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset final_state
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset h_update
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
            bool event0FromMte3[PING_PONG_STAGES] = {false, false};
            bool event2FromMte3[PING_PONG_STAGES] = {!(storeFinalState && std::is_same<ElementFinalState, float>::value),
                                                      !(storeFinalState && std::is_same<ElementFinalState, float>::value)};
            while (vecBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* V1:
                     * gmV = gmUInternal - gmVWorkspace
                     * g_buf = gmG[-1] - gmG
                     * g_buf = exp(g_buf)
                     * gmVWorkspace = g_buf * gmV
                     */
                    vecBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = vecBlockScheduler.GetStreamId(i);
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const KDAFwdHOffsets& vec1Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        AscendC::LocalTensor<ElementV> l1VUpdate = (i == 0) ? l1VUpdatePing : l1VUpdatePong;
                        bool tailVectorPath = vec1Offsets.blockTokens < 16;
                        if (tailVectorPath) {
                            Arch::CrossCoreWaitFlag(
                                vecBlockScheduler.cube1Done[streamId]);
                            ComputeTailVWorkspace(
                                vec1Offsets, EVENT_ID3 + (i == 0 ? 0 : pongBaseEvent));
                        }
                        bool waitWsFromMte3 = storeFinalState && std::is_same<ElementFinalState, float>::value &&
                                              event0FromMte3[streamId];
                        bool useDirectForTask = useDirectFp32Ub && !tailVectorPath;
                        uint64_t vUpdateBase = VUpdateBase(vec1Offsets);
                        uint64_t kDecayBase = HI_LO_C2
                            ? 2ULL * vec1Offsets.kDecayWorkOffset
                            : vec1Offsets.kDecayWorkOffset;
                        epilogueKDAFwdHVnew(
                            gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vUpdateBase], l1VUpdate,
                            gmG[vec1Offsets.gOffset], gmUInternal[vec1Offsets.uvOffset],
                            gmVWorkspace[vec1Offsets.vWorkOffset],
                            gmGk[vec1Offsets.gkOffset], gmK[vec1Offsets.wkOffset],
                            gmKResidual[vec1Offsets.uvOffset], gmKDecayWorkspace[kDecayBase],
                            vec1Offsets.blockTokens, kHeadDim, vec1Offsets.vBlockDim, vHeadDim,
                            vecBlockScheduler.cube1Done[streamId], vecBlockScheduler.vec1Done[streamId],
                            vec1Offsets.isInitialState, vec1Offsets.isFinalState, storeFinalState,
                            waitWsFromMte3, (i == 0), tailVectorPath, useDirectForTask,
                            DIRECT_UB_FREE_FLAG_BEGIN, DIRECT_UB_READY_FLAG_BEGIN
                        );
                        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                            event0FromMte3[streamId] = false;
                        }
                    }
                } else {
                    /* V2: h[i+1] += h_work if i < num_chunks - 1 else None */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = vecBlockScheduler.GetStreamId(i);
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const KDAFwdHOffsets& vec2Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        if (vecBlockScheduler.NeedProcessStage2(stream)) {
                            bool tailVectorPath = vec2Offsets.blockTokens < 16;
                            if (tailVectorPath) {
                                Arch::CrossCoreWaitFlag(
                                    vecBlockScheduler.cube2Done[streamId]);
                                ComputeTailHWorkspace(
                                    vec2Offsets, EVENT_ID3 + (i == 0 ? 0 : pongBaseEvent));
                            }
                            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                                // Update always writes the FP32 state through MTE3,
                                // including intermediate chunks.
                                event0FromMte3[streamId] = true;
                                event2FromMte3[streamId] = !vec2Offsets.isFinalState;
                            }
                            // step 4:  h[i+1] += h_work if i < num_chunks - 1 else None
                            epilogueKDAFwdHUpdate(
                                gmHInternal[vec2Offsets.hDstOffset], gmFinalState[vec2Offsets.finalStateOffset],
                                gmG[vec2Offsets.gOffset],
                                gmHInternal[vec2Offsets.hSrcOffset],
                                gmHWorkspace[vec2Offsets.hWorkOffset],
                                gmGk[vec2Offsets.gkOffset],
                                gmInitialState[vec2Offsets.initialStateOffset],
                                vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim, vHeadDim, vecBlockScheduler.cube2Done[streamId],
                                vec2Offsets.isInitialState, vec2Offsets.isFinalState, storeFinalState,
                                useInitialState, (i == 0), tailVectorPath,
                                useDirectFp32Ub && !tailVectorPath,
                                DIRECT_UB_FREE_FLAG_BEGIN, DIRECT_UB_READY_FLAG_BEGIN
                            );
                        } else {
                            if (!useDirectFp32Ub) {
                                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[streamId]);
                            }
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);
                    }
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
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset h_update
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
        if constexpr (FP32_H) {
            AscendC::SyncAll<false>();
            if ASCEND_IS_AIV {
                uint32_t subBlockNum = AscendC::GetSubBlockNum();
                uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
                uint32_t coreNum = AscendC::GetBlockNum();
                uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
                ExportPublicH(
                    coreIdx, coreNum, subBlockIdx, subBlockNum);
            }
            AscendC::SyncAll<false>();
        }
    }

};

}
