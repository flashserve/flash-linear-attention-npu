/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_w_u_fwd.h
 * \brief
 */


#ifndef RECOMPUTE_W_U_FWD_CUBE_H
#define RECOMPUTE_W_U_FWD_CUBE_H

#include "recompute_w_u_fwd_struct.h"

using GDN::RecomputeWUFwdTilingData;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
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
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
using namespace Catlass;
using namespace tla;
namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <class BlockMmadU_, class BlockMmadW_, bool kFlattenHeadTasks_ = false,
          bool kCoefficientGenerationTaskOrder_ = false>
class RecomputeWUFwdTla {
public:
    using BlockMmadU = BlockMmadU_;
    using BlockMmadW = BlockMmadW_;

    using ArchTag = typename BlockMmadU::ArchTag;
    using BdkL1TileShape = typename BlockMmadU::L1TileShape;
    using BdkbL1TileShape = typename BlockMmadU::L1TileShape;

    using ElementA = typename BlockMmadU::ElementA;
    using LayoutA = typename BlockMmadU::LayoutA;
    using ElementVb = typename BlockMmadU::ElementB;
    using LayoutVb = typename BlockMmadU::LayoutB;
    using ElementU = typename BlockMmadU::ElementC;
    using LayoutU = typename BlockMmadU::LayoutC;

    using ElementKbgExp = typename BlockMmadW::ElementB;
    using LayoutKbgExp = typename BlockMmadW::LayoutB;
    using ElementW = typename BlockMmadW::ElementC;
    using LayoutW = typename BlockMmadW::LayoutC;
    static constexpr bool kFlattenHeadTasks = kFlattenHeadTasks_;
    static constexpr bool kCoefficientGenerationTaskOrder = kCoefficientGenerationTaskOrder_;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr uint32_t GM_RING_DEPTH = 8;
    static_assert(GM_RING_DEPTH <= Arch::MAX_REVERSE_DEPTH,
                  "Ring window must keep every cross-core flag below the hardware credit limit.");
    Arch::CrossCoreFlag flagAivVbReady{SYNC_AIV_AIC_FLAG_3};
    Arch::CrossCoreFlag flagAivKbgExpReady{SYNC_AIV_AIC_FLAG_4};
    Arch::CrossCoreFlag flagAicSlotFree{SYNC_AIC_AIV_FLAG_5};
#else
    Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_AIC_AIV_FLAG_5, SYNC_AIV_AIC_FLAG_3};
#endif
    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrVb;
        LayoutVb layoutVb;
        GM_ADDR ptrU;
        LayoutU layoutU;
        GM_ADDR ptrKbgExp;
        LayoutKbgExp layoutKbgExp;
        GM_ADDR ptrW;
        LayoutW layoutW;
        GM_ADDR ptrCuSeqLens;
        GM_ADDR ptrChunkIndices;
        uint64_t chunkNum;
        uint64_t B = 1;
        uint64_t Hk = 1;
        uint64_t Hv = 1;
        uint64_t hvPerHk = 1;
        uint64_t T = 32768;
        uint64_t K = 128;
        uint64_t V = 128;
        uint64_t chunkSize = 64;

        // Methods
        CATLASS_DEVICE
        Params()
        {
        }

        CATLASS_DEVICE
        Params(GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrVb_, LayoutVb layoutVb_, GM_ADDR ptrU_,
               LayoutU layoutU_, GM_ADDR ptrKbgExp_, LayoutKbgExp layoutKbgExp_, GM_ADDR ptrW_, LayoutW layoutW_,
               GM_ADDR ptrCuSeqLens_, GM_ADDR ptrChunkIndices_, uint64_t chunkNum_, uint64_t B_,
               uint64_t Hk_, uint64_t Hv_, uint64_t hvPerHk_, uint64_t T_, uint64_t K_, uint64_t V_, uint64_t BT_)
            : ptrA(ptrA_), layoutA(layoutA_), ptrVb(ptrVb_), layoutVb(layoutVb_), ptrU(ptrU_),
              layoutU(layoutU_), ptrKbgExp(ptrKbgExp_), layoutKbgExp(layoutKbgExp_), ptrW(ptrW_), layoutW(layoutW_),
              ptrCuSeqLens(ptrCuSeqLens_), ptrChunkIndices(ptrChunkIndices_),
              chunkNum(chunkNum_), B(B_), Hk(Hk_), Hv(Hv_), hvPerHk(hvPerHk_), T(T_), K(K_), V(V_), chunkSize(BT_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    RecomputeWUFwdTla()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Arch::Resource<ArchTag> resource;
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t coreLoops = kFlattenHeadTasks ? params.chunkNum * params.Hv : params.chunkNum;
        uint32_t loopBegin = coreIdx;
        uint32_t loopEnd = coreLoops;
        uint32_t loopStep = coreNum;
        if constexpr (kCoefficientGenerationTaskOrder) {
            const uint32_t tasksPerCore = (coreLoops + coreNum - 1) / coreNum;
            loopBegin = coreIdx * tasksPerCore;
            loopEnd = (loopBegin + tasksPerCore) < coreLoops ? loopBegin + tasksPerCore : coreLoops;
            loopStep = 1;
        }

        // A5 computes U and W for one logical task back-to-back. Besides removing the
        // phase-wide barrier, this lets the L1-resident BlockMmad retain A while the
        // vector side publishes Vb and K*beta*exp(g) through independent GM rings.
        BlockMmadU blockMmad(resource);
        blockMmad.preSetFlags();
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementVb> gmVb;
        AscendC::GlobalTensor<ElementU> gmU;
        AscendC::GlobalTensor<ElementKbgExp> gmKbgExp;
        AscendC::GlobalTensor<ElementW> gmW;
        uint32_t bos = 0;
        uint32_t eos = 0;
        uint32_t taskIdx = 0;

        for (uint32_t loopIdx = loopBegin; loopIdx < loopEnd; loopIdx += loopStep) {
            uint32_t chunkIdx = 0;
            uint32_t hBegin = 0;
            uint32_t hEnd = 0;
            DecodeRecomputeTask<kFlattenHeadTasks, kCoefficientGenerationTaskOrder>(
                loopIdx, params.ptrCuSeqLens, params.Hv, params.T, params.chunkSize,
                params.chunkNum, chunkIdx, hBegin, hEnd);
            GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.Hv,
                           params.T, params.chunkSize, chunkIdx, bos, eos);
            const uint32_t curChunkSize = eos - bos;

            for (uint32_t h = hBegin; h < hEnd; ++h) {
                const uint32_t slotId = taskIdx % GM_RING_DEPTH;
                ++taskIdx;
                const uint64_t ringTask = static_cast<uint64_t>(coreIdx) * GM_RING_DEPTH + slotId;
                gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA +
                                    (h * params.T + bos) * params.chunkSize);
                auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});

                Arch::CrossCoreWaitFlag(flagAivVbReady);
                const uint32_t tileN = tla::get<1>(BdkL1TileShape{});
                for (uint32_t nOffset = 0; nOffset < params.V; nOffset += tileN) {
                    const uint32_t curN =
                        (nOffset + tileN > params.V) ? (params.V - nOffset) : tileN;
                    GemmCoord actualBlockShape{curChunkSize, curN, curChunkSize};
                    gmVb.SetGlobalBuffer((__gm__ ElementVb *)params.ptrVb +
                                         ringTask * params.chunkSize * params.V + nOffset);
                    gmU.SetGlobalBuffer((__gm__ ElementU *)params.ptrU +
                                        (h * params.T + bos) * params.V + nOffset);
                    auto tensorVb = tla::MakeTensor(gmVb, params.layoutVb, Arch::PositionGM{});
                    auto tensorU = tla::MakeTensor(gmU, params.layoutU, Arch::PositionGM{});
                    auto tensorBlockA = GetTile(
                        tensorA, tla::MakeCoord(0, 0),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockVb = GetTile(
                        tensorVb, tla::MakeCoord(0, 0),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockU = GetTile(
                        tensorU, tla::MakeCoord(0, 0),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockVb, tensorBlockU, actualBlockShape);
                }

                Arch::CrossCoreWaitFlag(flagAivKbgExpReady);
                GemmCoord actualBlockShape{
                    curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                gmKbgExp.SetGlobalBuffer((__gm__ ElementKbgExp *)params.ptrKbgExp +
                                         ringTask * params.chunkSize * params.K);
                gmW.SetGlobalBuffer((__gm__ ElementW *)params.ptrW +
                                    (h * params.T + bos) * params.K);
                auto tensorKbgExp =
                    tla::MakeTensor(gmKbgExp, params.layoutKbgExp, Arch::PositionGM{});
                auto tensorW = tla::MakeTensor(gmW, params.layoutW, Arch::PositionGM{});
                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(0, 0),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockKbgExp = GetTile(
                    tensorKbgExp, tla::MakeCoord(0, 0),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockW = GetTile(
                    tensorW, tla::MakeCoord(0, 0),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                blockMmad(tensorBlockA, tensorBlockKbgExp, tensorBlockW, actualBlockShape);
                // Dependency chain inside BlockMmad is GM->L1 (MTE2), MTE2_MTE1 wait,
                // then L1->L0 (MTE1). Issuing the free notification on PIPE_MTE1 here
                // orders it after that final L1 read, which transitively proves the GM
                // source is no longer in use. Mode 0x2 broadcasts one credit to each
                // paired AIV subcore, matching their independent FIFO waits.
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE1>(flagAicSlotFree);
            }
        }
        blockMmad.finalWaitFlags();
#else
        Arch::Resource<ArchTag> resource;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreLoops = kFlattenHeadTasks ? params.chunkNum * params.Hv : params.chunkNum;
        uint32_t bos = 0;
        uint32_t eos = 0;
        { //处理U     V->C
            BlockMmadU BlockMmadU(resource);
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementVb> gmVb;
            AscendC::GlobalTensor<ElementU> gmU;
            uint32_t loopBegin = coreIdx;
            uint32_t loopEnd = coreLoops;
            uint32_t loopStep = AscendC::GetBlockNum();
            if constexpr (kCoefficientGenerationTaskOrder) {
                // Match the contiguous coefficient-generation producer range in the fused core.
                const uint32_t tasksPerCore =
                    (coreLoops + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
                loopBegin = coreIdx * tasksPerCore;
                loopEnd = (loopBegin + tasksPerCore) < coreLoops ? loopBegin + tasksPerCore : coreLoops;
                loopStep = 1;
            }
            for (uint32_t loopIdx = loopBegin; loopIdx < loopEnd; loopIdx += loopStep) {
                uint32_t chunkIdx = 0;
                uint32_t hBegin = 0;
                uint32_t hEnd = 0;
                DecodeRecomputeTask<kFlattenHeadTasks, kCoefficientGenerationTaskOrder>(
                    loopIdx, params.ptrCuSeqLens, params.Hv, params.T, params.chunkSize,
                    params.chunkNum, chunkIdx, hBegin, hEnd);
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.Hv, params.T,
                               params.chunkSize, chunkIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                for (uint32_t h = hBegin; h < hEnd; ++h) {
                    // Represent the full gm
                    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA + (h * params.T + bos) * params.chunkSize);

                    // Represent the full tensors
                    auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
                    Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(flagAivFinishStore);
                    // N 维步进取自 Catlass tile 的 N 维：
                    // key=1 tileN=128 -> V=128 单次；key=2 tileN=256 -> V=256 单次
                    uint32_t tileN = tla::get<1>(BdkL1TileShape{});
                    for (uint32_t nOffset = 0; nOffset < params.V; nOffset += tileN) {
                        uint32_t curN = (nOffset + tileN > params.V) ? (params.V - nOffset) : tileN;
                        GemmCoord actualBlockShape{curChunkSize, curN, curChunkSize};
                        gmVb.SetGlobalBuffer((__gm__ ElementVb *)params.ptrVb + (h * params.T + bos) * params.V + nOffset);
                        gmU.SetGlobalBuffer((__gm__ ElementU *)params.ptrU + (h * params.T + bos) * params.V + nOffset);

                        auto tensorVb = tla::MakeTensor(gmVb, params.layoutVb, Arch::PositionGM{});
                        auto tensorU = tla::MakeTensor(gmU, params.layoutU, Arch::PositionGM{});
                        // Make tiled views
                        auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                                                     tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockVb = GetTile(tensorVb, tla::MakeCoord(0, 0),
                                                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        auto tensorBlockU = GetTile(tensorU, tla::MakeCoord(0, 0),
                                                     tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                        // Compute block-scoped matrix multiply-add
                        BlockMmadU(tensorBlockA, tensorBlockVb, tensorBlockU, actualBlockShape);
                    }
                }
            }
        }
        AscendC::SyncAll<false>();
        { //处理第二部分 AT@K -> DKB
            BlockMmadW BlockMmadW(resource);
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementKbgExp> gmKbgExp;
            AscendC::GlobalTensor<ElementW> gmW;
            uint32_t loopBegin = coreIdx;
            uint32_t loopEnd = coreLoops;
            uint32_t loopStep = AscendC::GetBlockNum();
            if constexpr (kCoefficientGenerationTaskOrder) {
                const uint32_t tasksPerCore =
                    (coreLoops + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
                loopBegin = coreIdx * tasksPerCore;
                loopEnd = (loopBegin + tasksPerCore) < coreLoops ? loopBegin + tasksPerCore : coreLoops;
                loopStep = 1;
            }
            for (uint32_t loopIdx = loopBegin; loopIdx < loopEnd; loopIdx += loopStep) {
                uint32_t chunkIdx = 0;
                uint32_t hBegin = 0;
                uint32_t hEnd = 0;
                DecodeRecomputeTask<kFlattenHeadTasks, kCoefficientGenerationTaskOrder>(
                    loopIdx, params.ptrCuSeqLens, params.Hv, params.T, params.chunkSize,
                    params.chunkNum, chunkIdx, hBegin, hEnd);
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.Hv, params.T,
                               params.chunkSize, chunkIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                for (uint32_t h = hBegin; h < hEnd; ++h) {
                    // Represent the full gm
                    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA + (h * params.T + bos) * params.chunkSize);
                    gmKbgExp.SetGlobalBuffer((__gm__ ElementKbgExp *)params.ptrKbgExp +
                                             (h * params.T + bos) * params.K);
                    gmW.SetGlobalBuffer((__gm__ ElementW *)params.ptrW + (h * params.T + bos) * params.K);

                    // Represent the full tensors
                    auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
                    auto tensorKbgExp = tla::MakeTensor(gmKbgExp, params.layoutKbgExp, Arch::PositionGM{});
                    auto tensorW = tla::MakeTensor(gmW, params.layoutW, Arch::PositionGM{});

                    Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(flagAivFinishStore);
                    // Make tiled views
                    auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockKbgExp = GetTile(tensorKbgExp, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    BlockMmadW(tensorBlockA, tensorBlockKbgExp, tensorBlockW, actualBlockShape);
                }
            }
        }
#endif
    }
};
} // namespace Catlass::Gemm::Kernel

template <class... Dims>
using GemmCubeTileShape = tla::Shape<Dims...>;

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape,
          bool kFlattenHeadTasks = false, bool kCoefficientGenerationTaskOrder = false>
class RecomputeWUFwdProcess {
public:
    /** @brief constructor */
    __aicore__ inline RecomputeWUFwdProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
                                                        GM_ADDR workspace_);

    __aicore__ inline void Process();

    __aicore__ inline void Init(const RecomputeWUFwdTilingData &tiling);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t Hv = 1;
    uint64_t Hk = 1;
    uint64_t hvPerHk = 1;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR workspace;
};

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape,
          bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ inline RecomputeWUFwdProcess<kType, betaType, L1TileShape, L0TileShape,
                                        kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::RecomputeWUFwdProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_){};

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape,
          bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ void inline RecomputeWUFwdProcess<kType, betaType, L1TileShape, L0TileShape,
                                             kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::Init(
    const RecomputeWUFwdTilingData &tiling)
{
    B = tiling.B;
    T = tiling.T;
    Hv = tiling.Hv;
    Hk = tiling.Hk;
    hvPerHk = tiling.hvPerHk;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    return;
}

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape,
          bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ void inline RecomputeWUFwdProcess<kType, betaType, L1TileShape, L0TileShape,
                                             kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::Process()
{
    //输入
    using LayoutTagA = layout::RowMajor;
    using LayoutTagKbgExp = layout::RowMajor;
    using LayoutTagVb = layout::RowMajor;
    using LayoutTagW = layout::RowMajor;
    using LayoutTagU = layout::RowMajor;

    //输入
    LayoutTagA tagA = LayoutTagA::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagKbgExp tagKbgExp = LayoutTagKbgExp::MakeLayout<kType>(chunkSize, K);
    LayoutTagVb tagVb = LayoutTagVb::MakeLayout<kType>(chunkSize, V);

    //输出
    LayoutTagW tagW = LayoutTagW::MakeLayout<kType>(chunkSize, K);
    LayoutTagU tagU = LayoutTagU::MakeLayout<kType>(chunkSize, V);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Arch::Ascend950;
    // UnitFlag keeps L0C single-buffered; L1-resident mode lets the back-to-back
    // U/W calls reuse A while retaining the multi-stage A5 load/compute pipeline.
    using DispatchPolicy = Gemm::MmadPingpongTlaMulti<
        ArchTag, true, false, 1, true, 1, 2, 2, 2>;
#else
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
#endif

    //计算U
    using TileCopyU =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagA, kType, LayoutTagVb, kType, LayoutTagU>;
    using BlockMmadU =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyU>;
    //计算W
    using TileCopyW =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagA, kType, LayoutTagKbgExp, kType, LayoutTagW>;
    using BlockMmadW =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyW>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static_assert(BlockMmadU::ENABLE_L1_RESIDENT, "A5 recompute requires L1-resident A.");
    static_assert(BlockMmadU::L1A_STAGES == 1,
                  "A5 recompute requires one L1A stage so U and W hit the same resident slot.");
    static_assert(BlockMmadU::L1B_STAGES == 2 && BlockMmadU::L0A_STAGES == 2 &&
                      BlockMmadU::L0B_STAGES == 2,
                  "A5 recompute pipeline stage counts must remain explicit and audited.");
#endif

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutVb = MakeLayoutFromTag(tagVb);
    auto layoutU = MakeLayoutFromTag(tagU);

    auto layoutKbgExp = MakeLayoutFromTag(tagKbgExp);
    auto layoutW = MakeLayoutFromTag(tagW);

    // kernel level
    using MatmulKernel = Gemm::Kernel::RecomputeWUFwdTla<
        BlockMmadU, BlockMmadW, kFlattenHeadTasks, kCoefficientGenerationTaskOrder>;

    MatmulKernel kernel;

    GM_ADDR vb = workspace;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    GM_ADDR kbgExp = workspace + static_cast<uint64_t>(AscendC::GetBlockNum()) *
        MatmulKernel::GM_RING_DEPTH * chunkSize * V * sizeof(kType);
#else
    GM_ADDR kbgExp = workspace;
#endif
    typename MatmulKernel::Params param{
        A, layoutA, vb, layoutVb, u,        layoutU,
        kbgExp, layoutKbgExp, w,        layoutW,
        cu_seqlens, chunk_indices, chunkNum, B,
        Hk, Hv, hvPerHk, T, K, V, chunkSize};
    kernel(param);
}


#endif // RECOMPUTE_W_U_FWD_CUBE_H
