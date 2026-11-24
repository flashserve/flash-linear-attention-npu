/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef RECOMPUTE_W_U_FWD_CUBE_L1RESIDENT_H
#define RECOMPUTE_W_U_FWD_CUBE_L1RESIDENT_H

#include "../recompute_w_u_fwd_struct.h"
#include "../recompute_w_u_fwd_common.h"

using GDN::RecomputeWUFwdTilingData;

#define CATLASS_ARCH 9201
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/arch/cross_core_sync.hpp"
using namespace Catlass;
using namespace tla;
namespace Catlass::Gemm::Kernel {

template <class BlockMmadU_, class BlockMmadW_>
class RecomputeWUFwdTlaL1Resident {
public:
    using BlockMmadU = BlockMmadU_;
    using BlockMmadW = BlockMmadW_;

    using ArchTag = typename BlockMmadU::ArchTag;
    using BdkL1TileShape = typename BlockMmadU::L1TileShape;

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
    static constexpr uint32_t GM_RING_DEPTH = GDN::RECOMPUTE_W_U_FWD_GM_RING_DEPTH;
    Arch::CrossCoreFlagWithReverse<GM_RING_DEPTH> flagAivVbReady{
        SYNC_AIC_AIV_FLAG_5, SYNC_AIV_AIC_FLAG_3};
    Arch::CrossCoreFlagWithReverse<GM_RING_DEPTH> flagAivKbgExpReady{
        SYNC_AIC_AIV_FLAG_6, SYNC_AIV_AIC_FLAG_4};
    Arch::CrossCoreFlagWithReverse<GM_RING_DEPTH> flagAicRingSlotFree{
        SYNC_AIC_AIV_RING_SLOT_FREE_FLAG, SYNC_AIV_AIC_RING_SLOT_FREE_REVERSE_FLAG};
    struct Params {
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

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrVb_, LayoutVb layoutVb_, GM_ADDR ptrU_,
               LayoutU layoutU_, GM_ADDR ptrKbgExp_, LayoutKbgExp layoutKbgExp_, GM_ADDR ptrW_, LayoutW layoutW_,
               GM_ADDR ptrCuSeqLens_, GM_ADDR ptrChunkIndices_, uint64_t chunkNum_, uint64_t B_,
               uint64_t Hk_, uint64_t Hv_, uint64_t hvPerHk_, uint64_t T_, uint64_t K_, uint64_t V_, uint64_t BT_)
            : ptrA(ptrA_), layoutA(layoutA_), ptrVb(ptrVb_), layoutVb(layoutVb_), ptrU(ptrU_),
              layoutU(layoutU_), ptrKbgExp(ptrKbgExp_), layoutKbgExp(layoutKbgExp_), ptrW(ptrW_), layoutW(layoutW_),
              ptrCuSeqLens(ptrCuSeqLens_), ptrChunkIndices(ptrChunkIndices_),
              chunkNum(chunkNum_), B(B_), Hk(Hk_), Hv(Hv_), hvPerHk(hvPerHk_), T(T_), K(K_), V(V_), chunkSize(BT_)
        {}
    };

    CATLASS_DEVICE
    RecomputeWUFwdTlaL1Resident() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Arch::Resource<ArchTag> resource;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreLoops = params.chunkNum;
        uint32_t bos = 0;
        uint32_t eos = 0;
        uint32_t taskIdx = 0;

        BlockMmadU blockMmad(resource);
        blockMmad.preSetFlags();
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementVb> gmVb;
        AscendC::GlobalTensor<ElementU> gmU;
        AscendC::GlobalTensor<ElementKbgExp> gmKbgExp;
        AscendC::GlobalTensor<ElementW> gmW;

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.Hv, params.T,
                           params.chunkSize, loopIdx, bos, eos);
            uint32_t curChunkSize = eos - bos;

            for (int h = 0; h < params.Hv; h++) {
                uint32_t slotId = taskIdx % GM_RING_DEPTH;
                ++taskIdx;
                uint64_t ringTask = static_cast<uint64_t>(coreIdx) * GM_RING_DEPTH + slotId;

                gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA + (h * params.T + bos) * params.chunkSize);
                auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});

                WaitVbReady();

                {
                    uint32_t tileN = tla::get<1>(BdkL1TileShape{});
                    for (uint32_t nOffset = 0; nOffset < params.V; nOffset += tileN) {
                        uint32_t curN = (nOffset + tileN > params.V) ? (params.V - nOffset) : tileN;
                        GemmCoord actualBlockShape{curChunkSize, curN, curChunkSize};
                        gmVb.SetGlobalBuffer((__gm__ ElementVb *)params.ptrVb +
                                             ringTask * params.chunkSize * params.V + nOffset);
                        gmU.SetGlobalBuffer((__gm__ ElementU *)params.ptrU + (h * params.T + bos) * params.V + nOffset);

                        auto tensorVb = tla::MakeTensor(gmVb, params.layoutVb, Arch::PositionGM{});
                        auto tensorU = tla::MakeTensor(gmU, params.layoutU, Arch::PositionGM{});
                        auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                                                     tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockVb = GetTile(tensorVb, tla::MakeCoord(0, 0),
                                                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        auto tensorBlockU = GetTile(tensorU, tla::MakeCoord(0, 0),
                                                     tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                        blockMmad(tensorBlockA, tensorBlockVb, tensorBlockU, actualBlockShape);
                    }
                }

                WaitKbgExpReady();
                {
                    GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                    gmKbgExp.SetGlobalBuffer((__gm__ ElementKbgExp *)params.ptrKbgExp +
                                             ringTask * params.chunkSize * params.K);
                    gmW.SetGlobalBuffer((__gm__ ElementW *)params.ptrW + (h * params.T + bos) * params.K);

                    auto tensorKbgExp = tla::MakeTensor(gmKbgExp, params.layoutKbgExp, Arch::PositionGM{});
                    auto tensorW = tla::MakeTensor(gmW, params.layoutW, Arch::PositionGM{});

                    auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockKbgExp = GetTile(tensorKbgExp, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockKbgExp, tensorBlockW, actualBlockShape);
                }

                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE2>(flagAicRingSlotFree.id);
                if (++flagAicRingSlotFree.count >= GM_RING_DEPTH) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(flagAicRingSlotFree.reverseId);
                    flagAicRingSlotFree.count = 0;
                }

                blockMmad.RestoreStatus();
            }
        }
        blockMmad.finalWaitFlags();
    }

    __aicore__ inline void WaitVbReady()
    {

        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(flagAivVbReady.id);
        if (++flagAivVbReady.count >= GM_RING_DEPTH) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(flagAivVbReady.reverseId);
            flagAivVbReady.count = 0;
        }
    }

    __aicore__ inline void WaitKbgExpReady()
    {

        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(flagAivKbgExpReady.id);
        if (++flagAivKbgExpReady.count >= GM_RING_DEPTH) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(flagAivKbgExpReady.reverseId);
            flagAivKbgExpReady.count = 0;
        }
    }
};
}

template <class... Dims>
using GemmCubeTileShape = tla::Shape<Dims...>;

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape>
class RecomputeWUFwdProcessL1Resident {
public:
    __aicore__ inline RecomputeWUFwdProcessL1Resident(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
                                                        GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_,
                                                        GM_ADDR u_, GM_ADDR workspace_);

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

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape>
__aicore__ inline RecomputeWUFwdProcessL1Resident<kType, betaType, L1TileShape, L0TileShape>::RecomputeWUFwdProcessL1Resident(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
    GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_, GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_) {}

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape>
__aicore__ void inline RecomputeWUFwdProcessL1Resident<kType, betaType, L1TileShape, L0TileShape>::Init(
    const RecomputeWUFwdTilingData &tiling)
{
    B = tiling.B; T = tiling.T; Hv = tiling.Hv; Hk = tiling.Hk; hvPerHk = tiling.hvPerHk;
    K = tiling.K; V = tiling.V; chunkSize = tiling.chunkSize; chunkNum = tiling.chunkNum;
}

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape>
__aicore__ void inline RecomputeWUFwdProcessL1Resident<kType, betaType, L1TileShape, L0TileShape>::Process()
{
    using LayoutTagA = layout::RowMajor;
    using LayoutTagKbgExp = layout::RowMajor;
    using LayoutTagVb = layout::RowMajor;
    using LayoutTagW = layout::RowMajor;
    using LayoutTagU = layout::RowMajor;

    LayoutTagA tagA = LayoutTagA::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagKbgExp tagKbgExp = LayoutTagKbgExp::MakeLayout<kType>(chunkSize, K);
    LayoutTagVb tagVb = LayoutTagVb::MakeLayout<kType>(chunkSize, V);
    LayoutTagW tagW = LayoutTagW::MakeLayout<kType>(chunkSize, K);
    LayoutTagU tagU = LayoutTagU::MakeLayout<kType>(chunkSize, V);

    using ArchTag = Arch::Ascend960DT;

    using DispatchPolicy = Gemm::MmadPingpongTlaMulti<ArchTag, true, false, 1, true>;

    using TileCopyU =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagA, kType, LayoutTagVb, kType, LayoutTagU>;
    using BlockMmadU =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyU>;

    using TileCopyW =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagA, kType, LayoutTagKbgExp, kType, LayoutTagW>;
    using BlockMmadW =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyW>;

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutVb = MakeLayoutFromTag(tagVb);
    auto layoutU = MakeLayoutFromTag(tagU);
    auto layoutKbgExp = MakeLayoutFromTag(tagKbgExp);
    auto layoutW = MakeLayoutFromTag(tagW);

    using MatmulKernel = Gemm::Kernel::RecomputeWUFwdTlaL1Resident<BlockMmadU, BlockMmadW>;
    MatmulKernel kernel;

    GM_ADDR vb = workspace;
    GM_ADDR kbgExp = workspace + static_cast<uint64_t>(AscendC::GetBlockNum()) *
        MatmulKernel::GM_RING_DEPTH * chunkSize * V * sizeof(kType);
    typename MatmulKernel::Params param{
        A, layoutA, vb, layoutVb, u, layoutU,
        kbgExp, layoutKbgExp, w, layoutW,
        cu_seqlens, chunk_indices, chunkNum, B,
        Hk, Hv, hvPerHk, T, K, V, chunkSize};
    kernel(param);
}

#endif
