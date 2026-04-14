/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd.h
 * \brief
 */

#include "recompute_wu_fwd_common.h"
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
#ifndef RECOMPUTE_WU_FWD_CUBE_H
#define RECOMPUTE_WU_FWD_CUBE_H


using namespace Catlass;

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <class BlockMmadU_, class BlockMmadW_>
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
    Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_AIC_AIV_FLAG_5, SYNC_AIV_AIC_FLAG_3};
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
        uint64_t T = 32768;
        uint64_t H = 32;
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
               LayoutW layoutW_, GM_ADDR ptrKbgExp_, LayoutKbgExp layoutKbgExp_, GM_ADDR ptrW_, LayoutU layoutU_,
               GM_ADDR ptrCuSeqLens_, GM_ADDR ptrChunkIndices_, uint64_t chunkNum_, uint64_t B_,
               uint64_t T_, uint64_t H_, uint64_t K_, uint64_t V_, uint64_t BT_)
            : ptrA(ptrA_), layoutA(layoutA_), ptrVb(ptrVb_), layoutVb(layoutVb_), ptrU(ptrU_),
              layoutW(layoutW_), ptrKbgExp(ptrKbgExp_), layoutKbgExp(layoutKbgExp_), ptrW(ptrW_), layoutU(layoutU_),
              ptrCuSeqLens(ptrCuSeqLens_), ptrChunkIndices(ptrChunkIndices_),
              chunkNum(chunkNum_), B(B_), T(T_), H(H_), K(K_), V(V_), chunkSize(BT_)
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
        Arch::Resource<ArchTag> resource;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreLoops = params.chunkNum;
        uint32_t bos = 0;
        uint32_t eos = 0;
        { //处理U     V->C
            BlockMmadU BlockMmadU(resource);
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementVb> gmVb;
            AscendC::GlobalTensor<ElementU> gmU;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.V), curChunkSize};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA + (h * params.T + bos) * params.chunkSize);
                    gmVb.SetGlobalBuffer((__gm__ ElementVb *)params.ptrVb + (h * params.T + bos) * params.V);
                    gmU.SetGlobalBuffer((__gm__ ElementU *)params.ptrU + (h * params.T + bos) * params.V);

                    // Represent the full tensors
                    auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
                    auto tensorVb = tla::MakeTensor(gmVb, params.layoutVb, Arch::PositionGM{});
                    auto tensorU = tla::MakeTensor(gmU, params.layoutU, Arch::PositionGM{});
                    Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(flagAivFinishStore);
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
        AscendC::SyncAll<false>();
        { //处理第二部分 AT@K -> DKB
            BlockMmadW BlockMmadW(resource);
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementKbgExp> gmKbgExp;
            AscendC::GlobalTensor<ElementW> gmW;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA + (h * params.T + bos) * params.chunkSize);
                    gmKbgExp.SetGlobalBuffer((__gm__ ElementKbgExp *)params.ptrKbgExp + (h * params.T + bos) * params.K);
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

    }
};
} // namespace Catlass::Gemm::Kernel

template <typename kType, typename betaType>
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
    uint64_t H = 0;
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

template <typename kType, typename betaType>
__aicore__ inline RecomputeWUFwdProcess<kType, betaType>::RecomputeWUFwdProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_){};

template <typename kType, typename betaType>
__aicore__ void inline RecomputeWUFwdProcess<kType, betaType>::Init(const RecomputeWUFwdTilingData &tiling)
{
    B = tiling.B;
    T = tiling.T;
    H = tiling.H;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline RecomputeWUFwdProcess<kType, betaType>::Process()
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

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = Shape<_128, _128, _256>;
    using L0TileShape = Shape<_128, _128, _128>;

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

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutVb = MakeLayoutFromTag(tagVb);
    auto layoutU = MakeLayoutFromTag(tagU);

    auto layoutKbgExp = MakeLayoutFromTag(tagKbgExp);
    auto layoutW = MakeLayoutFromTag(tagW);
    
    // kernel level
    using MatmulKernel =
        Gemm::Kernel::RecomputeWUFwdTla<BlockMmadU, BlockMmadW>;

    MatmulKernel kernel;

    GM_ADDR vb = workspace;
    GM_ADDR kbgExp = workspace;
    typename MatmulKernel::Params param{
        A, layoutA, vb, layoutVb, u,        layoutU,  
        kbgExp, layoutKbgExp, w,        layoutW,
        cu_seqlens, chunk_indices, chunkNum, B,
        T,         H,           K,  V,        chunkSize};
    kernel(param);
}


#endif // RECOMPUTE_WU_FWD_CUBE_H
