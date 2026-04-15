/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prepare_wy_repr_bwd_full.h
 * \brief
 */

#include "prepare_wy_repr_bwd_full_common.h"
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

#ifndef PREPARE_WY_REPR_BWD_FULL_CUBE_H
#define PREPARE_WY_REPR_BWD_FULL_CUBE_H


using namespace Catlass;

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <class BlockMmadBdk_, class BlockMmadBdkb_, class BlockMmadBdkbg_, class BlockMmadBdvb_, class BlockMmadBkkT_>
class PrepareWyReprBwdFullTla {
public:
    using BlockMmadBdk = BlockMmadBdk_;
    using BlockMmadBdkb = BlockMmadBdkb_;
    using BlockMmadBdkbg = BlockMmadBdkbg_;
    using BlockMmadBkkT = BlockMmadBkkT_;
    using BlockMmadBdvb = BlockMmadBdvb_;
    using ArchTag = typename BlockMmadBdkb::ArchTag;
    using BdkL1TileShape = typename BlockMmadBdk::L1TileShape;
    using BdkbL1TileShape = typename BlockMmadBdkb::L1TileShape;
    using ElementDA = typename BlockMmadBdk::ElementA;
    using LayoutDA = typename BlockMmadBdk::LayoutA;
    using ElementKbeta = typename BlockMmadBdk::ElementB;
    using LayoutKbeta = typename BlockMmadBdk::LayoutB;
    using ElementDk = typename BlockMmadBdk::ElementC;
    using LayoutDk = typename BlockMmadBdk::LayoutC;

    using ElementDAT = typename BlockMmadBdkb::ElementA;
    using LayoutDAT = typename BlockMmadBdkb::LayoutA;
    using ElementK = typename BlockMmadBdkb::ElementB;
    using LayoutK = typename BlockMmadBdkb::LayoutB;
    using ElementDkb = typename BlockMmadBdkb::ElementC;
    using LayoutDkb = typename BlockMmadBdkb::LayoutC;

    using ElementAT = typename BlockMmadBdkbg::ElementA;
    using LayoutAT = typename BlockMmadBdkbg::LayoutA;
    using ElementDw = typename BlockMmadBdkbg::ElementB;
    using LayoutDw = typename BlockMmadBdkbg::LayoutB;
    using ElementDkbg = typename BlockMmadBdkbg::ElementC;
    using LayoutDkbg = typename BlockMmadBdkbg::LayoutC;

    using ElementDu = typename BlockMmadBdvb::ElementB;
    using LayoutDu = typename BlockMmadBdvb::LayoutB;
    using ElementDvb = typename BlockMmadBdvb::ElementC;
    using LayoutDvb = typename BlockMmadBdvb::LayoutC;

    using ElementKT = typename BlockMmadBkkT::ElementB;
    using LayoutKT = typename BlockMmadBkkT::LayoutB;
    using ElementKKT = typename BlockMmadBkkT::ElementC;
    using LayoutKKT = typename BlockMmadBkkT::LayoutC;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{SYNC_FLAG_2, SYNC_FLAG_3};
    Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_FLAG_4, SYNC_FLAG_5};
    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR ptrKbeta;
        LayoutKbeta layoutKbeta;
        GM_ADDR ptrDA;
        LayoutDA layoutDA;
        GM_ADDR ptrDk;
        LayoutDk layoutDk;
        GM_ADDR ptrDAT;
        LayoutDAT layoutDAT;
        GM_ADDR ptrK;
        LayoutK layoutK;
        GM_ADDR ptrDkb;
        LayoutDkb layoutDkb;
        GM_ADDR ptrAT;
        LayoutAT layoutAT;
        GM_ADDR ptrDw;
        LayoutDw layoutDw;
        GM_ADDR ptrDkbg;
        LayoutDw layoutDkbg;
        GM_ADDR ptrDu;
        LayoutDu layoutDu;
        GM_ADDR ptrDvb;
        LayoutDvb layoutDvb;
        GM_ADDR ptrKT;
        LayoutKT layoutKT;
        GM_ADDR ptrKKT;
        LayoutKKT layoutKKT;
        GM_ADDR ptrCuSeqLens;
        GM_ADDR ptrChunkIndices;
        uint64_t chunkNum;
        uint64_t B = 1;
        uint64_t T = 32768;
        uint64_t H = 32;
        uint64_t K = 128;
        uint64_t V = 128;
        uint64_t chunkSize = 64;
        uint64_t stage = 2;

        // Methods
        CATLASS_DEVICE
        Params()
        {
        }

        CATLASS_DEVICE
        Params(GM_ADDR ptrptrKbeta_, LayoutKbeta layoutKbeta_, GM_ADDR ptrDA_, LayoutDA layoutDA_, GM_ADDR ptrDk_,
               LayoutKbeta layoutDk_, GM_ADDR ptrDAT_, LayoutDAT layoutDAT_, GM_ADDR ptrK_, LayoutK layoutK_,
               GM_ADDR ptrDkb_, LayoutDkb layoutDkb_, GM_ADDR ptrAT_, LayoutAT layoutAT_, GM_ADDR ptrDw_,
               LayoutDw layoutDw_, GM_ADDR ptrDkbg_, LayoutDkbg layoutDkbg_, GM_ADDR ptrDu_, LayoutDkbg layoutDu_,
               GM_ADDR ptrDvb_, LayoutDkbg layoutDvb_, GM_ADDR ptrKT_, LayoutKT layoutKT_, GM_ADDR ptrKKT_,
               LayoutKKT layoutKKT_, GM_ADDR ptrCuSeqLens_, GM_ADDR ptrChunkIndices_, uint64_t chunkNum_, uint64_t B_,
               uint64_t T_, uint64_t H_, uint64_t K_, uint64_t V_, uint64_t BT_, uint64_t stage_)
            : ptrKbeta(ptrptrKbeta_), layoutKbeta(layoutKbeta_), ptrDA(ptrDA_), layoutDA(layoutDA_), ptrDk(ptrDk_),
              layoutDk(layoutDk_), ptrDAT(ptrDAT_), layoutDAT(layoutDAT_), ptrK(ptrK_), layoutK(layoutK_),
              ptrDkb(ptrDkb_), layoutDkb(layoutDkb_), ptrAT(ptrAT_), layoutAT(layoutAT_), ptrDw(ptrDw_),
              layoutDw(layoutDw_), ptrDkbg(ptrDkbg_), layoutDkbg(layoutDkbg_), ptrDu(ptrDu_), layoutDu(layoutDu_),
              ptrDvb(ptrDvb_), layoutDvb(layoutDvb_), ptrKT(ptrKT_), layoutKT(layoutKT_), ptrKKT(ptrKKT_),
              layoutKKT(layoutKKT_), ptrCuSeqLens(ptrCuSeqLens_), ptrChunkIndices(ptrChunkIndices_),
              chunkNum(chunkNum_), B(B_), T(T_), H(H_), K(K_), V(V_), chunkSize(BT_), stage(stage_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    PrepareWyReprBwdFullTla()
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
        { //处理第一部分cube DA @ Kbeta     V->C
            BlockMmadBdk blockMmadBdk(resource);
            AscendC::GlobalTensor<ElementDA> gmDA;
            AscendC::GlobalTensor<ElementKbeta> gmKbeta;
            AscendC::GlobalTensor<ElementDk> gmDk;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    gmDA.SetGlobalBuffer((__gm__ ElementDA *)params.ptrDA + (h * params.T + bos) * params.chunkSize);
                    gmKbeta.SetGlobalBuffer((__gm__ ElementKbeta *)params.ptrKbeta + (h * params.T + bos) * params.K);
                    gmDk.SetGlobalBuffer((__gm__ ElementDk *)params.ptrDk + (h * params.T + bos) * params.K);

                    // Represent the full tensors
                    auto tensorDA = tla::MakeTensor(gmDA, params.layoutDA, Arch::PositionGM{});
                    auto tensorKbeta = tla::MakeTensor(gmKbeta, params.layoutKbeta, Arch::PositionGM{});
                    auto tensorDk = tla::MakeTensor(gmDk, params.layoutDk, Arch::PositionGM{});

                    Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(flagAivFinishStore);
                    // Make tiled views
                    auto tensorBlockDA = GetTile(tensorDA, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockKbeta = GetTile(tensorKbeta, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDk = GetTile(tensorDk, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdk(tensorBlockDA, tensorBlockKbeta, tensorBlockDk, actualBlockShape);
                }
            }
        }
        AscendC::SyncAll<false>();
        { //处理第二部分 DAT@K -> DKB
            BlockMmadBdkb blockMmadBdkb(resource);
            AscendC::GlobalTensor<ElementDAT> gmDAT;
            AscendC::GlobalTensor<ElementK> gmK;
            AscendC::GlobalTensor<ElementDkb> gmDkb;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    gmDAT.SetGlobalBuffer((__gm__ ElementDAT *)params.ptrDAT + (h * params.T + bos) * params.chunkSize);
                    gmK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK + (h * params.T + bos) * params.K);
                    gmDkb.SetGlobalBuffer((__gm__ ElementDkb *)params.ptrDkb + (h * params.T + bos) * params.K);

                    // Represent the full tensors
                    auto tensorDAT = tla::MakeTensor(gmDAT, params.layoutDAT, Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
                    auto tensorDkb = tla::MakeTensor(gmDkb, params.layoutDkb, Arch::PositionGM{});

                    // Make tiled views
                    auto tensorBlockDAT = GetTile(tensorDAT, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDkb = GetTile(tensorDkb, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdkb(tensorBlockDAT, tensorBlockK, tensorBlockDkb, actualBlockShape);
                    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                }
            }
        }
        AscendC::SyncAll<false>();
        { //处理第三部分 AT@dw -> DKBG
            BlockMmadBdkbg blockMmadBdkbg(resource);
            AscendC::GlobalTensor<ElementAT> gmAT;
            AscendC::GlobalTensor<ElementK> gmDw;
            AscendC::GlobalTensor<ElementDkbg> gmDkbg;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    gmAT.SetGlobalBuffer((__gm__ ElementAT *)params.ptrAT + (h * params.T + bos) * params.chunkSize);
                    gmDw.SetGlobalBuffer((__gm__ ElementDw *)params.ptrDw + (h * params.T + bos) * params.K);
                    gmDkbg.SetGlobalBuffer((__gm__ ElementDkbg *)params.ptrDkbg + (h * params.T + bos) * params.K);

                    // Represent the full tensors
                    auto tensorAT = tla::MakeTensor(gmAT, params.layoutAT, Arch::PositionGM{});
                    auto tensorDw = tla::MakeTensor(gmDw, params.layoutDw, Arch::PositionGM{});
                    auto tensorDkbg = tla::MakeTensor(gmDkbg, params.layoutDkbg, Arch::PositionGM{});

                    // Make tiled views
                    auto tensorBlockAT = GetTile(tensorAT, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockDw = GetTile(tensorDw, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDkbg = GetTile(tensorDkbg, tla::MakeCoord(0, 0),
                                                   tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdkbg(tensorBlockAT, tensorBlockDw, tensorBlockDkbg, actualBlockShape);
                    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                }
            }
        }
        AscendC::SyncAll<false>();
        { //处理第四部分 AT@du -> dvb
            BlockMmadBdvb blockMmadBdvb(resource);
            AscendC::GlobalTensor<ElementAT> gmAT;
            AscendC::GlobalTensor<ElementK> gmDu;
            AscendC::GlobalTensor<ElementDvb> gmDvb;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.V), curChunkSize};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    gmAT.SetGlobalBuffer((__gm__ ElementAT *)params.ptrAT + (h * params.T + bos) * params.chunkSize);
                    gmDu.SetGlobalBuffer((__gm__ ElementDu *)params.ptrDu + (h * params.T + bos) * params.V);
                    gmDvb.SetGlobalBuffer((__gm__ ElementDvb *)params.ptrDvb + (h * params.T + bos) * params.V);

                    // Represent the full tensors
                    auto tensorAT = tla::MakeTensor(gmAT, params.layoutAT, Arch::PositionGM{});
                    auto tensorDu = tla::MakeTensor(gmDu, params.layoutDu, Arch::PositionGM{});
                    auto tensorDvb = tla::MakeTensor(gmDvb, params.layoutDkbg, Arch::PositionGM{});

                    // Make tiled views
                    auto tensorBlockAT = GetTile(tensorAT, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockDu = GetTile(tensorDu, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDvb = GetTile(tensorDvb, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdvb(tensorBlockAT, tensorBlockDu, tensorBlockDvb, actualBlockShape);
                    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                }
            }
        }
        AscendC::SyncAll<false>();
        { //处理第五部分 K@KT -> kkT
            BlockMmadBkkT blockMmadkkT(resource);
            AscendC::GlobalTensor<ElementK> gmK;
            AscendC::GlobalTensor<ElementKT> gmKT;
            AscendC::GlobalTensor<ElementKKT> gmKKT;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.H, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, curChunkSize, static_cast<uint32_t>(params.K)};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    gmK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK + (h * params.T + bos) * params.K);
                    gmKT.SetGlobalBuffer((__gm__ ElementKT *)params.ptrKT + (h * params.T + bos) * params.K);
                    gmKKT.SetGlobalBuffer((__gm__ ElementKKT *)params.ptrKKT + (h * params.T + bos) * params.chunkSize);

                    // Represent the full tensors
                    auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
                    auto tensorKT = tla::MakeTensor(gmKT, params.layoutKT, Arch::PositionGM{});
                    auto tensorKKT = tla::MakeTensor(gmKKT, params.layoutKKT, Arch::PositionGM{});

                    // Make tiled views
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockKT = GetTile(tensorKT, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockKKT = GetTile(tensorKKT, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadkkT(tensorBlockK, tensorBlockKT, tensorBlockKKT, actualBlockShape);
                    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                }
            }
        }
    }
};
} // namespace Catlass::Gemm::Kernel

template <typename kType, typename betaType>
class PrepareWyReprBwdFullProcess {
public:
    /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdFullProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_,
                                                  GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                  GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_,
                                                  GM_ADDR dg_, GM_ADDR workspace_);

    __aicore__ inline void Process();

    __aicore__ inline void Init(const PrepareWyReprBwdFullTilingData &tiling);

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
    GM_ADDR dA;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR dbeta;
    GM_ADDR dg;
    GM_ADDR workspace;
};

template <typename kType, typename betaType>
__aicore__ inline PrepareWyReprBwdFullProcess<kType, betaType>::PrepareWyReprBwdFullProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), dA(dA_), dw(dw_), du(du_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), dk(dk_), dv(dv_), dbeta(dbeta_), dg(dg_), workspace(workspace_){};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Init(const PrepareWyReprBwdFullTilingData &tiling)
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
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Process()
{
    //输入
    using LayoutTagA = layout::RowMajor;
    using LayoutTagAT = layout::ColumnMajor;
    using LayoutTagDW = layout::RowMajor;
    using LayoutTagDA = layout::RowMajor;
    using LayoutTagDAT = layout::ColumnMajor;
    using LayoutTagBeta = layout::RowMajor;
    using LayoutTagK = layout::RowMajor;
    using LayoutTagV = layout::RowMajor;
    using LayoutTagKT = layout::ColumnMajor;
    using LayoutTagDu = layout::RowMajor;
    using LayoutTagDvb = layout::RowMajor;


    //输入
    LayoutTagA tagA = LayoutTagA::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagAT tagAT = LayoutTagAT::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagDW tagDW = LayoutTagDW::MakeLayout<kType>(chunkSize, K);
    LayoutTagDA tagDA = LayoutTagDA::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagDAT tagDAT = LayoutTagDAT::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagK tagK = LayoutTagK::MakeLayout<kType>(chunkSize, K);
    LayoutTagV tagV = LayoutTagV::MakeLayout<kType>(chunkSize, V);
    LayoutTagKT tagKT = LayoutTagKT::MakeLayout<kType>(K, chunkSize);
    LayoutTagDu tagDu = LayoutTagDu::MakeLayout<kType>(chunkSize, V);

    //中间结果
    using LayoutTagKbeta = layout::RowMajor;
    LayoutTagKbeta tagKbeta = LayoutTagKbeta::MakeLayout<kType>(chunkSize, K);

    using LayoutTagDkb = layout::RowMajor;
    LayoutTagDkb tagDkb = LayoutTagDkb::MakeLayout<kType>(chunkSize, K);

    using LayoutTagDkbg = layout::RowMajor;
    LayoutTagDkbg tagDkbg = LayoutTagDkbg::MakeLayout<kType>(chunkSize, K);

    using LayoutTagDvb = layout::RowMajor;
    LayoutTagDvb tagDvb = LayoutTagDvb::MakeLayout<kType>(chunkSize, V);

    using LayoutTagKKT = layout::RowMajor;
    LayoutTagKKT tagKKT = LayoutTagKKT::MakeLayout<kType>(chunkSize, chunkSize);

    //输出
    using LayoutTagDk = layout::RowMajor;
    LayoutTagDk tagDk = LayoutTagDk::MakeLayout<kType>(chunkSize, K);

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = Shape<_128, _128, _256>;
    using L0TileShape = Shape<_128, _128, _128>;

    //计算dk第一部分, dA @ Kbeta
    using TileCopyDk =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDA, kType, LayoutTagKbeta, kType, LayoutTagDk>;
    using BlockMmadDk =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDk>;

    using TileCopyDkb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDAT, kType, LayoutTagK, kType, LayoutTagDkb>;
    using BlockMmadDkb =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDkb>;

    using TileCopyDkbg =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDW, kType, LayoutTagDkbg>;
    using BlockMmadDkbg =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDkbg>;

    using TileCopyDvb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDu, kType, LayoutTagDvb>;
    using BlockMmadDvb =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDvb>;

    using TileCopyKKT =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagK, kType, LayoutTagKT, kType, LayoutTagKKT>;
    using BlockMmadKKT =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyKKT>;

    auto layoutKbeta = MakeLayoutFromTag(tagKbeta);
    auto layoutDA = MakeLayoutFromTag(tagDA);
    auto layoutDK = MakeLayoutFromTag(tagDk);
    auto layoutDAT = MakeLayoutFromTag(tagDAT);
    auto layoutK = MakeLayoutFromTag(tagK);
    auto layoutDkb = MakeLayoutFromTag(tagDkb);
    auto layoutAT = MakeLayoutFromTag(tagAT);
    auto layoutDw = MakeLayoutFromTag(tagDW);
    auto layoutDkbg = MakeLayoutFromTag(tagDkbg);
    auto layoutDu = MakeLayoutFromTag(tagDu);
    auto layoutDvb = MakeLayoutFromTag(tagDvb);
    auto layoutKT = MakeLayoutFromTag(tagKT);
    auto layoutKKT = MakeLayoutFromTag(tagKKT);
    // kernel level
    using MatmulKernel =
        Gemm::Kernel::PrepareWyReprBwdFullTla<BlockMmadDk, BlockMmadDkb, BlockMmadDkbg, BlockMmadDvb, BlockMmadKKT>;

    MatmulKernel kernel;

    typename MatmulKernel::Params param{
        workspace, layoutKbeta, dA, layoutDA, dk,        layoutDK,  dA,         layoutDAT,     k,        layoutK,
        workspace, layoutDkb,   A,  layoutAT, dw,        layoutDw,  workspace,  layoutDkbg,    du,       layoutDu,
        workspace, layoutDvb,   k,  layoutKT, workspace, layoutKKT, cu_seqlens, chunk_indices, chunkNum, B,
        T,         H,           K,  V,        chunkSize, 4};
    kernel(param);
}


#endif // PREPARE_WY_REPR_BWD_FULL_CUBE_H
