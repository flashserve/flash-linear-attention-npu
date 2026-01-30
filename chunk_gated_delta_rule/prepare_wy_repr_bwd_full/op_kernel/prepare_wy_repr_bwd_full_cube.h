/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#ifndef PREPARE_WY_REPR_BWD_FULL_CUBE_H
#define PREPARE_WY_REPR_BWD_FULL_CUBE_H


using namespace Catlass;

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmadBdk_,
    class BlockMmadBdkb_,
    class BlockMmadBdkbg_,
    class BlockMmadBdvb_,
    class BlockMmadBkkT_
>
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

    using ElementDAT= typename BlockMmadBdkb::ElementA;
    using LayoutDAT = typename BlockMmadBdkb::LayoutA;
    using ElementK = typename BlockMmadBdkb::ElementB;
    using LayoutK = typename BlockMmadBdkb::LayoutB;
    using ElementDkb = typename BlockMmadBdkb::ElementC;
    using LayoutDkb = typename BlockMmadBdkb::LayoutC;

    using ElementAT= typename BlockMmadBdkbg::ElementA;
    using LayoutAT= typename BlockMmadBdkbg::LayoutA;
    using ElementDw= typename BlockMmadBdkbg::ElementB;
    using LayoutDw= typename BlockMmadBdkbg::LayoutB;
    using ElementDkbg= typename BlockMmadBdkbg::ElementC;
    using LayoutDkbg= typename BlockMmadBdkbg::LayoutC;

    using ElementDu= typename BlockMmadBdvb::ElementB;
    using LayoutDu= typename BlockMmadBdvb::LayoutB;
    using ElementDvb= typename BlockMmadBdvb::ElementC;
    using LayoutDvb= typename BlockMmadBdvb::LayoutC;

    using ElementKT= typename BlockMmadBkkT::ElementB;
    using LayoutKT= typename BlockMmadBkkT::LayoutB;
    using ElementKKT= typename BlockMmadBkkT::ElementC;
    using LayoutKKT= typename BlockMmadBkkT::LayoutC;

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
        uint64_t B = 1;
        uint64_t T = 32768;
        uint64_t H = 32;
        uint64_t K = 128;
        uint64_t V = 128;
        uint64_t BT = 64;
        uint64_t stage = 2;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR ptrptrKbeta_, LayoutKbeta layoutKbeta_,GM_ADDR ptrDA_,LayoutDA layoutDA_,
            GM_ADDR ptrDk_,LayoutKbeta layoutDk_,GM_ADDR ptrDAT_,LayoutDAT layoutDAT_,GM_ADDR ptrK_,LayoutK layoutK_,
            GM_ADDR ptrDkb_,LayoutDkb layoutDkb_,GM_ADDR ptrAT_,LayoutAT layoutAT_,GM_ADDR ptrDw_, LayoutDw layoutDw_,
            GM_ADDR ptrDkbg_, LayoutDkbg layoutDkbg_,GM_ADDR ptrDu_, LayoutDkbg layoutDu_,GM_ADDR ptrDvb_, LayoutDkbg layoutDvb_,
            GM_ADDR ptrKT_, LayoutKT layoutKT_, GM_ADDR ptrKKT_, LayoutKKT layoutKKT_,
            uint64_t B_, uint64_t T_,uint64_t H_,uint64_t K_,uint64_t V_,uint64_t BT_, uint64_t stage_)
            : ptrKbeta(ptrptrKbeta_), 
            layoutKbeta(layoutKbeta_),
            ptrDA(ptrDA_),
            layoutDA(layoutDA_),
            ptrDk(ptrDk_), 
            layoutDk(layoutDk_),
            ptrDAT(ptrDAT_),
            layoutDAT(layoutDAT_),
            ptrK(ptrK_),
            layoutK(layoutK_),
            ptrDkb(ptrDkb_),
            layoutDkb(layoutDkb_),
            ptrAT(ptrAT_),
            layoutAT(layoutAT_),
            ptrDw(ptrDw_),
            layoutDw(layoutDw_),
            ptrDkbg(ptrDkbg_),
            layoutDkbg(layoutDkbg_),
            ptrDu(ptrDu_),
            layoutDu(layoutDu_),
            ptrDvb(ptrDvb_),
            layoutDvb(layoutDvb_),
            ptrKT(ptrKT_),
            layoutKT(layoutKT_),
            ptrKKT(ptrKKT_),
            layoutKKT(layoutKKT_),
            B(B_), 
            T(T_), 
            H(H_), 
            K(K_), 
            V(V_), 
            BT(BT_), 
            stage(stage_){}
    };

    // Methods
    CATLASS_DEVICE
    PrepareWyReprBwdFullTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {
        Arch::Resource<ArchTag> resource;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        {   //处理第一部分cube DA @ Kbeta     V->C
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
            //AscendC::printf("CrossCoreSetFlag\n");
            //AscendC::printf("CrossCoreSetFlag\n");
            BlockMmadBdk blockMmadBdk(resource);
            uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
            uint32_t coreLoops = params.B * coreLoopsInB;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord blockCoord{0,0,0};
                GemmCoord actualBlockShape{static_cast<uint32_t>(params.BT),static_cast<uint32_t>(params.K),static_cast<uint32_t>(params.BT)};
                // AscendC::printf("blockCoord.m(%d)  blockCoord.n(%d)\n",blockCoord.m(), blockCoord.n());
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    AscendC::GlobalTensor<ElementDA> gmDA;
                    gmDA.SetGlobalBuffer((__gm__ ElementDA *)params.ptrDA +((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);
                    AscendC::GlobalTensor<ElementKbeta> gmKbeta;
                    gmKbeta.SetGlobalBuffer((__gm__ ElementKbeta *)params.ptrKbeta + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);
                    AscendC::GlobalTensor<ElementDk> gmDk;
                    gmDk.SetGlobalBuffer((__gm__ ElementDk *)params.ptrDk + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);

                    // Represent the full tensors
                    auto tensorDA = tla::MakeTensor(gmDA, params.layoutDA, Arch::PositionGM{});
                    auto tensorKbeta = tla::MakeTensor(gmKbeta, params.layoutKbeta, Arch::PositionGM{});
                    auto tensorDk = tla::MakeTensor(gmDk, params.layoutDk, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockDA = GetTile(tensorDA,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockDk = GetTile(tensorDk,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add

                    auto tensorBlockKbeta = GetTile(tensorKbeta,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    //AscendC::printf("CrossCoreWaitFlag\n");
                    blockMmadBdk(tensorBlockDA, tensorBlockKbeta, tensorBlockDk, actualBlockShape);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
                    //AscendC::printf("CrossCoreSetFlag\n");
                }
            }
        }
        {//处理第二部分 DAT@K -> DKB
            uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
            uint32_t coreLoops = params.B * coreLoopsInB;
            BlockMmadBdkb blockMmadBdkb(resource);
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord blockCoord{0,0,0};
                GemmCoord actualBlockShape{static_cast<uint32_t>(params.BT),static_cast<uint32_t>(params.K),static_cast<uint32_t>(params.BT)};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    AscendC::GlobalTensor<ElementDAT> gmDAT;
                    gmDAT.SetGlobalBuffer((__gm__ ElementDAT *)params.ptrDAT + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);
                    AscendC::GlobalTensor<ElementK> gmK;
                    gmK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);
                    AscendC::GlobalTensor<ElementDkb> gmDkb;
                    gmDkb.SetGlobalBuffer((__gm__ ElementDkb *)params.ptrDkb + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);

                    // Represent the full tensors
                    auto tensorDAT = tla::MakeTensor(gmDAT, params.layoutDAT, Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
                    auto tensorDkb = tla::MakeTensor(gmDkb, params.layoutDkb, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockDAT = GetTile(tensorDAT,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockK = GetTile(tensorK,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDkb = GetTile(tensorDkb,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdkb(tensorBlockDAT, tensorBlockK, tensorBlockDkb, actualBlockShape);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
                }
            }
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
        }
        {//处理第三部分 AT@dw -> DKBG
            uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
            uint32_t coreLoops = params.B * coreLoopsInB;
            BlockMmadBdkbg blockMmadBdkbg(resource);
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord blockCoord{0,0,0};
                GemmCoord actualBlockShape{static_cast<uint32_t>(params.BT),static_cast<uint32_t>(params.K),static_cast<uint32_t>(params.BT)};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    AscendC::GlobalTensor<ElementAT> gmAT;
                    gmAT.SetGlobalBuffer((__gm__ ElementAT *)params.ptrAT + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);
                    AscendC::GlobalTensor<ElementK> gmDw;
                    gmDw.SetGlobalBuffer((__gm__ ElementDw *)params.ptrDw + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);
                    AscendC::GlobalTensor<ElementDkbg> gmDkbg;
                    gmDkbg.SetGlobalBuffer((__gm__ ElementDkbg *)params.ptrDkbg + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);

                    // Represent the full tensors
                    auto tensorAT = tla::MakeTensor(gmAT, params.layoutAT, Arch::PositionGM{});
                    auto tensorDw = tla::MakeTensor(gmDw, params.layoutDw, Arch::PositionGM{});
                    auto tensorDkbg = tla::MakeTensor(gmDkbg, params.layoutDkbg, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockAT = GetTile(tensorAT,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockDw = GetTile(tensorDw,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDkbg = GetTile(tensorDkbg,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdkbg(tensorBlockAT, tensorBlockDw, tensorBlockDkbg, actualBlockShape);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
                }
            }
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
        }
        {//处理第四部分 AT@du -> dvb
            uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
            uint32_t coreLoops = params.B * coreLoopsInB;
            BlockMmadBdvb blockMmadBdvb(resource);
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord blockCoord{0,0,0};
                GemmCoord actualBlockShape{static_cast<uint32_t>(params.BT),static_cast<uint32_t>(params.V),static_cast<uint32_t>(params.BT)};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    AscendC::GlobalTensor<ElementAT> gmAT;
                    gmAT.SetGlobalBuffer((__gm__ ElementAT *)params.ptrAT + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);
                    AscendC::GlobalTensor<ElementK> gmDu;
                    gmDu.SetGlobalBuffer((__gm__ ElementDu *)params.ptrDu + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.V);
                    AscendC::GlobalTensor<ElementDvb> gmDvb;
                    gmDvb.SetGlobalBuffer((__gm__ ElementDvb *)params.ptrDvb + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.V);

                    // Represent the full tensors
                    auto tensorAT = tla::MakeTensor(gmAT, params.layoutAT, Arch::PositionGM{});
                    auto tensorDu = tla::MakeTensor(gmDu, params.layoutDu, Arch::PositionGM{});
                    auto tensorDvb = tla::MakeTensor(gmDvb, params.layoutDkbg, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockAT = GetTile(tensorAT,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockDu = GetTile(tensorDu,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDvb = GetTile(tensorDvb,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdvb(tensorBlockAT, tensorBlockDu, tensorBlockDvb, actualBlockShape);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
                }
            }
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
        }
        {//处理第五部分 K@KT -> kkT
            uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
            uint32_t coreLoops = params.B * coreLoopsInB;
            BlockMmadBkkT blockMmadkkT(resource);
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord blockCoord{0,0,0};
                GemmCoord actualBlockShape{static_cast<uint32_t>(params.BT),static_cast<uint32_t>(params.BT),static_cast<uint32_t>(params.K)};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    AscendC::GlobalTensor<ElementK> gmK;
                    gmK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);
                    AscendC::GlobalTensor<ElementKT> gmKT;
                    gmKT.SetGlobalBuffer((__gm__ ElementKT *)params.ptrKT + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);
                    AscendC::GlobalTensor<ElementKKT> gmKKT;
                    gmKKT.SetGlobalBuffer((__gm__ ElementKKT *)params.ptrKKT + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);

                    // Represent the full tensors
                    auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
                    auto tensorKT = tla::MakeTensor(gmKT, params.layoutKT, Arch::PositionGM{});
                    auto tensorKKT = tla::MakeTensor(gmKKT, params.layoutKKT, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockK = GetTile(tensorK,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockKT = GetTile(tensorKT,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockKKT = GetTile(tensorKKT,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadkkT(tensorBlockK, tensorBlockKT, tensorBlockKKT, actualBlockShape);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
                }
            }
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
        }
    }
};
}

template <typename kType, typename betaType>
class PrepareWyReprBwdFullProcess {
 public:
     /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdFullProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,GM_ADDR workspace_);

    __aicore__ inline void Process();

    __aicore__ inline void Init(GM_ADDR tiling);
private:
    uint64_t B = 1;
    uint64_t T = 2048;
    uint64_t H = 4;
    uint64_t K = 128;
    uint64_t V = 128;
    uint64_t BT = 64;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR dA;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR dbeta;
    GM_ADDR dg;
    GM_ADDR workspace;
};

template <typename kType, typename betaType>
 __aicore__ inline PrepareWyReprBwdFullProcess<kType, betaType>::PrepareWyReprBwdFullProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,GM_ADDR workspace_)
 :
    k(k_),
    v(v_),
    beta(beta_),
    A(A_),
    dA(dA_),
    dw(dw_),
    du(du_),
    g(g_),
    dk(dk_),
    dv(dv_),
    dbeta(dbeta_),
    dg(dg_),
    workspace(workspace_)
    {};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Init(GM_ADDR tiling) {
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Process() {

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
    LayoutTagA tagA = LayoutTagA::MakeLayout<kType>(BT, BT);
    LayoutTagAT tagAT = LayoutTagAT::MakeLayout<kType>(BT, BT);
    LayoutTagDW tagDW = LayoutTagDW::MakeLayout<kType>(BT, K);
    LayoutTagDA tagDA = LayoutTagDA::MakeLayout<kType>(BT, BT);
    LayoutTagDAT tagDAT = LayoutTagDAT::MakeLayout<kType>(BT, BT);
    LayoutTagK tagK = LayoutTagK::MakeLayout<kType>(BT, K);
    LayoutTagV tagV = LayoutTagV::MakeLayout<kType>(BT, V);
    LayoutTagKT tagKT = LayoutTagKT::MakeLayout<kType>(BT, K);
    LayoutTagDu tagDu = LayoutTagDu::MakeLayout<kType>(BT, V);

    //中间结果
    using LayoutTagKbeta = layout::RowMajor;
    LayoutTagKbeta tagKbeta = LayoutTagKbeta::MakeLayout<kType>(BT, K);

    using LayoutTagDkb = layout::RowMajor;
    LayoutTagDkb tagDkb = LayoutTagDkb::MakeLayout<kType>(BT, K);

    using LayoutTagDkbg = layout::RowMajor;
    LayoutTagDkbg tagDkbg = LayoutTagDkbg::MakeLayout<kType>(BT, K);

    using LayoutTagDvb = layout::RowMajor;
    LayoutTagDvb tagDvb = LayoutTagDvb::MakeLayout<kType>(BT, V);

    using LayoutTagKKT = layout::RowMajor;
    LayoutTagKKT tagKKT = LayoutTagKKT::MakeLayout<kType>(BT, BT);

    //输出
    using LayoutTagDk = layout::RowMajor;
    LayoutTagDk tagDk = LayoutTagDk::MakeLayout<kType>(BT, K);

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = Shape<_128, _128, _256>;
    using L0TileShape = Shape<_128, _128, _128>;

    //计算dk第一部分, dA @ Kbeta
    using TileCopyDk =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDA, kType, LayoutTagKbeta, kType, LayoutTagDk>;
    using BlockMmadDk = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDk>;

    using TileCopyDkb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDAT, kType, LayoutTagK, kType, LayoutTagDkb>;
    using BlockMmadDkb = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDkb>;

    using TileCopyDkbg =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDW, kType, LayoutTagDkbg>;
    using BlockMmadDkbg = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDkbg>;

    using TileCopyDvb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDu, kType, LayoutTagDvb>;
    using BlockMmadDvb = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDvb>;

    using TileCopyKKT =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagK, kType, LayoutTagKT, kType, LayoutTagKKT>;
    using BlockMmadKKT = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyKKT>;

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
    using MatmulKernel = Gemm::Kernel::PrepareWyReprBwdFullTla<BlockMmadDk, BlockMmadDkb, BlockMmadDkbg, BlockMmadDvb, BlockMmadKKT>;

    MatmulKernel kernel;

    typename MatmulKernel::Params param{
        workspace, layoutKbeta,
        dA, layoutDA,
        dk, layoutDK,
        dA, layoutDAT,
        k, layoutK, 
        workspace, layoutDkb, 
        A, layoutAT, 
        dw, layoutDw, 
        workspace, layoutDkbg, 
        du, layoutDu, 
        workspace, layoutDvb, 
        k, layoutKT, 
        workspace, layoutKKT, 
        B, T, H, K, V, BT, 4};
    kernel(param);
}


#endif  // PREPARE_WY_REPR_BWD_FULL_CUBE_H
