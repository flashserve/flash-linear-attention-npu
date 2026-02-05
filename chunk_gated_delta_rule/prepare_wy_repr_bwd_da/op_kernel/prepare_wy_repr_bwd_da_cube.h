/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prepare_wy_repr_bwd_da_cube.h
 * \brief
 */

#include "prepare_wy_repr_bwd_da_common.h"
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

#ifndef PREPARE_WY_REPR_BWD_DA_CUBE_H
#define PREPARE_WY_REPR_BWD_DA_CUBE_H


using namespace Catlass;

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmadDA1_,
    class BlockMmadDA2_,
    class BlockMmadDA5_,
    class BlockMmadDA6_
>
class PrepareWyReprBwdDATla {
public:
    using BlockMmadDA1 = BlockMmadDA1_;
    using BlockMmadDA2 = BlockMmadDA2_;
    using BlockMmadDA5 = BlockMmadDA5_;
    using BlockMmadDA6 = BlockMmadDA6_;

    using ArchTag = typename BlockMmadDA1::ArchTag;

    using ElementDw = typename BlockMmadDA1::ElementA;
    using LayoutDw = typename BlockMmadDA1::LayoutA;
    using ElementKbg = typename BlockMmadDA1::ElementB;
    using LayoutKbg = typename BlockMmadDA1::LayoutB;
    using ElementDA1 = typename BlockMmadDA1::ElementC;
    using LayoutDA1 = typename BlockMmadDA1::LayoutC;

    using ElementDu = typename BlockMmadDA2::ElementA;
    using LayoutDu = typename BlockMmadDA2::LayoutA;
    using ElementVb = typename BlockMmadDA2::ElementB;
    using LayoutVb = typename BlockMmadDA2::LayoutB;
    using ElementDA2 = typename BlockMmadDA2::ElementC;
    using LayoutDA2 = typename BlockMmadDA2::LayoutC;

    using ElementDA4 = typename BlockMmadDA5::ElementA;
    using LayoutDA4 = typename BlockMmadDA5::LayoutA;
    using ElementAT = typename BlockMmadDA5::ElementB;
    using LayoutAT = typename BlockMmadDA5::LayoutB;
    using ElementDA5 = typename BlockMmadDA5::ElementC;
    using LayoutDA5 = typename BlockMmadDA5::LayoutC;

    using ElementDA6 = typename BlockMmadDA6::ElementC;
    using LayoutDA6 = typename BlockMmadDA6::LayoutC;

    // Parameters structure
    struct Params {
        // Data members
        GM_ADDR ptrDw;
        LayoutDw layoutDw;
        GM_ADDR ptrKbg;
        LayoutKbg layoutKbg;
        GM_ADDR ptrDA1;
        LayoutDA1 layoutDA1;
        GM_ADDR ptrDu;
        LayoutDu layoutDu;
        GM_ADDR ptrVb;
        LayoutVb layoutVb;
        GM_ADDR ptrDA2;
        LayoutDA2 layoutDA2;
        GM_ADDR ptrDA4;
        LayoutDA4 layoutDA4;
        GM_ADDR ptrAT;
        LayoutAT layoutAT;
        GM_ADDR ptrDA5;
        LayoutDA5 layoutDA5;
        GM_ADDR ptrDA6;
        LayoutDA6 layoutDA6;
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
        Params(GM_ADDR ptrDw_, LayoutDw layoutDw_,
               GM_ADDR ptrKbg_, LayoutKbg layoutKbg_,
               GM_ADDR ptrDA1_, LayoutDA1 layoutDA1_,
               GM_ADDR ptrDu_, LayoutDu layoutDu_,
               GM_ADDR ptrVb_, LayoutVb layoutVb_,
               GM_ADDR ptrDA2_, LayoutDA2 layoutDA2_,
               GM_ADDR ptrDA4_, LayoutDA4 layoutDA4_,
               GM_ADDR ptrAT_, LayoutAT layoutAT_,
               GM_ADDR ptrDA5_, LayoutDA5 layoutDA5_,
               GM_ADDR ptrDA6_, LayoutDA6 layoutDA6_,
               uint64_t B_, uint64_t T_, uint64_t H_, uint64_t K_, uint64_t V_, uint64_t BT_, uint64_t stage_)
            : ptrDw(ptrDw_),
              layoutDw(layoutDw_),
              ptrKbg(ptrKbg_),
              layoutKbg(layoutKbg_),
              ptrDA1(ptrDA1_),
              layoutDA1(layoutDA1_),
              ptrDu(ptrDu_),
              layoutDu(layoutDu_),
              ptrVb(ptrVb_),
              layoutVb(layoutVb_),
              ptrDA2(ptrDA2_),
              layoutDA2(layoutDA2_),
              ptrDA4(ptrDA4_),
              layoutDA4(layoutDA4_),
              ptrAT(ptrAT_),
              layoutAT(layoutAT_),
              ptrDA5(ptrDA5_),
              layoutDA5(layoutDA5_),
              ptrDA6(ptrDA6_),
              layoutDA6(layoutDA6_),
              B(B_),
              T(T_),
              H(H_),
              K(K_),
              V(V_),
              BT(BT_),
              stage(stage_) {}
    };

    // Methods
    CATLASS_DEVICE
    PrepareWyReprBwdDATla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    // Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {
        Arch::Resource<ArchTag> resource;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        {   // 计算第二个矩阵乘 dA_2 = du @ vb.T     V->C
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::printf("###yzq CrossCoreSetFlag ### 1\n");
            AscendC::printf("###yzq CrossCoreSetFlag ### 2\n");
            AscendC::printf("###yzq CrossCoreSetFlag ### 3\n");
            AscendC::printf("###yzq CrossCoreSetFlag ### 4\n");
            uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
            uint32_t coreLoops = params.B * coreLoopsInB;
            BlockMmadDA2 blockMmadDA2(resource);
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{static_cast<uint32_t>(params.BT), static_cast<uint32_t>(params.V), static_cast<uint32_t>(params.BT)};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    AscendC::GlobalTensor<ElementDu> gmDu;
                    gmDu.SetGlobalBuffer((__gm__ ElementDu *)params.ptrDu + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.V);
                    AscendC::GlobalTensor<ElementVb> gmVb;
                    gmVb.SetGlobalBuffer((__gm__ ElementVb *)params.ptrVb + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.V);
                    AscendC::GlobalTensor<ElementDA2> gmDA2;
                    gmDA2.SetGlobalBuffer((__gm__ ElementDA2 *)params.ptrDA2 + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);

                    // Represent the full tensors
                    auto tensorDu = tla::MakeTensor(gmDu, params.layoutDu, Arch::PositionGM{});
                    auto tensorVb = tla::MakeTensor(gmVb, params.layoutVb, Arch::PositionGM{});
                    auto tensorDA2 = tla::MakeTensor(gmDA2, params.layoutDA2, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    AscendC::printf("###yzq CrossCoreWaitFlag ### AIV finish\n");
                    // Make tiled views
                    auto tensorBlockDu = GetTile(tensorDu,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockVb = GetTile(tensorVb,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDA2 = GetTile(tensorDA2,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadDA2(tensorBlockDu, tensorBlockVb, tensorBlockDA2, actualBlockShape);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
                    AscendC::printf("###yzq CrossCoreSetFlag ### AIC finish\n");
                }
            }
        }
        AscendC::SyncAll<false>();
        {   // 计算第一个矩阵乘 dA_1 = dw @ kbg.T     V->C
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            BlockMmadDA1 blockMmadDA1(resource);
            uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
            uint32_t coreLoops = params.B * coreLoopsInB;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{static_cast<uint32_t>(params.BT), static_cast<uint32_t>(params.K), static_cast<uint32_t>(params.BT)};
                // AscendC::printf("blockCoord.m(%d)  blockCoord.n(%d)\n",blockCoord.m(), blockCoord.n());
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    AscendC::GlobalTensor<ElementDw> gmDw;
                    gmDw.SetGlobalBuffer((__gm__ ElementDw *)params.ptrDw + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);
                    AscendC::GlobalTensor<ElementKbg> gmKbg;
                    gmKbg.SetGlobalBuffer((__gm__ ElementKbg *)params.ptrKbg + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.K);
                    AscendC::GlobalTensor<ElementDA1> gmDA1;
                    gmDA1.SetGlobalBuffer((__gm__ ElementDA1 *)params.ptrDA1 + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);

                    // Represent the full tensors
                    auto tensorDw = tla::MakeTensor(gmDw, params.layoutDw, Arch::PositionGM{});
                    auto tensorKbg = tla::MakeTensor(gmKbg, params.layoutKbg, Arch::PositionGM{});
                    auto tensorDA1 = tla::MakeTensor(gmDA1, params.layoutDA1, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockDw = GetTile(tensorDw,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockKbg = GetTile(tensorKbg,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDA1 = GetTile(tensorDA1,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    // AscendC::printf("CrossCoreWaitFlag\n");
                    blockMmadDA1(tensorBlockDw, tensorBlockKbg, tensorBlockDA1, actualBlockShape);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
                    // AscendC::printf("CrossCoreSetFlag\n");
                }
            }
        }
        AscendC::SyncAll<false>();
        {   // 计算第三个矩阵乘 dA_5 = dA_4 @ A.T     V->C
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
            uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
            uint32_t coreLoops = params.B * coreLoopsInB;
            BlockMmadDA5 blockMmadDA5(resource);
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord actualBlockShape{static_cast<uint32_t>(params.BT), static_cast<uint32_t>(params.BT), static_cast<uint32_t>(params.BT)};
                
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    AscendC::GlobalTensor<ElementDA4> gmDA4;
                    gmDA4.SetGlobalBuffer((__gm__ ElementDA4 *)params.ptrDA4 + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);
                    AscendC::GlobalTensor<ElementAT> gmAT;
                    gmAT.SetGlobalBuffer((__gm__ ElementAT *)params.ptrAT + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);
                    AscendC::GlobalTensor<ElementDA5> gmDA5;
                    gmDA5.SetGlobalBuffer((__gm__ ElementDA5 *)params.ptrDA5 + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);

                    // Represent the full tensors
                    auto tensorDA4 = tla::MakeTensor(gmDA4, params.layoutDA4, Arch::PositionGM{});
                    auto tensorAT = tla::MakeTensor(gmAT, params.layoutAT, Arch::PositionGM{});
                    auto tensorDA5 = tla::MakeTensor(gmDA5, params.layoutDA5, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockDA4 = GetTile(tensorDA4,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockAT = GetTile(tensorAT,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDA5 = GetTile(tensorDA5,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    
                    // Compute block-scoped matrix multiply-add
                    blockMmadDA5(tensorBlockDA4, tensorBlockAT, tensorBlockDA5, actualBlockShape);
                    // 注意：这里可能需要设置计算完成的标志
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
                }
            }
        }
        AscendC::SyncAll<false>();
        {   // 计算第四个矩阵乘 dA_6 = A.T @ dA_5
            uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
            uint32_t coreLoops = params.B * coreLoopsInB;
            BlockMmadDA6 blockMmadDA6(resource);
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord actualBlockShape{static_cast<uint32_t>(params.BT), static_cast<uint32_t>(params.BT), static_cast<uint32_t>(params.BT)};
                for (int h = 0; h < params.H; h++) {
                    // Represent the full gm
                    AscendC::GlobalTensor<ElementAT> gmAT;
                    gmAT.SetGlobalBuffer((__gm__ ElementAT *)params.ptrAT + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);
                    AscendC::GlobalTensor<ElementDA5> gmDA5;
                    gmDA5.SetGlobalBuffer((__gm__ ElementDA5 *)params.ptrDA5 + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);
                    AscendC::GlobalTensor<ElementDA6> gmDA6;
                    gmDA6.SetGlobalBuffer((__gm__ ElementDA6 *)params.ptrDA6 + ((bIdx * params.H + h) * params.T + chunkIdx * params.BT) * params.BT);

                    // Represent the full tensors
                    auto tensorAT = tla::MakeTensor(gmAT, params.layoutAT, Arch::PositionGM{});
                    auto tensorDA5 = tla::MakeTensor(gmDA5, params.layoutDA5, Arch::PositionGM{});
                    auto tensorDA6 = tla::MakeTensor(gmDA6, params.layoutDA6, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockAT = GetTile(tensorAT,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockDA5 = GetTile(tensorDA5,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDA6 = GetTile(tensorDA6,
                                                tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    
                    // Compute block-scoped matrix multiply-add
                    blockMmadDA6(tensorBlockAT, tensorBlockDA5, tensorBlockDA6, actualBlockShape);
                    
                    // 注意：这里可能需要设置计算完成的标志
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
                }
            }
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
            AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
        }
    }
};
}


template <typename kType, typename betaType>
class PrepareWyReprBwdDAProcess {
public:
     /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdDAProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dA_, GM_ADDR workspace_);
    __aicore__ inline void Process();
    __aicore__ inline void Init(const PrepareWyReprBwdDaTilingData &tiling);
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
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR dA;
    GM_ADDR workspace;
};

template <typename kType, typename betaType>
__aicore__ inline PrepareWyReprBwdDAProcess<kType, betaType>::PrepareWyReprBwdDAProcess(
    GM_ADDR k_,
    GM_ADDR v_,
    GM_ADDR beta_,
    GM_ADDR A_,
    GM_ADDR dw_,
    GM_ADDR du_,
    GM_ADDR g_,
    GM_ADDR dA_,
    GM_ADDR workspace_)
    : k(k_),
      v(v_),
      beta(beta_),
      A(A_),
      dw(dw_),
      du(du_),
      g(g_),
      dA(dA_),
      workspace(workspace_)
{
};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAProcess<kType, betaType>::Init(const PrepareWyReprBwdDaTilingData &tiling) {
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAProcess<kType, betaType>::Process() {

    // 输入
    using LayoutTagDw = layout::RowMajor;
    LayoutTagDw tagDw = LayoutTagDw::MakeLayout<kType>(BT, K);
    using LayoutTagKbg = layout::ColumnMajor;
    LayoutTagKbg tagKbg = LayoutTagKbg::MakeLayout<kType>(BT, K);
    using LayoutTagDu = layout::RowMajor;
    LayoutTagDu tagDu = LayoutTagDu::MakeLayout<kType>(BT, V);
    using LayoutTagVb = layout::ColumnMajor;
    LayoutTagVb tagVb = LayoutTagVb::MakeLayout<kType>(BT, V);
    using LayoutTagAT = layout::ColumnMajor;
    LayoutTagAT tagAT = LayoutTagAT::MakeLayout<kType>(BT, BT);

    // 中间结果
    using LayoutTagDA1 = layout::RowMajor;
    LayoutTagDA1 tagDA1 = LayoutTagDA1::MakeLayout<kType>(BT, BT);
    using LayoutTagDA2 = layout::RowMajor;
    LayoutTagDA2 tagDA2 = LayoutTagDA2::MakeLayout<kType>(BT, BT);
    using LayoutTagDA4 = layout::RowMajor;
    LayoutTagDA4 tagDA4 = LayoutTagDA4::MakeLayout<kType>(BT, BT);
    using LayoutTagDA5 = layout::RowMajor;
    LayoutTagDA5 tagDA5 = LayoutTagDA5::MakeLayout<kType>(BT, BT);
    using LayoutTagDA6 = layout::RowMajor;
    LayoutTagDA6 tagDA6 = LayoutTagDA6::MakeLayout<kType>(BT, BT);

    // // 输出
    // using LayoutTagDA = layout::RowMajor;
    // LayoutTagDA tagDA = LayoutTagDA::MakeLayout<kType>(BT, BT);


    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = Shape<_128, _128, _256>;
    using L0TileShape = Shape<_128, _128, _128>;

    // 计算第一个矩阵乘 dA_1 = dw @ kbg.T
    using TileCopyDA1 =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDw, kType, LayoutTagKbg, kType, LayoutTagDA1>;
    using BlockMmadDA1 = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDA1>;

    // 计算第二个矩阵乘 dA_2 = du @ vb.T
    using TileCopyDA2 =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDu, kType, LayoutTagVb, kType, LayoutTagDA2>;
    using BlockMmadDA2 = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDA2>;

    // 计算第三个矩阵乘 dA_5 = dA_4 @ A.T
    using TileCopyDA5 =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDA4, kType, LayoutTagAT, kType, LayoutTagDA5>;
    using BlockMmadDA5 = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDA5>;

    // 计算第四个矩阵乘 dA_6 = A.T @ dA_5
    using TileCopyDA6 =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDA5, kType, LayoutTagDA6>;
    using BlockMmadDA6 = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDA6>;

    auto layoutDw = MakeLayoutFromTag(tagDw);
    auto layoutKbg = MakeLayoutFromTag(tagKbg);
    auto layoutDA1 = MakeLayoutFromTag(tagDA1);

    auto layoutDu = MakeLayoutFromTag(tagDu);
    auto layoutVb = MakeLayoutFromTag(tagVb);
    auto layoutDA2 = MakeLayoutFromTag(tagDA2);

    auto layoutDA4 = MakeLayoutFromTag(tagDA4);
    auto layoutAT = MakeLayoutFromTag(tagAT);
    auto layoutDA5 = MakeLayoutFromTag(tagDA5);

    auto layoutDA6 = MakeLayoutFromTag(tagDA6);

    // auto LayoutDA = MakeLayoutFromTag(tagDA);

    // kernel level
    using MatmulKernel = Gemm::Kernel::PrepareWyReprBwdDATla<BlockMmadDA1, BlockMmadDA2, BlockMmadDA5, BlockMmadDA6>;

    MatmulKernel kernel;

    typename MatmulKernel::Params param{
        dw, layoutDw,
        dw, layoutKbg,
        workspace, layoutDA1,
        du, layoutDu,
        du, layoutVb,
        workspace, layoutDA2,
        A, layoutDA4,
        A, layoutAT,
        workspace, layoutDA5,
        workspace, layoutDA6,
        B, T, H, K, V, BT, 4};

    kernel(param);
}


#endif  // PREPARE_WY_REPR_BWD_DA_CUBE_H
