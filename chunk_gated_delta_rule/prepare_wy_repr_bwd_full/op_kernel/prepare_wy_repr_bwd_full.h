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

#include "catlass/gemm/kernel/prepare_wy_repr_bwd_full_tla.hpp"

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

#ifndef PREPARE_WY_REPR_BWD_FULL_H
#define PREPARE_WY_REPR_BWD_FULL_H


using namespace Catlass;
template <typename ComputeType>
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

template <typename ComputeType>
 __aicore__ inline PrepareWyReprBwdFullProcess<ComputeType>::PrepareWyReprBwdFullProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,GM_ADDR workspace_)
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

template <typename ComputeType>
__aicore__ void inline PrepareWyReprBwdFullProcess<ComputeType>::Init(GM_ADDR tiling) {
    return;
}

template <typename ComputeType>
__aicore__ void inline PrepareWyReprBwdFullProcess<ComputeType>::Process() {
    using ElementHalf = half;
    using ElementFloat = float;
    using ElementBeta = half;

    //输入
    using LayoutTagA = layout::ColumnMajor;
    using LayoutTagDW = layout::RowMajor;
    using LayoutTagDA = layout::ColumnMajor;
    using LayoutTagDU = layout::RowMajor;
    using LayoutTagG = layout::RowMajor;
    using LayoutTagBeta = layout::RowMajor;
    using LayoutTagK = layout::RowMajor;
    using LayoutTagV = layout::RowMajor;
    using LayoutTagKT = layout::ColumnMajor;


    //输入
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementHalf>(T, BT);
    LayoutTagDW tagDW = LayoutTagDW::MakeLayout<ElementHalf>(T, K);
    LayoutTagDA tagDA = LayoutTagDA::MakeLayout<ElementHalf>(T, BT);
    LayoutTagDU tagDU = LayoutTagDU::MakeLayout<ElementHalf>(T, V);
    LayoutTagG tagG = LayoutTagG::MakeLayout<ElementHalf>(1,T);
    LayoutTagBeta tagBeta = LayoutTagBeta::MakeLayout<ElementBeta>(1,T);
    LayoutTagK tagK = LayoutTagK::MakeLayout<ElementHalf>(T, K);
    LayoutTagV tagV = LayoutTagDW::MakeLayout<ElementHalf>(T, V);
    LayoutTagKT tagKT = LayoutTagKT::MakeLayout<ElementHalf>(T, K);

    //中间结果
    using LayoutTagDkb = layout::RowMajor;
    LayoutTagDkb tagDkb = LayoutTagDkb::MakeLayout<ElementHalf>(T, K);

    using LayoutTagDkbg = layout::RowMajor;
    LayoutTagV tagDkbg = LayoutTagDkbg::MakeLayout<ElementHalf>(T, V);

    // size_t sizeA = B * H * tagA.Capacity() * sizeof(ElementHalf);
    // size_t sizeDW = B * H * tagDW.Capacity() * sizeof(ElementHalf);
    // size_t sizeDA = B * H * tagDA.Capacity() * sizeof(ElementHalf);
    // size_t sizeDU = B * H * tagDU.Capacity() * sizeof(ElementHalf);
    // size_t sizeG = B * H * tagG.Capacity() * sizeof(ElementHalf);
    // size_t sizeB = B * H * tagBeta.Capacity() * sizeof(ElementHalf);
    // size_t sizeK = B * H * tagK.Capacity() * sizeof(ElementHalf);
    // size_t sizeV = B * H * tagV.Capacity() * sizeof(ElementHalf);



    // size_t lenA = B * H * tagA.Capacity();
    // size_t lenDW = B * H * tagDW.Capacity();
    // size_t lenDA = B * H * tagDA.Capacity();
    // size_t lenDU = B * H * tagDU.Capacity();
    // size_t lenG = B * H * tagG.Capacity();
    // size_t lenB = B * H * tagBeta.Capacity();
    // size_t lenK = B * H * tagK.Capacity();
    // size_t lenV = B * H * tagV.Capacity();
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = Shape<_64, _256, _256>;
    using L0TileShape = Shape<_64, _256, _64>;

    using TileCopyDkb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementHalf, LayoutTagDA, ElementHalf, LayoutTagK, ElementHalf, LayoutTagDkbg>;
    using BlockMmadDkb = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementHalf, ElementHalf, ElementHalf, void, TileCopyDkb>;
    using CType = Gemm::GemmType<half, LayoutTagDkb>;
    // using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PrepareBdk;
    // using XType = CType;
    // using DType = CType;
    // using ComputeType = CType;
    // constexpr uint32_t computeLength = 16384;
    // using TileElemWiseEpilogue = Epilogue::Tile::TileElemwiseBrcbMul<ArchTag, ComputeType>;
    // using EpilogueTileCopy = Epilogue::Tile::TileCopyKBeta<ArchTag, CType, XType, DType>;
    // using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    //     EpilogueDispatchPolicy, CType, XType, DType, TileElemWiseEpilogue, EpilogueTileCopy>;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    
    auto LayoutDA = MakeLayoutFromTag(tagDA);
    auto LayoutK = MakeLayoutFromTag(tagK);
    auto LayoutBeta = MakeLayoutFromTag(tagBeta);

    // kernel level
    using MatmulKernel = Gemm::Kernel::PrepareWyReprBwdFullTla<BlockMmadDkb, BlockMmadDkb, BlockMmadDkb, BlockMmadDkb, BlockMmadDkb, BlockScheduler,ElementBeta, decltype(LayoutBeta)>;

    MatmulKernel kernel;
// Params(GM_ADDR ptrDA_,const LayoutDA &layoutDA_,GM_ADDR ptrK_,const LayoutK &layoutK_,GM_ADDR ptrBeta_,const LayoutBeta &layoutBeta_,GM_ADDR ptrWorkspace_,const LayoutDkb &layoutDkb_,
//         uint64_t B_, uint64_t T_,uint64_t H_,uint64_t K_,uint64_t V_,uint64_t BT_, uint64_t stage_)
    MatmulKernel::Params param{dA, LayoutDA, k, LayoutK, beta, LayoutBeta,dk, LayoutK, B, T, H, K, V, BT, 4};
    kernel(param);
}


#endif  // PREPARE_WY_REPR_BWD_FULL_H
