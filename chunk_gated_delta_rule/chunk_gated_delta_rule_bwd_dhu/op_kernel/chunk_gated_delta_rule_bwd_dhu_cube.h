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
 * \file chunk_gated_delta_rule_bwd_dhu_vec.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_CUBE_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_CUBE_H
#endif

#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "chunk_gated_delta_rule_bwd_dhu_base.h"

using namespace Catlass;
using namespace ChunkGDRBwdDhu;

namespace Catlass::Gemm::Kernel {

template <
    class ArchTag_,
    typename DT,
    // class BlockScheduler_,
    class L1TileShapeBdv_,
    class L0TileShapeBdv_,
    class TileCopyBdv_,
    class L1TileShapeDh_,
    class L0TileShapeDh_,
    class TileCopyDh1_,
    class TileCopyDh2_
>
class ChunkGDRBwdDhuTla{
public:
    using ArchTag = ArchTag_;
    using ElementK = DT;
    using ElementDh = DT;
    using ElementGq = DT;
    using ElementW = DT;
    using ElementDo = DT;
    using ElementDv2 = DT;
    using ElementAccumulator = float;
    using ElementInt = int64_t;

    using TileCopyBdv = TileCopyBdv_;
    using LayoutK = typename TileCopyBdv::LayoutA;
    using LayoutDh = typename TileCopyBdv::LayoutB;
    using LayoutBdv = typename TileCopyBdv::LayoutC;
    using CopyL1ToL0A_Bdv = typename TileCopyBdv::CopyL1ToL0A;
    using CopyL1ToL0B_Bdv = typename TileCopyBdv::CopyL1ToL0B;
    using LayoutTagL1A_Bdv = typename TileCopyBdv::LayoutTagL1A;
    using LayoutTagL1B_Bdv = typename TileCopyBdv::LayoutTagL1B;
    using LayoutTagL0A_Bdv = typename TileCopyBdv::LayoutTagL0A;
    using LayoutTagL0B_Bdv = typename TileCopyBdv::LayoutTagL0B;
    using L1AAlignHelper = typename TileCopyBdv::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopyBdv::L1BAlignHelper;

    using TileCopyDh1 = TileCopyDh1_;
    using LayoutGq = typename TileCopyDh1::LayoutA;
    using LayoutDo = typename TileCopyDh1::LayoutB;
    using LayoutBdh = typename TileCopyDh1::LayoutC;
    using CopyL1ToL0A_Dh1 = typename TileCopyDh1::CopyL1ToL0A;
    using CopyL1ToL0B_Dh1 = typename TileCopyDh1::CopyL1ToL0B;
    using LayoutTagL1A_Dh1 = typename TileCopyDh1::LayoutTagL1A;
    using LayoutTagL1B_Dh1 = typename TileCopyDh1::LayoutTagL1B;
    using LayoutTagL0A_Dh1 = typename TileCopyDh1::LayoutTagL0A;
    using LayoutTagL0B_Dh1 = typename TileCopyDh1::LayoutTagL0B;

    using TileCopyDh2 = TileCopyDh2_;
    using LayoutW = typename TileCopyDh2::LayoutA;
    using LayoutDv2 = typename TileCopyDh2::LayoutB;
    using CopyL1ToL0A_Dh2 = typename TileCopyDh2::CopyL1ToL0A;
    using CopyL1ToL0B_Dh2 = typename TileCopyDh2::CopyL1ToL0B;
    using LayoutTagL1A_Dh2 = typename TileCopyDh2::LayoutTagL1A;
    using LayoutTagL1B_Dh2 = typename TileCopyDh2::LayoutTagL1B;
    using LayoutTagL0A_Dh2 = typename TileCopyDh2::LayoutTagL0A;
    using LayoutTagL0B_Dh2 = typename TileCopyDh2::LayoutTagL0B;


    using L1TileShapeBdv = L1TileShapeBdv_;
    using L0TileShapeBdv = L0TileShapeBdv_;
    static constexpr uint32_t L1_TILE_M_BDV = tla::get<0>(L1TileShapeBdv{}); // BT 128
    static constexpr uint32_t L1_TILE_N_BDV = tla::get<1>(L1TileShapeBdv{}); // V 256
    static constexpr uint32_t L1_TILE_K_BDV = tla::get<2>(L1TileShapeBdv{}); // K 128
    
    static constexpr uint32_t L0_TILE_M_BDV = tla::get<0>(L0TileShapeBdv{}); // K 128
    static constexpr uint32_t L0_TILE_N_BDV = tla::get<1>(L0TileShapeBdv{}); // V 256
    static constexpr uint32_t L0_TILE_K_BDV = tla::get<2>(L0TileShapeBdv{}); // BT 128

    static constexpr auto L1A_LAYOUT_BDV = tla::MakeLayout<ElementK, LayoutTagL1A_Bdv>(Int<L1_TILE_M_BDV>{}, Int<L1_TILE_K_BDV>{});
    static constexpr auto L1B_LAYOUT_BDV = tla::MakeLayout<ElementDh, LayoutTagL1B_Bdv>(Int<L1_TILE_K_BDV>{}, Int<L1_TILE_N_BDV>{});

    static constexpr uint32_t L1A_TILE_SIZE_BDV = L1_TILE_M_BDV * L1_TILE_K_BDV * sizeof(ElementK);

    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using DhType = Gemm::GemmType<ElementDh, LayoutDh>;

    using TileMmadBdv = Gemm::Tile::TileMmadTla<ArchTag, KType, LayoutTagL1A_Bdv>;

    using L1TileShapeDh = L1TileShapeDh_;
    using L0TileShapeDh = L0TileShapeDh_;
    static constexpr uint32_t L1_TILE_M_DH = tla::get<0>(L1TileShapeDh{});
    static constexpr uint32_t L1_TILE_N_DH = tla::get<1>(L1TileShapeDh{});
    static constexpr uint32_t L1_TILE_K_DH = tla::get<2>(L1TileShapeDh{});
    static constexpr uint32_t L0_TILE_M_DH = tla::get<0>(L0TileShapeDh{});
    static constexpr uint32_t L0_TILE_N_DH = tla::get<1>(L0TileShapeDh{});
    static constexpr uint32_t L0_TILE_K_DH = tla::get<2>(L0TileShapeDh{});
    static constexpr auto L1A_LAYOUT_DH1 = tla::MakeLayout<ElementGq, LayoutTagL1A_Dh1>(Int<L1_TILE_M_DH>{}, Int<L1_TILE_K_DH>{});
    static constexpr auto L1B_LAYOUT_DH1 = tla::MakeLayout<ElementDo, LayoutTagL1B_Dh1>(Int<L1_TILE_K_DH>{}, Int<L1_TILE_N_DH>{});
    static constexpr uint32_t L1A_TILE_SIZE_DH = L1_TILE_M_DH * L1_TILE_K_DH * sizeof(ElementGq);

    using GqType = Gemm::GemmType<ElementGq, LayoutGq>;
    using DoType = Gemm::GemmType<ElementDo, LayoutDo>;
    using TileMmadDh1 = Gemm::Tile::TileMmadTla<ArchTag, GqType, LayoutTagL1A_Dh1>;

    using WType = Gemm::GemmType<ElementW, LayoutW>;
    using TileMmadDh2 = Gemm::Tile::TileMmadTla<ArchTag, WType, LayoutTagL1A_Dh2>;


    static constexpr auto L1A_LAYOUT_DH2 = tla::MakeLayout<ElementW, LayoutTagL1A_Dh2>(Int<L1_TILE_M_DH>{}, Int<L1_TILE_K_DH>{});
    static constexpr auto L1B_LAYOUT_DH2 = tla::MakeLayout<ElementDv2, LayoutTagL1B_Dh2>(Int<L1_TILE_K_DH>{}, Int<L1_TILE_N_DH>{});
    // using LayoutGq = LayoutGq_;
    // using LayoutDo = LayoutDo_;
    // // k @ dh -> bdv
    // static constexpr uint32_t L1_TILE_BDV_M = BT;
    // static constexpr uint32_t L1_TILE_BDK_N = K;
    // static constexpr uint32_t L1_TILE_BDK_K = N;
    // // gated_q @ do -> qdo
    // static constexpr uint32_t L1_TILE_QDO_M = BT;
    // static constexpr uint32_t L1_TILE_QDO_N = K;
    // static constexpr uint32_t L1_TILE_QDO_K = N;


    struct Params {
        GM_ADDR k;
        LayoutK layoutK;
        GM_ADDR dh; // from output dh
        LayoutDh layoutDh;
        GM_ADDR workspace; // gQ from ws
        LayoutBdv layoutBdv;
        LayoutGq layoutGq;
        GM_ADDR dO;
        LayoutDo layoutDo;
        GM_ADDR w;
        LayoutW layoutW;
        GM_ADDR dv2;
        LayoutDv2 layoutDv2;
        LayoutBdh layoutBdh;
        GM_ADDR cu_seqlens; 
        uint64_t B = 0;
        uint64_t T = 0;
        uint64_t H = 0;
        uint64_t K = 0;
        uint64_t V = 0;
        uint64_t BT = 0;
        uint64_t chunkNum = 0;
        uint64_t seqNum = 0;
        uint64_t usedCoreNum = 0;
        uint64_t bdvWorkspaceOffset = 0;
        uint64_t gQWorkspaceOffset = 0;
        uint64_t bdhTerm1WorkspaceOffset = 0;
        uint64_t bdhTerm2WorkspaceOffset = 0;


        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR k_, LayoutK layoutK_, GM_ADDR dh_, LayoutDh layoutDh_, GM_ADDR workspace_,  LayoutBdv layoutBdv_, 
               LayoutGq layoutGq_, GM_ADDR dO_, LayoutDo layoutDo_,    
               GM_ADDR w_ , LayoutW layoutW_, GM_ADDR dv2_ , LayoutDv2 layoutDv2_, LayoutBdh layoutBdh_,
               GM_ADDR cu_seqlens_, uint64_t B_, uint64_t T_, uint64_t H_, uint64_t K_, uint64_t V_,
               uint64_t BT_, uint64_t chunkNum_, uint64_t seqNum_, uint64_t usedCoreNum_,
               uint64_t bdvWorkspaceOffset_, uint64_t gQWorkspaceOffset_,uint64_t bdhTerm1WorkspaceOffset_, uint64_t bdhTerm2WorkspaceOffset_): 
            k(k_), 
            layoutK(layoutK_),
            dh(dh_),
            layoutDh(layoutDh_),
            workspace(workspace_), 
            layoutBdv(layoutBdv_),
            layoutGq(layoutGq_),
            dO(dO_),
            layoutDo(layoutDo_),
            w(w_),
            layoutW(layoutW_),
            dv2(dv2_),
            layoutDv2(layoutDv2_),
            layoutBdh(layoutBdh_),
            cu_seqlens(cu_seqlens_),
            B(B_), 
            T(T_), 
            H(H_), 
            K(K_), 
            V(V_), 
            BT(BT_),
            chunkNum(chunkNum_),
            seqNum(seqNum_),
            usedCoreNum(usedCoreNum_),
            bdvWorkspaceOffset(bdvWorkspaceOffset_),
            gQWorkspaceOffset(gQWorkspaceOffset_),
            bdhTerm1WorkspaceOffset(bdhTerm1WorkspaceOffset_),
            bdhTerm2WorkspaceOffset(bdhTerm2WorkspaceOffset_)
            {}
    };

    CATLASS_HOST_DEVICE
    ChunkGDRBwdDhuTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {
        GemmCoord ProblemShapeQdh{static_cast<uint32_t>(params.T),static_cast<uint32_t>(params.K), static_cast<uint32_t>(params.BT)}; 
        Arch::Resource<ArchTag> resource;
        l1ATensorBdv = resource.l1Buf.template GetBufferByByte<ElementK>(0);
        l1BTensorBdv = resource.l1Buf.template GetBufferByByte<ElementDh>(L1A_TILE_SIZE_BDV);
        l0ATensorBdv = resource.l0ABuf.template GetBufferByByte<ElementK>(0);
        l0BTensorBdv = resource.l0BBuf.template GetBufferByByte<ElementDh>(0);
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        l1ATensorDh = resource.l1Buf.template GetBufferByByte<ElementK>(0);
        l1BTensorDh = resource.l1Buf.template GetBufferByByte<ElementDh>(L1A_TILE_SIZE_DH);
        l0ATensorDh = resource.l0ABuf.template GetBufferByByte<ElementK>(0);
        l0BTensorDh = resource.l0BBuf.template GetBufferByByte<ElementDh>(0);
        {
            // >>>>>>>>>>>>>变长：
            // AscendC::CrossCoreWaitFlag<PIPE_MTE2>(SYNC_AIC_AIV_FLAG_0); // 等待前一个循环的Vec执行结束


            gmCuSeqlens.SetGlobalBuffer((__gm__ ElementInt *)params.cu_seqlens);
            // chunk_indices[seq_idx, chunk_idx]
            // cu_seqlens[0, seq_len_0, seq_len_0 + seq_len_1, ...]

            // 每个核完成一个batch里的一个head的一个seqence里的所有chunk
            uint32_t totalTaskNum = params.B * params.H * params.seqNum; // 2head 3 seqNum
            uint32_t coreIdx = GetBlockIdx();
            // AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
            // AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mm2mte1EventId);
            for (uint32_t i = coreIdx; i < totalTaskNum; i += params.usedCoreNum) {
                // 当前核要处理的seq
                uint32_t seqIdx = i / params.H; // 当前任务在第几个seq
                uint32_t h = i % params.H; // 当前任务在第几个h
                // {0, 96, 224, 320} [0, 2, 4， 6]
                uint64_t seqStartOffset = gmCuSeqlens.GetValue(seqIdx); // 当前seq在T中的起始索引
                uint64_t seqEndOffset = gmCuSeqlens.GetValue(seqIdx+1); // 当前seq在T中的结束索引
                uint64_t curSeqLen = seqEndOffset - seqStartOffset;
                uint64_t tailChunkLen = curSeqLen % params.BT; 
                // 计算当前seq的起始chunkIdx
                uint64_t preChunkNum = 0;
                uint64_t tmpStartOffset = 0;
                uint64_t tmpEndOffset = 0;
                uint64_t tmpChunkNum = 0;
                for (uint32_t seq = 0; seq < seqIdx; seq++) {
                    tmpStartOffset = gmCuSeqlens.GetValue(seq); // 当前seq在T中的起始索引
                    tmpEndOffset = gmCuSeqlens.GetValue(seq+1); // 当前seq在T中的结束索引
                    auto tmpChunkNum = ((tmpEndOffset - tmpStartOffset) + params.BT - 1) / params.BT;
                    preChunkNum += tmpChunkNum;
                }
                int32_t curChunkNum = (curSeqLen + params.BT - 1) / params.BT; // 当前seq的chunk数
                // calc offset
                uint64_t b = 0;
                uint64_t curSeqStartOffsetK = (b * params.H + h) * params.T * params.K + seqStartOffset * params.K;
                uint64_t dhPreBHOffset = (b * params.H + h) * params.chunkNum * params.K * params.V + 
                                        preChunkNum * params.K * params.V; // [B,H,chunk_num,K,V]
                uint64_t curSeqStartOffsetV = (b * params.H + h) * params.T * params.V + seqStartOffset * params.V;
                // AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l0AEventId);
                // AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l0BEventId);
                uint32_t curBT = 0;
                for (int32_t chunkIdx = curChunkNum - 1; chunkIdx >= 0; chunkIdx --) {
                    // cacl k @ dh
                    if (chunkIdx == curChunkNum -1) {
                        curBT = curSeqLen - chunkIdx * params.BT;
                        // skip k_i @ dh_0
                    } else {
                        CrossCoreWaitFlag(CROSS_CORE_V2C_BDH);
                        curBT = params.BT;  // BT = 64/128 is always 16 aligned
                        // init GlobalTensor
                        gmK.SetGlobalBuffer((__gm__ ElementK *)params.k + curSeqStartOffsetK + chunkIdx * params.BT * params.K);
                         // 用的是上一次chunk迭代的结果
                        gmDh.SetGlobalBuffer((__gm__ ElementDh *)params.dh + dhPreBHOffset + chunkIdx * params.K * params.V);
                        gmWsBdv.SetGlobalBuffer((__gm__ ElementDh *)params.workspace + coreIdx * params.BT * params.V);
                        
                        auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
                        auto tensorDh = tla::MakeTensor(gmDh, params.layoutDh, Arch::PositionGM{});
                        auto tensorBdv = tla::MakeTensor(gmWsBdv, params.layoutBdv, Arch::PositionGM{});
                        // Make tiled views
                        auto tensorBlockK = GetTile(tensorK,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(curBT, params.K));
                        auto tensorBlockDh = GetTile(tensorDh,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.K, params.V));
                        auto tensorBlockBdv = GetTile(tensorBdv,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(curBT, params.V));
                        using CopyGmToL1A_Bdv = typename TileCopyBdv::template CopyGmToL1A<decltype(tensorBlockK)>;
                        using CopyGmToL1B_Bdv = typename TileCopyBdv::template CopyGmToL1B<decltype(tensorBlockDh)>;
                        using CopyL0CToGm_Bdv = typename TileCopyBdv::template CopyL0CToGm<decltype(tensorBlockBdv)>;
                        CopyGmToL1A_Bdv copyGmToL1A_Bdv;
                        CopyGmToL1B_Bdv copyGmToL1B_Bdv;
                        CopyL0CToGm_Bdv copyL0CToGm_Bdv;

                        PipeBarrier<PIPE_ALL>();
                        // load L1A
                        auto tensorL1A = tla::MakeTensor(l1ATensorBdv, L1A_LAYOUT_BDV, Arch::PositionL1{});
                        auto tensorGmTileA = GetTile(tensorBlockK, tla::MakeCoord(0, 0), tla::MakeShape(curBT, params.K));
                        copyGmToL1A_Bdv(tensorL1A, tensorGmTileA);
                        PipeBarrier<PIPE_ALL>();

                        // load L1B
                        auto tensorL1B = tla::MakeTensor(l1BTensorBdv, L1B_LAYOUT_BDV, Arch::PositionL1{});
                        auto tensorGmTileB = GetTile(tensorBlockDh, tla::MakeCoord(0, 0), tla::MakeShape(params.K, params.V));
                        copyGmToL1B_Bdv(tensorL1B, tensorGmTileB);
                        PipeBarrier<PIPE_ALL>();

                        // copy L1A -> L0A
                        auto layoutAInL0 = tla::MakeLayout<ElementK, LayoutTagL0A_Bdv>(curBT, params.K);
                        auto tensorL0A = tla::MakeTensor(l0ATensorBdv, layoutAInL0, Arch::PositionL0A{});
                        auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(curBT, params.K));
                        copyL1ToL0A_Bdv(tensorL0A, tensorTileL1A);
                        PipeBarrier<PIPE_ALL>();

                        // copy L1B -> L0B
                        auto layoutBInL0 = tla::MakeLayout<ElementDh, LayoutTagL0B_Bdv>(params.K, params.V);
                        auto tensorL0B = tla::MakeTensor(l0BTensorBdv, layoutBInL0, Arch::PositionL0B{});
                        auto tensorTileL1B = GetTile(tensorL1B,  tla::MakeCoord(0, 0), tla::MakeShape(params.K, params.V));
                        copyL1ToL0B_Bdv(tensorL0B, tensorTileL1B);
                        PipeBarrier<PIPE_ALL>();

                        bool initC = true; //k方向没有循环
                        uint8_t unitFlag = 0;
                        auto layoutInL0C = tla::MakeLayoutL0C(curBT, params.V);
                        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});
                        auto tensorTileL0C = GetTile(tensorL0C,
                                                     tla::MakeCoord(0,0),
                                                     tla::MakeShape(curBT, params.V));
                        tileMmadBdv(tensorTileL0C, tensorL0A, tensorL0B,
                                    curBT, params.V, params.K, initC, unitFlag);
                        
                        PipeBarrier<PIPE_ALL>();
                        copyL0CToGm_Bdv(tensorBlockBdv, tensorL0C);
                        PipeBarrier<PIPE_ALL>();
                        CrossCoreSetFlag<0x2, PIPE_FIX>(CROSS_CORE_C2V_BDV); // 计算完一个chunk的bdv,通知vec可以开始计算对应的dv2
                    } // end chunk k @ dh
                    
                    // gatedQ @ do
                    // | bdv coreNum * K * V | gQ coreNum * BT * K | qDo coreNum * K * V | wDv2 coreNum * K * V | 
                    if (chunkIdx != 0)
                    {
                        CrossCoreWaitFlag(CROSS_CORE_V2C_GQ); // vec计算完一个chunk的gatedQ,通知cube可以开始计算对应的dh term1
                        gmGq.SetGlobalBuffer((__gm__ ElementGq *)params.workspace + params.gQWorkspaceOffset + coreIdx * params.BT * params.K);
                        gmDo.SetGlobalBuffer((__gm__ ElementDo *)params.dO + curSeqStartOffsetV + chunkIdx * params.BT * params.V);
                        gmDhTerm1.SetGlobalBuffer((__gm__ ElementDh *)params.workspace + params.bdhTerm1WorkspaceOffset + coreIdx * params.K * params.V);
                        auto tensorGq = tla::MakeTensor(gmGq, params.layoutGq, Arch::PositionGM{});
                        auto tensorDo = tla::MakeTensor(gmDo, params.layoutDo, Arch::PositionGM{});
                        auto tensorDh1 = tla::MakeTensor(gmDhTerm1, params.layoutDh, Arch::PositionGM{});
                        auto tensorBlockGq = GetTile(tensorGq,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.K, curBT));
                        auto tensorBlockDo = GetTile(tensorDo,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(curBT, params.V));
                        auto tensorBlockDh1 = GetTile(tensorDh1,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.K, params.V));
                        using CopyGmToL1A_Dh1 = typename TileCopyDh1::template CopyGmToL1A<decltype(tensorBlockGq)>;
                        using CopyGmToL1B_Dh1 = typename TileCopyDh1::template CopyGmToL1B<decltype(tensorBlockDo)>;
                        using CopyL0CToGm_Dh1 = typename TileCopyDh1::template CopyL0CToGm<decltype(tensorBlockDh1)>;
                        CopyGmToL1A_Dh1 copyGmToL1A_Dh1;
                        CopyGmToL1B_Dh1 copyGmToL1B_Dh1;
                        CopyL0CToGm_Dh1 copyL0CToGm_Dh1;

                        PipeBarrier<PIPE_ALL>();
                        // load L1A
                        auto tensorL1A = tla::MakeTensor(l1ATensorDh, L1A_LAYOUT_DH1, Arch::PositionL1{});
                        auto tensorGmTileA = GetTile(tensorBlockGq, tla::MakeCoord(0, 0), tla::MakeShape(params.K, curBT));
                        
                        copyGmToL1A_Dh1(tensorL1A, tensorGmTileA);
                        PipeBarrier<PIPE_ALL>();

                        // load L1B
                        auto tensorL1B = tla::MakeTensor(l1BTensorDh, L1B_LAYOUT_DH1, Arch::PositionL1{});
                        auto tensorGmTileB = GetTile(tensorBlockDo, tla::MakeCoord(0, 0), tla::MakeShape(curBT, params.V));
                        copyGmToL1B_Dh1(tensorL1B, tensorGmTileB);
                        PipeBarrier<PIPE_ALL>();

                        // copy L1A -> L0A
                        auto layoutAInL0 = tla::MakeLayout<ElementGq, LayoutTagL0A_Dh1>(params.K, curBT);
                        auto tensorL0A = tla::MakeTensor(l0ATensorDh, layoutAInL0, Arch::PositionL0A{});
                        auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(params.K, curBT));
                        copyL1ToL0A_Dh1(tensorL0A, tensorTileL1A);
                        PipeBarrier<PIPE_ALL>();

                        // copy L1B -> L0B
                        auto layoutBInL0 = tla::MakeLayout<ElementDo, LayoutTagL0B_Dh1>(curBT, params.V);
                        auto tensorL0B = tla::MakeTensor(l0BTensorDh, layoutBInL0, Arch::PositionL0B{});
                        auto tensorTileL1B = GetTile(tensorL1B,  tla::MakeCoord(0, 0), tla::MakeShape(curBT, params.V));
                        copyL1ToL0B_Dh1(tensorL0B, tensorTileL1B);
                        PipeBarrier<PIPE_ALL>();

                        bool initC = true; //k方向没有循环
                        uint8_t unitFlag = 0;
                        auto layoutInL0C = tla::MakeLayoutL0C(params.K, params.V);
                        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});
                        auto tensorTileL0C = GetTile(tensorL0C,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.K, params.V));
                        tileMmadDh1(tensorTileL0C, tensorL0A, tensorL0B,
                                    params.K, params.V, curBT, initC, unitFlag);
                        PipeBarrier<PIPE_ALL>();
                        copyL0CToGm_Dh1(tensorBlockDh1, tensorL0C);
                        PipeBarrier<PIPE_ALL>();
                        CrossCoreSetFlag<0x2, PIPE_FIX>(CROSS_CORE_C2V_TERM1);
                    }
                    if (chunkIdx != 0)
                    {
                        // w @ dv2 -> bdh_term2
                        CrossCoreWaitFlag(CROSS_CORE_V2C_DV2);
                        
                        gmW.SetGlobalBuffer((__gm__ ElementW *)params.w + curSeqStartOffsetK + chunkIdx * params.BT * params.K);
                        gmDv2.SetGlobalBuffer((__gm__ ElementDv2 *)params.dv2 + curSeqStartOffsetV + chunkIdx * params.BT * params.V);
                        gmDhTerm2.SetGlobalBuffer((__gm__ ElementDh *)params.workspace + params.bdhTerm2WorkspaceOffset + coreIdx * params.K * params.V);
                        auto tensorW = tla::MakeTensor(gmW, params.layoutW, Arch::PositionGM{});
                        auto tensorDv2 = tla::MakeTensor(gmDv2, params.layoutDv2, Arch::PositionGM{});
                        auto tensorDh2 = tla::MakeTensor(gmDhTerm2, params.layoutDh, Arch::PositionGM{});
                        auto tensorBlockW = GetTile(tensorW,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.K, curBT));
                        auto tensorBlockDv2 = GetTile(tensorDv2,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(curBT, params.V));
                        auto tensorBlockDh2 = GetTile(tensorDh2,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.K, params.V));
                        using CopyGmToL1A = typename TileCopyDh2::template CopyGmToL1A<decltype(tensorBlockW)>;
                        using CopyGmToL1B = typename TileCopyDh2::template CopyGmToL1B<decltype(tensorBlockDv2)>;
                        using CopyL0CToGm = typename TileCopyDh2::template CopyL0CToGm<decltype(tensorBlockDh2)>;
                        CopyGmToL1A copyGmToL1A;
                        CopyGmToL1B copyGmToL1B;
                        CopyL0CToGm copyL0CToGm;

                        PipeBarrier<PIPE_ALL>();
                        // load L1A
                        auto tensorL1A = tla::MakeTensor(l1ATensorDh, L1A_LAYOUT_DH1, Arch::PositionL1{});
                        auto tensorGmTileA = GetTile(tensorBlockW, tla::MakeCoord(0, 0), tla::MakeShape(params.K, curBT));
                        
                        copyGmToL1A(tensorL1A, tensorGmTileA);
                        PipeBarrier<PIPE_ALL>();


                        // load L1B
                        auto tensorL1B = tla::MakeTensor(l1BTensorDh, L1B_LAYOUT_DH1, Arch::PositionL1{});
                        auto tensorGmTileB = GetTile(tensorBlockDv2, tla::MakeCoord(0, 0), tla::MakeShape(curBT, params.V));
                        copyGmToL1B(tensorL1B, tensorGmTileB);
                        PipeBarrier<PIPE_ALL>();

                        // copy L1A -> L0A
                        auto layoutAInL0 = tla::MakeLayout<ElementW, LayoutTagL0A_Dh2>(params.K, curBT);
                        auto tensorL0A = tla::MakeTensor(l0ATensorDh, layoutAInL0, Arch::PositionL0A{});
                        auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(params.K, curBT));
                        copyL1ToL0A_Dh2(tensorL0A, tensorTileL1A);
                        PipeBarrier<PIPE_ALL>();

                        // copy L1B -> L0B
                        auto layoutBInL0 = tla::MakeLayout<ElementDv2, LayoutTagL0B_Dh2>(curBT, params.V);
                        auto tensorL0B = tla::MakeTensor(l0BTensorDh, layoutBInL0, Arch::PositionL0B{});
                        auto tensorTileL1B = GetTile(tensorL1B,  tla::MakeCoord(0, 0), tla::MakeShape(curBT, params.V));
                        copyL1ToL0B_Dh2(tensorL0B, tensorTileL1B);
                        PipeBarrier<PIPE_ALL>();

                        bool initC = true; //k方向没有循环
                        uint8_t unitFlag = 0;
                        auto layoutInL0C = tla::MakeLayoutL0C(params.K, params.V);
                        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});
                        auto tensorTileL0C = GetTile(tensorL0C,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.K, params.V));
                        tileMmadDh2(tensorTileL0C, tensorL0A, tensorL0B,
                                    params.K, params.V, curBT, initC, unitFlag);
                        PipeBarrier<PIPE_ALL>();
                        copyL0CToGm(tensorBlockDh2, tensorL0C);
                        PipeBarrier<PIPE_ALL>();
                        CrossCoreSetFlag<0x2, PIPE_FIX>(CROSS_CORE_C2V_TERM2);
                    }
                }
                // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mm2mte1EventId); // 等上次MM结束
                // AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
            }
        }
        return;
    }
    
private:
    AscendC::GlobalTensor<ElementInt> gmCuSeqlens;

    AscendC::GlobalTensor<ElementK> gmK; // [B,H,T,K]
    AscendC::GlobalTensor<ElementDh> gmDh; // [B,H,chunkNum,K,V]
    AscendC::GlobalTensor<ElementDh> gmWsBdv;

    AscendC::GlobalTensor<ElementGq> gmGq;
    AscendC::GlobalTensor<ElementDo> gmDo;
    AscendC::GlobalTensor<ElementDh> gmDhTerm1;
    
    AscendC::GlobalTensor<ElementDh> gmW;
    AscendC::GlobalTensor<ElementDh> gmDv2;
    AscendC::GlobalTensor<ElementDh> gmDhTerm2;

    AscendC::LocalTensor<DT> l1ATensorBdv;
    AscendC::LocalTensor<DT> l1BTensorBdv;
    AscendC::LocalTensor<DT> l0ATensorBdv;
    AscendC::LocalTensor<DT> l0BTensorBdv;
    AscendC::LocalTensor<DT> l1ATensorDh;
    AscendC::LocalTensor<DT> l1BTensorDh;
    AscendC::LocalTensor<DT> l0ATensorDh;
    AscendC::LocalTensor<DT> l0BTensorDh;
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    int32_t l1AEventId = 0;
    int32_t l1BEventId = 1;
    int32_t l0AEventId = 0;
    int32_t l0BEventId = 1;
    int32_t bdvMMEventId = 0; 
    int32_t mm2mte1EventId = 0;

    CopyL1ToL0A_Bdv copyL1ToL0A_Bdv;
    CopyL1ToL0B_Bdv copyL1ToL0B_Bdv;
    CopyL1ToL0A_Dh1 copyL1ToL0A_Dh1;
    CopyL1ToL0A_Dh2 copyL1ToL0A_Dh2;
    CopyL1ToL0B_Dh1 copyL1ToL0B_Dh1;
    CopyL1ToL0B_Dh2 copyL1ToL0B_Dh2;
    TileMmadBdv tileMmadBdv;
    TileMmadDh1 tileMmadDh1;
    TileMmadDh2 tileMmadDh2;
    // CopyGmToL1A copyGmToL1A;
    // CopyGmToL1B copyGmToL1B;
    // CopyL0CToGm copyL0CToGm;
};
}



template <typename DT>
class GDRCube : public GDRBase<DT>
{
public:
    __aicore__ inline GDRCube(GM_ADDR k_, GM_ADDR w_, GM_ADDR dO_, GM_ADDR dh_, GM_ADDR dv2_, GM_ADDR cu_seqlens_, 
                              GM_ADDR chunk_indices_, GM_ADDR workspace_);
    __aicore__ inline void Process();
    __aicore__ inline void Init(const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData);
private:
    GM_ADDR workspaceGq;
    GM_ADDR k;
    GM_ADDR w;
    GM_ADDR dO;
    GM_ADDR dh;
    GM_ADDR dv2;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR workspace;

}; // class GDRCube

template <typename DT>
__aicore__ inline GDRCube<DT>::GDRCube(GM_ADDR k_, GM_ADDR w_, GM_ADDR dO_, GM_ADDR dh_, GM_ADDR dv2_, GM_ADDR cu_seqlens_, 
                                       GM_ADDR chunk_indices_, GM_ADDR workspace_)
:
    k(k_),
    w(w_),
    dO(dO_),
    dh(dh_),
    dv2(dv2_),
    cu_seqlens(cu_seqlens_),
    chunk_indices(chunk_indices_),
    workspace(workspace_)
    {};

template <typename DT>
__aicore__ inline void GDRCube<DT>::Init(const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData)
{
    GDRBase<DT>::InitTilingData(tilingData);
    return;
}

template <typename DT>
__aicore__ inline void GDRCube<DT>::Process()
{
    uint64_t bdvWorkspaceOffset = 0;
    uint64_t gQWorkspaceOffset = this->bdvWs;
    uint64_t bdhTerm1WorkspaceOffset = gQWorkspaceOffset + this->qWs;
    uint64_t bdhTerm2WorkspaceOffset = bdhTerm1WorkspaceOffset + this->qDoWs;
    //输入
    using LayoutTagK = layout::RowMajor;
    using LayoutTagDh = layout::RowMajor;
    using LayoutTagBdv = layout::RowMajor;

    using LayoutTagGq = layout::ColumnMajor; // bt,k -> k, bt
    using LayoutTagDo = layout::RowMajor; // bt,v
    using LayoutTagBdh = layout::RowMajor; // k,v

    using LayoutTagW = layout::ColumnMajor;
    using LayoutTagDv2 = layout::RowMajor;

    using LayoutTagCuSeqlens = layout::RowMajor;
    using LayoutTagChunkIndices = layout::RowMajor;

    using ElementHalf = half;
    using ElementFloat = float;

    //输入
    LayoutTagK tagK = LayoutTagK::MakeLayout<ElementHalf>(this->T, this->K);
    LayoutTagDh tagDh = LayoutTagDh::MakeLayout<ElementHalf>(this->K, this->V);
    LayoutTagBdv tagBdv = LayoutTagBdv::MakeLayout<ElementHalf>(this->chunkSize, this->V);

    LayoutTagGq tagGq = LayoutTagGq::MakeLayout<ElementHalf>(this->chunkSize, this->K);
    LayoutTagDo tagDo = LayoutTagDo::MakeLayout<ElementHalf>(this->T, this->V);
    LayoutTagBdh tagBdh = LayoutTagBdh::MakeLayout<ElementHalf>(this->K, this->V);

    LayoutTagW tagW = LayoutTagW::MakeLayout<ElementHalf>(this->chunkSize, this->K);
    LayoutTagDv2 tagDv2 = LayoutTagDv2::MakeLayout<ElementHalf>(this->T, this->V);

    LayoutTagCuSeqlens tagCuSeqlens = LayoutTagCuSeqlens::MakeLayout<int64_t>(1, this->seqNum + 1);
    LayoutTagChunkIndices tagChunkIndices = LayoutTagChunkIndices::MakeLayout<int64_t>(this->chunkNum, 2);

    using ArchTag = Arch::AtlasA2;

    auto layoutK = MakeLayoutFromTag(tagK);
    auto layoutDh = MakeLayoutFromTag(tagDh);
    auto layoutBdv = MakeLayoutFromTag(tagBdv);

    auto layoutGq = MakeLayoutFromTag(tagGq);
    auto layoutDo = MakeLayoutFromTag(tagDo);
    
    auto layoutW = MakeLayoutFromTag(tagW);
    auto layoutDv2 = MakeLayoutFromTag(tagDv2);

    auto layoutBdh = MakeLayoutFromTag(tagBdh); // term1/term2相同

    using TileCopyBdv =
            Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutTagK, DT, LayoutTagDh, DT, LayoutTagBdv>;
    using L1TileShapeBdv = tla::Shape<_128, _256, _128>; // BT, V, K
    using L0TileShapeBdv = tla::Shape<_128, _256, _128>;

    using TileCopyDh1 = 
            Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutTagGq, DT, LayoutTagDo, DT, LayoutTagBdh>;
    using TileCopyDh2 = 
            Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutTagW, DT, LayoutTagDv2, DT, LayoutTagBdh>;
    using L1TileShapeDh = tla::Shape<_128, _256, _128>; // K,V, BT
    using L0TileShapeDh = tla::Shape<_128, _256, _128>;
    // kernel level
    using GDRKernel = Gemm::Kernel::ChunkGDRBwdDhuTla<ArchTag, DT,
                                                      L1TileShapeBdv, L0TileShapeBdv, TileCopyBdv,
                                                      L1TileShapeDh, L0TileShapeDh, TileCopyDh1, TileCopyDh2>;


    GDRKernel kernel;
    typename GDRKernel::Params param{k, layoutK, dh, layoutDh, workspace, layoutBdv, // k @ dh -> bdv[workspace ]
                                     layoutGq, dO, layoutDo,                // gatedQ^T[workspace ] @ do -> bdh[workspace]
                                     w, layoutW, dv2, layoutDv2, layoutBdh, // w^T @ dv2 -> bdh[workspace] 
                                     cu_seqlens, this->B, this->T, this->H, this->K, this->V, 
                                     this->chunkSize, this->chunkNum, this->seqNum, this->usedCoreNum,
                                     bdvWorkspaceOffset, gQWorkspaceOffset, bdhTerm1WorkspaceOffset, bdhTerm2WorkspaceOffset};
    kernel(param);
    return;
}/*  */