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
    class TileCopyDh_
>
class ChunkGDRBwdDhuTla{
public:
    // using BlockScheduler = BlockScheduler_;
    using ArchTag = ArchTag_;
    using TileCopyBdv = TileCopyBdv_;
    using LayoutK = typename TileCopyBdv::LayoutA;
    using LayoutDh = typename TileCopyBdv::LayoutB;
    using LayoutBdv = typename TileCopyBdv::LayoutC;
    using CopyL1ToL0A = typename TileCopyBdv::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopyBdv::CopyL1ToL0B;
    // using CopyGmToL1A = typename TileCopyBdv::CopyGmToL1A;
    // using CopyGmToL1B = typename TileCopyBdv::CopyGmToL1B;
    // using CopyL0CToGm = typename TileCopyBdv::CopyL0CToGm;
    using LayoutTagL1A = typename TileCopyBdv::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopyBdv::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopyBdv::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopyBdv::LayoutTagL0B;
    using L1AAlignHelper = typename TileCopyBdv::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopyBdv::L1BAlignHelper;

    using ElementK = DT;
    using ElementDh = DT;
    using ElementAccumulator = float;

    using ElementInt = int64_t;
    
    using L1TileShapeBdv = L1TileShapeBdv_;
    using L0TileShapeBdv = L0TileShapeBdv_;
    static constexpr uint32_t L1_TILE_M_BDV = tla::get<0>(L1TileShapeBdv{});
    static constexpr uint32_t L1_TILE_N_BDV = tla::get<1>(L1TileShapeBdv{});
    static constexpr uint32_t L1_TILE_K_BDV = tla::get<2>(L1TileShapeBdv{});
    
    static constexpr uint32_t L0_TILE_M_BDV = tla::get<0>(L0TileShapeBdv{});
    static constexpr uint32_t L0_TILE_N_BDV = tla::get<1>(L0TileShapeBdv{});
    static constexpr uint32_t L0_TILE_K_BDV = tla::get<2>(L0TileShapeBdv{});

    static constexpr auto L1A_LAYOUT_BDV = tla::MakeLayout<ElementK, LayoutTagL1A>(Int<L1_TILE_M_BDV>{}, Int<L1_TILE_K_BDV>{});
    static constexpr auto L1B_LAYOUT_BDV = tla::MakeLayout<ElementDh, LayoutTagL1B>(Int<L1_TILE_K_BDV>{}, Int<L1_TILE_N_BDV>{});

    static constexpr uint32_t L1A_TILE_SIZE_BDV = L1_TILE_M_BDV * L1_TILE_K_BDV * sizeof(ElementK);

    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using DhType = Gemm::GemmType<ElementDh, LayoutDh>;

    using TileMmadBdv = Gemm::Tile::TileMmad<ArchTag, KType, DhType, void>;

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
        GM_ADDR workspace; // gQ from ws
        // LayoutGq layoutGq;
        GM_ADDR dO;
        // LayoutDo layoutDo;
        GM_ADDR k;
        LayoutK layoutK;
        GM_ADDR dh; // from output dh
        LayoutDh layoutDh;
        LayoutBdv layoutBdv;
        GM_ADDR cu_seqlens; 
        GM_ADDR chunk_indices;
        uint64_t B;
        uint64_t T;
        uint64_t H;
        uint64_t K;
        uint64_t V;
        uint64_t BT;
        uint64_t chunkNum;
        uint64_t seqNum;
        uint64_t usedCoreNum;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR workspace_,  GM_ADDR dO_ , 
               GM_ADDR k_, LayoutK layoutK_, GM_ADDR dh_, LayoutDh layoutDh_, LayoutBdv layoutBdv_,
               GM_ADDR cu_seqlens_, 
               uint64_t B_, uint64_t T_, uint64_t H_, uint64_t K_, uint64_t V_,
               uint64_t BT_, uint64_t chunkNum_, uint64_t seqNum_, uint64_t usedCoreNum_): 
            workspace(workspace_), 
            // layoutGq(layoutGq_),
            dO(dO_),
            // layoutDo(layoutDo_),
            k(k_), 
            layoutK(layoutK_),
            dh(dh_),
            layoutDh(layoutDh_),
            layoutBdv(layoutBdv_),
            cu_seqlens(cu_seqlens_),
            B(B_), 
            T(T_), 
            H(H_), 
            K(K_), 
            V(V_), 
            BT(BT_),
            chunkNum(chunkNum_),
            seqNum(seqNum_),
            usedCoreNum(usedCoreNum_)
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
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementK>(0);
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementDh>(L1A_TILE_SIZE_BDV);
        l0ATensor = resource.l0ABuf.template GetBufferByByte<ElementK>(0);
        l0BTensor = resource.l0BBuf.template GetBufferByByte<ElementDh>(0);
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        {
            // >>>>>>>>>>>>>变长：
            // AscendC::CrossCoreWaitFlag<PIPE_MTE2>(SYNC_AIC_AIV_FLAG_0); // 等待前一个循环的Vec执行结束
            AscendC::GlobalTensor<ElementK> gmK; // [B,H,T,K]
            AscendC::GlobalTensor<ElementDh> gmDh; // [B,H,chunkNum,K,V]
            AscendC::GlobalTensor<ElementDh> gmWsBdv;
            AscendC::GlobalTensor<ElementInt> gmCuSeqlens;
            AscendC::GlobalTensor<ElementInt> gmChunkIndices;

            gmCuSeqlens.SetGlobalBuffer((__gm__ ElementInt *)params.cu_seqlens);
            // chunk_indices[seq_idx, chunk_idx]
            // cu_seqlens[0, seq_len_0, seq_len_0 + seq_len_1, ...]

            // 每个核完成一个batch里的一个head的一个seqence里的所有chunk
            uint32_t totalTaskNum = params.B * params.H * params.seqNum; // 2head 3 seqNum
            uint32_t coreIdx = GetBlockIdx();
            
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
                uint64_t curSeqStartOffset = (b * params.H + h) * params.T * params.K + seqStartOffset * params.K;
                uint64_t dhPreBHOffset = (b * params.H + h) * params.chunkNum * params.K * params.V + 
                                        preChunkNum * params.K * params.V;
                for (int32_t chunkIdx = curChunkNum - 1; chunkIdx >= 0; chunkIdx --) {
                    if (chunkIdx == curChunkNum -1) {
                        // skip k_i @ dh_0
                        continue;
                    } else {
                        // init GlobalTensor
                        gmK.SetGlobalBuffer((__gm__ ElementK *)params.k + curSeqStartOffset + chunkIdx * params.BT);
                         // 用的是上一次chunk迭代的结果
                        gmDh.SetGlobalBuffer((__gm__ ElementDh *)params.dh + dhPreBHOffset + (chunkIdx + 1) * params.K * params.V);
                        // printf("gmDh offset is %ld\n", dhPreBHOffset + (chunkIdx + 1) * params.K * params.V);
                        gmWsBdv.SetGlobalBuffer((__gm__ ElementDh *)params.workspace + coreIdx * params.BT * params.V);
                        auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
                        auto tensorDh = tla::MakeTensor(gmDh, params.layoutDh, Arch::PositionGM{});
                        auto tensorBdv = tla::MakeTensor(gmWsBdv, params.layoutBdv, Arch::PositionGM{});
                        // Make tiled views
                        auto tensorBlockK = GetTile(tensorK,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.BT, params.K));
                        auto tensorBlockDh = GetTile(tensorDh,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.K, params.V));
                        auto tensorBlockBdv = GetTile(tensorBdv,
                                                    tla::MakeCoord(0,0),
                                                    tla::MakeShape(params.BT, params.V));
                        using CopyGmToL1A = typename TileCopyBdv::template CopyGmToL1B<decltype(tensorBlockK)>;
                        using CopyGmToL1B = typename TileCopyBdv::template CopyGmToL1B<decltype(tensorBlockDh)>;
                        using CopyL0CToGm = typename TileCopyBdv::template CopyL0CToGm<decltype(tensorBlockBdv)>;
                        CopyGmToL1A copyGmToL1A;
                        CopyGmToL1B copyGmToL1B;
                        CopyL0CToGm copyL0CToGm;
                        // load L1A
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l0AEventId);
                        auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT_BDV, Arch::PositionL1{});
                        auto tensorTileA = GetTile(tensorBlockK, tla::MakeCoord(0, 0), tla::MakeShape(params.BT, params.K));
                        copyGmToL1A(tensorL1A, tensorTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);

                        // load L1B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l0BEventId);
                        auto tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT_BDV, Arch::PositionL1{});
                        auto tensorTileB = GetTile(tensorBlockDh, tla::MakeCoord(0, 0), tla::MakeShape(params.K, params.V));
                        copyGmToL1B(tensorL1B, tensorTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mm2mte1EventId);
                        // copy L1A -> L0A
                        auto layoutAInL0 = tla::MakeLayout<ElementK, LayoutTagL0A>(params.BT, params.K);
                        auto tensorL0A = tla::MakeTensor(l0ATensor, layoutAInL0, Arch::PositionL0A{});
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);
                        
                        auto tensorTileL1A = GetTile(tensorL1A,  tla::MakeCoord(0, 0), tla::MakeShape(params.BT, params.K));
                        copyL1ToL0A(tensorL0A, tensorTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l0AEventId);

                        // copy L1B -> L0B
                        auto layoutBInL0 = tla::MakeLayout<ElementDh, LayoutTagL0B>(params.K, params.V);
                        auto tensorL0B = tla::MakeTensor(l0BTensor, layoutBInL0, Arch::PositionL0B{});
                        auto tensorTileL1B = GetTile(tensorL1B,  tla::MakeCoord(0, 0), tla::MakeShape(params.BT, params.K));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
                        copyL1ToL0B(tensorL0B, tensorTileL1B);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l0BEventId);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(bdvMMEventId);

                        bool initC = true; //k方向没有循环
                        uint8_t unitFlag = 0b00;
                        auto layoutInL0C = tla::MakeLayoutL0C(params.K, params.V);
                        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});
                        auto tensorTileL0C = GetTile(tensorL0C,
                                                     tla::MakeCoord(0,0),
                                                     tla::MakeShape(params.K, params.V));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(bdvMMEventId);
                        tileMmadBdv(tensorTileL0C, tensorL0A, tensorL0B,
                                 params.BT, params.V, params.K, initC, unitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mm2mte1EventId);
                        DumpTensor(gmWsBdv, 128, 128);
                    }
                }
                
            }
        }
        return;
    }
    
private:
    AscendC::LocalTensor<DT> l1ATensor;
    AscendC::LocalTensor<DT> l1BTensor;
    AscendC::LocalTensor<DT> l0ATensor;
    AscendC::LocalTensor<DT> l0BTensor;
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    int32_t l1AEventId = 0;
    int32_t l1BEventId = 1;
    int32_t l0AEventId = 0;
    int32_t l0BEventId = 1;
    int32_t bdvMMEventId = 0; 
    int32_t mm2mte1EventId = 0;

    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    TileMmadBdv tileMmadBdv;
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
       //输入
    using LayoutTagK = layout::RowMajor;
    using LayoutTagDh = layout::RowMajor;
    using LayoutTagBdv = layout::RowMajor;

    using LayoutTagGq = layout::ColumnMajor;
    using LayoutTagDo = layout::RowMajor;

    using LayoutTagW = layout::ColumnMajor;
    using LayoutTagDv2 = layout::RowMajor;

    using LayoutTagCuSeqlens = layout::RowMajor;
    using LayoutTagChunkIndices = layout::RowMajor;

    using ElementHalf = half;
    using ElementFloat = float;

    //输入
    LayoutTagK tagK = LayoutTagK::MakeLayout<ElementHalf>(this->T, this->K);
    LayoutTagDh tagDh = LayoutTagDh::MakeLayout<ElementHalf>(this->K, this->V);
    LayoutTagBdv tagBdv = LayoutTagBdv::MakeLayout<ElementHalf>(this->T, this->V);

    LayoutTagGq tagGq = LayoutTagGq::MakeLayout<ElementHalf>(this->T, this->K);
    LayoutTagDo tagDo = LayoutTagDo::MakeLayout<ElementHalf>(this->T, this->V);

    LayoutTagW tagW = LayoutTagW::MakeLayout<ElementHalf>(this->T, this->K);
    LayoutTagDv2 tagDv2 = LayoutTagDv2::MakeLayout<ElementHalf>(this->T, this->V);

    LayoutTagCuSeqlens tagCuSeqlens = LayoutTagCuSeqlens::MakeLayout<int64_t>(1, this->seqNum + 1);
    LayoutTagChunkIndices tagChunkIndices = LayoutTagChunkIndices::MakeLayout<int64_t>(this->chunkNum, 2);

    using ArchTag = Arch::AtlasA2;

    auto layoutK = MakeLayoutFromTag(tagK);
    auto layoutDh = MakeLayoutFromTag(tagDh);
    auto layoutBdv = MakeLayoutFromTag(tagBdv);


    auto layoutGq = MakeLayoutFromTag(tagGq);
    auto layoutDo = MakeLayoutFromTag(tagDo);

    using TileCopyBdv =
            Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutTagK, DT, LayoutTagDh, DT, LayoutTagBdv>;
    using L1TileShapeBdv = tla::Shape<_128, _256, _128>; // BT, V, K
    using L0TileShapeBdv = tla::Shape<_128, _256, _128>;

    using TileCopyDh = 
            Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutTagGq, DT, LayoutTagDo, DT, LayoutTagDh>;
    using L1TileShapeDh = tla::Shape<_128, _256, _128>; // BT, V, K
    using L0TileShapeDh = tla::Shape<_128, _256, _128>;
    // kernel level
    using GDRKernel = Gemm::Kernel::ChunkGDRBwdDhuTla<ArchTag, DT,
                                                      L1TileShapeBdv, L0TileShapeBdv, TileCopyBdv,
                                                      L1TileShapeDh, L0TileShapeDh, TileCopyDh>;

    GDRKernel kernel;
    typename GDRKernel::Params param{workspace, dO, k, layoutK, dh, layoutDh, layoutBdv, cu_seqlens,
                                     this->B, this->T, this->H, this->K, this->V, this->chunkSize, 
                                     this->chunkNum, this->seqNum, this->usedCoreNum};
    kernel(param);
    return;
}/*  */