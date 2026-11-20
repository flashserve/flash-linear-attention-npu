/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_CUBE_H
#define CHUNK_KDA_BWD_RECOMPUTE_CUBE_H

#include "chunk_kda_bwd_recompute_struct.h"
#include "chunk_kda_bwd_recompute_common.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace tla;

namespace KDA {

template <class BlockMmadU_, class BlockMmadW_>
class ChunkKdaBwdRecomputeTla {
public:
    using BlockMmadU = BlockMmadU_;
    using BlockMmadW = BlockMmadW_;
    using ArchTag = typename BlockMmadU::ArchTag;
    using ElementA = typename BlockMmadU::ElementA;
    using LayoutA = typename BlockMmadU::LayoutA;
    using ElementVb = typename BlockMmadU::ElementB;
    using LayoutVb = typename BlockMmadU::LayoutB;
    using ElementU = typename BlockMmadU::ElementC;
    using LayoutU = typename BlockMmadU::LayoutC;
    using ElementKbg = typename BlockMmadW::ElementB;
    using LayoutKbg = typename BlockMmadW::LayoutB;
    using ElementW = typename BlockMmadW::ElementC;
    using LayoutW = typename BlockMmadW::LayoutC;

    struct Params {
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR workspace;
        LayoutVb layoutVb;
        GM_ADDR ptrU;
        LayoutU layoutU;
        LayoutKbg layoutKbg;
        GM_ADDR ptrW;
        LayoutW layoutW;
        GM_ADDR ptrCuSeqLens;
        GM_ADDR ptrChunkIndices;
        uint64_t chunkNum;
        uint64_t B;
        uint64_t Hv;
        uint64_t T;
        uint64_t K;
        uint64_t V;
        uint64_t chunkSize;
        uint64_t usedAic;
        int64_t isVariable;
        uint64_t headGroup;
    };

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Arch::Resource<ArchTag> resource;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNumAic = params.usedAic > 0 ? static_cast<uint32_t>(params.usedAic)
                                                        : AscendC::GetBlockNum();
        if (coreIdx >= coreNumAic) return;
        const uint64_t cg=params.headGroup;
        const uint64_t nh=(params.Hv+cg-1)/cg,total=params.chunkNum*nh;
        for(uint64_t work=coreIdx,ordinal=0;work<total;work+=coreNumAic,++ordinal) {
            uint64_t bos,eos;
            KdaBwdRecomputeGetChunkOffset(params.ptrCuSeqLens,params.ptrChunkIndices,params.B,params.Hv,
                params.T,params.chunkSize,work/nh,bos,eos,params.isVariable);
            uint32_t length=eos-bos;
            uint16_t page=ordinal%2;
            auto base=(__gm__ ElementVb*)params.workspace+KdaRecomputePageOffset(coreIdx,page,16384,cg);
            AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE2>(KDA_BWD_RECOMPUTE_PAGE_READY_BASE+page);
            for(uint64_t slot=0;slot<cg && (work%nh)*cg+slot<params.Hv;++slot) {
                const uint64_t h=(work%nh)*cg+slot;
                // One instance per head: retain A across u/w, reset B residency before page address reuse.
                if(length<64) {
                    auto padding=resource.l1Buf.template GetBufferByByte<int16_t>(0);
                    AscendC::InitConstValueParams<int16_t> init;
                    init.repeatTimes=1;init.blockNum=768;init.dstGap=0;init.initValue=0;
                    AscendC::InitConstValue(padding,init);
                    AscendC::PipeBarrier<PIPE_MTE2>();
                }
                BlockMmadU block(resource);
                AscendC::GlobalTensor<ElementA> ga;
                AscendC::GlobalTensor<ElementVb> gb;
                AscendC::GlobalTensor<ElementU> gc;
                ga.SetGlobalBuffer((__gm__ ElementA*)params.ptrA+(h*params.T+bos)*64);
                auto ta=tla::MakeTensor(ga,params.layoutA,Arch::PositionGM{});
                GemmCoord shape{length,128,length};
                auto at=GetTile(ta,tla::MakeCoord(0U,0U),tla::MakeShape(length,length));
                for(uint32_t op=0;op<2;++op) {
                    gb.SetGlobalBuffer(base+slot*16384+op*8192);
                    gc.SetGlobalBuffer((__gm__ ElementU*)(op?params.ptrW:params.ptrU)+(h*params.T+bos)*128);
                    auto tb=tla::MakeTensor(gb,params.layoutVb,Arch::PositionGM{});
                    auto tc=tla::MakeTensor(gc,params.layoutU,Arch::PositionGM{});
                    auto bt=GetTile(tb,tla::MakeCoord(0U,0U),tla::MakeShape(length,128));
                    auto ct=GetTile(tc,tla::MakeCoord(0U,0U),tla::MakeShape(length,128));
                    block(at,bt,ct,shape);
                }
            }
            if(work+2*coreNumAic<total) {
                AscendC::CrossCoreSetFlag<0x2,PIPE_MTE2>(KDA_BWD_RECOMPUTE_PAGE_FREE_BASE+page);
            }
        }
    }
};

template <class... Dims>
using KdaBwdRecomputeGemmTileShape = tla::Shape<Dims...>;

struct KdaBwdRecomputeTileShapes128 {
    using L1TileShape = KdaBwdRecomputeGemmTileShape<_64, _128, _64>;
    using L0TileShape = KdaBwdRecomputeGemmTileShape<_64, _128, _64>;
};

template <typename QkType>
class ChunkKdaBwdRecomputeCubeProcess {
public:
    __aicore__ inline ChunkKdaBwdRecomputeCubeProcess(
        GM_ADDR a, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace)
        : a_(a), cuSeqlens_(cu_seqlens), chunkIndices_(chunk_indices), w_(w), u_(u), workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkKdaBwdRecomputeTilingData *tiling)
    {
        B_ = static_cast<uint64_t>(tiling->B);
        Hv_ = static_cast<uint64_t>(tiling->Hv);
        T_ = static_cast<uint64_t>(tiling->T);
        K_ = static_cast<uint64_t>(tiling->K);
        V_ = static_cast<uint64_t>(tiling->V);
        chunkNum_ = static_cast<uint64_t>(tiling->chunkNum);
        chunkSize_ = static_cast<uint64_t>(tiling->chunkSize);
        usedAic_ = static_cast<uint64_t>(tiling->usedAic);
        headGroup_ = tiling->headGroup;
        isVariable_ = tiling->isVariable;
    }

    __aicore__ inline void Process()
    {
        using LayoutTagA = layout::RowMajor;
        using LayoutTagKbg = layout::RowMajor;
        using LayoutTagVb = layout::RowMajor;
        using LayoutTagW = layout::RowMajor;
        using LayoutTagU = layout::RowMajor;

        LayoutTagA tagA = LayoutTagA::MakeLayout<QkType>(chunkSize_, chunkSize_);
        LayoutTagKbg tagKbg = LayoutTagKbg::MakeLayout<QkType>(chunkSize_, K_);
        LayoutTagVb tagVb = LayoutTagVb::MakeLayout<QkType>(chunkSize_, V_);
        LayoutTagW tagW = LayoutTagW::MakeLayout<QkType>(chunkSize_, K_);
        LayoutTagU tagU = LayoutTagU::MakeLayout<QkType>(chunkSize_, V_);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        using ArchTag = Arch::Ascend950;
        using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
#else
        using ArchTag = Arch::AtlasA2;
        using DispatchPolicy = Gemm::MmadPingpong<ArchTag, false, false, 1, true, 1, 1, 1, 1>;
#endif

        using L1TileShape = typename KdaBwdRecomputeTileShapes128::L1TileShape;
        using L0TileShape = typename KdaBwdRecomputeTileShapes128::L0TileShape;
        using TileCopyU =
            Gemm::Tile::PackedTileCopyTla<ArchTag, QkType, LayoutTagA, QkType, LayoutTagVb, QkType, LayoutTagU>;
        using BlockMmadU = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, QkType, QkType, QkType,
                                                     void, TileCopyU>;
        using TileCopyW =
            Gemm::Tile::PackedTileCopyTla<ArchTag, QkType, LayoutTagA, QkType, LayoutTagKbg, QkType, LayoutTagW>;
        using BlockMmadW = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, QkType, QkType, QkType,
                                                     void, TileCopyW>;
        using MatmulKernel = ChunkKdaBwdRecomputeTla<BlockMmadU, BlockMmadW>;

        MatmulKernel kernel;
        typename MatmulKernel::Params param{
            a_, MakeLayoutFromTag(tagA), workspace_, MakeLayoutFromTag(tagVb), u_, MakeLayoutFromTag(tagU),
            MakeLayoutFromTag(tagKbg), w_, MakeLayoutFromTag(tagW),
            cuSeqlens_, chunkIndices_, chunkNum_, B_, Hv_, T_, K_, V_, chunkSize_, usedAic_, isVariable_, headGroup_};
        kernel(param);
    }

private:
    GM_ADDR a_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR w_;
    GM_ADDR u_;
    GM_ADDR workspace_;
    uint64_t B_ = 0;
    uint64_t Hv_ = 0;
    uint64_t T_ = 0;
    uint64_t K_ = 0;
    uint64_t V_ = 0;
    uint64_t chunkNum_ = 0;
    uint64_t chunkSize_ = 0;
    uint64_t usedAic_ = 0;
    uint64_t headGroup_ = 4;
    int64_t isVariable_ = 0;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_RECOMPUTE_CUBE_H
