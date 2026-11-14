/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_CUBE_H
#define CHUNK_KDA_BWD_RECOMPUTE_CUBE_H

#include "chunk_kda_bwd_recompute_struct.h"
#include "chunk_kda_bwd_recompute_common.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
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
        int64_t isVariable;
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
        uint32_t bos = 0;
        uint32_t eos = 0;
        const uint64_t groupsPerChunk = (params.Hv + RECOMPUTE_HEADS_PER_GROUP - 1) / RECOMPUTE_HEADS_PER_GROUP;
        const uint64_t groupCount = params.chunkNum * groupsPerChunk;
        for (uint64_t group = coreIdx; group < groupCount; group += coreNumAic) {
            const uint32_t loopIdx = group / groupsPerChunk;
            const uint64_t firstHead = group % groupsPerChunk * RECOMPUTE_HEADS_PER_GROUP;
            const uint32_t page = (group / coreNumAic) % RECOMPUTE_PAGE_COUNT;
            const uint64_t slotElements = params.chunkSize * (params.V + params.K);
            auto pageBase = (__gm__ ElementVb *)params.workspace + KdaRecomputePageOffset(coreIdx, page, slotElements);
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(KDA_BWD_RECOMPUTE_PAGE_READY_BASE + page);
            {
                BlockMmadU blockMmadU(resource);
                AscendC::GlobalTensor<ElementA> gmA;
                AscendC::GlobalTensor<ElementVb> gmVb;
                AscendC::GlobalTensor<ElementU> gmU;
                for (uint64_t slot = 0; slot < RECOMPUTE_HEADS_PER_GROUP && firstHead + slot < params.Hv; ++slot) {
                    const uint64_t h = firstHead + slot;
                    KdaBwdRecomputeGetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.Hv,
                                                  params.T, params.chunkSize, loopIdx, bos, eos, params.isVariable);
                    uint32_t curChunkSize = eos - bos;
                    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA + (h * params.T + bos) * params.chunkSize);
                    auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
                    uint32_t tileN = tla::get<1>(typename BlockMmadU::L1TileShape{});
                    for (uint32_t nOffset = 0; nOffset < params.V; nOffset += tileN) {
                        uint32_t curN = (nOffset + tileN > params.V) ? (params.V - nOffset) : tileN;
                        GemmCoord actualBlockShape{curChunkSize, curN, curChunkSize};
                        gmVb.SetGlobalBuffer(pageBase + slot * slotElements + nOffset);
                        gmU.SetGlobalBuffer((__gm__ ElementU *)params.ptrU + (h * params.T + bos) * params.V + nOffset);
                        auto tensorVb = tla::MakeTensor(gmVb, params.layoutVb, Arch::PositionGM{});
                        auto tensorU = tla::MakeTensor(gmU, params.layoutU, Arch::PositionGM{});
                        auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockVb = GetTile(tensorVb, tla::MakeCoord(0, 0),
                                                     tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        auto tensorBlockU = GetTile(tensorU, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                        blockMmadU(tensorBlockA, tensorBlockVb, tensorBlockU, actualBlockShape);
                    }
                }
            }
            {
                BlockMmadW blockMmadW(resource);
                AscendC::GlobalTensor<ElementA> gmA;
                AscendC::GlobalTensor<ElementKbg> gmKbg;
                AscendC::GlobalTensor<ElementW> gmW;
                for (uint64_t slot = 0; slot < RECOMPUTE_HEADS_PER_GROUP && firstHead + slot < params.Hv; ++slot) {
                    const uint64_t h = firstHead + slot;
                    KdaBwdRecomputeGetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.Hv,
                                                  params.T, params.chunkSize, loopIdx, bos, eos, params.isVariable);
                    uint32_t curChunkSize = eos - bos;
                    GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA + (h * params.T + bos) * params.chunkSize);
                    gmKbg.SetGlobalBuffer(pageBase + slot * slotElements + params.chunkSize * params.V);
                    gmW.SetGlobalBuffer((__gm__ ElementW *)params.ptrW + (h * params.T + bos) * params.K);
                    auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
                    auto tensorKbg = tla::MakeTensor(gmKbg, params.layoutKbg, Arch::PositionGM{});
                    auto tensorW = tla::MakeTensor(gmW, params.layoutW, Arch::PositionGM{});
                    auto tensorBlockA =
                        GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockKbg = GetTile(tensorKbg, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockW =
                        GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmadW(tensorBlockA, tensorBlockKbg, tensorBlockW, actualBlockShape);
                }
            }
            if (group + static_cast<uint64_t>(RECOMPUTE_PAGE_COUNT) * coreNumAic < groupCount) {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(KDA_BWD_RECOMPUTE_PAGE_FREE_BASE + page);
            }
        }
    }
};

template <class... Dims>
using KdaBwdRecomputeGemmTileShape = tla::Shape<Dims...>;

struct KdaBwdRecomputeTileShapes128 {
    using L1TileShape = KdaBwdRecomputeGemmTileShape<_128, _128, _256>;
    using L0TileShape = KdaBwdRecomputeGemmTileShape<_128, _128, _128>;
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

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ArchTag = Arch::Ascend950;
        using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
#else
        using ArchTag = Arch::AtlasA2;
        using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
#endif

        using L1TileShape = typename KdaBwdRecomputeTileShapes128::L1TileShape;
        using L0TileShape = typename KdaBwdRecomputeTileShapes128::L0TileShape;
        using TileCopyU =
            Gemm::Tile::PackedTileCopyTla<ArchTag, QkType, LayoutTagA, QkType, LayoutTagVb, QkType, LayoutTagU>;
        using BlockMmadU =
            Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, QkType, QkType, QkType, void, TileCopyU>;
        using TileCopyW =
            Gemm::Tile::PackedTileCopyTla<ArchTag, QkType, LayoutTagA, QkType, LayoutTagKbg, QkType, LayoutTagW>;
        using BlockMmadW =
            Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, QkType, QkType, QkType, void, TileCopyW>;
        using MatmulKernel = ChunkKdaBwdRecomputeTla<BlockMmadU, BlockMmadW>;

        MatmulKernel kernel;
        typename MatmulKernel::Params param{
            a_, MakeLayoutFromTag(tagA), workspace_, MakeLayoutFromTag(tagVb), u_, MakeLayoutFromTag(tagU),
            MakeLayoutFromTag(tagKbg), w_, MakeLayoutFromTag(tagW),
            cuSeqlens_, chunkIndices_, chunkNum_, B_, Hv_, T_, K_, V_, chunkSize_, usedAic_, isVariable_};
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
    int64_t isVariable_ = 0;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_RECOMPUTE_CUBE_H
