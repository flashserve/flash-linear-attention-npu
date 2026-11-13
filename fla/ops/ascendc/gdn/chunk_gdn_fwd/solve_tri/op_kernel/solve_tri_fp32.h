/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * BSD 3-Clause License.
 *
 * FP32 implementation for the 64 x 64 solve_tri path.
 *
 * The AIC performs only native FP32 GEMMs. The paired AIV cores prepare and
 * update the MCH blocks, assemble the two sparse block-inverse merges, and
 * cast the final 64 x 64 result back to the input dtype.
 */
#ifndef SOLVE_TRI_FP32_H
#define SOLVE_TRI_FP32_H

#include "kernel_operator.h"
#include <type_traits>
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace NsSolveTri {

using namespace AscendC;

constexpr int32_t FP32_MATRIX_SIZE = 64;
constexpr int32_t FP32_MATRIX_ELEMS = FP32_MATRIX_SIZE * FP32_MATRIX_SIZE;
constexpr int32_t FP32_MATRIX_STRIDE = FP32_MATRIX_SIZE;
constexpr int32_t FP32_SLOT_ELEMS = FP32_MATRIX_SIZE * FP32_MATRIX_STRIDE;
constexpr int32_t FP32_WORKSPACE_SLOTS = 4;
// Number of independent 64x64 tiles kept in flight between two AIC/AIV
// handshakes. The per-tile math is unchanged; batching only amortizes the
// cross-core flag round trips that used to dominate small chunk solves.
constexpr int32_t FP32_TILE_BATCH = 16;
// Four batch windows allow AIV to prepare up to three batches ahead of AIC.
constexpr int32_t FP32_PIPE_BATCHES = 4;

// Each AIV subcore owns two 16x16 diagonal leaves per tile. One batch packs
// TILE_BATCH * 2 leaves so the vector inverse runs at full lane width.
constexpr int32_t FP32_LEAF_SIZE = 16;
constexpr int32_t FP32_LEAF_ELEMS = FP32_LEAF_SIZE * FP32_LEAF_SIZE;
constexpr int32_t FP32_LEAFS_PER_AIV = FP32_TILE_BATCH * 2;
constexpr int32_t FP32_LEAF_BATCH_ELEMS = FP32_LEAFS_PER_AIV * FP32_LEAF_ELEMS;

// X holds the diagonal leaf inverses, TMP/Y hold merge products, and MNEG holds
// the full -A operand consumed by the two merge levels.
        constexpr int32_t FP32_SLOT_X = 0;       // diagonal inverse leaves
        constexpr int32_t FP32_SLOT_TMP = 1;     // 16->32/32->64 merge product
constexpr int32_t FP32_SLOT_MNEG = 2;    // full -M
constexpr int32_t FP32_SLOT_Y = 3;       // running power Y, then merge temporary

// Two independent ready/free pairs. The reverse flag prevents a producer from
// overflowing the hardware flag counter during long, fully asynchronous runs.
constexpr uint64_t FP32_SYNC_AIV_READY = 2;
constexpr uint64_t FP32_SYNC_AIV_FREE = 3;
constexpr uint64_t FP32_SYNC_AIC_READY = 4;
constexpr uint64_t FP32_SYNC_AIC_FREE = 5;
constexpr uint64_t FP32_SYNC_AIV2_READY = 6;
constexpr uint64_t FP32_SYNC_AIV2_FREE = 7;

class SolveTriFp32Base {
protected:
    __aicore__ inline void InitShape(const SolveTriTilingData* tilingData)
    {
        totalTiles_ = tilingData->totalTiles;
        numHeads_ = tilingData->numHeads;
        seqLen_ = tilingData->seqLen;
        batchSize_ = tilingData->batchSize;
        tilesPerCore_ = tilingData->tilesPerCore;
        numChunks_ = tilingData->numChunks;
        lastChunkValidSize_ = tilingData->lastChunkValidSize;
        layoutMode_ = tilingData->layoutMode;
        // 行间步长: BNSD(0)/NTD(3) chunk 内行连续 = 矩阵宽度; BSND(1)/TND(2) = H*矩阵宽度
        rowStride_ = (layoutMode_ == 0 || layoutMode_ == 3)
                         ? FP32_MATRIX_SIZE
                         : numHeads_ * FP32_MATRIX_SIZE;
    }

    __aicore__ inline int64_t GetTileGMOffset(int64_t tileIdx)
    {
        if (layoutMode_ == 3) {
            // NTD 变长格式: [H, total_T, BT]，head 在最外维，chunk 内行连续
            // 遍历顺序与 TND 一致: chunk_global → H (H 变化最快)
            int64_t chunkGlobalIdx = tileIdx / numHeads_;
            int64_t headIdx = tileIdx % numHeads_;
            int64_t seqIdx = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2);
            int64_t chunkInSeq = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2 + 1);
            int64_t bos = cuSeqlensGM_.GetValue(seqIdx);
            return headIdx * seqLen_ * FP32_MATRIX_SIZE +
                   (bos + chunkInSeq * FP32_MATRIX_SIZE) * FP32_MATRIX_SIZE;
        }

        if (layoutMode_ == 2) {
            int64_t chunkGlobalIdx = tileIdx / numHeads_;
            int64_t headIdx = tileIdx % numHeads_;
            int64_t seqIdx = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2);
            int64_t chunkInSeq = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2 + 1);
            int64_t bos = cuSeqlensGM_.GetValue(seqIdx);
            return (bos + chunkInSeq * FP32_MATRIX_SIZE) * numHeads_ * FP32_MATRIX_SIZE +
                   headIdx * FP32_MATRIX_SIZE;
        }

        if (layoutMode_ == 1) {
            int64_t headIdx = tileIdx % numHeads_;
            int64_t chunkIdx = (tileIdx / numHeads_) % numChunks_;
            int64_t batchIdx = tileIdx / (numHeads_ * numChunks_);
            return batchIdx * seqLen_ * numHeads_ * FP32_MATRIX_SIZE +
                   chunkIdx * FP32_MATRIX_SIZE * numHeads_ * FP32_MATRIX_SIZE +
                   headIdx * FP32_MATRIX_SIZE;
        }

        int64_t chunkIdx = tileIdx % numChunks_;
        int64_t headIdx = (tileIdx / numChunks_) % numHeads_;
        int64_t batchIdx = tileIdx / (numChunks_ * numHeads_);
        return batchIdx * numHeads_ * seqLen_ * FP32_MATRIX_SIZE +
               headIdx * seqLen_ * FP32_MATRIX_SIZE +
               chunkIdx * FP32_MATRIX_ELEMS;
    }

    __aicore__ inline int64_t GetTileValidSize(int64_t tileIdx)
    {
        if (layoutMode_ == 2 || layoutMode_ == 3) {
            int64_t chunkGlobalIdx = tileIdx / numHeads_;
            int64_t seqIdx = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2);
            int64_t chunkInSeq = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2 + 1);
            int64_t bos = cuSeqlensGM_.GetValue(seqIdx);
            int64_t eos = cuSeqlensGM_.GetValue(seqIdx + 1);
            int64_t remaining = eos - bos - chunkInSeq * FP32_MATRIX_SIZE;
            return (remaining >= FP32_MATRIX_SIZE) ? FP32_MATRIX_SIZE : remaining;
        }

        int64_t chunkIdx =
            (layoutMode_ == 1) ? (tileIdx / numHeads_) % numChunks_ : tileIdx % numChunks_;
        return (chunkIdx == numChunks_ - 1) ? lastChunkValidSize_ : FP32_MATRIX_SIZE;
    }

    // Each core owns FP32_PIPE_BATCHES * FP32_TILE_BATCH private tile slots.
    // Rebase the view so per-tile helpers operate on the selected slot.
    __aicore__ inline void SelectTileSlot(int64_t core, int64_t batchSlot)
    {
        constexpr int64_t SLOTS_PER_TILE =
            static_cast<int64_t>(FP32_WORKSPACE_SLOTS) * FP32_SLOT_ELEMS;
        constexpr int64_t PER_CORE_SLOTS =
            static_cast<int64_t>(FP32_PIPE_BATCHES) * FP32_TILE_BATCH;
        coreWorkspaceGM_ = workspaceGM_[
            core * (PER_CORE_SLOTS * SLOTS_PER_TILE) + batchSlot * SLOTS_PER_TILE];
    }

    int64_t totalTiles_;
    int64_t numHeads_;
    int64_t seqLen_;
    int64_t batchSize_;
    int64_t tilesPerCore_;
    int64_t numChunks_;
    int64_t lastChunkValidSize_;
    int64_t layoutMode_;
    int64_t rowStride_;
    GlobalTensor<float> workspaceGM_;
    GlobalTensor<float> coreWorkspaceGM_;
    GlobalTensor<int64_t> cuSeqlensGM_;
    GlobalTensor<int64_t> chunkIndicesGM_;
};

template <typename ArchTag, typename TensorSrc, bool StackBlocksByRow>
struct SolveTriPackedBlockGmToL1 {
    template <typename TensorDst, typename TensorTileSrc>
    __aicore__ inline void operator()(
        const TensorDst& dstTensor, const TensorTileSrc& srcTensor)
    {
        using namespace Catlass;
        constexpr int32_t MCH_BLOCK_SIZE = 16;
        int32_t blockCount;
        int32_t blockStride;
        if constexpr (StackBlocksByRow) {
            blockCount = static_cast<int32_t>(
                tla::get<0>(srcTensor.originShape())) / MCH_BLOCK_SIZE;
            blockStride = static_cast<int32_t>(
                tla::get<0, 1>(srcTensor.layout().stride()));
        } else {
            blockCount = static_cast<int32_t>(
                tla::get<1>(srcTensor.originShape())) / MCH_BLOCK_SIZE;
            blockStride = static_cast<int32_t>(
                tla::get<1, 1>(srcTensor.layout().stride()));
        }

        auto blockLayout = tla::MakeLayout(
            tla::MakeShape(MCH_BLOCK_SIZE, MCH_BLOCK_SIZE),
            tla::MakeStride(
                static_cast<int64_t>(FP32_MATRIX_STRIDE), tla::Int<1>{}),
            tla::MakeShape(MCH_BLOCK_SIZE, MCH_BLOCK_SIZE));
        for (int32_t block = 0; block < blockCount; ++block) {
            auto srcBlock = tla::MakeTensor(
                srcTensor.data()[
                    block * blockStride],
                blockLayout,
                Arch::PositionGM{});
            auto dstCoord = StackBlocksByRow
                ? tla::MakeCoord(block * MCH_BLOCK_SIZE, 0)
                : tla::MakeCoord(0, block * MCH_BLOCK_SIZE);
            auto dstBlock = GetTile(
                dstTensor,
                dstCoord,
                tla::MakeShape(MCH_BLOCK_SIZE, MCH_BLOCK_SIZE));
            using BlockCopy = Catlass::Gemm::Tile::TileCopyTla<
                ArchTag, decltype(srcBlock), decltype(dstBlock)>;
            BlockCopy blockCopy;
            blockCopy(dstBlock, srcBlock);
        }
    }
};

template <
    typename ArchTag,
    typename BaseTileCopy,
    typename TensorSrc,
    bool IsMchLayout = (TensorSrc::Layout::depth > 1)>
struct SolveTriBlockCopyASelector {
    using Type =
        typename BaseTileCopy::template CopyGmToL1A<TensorSrc>;
};

template <typename ArchTag, typename BaseTileCopy, typename TensorSrc>
struct SolveTriBlockCopyASelector<ArchTag, BaseTileCopy, TensorSrc, true> {
    using Type = SolveTriPackedBlockGmToL1<ArchTag, TensorSrc, true>;
};

template <
    typename ArchTag,
    typename BaseTileCopy,
    typename TensorSrc,
    bool IsMchLayout = (TensorSrc::Layout::depth > 1)>
struct SolveTriBlockCopyBSelector {
    using Type =
        typename BaseTileCopy::template CopyGmToL1B<TensorSrc>;
};

template <typename ArchTag, typename BaseTileCopy, typename TensorSrc>
struct SolveTriBlockCopyBSelector<ArchTag, BaseTileCopy, TensorSrc, true> {
    using Type = SolveTriPackedBlockGmToL1<ArchTag, TensorSrc, false>;
};

template <typename ArchTag>
struct SolveTriTileCopy
    : Catlass::Gemm::Tile::PackedTileCopyTla<
          ArchTag,
          float,
          Catlass::layout::RowMajor,
          float,
          Catlass::layout::RowMajor,
          float,
          Catlass::layout::RowMajor> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag,
        float,
        Catlass::layout::RowMajor,
        float,
        Catlass::layout::RowMajor,
        float,
        Catlass::layout::RowMajor>;

    template <typename TensorA>
    using CopyGmToL1A =
        typename SolveTriBlockCopyASelector<
            ArchTag, Base, TensorA>::Type;

    template <typename TensorB>
    using CopyGmToL1B =
        typename SolveTriBlockCopyBSelector<
            ArchTag, Base, TensorB>::Type;
};

template <typename T>
class SolveTriCubeFp32 : public SolveTriFp32Base {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                GM_ADDR xOut, GM_ADDR workspace,
                                const SolveTriTilingData* tilingData)
    {
        InitShape(tilingData);
        aicIdx_ = GetBlockIdx();
        workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace)));
        SelectTileSlot(aicIdx_, 0);
        if (layoutMode_ == 2 || layoutMode_ == 3) {
            cuSeqlensGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cuSeqlens));
            chunkIndicesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunkIndices));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t startTile = aicIdx_ * tilesPerCore_;
        int64_t endTile = startTile + tilesPerCore_;
        if (endTile > totalTiles_) {
            endTile = totalTiles_;
        }
        if (startTile >= endTile) {
            return;
        }

        // Native FP32 only. Do not enable the A2/A3 HF32 multiply mode.
        SetHF32Mode(false);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ArchTag = Catlass::Arch::Ascend950;
#else
        using ArchTag = Catlass::Arch::AtlasA2;
#endif
        // Match the fused GDN solve pipeline: UnitFlag lets the fixpipe consume
        // the L0C result without an extra per-GEMM FIX_M event round trip.
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false>;
        using L1TileShape = tla::Shape<tla::_64, tla::_64, tla::_64>;
        using L0TileShape = tla::Shape<tla::_64, tla::_64, tla::_64>;
        using TileCopy = SolveTriTileCopy<ArchTag>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape,
            float, float, float, void, TileCopy>;

        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        const int64_t numBatches =
            (endTile - startTile + FP32_TILE_BATCH - 1) / FP32_TILE_BATCH;
        // Software-pipelined ring: batch b always uses slot group b % PIPE.
        // The AIV keeps PIPE-1 batches prepared ahead, so once the cube starts it
        // runs batch after batch without waiting for the previous batch's
        // cast/store (no per-window drain).
        for (int64_t b = 0; b < numBatches; ++b) {
            const int64_t begin = startTile + b * FP32_TILE_BATCH;
            const int64_t count =
                endTile - begin < FP32_TILE_BATCH ? endTile - begin : FP32_TILE_BATCH;
            const int64_t slotBase = (b % FP32_PIPE_BATCHES) * FP32_TILE_BATCH;

            WaitAiv();  // diagonal inverses and -A published for this batch

            // 16->32 merge products (to TMP/Y, folded back by the AIV).
            for (int64_t i = 0; i < count; ++i) {
                SelectTileSlot(aicIdx_, slotBase + i);
                RunMergeFirstGemm(blockMmad, 16);
            }
            SetFlag<HardEvent::FIX_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID7);
            for (int64_t i = 0; i < count; ++i) {
                SelectTileSlot(aicIdx_, slotBase + i);
                RunMergeSecondGemm(blockMmad, 16);
            }
            SignalAiv();       // 16->32 products ready
            WaitFolded16();    // X now holds the 32x32 block inverses

            for (int64_t i = 0; i < count; ++i) {
                SelectTileSlot(aicIdx_, slotBase + i);
                RunMergeFirstGemm(blockMmad, 32);
            }
            SetFlag<HardEvent::FIX_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID7);
            // Keep the GEMM result separate from its X input until Fixpipe completes.
            for (int64_t i = 0; i < count; ++i) {
                SelectTileSlot(aicIdx_, slotBase + i);
                RunMergeSecondGemm(blockMmad, 32);
            }
            PipeBarrier<PIPE_FIX>();
            SignalAiv();       // 32->64 products ready
        }
    }

private:
    template <typename BlockMmad>
    __aicore__ inline void RunGemmRegion(
        BlockMmad& blockMmad,
        int32_t slotA,
        int32_t slotB,
        int32_t slotC,
        int32_t rowA,
        int32_t colA,
        int32_t rowB,
        int32_t colB,
        int32_t rowC,
        int32_t colC,
        int32_t m,
        int32_t n,
        int32_t k)
    {
        using namespace Catlass;
        auto layout = tla::MakeLayout(
            tla::MakeShape(FP32_MATRIX_SIZE, FP32_MATRIX_SIZE),
            tla::MakeStride(static_cast<int64_t>(FP32_MATRIX_STRIDE), tla::Int<1>{}),
            tla::MakeShape(FP32_MATRIX_SIZE, FP32_MATRIX_SIZE));
        auto tensorA = tla::MakeTensor(
            coreWorkspaceGM_[slotA * FP32_SLOT_ELEMS], layout, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(
            coreWorkspaceGM_[slotB * FP32_SLOT_ELEMS], layout, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(
            coreWorkspaceGM_[slotC * FP32_SLOT_ELEMS], layout, Arch::PositionGM{});
        Catlass::GemmCoord actualShape{
            static_cast<Catlass::GemmCoord::Index>(m),
            static_cast<Catlass::GemmCoord::Index>(n),
            static_cast<Catlass::GemmCoord::Index>(k)};
        auto blockA = GetTile(
            tensorA, tla::MakeCoord(rowA, colA),
            tla::MakeShape(actualShape.m(), actualShape.k()));
        auto blockB = GetTile(
            tensorB, tla::MakeCoord(rowB, colB),
            tla::MakeShape(actualShape.k(), actualShape.n()));
        auto blockC = GetTile(
            tensorC, tla::MakeCoord(rowC, colC),
            tla::MakeShape(actualShape.m(), actualShape.n()));
        blockMmad(blockA, blockB, blockC, actualShape);
    }

    template <typename BlockMmad>
    __aicore__ inline void RunPackedPairGemm(
        BlockMmad& blockMmad,
        int32_t slotA,
        int32_t baseA,
        int32_t blockStrideA,
        int32_t slotB,
        int32_t baseB,
        int32_t blockStrideB,
        int32_t slotC, int32_t baseC = 0)
    {
        using namespace Catlass;
        constexpr int32_t BLOCK_SIZE = 16;
        constexpr int32_t BLOCK_COUNT = 2;
        constexpr int32_t PACKED_SIZE = BLOCK_SIZE * BLOCK_COUNT;

        auto layoutA = tla::MakeLayout(
            tla::MakeShape(
                tla::MakeShape(BLOCK_SIZE, BLOCK_COUNT),
                BLOCK_SIZE),
            tla::MakeStride(
                tla::MakeStride(FP32_MATRIX_STRIDE, blockStrideA),
                1),
            tla::MakeShape(PACKED_SIZE, BLOCK_SIZE));
        auto layoutB = tla::MakeLayout(
            tla::MakeShape(
                BLOCK_SIZE,
                tla::MakeShape(BLOCK_SIZE, BLOCK_COUNT)),
            tla::MakeStride(
                FP32_MATRIX_STRIDE,
                tla::MakeStride(1, blockStrideB)),
            tla::MakeShape(BLOCK_SIZE, PACKED_SIZE));
        auto layoutC = tla::MakeLayout(
            tla::MakeShape(PACKED_SIZE, PACKED_SIZE),
            tla::MakeStride(
                static_cast<int64_t>(FP32_MATRIX_STRIDE), tla::Int<1>{}),
            tla::MakeShape(PACKED_SIZE, PACKED_SIZE));

        auto tensorA = tla::MakeTensor(
            coreWorkspaceGM_[slotA * FP32_SLOT_ELEMS + baseA],
            layoutA,
            Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(
            coreWorkspaceGM_[slotB * FP32_SLOT_ELEMS + baseB],
            layoutB,
            Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(
            coreWorkspaceGM_[slotC * FP32_SLOT_ELEMS + baseC],
            layoutC,
            Arch::PositionGM{});
        Catlass::GemmCoord actualShape{
            PACKED_SIZE, PACKED_SIZE, BLOCK_SIZE};
        blockMmad(tensorA, tensorB, tensorC, actualShape);
    }

    template <typename BlockMmad>
    __aicore__ inline void RunMergeFirstGemm(
        BlockMmad& blockMmad, int32_t blockSize)
    {
        if (blockSize == 16) {
            constexpr int32_t DIAGONAL_BLOCK_1 =
                16 * FP32_MATRIX_STRIDE + 16;
            constexpr int32_t LOWER_BLOCK_1 =
                16 * FP32_MATRIX_STRIDE;
            constexpr int32_t PAIR_STRIDE =
                32 * FP32_MATRIX_STRIDE + 32;
            RunPackedPairGemm(
                blockMmad,
                FP32_SLOT_X,
                DIAGONAL_BLOCK_1,
                PAIR_STRIDE,
                FP32_SLOT_MNEG,
                LOWER_BLOCK_1,
                PAIR_STRIDE,
                FP32_SLOT_TMP);
            return;
        }

        RunGemmRegion(
            blockMmad,
            FP32_SLOT_X,
            FP32_SLOT_MNEG,
            FP32_SLOT_TMP,
            32,
            32,
            32,
            0,
            0,
            0,
            32,
            32,
            32);
    }

    template <typename BlockMmad>
    __aicore__ inline void RunMergeSecondGemm(
        BlockMmad& blockMmad, int32_t blockSize)
    {
        if (blockSize == 16) {
            constexpr int32_t PACKED_DIAGONAL_STRIDE =
                16 * FP32_MATRIX_STRIDE + 16;
            constexpr int32_t EVEN_DIAGONAL_STRIDE =
                32 * FP32_MATRIX_STRIDE + 32;
            RunPackedPairGemm(
                blockMmad,
                FP32_SLOT_TMP,
                0,
                PACKED_DIAGONAL_STRIDE,
                FP32_SLOT_X,
                0,
                EVEN_DIAGONAL_STRIDE,
                FP32_SLOT_Y);
            return;
        }

        RunGemmRegion(
            blockMmad,
            FP32_SLOT_TMP,
            FP32_SLOT_X,
            FP32_SLOT_Y,
            0,
            0,
            0,
            0,
            0,
            0,
            32,
            32,
            32);
    }

    __aicore__ inline void WaitAiv()
    {
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(aivToAic_);
    }

    __aicore__ inline void SignalAiv()
    {
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(aicToAiv_);
    }

    __aicore__ inline void WaitFolded16()
    {
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(aivToAicFold_);
    }

    Catlass::Arch::CrossCoreFlagWithReverse<4096> aivToAicFold_{
        FP32_SYNC_AIV2_READY, FP32_SYNC_AIV2_FREE};
    int64_t aicIdx_;
    Catlass::Arch::CrossCoreFlagWithReverse<4096> aivToAic_{
        FP32_SYNC_AIV_READY, FP32_SYNC_AIV_FREE};
    Catlass::Arch::CrossCoreFlagWithReverse<4096> aicToAiv_{
        FP32_SYNC_AIC_READY, FP32_SYNC_AIC_FREE};
};

template <typename T>
class SolveTriVectorFp32 : public SolveTriFp32Base {
    static constexpr int32_t STRIP_ROWS = FP32_MATRIX_SIZE / 2;
    static constexpr int32_t STRIP_ELEMS = STRIP_ROWS * FP32_MATRIX_SIZE;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                GM_ADDR xOut, GM_ADDR workspace,
                                const SolveTriTilingData* tilingData)
    {
        InitShape(tilingData);
        inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xOut));
        workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace)));

        subBlockIdx_ = GetSubBlockIdx();
        aicIdx_ = GetBlockIdx() / GetSubBlockNum();
        rowBegin_ = subBlockIdx_ * STRIP_ROWS;
        SelectTileSlot(aicIdx_, 0);

        if (layoutMode_ == 2 || layoutMode_ == 3) {
            cuSeqlensGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cuSeqlens));
            chunkIndicesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunkIndices));
        }

        pipe_.InitBuffer(inputBuf_, STRIP_ELEMS * sizeof(T));
        inputLocal_ = inputBuf_.Get<T>();
        pipe_.InitBuffer(fp32BufA_, STRIP_ELEMS * sizeof(float));
        fp32LocalA_ = fp32BufA_.Get<float>();
        pipe_.InitBuffer(fp32BufB_, STRIP_ELEMS * sizeof(float));
        fp32LocalB_ = fp32BufB_.Get<float>();
        pipe_.InitBuffer(fp32BufC_, STRIP_ELEMS * sizeof(float));
        fp32LocalC_ = fp32BufC_.Get<float>();
        // Constant operands are built once per core instead of per tile.
        // A is staged once, then copied into the packed leaf workspace.
        pipe_.InitBuffer(packBuf_, STRIP_ELEMS * sizeof(float));
        packLocal_ = packBuf_.Get<float>();
        pipe_.InitBuffer(leafBuf_, FP32_LEAF_BATCH_ELEMS * sizeof(float));
        leafLocal_ = leafBuf_.Get<float>();
        pipe_.InitBuffer(leafProdBuf_, FP32_LEAF_BATCH_ELEMS * sizeof(float));
        leafProdLocal_ = leafProdBuf_.Get<float>();
        pipe_.InitBuffer(leafRowBuf_, FP32_LEAFS_PER_AIV * FP32_LEAF_SIZE * sizeof(float));
        leafRowLocal_ = leafRowBuf_.Get<float>();
        pipe_.InitBuffer(
            leafCoeffBuf_, FP32_LEAFS_PER_AIV * 8 * FP32_LEAF_SIZE * sizeof(float));
        leafCoeffLocal_ = leafCoeffBuf_.Get<float>();
    }

    __aicore__ inline void Process()
    {
        int64_t startTile = aicIdx_ * tilesPerCore_;
        int64_t endTile = startTile + tilesPerCore_;
        if (endTile > totalTiles_) {
            endTile = totalTiles_;
        }
        if (startTile >= endTile) {
            return;
        }

        const int64_t numBatches =
            (endTile - startTile + FP32_TILE_BATCH - 1) / FP32_TILE_BATCH;
        constexpr int64_t LOOKAHEAD = FP32_PIPE_BATCHES - 1;

        // Prologue: prepare the first LOOKAHEAD batches so the cube can start
        // immediately and keep running while the vector core works behind it.
        for (int64_t b = 0; b < numBatches && b < LOOKAHEAD; ++b) {
            const int64_t begin = startTile + b * FP32_TILE_BATCH;
            const int64_t count =
                endTile - begin < FP32_TILE_BATCH ? endTile - begin : FP32_TILE_BATCH;
            const int64_t slotBase = (b % FP32_PIPE_BATCHES) * FP32_TILE_BATCH;
            PrepareBatch(begin, count, slotBase);
            SignalAic();
        }

        for (int64_t b = 0; b < numBatches; ++b) {
            const int64_t begin = startTile + b * FP32_TILE_BATCH;
            const int64_t count =
                endTile - begin < FP32_TILE_BATCH ? endTile - begin : FP32_TILE_BATCH;
            const int64_t slotBase = (b % FP32_PIPE_BATCHES) * FP32_TILE_BATCH;

            // Fold the 16->32 products into X.
            WaitAic();
            for (int64_t i = 0; i < count; ++i) {
                SelectTileSlot(aicIdx_, slotBase + i);
                Merge16ResultToX();
            }
            SignalFolded16();

            // Prepare the batch LOOKAHEAD ahead while the cube does 32->64.
            const int64_t ahead = b + LOOKAHEAD;
            if (ahead < numBatches) {
                const int64_t abegin = startTile + ahead * FP32_TILE_BATCH;
                const int64_t acount =
                    endTile - abegin < FP32_TILE_BATCH ? endTile - abegin : FP32_TILE_BATCH;
                const int64_t aslotBase = (ahead % FP32_PIPE_BATCHES) * FP32_TILE_BATCH;
                PrepareBatch(abegin, acount, aslotBase);
                SignalAic();
            }

            // Consume the separate 32->64 product directly in the output buffer.
            WaitAic();
            for (int64_t i = 0; i < count; ++i) {
                SelectTileSlot(aicIdx_, slotBase + i);
                CastAndStore(GetTileGMOffset(begin + i), GetTileValidSize(begin + i));
            }
        }
    }

private:
    __aicore__ inline void PrepareBatch(
        int64_t begin, int32_t count, int64_t slotBase)
    {
        Duplicate(leafLocal_, 0.0f, FP32_LEAF_BATCH_ELEMS);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 0; i < count; ++i) {
            SelectTileSlot(aicIdx_, slotBase + i);
            PrepareInput(
                GetTileGMOffset(begin + i), GetTileValidSize(begin + i), i * 2);
        }
        BuildLeafInverses(count, slotBase);
    }

    __aicore__ inline int32_t LocalValidRows(int64_t validSize) const
    {
        int64_t rows = validSize - rowBegin_;
        if (rows <= 0) {
            return 0;
        }
        return (rows >= STRIP_ROWS) ? STRIP_ROWS : static_cast<int32_t>(rows);
    }

    __aicore__ inline void LoadSlot(int32_t slot, const LocalTensor<float>& dst)
    {
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(STRIP_ROWS),
            static_cast<uint32_t>(FP32_MATRIX_SIZE * sizeof(float)),
            static_cast<uint32_t>((FP32_MATRIX_STRIDE - FP32_MATRIX_SIZE) * sizeof(float)),
            0,
            0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(
            dst,
            coreWorkspaceGM_[slot * FP32_SLOT_ELEMS + rowBegin_ * FP32_MATRIX_STRIDE],
            copyParams,
            padParams);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void LoadMergeResult(
        const LocalTensor<float>& dst,
        int32_t srcRow,
        int32_t srcCol,
        int32_t dstRow,
        int32_t dstCol,
        int32_t rows,
        int32_t cols)
    {
        uint32_t blockBytes = cols * sizeof(float);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            blockBytes,
            static_cast<uint32_t>(
                (FP32_MATRIX_SIZE - cols) * sizeof(float)),
            static_cast<uint32_t>(
                (FP32_MATRIX_SIZE - cols) * sizeof(float) / 32),
            0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(
            dst[dstRow * FP32_MATRIX_SIZE + dstCol],
            coreWorkspaceGM_[
                FP32_SLOT_Y * FP32_SLOT_ELEMS +
                srcRow * FP32_MATRIX_STRIDE +
                srcCol],
            copyParams,
            padParams);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void Merge16ResultToX()
    {
        int32_t packedBlock = static_cast<int32_t>(subBlockIdx_);
        // The product fully replaces this 16x16 block, so only the merge
        // result has to reach X; the old block is never read back.
        LoadMergeResult(
            fp32LocalA_,
            packedBlock * 16,
            packedBlock * 16,
            16,
            packedBlock * 32,
            16,
            16);
        StoreMergeBlock(16, packedBlock * 32, rowBegin_ + 16, packedBlock * 32, 16, 16);
    }

    __aicore__ inline void LoadFinalResult()
    {
        if (subBlockIdx_ == 0) {
            LoadSlot(FP32_SLOT_X, fp32LocalA_);
            return;
        }

        // Assemble the lower strip in UB; the final X has no further GM readers.
        constexpr int32_t BLOCK_SIZE = FP32_MATRIX_SIZE / 2;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(STRIP_ROWS),
            static_cast<uint32_t>(BLOCK_SIZE * sizeof(float)),
            static_cast<uint32_t>((FP32_MATRIX_STRIDE - BLOCK_SIZE) * sizeof(float)),
            static_cast<uint32_t>((FP32_MATRIX_SIZE - BLOCK_SIZE) * sizeof(float) / 32),
            0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(
            fp32LocalA_,
            coreWorkspaceGM_[FP32_SLOT_Y * FP32_SLOT_ELEMS],
            copyParams,
            padParams);
        DataCopyPad(
            fp32LocalA_[BLOCK_SIZE],
            coreWorkspaceGM_[
                FP32_SLOT_X * FP32_SLOT_ELEMS + rowBegin_ * FP32_MATRIX_STRIDE + BLOCK_SIZE],
            copyParams,
            padParams);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void StoreMergeBlock(
        int32_t localRow,
        int32_t localCol,
        int32_t dstRow,
        int32_t dstCol,
        int32_t rows,
        int32_t cols)
    {
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(float)),
            static_cast<uint32_t>((FP32_MATRIX_SIZE - cols) * sizeof(float) / 32),
            static_cast<uint32_t>((FP32_MATRIX_SIZE - cols) * sizeof(float)),
            0};
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(
            coreWorkspaceGM_[
                FP32_SLOT_X * FP32_SLOT_ELEMS + dstRow * FP32_MATRIX_STRIDE + dstCol],
            fp32LocalA_[localRow * FP32_MATRIX_SIZE + localCol],
            copyParams);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void PrepareInput(
        int64_t gmOffset, int64_t validSize, int32_t leafOffset)
    {
        Duplicate(inputLocal_, T(0), STRIP_ELEMS);
        PipeBarrier<PIPE_V>();

        int32_t rows = LocalValidRows(validSize);
        if (rows > 0) {
            uint32_t blockBytes = static_cast<uint32_t>(validSize * sizeof(T));
            uint32_t alignedBlockBytes = (blockBytes + 31U) & ~31U;
            uint32_t localRowBytes = FP32_MATRIX_SIZE * sizeof(T);
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rows),
                blockBytes,
                static_cast<uint32_t>((rowStride_ - validSize) * sizeof(T)),
                (localRowBytes - alignedBlockBytes) / 32U,
                0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(
                inputLocal_,
                inputGM_[gmOffset + rowBegin_ * rowStride_],
                copyParams,
                padParams);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
        }

        // Publish -A for both merge levels and collect this subcore's two
        // diagonal leaves for the vectorized inverse below.
        LocalTensor<float> aLocal = packLocal_;
        LocalTensor<float> negLocal = fp32LocalB_;
        if constexpr (std::is_same_v<T, float>) {
            Adds(aLocal, inputLocal_, 0.0f, STRIP_ELEMS);
        } else {
            Cast(aLocal, inputLocal_, RoundMode::CAST_NONE, STRIP_ELEMS);
        }
        PipeBarrier<PIPE_V>();
        Muls(negLocal, aLocal, -1.0f, STRIP_ELEMS);
        Duplicate(fp32LocalC_, 0.0f, STRIP_ELEMS);
        PackLeafInputs(aLocal, leafOffset);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams fullParams{
            static_cast<uint16_t>(STRIP_ROWS),
            static_cast<uint32_t>(FP32_MATRIX_SIZE * sizeof(float)),
            0,
            static_cast<uint32_t>((FP32_MATRIX_STRIDE - FP32_MATRIX_SIZE) * sizeof(float)),
            0};
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(
            coreWorkspaceGM_[
                FP32_SLOT_X * FP32_SLOT_ELEMS + rowBegin_ * FP32_MATRIX_STRIDE],
            fp32LocalC_,
            fullParams);
        DataCopyPad(
            coreWorkspaceGM_[
                FP32_SLOT_MNEG * FP32_SLOT_ELEMS + rowBegin_ * FP32_MATRIX_STRIDE],
            negLocal,
            fullParams);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void PackLeafInputs(
        const LocalTensor<float>& a, int32_t leafOffset)
    {
        for (int32_t leaf = 0; leaf < 2; ++leaf) {
            const int32_t localRow = leaf * FP32_LEAF_SIZE;
            const int32_t localCol =
                static_cast<int32_t>(subBlockIdx_ * 2 + leaf) * FP32_LEAF_SIZE;
            Adds(
                leafLocal_[(leafOffset + leaf) * FP32_LEAF_ELEMS],
                a[localRow * FP32_MATRIX_SIZE + localCol],
                0.0f,
                FP32_LEAF_SIZE,
                FP32_LEAF_SIZE,
                {1, 1, 2, 8});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildLeafInverses(int32_t count, int64_t slotBase)
    {
        // The upper-triangular input is zero, so A^15 is the first nilpotent
        // power. The fused implementation packs all leaves into Vector lanes.
        Muls(leafLocal_, leafLocal_, -1.0f, FP32_LEAF_BATCH_ELEMS);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 2; i < FP32_LEAF_SIZE; ++i) {
            DataCopy(
                leafRowLocal_, leafLocal_[i * FP32_LEAF_SIZE],
                {static_cast<uint16_t>(FP32_LEAFS_PER_AIV), 2, 30, 0});
            PipeBarrier<PIPE_V>();
            Brcb(leafCoeffLocal_, leafRowLocal_, FP32_LEAFS_PER_AIV * 2, {1, 8});
            PipeBarrier<PIPE_V>();
            for (int32_t row = 0; row < FP32_LEAFS_PER_AIV * FP32_LEAF_SIZE; row += 240) {
                const int32_t remaining = FP32_LEAFS_PER_AIV * FP32_LEAF_SIZE - row;
                const uint8_t rows = remaining < 240 ? static_cast<uint8_t>(remaining) : 240;
                Mul(
                    leafProdLocal_[row * FP32_LEAF_SIZE],
                    leafLocal_[row * FP32_LEAF_SIZE],
                    leafCoeffLocal_[row * 8],
                    FP32_LEAF_SIZE,
                    rows,
                    {1, 1, 0, 2, 2, 1});
            }
            PipeBarrier<PIPE_V>();
            Add(
                leafProdLocal_, leafProdLocal_, leafProdLocal_[128],
                64, FP32_LEAFS_PER_AIV, {1, 1, 1, 32, 32, 32});
            Add(
                leafProdLocal_[64], leafProdLocal_[64], leafProdLocal_[192],
                64, FP32_LEAFS_PER_AIV, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(
                leafProdLocal_, leafProdLocal_, leafProdLocal_[64],
                64, FP32_LEAFS_PER_AIV, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(
                leafProdLocal_, leafProdLocal_, leafProdLocal_[32],
                32, FP32_LEAFS_PER_AIV, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(
                leafProdLocal_, leafProdLocal_, leafProdLocal_[16],
                16, FP32_LEAFS_PER_AIV, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(
                leafLocal_[i * FP32_LEAF_SIZE],
                leafLocal_[i * FP32_LEAF_SIZE],
                leafProdLocal_,
                16,
                FP32_LEAFS_PER_AIV,
                {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
        }
        for (int32_t i = 0; i < FP32_LEAF_SIZE; ++i) {
            uint64_t mask[2] = {1ULL << i, 0};
            Duplicate(
                leafLocal_[i * FP32_LEAF_SIZE], 1.0f, mask,
                FP32_LEAFS_PER_AIV, 1, 32);
        }
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(FP32_LEAF_SIZE),
            static_cast<uint32_t>(FP32_LEAF_SIZE * sizeof(float)),
            0,
            static_cast<uint32_t>((FP32_MATRIX_STRIDE - FP32_LEAF_SIZE) * sizeof(float)),
            0};
        for (int32_t tile = 0; tile < count; ++tile) {
            SelectTileSlot(aicIdx_, slotBase + tile);
            for (int32_t leaf = 0; leaf < 2; ++leaf) {
                const int32_t block = static_cast<int32_t>(subBlockIdx_ * 2 + leaf);
                DataCopyPad(
                    coreWorkspaceGM_[
                        FP32_SLOT_X * FP32_SLOT_ELEMS +
                        (rowBegin_ + leaf * FP32_LEAF_SIZE) * FP32_MATRIX_STRIDE +
                        block * FP32_LEAF_SIZE],
                    leafLocal_[(tile * 2 + leaf) * FP32_LEAF_ELEMS],
                    copyParams);
            }
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void CastAndStore(int64_t gmOffset, int64_t validSize)
    {
        LoadFinalResult();
        if constexpr (std::is_same_v<T, float>) {
            Adds(inputLocal_, fp32LocalA_, 0.0f, STRIP_ELEMS);
        } else {
            Cast(inputLocal_, fp32LocalA_, RoundMode::CAST_RINT, STRIP_ELEMS);
        }
        PipeBarrier<PIPE_V>();

        int32_t rows = LocalValidRows(validSize);
        if (rows == 0) {
            return;
        }

        uint32_t blockBytes = static_cast<uint32_t>(validSize * sizeof(T));
        uint32_t alignedBlockBytes = (blockBytes + 31U) & ~31U;
        uint32_t localRowBytes = FP32_MATRIX_SIZE * sizeof(T);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            blockBytes,
            (localRowBytes - alignedBlockBytes) / 32U,
            static_cast<uint32_t>((rowStride_ - validSize) * sizeof(T)),
            0};
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(
            outputGM_[gmOffset + rowBegin_ * rowStride_],
            inputLocal_,
            copyParams);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void WaitAic()
    {
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(aicToAiv_);
    }

    __aicore__ inline void SignalAic()
    {
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(aivToAic_);
    }

    __aicore__ inline void SignalFolded16()
    {
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(aivToAicFold_);
    }

    Catlass::Arch::CrossCoreFlagWithReverse<4096> aivToAicFold_{
        FP32_SYNC_AIV2_READY, FP32_SYNC_AIV2_FREE};

    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> fp32BufA_;
    TBuf<TPosition::VECCALC> fp32BufB_;
    TBuf<TPosition::VECCALC> fp32BufC_;
    TBuf<TPosition::VECCALC> packBuf_;
    TBuf<TPosition::VECCALC> leafBuf_;
    TBuf<TPosition::VECCALC> leafProdBuf_;
    TBuf<TPosition::VECCALC> leafRowBuf_;
    TBuf<TPosition::VECCALC> leafCoeffBuf_;
    LocalTensor<T> inputLocal_;
    LocalTensor<float> fp32LocalA_;
    LocalTensor<float> fp32LocalB_;
    LocalTensor<float> fp32LocalC_;
    LocalTensor<float> packLocal_;
    LocalTensor<float> leafLocal_;
    LocalTensor<float> leafProdLocal_;
    LocalTensor<float> leafRowLocal_;
    LocalTensor<float> leafCoeffLocal_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    int64_t subBlockIdx_;
    int64_t aicIdx_;
    int64_t rowBegin_;
    Catlass::Arch::CrossCoreFlagWithReverse<4096> aivToAic_{
        FP32_SYNC_AIV_READY, FP32_SYNC_AIV_FREE};
    Catlass::Arch::CrossCoreFlagWithReverse<4096> aicToAiv_{
        FP32_SYNC_AIC_READY, FP32_SYNC_AIC_FREE};
};

}  // namespace NsSolveTri

#endif  // SOLVE_TRI_FP32_H
