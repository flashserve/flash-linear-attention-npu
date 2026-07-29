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

constexpr int32_t FP32_SLOT_X = 0;       // MCH input A, then the running inverse X
constexpr int32_t FP32_SLOT_Y = 1;       // running power Y, then merge temporary Y
constexpr int32_t FP32_SLOT_TMP = 2;     // GEMM output
constexpr int32_t FP32_SLOT_MNEG = 3;    // full -M

// Two independent ready/free pairs. The reverse flag prevents a producer from
// overflowing the hardware flag counter during long, fully asynchronous runs.
constexpr uint64_t FP32_SYNC_AIV_READY = 2;
constexpr uint64_t FP32_SYNC_AIV_FREE = 3;
constexpr uint64_t FP32_SYNC_AIC_READY = 4;
constexpr uint64_t FP32_SYNC_AIC_FREE = 5;

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
        rowStride_ = (layoutMode_ == 0) ? FP32_MATRIX_SIZE : numHeads_ * FP32_MATRIX_SIZE;
    }

    __aicore__ inline int64_t GetTileGMOffset(int64_t tileIdx)
    {
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
        if (layoutMode_ == 2) {
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

    int64_t totalTiles_;
    int64_t numHeads_;
    int64_t seqLen_;
    int64_t batchSize_;
    int64_t tilesPerCore_;
    int64_t numChunks_;
    int64_t lastChunkValidSize_;
    int64_t layoutMode_;
    int64_t rowStride_;
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
        workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        coreWorkspaceGM_ =
            workspaceGM_[aicIdx_ * FP32_WORKSPACE_SLOTS * FP32_SLOT_ELEMS];
        if (layoutMode_ == 2) {
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
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, false, false>;
        using L1TileShape = tla::Shape<tla::_64, tla::_64, tla::_64>;
        using L0TileShape = tla::Shape<tla::_64, tla::_64, tla::_64>;
        using TileCopy = SolveTriTileCopy<ArchTag>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape,
            float, float, float, void, TileCopy>;

        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        for (int64_t tileIdx = startTile; tileIdx < endTile; ++tileIdx) {
            WaitAiv();

            // MCH initialization: Y=A^2. AIV converts A to X=I-A afterwards.
            RunMchGemm(blockMmad, FP32_SLOT_X, FP32_SLOT_X, FP32_SLOT_Y);
            SignalAiv();
            WaitAiv();

            // X <- X + X*Y, Y <- Y*Y. Three iterations invert each 16x16 MCH block.
            for (int32_t iter = 0; iter < 3; ++iter) {
                RunMchGemm(blockMmad, FP32_SLOT_X, FP32_SLOT_Y, FP32_SLOT_TMP);
                SignalAiv();
                WaitAiv();

                if (iter < 2) {
                    RunMchGemm(blockMmad, FP32_SLOT_Y, FP32_SLOT_Y, FP32_SLOT_TMP);
                    SignalAiv();
                    WaitAiv();
                }
            }

            // Merge 16->32 and 32->64. AIV produces the D/O selected matrices.
            for (int32_t blockSize = 16; blockSize < FP32_MATRIX_SIZE; blockSize *= 2) {
                WaitAiv();

                RunMergeFirstGemm(blockMmad, blockSize);
                SignalAiv();
                WaitAiv();

                RunMergeSecondGemm(blockMmad, blockSize);
                SignalAiv();
                WaitAiv();
            }

            // The AIV casts and writes the final result before this workspace is reused.
            WaitAiv();
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
    __aicore__ inline void RunMchGemm(
        BlockMmad& blockMmad, int32_t slotA, int32_t slotB, int32_t slotC)
    {
        RunDenseGemm(blockMmad, slotA, slotB, slotC);
    }

    template <typename BlockMmad>
    __aicore__ inline void RunDenseGemm(
        BlockMmad& blockMmad, int32_t slotA, int32_t slotB, int32_t slotC)
    {
        RunGemmRegion(
            blockMmad,
            slotA,
            slotB,
            slotC,
            0,
            0,
            0,
            0,
            0,
            0,
            FP32_MATRIX_SIZE,
            FP32_MATRIX_SIZE,
            FP32_MATRIX_SIZE);
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
        int32_t slotC)
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
            coreWorkspaceGM_[slotC * FP32_SLOT_ELEMS],
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

    int64_t aicIdx_;
    GlobalTensor<float> workspaceGM_;
    GlobalTensor<float> coreWorkspaceGM_;
    Catlass::Arch::CrossCoreFlagWithReverse<> aivToAic_{
        FP32_SYNC_AIV_READY, FP32_SYNC_AIV_FREE};
    Catlass::Arch::CrossCoreFlagWithReverse<> aicToAiv_{
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
        workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));

        subBlockIdx_ = GetSubBlockIdx();
        aicIdx_ = GetBlockIdx() / GetSubBlockNum();
        rowBegin_ = subBlockIdx_ * STRIP_ROWS;
        coreWorkspaceGM_ =
            workspaceGM_[aicIdx_ * FP32_WORKSPACE_SLOTS * FP32_SLOT_ELEMS];

        if (layoutMode_ == 2) {
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

        for (int64_t tileIdx = startTile; tileIdx < endTile; ++tileIdx) {
            int64_t gmOffset = GetTileGMOffset(tileIdx);
            int64_t validSize = GetTileValidSize(tileIdx);

            PrepareInput(gmOffset, validSize);
            SignalAic();

            WaitAic();
            InitX(validSize);
            SignalAic();

            for (int32_t iter = 0; iter < 3; ++iter) {
                WaitAic();
                AddProductToX();
                SignalAic();

                if (iter < 2) {
                    WaitAic();
                    CopySlot(FP32_SLOT_TMP, FP32_SLOT_Y);
                    SignalAic();
                }
            }

            for (int32_t blockSize = 16; blockSize < FP32_MATRIX_SIZE; blockSize *= 2) {
                SignalAic();

                WaitAic();
                SignalAic();

                WaitAic();
                MergeResultToX(blockSize);
                SignalAic();
            }

            CastAndStore(gmOffset, validSize);
            SignalAic();
        }
    }

private:
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

    __aicore__ inline void StoreSlot(int32_t slot, const LocalTensor<float>& src)
    {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(STRIP_ROWS),
            static_cast<uint32_t>(FP32_MATRIX_SIZE * sizeof(float)),
            0,
            static_cast<uint32_t>((FP32_MATRIX_STRIDE - FP32_MATRIX_SIZE) * sizeof(float)),
            0};
        DataCopyPad(
            coreWorkspaceGM_[slot * FP32_SLOT_ELEMS + rowBegin_ * FP32_MATRIX_STRIDE],
            src,
            copyParams);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
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

    __aicore__ inline void MergeResultToX(int32_t blockSize)
    {
        if (blockSize == 16) {
            LoadSlot(FP32_SLOT_X, fp32LocalA_);
            int32_t packedBlock = static_cast<int32_t>(subBlockIdx_);
            LoadMergeResult(
                fp32LocalA_,
                packedBlock * 16,
                packedBlock * 16,
                16,
                packedBlock * 32,
                16,
                16);
            StoreSlot(FP32_SLOT_X, fp32LocalA_);
            return;
        }

        if (subBlockIdx_ == 1) {
            LoadSlot(FP32_SLOT_X, fp32LocalA_);
            LoadMergeResult(
                fp32LocalA_,
                0,
                0,
                0,
                0,
                32,
                32);
            StoreSlot(FP32_SLOT_X, fp32LocalA_);
        }
    }

    __aicore__ inline void PrepareInput(int64_t gmOffset, int64_t validSize)
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

        Cast(fp32LocalA_, inputLocal_, RoundMode::CAST_NONE, STRIP_ELEMS);
        PipeBarrier<PIPE_V>();
        Muls(fp32LocalB_, fp32LocalA_, -1.0f, STRIP_ELEMS);
        Duplicate(fp32LocalC_, 0.0f, STRIP_ELEMS);
        PipeBarrier<PIPE_V>();

        // Keep only the four 16x16 diagonal blocks for MCH.
        for (int32_t localRow = 0; localRow < STRIP_ROWS; ++localRow) {
            int32_t globalRow = rowBegin_ + localRow;
            int32_t diagonalBlockStart = (globalRow / 16) * 16;
            Adds(
                fp32LocalC_[localRow * FP32_MATRIX_SIZE + diagonalBlockStart],
                fp32LocalA_[localRow * FP32_MATRIX_SIZE + diagonalBlockStart],
                0.0f,
                16);
        }
        PipeBarrier<PIPE_V>();

        StoreSlot(FP32_SLOT_X, fp32LocalC_);
        StoreSlot(FP32_SLOT_MNEG, fp32LocalB_);
    }

    __aicore__ inline void InitX(int64_t validSize)
    {
        LoadSlot(FP32_SLOT_X, fp32LocalA_);
        Muls(fp32LocalA_, fp32LocalA_, -1.0f, STRIP_ELEMS);
        BuildIdentity(fp32LocalB_, validSize);
        Add(fp32LocalA_, fp32LocalA_, fp32LocalB_, STRIP_ELEMS);
        PipeBarrier<PIPE_V>();
        StoreSlot(FP32_SLOT_X, fp32LocalA_);
    }

    __aicore__ inline void AddProductToX()
    {
        LoadSlot(FP32_SLOT_TMP, fp32LocalA_);
        LoadSlot(FP32_SLOT_X, fp32LocalB_);
        Add(fp32LocalA_, fp32LocalA_, fp32LocalB_, STRIP_ELEMS);
        PipeBarrier<PIPE_V>();
        StoreSlot(FP32_SLOT_X, fp32LocalA_);
    }

    __aicore__ inline void CopySlot(int32_t srcSlot, int32_t dstSlot)
    {
        LoadSlot(srcSlot, fp32LocalA_);
        StoreSlot(dstSlot, fp32LocalA_);
    }

    __aicore__ inline void BuildIdentity(
        const LocalTensor<float>& identity, int64_t validSize)
    {
        Duplicate(identity, 0.0f, STRIP_ELEMS);
        PipeBarrier<PIPE_V>();
        int32_t rows = LocalValidRows(validSize);
        for (int32_t localRow = 0; localRow < rows; ++localRow) {
            int32_t globalRow = rowBegin_ + localRow;
            uint64_t diagonalMask[1] = {1ULL << globalRow};
            Duplicate(
                identity[localRow * FP32_MATRIX_SIZE],
                1.0f,
                diagonalMask,
                1,
                1,
                8);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CastAndStore(int64_t gmOffset, int64_t validSize)
    {
        LoadSlot(FP32_SLOT_X, fp32LocalA_);
        Cast(inputLocal_, fp32LocalA_, RoundMode::CAST_RINT, STRIP_ELEMS);
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

    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> fp32BufA_;
    TBuf<TPosition::VECCALC> fp32BufB_;
    TBuf<TPosition::VECCALC> fp32BufC_;
    LocalTensor<T> inputLocal_;
    LocalTensor<float> fp32LocalA_;
    LocalTensor<float> fp32LocalB_;
    LocalTensor<float> fp32LocalC_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    GlobalTensor<float> workspaceGM_;
    GlobalTensor<float> coreWorkspaceGM_;
    int64_t subBlockIdx_;
    int64_t aicIdx_;
    int64_t rowBegin_;
    Catlass::Arch::CrossCoreFlagWithReverse<> aivToAic_{
        FP32_SYNC_AIV_READY, FP32_SYNC_AIV_FREE};
    Catlass::Arch::CrossCoreFlagWithReverse<> aicToAiv_{
        FP32_SYNC_AIC_READY, FP32_SYNC_AIC_FREE};
};

}  // namespace NsSolveTri

#endif  // SOLVE_TRI_FP32_H
