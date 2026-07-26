/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * BSD 3-Clause License.
 *
 * FP32 implementation for the 64 x 64 solve_tri path.
 *
 * The AIC performs only native FP32 GEMMs. The paired AIV cores prepare the
 * MCH blocks, apply the FP32 additions/masks between GEMMs, and cast the final
 * merged 64 x 64 result back to the input dtype.
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
constexpr int32_t FP32_MATRIX_STRIDE = 128;
constexpr int32_t FP32_SLOT_ELEMS = FP32_MATRIX_SIZE * FP32_MATRIX_STRIDE;
constexpr int32_t FP32_WORKSPACE_SLOTS = 6;

constexpr int32_t FP32_SLOT_X = 0;       // MCH input A, then the running inverse X
constexpr int32_t FP32_SLOT_Y = 1;       // running power Y, then merge temporary Y
constexpr int32_t FP32_SLOT_TMP = 2;     // GEMM output
constexpr int32_t FP32_SLOT_MNEG = 3;    // full -M
constexpr int32_t FP32_SLOT_DRIVE = 4;   // selected diagonal blocks that drive a merge
constexpr int32_t FP32_SLOT_OTHER = 5;   // the other selected diagonal blocks

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
        isLower_ = tilingData->isLower;
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
    int64_t isLower_;
    int64_t tilesPerCore_;
    int64_t numChunks_;
    int64_t lastChunkValidSize_;
    int64_t layoutMode_;
    int64_t rowStride_;
    GlobalTensor<int64_t> cuSeqlensGM_;
    GlobalTensor<int64_t> chunkIndicesGM_;
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
        using L1TileShape = tla::Shape<tla::_128, tla::_128, tla::_64>;
        using L0TileShape = tla::Shape<tla::_128, tla::_128, tla::_64>;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, float, Catlass::layout::RowMajor,
            float, Catlass::layout::RowMajor,
            float, Catlass::layout::RowMajor>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape,
            float, float, float, void, TileCopy>;

        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        for (int64_t tileIdx = startTile; tileIdx < endTile; ++tileIdx) {
            WaitAiv();

            // MCH initialization: Y=A^2. AIV converts A to X=I-A afterwards.
            RunGemm(blockMmad, FP32_SLOT_X, FP32_SLOT_X, FP32_SLOT_Y);
            SignalAiv();
            WaitAiv();

            // X <- X + X*Y, Y <- Y*Y. Three iterations invert each 16x16 MCH block.
            for (int32_t iter = 0; iter < 3; ++iter) {
                RunGemm(blockMmad, FP32_SLOT_X, FP32_SLOT_Y, FP32_SLOT_TMP);
                SignalAiv();
                WaitAiv();

                if (iter < 2) {
                    RunGemm(blockMmad, FP32_SLOT_Y, FP32_SLOT_Y, FP32_SLOT_TMP);
                    SignalAiv();
                    WaitAiv();
                }
            }

            // Merge 16->32 and 32->64. AIV produces the D/O selected matrices.
            for (int32_t blockSize = 16; blockSize < FP32_MATRIX_SIZE; blockSize *= 2) {
                WaitAiv();

                RunGemm(blockMmad, FP32_SLOT_DRIVE, FP32_SLOT_MNEG, FP32_SLOT_TMP);
                SignalAiv();
                WaitAiv();

                RunGemm(blockMmad, FP32_SLOT_Y, FP32_SLOT_OTHER, FP32_SLOT_TMP);
                SignalAiv();
                WaitAiv();
            }

            // The AIV casts and writes the final result before this workspace is reused.
            WaitAiv();
        }
    }

private:
    template <typename BlockMmad>
    __aicore__ inline void RunGemm(
        BlockMmad& blockMmad, int32_t slotA, int32_t slotB, int32_t slotC)
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
            FP32_MATRIX_SIZE, FP32_MATRIX_SIZE, FP32_MATRIX_SIZE};
        auto blockA = GetTile(
            tensorA, tla::MakeCoord(0, 0),
            tla::MakeShape(actualShape.m(), actualShape.k()));
        auto blockB = GetTile(
            tensorB, tla::MakeCoord(0, 0),
            tla::MakeShape(actualShape.k(), actualShape.n()));
        auto blockC = GetTile(
            tensorC, tla::MakeCoord(0, 0),
            tla::MakeShape(actualShape.m(), actualShape.n()));
        blockMmad(blockA, blockB, blockC, actualShape);
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
                SplitX(blockSize);
                SignalAic();

                WaitAic();
                AddIdentity(validSize);
                SignalAic();

                WaitAic();
                AddDrivingToX();
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

    __aicore__ inline void SplitX(int32_t blockSize)
    {
        LoadSlot(FP32_SLOT_X, fp32LocalA_);
        Duplicate(fp32LocalB_, 0.0f, STRIP_ELEMS);
        Duplicate(fp32LocalC_, 0.0f, STRIP_ELEMS);
        PipeBarrier<PIPE_V>();

        int32_t driveParity = isLower_ ? 1 : 0;
        for (int32_t localRow = 0; localRow < STRIP_ROWS; ++localRow) {
            int32_t globalRow = rowBegin_ + localRow;
            bool rowDrives = ((globalRow / blockSize) & 1) == driveParity;
            for (int32_t col = 0; col < FP32_MATRIX_SIZE; col += blockSize) {
                bool colDrives = ((col / blockSize) & 1) == driveParity;
                if (rowDrives && colDrives) {
                    Adds(
                        fp32LocalB_[localRow * FP32_MATRIX_SIZE + col],
                        fp32LocalA_[localRow * FP32_MATRIX_SIZE + col],
                        0.0f,
                        blockSize);
                } else if (!rowDrives && !colDrives) {
                    Adds(
                        fp32LocalC_[localRow * FP32_MATRIX_SIZE + col],
                        fp32LocalA_[localRow * FP32_MATRIX_SIZE + col],
                        0.0f,
                        blockSize);
                }
            }
        }
        PipeBarrier<PIPE_V>();
        StoreSlot(FP32_SLOT_DRIVE, fp32LocalB_);
        StoreSlot(FP32_SLOT_OTHER, fp32LocalC_);
    }

    __aicore__ inline void AddIdentity(int64_t validSize)
    {
        LoadSlot(FP32_SLOT_TMP, fp32LocalA_);
        BuildIdentity(fp32LocalB_, validSize);
        Add(fp32LocalA_, fp32LocalA_, fp32LocalB_, STRIP_ELEMS);
        PipeBarrier<PIPE_V>();
        StoreSlot(FP32_SLOT_Y, fp32LocalA_);
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

    __aicore__ inline void AddDrivingToX()
    {
        LoadSlot(FP32_SLOT_TMP, fp32LocalA_);
        LoadSlot(FP32_SLOT_DRIVE, fp32LocalB_);
        Add(fp32LocalA_, fp32LocalA_, fp32LocalB_, STRIP_ELEMS);
        PipeBarrier<PIPE_V>();
        StoreSlot(FP32_SLOT_X, fp32LocalA_);
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
