/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef SOLVE_TRIL_CUBE_H
#define SOLVE_TRIL_CUBE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/cross_core_sync.hpp"

// 内联 l0c_to_gm：L0C -> GM (NZ→ND, FP32→FP16)
namespace NsSolveTril {

__aicore__ inline void L0CToGM(AscendC::GlobalTensor<half> gmTensor,
                                AscendC::LocalTensor<float> l0cTensor,
                                uint32_t mTileActual,
                                uint32_t nTileActual,
                                uint32_t srcStride,
                                uint32_t dstStride)
{
    auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                  mTileActual, // mSize
                                                  srcStride,   // srcStride
                                                  dstStride,   // dstStride
                                                  false);      // enRelu
    intriParams.quantPre = QuantMode_t::F322F16;
    AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
}

}  // namespace NsSolveTril


#include "solve_tril_common.h"

namespace NsSolveTril {

using namespace AscendC;

constexpr int32_t FRAC = 16;
constexpr int32_t FRAC_LEN = FRAC * FRAC;

template <int MATRIX_SIZE>
class SolveTrilCube {
    static constexpr int32_t TILE_LEN = MATRIX_SIZE * MATRIX_SIZE;
    static constexpr int32_t NUM_FRACS = MATRIX_SIZE / FRAC;
    static constexpr int32_t L1_SLOT_ELEMS = TILE_LEN;

    static constexpr int32_t SLOT_INEG = 0;
    static constexpr int32_t SLOT_I = 1;
    static constexpr int32_t SLOT_MNEG = 2;
    static constexpr int32_t SLOT_X = 3;
    static constexpr int32_t SLOT_Y = 4;
    static constexpr int32_t SLOT_INPUT = 5;
    static constexpr int32_t L1_SLOT_COUNT = 6;
    static constexpr int32_t L1_TOTAL_ELEMS = L1_SLOT_COUNT * L1_SLOT_ELEMS;

    static constexpr int32_t EVT_MTE2_MTE1 = 0;
    static constexpr int32_t EVT_MTE1_M = 0;
    static constexpr int32_t EVT_M_MTE1 = 0;
    static constexpr int32_t EVT_M_FIX = 0;
    static constexpr int32_t EVT_FIX_MTE2 = 0;
    static constexpr int32_t EVT_MTE3_MTE2 = 0;
    static constexpr int32_t EVT_MTE2_MTE3 = 0;

public:
    __aicore__ inline SolveTrilCube() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                GM_ADDR x_out, GM_ADDR workspace,
                                const SolveTrilTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneTile(int64_t tileIdx);
    __aicore__ inline int64_t GetTileGMOffset(int64_t tileIdx);
    __aicore__ inline int64_t GetTileValidSize(int64_t tileIdx);
    __aicore__ inline void PrepareConstants();
    __aicore__ inline void LoadInputTile(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
    __aicore__ inline void LoadFullInputForMBH(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
    __aicore__ inline void StoreFinalResult(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
    __aicore__ inline void MCHInvertDiagonal();
    __aicore__ inline void RecursiveMerge();

    __aicore__ inline void ProcessPartialTile(int64_t gmOffset, int64_t validSize);

    __aicore__ inline void MatmulToSlot(int32_t slotA, int32_t slotB, int32_t slotDst, bool initC);
    __aicore__ inline void MatmulToL0C(int32_t slotA, int32_t slotB, bool initC);
    __aicore__ inline void MatmulToL0CTest(int32_t slotA, int32_t slotB, bool initC);
    __aicore__ inline void L0CToSlot(int32_t slotDst);
    __aicore__ inline void ExtractBlocksToSlot(int32_t srcSlot, int32_t dstSlot,
                                                int32_t blockSize, int32_t startBlock);
    __aicore__ inline void ClearSlot(int32_t slot);

private:
    TPipe pipe_;
    GlobalTensor<half> inputGM_;
    GlobalTensor<half> outputGM_;
    GlobalTensor<half> workspaceGM_;
    GlobalTensor<half> scratchGM_;

    TBuf<TPosition::A1> l1Buf_;
    LocalTensor<half> l1_;
    TBuf<TPosition::A2> l0aBuf_;
    LocalTensor<half> l0a_;
    TBuf<TPosition::B2> l0bBuf_;
    LocalTensor<half> l0b_;
    TBuf<TPosition::CO1> l0cBuf_;
    LocalTensor<float> l0c_;

    int64_t totalTiles_;
    int64_t matrixSize_;
    int64_t numHeads_;
    int64_t seqLen_;
    int64_t batchSize_;
    int64_t isLower_;
    int64_t hasCuSeqlens_;
    int64_t tilesPerCore_;
    int64_t aicIdx_;
    int64_t numChunks_;
    int64_t lastChunkValidSize_;
    int64_t isVarlen_;
    int64_t totalChunks_;
    int64_t rowStride_;
    int64_t layoutMode_;
    GlobalTensor<int64_t> cuSeqlensGM_;
    GlobalTensor<int64_t> chunkIndicesGM_;

    Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinish_{SYNC_AIC_AIV_FLAG_SOLVE, SYNC_AIV_AIC_FLAG_SOLVE};
};


// ============ Implementation ============

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::Init(
    GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR x_out, GM_ADDR workspace,
    const SolveTrilTilingData* tilingData)
{
    totalTiles_ = tilingData->totalTiles;
    matrixSize_ = tilingData->matrixSize;
    numHeads_ = tilingData->numHeads;
    seqLen_ = tilingData->seqLen;
    batchSize_ = tilingData->batchSize;
    isLower_ = tilingData->isLower;
    hasCuSeqlens_ = tilingData->hasCuSeqlens;
    tilesPerCore_ = tilingData->tilesPerCore;
    numChunks_ = tilingData->numChunks;
    lastChunkValidSize_ = tilingData->lastChunkValidSize;
    isVarlen_ = tilingData->isVarlen;
    totalChunks_ = tilingData->totalChunks;
    layoutMode_ = tilingData->layoutMode;

    if (layoutMode_ == 0) {
        rowStride_ = matrixSize_;  // BHTD
    } else {
        rowStride_ = numHeads_ * matrixSize_;  // BSND or THD
    }

    aicIdx_ = GetBlockIdx();

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x_out));
    workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace));

    if (isVarlen_) {
        cuSeqlensGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cu_seqlens));
        chunkIndicesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunk_indices));
    }
    int64_t scratchOffset = GM_NUM_SHARED_SLOTS * TILE_LEN + aicIdx_ * TILE_LEN;
    scratchGM_ = workspaceGM_[scratchOffset];

    pipe_.InitBuffer(l1Buf_, L1_TOTAL_ELEMS * sizeof(half));
    l1_ = l1Buf_.Get<half>();
    pipe_.InitBuffer(l0aBuf_, TILE_LEN * sizeof(half));
    l0a_ = l0aBuf_.Get<half>();
    pipe_.InitBuffer(l0bBuf_, TILE_LEN * sizeof(half));
    l0b_ = l0bBuf_.Get<half>();
    pipe_.InitBuffer(l0cBuf_, TILE_LEN * sizeof(float));
    l0c_ = l0cBuf_.Get<float>();
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::Process()
{
    int64_t startTile = aicIdx_ * tilesPerCore_;
    int64_t endTile = startTile + tilesPerCore_;

    if (endTile > totalTiles_) endTile = totalTiles_;
    if (startTile >= totalTiles_) {
        return;
    }

    SyncAll<false>();

    PrepareConstants();

    for (int64_t t = startTile; t < endTile; t++) {
        ProcessOneTile(t);
    }
}

template <int MATRIX_SIZE>
__aicore__ inline int64_t SolveTrilCube<MATRIX_SIZE>::GetTileGMOffset(int64_t tileIdx)
{
    int64_t H = numHeads_;
    int64_t BT = matrixSize_;

    if (layoutMode_ == 2) {
        int64_t chunk_global_idx = tileIdx / H;
        int64_t h = tileIdx % H;

        int64_t seq_idx = chunkIndicesGM_.GetValue(chunk_global_idx * 2);
        int64_t chunk_in_seq = chunkIndicesGM_.GetValue(chunk_global_idx * 2 + 1);

        int64_t bos = cuSeqlensGM_.GetValue(seq_idx);

        return (bos + chunk_in_seq * BT) * H * BT + h * BT;

    } else if (layoutMode_ == 1) {
        int64_t T = seqLen_;

        int64_t h = tileIdx % H;
        int64_t chunk = (tileIdx / H) % numChunks_;
        int64_t b = tileIdx / (H * numChunks_);

        return b * T * H * BT + chunk * BT * H * BT + h * BT;

    } else {
        int64_t T = seqLen_;
        int64_t chunk = tileIdx % numChunks_;
        int64_t h = (tileIdx / numChunks_) % H;
        int64_t b = tileIdx / (numChunks_ * H);
        return b * H * T * BT + h * T * BT + chunk * BT * BT;
    }
}

template <int MATRIX_SIZE>
__aicore__ inline int64_t SolveTrilCube<MATRIX_SIZE>::GetTileValidSize(int64_t tileIdx)
{
    if (layoutMode_ == 2) {
        int64_t H = numHeads_;
        int64_t BT = matrixSize_;
        int64_t chunk_global_idx = tileIdx / H;

        int64_t seq_idx = chunkIndicesGM_.GetValue(chunk_global_idx * 2);
        int64_t chunk_in_seq = chunkIndicesGM_.GetValue(chunk_global_idx * 2 + 1);

        int64_t bos = cuSeqlensGM_.GetValue(seq_idx);
        int64_t eos = cuSeqlensGM_.GetValue(seq_idx + 1);
        int64_t seq_len = eos - bos;

        int64_t chunk_start = chunk_in_seq * BT;
        int64_t remaining = seq_len - chunk_start;
        return (remaining >= BT) ? BT : remaining;
    } else {
        int64_t chunk;
        if (layoutMode_ == 1) {
            chunk = (tileIdx / numHeads_) % numChunks_;
        } else {
            chunk = tileIdx % numChunks_;
        }
        if (chunk == numChunks_ - 1) {
            return lastChunkValidSize_;
        }
        return matrixSize_;
    }
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::ProcessOneTile(int64_t tileIdx)
{
    int64_t gmOffset = GetTileGMOffset(tileIdx);
    int64_t validSize = GetTileValidSize(tileIdx);

    if (validSize < matrixSize_) {
        ProcessPartialTile(gmOffset, validSize);
    } else {
        LoadInputTile(gmOffset);

        MCHInvertDiagonal();

        if constexpr (MATRIX_SIZE > FRAC) {
            LoadFullInputForMBH(gmOffset);
            RecursiveMerge();
        } else {
            MatmulToL0C(SLOT_X, SLOT_I, true);
            SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
            WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
        }
        StoreFinalResult(gmOffset);
    }
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::MatmulToL0C(int32_t slotA, int32_t slotB, bool initC)
{
    LoadData2DParams loadParamsA;
    loadParamsA.startIndex = 0;
    loadParamsA.repeatTimes = NUM_FRACS;
    loadParamsA.srcStride = NUM_FRACS;
    loadParamsA.dstGap = 0;
    loadParamsA.ifTranspose = false;
    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t srcOffsetA = slotA * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dstOffsetA = i * NUM_FRACS * FRAC_LEN;
        LoadData(l0a_[dstOffsetA], l1_[srcOffsetA], loadParamsA);
    }
    LoadData2DParams loadParamsB;
    loadParamsB.startIndex = 0;
    loadParamsB.repeatTimes = NUM_FRACS;
    loadParamsB.srcStride = NUM_FRACS;
    loadParamsB.dstGap = 0;
    loadParamsB.ifTranspose = true;

    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t srcOffsetB = slotB * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dstOffsetB = i * NUM_FRACS * FRAC_LEN;
        LoadData(l0b_[dstOffsetB], l1_[srcOffsetB], loadParamsB);
    }

    SetFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
    WaitFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
    MmadParams mmadParams;
    mmadParams.m = MATRIX_SIZE;
    mmadParams.n = MATRIX_SIZE;
    mmadParams.k = MATRIX_SIZE;
    mmadParams.cmatrixInitVal = initC;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0;
    Mmad(l0c_, l0a_, l0b_, mmadParams);
}


template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::MatmulToL0CTest(int32_t slotA, int32_t slotB, bool initC)
{
    LoadData2DParams loadParamsA;
    loadParamsA.startIndex = 0;
    loadParamsA.repeatTimes = NUM_FRACS;
    loadParamsA.srcStride = NUM_FRACS;
    loadParamsA.dstGap = 0;
    loadParamsA.ifTranspose = false;

    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t srcOffsetA = slotA * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dstOffsetA = i * NUM_FRACS * FRAC_LEN;
        LoadData(l0a_[dstOffsetA], l1_[srcOffsetA], loadParamsA);
    }

    LoadData2DParams loadParamsB;
    loadParamsB.startIndex = 0;
    loadParamsB.repeatTimes = NUM_FRACS;
    loadParamsB.srcStride = NUM_FRACS;
    loadParamsB.dstGap = 0;
    loadParamsB.ifTranspose = true;

    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t srcOffsetB = slotB * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dstOffsetB = i * NUM_FRACS * FRAC_LEN;
        LoadData(l0b_[dstOffsetB], l1_[srcOffsetB], loadParamsB);
    }

    SetFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
    WaitFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
    MmadParams mmadParams;
    mmadParams.m = MATRIX_SIZE;
    mmadParams.n = MATRIX_SIZE;
    mmadParams.k = MATRIX_SIZE;
    mmadParams.cmatrixInitVal = initC;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0;
    Mmad(l0c_, l0a_, l0b_, mmadParams);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::L0CToSlot(int32_t slotDst)
{
    int32_t rowStride = MATRIX_SIZE;
    NsSolveTril::L0CToGM(
        scratchGM_,
        l0c_,
        MATRIX_SIZE,
        MATRIX_SIZE,
        MATRIX_SIZE,
        rowStride
    );
    SetFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
    WaitFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = MATRIX_SIZE;
    nd2nzParams.dValue = MATRIX_SIZE;
    nd2nzParams.srcDValue = rowStride;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(l1_[slotDst * L1_SLOT_ELEMS], scratchGM_, nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::MatmulToSlot(
    int32_t slotA, int32_t slotB, int32_t slotDst, bool initC)
{
    MatmulToL0C(slotA, slotB, initC);
    SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
    WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
    L0CToSlot(slotDst);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::ClearSlot(int32_t slot)
{
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = TILE_LEN * sizeof(half) / 32;
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopy(l1_[slot * L1_SLOT_ELEMS], workspaceGM_[GM_WS_ZERO * TILE_LEN], params);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::ExtractBlocksToSlot(
    int32_t srcSlot, int32_t dstSlot, int32_t blockSize, int32_t startBlock)
{
    int32_t numBlocks = MATRIX_SIZE / blockSize;
    int32_t fracsPerBlock = blockSize / FRAC;

    ClearSlot(dstSlot);
    SetFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    copyParams.blockLen = FRAC_LEN * sizeof(half) / 32;
    for (int32_t blk = startBlock; blk < numBlocks; blk += 2) {
        for (int32_t fi = 0; fi < fracsPerBlock; fi++) {
            for (int32_t fj = 0; fj < fracsPerBlock; fj++) {
                int32_t row = blk * fracsPerBlock + fi;
                int32_t col = blk * fracsPerBlock + fj;
                int32_t off = (col * NUM_FRACS + row) * FRAC_LEN;
                DataCopy(scratchGM_, l1_[srcSlot * L1_SLOT_ELEMS + off], copyParams);
                SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                DataCopy(l1_[dstSlot * L1_SLOT_ELEMS + off], scratchGM_, copyParams);
                SetFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
                WaitFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
            }
        }
    }
}


template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::PrepareConstants()
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = MATRIX_SIZE;
    nd2nzParams.dValue = MATRIX_SIZE;
    nd2nzParams.srcDValue = MATRIX_SIZE;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(l1_[SLOT_I * L1_SLOT_ELEMS], workspaceGM_[GM_WS_I * TILE_LEN], nd2nzParams);
    DataCopy(l1_[SLOT_INEG * L1_SLOT_ELEMS], workspaceGM_[GM_WS_INEG * TILE_LEN], nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::LoadInputTile(int64_t gmOffset, int64_t validSize)
{
    ClearSlot(SLOT_INPUT);
    PipeBarrier<PIPE_MTE2>();

    int32_t numDiagFracs = (static_cast<int32_t>(validSize) + FRAC - 1) / FRAC;

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = numDiagFracs;
    nd2nzParams.nValue = FRAC;
    nd2nzParams.dValue = FRAC;
    nd2nzParams.srcDValue = static_cast<uint32_t>(rowStride_);
    nd2nzParams.srcNdMatrixStride = FRAC * static_cast<int32_t>(rowStride_) + FRAC;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = FRAC;
    nd2nzParams.dstNzMatrixStride = (NUM_FRACS + 1) * FRAC_LEN;

    DataCopy(l1_[SLOT_INPUT * L1_SLOT_ELEMS], inputGM_[gmOffset], nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::StoreFinalResult(int64_t gmOffset, int64_t validSize)
{
    NsSolveTril::L0CToGM(
        outputGM_[gmOffset],
        l0c_,
        static_cast<uint32_t>(validSize),
        static_cast<uint32_t>(validSize),
        MATRIX_SIZE,
        static_cast<uint32_t>(rowStride_)
    );
    PipeBarrier<PIPE_FIX>();
}


template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::MCHInvertDiagonal()
{
    MatmulToSlot(SLOT_INPUT, SLOT_INPUT, SLOT_Y, true);
    PipeBarrier<PIPE_ALL>();

    MatmulToL0C(SLOT_I, SLOT_I, true);
    SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    MatmulToSlot(SLOT_INEG, SLOT_INPUT, SLOT_X, false);

    constexpr int32_t NUM_ITERS = 3;
    for (int32_t iter = 0; iter < NUM_ITERS - 1; iter++) {
        MatmulToL0C(SLOT_X, SLOT_Y, true);
        SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
        WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
        SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
        WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
        MatmulToSlot(SLOT_X, SLOT_I, SLOT_X, false);
        MatmulToSlot(SLOT_Y, SLOT_Y, SLOT_Y, true);
    }
    MatmulToL0C(SLOT_X, SLOT_Y, true);
    SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    MatmulToSlot(SLOT_X, SLOT_I, SLOT_X, false);
}


template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::LoadFullInputForMBH(int64_t gmOffset, int64_t validSize)
{
    ClearSlot(SLOT_INPUT);
    PipeBarrier<PIPE_MTE2>();

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = static_cast<uint32_t>(validSize);
    nd2nzParams.dValue = static_cast<uint32_t>(validSize);
    nd2nzParams.srcDValue = static_cast<uint32_t>(rowStride_);
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(l1_[SLOT_INPUT * L1_SLOT_ELEMS], inputGM_[gmOffset], nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);

    MatmulToSlot(SLOT_INEG, SLOT_INPUT, SLOT_MNEG, true);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::RecursiveMerge()
{
    for (int32_t blockSize = FRAC; blockSize < MATRIX_SIZE; blockSize *= 2) {
        int32_t drvStart = isLower_ ? 1 : 0;
        int32_t othStart = isLower_ ? 0 : 1;

        ExtractBlocksToSlot(SLOT_X, SLOT_Y, blockSize, drvStart);

        MatmulToL0C(SLOT_I, SLOT_I, true);
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
        MatmulToSlot(SLOT_Y, SLOT_MNEG, SLOT_Y, false);

        ExtractBlocksToSlot(SLOT_X, SLOT_INPUT, blockSize, othStart);
        SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);

        MatmulToL0C(SLOT_Y, SLOT_INPUT, true);
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
        SetFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
        WaitFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
        ExtractBlocksToSlot(SLOT_X, SLOT_INPUT, blockSize, drvStart);
        SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
        MatmulToL0C(SLOT_I, SLOT_INPUT, false);
        SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);

        if (blockSize < MATRIX_SIZE / 2) {
            L0CToSlot(SLOT_X);
            SetFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
        }
    }
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::ProcessPartialTile(int64_t gmOffset, int64_t validSize)
{
    LoadInputTile(gmOffset, validSize);

    MCHInvertDiagonal();

    if constexpr (MATRIX_SIZE > FRAC) {
        LoadFullInputForMBH(gmOffset, validSize);
        RecursiveMerge();
    } else {
        MatmulToL0C(SLOT_X, SLOT_I, true);
        SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
    }

    StoreFinalResult(gmOffset, validSize);
}

}  // namespace NsSolveTril

#endif  // SOLVE_TRIL_CUBE_H
