/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef SOLVE_TRIL_VECTOR_H
#define SOLVE_TRIL_VECTOR_H

#include "kernel_operator.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "solve_tril_common.h"

#if SOLVE_TRIL_PLATFORM_ASCEND950
// Ascend950: AIV 核不需要，辅助矩阵在 AIC 的 UB 上生成
// 此文件内容被条件编译跳过
#else

namespace NsSolveTril {

using namespace AscendC;

template <int MATRIX_SIZE>
class SolveTrilVector {
    static constexpr int32_t TILE_LEN = MATRIX_SIZE * MATRIX_SIZE;
    static constexpr int32_t NUM_FRACS = MATRIX_SIZE / 16;
    static constexpr int32_t STRIP_LEN = ROWS_PER_AIV_CORE * MATRIX_SIZE;
    static constexpr int32_t NUM_AUX_CORES = NUM_FRACS * 2;
    static constexpr int32_t UB_DIAG_I_OFF = STRIP_LEN;
    static constexpr int32_t UB_DIAG_INEG_OFF = STRIP_LEN + DIAG_BLOCK_ELEMS;
    static constexpr int32_t UB_AIV_ELEMS = STRIP_LEN + 2 * DIAG_BLOCK_ELEMS;
#if SOLVE_TRIL_MBH_UB_OPT
    // UB 优化协作所需常量（须与 cube !950 L1 槽布局完全一致）
    static constexpr int32_t FRAC = 16;
    static constexpr int32_t FRAC_LEN = FRAC * FRAC;
    static constexpr int32_t L1_SLOT_ELEMS = TILE_LEN;
    static constexpr int32_t SLOT_X = 3;
    static constexpr int32_t SLOT_INPUT = 5;
    static constexpr int32_t L1_SLOT_COUNT = 6;
#endif

public:
    __aicore__ inline SolveTrilVector() {}

    __aicore__ inline void Init(GM_ADDR workspace,
                                 GM_ADDR mch_out,
                                 int64_t totalTiles,
                                 int64_t matrixSize,
                                 int64_t tilesPerCore,
                                 int64_t isLower);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GenerateAuxMatrices();
#if SOLVE_TRIL_MBH_UB_OPT
    // UB 优化（AIV 侧）：MCH 输出 GM->UB(nd2nz) 暂存；每层把 xUB_ 的 drv/oth 块清零+提取到 L1。
    __aicore__ inline void CooperateMergeOneTile();
    __aicore__ inline void ExtractFromUB(int32_t dstSlot, int32_t blockSize, int32_t startBlock);
    __aicore__ inline void ClearSlotUB(int32_t slot);
#endif

    TPipe pipe_;
    GlobalTensor<half> workspaceGM_;
#if SOLVE_TRIL_MBH_UB_OPT
    GlobalTensor<half> mchOutGM_;
    // 与 cube 共享：xUB_(VECCALC) 与 l1_(A1) 须各为本核 TPipe 内首个同位分配 -> 物理偏移 0 对齐 cube。
    TBuf<TPosition::VECCALC> xUbBuf_;
    LocalTensor<half> xUB_;
    TBuf<TPosition::A1> l1Buf_;
    LocalTensor<half> l1_;
    TBuf<TPosition::VECCALC> zeroUbBuf_;   // 清 L1 槽用（zeroUB->L1，raw），分配在 xUbBuf_ 之后
    LocalTensor<half> zeroUB_;
#endif
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<half> ub_;

    int64_t totalTiles_;
    int64_t matrixSize_;
    int64_t tilesPerCore_;
    int64_t isLower_;

    Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinish_{SYNC_AIV_AIC_FLAG_SOLVE, SYNC_AIC_AIV_FLAG_SOLVE};
#if SOLVE_TRIL_MBH_UB_OPT
    Catlass::Arch::CrossCoreFlag ubFlagAicReady_{UBOPT_FLAG_AIC_READY};  // AIC -> AIV
    Catlass::Arch::CrossCoreFlag ubFlagAivReady_{UBOPT_FLAG_AIV_READY};  // AIV -> AIC
#endif
};

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilVector<MATRIX_SIZE>::Init(
    GM_ADDR workspace, GM_ADDR mch_out, int64_t totalTiles, int64_t matrixSize,
    int64_t tilesPerCore, int64_t isLower)
{
    totalTiles_ = totalTiles;
    matrixSize_ = matrixSize;
    tilesPerCore_ = tilesPerCore;
    isLower_ = isLower;

    workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace));

#if SOLVE_TRIL_MBH_UB_OPT
    mchOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(mch_out));
    // 顺序关键：xUB_ 与 l1_ 必须各为同位首个分配 -> 物理偏移与 cube 一致。
    pipe_.InitBuffer(xUbBuf_, TILE_LEN * sizeof(half));
    xUB_ = xUbBuf_.Get<half>();
    pipe_.InitBuffer(l1Buf_, L1_SLOT_COUNT * L1_SLOT_ELEMS * sizeof(half));
    l1_ = l1Buf_.Get<half>();
    pipe_.InitBuffer(zeroUbBuf_, TILE_LEN * sizeof(half));
    zeroUB_ = zeroUbBuf_.Get<half>();
#endif
    pipe_.InitBuffer(ubBuf_, UB_AIV_ELEMS * sizeof(half));
    ub_ = ubBuf_.Get<half>();
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilVector<MATRIX_SIZE>::Process()
{
    int32_t subIdx = static_cast<int32_t>(GetSubBlockIdx());
    int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());

    // 辅助矩阵：仅全局 block0 sub0 生成 I/-I/Zero 到共享 GM（AIC PrepareConstants/ClearSlot 用）
    if (blockIdx == 0 && subIdx == 0) {
        GenerateAuxMatrices();
    }
    SyncAll<false>();   // 所有 AIV 核各调用一次（与既有结构一致）

#if SOLVE_TRIL_MBH_UB_OPT
    // UB 协作：仅 sub0；仅 MATRIX_SIZE>FRAC（BT>16 才有 MBH 递归）；仅本组非空闲（与 cube 一致）。
    if (subIdx != 0) return;
    if constexpr (MATRIX_SIZE > 16) {
        // 清零 zeroUB_（一次），供后续清 L1 槽用
        Duplicate(zeroUB_, half(0), TILE_LEN);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        int64_t startTile = static_cast<int64_t>(blockIdx) * tilesPerCore_;
        int64_t endTile = startTile + tilesPerCore_;
        if (endTile > totalTiles_) endTile = totalTiles_;
        if (startTile >= totalTiles_) return;
        for (int64_t t = startTile; t < endTile; t++) {
            CooperateMergeOneTile();
        }
    }
#endif
}

#if SOLVE_TRIL_MBH_UB_OPT
template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilVector<MATRIX_SIZE>::CooperateMergeOneTile()
{
    int32_t drvStart = isLower_ ? 1 : 0;
    int32_t othStart = isLower_ ? 0 : 1;

    // task2：MCH 输出 GM(ND) -> xUB_(UB, NZ)（GM->UB nd2nz；mch_out 为独立 BT×BT，读基址）
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = static_cast<uint32_t>(MATRIX_SIZE);
    nd2nzParams.dValue = static_cast<uint32_t>(MATRIX_SIZE);
    nd2nzParams.srcDValue = MATRIX_SIZE;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(xUB_, mchOutGM_, nd2nzParams);
    PipeBarrier<PIPE_ALL>();

    // 与 cube RecursiveMerge 的层循环一一对应（每层 1 次握手）。
    for (int32_t blockSize = FRAC; blockSize < MATRIX_SIZE; blockSize *= 2) {
        // 等 AIC：常量就绪(L0)/本层结果已 Fixpipe 写回 xUB_(L>=1)
        Catlass::Arch::CrossCoreWaitFlag(ubFlagAicReady_);
        PipeBarrier<PIPE_ALL>();

        // 清 L1 提取目标槽（zeroUB->L1），再写选中分形（xUB->L1，raw）。全部 UB->L1，在 AIV。
        ClearSlotUB(SLOT_X);
        ClearSlotUB(SLOT_INPUT);
        PipeBarrier<PIPE_ALL>();
        ExtractFromUB(SLOT_X, blockSize, drvStart);
        ExtractFromUB(SLOT_INPUT, blockSize, othStart);
        PipeBarrier<PIPE_ALL>();

        // 通知 AIC：drv/oth 已提取到 L1，可做 Mmad
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ubFlagAivReady_);
    }
}

// 清 L1 槽：zeroUB_(UB) -> l1_[slot]（raw UB->L1，整槽置零）。
template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilVector<MATRIX_SIZE>::ClearSlotUB(int32_t slot)
{
    DataCopyParams p;
    p.blockCount = 1;
    p.blockLen = TILE_LEN * sizeof(half) / 32;
    p.srcStride = 0;
    p.dstStride = 0;
    DataCopy(l1_[slot * L1_SLOT_ELEMS], zeroUB_, p);
}

// 从 xUB_(NZ) 按块 raw UB->L1 提取选中对角块到 L1 dstSlot；非选中分形已由 ClearSlotUB 置零。
// xUB_ 与 L1 槽 NZ 布局一致：分形 (fr,fc) 偏移 = (fc*NUM_FRACS+fr)*FRAC_LEN。
template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilVector<MATRIX_SIZE>::ExtractFromUB(
    int32_t dstSlot, int32_t blockSize, int32_t startBlock)
{
    int32_t numBlocks = MATRIX_SIZE / blockSize;
    int32_t fracsPerBlock = blockSize / FRAC;

    DataCopyParams fracParams;
    fracParams.blockCount = 1;
    fracParams.blockLen = FRAC_LEN * sizeof(half) / 32;
    fracParams.srcStride = 0;
    fracParams.dstStride = 0;

    for (int32_t blk = startBlock; blk < numBlocks; blk += 2) {
        for (int32_t fi = 0; fi < fracsPerBlock; fi++) {
            for (int32_t fj = 0; fj < fracsPerBlock; fj++) {
                int32_t fr = blk * fracsPerBlock + fi;
                int32_t fc = blk * fracsPerBlock + fj;
                int32_t off = (fc * NUM_FRACS + fr) * FRAC_LEN;
                DataCopy(l1_[dstSlot * L1_SLOT_ELEMS + off], xUB_[off], fracParams);
            }
        }
    }
}
#endif  // SOLVE_TRIL_MBH_UB_OPT

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilVector<MATRIX_SIZE>::GenerateAuxMatrices()
{
    for (int32_t stripIdx = 0; stripIdx < NUM_AUX_CORES; stripIdx++) {
        Duplicate(ub_, half(0), UB_AIV_ELEMS);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams stripParams;
        stripParams.blockCount = 1;
        stripParams.blockLen = static_cast<uint32_t>(STRIP_LEN * sizeof(half));
        stripParams.srcStride = 0;
        stripParams.dstStride = 0;

        int32_t stripOff = stripIdx * STRIP_LEN;
        DataCopyPad(workspaceGM_[GM_WS_ZERO * TILE_LEN + stripOff], ub_, stripParams);
        DataCopyPad(workspaceGM_[GM_WS_I * TILE_LEN + stripOff], ub_, stripParams);
        DataCopyPad(workspaceGM_[GM_WS_INEG * TILE_LEN + stripOff], ub_, stripParams);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);

        uint64_t diagMask[2] = {
            DIAG_MASK_8X16_EVEN[0],
            DIAG_MASK_8X16_EVEN[1]
        };
        Duplicate(ub_[UB_DIAG_I_OFF], half(1.0f), diagMask, 1, 1, 8);
        Duplicate(ub_[UB_DIAG_INEG_OFF], half(-1.0f), diagMask, 1, 1, 8);
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);

        int32_t rowStart = stripIdx * ROWS_PER_AIV_CORE;
        int32_t colStart = stripIdx * ROWS_PER_AIV_CORE;
        int32_t gmDiagOff = rowStart * MATRIX_SIZE + colStart;

        DataCopyExtParams diagParams;
        diagParams.blockCount = ROWS_PER_AIV_CORE;
        diagParams.blockLen = static_cast<uint32_t>(ROWS_PER_AIV_CORE * sizeof(half));
        diagParams.srcStride = 0;
        diagParams.dstStride = static_cast<uint32_t>((MATRIX_SIZE - ROWS_PER_AIV_CORE) * sizeof(half));

        DataCopyPad(workspaceGM_[GM_WS_I * TILE_LEN + gmDiagOff], ub_[UB_DIAG_I_OFF], diagParams);
        DataCopyPad(workspaceGM_[GM_WS_INEG * TILE_LEN + gmDiagOff], ub_[UB_DIAG_INEG_OFF], diagParams);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}

}  // namespace NsSolveTril

#endif  // SOLVE_TRIL_PLATFORM_ASCEND950
#endif  // SOLVE_TRIL_VECTOR_H
