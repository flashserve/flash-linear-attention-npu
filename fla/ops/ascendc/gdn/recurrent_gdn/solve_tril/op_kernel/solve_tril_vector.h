/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef SOLVE_TRIL_VECTOR_H
#define SOLVE_TRIL_VECTOR_H

#include "kernel_operator.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "solve_tril_common.h"

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

public:
    __aicore__ inline SolveTrilVector() {}

    __aicore__ inline void Init(GM_ADDR workspace,
                                 int64_t totalTiles,
                                 int64_t matrixSize);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GenerateAuxMatrices();

    TPipe pipe_;
    GlobalTensor<half> workspaceGM_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<half> ub_;

    int64_t totalTiles_;
    int64_t matrixSize_;

    Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinish_{SYNC_AIV_AIC_FLAG_SOLVE, SYNC_AIC_AIV_FLAG_SOLVE};
};

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilVector<MATRIX_SIZE>::Init(
    GM_ADDR workspace, int64_t totalTiles, int64_t matrixSize)
{
    totalTiles_ = totalTiles;
    matrixSize_ = matrixSize;

    workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace));

    pipe_.InitBuffer(ubBuf_, UB_AIV_ELEMS * sizeof(half));
    ub_ = ubBuf_.Get<half>();
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilVector<MATRIX_SIZE>::Process()
{
    int32_t subIdx = static_cast<int32_t>(GetSubBlockIdx());
    int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
    if (subIdx != 0 || blockIdx != 0) {
        SyncAll<false>();
        return;
    }

    GenerateAuxMatrices();

    SyncAll<false>();
}

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

#endif  // SOLVE_TRIL_VECTOR_H
