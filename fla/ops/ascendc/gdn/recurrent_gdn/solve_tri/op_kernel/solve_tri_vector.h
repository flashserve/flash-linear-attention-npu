/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * BSD 3-Clause License.
 * 
 * SolveTri Vector 部分 - AIV 核执行
 * 负责在 UB 中生成辅助矩阵 (I, -I, ZERO)，然后写到 GM（ND 格式）
 */
 #ifndef SOLVE_TRI_VECTOR_H
 #define SOLVE_TRI_VECTOR_H
 
 #include "kernel_operator.h"
 #include "catlass/arch/cross_core_sync.hpp"
 #include "solve_tri_common.h"
 
 namespace NsSolveTri {
 
 using namespace AscendC;
 
template <int MATRIX_SIZE, typename T = half>
class SolveTriVector {
     static constexpr int32_t TILE_LEN = MATRIX_SIZE * MATRIX_SIZE;
     static constexpr int32_t NUM_FRACS = MATRIX_SIZE / 16;
     static constexpr int32_t STRIP_LEN = ROWS_PER_AIV_CORE * MATRIX_SIZE;
     static constexpr int32_t NUM_AUX_CORES = NUM_FRACS * 2;
     static constexpr int32_t UB_DIAG_I_OFF = STRIP_LEN;
     static constexpr int32_t UB_DIAG_INEG_OFF = STRIP_LEN + DIAG_BLOCK_ELEMS;
     static constexpr int32_t UB_AIV_ELEMS = STRIP_LEN + 2 * DIAG_BLOCK_ELEMS;
 
 public:
     __aicore__ inline SolveTriVector() {}
     
     __aicore__ inline void Init(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                  GM_ADDR x_out, GM_ADDR workspace,
                                  const SolveTriTilingData* tilingData);
     __aicore__ inline void Process();
 
 private:
     __aicore__ inline void GenerateAuxMatrices();
     __aicore__ inline void MaskInputToOutput();
     __aicore__ inline int64_t GetTileGMOffset(int64_t tileIdx);
     __aicore__ inline int64_t GetTileValidSize(int64_t tileIdx);
 
     TPipe pipe_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    GlobalTensor<T> workspaceGM_;
    GlobalTensor<int64_t> cuSeqlensGM_;
    GlobalTensor<int64_t> chunkIndicesGM_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<T> ub_;
 
     int64_t totalTiles_;
     int64_t matrixSize_;
     int64_t numHeads_;
     int64_t seqLen_;
     int64_t numChunks_;
     int64_t lastChunkValidSize_;
     int64_t rowStride_;
     int64_t layoutMode_;
     int64_t isVarlen_;
 
     Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinish_{SYNC_AIV_AIC_FLAG_SOLVE, SYNC_AIC_AIV_FLAG_SOLVE};
 };
 
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriVector<MATRIX_SIZE, T>::Init(
    GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR x_out, GM_ADDR workspace, const SolveTriTilingData* tilingData)
{
    totalTiles_ = tilingData->totalTiles;
    matrixSize_ = tilingData->matrixSize;
    numHeads_ = tilingData->numHeads;
    seqLen_ = tilingData->seqLen;
    numChunks_ = tilingData->numChunks;
    lastChunkValidSize_ = tilingData->lastChunkValidSize;
    layoutMode_ = tilingData->layoutMode;
    isVarlen_ = tilingData->isVarlen;
    rowStride_ = (layoutMode_ == 0) ? matrixSize_ : numHeads_ * matrixSize_;

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x_out));
    workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));
    if (isVarlen_) {
        cuSeqlensGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cu_seqlens));
        chunkIndicesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunk_indices));
    }

    pipe_.InitBuffer(ubBuf_, UB_AIV_ELEMS * sizeof(T));
    ub_ = ubBuf_.Get<T>();
}
 
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriVector<MATRIX_SIZE, T>::Process()
 {
     int32_t subIdx = static_cast<int32_t>(GetSubBlockIdx());
     int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
     if (subIdx == 0 && blockIdx == 0) {
         GenerateAuxMatrices();
     }
     MaskInputToOutput();
     SyncAll<false>();
#else
     // 只让核组 0 的 AIV 子核 0 生成辅助矩阵，其他核直接参与全核同步
     if (subIdx != 0 || blockIdx != 0) {
         SyncAll<false>();
         return;
     }
 
     GenerateAuxMatrices();
 
     SyncAll<false>();
#endif
 }

template <int MATRIX_SIZE, typename T>
__aicore__ inline int64_t SolveTriVector<MATRIX_SIZE, T>::GetTileGMOffset(int64_t tileIdx)
{
    int64_t H = numHeads_;
    int64_t BT = matrixSize_;
    if (layoutMode_ == 2) {
        int64_t chunkGlobalIdx = tileIdx / H;
        int64_t h = tileIdx % H;
        int64_t seqIdx = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2);
        int64_t chunkInSeq = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2 + 1);
        int64_t bos = cuSeqlensGM_.GetValue(seqIdx);
        return (bos + chunkInSeq * BT) * H * BT + h * BT;
    }
    if (layoutMode_ == 1) {
        int64_t h = tileIdx % H;
        int64_t chunk = (tileIdx / H) % numChunks_;
        int64_t b = tileIdx / (H * numChunks_);
        return b * seqLen_ * H * BT + chunk * BT * H * BT + h * BT;
    }
    int64_t chunk = tileIdx % numChunks_;
    int64_t h = (tileIdx / numChunks_) % H;
    int64_t b = tileIdx / (numChunks_ * H);
    return b * H * seqLen_ * BT + h * seqLen_ * BT + chunk * BT * BT;
}

template <int MATRIX_SIZE, typename T>
__aicore__ inline int64_t SolveTriVector<MATRIX_SIZE, T>::GetTileValidSize(int64_t tileIdx)
{
    if (layoutMode_ == 2) {
        int64_t H = numHeads_;
        int64_t BT = matrixSize_;
        int64_t chunkGlobalIdx = tileIdx / H;
        int64_t seqIdx = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2);
        int64_t chunkInSeq = chunkIndicesGM_.GetValue(chunkGlobalIdx * 2 + 1);
        int64_t bos = cuSeqlensGM_.GetValue(seqIdx);
        int64_t eos = cuSeqlensGM_.GetValue(seqIdx + 1);
        int64_t remaining = (eos - bos) - chunkInSeq * BT;
        return (remaining >= BT) ? BT : remaining;
    }
    int64_t chunk = (layoutMode_ == 1) ? ((tileIdx / numHeads_) % numChunks_) : (tileIdx % numChunks_);
    return (chunk == numChunks_ - 1) ? lastChunkValidSize_ : matrixSize_;
}

template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriVector<MATRIX_SIZE, T>::MaskInputToOutput()
{
    int64_t workerIdx = static_cast<int64_t>(GetBlockIdx()) * static_cast<int64_t>(GetSubBlockNum()) +
                        static_cast<int64_t>(GetSubBlockIdx());
    int64_t workerNum = static_cast<int64_t>(GetBlockNum()) * static_cast<int64_t>(GetSubBlockNum());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if (workerIdx != 0) {
        return;
    }
    workerNum = 1;
#endif
    int64_t totalRows = totalTiles_ * MATRIX_SIZE;

    for (int64_t rowTask = workerIdx; rowTask < totalRows; rowTask += workerNum) {
        int64_t tileIdx = rowTask / MATRIX_SIZE;
        int64_t row = rowTask - tileIdx * MATRIX_SIZE;
        int64_t validSize = GetTileValidSize(tileIdx);
        int64_t gmOffset = GetTileGMOffset(tileIdx);

        Duplicate(ub_, T(0.0f), MATRIX_SIZE);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);

        if (row < validSize) {
            DataCopy(ub_, inputGM_[gmOffset + row * rowStride_], MATRIX_SIZE);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            uint64_t upperMask[2] = {0, 0};
            if (row < 64) {
                upperMask[0] = 0xffffffffffffffffULL << row;
                upperMask[1] = (MATRIX_SIZE > 64) ? 0xffffffffffffffffULL : 0;
            } else {
                upperMask[1] = 0xffffffffffffffffULL << (row - 64);
            }
            Duplicate(ub_, T(0.0f), upperMask, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(outputGM_[gmOffset + row * rowStride_], ub_, MATRIX_SIZE);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}
 
template <int MATRIX_SIZE, typename T>
__aicore__ inline void SolveTriVector<MATRIX_SIZE, T>::GenerateAuxMatrices()
 {
     // 单核循环生成所有条带（block dim=1 时只有 1 个 AIV 核）
     for (int32_t stripIdx = 0; stripIdx < NUM_AUX_CORES; stripIdx++) {
         // Step 1: UB 全部清零
         Duplicate(ub_, T(0), UB_AIV_ELEMS);
         SetFlag<HardEvent::V_MTE3>(0);
         WaitFlag<HardEvent::V_MTE3>(0);
         PipeBarrier<PIPE_V>();
 
         // Step 2: 零条带写入 ZERO / I / -I 三个 GM slot
         DataCopyExtParams stripParams;
         stripParams.blockCount = 1;
         stripParams.blockLen = static_cast<uint32_t>(STRIP_LEN * sizeof(T));
         stripParams.srcStride = 0;
         stripParams.dstStride = 0;
 
         int32_t stripOff = stripIdx * STRIP_LEN;
         DataCopyPad(workspaceGM_[GM_WS_ZERO * TILE_LEN + stripOff], ub_, stripParams);
         DataCopyPad(workspaceGM_[GM_WS_I * TILE_LEN + stripOff], ub_, stripParams);
         DataCopyPad(workspaceGM_[GM_WS_INEG * TILE_LEN + stripOff], ub_, stripParams);
         SetFlag<HardEvent::MTE3_V>(0);
         WaitFlag<HardEvent::MTE3_V>(0);
 
         // Step 3: mask 写 8x16 对角块
         uint64_t diagMask[2] = {
             DIAG_MASK_8X16_EVEN[0],
             DIAG_MASK_8X16_EVEN[1]
         };
        Duplicate(ub_[UB_DIAG_I_OFF], T(1.0f), diagMask, 1, 1, 8);
        Duplicate(ub_[UB_DIAG_INEG_OFF], T(-1.0f), diagMask, 1, 1, 8);
         SetFlag<HardEvent::V_MTE3>(1);
         WaitFlag<HardEvent::V_MTE3>(1);
 
         // Step 4: 搬对角块到 GM
         int32_t rowStart = stripIdx * ROWS_PER_AIV_CORE;
         int32_t colStart = stripIdx * ROWS_PER_AIV_CORE;
         int32_t gmDiagOff = rowStart * MATRIX_SIZE + colStart;
 
         DataCopyExtParams diagParams;
         diagParams.blockCount = ROWS_PER_AIV_CORE;
         diagParams.blockLen = static_cast<uint32_t>(ROWS_PER_AIV_CORE * sizeof(T));
         diagParams.srcStride = 0;
         diagParams.dstStride = static_cast<uint32_t>((MATRIX_SIZE - ROWS_PER_AIV_CORE) * sizeof(T));
 
         DataCopyPad(workspaceGM_[GM_WS_I * TILE_LEN + gmDiagOff], ub_[UB_DIAG_I_OFF], diagParams);
         DataCopyPad(workspaceGM_[GM_WS_INEG * TILE_LEN + gmDiagOff], ub_[UB_DIAG_INEG_OFF], diagParams);
         SetFlag<HardEvent::MTE3_V>(0);
         WaitFlag<HardEvent::MTE3_V>(0);
     }
 }
 
 }  // namespace NsSolveTri
 
 #endif  // SOLVE_TRI_VECTOR_H
