/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

/*!
 * \file solve_tril.cpp
 * \brief SolveTril kernel entry point
 *
 * Input  A: [B, S, H, BT] (BSND) or [T, H, BT] (TND), dtype FLOAT16
 * Output out: same shape as A
 *
 * Each chunk is a BT×BT unit lower-triangular block at GM stride H*BT.
 * MCH on 16×16 leaf blocks, then MBH recursive merge.
 */

#include "kernel_operator.h"
#include "solve_tril_common.h"
#include "solve_tril_cube.h"
#include "solve_tril_tiling_data.h"

using namespace AscendC;

template <typename D_TYPE, int MBH_LEVELS>
class SolveTrilKernel {
public:
    __aicore__ inline void Init(GM_ADDR A_gm, GM_ADDR /*cu_seqlens_gm*/,
                                 GM_ADDR /*chunk_indices_out_gm*/,
                                 GM_ADDR out_gm, GM_ADDR workspace,
                                 const SolveTrilTilingData* tilingData)
    {
        BT_           = static_cast<uint32_t>(tilingData->chunkSize);
        totalTasks_   = static_cast<uint32_t>(tilingData->chunkNumTotal);
        H_            = static_cast<uint32_t>(tilingData->numHead);
        NT_           = static_cast<uint32_t>(tilingData->chunkNumInSeq);
        rowStride_    = static_cast<uint32_t>(tilingData->rowStride);
        blockDim_     = static_cast<uint32_t>(tilingData->blockDim);
        taskPerCore_  = static_cast<uint32_t>(tilingData->taskPerCore);
        blockIdx_     = GetBlockIdx();

        xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ D_TYPE*>(A_gm));
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ D_TYPE*>(out_gm));
        wsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));

        constexpr uint32_t BLOCK_IO_F16 = LEAF_ELEMENTS * sizeof(D_TYPE);

        if constexpr (MBH_LEVELS == 0) {
            static_assert(MCH_ONLY_WORK_BYTES <= UB_CAPACITY_BYTES,
                "MCH_ONLY workBuf exceeds UB capacity");
            pipe.InitBuffer(inQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(outQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(workBuf, MCH_ONLY_WORK_BYTES);
        } else if constexpr (MBH_LEVELS == 1) {
            static_assert(MBH1_WORK_BYTES <= UB_CAPACITY_BYTES,
                "MBH_1 workBuf exceeds UB capacity");
            constexpr uint32_t MAT32_F16 = 32 * 32 * sizeof(D_TYPE);
            pipe.InitBuffer(inQueue, 1, MAT32_F16);
            pipe.InitBuffer(outQueue, 1, MAT32_F16);
            pipe.InitBuffer(workBuf, MBH1_WORK_BYTES);
        } else if constexpr (MBH_LEVELS == 2) {
            static_assert(MBH2_WORK_BYTES <= UB_CAPACITY_BYTES,
                "MBH_2 workBuf exceeds UB capacity");
            pipe.InitBuffer(inQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(outQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(workBuf, MBH2_WORK_BYTES);
        } else if constexpr (MBH_LEVELS == 3) {
            static_assert(MBH3_WORK_BYTES <= UB_CAPACITY_BYTES,
                "MBH_3 workBuf exceeds UB capacity");
            pipe.InitBuffer(inQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(outQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(workBuf, MBH3_WORK_BYTES);
        }
    }

    __aicore__ inline void Process()
    {
        PipeBarrier<PIPE_ALL>();
        if constexpr (MBH_LEVELS == 0) {
            ProcessMchOnly();
        } else if constexpr (MBH_LEVELS == 1) {
            ProcessMbh1();
        } else if constexpr (MBH_LEVELS == 2) {
            ProcessMbh2();
        } else if constexpr (MBH_LEVELS == 3) {
            ProcessMbh3();
        }
    }

private:
    // task_id → (b, h, chunk_idx, gmBase)
    __aicore__ inline void ComputeTaskInfo(uint32_t taskId, uint32_t& baseGmOffset)
    {
        uint32_t hNT = H_ * NT_;
        uint32_t b = taskId / hNT;
        uint32_t rem = taskId % hNT;
        uint32_t h = rem / NT_;
        uint32_t chunkIdx = rem % NT_;
        // base = b * S * H * BT + chunkIdx * BT * H * BT + h * BT
        // S is unknown directly, but base = (b*S + chunkIdx*BT) * rowStride_ + h * BT_
        // Since S = NT * BT (with possible tail), we approximate:
        // base ≈ (b * (NT * BT) + chunkIdx * BT) * rowStride_ + h * BT_
        //      = (b * NT + chunkIdx) * BT_ * rowStride_ + h * BT_
        baseGmOffset = (b * NT_ + chunkIdx) * BT_ * rowStride_ + h * BT_;
    }

    // Load a dim×dim block from xGlobal with row stride rowStride_
    __aicore__ inline void LoadMatrixStrided(LocalTensor<D_TYPE>& dst,
        uint32_t base, uint32_t dim)
    {
        uint32_t rowBytes = dim * sizeof(D_TYPE);
        for (uint32_t row = 0; row < dim; row++) {
            DataCopyPad(dst[row * dim], xGlobal[base + row * rowStride_],
                {1, static_cast<uint16_t>(rowBytes), 0, 0}, {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
    }

    // Store a dim×dim block to yGlobal with row stride rowStride_
    __aicore__ inline void StoreMatrixStrided(uint32_t base,
        LocalTensor<D_TYPE>& src, uint32_t dim)
    {
        uint32_t rowBytes = dim * sizeof(D_TYPE);
        for (uint32_t row = 0; row < dim; row++) {
            DataCopyPad(yGlobal[base + row * rowStride_], src[row * dim],
                {1, static_cast<uint16_t>(rowBytes), 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
    }

    // ========================================================================
    // MCH_ONLY (BT=16)
    // ========================================================================
    __aicore__ inline void ProcessMchOnly()
    {
        uint32_t startTask = blockIdx_ * taskPerCore_;
        uint32_t endTask = startTask + taskPerCore_;
        if (endTask > totalTasks_) endTask = totalTasks_;

        for (uint32_t taskId = startTask; taskId < endTask; taskId++) {
            uint32_t base;
            ComputeTaskInfo(taskId, base);

            LocalTensor<D_TYPE> inputLocal = inQueue.AllocTensor<D_TYPE>();
            LoadMatrixStrided(inputLocal, base, LEAF_BLOCK_SIZE);
            inQueue.EnQue(inputLocal);
            LocalTensor<D_TYPE> L_f16 = inQueue.DeQue<D_TYPE>();

            LocalTensor<float> work = workBuf.Get<float>();
            LocalTensor<float> X     = work[0 * LEAF_ELEMENTS];
            LocalTensor<float> A     = work[1 * LEAF_ELEMENTS];
            LocalTensor<float> Y     = work[2 * LEAF_ELEMENTS];
            LocalTensor<float> temp  = work[3 * LEAF_ELEMENTS];
            LocalTensor<float> mmTmp = work[4 * LEAF_ELEMENTS];

            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
                A.SetValue(i, static_cast<float>(L_f16.GetValue(i)));
            }

            MchInverse(X, A, X, Y, temp, mmTmp);

            LocalTensor<D_TYPE> outLocal = outQueue.AllocTensor<D_TYPE>();
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
                outLocal.SetValue(i, static_cast<D_TYPE>(X.GetValue(i)));
            }
            outQueue.EnQue(outLocal);
            LocalTensor<D_TYPE> outData = outQueue.DeQue<D_TYPE>();
            StoreMatrixStrided(base, outData, LEAF_BLOCK_SIZE);
            PipeBarrier<PIPE_ALL>();
            outQueue.FreeTensor(outData);
            inQueue.FreeTensor(L_f16);
        }
    }

    // ========================================================================
    // MBH_1 (BT=32)
    // ========================================================================
    __aicore__ inline void ProcessMbh1()
    {
        uint32_t startTask = blockIdx_ * taskPerCore_;
        uint32_t endTask = startTask + taskPerCore_;
        if (endTask > totalTasks_) endTask = totalTasks_;

        constexpr uint32_t MATRIX_DIM = 32;
        constexpr uint32_t MATRIX_SIZE = MATRIX_DIM * MATRIX_DIM;

        for (uint32_t taskId = startTask; taskId < endTask; taskId++) {
            uint32_t base;
            ComputeTaskInfo(taskId, base);

            LocalTensor<D_TYPE> inputLocal = inQueue.AllocTensor<D_TYPE>();
            LoadMatrixStrided(inputLocal, base, MATRIX_DIM);
            inQueue.EnQue(inputLocal);
            LocalTensor<D_TYPE> L_f16 = inQueue.DeQue<D_TYPE>();

            LocalTensor<float> work = workBuf.Get<float>();
            LocalTensor<float> L11inv = work[0 * LEAF_ELEMENTS];
            LocalTensor<float> L22inv = work[1 * LEAF_ELEMENTS];
            LocalTensor<float> L21    = work[2 * LEAF_ELEMENTS];
            LocalTensor<float> T      = work[3 * LEAF_ELEMENTS];
            LocalTensor<float> botL   = work[4 * LEAF_ELEMENTS];
            LocalTensor<float> X      = work[5 * LEAF_ELEMENTS];
            LocalTensor<float> A      = work[6 * LEAF_ELEMENTS];
            LocalTensor<float> Y      = work[7 * LEAF_ELEMENTS];
            LocalTensor<float> temp   = work[8 * LEAF_ELEMENTS];
            LocalTensor<float> mmTmp  = work[9 * LEAF_ELEMENTS];

            ExtractBlock16(A, L_f16, 0, 0, MATRIX_DIM);
            MchInverse(L11inv, A, X, Y, temp, mmTmp);
            ExtractBlock16(A, L_f16, 16, 16, MATRIX_DIM);
            MchInverse(L22inv, A, X, Y, temp, mmTmp);
            ExtractBlock16(L21, L_f16, 16, 0, MATRIX_DIM);

            NsSolveTril::MatMul16x16(T, L21, L11inv);
            NsSolveTril::MatMul16x16(botL, L22inv, T);
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++)
                botL.SetValue(i, -botL.GetValue(i));

            LocalTensor<D_TYPE> outLocal = outQueue.AllocTensor<D_TYPE>();
            for (uint32_t i = 0; i < MATRIX_DIM; i++) {
                for (uint32_t j = 0; j < MATRIX_DIM; j++) {
                    float val = 0.0f;
                    if (i < LEAF_BLOCK_SIZE && j < LEAF_BLOCK_SIZE)
                        val = L11inv.GetValue(i * LEAF_BLOCK_SIZE + j);
                    else if (i >= LEAF_BLOCK_SIZE && j < LEAF_BLOCK_SIZE)
                        val = botL.GetValue((i - LEAF_BLOCK_SIZE) * LEAF_BLOCK_SIZE + j);
                    else if (i >= LEAF_BLOCK_SIZE && j >= LEAF_BLOCK_SIZE)
                        val = L22inv.GetValue((i - LEAF_BLOCK_SIZE) * LEAF_BLOCK_SIZE + (j - LEAF_BLOCK_SIZE));
                    outLocal.SetValue(i * MATRIX_DIM + j, static_cast<D_TYPE>(val));
                }
            }
            outQueue.EnQue(outLocal);
            LocalTensor<D_TYPE> outData = outQueue.DeQue<D_TYPE>();
            StoreMatrixStrided(base, outData, MATRIX_DIM);
            PipeBarrier<PIPE_ALL>();
            outQueue.FreeTensor(outData);
            inQueue.FreeTensor(L_f16);
        }
    }

    __aicore__ inline void ExtractBlock16(
        LocalTensor<float>& dest, LocalTensor<D_TYPE>& src,
        uint32_t rowOff, uint32_t colOff, uint32_t stride)
    {
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++)
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++)
                dest.SetValue(i * LEAF_BLOCK_SIZE + j,
                    static_cast<float>(src.GetValue((rowOff + i) * stride + (colOff + j))));
    }

    __aicore__ inline void MchInverse(
        LocalTensor<float>& result, LocalTensor<float>& A_block,
        LocalTensor<float>& X, LocalTensor<float>& Y,
        LocalTensor<float>& temp, LocalTensor<float>& mmTmp)
    {
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++)
            for (uint32_t j = i; j < LEAF_BLOCK_SIZE; j++)
                A_block.SetValue(i * LEAF_BLOCK_SIZE + j, 0.0f);

        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++)
            X.SetValue(i, -A_block.GetValue(i));
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++)
            X.SetValue(i * LEAF_BLOCK_SIZE + i, 1.0f);

        NsSolveTril::MatMul16x16(Y, A_block, A_block);

        for (uint32_t iter = 0; iter < MCH_ITERATIONS; iter++) {
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++)
                temp.SetValue(i, Y.GetValue(i));
            for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++)
                temp.SetValue(i * LEAF_BLOCK_SIZE + i, temp.GetValue(i * LEAF_BLOCK_SIZE + i) + 1.0f);

            NsSolveTril::MatMul16x16(mmTmp, X, temp);
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++)
                X.SetValue(i, mmTmp.GetValue(i));

            if (iter < MCH_ITERATIONS - 1) {
                NsSolveTril::MatMul16x16(mmTmp, Y, Y);
                for (uint32_t i = 0; i < LEAF_ELEMENTS; i++)
                    Y.SetValue(i, mmTmp.GetValue(i));
            }
        }

        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                if (j > i)       result.SetValue(i * LEAF_BLOCK_SIZE + j, 0.0f);
                else if (j == i) result.SetValue(i * LEAF_BLOCK_SIZE + j, 1.0f);
                else             result.SetValue(i * LEAF_BLOCK_SIZE + j, X.GetValue(i * LEAF_BLOCK_SIZE + j));
            }
        }
    }

    // ========================================================================
    // GM I/O helpers for 16×16 blocks (MBH_2/MBH_3)
    // ========================================================================

    __aicore__ inline void ReadBlock16FromX(LocalTensor<float>& dst,
        uint32_t gmBase, uint32_t row, uint32_t col)
    {
        LocalTensor<D_TYPE> stageIn = inQueue.AllocTensor<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            uint32_t gmIdx = gmBase + (row + i) * rowStride_ + col;
            DataCopyPad(stageIn[i * LEAF_BLOCK_SIZE], xGlobal[gmIdx],
                {1, static_cast<uint16_t>(LEAF_BLOCK_SIZE * sizeof(D_TYPE)), 0, 0}, {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        inQueue.EnQue(stageIn);
        LocalTensor<D_TYPE> stageData = inQueue.DeQue<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++)
            dst.SetValue(i, static_cast<float>(stageData.GetValue(i)));
        inQueue.FreeTensor(stageData);
    }

    __aicore__ inline void WriteBlock16ToY(LocalTensor<float>& src,
        uint32_t gmBase, uint32_t row, uint32_t col)
    {
        LocalTensor<D_TYPE> stageOut = outQueue.AllocTensor<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++)
            stageOut.SetValue(i, static_cast<D_TYPE>(src.GetValue(i)));
        outQueue.EnQue(stageOut);
        LocalTensor<D_TYPE> stageData = outQueue.DeQue<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            uint32_t gmIdx = gmBase + (row + i) * rowStride_ + col;
            DataCopyPad(yGlobal[gmIdx], stageData[i * LEAF_BLOCK_SIZE],
                {1, static_cast<uint16_t>(LEAF_BLOCK_SIZE * sizeof(D_TYPE)), 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        outQueue.FreeTensor(stageData);
    }

    __aicore__ inline void ReadBlock16FromY(LocalTensor<float>& dst,
        uint32_t gmBase, uint32_t row, uint32_t col)
    {
        LocalTensor<D_TYPE> stageIn = inQueue.AllocTensor<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            uint32_t gmIdx = gmBase + (row + i) * rowStride_ + col;
            DataCopyPad(stageIn[i * LEAF_BLOCK_SIZE], yGlobal[gmIdx],
                {1, static_cast<uint16_t>(LEAF_BLOCK_SIZE * sizeof(D_TYPE)), 0, 0}, {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        inQueue.EnQue(stageIn);
        LocalTensor<D_TYPE> stageData = inQueue.DeQue<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++)
            dst.SetValue(i, static_cast<float>(stageData.GetValue(i)));
        inQueue.FreeTensor(stageData);
    }

    __aicore__ inline void WriteBlock16ToResult(LocalTensor<float>& result,
        LocalTensor<float>& src, uint32_t row, uint32_t col, uint32_t stride)
    {
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++)
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++)
                result.SetValue((row + i) * stride + (col + j),
                    src.GetValue(i * LEAF_BLOCK_SIZE + j));
    }

    __aicore__ inline void ReadBlock16FromResult(LocalTensor<float>& dst,
        LocalTensor<float>& result, uint32_t row, uint32_t col, uint32_t stride)
    {
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++)
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++)
                dst.SetValue(i * LEAF_BLOCK_SIZE + j,
                    result.GetValue((row + i) * stride + (col + j)));
    }

    // ========================================================================
    // MBH Layer functions (UB float32 computation)
    // ========================================================================

    __aicore__ inline void MbhLayer1PairUB_GM(LocalTensor<float>& result,
        uint32_t gmBase, uint32_t blk1, uint32_t blk2, uint32_t stride,
        LocalTensor<float>& T, LocalTensor<float>& mmTmp)
    {
        uint32_t r1 = blk1 * LEAF_BLOCK_SIZE, r2 = blk2 * LEAF_BLOCK_SIZE;
        uint32_t c1 = blk1 * LEAF_BLOCK_SIZE, c2 = blk2 * LEAF_BLOCK_SIZE;
        ReadBlock16FromX(T, gmBase, r2, c1);
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++)
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < LEAF_BLOCK_SIZE; k++)
                    sum += T.GetValue(i * LEAF_BLOCK_SIZE + k) *
                           result.GetValue((r1 + k) * stride + (c1 + j));
                mmTmp.SetValue(i * LEAF_BLOCK_SIZE + j, sum);
            }
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++)
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < LEAF_BLOCK_SIZE; k++)
                    sum += result.GetValue((r2 + i) * stride + (c2 + k)) *
                           mmTmp.GetValue(k * LEAF_BLOCK_SIZE + j);
                result.SetValue((r2 + i) * stride + (c1 + j), -sum);
            }
    }

    __aicore__ inline void MbhLayer2UB_GM(LocalTensor<float>& result,
        uint32_t gmBase, uint32_t stride, uint32_t baseOff,
        LocalTensor<float>& T, LocalTensor<float>& mmTmp)
    {
        constexpr uint32_t HALF = 32, NUM = 2;
        for (uint32_t bj = 0; bj < NUM; bj++)
            for (uint32_t bi = 0; bi < NUM; bi++) {
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) mmTmp.SetValue(e, 0.0f);
                for (uint32_t bk = 0; bk < NUM; bk++) {
                    for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) T.SetValue(e, 0.0f);
                    for (uint32_t bm = 0; bm < NUM; bm++) {
                        LocalTensor<float> w = workBuf.Get<float>();
                        LocalTensor<float> tA = w[MBH2_RESULT_SIZE + 2 * LEAF_ELEMENTS];
                        ReadBlock16FromX(tA, gmBase,
                            baseOff + HALF + bk * LEAF_BLOCK_SIZE,
                            baseOff + bm * LEAF_BLOCK_SIZE);
                        LocalTensor<float> tB = w[MBH2_RESULT_SIZE + 3 * LEAF_ELEMENTS];
                        ReadBlock16FromResult(tB, result,
                            baseOff + bm * LEAF_BLOCK_SIZE,
                            baseOff + bj * LEAF_BLOCK_SIZE, stride);
                        NsSolveTril::MatMul16x16Acc(T, tA, tB);
                    }
                    LocalTensor<float> w2 = workBuf.Get<float>();
                    LocalTensor<float> tC = w2[MBH2_RESULT_SIZE + 2 * LEAF_ELEMENTS];
                    ReadBlock16FromResult(tC, result,
                        baseOff + HALF + bi * LEAF_BLOCK_SIZE,
                        baseOff + HALF + bk * LEAF_BLOCK_SIZE, stride);
                    NsSolveTril::MatMul16x16Acc(mmTmp, tC, T);
                }
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) mmTmp.SetValue(e, -mmTmp.GetValue(e));
                WriteBlock16ToResult(result, mmTmp,
                    baseOff + HALF + bi * LEAF_BLOCK_SIZE, baseOff + bj * LEAF_BLOCK_SIZE, stride);
            }
    }

    __aicore__ inline void MbhLayer3UB_GM(LocalTensor<float>& result,
        uint32_t gmBase, uint32_t stride,
        LocalTensor<float>& T, LocalTensor<float>& mmTmp)
    {
        constexpr uint32_t HALF = 64, NUM = 4;
        for (uint32_t bj = 0; bj < NUM; bj++)
            for (uint32_t bi = 0; bi < NUM; bi++) {
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) mmTmp.SetValue(e, 0.0f);
                for (uint32_t bk = 0; bk < NUM; bk++) {
                    for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) T.SetValue(e, 0.0f);
                    for (uint32_t bm = 0; bm < NUM; bm++) {
                        LocalTensor<float> w = workBuf.Get<float>();
                        LocalTensor<float> tA = w[MBH3_RESULT_SIZE + 2 * LEAF_ELEMENTS];
                        ReadBlock16FromX(tA, gmBase,
                            HALF + bk * LEAF_BLOCK_SIZE, bm * LEAF_BLOCK_SIZE);
                        LocalTensor<float> tB = w[MBH3_RESULT_SIZE + 3 * LEAF_ELEMENTS];
                        ReadBlock16FromResult(tB, result,
                            bm * LEAF_BLOCK_SIZE, bj * LEAF_BLOCK_SIZE, stride);
                        NsSolveTril::MatMul16x16Acc(T, tA, tB);
                    }
                    LocalTensor<float> w2 = workBuf.Get<float>();
                    LocalTensor<float> tC = w2[MBH3_RESULT_SIZE + 2 * LEAF_ELEMENTS];
                    ReadBlock16FromResult(tC, result,
                        HALF + bi * LEAF_BLOCK_SIZE,
                        HALF + bk * LEAF_BLOCK_SIZE, stride);
                    NsSolveTril::MatMul16x16Acc(mmTmp, tC, T);
                }
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) mmTmp.SetValue(e, -mmTmp.GetValue(e));
                WriteBlock16ToResult(result, mmTmp,
                    HALF + bi * LEAF_BLOCK_SIZE, bj * LEAF_BLOCK_SIZE, stride);
            }
    }

    // ========================================================================
    // MBH_2 (BT=64)
    // ========================================================================
    __aicore__ inline void ProcessMbh2()
    {
        uint32_t startTask = blockIdx_ * taskPerCore_;
        uint32_t endTask = startTask + taskPerCore_;
        if (endTask > totalTasks_) endTask = totalTasks_;

        constexpr uint32_t DIM = 64;
        constexpr uint32_t MSIZE = DIM * DIM;

        for (uint32_t taskId = startTask; taskId < endTask; taskId++) {
            uint32_t base;
            ComputeTaskInfo(taskId, base);

            constexpr uint32_t MCH_X_OFF     = MBH2_RESULT_SIZE + 0 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_A_OFF     = MBH2_RESULT_SIZE + 1 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_Y_OFF     = MBH2_RESULT_SIZE + 2 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_TEMP_OFF  = MBH2_RESULT_SIZE + 3 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_MMTMP_OFF = MBH2_RESULT_SIZE + 4 * LEAF_ELEMENTS;

            LocalTensor<float> work  = workBuf.Get<float>();
            LocalTensor<float> result = work[0];
            LocalTensor<float> X      = work[MCH_X_OFF];
            LocalTensor<float> A      = work[MCH_A_OFF];
            LocalTensor<float> Y      = work[MCH_Y_OFF];
            LocalTensor<float> temp   = work[MCH_TEMP_OFF];
            LocalTensor<float> mmTmp  = work[MCH_MMTMP_OFF];

            for (uint32_t i = 0; i < MSIZE; i++) result.SetValue(i, 0.0f);

            for (uint32_t blk = 0; blk < 4; blk++) {
                ReadBlock16FromX(A, base, blk * LEAF_BLOCK_SIZE, blk * LEAF_BLOCK_SIZE);
                MchInverse(X, A, X, Y, temp, mmTmp);
                WriteBlock16ToResult(result, X, blk * LEAF_BLOCK_SIZE, blk * LEAF_BLOCK_SIZE, DIM);
            }

            MbhLayer1PairUB_GM(result, base, 0, 1, DIM, X, mmTmp);
            MbhLayer1PairUB_GM(result, base, 2, 3, DIM, X, mmTmp);
            MbhLayer2UB_GM(result, base, DIM, 0, X, mmTmp);

            for (uint32_t br = 0; br < 4; br++)
                for (uint32_t bc = 0; bc < 4; bc++) {
                    LocalTensor<float> blkSrc = work[MSIZE];
                    if (bc <= br)
                        ReadBlock16FromResult(blkSrc, result, br * LEAF_BLOCK_SIZE, bc * LEAF_BLOCK_SIZE, DIM);
                    else
                        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) blkSrc.SetValue(i, 0.0f);
                    WriteBlock16ToY(blkSrc, base, br * LEAF_BLOCK_SIZE, bc * LEAF_BLOCK_SIZE);
                }
        }
    }

    // ========================================================================
    // MBH_3 (BT=128)
    // ========================================================================
    __aicore__ inline void ProcessMbh3()
    {
        uint32_t startTask = blockIdx_ * taskPerCore_;
        uint32_t endTask = startTask + taskPerCore_;
        if (endTask > totalTasks_) endTask = totalTasks_;

        constexpr uint32_t DIM = 128;
        constexpr uint32_t MSIZE = DIM * DIM;

        for (uint32_t taskId = startTask; taskId < endTask; taskId++) {
            uint32_t base;
            ComputeTaskInfo(taskId, base);

            constexpr uint32_t MCH_X_OFF     = MBH3_RESULT_SIZE + 0 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_A_OFF     = MBH3_RESULT_SIZE + 1 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_Y_OFF     = MBH3_RESULT_SIZE + 2 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_TEMP_OFF  = MBH3_RESULT_SIZE + 3 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_MMTMP_OFF = MBH3_RESULT_SIZE + 4 * LEAF_ELEMENTS;

            LocalTensor<float> work  = workBuf.Get<float>();
            LocalTensor<float> result = work[0];
            LocalTensor<float> X      = work[MCH_X_OFF];
            LocalTensor<float> A      = work[MCH_A_OFF];
            LocalTensor<float> Y      = work[MCH_Y_OFF];
            LocalTensor<float> temp   = work[MCH_TEMP_OFF];
            LocalTensor<float> mmTmp  = work[MCH_MMTMP_OFF];

            for (uint32_t i = 0; i < MSIZE; i++) result.SetValue(i, 0.0f);

            for (uint32_t blk = 0; blk < 8; blk++) {
                ReadBlock16FromX(A, base, blk * LEAF_BLOCK_SIZE, blk * LEAF_BLOCK_SIZE);
                MchInverse(X, A, X, Y, temp, mmTmp);
                WriteBlock16ToResult(result, X, blk * LEAF_BLOCK_SIZE, blk * LEAF_BLOCK_SIZE, DIM);
            }

            MbhLayer1PairUB_GM(result, base, 0, 1, DIM, X, mmTmp);
            MbhLayer1PairUB_GM(result, base, 2, 3, DIM, X, mmTmp);
            MbhLayer1PairUB_GM(result, base, 4, 5, DIM, X, mmTmp);
            MbhLayer1PairUB_GM(result, base, 6, 7, DIM, X, mmTmp);

            MbhLayer2UB_GM(result, base, DIM, 0, X, mmTmp);
            MbhLayer2UB_GM(result, base, DIM, 64, X, mmTmp);
            MbhLayer3UB_GM(result, base, DIM, X, mmTmp);

            for (uint32_t br = 0; br < 8; br++)
                for (uint32_t bc = 0; bc < 8; bc++) {
                    LocalTensor<float> blkSrc = work[MSIZE];
                    if (bc <= br)
                        ReadBlock16FromResult(blkSrc, result, br * LEAF_BLOCK_SIZE, bc * LEAF_BLOCK_SIZE, DIM);
                    else
                        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) blkSrc.SetValue(i, 0.0f);
                    WriteBlock16ToY(blkSrc, base, br * LEAF_BLOCK_SIZE, bc * LEAF_BLOCK_SIZE);
                }
        }
    }

    GlobalTensor<D_TYPE> xGlobal;
    GlobalTensor<D_TYPE> yGlobal;
    GlobalTensor<float> wsGlobal;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> workBuf;

    uint32_t BT_;
    uint32_t totalTasks_;
    uint32_t H_;
    uint32_t NT_;
    uint32_t rowStride_;
    uint32_t blockDim_;
    uint32_t taskPerCore_;
    uint32_t blockIdx_;
};

template <typename D_TYPE, int MBH_LEVELS>
__global__ __aicore__ void solve_tril(GM_ADDR A, GM_ADDR cu_seqlens,
    GM_ADDR chunk_indices_out, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SolveTrilTilingData);
    GET_TILING_DATA(tilingData, tiling);
    SolveTrilKernel<D_TYPE, MBH_LEVELS> op;
    op.Init(A, cu_seqlens, chunk_indices_out, out, workspace, &tilingData);
    op.Process();
}
