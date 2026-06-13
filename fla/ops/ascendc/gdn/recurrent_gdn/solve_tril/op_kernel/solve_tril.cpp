/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

/*!
 * \file solve_tril.cpp
 * \brief SolveTril kernel entry point
 *
 * Uses float32 intermediate computation for precision (proven in probe).
 * Input/output in fp16, all MCH iterations computed in fp32.
 * Supports MCH_ONLY (n=16), MBH_1 (n=32), MBH_2 (n=64), MBH_3 (n=128).
 *
 * Precision fix: MBH layer 2/3 fuse T computation into BL calculation
 * to avoid storing intermediate T blocks in yGlobal (fp16 truncation).
 */

#include "kernel_operator.h"
#include "solve_tril_common.h"
#include "solve_tril_cube.h"
#include "solve_tril_tiling_data.h"

using namespace AscendC;

template <typename D_TYPE, int MBH_LEVELS>
class SolveTrilKernel {
public:
    __aicore__ inline void Init(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR workspace,
                                 const SolveTrilTilingData* tilingData)
    {
        n_ = static_cast<uint32_t>(tilingData->n);
        batchSize_ = static_cast<uint32_t>(tilingData->batchSize);
        numLeafBlocks_ = static_cast<uint32_t>(tilingData->numLeafBlocks);
        blockDim_ = static_cast<uint32_t>(tilingData->blockDim);
        taskPerCore_ = static_cast<uint32_t>(tilingData->taskPerCore);
        blockIdx_ = GetBlockIdx();

        xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ D_TYPE*>(xGm));
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ D_TYPE*>(yGm));
        wsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));

        // Buffer sizes using named constants from solve_tril_common.h
        constexpr uint32_t BLOCK_IO_F16 = LEAF_ELEMENTS * sizeof(D_TYPE);

        if constexpr (MBH_LEVELS == 0) {
            static_assert(MCH_ONLY_WORK_BYTES <= UB_CAPACITY_BYTES,
                "MCH_ONLY workBuf exceeds UB capacity (192KB)");
            pipe.InitBuffer(inQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(outQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(workBuf, MCH_ONLY_WORK_BYTES);
        } else if constexpr (MBH_LEVELS == 1) {
            static_assert(MBH1_WORK_BYTES <= UB_CAPACITY_BYTES,
                "MBH_1 workBuf exceeds UB capacity (192KB)");
            constexpr uint32_t MAT32_F16 = 32 * 32 * sizeof(D_TYPE);
            pipe.InitBuffer(inQueue, 1, MAT32_F16);
            pipe.InitBuffer(outQueue, 1, MAT32_F16);
            pipe.InitBuffer(workBuf, MBH1_WORK_BYTES);
        } else if constexpr (MBH_LEVELS == 2) {
            static_assert(MBH2_WORK_BYTES <= UB_CAPACITY_BYTES,
                "MBH_2 workBuf exceeds UB capacity (192KB)");
            pipe.InitBuffer(inQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(outQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(workBuf, MBH2_WORK_BYTES);
        } else if constexpr (MBH_LEVELS == 3) {
            static_assert(MBH3_WORK_BYTES <= UB_CAPACITY_BYTES,
                "MBH_3 workBuf exceeds UB capacity (192KB)");
            pipe.InitBuffer(inQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(outQueue, 1, BLOCK_IO_F16);
            pipe.InitBuffer(workBuf, MBH3_WORK_BYTES);
        }
    }

    __aicore__ inline void Process()
    {
        // Ensure tiling data transfer is complete before computation
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
    // ========================================================================
    // MCH_ONLY path (n=16): process individual 16x16 leaf blocks
    // ========================================================================
    __aicore__ inline void ProcessMchOnly()
    {
        uint32_t startTask = blockIdx_ * taskPerCore_;
        uint32_t endTask = startTask + taskPerCore_;
        uint32_t totalTasks = batchSize_ * numLeafBlocks_;
        if (endTask > totalTasks) endTask = totalTasks;

        for (uint32_t taskId = startTask; taskId < endTask; taskId++) {
            uint32_t batchIdx = taskId / numLeafBlocks_;
            uint32_t gmOffset = batchIdx * LEAF_ELEMENTS;

            // CopyIn
            LocalTensor<D_TYPE> inputLocal = inQueue.AllocTensor<D_TYPE>();
            DataCopyPad(inputLocal, xGlobal[gmOffset],
                {1, static_cast<uint16_t>(LEAF_ELEMENTS * sizeof(D_TYPE)), 0, 0},
                {false, 0, 0, 0});
            inQueue.EnQue(inputLocal);
            LocalTensor<D_TYPE> L_f16 = inQueue.DeQue<D_TYPE>();

            // Work buffer: 5 float32 16x16 blocks
            // Layout: [X/result][A][Y][temp][mmTmp]
            // result aliases X (MchInverse copies X->result at end, self-copy is safe)
            LocalTensor<float> work = workBuf.Get<float>();
            LocalTensor<float> X      = work[0 * LEAF_ELEMENTS];
            LocalTensor<float> A      = work[1 * LEAF_ELEMENTS];
            LocalTensor<float> Y      = work[2 * LEAF_ELEMENTS];
            LocalTensor<float> temp   = work[3 * LEAF_ELEMENTS];
            LocalTensor<float> mmTmp  = work[4 * LEAF_ELEMENTS];

            // Convert fp16 to fp32 into A
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
                A.SetValue(i, static_cast<float>(L_f16.GetValue(i)));
            }

            // Compute MCH inverse: result written to X (slot 0)
            MchInverse(X, A, X, Y, temp, mmTmp);

            // CopyOut: convert fp32 result back to fp16
            LocalTensor<D_TYPE> outLocal = outQueue.AllocTensor<D_TYPE>();
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
                outLocal.SetValue(i, static_cast<D_TYPE>(X.GetValue(i)));
            }
            outQueue.EnQue(outLocal);
            LocalTensor<D_TYPE> outData = outQueue.DeQue<D_TYPE>();
            DataCopyPad(yGlobal[gmOffset], outData,
                {1, static_cast<uint16_t>(LEAF_ELEMENTS * sizeof(D_TYPE)), 0, 0});
            PipeBarrier<PIPE_ALL>();
            outQueue.FreeTensor(outData);
            inQueue.FreeTensor(L_f16);
        }
    }

    // ========================================================================
    // MBH_1 path (n=32): load full matrix into UB, compute locally, write out
    // Matches probe approach (MERE=0) - no GM read-back of intermediate results
    // ========================================================================
    __aicore__ inline void ProcessMbh1()
    {
        // For MBH_1, each task is one batch element (32x32 matrix)
        uint32_t startTask = blockIdx_ * taskPerCore_;
        uint32_t endTask = startTask + taskPerCore_;
        if (endTask > batchSize_) endTask = batchSize_;

        constexpr uint32_t MATRIX_DIM = 32;
        constexpr uint32_t MATRIX_SIZE = MATRIX_DIM * MATRIX_DIM;  // 1024

        for (uint32_t batchIdx = startTask; batchIdx < endTask; batchIdx++) {
            uint32_t batchGmOffset = batchIdx * MATRIX_SIZE;

            // --- CopyIn: load full 32x32 fp16 matrix ---
            LocalTensor<D_TYPE> inputLocal = inQueue.AllocTensor<D_TYPE>();
            DataCopyPad(inputLocal, xGlobal[batchGmOffset],
                {1, static_cast<uint16_t>(MATRIX_SIZE * sizeof(D_TYPE)), 0, 0},
                {false, 0, 0, 0});
            inQueue.EnQue(inputLocal);
            LocalTensor<D_TYPE> L_f16 = inQueue.DeQue<D_TYPE>();

            // Work buffer: 10 float32 16x16 blocks
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

            // --- Extract L11 (top-left 16x16) and compute MCH inverse ---
            ExtractBlock16(A, L_f16, 0, 0, MATRIX_DIM);
            MchInverse(L11inv, A, X, Y, temp, mmTmp);

            // --- Extract L22 (bottom-right 16x16) and compute MCH inverse ---
            ExtractBlock16(A, L_f16, 16, 16, MATRIX_DIM);
            MchInverse(L22inv, A, X, Y, temp, mmTmp);

            // --- Extract L21 (bottom-left 16x16) ---
            ExtractBlock16(L21, L_f16, 16, 0, MATRIX_DIM);

            // --- T = L21 * L11inv ---
            NsSolveTril::MatMul16x16(T, L21, L11inv);

            // --- botL = -(L22inv * T) ---
            NsSolveTril::MatMul16x16(botL, L22inv, T);
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
                botL.SetValue(i, -botL.GetValue(i));
            }

            // --- Assemble output 32x32: [L11inv, 0; botL, L22inv] ---
            LocalTensor<D_TYPE> outLocal = outQueue.AllocTensor<D_TYPE>();
            for (uint32_t i = 0; i < MATRIX_DIM; i++) {
                for (uint32_t j = 0; j < MATRIX_DIM; j++) {
                    float val = 0.0f;
                    if (i < LEAF_BLOCK_SIZE && j < LEAF_BLOCK_SIZE) {
                        val = L11inv.GetValue(i * LEAF_BLOCK_SIZE + j);
                    } else if (i >= LEAF_BLOCK_SIZE && j < LEAF_BLOCK_SIZE) {
                        val = botL.GetValue((i - LEAF_BLOCK_SIZE) * LEAF_BLOCK_SIZE + j);
                    } else if (i >= LEAF_BLOCK_SIZE && j >= LEAF_BLOCK_SIZE) {
                        val = L22inv.GetValue((i - LEAF_BLOCK_SIZE) * LEAF_BLOCK_SIZE + (j - LEAF_BLOCK_SIZE));
                    }
                    outLocal.SetValue(i * MATRIX_DIM + j, static_cast<D_TYPE>(val));
                }
            }
            outQueue.EnQue(outLocal);
            LocalTensor<D_TYPE> outData = outQueue.DeQue<D_TYPE>();
            DataCopyPad(yGlobal[batchGmOffset], outData,
                {1, static_cast<uint16_t>(MATRIX_SIZE * sizeof(D_TYPE)), 0, 0});
            PipeBarrier<PIPE_ALL>();
            outQueue.FreeTensor(outData);
            inQueue.FreeTensor(L_f16);
        }
    }

    // Extract a 16x16 block from fp16 matrix at (rowOff, colOff) into fp32 dest
    __aicore__ inline void ExtractBlock16(
        LocalTensor<float>& dest,
        LocalTensor<D_TYPE>& src,
        uint32_t rowOff, uint32_t colOff, uint32_t stride)
    {
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                uint32_t srcIdx = (rowOff + i) * stride + (colOff + j);
                dest.SetValue(i * LEAF_BLOCK_SIZE + j,
                    static_cast<float>(src.GetValue(srcIdx)));
            }
        }
    }

    // MCH inverse: compute inverse of 16x16 unit lower triangular block
    // Input: A_block (full block with diagonal 1s)
    // Output: result = inverse
    __aicore__ inline void MchInverse(
        LocalTensor<float>& result,
        LocalTensor<float>& A_block,
        LocalTensor<float>& X,
        LocalTensor<float>& Y,
        LocalTensor<float>& temp,
        LocalTensor<float>& mmTmp)
    {
        // Zero diagonal and upper triangular (keep strict lower only)
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = i; j < LEAF_BLOCK_SIZE; j++) {
                A_block.SetValue(i * LEAF_BLOCK_SIZE + j, 0.0f);
            }
        }

        // X = I - A
        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
            X.SetValue(i, -A_block.GetValue(i));
        }
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            X.SetValue(i * LEAF_BLOCK_SIZE + i, 1.0f);
        }

        // Y = A * A
        NsSolveTril::MatMul16x16(Y, A_block, A_block);

        // 3 iterations: X = X*(I+Y), Y = Y*Y
        for (uint32_t iter = 0; iter < MCH_ITERATIONS; iter++) {
            // temp = I + Y
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
                temp.SetValue(i, Y.GetValue(i));
            }
            for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
                temp.SetValue(i * LEAF_BLOCK_SIZE + i,
                              temp.GetValue(i * LEAF_BLOCK_SIZE + i) + 1.0f);
            }

            // X = X * temp
            NsSolveTril::MatMul16x16(mmTmp, X, temp);
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
                X.SetValue(i, mmTmp.GetValue(i));
            }

            // Y = Y * Y (skip last iteration)
            if (iter < MCH_ITERATIONS - 1) {
                NsSolveTril::MatMul16x16(mmTmp, Y, Y);
                for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
                    Y.SetValue(i, mmTmp.GetValue(i));
                }
            }
        }

        // Copy result and enforce unit lower triangular structure
        // (MCH float32 rounding can produce tiny non-zero upper triangular values)
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                if (j > i) {
                    result.SetValue(i * LEAF_BLOCK_SIZE + j, 0.0f);
                } else if (j == i) {
                    result.SetValue(i * LEAF_BLOCK_SIZE + j, 1.0f);
                } else {
                    result.SetValue(i * LEAF_BLOCK_SIZE + j, X.GetValue(i * LEAF_BLOCK_SIZE + j));
                }
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

    uint32_t n_;
    uint32_t batchSize_;
    uint32_t numLeafBlocks_;
    uint32_t blockDim_;
    uint32_t taskPerCore_;
    uint32_t blockIdx_;

    // ========================================================================
    // Helpers for MBH_2/MBH_3: DMA via proven queue path
    // ========================================================================

    __aicore__ inline void SafeCopyIn(
        LocalTensor<D_TYPE>& dst, uint32_t gmOffset, uint32_t totalElements)
    {
        uint32_t totalBytes = totalElements * sizeof(D_TYPE);
        if (totalBytes <= 65535) {
            DataCopyPad(dst, xGlobal[gmOffset],
                {1, static_cast<uint16_t>(totalBytes), 0, 0}, {false, 0, 0, 0});
        } else {
            uint32_t halfElems = totalElements / 2;
            uint32_t halfBytes = halfElems * sizeof(D_TYPE);
            DataCopyPad(dst, xGlobal[gmOffset],
                {1, static_cast<uint16_t>(halfBytes), 0, 0}, {false, 0, 0, 0});
            DataCopyPad(dst[halfElems], xGlobal[gmOffset + halfElems],
                {1, static_cast<uint16_t>(halfBytes), 0, 0}, {false, 0, 0, 0});
        }
    }

    __aicore__ inline void WriteBlock16ToY(
        LocalTensor<float>& src, uint32_t gmBase,
        uint32_t row, uint32_t col, uint32_t stride)
    {
        LocalTensor<D_TYPE> stageOut = outQueue.AllocTensor<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                stageOut.SetValue(i * LEAF_BLOCK_SIZE + j,
                    static_cast<D_TYPE>(src.GetValue(i * LEAF_BLOCK_SIZE + j)));
            }
        }
        outQueue.EnQue(stageOut);
        LocalTensor<D_TYPE> stageData = outQueue.DeQue<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            uint32_t gmIdx = gmBase + (row + i) * stride + col;
            DataCopyPad(yGlobal[gmIdx], stageData[i * LEAF_BLOCK_SIZE],
                {1, static_cast<uint16_t>(LEAF_BLOCK_SIZE * sizeof(D_TYPE)), 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        outQueue.FreeTensor(stageData);
    }

    __aicore__ inline void ReadBlock16FromY(
        LocalTensor<float>& dst, uint32_t gmBase,
        uint32_t row, uint32_t col, uint32_t stride)
    {
        LocalTensor<D_TYPE> stageIn = inQueue.AllocTensor<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            uint32_t gmIdx = gmBase + (row + i) * stride + col;
            DataCopyPad(stageIn[i * LEAF_BLOCK_SIZE], yGlobal[gmIdx],
                {1, static_cast<uint16_t>(LEAF_BLOCK_SIZE * sizeof(D_TYPE)), 0, 0},
                {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        inQueue.EnQue(stageIn);
        LocalTensor<D_TYPE> stageData = inQueue.DeQue<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
            dst.SetValue(i, static_cast<float>(stageData.GetValue(i)));
        }
        inQueue.FreeTensor(stageData);
    }

    __aicore__ inline void ReadBlock16FromX(
        LocalTensor<float>& dst, uint32_t gmBase,
        uint32_t row, uint32_t col, uint32_t stride)
    {
        LocalTensor<D_TYPE> stageIn = inQueue.AllocTensor<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            uint32_t gmIdx = gmBase + (row + i) * stride + col;
            DataCopyPad(stageIn[i * LEAF_BLOCK_SIZE], xGlobal[gmIdx],
                {1, static_cast<uint16_t>(LEAF_BLOCK_SIZE * sizeof(D_TYPE)), 0, 0},
                {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        inQueue.EnQue(stageIn);
        LocalTensor<D_TYPE> stageData = inQueue.DeQue<D_TYPE>();
        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
            dst.SetValue(i, static_cast<float>(stageData.GetValue(i)));
        }
        inQueue.FreeTensor(stageData);
    }

    // ========================================================================
    // New UB-based helpers for full float32 computation (MBH_2/MBH_3)
    // ========================================================================

    __aicore__ inline void SafeCopyOut(
        LocalTensor<D_TYPE>& src, uint32_t gmOffset, uint32_t totalElements)
    {
        uint32_t totalBytes = totalElements * sizeof(D_TYPE);
        if (totalBytes <= 65535) {
            DataCopyPad(yGlobal[gmOffset], src,
                {1, static_cast<uint16_t>(totalBytes), 0, 0});
        } else {
            // Split into quarters for large matrices
            uint32_t quarterElems = totalElements / 4;
            uint32_t quarterBytes = quarterElems * sizeof(D_TYPE);
            for (uint32_t q = 0; q < 4; q++) {
                DataCopyPad(yGlobal[gmOffset + q * quarterElems], src[q * quarterElems],
                    {1, static_cast<uint16_t>(quarterBytes), 0, 0});
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // Write a 16x16 float32 block into the result matrix at (row, col)
    __aicore__ inline void WriteBlock16ToResult(
        LocalTensor<float>& result, LocalTensor<float>& src,
        uint32_t row, uint32_t col, uint32_t stride)
    {
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                result.SetValue((row + i) * stride + (col + j),
                    src.GetValue(i * LEAF_BLOCK_SIZE + j));
            }
        }
    }

    // Read a 16x16 float32 block from the result matrix at (row, col)
    __aicore__ inline void ReadBlock16FromResult(
        LocalTensor<float>& dst, LocalTensor<float>& result,
        uint32_t row, uint32_t col, uint32_t stride)
    {
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                dst.SetValue(i * LEAF_BLOCK_SIZE + j,
                    result.GetValue((row + i) * stride + (col + j)));
            }
        }
    }

    // Extract a 16x16 block from D_TYPE input L_io at (row, col) into float32 dst
    __aicore__ inline void ExtractBlock16FromInput(
        LocalTensor<float>& dst, LocalTensor<D_TYPE>& L_io,
        uint32_t row, uint32_t col, uint32_t stride)
    {
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                dst.SetValue(i * LEAF_BLOCK_SIZE + j,
                    static_cast<float>(L_io.GetValue((row + i) * stride + (col + j))));
            }
        }
    }

    // MBH Layer 1 pair operating entirely in UB float32
    // Computes BL = -(L22inv * L21 * L11inv) for adjacent blocks
    __aicore__ inline void MbhLayer1PairUB(
        LocalTensor<float>& result, LocalTensor<D_TYPE>& L_io,
        uint32_t blk1, uint32_t blk2, uint32_t stride,
        LocalTensor<float>& T, LocalTensor<float>& mmTmp)
    {
        uint32_t r1 = blk1 * LEAF_BLOCK_SIZE;
        uint32_t r2 = blk2 * LEAF_BLOCK_SIZE;
        uint32_t c1 = blk1 * LEAF_BLOCK_SIZE;
        uint32_t c2 = blk2 * LEAF_BLOCK_SIZE;

        // L11inv from result[r1, c1] (already computed by MCH)
        // L22inv from result[r2, c2] (already computed by MCH)
        // L21 from original input L_io[r2, c1]

        // T = L21 * L11inv (use T as temp for L21, mmTmp for matmul result)
        // First extract L21 into T
        ExtractBlock16FromInput(T, L_io, r2, c1, stride);

        // mmTmp = T(L21) * L11inv - need to read L11inv from result
        // Do manual 16x16 matmul reading from result directly
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < LEAF_BLOCK_SIZE; k++) {
                    float a_val = T.GetValue(i * LEAF_BLOCK_SIZE + k);
                    float b_val = result.GetValue((r1 + k) * stride + (c1 + j));
                    sum += a_val * b_val;
                }
                mmTmp.SetValue(i * LEAF_BLOCK_SIZE + j, sum);
            }
        }

        // BL = -(L22inv * mmTmp) - read L22inv from result directly
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < LEAF_BLOCK_SIZE; k++) {
                    float a_val = result.GetValue((r2 + i) * stride + (c2 + k));
                    float b_val = mmTmp.GetValue(k * LEAF_BLOCK_SIZE + j);
                    sum += a_val * b_val;
                }
                // Write -BL directly into result at [r2, c1]
                result.SetValue((r2 + i) * stride + (c1 + j), -sum);
            }
        }
    }

    // MBH Layer 2 operating entirely in UB float32
    // Handles a 32x32 sub-problem starting at baseOff
    // BL[32x32] = -(L22inv[32x32] * L21_orig[32x32] * L11inv[32x32])
    // where L11inv/L22inv are the top/bottom 32x32 halves of the result
    __aicore__ inline void MbhLayer2UB(
        LocalTensor<float>& result, LocalTensor<D_TYPE>& L_io,
        uint32_t stride, uint32_t baseOff,
        LocalTensor<float>& T, LocalTensor<float>& mmTmp)
    {
        constexpr uint32_t HALF = 32;
        // Direct scalar computation: BL[i][j] = -sum_k(L22inv[i][k] * sum_m(L21[k][m] * L11inv[m][j]))
        for (uint32_t i = 0; i < HALF; i++) {
            for (uint32_t j = 0; j < HALF; j++) {
                float bl_val = 0.0f;
                for (uint32_t k = 0; k < HALF; k++) {
                    float l22inv_ik = result.GetValue(
                        (baseOff + HALF + i) * stride + (baseOff + HALF + k));
                    float t_kj = 0.0f;
                    for (uint32_t m = 0; m < HALF; m++) {
                        float l21_km = static_cast<float>(
                            L_io.GetValue((baseOff + HALF + k) * stride + (baseOff + m)));
                        float l11inv_mj = result.GetValue(
                            (baseOff + m) * stride + (baseOff + j));
                        t_kj += l21_km * l11inv_mj;
                    }
                    bl_val += l22inv_ik * t_kj;
                }
                result.SetValue(
                    (baseOff + HALF + i) * stride + (baseOff + j), -bl_val);
            }
        }
    }

    // MBH Layer 3 operating entirely in UB float32
    // Handles the full 64x64 sub-problem (top half = L11inv, bottom half = L22inv)
    __aicore__ inline void MbhLayer3UB(
        LocalTensor<float>& result, LocalTensor<D_TYPE>& L_io,
        uint32_t stride,
        LocalTensor<float>& T, LocalTensor<float>& mmTmp)
    {
        constexpr uint32_t HALF = 64;
        // Direct scalar computation: BL[i][j] = -sum_k(L22inv[i][k] * sum_m(L21[k][m] * L11inv[m][j]))
        for (uint32_t i = 0; i < HALF; i++) {
            for (uint32_t j = 0; j < HALF; j++) {
                float bl_val = 0.0f;
                for (uint32_t k = 0; k < HALF; k++) {
                    float l22inv_ik = result.GetValue(
                        (HALF + i) * stride + (HALF + k));
                    float t_kj = 0.0f;
                    for (uint32_t m = 0; m < HALF; m++) {
                        float l21_km = static_cast<float>(
                            L_io.GetValue((HALF + k) * stride + m));
                        float l11inv_mj = result.GetValue(m * stride + j);
                        t_kj += l21_km * l11inv_mj;
                    }
                    bl_val += l22inv_ik * t_kj;
                }
                result.SetValue((HALF + i) * stride + j, -bl_val);
            }
        }
    }

    // MBH Layer 1 pair: compute BL = -(L22inv * L21 * L11inv) for adjacent blocks
    // All in float32 UB, only final BL written to yGlobal
    __aicore__ inline void MbhLayer1Pair(
        uint32_t gmBase, uint32_t blk1, uint32_t blk2, uint32_t stride)
    {
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> L11inv = work[0 * LEAF_ELEMENTS];
        LocalTensor<float> L22inv = work[1 * LEAF_ELEMENTS];
        LocalTensor<float> L21    = work[2 * LEAF_ELEMENTS];
        LocalTensor<float> T      = work[3 * LEAF_ELEMENTS];
        LocalTensor<float> botL   = work[4 * LEAF_ELEMENTS];

        uint32_t r1 = blk1 * LEAF_BLOCK_SIZE;
        uint32_t r2 = blk2 * LEAF_BLOCK_SIZE;
        uint32_t c1 = blk1 * LEAF_BLOCK_SIZE;
        uint32_t c2 = blk2 * LEAF_BLOCK_SIZE;

        ReadBlock16FromY(L11inv, gmBase, r1, c1, stride);
        ReadBlock16FromY(L22inv, gmBase, r2, c2, stride);
        ReadBlock16FromX(L21, gmBase, r2, c1, stride);

        NsSolveTril::MatMul16x16(T, L21, L11inv);
        NsSolveTril::MatMul16x16(botL, L22inv, T);
        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
            botL.SetValue(i, -botL.GetValue(i));
        }
        WriteBlock16ToY(botL, gmBase, r2, c1, stride);
    }

    // Helper: recompute Layer 1 BL block in float32
    // BL = -(L22inv_diag * L21_orig * L11inv_diag) for a pair (blk1, blk2)
    // Returns the BL block in dst
    __aicore__ inline void RecomputeLayer1BL(
        LocalTensor<float>& dst,
        uint32_t gmBase, uint32_t blk1, uint32_t blk2, uint32_t stride)
    {
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> tmpA = work[6 * LEAF_ELEMENTS];
        LocalTensor<float> tmpB = work[7 * LEAF_ELEMENTS];
        LocalTensor<float> tmpC = work[8 * LEAF_ELEMENTS];

        uint32_t r1 = blk1 * LEAF_BLOCK_SIZE;
        uint32_t r2 = blk2 * LEAF_BLOCK_SIZE;
        uint32_t c1 = blk1 * LEAF_BLOCK_SIZE;
        uint32_t c2 = blk2 * LEAF_BLOCK_SIZE;

        // Read MCH diagonal blocks (these are accurate - final results)
        ReadBlock16FromY(tmpA, gmBase, r1, c1, stride); // L11inv_diag
        ReadBlock16FromY(tmpB, gmBase, r2, c2, stride); // L22inv_diag
        // Read L21 from original input
        ReadBlock16FromX(tmpC, gmBase, r2, c1, stride); // L21_orig

        // T = L21 * L11inv
        NsSolveTril::MatMul16x16(dst, tmpC, tmpA);
        // BL = -(L22inv * T)
        NsSolveTril::MatMul16x16(tmpC, tmpB, dst);
        for (uint32_t e = 0; e < LEAF_ELEMENTS; e++)
            dst.SetValue(e, -tmpC.GetValue(e));
    }

    // Fused MBH Layer 2 with full float32 recomputation of off-diagonal blocks
    __aicore__ inline void MbhLayer2Fused(
        uint32_t gmBase, uint32_t stride, uint32_t halfDim, uint32_t baseOff)
    {
        uint32_t numBlks = halfDim / LEAF_BLOCK_SIZE; // 2

        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> blkA = work[0 * LEAF_ELEMENTS];
        LocalTensor<float> blkB = work[1 * LEAF_ELEMENTS];
        LocalTensor<float> blkC = work[2 * LEAF_ELEMENTS];
        LocalTensor<float> Tcol0 = work[3 * LEAF_ELEMENTS];
        LocalTensor<float> Tcol1 = work[4 * LEAF_ELEMENTS];
        LocalTensor<float> L11inv_offdiag = work[5 * LEAF_ELEMENTS];
        // Use work[9] for L22inv_offdiag (MCH is done, slot available)
        LocalTensor<float> L22inv_offdiag = work[9 * LEAF_ELEMENTS];

        // Precompute both off-diagonal blocks in float32
        uint32_t topBlk1 = baseOff / LEAF_BLOCK_SIZE;
        uint32_t topBlk2 = topBlk1 + 1;
        RecomputeLayer1BL(L11inv_offdiag, gmBase, topBlk1, topBlk2, stride);

        uint32_t botBlk1 = (baseOff + halfDim) / LEAF_BLOCK_SIZE;
        uint32_t botBlk2 = botBlk1 + 1;
        RecomputeLayer1BL(L22inv_offdiag, gmBase, botBlk1, botBlk2, stride);

        // For each output column j of BL:
        for (uint32_t bj = 0; bj < numBlks; bj++) {
            // Compute T[:,j] = L21_big * L11inv[:,j]
            for (uint32_t bk = 0; bk < numBlks; bk++) {
                LocalTensor<float>& Tcur = (bk == 0) ? Tcol0 : Tcol1;
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) Tcur.SetValue(e, 0.0f);
                for (uint32_t bm = 0; bm < numBlks; bm++) {
                    ReadBlock16FromX(blkA, gmBase,
                        baseOff + halfDim + bk * LEAF_BLOCK_SIZE,
                        baseOff + bm * LEAF_BLOCK_SIZE, stride);
                    if (bm == 1 && bj == 0) {
                        // Off-diagonal block of L11inv: use float32 recomputed
                        NsSolveTril::MatMul16x16Acc(Tcur, blkA, L11inv_offdiag);
                    } else if (bm == bj) {
                        // Diagonal block of L11inv
                        ReadBlock16FromY(blkB, gmBase,
                            baseOff + bm * LEAF_BLOCK_SIZE,
                            baseOff + bj * LEAF_BLOCK_SIZE, stride);
                        NsSolveTril::MatMul16x16Acc(Tcur, blkA, blkB);
                    }
                    // bm > bj or (bm < bj and bm != bj): zero block, skip
                }
            }

            // Compute BL[:,j] = -L22inv * T[:,j]
            for (uint32_t bi = 0; bi < numBlks; bi++) {
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) blkC.SetValue(e, 0.0f);
                for (uint32_t bk = 0; bk < numBlks; bk++) {
                    if (bi == 1 && bk == 0) {
                        // Off-diagonal block of L22inv
                        NsSolveTril::MatMul16x16Acc(blkC, L22inv_offdiag, Tcol0);
                    } else if (bi == bk) {
                        // Diagonal block of L22inv
                        ReadBlock16FromY(blkA, gmBase,
                            baseOff + halfDim + bi * LEAF_BLOCK_SIZE,
                            baseOff + halfDim + bk * LEAF_BLOCK_SIZE, stride);
                        LocalTensor<float>& Tcur = (bk == 0) ? Tcol0 : Tcol1;
                        NsSolveTril::MatMul16x16Acc(blkC, blkA, Tcur);
                    }
                }
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++)
                    blkC.SetValue(e, -blkC.GetValue(e));
                WriteBlock16ToY(blkC, gmBase,
                    baseOff + halfDim + bi * LEAF_BLOCK_SIZE,
                    baseOff + bj * LEAF_BLOCK_SIZE, stride);
            }
        }
    }

    // Fused MBH Layer 3: same principle but numBlks=4
    // T column has 4 blocks - use work[3..6] for T[0..3][j]
    __aicore__ inline void MbhLayer3Fused(
        uint32_t gmBase, uint32_t stride, uint32_t halfDim)
    {
        uint32_t numBlks = halfDim / LEAF_BLOCK_SIZE; // 4

        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> blkA = work[0 * LEAF_ELEMENTS];
        LocalTensor<float> blkB = work[1 * LEAF_ELEMENTS];
        LocalTensor<float> blkC = work[2 * LEAF_ELEMENTS];
        // T column: T[0..3][j] stored in work[3..6]
        // (need 4 blocks for T column + 3 for computation = 7, fits in 10)

        for (uint32_t bj = 0; bj < numBlks; bj++) {
            // Compute T[:,j]
            for (uint32_t bk = 0; bk < numBlks; bk++) {
                LocalTensor<float> Tcur = work[(3 + bk) * LEAF_ELEMENTS];
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) Tcur.SetValue(e, 0.0f);
                for (uint32_t bm = 0; bm < numBlks; bm++) {
                    ReadBlock16FromX(blkA, gmBase,
                        halfDim + bk * LEAF_BLOCK_SIZE,
                        bm * LEAF_BLOCK_SIZE, stride);
                    ReadBlock16FromY(blkB, gmBase,
                        bm * LEAF_BLOCK_SIZE,
                        bj * LEAF_BLOCK_SIZE, stride);
                    NsSolveTril::MatMul16x16Acc(Tcur, blkA, blkB);
                }
            }

            // Compute BL[:,j] = -L22inv * T[:,j]
            for (uint32_t bi = 0; bi < numBlks; bi++) {
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) blkC.SetValue(e, 0.0f);
                for (uint32_t bk = 0; bk < numBlks; bk++) {
                    ReadBlock16FromY(blkA, gmBase,
                        halfDim + bi * LEAF_BLOCK_SIZE,
                        halfDim + bk * LEAF_BLOCK_SIZE, stride);
                    LocalTensor<float> Tcur = work[(3 + bk) * LEAF_ELEMENTS];
                    NsSolveTril::MatMul16x16Acc(blkC, blkA, Tcur);
                }
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++)
                    blkC.SetValue(e, -blkC.GetValue(e));
                WriteBlock16ToY(blkC, gmBase,
                    halfDim + bi * LEAF_BLOCK_SIZE,
                    bj * LEAF_BLOCK_SIZE, stride);
            }
        }
    }

    // MBH Layer 1 pair: reads L21 from xGlobal (no L_io needed)
    __aicore__ inline void MbhLayer1PairUB_GM(
        LocalTensor<float>& result, uint32_t gmBase,
        uint32_t blk1, uint32_t blk2, uint32_t stride,
        LocalTensor<float>& T, LocalTensor<float>& mmTmp)
    {
        uint32_t r1 = blk1 * LEAF_BLOCK_SIZE;
        uint32_t r2 = blk2 * LEAF_BLOCK_SIZE;
        uint32_t c1 = blk1 * LEAF_BLOCK_SIZE;
        uint32_t c2 = blk2 * LEAF_BLOCK_SIZE;

        // Read L21 from xGlobal into T
        ReadBlock16FromX(T, gmBase, r2, c1, stride);

        // mmTmp = T(L21) * L11inv (read L11inv from result)
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < LEAF_BLOCK_SIZE; k++) {
                    float a_val = T.GetValue(i * LEAF_BLOCK_SIZE + k);
                    float b_val = result.GetValue((r1 + k) * stride + (c1 + j));
                    sum += a_val * b_val;
                }
                mmTmp.SetValue(i * LEAF_BLOCK_SIZE + j, sum);
            }
        }

        // BL = -(L22inv * mmTmp) - read L22inv from result
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < LEAF_BLOCK_SIZE; k++) {
                    float a_val = result.GetValue((r2 + i) * stride + (c2 + k));
                    float b_val = mmTmp.GetValue(k * LEAF_BLOCK_SIZE + j);
                    sum += a_val * b_val;
                }
                result.SetValue((r2 + i) * stride + (c1 + j), -sum);
            }
        }
    }

    // MBH Layer 2 GM version: reads L21 from xGlobal block-by-block
    __aicore__ inline void MbhLayer2UB_GM(
        LocalTensor<float>& result, uint32_t gmBase,
        uint32_t stride, uint32_t baseOff,
        LocalTensor<float>& T, LocalTensor<float>& mmTmp)
    {
        constexpr uint32_t HALF = 32;
        constexpr uint32_t NUM_BLKS = HALF / LEAF_BLOCK_SIZE; // 2

        // BL = -(L22inv * L21 * L11inv) computed block-by-block
        // For each output block (bi, bj):
        for (uint32_t bj = 0; bj < NUM_BLKS; bj++) {
            for (uint32_t bi = 0; bi < NUM_BLKS; bi++) {
                // Accumulate BL[bi,bj] = sum_bk( L22inv[bi,bk] * sum_bm( L21[bk,bm] * L11inv[bm,bj] ) )
                // Initialize accumulator to 0
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) mmTmp.SetValue(e, 0.0f);

                for (uint32_t bk = 0; bk < NUM_BLKS; bk++) {
                    // Compute T_kj = sum_bm(L21[bk,bm] * L11inv[bm,bj])
                    // T stored in T buffer
                    for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) T.SetValue(e, 0.0f);
                    for (uint32_t bm = 0; bm < NUM_BLKS; bm++) {
                        // Read L21[bk,bm] from xGlobal
                        LocalTensor<float> work = workBuf.Get<float>();
                        LocalTensor<float> tmpBlk = work[stride * stride + 2 * LEAF_ELEMENTS]; // reuse Y slot
                        ReadBlock16FromX(tmpBlk, gmBase,
                            baseOff + HALF + bk * LEAF_BLOCK_SIZE,
                            baseOff + bm * LEAF_BLOCK_SIZE, stride);
                        // Read L11inv[bm,bj] from result
                        LocalTensor<float> tmpBlk2 = work[stride * stride + 3 * LEAF_ELEMENTS]; // reuse temp slot
                        ReadBlock16FromResult(tmpBlk2, result,
                            baseOff + bm * LEAF_BLOCK_SIZE,
                            baseOff + bj * LEAF_BLOCK_SIZE, stride);
                        // T += L21[bk,bm] * L11inv[bm,bj]
                        NsSolveTril::MatMul16x16Acc(T, tmpBlk, tmpBlk2);
                    }
                    // Read L22inv[bi,bk] from result
                    LocalTensor<float> work2 = workBuf.Get<float>();
                    LocalTensor<float> tmpBlk3 = work2[stride * stride + 2 * LEAF_ELEMENTS]; // reuse Y slot
                    ReadBlock16FromResult(tmpBlk3, result,
                        baseOff + HALF + bi * LEAF_BLOCK_SIZE,
                        baseOff + HALF + bk * LEAF_BLOCK_SIZE, stride);
                    // mmTmp += L22inv[bi,bk] * T_kj
                    NsSolveTril::MatMul16x16Acc(mmTmp, tmpBlk3, T);
                }
                // Write -BL[bi,bj] into result
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++)
                    mmTmp.SetValue(e, -mmTmp.GetValue(e));
                WriteBlock16ToResult(result, mmTmp,
                    baseOff + HALF + bi * LEAF_BLOCK_SIZE,
                    baseOff + bj * LEAF_BLOCK_SIZE, stride);
            }
        }
    }

    // MBH Layer 3 GM version: reads L21 from xGlobal block-by-block
    __aicore__ inline void MbhLayer3UB_GM(
        LocalTensor<float>& result, uint32_t gmBase,
        uint32_t stride,
        LocalTensor<float>& T, LocalTensor<float>& mmTmp)
    {
        constexpr uint32_t HALF = 64;
        constexpr uint32_t NUM_BLKS = HALF / LEAF_BLOCK_SIZE; // 4

        // BL = -(L22inv * L21 * L11inv) computed block-by-block
        for (uint32_t bj = 0; bj < NUM_BLKS; bj++) {
            for (uint32_t bi = 0; bi < NUM_BLKS; bi++) {
                // Accumulate BL[bi,bj]
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) mmTmp.SetValue(e, 0.0f);

                for (uint32_t bk = 0; bk < NUM_BLKS; bk++) {
                    // Compute T_kj = sum_bm(L21[bk,bm] * L11inv[bm,bj])
                    for (uint32_t e = 0; e < LEAF_ELEMENTS; e++) T.SetValue(e, 0.0f);
                    for (uint32_t bm = 0; bm < NUM_BLKS; bm++) {
                        LocalTensor<float> work = workBuf.Get<float>();
                        LocalTensor<float> tmpBlk = work[stride * stride + 2 * LEAF_ELEMENTS];
                        ReadBlock16FromX(tmpBlk, gmBase,
                            HALF + bk * LEAF_BLOCK_SIZE,
                            bm * LEAF_BLOCK_SIZE, stride);
                        LocalTensor<float> tmpBlk2 = work[stride * stride + 3 * LEAF_ELEMENTS];
                        ReadBlock16FromResult(tmpBlk2, result,
                            bm * LEAF_BLOCK_SIZE,
                            bj * LEAF_BLOCK_SIZE, stride);
                        NsSolveTril::MatMul16x16Acc(T, tmpBlk, tmpBlk2);
                    }
                    // Read L22inv[bi,bk] from result
                    LocalTensor<float> work2 = workBuf.Get<float>();
                    LocalTensor<float> tmpBlk3 = work2[stride * stride + 2 * LEAF_ELEMENTS];
                    ReadBlock16FromResult(tmpBlk3, result,
                        HALF + bi * LEAF_BLOCK_SIZE,
                        HALF + bk * LEAF_BLOCK_SIZE, stride);
                    NsSolveTril::MatMul16x16Acc(mmTmp, tmpBlk3, T);
                }
                for (uint32_t e = 0; e < LEAF_ELEMENTS; e++)
                    mmTmp.SetValue(e, -mmTmp.GetValue(e));
                WriteBlock16ToResult(result, mmTmp,
                    HALF + bi * LEAF_BLOCK_SIZE,
                    bj * LEAF_BLOCK_SIZE, stride);
            }
        }
    }

    // ========================================================================
    // MBH_2 path (n=64): Full UB float32 - no GM intermediate read-back
    // Same approach as MBH_1: keep entire result matrix in UB as float32
    // ========================================================================
    __aicore__ inline void ProcessMbh2()
    {
        uint32_t startTask = blockIdx_ * taskPerCore_;
        uint32_t endTask = startTask + taskPerCore_;
        if (endTask > batchSize_) endTask = batchSize_;

        constexpr uint32_t DIM = 64;
        constexpr uint32_t MATRIX_SIZE = DIM * DIM;  // 4096

        for (uint32_t batchIdx = startTask; batchIdx < endTask; batchIdx++) {
            uint32_t batchGmOffset = batchIdx * MATRIX_SIZE;

            // Work buffer layout (named offsets from solve_tril_common.h):
            // [result: MBH2_RESULT_SIZE floats][X][A][Y][temp][mmTmp]
            constexpr uint32_t MCH_X_OFFSET     = MBH2_RESULT_SIZE + 0 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_A_OFFSET     = MBH2_RESULT_SIZE + 1 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_Y_OFFSET     = MBH2_RESULT_SIZE + 2 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_TEMP_OFFSET  = MBH2_RESULT_SIZE + 3 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_MMTMP_OFFSET = MBH2_RESULT_SIZE + 4 * LEAF_ELEMENTS;

            LocalTensor<float> work = workBuf.Get<float>();
            LocalTensor<float> result = work[0];
            LocalTensor<float> X      = work[MCH_X_OFFSET];
            LocalTensor<float> A      = work[MCH_A_OFFSET];
            LocalTensor<float> Y      = work[MCH_Y_OFFSET];
            LocalTensor<float> temp   = work[MCH_TEMP_OFFSET];
            LocalTensor<float> mmTmp  = work[MCH_MMTMP_OFFSET];

            // Initialize result to 0
            for (uint32_t i = 0; i < MATRIX_SIZE; i++) result.SetValue(i, 0.0f);

            // Phase 1: MCH on 4 diagonal 16x16 blocks (read from xGlobal)
            for (uint32_t blk = 0; blk < 4; blk++) {
                uint32_t row = blk * LEAF_BLOCK_SIZE;
                uint32_t col = blk * LEAF_BLOCK_SIZE;
                ReadBlock16FromX(A, batchGmOffset, row, col, DIM);
                MchInverse(X, A, X, Y, temp, mmTmp);
                WriteBlock16ToResult(result, X, row, col, DIM);
            }

            // Phase 2: MBH Layer 1 - pairs (0,1) and (2,3)
            MbhLayer1PairUB_GM(result, batchGmOffset, 0, 1, DIM, X, mmTmp);
            MbhLayer1PairUB_GM(result, batchGmOffset, 2, 3, DIM, X, mmTmp);

            // Phase 3: MBH Layer 2 - 32x32 block multiply
            MbhLayer2UB_GM(result, batchGmOffset, DIM, 0, X, mmTmp);

            // CopyOut: block-by-block (write ALL blocks including upper triangle as 0)
            for (uint32_t br = 0; br < 4; br++) {
                for (uint32_t bc = 0; bc < 4; bc++) {
                    uint32_t row = br * LEAF_BLOCK_SIZE;
                    uint32_t col = bc * LEAF_BLOCK_SIZE;
                    LocalTensor<float> blkSrc = work[MATRIX_SIZE + 0 * LEAF_ELEMENTS];
                    if (bc <= br) {
                        ReadBlock16FromResult(blkSrc, result, row, col, DIM);
                    } else {
                        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) blkSrc.SetValue(i, 0.0f);
                    }
                    WriteBlock16ToY(blkSrc, batchGmOffset, row, col, DIM);
                }
            }
        }
    }

    // ========================================================================
    // MBH_3 path (n=128): Full UB float32 result, block-by-block GM I/O
    // inQueue/outQueue only hold 16x16 blocks to avoid UB overflow for fp32
    // ========================================================================
    __aicore__ inline void ProcessMbh3()
    {
        uint32_t startTask = blockIdx_ * taskPerCore_;
        uint32_t endTask = startTask + taskPerCore_;
        if (endTask > batchSize_) endTask = batchSize_;

        constexpr uint32_t DIM = 128;
        constexpr uint32_t MATRIX_SIZE = DIM * DIM;  // 16384

        for (uint32_t batchIdx = startTask; batchIdx < endTask; batchIdx++) {
            uint32_t batchGmOffset = batchIdx * MATRIX_SIZE;

            // Work buffer layout (named offsets from solve_tril_common.h):
            // [result: MBH3_RESULT_SIZE floats][X][A][Y][temp][mmTmp]
            constexpr uint32_t MCH_X_OFFSET     = MBH3_RESULT_SIZE + 0 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_A_OFFSET     = MBH3_RESULT_SIZE + 1 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_Y_OFFSET     = MBH3_RESULT_SIZE + 2 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_TEMP_OFFSET  = MBH3_RESULT_SIZE + 3 * LEAF_ELEMENTS;
            constexpr uint32_t MCH_MMTMP_OFFSET = MBH3_RESULT_SIZE + 4 * LEAF_ELEMENTS;

            LocalTensor<float> work = workBuf.Get<float>();
            LocalTensor<float> result = work[0];
            LocalTensor<float> X      = work[MCH_X_OFFSET];
            LocalTensor<float> A      = work[MCH_A_OFFSET];
            LocalTensor<float> Y      = work[MCH_Y_OFFSET];
            LocalTensor<float> temp   = work[MCH_TEMP_OFFSET];
            LocalTensor<float> mmTmp  = work[MCH_MMTMP_OFFSET];

            // Initialize result to 0
            for (uint32_t i = 0; i < MATRIX_SIZE; i++) result.SetValue(i, 0.0f);

            // Phase 1: MCH on 8 diagonal 16x16 blocks (read from xGlobal)
            for (uint32_t blk = 0; blk < 8; blk++) {
                uint32_t row = blk * LEAF_BLOCK_SIZE;
                uint32_t col = blk * LEAF_BLOCK_SIZE;
                ReadBlock16FromX(A, batchGmOffset, row, col, DIM);
                MchInverse(X, A, X, Y, temp, mmTmp);
                WriteBlock16ToResult(result, X, row, col, DIM);
            }

            // Phase 2: MBH Layer 1 - 4 pairs (read L21 from xGlobal)
            MbhLayer1PairUB_GM(result, batchGmOffset, 0, 1, DIM, X, mmTmp);
            MbhLayer1PairUB_GM(result, batchGmOffset, 2, 3, DIM, X, mmTmp);
            MbhLayer1PairUB_GM(result, batchGmOffset, 4, 5, DIM, X, mmTmp);
            MbhLayer1PairUB_GM(result, batchGmOffset, 6, 7, DIM, X, mmTmp);

            // Phase 3: MBH Layer 2 - two 32x32 sub-problems
            MbhLayer2UB_GM(result, batchGmOffset, DIM, 0, X, mmTmp);
            MbhLayer2UB_GM(result, batchGmOffset, DIM, 64, X, mmTmp);

            // Phase 4: MBH Layer 3 - one 64x64 sub-problem
            MbhLayer3UB_GM(result, batchGmOffset, DIM, X, mmTmp);

            // Phase 5: CopyOut block-by-block (write ALL blocks including upper triangle as 0)
            for (uint32_t br = 0; br < 8; br++) {
                for (uint32_t bc = 0; bc < 8; bc++) {
                    uint32_t row = br * LEAF_BLOCK_SIZE;
                    uint32_t col = bc * LEAF_BLOCK_SIZE;
                    LocalTensor<float> blkSrc = work[MATRIX_SIZE + 0 * LEAF_ELEMENTS];
                    if (bc <= br) {
                        ReadBlock16FromResult(blkSrc, result, row, col, DIM);
                    } else {
                        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) blkSrc.SetValue(i, 0.0f);
                    }
                    WriteBlock16ToY(blkSrc, batchGmOffset, row, col, DIM);
                }
            }
        }
    }
};

template <typename D_TYPE, int MBH_LEVELS>
__global__ __aicore__ void solve_tril(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SolveTrilTilingData);
    GET_TILING_DATA(tilingData, tiling);
    SolveTrilKernel<D_TYPE, MBH_LEVELS> op;
    op.Init(x, y, workspace, &tilingData);
    op.Process();
}
