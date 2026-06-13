/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

/*!
 * \file solve_tril_cube.h
 * \brief MCH algorithm implementation using scalar float32 matmul
 *        Proven approach from probe validation (MERE = 0 on NPU)
 *
 * ============================================================================
 * DESIGN NOTE: Why scalar matmul instead of Cube unit
 * ============================================================================
 *
 * Two approaches to use the Cube unit for 16x16 matmul were investigated:
 *
 * 1. Mmad (low-level Cube API):
 *    - Attempted in early probe phase. Output was all zeros on NPU.
 *    - Root cause: Mmad requires NZ format data layout and specific L0A/L0B/L0C
 *      buffer management that conflicts with the Vector-only kernel architecture.
 *    - The kernel uses VECIN/VECOUT/VECCALC queues exclusively; Cube buffers
 *      (A1/A2/B1/B2/C1/C2/CO1/CO2) are not allocated.
 *
 * 2. Matmul high-level API (lib/matmul_intf.h):
 *    - Requires TCubeTiling structure passed from host side.
 *    - Requires TPipe to manage Cube-specific buffers (L1, L0A, L0B, L0C).
 *    - Requires the kernel to be compiled as a Cube+Vector mixed-mode kernel.
 *    - The current kernel is a pure Vector kernel. Integrating Matmul API would
 *      require restructuring: host tiling (add TCubeTiling computation), kernel
 *      buffer management (add Cube queues), and build configuration.
 *    - For 16x16 float32 matmul (256 MACs per output element, 4096 total MACs),
 *      the overhead of Cube setup/teardown per call likely exceeds the scalar
 *      computation time, since MCH requires many small matmuls per batch element.
 *
 * Current approach: Scalar float32 triple-loop matmul.
 * - Proven correct on NPU (MERE = 0 across all ST cases).
 * - Simple, predictable, no hardware-specific buffer management needed.
 * - Performance is acceptable for the target use case (small matrices).
 *
 * Known limitation: For larger matrix sizes or performance-critical paths,
 * migrating to Cube units would require a kernel architecture redesign
 * (separate Cube kernel or mixed-mode kernel with proper buffer allocation).
 * ============================================================================
 */

#ifndef __SOLVE_TRIL_CUBE_H__
#define __SOLVE_TRIL_CUBE_H__

#include "kernel_operator.h"

namespace NsSolveTril {

using namespace AscendC;

// 16x16 matrix multiply in float32: C = A_mat * B_mat (scalar loop)
__aicore__ inline void MatMul16x16(
    LocalTensor<float>& C,
    LocalTensor<float>& A_mat,
    LocalTensor<float>& B_mat)
{
    for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
        for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < LEAF_BLOCK_SIZE; k++) {
                sum += A_mat.GetValue(i * LEAF_BLOCK_SIZE + k) *
                       B_mat.GetValue(k * LEAF_BLOCK_SIZE + j);
            }
            C.SetValue(i * LEAF_BLOCK_SIZE + j, sum);
        }
    }
}

// Run MCH algorithm entirely in float32
// Input: work buffer partitioned as X[256] + A[256] + Y[256] + temp[256] + mmTmp[256]
// All offsets are in float elements
__aicore__ inline void RunMCH_F32(LocalTensor<float>& work)
{
    LocalTensor<float> X = work[0];
    LocalTensor<float> A = work[LEAF_ELEMENTS];
    LocalTensor<float> Y = work[2 * LEAF_ELEMENTS];
    LocalTensor<float> temp = work[3 * LEAF_ELEMENTS];
    LocalTensor<float> mmTmp = work[4 * LEAF_ELEMENTS];

    // Y = A * A
    MatMul16x16(Y, A, A);

    // 3 iterations
    for (uint32_t iter = 0; iter < MCH_ITERATIONS; iter++) {
        // temp = I + Y
        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
            temp.SetValue(i, Y.GetValue(i));
        }
        for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
            temp.SetValue(i * LEAF_BLOCK_SIZE + i,
                          temp.GetValue(i * LEAF_BLOCK_SIZE + i) + 1.0f);
        }

        // X_new = X * temp
        MatMul16x16(mmTmp, X, temp);
        for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
            X.SetValue(i, mmTmp.GetValue(i));
        }

        // Y_new = Y * Y (skip on last iteration)
        if (iter < MCH_ITERATIONS - 1) {
            MatMul16x16(mmTmp, Y, Y);
            for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
                Y.SetValue(i, mmTmp.GetValue(i));
            }
        }
    }
}

// 16x16 matrix multiply-accumulate: C += A_mat * B_mat (scalar loop, float32)
// Used for block matmul decomposition in MBH layers 2/3
__aicore__ inline void MatMul16x16Acc(
    LocalTensor<float>& C,
    LocalTensor<float>& A_mat,
    LocalTensor<float>& B_mat)
{
    for (uint32_t i = 0; i < LEAF_BLOCK_SIZE; i++) {
        for (uint32_t j = 0; j < LEAF_BLOCK_SIZE; j++) {
            float sum = C.GetValue(i * LEAF_BLOCK_SIZE + j);
            for (uint32_t k = 0; k < LEAF_BLOCK_SIZE; k++) {
                sum += A_mat.GetValue(i * LEAF_BLOCK_SIZE + k) *
                       B_mat.GetValue(k * LEAF_BLOCK_SIZE + j);
            }
            C.SetValue(i * LEAF_BLOCK_SIZE + j, sum);
        }
    }
}

// MBH block multiply: C = A_mat * B_mat (16x16, float32)
// Used for MBH layer 1: T = L21 * L11inv, result = L22inv * T
__aicore__ inline void MbhBlockMul16x16(
    LocalTensor<float>& C,
    LocalTensor<float>& A_mat,
    LocalTensor<float>& B_mat)
{
    MatMul16x16(C, A_mat, B_mat);
}

// Generic NxN matrix multiply in float32: C = A_mat * B_mat (scalar loop)
// A_mat is at offset aOff, B_mat at bOff, C at cOff within the same tensor
// Used for MBH layer 2 (32x32) and layer 3 (64x64)
__aicore__ inline void MatMulNxN(
    LocalTensor<float>& C, uint32_t cOff,
    LocalTensor<float>& A_mat, uint32_t aOff,
    LocalTensor<float>& B_mat, uint32_t bOff,
    uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < dim; k++) {
                sum += A_mat.GetValue(aOff + i * dim + k) *
                       B_mat.GetValue(bOff + k * dim + j);
            }
            C.SetValue(cOff + i * dim + j, sum);
        }
    }
}

// Negate all elements of a 16x16 float32 matrix: dst[i] = -src[i]
__aicore__ inline void Negate16x16(LocalTensor<float>& dst, LocalTensor<float>& src)
{
    for (uint32_t i = 0; i < LEAF_ELEMENTS; i++) {
        dst.SetValue(i, -src.GetValue(i));
    }
}

// Negate N*N elements in-place at given offset
__aicore__ inline void NegateNxN(LocalTensor<float>& buf, uint32_t off, uint32_t dim)
{
    uint32_t count = dim * dim;
    for (uint32_t i = 0; i < count; i++) {
        buf.SetValue(off + i, -buf.GetValue(off + i));
    }
}

} // namespace NsSolveTril

#endif // __SOLVE_TRIL_CUBE_H__
