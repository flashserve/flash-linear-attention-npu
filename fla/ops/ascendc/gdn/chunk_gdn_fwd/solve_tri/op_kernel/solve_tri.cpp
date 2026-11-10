/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
#include "kernel_operator.h"
#include <type_traits>

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/solve_tri_ascend950_16.h"
#include "arch35/solve_tri_ascend950_32.h"
#include "arch35/solve_tri_ascend950_64.h"
#include "arch35/solve_tri_ascend950_128.h"
#else
#include "lib/matmul_intf.h"
#include "solve_tri_cube.h"
#include "solve_tri_vector.h"
#include "solve_tri_fp32.h"
#include "solve_tri_vector_fp32.h"
#endif

using namespace AscendC;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <typename T>
__aicore__ inline void RunSolveTriArch35(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR x_out,
                                     GM_ADDR workspace, GM_ADDR tiling)
{
    // tilingKey = chunkSize ∈ {16,32,64,128}
    // TILING_KEY_IS / KERNEL_TASK_TYPE 必须使用整型字面量，不能传 constexpr 变量
    if constexpr (std::is_same_v<T, float>) {
        return; // Host rejects FP32 on arch35.
    } else {
    if (TILING_KEY_IS(64)) {
        KERNEL_TASK_TYPE(64, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA(tilingData, tiling);
        SolveTri64<T, T> op;
        op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(128)) {
        KERNEL_TASK_TYPE(128, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA(tilingData, tiling);
        SolveTri128<T, T> op;
        op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(16)) {
        KERNEL_TASK_TYPE(16, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA(tilingData, tiling);
        SolveTri16<T, T> op;
        op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(32)) {
        KERNEL_TASK_TYPE(32, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA(tilingData, tiling);
        SolveTri32<T, T> op;
        op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
        op.Process();
    }
    }
}
__global__ __aicore__ void solve_tri(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR x_out,
                                     GM_ADDR workspace, GM_ADDR tiling)
{
    RunSolveTriArch35<DTYPE_X>(x, cu_seqlens, chunk_indices, x_out, workspace, tiling);
}
#else
extern "C" __global__ __aicore__ void solve_tri(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR x_out,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    // tilingKey = chunkSize；兼容旧 key=1
    if (TILING_KEY_IS(1) || TILING_KEY_IS(16) || TILING_KEY_IS(32) || TILING_KEY_IS(64) || TILING_KEY_IS(128)) {
        if (TILING_KEY_IS(1)) {
            KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        } else if (TILING_KEY_IS(16)) {
            KERNEL_TASK_TYPE(16, KERNEL_TYPE_MIX_AIC_1_2);
        } else if (TILING_KEY_IS(32)) {
            KERNEL_TASK_TYPE(32, KERNEL_TYPE_MIX_AIC_1_2);
        } else if (TILING_KEY_IS(64)) {
            KERNEL_TASK_TYPE(64, KERNEL_TYPE_MIX_AIC_1_2);
        } else {
            KERNEL_TASK_TYPE(128, KERNEL_TYPE_MIX_AIC_1_2);
        }

        if (tilingData.matrixSize == 64) {
            if ASCEND_IS_AIC {
                NsSolveTri::SolveTriCubeFp32<DTYPE_X> op;
                op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                op.Process();
            }
            if ASCEND_IS_AIV {
                NsSolveTri::SolveTriVectorFp32<DTYPE_X> op;
                op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                op.Process();
            }
            // Cross-core users and their local resources have finished.
            PipeBarrier<PIPE_ALL>();
            set_ffts_base_addr(0);
        } else {
            if ASCEND_IS_AIV {
                if (tilingData.matrixSize == 16) {
                    NsSolveTri::SolveTriVectorFullFp32<DTYPE_X, 16> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, &tilingData);
                    op.Process();
                } else if (tilingData.matrixSize == 32) {
                    NsSolveTri::SolveTriVectorFullFp32<DTYPE_X, 32> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, &tilingData);
                    op.Process();
                } else if (tilingData.matrixSize == 128) {
                    NsSolveTri::SolveTriVectorFullFp32<DTYPE_X, 128> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, &tilingData);
                    op.Process();
                }
            }
        }
    }
}
#endif
