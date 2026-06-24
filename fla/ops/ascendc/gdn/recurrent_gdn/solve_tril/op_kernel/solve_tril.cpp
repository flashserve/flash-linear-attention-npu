/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "solve_tril.h"

using namespace AscendC;

// __global__ __aicore__ void solve_tril(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
//                                        GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
// {
//     AscendC::printf("wangwei: entering solve_tril kernel");
//     AscendCUtils::SetOverflow(1);
//     if (TILING_KEY_IS(1)) {
//         KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
//         GET_TILING_DATA(tilingData, tiling);
//         if ASCEND_IS_AIC {
//             // SolveTrilCube<DTYPE_X, DTYPE_X> op;
//             // op.Init(x, cu_seqlens, chunk_indices, out, workspace, &tilingData);
//             // op.Process();
//         }
//         if ASCEND_IS_AIV {
//             SolveTrilVec<DTYPE_X, DTYPE_X> op;
//             op.Init(x, cu_seqlens, chunk_indices, out, workspace, &tilingData);
//             op.Process();
//         }
//     }
// }

__global__ __aicore__ void solve_tril(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                       GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::printf("wangwei: entering solve_tril kernel");
    // AscendCUtils::SetOverflow(1);
    // TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA(tilingData, tiling);
        
        SolveTril<DTYPE_X, DTYPE_X> op;
        op.Init(x, cu_seqlens, chunk_indices, out, workspace, &tilingData);
        op.Process();
        
    }
}
