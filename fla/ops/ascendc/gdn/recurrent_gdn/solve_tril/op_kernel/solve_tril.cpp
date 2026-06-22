/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "solve_tril_cube.h"
#if !SOLVE_TRIL_PLATFORM_ASCEND950
#include "solve_tril_vector.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void solve_tril(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                GM_ADDR x_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
#if SOLVE_TRIL_PLATFORM_ASCEND950
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIC);
#else
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
#endif

        int64_t ms = tilingData.matrixSize;
        int64_t totalTiles = tilingData.totalTiles;
        int64_t tilesPerCore = tilingData.tilesPerCore;

        if ASCEND_IS_AIC {
            if (ms == 16) {
                NsSolveTril::SolveTrilCube<16> op;
                op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                op.Process();
            } else if (ms == 32) {
                NsSolveTril::SolveTrilCube<32> op;
                op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                op.Process();
            } else if (ms == 64) {
                NsSolveTril::SolveTrilCube<64> op;
                op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                op.Process();
            } else if (ms == 128) {
                NsSolveTril::SolveTrilCube<128> op;
                op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                op.Process();
            }
        }

#if !SOLVE_TRIL_PLATFORM_ASCEND950
        if ASCEND_IS_AIV {
            if (ms == 16) {
                NsSolveTril::SolveTrilVector<16> op;
                op.Init(workspace, totalTiles, ms);
                op.Process();
            } else if (ms == 32) {
                NsSolveTril::SolveTrilVector<32> op;
                op.Init(workspace, totalTiles, ms);
                op.Process();
            } else if (ms == 64) {
                NsSolveTril::SolveTrilVector<64> op;
                op.Init(workspace, totalTiles, ms);
                op.Process();
            } else if (ms == 128) {
                NsSolveTril::SolveTrilVector<128> op;
                op.Init(workspace, totalTiles, ms);
                op.Process();
            }
        }
#endif
    }
}
