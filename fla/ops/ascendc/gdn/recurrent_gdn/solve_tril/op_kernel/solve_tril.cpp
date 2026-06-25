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
                                                GM_ADDR mch_out, GM_ADDR zero_mat, GM_ADDR eye_mat,
                                                GM_ADDR x_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        // Ascend950(arch35) 不支持纯 KERNEL_TYPE_AIC，cube 类算子统一使用
        // KERNEL_TYPE_MIX_AIC_1_2（与本仓库其他 arch35 cube 算子一致）。
        // AIC(cube) 做 Mmad/Fixpipe/L1 加载；AIV(vector) 生成辅助矩阵(I/-I/Zero)到 GM。
        // UB 优化(SOLVE_TRIL_MBH_UB_OPT, 950 自动开)下，AIV 还负责 MCH 输出 GM->UB 暂存与
        // 每层 raw UB->L1 提取（UB 位于 AIV），与 AIC 经 CrossCoreFlag 协作。
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);

        int64_t ms = tilingData.matrixSize;
        int64_t totalTiles = tilingData.totalTiles;
        int64_t tilesPerCore = tilingData.tilesPerCore;
        int64_t isLower = tilingData.isLower;

        if ASCEND_IS_AIC {
            if (ms == 16) {
                NsSolveTril::SolveTrilCube<16> op;
                op.Init(x, cu_seqlens, chunk_indices, mch_out, zero_mat, eye_mat, x_out, workspace, &tilingData);
                op.Process();
            } else if (ms == 32) {
                NsSolveTril::SolveTrilCube<32> op;
                op.Init(x, cu_seqlens, chunk_indices, mch_out, zero_mat, eye_mat, x_out, workspace, &tilingData);
                op.Process();
            } else if (ms == 64) {
                NsSolveTril::SolveTrilCube<64> op;
                op.Init(x, cu_seqlens, chunk_indices, mch_out, zero_mat, eye_mat, x_out, workspace, &tilingData);
                op.Process();
            } else if (ms == 128) {
                NsSolveTril::SolveTrilCube<128> op;
                op.Init(x, cu_seqlens, chunk_indices, mch_out, zero_mat, eye_mat, x_out, workspace, &tilingData);
                op.Process();
            }
        }

#if !SOLVE_TRIL_PLATFORM_ASCEND950
        if ASCEND_IS_AIV {
            if (ms == 16) {
                NsSolveTril::SolveTrilVector<16> op;
                op.Init(workspace, mch_out, totalTiles, ms, tilesPerCore, isLower);
                op.Process();
            } else if (ms == 32) {
                NsSolveTril::SolveTrilVector<32> op;
                op.Init(workspace, mch_out, totalTiles, ms, tilesPerCore, isLower);
                op.Process();
            } else if (ms == 64) {
                NsSolveTril::SolveTrilVector<64> op;
                op.Init(workspace, mch_out, totalTiles, ms, tilesPerCore, isLower);
                op.Process();
            } else if (ms == 128) {
                NsSolveTril::SolveTrilVector<128> op;
                op.Init(workspace, mch_out, totalTiles, ms, tilesPerCore, isLower);
                op.Process();
            }
        }
#endif
    }
}
