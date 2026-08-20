/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #include "kernel_operator.h"

 #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
 #include "arch35/solve_tri_ascend950.h"
 #else
 #include "lib/matmul_intf.h"
 #include "solve_tri_cube.h"
 #include "solve_tri_vector.h"
 #endif
 
 using namespace AscendC;
 
 #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
 __global__ __aicore__ void solve_tri(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                      GM_ADDR x_out, GM_ADDR workspace, GM_ADDR tiling)
 {
     // tilingKey = chunkSize ∈ {16,32,64,128}
     // TILING_KEY_IS / KERNEL_TASK_TYPE 必须使用整型字面量，不能传 constexpr 变量
     if (TILING_KEY_IS(64)) {
         KERNEL_TASK_TYPE(64, KERNEL_TYPE_MIX_AIC_1_2);
         GET_TILING_DATA(tilingData, tiling);
         SolveTri64<DTYPE_X, DTYPE_X> op;
         op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(128)) {
         KERNEL_TASK_TYPE(128, KERNEL_TYPE_MIX_AIC_1_2);
         GET_TILING_DATA(tilingData, tiling);
         SolveTri128<DTYPE_X, DTYPE_X> op;
         op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(16)) {
         KERNEL_TASK_TYPE(16, KERNEL_TYPE_MIX_AIC_1_2);
         GET_TILING_DATA(tilingData, tiling);
         SolveTri16<DTYPE_X, DTYPE_X> op;
         op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(32)) {
         KERNEL_TASK_TYPE(32, KERNEL_TYPE_MIX_AIC_1_2);
         GET_TILING_DATA(tilingData, tiling);
         SolveTri32<DTYPE_X, DTYPE_X> op;
         op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
         op.Process();
     }
 }
 #else
 extern "C" __global__ __aicore__ void solve_tri(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                 GM_ADDR x_out, GM_ADDR workspace, GM_ADDR tiling)
 {
     GET_TILING_DATA(tilingData, tiling);
     // tilingKey = chunkSize；兼容旧 key=1
     if (TILING_KEY_IS(1) || TILING_KEY_IS(16) || TILING_KEY_IS(32) ||
         TILING_KEY_IS(64) || TILING_KEY_IS(128)) {
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
 
         int64_t ms = tilingData.matrixSize;
         int64_t totalTiles = tilingData.totalTiles;
         int64_t tilesPerCore = tilingData.tilesPerCore;
         int64_t dtypeMode = tilingData.dtypeMode;  // 0=fp16, 1=bf16
 
         if ASCEND_IS_AIC {
             if (dtypeMode == 0) {
                 if (ms == 16) {
                     NsSolveTri::SolveTriCube<16, half> op;
                     op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                     op.Process();
                 } else if (ms == 32) {
                     NsSolveTri::SolveTriCube<32, half> op;
                     op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                     op.Process();
                 } else if (ms == 64) {
                     NsSolveTri::SolveTriCube<64, half> op;
                     op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                     op.Process();
                 } else if (ms == 128) {
                     NsSolveTri::SolveTriCube<128, half> op;
                     op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                     op.Process();
                 }
             } else {
                 if (ms == 16) {
                     NsSolveTri::SolveTriCube<16, bfloat16_t> op;
                     op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                     op.Process();
                 } else if (ms == 32) {
                     NsSolveTri::SolveTriCube<32, bfloat16_t> op;
                     op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                     op.Process();
                 } else if (ms == 64) {
                     NsSolveTri::SolveTriCube<64, bfloat16_t> op;
                     op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                     op.Process();
                 } else if (ms == 128) {
                     NsSolveTri::SolveTriCube<128, bfloat16_t> op;
                     op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                     op.Process();
                 }
             }
         }
 
         if ASCEND_IS_AIV {
             if (dtypeMode == 0) {
                 if (ms == 16) {
                     NsSolveTri::SolveTriVector<16, half> op;
                     op.Init(workspace, totalTiles, ms);
                     op.Process();
                 } else if (ms == 32) {
                     NsSolveTri::SolveTriVector<32, half> op;
                     op.Init(workspace, totalTiles, ms);
                     op.Process();
                 } else if (ms == 64) {
                     NsSolveTri::SolveTriVector<64, half> op;
                     op.Init(workspace, totalTiles, ms);
                     op.Process();
                 } else if (ms == 128) {
                     NsSolveTri::SolveTriVector<128, half> op;
                     op.Init(workspace, totalTiles, ms);
                     op.Process();
                 }
             } else {
                 if (ms == 16) {
                     NsSolveTri::SolveTriVector<16, bfloat16_t> op;
                     op.Init(workspace, totalTiles, ms);
                     op.Process();
                 } else if (ms == 32) {
                     NsSolveTri::SolveTriVector<32, bfloat16_t> op;
                     op.Init(workspace, totalTiles, ms);
                     op.Process();
                 } else if (ms == 64) {
                     NsSolveTri::SolveTriVector<64, bfloat16_t> op;
                     op.Init(workspace, totalTiles, ms);
                     op.Process();
                 } else if (ms == 128) {
                     NsSolveTri::SolveTriVector<128, bfloat16_t> op;
                     op.Init(workspace, totalTiles, ms);
                     op.Process();
                 }
             }
         }
     }
 }
 #endif
 