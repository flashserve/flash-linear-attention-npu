/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "solve_tril.h"

using namespace AscendC;

// SolveTril kernel 入口（MCH 上片后的 6 段形参）：
//   x / cu_seqlens / chunk_indices / x_out / workspace / tiling
//
// 关于结构（参考源仓 review）：
//   - 不使用 extern "C"：会抑制 C++ name mangling，与 KERNEL_TASK_TYPE 生成的
//     tiling-key 变体符号冲突。
//   - GET_TILING_DATA 必须在 TILING_KEY_IS(1) 块内、KERNEL_TASK_TYPE 之后。
//   - KERNEL_TYPE_MIX_AIC_1_2：AIC(cube) 与 AIV(vector) 协同；AIC/AIV 分支在
//     SolveTril::Process() 内部用 ASCEND_IS_AIC / ASCEND_IS_AIV 完成。
__global__ __aicore__ void solve_tril(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                      GM_ADDR x_out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA(tilingData, tiling);

        SolveTril<DTYPE_X, DTYPE_X> op;
        op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
        op.Process();
    }
}
