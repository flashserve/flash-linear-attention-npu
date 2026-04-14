/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "recompute_wu_fwd_common.h"
#include "recompute_wu_fwd_cube.h"
#include "recompute_wu_fwd_vector.h"


using namespace AscendC;
__global__ __aicore__ void recompute_wu_fwd(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR gk,
                                             GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w,
                                              GM_ADDR u, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIC {
            RecomputeWUFwdProcess<DTYPE_K, DTYPE_BETA>recomputeWUFwdProcess(
                k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, workspace);
           recomputeWUFwdProcess.Init(tilingData);
           recomputeWUFwdProcess.Process();
        }
        if ASCEND_IS_AIV {
            AscendC::TPipe tPipe;
            RecomputeWUFwdVectorProcess<DTYPE_K, DTYPE_BETA>recomputeWUFwdVectorProcess(
                k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, workspace);
           recomputeWUFwdVectorProcess.Init(tilingData, &tPipe);
           recomputeWUFwdVectorProcess.Process();
        }
    }
    return;
}
