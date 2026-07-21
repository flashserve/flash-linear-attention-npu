/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "kda/chunk_kda_fwd_kernel.hpp"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void chunk_kda_fwd_post_wu(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w_seed, GM_ADDR akk, GM_ADDR u_seed,
    GM_ADDR w, GM_ADDR u, GM_ADDR kg, GM_ADDR v_new, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(ChunkKdaFwdPostWuTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        TPipe pipe;
        RunChunkKdaPostWu<DTYPE_Q, DTYPE_GK, DTYPE_BETA>(
            q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices,
            w_seed, akk, u_seed, w, u, kg, v_new, userWorkspace, tilingData, pipe);
    }
}
