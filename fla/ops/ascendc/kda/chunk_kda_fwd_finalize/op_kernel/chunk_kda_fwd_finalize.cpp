/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_kda_fwd_finalize_kernel.hpp"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void chunk_kda_fwd_finalize(
    GM_ADDR qg_scaled, GM_ADDR aqk, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR attn_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(ChunkKdaFwdFinalizeTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        TPipe pipe;
        KdaFinalize::RunChunkKdaOutput<DTYPE_QG_SCALED, DTYPE_QG_SCALED, DTYPE_QG_SCALED>(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, cu_seqlens, chunk_indices,
            qg_scaled, aqk, v_new, h, attn_out, userWorkspace, tilingData, pipe);
    }
}
