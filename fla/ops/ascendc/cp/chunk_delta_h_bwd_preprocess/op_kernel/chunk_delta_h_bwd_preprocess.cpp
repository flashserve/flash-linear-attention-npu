/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess.cpp
 * \brief Device entry：模板参数直接给出 dtype / gate 模式，由 tilingKey 选择实例（不使用 TILING_KEY_IS）。
 */

#include "chunk_delta_h_bwd_preprocess_kernel.h"

#ifndef TORCH_MODE
#include "lib/matmul_intf.h"

template <int D_T_Q, int D_T_G, uint32_t GATE_MODE>
__global__ __aicore__ void chunk_delta_h_bwd_preprocess(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR cu_seqlens,
    GM_ADDR dhm, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(CP::ChunkDeltaHBwdPreprocessTilingData);
    GET_TILING_DATA_WITH_STRUCT(CP::ChunkDeltaHBwdPreprocessTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    CP::ChunkDeltaHBwdPreprocessDispatch<D_T_Q, D_T_G, GATE_MODE>::Invoke(
        q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS, &tilingData);
}
#endif
