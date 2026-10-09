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
 * \brief Device entry：模板参数直接给出 dtype / gate 模式，由 TPL tilingKey 选中模板实例（不使用 TILING_KEY_IS）。
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
    // A5（arch35）：1 AIC : 2 AIV，两个 AIV 各承包一个 head 的整条逆序链（参考 ChunkFwdH）。
    // A2/A3（arch22）：仍 1:1 —— 910B 的 0x2 是集合同步语义，三次接 1:2 都在 smoke 用例上挂死
    // （详见 design.md §17/§21 与 policy.h 注释）。
    // 三处必须一起切：这里的 KERNEL_TASK_TYPE、op_host 的 aivPerBlock、policy.h 的 flag 寻址口径。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#endif

    CP::ChunkDeltaHBwdPreprocessDispatch<D_T_Q, D_T_G, GATE_MODE>::Invoke(
        q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS, &tilingData);
}
#endif
