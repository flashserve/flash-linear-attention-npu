/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_bwd_dv_local.cpp
 * \brief
 */

#include "chunk_bwd_dv_local_base.h"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void chunk_bwd_dv_local(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g,
                                                         GM_ADDR upper_tri_matrix, GM_ADDR g_gamma, GM_ADDR A,
                                                         GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    AscendC::TPipe pipe;

    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        GDN::ChunkBwdDvLocalBase<DTYPE_Q, DTYPE_G> op;
        op.Init(q, k, d_o, g, upper_tri_matrix, cu_seqlens, chunk_indices, d_v, userWS, &tilingData, &pipe);
        op.Process();
    }
}
