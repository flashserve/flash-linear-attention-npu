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
 * \file chunk_gated_delta_rule_bwd_dhu.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_bwd_dhu_vec.h"
#include "chunk_gated_delta_rule_bwd_dhu_cube.h"

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#if defined(ORIG_DTYPE_Q) && defined(DT_FLOAT16) && ORIG_DTYPE_Q == DT_FLOAT16
    #define INPUT_DTYPE half
#elif defined(ORIG_DTYPE_Q) && defined(DT_BF16) && ORIG_DTYPE_Q == DT_BF16
    #define INPUT_DTYPE bfloat16_t
#endif
using namespace AscendC;
extern "C" __global__ __aicore__ void chunk_gated_delta_rule_bwd_dhu(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR h0, GM_ADDR dht, 
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    
    if (TILING_KEY_IS(1)) {
        if ASCEND_IS_AIC {
            GDRCube<INPUT_DTYPE> cubeOp(k, w, d_o, dh, dv2, cu_seqlens, chunk_indices, workspace);
            cubeOp.Init(tilingData);
            cubeOp.Process();
        }
        if ASCEND_IS_AIV {
            ChunkGDRBwdDhu::GDRVec<INPUT_DTYPE> op;
            op.Init(q, k, w, d_o, dv, g, cu_seqlens, chunk_indices, tilingData);
            op.Process();
        }

    }
}
