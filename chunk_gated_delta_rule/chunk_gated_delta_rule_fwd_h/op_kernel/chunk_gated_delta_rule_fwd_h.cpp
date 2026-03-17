/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_gated_delta_rule_fwd_h_common.h"
#include "chunk_gated_delta_rule_fwd_h_cube.h"
#include "chunk_gated_delta_rule_fwd_h_vector.h"

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g,
                                                         GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                         GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);
    if (gdnFwdHTilingData->dataType == 0) {

        if ASCEND_IS_AIC {
            ChunkGatedDeltaRuleFwdHCubeProcess<half> cubeProcess;
            cubeProcess.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            cubeProcess.Process();
        }
        if ASCEND_IS_AIV {
            ChunkGatedDeltaRuleFwdHVectorProcess<half> vectorProcess;
            vectorProcess.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            vectorProcess.Process();
        }

    } else {

        if ASCEND_IS_AIC {
            ChunkGatedDeltaRuleFwdHCubeProcess<bfloat16_t> cubeProcess;
            cubeProcess.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            cubeProcess.Process();
        }
        if ASCEND_IS_AIV {
            ChunkGatedDeltaRuleFwdHVectorProcess<bfloat16_t> vectorProcess;
            vectorProcess.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            vectorProcess.Process();
        }

    }
}
