/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h.cpp
 * \brief
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif

#include "lib/matmul_intf.h"

using namespace Catlass;

namespace {
template <typename InputType, typename GateType, typename StateType, typename OutputType, typename WorkspaceType>
__aicore__ inline void RunGdnFwdH(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state,
                                  GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                  GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user)
{
    using GDNFwdHKernel =
        Catlass::Gemm::Kernel::GDNFwdHKernel<InputType, GateType, StateType, WorkspaceType, OutputType>;
    GDNFwdHKernel gdnFwdH;
    gdnFwdH.Init(k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
    gdnFwdH.Process();
}
} // namespace

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g,
                                                         GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                         GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);
    using workspaceType = float;
    // dtype: 0 - fp16, 1 - bf16, 2 - fp32
    if (gdnFwdHTilingData->dataType == 2) {
        RunGdnFwdH<float, float, float, float, workspaceType>(
            k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
    } else if (gdnFwdHTilingData->dataType == 1) {
        if (gdnFwdHTilingData->stateDataType == 2) {
            if (gdnFwdHTilingData->gDataType == 2) {
                RunGdnFwdH<bfloat16_t, float, float, bfloat16_t, workspaceType>(
                    k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            } else {
                RunGdnFwdH<bfloat16_t, bfloat16_t, float, bfloat16_t, workspaceType>(
                    k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            }
        } else {
            if (gdnFwdHTilingData->gDataType == 2) {
                RunGdnFwdH<bfloat16_t, float, bfloat16_t, bfloat16_t, workspaceType>(
                    k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            } else {
                RunGdnFwdH<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, workspaceType>(
                    k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            }
        }
    } else {
        if (gdnFwdHTilingData->stateDataType == 2) {
            if (gdnFwdHTilingData->gDataType == 2) {
                RunGdnFwdH<half, float, float, half, workspaceType>(
                    k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            } else {
                RunGdnFwdH<half, half, float, half, workspaceType>(
                    k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            }
        } else {
            if (gdnFwdHTilingData->gDataType == 2) {
                RunGdnFwdH<half, float, half, half, workspaceType>(
                    k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            } else {
                RunGdnFwdH<half, half, half, half, workspaceType>(
                    k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
            }
        }
    }
}
