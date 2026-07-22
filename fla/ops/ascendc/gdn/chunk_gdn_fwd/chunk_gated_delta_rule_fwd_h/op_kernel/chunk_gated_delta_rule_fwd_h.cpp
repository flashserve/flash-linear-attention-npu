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

namespace GDN {

template <typename InputT, typename GT, typename StateT, typename WorkspaceT, typename TileShapes,
          bool kGated, bool scalarGated, bool useExp2>
__aicore__ inline void ChunkGatedDeltaRuleFwdHKernelImpl(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                                         GM_ADDR inital_state, GM_ADDR cu_seqlens,
                                                         GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                                         GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user)
{
    using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        InputT, GT, StateT, WorkspaceT, TileShapes, kGated, scalarGated, useExp2>;
    GDNFwdHKernel gdnFwdH;
    gdnFwdH.Init(k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
    gdnFwdH.Process();
}

template <typename DataT, typename GateT, typename StateT, typename TileShapes, bool useExp2>
__aicore__ inline void ChunkGatedDeltaRuleFwdHLaunchTyped(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                                         GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user,
                                                         bool useGk, bool useG)
{
    using WorkspaceT = float;
    if (useGk) {
        if (useG) {
            ChunkGatedDeltaRuleFwdHKernelImpl<DataT, GateT, StateT, WorkspaceT, TileShapes, true, true, useExp2>(
                k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
        } else {
            ChunkGatedDeltaRuleFwdHKernelImpl<DataT, GateT, StateT, WorkspaceT, TileShapes, true, false, useExp2>(
                k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
        }
    } else {
        ChunkGatedDeltaRuleFwdHKernelImpl<DataT, GateT, StateT, WorkspaceT, TileShapes, false, true, useExp2>(
            k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
    }
}

template <typename TileShapes, bool useExp2>
__aicore__ inline void ChunkGatedDeltaRuleFwdHDispatchExp(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                                       GM_ADDR inital_state, GM_ADDR cu_seqlens,
                                                       GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                                       GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user)
{
    __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData =
        reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);
    bool useGk = gdnFwdHTilingData->useGk;
    bool useG = gdnFwdHTilingData->useG;
    ChunkGatedDeltaRuleFwdHLaunchTyped<DTYPE_K, DTYPE_GK, DTYPE_INITAL_STATE, TileShapes, useExp2>(
        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new,
        final_state, tiling, user, useGk, useG);
}

template <typename TileShapes>
__aicore__ inline void ChunkGatedDeltaRuleFwdHDispatch(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                                       GM_ADDR inital_state, GM_ADDR cu_seqlens,
                                                       GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                                       GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user)
{
    __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict tilingData =
        reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);
    if (tilingData->useExp2) {
        ChunkGatedDeltaRuleFwdHDispatchExp<TileShapes, true>(
            k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new,
            final_state, tiling, user);
    } else {
        ChunkGatedDeltaRuleFwdHDispatchExp<TileShapes, false>(
            k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new,
            final_state, tiling, user);
    }
}

} // namespace GDN

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g,
                                                         GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                         GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::ChunkGatedDeltaRuleFwdHDispatch<Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::ChunkGatedDeltaRuleFwdHDispatch<Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
    }
}
