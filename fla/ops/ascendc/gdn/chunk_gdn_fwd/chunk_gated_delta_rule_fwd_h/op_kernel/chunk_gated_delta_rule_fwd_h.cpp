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

#include "chunk_gated_delta_rule_fwd_h_tiling_key.h"
#include "lib/matmul_intf.h"

using namespace Catlass;

namespace GDN {

template <uint32_t VTile>
struct FwdHTileSelector;

template <>
struct FwdHTileSelector<GDN_FWD_H_V_TILE_128> {
    using type = Catlass::Gemm::Kernel::GDNFwdHTileShapes128;
};

template <uint32_t GateMode>
struct FwdHGateTypeSelector;

template <>
struct FwdHGateTypeSelector<GDN_FWD_H_GATE_G> {
    using type = DTYPE_G;
};

template <>
struct FwdHGateTypeSelector<GDN_FWD_H_GATE_GK> {
    using type = DTYPE_GK;
};

} // namespace GDN

template <uint32_t V_TILE, uint32_t GATE_MODE, uint32_t EXP_MODE>
__global__ __aicore__ void chunk_gated_delta_rule_fwd_h(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    using TileShapes = typename GDN::FwdHTileSelector<V_TILE>::type;
    using GateT = typename GDN::FwdHGateTypeSelector<GATE_MODE>::type;
    using WorkspaceT = float;
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        DTYPE_K, GateT, DTYPE_FINAL_STATE, WorkspaceT, TileShapes, GATE_MODE, EXP_MODE>;
    Kernel op;
    op.Init(k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices,
            h, v_new, final_state, tiling, user);
    op.Process();
}
