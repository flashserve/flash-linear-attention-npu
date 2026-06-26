/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu.cpp
 * \brief
 */

#if defined(ORIG_DTYPE_G) && defined(DT_BF16) && ORIG_DTYPE_G == DT_BF16
    #define G_BF16
#endif

#include "chunk_gated_delta_rule_bwd_dhu_struct.h"
#include "chunk_gated_delta_rule_bwd_dhu_vec.h"
#include "chunk_gated_delta_rule_bwd_dhu_cube.h"

#ifndef TORCH_MODE
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

namespace GDN {

template <typename DT, typename GT>
__aicore__ inline void ChunkGatedDeltaRuleBwdDhuKernelImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2, GM_ADDR userWS, const ChunkGatedDeltaRuleBwdDhuTilingData *tilingData)
{
    (void)dh0;
    if ASCEND_IS_AIC {
        GDRCube<DT, GT> cubeOp(k, w, d_o, dh, dv2, cu_seqlens, chunk_indices, userWS);
        cubeOp.Init(*tilingData);
        cubeOp.Process();
    }
    if ASCEND_IS_AIV {
        ChunkGDRBwdDhu::GDRVec<DT, GT> op;
        op.Init(q, k, w, d_o, dv, g, cu_seqlens, dv2, dh, userWS, *tilingData);
        op.Process();
    }
}

} // namespace GDN

#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void chunk_gated_delta_rule_bwd_dhu(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR h0, GM_ADDR dht,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)gk;
    (void)h0;
    (void)dht;
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(GDN::ChunkGatedDeltaRuleBwdDhuTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkGatedDeltaRuleBwdDhuTilingData, tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::ChunkGatedDeltaRuleBwdDhuKernelImpl<DTYPE_Q, DTYPE_Q>(
            q, k, w, d_o, dv, g, cu_seqlens, chunk_indices, dh, dh0, dv2, userWS, &tilingData);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::ChunkGatedDeltaRuleBwdDhuKernelImpl<DTYPE_Q, float>(
            q, k, w, d_o, dv, g, cu_seqlens, chunk_indices, dh, dh0, dv2, userWS, &tilingData);
    }
}
#endif
