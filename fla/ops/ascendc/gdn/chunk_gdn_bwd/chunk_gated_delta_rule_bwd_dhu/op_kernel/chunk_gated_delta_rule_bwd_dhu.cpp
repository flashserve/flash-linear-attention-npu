/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu.cpp
 * \brief Kernel entry for chunk_gated_delta_rule_bwd_dhu.
 */

#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_gated_delta_rule_bwd_dhu_struct.h"
#include "arch35/chunk_gated_delta_rule_bwd_dhu_cube.h"
#include "arch35/chunk_gated_delta_rule_bwd_dhu_vector.h"
#else
#include "chunk_gated_delta_rule_bwd_dhu_struct.h"
#include "chunk_gated_delta_rule_bwd_dhu_cube.h"
#include "chunk_gated_delta_rule_bwd_dhu_vector.h"
#endif

namespace GDN {

template <typename DT, typename GT, int V_DIM, int USE_GK>
__aicore__ inline void ChunkGatedDeltaRuleBwdDhuKernelImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR gate, GM_ADDR dht,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2,
    GM_ADDR workspace, const ChunkGatedDeltaRuleBwdDhuTilingData *tilingData)
{
    if ASCEND_IS_AIC {
        ChunkGatedDeltaRuleBwdDhuCube<DT, V_DIM> cube;
        cube.Init(k, w, d_o, dh, dv2, cu_seqlens, chunk_indices, workspace, tilingData);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkGatedDeltaRuleBwdDhuVector<DT, GT, USE_GK> vec;
        vec.Init(q, gate, dv, dht, cu_seqlens, chunk_indices, dh, dh0, dv2, workspace, tilingData, &pipe);
        vec.Process();
    }
}

template <int D_T>
struct ChunkGatedDeltaRuleBwdDhuDTypeTraits;

template <>
struct ChunkGatedDeltaRuleBwdDhuDTypeTraits<TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct ChunkGatedDeltaRuleBwdDhuDTypeTraits<TPL_FP16> {
    using type = half;
};

template <>
struct ChunkGatedDeltaRuleBwdDhuDTypeTraits<TPL_FP32> {
    using type = float;
};

} // namespace GDN

#ifndef TORCH_MODE
template <int D_T_Q, int D_T_G, int V, int USE_GK>
__global__ __aicore__ void chunk_gated_delta_rule_bwd_dhu(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g,
    GM_ADDR gk, GM_ADDR h0, GM_ADDR dht, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)h0;

    AscendC::AscendCUtils::SetOverflow(1);

    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(GDN::ChunkGatedDeltaRuleBwdDhuTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkGatedDeltaRuleBwdDhuTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    using QType = typename GDN::ChunkGatedDeltaRuleBwdDhuDTypeTraits<D_T_Q>::type;
    using GType = typename GDN::ChunkGatedDeltaRuleBwdDhuDTypeTraits<D_T_G>::type;
    GM_ADDR gate = (USE_GK == 0) ? g : gk;
    GDN::ChunkGatedDeltaRuleBwdDhuKernelImpl<QType, GType, V, USE_GK>(
        q, k, w, d_o, dv, gate, dht, cu_seqlens, chunk_indices, dh, dh0, dv2, userWS, &tilingData);
}
#endif
