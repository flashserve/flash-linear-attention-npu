/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg.cpp
 */

#include "chunk_bwd_dqkwg_struct.h"
#include "chunk_bwd_dqkwg_common.h"
#include "chunk_bwd_dqkwg_cube.h"
#include "chunk_bwd_dqkwg_vector.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

namespace GDN {

template <typename QKVT, typename GT, int V, typename Strategy>
__aicore__ inline void ChunkBwdDqkwgKernelImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h,
    GM_ADDR do_, GM_ADDR dh, GM_ADDR dv,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR w, GM_ADDR g_gamma,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg,
    GM_ADDR workspace, const ChunkBwdDqkwgTilingData *tilingData,
    Strategy strategy)
{
    if ASCEND_IS_AIC {
        ChunkBwdDqkwgCubeProcess<QKVT, GT> cubeProcess(
            q, k, v, g, h,
            do_, dh, dv, cu_seqlens, chunk_indices,
            dq, dk, dw, dg,
            workspace
        );
        cubeProcess.Init(*tilingData);
        cubeProcess.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe tPipe;
        ChunkBwdDqkwgVectorProcess<QKVT, GT> vectorProcess(
            q, k, v, g, h,
            do_, dh, dv, cu_seqlens, chunk_indices, nullptr,
            dq, dk, dw, dg,
            workspace
        );
        vectorProcess.Init(*tilingData, &tPipe);
        vectorProcess.Process();
    }
}

template <int D_T>
struct DTypeTraitsDqkwg;

template <>
struct DTypeTraitsDqkwg<CHUNK_BWD_DQKWG_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct DTypeTraitsDqkwg<CHUNK_BWD_DQKWG_TPL_FP16> {
    using type = half;
};

template <>
struct DTypeTraitsDqkwg<CHUNK_BWD_DQKWG_TPL_FP32> {
    using type = float;
};

template <uint64_t strategy, int D_T_Q, int D_T_G, int V>
struct ChunkBwdDqkwgDispatch;

template <int D_T_Q, int D_T_G, int V>
struct ChunkBwdDqkwgDispatch<CHUNK_BWD_DQKWG_STRATEGY_FIX_LEN, D_T_Q, D_T_G, V> {
    __aicore__ inline static void Invoke(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h,
        GM_ADDR do_, GM_ADDR dh, GM_ADDR dv,
        GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
        GM_ADDR w, GM_ADDR g_gamma,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg,
        GM_ADDR userWS, const ChunkBwdDqkwgTilingData *tilingData)
    {
        FixedLengthStrategy fixedStrategy{tilingData->BT, tilingData->T, tilingData->numChunks, tilingData->HV};
        ChunkBwdDqkwgKernelImpl<typename DTypeTraitsDqkwg<D_T_Q>::type, typename DTypeTraitsDqkwg<D_T_G>::type, V>(
            q, k, v, g, h, do_, dh, dv, cu_seqlens, chunk_indices, w, g_gamma,
            dq, dk, dw, dg, userWS, tilingData, fixedStrategy);
    }
};

template <int D_T_Q, int D_T_G, int V>
struct ChunkBwdDqkwgDispatch<CHUNK_BWD_DQKWG_STRATEGY_VAR_LEN, D_T_Q, D_T_G, V> {
    __aicore__ inline static void Invoke(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h,
        GM_ADDR do_, GM_ADDR dh, GM_ADDR dv,
        GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
        GM_ADDR w, GM_ADDR g_gamma,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg,
        GM_ADDR userWS, const ChunkBwdDqkwgTilingData *tilingData)
    {
        VariableLengthStrategy variableStrategy{tilingData->BT, tilingData->T, tilingData->numChunks,
                                                 cu_seqlens, chunk_indices};
        ChunkBwdDqkwgKernelImpl<typename DTypeTraitsDqkwg<D_T_Q>::type, typename DTypeTraitsDqkwg<D_T_G>::type, V>(
            q, k, v, g, h, do_, dh, dv, cu_seqlens, chunk_indices, w, g_gamma,
            dq, dk, dw, dg, userWS, tilingData, variableStrategy);
    }
};

} // namespace GDN

#ifndef TORCH_MODE
template <uint64_t strategy, int D_T_Q, int D_T_G, int V>
__global__ __aicore__ void chunk_bwd_dqkwg(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h,
    GM_ADDR do_, GM_ADDR dh, GM_ADDR dv,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR w, GM_ADDR g_gamma,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(GDN::ChunkBwdDqkwgTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkBwdDqkwgTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    AscendCUtils::SetOverflow(1);

    GDN::ChunkBwdDqkwgDispatch<strategy, D_T_Q, D_T_G, V>::Invoke(
        q, k, v, g, h, do_, dh, dv, cu_seqlens, chunk_indices, w, g_gamma,
        dq, dk, dw, dg, userWS, &tilingData);
}
#endif
