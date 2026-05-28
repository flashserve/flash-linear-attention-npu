/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dv_local.cpp
 * \brief
 */

#include "chunk_bwd_dv_local_struct.h"
#include "chunk_bwd_dv_local_cube.h"
#include "chunk_bwd_dv_local_vector.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

namespace GDN {

template <typename QKVT, typename GT, int V, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalKernelImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g, GM_ADDR g_gamma,
    GM_ADDR A, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR d_v, GM_ADDR workspace, const ChunkBwdDvLocalTilingData *tilingData,
    Strategy strategy)
{
    if ASCEND_IS_AIC {
        ChunkBwdDvLocalCube<QKVT, GT, Strategy, V> cube(strategy);
        cube.Init(q, k, d_o, cu_seqlens, chunk_indices, d_v, workspace, tilingData);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkBwdDvLocalVector<QKVT, GT, Strategy> vec(strategy);
        vec.Init(d_o, g, cu_seqlens, chunk_indices, d_v, workspace, tilingData, &pipe);
        vec.Process();
    }
}

template <int D_T>
struct DTypeTraits;

template <>
struct DTypeTraits<CHUNK_BWD_DV_LOCAL_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct DTypeTraits<CHUNK_BWD_DV_LOCAL_TPL_FP16> {
    using type = half;
};

template <>
struct DTypeTraits<CHUNK_BWD_DV_LOCAL_TPL_FP32> {
    using type = float;
};

template <uint64_t strategy, int D_T_Q, int D_T_G, int V>
struct ChunkBwdDvLocalDispatch;

template <int D_T_Q, int D_T_G, int V>
struct ChunkBwdDvLocalDispatch<CHUNK_BWD_DV_LOCAL_STRATEGY_FIX_LEN, D_T_Q, D_T_G, V> {
    __aicore__ inline static void Invoke(
        GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g, GM_ADDR g_gamma,
        GM_ADDR A, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
        GM_ADDR d_v, GM_ADDR userWS, const ChunkBwdDvLocalTilingData *tilingData)
    {
        FixedLengthStrategy fixedStrategy{tilingData->chunkSize, tilingData->t, tilingData->chunkNumForT};
        ChunkBwdDvLocalKernelImpl<typename DTypeTraits<D_T_Q>::type, typename DTypeTraits<D_T_G>::type, V>(
            q, k, d_o, g, g_gamma, A, cu_seqlens, chunk_indices, d_v, userWS, tilingData, fixedStrategy);
    }
};

template <int D_T_Q, int D_T_G, int V>
struct ChunkBwdDvLocalDispatch<CHUNK_BWD_DV_LOCAL_STRATEGY_VAR_LEN, D_T_Q, D_T_G, V> {
    __aicore__ inline static void Invoke(
        GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g, GM_ADDR g_gamma,
        GM_ADDR A, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
        GM_ADDR d_v, GM_ADDR userWS, const ChunkBwdDvLocalTilingData *tilingData)
    {
        VariableLengthStrategy variableStrategy{tilingData->chunkSize, tilingData->t, tilingData->chunkNumForT,
                                                 cu_seqlens, chunk_indices};
        ChunkBwdDvLocalKernelImpl<typename DTypeTraits<D_T_Q>::type, typename DTypeTraits<D_T_G>::type, V>(
            q, k, d_o, g, g_gamma, A, cu_seqlens, chunk_indices, d_v, userWS, tilingData, variableStrategy);
    }
};

} // namespace GDN

#ifndef TORCH_MODE
template <uint64_t strategy, int D_T_Q, int D_T_G, int V>
__global__ __aicore__ void chunk_bwd_dv_local(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g, GM_ADDR g_gamma,
                                               GM_ADDR A, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                               GM_ADDR d_v, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(GDN::ChunkBwdDvLocalTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkBwdDvLocalTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GDN::ChunkBwdDvLocalDispatch<strategy, D_T_Q, D_T_G, V>::Invoke(
        q, k, d_o, g, g_gamma, A, cu_seqlens, chunk_indices, d_v, userWS, &tilingData);
}
#endif
