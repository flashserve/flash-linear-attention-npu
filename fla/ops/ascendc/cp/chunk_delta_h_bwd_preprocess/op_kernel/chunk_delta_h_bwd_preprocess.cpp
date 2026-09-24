/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess.cpp
 * \brief Device entry for chunk_delta_h_bwd_preprocess.
 */

#include "chunk_delta_h_bwd_preprocess_struct.h"
#include "chunk_delta_h_bwd_preprocess_vec.h"
#include "chunk_delta_h_bwd_preprocess_cube.h"

#ifndef TORCH_MODE
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

namespace CP {

template <typename DT, typename GT>
__aicore__ inline void ChunkDeltaHBwdPreprocessImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR cu_seqlens,
    GM_ADDR dhm, GM_ADDR userWS, const ChunkDeltaHBwdPreprocessTilingData *tilingData)
{
    if ASCEND_IS_AIC {
        ChunkDeltaHBwdPreCube<DT> cubeOp;
        cubeOp.Init(q, k, w, d_o, dv, cu_seqlens, dhm, userWS, *tilingData);
        cubeOp.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkDeltaHBwdPreVec<DT, GT> vecOp;
        vecOp.Init(q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS, *tilingData);
        vecOp.InitBuffer(&pipe);
        vecOp.Process();
    }
}

} // namespace CP

#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void chunk_delta_h_bwd_preprocess(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR cu_seqlens,
    GM_ADDR dhm, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(CP::ChunkDeltaHBwdPreprocessTilingData);
    GET_TILING_DATA_WITH_STRUCT(CP::ChunkDeltaHBwdPreprocessTilingData, tilingData, tiling);

    // tilingKey：1 无门控 / 2 USE_G(g 与 q 同 dtype) / 3 USE_G(g FP32) / 4 USE_GK
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
        CP::ChunkDeltaHBwdPreprocessImpl<DTYPE_Q, DTYPE_Q>(q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS,
                                                          &tilingData);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_1);
        CP::ChunkDeltaHBwdPreprocessImpl<DTYPE_Q, DTYPE_G>(q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS,
                                                          &tilingData);
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_1);
        CP::ChunkDeltaHBwdPreprocessImpl<DTYPE_Q, float>(q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS,
                                                        &tilingData);
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_1);
        CP::ChunkDeltaHBwdPreprocessImpl<DTYPE_Q, DTYPE_GK>(q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS,
                                                           &tilingData);
    }
}
#endif
