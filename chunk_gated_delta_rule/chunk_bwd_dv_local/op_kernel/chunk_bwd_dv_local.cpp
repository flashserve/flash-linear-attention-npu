/**
 * Copyright (c) 2025 Tianjin University, Ltd.
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

#include "chunk_bwd_dv_local_cube.h"
#include "chunk_bwd_dv_local_vector.h"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void chunk_bwd_dv_local(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g, GM_ADDR g_gamma,
                                                         GM_ADDR A, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                         GM_ADDR d_v, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(0)) {
        GDN::FixedLengthStrategy fixedStrategy{tilingData.chunkSize, tilingData.t, tilingData.chunkNumForT};
        if ASCEND_IS_AIC {
            GDN::ChunkBwdDvLocalCube<DTYPE_Q, DTYPE_G, GDN::FixedLengthStrategy> chunkBwdDvLocalCube(fixedStrategy);
            chunkBwdDvLocalCube.Init(q, k, d_o, cu_seqlens, chunk_indices, d_v, userWS, &tilingData);
            chunkBwdDvLocalCube.Process();
        }
        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            GDN::ChunkBwdDvLocalVector<DTYPE_Q, DTYPE_G, GDN::FixedLengthStrategy> chunkBwdDvLocalVector(fixedStrategy);
            chunkBwdDvLocalVector.Init(d_o, g, cu_seqlens, chunk_indices, d_v, userWS, &tilingData, &pipe);
            chunkBwdDvLocalVector.Process();
        }
    }
    if (TILING_KEY_IS(1)) {
        GDN::VariableLengthStrategy variableStrategy{tilingData.chunkSize, tilingData.t, tilingData.chunkNumForT,
                                                     cu_seqlens, chunk_indices};
        if ASCEND_IS_AIC {
            GDN::ChunkBwdDvLocalCube<DTYPE_Q, DTYPE_G, GDN::VariableLengthStrategy> chunkBwdDvLocalCube(
                variableStrategy);
            chunkBwdDvLocalCube.Init(q, k, d_o, cu_seqlens, chunk_indices, d_v, userWS, &tilingData);
            chunkBwdDvLocalCube.Process();
        }
        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            GDN::ChunkBwdDvLocalVector<DTYPE_Q, DTYPE_G, GDN::VariableLengthStrategy> chunkBwdDvLocalVector(
                variableStrategy);
            chunkBwdDvLocalVector.Init(d_o, g, cu_seqlens, chunk_indices, d_v, userWS, &tilingData, &pipe);
            chunkBwdDvLocalVector.Process();
        }
    }
}
