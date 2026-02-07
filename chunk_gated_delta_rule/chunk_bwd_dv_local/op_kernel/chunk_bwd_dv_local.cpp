/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_bwd_dv_local.cpp
 * \brief
 */

#include "chunk_bwd_dv_local_cube.h"
#include "chunk_bwd_dv_local_vector.h"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void chunk_bwd_dv_local(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g,
                                                         GM_ADDR upper_tri_matrix, GM_ADDR g_gamma, GM_ADDR A,
                                                         GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
        
        if ASCEND_IS_AIC {
            GDN::ChunkBwdDvLocalCube<DTYPE_Q, DTYPE_G> chunkBwdDvLocalCube;
            chunkBwdDvLocalCube.Init(q, k, d_o, cu_seqlens, chunk_indices, d_v, userWS, &tilingData);
            chunkBwdDvLocalCube.Process();
        }
        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            GDN::ChunkBwdDvLocalVector<DTYPE_Q, DTYPE_G> chunkBwdDvLocalVector;
            chunkBwdDvLocalVector.Init(d_o, g, upper_tri_matrix, cu_seqlens, chunk_indices, d_v, userWS, &tilingData,
                                       &pipe);
            chunkBwdDvLocalVector.Process();
        }
    }

    // if (TILING_KEY_IS(1)) {
    // if ASCEND_IS_AIC {
    //     uint32_t coreIdx = AscendC::GetBlockIdx();
    //     AscendC::printf("[参数打印] ASCEND_IS_AIC  coreIdx = %d  \n", coreIdx);
    //     uint32_t blockNum = AscendC::GetBlockNum();
    //     AscendC::printf("[参数打印] ASCEND_IS_AIC  blockNum = %d  \n", blockNum);
    // }
    // if ASCEND_IS_AIV {
    //     uint32_t BlockIdx = AscendC::GetBlockIdx();
    //     AscendC::printf("[参数打印] ASCEND_IS_AIC  BlockIdx = %d  \n", BlockIdx);
    //     uint32_t subBlockNum = AscendC::GetSubBlockNum();
    //     AscendC::printf("[参数打印] ASCEND_IS_AIC  subBlockNum = %d  \n", subBlockNum);
    //     uint32_t coreIdx = BlockIdx / subBlockNum;
    //     AscendC::printf("[参数打印] ASCEND_IS_AIC  coreIdx = %d  \n", coreIdx);
    //     uint32_t blockNum = AscendC::GetBlockNum();
    //     AscendC::printf("[参数打印] ASCEND_IS_AIC  blockNum = %d  \n", blockNum);
    // }}

    // GET_TILING_DATA(tilingData, tiling);
    // if (TILING_KEY_IS(1)) {
    //     if ASCEND_IS_AIC {
    //         GDN::ChunkBwdDvLocalBase<DTYPE_Q, DTYPE_G> op;
    //         op.Init(q, k, d_o, g, upper_tri_matrix, cu_seqlens, chunk_indices, d_v, userWS, &tilingData);
    //         op.Process();
    //     }
    //     if ASCEND_IS_AIV {
    //         AscendC::TPipe pipe;
    //         GDN::ChunkBwdDvLocalBase<DTYPE_Q, DTYPE_G> op;
    //         op.Init(q, k, d_o, g, upper_tri_matrix, cu_seqlens, chunk_indices, d_v, userWS, &tilingData, &pipe);
    //         op.Process();
    //     }

    // }
}
