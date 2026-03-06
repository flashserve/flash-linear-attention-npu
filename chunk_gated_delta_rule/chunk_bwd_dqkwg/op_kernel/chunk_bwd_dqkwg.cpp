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
 * \file chunk_bwd_dqkwg.cpp
 */

#include "kernel_operator.h"
#include "chunk_bwd_dqkwg_common.h"
#include "chunk_bwd_dqkwg_cube.h"
#include "chunk_bwd_dqkwg_vector.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

#ifndef DTYPE_Q
#define DTYPE_Q half
#endif

#ifndef DTYPE_G
#define DTYPE_G float
#endif


__global__ __aicore__ void chunk_bwd_dqkwg(
    GM_ADDR q,              // [B, H, T, K]
    GM_ADDR k,              // [B, H, T, K]
    GM_ADDR v,              // [B, H, T, V]
    GM_ADDR g,              // [B, H, T]
    GM_ADDR h,              // [B, num_chunks, H, K, V]
    GM_ADDR do_,            // [B, H, T, V]
    GM_ADDR dh,             // [B, num_chunks, H, K, V]
    GM_ADDR dv,             // [B, H, T, V]
    GM_ADDR cu_seqlens,     // [N+1] (optional)
    GM_ADDR chunk_indices,  // [num_chunks, 2] (optional)
    GM_ADDR dq,             // [B, H, T, K] - output
    GM_ADDR dk,             // [B, H, T, K] - output
    GM_ADDR dw,             // [B, H, T, K] - output
    GM_ADDR dg,             // [B, H, T] - output (fp32)
    GM_ADDR workspace,      // workspace buffer
    GM_ADDR tiling          // . data
)
{

    // 设置溢出处理
    AscendCUtils::SetOverflow(1);
    
    // 根据 TilingKey 选择执行路径
    if (TILING_KEY_IS(1)) {

        // 使用 C-V 融合模式
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA(tilingData, tiling);
        // AIC (Cube) 端执行

        if ASCEND_IS_AIC {
            ChunkBwdDqkwgCubeProcess<DTYPE_Q, DTYPE_G> cubeProcess(
                q, k, v, g, h,
                do_, dh, dv, cu_seqlens, chunk_indices,
                dq, dk, dw, dg,
                workspace
            );
            cubeProcess.Init(tilingData);
            cubeProcess.Process();
        }
        
        // AIV (Vector) 端执行
        if ASCEND_IS_AIV {
            TPipe tPipe; // 创建 TPipe 用于 Vector 端流水
            ChunkBwdDqkwgVectorProcess<DTYPE_Q, DTYPE_G> vectorProcess(
                q, k, v, g, h,
                do_, dh, dv, cu_seqlens, chunk_indices, nullptr,        //mask = nullptr
                dq, dk, dw, dg,
                workspace
            );
            vectorProcess.Init(tilingData, &tPipe);
            vectorProcess.Process();
        }

    }

    return;
}