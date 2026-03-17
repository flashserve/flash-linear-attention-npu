/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_fwd_o_common.h"
#include "chunk_fwd_o_cube.h"
#include "chunk_fwd_o_vector.h"

extern "C" __global__ __aicore__ void chunk_fwd_o(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
                                                         GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_offsets,
                                                         GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    __gm__ ChunkFwdOTilingData *__restrict gdnFwdOTilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);
    if (gdnFwdOTilingData->dataType == 0) {

        if ASCEND_IS_AIC {
            ChunkFwdOCubeProcess<half> cubeProcess;
            cubeProcess.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
            cubeProcess.Process();
        }
        if ASCEND_IS_AIV {
            ChunkFwdOVectorProcess<half> vectorProcess;
            vectorProcess.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
            vectorProcess.Process();
        }

    } else {

        if ASCEND_IS_AIC {
            ChunkFwdOCubeProcess<bfloat16_t> cubeProcess;
            cubeProcess.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
            cubeProcess.Process();
        }
        if ASCEND_IS_AIV {
            ChunkFwdOVectorProcess<bfloat16_t> vectorProcess;
            vectorProcess.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
            vectorProcess.Process();
        }

    }
}
