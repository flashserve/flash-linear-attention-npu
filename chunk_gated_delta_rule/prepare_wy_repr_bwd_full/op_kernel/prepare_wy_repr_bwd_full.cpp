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
 * \file prepare_wy_repr_bwd_full.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "prepare_wy_repr_bwd_full_common.h"
#include "prepare_wy_repr_bwd_full_cube.h"
#include "prepare_wy_repr_bwd_full_vector.h"
#include "lib/matmul_intf.h"
// #include "kernel_basic_intf.h"
using namespace AscendC;
__global__ __aicore__ void prepare_wy_repr_bwd_full(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dA, GM_ADDR dw, GM_ADDR du, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                    GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tPipe;
    AscendC::AscendCUtils::SetOverflow(1);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIC{
            PrepareWyReprBwdFullProcess<DTYPE_K,DTYPE_BETA> prepareWyReprBwdFullProcess(k, v, beta, A, dA, dw, du, g, dk, dv, dbeta, dg, workspace);
            prepareWyReprBwdFullProcess.Init(tilingData);
            prepareWyReprBwdFullProcess.Process();
        }
        if ASCEND_IS_AIV{
            PrepareWyReprBwdFullVectorProcess<DTYPE_K,DTYPE_BETA> prepareWyReprBwdFullVectorProcess(k, v, beta, A, dA, dw, du, g, dk, dv, dbeta, dg, workspace);
            prepareWyReprBwdFullVectorProcess.Init(tilingData, &tPipe);
            prepareWyReprBwdFullVectorProcess.Process();
        }
    }
    return;
}
