/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_da.cpp
 * \brief
 */

#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/prepare_wy_repr_bwd_da_common.h"
#include "arch35/prepare_wy_repr_bwd_da_cube.h"
#include "arch35/prepare_wy_repr_bwd_da_vector.h"
#include "arch35/prepare_wy_repr_bwd_da_tiling_data_apt.h"
#else
#include "prepare_wy_repr_bwd_da_common.h"
#include "prepare_wy_repr_bwd_da_cube.h"
#include "prepare_wy_repr_bwd_da_vector.h"
#endif
#include "lib/matmul_intf.h"
// #include "kernel_basic_intf.h"

using namespace AscendC;
__global__ __aicore__ void prepare_wy_repr_bwd_da(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dw, GM_ADDR du,
                                                  GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                  GM_ADDR dA, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    REGISTER_TILING_DEFAULT(PrepareWyReprBwdDaTilingDataA5);
#endif
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIC{
            PrepareWyReprBwdDAProcess<DTYPE_K, DTYPE_BETA> prepareWyReprBwdDAProcess(
                k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices, dA, workspace);
            prepareWyReprBwdDAProcess.Init(tilingData);
            prepareWyReprBwdDAProcess.Process();
        }
        if ASCEND_IS_AIV{
            AscendC::TPipe tPipe;
            PrepareWyReprBwdDAVectorProcess<DTYPE_K, DTYPE_BETA> prepareWyReprBwdDAVectorProcess(
                k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices, dA, workspace);
            prepareWyReprBwdDAVectorProcess.Init(tilingData, &tPipe);
            prepareWyReprBwdDAVectorProcess.Process();
        }
    }
    return;
}
