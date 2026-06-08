/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_bwd.cpp
 * \brief Causal Conv1D backward kernel entry point (FP32/FP16/BF16)
 */

#include "causal_conv1d_bwd_input_layout.h"

extern "C" __global__ __aicore__ void causal_conv1d_bwd(
    GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR dy,
    GM_ADDR initial_state, GM_ADDR dht, GM_ADDR query_start_loc,
    GM_ADDR dx, GM_ADDR dw, GM_ADDR db, GM_ADDR dh0,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dBwdTilingData);
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR userWorkspace = workspace;
    if (workspace != nullptr) {
        userWorkspace = AscendC::GetUserWorkspace(workspace);
    }

#if (ORIG_DTYPE_X == DT_FLOAT)
    CausalConv1dBwdInputLayoutKernel<float, float> op;
#elif (ORIG_DTYPE_X == DT_FLOAT16)
    CausalConv1dBwdInputLayoutKernel<half, float> op;
#else
    CausalConv1dBwdInputLayoutKernel<bfloat16_t, float> op;
#endif
    op.InitTilingData(&tilingData, x, y, weight, dy,
                      initial_state, dht, query_start_loc, dx, dw, db, dh0,
                      userWorkspace);
    op.InitBuffer(&pipe);
    op.Process();
}
