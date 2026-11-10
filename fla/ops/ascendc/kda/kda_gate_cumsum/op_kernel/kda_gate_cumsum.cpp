#include "fla_kernel_trace.h"
/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "kda_gate_cumsum_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void kda_gate_cumsum(GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias,
                                                       GM_ADDR cuSeqlens, GM_ADDR gk, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    FLA_KERNEL_TRACE("kda_gate_cumsum");
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (tilingData.dataType == 2) {
        KdaGateCumsum::DispatchKdaGateCumsum<float>(
            g, aLog, dtBias, cuSeqlens, gk, tilingData, &pipe);
    } else if (tilingData.dataType == 1) {
        KdaGateCumsum::DispatchKdaGateCumsum<bfloat16_t>(
            g, aLog, dtBias, cuSeqlens, gk, tilingData, &pipe);
    } else {
        KdaGateCumsum::DispatchKdaGateCumsum<half>(
            g, aLog, dtBias, cuSeqlens, gk, tilingData, &pipe);
    }
}
