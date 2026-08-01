/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/recurrent_kda_cube.h"
#include "arch35/recurrent_kda.h"
#include "arch35/recurrent_kda_tiling_data_apt.h"
#else
#include "recurrent_kda.h"
#endif
#include "recurrent_kda_tiling_data.h"


using namespace AscendC;
using namespace matmul;
using namespace RecurrentKda;


extern "C" __global__ __aicore__ void
recurrent_kda(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gate, GM_ADDR beta, GM_ADDR initialState,
              GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR numAcceptedTokens,
              GM_ADDR out, GM_ADDR initialStateOut, GM_ADDR finalState, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    REGISTER_TILING_DEFAULT(RecurrentKdaTilingDataA5);
    GET_TILING_DATA(tilingData, tilingGM);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspaceGM);
        if ASCEND_IS_AIC {
            RKDACube cubeOp(cuSeqlens, userWorkspace, &tilingData);
            cubeOp.Process();
        }
        if ASCEND_IS_AIV {
            TPipe pipe;
            RKDA<bfloat16_t, bfloat16_t, DTYPE_INITIAL_STATE> vectorOp(&tilingData.base);
            GM_ADDR stateOutput =
                tilingData.base.inplaceFinalState == 1 ? initialStateOut : finalState;
            RKDAInitParams initParams{query, key, value, gate, beta, initialState, cuSeqlens, ssmStateIndices,
                                      aLog, dtBias, numAcceptedTokens, out, stateOutput, userWorkspace,
                                      tilingData.workspaceStride};
            vectorOp.Init(initParams, &pipe);
            vectorOp.Process();
        }
    }
#else
    REGISTER_TILING_DEFAULT(RecurrentKdaTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    RKDA<bfloat16_t, bfloat16_t, DTYPE_INITIAL_STATE> op(&tilingData);
    GM_ADDR stateOutput = tilingData.inplaceFinalState == 1 ? initialStateOut : finalState;
    RKDAInitParams initParams{query, key, value, gate, beta, initialState, cuSeqlens, ssmStateIndices,
                              aLog, dtBias, numAcceptedTokens, out, stateOutput};
    op.Init(initParams, &pipe);
    op.Process();
#endif
}
