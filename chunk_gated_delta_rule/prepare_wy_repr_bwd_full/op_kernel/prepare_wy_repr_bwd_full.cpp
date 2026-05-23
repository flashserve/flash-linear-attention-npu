/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_full.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/prepare_wy_repr_bwd_full_tiling_data_apt.h"
#include "arch35/prepare_wy_repr_bwd_full_common.h"
#include "arch35/prepare_wy_repr_bwd_full_cube.h"
#include "arch35/prepare_wy_repr_bwd_full_vector.h"
#else
#include "prepare_wy_repr_bwd_full_common.h"
#include "prepare_wy_repr_bwd_full_cube.h"
#include "prepare_wy_repr_bwd_full_vector.h"
#endif

using namespace AscendC;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
// Arch35 cube path: tile shapes come from tiling (PrepareWyReprBwdFullTilingDataA5), not template parameters.
#else
/** Cube GEMM tile shape: must be `tla::Shape` (BlockMmadTla); keep explicit to avoid ambiguity with Catlass. */
template <class... Dims>
using GemmCubeTileShape = tla::Shape<Dims...>;
using namespace tla;
#endif
__global__ __aicore__ void prepare_wy_repr_bwd_full(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dA,
                                                    GM_ADDR dw, GM_ADDR du, GM_ADDR g, GM_ADDR cu_seqlens,
                                                    GM_ADDR chunk_indices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta,
                                                    GM_ADDR dg, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    REGISTER_TILING_DEFAULT(PrepareWyReprBwdFullTilingDataA5);
#endif
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIC {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            PrepareWyReprBwdFullProcess<DTYPE_K, DTYPE_BETA> prepareWyReprBwdFullProcess(
                k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg, workspace);
#else
            PrepareWyReprBwdFullProcess<DTYPE_K, DTYPE_BETA, GemmCubeTileShape<_128, _128, _256>,
                                        GemmCubeTileShape<_128, _128, _128>>
                prepareWyReprBwdFullProcess(k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg,
                                            workspace);
#endif
            prepareWyReprBwdFullProcess.Init(tilingData);
            prepareWyReprBwdFullProcess.Process();
        }
        if ASCEND_IS_AIV {
            AscendC::TPipe tPipe;
            PrepareWyReprBwdFullVectorProcess<DTYPE_K, DTYPE_BETA> prepareWyReprBwdFullVectorProcess(
                k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg, workspace);
            prepareWyReprBwdFullVectorProcess.Init(tilingData, &tPipe);
            prepareWyReprBwdFullVectorProcess.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIC {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            PrepareWyReprBwdFullProcess<DTYPE_K, DTYPE_BETA> prepareWyReprBwdFullProcess(
                k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg, workspace);
#else
            PrepareWyReprBwdFullProcess<DTYPE_K, DTYPE_BETA, GemmCubeTileShape<_128, _256, _256>,
                                        GemmCubeTileShape<_128, _256, _64>>
                prepareWyReprBwdFullProcess(k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg,
                                            workspace);
#endif
            prepareWyReprBwdFullProcess.Init(tilingData);
            prepareWyReprBwdFullProcess.Process();
        }
        if ASCEND_IS_AIV {
            AscendC::TPipe tPipe;
            PrepareWyReprBwdFullVectorProcess<DTYPE_K, DTYPE_BETA> prepareWyReprBwdFullVectorProcess(
                k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg, workspace);
            prepareWyReprBwdFullVectorProcess.Init(tilingData, &tPipe);
            prepareWyReprBwdFullVectorProcess.Process();
        }
    }
    return;
}
