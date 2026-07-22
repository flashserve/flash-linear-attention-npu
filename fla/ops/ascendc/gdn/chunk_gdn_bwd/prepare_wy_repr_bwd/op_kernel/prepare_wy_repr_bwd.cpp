/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd.cpp
 * \brief Fused prepare_wy_repr_bwd kernel entry.
 */

#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/prepare_wy_repr_bwd_struct.h"
#include "arch35/prepare_wy_repr_bwd_tiling_key.h"
#include "arch35/prepare_wy_repr_bwd_common.h"
#include "arch35/prepare_wy_repr_bwd_cube.h"
#include "arch35/prepare_wy_repr_bwd_vector.h"
#else
#include "prepare_wy_repr_bwd_struct.h"
#include "prepare_wy_repr_bwd_tiling_key.h"
#include "prepare_wy_repr_bwd_common.h"
#include "prepare_wy_repr_bwd_cube.h"
#include "prepare_wy_repr_bwd_vector.h"
#endif

using namespace AscendC;

namespace GDN {

template <uint32_t D_T>
struct PrepareWyReprBwdDTypeTraits;

template <>
struct PrepareWyReprBwdDTypeTraits<TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct PrepareWyReprBwdDTypeTraits<TPL_FP16> {
    using type = half;
};

template <>
struct PrepareWyReprBwdDTypeTraits<TPL_FP32> {
    using type = float;
};

template <typename KType, typename GType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdKernelImpl(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dw, GM_ADDR du, GM_ADDR g,
                           GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg,
                           GM_ADDR workspace, const PrepareWyReprBwdTilingData *tilingData)
{
    if ASCEND_IS_AIC {
        ::PrepareWyReprBwdCubeProcess<KType, GType, V_DIM, CHUNK_SIZE> cubeProcess(
            k, A, dw, du, cuSeqlens, chunkIndices, workspace);
        cubeProcess.Init(*tilingData);
        cubeProcess.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe tPipe;
        ::PrepareWyReprBwdVectorProcess<KType, GType, V_DIM, CHUNK_SIZE> vectorProcess(
            k, v, beta, g, cuSeqlens, chunkIndices, dk, dv, dbeta, dg, workspace);
        vectorProcess.Init(*tilingData, &tPipe);
        vectorProcess.Process();
    }
}

} // namespace GDN

#ifndef TORCH_MODE
template <uint32_t D_T_K, uint32_t D_T_G, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__global__ __aicore__ void prepare_wy_repr_bwd(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dw, GM_ADDR du,
                                               GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dk,
                                               GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg, GM_ADDR workspace,
                                               GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(GDN::PrepareWyReprBwdTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::PrepareWyReprBwdTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    using KType = typename GDN::PrepareWyReprBwdDTypeTraits<D_T_K>::type;
    using GType = typename GDN::PrepareWyReprBwdDTypeTraits<D_T_G>::type;
    GDN::PrepareWyReprBwdKernelImpl<KType, GType, V_DIM, CHUNK_SIZE>(
        k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg, userWorkspace, &tilingData);
}
#endif
