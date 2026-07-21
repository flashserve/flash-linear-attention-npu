/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg.cpp
 */

#include "chunk_bwd_dqkwg_struct.h"
#include "kernel_operator.h"
#include "chunk_bwd_dqkwg_common.h"
#include "chunk_bwd_dqkwg_cube.h"
#include "chunk_bwd_dqkwg_vector.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

 using namespace AscendC;

namespace GDN {

 template <typename DT, typename GT>
 __aicore__ inline void ChunkBwdDqkwgKernelImpl(
     GM_ADDR q,              // [B, HK, T, K]
     GM_ADDR k,              // [B, HK, T, K]
     GM_ADDR v,              // [B, HV, T, V]
     GM_ADDR g,              // [B, HV, T]
     GM_ADDR h,              // [B, HV, num_chunks, K, V]
     GM_ADDR do_,            // [B, HV, T, V]
     GM_ADDR dh,             // [B, HV, num_chunks, K, V]
     GM_ADDR dv,             // [B, HV, T, V]
     GM_ADDR cu_seqlens,     // [N+1] (optional)
     GM_ADDR chunk_indices,  // [num_chunks, 2] (optional)
     GM_ADDR w,
     GM_ADDR g_gamma,
     GM_ADDR dq,             // [B, HK, T, K] - output
     GM_ADDR dk,             // [B, HK, T, K] - output
     GM_ADDR dw,             // [B, HV, T, K] - output
     GM_ADDR dg,             // [B, HV, T] - output (fp32)
     GM_ADDR userWorkspace,
     const ChunkBwdDqkwgTilingData *tilingData
 )
 {
     (void)w;
     (void)g_gamma;

     if ASCEND_IS_AIC {
         ChunkBwdDqkwgCubeProcess<DT, GT> cubeProcess(
             q, k, v, g, h,
             do_, dh, dv, cu_seqlens, chunk_indices,
             dq, dk, dw, dg,
             userWorkspace
         );
         cubeProcess.Init(*tilingData);
         cubeProcess.Process();
     }

     if ASCEND_IS_AIV {
         TPipe tPipe;
         ChunkBwdDqkwgVectorProcess<DT, GT> vectorProcess(
             q, k, v, g, h,
             do_, dh, dv, cu_seqlens, chunk_indices, nullptr,
             dq, dk, dw, dg,
             userWorkspace
         );
         vectorProcess.Init(*tilingData, &tPipe);
         vectorProcess.Process();
     }
 }

} // namespace GDN

#ifndef TORCH_MODE
template <int D_T>
struct DqkwgDTypeTraits;

template <>
struct DqkwgDTypeTraits<CHUNK_BWD_DQKWG_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct DqkwgDTypeTraits<CHUNK_BWD_DQKWG_TPL_FP16> {
    using type = half;
};

template <>
struct DqkwgDTypeTraits<CHUNK_BWD_DQKWG_TPL_FP32> {
    using type = float;
};

template <uint64_t strategy, int D_T_Q, int D_T_G, int V>
__global__ __aicore__ void chunk_bwd_dqkwg(
     GM_ADDR q,              // [B, HK, T, K]
     GM_ADDR k,              // [B, HK, T, K]
     GM_ADDR v,              // [B, HV, T, V]
     GM_ADDR g,              // [B, HV, T]
     GM_ADDR h,              // [B, HV, num_chunks, K, V]
     GM_ADDR do_,            // [B, HV, T, V]
     GM_ADDR dh,             // [B, HV, num_chunks, K, V]
     GM_ADDR dv,             // [B, HV, T, V]
     GM_ADDR cu_seqlens,     // [N+1] (optional)
     GM_ADDR chunk_indices,  // [num_chunks, 2] (optional)
     GM_ADDR w,
     GM_ADDR g_gamma,
     GM_ADDR dq,             // [B, HK, T, K] - output
     GM_ADDR dk,             // [B, HK, T, K] - output
     GM_ADDR dw,             // [B, HV, T, K] - output
     GM_ADDR dg,             // [B, HV, T] - output (fp32)
     GM_ADDR workspace,      // workspace buffer
     GM_ADDR tiling          // . data
 )
{

     (void)strategy;
     (void)V;
     AscendCUtils::SetOverflow(1);
     GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
     if (userWorkspace == nullptr) {
         return;
     }

     REGISTER_TILING_DEFAULT(GDN::ChunkBwdDqkwgTilingData);
     GET_TILING_DATA_WITH_STRUCT(GDN::ChunkBwdDqkwgTilingData, tilingData, tiling);
     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

     GDN::ChunkBwdDqkwgKernelImpl<typename DqkwgDTypeTraits<D_T_Q>::type,
                                  typename DqkwgDTypeTraits<D_T_G>::type>(
         q, k, v, g, h, do_, dh, dv, cu_seqlens, chunk_indices, w, g_gamma,
         dq, dk, dw, dg, userWorkspace, &tilingData);

     return;
 }
#endif
