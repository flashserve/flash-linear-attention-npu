/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#include "chunk_kda_bwd_recompute_struct.h"
#include "chunk_kda_bwd_recompute_tiling_key.h"
#include "chunk_kda_bwd_recompute_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_bwd_recompute_cube.h"
#include "arch35/chunk_kda_bwd_recompute_vector.h"
#else
#include "chunk_kda_bwd_recompute_cube.h"
#include "chunk_kda_bwd_recompute_vector.h"
#endif

namespace KDA {

template <typename QkType, typename GateType, typename BetaType>
__aicore__ inline void ChunkKdaBwdRecomputeKernelImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR a, GM_ADDR aLog, GM_ADDR dtBias,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR gk,
    GM_ADDR workspace, const ChunkKdaBwdRecomputeTilingData *tiling)
{
    if ASCEND_IS_AIC {
        ChunkKdaBwdRecomputeCubeProcess<QkType> cube(a, cuSeqlens, chunkIndices, w, u, workspace);
        cube.Init(tiling);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        ChunkKdaBwdRecomputeVectorProcess<QkType, GateType, BetaType> vector(
            q, k, v, g, beta, aLog, dtBias, cuSeqlens, chunkIndices, w, u, qg, kg, gk, workspace, tiling);
        vector.Init(&pipe);
        vector.Process();
    }
}

} // namespace KDA

#ifndef TORCH_MODE
using namespace AscendC;

__aicore__ inline void CopyTilingFromGm(
    const __gm__ KDA::ChunkKdaBwdRecomputeTilingData *src, KDA::ChunkKdaBwdRecomputeTilingData &dst)
{
    dst.B = src->B;
    dst.Hk = src->Hk;
    dst.Hv = src->Hv;
    dst.hvPerHk = src->hvPerHk;
    dst.T = src->T;
    dst.K = src->K;
    dst.V = src->V;
    dst.chunkNum = src->chunkNum;
    dst.chunkSize = src->chunkSize;
    dst.isVariable = src->isVariable;
    dst.useGateInKernel = src->useGateInKernel;
    dst.useExp2 = src->useExp2;
    dst.hasALog = src->hasALog;
    dst.hasDtBias = src->hasDtBias;
    dst.lowerBoundBits = src->lowerBoundBits;
    dst.vecRow = src->vecRow;
}

__global__ __aicore__ void chunk_kda_bwd_recompute(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR a, GM_ADDR a_log, GM_ADDR dt_bias,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR gk,
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        userWorkspace = workspace;
    }
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdRecomputeTilingData);
    KDA::ChunkKdaBwdRecomputeTilingData tilingData;
    CopyTilingFromGm(
        reinterpret_cast<__gm__ KDA::ChunkKdaBwdRecomputeTilingData *>(tiling), tilingData);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        KDA::ChunkKdaBwdRecomputeKernelImpl<bfloat16_t, bfloat16_t, bfloat16_t>(
            q, k, v, g, beta, a, a_log, dt_bias, cu_seqlens, chunk_indices, w, u, qg, kg, gk, userWorkspace,
            &tilingData);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        KDA::ChunkKdaBwdRecomputeKernelImpl<bfloat16_t, bfloat16_t, float>(
            q, k, v, g, beta, a, a_log, dt_bias, cu_seqlens, chunk_indices, w, u, qg, kg, gk, userWorkspace,
            &tilingData);
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        KDA::ChunkKdaBwdRecomputeKernelImpl<bfloat16_t, float, bfloat16_t>(
            q, k, v, g, beta, a, a_log, dt_bias, cu_seqlens, chunk_indices, w, u, qg, kg, gk, userWorkspace,
            &tilingData);
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        KDA::ChunkKdaBwdRecomputeKernelImpl<bfloat16_t, float, float>(
            q, k, v, g, beta, a, a_log, dt_bias, cu_seqlens, chunk_indices, w, u, qg, kg, gk, userWorkspace,
            &tilingData);
    }
}
#endif
