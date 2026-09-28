/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
#error "chunk_kda_bwd_finalize is an Ascend 950 / arch35-only kernel"
#endif

#include "chunk_kda_bwd_finalize_struct.h"
#include "arch35/chunk_kda_bwd_finalize_cube.h"
#include "arch35/chunk_kda_bwd_finalize_vector.h"

namespace KDA {

__aicore__ inline void ChunkKdaBwdFinalizeImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR akk,
    GM_ADDR vNew, GM_ADDR h, GM_ADDR dh, GM_ADDR dvScan,
    GM_ADDR dAqk, GM_ADDR dqRaw, GM_ADDR qRstd, GM_ADDR dq, GM_ADDR dv,
    GM_ADDR kRstd, GM_ADDR dk, GM_ADDR dBeta,
    GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR dG,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
    const ChunkKdaBwdFinalizeTilingData *tiling)
{
    if ASCEND_IS_AIC {
        ChunkKdaBwdFinalizeCubeStage10 cube;
        cube.Init(v, akk, vNew, h, dh, dvScan, cuSeqlens, chunkIndices,
                  workspace, tiling);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkKdaBwdFinalizeVectorStage12 vector;
        vector.Init(q, k, v, gk, beta, h, dh, dAqk, dqRaw, qRstd, dq, dv,
                    kRstd, dk, dBeta,
                    rawG, aLog, dtBias, dG,
                    cuSeqlens, chunkIndices,
                    workspace, tiling, &pipe);
        vector.Process();
    }
}

} // namespace KDA

extern "C" __global__ __aicore__ void chunk_kda_bwd_finalize(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk,
    GM_ADDR rawG, GM_ADDR beta, GM_ADDR aLog, GM_ADDR dtBias,
    GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh,
    GM_ADDR dvScan, GM_ADDR dAqk, GM_ADDR dqRaw,
    GM_ADDR qRstd, GM_ADDR kRstd,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dBeta,
    GM_ADDR dG, GM_ADDR dALog, GM_ADDR dDtBias,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdFinalizeTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdFinalizeTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(1) || TILING_KEY_IS(2) ||
        TILING_KEY_IS(3) || TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        KDA::ChunkKdaBwdFinalizeImpl(
            q, k, v, gk, beta, akk, vNew, h, dh, dvScan,
            dAqk, dqRaw, qRstd, dq, dv, kRstd, dk, dBeta,
            rawG, aLog, dtBias, dG,
            cuSeqlens, chunkIndices, userWorkspace, &tilingData);
        // Only AIVs produce the partials. AIV-only SyncAll uses flag 14,
        // which no earlier AIC->AIV handoff uses. MIX SyncAll uses 12/13
        // and would consume the final, unspent ZB_FREE credits instead.
        AscendC::SyncAll<true>();
        if ASCEND_IS_AIV {
            KDA::FinalizeStage12(userWorkspace, dALog, dDtBias, tilingData);
        }
    }
}
