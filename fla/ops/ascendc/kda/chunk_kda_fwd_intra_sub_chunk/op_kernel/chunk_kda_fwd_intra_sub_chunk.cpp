/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaFwdIntraSubChunk — BNSD + optional TND varlen, GVA.
 * tiling key 1: KERNEL_TYPE_MIX_AIC_1_2
 *   Cube  : two score GEMMs (Aqk_raw, Akk_raw), Kg held in L1.
 *   Vector: gate prep + tril/scale + Forward Substitution (I+L)^{-1}.
 *
 * Aligns with GPU Triton chunk_kda_fwd_kernel_intra_sub_chunk (Forward Substitution,
 * NOT MCH). A5 (arch35 / Ascend950) compiles from the same source via CATLASS_ARCH.
 */

#include "chunk_kda_fwd_intra_sub_chunk_common.h"
#include "chunk_kda_fwd_intra_sub_chunk_cube.h"
#include "chunk_kda_fwd_intra_sub_chunk_vector.h"

extern "C" __global__ __aicore__ void chunk_kda_fwd_intra_sub_chunk(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta,
                                                                    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk,
                                                                    GM_ADDR akkd, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        if (tilingData.dataType == 1) {
            if ASCEND_IS_AIC {
                kda_isub::KdaSubChunkCube<bfloat16_t> op;
                op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tilingData, &pipe);
                op.Process();
            }
            if ASCEND_IS_AIV {
                kda_isub::KdaSubChunkVector<bfloat16_t> op;
                op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tilingData, &pipe);
                op.Process();
            }
        } else {
            if ASCEND_IS_AIC {
                kda_isub::KdaSubChunkCube<half> op;
                op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tilingData, &pipe);
                op.Process();
            }
            if ASCEND_IS_AIV {
                kda_isub::KdaSubChunkVector<half> op;
                op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tilingData, &pipe);
                op.Process();
            }
        }
    }
}
