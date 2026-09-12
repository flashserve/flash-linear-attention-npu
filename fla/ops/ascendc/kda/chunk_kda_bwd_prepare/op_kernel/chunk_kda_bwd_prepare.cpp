/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
#error "chunk_kda_bwd_prepare is an Ascend 950 / arch35-only kernel"
#endif

#include "chunk_kda_bwd_prepare_struct.h"
#include "arch35/chunk_kda_bwd_prepare_cube.h"
#include "arch35/chunk_kda_bwd_prepare_vector.h"

namespace KDA {

__aicore__ inline void ChunkKdaBwdPrepareImpl(
    GM_ADDR aqk, GM_ADDR vNew, GM_ADDR dO, GM_ADDR h,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dAqk, GM_ADDR dv, GM_ADDR dqRaw,
    const ChunkKdaBwdPrepareTilingData *tiling)
{
    if ASCEND_IS_AIC {
        if (tiling->stateVFirst != 0) {
            ChunkKdaBwdPrepareCube<true> cube;
            cube.Init(aqk, vNew, dO, h, cuSeqlens, chunkIndices,
                      dAqk, dv, dqRaw, tiling);
            cube.Process();
        } else {
            ChunkKdaBwdPrepareCube<false> cube;
            cube.Init(aqk, vNew, dO, h, cuSeqlens, chunkIndices,
                      dAqk, dv, dqRaw, tiling);
            cube.Process();
        }
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkKdaBwdPrepareVector vector;
        vector.Init(cuSeqlens, chunkIndices, dAqk, dv, dqRaw, tiling, &pipe);
        vector.Process();
    }
}

} // namespace KDA

extern "C" __global__ __aicore__ void chunk_kda_bwd_prepare(
    GM_ADDR aqk, GM_ADDR vNew, GM_ADDR dO, GM_ADDR h,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dAqk, GM_ADDR dv, GM_ADDR dqRaw,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdPrepareTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdPrepareTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(1) || TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        KDA::ChunkKdaBwdPrepareImpl(
            aqk, vNew, dO, h, cuSeqlens, chunkIndices,
            dAqk, dv, dqRaw, &tilingData);
    }
}
