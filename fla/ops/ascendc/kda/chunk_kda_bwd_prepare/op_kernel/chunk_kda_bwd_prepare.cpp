/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"
// CANN's generated MIX wrapper references matmul::clearWorkspace.
#include "lib/matmul_intf.h"
#include "chunk_kda_bwd_prepare_struct.h"
#include "chunk_kda_bwd_prepare_tiling_key.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_bwd_prepare_cube.h"
#include "arch35/chunk_kda_bwd_prepare_vector.h"
#else
#include "arch22/chunk_kda_bwd_prepare_cube.h"
#include "arch22/chunk_kda_bwd_prepare_vector.h"
#endif

template <uint32_t INPUT_LAYOUT, bool IS_VARIABLE>
__global__ __aicore__ void chunk_kda_bwd_prepare(
    GM_ADDR aqk, GM_ADDR vNew, GM_ADDR dO, GM_ADDR h,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dAqk, GM_ADDR dv, GM_ADDR dqRaw,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdPrepareTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdPrepareTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    (void)workspace;
    if ASCEND_IS_AIC {
        if (tilingData.stateVFirst != 0) {
            KDA::ChunkKdaBwdPrepareCube<INPUT_LAYOUT, true> cube;
            cube.Init(aqk, vNew, dO, h, cuSeqlens, chunkIndices,
                      dAqk, dv, dqRaw, &tilingData);
            cube.Process();
        } else {
            KDA::ChunkKdaBwdPrepareCube<INPUT_LAYOUT, false> cube;
            cube.Init(aqk, vNew, dO, h, cuSeqlens, chunkIndices,
                      dAqk, dv, dqRaw, &tilingData);
            cube.Process();
        }
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        KDA::ChunkKdaBwdPrepareVector vector;
        vector.Init(cuSeqlens, chunkIndices, dAqk, dv, dqRaw, &tilingData, &pipe);
        vector.Process();
    }
#else
    GM_ADDR scratch = AscendC::GetUserWorkspace(workspace);
    AscendC::TPipe pipe;
    if ASCEND_IS_AIC {
        if (tilingData.stateVFirst != 0) {
            KDA::ChunkKdaBwdPrepareCube<INPUT_LAYOUT, true> cube;
            cube.Init(aqk, vNew, dO, h, cuSeqlens, chunkIndices, dv, dqRaw, scratch, &tilingData, &pipe);
            cube.Process();
        } else {
            KDA::ChunkKdaBwdPrepareCube<INPUT_LAYOUT, false> cube;
            cube.Init(aqk, vNew, dO, h, cuSeqlens, chunkIndices, dv, dqRaw, scratch, &tilingData, &pipe);
            cube.Process();
        }
    }
    if ASCEND_IS_AIV {
        KDA::ChunkKdaBwdPrepareVector vector;
        vector.Init(cuSeqlens, chunkIndices, dAqk, scratch, &tilingData, &pipe);
        vector.Process();
    }
#endif
}
