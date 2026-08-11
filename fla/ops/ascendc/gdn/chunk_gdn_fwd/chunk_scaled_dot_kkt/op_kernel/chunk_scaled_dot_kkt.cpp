#include "chunk_scaled_dot_kkt.h"
#include "chunk_scaled_dot_kkt_tiling_key.h"

using namespace AscendC;

template <uint32_t D_T_K, uint32_t CHUNK_KEY>
__global__ __aicore__ void chunk_scaled_dot_kkt(GM_ADDR k,
                                                GM_ADDR g,
                                                GM_ADDR beta,
                                                GM_ADDR cuSeqlens,
                                                GM_ADDR chunkIndices,
                                                GM_ADDR A,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(ChunkScaledDotKktTilingData, tilingData, tiling);

    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if ASCEND_IS_AIC {
        TPipe pipe;
        NsChunkScaledDotKkt::ChunkScaledDotKkt<DTYPE_K, CHUNK_KEY> op;
        op.Init(k, g, beta, cuSeqlens, chunkIndices, A, userWorkspace, tilingData.B, tilingData.Hk, tilingData.Hv,
                tilingData.hvPerHk, tilingData.T, tilingData.K, tilingData.BT, tilingData.NT, tilingData.taskNum,
                tilingData.usedAicNum, tilingData.usedAivNum, tilingData.btAlign, tilingData.isVarlen,
                tilingData.useCatlassScore, &pipe);
        op.ProcessAic();
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        NsChunkScaledDotKkt::ChunkScaledDotKkt<DTYPE_K, CHUNK_KEY> op;
        op.Init(k, g, beta, cuSeqlens, chunkIndices, A, userWorkspace, tilingData.B, tilingData.Hk, tilingData.Hv,
                tilingData.hvPerHk, tilingData.T, tilingData.K, tilingData.BT, tilingData.NT, tilingData.taskNum,
                tilingData.usedAicNum, tilingData.usedAivNum, tilingData.btAlign, tilingData.isVarlen,
                tilingData.useCatlassScore, &pipe);
        op.ProcessAiv();
    }
#else
    TPipe pipe;
    NsChunkScaledDotKkt::ChunkScaledDotKkt<DTYPE_K, CHUNK_KEY> op;
    if constexpr (CHUNK_KEY == CHUNK_SCALED_DOT_KKT_BT16 || CHUNK_KEY == CHUNK_SCALED_DOT_KKT_BT32 ||
                  CHUNK_KEY == CHUNK_SCALED_DOT_KKT_BT64 || CHUNK_KEY == CHUNK_SCALED_DOT_KKT_BT128) {
        const bool useCatlassScore = tilingData.useCatlassScore != 0 && tilingData.T > 0 &&
                                     tilingData.BT >= NsChunkScaledDotKkt::CATLASS_SCORE_MIN_BT &&
                                     tilingData.K > 0 && (tilingData.K % 16) == 0;
        if (useCatlassScore) {
            op.Init(k, g, beta, cuSeqlens, chunkIndices, A, userWorkspace, tilingData.B, tilingData.Hk,
                    tilingData.Hv, tilingData.hvPerHk, tilingData.T, tilingData.K, tilingData.BT, tilingData.NT,
                    tilingData.taskNum, tilingData.usedAicNum, tilingData.usedAivNum, tilingData.btAlign,
                    tilingData.isVarlen, 1, &pipe);
            if ASCEND_IS_AIC {
                op.ProcessAic();
            }
            if ASCEND_IS_AIV {
                op.ProcessAiv();
            }
            return;
        }
    }

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.scoreMatmul, &tilingData.cubeTilingData);
    op.Init(k, g, beta, cuSeqlens, chunkIndices, A, userWorkspace, tilingData.B, tilingData.Hk, tilingData.Hv,
            tilingData.hvPerHk, tilingData.T, tilingData.K, tilingData.BT, tilingData.NT, tilingData.taskNum,
            tilingData.usedAicNum, tilingData.usedAivNum, tilingData.btAlign, tilingData.isVarlen, 0, &pipe);

    if ASCEND_IS_AIV {
        op.ProcessAiv();
    }
#endif
}
