#include "chunk_scaled_dot_kkt.h"
#include "chunk_scaled_dot_kkt_tiling_key.h"

using namespace AscendC;

__aicore__ inline int64_t MinI64(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline bool HasOnlyFullChunks(GM_ADDR cuSeqlens,
                                         GM_ADDR chunkIndices,
                                         uint64_t t,
                                         uint64_t bt,
                                         uint64_t nt,
                                         uint64_t isVarlen)
{
    if (t == 0 || bt == 0) {
        return false;
    }
    if (isVarlen == 0) {
        return (t % bt) == 0;
    }
    GlobalTensor<int64_t> cuSeqlensGm;
    GlobalTensor<int64_t> chunkIndicesGm;
    cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens);
    chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunkIndices, static_cast<int64_t>(nt) * 2);
    for (int64_t chunk = 0; chunk < static_cast<int64_t>(nt); ++chunk) {
        const int64_t seqId = chunkIndicesGm.GetValue(chunk * 2);
        const int64_t localChunk = chunkIndicesGm.GetValue(chunk * 2 + 1);
        const int64_t bos = cuSeqlensGm.GetValue(seqId);
        const int64_t eos = cuSeqlensGm.GetValue(seqId + 1);
        const int64_t rowStart = bos + localChunk * static_cast<int64_t>(bt);
        int64_t valid = MinI64(static_cast<int64_t>(bt), eos - rowStart);
        valid = MinI64(valid, static_cast<int64_t>(t) - rowStart);
        if (valid != static_cast<int64_t>(bt)) {
            return false;
        }
    }
    return true;
}

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

    TPipe pipe;
    NsChunkScaledDotKkt::ChunkScaledDotKkt<DTYPE_K, CHUNK_KEY> op;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    if constexpr (CHUNK_KEY == CHUNK_SCALED_DOT_KKT_BT16 || CHUNK_KEY == CHUNK_SCALED_DOT_KKT_BT32 ||
                  CHUNK_KEY == CHUNK_SCALED_DOT_KKT_BT64 || CHUNK_KEY == CHUNK_SCALED_DOT_KKT_BT128) {
        const bool useCatlassScore = tilingData.T > 0 && tilingData.BT > 0 && tilingData.K > 0 &&
                                     (tilingData.K % 16) == 0 &&
                                     HasOnlyFullChunks(cuSeqlens, chunkIndices, tilingData.T, tilingData.BT,
                                                       tilingData.NT, tilingData.isVarlen);
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
}
