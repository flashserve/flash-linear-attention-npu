#include "chunk_scaled_dot_kkt_fused_cumsum.h"
#include "chunk_scaled_dot_kkt_tiling_key.h"

using namespace AscendC;

template <uint32_t D_T_K, uint32_t CHUNK_KEY>
__global__ __aicore__ void chunk_cumsum_kkt(GM_ADDR k,
                                            GM_ADDR g,
                                            GM_ADDR beta,
                                            GM_ADDR cuSeqlens,
                                            GM_ADDR chunkIndices,
                                            GM_ADDR gCumsum,
                                            GM_ADDR A,
                                            GM_ADDR workspace,
                                            GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(ChunkScaledDotKktTilingData, tilingData, tiling);

    TPipe pipe;
    NsChunkScaledDotKktFusedCumsum::ChunkScaledDotKktFusedCumsum<DTYPE_K> op;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.scoreMatmul, &tilingData.cubeTilingData);
    GM_ADDR scoreWorkspace = GetUserWorkspace(workspace);
    op.InitFusedCumsum(k, g, beta, cuSeqlens, chunkIndices, gCumsum, A, scoreWorkspace,
                       tilingData.B, tilingData.Hk, tilingData.Hv, tilingData.hvPerHk,
                       tilingData.T, tilingData.K, tilingData.BT, tilingData.NT, tilingData.taskNum,
                       tilingData.usedAicNum, tilingData.usedAivNum, tilingData.btAlign,
                       tilingData.isVarlen, &pipe);
    if ASCEND_IS_AIV {
        op.ProcessAiv();
    }
}
