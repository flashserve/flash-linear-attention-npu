#include "chunk_scaled_dot_kkt.h"
#include "chunk_scaled_dot_kkt_tiling_key.h"

using namespace AscendC;

template <uint32_t D_T_K, uint32_t CHUNK_KEY>
__global__ __aicore__ void chunk_scaled_dot_kkt(GM_ADDR k,
                                                GM_ADDR g,
                                                GM_ADDR beta,
                                                GM_ADDR A,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(ChunkScaledDotKktTilingData, tilingData, tiling);

    TPipe pipe;
    NsChunkScaledDotKkt::ChunkScaledDotKkt<DTYPE_K> op;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.scoreMatmul, &tilingData.cubeTilingData);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    op.Init(k, g, beta, A, userWorkspace, tilingData.B, tilingData.H, tilingData.T, tilingData.K, tilingData.BT,
            tilingData.NT, tilingData.taskNum, tilingData.usedAicNum, tilingData.usedAivNum, tilingData.btAlign, &pipe);

    if ASCEND_IS_AIV {
        op.ProcessAiv();
    }
}
