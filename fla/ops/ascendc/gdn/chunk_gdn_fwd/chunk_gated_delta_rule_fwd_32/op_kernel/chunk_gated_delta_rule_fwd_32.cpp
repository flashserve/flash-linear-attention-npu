#include "kda_pipe/gdn_kda_pipe.h"

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd32(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g,
                                                                   GM_ADDR beta, GM_ADDR initialState, GM_ADDR o,
                                                                   GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleFwd32TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(ChunkGatedDeltaRuleFwd32TilingData, td, tiling);
    // Route to the KDA-scheduled kernel. arch35/ keeps the previous one.
    GdnKdaPipe::Kernel op;
    op.Init(q, k, v, g, beta, initialState, o, finalState, workspace, td);
    op.Process();
}
