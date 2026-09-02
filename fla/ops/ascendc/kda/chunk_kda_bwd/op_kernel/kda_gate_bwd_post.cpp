#include "kda_gate_bwd_post.h"

#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void kda_gate_bwd_post(
    GM_ADDR dg_act, GM_ADDR raw_g, GM_ADDR a_log, GM_ADDR dt_bias,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dg, GM_ADDR dA,
    GM_ADDR dbias, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(KDA::KdaGateBwdPostTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        KDA::KdaGateBwdPostTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

#define RUN_GATE_POST(SAFE, VARLEN)                                           \
    do {                                                                      \
        KDA::KdaGateBwdPostKernel<SAFE, DTYPE_RAW_G, VARLEN> process;         \
        process.Init(dg_act, raw_g, a_log, dt_bias, cu_seqlens,               \
                     chunk_indices, dg, dA, dbias, tilingData, &pipe);         \
        process.Process();                                                     \
    } while (0)

    if (TILING_KEY_IS(1)) {
        RUN_GATE_POST(true, false);
    } else if (TILING_KEY_IS(2)) {
        RUN_GATE_POST(true, true);
    } else if (TILING_KEY_IS(3)) {
        RUN_GATE_POST(false, false);
    } else if (TILING_KEY_IS(4)) {
        RUN_GATE_POST(false, true);
    }

#undef RUN_GATE_POST
}
#endif
