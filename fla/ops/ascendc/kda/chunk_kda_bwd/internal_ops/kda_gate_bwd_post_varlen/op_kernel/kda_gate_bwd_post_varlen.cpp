#include "../../chunk_kda_bwd/op_kernel/kda_gate_bwd_post.h"

#ifndef TORCH_MODE
constexpr bool kKdaGatePostCompileTimeVarlen = true;

extern "C" __global__ __aicore__ void kda_gate_bwd_post_varlen(
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

#define RUN_GATE_POST_VARLEN(SAFE)                                            \
    do {                                                                      \
        KDA::KdaGateBwdPostKernel<                                           \
            SAFE, DTYPE_RAW_G, kKdaGatePostCompileTimeVarlen> process;        \
        process.Init(dg_act, raw_g, a_log, dt_bias, cu_seqlens,               \
                     chunk_indices, dg, dA, dbias, tilingData, &pipe);         \
        process.Process();                                                     \
    } while (0)

    if (TILING_KEY_IS(1) || TILING_KEY_IS(2)) {
        RUN_GATE_POST_VARLEN(true);
    } else if (TILING_KEY_IS(3) || TILING_KEY_IS(4)) {
        RUN_GATE_POST_VARLEN(false);
    }

#undef RUN_GATE_POST_VARLEN
}
#endif
