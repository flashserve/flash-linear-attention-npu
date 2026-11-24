#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define KDA_COMPILE_CHUNK32_FUSED 1
#include "arch35/chunk_kda_fwd_fused/chunk_kda_fwd_32.h"
#else
#define KDA_COMPILE_CHUNK32_FUSED 0
#endif

extern "C" __global__ __aicore__ void chunk_kda_fwd_fused(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR attn_out, GM_ADDR final_state,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if KDA_COMPILE_CHUNK32_FUSED
    KdaForward::Chunk32::Run(
        q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
        nullptr, attn_out, final_state, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        workspace, tiling);
#endif
}

#undef KDA_COMPILE_CHUNK32_FUSED
