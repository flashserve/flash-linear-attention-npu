/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "kernel_operator.h"
#include "arch35/chunk_gated_delta_rule_fwd_prepare.h"

using namespace AscendC;

#ifndef DTYPE_A_LOG
#define DTYPE_A_LOG DTYPE_G
#endif

REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleFwdStageTilingData);

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_prepare(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR g_out, GM_ADDR w_out, GM_ADDR u_out, GM_ADDR a_out,
    GM_ADDR q_hat, GM_ADDR k_hat, GM_ADDR q_rstd, GM_ADDR k_rstd, GM_ADDR beta_eff,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(ChunkGatedDeltaRuleFwdStageTilingData, tilingData, tiling);
    ChunkGatedDeltaRuleFwdPrepare::ChunkGatedDeltaRuleFwdPrepareKernel<DTYPE_G, DTYPE_A_LOG> op;
    op.Init(q, k, v, g, beta, a_log, dt_bias, cu_seqlens, chunk_indices,
            g_out, w_out, u_out, a_out, q_hat, k_hat, q_rstd, k_rstd, beta_eff,
            workspace, tilingData);
    op.Process();
}
