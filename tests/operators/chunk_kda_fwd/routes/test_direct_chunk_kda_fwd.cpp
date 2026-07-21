// Canonical Ascend C stage-launch contract for ChunkKdaFwd.
// A complete direct route is stage1 -> scale/cast -> stage3 -> state -> stage2.
// Every launch configuration is derived from tests/op_cases/chunk_kda_fwd.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR stage_qg, GM_ADDR stage_aqk,
    GM_ADDR stage_v_new, GM_ADDR stage_h, GM_ADDR o, GM_ADDR final_state, GM_ADDR aqk,
    GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR workspace, GM_ADDR tiling);

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
    GM_ADDR workspace, GM_ADDR tiling);

void LaunchChunkKdaFwdStage(
    uint32_t blockDim, aclrtStream stream, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk,
    GM_ADDR beta, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR stage_qg, GM_ADDR stage_aqk, GM_ADDR stage_v_new, GM_ADDR stage_h, GM_ADDR o,
    GM_ADDR final_state, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg,
    GM_ADDR kg, GM_ADDR v_new, GM_ADDR h, GM_ADDR workspace, GM_ADDR tiling)
{
    chunk_kda_fwd<<<blockDim, nullptr, stream>>>(
        q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, stage_qg, stage_aqk,
        stage_v_new, stage_h, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
        workspace, tiling);
}

void LaunchChunkKdaStateStage(
    uint32_t blockDim, aclrtStream stream, GM_ADDR kg, GM_ADDR w, GM_ADDR u,
    GM_ADDR neutral_g, GM_ADDR gk, GM_ADDR initial_state, GM_ADDR cu_seqlens,
    GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
    GM_ADDR workspace, GM_ADDR tiling)
{
    chunk_gated_delta_rule_fwd_h<<<blockDim, nullptr, stream>>>(
        kg, w, u, neutral_g, gk, initial_state, cu_seqlens, chunk_indices,
        h, v_new, final_state, workspace, tiling);
}
