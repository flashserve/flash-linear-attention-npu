// Canonical Ascend C direct-launch contract for ChunkGatedDeltaRuleFwdH.
// blockDim/workspace/tiling are produced from tests/op_cases/chunk_gated_delta_rule_fwd_h.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling);

void LaunchChunkGatedDeltaRuleFwdH(uint32_t blockDim, aclrtStream stream, GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    chunk_gated_delta_rule_fwd_h<<<blockDim, nullptr, stream>>>(k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, workspace, tiling);
}
