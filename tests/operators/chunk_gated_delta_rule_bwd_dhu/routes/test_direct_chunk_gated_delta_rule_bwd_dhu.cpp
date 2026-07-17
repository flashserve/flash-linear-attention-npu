// Canonical Ascend C direct-launch contract for ChunkGatedDeltaRuleBwdDhu.
// blockDim/workspace/tiling are produced from tests/op_cases/chunk_gated_delta_rule_bwd_dhu.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_bwd_dhu(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR h0, GM_ADDR dht, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace, GM_ADDR tiling);

void LaunchChunkGatedDeltaRuleBwdDhu(uint32_t blockDim, aclrtStream stream, GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR h0, GM_ADDR dht, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace, GM_ADDR tiling)
{
    chunk_gated_delta_rule_bwd_dhu<<<blockDim, nullptr, stream>>>(q, k, w, d_o, dv, g, gk, h0, dht, cu_seqlens, chunk_indices, dh, dh0, dv2, workspace, tiling);
}
