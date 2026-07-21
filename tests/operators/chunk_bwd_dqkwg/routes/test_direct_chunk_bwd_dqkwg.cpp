// Canonical Ascend C direct-launch contract for ChunkBwdDqkwg.
// blockDim/workspace/tiling are produced from tests/op_cases/chunk_bwd_dqkwg.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void chunk_bwd_dqkwg(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h, GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR g_gamma, GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg, GM_ADDR workspace, GM_ADDR tiling);

void LaunchChunkBwdDqkwg(uint32_t blockDim, aclrtStream stream, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h, GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR g_gamma, GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg, GM_ADDR workspace, GM_ADDR tiling)
{
    chunk_bwd_dqkwg<<<blockDim, nullptr, stream>>>(q, k, v, g, h, do_, dh, dv, cu_seqlens, chunk_indices, w, g_gamma, dq, dk, dw, dg, workspace, tiling);
}
