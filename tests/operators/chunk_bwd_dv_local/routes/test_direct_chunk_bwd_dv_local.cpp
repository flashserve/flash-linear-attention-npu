// Canonical Ascend C direct-launch contract for ChunkBwdDvLocal.
// blockDim/workspace/tiling are produced from tests/op_cases/chunk_bwd_dv_local.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void chunk_bwd_dv_local(GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g, GM_ADDR g_gamma, GM_ADDR A, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace, GM_ADDR tiling);

void LaunchChunkBwdDvLocal(uint32_t blockDim, aclrtStream stream, GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g, GM_ADDR g_gamma, GM_ADDR A, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace, GM_ADDR tiling)
{
    chunk_bwd_dv_local<<<blockDim, nullptr, stream>>>(q, k, d_o, g, g_gamma, A, cu_seqlens, chunk_indices, d_v, workspace, tiling);
}
