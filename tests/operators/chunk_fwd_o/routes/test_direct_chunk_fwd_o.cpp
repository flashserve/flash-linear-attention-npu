// Canonical Ascend C direct-launch contract for ChunkFwdO.
// blockDim/workspace/tiling are produced from tests/op_cases/chunk_fwd_o.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void chunk_fwd_o(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling);

void LaunchChunkFwdO(uint32_t blockDim, aclrtStream stream, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
{
    chunk_fwd_o<<<blockDim, nullptr, stream>>>(q, k, v, h, g, cu_seqlens, chunk_offsets, o, workspace, tiling);
}
