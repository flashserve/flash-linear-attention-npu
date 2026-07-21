// Canonical Ascend C direct-launch contract for ChunkLocalCumsum.
// blockDim/workspace/tiling are produced from tests/op_cases/chunk_local_cumsum.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void chunk_local_cumsum(GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);

void LaunchChunkLocalCumsum(uint32_t blockDim, aclrtStream stream, GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    chunk_local_cumsum<<<blockDim, nullptr, stream>>>(g, cuSeqlens, chunkIndices, out, workspace, tiling);
}
