// Canonical Ascend C direct-launch contract for ChunkScaledDotKkt.
// blockDim/workspace/tiling are produced from tests/op_cases/chunk_scaled_dot_kkt.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void chunk_scaled_dot_kkt(GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR A, GM_ADDR workspace, GM_ADDR tiling);

void LaunchChunkScaledDotKkt(uint32_t blockDim, aclrtStream stream, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR A, GM_ADDR workspace, GM_ADDR tiling)
{
    chunk_scaled_dot_kkt<<<blockDim, nullptr, stream>>>(k, g, beta, cuSeqlens, chunkIndices, A, workspace, tiling);
}
