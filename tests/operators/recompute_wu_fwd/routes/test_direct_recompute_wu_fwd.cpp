// Canonical Ascend C direct-launch contract for RecomputeWUFwd.
// blockDim/workspace/tiling are produced from tests/op_cases/recompute_wu_fwd.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void recompute_wu_fwd(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR gk, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace, GM_ADDR tiling);

void LaunchRecomputeWUFwd(uint32_t blockDim, aclrtStream stream, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR gk, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace, GM_ADDR tiling)
{
    recompute_wu_fwd<<<blockDim, nullptr, stream>>>(k, v, beta, A, g, gk, cu_seqlens, chunk_indices, w, u, workspace, tiling);
}
