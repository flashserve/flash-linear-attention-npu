// Canonical Ascend C direct-launch contract for SolveTri.
// blockDim/workspace/tiling are produced from tests/op_cases/solve_tri.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void solve_tri(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR x_out, GM_ADDR workspace, GM_ADDR tiling);

void LaunchSolveTri(uint32_t blockDim, aclrtStream stream, GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR x_out, GM_ADDR workspace, GM_ADDR tiling)
{
    solve_tri<<<blockDim, nullptr, stream>>>(x, cu_seqlens, chunk_indices, x_out, workspace, tiling);
}
