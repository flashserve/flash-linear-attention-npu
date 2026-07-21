// Canonical Ascend C direct-launch contract for KdaGateCumsum.
// blockDim/workspace/tiling are produced from tests/op_cases/kda_gate_cumsum.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void kda_gate_cumsum(GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens, GM_ADDR gk, GM_ADDR workspace, GM_ADDR tiling);

void LaunchKdaGateCumsum(uint32_t blockDim, aclrtStream stream, GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens, GM_ADDR gk, GM_ADDR workspace, GM_ADDR tiling)
{
    kda_gate_cumsum<<<blockDim, nullptr, stream>>>(g, aLog, dtBias, cuSeqlens, gk, workspace, tiling);
}
