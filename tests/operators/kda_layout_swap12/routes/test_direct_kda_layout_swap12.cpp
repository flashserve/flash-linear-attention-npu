// Canonical Ascend C direct-launch contract for KdaLayoutSwap12.
// blockDim/workspace/tiling are produced from tests/op_cases/kda_layout_swap12.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void kda_layout_swap12(GM_ADDR x, GM_ADDR dependency, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

void LaunchKdaLayoutSwap12(uint32_t blockDim, aclrtStream stream, GM_ADDR x, GM_ADDR dependency, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    kda_layout_swap12<<<blockDim, nullptr, stream>>>(x, dependency, y, workspace, tiling);
}
