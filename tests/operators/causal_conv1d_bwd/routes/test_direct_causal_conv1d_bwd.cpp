// Canonical Ascend C direct-launch contract for CausalConv1dBwd.
// blockDim/workspace/tiling are produced from tests/op_cases/causal_conv1d_bwd.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void causal_conv1d_bwd(GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR dy, GM_ADDR initial_state, GM_ADDR dht, GM_ADDR query_start_loc, GM_ADDR dx, GM_ADDR dw, GM_ADDR db, GM_ADDR dh0, GM_ADDR workspace, GM_ADDR tiling);

void LaunchCausalConv1dBwd(uint32_t blockDim, aclrtStream stream, GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR dy, GM_ADDR initial_state, GM_ADDR dht, GM_ADDR query_start_loc, GM_ADDR dx, GM_ADDR dw, GM_ADDR db, GM_ADDR dh0, GM_ADDR workspace, GM_ADDR tiling)
{
    causal_conv1d_bwd<<<blockDim, nullptr, stream>>>(x, y, weight, dy, initial_state, dht, query_start_loc, dx, dw, db, dh0, workspace, tiling);
}
