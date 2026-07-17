// Canonical Ascend C direct-launch contract for CausalConv1d.
// blockDim/workspace/tiling are produced from tests/op_cases/causal_conv1d.json.
#include "acl/acl.h"
#include "kernel_operator.h"

template <uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey>
__global__ __aicore__ void causal_conv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

template <uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey>
void LaunchCausalConv1d(uint32_t blockDim, aclrtStream stream, GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    causal_conv1d<runModeKey, widthKey, fnPlanKey><<<blockDim, nullptr, stream>>>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace, tiling);
}
