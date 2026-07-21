// Canonical Ascend C direct-launch contract for RecurrentGatedDeltaRule.
// blockDim/workspace/tiling are produced from tests/op_cases/recurrent_gated_delta_rule.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void recurrent_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR state, GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR g, GM_ADDR gk, GM_ADDR numAcceptedTokens, GM_ADDR out, GM_ADDR stateOut, GM_ADDR workspaceGM, GM_ADDR tilingGM);

void LaunchRecurrentGatedDeltaRule(uint32_t blockDim, aclrtStream stream, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR state, GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR g, GM_ADDR gk, GM_ADDR numAcceptedTokens, GM_ADDR out, GM_ADDR stateOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    recurrent_gated_delta_rule<<<blockDim, nullptr, stream>>>(query, key, value, beta, state, cuSeqlens, ssmStateIndices, g, gk, numAcceptedTokens, out, stateOut, workspaceGM, tilingGM);
}
