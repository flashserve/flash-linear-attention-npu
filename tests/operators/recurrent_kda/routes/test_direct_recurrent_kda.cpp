// Canonical Ascend C direct-launch contract for RecurrentKda.
// blockDim/workspace/tiling are produced from tests/op_cases/recurrent_kda.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void recurrent_kda(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gate, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR numAcceptedTokens,
    GM_ADDR out, GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling);

void LaunchRecurrentKda(uint32_t blockDim,
                        aclrtStream stream,
                        GM_ADDR query,
                        GM_ADDR key,
                        GM_ADDR value,
                        GM_ADDR gate,
                        GM_ADDR beta,
                        GM_ADDR initialState,
                        GM_ADDR cuSeqlens,
                        GM_ADDR ssmStateIndices,
                        GM_ADDR aLog,
                        GM_ADDR dtBias,
                        GM_ADDR numAcceptedTokens,
                        GM_ADDR out,
                        GM_ADDR finalState,
                        GM_ADDR workspace,
                        GM_ADDR tiling)
{
    recurrent_kda<<<blockDim, nullptr, stream>>>(query, key, value, gate, beta, initialState, cuSeqlens,
                                                 ssmStateIndices, aLog, dtBias, numAcceptedTokens, out,
                                                 finalState, workspace, tiling);
}
