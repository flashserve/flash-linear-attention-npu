// RecurrentGatedDeltaRuleV2 A5 entry. Its engine and tiling ABI are private to
// V2 and independent from the legacy operator entry points.
#include "arch35/recurrent_gated_delta_rule_v2_engine.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
using namespace RecurrentGatedDeltaRuleV2;

namespace RecurrentGatedDeltaRuleV2 {
template <int D_T>
struct DTypeTraits;
template <>
struct DTypeTraits<RGDR_V2_TPL_BF16> {
    using type = bfloat16_t;
};
template <>
struct DTypeTraits<RGDR_V2_TPL_FP32> {
    using type = float;
};
} // namespace RecurrentGatedDeltaRuleV2

template <int D_T_STATE, int D_T_G, int D_T_BETA, bool USE_QK_L2NORM, bool USE_GATE_IN_KERNEL, bool USE_BETA_SIGMOID,
          bool ALLOW_NEG_EIGVAL>
__global__ __aicore__ void
recurrent_gated_delta_rule_v2(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR state, GM_ADDR cuSeqlens,
                              GM_ADDR ssmStateIndices, GM_ADDR g, GM_ADDR gk, GM_ADDR numAcceptedTokens, GM_ADDR aLog,
                              GM_ADDR dtBias, GM_ADDR out, GM_ADDR stateOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)gk;
    REGISTER_TILING_DEFAULT(RecurrentGatedDeltaRuleV2TilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using StateType = typename RecurrentGatedDeltaRuleV2::DTypeTraits<D_T_STATE>::type;
    using GType = typename RecurrentGatedDeltaRuleV2::DTypeTraits<D_T_G>::type;
    using BetaType = typename RecurrentGatedDeltaRuleV2::DTypeTraits<D_T_BETA>::type;
    RGDRV2<bfloat16_t, bfloat16_t, StateType, GType, BetaType, USE_QK_L2NORM, USE_GATE_IN_KERNEL, USE_BETA_SIGMOID,
           ALLOW_NEG_EIGVAL>
        op(&tilingData);
    op.Init(query, key, value, g, beta, state, cuSeqlens, ssmStateIndices, aLog, dtBias, numAcceptedTokens, out,
            stateOut);
    op.Process();
}
