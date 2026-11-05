#ifndef RECURRENT_GATED_DELTA_RULE_V2_STRUCT_H
#define RECURRENT_GATED_DELTA_RULE_V2_STRUCT_H

#include <cstdint>

#include "ascendc/host_api/tiling/template_argument.h"

namespace RecurrentGatedDeltaRuleV2 {
#pragma pack(push, 8)
struct alignas(8) RecurrentGatedDeltaRuleV2TilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t t;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t ssmStateStride;
    uint32_t b;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
    float compatibilityFloat0;
    uint32_t layout;
    uint32_t hasDtBias;
    uint32_t hasAcceptedTokens;
    uint32_t useQkL2norm;
    uint32_t useGateInKernel;
    uint32_t useBetaSigmoid;
    uint32_t allowNegEigval;
    uint32_t compatibilityUint0;
    uint32_t stateVFirst;
    uint32_t outputFinalState;
    uint32_t inplaceFinalState;
    uint64_t stateInStride0;
    uint64_t stateInStride1;
    uint64_t stateInStride2;
    uint64_t stateInStride3;
    uint64_t stateOutStride0;
    uint64_t stateOutStride1;
    uint64_t stateOutStride2;
    uint64_t stateOutStride3;
    uint64_t queryTokenStride;
    uint64_t queryHeadStride;
    uint64_t queryFeatureStride;
    uint64_t keyTokenStride;
    uint64_t keyHeadStride;
    uint64_t keyFeatureStride;
    uint64_t valueTokenStride;
    uint64_t valueHeadStride;
    uint64_t valueFeatureStride;
};
#pragma pack(pop)

#define RGDR_V2_TPL_BF16 10
#define RGDR_V2_TPL_FP32 30

ASCENDC_TPL_ARGS_DECL(RecurrentGatedDeltaRuleV2, ASCENDC_TPL_DTYPE_DECL(D_T_STATE, RGDR_V2_TPL_BF16, RGDR_V2_TPL_FP32),
                      ASCENDC_TPL_DTYPE_DECL(D_T_G, RGDR_V2_TPL_BF16, RGDR_V2_TPL_FP32),
                      ASCENDC_TPL_DTYPE_DECL(D_T_BETA, RGDR_V2_TPL_BF16, RGDR_V2_TPL_FP32),
                      ASCENDC_TPL_BOOL_DECL(USE_QK_L2NORM, 0, 1), ASCENDC_TPL_BOOL_DECL(USE_GATE_IN_KERNEL, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(USE_BETA_SIGMOID, 0, 1), ASCENDC_TPL_BOOL_DECL(ALLOW_NEG_EIGVAL, 0, 1), );

#define RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, USE_QK_L2NORM_VALUE, USE_GATE_IN_KERNEL_VALUE,              \
                            USE_BETA_SIGMOID_VALUE, ALLOW_NEG_EIGVAL_VALUE)                                            \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_STATE, STATE_TYPE), ASCENDC_TPL_DTYPE_SEL(D_T_G, G_TYPE),           \
                         ASCENDC_TPL_DTYPE_SEL(D_T_BETA, BETA_TYPE),                                                   \
                         ASCENDC_TPL_BOOL_SEL(USE_QK_L2NORM, USE_QK_L2NORM_VALUE),                                     \
                         ASCENDC_TPL_BOOL_SEL(USE_GATE_IN_KERNEL, USE_GATE_IN_KERNEL_VALUE),                           \
                         ASCENDC_TPL_BOOL_SEL(USE_BETA_SIGMOID, USE_BETA_SIGMOID_VALUE),                               \
                         ASCENDC_TPL_BOOL_SEL(ALLOW_NEG_EIGVAL, ALLOW_NEG_EIGVAL_VALUE), )

#define RGDR_V2_TPL_SEL_FLAGS(STATE_TYPE, G_TYPE, BETA_TYPE)                                                           \
    RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 0, 0, 0),                                                    \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 0, 0, 1),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 0, 1, 0),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 0, 1, 1),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 1, 0, 0),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 1, 0, 1),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 1, 1, 0),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 1, 1, 1),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 0, 0, 0),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 0, 0, 1),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 0, 1, 0),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 0, 1, 1),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 1, 0, 0),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 1, 0, 1),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 1, 1, 0),                                                \
        RGDR_V2_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 1, 1, 1)

#define RGDR_V2_TPL_SEL_SCALARS(STATE_TYPE)                                                                            \
    RGDR_V2_TPL_SEL_FLAGS(STATE_TYPE, RGDR_V2_TPL_BF16, RGDR_V2_TPL_BF16),                                             \
        RGDR_V2_TPL_SEL_FLAGS(STATE_TYPE, RGDR_V2_TPL_BF16, RGDR_V2_TPL_FP32),                                         \
        RGDR_V2_TPL_SEL_FLAGS(STATE_TYPE, RGDR_V2_TPL_FP32, RGDR_V2_TPL_BF16),                                         \
        RGDR_V2_TPL_SEL_FLAGS(STATE_TYPE, RGDR_V2_TPL_FP32, RGDR_V2_TPL_FP32)

ASCENDC_TPL_SEL(RGDR_V2_TPL_SEL_SCALARS(RGDR_V2_TPL_BF16), RGDR_V2_TPL_SEL_SCALARS(RGDR_V2_TPL_FP32), );

#undef RGDR_V2_TPL_SEL_SCALARS
#undef RGDR_V2_TPL_SEL_FLAGS
#undef RGDR_V2_TPL_SEL_ONE
} // namespace RecurrentGatedDeltaRuleV2

#endif // RECURRENT_GATED_DELTA_RULE_V2_STRUCT_H
