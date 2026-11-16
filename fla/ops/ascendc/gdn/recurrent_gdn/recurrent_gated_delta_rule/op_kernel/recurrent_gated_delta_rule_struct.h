/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule_struct.h
 * \brief Plain tiling struct shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECURRENT_GATED_DELTA_RULE_STRUCT_H
#define RECURRENT_GATED_DELTA_RULE_STRUCT_H

#include <cstdint>

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace RecurrentGatedDeltaRule {

#define RGDR_TPL_BF16 10
#define RGDR_TPL_FP32 30

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(RecurrentGatedDeltaRule, ASCENDC_TPL_DTYPE_DECL(D_T_STATE, RGDR_TPL_BF16, RGDR_TPL_FP32),
                      ASCENDC_TPL_DTYPE_DECL(D_T_G, RGDR_TPL_BF16, RGDR_TPL_FP32),
                      ASCENDC_TPL_DTYPE_DECL(D_T_BETA, RGDR_TPL_BF16, RGDR_TPL_FP32),
                      ASCENDC_TPL_BOOL_DECL(USE_QK_L2NORM, 0, 1), ASCENDC_TPL_BOOL_DECL(USE_GATE_IN_KERNEL, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(USE_BETA_SIGMOID, 0, 1), ASCENDC_TPL_BOOL_DECL(ALLOW_NEG_EIGVAL, 0, 1), );

#define RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, USE_QK_L2NORM_VALUE, USE_GATE_IN_KERNEL_VALUE,              \
                            USE_BETA_SIGMOID_VALUE, ALLOW_NEG_EIGVAL_VALUE)                                            \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_STATE, STATE_TYPE), ASCENDC_TPL_DTYPE_SEL(D_T_G, G_TYPE),           \
                         ASCENDC_TPL_DTYPE_SEL(D_T_BETA, BETA_TYPE),                                                   \
                         ASCENDC_TPL_BOOL_SEL(USE_QK_L2NORM, USE_QK_L2NORM_VALUE),                                     \
                         ASCENDC_TPL_BOOL_SEL(USE_GATE_IN_KERNEL, USE_GATE_IN_KERNEL_VALUE),                           \
                         ASCENDC_TPL_BOOL_SEL(USE_BETA_SIGMOID, USE_BETA_SIGMOID_VALUE),                               \
                         ASCENDC_TPL_BOOL_SEL(ALLOW_NEG_EIGVAL, ALLOW_NEG_EIGVAL_VALUE), )

#define RGDR_TPL_SEL_FLAGS(STATE_TYPE, G_TYPE, BETA_TYPE)                                                           \
    RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 0, 0, 0),                                                    \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 0, 0, 1),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 0, 1, 0),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 0, 1, 1),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 1, 0, 0),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 1, 0, 1),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 1, 1, 0),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 0, 1, 1, 1),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 0, 0, 0),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 0, 0, 1),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 0, 1, 0),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 0, 1, 1),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 1, 0, 0),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 1, 0, 1),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 1, 1, 0),                                                \
        RGDR_TPL_SEL_ONE(STATE_TYPE, G_TYPE, BETA_TYPE, 1, 1, 1, 1)

#define RGDR_TPL_SEL_SCALARS(STATE_TYPE)                                                                            \
    RGDR_TPL_SEL_FLAGS(STATE_TYPE, RGDR_TPL_BF16, RGDR_TPL_BF16),                                             \
        RGDR_TPL_SEL_FLAGS(STATE_TYPE, RGDR_TPL_BF16, RGDR_TPL_FP32),                                         \
        RGDR_TPL_SEL_FLAGS(STATE_TYPE, RGDR_TPL_FP32, RGDR_TPL_BF16),                                         \
        RGDR_TPL_SEL_FLAGS(STATE_TYPE, RGDR_TPL_FP32, RGDR_TPL_FP32)

ASCENDC_TPL_SEL(RGDR_TPL_SEL_SCALARS(RGDR_TPL_BF16), RGDR_TPL_SEL_SCALARS(RGDR_TPL_FP32), );

#undef RGDR_TPL_SEL_SCALARS
#undef RGDR_TPL_SEL_FLAGS
#undef RGDR_TPL_SEL_ONE

#endif

#pragma pack(push, 8)
struct alignas(8) RecurrentGatedDeltaRuleTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t t;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t b;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
    uint32_t hasGama;
    uint32_t hasAcceptedTokens;
    uint32_t stateStride0;
 	uint32_t stateStride1;
 	uint32_t stateStride2;
    uint64_t queryStrides[3] = {};
    uint64_t keyStrides[3] = {};
    uint64_t valueStrides[3] = {};
    uint64_t stateStrides[4] = {};
    uint32_t stateIndicesWidth = 0;
    uint32_t useQkL2NormInKernel = 0;
    uint32_t useGateInKernel = 0;
    uint32_t useBetaSigmoidInKernel = 0;
    uint32_t allowNegEigval = 0;
    uint32_t hasDtBias = 0;
    uint32_t hasALog = 0;
    uint32_t betaDtype = RGDR_TPL_BF16;
    uint32_t gDtype = RGDR_TPL_FP32;
    uint64_t taskCount = 0;
    uint64_t totalTokens = 0;
    uint64_t keyHeads = 0;
    uint64_t valueHeads = 0;
    uint64_t stateSlots = 0;
    uint64_t batchSize = 0;
    uint64_t checkpointWidth = 0;
};
#pragma pack(pop)

} // namespace RecurrentGatedDeltaRule

#endif // RECURRENT_GATED_DELTA_RULE_STRUCT_H
