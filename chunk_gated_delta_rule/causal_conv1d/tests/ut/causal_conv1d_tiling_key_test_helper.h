/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAUSAL_CONV1D_TILING_KEY_TEST_HELPER_H
#define CAUSAL_CONV1D_TILING_KEY_TEST_HELPER_H

#include "../../op_kernel/causal_conv1d_tiling_key.h"

inline constexpr uint32_t NormalizeFnPlanTilingKey(uint32_t runModeKey, FnExecutionPlan fnExecutionPlan)
{
    if (runModeKey != CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
        return CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
    }
    switch (fnExecutionPlan) {
        case FN_EXECUTION_PLAN_CUTBS:
            return CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS;
        case FN_EXECUTION_PLAN_CUTBSD:
            return CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD;
        default:
            return CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
    }
}

inline constexpr uint32_t NormalizeWidthTilingKey(uint32_t runModeKey, int32_t width)
{
    if (runModeKey != CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
        return CAUSAL_CONV1D_TPL_WIDTH_RUNTIME;
    }
    switch (width) {
        case 2:
            return CAUSAL_CONV1D_TPL_WIDTH_2;
        case 3:
            return CAUSAL_CONV1D_TPL_WIDTH_3;
        case 4:
            return CAUSAL_CONV1D_TPL_WIDTH_4;
        default:
            return CAUSAL_CONV1D_TPL_WIDTH_RUNTIME;
    }
}

inline uint64_t BuildCausalConv1dTilingKey(uint32_t runModeKey, FnExecutionPlan fnExecutionPlan, int32_t width = 4,
                                           int64_t activationMode = 0, bool hasBias = true)
{
    (void)activationMode;
    (void)hasBias;
    const uint32_t fnPlanKey = NormalizeFnPlanTilingKey(runModeKey, fnExecutionPlan);
    const uint32_t widthKey = NormalizeWidthTilingKey(runModeKey, width);
    static_assert(CAUSAL_CONV1D_TPL_RUN_MODE_FN == 0 && CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE == 1,
                  "test helper assumes runModeKey values match tiling-key indices");
    static_assert(CAUSAL_CONV1D_TPL_WIDTH_RUNTIME == 0 && CAUSAL_CONV1D_TPL_WIDTH_2 == 1 &&
                      CAUSAL_CONV1D_TPL_WIDTH_3 == 2 && CAUSAL_CONV1D_TPL_WIDTH_4 == 3,
                  "test helper assumes widthKey values match tiling-key indices");
    static_assert(CAUSAL_CONV1D_TPL_FN_PLAN_INVALID == 0 && CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS == 1 &&
                      CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD == 2,
                  "test helper assumes fnPlanKey values match tiling-key indices");
    return (static_cast<uint64_t>(fnPlanKey) << 3) | (static_cast<uint64_t>(widthKey) << 1) |
           static_cast<uint64_t>(runModeKey);
}

#endif // CAUSAL_CONV1D_TILING_KEY_TEST_HELPER_H
