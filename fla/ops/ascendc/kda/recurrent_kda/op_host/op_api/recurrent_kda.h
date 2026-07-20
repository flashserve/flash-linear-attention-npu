/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Adapted for flash-linear-attention-npu by Tianjin University.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_RECURRENT_KDA
#define PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_RECURRENT_KDA

#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include <array>

namespace l0op {
const std::array<const aclTensor *, 2> RecurrentKda(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    const aclTensor *initialState,
    const aclIntArray *cuSeqlensOptional,
    const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *numAcceptedTokensOptional,
    const char *layout,
    double scale,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool stateVFirst,
    const aclTensor *out,
    const aclTensor *finalState,
    aclOpExecutor *executor);
}

#endif // PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_RECURRENT_KDA
