/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Adapted for flash-linear-attention-npu by Tianjin University.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda.cpp
 * \brief
 */
#include "recurrent_kda.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(RecurrentKda);

const std::array<const aclTensor *, 2> RecurrentKda(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    aclTensor *stateRef,
    const aclTensor *cuSeqlens,
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
    aclOpExecutor *executor)
{
    L0_DFX(RecurrentKda, query, key, value, gate, beta, stateRef, cuSeqlens, ssmStateIndicesOptional,
           aLogOptional, dtBiasOptional, numAcceptedTokensOptional, layout, scale, useQkL2normInKernel,
           useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound, stateVFirst, out,
           stateRef);

    float scaleAttr = static_cast<float>(scale);
    float lowerBoundAttr = static_cast<float>(lowerBound);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        RecurrentKda,
        OP_INPUT(query, key, value, gate, beta, stateRef, cuSeqlens, ssmStateIndicesOptional, aLogOptional,
                 dtBiasOptional, numAcceptedTokensOptional),
        OP_OUTPUT(out, stateRef),
        OP_ATTR(layout, scaleAttr, useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval,
                safeGate, lowerBoundAttr, stateVFirst));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "RecurrentKda ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }

    return {out, stateRef};
}
} // namespace l0op
