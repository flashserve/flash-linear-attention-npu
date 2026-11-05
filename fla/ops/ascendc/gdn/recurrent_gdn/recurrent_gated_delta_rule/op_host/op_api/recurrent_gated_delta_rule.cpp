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
 * \file recurrent_gated_delta_rule.cpp
 * \brief
 */
#include "../recurrent_gated_delta_rule_error.h"
#include "recurrent_gated_delta_rule.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(RecurrentGatedDeltaRule);

const aclTensor *RecurrentGatedDeltaRule(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                         const aclTensor *beta, aclTensor *stateRef, const aclTensor *actualSeqLengths,
                                         const aclTensor *ssmStateIndices, const aclTensor *g, const aclTensor *gk,
                                         const aclTensor *numAcceptedTokens, float scaleValue, aclOpExecutor *executor)
{
    L0_DFX(RecurrentGatedDeltaRule, query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk,
           numAcceptedTokens, scaleValue);

    DataType outType = DataType::DT_BF16;
    Format format = Format::FORMAT_ND;
    auto out = executor->AllocTensor(outType, format, format);
    if (!(out != nullptr)) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: output allocation failed"); return nullptr; };

    const auto stateStorageShape = stateRef->GetStorageShape();
    const auto stateViewShape = stateRef->GetViewShape();
    const auto stateViewStrides = stateRef->GetViewStrides();
    const auto stateViewOffset = stateRef->GetViewOffset();

    auto ret = INFER_SHAPE(
        RecurrentGatedDeltaRule,
        OP_INPUT(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk, numAcceptedTokens,
                 static_cast<const aclTensor *>(nullptr), static_cast<const aclTensor *>(nullptr)),
        OP_OUTPUT(out, stateRef), OP_ATTR(scaleValue));
    if (ret != ACLNN_SUCCESS) { return nullptr; };

    stateRef->SetViewShape(stateViewShape);
    stateRef->SetViewStrides(stateViewStrides);
    stateRef->SetViewOffset(stateViewOffset);
    stateRef->SetStorageShape(stateStorageShape);

    // The shared kernel has two additional optional input slots. Preserve the
    // historical API while supplying their positions in the launch argument list.
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        RecurrentGatedDeltaRule,
        OP_INPUT(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk, numAcceptedTokens,
                 static_cast<const aclTensor *>(nullptr), static_cast<const aclTensor *>(nullptr)),
        OP_OUTPUT(out, stateRef), OP_ATTR(scaleValue));
    if (ret != ACLNN_SUCCESS) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: failed to create the AICore launcher"); return nullptr; };
    return out;
}

const aclTensor *RecurrentGatedDeltaRuleV2(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                           const aclTensor *beta, aclTensor *stateRef, const aclTensor *actualSeqLengths,
                                           const aclTensor *ssmStateIndices, const aclTensor *g, const aclTensor *gk,
                                           const aclTensor *numAcceptedTokens, const aclTensor *aLog, const aclTensor *dtBias,
                                           float scaleValue, bool useQkL2NormInKernel, bool useGateInKernel,
                                           bool useBetaSigmoidInKernel, bool allowNegEigval, aclOpExecutor *executor)
{
    L0_DFX(RecurrentGatedDeltaRuleV2, query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk,
           numAcceptedTokens, aLog, dtBias, scaleValue, useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval);

    DataType outType = DataType::DT_BF16;
    Format format = Format::FORMAT_ND;

    auto out = executor->AllocTensor(outType, format, format);

    if (!(out != nullptr)) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: output allocation failed"); return nullptr; };

    // State is an existing strided allocation, rather than a newly allocated
    // output. InferShape normalizes every output descriptor to its logical
    // shape and contiguous strides; preserve the caller's physical storage
    // extent and view metadata before it updates this aliased output.
    const auto stateStorageShape = stateRef->GetStorageShape();
    const auto stateViewShape = stateRef->GetViewShape();
    const auto stateViewStrides = stateRef->GetViewStrides();
    const auto stateViewOffset = stateRef->GetViewOffset();

    // infershape
    auto ret = INFER_SHAPE(
        RecurrentGatedDeltaRule,
        OP_INPUT(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk, numAcceptedTokens, aLog, dtBias),
        OP_OUTPUT(out, stateRef), OP_ATTR(scaleValue, useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, true));
    if (ret != ACLNN_SUCCESS) { return nullptr; };

    // SetViewShape also resets strides, so restore strides after the shape.
    // SetStorageShape restores both the actual backing extent and its cached
    // byte size, which the launcher and memory checker use for GM bounds.
    stateRef->SetViewShape(stateViewShape);
    stateRef->SetViewStrides(stateViewStrides);
    stateRef->SetViewOffset(stateViewOffset);
    stateRef->SetStorageShape(stateStorageShape);

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        RecurrentGatedDeltaRule,
        OP_INPUT(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk, numAcceptedTokens, aLog, dtBias),
        OP_OUTPUT(out, stateRef), OP_ATTR(scaleValue, useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, true));
    if (ret != ACLNN_SUCCESS) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: failed to create the AICore launcher"); return nullptr; };

    return out;
}
} // namespace l0op
