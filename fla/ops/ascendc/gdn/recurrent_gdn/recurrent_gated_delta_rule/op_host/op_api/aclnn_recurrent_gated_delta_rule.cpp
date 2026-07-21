/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file aclnn_recurrent_gated_delta_rule.cpp
 * \brief
 */
#include <cmath>
#include <dlfcn.h>
#include "aclnn_recurrent_gated_delta_rule.h"
#include "recurrent_gated_delta_rule.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t QUERY_DIM_NUM = 3;
constexpr size_t KEY_DIM_NUM = 3;
constexpr size_t VALUE_DIM_NUM = 3;
constexpr size_t BETA_DIM_NUM = 2;
constexpr size_t STATE_DIM_NUM = 4;

struct RecurrentGatedDeltaRuleParams {
    // madatory
    const aclTensor *query {nullptr};
    const aclTensor *key {nullptr};
    const aclTensor *value {nullptr};
    const aclTensor *beta {nullptr};
    const aclTensor *state {nullptr};
    const aclTensor *actual_seq_lengths {nullptr};
    const aclTensor *ssm_state_indices {nullptr};
    // optional
    const aclTensor *g {nullptr};
    const aclTensor *gk {nullptr};
    const aclTensor *num_accepted_tokens {nullptr};
    // attrs
    float scale {1.0f};
    //output
    const aclTensor *out {nullptr};
};

// support dtype
static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16,op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BETA_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> SEQ_LENS_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> SSM_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> G_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> ACC_TO_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};

static inline bool CheckNotNull(const RecurrentGatedDeltaRuleParams &params)
{
    // 必选参数
    OP_CHECK_NULL(params.query, return false);
    OP_CHECK_NULL(params.key, return false);
    OP_CHECK_NULL(params.value, return false);
    OP_CHECK_NULL(params.state, return false);
    OP_CHECK_NULL(params.beta, return false);
    OP_CHECK_NULL(params.actual_seq_lengths, return false);
    OP_CHECK_NULL(params.ssm_state_indices, return false);
    OP_CHECK_NULL(params.out, return false);

    return true;
}

static inline bool CheckDtypeVaild(const RecurrentGatedDeltaRuleParams &params)
{
    // 检查必选参数数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.state, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, BETA_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.actual_seq_lengths, SEQ_LENS_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.ssm_state_indices, SSM_TYPE_SUPPORT_LIST, return false);

    // 检查可选参数数据类型
    if (params.g != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.g, G_TYPE_SUPPORT_LIST, return false);
    }
    if (params.gk != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.gk, G_TYPE_SUPPORT_LIST, return false);
    }
    if (params.num_accepted_tokens != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.num_accepted_tokens, ACC_TO_TYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    return true;
}

static inline bool CheckShapeValid(const RecurrentGatedDeltaRuleParams &params)
{
    const auto queryShape = params.query->GetViewShape();
    const auto keyShape = params.key->GetViewShape();
    const auto valueShape = params.value->GetViewShape();
    const auto betaShape = params.beta->GetViewShape();
    const auto stateShape = params.state->GetViewShape();
    const auto seqShape = params.actual_seq_lengths->GetViewShape();
    const auto indexShape = params.ssm_state_indices->GetViewShape();
    const auto outShape = params.out->GetViewShape();
    CHECK_COND(queryShape.GetDimNum() == QUERY_DIM_NUM && keyShape.GetDimNum() == KEY_DIM_NUM &&
                   valueShape.GetDimNum() == VALUE_DIM_NUM && betaShape.GetDimNum() == BETA_DIM_NUM &&
                   stateShape.GetDimNum() == STATE_DIM_NUM,
               false, "query/key/value must be rank-3, beta rank-2 and state rank-4.");
    CHECK_COND(seqShape.GetDimNum() == 1 && seqShape.GetDim(0) >= 2, false,
               "actual_seq_lengths must be rank-1 with B+1 entries.");
    CHECK_COND(indexShape.GetDimNum() == 1 && indexShape.GetDim(0) == queryShape.GetDim(0), false,
               "ssm_state_indices must be rank-1 with T entries.");
    CHECK_COND(queryShape.GetDim(0) == keyShape.GetDim(0) && queryShape.GetDim(1) == keyShape.GetDim(1) &&
                   queryShape.GetDim(2) == keyShape.GetDim(2),
               false, "key must match query shape [T, HK, K].");
    CHECK_COND(valueShape.GetDim(0) == queryShape.GetDim(0) && betaShape.GetDim(0) == queryShape.GetDim(0) &&
                   betaShape.GetDim(1) == valueShape.GetDim(1),
               false, "value must be [T, HV, V] and beta must be [T, HV].");
    CHECK_COND(queryShape.GetDim(1) > 0 && valueShape.GetDim(1) > 0 &&
                   valueShape.GetDim(1) % queryShape.GetDim(1) == 0,
               false, "GVA requires HV divisible by HK.");
    CHECK_COND(queryShape.GetDim(0) > 0 && queryShape.GetDim(2) > 0 && valueShape.GetDim(2) > 0,
               false, "T, K and V must be positive.");
    CHECK_COND(stateShape.GetDim(0) > 0 && stateShape.GetDim(1) == valueShape.GetDim(1) &&
                   stateShape.GetDim(2) == valueShape.GetDim(2) && stateShape.GetDim(3) == queryShape.GetDim(2),
               false, "stateRef must be [D_s, HV, V, K] with D_s positive.");
    CHECK_COND(queryShape.GetDim(1) <= 256 && valueShape.GetDim(1) <= 256 &&
                   queryShape.GetDim(2) <= 512 && valueShape.GetDim(2) <= 512,
               false, "HK/HV must be <= 256 and K/V must be <= 512.");
    CHECK_COND(outShape.GetDimNum() == VALUE_DIM_NUM && outShape.GetDim(0) == valueShape.GetDim(0) &&
                   outShape.GetDim(1) == valueShape.GetDim(1) && outShape.GetDim(2) == valueShape.GetDim(2),
               false, "out must match value shape [T, HV, V].");
    if (params.g != nullptr) {
        const auto gShape = params.g->GetViewShape();
        CHECK_COND(gShape.GetDimNum() == 2 && gShape.GetDim(0) == valueShape.GetDim(0) &&
                       gShape.GetDim(1) == valueShape.GetDim(1),
                   false, "g must be [T, HV].");
    }
    if (params.gk != nullptr) {
        const auto gkShape = params.gk->GetViewShape();
        CHECK_COND(gkShape.GetDimNum() == 3 && gkShape.GetDim(0) == valueShape.GetDim(0) &&
                       gkShape.GetDim(1) == valueShape.GetDim(1) && gkShape.GetDim(2) == queryShape.GetDim(2),
                   false, "gk must be [T, HV, K].");
    }
    if (params.num_accepted_tokens != nullptr) {
        const auto acceptedShape = params.num_accepted_tokens->GetViewShape();
        CHECK_COND(acceptedShape.GetDimNum() == 1 && acceptedShape.GetDim(0) == seqShape.GetDim(0) - 1,
                   false, "num_accepted_tokens must be rank-1 with B entries.");
    }
    return true;
}

static aclnnStatus CheckParams(RecurrentGatedDeltaRuleParams &params)
{
    // 检查输入参数是否在支持的数据类型范围内
    CHECK_RET(CheckDtypeVaild(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(std::isfinite(params.scale), ACLNN_ERR_PARAM_INVALID, "scaleValue must be finite.");

    OP_LOGD("RecurrentGatedDeltaRule check params sucess.");

    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(RecurrentGatedDeltaRuleParams &params)
{
    params.query->SetOriginalShape(params.query->GetViewShape());
    params.key->SetOriginalShape(params.key->GetViewShape());
    params.value->SetOriginalShape(params.value->GetViewShape());
    params.beta->SetOriginalShape(params.beta->GetViewShape());
    params.state->SetOriginalShape(params.state->GetViewShape());
    params.actual_seq_lengths->SetOriginalShape(params.actual_seq_lengths->GetViewShape());
    params.ssm_state_indices->SetOriginalShape(params.ssm_state_indices->GetViewShape());

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(const aclTensor *query, const aclTensor *key,
                                                         const aclTensor *value, const aclTensor *beta,
                                                         aclTensor *stateRef, const aclTensor *actualSeqLengths,
                                                         const aclTensor *ssmStateIndices, const aclTensor *g,
                                                         const aclTensor *gk, const aclTensor *numAcceptedTokens,
                                                         float scaleValue, aclTensor *out, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    L2_DFX_PHASE_1(aclnnRecurrentGatedDeltaRule,
                   DFX_IN(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk,
                          numAcceptedTokens, scaleValue),
                   DFX_OUT(out, stateRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    RecurrentGatedDeltaRuleParams params {query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices,
                                          g, gk, numAcceptedTokens, scaleValue, out};

    CHECK_COND(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR, "required inputs and out must not be nullptr.");
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
    auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
    auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
    auto beta_ = l0op::Contiguous(beta, uniqueExecutor.get());
    auto actualSeqLengths_ = l0op::Contiguous(actualSeqLengths, uniqueExecutor.get());
    auto ssmStateIndices_ = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
    if (g != nullptr) {
        g = l0op::Contiguous(g, uniqueExecutor.get());
    }
    if (gk != nullptr) {
        gk = l0op::Contiguous(gk, uniqueExecutor.get());
    }
    if (numAcceptedTokens != nullptr) {
        numAcceptedTokens = l0op::Contiguous(numAcceptedTokens, uniqueExecutor.get());
    }
    if (!IsContiguous(stateRef)) {
        stateRef = uniqueExecutor.get()->CreateView(
            stateRef,
            stateRef->GetViewShape(),
            stateRef->GetStorageShape(),
            stateRef->GetViewStrides(),
            stateRef->GetViewOffset());
    }

    auto out_ = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(query_ != nullptr && key_ != nullptr && value_ != nullptr && beta_ != nullptr &&
                  actualSeqLengths_ != nullptr && ssmStateIndices_ != nullptr && out_ != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    // 调用l0接口
    auto outRet =
        l0op::RecurrentGatedDeltaRule(query_, key_, value_, beta_, stateRef, actualSeqLengths_, ssmStateIndices_, g, gk,
                                      numAcceptedTokens, scaleValue, uniqueExecutor.get());
    if (outRet == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ViewCopyResult = l0op::ViewCopy(outRet, out_, uniqueExecutor.get());
    if (ViewCopyResult == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // 获取计算过程中需要使用的workspace大小。
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecurrentGatedDeltaRule(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentGatedDeltaRule);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
