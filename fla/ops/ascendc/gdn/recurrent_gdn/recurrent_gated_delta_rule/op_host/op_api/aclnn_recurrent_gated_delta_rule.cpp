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
#include "../recurrent_gated_delta_rule_error.h"
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
#include "recurrent_gated_delta_rule_check.h"

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
    if (params.query == nullptr) { OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "query"); return false; };
    if (params.key == nullptr) { OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "key"); return false; };
    if (params.value == nullptr) { OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "value"); return false; };
    if (params.state == nullptr) { OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "state"); return false; };
    if (params.beta == nullptr) { OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "beta"); return false; };
    if (params.actual_seq_lengths == nullptr) { OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "actual_seq_lengths"); return false; };
    if (params.ssm_state_indices == nullptr) { OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "ssm_state_indices"); return false; };
    if (params.out == nullptr) { OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "out"); return false; };

    return true;
}

static inline bool CheckDtypeVaild(const RecurrentGatedDeltaRuleParams &params)
{
    // 检查必选参数数据类型
    if (!rgdn_check::Type(params.query, QKV_TYPE_SUPPORT_LIST, "query", op::ToString(QKV_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };
    if (!rgdn_check::Type(params.key, QKV_TYPE_SUPPORT_LIST, "key", op::ToString(QKV_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };
    if (!rgdn_check::Type(params.value, QKV_TYPE_SUPPORT_LIST, "value", op::ToString(QKV_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };
    if (!rgdn_check::Type(params.state, STATE_TYPE_SUPPORT_LIST, "state", op::ToString(STATE_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };
    if (!rgdn_check::Type(params.beta, BETA_TYPE_SUPPORT_LIST, "beta", op::ToString(BETA_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };
    if (!rgdn_check::Type(params.actual_seq_lengths, SEQ_LENS_TYPE_SUPPORT_LIST, "actual_seq_lengths", op::ToString(SEQ_LENS_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };
    if (!rgdn_check::Type(params.ssm_state_indices, SSM_TYPE_SUPPORT_LIST, "ssm_state_indices", op::ToString(SSM_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };

    // 检查可选参数数据类型
    if (params.g != nullptr) {
        if (!rgdn_check::Type(params.g, G_TYPE_SUPPORT_LIST, "g", op::ToString(G_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };
    }
    if (params.num_accepted_tokens != nullptr) {
        if (!rgdn_check::Type(params.num_accepted_tokens, ACC_TO_TYPE_SUPPORT_LIST, "num_accepted_tokens", op::ToString(ACC_TO_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };
    }

    if (!rgdn_check::Type(params.out, OUT_TYPE_SUPPORT_LIST, "out", op::ToString(OUT_TYPE_SUPPORT_LIST).GetString(), "aclnnRecurrentGatedDeltaRule")) { return false; };
    return true;
}

static aclnnStatus CheckParams(RecurrentGatedDeltaRuleParams &params)
{
    if (params.gk != nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            "aclnnRecurrentGatedDeltaRule", "gk", "gk is not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 检查输入参数是否在支持的数据类型范围内
    if (!(CheckDtypeVaild(params))) { return ACLNN_ERR_PARAM_INVALID; };

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
    // Validate before DFX: its phase-1 cache can return without executing
    // the function body, and its generic null guard loses the parameter name.
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRule", "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(aclnnRecurrentGatedDeltaRule,
                   DFX_IN(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk,
                          numAcceptedTokens, scaleValue),
                   DFX_OUT(stateRef, out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: CREATE_EXECUTOR failed"); return ACLNN_ERR_INNER_CREATE_EXECUTOR; }

    RecurrentGatedDeltaRuleParams params {query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk, numAcceptedTokens,scaleValue, out};

    if (!(CheckNotNull(params))) { return ACLNN_ERR_PARAM_INVALID; };
    if (!(CheckParams(params) == ACLNN_SUCCESS)) { return ACLNN_ERR_PARAM_INVALID; };
    auto ret = PreProcess(params);
    if (!(ret == ACLNN_SUCCESS)) { return ret; };

    auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
    auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
    auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
    auto beta_ = l0op::Contiguous(beta, uniqueExecutor.get());
    auto actualSeqLengths_ = l0op::Contiguous(actualSeqLengths, uniqueExecutor.get());
    auto ssmStateIndices_ = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
    if (g != nullptr) {
        g = l0op::Contiguous(g, uniqueExecutor.get());
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

    // 调用l0接口
    auto outRet = l0op::RecurrentGatedDeltaRule(query_, key_, value_, beta_, stateRef, actualSeqLengths_,
                                                ssmStateIndices_, g, gk, numAcceptedTokens, scaleValue,
                                                uniqueExecutor.get());
    if (outRet == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ViewCopyResult = l0op::ViewCopy(outRet, out_, uniqueExecutor.get());
    if (ViewCopyResult == nullptr) {
        REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: output ViewCopy failed");
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

aclnnStatus aclnnRecurrentGatedDeltaRuleV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
    aclTensor *stateRef, const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices,
    const aclTensor *g, const aclTensor *gk, const aclTensor *numAcceptedTokens,
    const aclTensor *aLog, const aclTensor *dtBias, float scaleValue,
    bool useQkL2NormInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel,
    bool allowNegEigval, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // Validate before DFX: its phase-1 cache can return without executing
    // the function body, and its generic null guard loses the parameter name.
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRuleV2", "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentGatedDeltaRuleV2", "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const rgdn_check::Params params{query, key, value, beta, stateRef, actualSeqLengths,
        ssmStateIndices, g, gk, numAcceptedTokens, aLog, dtBias, scaleValue,
        useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, out};
    if (!(rgdn_check::Check(params))) { return ACLNN_ERR_PARAM_INVALID; };

    L2_DFX_PHASE_1(aclnnRecurrentGatedDeltaRuleV2,
                  DFX_IN(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices,
                         g, gk, numAcceptedTokens, aLog, dtBias, scaleValue,
                         useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval),
                  DFX_OUT(stateRef, out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: CREATE_EXECUTOR failed"); return ACLNN_ERR_INNER_CREATE_EXECUTOR; }
    for (auto tensor : {query, key, value, beta, static_cast<const aclTensor *>(stateRef),
                        actualSeqLengths, ssmStateIndices, g, numAcceptedTokens, aLog, dtBias,
                        static_cast<const aclTensor *>(out)}) {
        if (tensor) tensor->SetOriginalShape(tensor->GetViewShape());
    }

    // Q/K/V and the writable State are genuine strided views. Never silently
    // materialize them: both their storage offset and their strides are ABI data.
    const auto preserveView = [&](const aclTensor *tensor) -> const aclTensor * {
        return uniqueExecutor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                          tensor->GetViewStrides(), tensor->GetViewOffset());
    };
    const auto queryView = preserveView(query);
    const auto keyView = preserveView(key);
    const auto valueView = preserveView(value);
    auto stateView = uniqueExecutor->CreateView(stateRef, stateRef->GetViewShape(), stateRef->GetStorageShape(),
                                               stateRef->GetViewStrides(), stateRef->GetViewOffset());
    if (!(queryView && keyView && valueView && stateView)) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRuleV2: failed to preserve query/key/value/state views"); return ACLNN_ERR_INNER_NULLPTR; }

    const auto contiguous = [&](const aclTensor *tensor) -> const aclTensor * {
        return tensor ? l0op::Contiguous(tensor, uniqueExecutor.get()) : nullptr;
    };
    const auto betaContiguous = contiguous(beta);
    const auto lengthsContiguous = contiguous(actualSeqLengths);
    const auto indicesContiguous = contiguous(ssmStateIndices);
    const auto gContiguous = contiguous(g);
    const auto acceptedContiguous = contiguous(numAcceptedTokens);
    const auto aLogContiguous = useGateInKernel ? contiguous(aLog) : nullptr;
    const auto biasContiguous = useGateInKernel ? contiguous(dtBias) : nullptr;
    if (!(betaContiguous && lengthsContiguous && indicesContiguous && (!g || gContiguous) &&
              (!numAcceptedTokens || acceptedContiguous) && (!useGateInKernel || aLogContiguous) &&
              (!useGateInKernel || !dtBias || biasContiguous))) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRuleV2: input Contiguous failed"); return ACLNN_ERR_INNER_NULLPTR; }
    const auto outRet = l0op::RecurrentGatedDeltaRuleV2(
        queryView, keyView, valueView, betaContiguous, stateView, lengthsContiguous, indicesContiguous,
        gContiguous, nullptr, acceptedContiguous, aLogContiguous, biasContiguous, scaleValue,
        useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, uniqueExecutor.get());
    if (!(outRet != nullptr)) { return ACLNN_ERR_INNER_NULLPTR; };
    if (l0op::ViewCopy(outRet, out, uniqueExecutor.get()) == nullptr) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRuleV2: output ViewCopy failed"); return ACLNN_ERR_INNER_NULLPTR; }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecurrentGatedDeltaRuleV2(void *workspace, uint64_t workspaceSize,
                                           aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentGatedDeltaRuleV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
