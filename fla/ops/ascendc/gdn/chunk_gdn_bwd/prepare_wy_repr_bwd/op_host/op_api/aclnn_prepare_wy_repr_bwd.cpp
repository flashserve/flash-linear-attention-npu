/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

// Load structured reporters before opdev's legacy logging macros. The opdev
// headers must retain their logging convention for CHECK_RET and DFX helpers.
#include "op_common/log/log.h"
#include "aclnn_prepare_wy_repr_bwd.h"
#include "prepare_wy_repr_bwd.h"

#include <dlfcn.h>
#include <new>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static constexpr const char *OP_NAME = "aclnnPrepareWyReprBwd";

#ifdef __cplusplus
extern "C" {
#endif

struct PrepareWyReprBwdParams {
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *a = nullptr;
    const aclTensor *dw = nullptr;
    const aclTensor *du = nullptr;
    const aclTensor *g = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    int64_t chunkSize = 64;
    const aclTensor *dkOut = nullptr;
    const aclTensor *dvOut = nullptr;
    const aclTensor *dbetaOut = nullptr;
    const aclTensor *dgOut = nullptr;
};

static aclnnStatus CheckNotNull(PrepareWyReprBwdParams params)
{
    if (params.k == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.v == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "v");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.beta == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "beta");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.a == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "a");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dw == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dw");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.du == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "du");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.g == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dkOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dkOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dvOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dvOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dbetaOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dbetaOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dgOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dgOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(PrepareWyReprBwdParams params)
{
    (void)params;
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(PrepareWyReprBwdParams params)
{
    (void)params;
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(PrepareWyReprBwdParams params)
{
    (void)params;
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(PrepareWyReprBwdParams params)
{
    {
        const aclnnStatus checkRet = CheckNotNull(params);
        if (checkRet != ACLNN_SUCCESS) {
            return checkRet;
        }
    }
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(PrepareWyReprBwdParams &params, aclOpExecutor *executor)
{
    if (DataContiguous(params.k, executor) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous k failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.v, executor) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous v failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.beta, executor) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous beta failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.a, executor) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous a failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.dw, executor) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous dw failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.du, executor) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous du failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.g, executor) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous g failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPrepareWyReprBwdGetWorkspaceSize(
    const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *a, const aclTensor *dw,
    const aclTensor *du, const aclTensor *g, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, int64_t chunkSize, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    PrepareWyReprBwdParams params{k, v, beta, a, dw, du, g, cuSeqlensOptional, chunkIndicesOptional, chunkSize,
                                  dkOut, dvOut, dbetaOut, dgOut};
    L2_DFX_PHASE_1_WITHOUT_CACHE(
        aclnnPrepareWyReprBwd,
        DFX_IN(k, v, beta, a, dw, du, g, cuSeqlensOptional, chunkIndicesOptional, chunkSize),
        DFX_OUT(dkOut, dvOut, dbetaOut, dgOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    OP_LOGW("npu_prepare_wy_repr_bwd",
            "npu_prepare_wy_repr_bwd is deprecated and will be removed on "
            "January 15, 2027 (v26.12.0); use the fused "
            "npu_chunk_gated_delta_rule_bwd instead.");

    {
        const aclnnStatus checkRet = CheckParams(params);
        if (checkRet != ACLNN_SUCCESS) {
            return checkRet;
        }
    }
    if (ParamsDataContiguous(params, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: input contiguous conversion failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto result = l0op::PrepareWyReprBwd(params.k, params.v, params.beta, params.a, params.dw, params.du, params.g,
                                         params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize,
                                         params.dkOut, params.dvOut, params.dbetaOut, params.dgOut, executorPtr);
    if (result[0] == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: operator returned no output", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclTensor *tailOutputs[] = {result[1], result[2], result[3]};
    const char *tailOutputNames[] = {"dv", "dbeta", "dg"};
    for (size_t i = 0; i < sizeof(tailOutputs) / sizeof(tailOutputs[0]); ++i) {
        if (tailOutputs[i] == nullptr) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: operator returned null output: %s", __func__,
                                   tailOutputNames[i]);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    auto viewCopyResult = l0op::ViewCopy(result[0], params.dkOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult is nullptr", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    viewCopyResult = l0op::ViewCopy(result[1], params.dvOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult is nullptr", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    viewCopyResult = l0op::ViewCopy(result[2], params.dbetaOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult is nullptr", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    viewCopyResult = l0op::ViewCopy(result[3], params.dgOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult is nullptr", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPrepareWyReprBwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPrepareWyReprBwd);
    const aclnnStatus runRet = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (runRet != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: operator launch failed, rc=%d", __func__,
                               static_cast<int>(runRet));
        return runRet;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
