/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "host_utils/log.h"
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

static constexpr const char *DFX_ENTITY = "AclnnPrepareWyReprBwd";

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
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(params.a != nullptr, ACLNN_ERR_PARAM_NULLPTR, "a must not be nullptr.");
    CHECK_COND(params.dw != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dw must not be nullptr.");
    CHECK_COND(params.du != nullptr, ACLNN_ERR_PARAM_NULLPTR, "du must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(params.dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dkOut must not be nullptr.");
    CHECK_COND(params.dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dvOut must not be nullptr.");
    CHECK_COND(params.dbetaOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dbetaOut must not be nullptr.");
    CHECK_COND(params.dgOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dgOut must not be nullptr.");
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
    if (!(CheckNotNull(params) == ACLNN_SUCCESS)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_WITH_REASON_COMPAT(DFX_ENTITY, "k,v,beta,a,dw,du,g,dkOut,dvOut,dbetaOut,dgOut", "CheckNotNull(params) == ACLNN_SUCCESS", "documented constraint");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(CheckFormat(params) == ACLNN_SUCCESS)) {
        OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(DFX_ENTITY, "format", "CheckFormat(params) == ACLNN_SUCCESS", "documented format");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(CheckShape(params) == ACLNN_SUCCESS)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(DFX_ENTITY, "CheckShape inputs", "CheckShape(params) == ACLNN_SUCCESS", "shape must match the documented layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(CheckDtype(params) == ACLNN_SUCCESS)) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(DFX_ENTITY, "dtype", "CheckDtype(params) == ACLNN_SUCCESS", "supported dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(DFX_ENTITY, "tensor");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(PrepareWyReprBwdParams &params, aclOpExecutor *executor)
{
    CHECK_COND(DataContiguous(params.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous v failed.");
    CHECK_COND(DataContiguous(params.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous beta failed.");
    CHECK_COND(DataContiguous(params.a, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous a failed.");
    CHECK_COND(DataContiguous(params.dw, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dw failed.");
    CHECK_COND(DataContiguous(params.du, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous du failed.");
    CHECK_COND(DataContiguous(params.g, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPrepareWyReprBwdGetWorkspaceSize(
    const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *a, const aclTensor *dw,
    const aclTensor *du, const aclTensor *g, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, int64_t chunkSize, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    PrepareWyReprBwdParams params{k, v, beta, a, dw, du, g, cuSeqlensOptional, chunkIndicesOptional, chunkSize,
                                  dkOut, dvOut, dbetaOut, dgOut};
    L2_DFX_PHASE_1_WITHOUT_CACHE(
        aclnnPrepareWyReprBwd,
        DFX_IN(k, v, beta, a, dw, du, g, cuSeqlensOptional, chunkIndicesOptional, chunkSize),
        DFX_OUT(dkOut, dvOut, dbetaOut, dgOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (!(uniqueExecutor.get() != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_CREATE_EXECUTOR, "CREATE_EXECUTOR returned nullptr");
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();

    if (!(CheckParams(params) == ACLNN_SUCCESS)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_WITH_REASON_COMPAT(DFX_ENTITY, "CheckParams inputs", "CheckParams(params) == ACLNN_SUCCESS", "documented constraint");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");

    auto result = l0op::PrepareWyReprBwd(params.k, params.v, params.beta, params.a, params.dw, params.du, params.g,
                                         params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize,
                                         params.dkOut, params.dvOut, params.dbetaOut, params.dgOut, executorPtr);
    if (result[0] == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_NULLPTR, "internal l0op output is null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (result[1] == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_NULLPTR, "internal l0op output is null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (result[2] == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_NULLPTR, "internal l0op output is null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (result[3] == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_NULLPTR, "internal l0op output is null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(result[0], params.dkOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(DFX_ENTITY, "viewCopyResult");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    viewCopyResult = l0op::ViewCopy(result[1], params.dvOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(DFX_ENTITY, "viewCopyResult");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    viewCopyResult = l0op::ViewCopy(result[2], params.dbetaOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(DFX_ENTITY, "viewCopyResult");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    viewCopyResult = l0op::ViewCopy(result[3], params.dgOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(DFX_ENTITY, "viewCopyResult");
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
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in PrepareWyReprBwd launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
