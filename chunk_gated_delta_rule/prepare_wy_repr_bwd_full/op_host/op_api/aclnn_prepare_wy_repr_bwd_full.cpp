/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_prepare_wy_repr_bwd_full.h"
#include "prepare_wy_repr_bwd_full.h"
#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"


using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct PrepareWyReprBwdFullParams {
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *a = nullptr;
    const aclTensor *dA = nullptr;
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

static aclnnStatus CheckNotNull(PrepareWyReprBwdFullParams params)
{
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(params.a != nullptr, ACLNN_ERR_PARAM_NULLPTR, "a must not be nullptr.");
    CHECK_COND(params.dA != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dA must not be nullptr.");
    CHECK_COND(params.dw != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dw must not be nullptr.");
    CHECK_COND(params.du != nullptr, ACLNN_ERR_PARAM_NULLPTR, "du must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");

    CHECK_COND(params.dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dkOut must not be nullptr.");
    CHECK_COND(params.dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dvOut must not be nullptr.");
    CHECK_COND(params.dbetaOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dbetaOut must not be nullptr.");
    CHECK_COND(params.dgOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dgOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(PrepareWyReprBwdFullParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(PrepareWyReprBwdFullParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(PrepareWyReprBwdFullParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous v failed.");
    CHECK_COND(DataContiguous(params.beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous beta failed.");
    CHECK_COND(DataContiguous(params.a, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous a failed.");
    CHECK_COND(DataContiguous(params.dA, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dA failed.");
    CHECK_COND(DataContiguous(params.dw, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dw failed.");
    CHECK_COND(DataContiguous(params.du, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous du failed.");
    CHECK_COND(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(PrepareWyReprBwdFullParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(PrepareWyReprBwdFullParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPrepareWyReprBwdFullGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dA,
    const aclTensor *dw,
    const aclTensor *du,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dbetaOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    PrepareWyReprBwdFullParams params{k, v, beta, a, dA, dw, du, g, cuSeqlensOptional, chunkIndicesOptional, chunkSize, dkOut, dvOut, dbetaOut, dgOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnPrepareWyReprBwdFull, DFX_IN(k, v, beta, a, dA, dw, du, g, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(dkOut, dvOut, dbetaOut, dgOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::PrepareWyReprBwdFull(params.k, params.v, params.beta, params.a, params.dA, params.dw, params.du, params.g, params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize, params.dkOut, params.dvOut, params.dbetaOut, params.dgOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[3] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result[0], params.dkOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[1], params.dvOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[2], params.dbetaOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[3], params.dgOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnPrepareWyReprBwdFull(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPrepareWyReprBwdFull);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in QuantGMMInplaceAdd launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
