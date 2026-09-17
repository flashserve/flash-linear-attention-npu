/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_recompute_w_u_fwd.h"
#include "recompute_w_u_fwd.h"
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

struct RecomputeWUFwdParams {
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *a = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *gk = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    int64_t chunkSize = 64;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
};

static aclnnStatus CheckNotNull(RecomputeWUFwdParams params)
{
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(params.a != nullptr, ACLNN_ERR_PARAM_NULLPTR, "a must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(params.gk == nullptr, ACLNN_ERR_PARAM_NULLPTR, "gk must be nullptr.");


    CHECK_COND(params.wOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "wOut must not be nullptr.");
    CHECK_COND(params.uOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "uOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(RecomputeWUFwdParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(RecomputeWUFwdParams params)
{
    const auto kShape = params.k->GetViewShape();
    const auto vShape = params.v->GetViewShape();
    const auto betaShape = params.beta->GetViewShape();
    const auto aShape = params.a->GetViewShape();
    const auto gShape = params.g->GetViewShape();
    const auto wOutShape = params.wOut->GetViewShape();
    const auto uOutShape = params.uOut->GetViewShape();

    CHECK_COND(kShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "k must be [B, HK, T, K].");
    CHECK_COND(vShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "v must be [B, HV, T, V].");
    CHECK_COND(betaShape.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID, "beta must be [B, HV, T].");
    CHECK_COND(aShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "A must be [B, HV, T, chunkSize].");
    CHECK_COND(gShape.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID, "g must be [B, HV, T].");
    CHECK_COND(wOutShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "wOut must be [B, HV, T, K].");
    CHECK_COND(uOutShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "uOut must be [B, HV, T, V].");

    const int64_t B = kShape.GetDim(0);
    const int64_t HK = kShape.GetDim(1);
    const int64_t T = kShape.GetDim(2);
    const int64_t K = kShape.GetDim(3);
    const int64_t HV = vShape.GetDim(1);
    const int64_t V = vShape.GetDim(3);

    CHECK_COND(HK > 0 && HV > 0 && HV % HK == 0, ACLNN_ERR_PARAM_INVALID,
               "GVA requires HV divisible by HK.");
    CHECK_COND(vShape.GetDim(0) == B && vShape.GetDim(2) == T, ACLNN_ERR_PARAM_INVALID,
               "v must match k batch and sequence dimensions.");
    CHECK_COND(betaShape.GetDim(0) == B && betaShape.GetDim(1) == HV && betaShape.GetDim(2) == T,
               ACLNN_ERR_PARAM_INVALID, "beta must be [B, HV, T].");
    CHECK_COND(aShape.GetDim(0) == B && aShape.GetDim(1) == HV && aShape.GetDim(2) == T &&
               aShape.GetDim(3) == params.chunkSize, ACLNN_ERR_PARAM_INVALID,
               "A must be [B, HV, T, chunkSize].");
    CHECK_COND(gShape.GetDim(0) == B && gShape.GetDim(1) == HV && gShape.GetDim(2) == T,
               ACLNN_ERR_PARAM_INVALID, "g must be [B, HV, T].");
    CHECK_COND(wOutShape.GetDim(0) == B && wOutShape.GetDim(1) == HV && wOutShape.GetDim(2) == T &&
               wOutShape.GetDim(3) == K, ACLNN_ERR_PARAM_INVALID, "wOut must be [B, HV, T, K].");
    CHECK_COND(uOutShape.GetDim(0) == B && uOutShape.GetDim(1) == HV && uOutShape.GetDim(2) == T &&
               uOutShape.GetDim(3) == V, ACLNN_ERR_PARAM_INVALID, "uOut must be [B, HV, T, V].");
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be 64 or 128.");
    CHECK_COND(K == 128, ACLNN_ERR_PARAM_INVALID, "K must be 128.");
    CHECK_COND(V == 128 || V == 256, ACLNN_ERR_PARAM_INVALID, "V must be 128 or 256.");
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(RecomputeWUFwdParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous v failed.");
    CHECK_COND(DataContiguous(params.beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous beta failed.");
    CHECK_COND(DataContiguous(params.a, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous a failed.");
    CHECK_COND(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(RecomputeWUFwdParams params)
{
    auto inputDtype = params.k->GetDataType();
    CHECK_COND(inputDtype == DataType::DT_FLOAT16 || inputDtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "k dtype must be float16 or bfloat16.");
    CHECK_COND(params.v->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "v dtype must match k.");
    CHECK_COND(params.a->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "a dtype must match k.");
    CHECK_COND(params.wOut->GetDataType() == inputDtype && params.uOut->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "wOut and uOut dtype must match k.");
    CHECK_COND(params.beta->GetDataType() == DataType::DT_FLOAT || params.beta->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "beta dtype must be float32 or match k dtype.");
    CHECK_COND(params.g->GetDataType() == DataType::DT_FLOAT || params.g->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "g dtype must be float32 or match k dtype.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(RecomputeWUFwdParams params)
{
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecomputeWUFwdGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *g,
    const aclTensor *gk,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *wOut,
    const aclTensor *uOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    RecomputeWUFwdParams params{k, v, beta, a, g, gk,cuSeqlensOptional, chunkIndicesOptional, chunkSize, wOut, uOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnRecomputeWUFwd,
                   DFX_IN(k, v, beta, a, g, gk, cuSeqlensOptional, chunkIndicesOptional, chunkSize),
                   DFX_OUT(wOut, uOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::RecomputeWUFwd(params.k, params.v, params.beta, params.a, params.g, params.gk, params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize, params.wOut, params.uOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result[0], params.wOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[1], params.uOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);


    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnRecomputeWUFwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecomputeWUFwd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in RecomputeWUFwd launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
