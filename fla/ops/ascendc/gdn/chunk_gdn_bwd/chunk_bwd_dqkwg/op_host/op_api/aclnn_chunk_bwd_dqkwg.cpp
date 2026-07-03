/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_chunk_bwd_dqkwg.h"
#include "chunk_bwd_dqkwg.h"
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

struct ChunkBwdDqkwgParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *h = nullptr;
    const aclTensor *dox = nullptr;
    const aclTensor *dh = nullptr;
    const aclTensor *dv = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *gGamma = nullptr;
    float scale = 0;
    int64_t chunkSize = 0;
    const aclTensor *dqOut = nullptr;
    const aclTensor *dkOut = nullptr;
    const aclTensor *dwOut = nullptr;
    const aclTensor *dgOut = nullptr;
};

static aclnnStatus CheckNotNull(ChunkBwdDqkwgParams params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(params.h != nullptr, ACLNN_ERR_PARAM_NULLPTR, "h must not be nullptr.");
    CHECK_COND(params.dox != nullptr, ACLNN_ERR_PARAM_NULLPTR, "do must not be nullptr.");
    CHECK_COND(params.dh != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dh must not be nullptr.");
    CHECK_COND(params.dv != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dv must not be nullptr.");

    CHECK_COND(params.dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dqOut must not be nullptr.");
    CHECK_COND(params.dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dkvOut must not be nullptr.");
    CHECK_COND(params.dwOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dwOut must not be nullptr.");
    CHECK_COND(params.dgOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dgOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkBwdDqkwgParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkBwdDqkwgParams params)
{
    const auto qShape = params.q->GetViewShape();
    const auto kShape = params.k->GetViewShape();
    const auto vShape = params.v->GetViewShape();
    const auto gShape = params.g->GetViewShape();
    const auto dqShape = params.dqOut->GetViewShape();
    const auto dkShape = params.dkOut->GetViewShape();
    const auto dwShape = params.dwOut->GetViewShape();
    const auto dgShape = params.dgOut->GetViewShape();

    CHECK_COND(qShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "q must be [B, HK, T, K].");
    CHECK_COND(kShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "k must be [B, HK, T, K].");
    CHECK_COND(vShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "v must be [B, HV, T, V].");
    CHECK_COND(gShape.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID, "g must be [B, HV, T].");
    CHECK_COND(dqShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "dqOut must be [B, HK, T, K].");
    CHECK_COND(dkShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "dkOut must be [B, HK, T, K].");
    CHECK_COND(dwShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "dwOut must be [B, HV, T, K].");
    CHECK_COND(dgShape.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID, "dgOut must be [B, HV, T].");

    const int64_t B = qShape.GetDim(0);
    const int64_t HK = qShape.GetDim(1);
    const int64_t T = qShape.GetDim(2);
    const int64_t K = qShape.GetDim(3);
    const int64_t HV = vShape.GetDim(1);

    CHECK_COND(HK > 0 && HV > 0 && HV % HK == 0, ACLNN_ERR_PARAM_INVALID,
               "GVA requires HV divisible by HK.");
    CHECK_COND(kShape.GetDim(0) == B && kShape.GetDim(1) == HK && kShape.GetDim(2) == T && kShape.GetDim(3) == K,
               ACLNN_ERR_PARAM_INVALID, "k must match q shape [B, HK, T, K].");
    CHECK_COND(vShape.GetDim(0) == B && vShape.GetDim(2) == T, ACLNN_ERR_PARAM_INVALID,
               "v must match q batch and sequence dimensions.");
    CHECK_COND(gShape.GetDim(0) == B && gShape.GetDim(1) == HV && gShape.GetDim(2) == T, ACLNN_ERR_PARAM_INVALID,
               "g must be [B, HV, T].");
    CHECK_COND(dqShape.GetDim(0) == B && dqShape.GetDim(1) == HK && dqShape.GetDim(2) == T && dqShape.GetDim(3) == K,
               ACLNN_ERR_PARAM_INVALID, "dqOut must be [B, HK, T, K].");
    CHECK_COND(dkShape.GetDim(0) == B && dkShape.GetDim(1) == HK && dkShape.GetDim(2) == T && dkShape.GetDim(3) == K,
               ACLNN_ERR_PARAM_INVALID, "dkOut must be [B, HK, T, K].");
    CHECK_COND(dwShape.GetDim(0) == B && dwShape.GetDim(1) == HV && dwShape.GetDim(2) == T && dwShape.GetDim(3) == K,
               ACLNN_ERR_PARAM_INVALID, "dwOut must be [B, HV, T, K].");
    CHECK_COND(dgShape.GetDim(0) == B && dgShape.GetDim(1) == HV && dgShape.GetDim(2) == T,
               ACLNN_ERR_PARAM_INVALID, "dgOut must be [B, HV, T].");
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkBwdDqkwgParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous q failed.");
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous v failed.");
    CHECK_COND(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");
    CHECK_COND(DataContiguous(params.h, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous h failed.");
    CHECK_COND(DataContiguous(params.dox, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous do failed.");
    CHECK_COND(DataContiguous(params.dh, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dh failed.");
    CHECK_COND(DataContiguous(params.dv, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dv failed.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkBwdDqkwgParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkBwdDqkwgParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkBwdDqkwgGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *h,
    const aclTensor *dox,
    const aclTensor *dh,
    const aclTensor *dv,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const aclTensor *w,
    const aclTensor *gGamma,
    float scale,
    int64_t chunkSize,
    bool use_exp2,
    bool transpose_state_layout,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dwOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkBwdDqkwgParams params{q, k, v, g, h, dox, dh, dv, cuSeqlensOptional, chunkIndicesOptional, w, gGamma, scale, chunkSize, dqOut, dkOut, dwOut, dgOut};
    CHECK_COND(use_exp2 == false && transpose_state_layout == false, ACLNN_ERR_INNER,
               "use_exp2 and transpose_state_layout must be false.");
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkBwdDqkwg, DFX_IN(q, k, v, g, h, dox, dh, dv, cuSeqlensOptional, chunkIndicesOptional, w, gGamma),
                   DFX_OUT(dqOut, dkOut, dwOut, dgOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::ChunkBwdDqkwg(params.q, params.k, params.v, params.g, params.h, params.dox, params.dh, params.dv, params.cuSeqlensOptional, params.chunkIndicesOptional, params.w, params.gGamma, params.scale, params.chunkSize, params.dqOut, params.dkOut, params.dwOut, params.dgOut, executorPtr);

    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[3] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result[0], params.dqOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[1], params.dkOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[2], params.dwOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[3], params.dgOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkBwdDqkwg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkBwdDqkwg);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkBwdDqkwg launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
