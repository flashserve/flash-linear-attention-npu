/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_recompute_wu_fwd.h"
#include "recompute_wu_fwd.h"
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

static constexpr int64_t RECOMPUTE_WU_K_DIM = 128;
static constexpr int64_t RECOMPUTE_WU_V_DIM_128 = 128;
static constexpr int64_t RECOMPUTE_WU_V_DIM_256 = 256;

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
    const auto wShape = params.wOut->GetViewShape();
    const auto uShape = params.uOut->GetViewShape();
    CHECK_COND(kShape.GetDimNum() == 4 && vShape.GetDimNum() == 4 && aShape.GetDimNum() == 4 &&
                   wShape.GetDimNum() == 4 && uShape.GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID, "k, v, A, wOut and uOut must be rank-4 tensors.");
    CHECK_COND(betaShape.GetDimNum() == 3 && gShape.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID,
               "beta and g must be rank-3 tensors.");
    const int64_t batch = kShape.GetDim(0);
    const int64_t keyHeads = kShape.GetDim(1);
    const int64_t seqlen = kShape.GetDim(2);
    const int64_t keyDim = kShape.GetDim(3);
    const int64_t valueHeads = vShape.GetDim(1);
    const int64_t valueDim = vShape.GetDim(3);
    CHECK_COND(batch > 0 && seqlen > 0 && keyDim > 0 && valueDim > 0, ACLNN_ERR_PARAM_INVALID,
               "B, T, K and V must be positive.");
    CHECK_COND(keyHeads > 0 && valueHeads > 0 && valueHeads % keyHeads == 0, ACLNN_ERR_PARAM_INVALID,
               "GVA requires HV divisible by HK.");
    CHECK_COND(vShape.GetDim(0) == batch && vShape.GetDim(2) == seqlen, ACLNN_ERR_PARAM_INVALID,
               "v must match k batch and sequence dimensions.");
    CHECK_COND(betaShape.GetDim(0) == batch && betaShape.GetDim(1) == valueHeads &&
                   betaShape.GetDim(2) == seqlen,
               ACLNN_ERR_PARAM_INVALID, "beta must be [B, HV, T].");
    CHECK_COND(gShape.GetDim(0) == batch && gShape.GetDim(1) == valueHeads && gShape.GetDim(2) == seqlen,
               ACLNN_ERR_PARAM_INVALID, "g must be [B, HV, T].");
    CHECK_COND(aShape.GetDim(0) == batch && aShape.GetDim(1) == valueHeads && aShape.GetDim(2) == seqlen &&
                   aShape.GetDim(3) == params.chunkSize,
               ACLNN_ERR_PARAM_INVALID, "A must be [B, HV, T, C] with C equal to chunkSize.");
    CHECK_COND(wShape.GetDim(0) == batch && wShape.GetDim(1) == valueHeads && wShape.GetDim(2) == seqlen &&
                   wShape.GetDim(3) == keyDim,
               ACLNN_ERR_PARAM_INVALID, "wOut must be [B, HV, T, K].");
    CHECK_COND(uShape.GetDim(0) == batch && uShape.GetDim(1) == valueHeads && uShape.GetDim(2) == seqlen &&
                   uShape.GetDim(3) == valueDim,
               ACLNN_ERR_PARAM_INVALID, "uOut must match v shape [B, HV, T, V].");
    CHECK_COND(keyDim == RECOMPUTE_WU_K_DIM, ACLNN_ERR_PARAM_INVALID,
               "K must be %ld, but got %ld.", RECOMPUTE_WU_K_DIM, keyDim);
    CHECK_COND(valueDim == RECOMPUTE_WU_V_DIM_128 || valueDim == RECOMPUTE_WU_V_DIM_256,
               ACLNN_ERR_PARAM_INVALID, "V must be 128 or 256, but got %ld.", valueDim);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckChunkMetadata(RecomputeWUFwdParams params)
{
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t batch = params.k->GetViewShape().GetDim(0);
    const int64_t seqlen = params.k->GetViewShape().GetDim(2);
    CHECK_COND(batch == 1, ACLNN_ERR_PARAM_INVALID,
               "B must be 1 when cuSeqlensOptional is provided, but got %ld.", batch);
    const aclIntArray &cu = *params.cuSeqlensOptional;
    const aclIntArray &indices = *params.chunkIndicesOptional;
    CHECK_COND(cu.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must contain at least [0, total_tokens].");
    CHECK_COND(cu[0] == 0 && cu[cu.Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must start at 0 and end at T=%ld.", seqlen);
    CHECK_COND(indices.Size() % 2 == 0, ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional must contain flattened (seq_id, chunk_id) pairs.");
    size_t index = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        CHECK_COND(cu[seq] >= 0 && cu[seq] <= cu[seq + 1] && cu[seq + 1] <= seqlen,
                   ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must be nondecreasing and within [0,T].");
        const int64_t localChunkCount =
            (cu[seq + 1] - cu[seq] + params.chunkSize - 1) / params.chunkSize;
        for (int64_t localChunk = 0; localChunk < localChunkCount; ++localChunk) {
            CHECK_COND(index + 1 < indices.Size(), ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOptional contains fewer pairs than required by cuSeqlensOptional.");
            CHECK_COND(indices[index] == static_cast<int64_t>(seq) && indices[index + 1] == localChunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOptional must use canonical sequence-major chunk order.");
            index += 2;
        }
    }
    CHECK_COND(index == indices.Size(), ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional pair count must match cuSeqlensOptional and chunkSize.");
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
    const auto inputDtype = params.k->GetDataType();
    CHECK_COND(inputDtype == DataType::DT_FLOAT16 || inputDtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "k dtype must be float16 or bfloat16.");
    CHECK_COND(params.v->GetDataType() == inputDtype && params.a->GetDataType() == inputDtype &&
                   params.wOut->GetDataType() == inputDtype && params.uOut->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "v, A, wOut and uOut dtype must match k.");
    CHECK_COND((params.beta->GetDataType() == inputDtype || params.beta->GetDataType() == DataType::DT_FLOAT) &&
                   params.g->GetDataType() == params.beta->GetDataType(),
               ACLNN_ERR_PARAM_INVALID, "beta and g must share float32 or the main input dtype.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(RecomputeWUFwdParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(params.gk == nullptr, ACLNN_ERR_PARAM_INVALID, "gk is reserved and must be nullptr.");
    CHECK_COND((params.cuSeqlensOptional == nullptr) == (params.chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional and chunkIndicesOptional must both be provided or both be nullptr.");
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be 64 or 128, but got %ld.", params.chunkSize);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckChunkMetadata(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
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
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    RecomputeWUFwdParams params{k, v, beta, a, g, gk, cuSeqlensOptional, chunkIndicesOptional, chunkSize, wOut, uOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnRecomputeWUFwd, DFX_IN(k, v, beta, a, g, gk, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(wOut, uOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
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
