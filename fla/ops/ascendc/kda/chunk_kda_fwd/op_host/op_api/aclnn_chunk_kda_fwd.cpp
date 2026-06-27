/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "aclnn_chunk_kda_fwd.h"
#include "chunk_kda_fwd.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t MAX_KDA_K_DIM = 256;

struct ChunkKdaFwdParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *gk = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    bool outputFinalState = false;
    int64_t totalChunks = 1;
    const aclTensor *oOut = nullptr;
    const aclTensor *finalStateOut = nullptr;
    const aclTensor *aqkOut = nullptr;
    const aclTensor *akkOut = nullptr;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *qgOut = nullptr;
    const aclTensor *kgOut = nullptr;
    const aclTensor *vNewOut = nullptr;
    const aclTensor *hOut = nullptr;
};

aclnnStatus KdaFwdDataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus KdaFwdCheckParams(const ChunkKdaFwdParams &params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.gk != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gk must not be nullptr.");
    CHECK_COND(params.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(params.oOut != nullptr && params.finalStateOut != nullptr && params.aqkOut != nullptr &&
                   params.akkOut != nullptr && params.wOut != nullptr && params.uOut != nullptr &&
                   params.qgOut != nullptr && params.kgOut != nullptr && params.vNewOut != nullptr &&
                   params.hOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "ChunkKdaFwd outputs must not be nullptr.");
    CHECK_COND(params.chunkSize > 0, ACLNN_ERR_PARAM_INVALID, "chunkSize must be positive.");
    CHECK_COND(params.totalChunks > 0, ACLNN_ERR_PARAM_INVALID, "totalChunks must be positive.");
    CHECK_COND(params.q->GetViewShape().GetDim(3) <= MAX_KDA_K_DIM, ACLNN_ERR_PARAM_INVALID,
               "k head dimension must be less than or equal to 256.");
    return ACLNN_SUCCESS;
}

aclnnStatus KdaFwdParamsDataContiguous(ChunkKdaFwdParams &params, aclOpExecutor *executor)
{
    CHECK_RET(KdaFwdDataContiguous(params.q, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFwdDataContiguous(params.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFwdDataContiguous(params.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFwdDataContiguous(params.gk, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFwdDataContiguous(params.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFwdDataContiguous(params.initialStateOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *gk,
    const aclTensor *beta,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool outputFinalState,
    int64_t totalChunks,
    const aclTensor *oOut,
    const aclTensor *finalStateOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkKdaFwdParams params{q, k, v, gk, beta, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional,
                             scale, chunkSize, outputFinalState, totalChunks, oOut, finalStateOut, aqkOut,
                             akkOut, wOut, uOut, qgOut, kgOut, vNewOut, hOut};
    L2_DFX_PHASE_1(aclnnChunkKdaFwd,
                   DFX_IN(q, k, v, gk, beta, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(oOut, finalStateOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut, vNewOut, hOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(KdaFwdCheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFwdParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto result = l0op::ChunkKdaFwd(params.q, params.k, params.v, params.gk, params.beta, params.initialStateOptional,
                                    params.cuSeqlensOptional, params.chunkIndicesOptional, params.scale,
                                    params.chunkSize, params.outputFinalState, params.totalChunks, params.oOut,
                                    params.finalStateOut, params.aqkOut, params.akkOut, params.wOut, params.uOut,
                                    params.qgOut, params.kgOut, params.vNewOut, params.hOut, executorPtr);
    for (auto tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(result[0], params.oOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[1], params.finalStateOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[2], params.aqkOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[3], params.akkOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[4], params.wOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[5], params.uOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[6], params.qgOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[7], params.kgOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[8], params.vNewOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[9], params.hOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ChunkKdaFwd launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
