/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "aclnn_chunk_kda_bwd.h"
#include "chunk_kda_bwd.h"
#include "../../../chunk_kda_fwd/op_host/op_api/chunk_kda_fwd.h"

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

struct ChunkKdaBwdParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *gk = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *aqk = nullptr;
    const aclTensor *akk = nullptr;
    const aclTensor *dO = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    const aclTensor *dhtOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    int64_t seqNum = 1;
    int64_t totalChunks = 1;
    const aclTensor *dqOut = nullptr;
    const aclTensor *dkOut = nullptr;
    const aclTensor *dvOut = nullptr;
    const aclTensor *dbetaOut = nullptr;
    const aclTensor *dgkOut = nullptr;
    const aclTensor *dh0Out = nullptr;
};

aclnnStatus KdaBwdDataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

op::Shape KdaBwdMakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

int64_t KdaBwdDim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

aclnnStatus KdaBwdCheckParams(const ChunkKdaBwdParams &params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.gk != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gk must not be nullptr.");
    CHECK_COND(params.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(params.aqk != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Aqk must not be nullptr.");
    CHECK_COND(params.akk != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Akk must not be nullptr.");
    CHECK_COND(params.dO != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dO must not be nullptr.");
    CHECK_COND(params.dqOut != nullptr && params.dkOut != nullptr && params.dvOut != nullptr &&
                   params.dbetaOut != nullptr && params.dgkOut != nullptr && params.dh0Out != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "ChunkKdaBwd outputs must not be nullptr.");
    CHECK_COND(params.chunkSize > 0, ACLNN_ERR_PARAM_INVALID, "chunkSize must be positive.");
    CHECK_COND(params.seqNum > 0, ACLNN_ERR_PARAM_INVALID, "seqNum must be positive.");
    CHECK_COND(params.totalChunks > 0, ACLNN_ERR_PARAM_INVALID, "totalChunks must be positive.");
    CHECK_COND(params.q->GetViewShape().GetDim(3) <= MAX_KDA_K_DIM, ACLNN_ERR_PARAM_INVALID,
               "k head dimension must be less than or equal to 256.");
    return ACLNN_SUCCESS;
}

aclnnStatus KdaBwdParamsDataContiguous(ChunkKdaBwdParams &params, aclOpExecutor *executor)
{
    CHECK_RET(KdaBwdDataContiguous(params.q, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdDataContiguous(params.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdDataContiguous(params.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdDataContiguous(params.gk, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdDataContiguous(params.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdDataContiguous(params.aqk, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdDataContiguous(params.akk, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdDataContiguous(params.dO, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdDataContiguous(params.initialStateOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdDataContiguous(params.dhtOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnChunkKdaBwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *gk,
    const aclTensor *beta,
    const aclTensor *aqk,
    const aclTensor *akk,
    const aclTensor *dO,
    const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    int64_t seqNum,
    int64_t totalChunks,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dbetaOut,
    const aclTensor *dgkOut,
    const aclTensor *dh0Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkKdaBwdParams params{q, k, v, gk, beta, aqk, akk, dO, initialStateOptional, dhtOptional,
                             cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, seqNum, totalChunks,
                             dqOut, dkOut, dvOut, dbetaOut, dgkOut, dh0Out};
    L2_DFX_PHASE_1(aclnnChunkKdaBwd,
                   DFX_IN(q, k, v, gk, beta, aqk, akk, dO, initialStateOptional, dhtOptional,
                          cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(dqOut, dkOut, dvOut, dbetaOut, dgkOut, dh0Out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(KdaBwdCheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaBwdParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    int64_t batch = KdaBwdDim(params.q, 0);
    int64_t hNum = KdaBwdDim(params.q, 1);
    int64_t seqlen = KdaBwdDim(params.q, 2);
    int64_t kDim = KdaBwdDim(params.q, 3);
    int64_t hvNum = KdaBwdDim(params.v, 1);
    int64_t vDim = KdaBwdDim(params.v, 3);

    auto oTmp = executorPtr->AllocTensor(params.v->GetViewShape(), params.v->GetDataType(), Format::FORMAT_ND);
    auto stateTmp = executorPtr->AllocTensor(KdaBwdMakeShape({params.seqNum, hvNum, kDim, vDim}),
                                             params.q->GetDataType(), Format::FORMAT_ND);
    auto aqkTmp = executorPtr->AllocTensor(params.aqk->GetViewShape(), params.aqk->GetDataType(), Format::FORMAT_ND);
    auto akkTmp = executorPtr->AllocTensor(params.akk->GetViewShape(), params.akk->GetDataType(), Format::FORMAT_ND);
    auto wTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, kDim}),
                                         params.q->GetDataType(), Format::FORMAT_ND);
    auto uTmp = executorPtr->AllocTensor(params.v->GetViewShape(), params.v->GetDataType(), Format::FORMAT_ND);
    auto qgTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, kDim}),
                                          params.q->GetDataType(), Format::FORMAT_ND);
    auto kgTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, kDim}),
                                          params.q->GetDataType(), Format::FORMAT_ND);
    auto vNewTmp = executorPtr->AllocTensor(params.v->GetViewShape(), params.v->GetDataType(), Format::FORMAT_ND);
    auto hTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, params.totalChunks, kDim, vDim}),
                                         params.q->GetDataType(), Format::FORMAT_ND);
    CHECK_RET(oTmp != nullptr && stateTmp != nullptr && aqkTmp != nullptr && akkTmp != nullptr && wTmp != nullptr &&
                  uTmp != nullptr && qgTmp != nullptr && kgTmp != nullptr && vNewTmp != nullptr && hTmp != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto fwdResult = l0op::ChunkKdaFwd(params.q, params.k, params.v, params.gk, params.beta,
                                       params.initialStateOptional, params.cuSeqlensOptional,
                                       params.chunkIndicesOptional, params.scale, params.chunkSize, true,
                                       params.totalChunks, oTmp, stateTmp, aqkTmp, akkTmp, wTmp, uTmp, qgTmp, kgTmp,
                                       vNewTmp, hTmp, executorPtr);
    for (auto tensor : fwdResult) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto bwdResult = l0op::ChunkKdaBwd(params.q, params.k, params.v, params.gk, params.beta, aqkTmp, akkTmp, wTmp,
                                       uTmp, qgTmp, kgTmp, vNewTmp, hTmp, params.dO, params.initialStateOptional,
                                       params.dhtOptional, params.cuSeqlensOptional, params.chunkIndicesOptional,
                                       params.scale, params.chunkSize, params.totalChunks, params.dqOut,
                                       params.dkOut, params.dvOut, params.dbetaOut, params.dgkOut, params.dh0Out,
                                       executorPtr);
    for (auto tensor : bwdResult) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(bwdResult[0], params.dqOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(bwdResult[1], params.dkOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(bwdResult[2], params.dvOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(bwdResult[3], params.dbetaOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(bwdResult[4], params.dgkOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(bwdResult[5], params.dh0Out, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    (void)hNum;
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaBwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ChunkKdaBwd launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
