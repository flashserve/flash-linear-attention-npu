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
#include "../../../kda_layout_swap12/op_host/op_api/kda_layout_swap12.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
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

size_t KdaBwdRank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
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
    size_t qRank = KdaBwdRank(params.q);
    size_t betaRank = KdaBwdRank(params.beta);
    CHECK_COND((qRank == 4 && betaRank == 3) || (qRank == 3 && betaRank == 2), ACLNN_ERR_PARAM_INVALID,
               "q/k/v/gk must be BSND rank4 with beta rank3, or TND rank3 with beta rank2.");
    size_t kDimIdx = (qRank == 4) ? 3 : 2;
    CHECK_COND(params.q->GetViewShape().GetDim(kDimIdx) <= MAX_KDA_K_DIM, ACLNN_ERR_PARAM_INVALID,
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

    bool isTnd = KdaBwdRank(params.q) == 3;
    int64_t batch = isTnd ? 1 : KdaBwdDim(params.q, 0);
    int64_t seqlen = isTnd ? KdaBwdDim(params.q, 0) : KdaBwdDim(params.q, 1);
    int64_t hNum = isTnd ? KdaBwdDim(params.q, 1) : KdaBwdDim(params.q, 2);
    int64_t kDim = isTnd ? KdaBwdDim(params.q, 2) : KdaBwdDim(params.q, 3);
    int64_t hvNum = isTnd ? KdaBwdDim(params.v, 1) : KdaBwdDim(params.v, 2);
    int64_t vDim = isTnd ? KdaBwdDim(params.v, 2) : KdaBwdDim(params.v, 3);

    const aclTensor *qBsnd = params.q;
    const aclTensor *kBsnd = params.k;
    const aclTensor *vBsnd = params.v;
    const aclTensor *gkBsnd = params.gk;
    const aclTensor *betaBsn = params.beta;
    const aclTensor *dOBsnd = params.dO;
    if (isTnd) {
        qBsnd = l0op::Reshape(params.q, KdaBwdMakeShape({1, seqlen, hNum, kDim}), executorPtr);
        kBsnd = l0op::Reshape(params.k, KdaBwdMakeShape({1, seqlen, hNum, kDim}), executorPtr);
        vBsnd = l0op::Reshape(params.v, KdaBwdMakeShape({1, seqlen, hvNum, vDim}), executorPtr);
        gkBsnd = l0op::Reshape(params.gk, KdaBwdMakeShape({1, seqlen, hvNum, kDim}), executorPtr);
        betaBsn = l0op::Reshape(params.beta, KdaBwdMakeShape({1, seqlen, hvNum}), executorPtr);
        dOBsnd = l0op::Reshape(params.dO, KdaBwdMakeShape({1, seqlen, hvNum, vDim}), executorPtr);
        CHECK_RET(qBsnd != nullptr && kBsnd != nullptr && vBsnd != nullptr && gkBsnd != nullptr &&
                      betaBsn != nullptr && dOBsnd != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    auto qBnsd = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hNum, seqlen, kDim}),
                                          params.q->GetDataType(), Format::FORMAT_ND);
    auto kBnsd = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hNum, seqlen, kDim}),
                                          params.k->GetDataType(), Format::FORMAT_ND);
    auto vBnsd = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, vDim}),
                                          params.v->GetDataType(), Format::FORMAT_ND);
    auto gkBnsdRaw = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, kDim}),
                                              params.gk->GetDataType(), Format::FORMAT_ND);
    auto betaBnsRaw = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen}),
                                               params.beta->GetDataType(), Format::FORMAT_ND);
    auto dOBnsd = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, vDim}),
                                           params.dO->GetDataType(), Format::FORMAT_ND);
    auto dqBnsd = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hNum, seqlen, kDim}),
                                           params.dqOut->GetDataType(), Format::FORMAT_ND);
    auto dkBnsd = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hNum, seqlen, kDim}),
                                           params.dkOut->GetDataType(), Format::FORMAT_ND);
    auto dvBnsd = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, vDim}),
                                           params.dvOut->GetDataType(), Format::FORMAT_ND);
    auto dbetaBns = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen}),
                                             params.dbetaOut->GetDataType(), Format::FORMAT_ND);
    auto dgkBnsd = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, kDim}),
                                            params.dgkOut->GetDataType(), Format::FORMAT_ND);
    CHECK_RET(qBnsd != nullptr && kBnsd != nullptr && vBnsd != nullptr && gkBnsdRaw != nullptr &&
                  betaBnsRaw != nullptr && dOBnsd != nullptr && dqBnsd != nullptr && dkBnsd != nullptr &&
                  dvBnsd != nullptr && dbetaBns != nullptr && dgkBnsd != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(l0op::KdaLayoutSwap12(qBsnd, qBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::KdaLayoutSwap12(kBsnd, kBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::KdaLayoutSwap12(vBsnd, vBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::KdaLayoutSwap12(gkBsnd, gkBnsdRaw, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::KdaLayoutSwap12(betaBsn, betaBnsRaw, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::KdaLayoutSwap12(dOBsnd, dOBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *gkBnsd = gkBnsdRaw;
    const aclTensor *betaBns = betaBnsRaw;
    if (gkBnsd->GetDataType() != DataType::DT_FLOAT) {
        gkBnsd = l0op::Cast(gkBnsd, DataType::DT_FLOAT, executorPtr);
        CHECK_RET(gkBnsd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (betaBns->GetDataType() != DataType::DT_FLOAT) {
        betaBns = l0op::Cast(betaBns, DataType::DT_FLOAT, executorPtr);
        CHECK_RET(betaBns != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto oTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, vDim}),
                                         params.v->GetDataType(), Format::FORMAT_ND);
    auto stateTmp = executorPtr->AllocTensor(KdaBwdMakeShape({params.seqNum, hvNum, kDim, vDim}),
                                             params.q->GetDataType(), Format::FORMAT_ND);
    auto aqkTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, params.chunkSize}),
                                           params.aqk->GetDataType(), Format::FORMAT_ND);
    auto akkTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, params.chunkSize}),
                                           params.akk->GetDataType(), Format::FORMAT_ND);
    auto wTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, kDim}),
                                         params.q->GetDataType(), Format::FORMAT_ND);
    auto uTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, vDim}),
                                         params.v->GetDataType(), Format::FORMAT_ND);
    auto qgTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, kDim}),
                                          params.q->GetDataType(), Format::FORMAT_ND);
    auto kgTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, kDim}),
                                          params.q->GetDataType(), Format::FORMAT_ND);
    auto vNewTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, vDim}),
                                            params.v->GetDataType(), Format::FORMAT_ND);
    auto hTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, params.totalChunks, kDim, vDim}),
                                         params.q->GetDataType(), Format::FORMAT_ND);
    auto dVNewGradTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, vDim}),
                                                DataType::DT_FLOAT, Format::FORMAT_ND);
    auto dWTmp = executorPtr->AllocTensor(KdaBwdMakeShape({batch, hvNum, seqlen, kDim}),
                                          DataType::DT_FLOAT, Format::FORMAT_ND);
    CHECK_RET(oTmp != nullptr && stateTmp != nullptr && aqkTmp != nullptr && akkTmp != nullptr && wTmp != nullptr &&
                  uTmp != nullptr && qgTmp != nullptr && kgTmp != nullptr && vNewTmp != nullptr && hTmp != nullptr &&
                  dVNewGradTmp != nullptr && dWTmp != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto fwdResult = l0op::ChunkKdaFwd(qBnsd, kBnsd, vBnsd, gkBnsd, betaBns, params.initialStateOptional,
                                       params.cuSeqlensOptional, params.chunkIndicesOptional, params.scale,
                                       params.chunkSize, true, params.totalChunks, oTmp, stateTmp, aqkTmp, akkTmp,
                                       wTmp, uTmp, qgTmp, kgTmp, vNewTmp, hTmp, executorPtr);
    for (auto tensor : fwdResult) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto bwdResult = l0op::ChunkKdaBwd(qBnsd, kBnsd, vBnsd, gkBnsd, betaBns, aqkTmp, akkTmp, wTmp,
                                       uTmp, qgTmp, kgTmp, vNewTmp, hTmp, dOBnsd, params.initialStateOptional,
                                       params.dhtOptional, params.cuSeqlensOptional, params.chunkIndicesOptional,
                                       params.scale, params.chunkSize, params.totalChunks, dqBnsd, dkBnsd, dvBnsd,
                                       dbetaBns, dgkBnsd, params.dh0Out, dVNewGradTmp, dWTmp, executorPtr);
    for (auto tensor : bwdResult) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (isTnd) {
        auto dqBsnd = executorPtr->AllocTensor(KdaBwdMakeShape({1, seqlen, hNum, kDim}),
                                               params.dqOut->GetDataType(), Format::FORMAT_ND);
        auto dkBsnd = executorPtr->AllocTensor(KdaBwdMakeShape({1, seqlen, hNum, kDim}),
                                               params.dkOut->GetDataType(), Format::FORMAT_ND);
        auto dvBsnd = executorPtr->AllocTensor(KdaBwdMakeShape({1, seqlen, hvNum, vDim}),
                                               params.dvOut->GetDataType(), Format::FORMAT_ND);
        auto dbetaBsn = executorPtr->AllocTensor(KdaBwdMakeShape({1, seqlen, hvNum}),
                                                 params.dbetaOut->GetDataType(), Format::FORMAT_ND);
        auto dgkBsnd = executorPtr->AllocTensor(KdaBwdMakeShape({1, seqlen, hvNum, kDim}),
                                                params.dgkOut->GetDataType(), Format::FORMAT_ND);
        CHECK_RET(dqBsnd != nullptr && dkBsnd != nullptr && dvBsnd != nullptr &&
                      dbetaBsn != nullptr && dgkBsnd != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[0], dqBsnd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[1], dkBsnd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[2], dvBsnd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[3], dbetaBsn, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[4], dgkBsnd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(dqBsnd, KdaBwdMakeShape({seqlen, hNum, kDim}), executorPtr),
                                 params.dqOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(dkBsnd, KdaBwdMakeShape({seqlen, hNum, kDim}), executorPtr),
                                 params.dkOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(dvBsnd, KdaBwdMakeShape({seqlen, hvNum, vDim}), executorPtr),
                                 params.dvOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(dbetaBsn, KdaBwdMakeShape({seqlen, hvNum}), executorPtr),
                                 params.dbetaOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(dgkBsnd, KdaBwdMakeShape({seqlen, hvNum, kDim}), executorPtr),
                                 params.dgkOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[0], params.dqOut, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[1], params.dkOut, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[2], params.dvOut, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[3], params.dbetaOut, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(bwdResult[4], params.dgkOut, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }
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
