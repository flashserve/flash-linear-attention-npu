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

op::Shape KdaFwdMakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

int64_t KdaFwdDim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

size_t KdaFwdRank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
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
    size_t qRank = KdaFwdRank(params.q);
    size_t betaRank = KdaFwdRank(params.beta);
    CHECK_COND((qRank == 4 && betaRank == 3) || (qRank == 3 && betaRank == 2), ACLNN_ERR_PARAM_INVALID,
               "q/k/v/gk must be BSND rank4 with beta rank3, or TND rank3 with beta rank2.");
    size_t kDimIdx = (qRank == 4) ? 3 : 2;
    CHECK_COND(params.q->GetViewShape().GetDim(kDimIdx) <= MAX_KDA_K_DIM, ACLNN_ERR_PARAM_INVALID,
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

    bool isTnd = KdaFwdRank(params.q) == 3;
    int64_t batch = isTnd ? 1 : KdaFwdDim(params.q, 0);
    int64_t seqlen = isTnd ? KdaFwdDim(params.q, 0) : KdaFwdDim(params.q, 1);
    int64_t hNum = isTnd ? KdaFwdDim(params.q, 1) : KdaFwdDim(params.q, 2);
    int64_t kDim = isTnd ? KdaFwdDim(params.q, 2) : KdaFwdDim(params.q, 3);
    int64_t hvNum = isTnd ? KdaFwdDim(params.v, 1) : KdaFwdDim(params.v, 2);
    int64_t vDim = isTnd ? KdaFwdDim(params.v, 2) : KdaFwdDim(params.v, 3);

    const aclTensor *qBsnd = params.q;
    const aclTensor *kBsnd = params.k;
    const aclTensor *vBsnd = params.v;
    const aclTensor *gkBsnd = params.gk;
    const aclTensor *betaBsn = params.beta;
    if (isTnd) {
        qBsnd = l0op::Reshape(params.q, KdaFwdMakeShape({1, seqlen, hNum, kDim}), executorPtr);
        kBsnd = l0op::Reshape(params.k, KdaFwdMakeShape({1, seqlen, hNum, kDim}), executorPtr);
        vBsnd = l0op::Reshape(params.v, KdaFwdMakeShape({1, seqlen, hvNum, vDim}), executorPtr);
        gkBsnd = l0op::Reshape(params.gk, KdaFwdMakeShape({1, seqlen, hvNum, kDim}), executorPtr);
        betaBsn = l0op::Reshape(params.beta, KdaFwdMakeShape({1, seqlen, hvNum}), executorPtr);
        CHECK_RET(qBsnd != nullptr && kBsnd != nullptr && vBsnd != nullptr && gkBsnd != nullptr && betaBsn != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    auto qBnsd = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hNum, seqlen, kDim}),
                                          params.q->GetDataType(), Format::FORMAT_ND);
    auto kBnsd = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hNum, seqlen, kDim}),
                                          params.k->GetDataType(), Format::FORMAT_ND);
    auto vBnsd = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, vDim}),
                                          params.v->GetDataType(), Format::FORMAT_ND);
    auto gkBnsdRaw = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, kDim}),
                                              params.gk->GetDataType(), Format::FORMAT_ND);
    auto betaBnsRaw = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen}),
                                               params.beta->GetDataType(), Format::FORMAT_ND);
    auto oBnsd = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, vDim}),
                                          params.oOut->GetDataType(), Format::FORMAT_ND);
    auto aqkBnst = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, params.chunkSize}),
                                            params.aqkOut->GetDataType(), Format::FORMAT_ND);
    auto akkBnst = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, params.chunkSize}),
                                            params.akkOut->GetDataType(), Format::FORMAT_ND);
    auto wBnsd = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, kDim}),
                                          params.wOut->GetDataType(), Format::FORMAT_ND);
    auto uBnsd = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, vDim}),
                                          params.uOut->GetDataType(), Format::FORMAT_ND);
    auto qgBnsd = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, kDim}),
                                           params.qgOut->GetDataType(), Format::FORMAT_ND);
    auto kgBnsd = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, kDim}),
                                           params.kgOut->GetDataType(), Format::FORMAT_ND);
    auto vNewBnsd = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, vDim}),
                                             params.vNewOut->GetDataType(), Format::FORMAT_ND);
    auto hBnst = executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, params.totalChunks, kDim, vDim}),
                                          params.hOut->GetDataType(), Format::FORMAT_ND);
    CHECK_RET(qBnsd != nullptr && kBnsd != nullptr && vBnsd != nullptr && gkBnsdRaw != nullptr &&
                  betaBnsRaw != nullptr && oBnsd != nullptr && aqkBnst != nullptr && akkBnst != nullptr &&
                  wBnsd != nullptr && uBnsd != nullptr && qgBnsd != nullptr && kgBnsd != nullptr &&
                  vNewBnsd != nullptr && hBnst != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(l0op::KdaLayoutSwap12(qBsnd, qBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::KdaLayoutSwap12(kBsnd, kBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::KdaLayoutSwap12(vBsnd, vBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::KdaLayoutSwap12(gkBsnd, gkBnsdRaw, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::KdaLayoutSwap12(betaBsn, betaBnsRaw, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

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

    auto result = l0op::ChunkKdaFwd(qBnsd, kBnsd, vBnsd, gkBnsd, betaBns, params.initialStateOptional,
                                    params.cuSeqlensOptional, params.chunkIndicesOptional, params.scale,
                                    params.chunkSize, params.outputFinalState, params.totalChunks, oBnsd,
                                    params.finalStateOut, aqkBnst, akkBnst, wBnsd, uBnsd, qgBnsd, kgBnsd,
                                    vNewBnsd, hBnst, executorPtr);
    for (auto tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (isTnd) {
        auto oBsnd = executorPtr->AllocTensor(KdaFwdMakeShape({1, seqlen, hvNum, vDim}),
                                              params.oOut->GetDataType(), Format::FORMAT_ND);
        auto aqkBsnt = executorPtr->AllocTensor(KdaFwdMakeShape({1, seqlen, hvNum, params.chunkSize}),
                                                params.aqkOut->GetDataType(), Format::FORMAT_ND);
        auto akkBsnt = executorPtr->AllocTensor(KdaFwdMakeShape({1, seqlen, hvNum, params.chunkSize}),
                                                params.akkOut->GetDataType(), Format::FORMAT_ND);
        auto wBsnd = executorPtr->AllocTensor(KdaFwdMakeShape({1, seqlen, hvNum, kDim}),
                                              params.wOut->GetDataType(), Format::FORMAT_ND);
        auto uBsnd = executorPtr->AllocTensor(KdaFwdMakeShape({1, seqlen, hvNum, vDim}),
                                              params.uOut->GetDataType(), Format::FORMAT_ND);
        auto qgBsnd = executorPtr->AllocTensor(KdaFwdMakeShape({1, seqlen, hvNum, kDim}),
                                               params.qgOut->GetDataType(), Format::FORMAT_ND);
        auto kgBsnd = executorPtr->AllocTensor(KdaFwdMakeShape({1, seqlen, hvNum, kDim}),
                                               params.kgOut->GetDataType(), Format::FORMAT_ND);
        auto vNewBsnd = executorPtr->AllocTensor(KdaFwdMakeShape({1, seqlen, hvNum, vDim}),
                                                 params.vNewOut->GetDataType(), Format::FORMAT_ND);
        auto hBsnt = executorPtr->AllocTensor(KdaFwdMakeShape({1, params.totalChunks, hvNum, kDim, vDim}),
                                              params.hOut->GetDataType(), Format::FORMAT_ND);
        CHECK_RET(oBsnd != nullptr && aqkBsnt != nullptr && akkBsnt != nullptr && wBsnd != nullptr &&
                      uBsnd != nullptr && qgBsnd != nullptr && kgBsnd != nullptr && vNewBsnd != nullptr &&
                      hBsnt != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[0], oBsnd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[2], aqkBsnt, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[3], akkBsnt, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[4], wBsnd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[5], uBsnd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[6], qgBsnd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[7], kgBsnd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[8], vNewBsnd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[9], hBsnt, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(oBsnd, KdaFwdMakeShape({seqlen, hvNum, vDim}), executorPtr),
                                 params.oOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(aqkBsnt, KdaFwdMakeShape({seqlen, hvNum, params.chunkSize}),
                                                executorPtr),
                                 params.aqkOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(akkBsnt, KdaFwdMakeShape({seqlen, hvNum, params.chunkSize}),
                                                executorPtr),
                                 params.akkOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(wBsnd, KdaFwdMakeShape({seqlen, hvNum, kDim}), executorPtr),
                                 params.wOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(uBsnd, KdaFwdMakeShape({seqlen, hvNum, vDim}), executorPtr),
                                 params.uOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(qgBsnd, KdaFwdMakeShape({seqlen, hvNum, kDim}), executorPtr),
                                 params.qgOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(kgBsnd, KdaFwdMakeShape({seqlen, hvNum, kDim}), executorPtr),
                                 params.kgOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(vNewBsnd, KdaFwdMakeShape({seqlen, hvNum, vDim}), executorPtr),
                                 params.vNewOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(l0op::Reshape(hBsnt, KdaFwdMakeShape({params.totalChunks, hvNum, kDim, vDim}),
                                                executorPtr),
                                 params.hOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        CHECK_RET(l0op::KdaLayoutSwap12(result[0], params.oOut, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[2], params.aqkOut, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[3], params.akkOut, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[4], params.wOut, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[5], params.uOut, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[6], params.qgOut, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[7], params.kgOut, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[8], params.vNewOut, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[9], params.hOut, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(result[1], params.finalStateOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);

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
