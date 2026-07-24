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

#include <cstring>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
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
constexpr int64_t MAX_KDA_HEAD_NUM = 128;
constexpr int64_t MAX_KDA_VARLEN_SEQUENCES = 1024;

struct ChunkKdaFwdParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *gk = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const char *layout = "BSND";
    double scale = 1.0;
    int64_t chunkSize = 64;
    bool outputFinalState = false;
    bool safeGate = false;
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

int64_t KdaFwdNumel(const aclTensor *tensor)
{
    const auto shape = tensor->GetViewShape();
    int64_t numel = 1;
    for (size_t idx = 0; idx < shape.GetDimNum(); ++idx) {
        numel *= shape.GetDim(idx);
    }
    return numel;
}

size_t KdaFwdRank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

aclnnStatus KdaFwdCheckCuSeqlens(const aclIntArray *cuSeqlensOptional, int64_t seqlen)
{
    if (cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const aclIntArray &cu = *cuSeqlensOptional;
    CHECK_COND(cu.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must contain at least [0, total_tokens].");
    CHECK_COND(cu[0] == 0, ACLNN_ERR_PARAM_INVALID, "cuSeqlensOptional[0] must be 0.");
    CHECK_COND(cu[cu.Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional last element must equal the sequence length.");
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        CHECK_COND(cu[idx] <= cu[idx + 1], ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must be nondecreasing.");
    }
    return ACLNN_SUCCESS;
}

int64_t KdaFwdExpectedChunks(const aclIntArray *cuSeqlensOptional, int64_t seqlen, int64_t chunkSize)
{
    if (cuSeqlensOptional == nullptr) {
        return (seqlen + chunkSize - 1) / chunkSize;
    }
    int64_t total = 0;
    const aclIntArray &cu = *cuSeqlensOptional;
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        int64_t length = cu[idx + 1] - cu[idx];
        total += (length + chunkSize - 1) / chunkSize;
    }
    return total;
}

aclnnStatus KdaFwdCheckChunkIndices(const aclIntArray *chunkIndicesOptional,
                                    const aclIntArray *cuSeqlensOptional,
                                    int64_t totalChunks,
                                    int64_t expectedChunks,
                                    int64_t chunkSize)
{
    CHECK_COND(totalChunks == expectedChunks, ACLNN_ERR_PARAM_INVALID,
               "totalChunks must equal the number of chunks derived from sequence lengths and chunkSize.");
    if (chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(cuSeqlensOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional is only valid when cuSeqlensOptional is provided.");
    CHECK_COND(chunkIndicesOptional->Size() % 2 == 0, ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional must contain (seq_id, chunk_id) pairs.");
    CHECK_COND(static_cast<int64_t>(chunkIndicesOptional->Size() / 2) == expectedChunks,
               ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional must contain exactly totalChunks (seq_id, chunk_id) pairs.");
    const aclIntArray &indices = *chunkIndicesOptional;
    const aclIntArray &cu = *cuSeqlensOptional;
    int64_t seqNum = static_cast<int64_t>(cu.Size()) - 1;
    for (size_t idx = 0; idx < indices.Size(); idx += 2) {
        int64_t seq = indices[idx];
        int64_t localChunk = indices[idx + 1];
        CHECK_COND(seq >= 0 && seq < seqNum, ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional seq_id must be in [0, seq_num).");
        int64_t seqLength = cu[seq + 1] - cu[seq];
        int64_t seqChunks = (seqLength + chunkSize - 1) / chunkSize;
        CHECK_COND(localChunk >= 0 && localChunk < seqChunks, ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional chunk_id is outside the selected sequence.");
    }
    size_t expectedIdx = 0;
    for (int64_t seq = 0; seq < seqNum; ++seq) {
        int64_t seqLength = cu[seq + 1] - cu[seq];
        int64_t seqChunks = (seqLength + chunkSize - 1) / chunkSize;
        for (int64_t localChunk = 0; localChunk < seqChunks; ++localChunk) {
            CHECK_COND(indices[expectedIdx] == seq && indices[expectedIdx + 1] == localChunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOptional must use canonical sequence-major chunk order.");
            expectedIdx += 2;
        }
    }
    return ACLNN_SUCCESS;
}

int64_t KdaFwdSeqNum(int64_t batch, const aclIntArray *cuSeqlensOptional)
{
    if (cuSeqlensOptional == nullptr) {
        return batch;
    }
    return static_cast<int64_t>(cuSeqlensOptional->Size()) - 1;
}

aclnnStatus KdaFwdCheckStateShape(const aclTensor *state, const char *name, int64_t seqNum, int64_t hvNum,
                                  int64_t kDim, int64_t vDim)
{
    if (state == nullptr) {
        return ACLNN_SUCCESS;
    }
    const auto shape = state->GetViewShape();
    CHECK_COND(shape.GetDimNum() == 4 && shape.GetDim(0) == seqNum && shape.GetDim(1) == hvNum &&
                   shape.GetDim(2) == kDim && shape.GetDim(3) == vDim,
               ACLNN_ERR_PARAM_INVALID,
               "%s must be [seq_num, HV, K, V], where seq_num is batch for dense input or "
               "len(cuSeqlensOptional)-1 for varlen input.",
               name);
    return ACLNN_SUCCESS;
}

enum class KdaFwdLayout {
    BSND,
    BNSD,
    TND,
    NTD,
};

bool KdaFwdSameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (KdaFwdRank(lhs) != KdaFwdRank(rhs)) {
        return false;
    }
    for (size_t idx = 0; idx < KdaFwdRank(lhs); ++idx) {
        if (KdaFwdDim(lhs, idx) != KdaFwdDim(rhs, idx)) {
            return false;
        }
    }
    return true;
}

aclnnStatus KdaFwdParseLayout(const char *layout, KdaFwdLayout &parsed)
{
    CHECK_COND(layout != nullptr, ACLNN_ERR_PARAM_INVALID,
               "layout must not be nullptr and must be one of BSND, BNSD, TND, NTD.");
    if (std::strcmp(layout, "BSND") == 0) {
        parsed = KdaFwdLayout::BSND;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(layout, "BNSD") == 0) {
        parsed = KdaFwdLayout::BNSD;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(layout, "TND") == 0) {
        parsed = KdaFwdLayout::TND;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(layout, "NTD") == 0) {
        parsed = KdaFwdLayout::NTD;
        return ACLNN_SUCCESS;
    }
    CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
               "layout must be one of BSND, BNSD, TND, NTD and must be uppercase.");
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus KdaFwdCheckLayoutShape(const ChunkKdaFwdParams &params, KdaFwdLayout layout)
{
    CHECK_COND(KdaFwdSameShape(params.q, params.k), ACLNN_ERR_PARAM_INVALID,
               "q and k must have identical shape.");
    if (layout == KdaFwdLayout::TND) {
        CHECK_COND(KdaFwdRank(params.q) == 3 && KdaFwdRank(params.v) == 3 &&
                       KdaFwdRank(params.gk) == 3 && KdaFwdRank(params.beta) == 2,
                   ACLNN_ERR_PARAM_INVALID,
                   "layout TND expects q/k [T,H,K], v [T,HV,V], gk [T,HV,K], beta [T,HV].");
        CHECK_COND(KdaFwdDim(params.v, 0) == KdaFwdDim(params.q, 0) &&
                       KdaFwdDim(params.gk, 0) == KdaFwdDim(params.q, 0) &&
                       KdaFwdDim(params.beta, 0) == KdaFwdDim(params.q, 0) &&
                       KdaFwdDim(params.gk, 1) == KdaFwdDim(params.v, 1) &&
                       KdaFwdDim(params.beta, 1) == KdaFwdDim(params.v, 1) &&
                       KdaFwdDim(params.gk, 2) == KdaFwdDim(params.q, 2),
                   ACLNN_ERR_PARAM_INVALID,
                   "layout TND shape mismatch.");
    } else if (layout == KdaFwdLayout::NTD) {
        CHECK_COND(KdaFwdRank(params.q) == 3 && KdaFwdRank(params.v) == 3 &&
                       KdaFwdRank(params.gk) == 3 && KdaFwdRank(params.beta) == 2,
                   ACLNN_ERR_PARAM_INVALID,
                   "layout NTD expects q/k [H,T,K], v [HV,T,V], gk [HV,T,K], beta [HV,T].");
        CHECK_COND(KdaFwdDim(params.v, 1) == KdaFwdDim(params.q, 1) &&
                       KdaFwdDim(params.gk, 0) == KdaFwdDim(params.v, 0) &&
                       KdaFwdDim(params.beta, 0) == KdaFwdDim(params.v, 0) &&
                       KdaFwdDim(params.gk, 1) == KdaFwdDim(params.q, 1) &&
                       KdaFwdDim(params.beta, 1) == KdaFwdDim(params.q, 1) &&
                       KdaFwdDim(params.gk, 2) == KdaFwdDim(params.q, 2),
                   ACLNN_ERR_PARAM_INVALID,
                   "layout NTD shape mismatch.");
    } else if (layout == KdaFwdLayout::BSND) {
        CHECK_COND(KdaFwdRank(params.q) == 4 && KdaFwdRank(params.v) == 4 &&
                       KdaFwdRank(params.gk) == 4 && KdaFwdRank(params.beta) == 3,
                   ACLNN_ERR_PARAM_INVALID,
                   "layout BSND expects q/k [B,T,H,K], v [B,T,HV,V], gk [B,T,HV,K], beta [B,T,HV].");
        CHECK_COND(KdaFwdDim(params.v, 0) == KdaFwdDim(params.q, 0) &&
                       KdaFwdDim(params.v, 1) == KdaFwdDim(params.q, 1) &&
                       KdaFwdDim(params.gk, 0) == KdaFwdDim(params.q, 0) &&
                       KdaFwdDim(params.gk, 1) == KdaFwdDim(params.q, 1) &&
                       KdaFwdDim(params.beta, 0) == KdaFwdDim(params.q, 0) &&
                       KdaFwdDim(params.beta, 1) == KdaFwdDim(params.q, 1) &&
                       KdaFwdDim(params.gk, 2) == KdaFwdDim(params.v, 2) &&
                       KdaFwdDim(params.beta, 2) == KdaFwdDim(params.v, 2) &&
                       KdaFwdDim(params.gk, 3) == KdaFwdDim(params.q, 3),
                   ACLNN_ERR_PARAM_INVALID,
                   "layout BSND shape mismatch.");
    } else {
        CHECK_COND(KdaFwdRank(params.q) == 4 && KdaFwdRank(params.v) == 4 &&
                       KdaFwdRank(params.gk) == 4 && KdaFwdRank(params.beta) == 3,
                   ACLNN_ERR_PARAM_INVALID,
                   "layout BNSD expects q/k [B,H,T,K], v [B,HV,T,V], gk [B,HV,T,K], beta [B,HV,T].");
        CHECK_COND(KdaFwdDim(params.v, 0) == KdaFwdDim(params.q, 0) &&
                       KdaFwdDim(params.v, 2) == KdaFwdDim(params.q, 2) &&
                       KdaFwdDim(params.gk, 0) == KdaFwdDim(params.q, 0) &&
                       KdaFwdDim(params.gk, 1) == KdaFwdDim(params.v, 1) &&
                       KdaFwdDim(params.beta, 0) == KdaFwdDim(params.q, 0) &&
                       KdaFwdDim(params.beta, 1) == KdaFwdDim(params.v, 1) &&
                       KdaFwdDim(params.gk, 2) == KdaFwdDim(params.q, 2) &&
                       KdaFwdDim(params.beta, 2) == KdaFwdDim(params.q, 2) &&
                       KdaFwdDim(params.gk, 3) == KdaFwdDim(params.q, 3),
                   ACLNN_ERR_PARAM_INVALID,
                   "layout BNSD shape mismatch.");
    }
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
    const bool returnIntermediates = KdaFwdNumel(params.aqkOut) != 0;
    const aclTensor *intermediateOutputs[] = {
        params.akkOut, params.wOut, params.uOut, params.qgOut,
        params.kgOut, params.vNewOut, params.hOut};
    for (const aclTensor *output : intermediateOutputs) {
        CHECK_COND((KdaFwdNumel(output) != 0) == returnIntermediates, ACLNN_ERR_PARAM_INVALID,
                   "Aqk, Akk, w, u, qg, kg, v_new and h outputs must be all present or all empty.");
    }
    CHECK_COND(params.chunkSize > 0, ACLNN_ERR_PARAM_INVALID, "chunkSize must be positive.");
    CHECK_COND(params.totalChunks > 0, ACLNN_ERR_PARAM_INVALID, "totalChunks must be positive.");
    size_t qRank = KdaFwdRank(params.q);
    size_t betaRank = KdaFwdRank(params.beta);
    CHECK_COND((qRank == 4 && betaRank == 3) || (qRank == 3 && betaRank == 2), ACLNN_ERR_PARAM_INVALID,
               "q/k/v/gk must be BSND/BNSD rank4 with beta rank3, or TND/NTD rank3 with beta rank2.");
    size_t kDimIdx = (qRank == 4) ? 3 : 2;
    CHECK_COND(params.q->GetViewShape().GetDim(kDimIdx) <= MAX_KDA_K_DIM, ACLNN_ERR_PARAM_INVALID,
               "k head dimension must be less than or equal to 256.");
    return ACLNN_SUCCESS;
}

bool KdaFwdSplitCubePathSupported(const ChunkKdaFwdParams &params, int64_t kDim, int64_t vDim)
{
    auto qDtype = params.q->GetDataType();
    auto kDtype = params.k->GetDataType();
    auto vDtype = params.v->GetDataType();
    bool dataDtypeSupported = (qDtype == DataType::DT_FLOAT16 || qDtype == DataType::DT_BF16) &&
                              kDtype == qDtype && vDtype == qDtype;
    return dataDtypeSupported &&
           (params.chunkSize == 64 || params.chunkSize == 128) && kDim >= 16 && vDim >= 16 &&
           kDim % 16 == 0 && vDim % 16 == 0 && vDim <= 256;
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
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool outputFinalState,
    bool safeGate,
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
    ChunkKdaFwdParams params{q, k, v, gk, beta, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional, layout,
                             scale, chunkSize, outputFinalState, safeGate, totalChunks, oOut, finalStateOut, aqkOut,
                             akkOut, wOut, uOut, qgOut, kgOut, vNewOut, hOut};
    L2_DFX_PHASE_1(aclnnChunkKdaFwd,
                   DFX_IN(q, k, v, gk, beta, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(oOut, finalStateOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut, vNewOut, hOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(KdaFwdCheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFwdParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    KdaFwdLayout parsedLayout = KdaFwdLayout::BSND;
    CHECK_RET(KdaFwdParseLayout(params.layout, parsedLayout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFwdCheckLayoutShape(params, parsedLayout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    bool isTnd = parsedLayout == KdaFwdLayout::TND || parsedLayout == KdaFwdLayout::NTD;
    bool isInternalLayout = parsedLayout == KdaFwdLayout::BNSD || parsedLayout == KdaFwdLayout::NTD;
    int64_t batch = isTnd ? 1 : KdaFwdDim(params.q, 0);
    int64_t seqlen = parsedLayout == KdaFwdLayout::TND ? KdaFwdDim(params.q, 0) :
                     (parsedLayout == KdaFwdLayout::NTD ? KdaFwdDim(params.q, 1) :
                     (parsedLayout == KdaFwdLayout::BNSD ? KdaFwdDim(params.q, 2) : KdaFwdDim(params.q, 1)));
    int64_t hNum = parsedLayout == KdaFwdLayout::TND ? KdaFwdDim(params.q, 1) :
                   (parsedLayout == KdaFwdLayout::NTD ? KdaFwdDim(params.q, 0) :
                   (parsedLayout == KdaFwdLayout::BNSD ? KdaFwdDim(params.q, 1) : KdaFwdDim(params.q, 2)));
    int64_t kDim = isTnd ? KdaFwdDim(params.q, 2) : KdaFwdDim(params.q, 3);
    int64_t hvNum = parsedLayout == KdaFwdLayout::TND ? KdaFwdDim(params.v, 1) :
                    (parsedLayout == KdaFwdLayout::NTD ? KdaFwdDim(params.v, 0) :
                    (parsedLayout == KdaFwdLayout::BNSD ? KdaFwdDim(params.v, 1) : KdaFwdDim(params.v, 2)));
    int64_t vDim = isTnd ? KdaFwdDim(params.v, 2) : KdaFwdDim(params.v, 3);
    int64_t seqNum = KdaFwdSeqNum(batch, params.cuSeqlensOptional);
    CHECK_COND(hNum <= MAX_KDA_HEAD_NUM && hvNum <= MAX_KDA_HEAD_NUM, ACLNN_ERR_PARAM_INVALID,
               "H and HV must be less than or equal to 128.");
    CHECK_COND(hNum > 0 && hvNum >= hNum && hvNum % hNum == 0, ACLNN_ERR_PARAM_INVALID,
               "H and HV must be positive, HV must be greater than or equal to H, and HV must be divisible by H.");
    CHECK_COND(parsedLayout != KdaFwdLayout::TND || hNum == 1, ACLNN_ERR_PARAM_INVALID,
               "TND layout with H > 1 is not supported by npu_chunk_kda_fwd; use NTD [H,T,D] layout "
               "for multi-head rank3 input.");
    CHECK_RET(KdaFwdCheckCuSeqlens(params.cuSeqlensOptional, seqlen) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    int64_t expectedChunks = KdaFwdExpectedChunks(params.cuSeqlensOptional, seqlen, params.chunkSize);
    CHECK_COND(params.cuSeqlensOptional == nullptr || seqNum <= MAX_KDA_VARLEN_SEQUENCES,
               ACLNN_ERR_PARAM_INVALID,
               "varlen input supports at most 1024 sequences in one call; split a larger request at sequence "
               "boundaries.");
    CHECK_RET(KdaFwdCheckChunkIndices(params.chunkIndicesOptional, params.cuSeqlensOptional, params.totalChunks,
                                      expectedChunks, params.chunkSize) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(params.cuSeqlensOptional == nullptr || isTnd || batch == 1, ACLNN_ERR_PARAM_INVALID,
               "rank4 varlen input with cuSeqlensOptional currently requires B=1.");
    CHECK_RET(KdaFwdCheckStateShape(params.initialStateOptional, "initialStateOptional", seqNum, hvNum, kDim, vDim) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFwdCheckStateShape(params.finalStateOut, "finalStateOut", seqNum, hvNum, kDim, vDim) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(KdaFwdSplitCubePathSupported(params, kDim, vDim), ACLNN_ERR_PARAM_INVALID,
               "npu_chunk_kda_fwd only supports the AscendC split cube/vector path: q/k/v dtype must be the same "
               "fp16/bf16 type, chunkSize must be 64 or 128, K/V must be multiples of 16, and V must be <= 256.");

    const aclTensor *qBsnd = params.q;
    const aclTensor *kBsnd = params.k;
    const aclTensor *vBsnd = params.v;
    const aclTensor *gkBsnd = params.gk;
    const aclTensor *betaBsn = params.beta;
    if (parsedLayout == KdaFwdLayout::TND) {
        qBsnd = l0op::Reshape(params.q, KdaFwdMakeShape({1, seqlen, hNum, kDim}), executorPtr);
        kBsnd = l0op::Reshape(params.k, KdaFwdMakeShape({1, seqlen, hNum, kDim}), executorPtr);
        vBsnd = l0op::Reshape(params.v, KdaFwdMakeShape({1, seqlen, hvNum, vDim}), executorPtr);
        gkBsnd = l0op::Reshape(params.gk, KdaFwdMakeShape({1, seqlen, hvNum, kDim}), executorPtr);
        betaBsn = l0op::Reshape(params.beta, KdaFwdMakeShape({1, seqlen, hvNum}), executorPtr);
        CHECK_RET(qBsnd != nullptr && kBsnd != nullptr && vBsnd != nullptr && gkBsnd != nullptr && betaBsn != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    } else if (parsedLayout == KdaFwdLayout::NTD) {
        qBsnd = l0op::Reshape(params.q, KdaFwdMakeShape({1, hNum, seqlen, kDim}), executorPtr);
        kBsnd = l0op::Reshape(params.k, KdaFwdMakeShape({1, hNum, seqlen, kDim}), executorPtr);
        vBsnd = l0op::Reshape(params.v, KdaFwdMakeShape({1, hvNum, seqlen, vDim}), executorPtr);
        gkBsnd = l0op::Reshape(params.gk, KdaFwdMakeShape({1, hvNum, seqlen, kDim}), executorPtr);
        betaBsn = l0op::Reshape(params.beta, KdaFwdMakeShape({1, hvNum, seqlen}), executorPtr);
        CHECK_RET(qBsnd != nullptr && kBsnd != nullptr && vBsnd != nullptr && gkBsnd != nullptr && betaBsn != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *qBnsd = isInternalLayout ? qBsnd :
        executorPtr->AllocTensor(KdaFwdMakeShape({batch, hNum, seqlen, kDim}),
                                 params.q->GetDataType(), Format::FORMAT_ND);
    const aclTensor *kBnsd = isInternalLayout ? kBsnd :
        executorPtr->AllocTensor(KdaFwdMakeShape({batch, hNum, seqlen, kDim}),
                                 params.k->GetDataType(), Format::FORMAT_ND);
    const aclTensor *vBnsd = isInternalLayout ? vBsnd :
        executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, vDim}),
                                 params.v->GetDataType(), Format::FORMAT_ND);
    const aclTensor *gkBnsdRaw = isInternalLayout ? gkBsnd :
        executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen, kDim}),
                                 params.gk->GetDataType(), Format::FORMAT_ND);
    const aclTensor *betaBnsRaw = isInternalLayout ? betaBsn :
        executorPtr->AllocTensor(KdaFwdMakeShape({batch, hvNum, seqlen}),
                                 params.beta->GetDataType(), Format::FORMAT_ND);
    bool returnIntermediates = KdaFwdNumel(params.aqkOut) != 0;
    const aclTensor *oBnsd = nullptr;
    const aclTensor *aqkBnst = nullptr;
    const aclTensor *akkBnst = nullptr;
    const aclTensor *wBnsd = nullptr;
    const aclTensor *uBnsd = nullptr;
    const aclTensor *qgBnsd = nullptr;
    const aclTensor *kgBnsd = nullptr;
    const aclTensor *vNewBnsd = nullptr;
    const aclTensor *hBnst = nullptr;
    oBnsd = params.oOut;
    if (parsedLayout == KdaFwdLayout::NTD) {
        oBnsd = l0op::Reshape(params.oOut, KdaFwdMakeShape({1, hvNum, seqlen, vDim}), executorPtr);
    }
    if (returnIntermediates) {
        // Canonical graph views preserve BNSD metadata between independently queued stages even when
        // the caller's output tensor has a flat physical storage shape.
        aqkBnst = l0op::Reshape(params.aqkOut,
                                KdaFwdMakeShape({batch, hvNum, seqlen, params.chunkSize}), executorPtr);
        akkBnst = l0op::Reshape(params.akkOut,
                                KdaFwdMakeShape({batch, hvNum, seqlen, params.chunkSize}), executorPtr);
        wBnsd = l0op::Reshape(params.wOut, KdaFwdMakeShape({batch, hvNum, seqlen, kDim}), executorPtr);
        uBnsd = l0op::Reshape(params.uOut, KdaFwdMakeShape({batch, hvNum, seqlen, vDim}), executorPtr);
        qgBnsd = l0op::Reshape(params.qgOut, KdaFwdMakeShape({batch, hvNum, seqlen, kDim}), executorPtr);
        kgBnsd = l0op::Reshape(params.kgOut, KdaFwdMakeShape({batch, hvNum, seqlen, kDim}), executorPtr);
        vNewBnsd = l0op::Reshape(params.vNewOut, KdaFwdMakeShape({batch, hvNum, seqlen, vDim}), executorPtr);
        hBnst = l0op::Reshape(params.hOut,
                              KdaFwdMakeShape({batch, hvNum, params.totalChunks, kDim, vDim}), executorPtr);
    }
    const bool internalIntermediateOutputsReady = !returnIntermediates ||
        (aqkBnst != nullptr && akkBnst != nullptr && wBnsd != nullptr && uBnsd != nullptr &&
         qgBnsd != nullptr && kgBnsd != nullptr && vNewBnsd != nullptr && hBnst != nullptr);
    CHECK_RET(qBnsd != nullptr && kBnsd != nullptr && vBnsd != nullptr && gkBnsdRaw != nullptr &&
                  betaBnsRaw != nullptr && oBnsd != nullptr && internalIntermediateOutputsReady,
              ACLNN_ERR_INNER_NULLPTR);

    if (!isInternalLayout) {
        CHECK_RET(l0op::KdaLayoutSwap12(qBsnd, qBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(kBsnd, kBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(vBsnd, vBnsd, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(gkBsnd, gkBnsdRaw, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(betaBsn, betaBnsRaw, executorPtr)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *gkBnsd = gkBnsdRaw;
    const aclTensor *betaBns = betaBnsRaw;

    std::array<const aclTensor *, 10> result;
    const aclTensor *aqkComputeBnst = aqkBnst;
    const aclTensor *akkComputeBnst = akkBnst;
    const aclTensor *wComputeBnsd = wBnsd;
    const aclTensor *uComputeBnsd = uBnsd;
    const aclTensor *qgComputeBnsd = qgBnsd;
    const aclTensor *kgComputeBnsd = kgBnsd;
    const aclTensor *vNewComputeBnsd = vNewBnsd;
    const aclTensor *hComputeBnst = hBnst;
    const aclTensor *finalStateCompute = params.finalStateOut;

    if (!returnIntermediates) {
        aqkComputeBnst = executorPtr->AllocTensor(
            KdaFwdMakeShape({batch, hvNum, seqlen, params.chunkSize}),
            qBnsd->GetDataType(), Format::FORMAT_ND);
        akkComputeBnst = executorPtr->AllocTensor(
            KdaFwdMakeShape({batch, hvNum, seqlen, params.chunkSize}),
            qBnsd->GetDataType(), Format::FORMAT_ND);
        wComputeBnsd = executorPtr->AllocTensor(
            KdaFwdMakeShape({batch, hvNum, seqlen, kDim}), qBnsd->GetDataType(), Format::FORMAT_ND);
        uComputeBnsd = executorPtr->AllocTensor(
            KdaFwdMakeShape({batch, hvNum, seqlen, vDim}), qBnsd->GetDataType(), Format::FORMAT_ND);
        qgComputeBnsd = executorPtr->AllocTensor(
            KdaFwdMakeShape({batch, hvNum, seqlen, kDim}), qBnsd->GetDataType(), Format::FORMAT_ND);
        kgComputeBnsd = executorPtr->AllocTensor(
            KdaFwdMakeShape({batch, hvNum, seqlen, kDim}), qBnsd->GetDataType(), Format::FORMAT_ND);
        vNewComputeBnsd = executorPtr->AllocTensor(
            KdaFwdMakeShape({batch, hvNum, seqlen, vDim}), qBnsd->GetDataType(), Format::FORMAT_ND);
        hComputeBnst = executorPtr->AllocTensor(
            KdaFwdMakeShape({batch, hvNum, params.totalChunks, kDim, vDim}),
            qBnsd->GetDataType(), Format::FORMAT_ND);
    }
    if (!params.outputFinalState) {
        finalStateCompute = executorPtr->AllocTensor(
            KdaFwdMakeShape({seqNum, hvNum, kDim, vDim}),
            DataType::DT_FLOAT, Format::FORMAT_ND);
    }
    CHECK_RET(aqkComputeBnst != nullptr && akkComputeBnst != nullptr && wComputeBnsd != nullptr &&
                  uComputeBnsd != nullptr && qgComputeBnsd != nullptr && kgComputeBnsd != nullptr &&
                  vNewComputeBnsd != nullptr && hComputeBnst != nullptr && finalStateCompute != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    result = l0op::KdaChunkForward(
        qBnsd, kBnsd, vBnsd, gkBnsd, betaBns, params.initialStateOptional,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.scale, params.chunkSize,
        params.outputFinalState, params.totalChunks, params.safeGate, !isInternalLayout, oBnsd, finalStateCompute,
        aqkComputeBnst, akkComputeBnst, wComputeBnsd, uComputeBnsd, qgComputeBnsd,
        kgComputeBnsd, vNewComputeBnsd, hComputeBnst, executorPtr);
    for (auto tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
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
