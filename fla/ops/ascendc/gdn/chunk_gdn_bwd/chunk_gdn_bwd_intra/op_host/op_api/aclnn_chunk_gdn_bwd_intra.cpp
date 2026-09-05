/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "aclnn_chunk_gdn_bwd_intra.h"
#include "chunk_gdn_bwd_intra.h"

#include <array>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

struct Params {
    const aclTensor *q;
    const aclTensor *k;
    const aclTensor *v;
    const aclTensor *g;
    const aclTensor *beta;
    const aclTensor *a;
    const aclTensor *dO;
    const aclIntArray *cuSeqlens;
    const aclIntArray *chunkIndices;
    double scale;
    int64_t chunkSize;
    bool useExp2;
    int64_t stage;
    const aclTensor *wOut;
    const aclTensor *uOut;
    const aclTensor *dvLocalOut;
};

bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    const auto a = lhs->GetViewShape();
    const auto b = rhs->GetViewShape();
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }
    return true;
}

aclnnStatus CheckMetadata(const Params &p, int64_t totalTokens)
{
    const bool hasCu = p.cuSeqlens != nullptr;
    CHECK_COND(hasCu == (p.chunkIndices != nullptr), ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must either both be provided or both be nullptr.");
    if (!hasCu) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(p.cuSeqlens->Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must contain at least two values.");
    CHECK_COND((*p.cuSeqlens)[0] == 0, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must start at zero.");
    CHECK_COND((*p.cuSeqlens)[p.cuSeqlens->Size() - 1] == totalTokens,
               ACLNN_ERR_PARAM_INVALID, "cu_seqlens must end at T.");
    size_t expectedValues = 0;
    for (size_t seq = 0; seq + 1 < p.cuSeqlens->Size(); ++seq) {
        const int64_t begin = (*p.cuSeqlens)[seq];
        const int64_t end = (*p.cuSeqlens)[seq + 1];
        CHECK_COND(end >= begin, ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens must be nondecreasing.");
        expectedValues += static_cast<size_t>((end - begin + p.chunkSize - 1) / p.chunkSize) * 2;
    }
    CHECK_COND(p.chunkIndices->Size() == expectedValues, ACLNN_ERR_PARAM_INVALID,
               "chunk_indices must contain one [sequence, local_chunk] pair per chunk.");
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < p.cuSeqlens->Size(); ++seq) {
        const int64_t length = (*p.cuSeqlens)[seq + 1] - (*p.cuSeqlens)[seq];
        const int64_t count = (length + p.chunkSize - 1) / p.chunkSize;
        for (int64_t chunk = 0; chunk < count; ++chunk) {
            CHECK_COND((*p.chunkIndices)[offset] == static_cast<int64_t>(seq) &&
                           (*p.chunkIndices)[offset + 1] == chunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunk_indices must use canonical sequence-major order.");
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Check(const Params &p)
{
    const std::array<const aclTensor *, 10> tensors = {
        p.q, p.k, p.v, p.g, p.beta, p.a, p.dO, p.wOut, p.uOut, p.dvLocalOut};
    for (const aclTensor *tensor : tensors) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "ChunkGdnBwdIntra tensor arguments must not be nullptr.");
        CHECK_COND(IsContiguous(tensor), ACLNN_ERR_PARAM_INVALID,
                   "ChunkGdnBwdIntra only supports contiguous BNSD tensors.");
    }
    CHECK_COND(p.chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "chunk_size must be 64.");
    CHECK_COND(p.stage >= 0 && p.stage <= 2, ACLNN_ERR_PARAM_INVALID,
               "stage must be 0, 1, or 2.");

    const auto qShape = p.q->GetViewShape();
    const auto vShape = p.v->GetViewShape();
    const auto gShape = p.g->GetViewShape();
    const auto aShape = p.a->GetViewShape();
    CHECK_COND(qShape.GetDimNum() == 4 && vShape.GetDimNum() == 4 &&
                   gShape.GetDimNum() == 3 && aShape.GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID,
               "q/k/v/A/d_o must be rank 4 and g/beta rank 3 in BNSD layout.");
    CHECK_COND(SameShape(p.q, p.k), ACLNN_ERR_PARAM_INVALID,
               "q and k must have the same shape.");
    CHECK_COND(SameShape(p.v, p.dO) && SameShape(p.v, p.uOut) &&
                   SameShape(p.v, p.dvLocalOut), ACLNN_ERR_PARAM_INVALID,
               "v, d_o, u and dv_local must have the same shape.");
    CHECK_COND(SameShape(p.g, p.beta), ACLNN_ERR_PARAM_INVALID,
               "g and beta must have the same shape.");

    const int64_t b = qShape.GetDim(0);
    const int64_t hk = qShape.GetDim(1);
    const int64_t t = qShape.GetDim(2);
    const int64_t kDim = qShape.GetDim(3);
    const int64_t hv = vShape.GetDim(1);
    CHECK_COND(b > 0 && hk > 0 && hv > 0 && t > 0, ACLNN_ERR_PARAM_INVALID,
               "B, HK, HV and T must be positive.");
    CHECK_COND(kDim == 128 && vShape.GetDim(3) == 128, ACLNN_ERR_PARAM_INVALID,
               "K and V must both be 128.");
    CHECK_COND(vShape.GetDim(0) == b && vShape.GetDim(2) == t,
               ACLNN_ERR_PARAM_INVALID, "q/k and value tensors must share B and T.");
    CHECK_COND(hv % hk == 0 && hv / hk <= 4, ACLNN_ERR_PARAM_INVALID,
               "HV/HK must be an integer in [1, 4].");
    CHECK_COND(gShape.GetDim(0) == b && gShape.GetDim(1) == hv && gShape.GetDim(2) == t,
               ACLNN_ERR_PARAM_INVALID, "g/beta shape must be [B, HV, T].");
    CHECK_COND(aShape.GetDim(0) == b && aShape.GetDim(1) == hv &&
                   aShape.GetDim(2) == t && aShape.GetDim(3) == p.chunkSize,
               ACLNN_ERR_PARAM_INVALID, "A shape must be [B, HV, T, chunk_size].");
    const auto wShape = p.wOut->GetViewShape();
    CHECK_COND(wShape.GetDimNum() == 4 && wShape.GetDim(0) == b &&
                   wShape.GetDim(1) == hv && wShape.GetDim(2) == t &&
                   wShape.GetDim(3) == kDim,
               ACLNN_ERR_PARAM_INVALID, "w shape must be [B, HV, T, K].");

    const DataType mainType = p.q->GetDataType();
    CHECK_COND(mainType == DataType::DT_BF16 || mainType == DataType::DT_FLOAT16,
               ACLNN_ERR_PARAM_INVALID, "q/k/v/A/d_o must be BF16 or FP16.");
    const std::array<const aclTensor *, 7> mainTensors = {
        p.k, p.v, p.a, p.dO, p.wOut, p.uOut, p.dvLocalOut};
    for (const aclTensor *tensor : mainTensors) {
        CHECK_COND(tensor->GetDataType() == mainType, ACLNN_ERR_PARAM_INVALID,
                   "all main tensors and outputs must use the same dtype.");
    }
    const DataType gateType = p.g->GetDataType();
    CHECK_COND(gateType == DataType::DT_BF16 || gateType == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "g must be BF16 or FP32.");
    const DataType betaType = p.beta->GetDataType();
    CHECK_COND(betaType == DataType::DT_BF16 || betaType == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "beta must be BF16 or FP32.");
    CHECK_RET(CheckMetadata(p, t) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(p.cuSeqlens == nullptr || b == 1, ACLNN_ERR_PARAM_INVALID,
               "varlen BNSD requires B=1.");
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnChunkGdnBwdIntraGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta, const aclTensor *a,
    const aclTensor *dO, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, double scale, int64_t chunkSize,
    bool useExp2, int64_t stage, const aclTensor *wOut, const aclTensor *uOut,
    const aclTensor *dvLocalOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGdnBwdIntra,
                   DFX_IN(q, k, v, g, beta, a, dO, cuSeqlensOptional,
                          chunkIndicesOptional, scale, chunkSize, useExp2, stage),
                   DFX_OUT(wOut, uOut, dvLocalOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "workspaceSize and executor must not be nullptr.");
    const Params params{q, k, v, g, beta, a, dO, cuSeqlensOptional,
                        chunkIndicesOptional, scale, chunkSize, useExp2, stage,
                        wOut, uOut, dvLocalOut};
    CHECK_RET(Check(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();
    const auto result = l0op::ChunkGdnBwdIntra(
        q, k, v, g, beta, a, dO, cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, useExp2, stage, wOut, uOut, dvLocalOut, executorPtr);
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkGdnBwdIntra(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGdnBwdIntra);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkGdnBwdIntra launch failed.");
    return ACLNN_SUCCESS;
}
