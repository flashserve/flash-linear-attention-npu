/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "aclnn_chunk_gdn_bwd_intra.h"
#include "chunk_gdn_bwd_intra.h"

#include <array>
#include <set>
#include <utility>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "op_common/log/log.h"
#include <string>

static constexpr const char *OP_NAME = "aclnnChunkGdnBwdIntra";

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

aclnnStatus Check(const Params &p)
{
    const std::array<const aclTensor *, 10> tensors = {
        p.q, p.k, p.v, p.g, p.beta, p.a, p.dO, p.wOut, p.uOut, p.dvLocalOut};
    for (const aclTensor *tensor : tensors) {
        if (!(tensor != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "arguments");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
        if (!(IsContiguous(tensor))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "input tensors", "non-contiguous", "ChunkGdnBwdIntra only supports contiguous BNSD tensors");
        return ACLNN_ERR_PARAM_INVALID;
    }
    }
    if (!(p.chunkSize == 64)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "chunk_size", std::to_string(p.chunkSize), "chunk_size must be 64.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto qShape = p.q->GetViewShape();
    const auto vShape = p.v->GetViewShape();
    const auto gShape = p.g->GetViewShape();
    const auto aShape = p.a->GetViewShape();
    if (!(qShape.GetDimNum() == 4 && vShape.GetDimNum() == 4 &&
                   gShape.GetDimNum() == 3 && aShape.GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(OP_NAME, "layout", "invalid", "q/k/v/A/d_o must be rank 4 and g/beta rank 3 in BNSD layout.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(SameShape(p.q, p.k))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "q", "invalid", "q and k must have the same shape.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(SameShape(p.v, p.dO) && SameShape(p.v, p.uOut) &&
                   SameShape(p.v, p.dvLocalOut))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "v", "invalid", "v, d_o, u and dv_local must have the same shape.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(SameShape(p.g, p.beta))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "g", "invalid", "g and beta must have the same shape.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t b = qShape.GetDim(0);
    const int64_t hk = qShape.GetDim(1);
    const int64_t t = qShape.GetDim(2);
    const int64_t kDim = qShape.GetDim(3);
    const int64_t hv = vShape.GetDim(1);
    if (!(b > 0 && hk > 0 && hv > 0 && t > 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "HV", "invalid", "B, HK, HV and T must be positive.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(kDim == 128 && vShape.GetDim(3) == 128)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "K", "invalid", "K and V must both be 128.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(vShape.GetDim(0) == b && vShape.GetDim(2) == t)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "q", "invalid", "q/k and value tensors must share B and T.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(hv % hk == 0 && hv / hk <= 4)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "HV", "invalid", "HV/HK must be an integer in [1, 4].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(gShape.GetDim(0) == b && gShape.GetDim(1) == hv && gShape.GetDim(2) == t)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "HV", "invalid", "g/beta shape must be [B, HV, T].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(aShape.GetDim(0) == b && aShape.GetDim(1) == hv &&
                   aShape.GetDim(2) == t && aShape.GetDim(3) == p.chunkSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "chunk_size", "invalid", "A shape must be [B, HV, T, chunk_size].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto wShape = p.wOut->GetViewShape();
    if (!(wShape.GetDimNum() == 4 && wShape.GetDim(0) == b &&
                   wShape.GetDim(1) == hv && wShape.GetDim(2) == t &&
                   wShape.GetDim(3) == kDim)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "HV", "invalid", "w shape must be [B, HV, T, K].");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const DataType mainType = p.q->GetDataType();
    if (!(mainType == DataType::DT_BF16 || mainType == DataType::DT_FLOAT16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "q", "invalid", "q/k/v/A/d_o must be BF16 or FP16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const std::array<const aclTensor *, 7> mainTensors = {
        p.k, p.v, p.a, p.dO, p.wOut, p.uOut, p.dvLocalOut};
    for (const aclTensor *tensor : mainTensors) {
        if (!(tensor->GetDataType() == mainType)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "dtype", "invalid", "all main tensors and outputs must use the same dtype.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    }
    const DataType gateType = p.g->GetDataType();
    if (!(gateType == DataType::DT_BF16 || gateType == DataType::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "g", "invalid", "g must be BF16 or FP32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const DataType betaType = p.beta->GetDataType();
    if (!(betaType == DataType::DT_BF16 || betaType == DataType::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "beta", "invalid", "beta must be BF16 or FP32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(p.cuSeqlens != nullptr) == (p.chunkIndices != nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "cu_seqlens", "cu_seqlens and chunk_indices must be provided together.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (p.cuSeqlens != nullptr) {
        if (!(b == 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "B", std::to_string(b), "varlen BNSD requires B=1.");
        return ACLNN_ERR_PARAM_INVALID;
    }
        if (!(p.cuSeqlens->Size() >= 2)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "cu_seqlens", std::to_string(p.cuSeqlens->Size()), "cu_seqlens must contain at least two boundaries.");
        return ACLNN_ERR_PARAM_INVALID;
    }
        if (!(p.chunkIndices != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "cu_seqlens");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
        if (!(p.chunkIndices->Size() > 0 && p.chunkIndices->Size() % 2 == 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "chunk_indices", "invalid", "chunk_indices flattened length must be a positive multiple of 2.");
        return ACLNN_ERR_PARAM_INVALID;
    }
        const aclIntArray &cu = *p.cuSeqlens;
        const aclIntArray &indices = *p.chunkIndices;
        if (!(cu[0] == 0 && cu[cu.Size() - 1] == t)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "cu_seqlens", "invalid", "cu_seqlens must span [0,T].");
        return ACLNN_ERR_PARAM_INVALID;
    }
        for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
            if (!(cu[seq] >= 0 && cu[seq + 1] >= cu[seq])) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "cu_seqlens", "invalid", "cu_seqlens must be nonnegative and nondecreasing.");
        return ACLNN_ERR_PARAM_INVALID;
    }
        }
        std::set<std::pair<int64_t, int64_t>> seen;
        for (size_t offset = 0; offset < indices.Size(); offset += 2) {
            const int64_t seq = indices[offset];
            const int64_t chunk = indices[offset + 1];
            if (!(seq >= 0 && static_cast<uint64_t>(seq) < cu.Size() - 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "index", "invalid", "chunk sequence index is out of bounds.");
        return ACLNN_ERR_PARAM_INVALID;
    }
            const int64_t length = cu[seq + 1] - cu[seq];
            const int64_t chunks = length / p.chunkSize + (length % p.chunkSize != 0);
            if (!(chunk >= 0 && chunk < chunks)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "index", "invalid", "local chunk index is out of bounds.");
        return ACLNN_ERR_PARAM_INVALID;
    }
            if (!(seen.emplace(seq, chunk).second)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "chunkIndices", "duplicate", "duplicate chunk indices would write overlapping outputs.");
        return ACLNN_ERR_PARAM_INVALID;
    }
        }
    }
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnChunkGdnBwdIntraGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta, const aclTensor *a,
    const aclTensor *dO, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, double scale, int64_t chunkSize,
    bool useExp2, const aclTensor *wOut, const aclTensor *uOut,
    const aclTensor *dvLocalOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGdnBwdIntra,
                   DFX_IN(q, k, v, g, beta, a, dO, cuSeqlensOptional,
                          chunkIndicesOptional, scale, chunkSize, useExp2),
                   DFX_OUT(wOut, uOut, dvLocalOut));
    if (!(workspaceSize != nullptr && executor != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const Params params{q, k, v, g, beta, a, dO, cuSeqlensOptional,
                        chunkIndicesOptional, scale, chunkSize, useExp2,
                        wOut, uOut, dvLocalOut};
    if (!(Check(params) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    aclOpExecutor *executorPtr = uniqueExecutor.get();
    const auto result = l0op::ChunkGdnBwdIntra(
        q, k, v, g, beta, a, dO, cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, useExp2, wOut, uOut, dvLocalOut, executorPtr);
    for (const aclTensor *tensor : result) {
        if (!(tensor != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: tensor != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkGdnBwdIntra(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGdnBwdIntra);
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkGdnBwdIntra launch failed.", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
