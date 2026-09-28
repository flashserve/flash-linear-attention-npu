/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_chunk_scaled_dot_kkt.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
struct ChunkScaledDotKktParams {
    const aclTensor *k;
    const aclTensor *g;
    const aclTensor *beta;
    const aclIntArray *cuSeqlens;
    const aclIntArray *chunkIndices;
    int64_t chunkSize;
    const aclTensor *a;
};

aclnnStatus CheckShape(const ChunkScaledDotKktParams &p)
{
    const auto &k = p.k->GetViewShape();
    const auto &g = p.g->GetViewShape();
    const auto &beta = p.beta->GetViewShape();
    const auto &a = p.a->GetViewShape();
    CHECK_COND(k.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "k must be [B,Hk,T,K] (rank 4).");
    CHECK_COND(g.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID, "g must be [B,Hv,T] (rank 3).");
    CHECK_COND(beta.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID, "beta must be [B,Hv,T] (rank 3).");
    CHECK_COND(a.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "A must be [B,Hv,T,chunkSize] (rank 4).");

    constexpr int64_t maxDim = std::numeric_limits<int32_t>::max();
    for (size_t dim = 0; dim < 4; ++dim) {
        CHECK_COND(k.GetDim(dim) > 0 && k.GetDim(dim) <= maxDim, ACLNN_ERR_PARAM_INVALID,
                   "k dimension %zu must be in [1, %ld], got %ld.", dim, maxDim, k.GetDim(dim));
    }
    CHECK_COND(k.GetDim(3) >= 8, ACLNN_ERR_PARAM_INVALID,
               "k feature dimension K must be at least 8, got %ld.", k.GetDim(3));
    const int64_t b = k.GetDim(0);
    const int64_t hk = k.GetDim(1);
    const int64_t t = k.GetDim(2);
    const int64_t hv = g.GetDim(1);
    CHECK_COND(hv > 0 && hv <= maxDim && g.GetDim(0) == b && g.GetDim(2) == t,
               ACLNN_ERR_PARAM_INVALID, "g must be [B,Hv,T] with positive Hv and B/T matching k.");
    CHECK_COND(beta.GetDim(0) == b && beta.GetDim(1) == hv && beta.GetDim(2) == t,
               ACLNN_ERR_PARAM_INVALID, "beta must have the same [B,Hv,T] shape as g.");
    CHECK_COND(hv % hk == 0, ACLNN_ERR_PARAM_INVALID,
               "Hv must be divisible by Hk, got Hv=%ld and Hk=%ld.", hv, hk);
    CHECK_COND(a.GetDim(0) == b && a.GetDim(1) == hv && a.GetDim(2) == t && a.GetDim(3) == p.chunkSize,
               ACLNN_ERR_PARAM_INVALID, "A must have shape [B,Hv,T,chunkSize].");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMetadata(const ChunkScaledDotKktParams &p)
{
    CHECK_COND((p.cuSeqlens == nullptr) == (p.chunkIndices == nullptr), ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be provided together.");
    if (p.cuSeqlens == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(p.cuSeqlens->Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must contain at least [0, T].");
    const int64_t t = p.k->GetViewShape().GetDim(2);
    CHECK_COND((*p.cuSeqlens)[0] == 0 && (*p.cuSeqlens)[p.cuSeqlens->Size() - 1] == t,
               ACLNN_ERR_PARAM_INVALID, "cu_seqlens must start at 0 and end at T=%ld.", t);
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < p.cuSeqlens->Size(); ++seq) {
        const int64_t length = (*p.cuSeqlens)[seq + 1] - (*p.cuSeqlens)[seq];
        CHECK_COND(length > 0, ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens must be strictly increasing (empty sequences are unsupported).");
        totalChunks += (length - 1) / p.chunkSize + 1;
    }
    CHECK_COND(p.chunkIndices->Size() == static_cast<size_t>(totalChunks) * 2,
               ACLNN_ERR_PARAM_INVALID, "chunk_indices must contain one [seq_id, chunk_id] pair per chunk.");
    for (size_t i = 0; i < p.chunkIndices->Size(); i += 2) {
        const int64_t seq = (*p.chunkIndices)[i];
        CHECK_COND(seq >= 0 && static_cast<uint64_t>(seq) < p.cuSeqlens->Size() - 1,
                   ACLNN_ERR_PARAM_INVALID, "chunk_indices[%zu] has invalid seq_id=%ld.", i, seq);
        const int64_t length = (*p.cuSeqlens)[seq + 1] - (*p.cuSeqlens)[seq];
        const int64_t chunks = (length - 1) / p.chunkSize + 1;
        CHECK_COND((*p.chunkIndices)[i + 1] >= 0 && (*p.chunkIndices)[i + 1] < chunks,
                   ACLNN_ERR_PARAM_INVALID, "chunk_indices[%zu] has invalid chunk_id=%ld.",
                   i + 1, (*p.chunkIndices)[i + 1]);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const ChunkScaledDotKktParams &p)
{
    CHECK_COND(p.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(p.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(p.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(p.a != nullptr, ACLNN_ERR_PARAM_NULLPTR, "A must not be nullptr.");
    CHECK_COND(p.chunkSize == 16 || p.chunkSize == 32 || p.chunkSize == 64 || p.chunkSize == 128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be one of 16, 32, 64, 128, got %ld.", p.chunkSize);
    CHECK_COND(p.k->GetDataType() == DataType::DT_FLOAT16 || p.k->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "k must use float16 or bfloat16.");
    CHECK_COND(p.g->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID, "g must use float32.");
    CHECK_COND(p.beta->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID, "beta must use float32.");
    CHECK_COND(p.a->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID, "A must use float32.");
    auto status = CheckShape(p);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    CHECK_COND(IsContiguous(p.a), ACLNN_ERR_PARAM_INVALID, "A must be contiguous.");
    return CheckMetadata(p);
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_COND(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous input conversion failed.");
    return ACLNN_SUCCESS;
}
}  // namespace

extern "C" {
aclnnStatus aclnnChunkScaledDotKktGetWorkspaceSize(
    const aclTensor *k, const aclTensor *g, const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize, const aclTensor *a, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    ChunkScaledDotKktParams p{k, g, beta, cuSeqlensOptional, chunkIndicesOptional, chunkSize, a};
    L2_DFX_PHASE_1(aclnnChunkScaledDotKkt,
                   DFX_IN(k, g, beta, cuSeqlensOptional, chunkIndicesOptional, chunkSize), DFX_OUT(a));
    const auto status = CheckParams(p);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = MakeContiguous(p.k, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = MakeContiguous(p.g, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = MakeContiguous(p.beta, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    const auto result = l0op::ChunkScaledDotKkt(
        p.k, p.g, p.beta, p.cuSeqlens, p.chunkIndices, p.chunkSize, p.a, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const auto copied = l0op::ViewCopy(result, p.a, executorPtr);
    CHECK_RET(copied != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkScaledDotKkt(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkScaledDotKkt);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkScaledDotKkt launch failed.");
    return ACLNN_SUCCESS;
}
}
