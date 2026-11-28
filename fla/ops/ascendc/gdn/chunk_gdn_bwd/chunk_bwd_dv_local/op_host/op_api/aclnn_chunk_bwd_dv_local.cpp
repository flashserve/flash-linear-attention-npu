/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
// Load structured reporters before opdev's legacy logging macros. The opdev
// headers must retain their logging convention for CHECK_RET and DFX helpers.
#include "op_common/log/log.h"
#include "aclnn_chunk_bwd_dv_local.h"
#include "chunk_bwd_dv_local.h"
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

static constexpr const char *OP_NAME = "aclnnChunkBwdDvLocal";

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkBwdDvLocalParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *dO = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *gGammaOptional = nullptr;
    const aclTensor *aOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64L;
    const aclTensor *out = nullptr;
};

static aclnnStatus CheckNotNull(ChunkBwdDvLocalParams params)
{
    if (params.q == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.k == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dO == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dO");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.g == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.out == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "out");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkBwdDvLocalParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkBwdDvLocalParams params)
{
    CHECK_COND(params.chunkSize > 0, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be positive.");
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be 64 or 128.");
    CHECK_COND(params.q->GetViewShape().GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID, "q must be a rank-4 tensor [B, H, T, K].");
    if (params.chunkIndicesOptional != nullptr) {
        CHECK_COND(params.cuSeqlensOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "chunk_indices requires cu_seqlens in variable-length mode.");
    }
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const aclIntArray &cuSeqlens = *params.cuSeqlensOptional;
    CHECK_COND(cuSeqlens.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must contain at least [0, total_tokens].");
    const int64_t totalTokens = params.q->GetViewShape().GetDim(2);
    CHECK_COND(cuSeqlens[0] == 0, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens[0] must be 0.");
    CHECK_COND(cuSeqlens[cuSeqlens.Size() - 1] == totalTokens,
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens[-1] must equal q's sequence length.");
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens.Size(); ++seq) {
        CHECK_COND(cuSeqlens[seq] >= 0 && cuSeqlens[seq + 1] >= cuSeqlens[seq],
                   ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens must be nondecreasing and non-negative.");
        const int64_t length = cuSeqlens[seq + 1] - cuSeqlens[seq];
        totalChunks += (length + params.chunkSize - 1) / params.chunkSize;
    }
    CHECK_COND(totalChunks > 0, ACLNN_ERR_PARAM_INVALID,
               "variable-length input must contain at least one chunk.");
    CHECK_COND(params.chunkIndicesOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "variable-length mode requires chunk_indices.");
    const aclIntArray &chunkIndices = *params.chunkIndicesOptional;
    CHECK_COND(chunkIndices.Size() == static_cast<size_t>(totalChunks) * 2,
               ACLNN_ERR_PARAM_INVALID,
               "chunk_indices must contain exactly one (sequence, chunk) pair per chunk.");
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens.Size(); ++seq) {
        const int64_t length = cuSeqlens[seq + 1] - cuSeqlens[seq];
        const int64_t chunks = (length + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            CHECK_COND(chunkIndices[offset] == static_cast<int64_t>(seq) &&
                           chunkIndices[offset + 1] == chunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunk_indices must use canonical sequence-major chunk order.");
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkBwdDvLocalParams &params, aclOpExecutor *executorPtr)
{
    if (DataContiguous(params.q, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(q) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.k, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(k) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.dO, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(dO) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.g, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(g) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkBwdDvLocalParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkBwdDvLocalParams params)
{
    {
        const aclnnStatus checkRet = CheckNotNull(params);
        if (checkRet != ACLNN_SUCCESS) {
            return checkRet;
        }
    }
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkBwdDvLocalGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *dO,
    const aclTensor *g,
    const aclTensor *gGammaOptional,
    const aclTensor *aOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkBwdDvLocalParams params{q, k, dO, g, gGammaOptional, aOptional, cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, out};
    // Standard syntax, Check parameters.
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    L2_DFX_PHASE_1(aclnnChunkBwdDvLocal, DFX_IN(q, k, dO, g, gGammaOptional, aOptional, cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize),
                   DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    OP_LOGW("npu_chunk_bwd_dv_local",
            "npu_chunk_bwd_dv_local is deprecated and will be removed on "
            "January 15, 2027 (v26.12.0); use the fused npu_chunk_gated_delta_rule_bwd instead.");

    // 固定写法，参数检查
    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    if (ParamsDataContiguous(params, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(conversion) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto result = l0op::ChunkBwdDvLocal(params.q, params.k,  params.dO, params.g, params.gGammaOptional, params.aOptional, params.cuSeqlensOptional, params.chunkIndicesOptional, params.scale, params.chunkSize, params.out, executorPtr);
    if (result == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "l0op",
            "l0op(ChunkBwdDvLocal) returned nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result, params.out, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "view_copy",
            "ViewCopy(out) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkBwdDvLocal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkBwdDvLocal);
    const aclnnStatus runRet = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (runRet != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "executor",
            "CommonOpExecutorRun failed, ret = " +
                std::to_string(static_cast<int64_t>(static_cast<int>(runRet))));
        return runRet;
    }
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
