/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
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
#include "../../../../../common/host_utils/log.h"


using namespace op;

static constexpr const char *CHUNK_BWD_DV_LOCAL_ENTITY = "aclnnChunkBwdDvLocal";

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
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "q");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.k == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dO == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "dO");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.g == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.out == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "out");
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
    if (params.chunkSize <= 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "chunkSize", std::to_string(params.chunkSize), "positive");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.chunkSize != 64 && params.chunkSize != 128) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "chunkSize", std::to_string(params.chunkSize), "64 or 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.q->GetViewShape().GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "q",
            "rank=" + std::to_string(params.q->GetViewShape().GetDimNum()), "4 [B, H, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.chunkIndicesOptional != nullptr && params.cuSeqlensOptional == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "chunkIndices", "present", "requires cu_seqlens in variable-length mode");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const aclIntArray &cuSeqlens = *params.cuSeqlensOptional;
    if (cuSeqlens.Size() < 2) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "cuSeqlens",
            std::to_string(cuSeqlens.Size()), "at least 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t totalTokens = params.q->GetViewShape().GetDim(2);
    if (cuSeqlens[0] != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "cuSeqlens[0]", std::to_string(cuSeqlens[0]), "0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cuSeqlens[cuSeqlens.Size() - 1] != totalTokens) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "cuSeqlens[-1]",
            std::to_string(cuSeqlens[cuSeqlens.Size() - 1]),
            "equal q sequence length " + std::to_string(totalTokens));
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens.Size(); ++seq) {
        if (cuSeqlens[seq] < 0 || cuSeqlens[seq + 1] < cuSeqlens[seq]) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "cuSeqlens", "nondecreasing, non-negative",
                "cu_seqlens must be nondecreasing and non-negative");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t length = cuSeqlens[seq + 1] - cuSeqlens[seq];
        totalChunks += (length + params.chunkSize - 1) / params.chunkSize;
    }
    if (totalChunks <= 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "cuSeqlens", "empty", "at least one chunk");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.chunkIndicesOptional == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "chunkIndices");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclIntArray &chunkIndices = *params.chunkIndicesOptional;
    if (chunkIndices.Size() != static_cast<size_t>(totalChunks) * 2) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "chunkIndices",
            std::to_string(chunkIndices.Size()),
            std::to_string(totalChunks * 2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens.Size(); ++seq) {
        const int64_t length = cuSeqlens[seq + 1] - cuSeqlens[seq];
        const int64_t chunks = (length + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            if (chunkIndices[offset] != static_cast<int64_t>(seq) ||
                chunkIndices[offset + 1] != chunk) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DV_LOCAL_ENTITY, "chunkIndices", "non-canonical",
                    "canonical sequence-major chunk order");
                return ACLNN_ERR_PARAM_INVALID;
            }
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
    auto dfxRetStatus1 = DataContiguous(params.q, executorPtr);
    if (dfxRetStatus1 != ACLNN_SUCCESS) {
        return dfxRetStatus1;
    }
    auto dfxRetStatus2 = DataContiguous(params.k, executorPtr);
    if (dfxRetStatus2 != ACLNN_SUCCESS) {
        return dfxRetStatus2;
    }
    auto dfxRetStatus3 = DataContiguous(params.dO, executorPtr);
    if (dfxRetStatus3 != ACLNN_SUCCESS) {
        return dfxRetStatus3;
    }
    auto dfxRetStatus4 = DataContiguous(params.g, executorPtr);
    if (dfxRetStatus4 != ACLNN_SUCCESS) {
        return dfxRetStatus4;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkBwdDvLocalParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkBwdDvLocalParams params)
{
    auto status = CheckNotNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckFormat(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckShape(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckDtype(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
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
    auto dfxRetStatus5 = ParamsDataContiguous(params, executorPtr);
    if (dfxRetStatus5 != ACLNN_SUCCESS) {
        return dfxRetStatus5;
    }
    auto result = l0op::ChunkBwdDvLocal(params.q, params.k,  params.dO, params.g, params.gGammaOptional, params.aOptional, params.cuSeqlensOptional, params.chunkIndicesOptional, params.scale, params.chunkSize, params.out, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result, params.out, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkBwdDvLocal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkBwdDvLocal);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkBwdDvLocal launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
