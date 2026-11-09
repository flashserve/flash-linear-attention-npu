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
#include "../../../../../common/host_utils/log.h"

using namespace op;

static constexpr const char *CHUNK_SCALED_DOT_KKT_ENTITY = "aclnnChunkScaledDotKkt";

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
    if (k.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "k",
            "rank=" + std::to_string(k.GetDimNum()), "4 [B,Hk,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (g.GetDimNum() != 3) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "g",
            "rank=" + std::to_string(g.GetDimNum()), "3 [B,Hv,T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (beta.GetDimNum() != 3) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "beta",
            "rank=" + std::to_string(beta.GetDimNum()), "3 [B,Hv,T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (a.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "A",
            "rank=" + std::to_string(a.GetDimNum()), "4 [B,Hv,T,chunkSize]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    constexpr int64_t maxDim = std::numeric_limits<int32_t>::max();
    for (size_t dim = 0; dim < 4; ++dim) {
        if (k.GetDim(dim) <= 0 || k.GetDim(dim) > maxDim) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "k.dim",
                std::to_string(k.GetDim(dim)), "[1, " + std::to_string(maxDim) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (k.GetDim(3) < 8) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "k.featureDim",
            std::to_string(k.GetDim(3)), "at least 8");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t b = k.GetDim(0);
    const int64_t hk = k.GetDim(1);
    const int64_t t = k.GetDim(2);
    const int64_t hv = g.GetDim(1);
    if (hv <= 0 || hv > maxDim || g.GetDim(0) != b || g.GetDim(2) != t) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "g", "mismatch", "[B,Hv,T] with positive Hv, B/T matching k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (beta.GetDim(0) != b || beta.GetDim(1) != hv || beta.GetDim(2) != t) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "beta", "mismatch", "same [B,Hv,T] as g");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hv % hk != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "Hv,Hk", "not divisible",
            "Hv % Hk == 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (a.GetDim(0) != b || a.GetDim(1) != hv || a.GetDim(2) != t || a.GetDim(3) != p.chunkSize) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "A", "mismatch", "[B,Hv,T,chunkSize]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMetadata(const ChunkScaledDotKktParams &p)
{
    if ((p.cuSeqlens == nullptr) != (p.chunkIndices == nullptr)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "cuSeqlens,chunkIndices",
            "not together", "must be provided together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (p.cuSeqlens == nullptr) {
        return ACLNN_SUCCESS;
    }
    if (p.cuSeqlens->Size() < 2) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "cuSeqlens",
            std::to_string(p.cuSeqlens->Size()), "at least 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t t = p.k->GetViewShape().GetDim(2);
    if ((*p.cuSeqlens)[0] != 0 || (*p.cuSeqlens)[p.cuSeqlens->Size() - 1] != t) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "cuSeqlens", "invalid bounds",
            "start at 0, end at T=" + std::to_string(t));
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < p.cuSeqlens->Size(); ++seq) {
        const int64_t length = (*p.cuSeqlens)[seq + 1] - (*p.cuSeqlens)[seq];
        if (length <= 0) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "cuSeqlens", "non-increasing",
                "strictly increasing (empty sequences unsupported)");
            return ACLNN_ERR_PARAM_INVALID;
        }
        totalChunks += (length - 1) / p.chunkSize + 1;
    }
    if (p.chunkIndices->Size() != static_cast<size_t>(totalChunks) * 2) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "chunkIndices",
            std::to_string(p.chunkIndices->Size()),
            std::to_string(totalChunks * 2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t i = 0; i < p.chunkIndices->Size(); i += 2) {
        const int64_t seq = (*p.chunkIndices)[i];
        if (seq < 0 || static_cast<uint64_t>(seq) >= p.cuSeqlens->Size() - 1) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "chunkIndices",
                "invalid seq_id=" + std::to_string(seq), "valid seq_id");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t length = (*p.cuSeqlens)[seq + 1] - (*p.cuSeqlens)[seq];
        const int64_t chunks = (length - 1) / p.chunkSize + 1;
        if ((*p.chunkIndices)[i + 1] < 0 || (*p.chunkIndices)[i + 1] >= chunks) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "chunkIndices",
                "invalid chunk_id", "valid chunk_id");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const ChunkScaledDotKktParams &p)
{
    if (p.k == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (p.g == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (p.beta == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "beta");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (p.a == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "A");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (p.chunkSize != 16 && p.chunkSize != 32 && p.chunkSize != 64 && p.chunkSize != 128) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "chunkSize",
            std::to_string(p.chunkSize), "16, 32, 64 or 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (p.k->GetDataType() != DataType::DT_FLOAT16 && p.k->GetDataType() != DataType::DT_BF16) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "k",
            fla_npu::host_utils::DfxDtypeString(p.k->GetDataType()), "FLOAT16,BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (p.g->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "g",
            fla_npu::host_utils::DfxDtypeString(p.g->GetDataType()), "FLOAT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (p.beta->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "beta",
            fla_npu::host_utils::DfxDtypeString(p.beta->GetDataType()), "FLOAT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (p.a->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "A",
            fla_npu::host_utils::DfxDtypeString(p.a->GetDataType()), "FLOAT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto status = CheckShape(p);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (!IsContiguous(p.a)) {
        OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "A", "non-contiguous", "contiguous");
        return ACLNN_ERR_PARAM_INVALID;
    }
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
    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_SCALED_DOT_KKT_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
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

    OP_LOGW("npu_chunk_scaled_dot_kkt",
            "npu_chunk_scaled_dot_kkt is deprecated and will be removed on "
            "January 15, 2027 (v26.12.0); use the fused npu_chunk_gated_delta_rule_fwd instead.");

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
