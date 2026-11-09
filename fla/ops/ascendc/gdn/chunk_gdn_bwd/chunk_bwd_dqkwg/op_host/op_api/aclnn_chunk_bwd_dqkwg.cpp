/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_chunk_bwd_dqkwg.h"
#include "chunk_bwd_dqkwg.h"
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

static constexpr const char *CHUNK_BWD_DQKWG_ENTITY = "aclnnChunkBwdDqkwg";

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkBwdDqkwgParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *h = nullptr;
    const aclTensor *dox = nullptr;
    const aclTensor *dh = nullptr;
    const aclTensor *dv = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *gGamma = nullptr;
    float scale = 0;
    int64_t chunkSize = 0;
    const aclTensor *dqOut = nullptr;
    const aclTensor *dkOut = nullptr;
    const aclTensor *dwOut = nullptr;
    const aclTensor *dgOut = nullptr;
};

static aclnnStatus CheckNotNull(ChunkBwdDqkwgParams params)
{
    if (params.q == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "q");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.k == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.v == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "v");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.g == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.h == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "h");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dox == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "do");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dh == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dh");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dv == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dv");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dqOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dqOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dkOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dkOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dwOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dwOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dgOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dgOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkBwdDqkwgParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkBwdDqkwgParams params)
{
    const auto qShape = params.q->GetViewShape();
    const auto kShape = params.k->GetViewShape();
    const auto vShape = params.v->GetViewShape();
    const auto gShape = params.g->GetViewShape();
    const auto dqShape = params.dqOut->GetViewShape();
    const auto dkShape = params.dkOut->GetViewShape();
    const auto dwShape = params.dwOut->GetViewShape();
    const auto dgShape = params.dgOut->GetViewShape();

    if (qShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "q", "rank=" + std::to_string(qShape.GetDimNum()), "4 [B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "k", "rank=" + std::to_string(kShape.GetDimNum()), "4 [B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (vShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "v", "rank=" + std::to_string(vShape.GetDimNum()), "4 [B, HV, T, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gShape.GetDimNum() != 3) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "g", "rank=" + std::to_string(gShape.GetDimNum()), "3 [B, HV, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dqShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dqOut", "rank=" + std::to_string(dqShape.GetDimNum()), "4 [B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dkShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dkOut", "rank=" + std::to_string(dkShape.GetDimNum()), "4 [B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dwShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dwOut", "rank=" + std::to_string(dwShape.GetDimNum()), "4 [B, HV, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dgShape.GetDimNum() != 3) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dgOut", "rank=" + std::to_string(dgShape.GetDimNum()), "3 [B, HV, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t B = qShape.GetDim(0);
    const int64_t HK = qShape.GetDim(1);
    const int64_t T = qShape.GetDim(2);
    const int64_t K = qShape.GetDim(3);
    const int64_t HV = vShape.GetDim(1);

    if (HK > 0 && HV > 0 && HV % HK == 0) {
        // valid
    } else {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "HK,HV", "invalid", "positive and HV % HK == 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kShape.GetDim(0) != B || kShape.GetDim(1) != HK || kShape.GetDim(2) != T || kShape.GetDim(3) != K) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "k", "mismatch", "[B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (vShape.GetDim(0) != B || vShape.GetDim(2) != T) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "v", "mismatch", "match q batch and sequence");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gShape.GetDim(0) != B || gShape.GetDim(1) != HV || gShape.GetDim(2) != T) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "g", "mismatch", "[B, HV, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dqShape.GetDim(0) != B || dqShape.GetDim(1) != HK || dqShape.GetDim(2) != T || dqShape.GetDim(3) != K) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dqOut", "mismatch", "[B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dkShape.GetDim(0) != B || dkShape.GetDim(1) != HK || dkShape.GetDim(2) != T || dkShape.GetDim(3) != K) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dkOut", "mismatch", "[B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dwShape.GetDim(0) != B || dwShape.GetDim(1) != HV || dwShape.GetDim(2) != T || dwShape.GetDim(3) != K) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dwOut", "mismatch", "[B, HV, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dgShape.GetDim(0) != B || dgShape.GetDim(1) != HV || dgShape.GetDim(2) != T) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "dgOut", "mismatch", "[B, HV, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkBwdDqkwgParams &params, aclOpExecutor *executorPtr)
{
    auto dfxRetStatus1 = DataContiguous(params.q, executorPtr);
    if (dfxRetStatus1 != ACLNN_SUCCESS) {
        return dfxRetStatus1;
    }
    auto dfxRetStatus2 = DataContiguous(params.k, executorPtr);
    if (dfxRetStatus2 != ACLNN_SUCCESS) {
        return dfxRetStatus2;
    }
    auto dfxRetStatus3 = DataContiguous(params.v, executorPtr);
    if (dfxRetStatus3 != ACLNN_SUCCESS) {
        return dfxRetStatus3;
    }
    auto dfxRetStatus4 = DataContiguous(params.g, executorPtr);
    if (dfxRetStatus4 != ACLNN_SUCCESS) {
        return dfxRetStatus4;
    }
    auto dfxRetStatus5 = DataContiguous(params.h, executorPtr);
    if (dfxRetStatus5 != ACLNN_SUCCESS) {
        return dfxRetStatus5;
    }
    auto dfxRetStatus6 = DataContiguous(params.dox, executorPtr);
    if (dfxRetStatus6 != ACLNN_SUCCESS) {
        return dfxRetStatus6;
    }
    auto dfxRetStatus7 = DataContiguous(params.dh, executorPtr);
    if (dfxRetStatus7 != ACLNN_SUCCESS) {
        return dfxRetStatus7;
    }
    auto dfxRetStatus8 = DataContiguous(params.dv, executorPtr);
    if (dfxRetStatus8 != ACLNN_SUCCESS) {
        return dfxRetStatus8;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkBwdDqkwgParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkBwdDqkwgParams params)
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

aclnnStatus aclnnChunkBwdDqkwgGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *h,
    const aclTensor *dox,
    const aclTensor *dh,
    const aclTensor *dv,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const aclTensor *w,
    const aclTensor *gGamma,
    float scale,
    int64_t chunkSize,
    bool use_exp2,
    bool transpose_state_layout,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dwOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkBwdDqkwgParams params{q, k, v, g, h, dox, dh, dv, cuSeqlensOptional, chunkIndicesOptional, w, gGamma, scale, chunkSize, dqOut, dkOut, dwOut, dgOut};
    if (use_exp2 != false || transpose_state_layout != false) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_BWD_DQKWG_ENTITY, "useExp2,transposeStateLayout",
            "not matching", "both false");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(
        aclnnChunkBwdDqkwg,
        DFX_IN(q, k, v, g, h, dox, dh, dv, cuSeqlensOptional, chunkIndicesOptional, w, gGamma, scale, chunkSize,
               use_exp2, transpose_state_layout),
        DFX_OUT(dqOut, dkOut, dwOut, dgOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    OP_LOGW("npu_chunk_bwd_dqkwg",
            "npu_chunk_bwd_dqkwg is deprecated and will be removed on "
            "January 15, 2027 (v26.12.0); use the fused npu_chunk_gated_delta_rule_bwd instead.");

    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto dfxRetStatus9 = ParamsDataContiguous(params, executorPtr);
    if (dfxRetStatus9 != ACLNN_SUCCESS) {
        return dfxRetStatus9;
    }
    auto result = l0op::ChunkBwdDqkwg(params.q, params.k, params.v, params.g, params.h, params.dox, params.dh, params.dv, params.cuSeqlensOptional, params.chunkIndicesOptional, params.w, params.gGamma, params.scale, params.chunkSize, executorPtr);

    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[3] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result[0], params.dqOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[1], params.dkOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[2], params.dwOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[3], params.dgOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkBwdDqkwg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkBwdDqkwg);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkBwdDqkwg launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
