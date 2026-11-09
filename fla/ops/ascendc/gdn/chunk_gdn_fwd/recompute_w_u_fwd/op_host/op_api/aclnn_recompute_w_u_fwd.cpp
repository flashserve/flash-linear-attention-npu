/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_recompute_w_u_fwd.h"
#include "recompute_w_u_fwd.h"
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

static constexpr const char *RECOMPUTE_W_U_FWD_ENTITY = "aclnnRecomputeWUFwd";

#ifdef __cplusplus
extern "C" {
#endif

struct RecomputeWUFwdParams {
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *a = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *gk = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    int64_t chunkSize = 64;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
};

static aclnnStatus CheckNotNull(RecomputeWUFwdParams params)
{
    if (params.k == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.v == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "v");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.beta == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "beta");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.a == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "a");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.g == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.gk != nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "gk");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.wOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "wOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.uOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "uOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(RecomputeWUFwdParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(RecomputeWUFwdParams params)
{
    const auto kShape = params.k->GetViewShape();
    const auto vShape = params.v->GetViewShape();
    const auto betaShape = params.beta->GetViewShape();
    const auto aShape = params.a->GetViewShape();
    const auto gShape = params.g->GetViewShape();
    const auto wOutShape = params.wOut->GetViewShape();
    const auto uOutShape = params.uOut->GetViewShape();

    if (kShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "k", "rank=" + std::to_string(kShape.GetDimNum()), "4 [B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (vShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "v", "rank=" + std::to_string(vShape.GetDimNum()), "4 [B, HV, T, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (betaShape.GetDimNum() != 3) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "beta", "rank=" + std::to_string(betaShape.GetDimNum()), "3 [B, HV, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "A", "rank=" + std::to_string(aShape.GetDimNum()), "4 [B, HV, T, chunkSize]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gShape.GetDimNum() != 3) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "g", "rank=" + std::to_string(gShape.GetDimNum()), "3 [B, HV, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (wOutShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "wOut", "rank=" + std::to_string(wOutShape.GetDimNum()), "4 [B, HV, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (uOutShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "uOut", "rank=" + std::to_string(uOutShape.GetDimNum()), "4 [B, HV, T, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t B = kShape.GetDim(0);
    const int64_t HK = kShape.GetDim(1);
    const int64_t T = kShape.GetDim(2);
    const int64_t K = kShape.GetDim(3);
    const int64_t HV = vShape.GetDim(1);
    const int64_t V = vShape.GetDim(3);

    if (HK <= 0 || HV <= 0 || HV % HK != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "HK,HV", "invalid", "positive and HV % HK == 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (vShape.GetDim(0) != B || vShape.GetDim(2) != T) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "v", "mismatch", "match k batch/sequence");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (betaShape.GetDim(0) != B || betaShape.GetDim(1) != HV || betaShape.GetDim(2) != T) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "beta", "mismatch", "[B, HV, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aShape.GetDim(0) != B || aShape.GetDim(1) != HV || aShape.GetDim(2) != T ||
        aShape.GetDim(3) != params.chunkSize) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "A", "mismatch", "[B, HV, T, chunkSize]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gShape.GetDim(0) != B || gShape.GetDim(1) != HV || gShape.GetDim(2) != T) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "g", "mismatch", "[B, HV, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (wOutShape.GetDim(0) != B || wOutShape.GetDim(1) != HV || wOutShape.GetDim(2) != T ||
        wOutShape.GetDim(3) != K) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "wOut", "mismatch", "[B, HV, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (uOutShape.GetDim(0) != B || uOutShape.GetDim(1) != HV || uOutShape.GetDim(2) != T ||
        uOutShape.GetDim(3) != V) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "uOut", "mismatch", "[B, HV, T, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.chunkSize != 64 && params.chunkSize != 128) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "chunkSize", std::to_string(params.chunkSize), "64 or 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (K != 128) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "K", std::to_string(K), "128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (V != 128 && V != 256) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "V", std::to_string(V), "128 or 256");
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

static aclnnStatus ParamsDataContiguous(RecomputeWUFwdParams &params, aclOpExecutor *executorPtr)
{
    auto dfxRetStatus1 = DataContiguous(params.k, executorPtr);
    if (dfxRetStatus1 != ACLNN_SUCCESS) {
        return dfxRetStatus1;
    }
    auto dfxRetStatus2 = DataContiguous(params.v, executorPtr);
    if (dfxRetStatus2 != ACLNN_SUCCESS) {
        return dfxRetStatus2;
    }
    auto dfxRetStatus3 = DataContiguous(params.beta, executorPtr);
    if (dfxRetStatus3 != ACLNN_SUCCESS) {
        return dfxRetStatus3;
    }
    auto dfxRetStatus4 = DataContiguous(params.a, executorPtr);
    if (dfxRetStatus4 != ACLNN_SUCCESS) {
        return dfxRetStatus4;
    }
    auto dfxRetStatus5 = DataContiguous(params.g, executorPtr);
    if (dfxRetStatus5 != ACLNN_SUCCESS) {
        return dfxRetStatus5;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(RecomputeWUFwdParams params)
{
    auto inputDtype = params.k->GetDataType();
    if (inputDtype != DataType::DT_FLOAT16 && inputDtype != DataType::DT_BF16) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "k",
            fla_npu::host_utils::DfxDtypeString(inputDtype), "FLOAT16,BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.v->GetDataType() != inputDtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "v",
            fla_npu::host_utils::DfxDtypeString(params.v->GetDataType()), "match k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.a->GetDataType() != inputDtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "a",
            fla_npu::host_utils::DfxDtypeString(params.a->GetDataType()), "match k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.wOut->GetDataType() != inputDtype || params.uOut->GetDataType() != inputDtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "wOut,uOut", "dtype mismatch", "match k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.beta->GetDataType() != DataType::DT_FLOAT && params.beta->GetDataType() != inputDtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "beta",
            fla_npu::host_utils::DfxDtypeString(params.beta->GetDataType()), "FLOAT32 or match k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.g->GetDataType() != DataType::DT_FLOAT && params.g->GetDataType() != inputDtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(RECOMPUTE_W_U_FWD_ENTITY, "g",
            fla_npu::host_utils::DfxDtypeString(params.g->GetDataType()), "FLOAT32 or match k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(RecomputeWUFwdParams params)
{
    auto dfxRetStatus6 = CheckFormat(params);
    if (dfxRetStatus6 != ACLNN_SUCCESS) {
        return dfxRetStatus6;
    }
    auto dfxRetStatus7 = CheckShape(params);
    if (dfxRetStatus7 != ACLNN_SUCCESS) {
        return dfxRetStatus7;
    }
    auto dfxRetStatus8 = CheckDtype(params);
    if (dfxRetStatus8 != ACLNN_SUCCESS) {
        return dfxRetStatus8;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecomputeWUFwdGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *g,
    const aclTensor *gk,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *wOut,
    const aclTensor *uOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    RecomputeWUFwdParams params{k, v, beta, a, g, gk,cuSeqlensOptional, chunkIndicesOptional, chunkSize, wOut, uOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnRecomputeWUFwd,
                   DFX_IN(k, v, beta, a, g, gk, cuSeqlensOptional, chunkIndicesOptional, chunkSize),
                   DFX_OUT(wOut, uOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    OP_LOGW("npu_recompute_w_u_fwd",
            "npu_recompute_w_u_fwd is deprecated and will be removed on "
            "January 15, 2027 (v26.12.0); use the fused npu_chunk_gated_delta_rule_fwd / "
            "npu_chunk_gated_delta_rule_bwd instead.");

    // 固定写法，参数检查
    auto status = CheckNotNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (ParamsDataContiguous(params, executorPtr) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto result = l0op::RecomputeWUFwd(params.k, params.v, params.beta, params.a, params.g, params.gk, params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize, params.wOut, params.uOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result[0], params.wOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[1], params.uOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);


    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnRecomputeWUFwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecomputeWUFwd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in RecomputeWUFwd launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
