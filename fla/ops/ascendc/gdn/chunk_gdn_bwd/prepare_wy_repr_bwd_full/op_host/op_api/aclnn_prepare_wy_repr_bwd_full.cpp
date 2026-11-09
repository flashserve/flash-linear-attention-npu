/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_prepare_wy_repr_bwd_full.h"
#include "prepare_wy_repr_bwd_full.h"
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

static constexpr const char *PREPARE_WY_REPR_BWD_FULL_ENTITY = "aclnnPrepareWyReprBwdFull";

#ifdef __cplusplus
extern "C" {
#endif

struct PrepareWyReprBwdFullParams {
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *a = nullptr;
    const aclTensor *dA = nullptr;
    const aclTensor *dw = nullptr;
    const aclTensor *du = nullptr;
    const aclTensor *g = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    int64_t chunkSize = 64;
    const aclTensor *dkOut = nullptr;
    const aclTensor *dvOut = nullptr;
    const aclTensor *dbetaOut = nullptr;
    const aclTensor *dgOut = nullptr;
};

static aclnnStatus CheckNotNull(PrepareWyReprBwdFullParams params)
{
    if (params.k == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.v == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "v");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.beta == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "beta");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.a == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "a");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dA == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "dA");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dw == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "dw");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.du == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "du");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.g == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dkOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "dkOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dvOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "dvOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dbetaOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "dbetaOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dgOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_WY_REPR_BWD_FULL_ENTITY, "dgOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(PrepareWyReprBwdFullParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(PrepareWyReprBwdFullParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(PrepareWyReprBwdFullParams &params, aclOpExecutor *executorPtr)
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
    auto dfxRetStatus5 = DataContiguous(params.dA, executorPtr);
    if (dfxRetStatus5 != ACLNN_SUCCESS) {
        return dfxRetStatus5;
    }
    auto dfxRetStatus6 = DataContiguous(params.dw, executorPtr);
    if (dfxRetStatus6 != ACLNN_SUCCESS) {
        return dfxRetStatus6;
    }
    auto dfxRetStatus7 = DataContiguous(params.du, executorPtr);
    if (dfxRetStatus7 != ACLNN_SUCCESS) {
        return dfxRetStatus7;
    }
    auto dfxRetStatus8 = DataContiguous(params.g, executorPtr);
    if (dfxRetStatus8 != ACLNN_SUCCESS) {
        return dfxRetStatus8;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(PrepareWyReprBwdFullParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(PrepareWyReprBwdFullParams params)
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

aclnnStatus aclnnPrepareWyReprBwdFullGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dA,
    const aclTensor *dw,
    const aclTensor *du,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dbetaOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    PrepareWyReprBwdFullParams params{k, v, beta, a, dA, dw, du, g, cuSeqlensOptional, chunkIndicesOptional, chunkSize, dkOut, dvOut, dbetaOut, dgOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnPrepareWyReprBwdFull,
                   DFX_IN(k, v, beta, a, dA, dw, du, g, cuSeqlensOptional, chunkIndicesOptional, chunkSize),
                   DFX_OUT(dkOut, dvOut, dbetaOut, dgOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    OP_LOGW("npu_prepare_wy_repr_bwd_full",
            "npu_prepare_wy_repr_bwd_full is deprecated and will be removed "
            "on January 15, 2027 (v26.12.0); use the fused npu_chunk_gated_delta_rule_bwd "
            "instead.");

    // 固定写法，参数检查
    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto dfxRetStatus9 = ParamsDataContiguous(params, executorPtr);
    if (dfxRetStatus9 != ACLNN_SUCCESS) {
        return dfxRetStatus9;
    }
    auto result = l0op::PrepareWyReprBwdFull(params.k, params.v, params.beta, params.a, params.dA, params.dw, params.du, params.g, params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize, params.dkOut, params.dvOut, params.dbetaOut, params.dgOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[3] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result[0], params.dkOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[1], params.dvOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[2], params.dbetaOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[3], params.dgOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnPrepareWyReprBwdFull(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPrepareWyReprBwdFull);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in PrepareWyReprBwdFull launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
