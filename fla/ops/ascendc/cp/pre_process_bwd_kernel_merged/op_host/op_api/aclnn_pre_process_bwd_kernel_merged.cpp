/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_pre_process_bwd_kernel_merged.h"
#if defined(__has_include)
#if __has_include("op_common/log/log.h")
#include "op_common/log/log.h"
#undef OP_LOGE
#undef OP_LOGE_WITHOUT_REPORT
#endif
#endif
#include "pre_process_bwd_kernel_merged.h"
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

namespace {
constexpr const char *CP_BWD_ENTITY = "aclnnPreProcessBwdKernelMerged";

void ReportInvalidInput(const char *paramName)
{
#if defined(OP_LOGE_LIBOPAPI_REPORT) && defined(OP_LOGE_WITH_INVALID_INPUT)
    OP_LOGE_WITH_INVALID_INPUT(CP_BWD_ENTITY, paramName);
#else
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Parameter %s of %s is required, but it is empty.", paramName, CP_BWD_ENTITY);
#endif
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

struct PreProcessBwdKernelMergedParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *dO = nullptr;
    const aclTensor *dv = nullptr;
    const aclTensor *gOptional = nullptr;
    const aclTensor *gkOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    const aclTensor *dhmOut = nullptr;
};

static aclnnStatus CheckNotNull(PreProcessBwdKernelMergedParams params)
{
    const aclTensor *required[] = {params.q, params.k, params.w, params.dO, params.dv, params.dhmOut};
    const char *names[] = {"q", "k", "w", "d_o", "dv", "dhm"};
    for (size_t idx = 0; idx < sizeof(required) / sizeof(required[0]); ++idx) {
        if (required[idx] == nullptr) {
            ReportInvalidInput(names[idx]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const char *name, const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Making input %s contiguous failed in %s.", name, CP_BWD_ENTITY);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(PreProcessBwdKernelMergedParams &params, aclOpExecutor *executorPtr)
{
    auto status = DataContiguous("q", params.q, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous("k", params.k, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous("w", params.w, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous("d_o", params.dO, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous("dv", params.dv, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (params.gOptional != nullptr) {
        status = DataContiguous("g", params.gOptional, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    if (params.gkOptional != nullptr) {
        status = DataContiguous("gk", params.gkOptional, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    return ACLNN_SUCCESS;
}

// shape/dtype 的完整约束由 host tiling 统一拦截（见 tiling_processor），此处只做非空与连续性检查，
// 避免同一套规则在两处漂移。
static aclnnStatus CheckParams(PreProcessBwdKernelMergedParams params)
{
    return CheckNotNull(params);
}

aclnnStatus aclnnPreProcessBwdKernelMergedGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *dO,
    const aclTensor *dv,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclIntArray *cuSeqlensOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *dhmOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    PreProcessBwdKernelMergedParams params{q, k, w, dO, dv, gOptional, gkOptional, cuSeqlensOptional, scale, chunkSize,
                                          dhmOut};
    if (workspaceSize == nullptr) {
        ReportInvalidInput("workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        ReportInvalidInput("executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(aclnnPreProcessBwdKernelMerged,
                   DFX_IN(q, k, w, dO, dv, gOptional, gkOptional, cuSeqlensOptional), DFX_OUT(dhmOut));
    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    ret = ParamsDataContiguous(params, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto result = l0op::PreProcessBwdKernelMerged(params.q, params.k, params.w, params.dO, params.dv, params.gOptional,
                                                 params.gkOptional, params.cuSeqlensOptional, params.scale,
                                                 params.chunkSize, params.dhmOut, executorPtr);
    if (result == nullptr) {
        // Public shape/dtype/value constraints are enforced by tiling. Keep the structured error
        // reported there as the most recent message and expose the documented aclnn return code.
        return ACLNN_ERR_PARAM_INVALID;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPreProcessBwdKernelMerged(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPreProcessBwdKernelMerged);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in aclnnPreProcessBwdKernelMerged launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
