/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_pre_process_bwd_kernel_merged.h"
#include "pre_process_bwd_kernel_merged.h"
#include <dlfcn.h>
#include <new>
#include <limits>
#include <sstream>
#include "../dfx_report.h"

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
    const aclTensor *tensors[] = {params.q, params.k, params.w, params.dO, params.dv, params.dhmOut};
    const char *names[] = {"q", "k", "w", "d_o", "dv", "dhm"};
    for (size_t i = 0; i < 6; ++i) {
        if (tensors[i] == nullptr) {
            CP_DFX_NULL("aclnnPreProcessBwdKernelMerged", names[i]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_NULLPTR, "PreProcessBwdKernelMerged: Contiguous returned nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(PreProcessBwdKernelMergedParams &params, aclOpExecutor *executorPtr)
{
    const aclTensor **tensors[] = {&params.q, &params.k, &params.w, &params.dO, &params.dv,
                                  &params.gOptional, &params.gkOptional};
    const char *names[] = {"q", "k", "w", "d_o", "dv", "g", "gk"};
    for (size_t i = 0; i < 7; ++i) {
        if (*tensors[i] != nullptr && DataContiguous(*tensors[i], executorPtr) != ACLNN_SUCCESS) {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_NULLPTR, "PreProcessBwdKernelMerged: Contiguous parameter=%s.", names[i]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

// Full input shape/dtype validation remains owned by the host tiling processor.
static aclnnStatus CheckParams(PreProcessBwdKernelMergedParams params)
{
    auto ret = CheckNotNull(params);
    if (ret != ACLNN_SUCCESS) return ret;
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckBoundaryParams(const PreProcessBwdKernelMergedParams &params)
{
    const char *opName = "aclnnPreProcessBwdKernelMerged";
    const auto &qShape = params.q->GetViewShape();
    const auto &doShape = params.dO->GetViewShape();
    // Validate the caller-owned output only once its expected dimensions exist.
    if (qShape.GetDimNum() == 4 && doShape.GetDimNum() == 4) {
        const auto &shape = params.dhmOut->GetViewShape();
        const int64_t hv = doShape.GetDim(1), k = qShape.GetDim(3), v = doShape.GetDim(3);
        if (shape.GetDimNum() != 3 || shape.GetDim(0) != hv || shape.GetDim(1) != k ||
            k < 0 || v < 0 || v > std::numeric_limits<int64_t>::max() - k || shape.GetDim(2) != v + k) {
            std::ostringstream actual;
            for (size_t i = 0; i < shape.GetDimNum(); ++i) {
                if (i != 0) actual << ',';
                actual << shape.GetDim(i);
            }
            CP_DFX_SHAPE_REASON(opName, "dhm", actual.str(), "output must be [Hv,K,V+K] matching q and d_o");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.dhmOut->GetDataType() != DataType::DT_FLOAT) {
            CP_DFX_DTYPE(opName, "dhm", std::to_string(params.dhmOut->GetDataType()), "dhm must be float32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!IsContiguous(params.dhmOut)) {
            const auto &strides = params.dhmOut->GetViewStrides();
            std::ostringstream actual;
            for (size_t i = 0; i < shape.GetDimNum(); ++i) {
                if (i != 0) actual << ',';
                actual << strides[i];
            }
            CP_DFX_STRIDE(opName, "dhm", actual.str(), "contiguous [K*(V+K),V+K,1]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.cuSeqlensOptional != nullptr && params.cuSeqlensOptional->Size() >= 2) {
            const int64_t bos = (*params.cuSeqlensOptional)[0], eos = (*params.cuSeqlensOptional)[1];
            if (bos < 0 || eos < bos || eos > qShape.GetDim(2) || eos > std::numeric_limits<uint32_t>::max()) {
                CP_DFX_VALUE_REASON(opName, "cu_seqlens[0:2]", std::to_string(bos) + "," + std::to_string(eos),
                    "requires 0 <= bos <= eos <= T=" + std::to_string(qShape.GetDim(2)) + "; offsets must fit uint32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    return ACLNN_SUCCESS;
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
    if (workspaceSize == nullptr || executor == nullptr) {
        const char *parameter = workspaceSize == nullptr ? "workspaceSize" : "executor";
        CP_DFX_ARGUMENT("aclnnPreProcessBwdKernelMerged", parameter,
            "pass a writable output pointer for the two-stage ACLNN API");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    PreProcessBwdKernelMergedParams params{q, k, w, dO, dv, gOptional, gkOptional, cuSeqlensOptional, scale, chunkSize,
                                          dhmOut};
    auto ret = CheckParams(params);
    // Preserve the existing public errno mapping independently of the EZ code.
    if (ret != ACLNN_SUCCESS) return ACLNN_ERR_PARAM_INVALID;
    ret = CheckBoundaryParams(params);
    if (ret != ACLNN_SUCCESS) return ret;
    L2_DFX_PHASE_1(aclnnPreProcessBwdKernelMerged,
                   DFX_IN(q, k, w, dO, dv, gOptional, gkOptional, cuSeqlensOptional), DFX_OUT(dhmOut));
    // CREATE_EXECUTOR uses the allocator's exception path; a null check here
    // cannot add a distinct root cause and would turn allocation failure into
    // a generic internal report.
    auto uniqueExecutor = CREATE_EXECUTOR();
    auto executorPtr = uniqueExecutor.get();
    if (ParamsDataContiguous(params, executorPtr) != ACLNN_SUCCESS) return ACLNN_ERR_PARAM_INVALID;
    auto result = l0op::PreProcessBwdKernelMerged(params.q, params.k, params.w, params.dO, params.dv, params.gOptional,
                                                 params.gkOptional, params.cuSeqlensOptional, params.scale,
                                                 params.chunkSize, params.dhmOut, executorPtr);
    if (result == nullptr) return ACLNN_ERR_PARAM_NULLPTR;
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPreProcessBwdKernelMerged(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPreProcessBwdKernelMerged);
    const auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(ret, "PreProcessBwdKernelMerged: CommonOpExecutorRun returned rc=%u.", ret);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
