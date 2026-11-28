/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_pre_process_fwd_kernel_merged.h"
#include "pre_process_fwd_kernel_merged.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_common/log/log.h"
#include <string>
#include "graph/utils/type_utils.h"

static constexpr const char *OP_NAME = "aclnnPreProcessFwdKernelMerged";

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t PPFM_K = 128;
constexpr int64_t PPFM_V = 128;
constexpr int64_t PPFM_CHUNK = 64;

struct PreProcessFwdKernelMergedParams {
    const aclTensor *k = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *u = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *gk = nullptr;
    const aclTensor *bg = nullptr;
    const aclTensor *v = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    int64_t chunkSize = PPFM_CHUNK;
    const aclTensor *hmOut = nullptr;
};

static aclnnStatus CheckNotNull(PreProcessFwdKernelMergedParams params)
{
    if (!(params.k != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(params.w != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "w");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(params.u != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "u");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(params.hmOut != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "hmOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(params.cuSeqlensOptional != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "cuSeqlensOptional");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(PreProcessFwdKernelMergedParams params)
{
    const auto bf16 = DataType::DT_BF16;
    if (!(params.k->GetDataType() == bf16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "k",
            ge::TypeUtils::DataTypeToSerialString(params.k->GetDataType()), "k must be BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.w->GetDataType() == bf16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "w",
            ge::TypeUtils::DataTypeToSerialString(params.w->GetDataType()), "w must be BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.u->GetDataType() == bf16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "u",
            ge::TypeUtils::DataTypeToSerialString(params.u->GetDataType()), "u must be BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // DPLR 不支持：bg / v 是 DPLR 专用输入。两者留着 ABI 槽位，但必须为空——否则会被带到
    // 只注册未实现的 USE_BG 分派上，或者被静默当成 GDN/KDA 算错。host 侧显式拒绝。
    if (!(params.bg == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "DPLR", "bg must be nullptr: DPLR is not implemented in this release (GDN/KDA only)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.v == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "v", "v must be nullptr: it is DPLR-only; GDN/KDA takes the values from u");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool gOk = params.g != nullptr &&
                     (params.g->GetDataType() == DataType::DT_FLOAT || params.g->GetDataType() == bf16);
    const bool gkOk = params.gk != nullptr &&
                      (params.gk->GetDataType() == DataType::DT_FLOAT || params.gk->GetDataType() == bf16);
    if (!(gOk != gkOk)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(OP_NAME, "g,gk",
            std::string("g=") + (params.g != nullptr ? "present" : "absent") + ",gk=" +
                (params.gk != nullptr ? "present" : "absent"),
            "exactly one of g / gk must be given");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.hmOut->GetDataType() == DataType::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "hm",
            ge::TypeUtils::DataTypeToSerialString(params.hmOut->GetDataType()), "hm must be FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(PreProcessFwdKernelMergedParams params)
{
    const auto kShape = params.k->GetViewShape();
    const auto wShape = params.w->GetViewShape();
    const auto uShape = params.u->GetViewShape();
    const auto hmShape = params.hmOut->GetViewShape();

    if (!(kShape.GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "k", std::to_string(kShape.GetDimNum()), "k must be rank 4 ([1,HK,T,K])");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(wShape.GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "w", std::to_string(wShape.GetDimNum()), "w must be rank 4 ([1,HV,T,K])");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(uShape.GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "u", std::to_string(uShape.GetDimNum()), "u must be rank 4 ([1,HV,T,V])");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(hmShape.GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "hm", std::to_string(hmShape.GetDimNum()), "hm must be rank 4 ([Nseq,HV,K,V+K])");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t B = kShape.GetDim(0);
    const int64_t HK = kShape.GetDim(1);
    const int64_t T = kShape.GetDim(2);
    const int64_t K = kShape.GetDim(3);
    const int64_t HV = uShape.GetDim(1);
    const int64_t V = uShape.GetDim(3);

    if (!(B == 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "B", std::to_string(B), "B must be 1 (varlen packed); pack a dense batch first");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(K == PPFM_K && V == PPFM_V)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(OP_NAME, "K/V",
            "K=" + std::to_string(K) + ",V=" + std::to_string(V), "K and V must be 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.chunkSize == PPFM_CHUNK)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "chunkSize",
            std::to_string(params.chunkSize), "chunkSize must be 64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HK > 0 && HV > 0 && HV % HK == 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(OP_NAME, "HV/HK",
            "HV=" + std::to_string(HV) + ",HK=" + std::to_string(HK), "GVA requires HV % HK == 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(wShape.GetDim(0) == 1 && wShape.GetDim(2) == T && wShape.GetDim(3) == K)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "w",
            std::to_string(wShape.GetDim(0)) + "," + std::to_string(wShape.GetDim(2)) + "," +
                std::to_string(wShape.GetDim(3)) + " vs " + std::to_string(T) + "," + std::to_string(K),
            "w must be [1, HV, T, K] matching k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(uShape.GetDim(0) == 1 && uShape.GetDim(2) == T)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "u",
            std::to_string(uShape.GetDim(0)) + "," + std::to_string(uShape.GetDim(2)) + " vs " +
                std::to_string(T),
            "u must be [1, HV, T, V] matching k");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t cuNumel = params.cuSeqlensOptional->Size();
    if (!(cuNumel >= 2)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "cu_seqlens", std::to_string(cuNumel), "cu_seqlens must have >= 2 elements");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t Nseq = cuNumel - 1;
    if (!(hmShape.GetDim(0) == Nseq && hmShape.GetDim(1) == HV && hmShape.GetDim(2) == K &&
               hmShape.GetDim(3) == V + K)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "hm",
            std::to_string(hmShape.GetDim(0)) + "," + std::to_string(hmShape.GetDim(1)) + "," +
                std::to_string(hmShape.GetDim(2)) + "," + std::to_string(hmShape.GetDim(3)) +
                " vs " + std::to_string(Nseq) + "," + std::to_string(HV) + "," +
                std::to_string(K) + "," + std::to_string(V + K),
            "hm must be [Nseq, HV, K, V+K]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t *cuData = params.cuSeqlensOptional->GetData();
    int64_t prev = -1;
    for (int64_t i = 0; i < cuNumel; ++i) {
        const int64_t cur = cuData[i];
        if (!(cur >= 0 && cur <= T)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "cu_seqlens",
            std::to_string(cur) + "," + std::to_string(i) + "," + std::to_string(T),
            "cu_seqlens must satisfy 0 <= cu[i] <= T (sub-interval windows are allowed)");
        return ACLNN_ERR_PARAM_INVALID;
    }
        if (!(cur > prev)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "cu_seqlens",
            std::to_string(cur) + "," + std::to_string(prev) + "," + std::to_string(i),
            "cu_seqlens must be strictly increasing");
        return ACLNN_ERR_PARAM_INVALID;
    }
        prev = cur;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    if (!(tensor != nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(PreProcessFwdKernelMergedParams &params, aclOpExecutor *executor)
{
    if (!(DataContiguous(params.k, executor) == ACLNN_SUCCESS)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(k) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.w, executor) == ACLNN_SUCCESS)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(w) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.u, executor) == ACLNN_SUCCESS)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(u) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.g, executor) == ACLNN_SUCCESS)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(g) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.gk, executor) == ACLNN_SUCCESS)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(gk) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.bg, executor) == ACLNN_SUCCESS)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(bg) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.v, executor) == ACLNN_SUCCESS)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(v) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

// gate 统一成 FP32：BF16 gate 先 Cast（kernel v1 只处理 FP32 gate；bf16 标量转换在设备侧不合法）
static aclnnStatus ParamsGateToFp32(PreProcessFwdKernelMergedParams &params, aclOpExecutor *executor)
{
    if (params.g != nullptr && params.g->GetDataType() != DataType::DT_FLOAT) {
        params.g = l0op::Cast(params.g, DataType::DT_FLOAT, executor);
        if (!(params.g != nullptr)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, "l0op",
                "l0op(g) returned nullptr");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (params.gk != nullptr && params.gk->GetDataType() != DataType::DT_FLOAT) {
        params.gk = l0op::Cast(params.gk, DataType::DT_FLOAT, executor);
        if (!(params.gk != nullptr)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, "l0op",
                "l0op(gk) returned nullptr");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnPreProcessFwdKernelMergedGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *bgOptional,
    const aclTensor *vOptional,
    const aclIntArray *cuSeqlensOptional,
    int64_t chunkSize,
    const aclTensor *hmOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    PreProcessFwdKernelMergedParams params{k,          w,   u,   gOptional, gkOptional,
                                           bgOptional, vOptional, cuSeqlensOptional, chunkSize, hmOut};
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    L2_DFX_PHASE_1(aclnnPreProcessFwdKernelMerged,
                   DFX_IN(k, w, u, gOptional, gkOptional, bgOptional, vOptional, cuSeqlensOptional, chunkSize),
                   DFX_OUT(hmOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "executor",
            "CREATE_EXECUTOR returned nullptr");
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();

    if (!(CheckNotNull(params) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    {
        const aclnnStatus checkStatus = CheckDtype(params);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckShape(params);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    if (!(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(conversion) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(ParamsGateToFp32(params, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "internal",
            "gate data type conversion to fp32 failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto result = l0op::PreProcessFwdKernelMerged(params.k, params.w, params.u, params.g, params.gk, params.bg,
                                                 params.v, params.cuSeqlensOptional, params.chunkSize,
                                                 params.hmOut, executorPtr);
    if (!(result[0] != nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "l0op",
            "l0op(PreProcessFwdKernelMerged) returned nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(result[0], params.hmOut, executorPtr);
    if (!(viewCopyResult != nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "view_copy",
            "ViewCopy(hm) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPreProcessFwdKernelMerged(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPreProcessFwdKernelMerged);
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
