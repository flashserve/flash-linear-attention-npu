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

#include "../../../../common/host_utils/log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t PPFM_K = 128;
constexpr int64_t PPFM_V = 128;
constexpr int64_t PPFM_CHUNK = 64;
constexpr const char *PPFM_ENTITY = "aclnnPreProcessFwdKernelMerged";

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
    const aclTensor *required[] = {params.k, params.w, params.u, params.hmOut};
    const char *names[] = {"k", "w", "u", "hmOut"};
    for (size_t idx = 0; idx < sizeof(required) / sizeof(required[0]); ++idx) {
        if (required[idx] == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(PPFM_ENTITY, names[idx]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (params.cuSeqlensOptional == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PPFM_ENTITY, "cuSeqlensOptional");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(PreProcessFwdKernelMergedParams params)
{
    const auto bf16 = DataType::DT_BF16;
    const auto kDtype = params.k->GetDataType();
    if (kDtype != bf16) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PPFM_ENTITY, "k", fla_npu::host_utils::DfxDtypeString(kDtype), "BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.w->GetDataType() != bf16) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PPFM_ENTITY, "w", fla_npu::host_utils::DfxDtypeString(params.w->GetDataType()), "BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.u->GetDataType() != bf16) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PPFM_ENTITY, "u", fla_npu::host_utils::DfxDtypeString(params.u->GetDataType()), "BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // DPLR 不支持：bg / v 是 DPLR 专用输入。两者留着 ABI 槽位，但必须为空——否则会被带到
    // 只注册未实现的 USE_BG 分派上，或者被静默当成 GDN/KDA 算错。host 侧显式拒绝。
    if (params.bg != nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PPFM_ENTITY, "bg", "non-null",
                                                   "nullptr (DPLR not implemented in this release)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.v != nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PPFM_ENTITY, "v", "non-null",
                                                   "nullptr (DPLR-only; GDN/KDA takes values from u)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool gOk = params.g != nullptr &&
                     (params.g->GetDataType() == DataType::DT_FLOAT || params.g->GetDataType() == bf16);
    const bool gkOk = params.gk != nullptr &&
                      (params.gk->GetDataType() == DataType::DT_FLOAT || params.gk->GetDataType() == bf16);
    if (gOk == gkOk) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(PPFM_ENTITY, "g,gk",
                                                    "gOk=" + std::to_string(gOk) + ",gkOk=" + std::to_string(gkOk),
                                                    "exactly one of g / gk must be given");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.hmOut->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PPFM_ENTITY, "hmOut",
                                                   fla_npu::host_utils::DfxDtypeString(params.hmOut->GetDataType()), "FLOAT");
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

    if (kShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(PPFM_ENTITY, "k",
                                                             "rank=" + std::to_string(kShape.GetDimNum()), "rank must be 4 [1,HK,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (wShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(PPFM_ENTITY, "w",
                                                             "rank=" + std::to_string(wShape.GetDimNum()), "rank must be 4 [1,HV,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (uShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(PPFM_ENTITY, "u",
                                                             "rank=" + std::to_string(uShape.GetDimNum()), "rank must be 4 [1,HV,T,V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hmShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(PPFM_ENTITY, "hmOut",
                                                             "rank=" + std::to_string(hmShape.GetDimNum()), "rank must be 4 [Nseq,HV,K,V+K]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t B = kShape.GetDim(0);
    const int64_t HK = kShape.GetDim(1);
    const int64_t T = kShape.GetDim(2);
    const int64_t K = kShape.GetDim(3);
    const int64_t HV = uShape.GetDim(1);
    const int64_t V = uShape.GetDim(3);

    if (B != 1) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PPFM_ENTITY, "k.shape[B]", std::to_string(B),
                                                   "1 (varlen packed)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (K != PPFM_K || V != PPFM_V) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(PPFM_ENTITY, "k.shape[K],u.shape[V]",
                                                    "K=" + std::to_string(K) + ",V=" + std::to_string(V), "K=128, V=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.chunkSize != PPFM_CHUNK) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PPFM_ENTITY, "chunkSize", std::to_string(params.chunkSize), "64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HK > 0 && HV > 0 && HV % HK == 0)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(PPFM_ENTITY, "k.shape[HK],u.shape[HV]",
                                                    "HK=" + std::to_string(HK) + ",HV=" + std::to_string(HV), "HV % HK == 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(wShape.GetDim(0) == 1 && wShape.GetDim(2) == T && wShape.GetDim(3) == K)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(PPFM_ENTITY, "w", "mismatch with k",
                                                             "w must be [1,HV,T,K] matching k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(uShape.GetDim(0) == 1 && uShape.GetDim(2) == T)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(PPFM_ENTITY, "u", "mismatch with k",
                                                             "u must be [1,HV,T,V] matching k");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t cuNumel = params.cuSeqlensOptional->Size();
    if (cuNumel < 2) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(PPFM_ENTITY, "cuSeqlensOptional", std::to_string(cuNumel), ">=2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t Nseq = cuNumel - 1;
    if (!(hmShape.GetDim(0) == Nseq && hmShape.GetDim(1) == HV && hmShape.GetDim(2) == K &&
          hmShape.GetDim(3) == V + K)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(PPFM_ENTITY, "hmOut", "mismatch",
                                                             "hm must be [Nseq=" + std::to_string(Nseq) + ", HV=" + std::to_string(HV) + ", K, V+K]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t *cuData = params.cuSeqlensOptional->GetData();
    int64_t prev = -1;
    for (int64_t i = 0; i < cuNumel; ++i) {
        const int64_t cur = cuData[i];
        if (!(cur >= 0 && cur <= T)) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PPFM_ENTITY, "cuSeqlensOptional[i]",
                                                       "i=" + std::to_string(i) + ",val=" + std::to_string(cur),
                                                       "0 <= cu[i] <= T=" + std::to_string(T));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(cur > prev)) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PPFM_ENTITY, "cuSeqlensOptional", "non-increasing",
                                                       "strictly increasing");
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
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(PreProcessFwdKernelMergedParams &params, aclOpExecutor *executor)
{
    auto status = DataContiguous(params.k, executor);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.w, executor);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.u, executor);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.g, executor);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.gk, executor);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.bg, executor);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.v, executor);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return ACLNN_SUCCESS;
}

// gate 统一成 FP32：BF16 gate 先 Cast（kernel v1 只处理 FP32 gate；bf16 标量转换在设备侧不合法）
static aclnnStatus ParamsGateToFp32(PreProcessFwdKernelMergedParams &params, aclOpExecutor *executor)
{
    if (params.g != nullptr && params.g->GetDataType() != DataType::DT_FLOAT) {
        params.g = l0op::Cast(params.g, DataType::DT_FLOAT, executor);
        CHECK_RET(params.g != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (params.gk != nullptr && params.gk->GetDataType() != DataType::DT_FLOAT) {
        params.gk = l0op::Cast(params.gk, DataType::DT_FLOAT, executor);
        CHECK_RET(params.gk != nullptr, ACLNN_ERR_INNER_NULLPTR);
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
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PPFM_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PPFM_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(aclnnPreProcessFwdKernelMerged,
                   DFX_IN(k, w, u, gOptional, gkOptional, bgOptional, vOptional, cuSeqlensOptional, chunkSize),
                   DFX_OUT(hmOut));
    auto ret = CheckNotNull(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = CheckDtype(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = CheckShape(params);
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
    ret = ParamsGateToFp32(params, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto result = l0op::PreProcessFwdKernelMerged(params.k, params.w, params.u, params.g, params.gk, params.bg,
                                                 params.v, params.cuSeqlensOptional, params.chunkSize,
                                                 params.hmOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result[0], params.hmOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPreProcessFwdKernelMerged(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPreProcessFwdKernelMerged);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in PreProcessFwdKernelMerged launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
