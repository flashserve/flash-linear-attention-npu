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
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.w != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w must not be nullptr.");
    CHECK_COND(params.u != nullptr, ACLNN_ERR_PARAM_NULLPTR, "u must not be nullptr.");
    CHECK_COND(params.hmOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hmOut must not be nullptr.");
    CHECK_COND(params.cuSeqlensOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "cuSeqlensOptional must not be nullptr (this operator is varlen only).");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(PreProcessFwdKernelMergedParams params)
{
    const auto bf16 = DataType::DT_BF16;
    CHECK_COND(params.k->GetDataType() == bf16, ACLNN_ERR_PARAM_INVALID, "k must be BF16.");
    CHECK_COND(params.w->GetDataType() == bf16, ACLNN_ERR_PARAM_INVALID, "w must be BF16.");
    CHECK_COND(params.u->GetDataType() == bf16, ACLNN_ERR_PARAM_INVALID, "u must be BF16.");
    // DPLR 不支持：bg / v 是 DPLR 专用输入。两者留着 ABI 槽位，但必须为空——否则会被带到
    // 只注册未实现的 USE_BG 分派上，或者被静默当成 GDN/KDA 算错。host 侧显式拒绝。
    CHECK_COND(params.bg == nullptr, ACLNN_ERR_PARAM_INVALID,
               "bg must be nullptr: DPLR is not implemented in this release (GDN/KDA only).");
    CHECK_COND(params.v == nullptr, ACLNN_ERR_PARAM_INVALID,
               "v must be nullptr: it is DPLR-only; GDN/KDA takes the values from u.");
    const bool gOk = params.g != nullptr &&
                     (params.g->GetDataType() == DataType::DT_FLOAT || params.g->GetDataType() == bf16);
    const bool gkOk = params.gk != nullptr &&
                      (params.gk->GetDataType() == DataType::DT_FLOAT || params.gk->GetDataType() == bf16);
    CHECK_COND(gOk != gkOk, ACLNN_ERR_PARAM_INVALID, "exactly one of g / gk must be given.");
    CHECK_COND(params.hmOut->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID, "hm must be FP32.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(PreProcessFwdKernelMergedParams params)
{
    const auto kShape = params.k->GetViewShape();
    const auto wShape = params.w->GetViewShape();
    const auto uShape = params.u->GetViewShape();
    const auto hmShape = params.hmOut->GetViewShape();

    CHECK_COND(kShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "k must be [1, HK, T, K].");
    CHECK_COND(wShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "w must be [1, HV, T, K].");
    CHECK_COND(uShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "u must be [1, HV, T, V].");
    CHECK_COND(hmShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "hm must be [Nseq, HV, K, V+K].");

    const int64_t B = kShape.GetDim(0);
    const int64_t HK = kShape.GetDim(1);
    const int64_t T = kShape.GetDim(2);
    const int64_t K = kShape.GetDim(3);
    const int64_t HV = uShape.GetDim(1);
    const int64_t V = uShape.GetDim(3);

    CHECK_COND(B == 1, ACLNN_ERR_PARAM_INVALID, "B must be 1 (varlen packed); pack a dense batch first.");
    CHECK_COND(K == PPFM_K && V == PPFM_V, ACLNN_ERR_PARAM_INVALID, "K and V must be 128.");
    CHECK_COND(params.chunkSize == PPFM_CHUNK, ACLNN_ERR_PARAM_INVALID, "chunk_size must be 64.");
    CHECK_COND(HK > 0 && HV > 0 && HV % HK == 0, ACLNN_ERR_PARAM_INVALID, "GVA requires HV % HK == 0.");
    CHECK_COND(wShape.GetDim(0) == 1 && wShape.GetDim(2) == T && wShape.GetDim(3) == K,
               ACLNN_ERR_PARAM_INVALID, "w must be [1, HV, T, K] matching k.");
    CHECK_COND(wShape.GetDim(1) == HV, ACLNN_ERR_PARAM_INVALID,
               "w head dim must equal HV (u dim 1).");
    CHECK_COND(uShape.GetDim(0) == 1 && uShape.GetDim(2) == T,
               ACLNN_ERR_PARAM_INVALID, "u must be [1, HV, T, V] matching k.");
    // 可选输入 g / gk 的 shape 必须与 docs/api.md §2.1 的契约一致：
    //   g  [1, HV, T]      gk [1, HV, T, K]     （gate 按 value head HV，不是 HK）
    // 两者的存在性与互斥已由 CheckDtype 负责，这里只补 shape 拦截。
    if (params.g != nullptr) {
        const auto gShape = params.g->GetViewShape();
        CHECK_COND(gShape.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID, "g must be [1, HV, T].");
        CHECK_COND(gShape.GetDim(0) == 1 && gShape.GetDim(1) == HV && gShape.GetDim(2) == T,
                   ACLNN_ERR_PARAM_INVALID, "g must be [1, HV, T] matching u's HV and k's T.");
    }
    if (params.gk != nullptr) {
        const auto gkShape = params.gk->GetViewShape();
        CHECK_COND(gkShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "gk must be [1, HV, T, K].");
        CHECK_COND(gkShape.GetDim(0) == 1 && gkShape.GetDim(1) == HV && gkShape.GetDim(2) == T &&
                   gkShape.GetDim(3) == K,
                   ACLNN_ERR_PARAM_INVALID, "gk must be [1, HV, T, K] matching u's HV and k's T/K.");
    }

    const int64_t cuNumel = params.cuSeqlensOptional->Size();
    CHECK_COND(cuNumel >= 2, ACLNN_ERR_PARAM_INVALID, "cu_seqlens must have >= 2 elements.");
    const int64_t Nseq = cuNumel - 1;
    CHECK_COND(hmShape.GetDim(0) == Nseq && hmShape.GetDim(1) == HV && hmShape.GetDim(2) == K &&
               hmShape.GetDim(3) == V + K,
               ACLNN_ERR_PARAM_INVALID, "hm must be [Nseq, HV, K, V+K].");

    const int64_t *cuData = params.cuSeqlensOptional->GetData();
    int64_t prev = -1;
    for (int64_t i = 0; i < cuNumel; ++i) {
        const int64_t cur = cuData[i];
        CHECK_COND(cur >= 0 && cur <= T, ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens must satisfy 0 <= cu[i] <= T (sub-interval windows are allowed).");
        CHECK_COND(cur > prev, ACLNN_ERR_PARAM_INVALID, "cu_seqlens must be strictly increasing.");
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
    CHECK_COND(DataContiguous(params.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.w, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous w failed.");
    CHECK_COND(DataContiguous(params.u, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous u failed.");
    CHECK_COND(DataContiguous(params.g, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous g failed.");
    CHECK_COND(DataContiguous(params.gk, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous gk failed.");
    CHECK_COND(DataContiguous(params.bg, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous bg failed.");
    CHECK_COND(DataContiguous(params.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous v failed.");
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
    L2_DFX_PHASE_1(aclnnPreProcessFwdKernelMerged,
                   DFX_IN(k, w, u, gOptional, gkOptional, bgOptional, vOptional, cuSeqlensOptional, chunkSize),
                   DFX_OUT(hmOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    CHECK_COND(ParamsGateToFp32(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsGateToFp32 failed.");

    auto result = l0op::PreProcessFwdKernelMerged(params.k, params.w, params.u, params.g, params.gk, params.bg,
                                                 params.v, params.cuSeqlensOptional, params.chunkSize,
                                                 params.hmOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

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
