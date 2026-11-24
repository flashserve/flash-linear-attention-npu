/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
// Load structured reporters before opdev's legacy logging macros. The opdev
// headers must retain their logging convention for CHECK_RET and DFX helpers.
#include "op_common/log/log.h"
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

using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkBwdDqkwg";

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
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.k == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.v == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "v");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.g == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.h == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "h");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dox == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dO");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dh == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dh");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dv == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dv");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (params.dqOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dqOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dkOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dkOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dwOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dwOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dgOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dgOut");
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

    const auto checkRank = [](const op::Shape &shape, const char *name, int64_t expect) -> aclnnStatus {
        if (shape.GetDimNum() != static_cast<size_t>(expect)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, name, std::to_string(shape.GetDimNum()),
                std::string(name) + " must be rank " + std::to_string(expect));
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    };
    const std::pair<const op::Shape *, const char *> rankChecks[] = {
        {&qShape, "q"}, {&kShape, "k"}, {&vShape, "v"}, {&gShape, "g"},
        {&dqShape, "dqOut"}, {&dkShape, "dkOut"}, {&dwShape, "dwOut"}, {&dgShape, "dgOut"}};
    const int64_t rankExpects[] = {4, 4, 4, 3, 4, 4, 4, 3};
    for (size_t i = 0; i < sizeof(rankChecks) / sizeof(rankChecks[0]); ++i) {
        const aclnnStatus rankStatus = checkRank(*rankChecks[i].first, rankChecks[i].second, rankExpects[i]);
        if (rankStatus != ACLNN_SUCCESS) {
            return rankStatus;
        }
    }

    const int64_t B = qShape.GetDim(0);
    const int64_t HK = qShape.GetDim(1);
    const int64_t T = qShape.GetDim(2);
    const int64_t K = qShape.GetDim(3);
    const int64_t HV = vShape.GetDim(1);

    // 先判正数：HK/HV 为 0 时既非法也会让下面的取模除零。
    if (!(HK > 0 && HV > 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "HV/HK", "HV=" + std::to_string(HV) + ",HK=" + std::to_string(HK),
            "HV and HK must be positive");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // GVA 的 head 分组要求 HV 是 HK 的整数倍（n_ratio = HV / HK）。
    if (!(HV % HK == 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "HV/HK", "HV=" + std::to_string(HV) + ",HK=" + std::to_string(HK),
            "HV must be divisible by HK (GVA head grouping)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto shapeMatches = [](const op::Shape &shape, std::initializer_list<int64_t> dims) {
        if (shape.GetDimNum() != dims.size()) {
            return false;
        }
        size_t i = 0;
        for (auto d : dims) {
            if (shape.GetDim(i++) != d) {
                return false;
            }
        }
        return true;
    };
    const auto shapeText = [](const op::Shape &shape) {
        std::string text = "[";
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            if (i != 0) {
                text += ",";
            }
            text += std::to_string(shape.GetDim(i));
        }
        return text + "]";
    };
    const auto requireSameShape = [&](const op::Shape &shape, const char *name,
                                      std::initializer_list<int64_t> dims,
                                      const char *expect) -> aclnnStatus {
        if (shapeMatches(shape, dims)) {
            return ACLNN_SUCCESS;
        }
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, name, shapeText(shape), expect);
        return ACLNN_ERR_PARAM_INVALID;
    };
    {
        const aclnnStatus kStatus = requireSameShape(kShape, "k", {B, HK, T, K}, "k must match q shape [B,HK,T,K]");
        if (kStatus != ACLNN_SUCCESS) {
            return kStatus;
        }
        const aclnnStatus vStatus = requireSameShape(vShape, "v", {B, HV, T, vShape.GetDim(3)}, "v must be [B,HV,T,V]");
        if (vStatus != ACLNN_SUCCESS) {
            return vStatus;
        }
        const aclnnStatus gStatus = requireSameShape(gShape, "g", {B, HV, T}, "g must be [B,HV,T]");
        if (gStatus != ACLNN_SUCCESS) {
            return gStatus;
        }
        const aclnnStatus dqStatus = requireSameShape(dqShape, "dqOut", {B, HK, T, K}, "dqOut must be [B,HK,T,K]");
        if (dqStatus != ACLNN_SUCCESS) {
            return dqStatus;
        }
        const aclnnStatus dkStatus = requireSameShape(dkShape, "dkOut", {B, HK, T, K}, "dkOut must be [B,HK,T,K]");
        if (dkStatus != ACLNN_SUCCESS) {
            return dkStatus;
        }
        const aclnnStatus dwStatus = requireSameShape(dwShape, "dwOut", {B, HV, T, K}, "dwOut must be [B,HV,T,K]");
        if (dwStatus != ACLNN_SUCCESS) {
            return dwStatus;
        }
        const aclnnStatus dgStatus = requireSameShape(dgShape, "dgOut", {B, HV, T}, "dgOut must be [B,HV,T]");
        if (dgStatus != ACLNN_SUCCESS) {
            return dgStatus;
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

static aclnnStatus ParamsDataContiguous(ChunkBwdDqkwgParams &params, aclOpExecutor *executorPtr)
{
    if (DataContiguous(params.q, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous q failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.k, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous k failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.v, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous v failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.g, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous g failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.h, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous h failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.dox, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous dox failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.dh, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous dh failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.dv, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous dv failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkBwdDqkwgParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkBwdDqkwgParams params)
{
    {
        const aclnnStatus checkRet = CheckNotNull(params);
        if (checkRet != ACLNN_SUCCESS) {
            return checkRet;
        }
    }
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
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
    if (use_exp2 || transpose_state_layout) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "use_exp2,transpose_state_layout",
            std::string("use_exp2=") + (use_exp2 ? "true" : "false") +
                ",transpose_state_layout=" + (transpose_state_layout ? "true" : "false"),
            "only false is supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Standard syntax, Check parameters.
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

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
    if (ParamsDataContiguous(params, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: input contiguous conversion failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto result = l0op::ChunkBwdDqkwg(params.q, params.k, params.v, params.g, params.h, params.dox, params.dh, params.dv, params.cuSeqlensOptional, params.chunkIndicesOptional, params.w, params.gGamma, params.scale, params.chunkSize, executorPtr);

    if (result[0] == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: operator returned no output", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (result[1] == nullptr || result[2] == nullptr || result[3] == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: operator returned null output (dk/dw/dg)", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(result[0], params.dqOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult is nullptr", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    viewCopyResult = l0op::ViewCopy(result[1], params.dkOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult is nullptr", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    viewCopyResult = l0op::ViewCopy(result[2], params.dwOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult is nullptr", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    viewCopyResult = l0op::ViewCopy(result[3], params.dgOut, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult is nullptr", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkBwdDqkwg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkBwdDqkwg);
    const aclnnStatus runRet = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (runRet != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: operator launch failed, rc=%d", __func__,
                               static_cast<int>(runRet));
        return runRet;
    }
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
