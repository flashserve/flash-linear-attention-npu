/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_chunk_delta_h_bwd_preprocess.h"
#include "chunk_delta_h_bwd_preprocess.h"
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

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkDeltaHBwdPreprocessParams {
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

static aclnnStatus CheckNotNull(ChunkDeltaHBwdPreprocessParams params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q is nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k is nullptr.");
    CHECK_COND(params.w != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w is nullptr.");
    CHECK_COND(params.dO != nullptr, ACLNN_ERR_PARAM_NULLPTR, "d_o is nullptr.");
    CHECK_COND(params.dv != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dv is nullptr.");
    CHECK_COND(params.dhmOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dhm is nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous failed.");
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkDeltaHBwdPreprocessParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR, "q Contiguous failed.");
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR, "k Contiguous failed.");
    CHECK_COND(DataContiguous(params.w, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR, "w Contiguous failed.");
    CHECK_COND(DataContiguous(params.dO, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
               "d_o Contiguous failed.");
    CHECK_COND(DataContiguous(params.dv, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
               "dv Contiguous failed.");
    if (params.gOptional != nullptr) {
        CHECK_COND(DataContiguous(params.gOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
                   "g Contiguous failed.");
    }
    if (params.gkOptional != nullptr) {
        CHECK_COND(DataContiguous(params.gkOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
                   "gk Contiguous failed.");
    }
    return ACLNN_SUCCESS;
}

// shape/dtype 的完整约束由 host tiling 统一拦截（见 tiling_processor），此处只做非空与连续性检查，
// 避免同一套规则在两处漂移。
static aclnnStatus CheckParams(ChunkDeltaHBwdPreprocessParams params)
{
    auto ret = CheckNotNull(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkDeltaHBwdPreprocessGetWorkspaceSize(
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
    ChunkDeltaHBwdPreprocessParams params{q, k, w, dO, dv, gOptional, gkOptional, cuSeqlensOptional, scale, chunkSize,
                                          dhmOut};
    L2_DFX_PHASE_1(aclnnChunkDeltaHBwdPreprocess,
                   DFX_IN(q, k, w, dO, dv, gOptional, gkOptional, cuSeqlensOptional), DFX_OUT(dhmOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::ChunkDeltaHBwdPreprocess(params.q, params.k, params.w, params.dO, params.dv, params.gOptional,
                                                 params.gkOptional, params.cuSeqlensOptional, params.scale,
                                                 params.chunkSize, params.dhmOut, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkDeltaHBwdPreprocess(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkDeltaHBwdPreprocess);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in aclnnChunkDeltaHBwdPreprocess launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
