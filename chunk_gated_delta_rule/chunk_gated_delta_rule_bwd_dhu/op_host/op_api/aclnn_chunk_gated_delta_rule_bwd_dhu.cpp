/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_chunk_gated_delta_rule_bwd_dhu.h"
#include "chunk_gated_delta_rule_bwd_dhu.h"
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

struct ChunkGatedDeltaRuleBwdDhuParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *dO = nullptr;
    const aclTensor *dv = nullptr;
    const aclTensor *gOptional = nullptr;
    const aclTensor *gkOptional = nullptr;
    const aclTensor *h0Optional = nullptr;
    const aclTensor *dhtOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    double scale = 0.0;
    int64_t chunkSize = 64;
    const aclTensor *dhOut = nullptr;
    const aclTensor *dh0Out = nullptr;
    const aclTensor *dv2Out = nullptr;
};

static aclnnStatus CheckNotNull(ChunkGatedDeltaRuleBwdDhuParams params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.w != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w must not be nullptr.");
    CHECK_COND(params.dO != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dO must not be nullptr.");
    CHECK_COND(params.dv != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dv must not be nullptr.");

    CHECK_COND(params.dhOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dhOut must not be nullptr.");
    CHECK_COND(params.dh0Out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dh0Out must not be nullptr.");
    CHECK_COND(params.dv2Out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dv2Out must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkGatedDeltaRuleBwdDhuParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkGatedDeltaRuleBwdDhuParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkGatedDeltaRuleBwdDhuParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous q failed.");
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.w, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous w failed.");
    CHECK_COND(DataContiguous(params.dO, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dO failed.");
    CHECK_COND(DataContiguous(params.dv, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dv failed.");
    if (params.gOptional != nullptr) {
        CHECK_COND(DataContiguous(params.gOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous gOptional failed.");
    }
    if (params.gkOptional != nullptr) {
        CHECK_COND(DataContiguous(params.gkOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous gkOptional failed.");
    }
    if (params.h0Optional != nullptr) {
        CHECK_COND(DataContiguous(params.h0Optional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous h0Optional failed.");
    }
    if (params.dhtOptional != nullptr) {
        CHECK_COND(DataContiguous(params.dhtOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous dhtOptional failed.");
    }
    CHECK_COND(DataContiguous(params.dhOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dhOut failed.");
    CHECK_COND(DataContiguous(params.dh0Out, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dh0Out failed.");
    CHECK_COND(DataContiguous(params.dv2Out, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dv2Out failed.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkGatedDeltaRuleBwdDhuParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkGatedDeltaRuleBwdDhuParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *dO,
    const aclTensor *dv,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *h0Optional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *dhOut,
    const aclTensor *dh0Out,
    const aclTensor *dv2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkGatedDeltaRuleBwdDhuParams params{
        q, k, w, dO, dv, gOptional, gkOptional, h0Optional, dhtOptional,
        cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, dhOut, dh0Out, dv2Out
    };
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleBwdDhu, DFX_IN(q, k, w, dO, dv, gOptional, gkOptional, h0Optional, dhtOptional, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(dhOut, dh0Out, dv2Out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::ChunkGatedDeltaRuleBwdDhu(params.q, params.k, params.w, params.dO, params.dv, params.gOptional, params.gkOptional, params.h0Optional, params.dhtOptional, params.cuSeqlensOptional, params.chunkIndicesOptional, params.scale, params.chunkSize, params.dhOut, params.dh0Out, params.dv2Out, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result[0], params.dhOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[1], params.dh0Out, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(result[2], params.dv2Out, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkGatedDeltaRuleBwdDhu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleBwdDhu);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in aclnnChunkGatedDeltaRuleBwdDhu launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
