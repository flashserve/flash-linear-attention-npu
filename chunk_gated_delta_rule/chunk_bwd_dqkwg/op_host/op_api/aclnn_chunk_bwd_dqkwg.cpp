/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
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
    float scale = 0;
    int64_t chunkSize = 0;
    const aclTensor *dqOut = nullptr;
    const aclTensor *dkOut = nullptr;
    const aclTensor *dwOut = nullptr;
    const aclTensor *dgOut = nullptr;
};

static aclnnStatus CheckNotNull(ChunkBwdDqkwgParams params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(params.h != nullptr, ACLNN_ERR_PARAM_NULLPTR, "h must not be nullptr.");
    CHECK_COND(params.dox != nullptr, ACLNN_ERR_PARAM_NULLPTR, "do must not be nullptr.");
    CHECK_COND(params.dh != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dh must not be nullptr.");
    CHECK_COND(params.dv != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dv must not be nullptr.");

    CHECK_COND(params.dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dqOut must not be nullptr.");
    CHECK_COND(params.dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dkvOut must not be nullptr.");
    CHECK_COND(params.dwOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dwOut must not be nullptr.");
    CHECK_COND(params.dgOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dgOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkBwdDqkwgParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkBwdDqkwgParams params)
{
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
    CHECK_COND(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous q failed.");
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous v failed.");
    CHECK_COND(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");
    CHECK_COND(DataContiguous(params.h, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous h failed.");
    CHECK_COND(DataContiguous(params.dox, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous do failed.");
    CHECK_COND(DataContiguous(params.dh, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dh failed.");
    CHECK_COND(DataContiguous(params.dv, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dv failed.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkBwdDqkwgParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkBwdDqkwgParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
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
    float scale,
    int64_t chunkSize,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dwOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
// std::cout << "1111\n";
    ChunkBwdDqkwgParams params{q, k, v, g, h, dox, dh, dv, cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, dqOut, dkOut, dwOut, dgOut};

    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkBwdDqkwg, DFX_IN(q, k, v, g, h, dox, dh, dv, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(dqOut, dkOut, dwOut, dgOut));
// std::cout << "11111111111--22222\n";
    // 固定写法，创建OpExecutor
// std::cout << "11111111111--2222233\n";
    auto uniqueExecutor = CREATE_EXECUTOR();
// std::cout << "11111111111--2222244\n";
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
// std::cout << "11111111111--222255\n";
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查
// std::cout << "11111111111--2222266\n";
    auto ret = CheckParams(params);
// std::cout << "11111111111--2222277\n";
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
// std::cout << "11111111111--2222288\n";
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
// std::cout << "11111111111--2222299\n";
    auto result = l0op::ChunkBwdDqkwg(params.q, params.k, params.v, params.g, params.h, params.dox, params.dh, params.dv, params.cuSeqlensOptional, params.chunkIndicesOptional, params.scale, params.chunkSize, params.dqOut, params.dkOut, params.dwOut, params.dgOut, executorPtr);
// std::cout << "11111111111--2222200\n";
// printf("result[0] %p, result[1] %p, result[2] : %p, result[3] %p\n",result[0],result[1],result[2],result[3]);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[3] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
// std::cout << "22222AA\n";

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result[0], params.dqOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
// std::cout << "22222BB\n";

    viewCopyResult = l0op::ViewCopy(result[1], params.dkOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
// std::cout << "22222VV\n";

    viewCopyResult = l0op::ViewCopy(result[2], params.dwOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
// std::cout << "22222CC\n";

    viewCopyResult = l0op::ViewCopy(result[3], params.dgOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
// std::cout << "3333\n";

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
// std::cout << "5555\n";

    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkBwdDqkwg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkBwdDqkwg);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkBwdDqkwg launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
