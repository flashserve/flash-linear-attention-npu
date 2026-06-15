/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_chunk_bwd_dv_local.h"
#include "chunk_bwd_dv_local.h"
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

struct ChunkBwdDvLocalParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *dO = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *gGammaOptional = nullptr;
    const aclTensor *aOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64L;
    const aclTensor *out = nullptr;
};

static constexpr size_t Q_K_DO_DIM_NUM = 4;
static constexpr size_t G_DIM_NUM = 3;
static constexpr size_t ACLNN_DIM_3 = 3;
static constexpr int64_t K_DIM_SUPPORTED = 128;
static constexpr int64_t V_DIM_128 = 128;
static constexpr int64_t V_DIM_256 = 256;
static constexpr int64_t CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_SIZE_128 = 128;

static aclnnStatus CheckNotNull(ChunkBwdDvLocalParams params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.dO != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dO must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(params.out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "out must not be nullptr.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDimNumAndNonZero(const aclTensor *tensor, size_t expectedDimNum, const char *tensorName)
{
    const auto &shape = tensor->GetViewShape();
    size_t dimNum = shape.GetDimNum();
    CHECK_COND(dimNum == expectedDimNum, ACLNN_ERR_PARAM_INVALID,
               "Check %s shape failed, dim num should be %zu, but got %zu.", tensorName, expectedDimNum, dimNum);
    for (size_t dimIndex = 0; dimIndex < dimNum; ++dimIndex) {
        CHECK_COND(shape.GetDim(dimIndex) > 0, ACLNN_ERR_PARAM_INVALID,
                   "Check %s shape failed, dim %zu should be positive, but got %ld.", tensorName, dimIndex,
                   shape.GetDim(dimIndex));
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSameDim(const aclTensor *lhs, const aclTensor *rhs, const char *lhsName, const char *rhsName,
                                size_t dimIndex)
{
    int64_t lhsDim = lhs->GetViewShape().GetDim(dimIndex);
    int64_t rhsDim = rhs->GetViewShape().GetDim(dimIndex);
    CHECK_COND(lhsDim == rhsDim, ACLNN_ERR_PARAM_INVALID,
               "Compare %s and %s shape failed, dim %zu should be same, but got %ld and %ld.", lhsName, rhsName,
               dimIndex, lhsDim, rhsDim);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSameLeadingDims(const aclTensor *lhs, const aclTensor *rhs, const char *lhsName,
                                        const char *rhsName, size_t dimNum)
{
    for (size_t dimIndex = 0; dimIndex < dimNum; ++dimIndex) {
        CHECK_RET(CheckSameDim(lhs, rhs, lhsName, rhsName, dimIndex) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkBwdDvLocalParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkBwdDvLocalParams params)
{
    CHECK_COND(params.gGammaOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "gGammaOptional is not supported by chunk_bwd_dv_local currently.");
    CHECK_COND(params.aOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "aOptional is not supported by chunk_bwd_dv_local currently.");
    CHECK_COND(params.chunkSize == CHUNK_SIZE_64 || params.chunkSize == CHUNK_SIZE_128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize should be 64 or 128, but got %ld.", params.chunkSize);

    CHECK_RET(CheckDimNumAndNonZero(params.q, Q_K_DO_DIM_NUM, "q") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDimNumAndNonZero(params.k, Q_K_DO_DIM_NUM, "k") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDimNumAndNonZero(params.dO, Q_K_DO_DIM_NUM, "dO") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDimNumAndNonZero(params.g, G_DIM_NUM, "g") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDimNumAndNonZero(params.out, Q_K_DO_DIM_NUM, "out") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckSameLeadingDims(params.q, params.k, "q", "k", Q_K_DO_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSameLeadingDims(params.q, params.dO, "q", "dO", G_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSameLeadingDims(params.q, params.g, "q", "g", G_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSameLeadingDims(params.dO, params.out, "dO", "out", Q_K_DO_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    int64_t kDim = params.q->GetViewShape().GetDim(ACLNN_DIM_3);
    int64_t vDim = params.dO->GetViewShape().GetDim(ACLNN_DIM_3);
    CHECK_COND(kDim == K_DIM_SUPPORTED, ACLNN_ERR_PARAM_INVALID, "K should be 128, but got %ld.", kDim);
    CHECK_COND(vDim == V_DIM_128 || vDim == V_DIM_256, ACLNN_ERR_PARAM_INVALID,
               "V should be 128 or 256, but got %ld.", vDim);
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkBwdDvLocalParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous q failed.");
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.dO, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dO failed.");
    CHECK_COND(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkBwdDvLocalParams params)
{
    auto qDtype = params.q->GetDataType();
    CHECK_COND(qDtype == DataType::DT_FLOAT16 || qDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
               "q dtype should be float16 or bfloat16.");
    CHECK_COND(params.k->GetDataType() == qDtype, ACLNN_ERR_PARAM_INVALID, "k dtype should be same as q.");
    CHECK_COND(params.dO->GetDataType() == qDtype, ACLNN_ERR_PARAM_INVALID, "dO dtype should be same as q.");
    CHECK_COND(params.out->GetDataType() == qDtype, ACLNN_ERR_PARAM_INVALID, "out dtype should be same as q.");
    CHECK_COND(params.g->GetDataType() == qDtype || params.g->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "g dtype should be same as q or float32.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkBwdDvLocalParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkBwdDvLocalGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *dO,
    const aclTensor *g,
    const aclTensor *gGammaOptional,
    const aclTensor *aOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkBwdDvLocalParams params{q, k, dO, g, gGammaOptional, aOptional, cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, out};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkBwdDvLocal, DFX_IN(q, k, dO, g, gGammaOptional, aOptional, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::ChunkBwdDvLocal(params.q, params.k,  params.dO, params.g, params.gGammaOptional, params.aOptional, params.cuSeqlensOptional, params.chunkIndicesOptional, params.scale, params.chunkSize, params.out, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result, params.out, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkBwdDvLocal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkBwdDvLocal);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkBwdDvLocal launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
