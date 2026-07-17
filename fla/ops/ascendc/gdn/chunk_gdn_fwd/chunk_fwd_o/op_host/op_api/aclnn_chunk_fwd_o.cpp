/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_chunk_fwd_o.h"
#include "chunk_fwd_o.h"
#include <cmath>
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

static constexpr size_t CHUNK_FWD_O_QKV_DIM_NUM = 4;
static constexpr size_t CHUNK_FWD_O_H_DIM_NUM = 5;
static constexpr size_t CHUNK_FWD_O_G_DIM_NUM = 3;
static constexpr size_t CHUNK_FWD_O_DIM_HEAD_DIM = 3;
static constexpr int64_t CHUNK_FWD_O_MAX_V_HEAD_DIM = 256;
static constexpr int64_t CHUNK_FWD_O_K_HEAD_DIM = 128;
static constexpr int64_t CHUNK_FWD_O_V_HEAD_DIM_128 = 128;
static constexpr int64_t CHUNK_FWD_O_V_HEAD_DIM_256 = 256;
static constexpr int64_t CHUNK_FWD_O_CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_FWD_O_CHUNK_SIZE_128 = 128;

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkFwdOParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *h = nullptr;
    const aclTensor *g = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkOffsetsOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    const aclTensor *oOut = nullptr;
};

static aclnnStatus CheckNotNull(ChunkFwdOParams params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.h != nullptr, ACLNN_ERR_PARAM_NULLPTR, "h must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");

    CHECK_COND(params.oOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "oOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkFwdOParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkFwdOParams params)
{
    const auto &qShape = params.q->GetViewShape();
    const auto &kShape = params.k->GetViewShape();
    const auto &vShape = params.v->GetViewShape();
    const auto &hShape = params.h->GetViewShape();
    const auto &gShape = params.g->GetViewShape();
    const auto &oShape = params.oOut->GetViewShape();
    CHECK_COND(qShape.GetDimNum() == CHUNK_FWD_O_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "q must be [B, HK, T, K].");
    CHECK_COND(kShape.GetDimNum() == CHUNK_FWD_O_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "k must be [B, HK, T, K].");
    CHECK_COND(vShape.GetDimNum() == CHUNK_FWD_O_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "v must be [B, HV, T, V].");
    CHECK_COND(hShape.GetDimNum() == CHUNK_FWD_O_H_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "h must be [B, HV, NC, K, V].");
    CHECK_COND(gShape.GetDimNum() == CHUNK_FWD_O_G_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "g must be [B, HV, T].");
    CHECK_COND(oShape.GetDimNum() == CHUNK_FWD_O_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "oOut must be [B, HV, T, V].");

    const int64_t batch = qShape.GetDim(0);
    const int64_t keyHeads = qShape.GetDim(1);
    const int64_t seqlen = qShape.GetDim(2);
    const int64_t keyDim = qShape.GetDim(3);
    const int64_t valueHeads = vShape.GetDim(1);
    const int64_t valueDim = vShape.GetDim(3);
    CHECK_COND(batch > 0 && seqlen > 0 && keyDim > 0 && valueDim > 0, ACLNN_ERR_PARAM_INVALID,
               "B, T, K and V must be positive.");
    CHECK_COND(keyHeads > 0 && valueHeads > 0 && valueHeads % keyHeads == 0, ACLNN_ERR_PARAM_INVALID,
               "GVA requires HV divisible by HK.");
    CHECK_COND(kShape.GetDim(0) == batch && kShape.GetDim(1) == keyHeads && kShape.GetDim(2) == seqlen &&
                   kShape.GetDim(3) == keyDim,
               ACLNN_ERR_PARAM_INVALID, "k must match q shape [B, HK, T, K].");
    CHECK_COND(vShape.GetDim(0) == batch && vShape.GetDim(2) == seqlen, ACLNN_ERR_PARAM_INVALID,
               "v must match q batch and sequence dimensions.");
    CHECK_COND(hShape.GetDim(0) == batch && hShape.GetDim(1) == valueHeads && hShape.GetDim(2) > 0 &&
                   hShape.GetDim(3) == keyDim && hShape.GetDim(4) == valueDim,
               ACLNN_ERR_PARAM_INVALID, "h must be [B, HV, NC, K, V] and NC must be positive.");
    CHECK_COND(gShape.GetDim(0) == batch && gShape.GetDim(1) == valueHeads && gShape.GetDim(2) == seqlen,
               ACLNN_ERR_PARAM_INVALID, "g must be [B, HV, T].");
    CHECK_COND(oShape.GetDim(0) == batch && oShape.GetDim(1) == valueHeads && oShape.GetDim(2) == seqlen &&
                   oShape.GetDim(3) == valueDim,
               ACLNN_ERR_PARAM_INVALID, "oOut must match v shape [B, HV, T, V].");
    CHECK_COND(keyDim == CHUNK_FWD_O_K_HEAD_DIM, ACLNN_ERR_PARAM_INVALID,
               "K must be %ld, but got %ld.", CHUNK_FWD_O_K_HEAD_DIM, keyDim);
    CHECK_COND(valueDim == CHUNK_FWD_O_V_HEAD_DIM_128 || valueDim == CHUNK_FWD_O_V_HEAD_DIM_256,
               ACLNN_ERR_PARAM_INVALID, "V must be 128 or %ld, but got %ld.",
               CHUNK_FWD_O_MAX_V_HEAD_DIM, valueDim);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckChunkMetadata(ChunkFwdOParams params)
{
    const auto &qShape = params.q->GetViewShape();
    const auto &hShape = params.h->GetViewShape();
    const int64_t batch = qShape.GetDim(0);
    const int64_t seqlen = qShape.GetDim(2);
    int64_t expectedChunks = 0;
    if (params.cuSeqlensOptional == nullptr) {
        expectedChunks = (seqlen + params.chunkSize - 1) / params.chunkSize;
    } else {
        CHECK_COND(batch == 1, ACLNN_ERR_PARAM_INVALID,
                   "B must be 1 when cuSeqlensOptional is provided, but got %ld.", batch);
        const aclIntArray &cu = *params.cuSeqlensOptional;
        const aclIntArray &offsets = *params.chunkOffsetsOptional;
        CHECK_COND(cu.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must contain at least [0, total_tokens].");
        CHECK_COND(cu[0] == 0 && cu[cu.Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must start at 0 and end at T=%ld.", seqlen);
        CHECK_COND(offsets.Size() % 2 == 0, ACLNN_ERR_PARAM_INVALID,
                   "chunkOffsetsOptional must contain flattened (seq_id, chunk_id) pairs.");
        size_t offsetIndex = 0;
        for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
            CHECK_COND(cu[seq] >= 0 && cu[seq] <= cu[seq + 1] && cu[seq + 1] <= seqlen,
                       ACLNN_ERR_PARAM_INVALID,
                       "cuSeqlensOptional must be nondecreasing and within [0,T].");
            const int64_t localChunkCount =
                (cu[seq + 1] - cu[seq] + params.chunkSize - 1) / params.chunkSize;
            for (int64_t localChunk = 0; localChunk < localChunkCount; ++localChunk) {
                CHECK_COND(offsetIndex + 1 < offsets.Size(), ACLNN_ERR_PARAM_INVALID,
                           "chunkOffsetsOptional contains fewer pairs than required by cuSeqlensOptional.");
                CHECK_COND(offsets[offsetIndex] == static_cast<int64_t>(seq) &&
                               offsets[offsetIndex + 1] == localChunk,
                           ACLNN_ERR_PARAM_INVALID,
                           "chunkOffsetsOptional must use canonical sequence-major chunk order.");
                offsetIndex += 2;
                ++expectedChunks;
            }
        }
        CHECK_COND(offsetIndex == offsets.Size(), ACLNN_ERR_PARAM_INVALID,
                   "chunkOffsetsOptional pair count must match cuSeqlensOptional and chunkSize.");
    }
    CHECK_COND(hShape.GetDim(2) == expectedChunks, ACLNN_ERR_PARAM_INVALID,
               "h N_c must equal the chunk count derived from T/cuSeqlensOptional and chunkSize; expected %ld, got %ld.",
               expectedChunks, hShape.GetDim(2));
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkFwdOParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous q failed.");
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous v failed.");
    CHECK_COND(DataContiguous(params.h, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous h failed.");
    CHECK_COND(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkFwdOParams params)
{
    const auto inputDtype = params.q->GetDataType();
    CHECK_COND(inputDtype == DataType::DT_FLOAT16 || inputDtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "q dtype must be float16 or bfloat16.");
    CHECK_COND(params.k->GetDataType() == inputDtype && params.v->GetDataType() == inputDtype &&
                   params.h->GetDataType() == inputDtype && params.oOut->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "k, v, h and oOut dtype must match q.");
    CHECK_COND(params.g->GetDataType() == DataType::DT_FLOAT || params.g->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "g dtype must be float32 or match q dtype.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkFwdOParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND((params.cuSeqlensOptional == nullptr) == (params.chunkOffsetsOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional and chunkOffsetsOptional must both be provided or both be nullptr.");
    CHECK_COND(params.chunkSize == CHUNK_FWD_O_CHUNK_SIZE_64 || params.chunkSize == CHUNK_FWD_O_CHUNK_SIZE_128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be 64 or 128, but got %ld.", params.chunkSize);
    CHECK_COND(std::isfinite(params.scale), ACLNN_ERR_PARAM_INVALID, "scale must be finite.");
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckChunkMetadata(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkFwdOGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *h,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkOffsetsOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *oOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    ChunkFwdOParams params{q, k, v, h, g, cuSeqlensOptional, chunkOffsetsOptional, scale, chunkSize, oOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkFwdO, DFX_IN(q, k, v, h, g, cuSeqlensOptional, chunkOffsetsOptional),
                   DFX_OUT(oOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::ChunkFwdO(params.q, params.k, params.v, params.h, params.g, params.cuSeqlensOptional, params.chunkOffsetsOptional, params.scale, params.chunkSize, params.oOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result[0], params.oOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkFwdO(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkFwdO);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkFwdO launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
