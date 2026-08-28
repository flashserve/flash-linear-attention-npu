/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_scaled_dot_kkt.h"

#include "chunk_scaled_dot_kkt.h"

#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
constexpr int64_t MIN_RANK_K = 4;
constexpr int64_t MIN_RANK_GATE = 3;

struct ChunkScaledDotKktParams {
    const aclTensor *k = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    int64_t chunkSize = 64;
    const aclTensor *aOut = nullptr;
};

bool IsSupportedChunkSize(int64_t chunkSize)
{
    return chunkSize == 16 || chunkSize == 32 || chunkSize == 64 || chunkSize == 128;
}

int64_t Dim(const aclTensor *tensor, size_t index)
{
    return tensor->GetViewShape().GetDim(index);
}

aclnnStatus CheckParams(const ChunkScaledDotKktParams &params)
{
    CHECK_COND(params.k != nullptr && params.g != nullptr && params.beta != nullptr && params.aOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "k, g, beta and aOut must not be nullptr.");
    CHECK_COND(params.k->GetViewShape().GetDimNum() == MIN_RANK_K &&
                   params.g->GetViewShape().GetDimNum() == MIN_RANK_GATE &&
                   params.beta->GetViewShape().GetDimNum() == MIN_RANK_GATE &&
                   params.aOut->GetViewShape().GetDimNum() == MIN_RANK_K,
               ACLNN_ERR_PARAM_INVALID, "expected k/aOut rank 4 and g/beta rank 3.");
    CHECK_COND(IsSupportedChunkSize(params.chunkSize), ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be one of 16, 32, 64 or 128.");
    CHECK_COND((params.cuSeqlensOptional == nullptr) == (params.chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID, "cuSeqlensOptional and chunkIndicesOptional must be provided together.");
    if (params.chunkIndicesOptional != nullptr) {
        CHECK_COND(params.chunkIndicesOptional->Size() % 2 == 0, ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional must contain flattened [sequence, chunk] pairs.");
    }

    const auto kDtype = params.k->GetDataType();
    CHECK_COND(kDtype == DataType::DT_FLOAT16 || kDtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "k must be float16 or bfloat16.");
    CHECK_COND(params.g->GetDataType() == DataType::DT_FLOAT &&
                   params.beta->GetDataType() == DataType::DT_FLOAT &&
                   params.aOut->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "g, beta and aOut must be float32.");

    const int64_t batch = Dim(params.k, 0);
    const int64_t keyHeads = Dim(params.k, 1);
    const int64_t tokens = Dim(params.k, 2);
    const int64_t valueHeads = Dim(params.g, 1);
    CHECK_COND(batch > 0 && keyHeads > 0 && tokens > 0 && Dim(params.k, 3) > 0,
               ACLNN_ERR_PARAM_INVALID, "k dimensions must be positive.");
    CHECK_COND(Dim(params.g, 0) == batch && Dim(params.g, 2) == tokens &&
                   Dim(params.beta, 0) == batch && Dim(params.beta, 1) == valueHeads &&
                   Dim(params.beta, 2) == tokens && valueHeads >= keyHeads && valueHeads % keyHeads == 0,
               ACLNN_ERR_PARAM_INVALID, "g/beta batch, token and head dimensions are incompatible with k.");
    CHECK_COND(Dim(params.aOut, 0) == batch && Dim(params.aOut, 1) == keyHeads &&
                   Dim(params.aOut, 2) == tokens && Dim(params.aOut, 3) == params.chunkSize,
               ACLNN_ERR_PARAM_INVALID, "aOut must have shape [B, Hk, T, chunkSize].");
    return ACLNN_SUCCESS;
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkScaledDotKktGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *aOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkScaledDotKktParams params{k, g, beta, cuSeqlensOptional, chunkIndicesOptional, chunkSize, aOut};
    L2_DFX_PHASE_1(aclnnChunkScaledDotKkt,
                   DFX_IN(k, g, beta, cuSeqlensOptional, chunkIndicesOptional, chunkSize),
                   DFX_OUT(aOut));

    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(MakeContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto result = l0op::ChunkScaledDotKkt(params.k, params.g, params.beta,
                                          params.cuSeqlensOptional, params.chunkIndicesOptional,
                                          params.chunkSize, params.aOut, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result, params.aOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkScaledDotKkt(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkScaledDotKkt);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkScaledDotKkt launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
