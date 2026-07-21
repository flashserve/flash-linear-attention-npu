/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #include "aclnn_solve_tri.h"
 #include <cstring>
 #include <new>

 #include "aclnn_kernels/contiguous.h"
 #include "acl/acl.h"
 #include "aclnn/aclnn_base.h"
 #include "aclnn_kernels/common/op_error_check.h"
 #include "opdev/common_types.h"
 #include "opdev/op_executor.h"
 #include "opdev/op_log.h"
 #include "opdev/op_dfx.h"
 #include "opdev/make_op_executor.h"
 #include "opdev/tensor_view_utils.h"
 
 using namespace op;
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 struct SolveTriParams {
     const aclTensor *x = nullptr;
     const aclIntArray *cuSeqlens = nullptr;
     const aclIntArray *chunkIndices = nullptr;
     const char *layout = "bsnd";
     const aclTensor *xOut = nullptr;
 };
 
 static aclnnStatus CheckNotNull(SolveTriParams params)
 {
     CHECK_COND(params.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
     CHECK_COND(params.layout != nullptr, ACLNN_ERR_PARAM_NULLPTR, "layout must not be nullptr.");
     CHECK_COND(params.xOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "xOut must not be nullptr.");
     return ACLNN_SUCCESS;
 }

static aclnnStatus CheckLayoutAndShape(SolveTriParams params)
{
     const bool isBhtd = std::strcmp(params.layout, "bhtd") == 0;
     const bool isBsnd = std::strcmp(params.layout, "bsnd") == 0;
     const bool isTnd = std::strcmp(params.layout, "tnd") == 0;
     CHECK_COND(isBhtd || isBsnd || isTnd, ACLNN_ERR_PARAM_INVALID,
                "layout must be lowercase bhtd, bsnd or tnd, but got %s.", params.layout);
     CHECK_COND((params.cuSeqlens == nullptr) == (params.chunkIndices == nullptr), ACLNN_ERR_PARAM_INVALID,
                "cuSeqlens and chunkIndices must both be provided or both be nullptr.");
     CHECK_COND(!isTnd || params.cuSeqlens != nullptr, ACLNN_ERR_PARAM_INVALID,
                "tnd layout requires cuSeqlens and chunkIndices.");
     CHECK_COND(isTnd || params.cuSeqlens == nullptr, ACLNN_ERR_PARAM_INVALID,
                "cuSeqlens and chunkIndices are only valid with tnd layout.");

    const auto xShape = params.x->GetViewShape();
    const auto outShape = params.xOut->GetViewShape();
     const size_t expectedRank = isTnd ? 3 : 4;
     CHECK_COND(xShape.GetDimNum() == expectedRank, ACLNN_ERR_PARAM_INVALID,
                "layout %s requires rank %zu, but got rank %zu.", params.layout, expectedRank, xShape.GetDimNum());
     CHECK_COND(outShape.GetDimNum() == xShape.GetDimNum(), ACLNN_ERR_PARAM_INVALID,
                "xOut rank must match x rank.");
    for (size_t dim = 0; dim < xShape.GetDimNum(); ++dim) {
        CHECK_COND(xShape.GetDim(dim) > 0, ACLNN_ERR_PARAM_INVALID,
                   "x dimensions must be positive; dimension %zu is %ld.", dim, xShape.GetDim(dim));
        CHECK_COND(outShape.GetDim(dim) == xShape.GetDim(dim), ACLNN_ERR_PARAM_INVALID,
                    "xOut shape must match x shape; dimension %zu differs.", dim);
     }
     const int64_t chunkSize = xShape.GetDim(xShape.GetDimNum() - 1);
     CHECK_COND(chunkSize == 16 || chunkSize == 32 || chunkSize == 64 || chunkSize == 128,
                ACLNN_ERR_PARAM_INVALID, "the last dimension C must be 16, 32, 64 or 128, but got %ld.", chunkSize);
     CHECK_COND(params.x->GetDataType() == DataType::DT_FLOAT16 || params.x->GetDataType() == DataType::DT_BF16,
                ACLNN_ERR_PARAM_INVALID, "x dtype must be float16 or bfloat16.");
     CHECK_COND(params.xOut->GetDataType() == params.x->GetDataType(), ACLNN_ERR_PARAM_INVALID,
                "xOut dtype must match x dtype.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckVarlenMetadata(SolveTriParams params)
{
    if (params.cuSeqlens == nullptr) {
        return ACLNN_SUCCESS;
    }
    const auto xShape = params.x->GetViewShape();
    const int64_t totalTokens = xShape.GetDim(0);
    const int64_t chunkSize = xShape.GetDim(xShape.GetDimNum() - 1);
    const aclIntArray &cu = *params.cuSeqlens;
    const aclIntArray &indices = *params.chunkIndices;
    CHECK_COND(cu.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlens must contain at least [0, total_tokens].");
    CHECK_COND(cu[0] == 0 && cu[cu.Size() - 1] == totalTokens, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlens must start at 0 and end at T=%ld.", totalTokens);
    CHECK_COND(indices.Size() % 2 == 0, ACLNN_ERR_PARAM_INVALID,
               "chunkIndices must contain flattened (seq_id, chunk_id) pairs.");

    size_t expectedIndex = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        CHECK_COND(cu[seq] >= 0 && cu[seq] <= cu[seq + 1] && cu[seq + 1] <= totalTokens,
                   ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlens must be nondecreasing and within [0,T].");
        const int64_t chunkCount = (cu[seq + 1] - cu[seq] + chunkSize - 1) / chunkSize;
        for (int64_t localChunk = 0; localChunk < chunkCount; ++localChunk) {
            CHECK_COND(expectedIndex + 1 < indices.Size(), ACLNN_ERR_PARAM_INVALID,
                       "chunkIndices contains fewer pairs than required by cuSeqlens.");
            CHECK_COND(indices[expectedIndex] == static_cast<int64_t>(seq) &&
                           indices[expectedIndex + 1] == localChunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndices must list every chunk in canonical sequence-major order.");
            expectedIndex += 2;
        }
    }
    CHECK_COND(expectedIndex == indices.Size(), ACLNN_ERR_PARAM_INVALID,
               "chunkIndices pair count must equal the chunk count derived from cuSeqlens and C.");
    return ACLNN_SUCCESS;
}
 
 static aclnnStatus CheckParams(SolveTriParams params)
 {
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckLayoutAndShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckVarlenMetadata(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
 
 static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
 {
     tensor = l0op::Contiguous(tensor, executor);
     CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
     return ACLNN_SUCCESS;
 }
 
 static aclnnStatus ParamsDataContiguous(SolveTriParams &params, aclOpExecutor *executorPtr)
 {
     CHECK_COND(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                "Contiguous x failed.");
     return ACLNN_SUCCESS;
 }
 
 aclnnStatus aclnnSolveTriGetWorkspaceSize(
     const aclTensor *x,
     const aclIntArray *cuSeqlens,
     const aclIntArray *chunkIndices,
     const char *layout,
     const aclTensor *xOut,
     uint64_t *workspaceSize,
     aclOpExecutor **executor)
 {
     CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
     CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
     SolveTriParams params{x, cuSeqlens, chunkIndices, layout, xOut};
 
     L2_DFX_PHASE_1(aclnnSolveTri, DFX_IN(x, cuSeqlens, chunkIndices), DFX_OUT(xOut));
 
     auto uniqueExecutor = CREATE_EXECUTOR();
     CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
     auto executorPtr = uniqueExecutor.get();
 
     auto ret = CheckParams(params);
     CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
 
     CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                "ParamsDataContiguous failed.");
 
     auto result = l0op::SolveTri(params.x, params.cuSeqlens, params.chunkIndices,
                                   params.layout, params.xOut, executorPtr);
     CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
 
     auto viewCopyResult = l0op::ViewCopy(result, params.xOut, executorPtr);
     CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
 
     *workspaceSize = uniqueExecutor->GetWorkspaceSize();
     uniqueExecutor.ReleaseTo(executor);
     return ACLNN_SUCCESS;
 }
 
 aclnnStatus aclnnSolveTri(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
 {
     L2_DFX_PHASE_2(aclnnSolveTri);
     CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
                "SolveTri launch failed.");
     return ACLNN_SUCCESS;
 }
 
 #ifdef __cplusplus
 }
 #endif
