/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #include "aclnn_solve_tri.h"
 #include <new>
#include <string>
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
     CHECK_COND(params.xOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "xOut must not be nullptr.");
     return ACLNN_SUCCESS;
 }
 
 static aclnnStatus CheckParams(SolveTriParams params)
 {
    auto nullStatus = CheckNotNull(params);
    CHECK_RET(nullStatus == ACLNN_SUCCESS, nullStatus);
     const auto dtype = params.x->GetDataType();
     CHECK_COND(dtype == DataType::DT_FLOAT16 || dtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
                "SolveTri requires FP16, BF16 or FP32 input.");
     CHECK_COND(params.xOut->GetDataType() == dtype, ACLNN_ERR_PARAM_INVALID,
                "SolveTri input and output dtype must match.");
    const auto xShape = params.x->GetViewShape();
    const auto xOutShape = params.xOut->GetViewShape();
    CHECK_COND(xShape.GetDimNum() == 3 || xShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
               "x must be rank 3 for tnd/ntd or rank 4 for bsnd/bnsd, got rank %zu.",
               xShape.GetDimNum());
    CHECK_COND(xShape == xOutShape, ACLNN_ERR_PARAM_INVALID,
               "xOut shape must exactly match x shape.");
    CHECK_COND(IsContiguous(params.xOut), ACLNN_ERR_PARAM_INVALID, "xOut must be contiguous.");
    const int64_t chunkSize = xShape.GetDim(xShape.GetDimNum() - 1);
    CHECK_COND(chunkSize == 16 || chunkSize == 32 || chunkSize == 64 || chunkSize == 128,
               ACLNN_ERR_PARAM_INVALID,
               "last dimension must be one of 16, 32, 64, 128, but got %ld.", chunkSize);
    const std::string layout = params.layout == nullptr ? "bsnd" : params.layout;
    const bool isDenseLayout = layout == "bsnd" || layout == "bnsd" || layout == "bhtd";
    const bool isVarlenLayout = layout == "tnd" || layout == "ntd";
    CHECK_COND(isDenseLayout || isVarlenLayout, ACLNN_ERR_PARAM_INVALID,
               "layout must be one of bsnd, bnsd, bhtd, tnd, ntd, but got %s.", layout.c_str());
    CHECK_COND((isDenseLayout && xShape.GetDimNum() == 4) ||
                   (isVarlenLayout && xShape.GetDimNum() == 3),
               ACLNN_ERR_PARAM_INVALID, "layout %s is incompatible with rank %zu.",
               layout.c_str(), xShape.GetDimNum());
    CHECK_COND((params.cuSeqlens != nullptr) == (params.chunkIndices != nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be provided together.");
    if (isDenseLayout) {
        CHECK_COND(params.cuSeqlens == nullptr && params.chunkIndices == nullptr,
                   ACLNN_ERR_PARAM_INVALID,
                   "dense layouts must not provide cu_seqlens or chunk_indices.");
        return ACLNN_SUCCESS;
    }
    CHECK_COND(params.cuSeqlens != nullptr && params.chunkIndices != nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "tnd/ntd layout requires cu_seqlens and chunk_indices.");
    CHECK_COND(params.cuSeqlens->Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must contain at least [0, total_T].");
    const int64_t totalTokens = layout == "tnd" ? xShape.GetDim(0) : xShape.GetDim(1);
    CHECK_COND((*params.cuSeqlens)[0] == 0 &&
                   (*params.cuSeqlens)[params.cuSeqlens->Size() - 1] == totalTokens,
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must start at 0 and end at total_T=%ld.", totalTokens);
    int64_t expectedChunks = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlens->Size(); ++seq) {
        const int64_t begin = (*params.cuSeqlens)[seq];
        const int64_t end = (*params.cuSeqlens)[seq + 1];
        CHECK_COND(begin >= 0 && end > begin && end <= totalTokens, ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens sequence %zu must satisfy 0 <= begin < end <= %ld, got [%ld, %ld].",
                   seq, totalTokens, begin, end);
        const int64_t length = end - begin;
        expectedChunks += (length - 1) / chunkSize + 1;
    }
    CHECK_COND(params.chunkIndices->Size() == static_cast<size_t>(expectedChunks) * 2,
               ACLNN_ERR_PARAM_INVALID,
               "chunk_indices must contain one [seq_id, chunk_id] pair per chunk.");
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlens->Size(); ++seq) {
        const int64_t chunks =
            ((*params.cuSeqlens)[seq + 1] - (*params.cuSeqlens)[seq] - 1) / chunkSize + 1;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            CHECK_COND((*params.chunkIndices)[offset] == static_cast<int64_t>(seq) &&
                           (*params.chunkIndices)[offset + 1] == chunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunk_indices[%zu] must be [seq_id=%zu, chunk_id=%ld], but got [%ld, %ld].",
                       offset / 2, seq, chunk, (*params.chunkIndices)[offset],
                       (*params.chunkIndices)[offset + 1]);
            offset += 2;
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
     SolveTriParams params{x, cuSeqlens, chunkIndices, layout, xOut};

     CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                "workspaceSize and executor must not be nullptr.");
 
     L2_DFX_PHASE_1(aclnnSolveTri, DFX_IN(x, cuSeqlens, chunkIndices, layout), DFX_OUT(xOut));
 
     auto uniqueExecutor = CREATE_EXECUTOR();
     CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
     auto executorPtr = uniqueExecutor.get();
 
     auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
 
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
