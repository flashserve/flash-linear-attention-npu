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
     CHECK_COND(params.xOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "xOut must not be nullptr.");
     return ACLNN_SUCCESS;
 }
 
 // layout 取值与 solve_tri_tiling.cpp 的 layoutMode 映射保持一致：bnsd/bhtd->0，bsnd->1，tnd->2，ntd->3；
 // 未知取值在 tiling 会静默落入 BSND，必须在 host 前置拒绝。
 static bool IsDenseLayout(const char *layout)
 {
     return std::strcmp(layout, "bsnd") == 0 || std::strcmp(layout, "bnsd") == 0 ||
            std::strcmp(layout, "bhtd") == 0;
 }
 
 static bool IsVarlenLayout(const char *layout)
 {
     return std::strcmp(layout, "tnd") == 0 || std::strcmp(layout, "ntd") == 0;
 }
 
 static aclnnStatus CheckLayoutAndMetadata(const SolveTriParams &params)
 {
     const char *layout = params.layout != nullptr ? params.layout : "bsnd";
     CHECK_COND(IsDenseLayout(layout) || IsVarlenLayout(layout), ACLNN_ERR_PARAM_INVALID,
                "layout must be one of [bsnd, bnsd, bhtd, tnd, ntd], but got %s.", layout);
     CHECK_COND((params.cuSeqlens == nullptr) == (params.chunkIndices == nullptr),
                ACLNN_ERR_PARAM_INVALID, "cu_seqlens and chunk_indices must be provided together.");
     if (IsDenseLayout(layout)) {
         // dense 布局的 tiling/kernel 不读 varlen 元数据：传入会被静默忽略并按全局 chunk
         // 对齐切分，跨序列边界读出错误结果（issue #819），直接拒绝。
         CHECK_COND(params.cuSeqlens == nullptr, ACLNN_ERR_PARAM_INVALID,
                    "layout %s does not support cu_seqlens/chunk_indices; "
                    "use layout tnd/ntd for varlen input.", layout);
     } else {
         // varlen 布局的 tiling 无条件读取 chunk_indices/cu_seqlens，缺失将解引用空指针。
         CHECK_COND(params.cuSeqlens != nullptr, ACLNN_ERR_PARAM_INVALID,
                    "layout %s requires cu_seqlens and chunk_indices for varlen input.", layout);
     }
     return ACLNN_SUCCESS;
 }
 
 static aclnnStatus CheckParams(SolveTriParams params)
 {
     CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
     return CheckLayoutAndMetadata(params);
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
 
     L2_DFX_PHASE_1(aclnnSolveTri, DFX_IN(x, cuSeqlens, chunkIndices, layout), DFX_OUT(xOut));
 
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
 