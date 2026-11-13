/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
#include "aclnn_solve_tri.h"
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

static constexpr const char *OP_NAME = "aclnnSolveTri";
 
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
     if (params.x == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "x");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
     if (params.xOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "xOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
     return ACLNN_SUCCESS;
 }
 
 static aclnnStatus CheckParams(SolveTriParams params)
 {
     {
         const aclnnStatus checkStatus = CheckNotNull(params);
         if (checkStatus != ACLNN_SUCCESS) {
             return checkStatus;
         }
     };
     return ACLNN_SUCCESS;
 }
 
 static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
 {
     tensor = l0op::Contiguous(tensor, executor);
     if (!(tensor != nullptr)) {
         OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: tensor != nullptr failed", __func__);
         return ACLNN_ERR_INNER_NULLPTR;
     }
     return ACLNN_SUCCESS;
 }
 
 static aclnnStatus ParamsDataContiguous(SolveTriParams &params, aclOpExecutor *executorPtr)
 {
     if (!(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous x failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
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
     if (uniqueExecutor.get() == nullptr) {
         OP_LOGE_WITHOUT_REPORT(OP_NAME,
                            "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
         return ACLNN_ERR_INNER_CREATE_EXECUTOR;
     }
     auto executorPtr = uniqueExecutor.get();
 
     auto ret = CheckParams(params);
     if (!(ret == ACLNN_SUCCESS)) {
        return ret;
    }
 
     if (!(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
 
     auto result = l0op::SolveTri(params.x, params.cuSeqlens, params.chunkIndices,
                                   params.layout, params.xOut, executorPtr);
     if (!(result != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(
            OP_NAME, "%s: operator returned no output; see the preceding tiling/impl error for the root cause",
            __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
 
     auto viewCopyResult = l0op::ViewCopy(result, params.xOut, executorPtr);
     if (!(viewCopyResult != nullptr)) {
         OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult != nullptr failed", __func__);
         return ACLNN_ERR_INNER_NULLPTR;
     }
 
     *workspaceSize = uniqueExecutor->GetWorkspaceSize();
     uniqueExecutor.ReleaseTo(executor);
     return ACLNN_SUCCESS;
 }
 
 aclnnStatus aclnnSolveTri(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
 {
     L2_DFX_PHASE_2(aclnnSolveTri);
     if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "SolveTri launch failed.", __func__);
        return ACLNN_ERR_INNER;
    }
     return ACLNN_SUCCESS;
 }
 
 #ifdef __cplusplus
 }
 #endif
 