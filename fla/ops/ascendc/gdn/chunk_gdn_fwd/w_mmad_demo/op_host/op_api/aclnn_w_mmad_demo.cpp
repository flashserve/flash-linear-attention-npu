/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "aclnn_w_mmad_demo.h"
#include "w_mmad_demo.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnWMmadDemoGetWorkspaceSize(
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *pre,
    aclTensor *c,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnWMmadDemo, DFX_IN(a, b, pre), DFX_OUT(c));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_COND(a != nullptr, ACLNN_ERR_PARAM_NULLPTR, "a must not be nullptr.");
    CHECK_COND(b != nullptr, ACLNN_ERR_PARAM_NULLPTR, "b must not be nullptr.");
    CHECK_COND(pre != nullptr, ACLNN_ERR_PARAM_NULLPTR, "pre must not be nullptr.");
    CHECK_COND(c != nullptr, ACLNN_ERR_PARAM_NULLPTR, "c must not be nullptr.");

    if (!IsContiguous(a)) {
        a = l0op::Contiguous(a, executorPtr);
        CHECK_RET(a != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!IsContiguous(b)) {
        b = l0op::Contiguous(b, executorPtr);
        CHECK_RET(b != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!IsContiguous(pre)) {
        pre = l0op::Contiguous(pre, executorPtr);
        CHECK_RET(pre != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto result = l0op::WMmadDemo(a, b, pre, c, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto copied = l0op::ViewCopy(result, c, executorPtr);
    CHECK_RET(copied != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnWMmadDemo(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnWMmadDemo);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "error in WMmadDemo launch.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
