#include "aclnn_recompute_w_u_fwd_experimental.h"
#include "recompute_w_u_fwd_experimental.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"

using namespace op;

extern "C" aclnnStatus aclnnRecomputeWUFwdExperimentalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *y, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(x != nullptr && y != nullptr && workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "experimental build probe requires non-null arguments");
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(l0op::RecomputeWUFwdExperimental(x, y, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_PARAM_INVALID);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnRecomputeWUFwdExperimental(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
