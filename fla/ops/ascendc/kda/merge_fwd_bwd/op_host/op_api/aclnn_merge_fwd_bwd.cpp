/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "aclnn_merge_fwd_bwd.h"
#include "merge_fwd_bwd.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMergeFwdBwdGetWorkspaceSize(
    const aclTensor *agHm, int64_t rank, int64_t preOrPostNumRanks, bool forward, const aclTensor *hOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(agHm != nullptr && hOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "agHm and hOut must not be nullptr.");
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    const aclTensor *agContig = l0op::Contiguous(agHm, executorPtr);
    CHECK_RET(agContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *hContig = l0op::Contiguous(hOut, executorPtr);
    CHECK_RET(hContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outputs = l0op::MergeFwdBwd(agContig, rank, preOrPostNumRanks, forward, hContig, executorPtr);
    CHECK_COND(outputs[0] != nullptr, ACLNN_ERR_INNER_NULLPTR, "MergeFwdBwd launch failed.");

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMergeFwdBwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
