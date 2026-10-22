#pragma once

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnRecomputeWUFwdExperimentalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *y, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnRecomputeWUFwdExperimental(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif
