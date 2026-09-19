/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef ACLNN_MERGE_FWD_BWD_KERNEL_H
#define ACLNN_MERGE_FWD_BWD_KERNEL_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnMergeFwdBwdKernelGetWorkspaceSize(
    const aclTensor *agHm,
    int64_t rank,
    int64_t preOrPostNumRanks,
    bool forward,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnMergeFwdBwdKernel(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
