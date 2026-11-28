// SPDX-License-Identifier: BSD-3-Clause
#ifndef ACLNN_VDN_SCAN_H
#define ACLNN_VDN_SCAN_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnVdnScanGetWorkspaceSize(
    const aclTensor *transition, const aclTensor *prefix, const aclTensor *suffix, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnVdnScan(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_VDN_SCAN_H
