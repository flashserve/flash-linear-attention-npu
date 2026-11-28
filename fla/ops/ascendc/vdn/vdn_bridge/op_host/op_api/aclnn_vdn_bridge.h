// SPDX-License-Identifier: BSD-3-Clause
#ifndef ACLNN_VDN_BRIDGE_H
#define ACLNN_VDN_BRIDGE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnVdnBridgeGetWorkspaceSize(const aclTensor *prefix, const aclTensor *suffix,
                                          const aclTensor *bounds, const aclTensor *logs, int64_t start,
                                          const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnVdnBridge(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
