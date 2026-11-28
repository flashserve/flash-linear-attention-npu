// SPDX-License-Identifier: BSD-3-Clause
#ifndef ACLNN_VDN_ADD_DIAGONAL_H
#define ACLNN_VDN_ADD_DIAGONAL_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnVdnAddDiagonalGetWorkspaceSize(
    const aclTensor *correction, const aclTensor *twoEye, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnVdnAddDiagonal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_VDN_ADD_DIAGONAL_H
