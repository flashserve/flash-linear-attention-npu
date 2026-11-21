/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OP_API_ACLNN_GDN_RECOVERSSM_COMMIT_H
#define OP_API_ACLNN_GDN_RECOVERSSM_COMMIT_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnGdnRecoverssmCommitGetWorkspaceSize(
    const aclTensor *state, const aclTensor *replay, const aclTensor *stateCommitCacheAddressStrideTable,
    const aclTensor *stateIndices, const aclTensor *commitLens, const aclTensor *finalStateIndices,
    const aclTensor *boundaryStateIndices, const aclTensor *boundaryRecoveryLens, int64_t specQueryLen, bool alignMode,
    const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnGdnRecoverssmCommit(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_GDN_RECOVERSSM_COMMIT_H
