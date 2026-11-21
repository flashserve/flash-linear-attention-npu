/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OP_API_GDN_RECOVERSSM_COMMIT_INNER_H
#define OP_API_GDN_RECOVERSSM_COMMIT_INNER_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus GdnRecoverssmCommitInnerGetWorkspaceSize(
    const aclTensor *state, const aclTensor *replay, const aclTensor *table,
    const aclTensor *stateIndices, const aclTensor *commitLens,
    const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices,
    const aclTensor *boundaryRecoveryLens, int64_t specQueryLen,
    bool alignMode, aclTensor *stateOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus GdnRecoverssmCommitInner(void *workspace, uint64_t workspaceSize,
                                     aclOpExecutor *executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_GDN_RECOVERSSM_COMMIT_INNER_H
