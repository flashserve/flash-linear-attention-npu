/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OP_API_ACLNN_COMPACT_CONV_STATE_H
#define OP_API_ACLNN_COMPACT_CONV_STATE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnCompactConvStateGetWorkspaceSize(
    const aclTensor *convState, const aclTensor *baseAddrs, const aclTensor *blockStrides, const aclTensor *dimStrides,
    const aclTensor *tokenStrides, const aclTensor *stateIndices, const aclTensor *commitLens,
    const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices, const aclTensor *boundaryRecoveryLens,
    int64_t convHistoryLen, bool alignMode, int64_t dimStride, int64_t nullBlockId, const aclTensor *convStateOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnCompactConvState(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_COMPACT_CONV_STATE_H
