/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OP_API_COMPACT_CONV_STATE_INNER_H
#define OP_API_COMPACT_CONV_STATE_INNER_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus CompactConvStateInnerGetWorkspaceSize(
    const aclTensor *convState, const aclTensor *baseAddrs, const aclTensor *blockStrides, const aclTensor *dimStrides,
    const aclTensor *tokenStrides, const aclTensor *stateIndices, const aclTensor *commitLens,
    const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices, const aclTensor *boundaryRecoveryLens,
    int64_t convHistoryLen, bool alignMode, int64_t dimStride, int64_t nullBlockId, aclTensor *convStateOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus CompactConvStateInner(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_COMPACT_CONV_STATE_INNER_H
