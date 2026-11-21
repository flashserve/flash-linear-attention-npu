/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "gdn_recoverssm_commit.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GdnRecoverssmCommit);

const aclTensor *GdnRecoverssmCommit(
    const aclTensor *state, const aclTensor *replay, const aclTensor *table,
    const aclTensor *stateIndices, const aclTensor *commitLens,
    const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices,
    const aclTensor *boundaryRecoveryLens, int64_t specQueryLen,
    bool alignMode, aclOpExecutor *executor)
{
    L0_DFX(GdnRecoverssmCommit, state, replay, table, stateIndices, commitLens,
           finalStateIndices, boundaryStateIndices, boundaryRecoveryLens,
           specQueryLen, alignMode);
    auto *mutableState = const_cast<aclTensor *>(state);
    const auto storageShape = mutableState->GetStorageShape();
    const auto viewShape = mutableState->GetViewShape();
    const auto viewStrides = mutableState->GetViewStrides();
    const auto viewOffset = mutableState->GetViewOffset();
    auto status = INFER_SHAPE(
        GdnRecoverssmCommit,
        OP_INPUT(state, replay, table, stateIndices, commitLens,
                 finalStateIndices, boundaryStateIndices,
                 boundaryRecoveryLens),
        OP_OUTPUT(mutableState), OP_ATTR(specQueryLen, alignMode));
    if (status != ACLNN_SUCCESS) {
        return nullptr;
    }
    mutableState->SetViewShape(viewShape);
    mutableState->SetViewStrides(viewStrides);
    mutableState->SetViewOffset(viewOffset);
    mutableState->SetStorageShape(storageShape);
    status = ADD_TO_LAUNCHER_LIST_AICORE(
        GdnRecoverssmCommit,
        OP_INPUT(state, replay, table, stateIndices, commitLens,
                 finalStateIndices, boundaryStateIndices,
                 boundaryRecoveryLens),
        OP_OUTPUT(mutableState), OP_ATTR(specQueryLen, alignMode));
    if (status != ACLNN_SUCCESS) {
        return nullptr;
    }
    return mutableState;
}
} // namespace l0op
