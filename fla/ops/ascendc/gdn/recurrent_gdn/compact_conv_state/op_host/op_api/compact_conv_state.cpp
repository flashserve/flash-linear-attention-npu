/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "compact_conv_state.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(CompactConvState);

const aclTensor *CompactConvState(const aclTensor *convState, const aclTensor *baseAddrs, const aclTensor *blockStrides,
                                  const aclTensor *dimStrides, const aclTensor *tokenStrides,
                                  const aclTensor *stateIndices, const aclTensor *commitLens,
                                  const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices,
                                  const aclTensor *boundaryRecoveryLens, int64_t convHistoryLen, bool alignMode,
                                  int64_t dimStride, int64_t nullBlockId, aclOpExecutor *executor)
{
    L0_DFX(CompactConvState, convState, baseAddrs, blockStrides, dimStrides, tokenStrides, stateIndices, commitLens,
           finalStateIndices, boundaryStateIndices, boundaryRecoveryLens, convHistoryLen, alignMode, dimStride,
           nullBlockId);
    auto *mutableState = const_cast<aclTensor *>(convState);
    const auto storageShape = mutableState->GetStorageShape();
    const auto viewShape = mutableState->GetViewShape();
    const auto viewStrides = mutableState->GetViewStrides();
    const auto viewOffset = mutableState->GetViewOffset();
    auto status = INFER_SHAPE(CompactConvState,
                              OP_INPUT(convState, baseAddrs, blockStrides, dimStrides, tokenStrides, stateIndices,
                                       commitLens, finalStateIndices, boundaryStateIndices, boundaryRecoveryLens),
                              OP_OUTPUT(mutableState), OP_ATTR(convHistoryLen, alignMode, dimStride, nullBlockId));
    if (status != ACLNN_SUCCESS) {
        return nullptr;
    }
    mutableState->SetViewShape(viewShape);
    mutableState->SetViewStrides(viewStrides);
    mutableState->SetViewOffset(viewOffset);
    mutableState->SetStorageShape(storageShape);
    status = ADD_TO_LAUNCHER_LIST_AICORE(
        CompactConvState,
        OP_INPUT(convState, baseAddrs, blockStrides, dimStrides, tokenStrides, stateIndices, commitLens,
                 finalStateIndices, boundaryStateIndices, boundaryRecoveryLens),
        OP_OUTPUT(mutableState), OP_ATTR(convHistoryLen, alignMode, dimStride, nullBlockId));
    return status == ACLNN_SUCCESS ? mutableState : nullptr;
}
} // namespace l0op
