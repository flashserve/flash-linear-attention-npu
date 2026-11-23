/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "compact_conv_state_inner.h"
#include "compact_conv_state.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "op_common/log/log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr const char *OP_NAME = "CompactConvStateInner";

aclnnStatus CheckCompactNull(const aclTensor *convState, const aclTensor *baseAddrs, const aclTensor *blockStrides,
                             const aclTensor *dimStrides, const aclTensor *tokenStrides, const aclTensor *stateIndices,
                             const aclTensor *commitLens, const aclTensor *finalStateIndices,
                             const aclTensor *boundaryStateIndices, const aclTensor *boundaryRecoveryLens,
                             aclTensor *convStateOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (convState == nullptr || baseAddrs == nullptr || blockStrides == nullptr || dimStrides == nullptr ||
        tokenStrides == nullptr || stateIndices == nullptr || commitLens == nullptr || finalStateIndices == nullptr ||
        boundaryStateIndices == nullptr || boundaryRecoveryLens == nullptr || convStateOut == nullptr ||
        workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "required argument", "nullptr");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" __attribute__((visibility("default"))) aclnnStatus CompactConvStateInnerGetWorkspaceSize(
    const aclTensor *convState, const aclTensor *baseAddrs, const aclTensor *blockStrides, const aclTensor *dimStrides,
    const aclTensor *tokenStrides, const aclTensor *stateIndices, const aclTensor *commitLens,
    const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices, const aclTensor *boundaryRecoveryLens,
    int64_t convHistoryLen, bool alignMode, int64_t dimStride, int64_t nullBlockId, aclTensor *convStateOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const auto nullStatus = CheckCompactNull(convState, baseAddrs, blockStrides, dimStrides, tokenStrides, stateIndices,
                                             commitLens, finalStateIndices, boundaryStateIndices, boundaryRecoveryLens,
                                             convStateOut, workspaceSize, executor);
    if (nullStatus != ACLNN_SUCCESS) {
        return nullStatus;
    }
    if (convState->GetDataType() != DataType::DT_BF16 && convState->GetDataType() != DataType::DT_FLOAT) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (convStateOut->GetDataType() != convState->GetDataType()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    L2_DFX_PHASE_1(CompactConvStateInner,
                   DFX_IN(convState, baseAddrs, blockStrides, dimStrides, tokenStrides, stateIndices, commitLens,
                          finalStateIndices, boundaryStateIndices, boundaryRecoveryLens, convHistoryLen, alignMode,
                          dimStride, nullBlockId),
                   DFX_OUT(convStateOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    const auto view = [&](const aclTensor *tensor) -> const aclTensor * {
        tensor->SetOriginalShape(tensor->GetViewShape());
        return uniqueExecutor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                          tensor->GetViewStrides(), tensor->GetViewOffset());
    };
    const auto stateView = view(convState);
    const auto baseView = view(baseAddrs);
    const auto blockView = view(blockStrides);
    const auto dimView = view(dimStrides);
    const auto tokenView = view(tokenStrides);
    const auto sourceView = view(stateIndices);
    const auto lengthsView = view(commitLens);
    const auto finalView = view(finalStateIndices);
    const auto boundaryView = view(boundaryStateIndices);
    const auto recoveryView = view(boundaryRecoveryLens);
    auto stateOutView =
        uniqueExecutor->CreateView(convStateOut, convStateOut->GetViewShape(), convStateOut->GetStorageShape(),
                                   convStateOut->GetViewStrides(), convStateOut->GetViewOffset());
    if (!stateView || !baseView || !blockView || !dimView || !tokenView || !sourceView || !lengthsView || !finalView ||
        !boundaryView || !recoveryView || !stateOutView) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const auto result = l0op::CompactConvState(stateView, baseView, blockView, dimView, tokenView, sourceView,
                                               lengthsView, finalView, boundaryView, recoveryView, convHistoryLen,
                                               alignMode, dimStride, nullBlockId, uniqueExecutor.get());
    if (result == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) aclnnStatus CompactConvStateInner(void *workspace,
                                                                                    uint64_t workspaceSize,
                                                                                    aclOpExecutor *executor,
                                                                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(CompactConvStateInner);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
