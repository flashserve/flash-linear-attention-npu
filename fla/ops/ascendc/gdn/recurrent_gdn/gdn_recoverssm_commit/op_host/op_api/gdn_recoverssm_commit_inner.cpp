/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "gdn_recoverssm_commit_inner.h"
#include "gdn_recoverssm_commit.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "op_common/log/log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr const char *OP_NAME = "GdnRecoverssmCommitInner";

aclnnStatus CheckNull(
    const aclTensor *state, const aclTensor *replay, const aclTensor *table,
    const aclTensor *stateIndices, const aclTensor *commitLens,
    const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices,
    const aclTensor *boundaryRecoveryLens, aclTensor *stateOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (state == nullptr || replay == nullptr || table == nullptr ||
        stateIndices == nullptr || commitLens == nullptr ||
        finalStateIndices == nullptr || boundaryStateIndices == nullptr ||
        boundaryRecoveryLens == nullptr || stateOut == nullptr ||
        workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "required argument",
                                                 "nullptr");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const aclTensor *tensor, DataType expected,
                       const char *name)
{
    if (tensor->GetDataType() != expected) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, name,
            "expected=" + std::string(ToString(expected).GetString()) +
                ", actual=" +
                std::string(ToString(tensor->GetDataType()).GetString()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" __attribute__((visibility("default"))) aclnnStatus
GdnRecoverssmCommitInnerGetWorkspaceSize(
    const aclTensor *state, const aclTensor *replay, const aclTensor *table,
    const aclTensor *stateIndices, const aclTensor *commitLens,
    const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices,
    const aclTensor *boundaryRecoveryLens, int64_t specQueryLen, bool alignMode,
    aclTensor *stateOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const auto nullStatus =
        CheckNull(state, replay, table, stateIndices, commitLens,
                  finalStateIndices, boundaryStateIndices,
                  boundaryRecoveryLens, stateOut, workspaceSize, executor);
    if (nullStatus != ACLNN_SUCCESS) {
        return nullStatus;
    }
    if (CheckDtype(replay, DataType::DT_FLOAT, "replay") != ACLNN_SUCCESS ||
        CheckDtype(table, DataType::DT_INT64, "table") != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto stateType = state->GetDataType();
    if (stateType != DataType::DT_BF16 && stateType != DataType::DT_FLOAT) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (stateOut->GetDataType() != stateType) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    L2_DFX_PHASE_1(
        GdnRecoverssmCommitInner,
        DFX_IN(state, replay, table, stateIndices, commitLens,
               finalStateIndices, boundaryStateIndices, boundaryRecoveryLens,
               specQueryLen, alignMode),
        DFX_OUT(stateOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    const auto view = [&](const aclTensor *tensor) -> const aclTensor * {
        tensor->SetOriginalShape(tensor->GetViewShape());
        return uniqueExecutor->CreateView(
            tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
            tensor->GetViewStrides(), tensor->GetViewOffset());
    };
    const auto stateView = view(state);
    const auto replayView = view(replay);
    const auto tableView = view(table);
    const auto sourceView = view(stateIndices);
    const auto lengthsView = view(commitLens);
    const auto finalView = view(finalStateIndices);
    const auto boundaryView = view(boundaryStateIndices);
    const auto recoveryView = view(boundaryRecoveryLens);
    auto stateOutView = uniqueExecutor->CreateView(
        stateOut, stateOut->GetViewShape(), stateOut->GetStorageShape(),
        stateOut->GetViewStrides(), stateOut->GetViewOffset());
    if (!stateView || !replayView || !tableView || !sourceView ||
        !lengthsView || !finalView || !boundaryView || !recoveryView ||
        !stateOutView) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const auto result = l0op::GdnRecoverssmCommit(
        stateView, replayView, tableView, sourceView, lengthsView, finalView,
        boundaryView, recoveryView, specQueryLen, alignMode,
        uniqueExecutor.get());
    if (result == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) aclnnStatus
GdnRecoverssmCommitInner(void *workspace, uint64_t workspaceSize,
                         aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(GdnRecoverssmCommitInner);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
