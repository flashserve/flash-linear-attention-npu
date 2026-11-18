/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "gdn_recoverssm_verify_inner.h"
#include "gdn_recoverssm_verify.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "op_common/log/log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr const char *OP_NAME = "GdnRecoverssmVerifyInner";

aclnnStatus CheckNull(const aclTensor *A_log, const aclTensor *a, const aclTensor *b, const aclTensor *dt_bias,
                      const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *checkpoint,
                      aclTensor *replay, const aclTensor *starts, const aclTensor *indices, aclTensor *out,
                      uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (A_log == nullptr || a == nullptr || b == nullptr || dt_bias == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || checkpoint == nullptr || replay == nullptr || starts == nullptr || indices == nullptr ||
        out == nullptr || workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "required argument", "nullptr");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const aclTensor *tensor, DataType expected, const char *name)
{
    if (tensor->GetDataType() != expected) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, name,
            "expected=" + std::string(ToString(expected).GetString()) +
                ", actual=" + std::string(ToString(tensor->GetDataType()).GetString()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" __attribute__((visibility("default"))) aclnnStatus GdnRecoverssmVerifyInnerGetWorkspaceSize(
    const aclTensor *A_log, const aclTensor *a, const aclTensor *b, const aclTensor *dt_bias, const aclTensor *q,
    const aclTensor *k, const aclTensor *v, const aclTensor *checkpoint, aclTensor *replay, const aclTensor *starts,
    const aclTensor *indices, float scale, float softplusBeta, float softplusThreshold, bool useQkL2normInKernel,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const auto nullStatus =
        CheckNull(A_log, a, b, dt_bias, q, k, v, checkpoint, replay, starts, indices, out, workspaceSize, executor);
    if (nullStatus != ACLNN_SUCCESS) {
        return nullStatus;
    }
    const std::pair<const aclTensor *, DataType> dtypes[] = {
        {A_log, DataType::DT_FLOAT},   {a, DataType::DT_BF16},       {b, DataType::DT_BF16},
        {dt_bias, DataType::DT_FLOAT}, {q, DataType::DT_BF16},       {k, DataType::DT_BF16},
        {v, DataType::DT_BF16},        {replay, DataType::DT_FLOAT}, {out, DataType::DT_BF16}};
    for (const auto &item : dtypes) {
        const auto status = CheckDtype(item.first, item.second, "tensor dtype");
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    if (checkpoint->GetDataType() != DataType::DT_BF16 && checkpoint->GetDataType() != DataType::DT_FLOAT) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (starts->GetDataType() != DataType::DT_INT32 && starts->GetDataType() != DataType::DT_INT64) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    L2_DFX_PHASE_1(GdnRecoverssmVerifyInner,
                   DFX_IN(A_log, a, b, dt_bias, q, k, v, checkpoint, replay, starts, indices, scale, softplusBeta,
                          softplusThreshold, useQkL2normInKernel),
                   DFX_OUT(replay, out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }

    const auto view = [&](const aclTensor *tensor) -> const aclTensor * {
        tensor->SetOriginalShape(tensor->GetViewShape());
        return uniqueExecutor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                          tensor->GetViewStrides(), tensor->GetViewOffset());
    };
    const auto A_logView = view(A_log);
    const auto aView = view(a);
    const auto bView = view(b);
    const auto dtBiasView = view(dt_bias);
    const auto qView = view(q);
    const auto kView = view(k);
    const auto vView = view(v);
    const auto checkpointView = view(checkpoint);
    auto replayView = uniqueExecutor->CreateView(replay, replay->GetViewShape(), replay->GetStorageShape(),
                                                 replay->GetViewStrides(), replay->GetViewOffset());
    const auto startsView = view(starts);
    const auto indicesView = view(indices);
    const auto outView = view(out);
    if (!A_logView || !aView || !bView || !dtBiasView || !qView || !kView || !vView || !checkpointView || !replayView ||
        !startsView || !indicesView || !outView) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const auto result = l0op::GdnRecoverssmVerify(
        A_logView, aView, bView, dtBiasView, qView, kView, vView, checkpointView, replayView, startsView, indicesView,
        scale, softplusBeta, softplusThreshold, useQkL2normInKernel, uniqueExecutor.get());
    if (result == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (l0op::ViewCopy(result, outView, uniqueExecutor.get()) == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) aclnnStatus GdnRecoverssmVerifyInner(void *workspace,
                                                                                       uint64_t workspaceSize,
                                                                                       aclOpExecutor *executor,
                                                                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(GdnRecoverssmVerifyInner);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
