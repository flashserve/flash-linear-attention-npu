/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "aclnn_merge_fwd_bwd_kernel.h"
#include "merge_fwd_bwd_kernel.h"

#include <cstddef>
#include <initializer_list>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "op_common/log/log.h"
#include <string>

static constexpr const char *OP_NAME = "aclnnMergeFwdBwdKernel";

using namespace op;

namespace {

constexpr int64_t kK = 128;
constexpr int64_t kV = 128;
constexpr int64_t kVK = kV + kK;
constexpr int64_t kSMax = 1024;
constexpr int64_t kHvMax = 256;

bool MatchesShape(const Shape &shape, std::initializer_list<int64_t> expect)
{
    if (shape.GetDimNum() != expect.size()) {
        return false;
    }
    size_t axis = 0;
    for (const int64_t dim : expect) {
        if (shape.GetDim(axis) != dim) {
            return false;
        }
        ++axis;
    }
    return true;
}

aclnnStatus CheckPublicFormat(const aclTensor *tensor, const char *name)
{
    const auto storageFormat = tensor->GetStorageFormat();
    const auto viewFormat = tensor->GetViewFormat();
    // ND/NCHW/NCL/NHWC name the same packed layout. NZ and other private formats do not.
    if (!(!IsPrivateFormat(storageFormat) && !IsPrivateFormat(viewFormat))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "format", "private",
            "storage and view format must be public (ND/NCHW/NCL/NHWC), private formats such as NZ are not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMergeFwdBwdKernelParams(
    const aclTensor *h, const aclTensor *agHm, int64_t preOrPostNumRanks, int64_t rank, bool forward)
{
    const aclnnStatus hFormat = CheckPublicFormat(h, "h");
    if (hFormat != ACLNN_SUCCESS) {
        return hFormat;
    }
    const aclnnStatus agFormat = CheckPublicFormat(agHm, "agHm");
    if (agFormat != ACLNN_SUCCESS) {
        return agFormat;
    }
    // h is inplace. A contiguous copy would be written and then dropped.
    if (!(IsContiguous(h))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "and", std::to_string(IsContiguous(h)), "h is written in place and must be contiguous.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const DataType dtype = agHm->GetDataType();
    if (!(dtype == DataType::DT_FLOAT || dtype == DataType::DT_BF16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "dtype", "invalid", "agHm dtype must be FP32 or BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(h->GetDataType() == dtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "dtype", "invalid", "h dtype must match agHm.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const Shape agView = agHm->GetViewShape();
    if (!(agView.GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "HV", std::to_string(agView.GetDimNum()), "agHm must be rank-4 [S,HV,128,256].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t s = agView.GetDim(0);
    const int64_t hv = agView.GetDim(1);
    const int64_t k = agView.GetDim(2);
    const int64_t vk = agView.GetDim(3);
    if (!(s >= 1 && s <= kSMax && hv >= 1 && hv <= kHvMax)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "HV", "invalid", "S must be in [1,1024] and HV must be in [1,256].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(k == kK && vk == kVK)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "K", "invalid", "K must be 128 and the last dim must be 256.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Tiling reads the storage shape. Both must be the dense [HV,128,128] buffer.
    if (!(MatchesShape(h->GetViewShape(), {hv, kK, kV}) && MatchesShape(h->GetStorageShape(), {hv, kK, kV}))) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(OP_NAME, "HV", "invalid", "h must be [HV,128,128] for both state layouts.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!(preOrPostNumRanks >= 1 && preOrPostNumRanks <= s)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "preOrPostNumRanks", "invalid", "preOrPostNumRanks must be in [1, S].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(rank >= 0 && rank < s)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "rank", "invalid", "rank must be in [0, S).");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool unitWorld = (s == 1 && preOrPostNumRanks == 1 && rank == 0);
    if (!unitWorld) {
        if (forward) {
            if (!(rank >= preOrPostNumRanks)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "rank", "invalid", "forward merge requires rank >= N.");
        return ACLNN_ERR_PARAM_INVALID;
    }
        } else {
            if (!(rank + preOrPostNumRanks < s)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "rank", "invalid", "backward merge requires rank + N < S.");
        return ACLNN_ERR_PARAM_INVALID;
    }
        }
    }
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMergeFwdBwdKernelGetWorkspaceSize(
    const aclTensor *h, const aclTensor *agHm, int64_t preOrPostNumRanks, int64_t rank, bool forward,
    bool stateVFirst, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (!(agHm != nullptr && h != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "agHm/h");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(workspaceSize != nullptr && executor != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const aclnnStatus paramStatus = CheckMergeFwdBwdKernelParams(h, agHm, preOrPostNumRanks, rank, forward);
    if (paramStatus != ACLNN_SUCCESS) {
        return paramStatus;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();

    // ag_hm is read-only, so a contiguous copy is safe. h is not copied.
    const aclTensor *agContig = l0op::Contiguous(agHm, executorPtr);
    if (!(agContig != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: agContig != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto outputs = l0op::MergeFwdBwdKernel(
        h, agContig, preOrPostNumRanks, rank, forward, stateVFirst, executorPtr);
    if (!(outputs[0] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: MergeFwdBwdKernel launch failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMergeFwdBwdKernel(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
