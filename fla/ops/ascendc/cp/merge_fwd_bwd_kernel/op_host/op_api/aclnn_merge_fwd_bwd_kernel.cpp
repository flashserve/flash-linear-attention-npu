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
    CHECK_COND(!IsPrivateFormat(storageFormat) && !IsPrivateFormat(viewFormat), ACLNN_ERR_PARAM_INVALID,
               "%s must use a non-private format, but got storage=%d and view=%d.", name,
               static_cast<int>(storageFormat), static_cast<int>(viewFormat));
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
    CHECK_COND(IsContiguous(h), ACLNN_ERR_PARAM_INVALID, "h is written in place and must be contiguous.");

    const DataType dtype = agHm->GetDataType();
    CHECK_COND(dtype == DataType::DT_FLOAT || dtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
               "agHm dtype must be FP32 or BF16.");
    CHECK_COND(h->GetDataType() == dtype, ACLNN_ERR_PARAM_INVALID, "h dtype must match agHm.");

    const Shape agView = agHm->GetViewShape();
    CHECK_COND(agView.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "agHm must be rank-4 [S,HV,128,256].");
    const int64_t s = agView.GetDim(0);
    const int64_t hv = agView.GetDim(1);
    const int64_t k = agView.GetDim(2);
    const int64_t vk = agView.GetDim(3);
    CHECK_COND(s >= 1 && s <= kSMax && hv >= 1 && hv <= kHvMax, ACLNN_ERR_PARAM_INVALID,
               "S must be in [1,1024] and HV must be in [1,256].");
    CHECK_COND(k == kK && vk == kVK, ACLNN_ERR_PARAM_INVALID, "K must be 128 and the last dim must be 256.");
    // Tiling reads the storage shape. Both must be the dense [HV,128,128] buffer.
    CHECK_COND(MatchesShape(h->GetViewShape(), {hv, kK, kV}) && MatchesShape(h->GetStorageShape(), {hv, kK, kV}),
               ACLNN_ERR_PARAM_INVALID, "h must be [HV,128,128] for both state layouts.");

    CHECK_COND(preOrPostNumRanks >= 1 && preOrPostNumRanks <= s, ACLNN_ERR_PARAM_INVALID,
               "preOrPostNumRanks must be in [1, S].");
    CHECK_COND(rank >= 0 && rank < s, ACLNN_ERR_PARAM_INVALID, "rank must be in [0, S).");
    const bool unitWorld = (s == 1 && preOrPostNumRanks == 1 && rank == 0);
    if (!unitWorld) {
        if (forward) {
            CHECK_COND(rank >= preOrPostNumRanks, ACLNN_ERR_PARAM_INVALID, "forward merge requires rank >= N.");
        } else {
            CHECK_COND(rank + preOrPostNumRanks < s, ACLNN_ERR_PARAM_INVALID,
                       "backward merge requires rank + N < S.");
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
    CHECK_COND(agHm != nullptr && h != nullptr, ACLNN_ERR_PARAM_NULLPTR, "agHm and h must not be nullptr.");
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    const aclnnStatus paramStatus = CheckMergeFwdBwdKernelParams(h, agHm, preOrPostNumRanks, rank, forward);
    if (paramStatus != ACLNN_SUCCESS) {
        return paramStatus;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    // ag_hm is read-only, so a contiguous copy is safe. h is not copied.
    const aclTensor *agContig = l0op::Contiguous(agHm, executorPtr);
    CHECK_RET(agContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outputs = l0op::MergeFwdBwdKernel(
        h, agContig, preOrPostNumRanks, rank, forward, stateVFirst, executorPtr);
    CHECK_COND(outputs[0] != nullptr, ACLNN_ERR_INNER_NULLPTR, "MergeFwdBwdKernel launch failed.");

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
