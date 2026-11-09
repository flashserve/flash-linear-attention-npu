/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "aclnn_merge_fwd_bwd_kernel.h"
#include "merge_fwd_bwd_kernel.h"

#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <string>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include "../../../../common/host_utils/log.h"

using namespace op;

namespace {

constexpr int64_t kK = 128;
constexpr int64_t kV = 128;
constexpr int64_t kVK = kV + kK;
constexpr int64_t kSMax = 1024;
constexpr int64_t kHvMax = 256;

constexpr const char *MERGE_FWD_BWD_ENTITY = "aclnnMergeFwdBwdKernel";

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

std::string DfxTensorShapeString(const aclTensor *tensor)
{
    const auto &shape = tensor->GetViewShape();
    std::ostringstream stream;
    stream << "[";
    for (size_t idx = 0; idx < shape.GetDimNum(); ++idx) {
        if (idx != 0) {
            stream << ",";
        }
        stream << shape.GetDim(idx);
    }
    stream << "]";
    return stream.str();
}

aclnnStatus CheckPublicFormat(const aclTensor *tensor, const char *name)
{
    const auto storageFormat = tensor->GetStorageFormat();
    const auto viewFormat = tensor->GetViewFormat();
    // ND/NCHW/NCL/NHWC name the same packed layout. NZ and other private formats do not.
    if (IsPrivateFormat(storageFormat) || IsPrivateFormat(viewFormat)) {
        OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(
            MERGE_FWD_BWD_ENTITY, name,
            "storage=" + std::to_string(static_cast<int>(storageFormat)) +
                ",view=" + std::to_string(static_cast<int>(viewFormat)),
            "non-private format");
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
    if (!IsContiguous(h)) {
        OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(MERGE_FWD_BWD_ENTITY, "h", "non-contiguous", "contiguous");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const DataType dtype = agHm->GetDataType();
    if (dtype != DataType::DT_FLOAT && dtype != DataType::DT_BF16) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(MERGE_FWD_BWD_ENTITY, "agHm",
                                                   fla_npu::host_utils::DfxDtypeString(dtype), "FLOAT,BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (h->GetDataType() != dtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(MERGE_FWD_BWD_ENTITY, "h,agHm",
                                                    "h=" + fla_npu::host_utils::DfxDtypeString(h->GetDataType()) +
                                                        ",agHm=" + fla_npu::host_utils::DfxDtypeString(dtype),
                                                    "both same");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const Shape agView = agHm->GetViewShape();
    if (agView.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(
            MERGE_FWD_BWD_ENTITY, "agHm", "rank=" + std::to_string(agView.GetDimNum()),
            "rank must be 4 [S,HV,128,256]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t s = agView.GetDim(0);
    const int64_t hv = agView.GetDim(1);
    const int64_t k = agView.GetDim(2);
    const int64_t vk = agView.GetDim(3);
    if (s < 1 || s > kSMax || hv < 1 || hv > kHvMax) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(
            MERGE_FWD_BWD_ENTITY, "agHm.shape[S,HV]", "S=" + std::to_string(s) + ",HV=" + std::to_string(hv),
            "S in [1,1024], HV in [1,256]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (k != kK || vk != kVK) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(
            MERGE_FWD_BWD_ENTITY, "agHm.shape[K],agHm.shape[V+K]",
            "K=" + std::to_string(k) + ",V+K=" + std::to_string(vk), "K=128, V+K=256");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Tiling reads the storage shape. Both must be the dense [HV,128,128] buffer.
    if (!MatchesShape(h->GetViewShape(), {hv, kK, kV}) || !MatchesShape(h->GetStorageShape(), {hv, kK, kV})) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(
            MERGE_FWD_BWD_ENTITY, "h", DfxTensorShapeString(h),
            "[" + std::to_string(hv) + ",128,128]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (preOrPostNumRanks < 1 || preOrPostNumRanks > s) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(
            MERGE_FWD_BWD_ENTITY, "preOrPostNumRanks", std::to_string(preOrPostNumRanks),
            "in [1, S=" + std::to_string(s) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (rank < 0 || rank >= s) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(MERGE_FWD_BWD_ENTITY, "rank", std::to_string(rank),
                                                   "in [0, S=" + std::to_string(s) + ")");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool unitWorld = (s == 1 && preOrPostNumRanks == 1 && rank == 0);
    if (!unitWorld) {
        if (forward) {
            if (rank < preOrPostNumRanks) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(
                    MERGE_FWD_BWD_ENTITY, "rank", std::to_string(rank),
                    ">= N=" + std::to_string(preOrPostNumRanks) + " for forward merge");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (rank + preOrPostNumRanks >= s) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(
                    MERGE_FWD_BWD_ENTITY, "rank", std::to_string(rank),
                    "rank + N < S=" + std::to_string(s) + " for backward merge");
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
    if (agHm == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(MERGE_FWD_BWD_ENTITY, "agHm");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (h == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(MERGE_FWD_BWD_ENTITY, "h");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(MERGE_FWD_BWD_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(MERGE_FWD_BWD_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
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
