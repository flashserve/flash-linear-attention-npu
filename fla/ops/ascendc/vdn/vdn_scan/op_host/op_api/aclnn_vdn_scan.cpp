// SPDX-License-Identifier: BSD-3-Clause
#include "aclnn_vdn_scan.h"
#include "vdn_scan.h"
#include "../vdn_scan_tiling_processor.h"

#include <initializer_list>
#include <string>
#include "graph/utils/type_utils.h"
#include "op_common/log/log.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

namespace {
using namespace op;
using namespace VdnScanHost;
constexpr const char *kOpName = "aclnnVdnScan";

void ReportInvalid(const char *name, const std::string &actual, const std::string &expected)
{
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, name, actual, expected);
}

bool CheckFormatAndType(const aclTensor *tensor, const char *name)
{
    if (tensor->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, name,
            ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()), "expected FP32");
        return false;
    }
    if (IsPrivateFormat(tensor->GetStorageFormat()) || IsPrivateFormat(tensor->GetViewFormat()) ||
        !IsContiguous(tensor)) {
        ReportInvalid(name, "private format or strided view",
            "expected a contiguous tensor in a public ND-compatible format");
        return false;
    }
    return true;
}


struct StorageRange {
    uintptr_t first;
    uintptr_t last;

    bool Overlaps(const StorageRange &other) const
    {
        return first < other.last && other.first < last;
    }
};

// All inputs are contiguous FP32. Use view offsets to allow adjacent banks in one allocation.
bool GetRange(const aclTensor *tensor, const char *name, StorageRange &range)
{
    const auto base = reinterpret_cast<uintptr_t>(tensor->GetStorageAddr());
    const auto offset = tensor->GetViewOffset();
    const auto elements = tensor->GetViewShape().GetShapeSize();
    const auto capacity = tensor->GetStorageShape().GetShapeSize();
    const auto maxElements = (std::numeric_limits<uintptr_t>::max() - base) / sizeof(float);
    if (base == 0 || offset < 0 || elements <= 0 || capacity < elements || offset > capacity - elements ||
        static_cast<uint64_t>(offset) > maxElements ||
        static_cast<uint64_t>(elements) > maxElements - static_cast<uint64_t>(offset)) {
        ReportInvalid(name, "invalid storage interval", "the packed view must fit within its backing allocation");
        return false;
    }
    range.first = base + static_cast<uintptr_t>(offset) * sizeof(float);
    range.last = range.first + static_cast<uintptr_t>(elements) * sizeof(float);
    return true;
}

bool CheckBuffers(const aclTensor *transition, const aclTensor *prefix, const aclTensor *suffix)
{
    StorageRange transitionRange, prefixRange, suffixRange;
    if (!GetRange(transition, "transition", transitionRange) ||
        !GetRange(prefix, "prefix", prefixRange) ||
        !GetRange(suffix, "suffix", suffixRange)) {
        return false;
    }
    if (prefixRange.Overlaps(suffixRange) || transitionRange.Overlaps(prefixRange) ||
        transitionRange.Overlaps(suffixRange)) {
        ReportInvalid("transition, prefix, suffix", "overlapping views", "input storage intervals must be disjoint");
        return false;
    }
    return true;
}

bool HasShape(const aclTensor *tensor, const char *name, std::initializer_list<int64_t> dims)
{
    const auto &shape = tensor->GetViewShape();
    if (shape.GetDimNum() != dims.size()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, name, std::to_string(shape.GetDimNum()),
                                                "expected rank " + std::to_string(dims.size()));
        return false;
    }
    size_t axis = 0;
    for (auto expected : dims) {
        if (shape.GetDim(axis) != expected) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, name, std::to_string(shape.GetDim(axis)),
                "axis " + std::to_string(axis) + " must be " + std::to_string(expected));
            return false;
        }
        ++axis;
    }
    return true;
}

const aclTensor *AsNd(const aclTensor *tensor, aclOpExecutor *executor)
{
    auto *view = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                      tensor->GetViewStrides(), tensor->GetViewOffset());
    if (view != nullptr) {
        view->SetStorageFormat(Format::FORMAT_ND);
        view->SetViewFormat(Format::FORMAT_ND);
        view->SetOriginalFormat(Format::FORMAT_ND);
        view->SetOriginalShape(tensor->GetViewShape());
    }
    return view;
}

bool CheckShapes(const aclTensor *transition, const aclTensor *prefix, const aclTensor *suffix)
{
    const auto &shape = transition->GetViewShape();
    if (shape.GetDimNum() != 4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "transition", std::to_string(shape.GetDimNum()),
                                                "expected [frames, heads, 128, 128]");
        return false;
    }
    const int64_t frames = shape.GetDim(0), heads = shape.GetDim(1);
    if (!ValidDimensions(frames, heads)) {
        ReportInvalid("frames, heads",
            std::to_string(frames) + "," + std::to_string(heads),
            "frames must be in [1, INT32_MAX) and heads in [1, INT32_MAX/2)");
        return false;
    }
    return HasShape(transition, "transition", {frames, heads, kHeadDim, kHeadDim}) &&
           HasShape(prefix, "prefix", {frames + 1, heads, kHeadDim, kHeadDim}) &&
           HasShape(suffix, "suffix", {frames + 1, heads, kHeadDim, kHeadDim});
}
} // namespace

extern "C" aclnnStatus aclnnVdnScanGetWorkspaceSize(
    const aclTensor *transition, const aclTensor *prefix, const aclTensor *suffix, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (transition == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "transition");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (prefix == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "prefix");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (suffix == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "suffix");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckFormatAndType(transition, "transition") ||
        !CheckFormatAndType(prefix, "prefix") ||
        !CheckFormatAndType(suffix, "suffix") ||
        !CheckShapes(transition, prefix, suffix) || !CheckBuffers(transition, prefix, suffix)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto op = CREATE_EXECUTOR();
    if (op.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    const auto *transitionView = AsNd(transition, op.get());
    const auto *prefixView = AsNd(prefix, op.get());
    const auto *suffixView = AsNd(suffix, op.get());
    if (transitionView == nullptr || prefixView == nullptr || suffixView == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const auto status = l0op::VdnScan(transitionView, prefixView, suffixView, op.get());
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    *workspaceSize = op->GetWorkspaceSize();
    op.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnVdnScan(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
