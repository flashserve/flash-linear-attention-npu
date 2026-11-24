// SPDX-License-Identifier: BSD-3-Clause
#include "op_common/log/log.h"
#include "aclnn_vdn_bridge.h"
#include "vdn_bridge.h"
#include "../vdn_bridge_tiling_processor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include <string>

namespace {
using namespace op;
using namespace VdnBridgeHost;
using VdnBridgeKernel::HEAD_DIM;
constexpr const char *OP_NAME = "aclnnVdnBridge";

aclnnStatus Invalid(const char *name, const std::string &reason)
{
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, name, reason);
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus CheckTensor(const aclTensor *tensor, const char *name, op::DataType dtype, size_t rank)
{
    if (tensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (tensor->GetDataType() != dtype || tensor->GetViewShape().GetDimNum() != rank || !op::IsContiguous(tensor) ||
        op::IsPrivateFormat(tensor->GetStorageFormat()) || op::IsPrivateFormat(tensor->GetViewFormat())) {
        const auto actualRank = tensor->GetViewShape().GetDimNum();
        return Invalid(name, "expected a public, contiguous rank " + std::to_string(rank) + " with dtype " +
                             std::to_string(static_cast<int>(dtype)) + "; actual rank " + std::to_string(actualRank) +
                             ", dtype " + std::to_string(static_cast<int>(tensor->GetDataType())) +
                             ", contiguous " + std::to_string(op::IsContiguous(tensor)));
    }
    return ACLNN_SUCCESS;
}

struct AddressRange {
    uintptr_t begin;
    uintptr_t end;
};

// All views use dense public formats here. Separate views into the same allocation
// can be valid arguments, so compare their occupied bytes instead of storage identities.
bool GetRange(const aclTensor *tensor, AddressRange &range)
{
    const int64_t itemBytes = op::CalcShapeBytes(1, tensor->GetDataType());
    const int64_t offset = tensor->GetViewOffset();
    const int64_t elements = tensor->Numel(); // CheckArgs validated shape products first.
    if (itemBytes < 1 || offset < 0 || elements < 1) {
        return false;
    }
    const auto base = reinterpret_cast<uintptr_t>(tensor->GetStorageAddr());
    const auto limit = std::numeric_limits<uintptr_t>::max();
    if (base == 0 || static_cast<uint64_t>(offset) > (limit - base) / itemBytes) {
        return false;
    }
    const int64_t maxStorage = std::numeric_limits<int64_t>::max() / itemBytes;
    const auto &storageShape = tensor->GetStorageShape();
    int64_t storage = 1;
    for (size_t axis = 0; axis < storageShape.GetDimNum(); ++axis) {
        const int64_t dim = storageShape.GetDim(axis);
        if (dim <= 0 || dim > maxStorage / storage) {
            return false;
        }
        storage *= dim;
    }
    if (offset > storage || elements > storage - offset) {
        return false;
    }
    range.begin = base + static_cast<uintptr_t>(offset) * itemBytes;
    if (static_cast<uint64_t>(elements) > (limit - range.begin) / itemBytes) {
        return false;
    }
    range.end = range.begin + static_cast<uintptr_t>(elements) * itemBytes;
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

aclnnStatus CheckArgs(const aclTensor *prefix, const aclTensor *suffix, const aclTensor *bounds,
                      const aclTensor *logs, int64_t start, const aclTensor *out)
{
    if (out == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "out");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto outType = out->GetDataType();
    if (outType != DataType::DT_FLOAT && outType != DataType::DT_BF16) {
        return Invalid("out", "expected FP32 or BF16; actual dtype " + std::to_string(static_cast<int>(outType)));
    }
    struct Arg {
        const aclTensor *tensor;
        const char *name;
        DataType dtype;
        size_t rank;
    };
    const Arg args[] = {{prefix, "prefix", DataType::DT_FLOAT, 4}, {suffix, "suffix", DataType::DT_FLOAT, 4},
                        {bounds, "bounds", DataType::DT_INT64, 2}, {logs, "logs", DataType::DT_FLOAT, 3},
                        {out, "out", outType, 4}};
    for (const auto &arg : args) {
        const auto status = CheckTensor(arg.tensor, arg.name, arg.dtype, arg.rank);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    const auto &bank = prefix->GetViewShape();
    const int64_t bankRows = bank.GetDim(0);
    const int64_t heads = bank.GetDim(1);
    const int64_t count = out->GetViewShape().GetDim(0);
    if (!ValidTile(bankRows, heads, start, count)) {
        return Invalid("start/count", "expected a nonempty subrange of frames and representable offsets; bank rows " +
                       std::to_string(bankRows) + ", heads " + std::to_string(heads) + ", start " +
                       std::to_string(start) + ", count " + std::to_string(count));
    }
    if (!HasShape(bank, {bankRows, heads, HEAD_DIM, HEAD_DIM})) {
        return Invalid("prefix", "expected [F+1,H,128,128], got matrix axes " + std::to_string(bank.GetDim(2)) +
                       " and " + std::to_string(bank.GetDim(3)));
    }
    if (!HasShape(suffix->GetViewShape(), {bankRows, heads, HEAD_DIM, HEAD_DIM})) {
        return Invalid("suffix", "expected same shape as prefix");
    }
    if (!HasShape(bounds->GetViewShape(), {bankRows - 1, 2})) {
        return Invalid("bounds", "expected [F,2] where F=" + std::to_string(bankRows - 1));
    }
    if (!HasShape(logs->GetViewShape(), {bankRows, heads, HEAD_DIM})) {
        return Invalid("logs", "expected [F+1,H,128] matching prefix");
    }
    if (!HasShape(out->GetViewShape(), {count, heads, HEAD_DIM, HEAD_DIM})) {
        return Invalid("out", "expected [count,H,128,128] matching prefix");
    }
    AddressRange write;
    if (!GetRange(out, write)) {
        return Invalid("out", "view must address a nonempty representable range in its storage");
    }
    for (size_t index = 0; index < 4; ++index) {
        const auto &arg = args[index];
        AddressRange read;
        if (!GetRange(arg.tensor, read)) {
            return Invalid(arg.name, "view must address a nonempty representable range in its storage");
        }
        if (read.begin < write.end && write.begin < read.end) {
            return Invalid("out", std::string("output overlaps input ") + arg.name);
        }
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" aclnnStatus aclnnVdnBridgeGetWorkspaceSize(const aclTensor *prefix, const aclTensor *suffix,
                                                      const aclTensor *bounds, const aclTensor *logs, int64_t start,
                                                      const aclTensor *out, uint64_t *workspaceSize,
                                                      aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, workspaceSize == nullptr ? "workspaceSize" : "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(aclnnVdnBridge, DFX_IN(prefix, suffix, bounds, logs, start), DFX_OUT(out));
    const auto status = CheckArgs(prefix, suffix, bounds, logs, start, out);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *plan = uniqueExecutor.get();
    const auto *prefixView = AsNd(prefix, plan);
    const auto *suffixView = AsNd(suffix, plan);
    const auto *boundsView = AsNd(bounds, plan);
    const auto *logsView = AsNd(logs, plan);
    const auto *outView = AsNd(out, plan);
    if (prefixView == nullptr || suffixView == nullptr || boundsView == nullptr || logsView == nullptr ||
        outView == nullptr || l0op::VdnBridge(prefixView, suffixView, boundsView, logsView, start, outView, plan) == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnVdnBridge(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnVdnBridge);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
