#ifndef ACLNN_CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_OUTPUT_ARCH22_H
#define ACLNN_CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_OUTPUT_ARCH22_H

#include <array>
#include <cstdint>
#include <limits>

#include "opdev/data_type_utils.h"
#include "opdev/framework_op.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "op_common/log/log.h"

namespace GDN::FinalizeHostArch22 {

inline bool TensorViewsMayOverlap(const aclTensor *left, const aclTensor *right)
{
    if (left == nullptr || right == nullptr || left->Numel() == 0 || right->Numel() == 0) {
        return false;
    }
    if (left == right) {
        return true;
    }
    const bool leftUnbound = left->GetStorageAddr() == nullptr;
    const bool rightUnbound = right->GetStorageAddr() == nullptr;
    if (leftUnbound || rightUnbound) {
        // executor 统一规划尚未分配的内部存储；同一 storage 的 view 仍按重叠处理。
        if ((leftUnbound && !left->IsFromWorkspace()) || (rightUnbound && !right->IsFromWorkspace())) {
            return true;
        }
        return left->GetStorage() == right->GetStorage();
    }
    // 独立 Finalize 和完整 Bwd 共用此 ACLNN 校验，非连续 view 按可能重叠处理。
    if (!op::IsContiguous(left) || !op::IsContiguous(right) || left->Numel() < 0 || right->Numel() < 0) {
        return true;
    }
    const int64_t leftElementBytes = op::CalcShapeBytes(1, left->GetDataType());
    const int64_t rightElementBytes = op::CalcShapeBytes(1, right->GetDataType());
    const int64_t sizeLimit = std::numeric_limits<int64_t>::max();
    if (leftElementBytes <= 0 || rightElementBytes <= 0 ||
        left->Numel() > sizeLimit / leftElementBytes || right->Numel() > sizeLimit / rightElementBytes) {
        return true;
    }
    const int64_t leftBytes = op::CalcShapeBytes(left->Numel(), left->GetDataType());
    const int64_t rightBytes = op::CalcShapeBytes(right->Numel(), right->GetDataType());
    const uintptr_t leftBegin = reinterpret_cast<uintptr_t>(left->GetData());
    const uintptr_t rightBegin = reinterpret_cast<uintptr_t>(right->GetData());
    const uintptr_t addressLimit = std::numeric_limits<uintptr_t>::max();
    if (leftBytes <= 0 || rightBytes <= 0 || leftBegin == 0 || rightBegin == 0 ||
        static_cast<uint64_t>(leftBytes) > addressLimit - leftBegin ||
        static_cast<uint64_t>(rightBytes) > addressLimit - rightBegin) {
        return true;
    }
    return leftBegin < rightBegin + static_cast<uintptr_t>(rightBytes) &&
           rightBegin < leftBegin + static_cast<uintptr_t>(leftBytes);
}

inline bool OutputNeedsTemporary(size_t outputIndex, const std::array<const aclTensor *, 5> &outputs,
                          const std::array<const aclTensor *, 16> &inputs)
{
    for (const aclTensor *input : inputs) {
        if (TensorViewsMayOverlap(outputs[outputIndex], input)) {
            return true;
        }
    }
    for (size_t other = 0; other < outputs.size(); ++other) {
        if (other != outputIndex && TensorViewsMayOverlap(outputs[outputIndex], outputs[other])) {
            return true;
        }
    }
    return false;
}

inline uint64_t GetOutputAliasMask(const std::array<const aclTensor *, 16> &inputs,
                                   const std::array<const aclTensor *, 5> &outputs)
{
    uint64_t mask = 0;
    for (size_t index = 0; index < outputs.size(); ++index) {
        if (OutputNeedsTemporary(index, outputs, inputs)) {
            mask |= uint64_t{1} << index;
        }
    }
    return mask;
}

inline aclnnStatus PrepareOutputs(const std::array<const aclTensor *, 5> &outputs,
                                  uint64_t outputAliasMask, aclOpExecutor *executor,
                                  std::array<const aclTensor *, 5> &kernelOutputs)
{
    kernelOutputs = outputs;
    for (size_t index = 0; index < outputs.size(); ++index) {
        if ((outputAliasMask & (uint64_t{1} << index)) != 0) {
            kernelOutputs[index] = executor->AllocTensor(
                outputs[index]->GetViewShape(), outputs[index]->GetDataType(), op::Format::FORMAT_ND);
            if (kernelOutputs[index] == nullptr) {
                OP_LOGE_WITHOUT_REPORT("ChunkGatedDeltaRuleBwdFinalize",
                                      "Failed to allocate a temporary Finalize output[%zu].", index);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
    }
    return ACLNN_SUCCESS;
}

inline aclnnStatus CopyOutputs(const std::array<const aclTensor *, 5> &kernelOutputs,
                               const std::array<const aclTensor *, 5> &outputs, aclOpExecutor *executor)
{
    for (size_t index = 0; index < outputs.size(); ++index) {
        if (kernelOutputs[index] != outputs[index] &&
            op::CopyNpuToNpu(kernelOutputs[index], outputs[index], executor) != ACLNN_SUCCESS) {
            // 显式 copy 保留临时存储，连续 ViewCopy 会把 kernel 输出重新绑定到原输入。
            OP_LOGE_WITHOUT_REPORT("ChunkGatedDeltaRuleBwdFinalize",
                                  "Failed to copy a temporary Finalize output[%zu].", index);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

} // namespace GDN::FinalizeHostArch22

#endif
