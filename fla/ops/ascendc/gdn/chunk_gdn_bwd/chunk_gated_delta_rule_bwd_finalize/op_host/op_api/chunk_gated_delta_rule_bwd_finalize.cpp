#include "chunk_gated_delta_rule_bwd_finalize.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/contiguous.h"
#include <cstdint>
#include <limits>

#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/data_type_utils.h"
#include "opdev/framework_op.h"
#include "op_common/log/log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleBwdFinalize);

namespace {

constexpr const char *OP_NAME = "ChunkGatedDeltaRuleBwdFinalize";

bool TensorViewsMayOverlap(const aclTensor *left, const aclTensor *right)
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
    // L0 也被完整 Bwd 直接调用，非连续视图不能按 Numel 推断实际地址跨度。
    if (!IsContiguous(left) || !IsContiguous(right) || left->Numel() < 0 || right->Numel() < 0) {
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

bool OutputNeedsTemporary(size_t outputIndex, const std::array<const aclTensor *, 5> &outputs,
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

const aclTensor *ConvertIntArray(const aclIntArray *array, aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}

} // namespace

uint64_t GetFinalizeOutputAliasMask(const std::array<const aclTensor *, 16> &inputs,
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

const std::array<const aclTensor *, 5> ChunkGatedDeltaRuleBwdFinalize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *dO, const aclTensor *du,
    const aclTensor *g, const aclTensor *beta, const aclTensor *h,
    const aclTensor *dh, const aclTensor *a,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclTensor *betaRawOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool useQkL2NormInKernel,
    bool useBetaSigmoidInKernel, bool useGateInKernel, bool stateVFirst,
    bool useExp2,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut,
    aclOpExecutor *executor, uint64_t outputTemporaryMask)
{
    L0_DFX(ChunkGatedDeltaRuleBwdFinalize, q, k, v, vNew, dO, du, g, beta, h, dh, a,
           qRstdOptional, kRstdOptional, betaRawOptional,
           cuSeqlensOptional, chunkIndicesOptional,
           scale, chunkSize, useQkL2NormInKernel,
           useBetaSigmoidInKernel, useGateInKernel, stateVFirst, useExp2,
           dqOut, dkOut, dvOut, dbetaOut, dgOut, outputTemporaryMask);

    const aclTensor *cuSeqlens = ConvertIntArray(cuSeqlensOptional, executor);
    const aclTensor *chunkIndices = ConvertIntArray(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && chunkIndices == nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "Convert varlen index array failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    const std::array<const aclTensor *, 5> outputs = {dqOut, dkOut, dvOut, dbetaOut, dgOut};
    auto kernelOutputs = outputs;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        const std::array<const aclTensor *, 16> inputs = {
            q, k, v, vNew, dO, du, g, beta, h, dh, a,
            qRstdOptional, kRstdOptional, betaRawOptional, cuSeqlens, chunkIndices};
        // 公共入口透传缓存身份中的 mask，确保临时输出选择与缓存图一致。
        const uint64_t outputAliasMask = outputTemporaryMask == ~uint64_t{0}
                                            ? GetFinalizeOutputAliasMask(inputs, outputs)
                                            : outputTemporaryMask;
        for (size_t index = 0; index < outputs.size(); ++index) {
            if ((outputAliasMask & (uint64_t{1} << index)) != 0) {
                kernelOutputs[index] = executor->AllocTensor(
                    outputs[index]->GetViewShape(), outputs[index]->GetDataType(), Format::FORMAT_ND);
                if (kernelOutputs[index] == nullptr) {
                    OP_LOGE_WITHOUT_REPORT(OP_NAME, "Failed to allocate a temporary Finalize output[%zu].", index);
                    return {nullptr, nullptr, nullptr, nullptr, nullptr};
                }
            }
        }
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleBwdFinalize,
        OP_INPUT(q, k, v, vNew, dO, du, g, beta, h, dh, a,
                 qRstdOptional, kRstdOptional, betaRawOptional,
                 cuSeqlens, chunkIndices),
        OP_OUTPUT(kernelOutputs[0], kernelOutputs[1], kernelOutputs[2], kernelOutputs[3], kernelOutputs[4]),
        OP_ATTR(scale, chunkSize, useQkL2NormInKernel,
                useBetaSigmoidInKernel, useGateInKernel, stateVFirst, useExp2));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    for (size_t index = 0; index < outputs.size(); ++index) {
        if (kernelOutputs[index] != outputs[index] &&
            op::CopyNpuToNpu(kernelOutputs[index], outputs[index], executor) != ACLNN_SUCCESS) {
            // 显式 copy 保留临时存储，连续 ViewCopy 会把 kernel 输出重新绑定到原输入。
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "Failed to copy a temporary Finalize output[%zu].", index);
            return {nullptr, nullptr, nullptr, nullptr, nullptr};
        }
    }
    return outputs;
}

} // namespace l0op
