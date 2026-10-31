#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_OP_API_ARCH22_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_OP_API_ARCH22_H

#include "../op_api/aclnn_chunk_gated_delta_rule_bwd_finalize.h"
#include "../op_api/chunk_gated_delta_rule_bwd_finalize.h"
#include "chunk_gated_delta_rule_bwd_finalize_validation_arch22.h"

#include <cmath>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

#include "op_common/log/log.h"

namespace {
namespace gdn_finalize_arch22 {

using namespace op;
using GDN::FinalizeHostArch22::ShapeString;
constexpr const char *OP_NAME = "ChunkGatedDeltaRuleBwdFinalize";

// A2/A3 的校验和 view 规范化独立于 A5 基线路径。
bool SameShape(const aclTensor *left, const aclTensor *right)
{
    const auto &leftShape = left->GetViewShape();
    const auto &rightShape = right->GetViewShape();
    if (leftShape.GetDimNum() != rightShape.GetDimNum()) {
        return false;
    }
    for (size_t index = 0; index < leftShape.GetDimNum(); ++index) {
        if (leftShape.GetDim(index) != rightShape.GetDim(index)) {
            return false;
        }
    }
    return true;
}

bool IsDenseFormat(Format format)
{
    return format == Format::FORMAT_ND || format == Format::FORMAT_NCHW ||
           format == Format::FORMAT_NCDHW || format == Format::FORMAT_NCL;
}

aclnnStatus CheckInputShape(const aclTensor *tensor, const std::string &parameter,
                            std::initializer_list<int64_t> expected, size_t inputIndex)
{
    if (!GDN::FinalizeHostArch22::HasShape(tensor->GetViewShape(), expected)) {
        const std::string actualValue = parameter + "=" + ShapeString(tensor->GetViewShape());
        const std::string expectedValue = GDN::FinalizeHostArch22::DimensionsString(expected);
        OP_LOGE_WITH_INVALID_INPUT_SHAPE(OP_NAME, inputIndex, actualValue.c_str(), expectedValue.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    // 连续 view 也需由 Contiguous 规范化 storage shape，并保留 view offset。
    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) {
        OP_LOGE(OP_NAME, "Failed to create a contiguous input.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}


aclnnStatus GetWorkspaceSize(
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
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (!SameShape(q, k)) {
        const std::string actualValue = "k.shape=" + ShapeString(k->GetViewShape());
        const std::string expectedValue = "q.shape=" + ShapeString(q->GetViewShape());
        OP_LOGE_WITH_INVALID_INPUT_SHAPE(OP_NAME, 1, actualValue.c_str(), expectedValue.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (chunkSize != 64 && chunkSize != 128) {
        const std::string actualValue = std::to_string(chunkSize);
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "chunkSize", actualValue.c_str(), "64 or 128 on A2/A3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(scale) || !std::isfinite(static_cast<float>(scale))) {
        const std::string actualValue = std::to_string(scale);
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "scale", actualValue.c_str(), "finite after conversion to FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (useGateInKernel) {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "useGateInKernel", "true", "false on A2/A3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((cuSeqlensOptional == nullptr) != (chunkIndicesOptional == nullptr)) {
        const std::string actualValue = std::string(cuSeqlensOptional == nullptr ? "absent/" : "present/") +
            (chunkIndicesOptional == nullptr ? "absent" : "present");
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "cuSeqlensOptional/chunkIndicesOptional",
                                 actualValue.c_str(), "both present or both absent");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *requiredInputs[] = {q, k, v, vNew, dO, du, g, beta, h, dh, a};
    const char *requiredNames[] = {"q", "k", "v", "vNew", "dO", "du", "g", "beta", "h", "dh", "a"};
    const size_t requiredRanks[] = {4, 4, 4, 4, 4, 4, 3, 3, 5, 5, 4};
    for (size_t index = 0; index < 11; ++index) {
        if (requiredInputs[index]->GetViewShape().GetDimNum() != requiredRanks[index]) {
            const std::string actualValue = std::to_string(requiredInputs[index]->GetViewShape().GetDimNum());
            const std::string expectedValue = std::to_string(requiredRanks[index]);
            OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE(OP_NAME, index, actualValue.c_str(), expectedValue.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const auto &qShape = q->GetViewShape();
    const int64_t tokens = qShape.GetDim(2);
    if (tokens <= 0 || tokens > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        const std::string actualValue = std::to_string(tokens);
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "q.T", actualValue.c_str(), "1 <= T <= UINT32_MAX on A2/A3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t batch = qShape.GetDim(0);
    const int64_t keyHeads = qShape.GetDim(1);
    const int64_t keyDim = qShape.GetDim(3);
    const int64_t valueHeads = v->GetViewShape().GetDim(1);
    const int64_t valueDim = v->GetViewShape().GetDim(3);
    if (batch <= 0 || keyHeads <= 0 || valueHeads <= 0) {
        const std::string actualValue = GDN::FinalizeHostArch22::DimensionsString({batch, keyHeads, valueHeads});
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "B/HK/HV", actualValue.c_str(), "all dimensions > 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t maxScheduleDimension = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    if (batch > maxScheduleDimension || keyHeads > maxScheduleDimension || valueHeads > maxScheduleDimension) {
        const std::string actualValue = GDN::FinalizeHostArch22::DimensionsString({batch, keyHeads, valueHeads});
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "B/HK/HV", actualValue.c_str(), "each dimension <= UINT32_MAX on A2/A3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (keyDim != 128) {
        const std::string actualValue = std::to_string(keyDim);
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "q.K", actualValue.c_str(), "128 on A2/A3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (valueDim != 128 && valueDim != 256) {
        const std::string actualValue = std::to_string(valueDim);
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "v.V", actualValue.c_str(), "128 or 256 on A2/A3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (valueHeads % keyHeads != 0) {
        const std::string actualValue = "HV=" + std::to_string(valueHeads) + ", HK=" + std::to_string(keyHeads);
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "v.HV/q.HK", actualValue.c_str(), "HV % HK == 0 with positive heads");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cuSeqlensOptional == nullptr) {
        const uint64_t lastStart = static_cast<uint64_t>((tokens - 1) / chunkSize * chunkSize);
        if (!GDN::FinalizeHostArch22::ChunkEndFitsUint32(lastStart, static_cast<uint64_t>(chunkSize))) {
            const std::string actualValue = "T=" + std::to_string(tokens) +
                ", last chunk start=" + std::to_string(lastStart) + ", chunkSize=" + std::to_string(chunkSize);
            OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "q.T/chunkSize", actualValue.c_str(),
                                     "last chunk start + chunkSize <= UINT32_MAX on A2/A3");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (qShape.GetDim(0) != 1) {
            const std::string actualValue = std::to_string(qShape.GetDim(0));
            OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "q.B", actualValue.c_str(), "1 in varlen mode");
            return ACLNN_ERR_PARAM_INVALID;
        }
        GDN::FinalizeHostArch22::MetadataError error;
        if (!GDN::FinalizeHostArch22::ValidateVarlenMetadata(
                *cuSeqlensOptional, *chunkIndicesOptional, qShape.GetDim(2), chunkSize, error)) {
            const std::string actualValue = error.actual;
            const std::string expectedValue = error.expected;
            const std::string parameterName = error.parameter;
            OP_LOGE_WITH_INVALID_ATTR(OP_NAME, parameterName.c_str(), actualValue.c_str(), expectedValue.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    const size_t valueIndices[] = {2, 3, 4, 5};
    for (const size_t index : valueIndices) {
        const aclnnStatus status = CheckInputShape(requiredInputs[index], std::string(requiredNames[index]) + ".shape",
                                                   {batch, valueHeads, tokens, valueDim}, index);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    const size_t gateIndices[] = {6, 7};
    for (const size_t index : gateIndices) {
        const aclnnStatus status = CheckInputShape(requiredInputs[index], std::string(requiredNames[index]) + ".shape",
                                                   {batch, valueHeads, tokens}, index);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    const aclnnStatus aShapeStatus = CheckInputShape(a, "a.shape", {batch, valueHeads, tokens, chunkSize}, 10);
    if (aShapeStatus != ACLNN_SUCCESS) {
        return aShapeStatus;
    }
    const int64_t stateChunks = chunkIndicesOptional == nullptr ?
        tokens / chunkSize + (tokens % chunkSize != 0) : static_cast<int64_t>(chunkIndicesOptional->Size() / 2);
    if (chunkIndicesOptional == nullptr && batch > maxScheduleDimension / stateChunks) {
        const std::string actualValue = "B=" + std::to_string(batch) + ", NT=" + std::to_string(stateChunks);
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "totalChunkNum", actualValue.c_str(), "B * NT <= UINT32_MAX on A2/A3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t stateDim3 = stateVFirst ? valueDim : keyDim;
    const int64_t stateDim4 = stateVFirst ? keyDim : valueDim;
    const size_t stateIndices[] = {8, 9};
    for (const size_t index : stateIndices) {
        const aclnnStatus status = CheckInputShape(requiredInputs[index], std::string(requiredNames[index]) + ".shape",
                                                   {batch, stateChunks, valueHeads, stateDim3, stateDim4}, index);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }

    const aclTensor *mainInputs[] = {q, k, v, vNew, dO, du, h, dh, a};
    const char *mainInputNames[] = {"q", "k", "v", "vNew", "dO", "du", "h", "dh", "a"};
    for (size_t index = 0; index < 9; ++index) {
        if (mainInputs[index]->GetDataType() != DataType::DT_BF16) {
            const std::string actualValue = std::to_string(static_cast<int>(mainInputs[index]->GetDataType()));
            const std::string parameterName = std::string(mainInputNames[index]) + ".dtype";
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(OP_NAME, parameterName.c_str(), actualValue.c_str(), "BF16");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const auto gateType = g->GetDataType();
    if (gateType != DataType::DT_BF16 && gateType != DataType::DT_FLOAT) {
        const std::string actualValue = std::to_string(static_cast<int>(gateType));
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(OP_NAME, "g.dtype", actualValue.c_str(), "BF16 or FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (beta->GetDataType() != gateType) {
        const std::string actualValue = std::to_string(static_cast<int>(beta->GetDataType()));
        const std::string expectedValue = "g.dtype=" + std::to_string(static_cast<int>(gateType));
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(OP_NAME, "beta.dtype", actualValue.c_str(), expectedValue.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (useQkL2NormInKernel && (qRstdOptional == nullptr || kRstdOptional == nullptr)) {
        const std::string actualValue = std::string(qRstdOptional == nullptr ? "absent/" : "present/") +
            (kRstdOptional == nullptr ? "absent" : "present");
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "qRstdOptional/kRstdOptional",
                                 actualValue.c_str(), "both present when useQkL2NormInKernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *rstdInputs[] = {qRstdOptional, kRstdOptional};
    const char *rstdNames[] = {"qRstdOptional", "kRstdOptional"};
    for (size_t index = 0; index < 2; ++index) {
        if (rstdInputs[index] != nullptr && rstdInputs[index]->GetDataType() != DataType::DT_FLOAT) {
            const std::string actualValue = std::to_string(static_cast<int>(rstdInputs[index]->GetDataType()));
            const std::string parameterName = std::string(rstdNames[index]) + ".dtype";
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(OP_NAME, parameterName.c_str(), actualValue.c_str(), "FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (useBetaSigmoidInKernel && betaRawOptional == nullptr) {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "betaRawOptional", "absent", "present when useBetaSigmoidInKernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (betaRawOptional != nullptr && betaRawOptional->GetDataType() != beta->GetDataType()) {
        const std::string actualValue = std::to_string(static_cast<int>(betaRawOptional->GetDataType()));
        const std::string expectedValue = "beta.dtype=" + std::to_string(static_cast<int>(beta->GetDataType()));
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(OP_NAME, "betaRawOptional.dtype", actualValue.c_str(), expectedValue.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (useQkL2NormInKernel) {
        for (size_t index = 0; index < 2; ++index) {
            const aclnnStatus status = CheckInputShape(rstdInputs[index], std::string(rstdNames[index]) + ".shape",
                                                       {batch, keyHeads, tokens}, 11 + index);
            if (status != ACLNN_SUCCESS) {
                return status;
            }
        }
    }
    if (useBetaSigmoidInKernel) {
        const aclnnStatus status = CheckInputShape(betaRawOptional, "betaRawOptional.shape",
                                                   {batch, valueHeads, tokens}, 13);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }

    // L0 直接写五个输出；ViewCopy 不负责把 kernel 的连续写入变成 strided 写入。
    // Stable/ctypes 对普通 tensor 使用按 rank 推断的稠密 format；NCHW 等标签
    // 不改变这里由逻辑 shape/stride 定义的连续地址，不能只接受 ND 标签。
    const aclTensor *inputs[] = {q, k, v, beta, g};
    const aclTensor *outputs[] = {dqOut, dkOut, dvOut, dbetaOut, dgOut};
    const char *outputNames[] = {"dqOut", "dkOut", "dvOut", "dbetaOut", "dgOut"};
    for (size_t index = 0; index < 5; ++index) {
        if (!IsContiguous(outputs[index])) {
            const std::string actualValue = "false, shape=" + ShapeString(outputs[index]->GetViewShape());
            const std::string parameterName = std::string(outputNames[index]) + ".contiguous";
            OP_LOGE_WITH_INVALID_ATTR(OP_NAME, parameterName.c_str(), actualValue.c_str(),
                                     "true for direct A2/A3 kernel output writes");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const auto storageFormat = outputs[index]->GetStorageFormat();
        const auto viewFormat = outputs[index]->GetViewFormat();
        if (!IsDenseFormat(storageFormat) || !IsDenseFormat(viewFormat)) {
            const std::string actualValue = std::to_string(static_cast<int>(storageFormat)) + "/" +
                std::to_string(static_cast<int>(viewFormat));
            const std::string parameterName = std::string(outputNames[index]) + ".storageFormat/viewFormat";
            OP_LOGE_WITH_INVALID_INPUT_FORMAT(OP_NAME, parameterName.c_str(), actualValue.c_str(),
                                             "dense ND/NCHW/NCDHW/NCL");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!SameShape(inputs[index], outputs[index])) {
            const std::string actualValue = ShapeString(outputs[index]->GetViewShape());
            const std::string expectedValue = "differentiated input shape=" + ShapeString(inputs[index]->GetViewShape());
            const std::string parameterName = std::string(outputNames[index]) + ".shape";
            OP_LOGE_WITH_INVALID_ATTR(OP_NAME, parameterName.c_str(), actualValue.c_str(), expectedValue.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (inputs[index]->GetDataType() != outputs[index]->GetDataType()) {
            const std::string actualValue = std::to_string(static_cast<int>(outputs[index]->GetDataType()));
            const std::string expectedValue = "differentiated input dtype=" +
                std::to_string(static_cast<int>(inputs[index]->GetDataType()));
            const std::string parameterName = std::string(outputNames[index]) + ".dtype";
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(OP_NAME, parameterName.c_str(), actualValue.c_str(), expectedValue.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE(OP_NAME, "Failed to create the operator executor.");
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto *executorPtr = uniqueExecutor.get();

    const aclTensor **contiguousInputs[] = {&q, &k, &v, &vNew, &dO, &du, &g, &beta, &h, &dh, &a,
                                            &qRstdOptional, &kRstdOptional, &betaRawOptional};
    for (auto **tensor : contiguousInputs) {
        const aclnnStatus status = MakeContiguous(*tensor, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }

    const auto result = l0op::ChunkGatedDeltaRuleBwdFinalize(
        q, k, v, vNew, dO, du, g, beta, h, dh, a,
        qRstdOptional, kRstdOptional, betaRawOptional,
        cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, useQkL2NormInKernel,
        useBetaSigmoidInKernel, useGateInKernel, stateVFirst, useExp2,
        dqOut, dkOut, dvOut, dbetaOut, dgOut,
        executorPtr);
    for (const aclTensor *tensor : result) {
        if (tensor == nullptr) {
            OP_LOGE(OP_NAME, "Failed to create a Finalize output.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    for (size_t index = 0; index < result.size(); ++index) {
        const aclTensor *copied = l0op::ViewCopy(result[index], outputs[index], executorPtr);
        if (copied == nullptr) {
            OP_LOGE(OP_NAME, "Failed to copy output[%zu] to the output view.", index);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace gdn_finalize_arch22
} // namespace

#endif
