#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_OP_API_ARCH22_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_OP_API_ARCH22_H

#include "../op_api/aclnn_chunk_gated_delta_rule_bwd_finalize.h"
#include "../op_api/chunk_gated_delta_rule_bwd_finalize.h"
#include "chunk_gated_delta_rule_bwd_finalize_validation_arch22.h"
#include "chunk_gated_delta_rule_bwd_finalize_error_arch22.h"

#include <cmath>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

namespace {
namespace gdn_finalize_arch22 {

using namespace op;
using GDN::FinalizeHostArch22::ShapeString;
using GDN::FinalizeHostArch22::ParameterKind;

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

aclnnStatus InvalidParameter(const std::string &parameter, const std::string &actual,
                             const std::string &expected,
                             ParameterKind kind = ParameterKind::VALUE, size_t inputIndex = 0)
{
    OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "A2/A3 parameter[%s]: actual=%s; expected=%s.",
            parameter.c_str(), actual.c_str(), expected.c_str());
    if (!GDN::FinalizeHostArch22::ReportParameterError(parameter, actual, expected, kind, inputIndex)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A2/A3 parameter[%s]: actual=%s; expected=%s.",
                parameter.c_str(), actual.c_str(), expected.c_str());
    }
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus CheckInputShape(const aclTensor *tensor, const std::string &parameter,
                            std::initializer_list<int64_t> expected, size_t inputIndex)
{
    if (!GDN::FinalizeHostArch22::HasShape(tensor->GetViewShape(), expected)) {
        return InvalidParameter(parameter, ShapeString(tensor->GetViewShape()),
                                GDN::FinalizeHostArch22::DimensionsString(expected), ParameterKind::SHAPE, inputIndex);
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
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
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
        return InvalidParameter("k.shape", ShapeString(k->GetViewShape()),
                                "q.shape=" + ShapeString(q->GetViewShape()), ParameterKind::SHAPE, 1);
    }
    if (chunkSize != 64 && chunkSize != 128) {
        return InvalidParameter("chunkSize", std::to_string(chunkSize), "64 or 128 on A2/A3");
    }
    if (!std::isfinite(scale) || !std::isfinite(static_cast<float>(scale))) {
        return InvalidParameter("scale", std::to_string(scale), "finite after conversion to FP32");
    }
    if (useGateInKernel) {
        return InvalidParameter("useGateInKernel", "true", "false on A2/A3");
    }
    if ((cuSeqlensOptional == nullptr) != (chunkIndicesOptional == nullptr)) {
        return InvalidParameter("cuSeqlensOptional/chunkIndicesOptional",
                                std::string(cuSeqlensOptional == nullptr ? "absent/" : "present/") +
                                (chunkIndicesOptional == nullptr ? "absent" : "present"),
                                "both present or both absent");
    }
    const aclTensor *requiredInputs[] = {q, k, v, vNew, dO, du, g, beta, h, dh, a};
    const char *requiredNames[] = {"q", "k", "v", "vNew", "dO", "du", "g", "beta", "h", "dh", "a"};
    const size_t requiredRanks[] = {4, 4, 4, 4, 4, 4, 3, 3, 5, 5, 4};
    for (size_t index = 0; index < 11; ++index) {
        if (requiredInputs[index]->GetViewShape().GetDimNum() != requiredRanks[index]) {
            return InvalidParameter(std::string(requiredNames[index]) + ".rank",
                                    ShapeString(requiredInputs[index]->GetViewShape()),
                                    "rank " + std::to_string(requiredRanks[index]), ParameterKind::SHAPE, index);
        }
    }
    const auto &qShape = q->GetViewShape();
    const int64_t tokens = qShape.GetDim(2);
    if (tokens <= 0 || tokens > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return InvalidParameter("q.T", std::to_string(tokens), "1 <= T <= UINT32_MAX on A2/A3");
    }
    const int64_t batch = qShape.GetDim(0);
    const int64_t keyHeads = qShape.GetDim(1);
    const int64_t keyDim = qShape.GetDim(3);
    const int64_t valueHeads = v->GetViewShape().GetDim(1);
    const int64_t valueDim = v->GetViewShape().GetDim(3);
    if (batch <= 0 || keyHeads <= 0 || valueHeads <= 0) {
        return InvalidParameter("B/HK/HV", GDN::FinalizeHostArch22::DimensionsString({batch, keyHeads, valueHeads}),
                                "all dimensions > 0");
    }
    const int64_t maxScheduleDimension = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    if (batch > maxScheduleDimension || keyHeads > maxScheduleDimension || valueHeads > maxScheduleDimension) {
        return InvalidParameter("B/HK/HV", GDN::FinalizeHostArch22::DimensionsString({batch, keyHeads, valueHeads}),
                                "each dimension <= UINT32_MAX on A2/A3");
    }
    if (keyDim != 128) {
        return InvalidParameter("q.K", std::to_string(keyDim), "128 on A2/A3");
    }
    if (valueDim != 128 && valueDim != 256) {
        return InvalidParameter("v.V", std::to_string(valueDim), "128 or 256 on A2/A3");
    }
    if (valueHeads % keyHeads != 0) {
        return InvalidParameter("v.HV/q.HK", "HV=" + std::to_string(valueHeads) +
                                ", HK=" + std::to_string(keyHeads), "HV % HK == 0 with positive heads");
    }
    if (cuSeqlensOptional == nullptr) {
        const uint64_t lastStart = static_cast<uint64_t>((tokens - 1) / chunkSize * chunkSize);
        if (!GDN::FinalizeHostArch22::ChunkEndFitsUint32(lastStart, static_cast<uint64_t>(chunkSize))) {
            return InvalidParameter("q.T/chunkSize", "T=" + std::to_string(tokens) +
                                    ", last chunk start=" + std::to_string(lastStart) +
                                    ", chunkSize=" + std::to_string(chunkSize),
                                    "last chunk start + chunkSize <= UINT32_MAX on A2/A3");
        }
    } else {
        if (qShape.GetDim(0) != 1) {
            return InvalidParameter("q.B", std::to_string(qShape.GetDim(0)), "1 in varlen mode");
        }
        GDN::FinalizeHostArch22::MetadataError error;
        if (!GDN::FinalizeHostArch22::ValidateVarlenMetadata(
                *cuSeqlensOptional, *chunkIndicesOptional, qShape.GetDim(2), chunkSize, error)) {
            return InvalidParameter(error.parameter, error.actual, error.expected);
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
        return InvalidParameter("totalChunkNum", "B=" + std::to_string(batch) +
                                ", NT=" + std::to_string(stateChunks), "B * NT <= UINT32_MAX on A2/A3");
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
            return InvalidParameter(std::string(mainInputNames[index]) + ".dtype",
                                    std::to_string(static_cast<int>(mainInputs[index]->GetDataType())), "BF16",
                                    ParameterKind::DTYPE);
        }
    }
    const auto gateType = g->GetDataType();
    if (gateType != DataType::DT_BF16 && gateType != DataType::DT_FLOAT) {
        return InvalidParameter("g.dtype", std::to_string(static_cast<int>(gateType)),
                                "BF16 or FP32",
        ParameterKind::DTYPE);
    }
    if (beta->GetDataType() != gateType) {
        return InvalidParameter("beta.dtype", std::to_string(static_cast<int>(beta->GetDataType())),
                                "g.dtype=" + std::to_string(static_cast<int>(gateType)), ParameterKind::DTYPE);
    }
    if (useQkL2NormInKernel && (qRstdOptional == nullptr || kRstdOptional == nullptr)) {
        return InvalidParameter("qRstdOptional/kRstdOptional",
                                std::string(qRstdOptional == nullptr ? "absent/" : "present/") +
                                (kRstdOptional == nullptr ? "absent" : "present"),
                                "both present when useQkL2NormInKernel=true");
    }
    const aclTensor *rstdInputs[] = {qRstdOptional, kRstdOptional};
    const char *rstdNames[] = {"qRstdOptional", "kRstdOptional"};
    for (size_t index = 0; index < 2; ++index) {
        if (rstdInputs[index] != nullptr && rstdInputs[index]->GetDataType() != DataType::DT_FLOAT) {
            return InvalidParameter(std::string(rstdNames[index]) + ".dtype",
                                    std::to_string(static_cast<int>(rstdInputs[index]->GetDataType())), "FP32",
                                    ParameterKind::DTYPE);
        }
    }
    if (useBetaSigmoidInKernel && betaRawOptional == nullptr) {
        return InvalidParameter("betaRawOptional", "absent", "present when useBetaSigmoidInKernel=true");
    }
    if (betaRawOptional != nullptr && betaRawOptional->GetDataType() != beta->GetDataType()) {
        return InvalidParameter("betaRawOptional.dtype",
                                std::to_string(static_cast<int>(betaRawOptional->GetDataType())),
                                "beta.dtype=" + std::to_string(static_cast<int>(beta->GetDataType())),
                                ParameterKind::DTYPE);
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
            return InvalidParameter(std::string(outputNames[index]) + ".contiguous",
                                    "false, shape=" + ShapeString(outputs[index]->GetViewShape()),
                                    "true for direct A2/A3 kernel output writes");
        }
        const auto storageFormat = outputs[index]->GetStorageFormat();
        const auto viewFormat = outputs[index]->GetViewFormat();
        if (!IsDenseFormat(storageFormat) || !IsDenseFormat(viewFormat)) {
            return InvalidParameter(std::string(outputNames[index]) + ".storageFormat/viewFormat",
                                    std::to_string(static_cast<int>(storageFormat)) + "/" +
                                    std::to_string(static_cast<int>(viewFormat)), "dense ND/NCHW/NCDHW/NCL",
                                    ParameterKind::FORMAT);
        }
        if (!SameShape(inputs[index], outputs[index])) {
            return InvalidParameter(std::string(outputNames[index]) + ".shape",
                                    ShapeString(outputs[index]->GetViewShape()),
                                    "differentiated input shape=" + ShapeString(inputs[index]->GetViewShape()));
        }
        if (inputs[index]->GetDataType() != outputs[index]->GetDataType()) {
            return InvalidParameter(std::string(outputNames[index]) + ".dtype",
                                    std::to_string(static_cast<int>(outputs[index]->GetDataType())),
                                    "differentiated input dtype=" +
                                    std::to_string(static_cast<int>(inputs[index]->GetDataType())),
                                    ParameterKind::DTYPE);
        }
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
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
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    for (size_t index = 0; index < result.size(); ++index) {
        const aclTensor *copied = l0op::ViewCopy(result[index], outputs[index], executorPtr);
        CHECK_RET(copied != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace gdn_finalize_arch22
} // namespace

#endif
