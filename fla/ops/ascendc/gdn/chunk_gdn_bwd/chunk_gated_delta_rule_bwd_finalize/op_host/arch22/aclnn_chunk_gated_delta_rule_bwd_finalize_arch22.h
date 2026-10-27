#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_OP_API_ARCH22_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_OP_API_ARCH22_H

#include "../op_api/aclnn_chunk_gated_delta_rule_bwd_finalize.h"
#include "../op_api/chunk_gated_delta_rule_bwd_finalize.h"

#include <cmath>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

namespace {
namespace gdn_finalize_arch22 {

using namespace op;

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
    CHECK_COND(SameShape(q, k), ACLNN_ERR_PARAM_INVALID,
               "k must have the same shape as q.");
    CHECK_COND(chunkSize == 64 || chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize supports 64/128 on A2/A3.");
    CHECK_COND(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)),
               ACLNN_ERR_PARAM_INVALID, "scale must be finite in FP32.");
    CHECK_COND(!useGateInKernel, ACLNN_ERR_PARAM_INVALID,
               "useGateInKernel only supports false.");
    CHECK_COND((cuSeqlensOptional == nullptr) == (chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional and chunkIndicesOptional must be both present or absent.");

    const aclTensor *mainInputs[] = {q, k, v, vNew, dO, du, h, dh, a};
    const char *mainInputNames[] = {"q", "k", "v", "vNew", "dO", "du", "h", "dh", "a"};
    for (size_t index = 0; index < 9; ++index) {
        CHECK_COND(mainInputs[index]->GetDataType() == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID, "%s must use BF16, but got %d.",
                   mainInputNames[index], static_cast<int>(mainInputs[index]->GetDataType()));
    }
    const auto gateType = g->GetDataType();
    CHECK_COND(gateType == DataType::DT_BF16 || gateType == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "g must use BF16 or FP32, but got %d.",
               static_cast<int>(gateType));
    CHECK_COND(beta->GetDataType() == gateType, ACLNN_ERR_PARAM_INVALID,
               "beta must match g dtype, but got g=%d and beta=%d.",
               static_cast<int>(gateType), static_cast<int>(beta->GetDataType()));
    CHECK_COND(!useQkL2NormInKernel || (qRstdOptional != nullptr && kRstdOptional != nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "qRstdOptional and kRstdOptional are required when qk L2Norm is enabled.");
    CHECK_COND(qRstdOptional == nullptr || qRstdOptional->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "qRstdOptional must use FP32 when provided.");
    CHECK_COND(kRstdOptional == nullptr || kRstdOptional->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "kRstdOptional must use FP32 when provided.");
    CHECK_COND(!useBetaSigmoidInKernel || betaRawOptional != nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "betaRawOptional is required when beta sigmoid backward is enabled.");
    CHECK_COND(betaRawOptional == nullptr || betaRawOptional->GetDataType() == beta->GetDataType(),
               ACLNN_ERR_PARAM_INVALID, "betaRawOptional must match beta dtype when provided.");

    // L0 直接写五个输出；ViewCopy 不负责把 kernel 的连续写入变成 strided 写入。
    // Stable/ctypes 对普通 tensor 使用按 rank 推断的稠密 format；NCHW 等标签
    // 不改变这里由逻辑 shape/stride 定义的连续地址，不能只接受 ND 标签。
    const aclTensor *inputs[] = {q, k, v, beta, g};
    const aclTensor *outputs[] = {dqOut, dkOut, dvOut, dbetaOut, dgOut};
    const char *outputNames[] = {"dqOut", "dkOut", "dvOut", "dbetaOut", "dgOut"};
    for (size_t index = 0; index < 5; ++index) {
        CHECK_COND(IsContiguous(outputs[index]), ACLNN_ERR_PARAM_INVALID,
                   "%s must be contiguous because the kernel writes it directly.", outputNames[index]);
        const auto storageFormat = outputs[index]->GetStorageFormat();
        const auto viewFormat = outputs[index]->GetViewFormat();
        CHECK_COND(IsDenseFormat(storageFormat) && IsDenseFormat(viewFormat),
                   ACLNN_ERR_PARAM_INVALID,
                   "%s must use dense ND/NCHW/NCDHW/NCL storage/view format, but got %d/%d.",
                   outputNames[index], static_cast<int>(storageFormat), static_cast<int>(viewFormat));
        CHECK_COND(SameShape(inputs[index], outputs[index]), ACLNN_ERR_PARAM_INVALID,
                   "%s must match its differentiated input shape.", outputNames[index]);
        CHECK_COND(inputs[index]->GetDataType() == outputs[index]->GetDataType(),
                   ACLNN_ERR_PARAM_INVALID,
                   "%s must match its differentiated input dtype, but got input=%d and output=%d.",
                   outputNames[index], static_cast<int>(inputs[index]->GetDataType()),
                   static_cast<int>(outputs[index]->GetDataType()));
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();

    CHECK_RET(MakeContiguous(q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(vNew, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(dO, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(du, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(h, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(dh, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(a, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(qRstdOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(kRstdOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(betaRawOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

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
