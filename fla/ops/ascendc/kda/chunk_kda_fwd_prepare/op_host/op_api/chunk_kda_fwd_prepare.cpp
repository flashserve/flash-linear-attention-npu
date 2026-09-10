/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "chunk_kda_fwd_prepare.h"

#include <cstdint>

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkKdaFwdPrepare);

namespace {

const aclTensor *ConvertIntArrayToTensor(const aclIntArray *array,
                                         aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor =
        executor->ConvertToTensor(array, DataType::DT_INT64);
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

ChunkKdaFwdPrepareOutputs ChunkKdaFwdPrepare(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    double epsilon,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool useExp2,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *qgScaledOut,
    const aclTensor *qHatOut,
    const aclTensor *kHatOut,
    const aclTensor *qRstdOut,
    const aclTensor *kRstdOut,
    const aclTensor *betaEffOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaFwdPrepare, q, k, v, g, beta, aLogOptional,
           dtBiasOptional, cuSeqlensOptional, chunkIndicesOptional, layout,
           scale, chunkSize, epsilon, useQkL2normInKernel, useGateInKernel,
           useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound,
           useExp2, gkOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut,
           qgScaledOut, qHatOut, kHatOut, qRstdOut, kRstdOut, betaEffOut);

    const aclTensor *cuSeqlens =
        ConvertIntArrayToTensor(cuSeqlensOptional, executor);
    const aclTensor *chunkIndices =
        ConvertIntArrayToTensor(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && chunkIndices == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "转换 cu_seqlens/chunk_indices 到 Tensor 失败。");
        return {};
    }

    const auto status = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaFwdPrepare,
        OP_INPUT(q, k, v, g, beta, aLogOptional, dtBiasOptional,
                 cuSeqlens, chunkIndices),
        OP_OUTPUT(gkOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut,
                  qgScaledOut, qHatOut, kHatOut, qRstdOut, kRstdOut,
                  betaEffOut),
        OP_ATTR(layout, scale, chunkSize, epsilon, useQkL2normInKernel,
                useGateInKernel, useBetaSigmoidInKernel, allowNegEigval,
                safeGate, lowerBound, useExp2));
    if (status != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "添加 ChunkKdaFwdPrepare AI Core 任务失败。");
        return {};
    }

    return {gkOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut, qgScaledOut,
            qHatOut, kHatOut, qRstdOut, kRstdOut, betaEffOut};
}

} // namespace l0op
