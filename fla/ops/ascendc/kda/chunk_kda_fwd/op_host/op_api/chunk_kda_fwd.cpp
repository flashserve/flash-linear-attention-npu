/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "chunk_kda_fwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

// l0op::Exp 在 CANN libopapi.so 有导出符号，仓库 vendor 头未含 exp.h。
namespace l0op {
const aclTensor *Exp(const aclTensor *x, aclOpExecutor *executor);
}

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaFwd);

KdaCoreOutputs KdaChunkForward(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    bool inputSequenceMajor,
    bool useGateInKernel,
    double lowerBound,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    const aclTensor *qgScaledOut,
    const aclTensor *uSeedOut,
    int64_t stage,
    aclOpExecutor *executor)
{
    L0_DFX(KdaChunkForward, q, k, v, g, beta, aLogOptional, dtBiasOptional,
           initialStateOptional, cuSeqlensOptional, chunkIndicesOptional,
           scale, chunkSize, safeGate, inputSequenceMajor, useGateInKernel,
           lowerBound, attnOut, finalStateOut, gkOut, aqkOut, akkOut,
           wOut, uOut, qgOut, kgOut, vNewOut, hOut, qgScaledOut, uSeedOut,
           stage);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional != nullptr) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        if (actualChunkIndices == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to convert chunk metadata to tensor.");
            return {};
        }
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaFwd,
        OP_INPUT(q, k, v, g, beta, aLogOptional, dtBiasOptional,
                 initialStateOptional, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut,
                  qgOut, kgOut, vNewOut, hOut, qgScaledOut, uSeedOut),
        OP_ATTR(inputSequenceMajor ? "BSND" : "BNSD", scale, chunkSize,
                safeGate, static_cast<float>(lowerBound), useGateInKernel,
                false, stage));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwd failed.");
        return {};
    }
    return {attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut,
            qgOut, kgOut, vNewOut, hOut, qgScaledOut, uSeedOut};
}

const aclTensor *KdaChunkForward32(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
    const aclTensor *beta, const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *initialState, double scale, double lowerBound,
    const aclTensor *o, const aclTensor *finalState, const aclIntArray *cuSeqlens,
    aclOpExecutor *executor, const char *layout)
{
    L0_DFX(KdaChunkForward32, q, k, v, g, beta, aLog, dtBias, initialState,
           scale, lowerBound, o, finalState);

    const aclTensor *aLogExp = l0op::Exp(aLog, executor);
    if (aLogExp == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0op::Exp on aLog failed.");
        return nullptr;
    }
    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlens != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlens, DataType::DT_INT64);
        if (actualCuSeqlens == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to convert cu_seqlens to tensor.");
            return nullptr;
        }
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    }

    // Aqk/Akk 在现有 def 中为 REQUIRED；chunk32 不导出有效中间量，仅提供内部占位。
    const aclTensor *aqkPlaceholder = executor->AllocTensor(op::Shape({1}), q->GetDataType(), Format::FORMAT_ND);
    const aclTensor *akkPlaceholder = executor->AllocTensor(op::Shape({1}), q->GetDataType(), Format::FORMAT_ND);
    const aclTensor *unusedQ = executor->AllocTensor(op::Shape({1}), q->GetDataType(), Format::FORMAT_ND);
    const aclTensor *unusedF32 = executor->AllocTensor(op::Shape({1}), DataType::DT_FLOAT, Format::FORMAT_ND);
    const aclTensor *unusedI64 = executor->AllocTensor(op::Shape({1}), DataType::DT_INT64, Format::FORMAT_ND);
    if (aqkPlaceholder == nullptr || akkPlaceholder == nullptr || unusedQ == nullptr ||
        unusedF32 == nullptr || unusedI64 == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to allocate chunk32 output placeholders.");
        return nullptr;
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaFwd,
        OP_INPUT(q, k, v, g, beta, aLogExp, dtBias, initialState, actualCuSeqlens, unusedI64),
        OP_OUTPUT(o, finalState, unusedF32, aqkPlaceholder, akkPlaceholder, unusedQ, unusedQ,
                  unusedQ, unusedQ, unusedQ, unusedQ, unusedQ, unusedQ),
        OP_ATTR(layout, scale, 32, true, static_cast<float>(lowerBound), true, false, -1));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwd chunk32 failed.");
        return nullptr;
    }
    return o;
}

} // namespace l0op
