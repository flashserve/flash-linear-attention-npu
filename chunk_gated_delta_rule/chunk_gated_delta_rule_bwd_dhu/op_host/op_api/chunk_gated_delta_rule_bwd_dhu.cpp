/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "chunk_gated_delta_rule_bwd_dhu.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkGatedDeltaRuleBwdDhu);

const std::array<const aclTensor *, 3> ChunkGatedDeltaRuleBwdDhu(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *dO,
    const aclTensor *dv,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *h0Optional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *dhOut,
    const aclTensor *dh0Out,
    const aclTensor *dv2Out,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleBwdDhu, q, k, w, dO, dv, gOptional, gkOptional, h0Optional, dhtOptional, cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, dhOut, dh0Out, dv2Out);
    
    const aclTensor *actualCuSeqQLen = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqQLen = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqQLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualCuSeqQLen = nullptr;
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional != nullptr) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualChunkIndices = nullptr;
    }

    const aclTensor* dh0OutKernel = nullptr;
    if (dh0Out == nullptr) {
        op::Shape zeroShape;
        zeroShape.AppendDim(0);
        dh0OutKernel = executor->AllocTensor(zeroShape, q->GetDataType(), op::Format::FORMAT_ND);
    } else {
        dh0OutKernel = dh0Out;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkGatedDeltaRuleBwdDhu,
        OP_INPUT(q, k, w, dO, dv, gOptional, gkOptional, h0Optional, dhtOptional, actualCuSeqQLen, actualChunkIndices),
        OP_OUTPUT(dhOut, dh0OutKernel, dv2Out),
        OP_ATTR(scale, chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {dhOut, dh0OutKernel, dv2Out};
}

} // namespace l0op