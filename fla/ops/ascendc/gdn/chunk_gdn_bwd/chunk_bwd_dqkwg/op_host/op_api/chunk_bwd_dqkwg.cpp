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
#include "chunk_bwd_dqkwg.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkBwdDqkwg);

const std::array<const aclTensor *, 4> ChunkBwdDqkwg(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *h,
    const aclTensor *dox,
    const aclTensor *dh,
    const aclTensor *dv,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const aclTensor *w,
    const aclTensor *gGamma,
    float scale,
    int64_t chunkSize,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dwOut,
    const aclTensor *dgOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkBwdDqkwg, q, k, v, g, h, dox, dh, dv, cuSeqlensOptional, chunkIndicesOptional, w, gGamma, scale, chunkSize, dqOut, dkOut, dwOut, dgOut);

    const aclTensor *actualCuSeqQLen = nullptr;
    if (cuSeqlensOptional) {
        actualCuSeqQLen = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqQLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualCuSeqQLen = nullptr;
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualChunkIndices = nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkBwdDqkwg,
        OP_INPUT(q, k, v, g, h, dox, dh, dv, actualCuSeqQLen, actualChunkIndices, w, gGamma),
        OP_OUTPUT(dqOut, dkOut, dwOut, dgOut),
        OP_ATTR(scale, chunkSize, 0));

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    return {dqOut, dkOut, dwOut, dgOut};
}

} // namespace l0op
