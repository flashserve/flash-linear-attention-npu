/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "chunk_kda_bwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwd);

const std::array<const aclTensor *, 6> ChunkKdaBwd(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *gk,
    const aclTensor *beta,
    const aclTensor *aqk,
    const aclTensor *akk,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *qg,
    const aclTensor *kg,
    const aclTensor *vNew,
    const aclTensor *h,
    const aclTensor *dO,
    const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    int64_t totalChunks,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dbetaOut,
    const aclTensor *dgkOut,
    const aclTensor *dh0Out,
    const aclTensor *dVNewGrad,
    const aclTensor *dW,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwd, q, k, v, gk, beta, aqk, akk, w, u, qg, kg, vNew, h, dO, initialStateOptional,
           dhtOptional, cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, totalChunks, dqOut, dkOut, dvOut,
           dbetaOut, dgkOut, dh0Out, dVNewGrad, dW);

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
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwd,
        OP_INPUT(q, k, v, gk, beta, aqk, akk, w, u, qg, kg, vNew, h, dO, initialStateOptional, dhtOptional,
                 actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(dqOut, dkOut, dvOut, dbetaOut, dgkOut, dh0Out, dVNewGrad, dW),
        OP_ATTR(scale, chunkSize, totalChunks));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwd failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {dqOut, dkOut, dvOut, dbetaOut, dgkOut, dh0Out};
}

} // namespace l0op
