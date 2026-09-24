/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "chunk_delta_h_bwd_preprocess.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkDeltaHBwdPreprocess);

const aclTensor *ChunkDeltaHBwdPreprocess(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *dO,
    const aclTensor *dv,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclIntArray *cuSeqlensOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *dhmOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkDeltaHBwdPreprocess, q, k, w, dO, dv, gOptional, gkOptional, cuSeqlensOptional, scale, chunkSize,
           dhmOut);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkDeltaHBwdPreprocess,
        OP_INPUT(q, k, w, dO, dv, gOptional, gkOptional, actualCuSeqlens),
        OP_OUTPUT(dhmOut),
        OP_ATTR(scale, chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return dhmOut;
}

} // namespace l0op
