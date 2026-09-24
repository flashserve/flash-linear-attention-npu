/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_chunk_scaled_dot_kkt.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkScaledDotKkt);

const aclTensor *ChunkScaledDotKkt(
    const aclTensor *k, const aclTensor *g, const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize, const aclTensor *a, aclOpExecutor *executor)
{
    L0_DFX(ChunkScaledDotKkt, k, g, beta, cuSeqlensOptional, chunkIndicesOptional, chunkSize, a);

    const aclTensor *cuSeqlens = nullptr;
    const aclTensor *chunkIndices = nullptr;
    if (cuSeqlensOptional != nullptr) {
        cuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        if (cuSeqlens == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert cu_seqlens to tensor failed.");
            return nullptr;
        }
        const_cast<aclTensor *>(cuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(cuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(cuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);

        chunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        if (chunkIndices == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert chunk_indices to tensor failed.");
            return nullptr;
        }
        const_cast<aclTensor *>(chunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(chunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(chunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkScaledDotKkt, OP_INPUT(k, g, beta, cuSeqlens, chunkIndices),
        OP_OUTPUT(a), OP_ATTR(chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return a;
}
}  // namespace l0op
