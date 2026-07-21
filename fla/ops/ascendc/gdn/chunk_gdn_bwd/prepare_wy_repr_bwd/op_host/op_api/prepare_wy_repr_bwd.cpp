/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "prepare_wy_repr_bwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(PrepareWyReprBwd);

const std::array<const aclTensor *, 10> PrepareWyReprBwd(
    const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *a, const aclTensor *dw,
    const aclTensor *du, const aclTensor *g, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, int64_t chunkSize, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut, const aclTensor *debugKbgOut,
    const aclTensor *debugVbOut, const aclTensor *debugKbetaOut, const aclTensor *debugDkbgOut,
    const aclTensor *debugDvbOut, const aclTensor *debugKktOut, aclOpExecutor *executor)
{
    L0_DFX(PrepareWyReprBwd, k, v, beta, a, dw, du, g, cuSeqlensOptional, chunkIndicesOptional, chunkSize, dkOut,
           dvOut, dbetaOut, dgOut, debugKbgOut, debugVbOut, debugKbetaOut, debugDkbgOut, debugDvbOut,
           debugKktOut);

    const aclTensor *actualCuSeqQLen = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqQLen = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqQLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional != nullptr) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(PrepareWyReprBwd,
        OP_INPUT(k, v, beta, a, dw, du, g, actualCuSeqQLen, actualChunkIndices),
        OP_OUTPUT(dkOut, dvOut, dbetaOut, dgOut, debugKbgOut, debugVbOut, debugKbetaOut, debugDkbgOut,
                  debugDvbOut, debugKktOut),
        OP_ATTR(chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {dkOut, dvOut, dbetaOut, dgOut, debugKbgOut, debugVbOut, debugKbetaOut, debugDkbgOut, debugDvbOut,
            debugKktOut};
}

} // namespace l0op
