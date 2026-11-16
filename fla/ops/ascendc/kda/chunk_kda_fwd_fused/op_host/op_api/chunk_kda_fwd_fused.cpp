/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "chunk_kda_fwd_fused.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

// l0op::Exp 在 CANN libopapi.so 有导出符号，仓库 vendor 头未含 exp.h。
namespace l0op {
const aclTensor *Exp(const aclTensor *x, aclOpExecutor *executor);
}

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaFwdFused);

const aclTensor *KdaChunkForwardFused(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
    const aclTensor *beta, const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *initialState, double scale, double lowerBound,
    const aclTensor *o, const aclTensor *finalState, const aclIntArray *cuSeqlens,
    aclOpExecutor *executor, const char *layout)
{
    L0_DFX(KdaChunkForwardFused, q, k, v, g, beta, aLog, dtBias, initialState,
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

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaFwdFused,
        OP_INPUT(q, k, v, g, beta, aLogExp, dtBias, initialState, actualCuSeqlens),
        OP_OUTPUT(o, finalState),
        OP_ATTR(layout, scale, static_cast<float>(lowerBound)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwdFused failed.");
        return nullptr;
    }
    return o;
}

} // namespace l0op
