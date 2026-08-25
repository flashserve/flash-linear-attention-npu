/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "chunk_gated_delta_rule_bwd_dhu.h"

#include <initializer_list>

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleBwdDhu);

namespace {

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

const aclTensor *ConvertIntArrayToTensor(const aclIntArray *array, aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    const_cast<aclTensor *>(tensor)->SetStorageFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(tensor)->SetViewFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(tensor)->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}

} // namespace

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
    bool useExp2,
    const aclTensor *dhOut,
    const aclTensor *dh0Out,
    const aclTensor *dv2Out,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleBwdDhu, q, k, w, dO, dv, gOptional, gkOptional, h0Optional, dhtOptional,
           cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, useExp2, dhOut, dh0Out, dv2Out);

    const aclTensor *actualCuSeqlens = ConvertIntArrayToTensor(cuSeqlensOptional, executor);
    const aclTensor *actualChunkIndices = ConvertIntArrayToTensor(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && actualCuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && actualChunkIndices == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert optional int array to tensor failed.");
        return {nullptr, nullptr, nullptr};
    }

    const aclTensor *dh0OutKernel = dh0Out;
    if (dh0OutKernel == nullptr) {
        dh0OutKernel = executor->AllocTensor(MakeShape({0}), dhOut->GetDataType(), Format::FORMAT_ND);
        if (dh0OutKernel == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Alloc dh0 placeholder failed.");
            return {nullptr, nullptr, nullptr};
        }
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleBwdDhu,
        OP_INPUT(q, k, w, dO, dv, gOptional, gkOptional, h0Optional, dhtOptional,
                 actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(dhOut, dh0OutKernel, dv2Out),
        OP_ATTR(scale, chunkSize, useExp2));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {dhOut, dh0OutKernel, dv2Out};
}

} // namespace l0op
