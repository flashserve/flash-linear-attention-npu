/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
    float scale,
    int64_t chunkSize,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dwOut,
    const aclTensor *dgOut,
    aclOpExecutor *executor)
{
// std::cout << "2222222222--2222200\n";
    L0_DFX(ChunkBwdDqkwg, q, k, v, g, h, dox, dh, dv, cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, dqOut, dkOut, dwOut, dgOut);
// std::cout << "2222222222--2222200111\n";
    
    const aclTensor *actualCuSeqQLen = nullptr;
    if (cuSeqlensOptional) {
// printf("dh %p, dv %p, cuSeqlensOptional : %p, chunkIndicesOptional %p\n",dh,dv,cuSeqlensOptional,chunkIndicesOptional);
        actualCuSeqQLen = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
// std::cout << "2222222222--2222200222-A" << actualCuSeqQLen <<"\n";
        const_cast<aclTensor *>(actualCuSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
// std::cout << "2222222222--2222200222-B\n";
        const_cast<aclTensor *>(actualCuSeqQLen)->SetViewFormat(Format::FORMAT_ND);
// std::cout << "2222222222--2222200222-C\n";
        const_cast<aclTensor *>(actualCuSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualCuSeqQLen = nullptr;
    }
// std::cout << "2222222222--2222200222\n";

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional) {
// std::cout << "2222222222--2222200222\n";
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
// std::cout << "1111111111\n";
        actualChunkIndices = nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkBwdDqkwg,
        OP_INPUT(q, k, v, g, h, dox, dh, dv, actualCuSeqQLen, actualChunkIndices),
        OP_OUTPUT(dqOut, dkOut, dwOut, dgOut),
        OP_ATTR(scale, chunkSize));

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    return {dqOut, dkOut, dwOut, dgOut};
}

} // namespace l0op