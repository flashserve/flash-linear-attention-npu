/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef OP_API_INC_LEVEL0_OP_CHUNK_BWD_DQKWG_OP_H
#define OP_API_INC_LEVEL0_OP_CHUNK_BWD_DQKWG_OP_H

#include "opdev/op_executor.h"

namespace l0op {
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
    aclOpExecutor *executor);
}

#endif