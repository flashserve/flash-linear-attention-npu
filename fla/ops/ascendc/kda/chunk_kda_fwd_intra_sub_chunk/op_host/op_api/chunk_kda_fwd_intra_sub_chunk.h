/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details.
 */

#ifndef OP_API_INC_CHUNK_KDA_FWD_INTRA_SUB_CHUNK_H
#define OP_API_INC_CHUNK_KDA_FWD_INTRA_SUB_CHUNK_H

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 2> ChunkKdaFwdIntraSubChunk(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    float scale,
    int64_t chunkSize,
    const aclTensor *aqkOut,
    const aclTensor *akkdOut,
    aclOpExecutor *executor);
} // namespace l0op

#endif
