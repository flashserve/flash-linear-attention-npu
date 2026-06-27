/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CHUNK_KDA_BWD_OP_API_H
#define CHUNK_KDA_BWD_OP_API_H

#include <array>
#include "aclnn/aclnn_base.h"

struct aclOpExecutor;

namespace l0op {
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
    aclOpExecutor *executor);
} // namespace l0op

#endif // CHUNK_KDA_BWD_OP_API_H
