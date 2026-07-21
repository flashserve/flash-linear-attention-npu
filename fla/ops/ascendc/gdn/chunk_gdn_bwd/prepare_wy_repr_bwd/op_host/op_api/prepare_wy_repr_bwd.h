/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OP_API_INC_LEVEL0_OP_PREPARE_WY_REPR_BWD_H
#define OP_API_INC_LEVEL0_OP_PREPARE_WY_REPR_BWD_H

#include <array>

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 10> PrepareWyReprBwd(
    const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *a, const aclTensor *dw,
    const aclTensor *du, const aclTensor *g, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, int64_t chunkSize, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut, const aclTensor *debugKbgOut,
    const aclTensor *debugVbOut, const aclTensor *debugKbetaOut, const aclTensor *debugDkbgOut,
    const aclTensor *debugDvbOut, const aclTensor *debugKktOut, aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_OP_PREPARE_WY_REPR_BWD_H
