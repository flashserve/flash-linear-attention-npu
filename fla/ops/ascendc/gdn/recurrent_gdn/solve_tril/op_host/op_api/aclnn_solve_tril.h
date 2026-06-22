/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef OP_API_INC_LEVEL0_OP_SOLVE_TRIL_H
#define OP_API_INC_LEVEL0_OP_SOLVE_TRIL_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* SolveTril(
    const aclTensor *x,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const char *layout,
    const aclTensor *xOut,
    aclOpExecutor *executor);
}

#endif