/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef OP_API_INC_LEVEL0_OP_MERGE_FWD_BWD_KERNEL_H
#define OP_API_INC_LEVEL0_OP_MERGE_FWD_BWD_KERNEL_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 1> MergeFwdBwdKernel(
    const aclTensor *agHm,
    int64_t rank,
    int64_t preOrPostNumRanks,
    bool forward,
    const aclTensor *hOut,
    aclOpExecutor *executor);
} // namespace l0op

#endif
