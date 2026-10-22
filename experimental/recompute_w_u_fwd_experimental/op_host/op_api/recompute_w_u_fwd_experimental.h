#pragma once

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *RecomputeWUFwdExperimental(
    const aclTensor *x, const aclTensor *y, aclOpExecutor *executor);
} // namespace l0op
