/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef OP_API_WMMADDEMO_H
#define OP_API_WMMADDEMO_H

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *WMmadDemo(const aclTensor *a, const aclTensor *b, const aclTensor *pre,
                           const aclTensor *c, aclOpExecutor *executor);

} // namespace l0op

#endif
