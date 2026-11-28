// SPDX-License-Identifier: BSD-3-Clause
#ifndef L0_VDN_BRIDGE_H
#define L0_VDN_BRIDGE_H

#include "aclnn/aclnn_base.h"

namespace l0op {
const aclTensor *VdnBridge(const aclTensor *prefix, const aclTensor *suffix, const aclTensor *bounds,
                           const aclTensor *logs, int64_t start, const aclTensor *out, aclOpExecutor *executor);
} // namespace l0op
#endif
