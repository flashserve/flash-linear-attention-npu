// SPDX-License-Identifier: BSD-3-Clause
#ifndef L0_VDN_SCAN_H
#define L0_VDN_SCAN_H

#include "opdev/op_executor.h"

namespace l0op {
aclnnStatus VdnScan(const aclTensor *transition, const aclTensor *prefix, const aclTensor *suffix, aclOpExecutor *executor);
} // namespace l0op

#endif // L0_VDN_SCAN_H
