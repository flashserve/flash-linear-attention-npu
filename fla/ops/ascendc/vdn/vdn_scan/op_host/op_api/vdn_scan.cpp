// SPDX-License-Identifier: BSD-3-Clause
#include "vdn_scan.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

namespace l0op {
OP_TYPE_REGISTER(VdnScan);

aclnnStatus VdnScan(const aclTensor *transition, const aclTensor *prefix, const aclTensor *suffix, aclOpExecutor *executor)
{
    L0_DFX(VdnScan, transition, prefix, suffix, prefix, suffix);
    // Bind writable banks directly. The kernel consumes their original contents.
    return ADD_TO_LAUNCHER_LIST_AICORE(VdnScan, OP_INPUT(transition, prefix, suffix), OP_OUTPUT(prefix, suffix));
}
} // namespace l0op
