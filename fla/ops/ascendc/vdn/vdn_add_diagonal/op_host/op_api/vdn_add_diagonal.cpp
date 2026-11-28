// SPDX-License-Identifier: BSD-3-Clause
#include "vdn_add_diagonal.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

namespace l0op {
OP_TYPE_REGISTER(VdnAddDiagonal);

aclnnStatus VdnAddDiagonal(const aclTensor *correction, const aclTensor *twoEye, aclOpExecutor *executor)
{
    L0_DFX(VdnAddDiagonal, correction, twoEye, correction);
    // Bind writable banks directly. The kernel consumes their original contents.
    return ADD_TO_LAUNCHER_LIST_AICORE(VdnAddDiagonal, OP_INPUT(correction, twoEye), OP_OUTPUT(correction));
}
} // namespace l0op
