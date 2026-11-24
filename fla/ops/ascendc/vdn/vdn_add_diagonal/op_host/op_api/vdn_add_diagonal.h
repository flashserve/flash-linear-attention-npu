// SPDX-License-Identifier: BSD-3-Clause
#ifndef L0_VDN_ADD_DIAGONAL_H
#define L0_VDN_ADD_DIAGONAL_H

#include "opdev/op_executor.h"

namespace l0op {
aclnnStatus VdnAddDiagonal(const aclTensor *correction, const aclTensor *twoEye, aclOpExecutor *executor);
} // namespace l0op

#endif // L0_VDN_ADD_DIAGONAL_H
