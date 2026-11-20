/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef PTA_NPU_OP_API_GDN_RECOVERSSM_COMMIT_H
#define PTA_NPU_OP_API_GDN_RECOVERSSM_COMMIT_H

#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *GdnRecoverssmCommit(
    const aclTensor *state, const aclTensor *replay, const aclTensor *table,
    const aclTensor *stateIndices, const aclTensor *commitLens,
    const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices,
    const aclTensor *boundaryRecoveryLens, int64_t specQueryLen,
    bool alignMode, aclOpExecutor *executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_GDN_RECOVERSSM_COMMIT_H
