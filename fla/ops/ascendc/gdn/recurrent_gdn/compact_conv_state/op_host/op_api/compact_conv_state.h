/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef PTA_NPU_OP_API_COMPACT_CONV_STATE_H
#define PTA_NPU_OP_API_COMPACT_CONV_STATE_H

#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *CompactConvState(const aclTensor *convState, const aclTensor *baseAddrs, const aclTensor *blockStrides,
                                  const aclTensor *dimStrides, const aclTensor *tokenStrides,
                                  const aclTensor *stateIndices, const aclTensor *commitLens,
                                  const aclTensor *finalStateIndices, const aclTensor *boundaryStateIndices,
                                  const aclTensor *boundaryRecoveryLens, int64_t convHistoryLen, bool alignMode,
                                  int64_t dimStride, int64_t nullBlockId, aclOpExecutor *executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_COMPACT_CONV_STATE_H
