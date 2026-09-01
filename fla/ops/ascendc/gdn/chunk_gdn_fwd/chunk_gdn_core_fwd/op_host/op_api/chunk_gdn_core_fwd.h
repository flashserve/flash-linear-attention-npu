/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OP_API_INC_CHUNK_GDN_CORE_FWD_H
#define OP_API_INC_CHUNK_GDN_CORE_FWD_H

#include "aclnn/aclnn_base.h"
#include <array>

namespace l0op {

const std::array<const aclTensor *, 4> ChunkGdnCoreFwd(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *rawG,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    double scale,
    int64_t outputMask,
    const aclTensor *oOut,
    const aclTensor *finalStateOut,
    const aclTensor *gCumsumBthOut,
    const aclTensor *aOut,
    aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_CHUNK_GDN_CORE_FWD_H
