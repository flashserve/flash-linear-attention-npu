/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef OP_API_CHUNK_GATED_DELTA_RULE_FWD_PREPARE_H
#define OP_API_CHUNK_GATED_DELTA_RULE_FWD_PREPARE_H

#include <array>

#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 9> ChunkGatedDeltaRuleFwdPrepare(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    bool allowNegEigval,
    bool useExp2,
    bool useQkL2norm,
    bool useGateInKernel,
    bool useBetaSigmoid,
    const aclTensor *gOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *aOut,
    const aclTensor *qHatOptional,
    const aclTensor *kHatOptional,
    const aclTensor *qRstdOptional,
    const aclTensor *kRstdOptional,
    const aclTensor *betaEffOptional,
    aclOpExecutor *executor);

} // namespace l0op

#endif
