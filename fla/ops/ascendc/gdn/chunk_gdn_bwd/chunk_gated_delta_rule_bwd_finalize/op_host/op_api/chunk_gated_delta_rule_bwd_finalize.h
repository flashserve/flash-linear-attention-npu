#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_L0_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_L0_H

#include <array>
#include <cstdint>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

namespace l0op {

// 每位对应一个输出，供调用层区分需要临时输出的潜在重叠关系。
uint64_t GetFinalizeOutputAliasMask(const std::array<const aclTensor *, 16> &inputs,
                                   const std::array<const aclTensor *, 5> &outputs);

const std::array<const aclTensor *, 5> ChunkGatedDeltaRuleBwdFinalize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *dO, const aclTensor *du,
    const aclTensor *g, const aclTensor *beta, const aclTensor *h,
    const aclTensor *dh, const aclTensor *a,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclTensor *betaRawOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool useQkL2NormInKernel,
    bool useBetaSigmoidInKernel, bool useGateInKernel, bool stateVFirst,
    bool useExp2,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut,
    aclOpExecutor *executor, uint64_t outputTemporaryMask = ~uint64_t{0});

} // namespace l0op

#endif
