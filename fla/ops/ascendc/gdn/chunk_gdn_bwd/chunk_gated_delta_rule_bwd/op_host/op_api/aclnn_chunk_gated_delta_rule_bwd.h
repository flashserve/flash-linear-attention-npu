/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef ACLNN_CHUNK_GATED_DELTA_RULE_BWD_H
#define ACLNN_CHUNK_GATED_DELTA_RULE_BWD_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnChunkGatedDeltaRuleBwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dO,
    const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclTensor *qRstdOptional,
    const aclTensor *kRstdOptional,
    const aclTensor *betaRawOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    bool useGateInKernel,
    bool useQkL2normInKernel,
    bool useBetaSigmoidInKernel,
    bool stateVFirst,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dBetaOut,
    const aclTensor *dGOut,
    const aclTensor *dh0OutOptional,
    const aclTensor *dALogOutOptional,
    const aclTensor *dDtBiasOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnChunkGatedDeltaRuleBwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
