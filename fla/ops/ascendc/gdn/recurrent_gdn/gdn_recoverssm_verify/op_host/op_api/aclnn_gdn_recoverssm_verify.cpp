/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "aclnn_gdn_recoverssm_verify.h"

#include "gdn_recoverssm_verify_inner.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGdnRecoverssmVerifyGetWorkspaceSize(
    const aclTensor *A_log, const aclTensor *a, const aclTensor *b, const aclTensor *dt_bias,
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *checkpoint_state,
    aclTensor *replayCacheRef, const aclTensor *queryStartLoc, const aclTensor *stateIndices,
    float scale, float softplusBeta, float softplusThreshold, bool useQkL2normInKernel,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return GdnRecoverssmVerifyInnerGetWorkspaceSize(
        A_log, a, b, dt_bias, q, k, v, checkpoint_state, replayCacheRef, queryStartLoc, stateIndices,
        scale, softplusBeta, softplusThreshold, useQkL2normInKernel, out, workspaceSize, executor);
}

aclnnStatus aclnnGdnRecoverssmVerify(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    return GdnRecoverssmVerifyInner(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
