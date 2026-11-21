/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OP_API_ACLNN_GDN_RECOVERSSM_VERIFY_H
#define OP_API_ACLNN_GDN_RECOVERSSM_VERIFY_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnGdnRecoverssmVerifyGetWorkspaceSize(
    const aclTensor *A_log, const aclTensor *a, const aclTensor *b, const aclTensor *dt_bias,
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *checkpoint_state,
    aclTensor *replayCacheRef, const aclTensor *queryStartLoc, const aclTensor *stateIndices,
    float scale, float softplusBeta, float softplusThreshold, bool useQkL2normInKernel,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGdnRecoverssmVerify(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_GDN_RECOVERSSM_VERIFY_H
