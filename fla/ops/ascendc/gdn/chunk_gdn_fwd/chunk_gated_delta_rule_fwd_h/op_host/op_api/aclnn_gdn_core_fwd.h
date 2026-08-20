/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OP_API_INC_ACLNN_GDN_CORE_FWD_H
#define OP_API_INC_ACLNN_GDN_CORE_FWD_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Composite GDN core forward:
 * local_cumsum -> KKT -> solve_tri -> recompute_w_u -> fwd_h -> fwd_o.
 *
 * Phase 2 replaces KKT + solve_tri with one fused kernel while keeping this ABI
 * and the other four kernel boundaries unchanged.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGdnCoreFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    const aclTensor *oOut,
    const aclTensor *finalStateOutOptional,
    const aclTensor *gCumsumOut,
    const aclTensor *aOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnGdnCoreFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
