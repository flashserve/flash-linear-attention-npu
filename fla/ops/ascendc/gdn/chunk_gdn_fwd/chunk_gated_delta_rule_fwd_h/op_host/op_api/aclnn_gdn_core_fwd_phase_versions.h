/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OP_API_INC_ACLNN_GDN_CORE_FWD_PHASE_VERSIONS_H
#define OP_API_INC_ACLNN_GDN_CORE_FWD_PHASE_VERSIONS_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Phase 1 checkpoint: one ACLNN executor scheduling six independent kernels. */
__attribute__((visibility("default")))
aclnnStatus aclnnGdnCoreFwdPhase1GetWorkspaceSize(
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
aclnnStatus aclnnGdnCoreFwdPhase1(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

/** Phase 2 checkpoint: KKT + solve_tri use the fused ChunkKktSolveTri kernel. */
__attribute__((visibility("default")))
aclnnStatus aclnnGdnCoreFwdPhase2GetWorkspaceSize(
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
aclnnStatus aclnnGdnCoreFwdPhase2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

/** Phase 3 checkpoint: local cumsum + KKT + solve_tri use ChunkCumsumKktSolveTri. */
__attribute__((visibility("default")))
aclnnStatus aclnnGdnCoreFwdPhase3GetWorkspaceSize(
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
aclnnStatus aclnnGdnCoreFwdPhase3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

/** Phase 4 checkpoint: fwd_h + fwd_o use ChunkGatedDeltaRuleFwdHO. */
__attribute__((visibility("default")))
aclnnStatus aclnnGdnCoreFwdPhase4GetWorkspaceSize(
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
aclnnStatus aclnnGdnCoreFwdPhase4(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

/** Phase 5: recompute_w_u and fused fwd_h/fwd_o share one kernel boundary. */
__attribute__((visibility("default")))
aclnnStatus aclnnGdnCoreFwdPhase5GetWorkspaceSize(
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
aclnnStatus aclnnGdnCoreFwdPhase5(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

/** Phase 6 P0a: ABC + DEF and public BTH g_cumsum use one kernel boundary. */
__attribute__((visibility("default")))
aclnnStatus aclnnGdnCoreFwdPhase6GetWorkspaceSize(
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
aclnnStatus aclnnGdnCoreFwdPhase6(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
