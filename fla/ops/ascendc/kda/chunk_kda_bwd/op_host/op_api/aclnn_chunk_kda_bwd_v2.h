#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_V2_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_V2_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

// A5 V2 single-layout backward: layout controls q/k/v and dq/dk/dv.
// With rstd, q/k are saved head-major normalized tensors. All other saved
// caches are head-major; do/raw_g/beta and dg/dbeta are token-major.
// beta_eff and beta sigmoid flags are reserved (null/false only).
__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdV2GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *gk,
    const aclTensor *aqk,
    const aclTensor *akk,
    const aclTensor *wOptional,
    const aclTensor *qgOptional,
    const aclTensor *kgOptional,
    const aclTensor *vNewOptional,
    const aclTensor *hOptional,
    const aclTensor *dO,
    const aclTensor *rawGOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    bool useGateInKernel,
    double lowerBound,
    bool disableRecompute,
    bool useExp2,
    bool stateVFirst,
    const aclTensor *qRstdOptional,
    const aclTensor *kRstdOptional,
    const aclTensor *betaEffOptional,
    const char *layout,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dbOut,
    const aclTensor *dgOut,
    const aclTensor *dh0OutOptional,
    const aclTensor *dAOutOptional,
    const aclTensor *dBiasOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CHUNK_KDA_BWD_V2_H
