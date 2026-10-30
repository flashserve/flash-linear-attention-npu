#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

// A2/A3 single-layout contract: q/k/v and dq/dk/dv follow layout.
// A supplied rstd pair identifies saved head-major q_hat/k_hat.
// dO/rawG and dg/db are token-major; saved caches remain head-major.
// beta is always original token-major input; fused sigmoid additionally requires
// saved FP32 head-major betaEffOptional. db follows original beta dtype.
// Gate gradients follow input dtype; A_log/dt_bias and their gradients are FP32.
__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdGetWorkspaceSize(
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
aclnnStatus aclnnChunkKdaBwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CHUNK_KDA_BWD_H
