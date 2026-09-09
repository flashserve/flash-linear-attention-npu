#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single-launch KDA backward megakernel. Inputs use the canonical internal
// head-major token layout consumed by the fused forward exports:
// dense [B,H,T,D], varlen [H,T,D]. Saved h remains chunk-major.
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
