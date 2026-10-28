#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

// A2/A3 compose backward L0 operators; A5 keeps its fused kernel.
// Saved caches remain head-major, with chunk-major h. External token inputs
// follow inputLayout. On A2/A3, dq/dk/dv follow their input dtypes and
// outputLayout; db/dg always use BSND/TND. A5 keeps its output contract.
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
    const char *inputLayout,
    const char *outputLayout,
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
