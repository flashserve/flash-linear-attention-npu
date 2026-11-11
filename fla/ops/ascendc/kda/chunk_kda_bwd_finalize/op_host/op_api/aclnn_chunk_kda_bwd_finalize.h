#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_FINALIZE_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_FINALIZE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// beta accepts BF16/FP32; dBetaOut must have the same dtype.
// Fused sigmoid backward requires saved FP32 beta_eff on A2/A3; A5 rejects this flag.
// allowNegEigval=true requires useBetaSigmoidInKernel=true.
__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdFinalizeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *gk, const aclTensor *rawG, const aclTensor *beta,
    const aclTensor *aLog, const aclTensor *dtBias, const aclTensor *akk,
    const aclTensor *vNew, const aclTensor *h, const aclTensor *dh,
    const aclTensor *dvScan, const aclTensor *dAqk, const aclTensor *dqRaw,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, double lowerBound, int64_t chunkSize,
    bool safeGate, bool useGateInKernel, bool useExp2, bool stateVFirst,
    bool useBetaSigmoidInKernel, bool allowNegEigval, const char *outputLayout,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dBetaOut, const aclTensor *dGOut,
    const aclTensor *dALogOut, const aclTensor *dDtBiasOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdFinalize(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
