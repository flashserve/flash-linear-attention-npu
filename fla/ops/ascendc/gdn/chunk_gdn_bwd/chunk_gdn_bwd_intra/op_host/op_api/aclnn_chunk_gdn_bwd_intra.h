#ifndef OP_API_INC_ACLNN_CHUNK_GDN_BWD_INTRA_H
#define OP_API_INC_ACLNN_CHUNK_GDN_BWD_INTRA_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnChunkGdnBwdIntraGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dO,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    int64_t stage,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *dvLocalOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkGdnBwdIntra(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
