#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_INTRA_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_INTRA_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdIntraGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *gk,
    const aclTensor *beta,
    const aclTensor *dAqk,
    const aclTensor *dAkk,
    const aclTensor *dq,
    const aclTensor *dk,
    const aclTensor *db,
    const aclTensor *dg,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    bool safeGate,
    const char *layout,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dbOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdIntra(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
