#ifndef ACLNN_CHUNK_KKT_SOLVE_TRI_H
#define ACLNN_CHUNK_KKT_SOLVE_TRI_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKktSolveTriGetWorkspaceSize(
    const aclTensor *k, const aclTensor *g, const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize, const aclTensor *aOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKktSolveTri(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
