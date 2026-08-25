#ifndef ACLNN_CHUNK_SCALED_DOT_KKT_H_
#define ACLNN_CHUNK_SCALED_DOT_KKT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnChunkScaledDotKktGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkScaledDotKkt(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif