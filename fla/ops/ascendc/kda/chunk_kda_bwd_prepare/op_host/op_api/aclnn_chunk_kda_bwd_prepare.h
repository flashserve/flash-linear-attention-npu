#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_PREPARE_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_PREPARE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdPrepareGetWorkspaceSize(
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *dO,
    const aclTensor *h,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool stateVFirst,
    const aclTensor *dAqkOut,
    const aclTensor *dvOut,
    const aclTensor *dqRawOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdPrepare(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
