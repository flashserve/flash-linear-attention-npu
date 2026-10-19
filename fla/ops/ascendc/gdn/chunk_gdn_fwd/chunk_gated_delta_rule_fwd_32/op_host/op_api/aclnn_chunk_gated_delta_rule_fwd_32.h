#ifndef OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_FWD_32_H
#define OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_FWD_32_H

#include <cstdint>

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRuleFwd32GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *initialStateOptional, double scale, const aclTensor *o, const aclTensor *finalState,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRuleFwd32(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace l0op {
const aclTensor *ChunkGatedDeltaRuleFwd32(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *initialStateOptional, double scale, const aclTensor *o, const aclTensor *finalState,
    aclOpExecutor *executor);
}

#endif
