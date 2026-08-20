/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OP_API_INC_LEVEL0_OP_CHUNK_SCALED_DOT_KKT_H
#define OP_API_INC_LEVEL0_OP_CHUNK_SCALED_DOT_KKT_H

#include <cstdint>

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

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
    const aclTensor *aOut,
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
