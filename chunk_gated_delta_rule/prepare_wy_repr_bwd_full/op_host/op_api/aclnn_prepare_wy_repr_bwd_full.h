/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_ADD_H
#define OP_API_INC_GROUPED_MATMUL_ADD_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnPrepareWyReprBwdFullGetWorkspaceSize
 * parameters :
 * k : required
 * v : required
 * beta : required
 * a : required
 * dA : required
 * dw : required
 * du : required
 * g : required
 * cuSeqlensOptional : optional
 * chunkIndicesOptional : optional
 * chunkSize : required
 * dkOut : required
 * dvOut : required
 * dbetaOut : required
 * dgOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPrepareWyReprBwdFullGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dA,
    const aclTensor *dw,
    const aclTensor *du,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dbetaOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnPrepareWyReprBwdFull
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPrepareWyReprBwdFull(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif