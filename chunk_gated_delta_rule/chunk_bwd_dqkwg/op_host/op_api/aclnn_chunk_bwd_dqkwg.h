/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_CHUNK_BWD_DQKWG_H
#define OP_API_INC_CHUNK_BWD_DQKWG_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnChunkBwdDqkwgGetWorkspaceSize
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
aclnnStatus aclnnChunkBwdDqkwgGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *h,
    const aclTensor *dox,
    const aclTensor *dh,
    const aclTensor *dv,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    float scale,
    int64_t chunkSize,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dwOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnChunkBwdDqkwg
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkBwdDqkwg(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif