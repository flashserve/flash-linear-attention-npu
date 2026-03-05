/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_CHUNK_GATED_DELTA_RULE_BWD_DHU_H_
#define ACLNN_CHUNK_GATED_DELTA_RULE_BWD_DHU_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * w : required
 * dO : required
 * dv : required
 * gOptional : optional
 * gkOptional : optional
 * h0Optional : optional
 * dhtOptional : optional
 * cuSeqlensOptional : optional
 * chunkIndicesOptional : optional
 * scale : optional
 * chunkSize : optional
 * dhOut : required
 * dh0Out : required
 * dv2Out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *dO,
    const aclTensor *dv,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *h0Optional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *dhOut,
    const aclTensor *dh0Out,
    const aclTensor *dv2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnChunkGatedDeltaRuleBwdDhu
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleBwdDhu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
