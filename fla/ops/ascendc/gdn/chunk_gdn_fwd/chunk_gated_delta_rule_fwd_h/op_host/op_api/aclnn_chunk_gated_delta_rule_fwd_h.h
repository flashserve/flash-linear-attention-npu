/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_FWD_H_H
#define OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_FWD_H_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize
 * parameters (order aligned with chunk_gated_delta_rule_fwd_h Python API):
 * k : required
 * w : required
 * u : required
 * gOptional : optional, scalar gate tensor; exactly one of gOptional and gkOptional must be non-null
 * gkOptional : optional, key-wise gate tensor; exactly one of gOptional and gkOptional must be non-null
 * initialStateOptional : optional, float32 or bfloat16
 * outputFinalState : required
 * chunkSize : required
 * saveNewValue : reserved, only true is supported
 * cuSeqlensOptional : optional
 * chunkIndicesOptional : optional
 * useExp2 : exponent implementation/domain selector, independent of the g/gk gate mode
 * stateVFirst : reserved by the physical aclnn interface, only false is supported
 * hOut : required
 * vNewOut : required
 * finalStateOut : required, float32 or bfloat16 and matching initialStateOptional when present;
 *                 its dtype controls the chunk-to-chunk rolling state dtype; use an empty tensor
 *                 with shape [0] when outputFinalState is false
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
    bool stateVFirst,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnChunkGatedDeltaRuleFwdH
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwdH(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
