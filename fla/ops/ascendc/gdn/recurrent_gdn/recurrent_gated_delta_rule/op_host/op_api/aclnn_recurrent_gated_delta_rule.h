/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OP_API_ACLNN_RECURRENT_GATED_DELTA_RULE_H
#define OP_API_ACLNN_RECURRENT_GATED_DELTA_RULE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First phase: validate descriptors and prepare the fused recurrence.
 * @param [in] query BF16 [T, HK, 128]; positive strides, last stride 1.
 * @param [in] key Same dtype, shape and view constraints as query.
 * @param [in] value BF16 [T, HV, 128]; positive strides, last stride 1.
 * @param [in] beta BF16/FP32 [T, HV].
 * @param [in,out] stateRef BF16/FP32 [Cstate, HV, 128, 128], updated in place.
 * @param [in] actualSeqLengths INT32 [B+1]: prefix, then per-request lengths.
 * @param [in] ssmStateIndices INT32 packed [T] or checkpoint table [B, Ckpt].
 * @param [in] g Optional BF16/FP32 [T, HV].
 * @param [in] gk Reserved; must be nullptr.
 * @param [in] numAcceptedTokens Optional INT32 [B].
 * @param [in] aLog FP32 [HV]; required when useGateInKernel is true.
 * @param [in] dtBias Optional FP32 [HV].
 * @param [in] scaleValue Finite FP32 query scale.
 * @param [in] useQkL2NormInKernel Normalize Q/K in the kernel.
 * @param [in] useGateInKernel Apply the fused gate transform; requires g/aLog.
 * @param [in] useBetaSigmoidInKernel Apply sigmoid to beta.
 * @param [in] allowNegEigval Double sigmoid(beta); ignored without sigmoid.
 * @param [out] out Contiguous BF16 [T, HV, 128].
 * @param [out] workspaceSize Required device workspace size in bytes.
 * @param [out] executor Prepared operator executor.
 * @return aclnnStatus
 *
 * This is the full formerly-V2 signature. Callers of the historical reduced
 * signature must be rebuilt; there is no separate V2 entry point.
 */
ACLNN_API aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
    aclTensor *stateRef, const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices,
    const aclTensor *g, const aclTensor *gk, const aclTensor *numAcceptedTokens,
    const aclTensor *aLog, const aclTensor *dtBias, float scaleValue,
    bool useQkL2NormInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel,
    bool allowNegEigval, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/** Second phase: execute the prepared recurrence on stream. */
ACLNN_API aclnnStatus aclnnRecurrentGatedDeltaRule(void *workspace, uint64_t workspaceSize,
                                                   aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_RECURRENT_GATED_DELTA_RULE_H
