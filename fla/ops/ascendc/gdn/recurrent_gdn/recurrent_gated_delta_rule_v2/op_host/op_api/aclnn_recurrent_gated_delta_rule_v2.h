#ifndef ACLNN_RECURRENT_GATED_DELTA_RULE_V2_H
#define ACLNN_RECURRENT_GATED_DELTA_RULE_V2_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnRecurrentGatedDeltaRuleV2A5GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta, aclTensor *stateRef,
    const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices, const aclTensor *g, const aclTensor *gk,
    const aclTensor *numAcceptedTokens, const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    float scaleValue, bool useQkL2normInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel, bool allowNegEigval,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnRecurrentGatedDeltaRuleV2A5(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_RECURRENT_GATED_DELTA_RULE_V2_H
