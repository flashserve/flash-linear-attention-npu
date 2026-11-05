#ifndef RECURRENT_GATED_DELTA_RULE_V2_L0_H
#define RECURRENT_GATED_DELTA_RULE_V2_L0_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *RecurrentGatedDeltaRuleV2A5(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
    aclTensor *stateRef, const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices,
    const aclTensor *g, const aclTensor *gk, const aclTensor *numAcceptedTokens,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional, float scaleValue,
    bool useQkL2normInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel,
    bool allowNegEigval, aclOpExecutor *executor);
}

#endif // RECURRENT_GATED_DELTA_RULE_V2_L0_H
