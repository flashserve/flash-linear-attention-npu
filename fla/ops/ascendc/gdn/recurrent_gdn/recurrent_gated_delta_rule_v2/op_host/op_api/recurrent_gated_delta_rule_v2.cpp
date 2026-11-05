#include "recurrent_gated_delta_rule_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RecurrentGatedDeltaRuleV2);

const aclTensor *RecurrentGatedDeltaRuleV2A5(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
    aclTensor *stateRef, const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices,
    const aclTensor *g, const aclTensor *gk, const aclTensor *numAcceptedTokens,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional, float scaleValue,
    bool useQkL2normInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel,
    bool allowNegEigval, aclOpExecutor *executor)
{
    L0_DFX(RecurrentGatedDeltaRuleV2A5, query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk,
           numAcceptedTokens, aLogOptional, dtBiasOptional, scaleValue, useQkL2normInKernel, useGateInKernel,
           useBetaSigmoidInKernel, allowNegEigval);
    auto out = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed."), return nullptr);
    auto ret =
        INFER_SHAPE(RecurrentGatedDeltaRuleV2,
                    OP_INPUT(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk,
                             numAcceptedTokens, aLogOptional, dtBiasOptional),
                    OP_OUTPUT(out, stateRef),
                    OP_ATTR(scaleValue, useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "RecurrentGatedDeltaRuleV2 InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        RecurrentGatedDeltaRuleV2,
        OP_INPUT(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk, numAcceptedTokens,
                 aLogOptional, dtBiasOptional),
        OP_OUTPUT(out, stateRef),
        OP_ATTR(scaleValue, useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "RecurrentGatedDeltaRuleV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}
} // namespace l0op
