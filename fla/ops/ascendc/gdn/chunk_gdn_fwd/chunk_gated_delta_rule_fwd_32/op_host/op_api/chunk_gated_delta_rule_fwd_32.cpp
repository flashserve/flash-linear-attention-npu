#include "aclnn_chunk_gated_delta_rule_fwd_32.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkGatedDeltaRuleFwd32);

const aclTensor *ChunkGatedDeltaRuleFwd32(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *initialStateOptional, double scale, const aclTensor *o, const aclTensor *finalState,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleFwd32, q, k, v, g, beta, initialStateOptional, scale, o, finalState);
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleFwd32, OP_INPUT(q, k, v, g, beta, initialStateOptional), OP_OUTPUT(o, finalState),
        OP_ATTR(scale));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return o;
}
}  // namespace l0op
