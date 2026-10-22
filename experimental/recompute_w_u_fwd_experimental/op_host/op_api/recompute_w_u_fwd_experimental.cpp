#include "recompute_w_u_fwd_experimental.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RecomputeWUFwdExperimental);

const aclTensor *RecomputeWUFwdExperimental(
    const aclTensor *x, const aclTensor *y, aclOpExecutor *executor)
{
    (void)executor;
    const auto status = ADD_TO_LAUNCHER_LIST_AICORE(
        RecomputeWUFwdExperimental, OP_INPUT(x), OP_OUTPUT(y));
    return status == ACLNN_SUCCESS ? y : nullptr;
}
} // namespace l0op
