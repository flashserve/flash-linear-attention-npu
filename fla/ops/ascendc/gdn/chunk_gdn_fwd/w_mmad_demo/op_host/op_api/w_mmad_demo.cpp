/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "w_mmad_demo.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(WMmadDemo);

const aclTensor *WMmadDemo(const aclTensor *a, const aclTensor *b, const aclTensor *pre,
                           const aclTensor *c, aclOpExecutor *executor)
{
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(WMmadDemo, OP_INPUT(a, b, pre), OP_OUTPUT(c));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE WMmadDemo failed.");
        return nullptr;
    }
    return c;
}

} // namespace l0op
