// SPDX-License-Identifier: BSD-3-Clause
#include "vdn_bridge.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

namespace l0op {
using namespace op;
OP_TYPE_REGISTER(VdnBridge);

const aclTensor *VdnBridge(const aclTensor *prefix, const aclTensor *suffix, const aclTensor *bounds,
                           const aclTensor *logs, int64_t start, const aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(VdnBridge, prefix, suffix, bounds, logs, start, out);
    const int64_t count = out->GetViewShape().GetDim(0);
    const bool outBf16 = out->GetDataType() == DataType::DT_BF16;
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(VdnBridge, OP_INPUT(prefix, suffix, bounds, logs), OP_OUTPUT(out),
                                                 OP_ATTR(start, count, outBf16));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "Failed to enqueue VdnBridge.");
        return nullptr;
    }
    return out;
}
} // namespace l0op
