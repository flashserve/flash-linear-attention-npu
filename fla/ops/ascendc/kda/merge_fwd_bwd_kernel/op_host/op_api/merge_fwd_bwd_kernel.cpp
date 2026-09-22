/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "merge_fwd_bwd_kernel.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MergeFwdBwdKernel);

const std::array<const aclTensor *, 1> MergeFwdBwdKernel(
    const aclTensor *h, const aclTensor *agHm, int64_t preOrPostNumRanks, int64_t rank, bool forward,
    bool stateVFirst, aclOpExecutor *executor)
{
    L0_DFX(MergeFwdBwdKernel, h, agHm, preOrPostNumRanks, rank, forward, stateVFirst, h);
    auto *mutableH = const_cast<aclTensor *>(h);
    mutableH->SetStorageFormat(Format::FORMAT_ND);
    mutableH->SetViewFormat(Format::FORMAT_ND);
    mutableH->SetOriginalFormat(Format::FORMAT_ND);
    auto *mutableAg = const_cast<aclTensor *>(agHm);
    mutableAg->SetStorageFormat(Format::FORMAT_ND);
    mutableAg->SetViewFormat(Format::FORMAT_ND);
    mutableAg->SetOriginalFormat(Format::FORMAT_ND);

    // Same tensor for the inplace input and the op output. FLA stores into h;
    // CANN still lists that buffer as Output("h") so the write is visible.
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MergeFwdBwdKernel,
        OP_INPUT(agHm, h),
        OP_OUTPUT(h),
        OP_ATTR(forward, rank, preOrPostNumRanks, stateVFirst));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE MergeFwdBwdKernel failed.");
        return {nullptr};
    }
    return {h};
}
} // namespace l0op
