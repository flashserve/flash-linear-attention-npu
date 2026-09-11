/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "merge_fwd_bwd.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MergeFwdBwd);

const std::array<const aclTensor *, 1> MergeFwdBwd(
    const aclTensor *agHm, int64_t rank, int64_t preOrPostNumRanks, bool forward, const aclTensor *hOut,
    aclOpExecutor *executor)
{
    L0_DFX(MergeFwdBwd, agHm, rank, preOrPostNumRanks, forward, hOut);
    auto *mutableH = const_cast<aclTensor *>(hOut);
    mutableH->SetStorageFormat(Format::FORMAT_ND);
    mutableH->SetViewFormat(Format::FORMAT_ND);
    mutableH->SetOriginalFormat(Format::FORMAT_ND);
    auto *mutableAg = const_cast<aclTensor *>(agHm);
    mutableAg->SetStorageFormat(Format::FORMAT_ND);
    mutableAg->SetViewFormat(Format::FORMAT_ND);
    mutableAg->SetOriginalFormat(Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MergeFwdBwd,
        OP_INPUT(agHm),
        OP_OUTPUT(hOut),
        OP_ATTR(forward, rank, preOrPostNumRanks));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE MergeFwdBwd failed.");
        return {nullptr};
    }
    return {hOut};
}
} // namespace l0op
