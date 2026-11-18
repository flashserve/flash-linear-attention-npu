/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "gdn_recoverssm_verify.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GdnRecoverssmVerify);

const aclTensor *GdnRecoverssmVerify(const aclTensor *A_log, const aclTensor *a, const aclTensor *b,
                                     const aclTensor *dt_bias, const aclTensor *q, const aclTensor *k,
                                     const aclTensor *v, const aclTensor *checkpoint_state, aclTensor *replay_cache,
                                     const aclTensor *query_start_loc, const aclTensor *state_indices, float scale,
                                     float softplus_beta, float softplus_threshold, bool use_qk_l2norm_in_kernel,
                                     aclOpExecutor *executor)
{
    L0_DFX(GdnRecoverssmVerify, A_log, a, b, dt_bias, q, k, v, checkpoint_state, replay_cache, query_start_loc,
           state_indices, scale, softplus_beta, softplus_threshold, use_qk_l2norm_in_kernel);
    auto out = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);
    if (out == nullptr) {
        return nullptr;
    }
    const auto replayStorage = replay_cache->GetStorageShape();
    const auto replayShape = replay_cache->GetViewShape();
    const auto replayStrides = replay_cache->GetViewStrides();
    const auto replayOffset = replay_cache->GetViewOffset();
    auto status = INFER_SHAPE(
        GdnRecoverssmVerify,
        OP_INPUT(A_log, a, b, dt_bias, q, k, v, checkpoint_state, replay_cache, query_start_loc, state_indices),
        OP_OUTPUT(out, replay_cache), OP_ATTR(scale, softplus_beta, softplus_threshold, use_qk_l2norm_in_kernel));
    if (status != ACLNN_SUCCESS) {
        return nullptr;
    }
    replay_cache->SetViewShape(replayShape);
    replay_cache->SetViewStrides(replayStrides);
    replay_cache->SetViewOffset(replayOffset);
    replay_cache->SetStorageShape(replayStorage);
    status = ADD_TO_LAUNCHER_LIST_AICORE(
        GdnRecoverssmVerify,
        OP_INPUT(A_log, a, b, dt_bias, q, k, v, checkpoint_state, replay_cache, query_start_loc, state_indices),
        OP_OUTPUT(out, replay_cache), OP_ATTR(scale, softplus_beta, softplus_threshold, use_qk_l2norm_in_kernel));
    if (status != ACLNN_SUCCESS) {
        return nullptr;
    }
    return out;
}
} // namespace l0op
