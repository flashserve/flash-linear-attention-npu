/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef PTA_NPU_OP_API_GDN_RECOVERSSM_VERIFY_H
#define PTA_NPU_OP_API_GDN_RECOVERSSM_VERIFY_H

#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *GdnRecoverssmVerify(const aclTensor *A_log, const aclTensor *a, const aclTensor *b,
                                     const aclTensor *dt_bias, const aclTensor *q, const aclTensor *k,
                                     const aclTensor *v, const aclTensor *checkpoint_state, aclTensor *replay_cache,
                                     const aclTensor *query_start_loc, const aclTensor *state_indices, float scale,
                                     float softplus_beta, float softplus_threshold, bool use_qk_l2norm_in_kernel,
                                     aclOpExecutor *executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_GDN_RECOVERSSM_VERIFY_H
