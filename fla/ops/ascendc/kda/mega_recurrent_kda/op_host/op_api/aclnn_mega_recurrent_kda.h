#pragma once
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API aclnnStatus aclnnMegaRecurrentKdaGetWorkspaceSize(
    const aclTensor *hidden_states, const aclTensor *qkv_projection_weight, const aclTensor *decay_projection_a_weight,
    const aclTensor *decay_projection_b_weight, const aclTensor *beta_projection_weight,
    const aclTensor *output_gate_projection_weight, const aclTensor *output_norm_weight,
    const aclTensor *output_projection_weight, const aclTensor *conv1d_weight, aclTensor *conv_state,
    const aclTensor *checkpoint_state, aclTensor *replay_u, aclTensor *replay_k, aclTensor *replay_decay,
    const aclIntArray *state_indices, const aclIntArray *conv_num_accepted_tokens, const aclTensor *a_log,
    const aclTensor *dt_bias, const aclIntArray *cu_seqlens, double scale, double lower_bound, double rms_norm_eps,
    const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnMegaRecurrentKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream);
ACLNN_API aclnnStatus aclnnMegaRecurrentKdaV2GetWorkspaceSize(
    const aclTensor *hidden_states, const aclTensor *qkv_projection_weight, const aclTensor *decay_projection_a_weight,
    const aclTensor *decay_projection_b_weight, const aclTensor *beta_projection_weight,
    const aclTensor *output_gate_projection_weight, const aclTensor *output_norm_weight,
    const aclTensor *output_projection_weight, const aclTensor *conv1d_weight, aclTensor *conv_state,
    const aclTensor *checkpoint_state, aclTensor *replay_u, aclTensor *replay_k, aclTensor *replay_decay,
    const aclTensor *state_indices, const aclTensor *conv_num_accepted_tokens, const aclTensor *a_log,
    const aclTensor *dt_bias, const aclTensor *cu_seqlens, const aclTensor *plan, double scale, double lower_bound,
    double rms_norm_eps, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnMegaRecurrentKdaV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);
#ifdef __cplusplus
}
#endif
