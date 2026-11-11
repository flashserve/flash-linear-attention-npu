// Stable-ABI adapter for the fused RecurrentGDN V2 contract.
// Included by stable_ops.cpp; no ATen/c10 or libtorch C++ ABI dependencies.
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>

namespace {
using torch::stable::Tensor;
using fla_npu_stable::stable::TensorArg;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::kAclFormatNd;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_out_tensor;
using fla_npu_stable::stable::scalar;

constexpr const char* kSchemaRecurrentGdr =
    "npu_recurrent_gated_delta_rule(Tensor query, Tensor key, Tensor value, "
    "Tensor(a!) state, Tensor beta, Tensor actual_seq_lengths, "
    "Tensor ssm_state_indices, Tensor? num_accepted_tokens, Tensor? g, Tensor? gk, "
    "Tensor? A_log, Tensor? dt_bias, float scale, bool use_qk_l2norm_in_kernel, "
    "bool use_gate_in_kernel, bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, "
    "int stream) -> Tensor";

Tensor run_recurrent_gated_delta_rule(
    Tensor query, Tensor key, Tensor value, Tensor state, Tensor beta,
    Tensor actual_seq_lengths, Tensor ssm_state_indices,
    std::optional<Tensor> num_accepted_tokens, std::optional<Tensor> g,
    std::optional<Tensor> gk, std::optional<Tensor> A_log, std::optional<Tensor> dt_bias,
    double scale, bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, int64_t stream) {
  const TensorMeta value_meta = meta_of(value);
  Tensor out = allocate_like(value_meta);

  // Use the actual allocation extent even for contiguous slices with an offset.
  // TensorArg's default logical_storage=false preserves the storage base plus
  // the offset exactly once; nd_tensor's logical storage is unsuitable here.
  FLA_STABLE_EXEC(
      "aclnnRecurrentGatedDeltaRuleV2", value_meta, stream,
      TensorArg(meta_of(query), kAclFormatNd),
      TensorArg(meta_of(key), kAclFormatNd),
      TensorArg(value_meta, kAclFormatNd),
      TensorArg(meta_of(beta), kAclFormatNd),
      TensorArg(meta_of(state), kAclFormatNd),
      TensorArg(meta_of(actual_seq_lengths), kAclFormatNd),
      TensorArg(meta_of(ssm_state_indices), kAclFormatNd),
      TensorArg(g ? meta_of(*g) : TensorMeta(), kAclFormatNd),
      TensorArg(gk ? meta_of(*gk) : TensorMeta(), kAclFormatNd),
      TensorArg(num_accepted_tokens ? meta_of(*num_accepted_tokens) : TensorMeta(), kAclFormatNd),
      TensorArg(A_log ? meta_of(*A_log) : TensorMeta(), kAclFormatNd),
      TensorArg(dt_bias ? meta_of(*dt_bias) : TensorMeta(), kAclFormatNd),
      scalar(static_cast<float>(scale)), scalar(use_qk_l2norm_in_kernel),
      scalar(use_gate_in_kernel), scalar(use_beta_sigmoid_in_kernel),
      scalar(allow_neg_eigval), nd_out_tensor(meta_of(out)));
  return out;
}
}  // namespace
