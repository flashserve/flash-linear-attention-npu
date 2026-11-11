// Stable-ABI adapter for the fused RecurrentGDN V2 contract.
// Included by stable_ops.cpp; no ATen/c10 or libtorch C++ ABI dependencies.
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

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
  const TensorMeta query_meta = meta_of(query);
  const int32_t npu_device_type = aoti_torch_device_type_privateuse1();
  // ACL descriptors retain storage metadata but not the torch device. Reject
  // host pointers and cross-device inputs before allocating output or creating
  // those descriptors; checking metadata never synchronizes tensor contents.
  const auto check_device = [&](const TensorMeta& meta, const char* name) {
    if (!meta.defined) {
      return;
    }
    constexpr const char* prefix = "fla_npu(stable): npu_recurrent_gated_delta_rule: ";
    if (meta.device_type != npu_device_type) {
      throw std::runtime_error(
          std::string(prefix) + name + " must be on NPU (got device_type=" +
          std::to_string(meta.device_type) + ", device_index=" +
          std::to_string(meta.device_index) + ")");
    }
    if (meta.device_index != query_meta.device_index) {
      throw std::runtime_error(
          std::string(prefix) + name + " must be on the same device as query (expected npu:" +
          std::to_string(query_meta.device_index) + ", got npu:" +
          std::to_string(meta.device_index) + ")");
    }
  };
  check_device(query_meta, "query");
  const auto checked = [&](TensorMeta meta, const char* name) {
    check_device(meta, name);
    return meta;
  };
  const TensorMeta key_meta = checked(meta_of(key), "key");
  const TensorMeta value_meta = checked(meta_of(value), "value");
  const TensorMeta state_meta = checked(meta_of(state), "state");
  const TensorMeta beta_meta = checked(meta_of(beta), "beta");
  const TensorMeta lengths_meta = checked(meta_of(actual_seq_lengths), "actual_seq_lengths");
  const TensorMeta indices_meta = checked(meta_of(ssm_state_indices), "ssm_state_indices");
  const TensorMeta accepted_meta = checked(
      num_accepted_tokens ? meta_of(*num_accepted_tokens) : TensorMeta(), "num_accepted_tokens");
  const TensorMeta g_meta = checked(g ? meta_of(*g) : TensorMeta(), "g");
  const TensorMeta gk_meta = checked(gk ? meta_of(*gk) : TensorMeta(), "gk");
  const TensorMeta a_log_meta = checked(A_log ? meta_of(*A_log) : TensorMeta(), "A_log");
  const TensorMeta bias_meta = checked(dt_bias ? meta_of(*dt_bias) : TensorMeta(), "dt_bias");
  Tensor out = allocate_like(value_meta);

  // Use the actual allocation extent even for contiguous slices with an offset.
  // TensorArg's default logical_storage=false preserves the storage base plus
  // the offset exactly once; nd_tensor's logical storage is unsuitable here.
  FLA_STABLE_EXEC(
      "aclnnRecurrentGatedDeltaRuleV2", value_meta, stream,
      TensorArg(query_meta, kAclFormatNd),
      TensorArg(key_meta, kAclFormatNd),
      TensorArg(value_meta, kAclFormatNd),
      TensorArg(beta_meta, kAclFormatNd),
      TensorArg(state_meta, kAclFormatNd),
      TensorArg(lengths_meta, kAclFormatNd),
      TensorArg(indices_meta, kAclFormatNd),
      TensorArg(g_meta, kAclFormatNd),
      TensorArg(gk_meta, kAclFormatNd),
      TensorArg(accepted_meta, kAclFormatNd),
      TensorArg(a_log_meta, kAclFormatNd),
      TensorArg(bias_meta, kAclFormatNd),
      scalar(static_cast<float>(scale)), scalar(use_qk_l2norm_in_kernel),
      scalar(use_gate_in_kernel), scalar(use_beta_sigmoid_in_kernel),
      scalar(allow_neg_eigval), nd_out_tensor(meta_of(out)));
  return out;
}
}  // namespace
