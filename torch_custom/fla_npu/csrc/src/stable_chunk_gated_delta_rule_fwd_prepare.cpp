// Stable-ABI adapter for npu_chunk_gated_delta_rule_fwd_prepare.
// aclnn: aclnnChunkGatedDeltaRuleFwdPrepare
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/layout_math.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::CStrArg;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::enum_name;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::logical_optional_tensor;
using fla_npu_stable::stable::logical_out_tensor;
using fla_npu_stable::stable::logical_tensor;
using fla_npu_stable::stable::nd_optional_tensor;
using fla_npu_stable::stable::nd_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_chunk_gated_delta_rule_fwd_prepare
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_gated_delta_rule_fwd_prepare =
    "npu_chunk_gated_delta_rule_fwd_prepare(Tensor q, Tensor k, Tensor v, "
    "Tensor g, Tensor beta, Tensor? a_log, Tensor? dt_bias, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int chunk_size, "
    "bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel, "
    "bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, bool use_exp2, "
    "bool output_a, int stream) "
    "-> (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor, Tensor, Tensor, "
    "Tensor)";

std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>, Tensor, Tensor, Tensor, Tensor>
run_npu_chunk_gated_delta_rule_fwd_prepare(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
    std::optional<Tensor> a_log, std::optional<Tensor> dt_bias,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t chunk_size, bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, bool use_exp2,
    bool output_a, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  const int64_t batch = size_of(q_meta, 0);
  const int64_t key_heads = size_of(q_meta, 1);
  const int64_t tokens = size_of(q_meta, 2);
  const int64_t key_dim = size_of(q_meta, 3);
  const int64_t value_heads = size_of(v_meta, 1);

  // Without the in-kernel Q/K normalisation the reference hands `q`/`k` back
  // unchanged and leaves the operator's slots null; the two cases have to be
  // spelled separately because a null handle cannot be an output.
  Tensor out_q_hat = use_qk_l2norm_in_kernel
                         ? allocate_sizes(q_meta.sizes, q_meta.scalar_type,
                                          q_meta)
                         : q;
  Tensor out_k_hat = use_qk_l2norm_in_kernel
                         ? allocate_sizes(k_meta.sizes, k_meta.scalar_type,
                                          k_meta)
                         : k;
  std::optional<Tensor> out_q_rstd;
  std::optional<Tensor> out_k_rstd;
  if (use_qk_l2norm_in_kernel) {
    out_q_rstd = allocate_sizes({batch, key_heads, tokens}, kFloat, q_meta);
    out_k_rstd = allocate_sizes({batch, key_heads, tokens}, kFloat, k_meta);
  }
  std::optional<Tensor> out_beta_eff;
  if (use_beta_sigmoid_in_kernel) {
    out_beta_eff = allocate_sizes({batch, value_heads, tokens}, kFloat,
                                  meta_of(beta));
  }
  Tensor out_g_cumsum =
      allocate_sizes({batch, value_heads, tokens}, kFloat, meta_of(g));
  Tensor out_w = allocate_sizes({batch, value_heads, tokens, key_dim},
                                k_meta.scalar_type, k_meta);
  Tensor out_u = allocate_sizes(v_meta.sizes, v_meta.scalar_type, v_meta);
  Tensor out_a = allocate_sizes({batch, value_heads, tokens, chunk_size},
                                k_meta.scalar_type, k_meta);

  FLA_STABLE_EXEC(
      // ND descriptors: this operator's reference passes
      // `acl_format_override=ACL_FORMAT_ND` for every argument.
      "aclnnChunkGatedDeltaRuleFwdPrepare", q_meta, stream, nd_tensor(q_meta),
      nd_tensor(k_meta), nd_tensor(v_meta), nd_tensor(meta_of(g)),
      nd_tensor(meta_of(beta)),
      nd_optional_tensor(use_gate_in_kernel ? a_log : std::nullopt),
      nd_optional_tensor(use_gate_in_kernel ? dt_bias : std::nullopt),
      int_array(cu_seqlens), int_array(chunk_indices), scalar(chunk_size),
      scalar(allow_neg_eigval), scalar(use_exp2), scalar(output_a),
      nd_out_tensor(meta_of(out_g_cumsum)), nd_out_tensor(meta_of(out_w)),
      nd_out_tensor(meta_of(out_u)), nd_out_tensor(meta_of(out_a)),
      nd_out_tensor(use_qk_l2norm_in_kernel ? meta_of(out_q_hat)
                                            : TensorMeta()),
      nd_out_tensor(use_qk_l2norm_in_kernel ? meta_of(out_k_hat)
                                            : TensorMeta()),
      nd_out_tensor(out_q_rstd.has_value() ? meta_of(*out_q_rstd)
                                           : TensorMeta()),
      nd_out_tensor(out_k_rstd.has_value() ? meta_of(*out_k_rstd)
                                           : TensorMeta()),
      nd_out_tensor(out_beta_eff.has_value() ? meta_of(*out_beta_eff)
                                             : TensorMeta()));
  return std::make_tuple(out_q_hat, out_k_hat, out_q_rstd, out_k_rstd,
                         out_beta_eff, out_g_cumsum, out_w, out_u, out_a);
}

}  // namespace
