// Stable-ABI adapter for npu_chunk_gated_delta_rule_bwd_finalize.
// aclnn: aclnnChunkGatedDeltaRuleBwdFinalize
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
// npu_chunk_gated_delta_rule_bwd_finalize
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_gated_delta_rule_bwd_finalize =
    "npu_chunk_gated_delta_rule_bwd_finalize(Tensor q, Tensor k, Tensor v, "
    "Tensor v_new, Tensor do, Tensor du, Tensor g, Tensor beta, Tensor h, "
    "Tensor dh, Tensor a, Tensor? q_rstd, Tensor? k_rstd, Tensor? beta_raw, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size, "
    "bool use_qk_l2norm_in_kernel, bool use_beta_sigmoid_in_kernel, "
    "bool use_gate_in_kernel, bool state_v_first, bool use_exp2, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
run_npu_chunk_gated_delta_rule_bwd_finalize(
    Tensor q, Tensor k, Tensor v, Tensor v_new, Tensor d_o, Tensor du,
    Tensor g, Tensor beta, Tensor h, Tensor dh, Tensor a,
    std::optional<Tensor> q_rstd, std::optional<Tensor> k_rstd,
    std::optional<Tensor> beta_raw, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> chunk_indices, double scale, int64_t chunk_size,
    bool use_qk_l2norm_in_kernel, bool use_beta_sigmoid_in_kernel,
    bool use_gate_in_kernel, bool state_v_first, bool use_exp2,
    int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  // Each gradient follows the tensor it differentiates.
  Tensor out_dq = allocate_sizes(meta_of(q).sizes, meta_of(q).scalar_type,
                                 meta_of(q));
  Tensor out_dk = allocate_sizes(meta_of(k).sizes, meta_of(k).scalar_type,
                                 meta_of(k));
  Tensor out_dv = allocate_sizes(meta_of(v).sizes, meta_of(v).scalar_type,
                                 meta_of(v));
  Tensor out_dbeta =
      allocate_sizes(meta_of(beta).sizes, meta_of(beta).scalar_type,
                     meta_of(beta));
  Tensor out_dg = allocate_sizes(meta_of(g).sizes, meta_of(g).scalar_type,
                                 meta_of(g));

  FLA_STABLE_EXEC(
      // Logical storage shape (this reference's `logical_tensor`).
      "aclnnChunkGatedDeltaRuleBwdFinalize", q_meta, stream,
      logical_tensor(q_meta), logical_tensor(meta_of(k)),
      logical_tensor(meta_of(v)), logical_tensor(meta_of(v_new)),
      logical_tensor(meta_of(d_o)), logical_tensor(meta_of(du)),
      logical_tensor(meta_of(g)), logical_tensor(meta_of(beta)),
      logical_tensor(meta_of(h)), logical_tensor(meta_of(dh)),
      logical_tensor(meta_of(a)), logical_optional_tensor(q_rstd),
      logical_optional_tensor(k_rstd), logical_optional_tensor(beta_raw),
      int_array(cu_seqlens),
      int_array(chunk_indices), scalar(scale), scalar(chunk_size),
      scalar(use_qk_l2norm_in_kernel), scalar(use_beta_sigmoid_in_kernel),
      scalar(use_gate_in_kernel), scalar(state_v_first), scalar(use_exp2),
      logical_out_tensor(meta_of(out_dq)), logical_out_tensor(meta_of(out_dk)),
      logical_out_tensor(meta_of(out_dv)),
      logical_out_tensor(meta_of(out_dbeta)),
      logical_out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dq, out_dk, out_dv, out_dbeta, out_dg);
}

}  // namespace
