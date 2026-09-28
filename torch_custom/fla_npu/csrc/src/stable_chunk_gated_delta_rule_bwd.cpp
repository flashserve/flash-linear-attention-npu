// Stable-ABI adapter for npu_chunk_gated_delta_rule_bwd.
// aclnn: aclnnChunkGatedDeltaRuleBwd
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
using fla_npu_stable::stable::cstr;
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

constexpr const char* kGdnBwdLayoutNames[] = {"BSND", "BNSD", "TND", "NTD"};

// ---------------------------------------------------------------------------
// npu_chunk_gated_delta_rule_bwd
// ---------------------------------------------------------------------------

// The composite backward: one aclnn call for the whole graph.  `d_a_log` and
// `d_dt_bias` are reserved slots the operator does not fill yet, so they stay
// null and the public tuple returns None for them.
constexpr const char* kSchema_chunk_gated_delta_rule_bwd =
    "npu_chunk_gated_delta_rule_bwd(Tensor q, Tensor k, Tensor v, Tensor g, "
    "Tensor beta, Tensor A, Tensor d_o, Tensor? initial_state, Tensor? dht, "
    "Tensor? q_rstd, Tensor? k_rstd, Tensor? beta_raw, Tensor? a_log, "
    "Tensor? dt_bias, Tensor? cu_seqlens, Tensor? chunk_indices, int layout, "
    "float scale, int chunk_size, bool use_exp2, bool use_gate_in_kernel, "
    "bool use_qk_l2norm_in_kernel, bool use_beta_sigmoid_in_kernel, "
    "bool state_v_first, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor?, Tensor?)";

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>>
run_npu_chunk_gated_delta_rule_bwd(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor A, Tensor d_o,
    std::optional<Tensor> initial_state, std::optional<Tensor> dht,
    std::optional<Tensor> q_rstd, std::optional<Tensor> k_rstd,
    std::optional<Tensor> beta_raw, std::optional<Tensor> a_log,
    std::optional<Tensor> dt_bias, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> chunk_indices, int64_t layout, double scale,
    int64_t chunk_size, bool use_exp2, bool use_gate_in_kernel,
    bool use_qk_l2norm_in_kernel, bool use_beta_sigmoid_in_kernel,
    bool state_v_first, int64_t stream) {
  namespace layout_math = fla_npu_stable::stable::layout_math;
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  const int64_t batch = SIZE_OF(q_meta, 0);
  const int64_t tokens = SIZE_OF(q_meta, layout_math::token_axis4(layout));
  const int64_t heads = SIZE_OF(v_meta, layout_math::head_axis4(layout));

  Tensor out_dq = allocate_sizes(q_meta.sizes, q_meta.scalar_type, q_meta);
  Tensor out_dk = allocate_sizes(meta_of(k).sizes, meta_of(k).scalar_type,
                                 meta_of(k));
  Tensor out_dv = allocate_sizes(v_meta.sizes, v_meta.scalar_type, v_meta);
  Tensor out_d_beta =
      allocate_sizes({batch, tokens, heads}, meta_of(beta).scalar_type,
                     meta_of(beta));
  Tensor out_d_g = allocate_sizes({batch, tokens, heads},
                                  meta_of(g).scalar_type, meta_of(g));
  std::optional<Tensor> out_dh0;
  if (initial_state.has_value()) {
    out_dh0 = allocate_sizes(meta_of(*initial_state).sizes,
                             meta_of(*initial_state).scalar_type,
                             meta_of(*initial_state));
  }

  FLA_STABLE_EXEC(
      // Logical storage shape (this reference's `logical_tensor`).
      "aclnnChunkGatedDeltaRuleBwd", q_meta, stream, logical_tensor(q_meta),
      logical_tensor(meta_of(k)), logical_tensor(v_meta),
      logical_tensor(meta_of(g)), logical_tensor(meta_of(beta)),
      logical_tensor(meta_of(A)), logical_tensor(meta_of(d_o)),
      logical_optional_tensor(initial_state), logical_optional_tensor(dht),
      logical_optional_tensor(q_rstd), logical_optional_tensor(k_rstd),
      logical_optional_tensor(beta_raw), logical_optional_tensor(a_log),
      logical_optional_tensor(dt_bias), int_array(cu_seqlens),
      int_array(chunk_indices), cstr(kGdnBwdLayoutNames, layout),
      scalar(scale), scalar(chunk_size), scalar(use_exp2),
      scalar(use_gate_in_kernel), scalar(use_qk_l2norm_in_kernel),
      scalar(use_beta_sigmoid_in_kernel), scalar(state_v_first),
      logical_out_tensor(meta_of(out_dq)), logical_out_tensor(meta_of(out_dk)),
      logical_out_tensor(meta_of(out_dv)),
      logical_out_tensor(meta_of(out_d_beta)),
      logical_out_tensor(meta_of(out_d_g)),
      logical_out_tensor(out_dh0.has_value() ? meta_of(*out_dh0)
                                             : TensorMeta()),
      /*d_a_log=*/logical_out_tensor(TensorMeta()),
      /*d_dt_bias=*/logical_out_tensor(TensorMeta()));
  return std::make_tuple(out_dq, out_dk, out_dv, out_d_beta, out_d_g, out_dh0,
                         std::nullopt, std::nullopt);
}

}  // namespace
