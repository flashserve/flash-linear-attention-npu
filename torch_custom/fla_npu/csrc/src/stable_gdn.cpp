// Stable-ABI adapters: npu_chunk_fwd_o and npu_chunk_gdn_bwd_intra.
//
// Both allocate several outputs whose shapes come from more than one input,
// which is the only thing that separates them from the batch in
// stable_chunk.cpp:
//
//   * chunk_fwd_o's output shape depends on the layout string, so the enum name
//     is resolved once and used both for the allocation and for aclnn.
//   * chunk_gdn_bwd_intra mixes an output shaped from q and v with two that
//     copy v.
//
// Included by stable_ops.cpp (single TU); registration lives there.

// Owns the gated-delta-rule composites and their pieces:
// npu_chunk_gated_delta_rule_fwd/_bwd/_bwd_finalize/_fwd_prepare,
// npu_chunk_fwd_o, npu_chunk_gdn_bwd_intra.  The h/dh recurrence of the
// same family lives in stable_fwd_h.cpp.

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

namespace layout_math = fla_npu_stable::stable::layout_math;

// Table order is the code order; _stable.py's _char_code tables must match.
// Every layout name table in this file uses the code order
// stable/layout_math.h documents (BSND, BNSD, TND, NTD); the Python side
// carries the same table in _stable.py's _ENUM.
constexpr const char* kChunkFwdOOutputLayoutNames[] = {"BSND", "BNSD", "TND",
                                                       "NTD"};

// ---------------------------------------------------------------------------
// npu_chunk_fwd_o
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_fwd_o =
    "npu_chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, Tensor? g, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size, "
    "bool use_exp2, bool transpose_state_layout, int output_layout, "
    "int stream) -> Tensor";

Tensor run_npu_chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h,
                           std::optional<Tensor> g,
                           std::optional<Tensor> cu_seqlens,
                           std::optional<Tensor> chunk_indices, double scale,
                           int64_t chunk_size, bool use_exp2,
                           bool transpose_state_layout, int64_t output_layout,
                           int64_t stream) {
  const TensorMeta v_meta = meta_of(v);
  const char* layout = enum_name(kChunkFwdOOutputLayoutNames, output_layout);
  const int32_t dtype = v_meta.scalar_type;

  Tensor out;
  if (std::strcmp(layout, "BNSD") == 0) {
    out = allocate_sizes({size_of(v_meta, 0), size_of(v_meta, 1),
                          size_of(v_meta, 2), size_of(v_meta, 3)},
                         dtype, v_meta);
  } else if (std::strcmp(layout, "BSND") == 0) {
    out = allocate_sizes({size_of(v_meta, 0), size_of(v_meta, 2),
                          size_of(v_meta, 1), size_of(v_meta, 3)},
                         dtype, v_meta);
  } else if (std::strcmp(layout, "TND") == 0) {
    out = allocate_sizes({size_of(v_meta, 2), size_of(v_meta, 1),
                          size_of(v_meta, 3)},
                         dtype, v_meta);
  } else {
    out = allocate_sizes({size_of(v_meta, 1), size_of(v_meta, 2),
                          size_of(v_meta, 3)},
                         dtype, v_meta);
  }

  FLA_STABLE_EXEC("aclnnChunkFwdO", meta_of(v), stream, tensor(meta_of(q)),
                  tensor(meta_of(k)), tensor(v_meta), tensor(meta_of(h)),
                  optional_tensor(g), int_array(cu_seqlens),
                  int_array(chunk_indices), scalar(scale), scalar(chunk_size),
                  scalar(use_exp2), scalar(transpose_state_layout),
                  CStrArg(layout), out_tensor(meta_of(out)));
  return out;
}

// ---------------------------------------------------------------------------
// npu_chunk_gdn_bwd_intra
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_gdn_bwd_intra =
    "npu_chunk_gdn_bwd_intra(Tensor q, Tensor k, Tensor v, Tensor g, "
    "Tensor beta, Tensor A, Tensor d_o, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, float scale, int chunk_size, bool use_exp2, "
    "int stream) -> (Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor> run_npu_chunk_gdn_bwd_intra(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor A, Tensor d_o,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    double scale, int64_t chunk_size, bool use_exp2, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  Tensor out_dq = allocate_sizes(
      {size_of(q_meta, 0), size_of(v_meta, 1), size_of(q_meta, 2),
       size_of(q_meta, 3)},
      q_meta.scalar_type, q_meta);
  Tensor out_dk = allocate_like(v_meta);
  Tensor out_dv = allocate_like(v_meta);

  // ND again: this reference passes `acl_format_override=ACL_FORMAT_ND` ("BNSD
  // tensors are already contiguous; expose that physical shape to tiling").
  FLA_STABLE_EXEC("aclnnChunkGdnBwdIntra", q_meta, stream, nd_tensor(q_meta),
                  nd_tensor(meta_of(k)), nd_tensor(v_meta),
                  nd_tensor(meta_of(g)), nd_tensor(meta_of(beta)),
                  nd_tensor(meta_of(A)), nd_tensor(meta_of(d_o)),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  scalar(scale), scalar(chunk_size), scalar(use_exp2),
                  nd_out_tensor(meta_of(out_dq)),
                  nd_out_tensor(meta_of(out_dk)),
                  nd_out_tensor(meta_of(out_dv)));
  return std::make_tuple(out_dq, out_dk, out_dv);
}

// ---------------------------------------------------------------------------
// npu_chunk_gated_delta_rule_fwd
// ---------------------------------------------------------------------------

constexpr const char* kGdnFwdLayoutNames[] = {"BSND", "BNSD", "TND", "NTD"};

// The operator materializes its optional outputs instead of taking flags:
// `a_log` is the gate output and `beta_eff` the sigmoid'd beta, so both are
// allocated here from the caller's flags and passed as trailing inputs.  The
// declared return follows the reference's *public* tuple, which grows with
// those flags, rather than the fixed aclnn slot list.
constexpr const char* kSchema_chunk_gated_delta_rule_fwd =
    "npu_chunk_gated_delta_rule_fwd(Tensor q, Tensor k, Tensor v, Tensor g, "
    "Tensor beta, Tensor? initial_state, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, int layout, float scale, int chunk_size, "
    "bool use_exp2, bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel, "
    "bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, "
    "bool disable_recompute, bool output_final_state, "
    "bool return_intermediate_states, bool state_v_first, int stream) "
    "-> (Tensor, Tensor?, Tensor?, Tensor?, Tensor?)";

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>>
run_npu_chunk_gated_delta_rule_fwd(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
    std::optional<Tensor> initial_state,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t layout, double scale, int64_t chunk_size, bool use_exp2,
    bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval,
    bool disable_recompute, bool output_final_state,
    bool return_intermediate_states, bool state_v_first, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  // This operator accepts the TND/NTD *names* but always reads a rank-4
  // tensor: for TND the token axis is dim 1 and the heads are dim 2, exactly
  // like BSND.  Using the packed (rank-3) helpers here produced o/A/g_cumsum/
  // final_state with the wrong shapes, and the Ascend950 tiling rejected the
  // call with 161002 (the shapes were visible in a descriptor dump: e.g. A came
  // out as [1, 128, 1, 64] instead of [1, 4, 128, 64]).
  const int64_t batch = size_of(q_meta, 0);
  const int64_t tokens = layout_math::tokens4(q_meta, layout);
  const int64_t heads = layout_math::value_heads4(v_meta, layout);
  const int64_t k_dim = size_of(q_meta, 3);
  const int64_t v_dim = size_of(v_meta, 3);
  const int64_t state_tail_k = state_v_first ? v_dim : k_dim;
  const int64_t state_tail_v = state_v_first ? k_dim : v_dim;

  // `o` copies v's dtype, the states follow q -- same as the reference.
  Tensor out_o = allocate_sizes({batch, tokens, heads, v_dim},
                                v_meta.scalar_type, v_meta);
  std::optional<Tensor> out_final_state;
  if (output_final_state) {
    // The state follows the caller's dtype when one was supplied.
    const int64_t state_dtype = initial_state.has_value()
                                    ? meta_of(*initial_state).scalar_type
                                    : kFloat;
    out_final_state = allocate_sizes(
        {layout_math::sequences(cu, batch), heads, state_tail_k, state_tail_v},
        state_dtype, q_meta);
  }
  std::optional<Tensor> out_a_log;
  if (use_gate_in_kernel) {
    out_a_log = allocate_sizes({heads}, kFloat, meta_of(g));
  }
  std::optional<Tensor> out_beta_eff;
  if (use_beta_sigmoid_in_kernel) {
    out_beta_eff =
        allocate_sizes({batch, tokens, heads}, kFloat, meta_of(beta));
  }
  std::optional<Tensor> out_g_cumsum;
  std::optional<Tensor> out_a;
  if (disable_recompute) {
    out_g_cumsum = allocate_sizes({batch, tokens, heads}, kFloat, meta_of(g));
    out_a = allocate_sizes({batch, heads, tokens, chunk_size},
                           q_meta.scalar_type, q_meta);
  }
  std::optional<Tensor> out_h;
  if (return_intermediate_states) {
    out_h = allocate_sizes({batch, heads,
                            layout_math::chunks(cu, ci, chunk_size, tokens),
                            state_tail_k, state_tail_v},
                           q_meta.scalar_type, q_meta);
  }

  FLA_STABLE_EXEC(
      "aclnnChunkGatedDeltaRuleFwd", q_meta, stream, tensor(q_meta),
      tensor(meta_of(k)), tensor(v_meta), tensor(meta_of(g)),
      tensor(meta_of(beta)),
      out_tensor(out_a_log.has_value() ? meta_of(*out_a_log) : TensorMeta()),
      /*dt_bias=*/optional_tensor(std::nullopt),
      optional_tensor(initial_state), int_array(cu), int_array(ci),
      cstr(kGdnFwdLayoutNames, layout), scalar(scale), scalar(chunk_size),
      scalar(use_exp2), scalar(use_qk_l2norm_in_kernel),
      scalar(allow_neg_eigval), scalar(state_v_first),
      out_tensor(meta_of(out_o)),
      out_tensor(out_final_state.has_value() ? meta_of(*out_final_state)
                                             : TensorMeta()),
      /*q_hat=*/out_tensor(TensorMeta()),
      /*k_hat=*/out_tensor(TensorMeta()),
      /*q_rstd=*/out_tensor(TensorMeta()),
      /*k_rstd=*/out_tensor(TensorMeta()),
      out_tensor(out_beta_eff.has_value() ? meta_of(*out_beta_eff)
                                          : TensorMeta()),
      out_tensor(out_g_cumsum.has_value() ? meta_of(*out_g_cumsum)
                                          : TensorMeta()),
      out_tensor(out_a.has_value() ? meta_of(*out_a) : TensorMeta()),
      out_tensor(out_h.has_value() ? meta_of(*out_h) : TensorMeta()));
  return std::make_tuple(out_o, out_final_state, out_g_cumsum, out_a, out_h);
}

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
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  const int64_t batch = size_of(q_meta, 0);
  const int64_t tokens = layout_math::tokens4(q_meta, layout);
  const int64_t heads = layout_math::value_heads4(v_meta, layout);

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
      int_array(chunk_indices), cstr(kGdnFwdLayoutNames, layout),
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
