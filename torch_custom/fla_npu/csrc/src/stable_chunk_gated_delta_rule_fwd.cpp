// Stable-ABI adapter for npu_chunk_gated_delta_rule_fwd.
// aclnn: aclnnChunkGatedDeltaRuleFwd
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

// ---------------------------------------------------------------------------
// npu_chunk_gated_delta_rule_fwd
// ---------------------------------------------------------------------------

constexpr const char* kGdnFwdLayoutNames[] = {"BSND", "BNSD", "TND", "NTD"};

// The declared return is the reference's *public* tuple, which is fixed at ten
// slots rather than growing with the flags: `(o, final_state, g_cumsum, A,
// beta_eff, h, q_hat, k_hat, q_rstd, k_rstd)`, with None in the slots a flag
// switched off.  `q_hat`/`k_hat` alias the caller's own q/k when the kernel is
// not asked to normalise them, and both rstd slots are null in that case --
// exactly what the reference hands back.  Gate-in-kernel is still unsupported,
// so `a_log`/`dt_bias` are always null here; the wrapper refuses them, as the
// reference does, before the launch.
constexpr const char* kSchema_chunk_gated_delta_rule_fwd =
    "npu_chunk_gated_delta_rule_fwd(Tensor q, Tensor k, Tensor v, Tensor g, "
    "Tensor beta, Tensor? initial_state, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, int layout, float scale, int chunk_size, "
    "bool use_exp2, bool use_qk_l2norm_in_kernel, "
    "bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, "
    "bool disable_recompute, bool output_final_state, "
    "bool return_intermediate_states, bool state_v_first, int stream) "
    "-> (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor, Tensor, "
    "Tensor?, Tensor?)";

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>, Tensor, Tensor, std::optional<Tensor>,
           std::optional<Tensor>>
run_npu_chunk_gated_delta_rule_fwd(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
    std::optional<Tensor> initial_state,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t layout, double scale, int64_t chunk_size, bool use_exp2,
    bool use_qk_l2norm_in_kernel, bool use_beta_sigmoid_in_kernel,
    bool allow_neg_eigval, bool disable_recompute, bool output_final_state,
    bool return_intermediate_states, bool state_v_first, int64_t stream) {
  namespace layout_math = fla_npu_stable::stable::layout_math;
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  // This operator accepts the TND/NTD *names* but always reads a rank-4
  // tensor: for TND the token axis is dim 1 and the heads are dim 2, exactly
  // like BSND.  Using the packed (rank-3) helpers here produced o/A/g_cumsum/
  // final_state with the wrong shapes, and the Ascend950 tiling rejected the
  // call with 161002 (the shapes were visible in a descriptor dump: e.g. A came
  // out as [1, 128, 1, 64] instead of [1, 4, 128, 64]).
  const int64_t batch = SIZE_OF(q_meta, 0);
  const int64_t tokens = SIZE_OF(q_meta, layout_math::token_axis4(layout));
  const int64_t heads = SIZE_OF(v_meta, layout_math::head_axis4(layout));
  // Same rank-4 convention as the token axis above, applied to q: HK is dim 2
  // for the sequence-major names and dim 1 for the others.
  const int64_t key_heads = SIZE_OF(q_meta, layout_math::head_axis4(layout));
  const int64_t k_dim = SIZE_OF(q_meta, 3);
  const int64_t v_dim = SIZE_OF(v_meta, 3);
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
    const std::vector<int64_t> h_shape = {
        batch, layout_math::chunks(cu, ci, chunk_size, tokens), heads,
        state_tail_k, state_tail_v};
    out_h = allocate_sizes(h_shape, q_meta.scalar_type, q_meta);
  }
  // The Q/K normalisation results are part of the public tuple now.  The
  // reference allocates them only when the kernel does the normalisation;
  // otherwise the hats *are* the caller's q/k and both rstd slots stay null.
  const Tensor out_q_hat =
      use_qk_l2norm_in_kernel
          ? allocate_sizes(q_meta.sizes, q_meta.scalar_type, q_meta)
          : q;
  const Tensor out_k_hat =
      use_qk_l2norm_in_kernel
          ? allocate_sizes(k_meta.sizes, k_meta.scalar_type, k_meta)
          : k;
  std::optional<Tensor> out_q_rstd;
  std::optional<Tensor> out_k_rstd;
  if (use_qk_l2norm_in_kernel) {
    // BNS regardless of the input layout -- the kernel enforces this shape.
    out_q_rstd = allocate_sizes({batch, key_heads, tokens}, kFloat, q_meta);
    out_k_rstd = allocate_sizes({batch, key_heads, tokens}, kFloat, q_meta);
  }

  FLA_STABLE_EXEC(
      "aclnnChunkGatedDeltaRuleFwd", q_meta, stream, tensor(q_meta),
      tensor(k_meta), tensor(v_meta), tensor(meta_of(g)),
      tensor(meta_of(beta)),
      /*a_log=*/optional_tensor(std::nullopt),
      /*dt_bias=*/optional_tensor(std::nullopt),
      optional_tensor(initial_state), int_array(cu), int_array(ci),
      cstr(kGdnFwdLayoutNames, layout), scalar(scale), scalar(chunk_size),
      scalar(use_exp2), scalar(use_qk_l2norm_in_kernel),
      scalar(allow_neg_eigval), scalar(state_v_first),
      out_tensor(meta_of(out_o)),
      out_tensor(out_final_state.has_value() ? meta_of(*out_final_state)
                                             : TensorMeta()),
      out_tensor(use_qk_l2norm_in_kernel ? meta_of(out_q_hat) : TensorMeta()),
      out_tensor(use_qk_l2norm_in_kernel ? meta_of(out_k_hat) : TensorMeta()),
      out_tensor(out_q_rstd.has_value() ? meta_of(*out_q_rstd) : TensorMeta()),
      out_tensor(out_k_rstd.has_value() ? meta_of(*out_k_rstd) : TensorMeta()),
      out_tensor(out_beta_eff.has_value() ? meta_of(*out_beta_eff)
                                          : TensorMeta()),
      out_tensor(out_g_cumsum.has_value() ? meta_of(*out_g_cumsum)
                                          : TensorMeta()),
      out_tensor(out_a.has_value() ? meta_of(*out_a) : TensorMeta()),
      out_tensor(out_h.has_value() ? meta_of(*out_h) : TensorMeta()));
  return std::make_tuple(out_o, out_final_state, out_g_cumsum, out_a,
                         out_beta_eff, out_h, out_q_hat, out_k_hat, out_q_rstd,
                         out_k_rstd);
}

}  // namespace
