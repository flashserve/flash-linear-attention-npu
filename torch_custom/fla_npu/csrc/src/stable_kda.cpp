// Stable-ABI adapters: npu_kda_gate_cumsum, npu_chunk_kda_bwd_intra.
//
// These two cover the shapes a KDA-family operator runs into:
//
//   * kda_gate_cumsum -- optional tensors, an int_array, bool/double scalars,
//     and an output whose dtype differs from its source (fp32 over `g`).
//   * chunk_kda_bwd_intra -- ten tensor inputs, two int_arrays, a `char*`
//     enum argument carried as an int code plus a name table, and four outputs
//     that are allocated from their matching inputs.
//
// Included by stable_ops.cpp (single TU); registration lives there.

// Owns the KDA family: npu_chunk_kda_fwd, npu_chunk_kda_bwd_intra and
// npu_kda_gate_cumsum.  npu_chunk_kda_bwd and _bwd_recompute have no kernel on
// this branch, so no adapter for them is carried here.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/layout_math.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kBFloat16;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_optional_tensor;
using fla_npu_stable::stable::nd_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// The layout arithmetic lives in a named namespace so its helpers cannot
// collide with the adapter-local ones; alias it here because these files sit in
// an anonymous namespace at global scope.
namespace layout_math = fla_npu_stable::stable::layout_math;


// ---------------------------------------------------------------------------
// npu_kda_gate_cumsum
// ---------------------------------------------------------------------------

constexpr const char* kSchema_kda_gate_cumsum =
    "npu_kda_gate_cumsum(Tensor g, Tensor? A_log, Tensor? dt_bias, "
    "Tensor? cu_seqlens, int chunk_size, bool use_gate_in_kernel, "
    "bool safe_gate, float lower_bound, int stream) -> Tensor";

Tensor run_npu_kda_gate_cumsum(Tensor g, std::optional<Tensor> A_log,
                               std::optional<Tensor> dt_bias,
                               std::optional<Tensor> cu_seqlens,
                               int64_t chunk_size, bool use_gate_in_kernel,
                               bool safe_gate, double lower_bound,
                               int64_t stream) {
  const TensorMeta g_meta = meta_of(g);
  // The kernel accumulates in fp32 regardless of `g`'s dtype.
  Tensor out = allocate_sizes(g_meta.sizes, kFloat, g_meta);
  FLA_STABLE_EXEC("aclnnKdaGateCumsum", g_meta, stream, tensor(g_meta),
                  optional_tensor(A_log), optional_tensor(dt_bias),
                  int_array(cu_seqlens), scalar(chunk_size),
                  scalar(use_gate_in_kernel), scalar(safe_gate),
                  scalar(lower_bound), out_tensor(meta_of(out)));
  return out;
}

// ---------------------------------------------------------------------------
// npu_chunk_kda_bwd_intra
// ---------------------------------------------------------------------------

// The aclnn entry point takes `layout` as a string; the stable value
// conversions cannot carry one, so the caller passes a code and this table is
// the single source of the legal values.  The order must match the Python
// _char_code table -- tools/op_abi_parity.py checks exactly that.
// The kernel takes the layout as a string.  TND is the packed varlen spelling;
// NTD is not part of this operator's domain (the reference rejects it), so it
// has no code.
constexpr const char* kChunkKdaBwdIntraLayoutNames[] = {"BSND", "BNSD", "TND"};

constexpr const char* kSchema_chunk_kda_bwd_intra =
    "npu_chunk_kda_bwd_intra(Tensor q, Tensor k, Tensor gk, Tensor beta, "
    "Tensor dAqk, Tensor dAkk, Tensor dq, Tensor dk, Tensor db, Tensor dg, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int chunk_size, bool safe_gate, "
    "int layout, int stream) -> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_chunk_kda_bwd_intra(
    Tensor q, Tensor k, Tensor gk, Tensor beta, Tensor dAqk, Tensor dAkk,
    Tensor dq, Tensor dk, Tensor db, Tensor dg,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t chunk_size, bool safe_gate, int64_t layout, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  // Each gradient output has the shape and dtype of its own input.
  Tensor out_dq = allocate_like(meta_of(dq));
  Tensor out_dk = allocate_like(meta_of(dk));
  Tensor out_db = allocate_like(meta_of(db));
  Tensor out_dg = allocate_like(meta_of(dg));
  FLA_STABLE_EXEC(
      // ND descriptors, like this operator's reference (it passes
      // `acl_format_override=ACL_FORMAT_ND` for every argument).
      "aclnnChunkKdaBwdIntra", q_meta, stream,
      nd_tensor(meta_of(q)), nd_tensor(meta_of(k)), nd_tensor(meta_of(gk)),
      nd_tensor(meta_of(beta)), nd_tensor(meta_of(dAqk)),
      nd_tensor(meta_of(dAkk)), nd_tensor(meta_of(dq)), nd_tensor(meta_of(dk)),
      nd_tensor(meta_of(db)), nd_tensor(meta_of(dg)),
      int_array(cu_seqlens), int_array(chunk_indices), scalar(chunk_size),
      scalar(safe_gate), cstr(kChunkKdaBwdIntraLayoutNames, layout),
      nd_out_tensor(meta_of(out_dq)), nd_out_tensor(meta_of(out_dk)),
      nd_out_tensor(meta_of(out_db)), nd_out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dq, out_dk, out_db, out_dg);
}

// ---------------------------------------------------------------------------
// npu_chunk_kda_fwd
// ---------------------------------------------------------------------------

constexpr const char* kChunkKdaFwdLayoutNames[] = {"BSND", "BNSD", "TND",
                                                   "NTD"};

constexpr const char* kSchema_chunk_kda_fwd =
    "npu_chunk_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
    "Tensor? A_log, Tensor? dt_bias, Tensor? initial_state, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int layout, float scale, "
    "int chunk_size, bool safe_gate, float lower_bound, "
    "bool use_gate_in_kernel, bool state_v_first, bool output_final_state, "
    "bool disable_recompute, bool return_intermediate_states, int stream) "
    "-> (Tensor, Tensor?, Tensor?, Tensor, Tensor, Tensor?, Tensor?, Tensor?, "
    "Tensor?, Tensor?, Tensor?)";

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>, Tensor, Tensor,
           std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>>
run_npu_chunk_kda_fwd(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
    std::optional<Tensor> A_log, std::optional<Tensor> dt_bias,
    std::optional<Tensor> initial_state,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t layout, double scale, int64_t chunk_size, bool safe_gate,
    double lower_bound, bool use_gate_in_kernel, bool state_v_first,
    bool output_final_state, bool disable_recompute,
    bool return_intermediate_states, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  const int64_t tokens = layout_math::tokens(q_meta, layout);
  const int64_t heads = layout_math::value_heads(v_meta, layout);
  const int64_t k_dim = layout_math::key_dim(q_meta, layout);
  const int64_t v_dim = layout_math::value_dim(v_meta, layout);
  const bool rank3 = layout_math::packed(layout);
  const int64_t batch_size = layout_math::batch(q_meta, layout);

  // The head-major spellings put the batch dimension in front of the chunk
  // count; the packed ones do not have one.
  std::vector<int64_t> leading;
  if (!rank3) {
    leading.push_back(batch_size);
  }
  const int64_t q_dtype = q_meta.scalar_type;

  auto head_sizes = [&](int64_t channel_dim,
                        std::vector<int64_t> prefix) {
    prefix.insert(prefix.end(), {heads, tokens, channel_dim});
    return prefix;
  };

  Tensor out_attn = allocate_sizes(
      rank3 ? std::vector<int64_t>{tokens, heads, v_dim}
            : std::vector<int64_t>{batch_size, tokens, heads, v_dim},
      q_dtype, q_meta);
  std::optional<Tensor> out_final_state;
  if (output_final_state) {
    out_final_state = allocate_sizes(
        {layout_math::sequences(cu, batch_size), heads,
         state_v_first ? v_dim : k_dim, state_v_first ? k_dim : v_dim},
        kFloat, q_meta);
  }
  std::optional<Tensor> out_gk;
  if (!use_gate_in_kernel || disable_recompute) {
    out_gk = allocate_sizes(head_sizes(k_dim, leading), kFloat, q_meta);
  }
  Tensor out_aqk = allocate_sizes(head_sizes(chunk_size, leading), q_dtype,
                                  q_meta);
  Tensor out_akk = allocate_sizes(head_sizes(chunk_size, leading), q_dtype,
                                  q_meta);
  std::optional<Tensor> out_w;
  std::optional<Tensor> out_u;
  std::optional<Tensor> out_qg;
  std::optional<Tensor> out_kg;
  std::optional<Tensor> out_v_new;
  if (disable_recompute) {
    out_w = allocate_sizes(head_sizes(k_dim, leading), q_dtype, q_meta);
    out_u = allocate_sizes(head_sizes(v_dim, leading), q_dtype, q_meta);
    out_qg = allocate_sizes(head_sizes(k_dim, leading), q_dtype, q_meta);
    out_kg = allocate_sizes(head_sizes(k_dim, leading), q_dtype, q_meta);
    out_v_new = allocate_sizes(head_sizes(v_dim, leading), q_dtype, q_meta);
  }
  std::optional<Tensor> out_h;
  if (disable_recompute || return_intermediate_states) {
    std::vector<int64_t> h_sizes = leading;
    h_sizes.insert(h_sizes.end(),
                   {layout_math::chunks(cu, ci, chunk_size, tokens), heads,
                    state_v_first ? v_dim : k_dim,
                    state_v_first ? k_dim : v_dim});
    out_h = allocate_sizes(h_sizes, q_dtype, q_meta);
  }

  FLA_STABLE_EXEC(
      "aclnnChunkKdaFwd", q_meta, stream, tensor(q_meta), tensor(meta_of(k)),
      tensor(v_meta), tensor(meta_of(g)), tensor(meta_of(beta)),
      optional_tensor(A_log), optional_tensor(dt_bias),
      optional_tensor(initial_state), int_array(cu), int_array(ci),
      cstr(kChunkKdaFwdLayoutNames, layout), scalar(scale),
      scalar(chunk_size), scalar(safe_gate), scalar(lower_bound),
      scalar(use_gate_in_kernel), scalar(state_v_first),
      out_tensor(meta_of(out_attn)),
      out_tensor(out_final_state.has_value() ? meta_of(*out_final_state)
                                             : TensorMeta()),
      out_tensor(out_gk.has_value() ? meta_of(*out_gk) : TensorMeta()),
      out_tensor(meta_of(out_aqk)), out_tensor(meta_of(out_akk)),
      out_tensor(out_w.has_value() ? meta_of(*out_w) : TensorMeta()),
      out_tensor(out_u.has_value() ? meta_of(*out_u) : TensorMeta()),
      out_tensor(out_qg.has_value() ? meta_of(*out_qg) : TensorMeta()),
      out_tensor(out_kg.has_value() ? meta_of(*out_kg) : TensorMeta()),
      out_tensor(out_v_new.has_value() ? meta_of(*out_v_new) : TensorMeta()),
      out_tensor(out_h.has_value() ? meta_of(*out_h) : TensorMeta()));
  return std::make_tuple(out_attn, out_final_state, out_gk, out_aqk, out_akk,
                         out_w, out_u, out_qg, out_kg, out_v_new, out_h);
}

}  // namespace
