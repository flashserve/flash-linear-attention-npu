// Stable-ABI adapters: npu_chunk_gated_delta_rule_fwd_h and
// npu_chunk_gated_delta_rule_bwd_dhu.
//
// These are the first adapters with a conditional output.  The slot exists in
// the schema as `Tensor?` and is only allocated when the caller asks for it
// (fwd_h, on output_final_state) or when an initial state was supplied
// (bwd_dhu); an undefined meta for that slot becomes a null aclTensor, which is
// how the generated adapters spelled "absent" too.
//
// Their output shapes count chunks, so the chunk-count helpers the generated
// code spliced in from the specs live here as ordinary functions.
//
// Included by stable_ops.cpp (single TU); registration lives there.

// Owns the h/dh recurrence family: npu_chunk_gated_delta_rule_fwd_h,
// npu_chunk_gated_delta_rule_bwd_dhu.  npu_chunk_fwd_h has no kernel on this
// branch, so its adapter is not carried here.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::logical_optional_tensor;
using fla_npu_stable::stable::logical_out_tensor;
using fla_npu_stable::stable::logical_tensor;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// --- chunk counting ---------------------------------------------------------

inline int64_t count_chunks(const std::vector<int64_t>& cu_seqlens,
                            const std::vector<int64_t>& chunk_indices,
                            int64_t chunk_size, int64_t total_tokens) {
  if (!chunk_indices.empty()) {
    return static_cast<int64_t>(chunk_indices.size()) / 2;
  }
  if (!cu_seqlens.empty()) {
    int64_t chunks = 0;
    for (size_t i = 0; i + 1 < cu_seqlens.size(); ++i) {
      const int64_t len = cu_seqlens[i + 1] - cu_seqlens[i];
      chunks += (len + chunk_size - 1) / chunk_size;
    }
    return chunks;
  }
  return (total_tokens + chunk_size - 1) / chunk_size;
}

// Same as above for the backward, which only ever sees chunk_indices.
inline int64_t count_chunks(const std::vector<int64_t>& chunk_indices,
                            int64_t chunk_size, int64_t total_tokens) {
  return count_chunks({}, chunk_indices, chunk_size, total_tokens);
}

// [batch-or-segments, v_heads, k_dim, v_dim] with the head layout following
// state_v_first: from the initial state's dtype when one was supplied, fp32
// otherwise.
inline Tensor allocate_final_state(const TensorMeta& k_meta,
                                   const TensorMeta& u_meta,
                                   const std::vector<int64_t>& cu_seqlens,
                                   bool state_v_first,
                                   const std::optional<Tensor>& initial_state) {
  const int64_t rows = cu_seqlens.empty()
                           ? size_of(k_meta, 0)
                           : static_cast<int64_t>(cu_seqlens.size()) - 1;
  const std::vector<int64_t> sizes = {
      rows, size_of(u_meta, 1),
      state_v_first ? size_of(u_meta, 3) : size_of(k_meta, 3),
      state_v_first ? size_of(k_meta, 3) : size_of(u_meta, 3)};
  if (initial_state.has_value()) {
    return allocate_sizes(sizes, meta_of(*initial_state).scalar_type,
                          meta_of(*initial_state));
  }
  return allocate_sizes(sizes, kFloat, k_meta);
}

struct FwdHOutputs {
  Tensor h;
  Tensor v_new;
  std::optional<Tensor> final_state;
};

inline FwdHOutputs allocate_fwd_h(const TensorMeta& k_meta,
                                  const TensorMeta& u_meta,
                                  const std::vector<int64_t>& cu_seqlens,
                                  const std::vector<int64_t>& chunk_indices,
                                  int64_t chunk_size, bool output_final_state,
                                  bool state_v_first,
                                  const std::optional<Tensor>& initial_state) {
  FwdHOutputs out;
  out.h = allocate_sizes(
      {size_of(k_meta, 0), size_of(u_meta, 1),
       count_chunks(cu_seqlens, chunk_indices, chunk_size,
                    size_of(k_meta, 2)),
       state_v_first ? size_of(u_meta, 3) : size_of(k_meta, 3),
       state_v_first ? size_of(k_meta, 3) : size_of(u_meta, 3)},
      k_meta.scalar_type, k_meta);
  out.v_new = allocate_like(u_meta);
  if (output_final_state) {
    out.final_state =
        allocate_final_state(k_meta, u_meta, cu_seqlens, state_v_first,
                             initial_state);
  }
  return out;
}

inline TensorMeta meta_or_undefined(const std::optional<Tensor>& value) {
  return value.has_value() ? meta_of(*value) : TensorMeta();
}

// ---------------------------------------------------------------------------
// npu_chunk_gated_delta_rule_fwd_h
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_gated_delta_rule_fwd_h =
    "npu_chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, Tensor? g, "
    "Tensor? gk, Tensor? initial_state, bool output_final_state, "
    "int chunk_size, Tensor? cu_seqlens, Tensor? chunk_indices, "
    "bool state_v_first, int stream) -> (Tensor, Tensor, Tensor?)";

std::tuple<Tensor, Tensor, std::optional<Tensor>>
run_npu_chunk_gated_delta_rule_fwd_h(
    Tensor k, Tensor w, Tensor u, std::optional<Tensor> g,
    std::optional<Tensor> gk, std::optional<Tensor> initial_state,
    bool output_final_state, int64_t chunk_size,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    bool state_v_first, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta u_meta = meta_of(u);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  const FwdHOutputs out = allocate_fwd_h(k_meta, u_meta, cu, ci, chunk_size,
                                         output_final_state, state_v_first,
                                         initial_state);

  FLA_STABLE_EXEC("aclnnChunkGatedDeltaRuleFwdH", k_meta, stream,
                  tensor(k_meta), tensor(meta_of(w)), tensor(u_meta),
                  optional_tensor(g), optional_tensor(gk),
                  optional_tensor(initial_state), scalar(output_final_state),
                  scalar(chunk_size), int_array(cu), int_array(ci),
                  scalar(state_v_first), out_tensor(meta_of(out.h)),
                  out_tensor(meta_of(out.v_new)),
                  out_tensor(meta_or_undefined(out.final_state)));
  return std::make_tuple(out.h, out.v_new, out.final_state);
}

// ---------------------------------------------------------------------------
// npu_chunk_gated_delta_rule_bwd_dhu
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_gated_delta_rule_bwd_dhu =
    "npu_chunk_gated_delta_rule_bwd_dhu(Tensor q, Tensor k, Tensor w, "
    "Tensor d_o, Tensor dv, Tensor? g, Tensor? gK, Tensor? h0, Tensor? dht, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size, "
    "bool use_exp2, int stream) "
    "-> (Tensor, Tensor?, Tensor)";

std::tuple<Tensor, std::optional<Tensor>, Tensor>
run_npu_chunk_gated_delta_rule_bwd_dhu(
    Tensor q, Tensor k, Tensor w, Tensor d_o, Tensor dv,
    std::optional<Tensor> g, std::optional<Tensor> gK,
    std::optional<Tensor> h0, std::optional<Tensor> dht,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    double scale, int64_t chunk_size, bool use_exp2, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta dv_meta = meta_of(dv);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  const int64_t chunks = count_chunks(ci, chunk_size, size_of(q_meta, 2));
  const std::vector<int64_t> dh_sizes = {
      size_of(q_meta, 0), size_of(dv_meta, 1), chunks, size_of(q_meta, 3),
      size_of(dv_meta, 3)};

  Tensor out_dh = allocate_sizes(dh_sizes, q_meta.scalar_type, q_meta);
  // dh0 mirrors h0's presence: without an initial state there is nothing to
  // differentiate against, so the slot stays absent.  Its shape is one state
  // per *sequence* (not per chunk); this branch's entry point has no
  // stateVFirst flag, so the tail is always (K, V) -- exactly the shape the
  // reference allocates, which is why the caller's `transpose_state_layout`
  // kwarg is accepted and ignored.
  std::optional<Tensor> out_dh0;
  if (h0.has_value()) {
    const int64_t sequences = cu.empty()
                                  ? size_of(q_meta, 0)
                                  : static_cast<int64_t>(cu.size()) - 1;
    out_dh0 = allocate_sizes(
        {sequences, size_of(dv_meta, 1), size_of(q_meta, 3),
         size_of(dv_meta, 3)},
        q_meta.scalar_type, q_meta);
  }
  Tensor out_dv = allocate_like(dv_meta);

  FLA_STABLE_EXEC("aclnnChunkGatedDeltaRuleBwdDhu", q_meta, stream,
                  // Logical storage shape, format left to the tensor -- this
                  // operator's reference spells it `logical_tensor`.
                  logical_tensor(q_meta), logical_tensor(meta_of(k)),
                  logical_tensor(meta_of(w)), logical_tensor(meta_of(d_o)),
                  logical_tensor(dv_meta), logical_optional_tensor(g),
                  logical_optional_tensor(gK), logical_optional_tensor(h0),
                  logical_optional_tensor(dht), int_array(cu), int_array(ci),
                  scalar(scale), scalar(chunk_size), scalar(use_exp2),
                  logical_out_tensor(meta_of(out_dh)),
                  logical_out_tensor(meta_or_undefined(out_dh0)),
                  logical_out_tensor(meta_of(out_dv)));
  return std::make_tuple(out_dh, out_dh0, out_dv);
}

}  // namespace
