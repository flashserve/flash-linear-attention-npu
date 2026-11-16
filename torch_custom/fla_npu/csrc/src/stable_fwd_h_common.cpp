// Shared helpers for the operators whose adapters are
// chunk_fwd_h, chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu.
//
// Only declarations used by more than one operator live here; everything
// else belongs to the operator's own stable_<op>.cpp.  csrc/src/stable_ops.cpp
// includes this file first, so its names are declared before every user.

// The h/dh adapters are the ones with a conditional output: the slot is a
// `Tensor?` in the schema and only allocated when the caller asks for it
// (on output_final_state, or because an initial state was supplied), and an
// undefined meta becomes a null aclTensor.
// Their output shapes count chunks, so the chunk-count helpers the generated
// adapters used to splice in from the specs live here as ordinary functions.

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
using fla_npu_stable::stable::nd_optional_tensor;
using fla_npu_stable::stable::nd_out_tensor;
using fla_npu_stable::stable::nd_tensor;
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
                           ? SIZE_OF(k_meta, 0)
                           : static_cast<int64_t>(cu_seqlens.size()) - 1;
  const std::vector<int64_t> sizes = {
      rows, SIZE_OF(u_meta, 1),
      state_v_first ? SIZE_OF(u_meta, 3) : SIZE_OF(k_meta, 3),
      state_v_first ? SIZE_OF(k_meta, 3) : SIZE_OF(u_meta, 3)};
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
  const int64_t chunks = count_chunks(cu_seqlens, chunk_indices, chunk_size,
                                     SIZE_OF(k_meta, 2));
  const std::vector<int64_t> h_shape =
      {SIZE_OF(k_meta, 0), chunks, SIZE_OF(u_meta, 1),
       state_v_first ? SIZE_OF(u_meta, 3) : SIZE_OF(k_meta, 3),
       state_v_first ? SIZE_OF(k_meta, 3) : SIZE_OF(u_meta, 3)};
  out.h = allocate_sizes(h_shape, k_meta.scalar_type, k_meta);
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

}  // namespace
