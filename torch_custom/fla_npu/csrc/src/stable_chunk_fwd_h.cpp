// Stable-ABI adapter for npu_chunk_fwd_h.
// aclnn: aclnnChunkFwdH
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

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

// ---------------------------------------------------------------------------
// npu_chunk_fwd_h
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_fwd_h =
    "npu_chunk_fwd_h(Tensor k, Tensor w, Tensor u, Tensor? g, Tensor? gk, "
    "Tensor? initial_state, bool output_final_state, int chunk_size, "
    "bool save_new_value, Tensor? cu_seqlens, Tensor? chunk_indices, "
    "bool use_exp2, bool state_v_first, int stream) "
    "-> (Tensor, Tensor, Tensor?)";

std::tuple<Tensor, Tensor, std::optional<Tensor>> run_npu_chunk_fwd_h(
    Tensor k, Tensor w, Tensor u, std::optional<Tensor> g,
    std::optional<Tensor> gk, std::optional<Tensor> initial_state,
    bool output_final_state, int64_t chunk_size, bool save_new_value,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    bool use_exp2, bool state_v_first, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta u_meta = meta_of(u);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  const FwdHOutputs out = allocate_fwd_h(k_meta, u_meta, cu, ci, chunk_size,
                                         output_final_state, state_v_first,
                                         initial_state);

  // ND descriptors: `npu_chunk_fwd_h`'s reference passes
  // `storage_shape_override=_shape(tensor)` together with an ND format, while
  // the GDN spelling below keeps the rank-inferred format like its own
  // reference does.
  FLA_STABLE_EXEC("aclnnChunkFwdH", k_meta, stream, nd_tensor(k_meta),
                  nd_tensor(meta_of(w)), nd_tensor(u_meta), nd_optional_tensor(g),
                  nd_optional_tensor(gk),
                  nd_optional_tensor(initial_state),
                  scalar(output_final_state), scalar(chunk_size),
                  scalar(save_new_value), int_array(cu), int_array(ci),
                  scalar(use_exp2), scalar(state_v_first),
                  nd_out_tensor(meta_of(out.h)),
                  nd_out_tensor(meta_of(out.v_new)),
                  nd_out_tensor(meta_or_undefined(out.final_state)));
  return std::make_tuple(out.h, out.v_new, out.final_state);
}

}  // namespace
