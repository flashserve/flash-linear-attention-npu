// Stable-ABI adapter for npu_chunk_gated_delta_rule_bwd_dhu.
// aclnn: aclnnChunkGatedDeltaRuleBwdDhu
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
// npu_chunk_gated_delta_rule_bwd_dhu
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_gated_delta_rule_bwd_dhu =
    "npu_chunk_gated_delta_rule_bwd_dhu(Tensor q, Tensor k, Tensor w, "
    "Tensor d_o, Tensor dv, Tensor? g, Tensor? gK, Tensor? h0, Tensor? dht, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size, "
    "bool use_exp2, bool transpose_state_layout, int stream) "
    "-> (Tensor, Tensor?, Tensor)";

std::tuple<Tensor, std::optional<Tensor>, Tensor>
run_npu_chunk_gated_delta_rule_bwd_dhu(
    Tensor q, Tensor k, Tensor w, Tensor d_o, Tensor dv,
    std::optional<Tensor> g, std::optional<Tensor> gK,
    std::optional<Tensor> h0, std::optional<Tensor> dht,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    double scale, int64_t chunk_size, bool use_exp2,
    bool transpose_state_layout, int64_t stream) {
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
  // per *sequence* (not per chunk), and `transpose_state_layout` is what the
  // operator calls `stateVFirst`: it swaps the two state dimensions, which is
  // why the same flag decides whether the tail is (K, V) or (V, K).
  std::optional<Tensor> out_dh0;
  if (h0.has_value()) {
    const int64_t sequences = cu.empty()
                                  ? size_of(q_meta, 0)
                                  : static_cast<int64_t>(cu.size()) - 1;
    out_dh0 = allocate_sizes(
        {sequences, size_of(dv_meta, 1),
         transpose_state_layout ? size_of(dv_meta, 3) : size_of(q_meta, 3),
         transpose_state_layout ? size_of(q_meta, 3) : size_of(dv_meta, 3)},
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
                  scalar(transpose_state_layout),
                  logical_out_tensor(meta_of(out_dh)),
                  logical_out_tensor(meta_or_undefined(out_dh0)),
                  logical_out_tensor(meta_of(out_dv)));
  return std::make_tuple(out_dh, out_dh0, out_dv);
}

}  // namespace
