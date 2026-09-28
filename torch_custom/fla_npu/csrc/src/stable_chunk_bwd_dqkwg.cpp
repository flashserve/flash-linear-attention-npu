// Stable-ABI adapter for npu_chunk_bwd_dqkwg.
// aclnn: aclnnChunkBwdDqkwg
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

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kBFloat16;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_chunk_bwd_dqkwg
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_bwd_dqkwg =
    "npu_chunk_bwd_dqkwg(Tensor q, Tensor k, Tensor v, Tensor g, Tensor h, "
    "Tensor dox, Tensor dh, Tensor dv, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, Tensor? w, Tensor? g_gamma, float scale, "
    "int chunk_size, bool use_exp2, bool transpose_state_layout, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_chunk_bwd_dqkwg(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor h, Tensor dox, Tensor dh,
    Tensor dv, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> chunk_indices, std::optional<Tensor> w,
    std::optional<Tensor> g_gamma, double scale, int64_t chunk_size,
    bool use_exp2, bool transpose_state_layout, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  const TensorMeta g_meta = meta_of(g);
  Tensor out_dq = allocate_like(q_meta);
  Tensor out_dk = allocate_like(k_meta);
  Tensor out_dw = allocate_sizes(
      {SIZE_OF(q_meta, 0), SIZE_OF(v_meta, 1), SIZE_OF(q_meta, 2),
       SIZE_OF(q_meta, 3)},
      q_meta.scalar_type, q_meta);
  Tensor out_dg = allocate_like(g_meta);
  FLA_STABLE_EXEC(
      "aclnnChunkBwdDqkwg", q_meta, stream,
      tensor(q_meta), tensor(k_meta), tensor(v_meta), tensor(g_meta),
      tensor(meta_of(h)), tensor(meta_of(dox)), tensor(meta_of(dh)),
      tensor(meta_of(dv)), int_array(cu_seqlens), int_array(chunk_indices),
      optional_tensor(w), optional_tensor(g_gamma),
      scalar(static_cast<float>(scale)), scalar(chunk_size), scalar(use_exp2),
      scalar(transpose_state_layout), out_tensor(meta_of(out_dq)),
      out_tensor(meta_of(out_dk)), out_tensor(meta_of(out_dw)),
      out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dq, out_dk, out_dw, out_dg);
}

}  // namespace
