// Stable-ABI adapter for npu_chunk_bwd_dv_local.
// aclnn: aclnnChunkBwdDvLocal
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
// npu_chunk_bwd_dv_local
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_bwd_dv_local =
    "npu_chunk_bwd_dv_local(Tensor q, Tensor k, Tensor d_o, Tensor g, "
    "Tensor? g_gamma, Tensor? A, Tensor? cu_seqlens, Tensor? chunk_indices, "
    "float scale, int chunk_size, int stream) -> Tensor";

Tensor run_npu_chunk_bwd_dv_local(Tensor q, Tensor k, Tensor d_o, Tensor g,
                                  std::optional<Tensor> g_gamma,
                                  std::optional<Tensor> A,
                                  std::optional<Tensor> cu_seqlens,
                                  std::optional<Tensor> chunk_indices,
                                  double scale, int64_t chunk_size,
                                  int64_t stream) {
  const TensorMeta d_o_meta = meta_of(d_o);
  Tensor out = allocate_like(d_o_meta);
  FLA_STABLE_EXEC("aclnnChunkBwdDvLocal", d_o_meta, stream, tensor(meta_of(q)),
                  tensor(meta_of(k)), tensor(d_o_meta), tensor(meta_of(g)),
                  optional_tensor(g_gamma), optional_tensor(A),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  scalar(scale), scalar(chunk_size), out_tensor(meta_of(out)));
  return out;
}

}  // namespace
