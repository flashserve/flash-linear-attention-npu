// Stable-ABI adapter for npu_recompute_w_u_fwd.
// aclnn: aclnnRecomputeWUFwd
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
// npu_recompute_w_u_fwd
// ---------------------------------------------------------------------------

constexpr const char* kSchema_recompute_w_u_fwd =
    "npu_recompute_w_u_fwd(Tensor k, Tensor v, Tensor beta, Tensor A, "
    "Tensor? g, Tensor? gk, Tensor? cu_seqlens, Tensor? chunk_indices, "
    "int chunk_size, int stream) -> (Tensor, Tensor)";

std::tuple<Tensor, Tensor> run_npu_recompute_w_u_fwd(
    Tensor k, Tensor v, Tensor beta, Tensor A, std::optional<Tensor> g,
    std::optional<Tensor> gk, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> chunk_indices, int64_t chunk_size, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  Tensor out_w = allocate_sizes(
      {size_of(v_meta, 0), size_of(v_meta, 1), size_of(v_meta, 2),
       size_of(k_meta, 3)},
      k_meta.scalar_type, k_meta);
  Tensor out_u = allocate_like(v_meta);
  FLA_STABLE_EXEC("aclnnRecomputeWUFwd", k_meta, stream, tensor(k_meta),
                  tensor(v_meta), tensor(meta_of(beta)), tensor(meta_of(A)),
                  optional_tensor(g), optional_tensor(gk),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  scalar(chunk_size), out_tensor(meta_of(out_w)),
                  out_tensor(meta_of(out_u)));
  return std::make_tuple(out_w, out_u);
}

}  // namespace
