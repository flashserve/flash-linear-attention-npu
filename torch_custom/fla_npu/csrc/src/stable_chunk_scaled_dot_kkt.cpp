// Stable-ABI adapter for npu_chunk_scaled_dot_kkt.
// aclnn: aclnnChunkScaledDotKkt
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
// npu_chunk_scaled_dot_kkt
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_scaled_dot_kkt =
    "npu_chunk_scaled_dot_kkt(Tensor k, Tensor g, Tensor beta, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int chunk_size, int stream) "
    "-> Tensor";

Tensor run_npu_chunk_scaled_dot_kkt(Tensor k, Tensor g, Tensor beta,
                                    std::optional<Tensor> cu_seqlens,
                                    std::optional<Tensor> chunk_indices,
                                    int64_t chunk_size, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta g_meta = meta_of(g);
  Tensor out = allocate_sizes(
      {SIZE_OF(k_meta, 0), SIZE_OF(g_meta, 1), SIZE_OF(k_meta, 2), chunk_size},
      kFloat, k_meta);
  FLA_STABLE_EXEC("aclnnChunkScaledDotKkt", k_meta, stream, tensor(k_meta),
                  tensor(g_meta), tensor(meta_of(beta)),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  scalar(chunk_size), out_tensor(meta_of(out)));
  return out;
}

}  // namespace
