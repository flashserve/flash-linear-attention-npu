// Stable-ABI adapter for npu_prepare_wy_repr_bwd_da.
// aclnn: aclnnPrepareWyReprBwdDa
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
// npu_prepare_wy_repr_bwd_da / _full / npu_prepare_wy_repr_bwd
// ---------------------------------------------------------------------------

constexpr const char* kSchema_prepare_wy_repr_bwd_da =
    "npu_prepare_wy_repr_bwd_da(Tensor k, Tensor v, Tensor beta, Tensor A, "
    "Tensor dw, Tensor du, Tensor g, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, int chunk_size, int stream) -> Tensor";

Tensor run_npu_prepare_wy_repr_bwd_da(Tensor k, Tensor v, Tensor beta, Tensor A,
                                      Tensor dw, Tensor du, Tensor g,
                                      std::optional<Tensor> cu_seqlens,
                                      std::optional<Tensor> chunk_indices,
                                      int64_t chunk_size, int64_t stream) {
  const TensorMeta A_meta = meta_of(A);
  Tensor out = allocate_like(A_meta);
  FLA_STABLE_EXEC("aclnnPrepareWyReprBwdDa", A_meta, stream,
                  tensor(meta_of(k)), tensor(meta_of(v)), tensor(meta_of(beta)),
                  tensor(A_meta), tensor(meta_of(dw)), tensor(meta_of(du)),
                  tensor(meta_of(g)), int_array(cu_seqlens),
                  int_array(chunk_indices), scalar(chunk_size),
                  out_tensor(meta_of(out)));
  return out;
}

}  // namespace
