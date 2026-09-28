// Stable-ABI adapter for npu_prepare_wy_repr_bwd.
// aclnn: aclnnPrepareWyReprBwd
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

constexpr const char* kSchema_prepare_wy_repr_bwd =
    "npu_prepare_wy_repr_bwd(Tensor k, Tensor v, Tensor beta, Tensor A, "
    "Tensor dw, Tensor du, Tensor g, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, int chunk_size, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_prepare_wy_repr_bwd(
    Tensor k, Tensor v, Tensor beta, Tensor A, Tensor dw, Tensor du, Tensor g,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t chunk_size, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  const TensorMeta beta_meta = meta_of(beta);
  const TensorMeta g_meta = meta_of(g);
  Tensor out_dk = allocate_like(k_meta);
  Tensor out_dv = allocate_like(v_meta);
  Tensor out_dbeta = allocate_like(beta_meta);
  Tensor out_dg = allocate_like(g_meta);
  FLA_STABLE_EXEC(
      "aclnnPrepareWyReprBwd", k_meta, stream, tensor(k_meta), tensor(v_meta),
      tensor(beta_meta), tensor(meta_of(A)), tensor(meta_of(dw)),
      tensor(meta_of(du)), tensor(g_meta), int_array(cu_seqlens),
      int_array(chunk_indices), scalar(chunk_size), out_tensor(meta_of(out_dk)),
      out_tensor(meta_of(out_dv)), out_tensor(meta_of(out_dbeta)),
      out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dk, out_dv, out_dbeta, out_dg);
}

}  // namespace
