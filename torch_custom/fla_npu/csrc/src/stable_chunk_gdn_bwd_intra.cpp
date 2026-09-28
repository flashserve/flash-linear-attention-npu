// Stable-ABI adapter for npu_chunk_gdn_bwd_intra.
// aclnn: aclnnChunkGdnBwdIntra
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

// Mixes an output shaped from q and v with two that copy v.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/layout_math.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::CStrArg;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::enum_name;
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
// npu_chunk_gdn_bwd_intra
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_gdn_bwd_intra =
    "npu_chunk_gdn_bwd_intra(Tensor q, Tensor k, Tensor v, Tensor g, "
    "Tensor beta, Tensor A, Tensor d_o, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, float scale, int chunk_size, bool use_exp2, "
    "int stream) -> (Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor> run_npu_chunk_gdn_bwd_intra(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor A, Tensor d_o,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    double scale, int64_t chunk_size, bool use_exp2, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  Tensor out_dq = allocate_sizes(
      {SIZE_OF(q_meta, 0), SIZE_OF(v_meta, 1), SIZE_OF(q_meta, 2),
       SIZE_OF(q_meta, 3)},
      q_meta.scalar_type, q_meta);
  Tensor out_dk = allocate_like(v_meta);
  Tensor out_dv = allocate_like(v_meta);

  // ND again: this reference passes `acl_format_override=ACL_FORMAT_ND` ("BNSD
  // tensors are already contiguous; expose that physical shape to tiling").
  FLA_STABLE_EXEC("aclnnChunkGdnBwdIntra", q_meta, stream, nd_tensor(q_meta),
                  nd_tensor(meta_of(k)), nd_tensor(v_meta),
                  nd_tensor(meta_of(g)), nd_tensor(meta_of(beta)),
                  nd_tensor(meta_of(A)), nd_tensor(meta_of(d_o)),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  scalar(scale), scalar(chunk_size), scalar(use_exp2),
                  nd_out_tensor(meta_of(out_dq)),
                  nd_out_tensor(meta_of(out_dk)),
                  nd_out_tensor(meta_of(out_dv)));
  return std::make_tuple(out_dq, out_dk, out_dv);
}

}  // namespace
