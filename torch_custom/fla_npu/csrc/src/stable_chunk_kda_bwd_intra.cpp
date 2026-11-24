// Stable-ABI adapter for npu_chunk_kda_bwd_intra.
// aclnn: aclnnChunkKdaBwdIntra
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

// Ten tensor inputs, two int_arrays, a `char*` enum argument carried as an
// int code plus a name table, and four outputs allocated from their inputs.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/layout_math.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kBFloat16;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_optional_tensor;
using fla_npu_stable::stable::nd_logical_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_chunk_kda_bwd_intra
// ---------------------------------------------------------------------------

// The aclnn entry point takes `layout` as a string; the stable value
// conversions cannot carry one, so the caller passes a code and this table is
// the single source of the legal values.  The order must match the Python
// _char_code table -- tools/op_abi_parity.py checks exactly that.
// The kernel takes the layout as a string.  TND is the packed varlen spelling;
// NTD is not part of this operator's domain (the reference rejects it), so it
// has no code.
constexpr const char* kChunkKdaBwdIntraLayoutNames[] = {"BSND", "BNSD", "TND"};

constexpr const char* kSchema_chunk_kda_bwd_intra =
    "npu_chunk_kda_bwd_intra(Tensor q, Tensor k, Tensor gk, Tensor beta, "
    "Tensor dAqk, Tensor dAkk, Tensor dq, Tensor dk, Tensor db, Tensor dg, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int chunk_size, bool safe_gate, "
    "int layout, int stream) -> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_chunk_kda_bwd_intra(
    Tensor q, Tensor k, Tensor gk, Tensor beta, Tensor dAqk, Tensor dAkk,
    Tensor dq, Tensor dk, Tensor db, Tensor dg,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t chunk_size, bool safe_gate, int64_t layout, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  // Each gradient output has the shape and dtype of its own input.
  Tensor out_dq = allocate_like(meta_of(dq));
  Tensor out_dk = allocate_like(meta_of(dk));
  Tensor out_db = allocate_like(meta_of(db));
  Tensor out_dg = allocate_like(meta_of(dg));
  FLA_STABLE_EXEC(
      // ND descriptors, like this operator's reference (it passes
      // `acl_format_override=ACL_FORMAT_ND` for every argument).
      "aclnnChunkKdaBwdIntra", q_meta, stream,
      nd_tensor(meta_of(q)), nd_tensor(meta_of(k)), nd_tensor(meta_of(gk)),
      nd_tensor(meta_of(beta)), nd_tensor(meta_of(dAqk)),
      nd_tensor(meta_of(dAkk)), nd_tensor(meta_of(dq)), nd_tensor(meta_of(dk)),
      nd_tensor(meta_of(db)), nd_tensor(meta_of(dg)),
      int_array(cu_seqlens), int_array(chunk_indices), scalar(chunk_size),
      scalar(safe_gate), cstr(kChunkKdaBwdIntraLayoutNames, layout),
      nd_logical_out_tensor(meta_of(out_dq)),
      nd_logical_out_tensor(meta_of(out_dk)),
      nd_logical_out_tensor(meta_of(out_db)),
      nd_logical_out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dq, out_dk, out_db, out_dg);
}

}  // namespace
