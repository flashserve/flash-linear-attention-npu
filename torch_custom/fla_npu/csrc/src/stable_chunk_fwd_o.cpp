// Stable-ABI adapter for npu_chunk_fwd_o.
// aclnn: aclnnChunkFwdO
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

// The output shape depends on the layout string, so the enum name is
// resolved once and used both for the allocation and for aclnn.

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

// Table order is the code order; _stable.py's _char_code tables must match.
// Every layout name table in this file uses the code order
// stable/layout_math.h documents (BSND, BNSD, TND, NTD); the Python side
// carries the same table in _stable.py's _ENUM.
constexpr const char* kChunkFwdOOutputLayoutNames[] = {"BSND", "BNSD", "TND",
                                                       "NTD"};

// ---------------------------------------------------------------------------
// npu_chunk_fwd_o
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_fwd_o =
    "npu_chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, Tensor? g, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size, "
    "bool use_exp2, bool transpose_state_layout, int output_layout, "
    "int stream) -> Tensor";

Tensor run_npu_chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h,
                           std::optional<Tensor> g,
                           std::optional<Tensor> cu_seqlens,
                           std::optional<Tensor> chunk_indices, double scale,
                           int64_t chunk_size, bool use_exp2,
                           bool transpose_state_layout, int64_t output_layout,
                           int64_t stream) {
  const TensorMeta v_meta = meta_of(v);
  const char* layout = enum_name(kChunkFwdOOutputLayoutNames, output_layout);
  const int32_t dtype = v_meta.scalar_type;

  Tensor out;
  if (std::strcmp(layout, "BNSD") == 0) {
    out = allocate_sizes({SIZE_OF(v_meta, 0), SIZE_OF(v_meta, 1),
                          SIZE_OF(v_meta, 2), SIZE_OF(v_meta, 3)},
                         dtype, v_meta);
  } else if (std::strcmp(layout, "BSND") == 0) {
    out = allocate_sizes({SIZE_OF(v_meta, 0), SIZE_OF(v_meta, 2),
                          SIZE_OF(v_meta, 1), SIZE_OF(v_meta, 3)},
                         dtype, v_meta);
  } else if (std::strcmp(layout, "TND") == 0) {
    out = allocate_sizes({SIZE_OF(v_meta, 2), SIZE_OF(v_meta, 1),
                          SIZE_OF(v_meta, 3)},
                         dtype, v_meta);
  } else {
    out = allocate_sizes({SIZE_OF(v_meta, 1), SIZE_OF(v_meta, 2),
                          SIZE_OF(v_meta, 3)},
                         dtype, v_meta);
  }

  FLA_STABLE_EXEC("aclnnChunkFwdO", meta_of(v), stream, tensor(meta_of(q)),
                  tensor(meta_of(k)), tensor(v_meta), tensor(meta_of(h)),
                  optional_tensor(g), int_array(cu_seqlens),
                  int_array(chunk_indices), scalar(scale), scalar(chunk_size),
                  scalar(use_exp2), scalar(transpose_state_layout),
                  CStrArg(layout), out_tensor(meta_of(out)));
  return out;
}

}  // namespace
