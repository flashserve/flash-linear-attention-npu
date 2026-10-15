// Stable-ABI adapter: npu_chunk_fwd_o.
//
// The operator allocates nothing itself: `o` follows `v` in shape, dtype and
// device, and this branch's aclnn entry point takes neither a layout nor a
// useExp2 flag, so the adapter is a straight marshal of the reference call.
//
// Included by stable_ops.cpp (single TU); registration lives there.
//
// Owns npu_chunk_fwd_o.  The rest of this family -- the gated-delta-rule
// composites and the h/dh recurrence -- has no kernel on this branch, so no
// adapter for them is carried here.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_chunk_fwd_o
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_fwd_o =
    "npu_chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, Tensor? g, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size, "
    "int stream) -> Tensor";

Tensor run_npu_chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h,
                           std::optional<Tensor> g,
                           std::optional<Tensor> cu_seqlens,
                           std::optional<Tensor> chunk_indices, double scale,
                           int64_t chunk_size, int64_t stream) {
  const TensorMeta v_meta = meta_of(v);
  Tensor out = allocate_like(v_meta);

  FLA_STABLE_EXEC("aclnnChunkFwdO", meta_of(v), stream, tensor(meta_of(q)),
                  tensor(meta_of(k)), tensor(v_meta), tensor(meta_of(h)),
                  optional_tensor(g), int_array(cu_seqlens),
                  int_array(chunk_indices), scalar(scale), scalar(chunk_size),
                  out_tensor(meta_of(out)));
  return out;
}

}  // namespace
