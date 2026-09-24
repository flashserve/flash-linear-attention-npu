// Stable-ABI adapter for npu_chunk_kda_fwd_finalize.
// aclnn: aclnnChunkKdaFwdFinalize
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

// Optional int_arrays, a layout enum, and an output that must be described by
// its logical ND storage shape (nd_logical_out_tensor) because this kernel
// reads the storage shape when tiling.

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
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_logical_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;

// ---------------------------------------------------------------------------
// npu_chunk_kda_fwd_finalize
// ---------------------------------------------------------------------------

constexpr const char* kChunkKdaFwdFinalizeLayoutNames[] = {"BSND", "BNSD",
                                                           "TND", "NTD"};

// 输出布局由参数指定；h 为 [B,NT,HV,K,V]。
constexpr const char* kSchema_chunk_kda_fwd_finalize =
    "npu_chunk_kda_fwd_finalize(Tensor qg_scaled, Tensor aqk, Tensor v_new, "
    "Tensor h, Tensor? cu_seqlens, Tensor? chunk_indices, int output_layout, "
    "bool state_v_first, int stream) -> Tensor";

Tensor run_npu_chunk_kda_fwd_finalize(
    Tensor qg_scaled, Tensor aqk, Tensor v_new, Tensor h,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t output_layout, bool state_v_first, int64_t stream) {
  namespace layout_math = fla_npu_stable::stable::layout_math;
  const TensorMeta qg_meta = meta_of(qg_scaled);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  // BSND/TND are sequence-major outputs; BNSD/NTD are head-major.  The inputs
  // are head-major either way, so batch/heads/tokens are read from the input
  // with the head-major spelling and only the output layout is switched.
  const bool packed = layout_math::packed(output_layout);
  const int64_t batch = packed ? 1 : SIZE_OF(qg_meta, 0);
  const int64_t heads = packed ? SIZE_OF(qg_meta, 0) : SIZE_OF(qg_meta, 1);
  const int64_t tokens = packed ? SIZE_OF(qg_meta, 1) : SIZE_OF(qg_meta, 2);
  const int64_t head_dim = 128;
  const bool sequence_major = output_layout == 0 || output_layout == 2;
  std::vector<int64_t> attn_sizes;
  if (packed) {
    attn_sizes = sequence_major ? std::vector<int64_t>{tokens, heads, head_dim}
                                : std::vector<int64_t>{heads, tokens, head_dim};
  } else {
    attn_sizes = sequence_major
                     ? std::vector<int64_t>{batch, tokens, heads, head_dim}
                     : std::vector<int64_t>{batch, heads, tokens, head_dim};
  }
  Tensor out_attn = allocate_sizes(attn_sizes, qg_meta.scalar_type, qg_meta);
  FLA_STABLE_EXEC(
      "aclnnChunkKdaFwdFinalize", qg_meta, stream, nd_tensor(qg_meta),
      nd_tensor(meta_of(aqk)), nd_tensor(meta_of(v_new)), nd_tensor(meta_of(h)),
      int_array(cu), int_array(ci),
      cstr(kChunkKdaFwdFinalizeLayoutNames, output_layout),
      scalar(state_v_first), nd_logical_out_tensor(meta_of(out_attn)));
  return out_attn;
}

}  // namespace
