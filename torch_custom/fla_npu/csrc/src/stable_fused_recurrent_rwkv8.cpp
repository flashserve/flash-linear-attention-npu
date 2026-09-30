// Stable-ABI adapter for npu_fused_recurrent_rwkv8.
// aclnn: aclnnFusedRecurrentRwkv8
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

// Six same-dtype io tensors, an optional fp32 initial state, float/bool/int64
// scalars, and two conditional fp32 outputs.  The `s` snapshot output has a
// zero-slot corner (T < chunk_len): aclnn wants an (B,H,0,K,V) view but
// rejects a 0-byte allocation, so the adapter allocates a one-row buffer and
// hands aclnn a view meta with sizes[2] = 0 -- the same buffer-and-slice
// trick the ctypes reference plays in Python.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_fused_recurrent_rwkv8
// ---------------------------------------------------------------------------

constexpr const char* kSchema_fused_recurrent_rwkv8 =
    "npu_fused_recurrent_rwkv8(Tensor q, Tensor w, Tensor k, Tensor v, "
    "Tensor z, Tensor b, Tensor? initial_state, float scale, bool reverse, "
    "bool output_chunk_state, bool output_sa, int chunk_len, int stream) "
    "-> (Tensor, Tensor?, Tensor?)";

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>>
run_npu_fused_recurrent_rwkv8(Tensor q, Tensor w, Tensor k, Tensor v,
                              Tensor z, Tensor b,
                              std::optional<Tensor> initial_state,
                              double scale, bool reverse,
                              bool output_chunk_state, bool output_sa,
                              int64_t chunk_len, int64_t stream) {
  if (chunk_len < 1) {
    throw std::runtime_error(
        "fla_npu(stable): npu_fused_recurrent_rwkv8 chunk_len must be >= 1");
  }
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  const int64_t batch = SIZE_OF(q_meta, 0);
  const int64_t heads = SIZE_OF(q_meta, 1);
  const int64_t seqlen = SIZE_OF(v_meta, 2);
  const int64_t k_dim = SIZE_OF(q_meta, 3);
  const int64_t v_dim = SIZE_OF(v_meta, 3);

  // o: (B,H,T,V), dtype follows q (== v, the wrapper contract pins one dtype).
  Tensor out = allocate_sizes(v_meta.sizes, q_meta.scalar_type, q_meta);

  // s: chunk snapshots (B,H,T//chunk_len,K,V) fp32, only when asked.
  std::optional<Tensor> s;
  TensorMeta s_meta;  // stays undefined (null aclTensor) when output is off
  if (output_chunk_state) {
    const int64_t num_chunks = seqlen / chunk_len;
    Tensor s_buf = allocate_sizes(
        {batch, heads, num_chunks > 0 ? num_chunks : 1, k_dim, v_dim}, kFloat,
        q_meta);
    s_meta = meta_of(s_buf);
    if (num_chunks == 0) {
      // T < chunk_len: hand aclnn the 0-slot view over the one-row buffer
      // (strides/storage_numel stay the buffer's, like the ctypes slice), and
      // return a genuinely 0-sized tensor.  The kernel writes nothing when
      // there are no slots, so the unreturned buffer may leave the frame.
      s_meta.sizes[2] = 0;
      s = allocate_sizes({batch, heads, 0, k_dim, v_dim}, kFloat, q_meta);
    } else {
      s = std::move(s_buf);
    }
  }

  // sa: per-token state@z (B,H,T,V) fp32, only when asked.
  std::optional<Tensor> sa;
  if (output_sa) {
    sa = allocate_sizes(v_meta.sizes, kFloat, q_meta);
  }

  FLA_STABLE_EXEC("aclnnFusedRecurrentRwkv8", q_meta, stream, tensor(q_meta),
                  tensor(meta_of(w)), tensor(meta_of(k)), tensor(v_meta),
                  tensor(meta_of(z)), tensor(meta_of(b)),
                  optional_tensor(initial_state),
                  scalar(static_cast<float>(scale)), scalar(reverse),
                  scalar(output_chunk_state), scalar(output_sa),
                  scalar(chunk_len), out_tensor(meta_of(out)),
                  out_tensor(s.has_value() ? s_meta : TensorMeta()),
                  out_tensor(sa.has_value() ? meta_of(*sa) : TensorMeta()));
  return {std::move(out), std::move(s), std::move(sa)};
}

}  // namespace
