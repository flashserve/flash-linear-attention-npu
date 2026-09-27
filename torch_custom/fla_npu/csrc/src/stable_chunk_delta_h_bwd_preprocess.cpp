// Stable-ABI adapter for npu_chunk_delta_h_bwd_preprocess.
// aclnn: aclnnChunkDeltaHBwdPreprocess
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
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_chunk_delta_h_bwd_preprocess
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_delta_h_bwd_preprocess =
    "npu_chunk_delta_h_bwd_preprocess(Tensor q, Tensor k, Tensor w, "
    "Tensor d_o, Tensor dv, Tensor? g, Tensor? gk, Tensor? cu_seqlens, "
    "float scale, int chunk_size, int stream) -> Tensor";

Tensor run_npu_chunk_delta_h_bwd_preprocess(
    Tensor q, Tensor k, Tensor w, Tensor d_o, Tensor dv,
    std::optional<Tensor> g, std::optional<Tensor> gk,
    std::optional<Tensor> cu_seqlens, double scale, int64_t chunk_size,
    int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta w_meta = meta_of(w);
  const TensorMeta d_o_meta = meta_of(d_o);
  // dhm = [E_r | P_r] 是 FP32 的 [Hv, K, V + K]：Hv 取自 value/gate 侧，K 取自 q/k，
  // V 取自 d_o/dv；与 aclnn 文档里的输出 shape 规则一致。
  const std::vector<int64_t> dhm_sizes = {
      SIZE_OF(w_meta, 1), SIZE_OF(q_meta, 3),
      SIZE_OF(d_o_meta, 3) + SIZE_OF(q_meta, 3)};
  Tensor out = allocate_sizes(dhm_sizes, kFloat, q_meta);
  FLA_STABLE_EXEC("aclnnChunkDeltaHBwdPreprocess", q_meta, stream,
                  tensor(q_meta), tensor(meta_of(k)), tensor(w_meta),
                  tensor(d_o_meta), tensor(meta_of(dv)), optional_tensor(g),
                  optional_tensor(gk), int_array(cu_seqlens), scalar(scale),
                  scalar(chunk_size), out_tensor(meta_of(out)));
  return out;
}

}  // namespace
