// Stable-ABI adapter for npu_chunk_kda_bwd_recompute.
// aclnn: aclnnChunkKdaBwdRecompute
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

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
using fla_npu_stable::stable::nd_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_chunk_kda_bwd_recompute
// ---------------------------------------------------------------------------

// The declared order is the *public* one (`gk` first); aclnn wants the
// recomputed tensors first and the gate cumsum last, so the return tuple is
// reordered at the end rather than in Python.
constexpr const char* kSchema_chunk_kda_bwd_recompute =
    "npu_chunk_kda_bwd_recompute(Tensor q, Tensor k, Tensor v, Tensor g, "
    "Tensor beta, Tensor a, Tensor? A_log, Tensor? dt_bias, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int chunk_size, bool use_exp2, "
    "float lower_bound, bool use_gate_in_kernel, int stream) "
    "-> (Tensor?, Tensor, Tensor, Tensor, Tensor)";

std::tuple<std::optional<Tensor>, Tensor, Tensor, Tensor, Tensor>
run_npu_chunk_kda_bwd_recompute(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor a,
    std::optional<Tensor> A_log, std::optional<Tensor> dt_bias,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t chunk_size, bool use_exp2, double lower_bound,
    bool use_gate_in_kernel, int64_t stream) {
  const TensorMeta v_meta = meta_of(v);
  const TensorMeta g_meta = meta_of(g);
  // `w`/`u` follow their source `v`; `qg`/`kg` are the bfloat16 gate tensors
  // (even when `g` itself is fp32), and the gate cumsum is fp32.
  Tensor out_w = allocate_sizes(v_meta.sizes, v_meta.scalar_type, v_meta);
  Tensor out_u = allocate_sizes(v_meta.sizes, v_meta.scalar_type, v_meta);
  Tensor out_qg = allocate_sizes(g_meta.sizes, kBFloat16, g_meta);
  Tensor out_kg = allocate_sizes(g_meta.sizes, kBFloat16, g_meta);
  std::optional<Tensor> out_gk;
  if (use_gate_in_kernel) {
    out_gk = allocate_sizes(g_meta.sizes, kFloat, g_meta);
  }

  FLA_STABLE_EXEC("aclnnChunkKdaBwdRecompute", v_meta, stream,
                  tensor(meta_of(q)), tensor(meta_of(k)), tensor(v_meta),
                  tensor(g_meta), tensor(meta_of(beta)), tensor(meta_of(a)),
                  optional_tensor(A_log), optional_tensor(dt_bias),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  scalar(chunk_size), scalar(use_exp2), scalar(lower_bound),
                  out_tensor(meta_of(out_w)), out_tensor(meta_of(out_u)),
                  out_tensor(meta_of(out_qg)), out_tensor(meta_of(out_kg)),
                  out_tensor(out_gk.has_value() ? meta_of(*out_gk)
                                                : TensorMeta()));
  return std::make_tuple(out_gk, out_w, out_u, out_qg, out_kg);
}

}  // namespace
