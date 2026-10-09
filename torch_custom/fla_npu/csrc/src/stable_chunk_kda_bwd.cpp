// Stable-ABI adapter for npu_chunk_kda_bwd.
// aclnn: aclnnChunkKdaBwd
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
#include <stdexcept>
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
// npu_chunk_kda_bwd
// ---------------------------------------------------------------------------

// One fused launch.  Everything a caller sees beyond this -- the per-sequence
// split for the packed V=256 shape, the tail padding, the partner head -- is
// host-side policy that lives in the Python wrapper because it decides *how
// many* calls to make, not what a single call looks like.
//
// `dh0` is always a null slot: the fused backward does not differentiate an
// initial state, and the public API says so by returning None for it.
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>>
run_kda_bwd_v2(
    Tensor q, Tensor k, Tensor v, Tensor beta, std::optional<Tensor> gk, Tensor Aqk, Tensor Akk,
    std::optional<Tensor> w, std::optional<Tensor> qg,
    std::optional<Tensor> kg, std::optional<Tensor> v_new,
    std::optional<Tensor> h, Tensor d_o, std::optional<Tensor> raw_g,
    std::optional<Tensor> A_log, std::optional<Tensor> dt_bias,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    double scale, int64_t chunk_size, bool safe_gate, bool use_gate_in_kernel,
    double lower_bound, bool disable_recompute, bool use_exp2,
    bool state_v_first, std::optional<Tensor> q_rstd,
    std::optional<Tensor> k_rstd, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  // V2 token gradients follow their BF16 inputs; db follows beta.
  Tensor out_dq = allocate_like(q_meta);
  Tensor out_dk = allocate_like(k_meta);
  Tensor out_dv = allocate_sizes(v_meta.sizes, v_meta.scalar_type, v_meta);
  Tensor out_db = allocate_like(meta_of(beta));
  Tensor out_dg = allocate_sizes(q_meta.sizes, kFloat, q_meta);
  std::optional<Tensor> out_d_a_log;
  std::optional<Tensor> out_d_dt_bias;
  // Packed inputs are [H, T, D]; dense ones are [B, H, T, D].
  const int64_t key_heads = SIZE_OF(q_meta, q_meta.ndim == 3 ? 0 : 1);
  if (use_gate_in_kernel) {
    out_d_a_log = allocate_sizes({key_heads}, kFloat, q_meta);
    if (dt_bias.has_value()) {
      out_d_dt_bias = allocate_sizes({key_heads, SIZE_OF(q_meta, q_meta.ndim - 1)}, kFloat,
                                     q_meta);
    }
  }

  // Use logical ND descriptors for every V2 tensor.
  FLA_STABLE_EXEC(
      "aclnnChunkKdaBwdV2", q_meta, stream, nd_tensor(q_meta),
      nd_tensor(k_meta), nd_tensor(v_meta), nd_tensor(meta_of(beta)),
      nd_optional_tensor(gk), nd_tensor(meta_of(Aqk)), nd_tensor(meta_of(Akk)),
      nd_optional_tensor(w), nd_optional_tensor(qg), nd_optional_tensor(kg),
      nd_optional_tensor(v_new), nd_optional_tensor(h),
      nd_tensor(meta_of(d_o)), nd_optional_tensor(raw_g),
      nd_optional_tensor(A_log), nd_optional_tensor(dt_bias),
      /*initial_state=*/nd_optional_tensor(std::nullopt),
      /*dht=*/nd_optional_tensor(std::nullopt), int_array(cu_seqlens),
      int_array(chunk_indices), scalar(scale), scalar(chunk_size),
      scalar(safe_gate), scalar(use_gate_in_kernel), scalar(lower_bound),
      scalar(disable_recompute), scalar(use_exp2), scalar(state_v_first),
      nd_optional_tensor(q_rstd), nd_optional_tensor(k_rstd),
      nd_logical_out_tensor(meta_of(out_dq)),
      nd_logical_out_tensor(meta_of(out_dk)),
      nd_logical_out_tensor(meta_of(out_dv)),
      nd_logical_out_tensor(meta_of(out_db)),
      nd_logical_out_tensor(meta_of(out_dg)),
      /*dh0=*/nd_logical_out_tensor(TensorMeta()),
      nd_logical_out_tensor(out_d_a_log.has_value() ? meta_of(*out_d_a_log)
                                                    : TensorMeta()),
      nd_logical_out_tensor(out_d_dt_bias.has_value() ? meta_of(*out_d_dt_bias)
                                                      : TensorMeta()));
  return std::make_tuple(out_dq, out_dk, out_dv, out_db, out_dg,
                         std::nullopt, out_d_a_log, out_d_dt_bias);
}

constexpr const char* kSchema_chunk_kda_bwd =
    "npu_chunk_kda_bwd(Tensor q, Tensor k, Tensor v, Tensor beta, Tensor? gk, "
    "Tensor Aqk, Tensor Akk, Tensor? w, Tensor? qg, Tensor? kg, "
    "Tensor? v_new, Tensor? h, Tensor d_o, Tensor? raw_g, Tensor? A_log, "
    "Tensor? dt_bias, Tensor? cu_seqlens, Tensor? chunk_indices, float scale, "
    "int chunk_size, bool safe_gate, bool use_gate_in_kernel, "
    "float lower_bound, bool disable_recompute, bool use_exp2, "
    "bool state_v_first, bool optimized, Tensor? q_rstd, Tensor? k_rstd, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor?, Tensor?)";

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>>
run_npu_chunk_kda_bwd(
    Tensor q, Tensor k, Tensor v, Tensor beta, std::optional<Tensor> gk, Tensor Aqk, Tensor Akk,
    std::optional<Tensor> w, std::optional<Tensor> qg,
    std::optional<Tensor> kg, std::optional<Tensor> v_new,
    std::optional<Tensor> h, Tensor d_o, std::optional<Tensor> raw_g,
    std::optional<Tensor> A_log, std::optional<Tensor> dt_bias,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    double scale, int64_t chunk_size, bool safe_gate, bool use_gate_in_kernel,
    double lower_bound, bool disable_recompute, bool use_exp2,
    bool state_v_first, bool optimized, std::optional<Tensor> q_rstd,
    std::optional<Tensor> k_rstd, int64_t stream) {
  if (optimized) {
    return run_kda_bwd_v2(q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h,
        d_o, raw_g, A_log, dt_bias, cu_seqlens, chunk_indices, scale,
        chunk_size, safe_gate, use_gate_in_kernel, lower_bound,
        disable_recompute, use_exp2, state_v_first, q_rstd, k_rstd, stream);
  }
  if (!gk.has_value()) {
    throw std::runtime_error("legacy KDA backward requires gk");
  }
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  const TensorMeta gk_meta = meta_of(*gk);
  // The token gradients are fp32, dv follows v, dg follows the fp32 `gk`.
  Tensor out_dq = allocate_sizes(q_meta.sizes, kFloat, q_meta);
  Tensor out_dk = allocate_sizes(k_meta.sizes, kFloat, k_meta);
  Tensor out_dv = allocate_sizes(v_meta.sizes, v_meta.scalar_type, v_meta);
  Tensor out_db = allocate_sizes(meta_of(beta).sizes, kFloat, q_meta);
  Tensor out_dg = allocate_sizes(gk_meta.sizes, gk_meta.scalar_type, gk_meta);
  std::optional<Tensor> out_d_a_log;
  std::optional<Tensor> out_d_dt_bias;
  // Packed inputs are [H, T, D]; dense ones are [B, H, T, D].
  const int64_t key_heads = SIZE_OF(q_meta, q_meta.ndim == 3 ? 0 : 1);
  if (use_gate_in_kernel) {
    out_d_a_log = allocate_sizes({key_heads}, kFloat, q_meta);
    if (dt_bias.has_value()) {
      out_d_dt_bias = allocate_sizes(
          {key_heads, SIZE_OF(q_meta, q_meta.ndim - 1)}, kFloat, q_meta);
    }
  }

  FLA_STABLE_EXEC(
      // ND descriptors: the reference's `nd_tensor` helper overrides the format
      // for every argument of this call ("consume the canonical dense
      // BNSD/varlen NTD tensors as ND").
      "aclnnChunkKdaBwd", q_meta, stream, nd_tensor(q_meta),
      nd_tensor(k_meta), nd_tensor(v_meta), nd_tensor(meta_of(beta)),
      nd_tensor(gk_meta), nd_tensor(meta_of(Aqk)), nd_tensor(meta_of(Akk)),
      nd_optional_tensor(w), nd_optional_tensor(qg), nd_optional_tensor(kg),
      nd_optional_tensor(v_new), nd_optional_tensor(h),
      nd_tensor(meta_of(d_o)), nd_optional_tensor(raw_g),
      nd_optional_tensor(A_log), nd_optional_tensor(dt_bias),
      /*initial_state=*/nd_optional_tensor(std::nullopt),
      /*dht=*/nd_optional_tensor(std::nullopt), int_array(cu_seqlens),
      int_array(chunk_indices), scalar(scale), scalar(chunk_size),
      scalar(safe_gate), scalar(use_gate_in_kernel), scalar(lower_bound),
      scalar(disable_recompute), scalar(use_exp2), scalar(state_v_first),
      nd_logical_out_tensor(meta_of(out_dq)),
      nd_logical_out_tensor(meta_of(out_dk)),
      nd_logical_out_tensor(meta_of(out_dv)),
      nd_logical_out_tensor(meta_of(out_db)),
      nd_logical_out_tensor(meta_of(out_dg)),
      /*dh0=*/nd_logical_out_tensor(TensorMeta()),
      nd_logical_out_tensor(out_d_a_log.has_value() ? meta_of(*out_d_a_log)
                                                    : TensorMeta()),
      nd_logical_out_tensor(out_d_dt_bias.has_value() ? meta_of(*out_d_dt_bias)
                                                      : TensorMeta()));
  return std::make_tuple(out_dq, out_dk, out_dv, out_db, out_dg,
                         std::nullopt, out_d_a_log, out_d_dt_bias);
}

}  // namespace
