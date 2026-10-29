// Stable-ABI adapter for npu_chunk_kda_fwd.
// aclnn: aclnnChunkKdaFwd
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
using fla_npu_stable::stable::nd_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_chunk_kda_fwd
// ---------------------------------------------------------------------------

constexpr const char* kChunkKdaFwdLayoutNames[] = {"BSND", "BNSD", "TND",
                                                   "NTD"};

// V2 的三算子组合（ChunkKdaFwdPrepare + ChunkFwdH + ChunkKdaFwdFinalize）与融合
// 入口共用同一套数学；组合入口在大工作量下更快，但 ChunkFwdH 的耗时对 head 数
// 不敏感，当 (chunk, head) 总工作量偏小时整链会慢于单 kernel 的融合实现。
// 这里只按工作量门控，并与 ctypes 参考
// （_aclnn_ctypes.py 的 _CHUNK_KDA_FWD_V2_MIN_WORK_ITEMS）保持同一条判据，
// 两条后端才会逐位一致。门控值取自 A2 实测：head 数 16、T=8192（2048 work item）
// 时组合略慢，head 数 32 及以上组合领先 15% 以上。
constexpr int64_t kChunkKdaFwdV2MinWorkItems = 4096;
constexpr double kChunkKdaFwdDefaultEpsilon = 1e-6;

constexpr const char* kSchema_chunk_kda_fwd =
    "npu_chunk_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
    "Tensor? A_log, Tensor? dt_bias, Tensor? initial_state, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int layout, float scale, "
    "int chunk_size, bool safe_gate, float lower_bound, "
    "bool use_gate_in_kernel, bool state_v_first, float epsilon, "
    "bool use_qk_l2norm_in_kernel, bool use_beta_sigmoid_in_kernel, "
    "bool allow_neg_eigval, bool use_exp2, bool output_final_state, "
    "bool disable_recompute, bool return_intermediate_states, "
    "Tensor? q_hat_out, Tensor? k_hat_out, Tensor? q_rstd_out, "
    "Tensor? k_rstd_out, Tensor? beta_eff_out, int stream) "
    "-> (Tensor, Tensor?, Tensor?, Tensor, Tensor, Tensor?, Tensor?, Tensor?, "
    "Tensor?, Tensor?, Tensor?)";

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>, Tensor, Tensor,
           std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>>
run_npu_chunk_kda_fwd(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
    std::optional<Tensor> A_log, std::optional<Tensor> dt_bias,
    std::optional<Tensor> initial_state,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t layout, double scale, int64_t chunk_size, bool safe_gate,
    double lower_bound, bool use_gate_in_kernel, bool state_v_first,
    double epsilon, bool use_qk_l2norm_in_kernel,
    bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, bool use_exp2,
    bool output_final_state, bool disable_recompute,
    bool return_intermediate_states,
    std::optional<Tensor> q_hat_out, std::optional<Tensor> k_hat_out,
    std::optional<Tensor> q_rstd_out, std::optional<Tensor> k_rstd_out,
    std::optional<Tensor> beta_eff_out, int64_t stream) {
  // 反向 L2 norm 保存值出口：调用方给了才导出，不给就是空槽（nullptr），
  // 因此"不传这五个输出"与改动前的 11 项返回逐位一致。
  const bool wants_saved =
      q_hat_out.has_value() || k_hat_out.has_value() || q_rstd_out.has_value() ||
      k_rstd_out.has_value() || beta_eff_out.has_value();
  namespace layout_math = fla_npu_stable::stable::layout_math;
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  const bool rank3 = layout_math::packed(layout);
  const int64_t tokens = SIZE_OF(q_meta, layout_math::token_axis(layout));
  const int64_t heads = SIZE_OF(v_meta, layout_math::head_axis(layout));
  const int64_t k_dim = SIZE_OF(q_meta, layout_math::dim_axis(layout));
  const int64_t v_dim = SIZE_OF(v_meta, layout_math::dim_axis(layout));
  // A packed spelling has no batch dimension; the shape math still wants one.
  const int64_t batch_size = rank3 ? 1 : SIZE_OF(q_meta, 0);

  // The head-major spellings put the batch dimension in front of the chunk
  // count; the packed ones do not have one.
  std::vector<int64_t> leading;
  if (!rank3) {
    leading.push_back(batch_size);
  }
  const int64_t q_dtype = q_meta.scalar_type;

  auto head_sizes = [&](int64_t channel_dim,
                        std::vector<int64_t> prefix) {
    prefix.insert(prefix.end(), {heads, tokens, channel_dim});
    return prefix;
  };

  Tensor out_attn = allocate_sizes(
      rank3 ? std::vector<int64_t>{tokens, heads, v_dim}
            : std::vector<int64_t>{batch_size, tokens, heads, v_dim},
      q_dtype, q_meta);
  std::optional<Tensor> out_final_state;
  if (output_final_state) {
    out_final_state = allocate_sizes(
        {layout_math::sequences(cu, batch_size), heads,
         state_v_first ? v_dim : k_dim, state_v_first ? k_dim : v_dim},
        kFloat, q_meta);
  }
  std::optional<Tensor> out_gk;
  if (!use_gate_in_kernel || disable_recompute) {
    out_gk = allocate_sizes(head_sizes(k_dim, leading), kFloat, q_meta);
  }
  Tensor out_aqk = allocate_sizes(head_sizes(chunk_size, leading), q_dtype,
                                  q_meta);
  Tensor out_akk = allocate_sizes(head_sizes(chunk_size, leading), q_dtype,
                                  q_meta);
  std::optional<Tensor> out_w;
  std::optional<Tensor> out_u;
  std::optional<Tensor> out_qg;
  std::optional<Tensor> out_kg;
  std::optional<Tensor> out_v_new;
  if (disable_recompute) {
    out_w = allocate_sizes(head_sizes(k_dim, leading), q_dtype, q_meta);
    out_u = allocate_sizes(head_sizes(v_dim, leading), q_dtype, q_meta);
    out_qg = allocate_sizes(head_sizes(k_dim, leading), q_dtype, q_meta);
    out_kg = allocate_sizes(head_sizes(k_dim, leading), q_dtype, q_meta);
    out_v_new = allocate_sizes(head_sizes(v_dim, leading), q_dtype, q_meta);
  }
  std::optional<Tensor> out_h;
  if (disable_recompute || return_intermediate_states) {
    std::vector<int64_t> h_sizes = leading;
    h_sizes.insert(h_sizes.end(),
                   {layout_math::chunks(cu, ci, chunk_size, tokens), heads,
                    state_v_first ? v_dim : k_dim,
                    state_v_first ? k_dim : v_dim});
    out_h = allocate_sizes(h_sizes, q_dtype, q_meta);
  }

  // 场景选择：命中三个独立算子的组合场景且工作量足够时走 aclnnChunkKdaFwdV2，
  // 其余场景回落到签名未变的 aclnnChunkKdaFwd（私有 L0 融合实现）。
  // 非默认 gate/L2norm 开关只有组合入口支持，此时必须命中组合场景（wrapper
  // 已在 Python 侧按参考实现拦截非法组合）。
  bool cu_strictly_increasing = true;
  for (size_t idx = 0; idx + 1 < cu.size(); ++idx) {
    if (cu[idx] >= cu[idx + 1]) {
      cu_strictly_increasing = false;
      break;
    }
  }
  const bool switches_requested =
      epsilon != kChunkKdaFwdDefaultEpsilon || use_qk_l2norm_in_kernel ||
      use_beta_sigmoid_in_kernel || allow_neg_eigval || !use_exp2;
  const bool v2_scenario =
      q_meta.scalar_type == kBFloat16 && k_dim == 128 && v_dim == 128 &&
      chunk_size == 64 && cu_strictly_increasing;
  const int64_t work_items =
      heads * layout_math::chunks(cu, ci, chunk_size, tokens);
  const bool use_v2 = v2_scenario &&
                      (switches_requested ||
                       work_items >= kChunkKdaFwdV2MinWorkItems);

  if (use_v2) {
    FLA_STABLE_EXEC(
        "aclnnChunkKdaFwdV2", q_meta, stream, tensor(q_meta),
        tensor(meta_of(k)), tensor(v_meta), tensor(meta_of(g)),
        tensor(meta_of(beta)), optional_tensor(A_log), optional_tensor(dt_bias),
        optional_tensor(initial_state), int_array(cu), int_array(ci),
        cstr(kChunkKdaFwdLayoutNames, layout), scalar(scale),
        scalar(chunk_size), scalar(safe_gate), scalar(lower_bound),
        scalar(use_gate_in_kernel), scalar(state_v_first), scalar(epsilon),
        scalar(use_qk_l2norm_in_kernel), scalar(use_beta_sigmoid_in_kernel),
        scalar(allow_neg_eigval), scalar(use_exp2),
        out_tensor(meta_of(out_attn)),
        out_tensor(out_final_state.has_value() ? meta_of(*out_final_state)
                                               : TensorMeta()),
        out_tensor(out_gk.has_value() ? meta_of(*out_gk) : TensorMeta()),
        out_tensor(meta_of(out_aqk)), out_tensor(meta_of(out_akk)),
        out_tensor(out_w.has_value() ? meta_of(*out_w) : TensorMeta()),
        out_tensor(out_u.has_value() ? meta_of(*out_u) : TensorMeta()),
        out_tensor(out_qg.has_value() ? meta_of(*out_qg) : TensorMeta()),
        out_tensor(out_kg.has_value() ? meta_of(*out_kg) : TensorMeta()),
        out_tensor(out_v_new.has_value() ? meta_of(*out_v_new) : TensorMeta()),
        out_tensor(out_h.has_value() ? meta_of(*out_h) : TensorMeta()),
        out_tensor(q_hat_out.has_value() ? meta_of(*q_hat_out) : TensorMeta()),
        out_tensor(k_hat_out.has_value() ? meta_of(*k_hat_out) : TensorMeta()),
        out_tensor(q_rstd_out.has_value() ? meta_of(*q_rstd_out) : TensorMeta()),
        out_tensor(k_rstd_out.has_value() ? meta_of(*k_rstd_out) : TensorMeta()),
        out_tensor(beta_eff_out.has_value() ? meta_of(*beta_eff_out)
                                           : TensorMeta()));
    return std::make_tuple(out_attn, out_final_state, out_gk, out_aqk, out_akk,
                           out_w, out_u, out_qg, out_kg, out_v_new, out_h);
  }

  // 保存值只有组合入口（V2）产出；落在融合入口的场景里给了输出槽就拒绝，
  // 与 ctypes 后端同一句话（_aclnn_ctypes.py 的同名判据）。
  if (wants_saved) {
    throw std::runtime_error(
        "npu_chunk_kda_fwd: q_hat/k_hat/q_rstd/k_rstd/beta_eff are only "
        "exported by the three-stage entry (bfloat16, K=V=128, "
        "chunk_size=64); do not pass these outputs in the current scenario.");
  }

  FLA_STABLE_EXEC(
      "aclnnChunkKdaFwd", q_meta, stream, tensor(q_meta), tensor(meta_of(k)),
      tensor(v_meta), tensor(meta_of(g)), tensor(meta_of(beta)),
      optional_tensor(A_log), optional_tensor(dt_bias),
      optional_tensor(initial_state), int_array(cu), int_array(ci),
      cstr(kChunkKdaFwdLayoutNames, layout), scalar(scale),
      scalar(chunk_size), scalar(safe_gate), scalar(lower_bound),
      scalar(use_gate_in_kernel), scalar(state_v_first),
      out_tensor(meta_of(out_attn)),
      out_tensor(out_final_state.has_value() ? meta_of(*out_final_state)
                                             : TensorMeta()),
      out_tensor(out_gk.has_value() ? meta_of(*out_gk) : TensorMeta()),
      out_tensor(meta_of(out_aqk)), out_tensor(meta_of(out_akk)),
      out_tensor(out_w.has_value() ? meta_of(*out_w) : TensorMeta()),
      out_tensor(out_u.has_value() ? meta_of(*out_u) : TensorMeta()),
      out_tensor(out_qg.has_value() ? meta_of(*out_qg) : TensorMeta()),
      out_tensor(out_kg.has_value() ? meta_of(*out_kg) : TensorMeta()),
      out_tensor(out_v_new.has_value() ? meta_of(*out_v_new) : TensorMeta()),
      out_tensor(out_h.has_value() ? meta_of(*out_h) : TensorMeta()));
  return std::make_tuple(out_attn, out_final_state, out_gk, out_aqk, out_akk,
                         out_w, out_u, out_qg, out_kg, out_v_new, out_h);
}

}  // namespace
