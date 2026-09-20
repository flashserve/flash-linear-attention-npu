// Stable-ABI adapters: npu_kda_gate_cumsum, npu_chunk_kda_bwd_intra.
//
// These two cover the shapes a KDA-family operator runs into:
//
//   * kda_gate_cumsum -- optional tensors, an int_array, bool/double scalars,
//     and an output whose dtype differs from its source (fp32 over `g`).
//   * chunk_kda_bwd_intra -- ten tensor inputs, two int_arrays, a `char*`
//     enum argument carried as an int code plus a name table, and four outputs
//     that are allocated from their matching inputs.
//
// Included by stable_ops.cpp (single TU); registration lives there.

// Owns the KDA family: npu_chunk_kda_fwd/_bwd/_bwd_intra/_bwd_recompute,
// npu_kda_gate_cumsum.

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
using fla_npu_stable::stable::nd_logical_out_tensor;
using fla_npu_stable::stable::nd_optional_tensor;
using fla_npu_stable::stable::nd_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// The layout arithmetic lives in a named namespace so its helpers cannot
// collide with the adapter-local ones; alias it here because these files sit in
// an anonymous namespace at global scope.
namespace layout_math = fla_npu_stable::stable::layout_math;


// ---------------------------------------------------------------------------
// npu_kda_gate_cumsum
// ---------------------------------------------------------------------------

constexpr const char* kSchema_kda_gate_cumsum =
    "npu_kda_gate_cumsum(Tensor g, Tensor? A_log, Tensor? dt_bias, "
    "Tensor? cu_seqlens, int chunk_size, bool use_gate_in_kernel, "
    "bool safe_gate, float lower_bound, int stream) -> Tensor";

Tensor run_npu_kda_gate_cumsum(Tensor g, std::optional<Tensor> A_log,
                               std::optional<Tensor> dt_bias,
                               std::optional<Tensor> cu_seqlens,
                               int64_t chunk_size, bool use_gate_in_kernel,
                               bool safe_gate, double lower_bound,
                               int64_t stream) {
  const TensorMeta g_meta = meta_of(g);
  // The kernel accumulates in fp32 regardless of `g`'s dtype.
  Tensor out = allocate_sizes(g_meta.sizes, kFloat, g_meta);
  FLA_STABLE_EXEC("aclnnKdaGateCumsum", g_meta, stream, tensor(g_meta),
                  optional_tensor(A_log), optional_tensor(dt_bias),
                  int_array(cu_seqlens), scalar(chunk_size),
                  scalar(use_gate_in_kernel), scalar(safe_gate),
                  scalar(lower_bound), out_tensor(meta_of(out)));
  return out;
}

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
      nd_out_tensor(meta_of(out_dq)), nd_out_tensor(meta_of(out_dk)),
      nd_out_tensor(meta_of(out_db)), nd_out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dq, out_dk, out_db, out_dg);
}

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
  // 反向 L2 norm 保存值出口：调用方给了才导出，不给就是空槽（nullptr）。
  const bool wants_saved =
      q_hat_out.has_value() || k_hat_out.has_value() || q_rstd_out.has_value() ||
      k_rstd_out.has_value() || beta_eff_out.has_value();
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  const int64_t tokens = layout_math::tokens(q_meta, layout);
  const int64_t heads = layout_math::value_heads(v_meta, layout);
  const int64_t k_dim = layout_math::key_dim(q_meta, layout);
  const int64_t v_dim = layout_math::value_dim(v_meta, layout);
  const bool rank3 = layout_math::packed(layout);
  const int64_t batch_size = layout_math::batch(q_meta, layout);

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
        out_tensor(beta_eff_out.has_value() ? meta_of(*beta_eff_out) : TensorMeta()));
    return std::make_tuple(out_attn, out_final_state, out_gk, out_aqk, out_akk,
                           out_w, out_u, out_qg, out_kg, out_v_new, out_h);
  }

  if (wants_saved) {
    throw std::runtime_error(
        "npu_chunk_kda_fwd: q_hat/k_hat/q_rstd/k_rstd/beta_eff 只在三算子组合入口"
        "（bfloat16、K=V=128、chunk_size=64）上产出，当前场景请勿传入这些输出。");
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

// ---------------------------------------------------------------------------
// npu_chunk_kda_fwd_finalize
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// npu_chunk_kda_fwd_prepare
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_kda_fwd_prepare =
    "npu_chunk_kda_fwd_prepare(Tensor q, Tensor k, Tensor v, Tensor g, "
    "Tensor beta, Tensor? A_log, Tensor? dt_bias, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, int layout, float scale, int chunk_size, "
    "float epsilon, bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel, "
    "bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, bool safe_gate, "
    "float lower_bound, bool use_exp2, Tensor? gk_out, Tensor? aqk_out, "
    "Tensor? akk_out, Tensor? w_out, Tensor? u_out, Tensor? qg_out, "
    "Tensor? kg_out, Tensor? qg_scaled_out, Tensor? q_hat_out, "
    "Tensor? k_hat_out, Tensor? q_rstd_out, Tensor? k_rstd_out, "
    "Tensor? beta_eff_out, int stream) "
    "-> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, "
    "Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)";

std::tuple<std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>,
           std::optional<Tensor>>
run_npu_chunk_kda_fwd_prepare(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
    std::optional<Tensor> A_log, std::optional<Tensor> dt_bias,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t layout, double scale, int64_t chunk_size, double epsilon,
    bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, bool safe_gate,
    double lower_bound, bool use_exp2, std::optional<Tensor> gk_out,
    std::optional<Tensor> aqk_out, std::optional<Tensor> akk_out,
    std::optional<Tensor> w_out, std::optional<Tensor> u_out,
    std::optional<Tensor> qg_out, std::optional<Tensor> kg_out,
    std::optional<Tensor> qg_scaled_out, std::optional<Tensor> q_hat_out,
    std::optional<Tensor> k_hat_out, std::optional<Tensor> q_rstd_out,
    std::optional<Tensor> k_rstd_out, std::optional<Tensor> beta_eff_out,
    int64_t stream) {
  // 13 个输出槽全部可选：给了就写进调用方张量，没给就是空槽（nullptr），
  // 因此"不给某个槽"不会报错。op def 侧这些槽仍是 REQUIRED。
  const TensorMeta q_meta = meta_of(q);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  // Prepare 的 tiling 按逻辑 shape 校验；stable 描述符默认带展平 storage shape，
  // 这里用带逻辑 storage 的 ND 拼写（与 finalize 适配同一套 helper）。
  // 注意：L2 只拦私有格式，非私有拼写（NCHW/NCL/NHWC/ND）都接受，这里选 ND
  // 只是本适配层的既有约定，不是算子侧的强制要求。
  const auto out_slot = [](const std::optional<Tensor>& value) {
    return value.has_value() ? nd_logical_out_tensor(meta_of(*value))
                             : out_tensor(TensorMeta());
  };
  FLA_STABLE_EXEC(
      "aclnnChunkKdaFwdPrepare", q_meta, stream, nd_tensor(q_meta),
      nd_tensor(meta_of(k)), nd_tensor(meta_of(v)), nd_tensor(meta_of(g)),
      nd_tensor(meta_of(beta)), optional_tensor(A_log),
      optional_tensor(dt_bias),
      int_array(cu), int_array(ci), cstr(kChunkKdaFwdLayoutNames, layout),
      scalar(scale), scalar(chunk_size), scalar(epsilon),
      scalar(use_qk_l2norm_in_kernel), scalar(use_gate_in_kernel),
      scalar(use_beta_sigmoid_in_kernel), scalar(allow_neg_eigval),
      scalar(safe_gate), scalar(lower_bound), scalar(use_exp2),
      out_slot(gk_out), out_slot(aqk_out), out_slot(akk_out), out_slot(w_out),
      out_slot(u_out), out_slot(qg_out), out_slot(kg_out),
      out_slot(qg_scaled_out), out_slot(q_hat_out), out_slot(k_hat_out),
      out_slot(q_rstd_out), out_slot(k_rstd_out), out_slot(beta_eff_out));
  return std::make_tuple(gk_out, aqk_out, akk_out, w_out, u_out, qg_out, kg_out,
                         qg_scaled_out, q_hat_out, k_hat_out, q_rstd_out,
                         k_rstd_out, beta_eff_out);
}

// ---------------------------------------------------------------------------
// npu_chunk_kda_fwd_finalize
// ---------------------------------------------------------------------------

constexpr const char* kChunkKdaFwdFinalizeLayoutNames[] = {"BSND", "BNSD",
                                                           "TND", "NTD"};

// 只有 attn_out 是输出：它是 sequence-major（BSND/TND）或 head-major
// （BNSD/NTD）的 rank-4，packed 拼写是 rank-3。qg_scaled/aqk/v_new/h 始终是
// head-major，packed 时没有 batch 轴——与 ctypes 参考完全一致。
constexpr const char* kSchema_chunk_kda_fwd_finalize =
    "npu_chunk_kda_fwd_finalize(Tensor qg_scaled, Tensor aqk, Tensor v_new, "
    "Tensor h, Tensor? cu_seqlens, Tensor? chunk_indices, int output_layout, "
    "bool state_v_first, int stream) -> Tensor";

Tensor run_npu_chunk_kda_fwd_finalize(
    Tensor qg_scaled, Tensor aqk, Tensor v_new, Tensor h,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t output_layout, bool state_v_first, int64_t stream) {
  const TensorMeta qg_meta = meta_of(qg_scaled);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  const std::vector<int64_t> ci = int_values(chunk_indices);
  // BSND/TND are sequence-major outputs; BNSD/NTD are head-major.  The inputs
  // are head-major either way, so batch/heads/tokens are read from the input
  // with the head-major spelling and only the output layout is switched.
  const bool packed = layout_math::packed(output_layout);
  const int64_t batch = packed ? 1 : size_of(qg_meta, 0);
  const int64_t heads = packed ? size_of(qg_meta, 0) : size_of(qg_meta, 1);
  const int64_t tokens = packed ? size_of(qg_meta, 1) : size_of(qg_meta, 2);
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
  const int64_t key_heads = size_of(q_meta, q_meta.ndim == 3 ? 0 : 1);
  if (use_gate_in_kernel) {
    out_d_a_log = allocate_sizes({key_heads}, kFloat, q_meta);
    if (dt_bias.has_value()) {
      out_d_dt_bias = allocate_sizes({key_heads, size_of(q_meta, q_meta.ndim - 1)}, kFloat,
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
      nd_out_tensor(meta_of(out_dq)), nd_out_tensor(meta_of(out_dk)),
      nd_out_tensor(meta_of(out_dv)), nd_out_tensor(meta_of(out_db)),
      nd_out_tensor(meta_of(out_dg)),
      /*dh0=*/nd_out_tensor(TensorMeta()),
      nd_out_tensor(out_d_a_log.has_value() ? meta_of(*out_d_a_log)
                                            : TensorMeta()),
      nd_out_tensor(out_d_dt_bias.has_value() ? meta_of(*out_d_dt_bias)
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
  const int64_t key_heads = size_of(q_meta, q_meta.ndim == 3 ? 0 : 1);
  if (use_gate_in_kernel) {
    out_d_a_log = allocate_sizes({key_heads}, kFloat, q_meta);
    if (dt_bias.has_value()) {
      out_d_dt_bias = allocate_sizes({key_heads, size_of(q_meta, 3)}, kFloat,
                                     q_meta);
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
      nd_out_tensor(meta_of(out_dq)), nd_out_tensor(meta_of(out_dk)),
      nd_out_tensor(meta_of(out_dv)), nd_out_tensor(meta_of(out_db)),
      nd_out_tensor(meta_of(out_dg)),
      /*dh0=*/nd_out_tensor(TensorMeta()),
      nd_out_tensor(out_d_a_log.has_value() ? meta_of(*out_d_a_log)
                                            : TensorMeta()),
      nd_out_tensor(out_d_dt_bias.has_value() ? meta_of(*out_d_dt_bias)
                                              : TensorMeta()));
  return std::make_tuple(out_dq, out_dk, out_dv, out_db, out_dg,
                         std::nullopt, out_d_a_log, out_d_dt_bias);
}

}  // namespace
