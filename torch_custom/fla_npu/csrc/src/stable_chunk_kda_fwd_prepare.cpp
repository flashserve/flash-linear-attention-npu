// Stable-ABI adapter for npu_chunk_kda_fwd_prepare.
// aclnn: aclnnChunkKdaFwdPrepare
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

// 13 个输出槽全部可选，由 backward_mode（none/forward/recompute/save）决定调用方
// 给哪几个；输入按逻辑 ND 形状交给 aclnn，Prepare 的 tiling 按逻辑 shape 校验。

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
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_logical_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;

// ---------------------------------------------------------------------------
// npu_chunk_kda_fwd_prepare
// ---------------------------------------------------------------------------

// 与 `_stable._ENUM` 的 layout 表同序；表名带算子前缀，避免与同 TU 里
// `stable_chunk_kda_fwd.cpp` 的名表重名。
constexpr const char* kChunkKdaFwdPrepareLayoutNames[] = {"BSND", "BNSD", "TND",
                                                          "NTD"};

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
      int_array(cu), int_array(ci), cstr(kChunkKdaFwdPrepareLayoutNames, layout),
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

}  // namespace
