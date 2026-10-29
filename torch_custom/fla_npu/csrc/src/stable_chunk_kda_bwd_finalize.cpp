// Stable-ABI adapter for the unified ChunkKdaBwdFinalize interface.
#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using torch::stable::Tensor;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_optional_tensor;
using fla_npu_stable::stable::nd_logical_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::at_shim::kBFloat16;
using fla_npu_stable::stable::at_shim::kFloat;

constexpr const char* kFinalizeOutputLayoutNames[] = {
    "BSND", "BNSD", "TND", "NTD"};
constexpr const char* kSchema_chunk_kda_bwd_finalize =
    "npu_chunk_kda_bwd_finalize(Tensor q, Tensor k, Tensor v, Tensor gk, Tensor raw_g, "
    "Tensor beta, Tensor a_log, Tensor dt_bias, Tensor akk, Tensor v_new, Tensor h, Tensor dh, "
    "Tensor dv_scan, Tensor d_aqk, Tensor dq_raw, Tensor? q_rstd, Tensor? k_rstd, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, float scale, float lower_bound, int chunk_size, "
    "bool safe_gate, bool use_gate_in_kernel, bool use_exp2, bool state_v_first, "
    "bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, int output_layout, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> run_npu_chunk_kda_bwd_finalize(
    Tensor q, Tensor k, Tensor v, Tensor gk, Tensor raw_g, Tensor beta, Tensor a_log, Tensor dt_bias,
    Tensor akk, Tensor v_new, Tensor h, Tensor dh, Tensor dv_scan, Tensor d_aqk, Tensor dq_raw,
    std::optional<Tensor> q_rstd, std::optional<Tensor> k_rstd,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    double scale, double lower_bound, int64_t chunk_size, bool safe_gate, bool use_gate_in_kernel,
    bool use_exp2, bool state_v_first, bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval,
    int64_t output_layout, int64_t stream)
{
    const auto meta = meta_of(q);
    auto qkvShape = meta.sizes;
    if (meta.sizes.size() != 3 && meta.sizes.size() != 4) {
        throw std::invalid_argument("finalize q must have rank 3 or 4");
    }
    const size_t first = meta.sizes.size() == 3 ? 0 : 1;
    if (output_layout == 0 || output_layout == 2) {
        std::swap(qkvShape.at(first), qkvShape.at(first + 1));
    }
    auto betaShape = qkvShape;
    betaShape.pop_back();
    Tensor dq = allocate_sizes(qkvShape, kBFloat16, meta);
    Tensor dk = allocate_sizes(qkvShape, kBFloat16, meta);
    Tensor dv = allocate_sizes(qkvShape, kBFloat16, meta);
    Tensor db = allocate_sizes(betaShape, meta_of(beta).scalar_type, meta);
    Tensor dg = allocate_sizes(qkvShape, meta_of(raw_g).scalar_type, meta);
    Tensor da = allocate_sizes(meta_of(a_log).sizes, kFloat, meta);
    Tensor dbias = allocate_sizes(meta_of(dt_bias).sizes, kFloat, meta);
    FLA_STABLE_EXEC(
        "aclnnChunkKdaBwdFinalize", meta, stream,
        nd_tensor(meta), nd_tensor(meta_of(k)), nd_tensor(meta_of(v)), nd_tensor(meta_of(gk)),
        nd_tensor(meta_of(raw_g)), nd_tensor(meta_of(beta)), nd_tensor(meta_of(a_log)), nd_tensor(meta_of(dt_bias)),
        nd_tensor(meta_of(akk)), nd_tensor(meta_of(v_new)), nd_tensor(meta_of(h)), nd_tensor(meta_of(dh)),
        nd_tensor(meta_of(dv_scan)), nd_tensor(meta_of(d_aqk)), nd_tensor(meta_of(dq_raw)),
        nd_optional_tensor(q_rstd), nd_optional_tensor(k_rstd), int_array(cu_seqlens), int_array(chunk_indices),
        scalar(scale), scalar(lower_bound), scalar(chunk_size), scalar(safe_gate), scalar(use_gate_in_kernel),
        scalar(use_exp2), scalar(state_v_first), scalar(use_beta_sigmoid_in_kernel), scalar(allow_neg_eigval),
        cstr(kFinalizeOutputLayoutNames, output_layout), nd_logical_out_tensor(meta_of(dq)), nd_logical_out_tensor(meta_of(dk)),
        nd_logical_out_tensor(meta_of(dv)), nd_logical_out_tensor(meta_of(db)), nd_logical_out_tensor(meta_of(dg)),
        nd_logical_out_tensor(meta_of(da)), nd_logical_out_tensor(meta_of(dbias)));
    return std::make_tuple(dq, dk, dv, db, dg, da, dbias);
}
} // namespace
