#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include <optional>
#include <tuple>

namespace {
using torch::stable::Tensor;
using namespace fla_npu_stable::stable;
constexpr int32_t kKdaCpuDevice = 0;
constexpr int32_t kKdaInt32 = 3;
constexpr int32_t kKdaInt64 = 4;
constexpr const char* kSchema_mega_recurrent_kda =
    "npu_mega_recurrent_kda(Tensor hidden_states, Tensor qkv_projection_weight, "
    "Tensor decay_projection_a_weight, Tensor decay_projection_b_weight, Tensor beta_projection_weight, "
    "Tensor output_gate_projection_weight, Tensor output_norm_weight, Tensor output_projection_weight, "
    "Tensor conv1d_weight, Tensor(a!) conv_state, Tensor checkpoint_state, Tensor(b!) replay_u, "
    "Tensor(c!) replay_k, Tensor(d!) replay_decay, Tensor state_indices, Tensor conv_num_accepted_tokens, "
    "Tensor a_log, Tensor dt_bias, Tensor? cu_seqlens, float scale, float lower_bound, float rms_norm_eps, "
    "int stream) -> (Tensor, Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))";

constexpr const char *kSchema_mega_recurrent_kda_v2 =
    "npu_mega_recurrent_kda_v2(Tensor hidden_states, Tensor "
    "qkv_projection_weight, "
    "Tensor decay_projection_a_weight, Tensor decay_projection_b_weight, "
    "Tensor beta_projection_weight, "
    "Tensor output_gate_projection_weight, Tensor output_norm_weight, Tensor "
    "output_projection_weight, "
    "Tensor conv1d_weight, Tensor(a!) conv_state, Tensor checkpoint_state, "
    "Tensor(b!) replay_u, "
    "Tensor(c!) replay_k, Tensor(d!) replay_decay, Tensor state_indices, "
    "Tensor conv_num_accepted_tokens, "
    "Tensor a_log, Tensor dt_bias, Tensor? cu_seqlens, Tensor? plan, float "
    "scale, float lower_bound, float rms_norm_eps, "
    "int stream) -> (Tensor, Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))";

constexpr const char *kSchema_mega_recurrent_kda_mx =
    "npu_mega_recurrent_kda_mx(Tensor hidden_states, Tensor "
    "qkv_projection_weight, "
    "Tensor decay_projection_a_weight, Tensor decay_projection_b_weight, "
    "Tensor beta_projection_weight, "
    "Tensor output_gate_projection_weight, Tensor output_norm_weight, Tensor "
    "output_projection_weight, "
    "Tensor conv1d_weight, Tensor(a!) conv_state, Tensor checkpoint_state, "
    "Tensor(b!) replay_u, "
    "Tensor(c!) replay_k, Tensor(d!) replay_decay, Tensor state_indices, "
    "Tensor conv_num_accepted_tokens, "
    "Tensor a_log, Tensor dt_bias, Tensor? cu_seqlens, Tensor? plan, Tensor qkv_projection_weight_scale, int qkv_scale_alg, float "
    "scale, float lower_bound, float rms_norm_eps, "
    "int stream) -> (Tensor, Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))";

constexpr const char *kSchema_mega_recurrent_kda_int8 =
    "npu_mega_recurrent_kda_int8(Tensor hidden_states, Tensor "
    "qkv_projection_weight, "
    "Tensor decay_projection_a_weight, Tensor decay_projection_b_weight, "
    "Tensor beta_projection_weight, "
    "Tensor output_gate_projection_weight, Tensor output_norm_weight, Tensor "
    "output_projection_weight, "
    "Tensor conv1d_weight, Tensor(a!) conv_state, Tensor checkpoint_state, "
    "Tensor(b!) replay_u, "
    "Tensor(c!) replay_k, Tensor(d!) replay_decay, Tensor state_indices, "
    "Tensor conv_num_accepted_tokens, "
    "Tensor a_log, Tensor dt_bias, Tensor? cu_seqlens, Tensor? plan, Tensor qkv_projection_weight_scale, float "
    "scale, float lower_bound, float rms_norm_eps, "
    "int stream) -> (Tensor, Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))";

constexpr const char *kSchema_kda_metadata = "npu_mega_recurrent_kda_metadata("
                                             "Tensor cu_seqlens, int tokens, "
                                             "int stream) -> Tensor";

Tensor run_kda_metadata(Tensor cu_seqlens, int64_t tokens, int64_t stream)
{
    if (tokens <= 0 || tokens > INT32_MAX) {
        throw std::invalid_argument("tokens must fit a positive int32");
    }
    const auto prefix = meta_of(cu_seqlens);
    auto out = allocate_sizes({2 + (tokens + 15) / 16}, kKdaInt32, prefix);
    FLA_STABLE_EXEC("aclnnMegaRecurrentKdaMetadata", prefix, stream, nd_tensor(prefix), scalar(tokens),
                    nd_out_tensor(meta_of(out)));
    return out;
}

std::vector<int64_t> ReadKdaMetadata(const Tensor &tensor)
{
    const auto meta = meta_of(tensor);
    if (meta.device_type != kKdaCpuDevice || meta.ndim != 1 || !meta.contiguous ||
        (meta.scalar_type != kKdaInt32 && meta.scalar_type != kKdaInt64)) {
        throw std::invalid_argument("expected contiguous CPU integer metadata");
    }
    if (meta.sizes[0] == 0) {
        return {};
    }
    if (meta.scalar_type == kKdaInt64) {
        auto *values = static_cast<const int64_t *>(meta.data);
        return {values, values + meta.sizes[0]};
    }
    auto *values = static_cast<const int32_t *>(meta.data);
    return {values, values + meta.sizes[0]};
}

template <class Ids, class Counts, class Prefix, class... Extra>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
launch_kda(Tensor hidden_states, TensorArg qkv, Tensor decay_projection_a_weight,
           Tensor decay_projection_b_weight, Tensor beta_projection_weight, Tensor output_gate_projection_weight,
           Tensor output_norm_weight, Tensor output_projection_weight, Tensor conv1d_weight, Tensor conv_state,
           Tensor checkpoint_state, Tensor replay_u, Tensor replay_k, Tensor replay_decay, Ids ids, Counts counts,
           Tensor a_log, Tensor dt_bias, Prefix prefix, double scale, double lower_bound, double rms_norm_eps,
           int64_t stream, const char *symbol, Extra... extra)
{
    const auto hidden = meta_of(hidden_states);
    auto out = allocate_like(hidden);
    FLA_STABLE_EXEC(symbol, hidden, stream, nd_tensor(hidden), qkv,
                    nd_tensor(meta_of(decay_projection_a_weight)), nd_tensor(meta_of(decay_projection_b_weight)),
                    nd_tensor(meta_of(beta_projection_weight)), nd_tensor(meta_of(output_gate_projection_weight)),
                    nd_tensor(meta_of(output_norm_weight)), nd_tensor(meta_of(output_projection_weight)),
                    nd_tensor(meta_of(conv1d_weight)), nd_out_tensor(meta_of(conv_state)),
                    nd_tensor(meta_of(checkpoint_state)), nd_out_tensor(meta_of(replay_u)),
                    nd_out_tensor(meta_of(replay_k)), nd_out_tensor(meta_of(replay_decay)), ids, counts,
                    nd_tensor(meta_of(a_log)), nd_tensor(meta_of(dt_bias)), prefix, extra..., scalar(scale),
                    scalar(lower_bound), scalar(rms_norm_eps), nd_out_tensor(meta_of(out)));
    return {out, conv_state, replay_u, replay_k, replay_decay};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
run_mega_recurrent_kda(Tensor hidden_states, Tensor qkv_projection_weight, Tensor decay_projection_a_weight,
                       Tensor decay_projection_b_weight, Tensor beta_projection_weight,
                       Tensor output_gate_projection_weight, Tensor output_norm_weight, Tensor output_projection_weight,
                       Tensor conv1d_weight, Tensor conv_state, Tensor checkpoint_state, Tensor replay_u,
                       Tensor replay_k, Tensor replay_decay, Tensor state_indices, Tensor conv_num_accepted_tokens,
                       Tensor a_log, Tensor dt_bias, std::optional<Tensor> cu_seqlens, double scale, double lower_bound,
                       double rms_norm_eps, int64_t stream)
{
    return launch_kda(hidden_states, nd_tensor(meta_of(qkv_projection_weight)), decay_projection_a_weight, decay_projection_b_weight,
                      beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                      output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                      replay_decay, int_array(ReadKdaMetadata(state_indices)),
                      int_array(ReadKdaMetadata(conv_num_accepted_tokens)), a_log, dt_bias,
                      int_array(cu_seqlens ? ReadKdaMetadata(*cu_seqlens) : std::vector<int64_t>{}), scale, lower_bound,
                      rms_norm_eps, stream, "aclnnMegaRecurrentKda");
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> run_mega_recurrent_kda_v2(
    Tensor hidden_states, Tensor qkv_projection_weight, Tensor decay_projection_a_weight,
    Tensor decay_projection_b_weight, Tensor beta_projection_weight, Tensor output_gate_projection_weight,
    Tensor output_norm_weight, Tensor output_projection_weight, Tensor conv1d_weight, Tensor conv_state,
    Tensor checkpoint_state, Tensor replay_u, Tensor replay_k, Tensor replay_decay, Tensor state_indices,
    Tensor conv_num_accepted_tokens, Tensor a_log, Tensor dt_bias, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> plan, double scale, double lower_bound, double rms_norm_eps, int64_t stream)
{
    return launch_kda(hidden_states, nd_tensor(meta_of(qkv_projection_weight)), decay_projection_a_weight, decay_projection_b_weight,
                      beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                      output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                      replay_decay, nd_tensor(meta_of(state_indices)), nd_tensor(meta_of(conv_num_accepted_tokens)),
                      a_log, dt_bias, nd_optional_tensor(cu_seqlens), scale, lower_bound, rms_norm_eps, stream,
                      "aclnnMegaRecurrentKdaV2", nd_optional_tensor(plan));
}
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> run_mega_recurrent_kda_mx(
    Tensor hidden_states, Tensor qkv_projection_weight, Tensor decay_projection_a_weight,
    Tensor decay_projection_b_weight, Tensor beta_projection_weight, Tensor output_gate_projection_weight,
    Tensor output_norm_weight, Tensor output_projection_weight, Tensor conv1d_weight, Tensor conv_state,
    Tensor checkpoint_state, Tensor replay_u, Tensor replay_k, Tensor replay_decay, Tensor state_indices,
    Tensor conv_num_accepted_tokens, Tensor a_log, Tensor dt_bias, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> plan, Tensor weight_scale, int64_t mx_alg, double scale, double lower_bound, double rms_norm_eps, int64_t stream)
{
    const auto w = meta_of(qkv_projection_weight), sw = meta_of(weight_scale);
    if (w.scalar_type != 24 || (sw.scalar_type != 0 && sw.scalar_type != 44) ||
        w.ndim != 3 || w.sizes[0] != 1 || w.sizes[1] != 7168 || w.sizes[2] <= 0 || w.sizes[2] % 384 ||
        sw.sizes != std::vector<int64_t>{1,112,w.sizes[2],2} || !w.contiguous || !sw.contiguous ||
        w.storage_offset || sw.device_index != w.device_index || (mx_alg != 0 && mx_alg != 1)) {
        throw std::invalid_argument("invalid MXFP8 QKV weight, paired E8M0 scales or qkv_scale_alg");
    }
    TensorArg qkv(w, 29, acl_dtype(w.scalar_type), {1,w.sizes[2]/32,448,16,32});
    return launch_kda(hidden_states, std::move(qkv), decay_projection_a_weight, decay_projection_b_weight,
                      beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                      output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                      replay_decay, nd_tensor(meta_of(state_indices)), nd_tensor(meta_of(conv_num_accepted_tokens)),
                      a_log, dt_bias, nd_optional_tensor(cu_seqlens), scale, lower_bound, rms_norm_eps, stream,
                      "aclnnMegaRecurrentKdaMx", nd_optional_tensor(plan), TensorArg(sw, kAclFormatNd, 37, sw.sizes), scalar(mx_alg));
}
constexpr const char *kSchema_mega_recurrent_kda_verify_a3 =
    "npu_mega_recurrent_kda_verify_a3(Tensor hidden_states, Tensor "
    "qkv_projection_weight, "
    "Tensor decay_projection_a_weight, Tensor decay_projection_b_weight, "
    "Tensor beta_projection_weight, "
    "Tensor output_gate_projection_weight, Tensor output_norm_weight, Tensor "
    "output_projection_weight, "
    "Tensor conv1d_weight, Tensor(a!) conv_state, Tensor checkpoint_state, "
    "Tensor(b!) replay_u, "
    "Tensor(c!) replay_k, Tensor(d!) replay_decay, Tensor state_indices, "
    "Tensor conv_num_accepted_tokens, "
    "Tensor a_log, Tensor dt_bias, Tensor? cu_seqlens, Tensor? plan, Tensor qkv_projection_weight_scale, float "
    "scale, float lower_bound, float rms_norm_eps, "
    "Tensor source_conv, Tensor starts, Tensor(e!) flag, Tensor(f!) result, int stream) -> Tensor(f!)";

Tensor run_mega_recurrent_kda_verify_a3(
    Tensor hidden_states, Tensor qkv_projection_weight, Tensor decay_projection_a_weight,
    Tensor decay_projection_b_weight, Tensor beta_projection_weight, Tensor output_gate_projection_weight,
    Tensor output_norm_weight, Tensor output_projection_weight, Tensor conv1d_weight, Tensor conv_state,
    Tensor checkpoint_state, Tensor replay_u, Tensor replay_k, Tensor replay_decay, Tensor state_indices,
    Tensor conv_num_accepted_tokens, Tensor a_log, Tensor dt_bias, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> plan, Tensor weight_scale, double scale, double lower_bound, double rms_norm_eps, Tensor source_conv, Tensor starts, Tensor flag, Tensor result, int64_t stream)
{
    auto w = meta_of(qkv_projection_weight);
    const auto sw = meta_of(weight_scale);
    // Explicit packed weights from QuantMatmulWeightNz have four storage axes.
    // Normalise their ACL view to the same logical matrix as an NZ-format Tensor.
    if (w.ndim == 4 && w.sizes[0] > 0 && w.sizes[0] <= INT32_MAX / 32 &&
        w.sizes[1] == 448 && w.sizes[2] == 16 && w.sizes[3] == 32) {
        w.ndim = 2;
        w.sizes = {7168, w.sizes[0] * 32};
        w.strides = {w.sizes[1], 1};
    }
    if (w.scalar_type != 1 || sw.scalar_type != 15 || w.ndim != 2 || w.sizes[0] != 7168 || w.sizes[1] <= 0 ||
        w.sizes[1] % 384 || sw.sizes != std::vector<int64_t>{w.sizes[1]} || !w.contiguous || !sw.contiguous ||
        w.storage_offset || sw.device_index != w.device_index) {
        throw std::invalid_argument("expected packed INT8 NZ QKV weights and one BF16 scale per column");
    }
    TensorArg qkv(w, 29, acl_dtype(w.scalar_type), {w.sizes[1] / 32, 448, 16, 32});
    const auto hidden = meta_of(hidden_states);
    FLA_STABLE_EXEC("aclnnMegaRecurrentKdaVerifyA3", hidden, stream, nd_tensor(hidden), qkv,
                    nd_tensor(meta_of(decay_projection_a_weight)), nd_tensor(meta_of(decay_projection_b_weight)),
                    nd_tensor(meta_of(beta_projection_weight)), nd_tensor(meta_of(output_gate_projection_weight)),
                    nd_tensor(meta_of(output_norm_weight)), nd_tensor(meta_of(output_projection_weight)),
                    nd_tensor(meta_of(conv1d_weight)), nd_out_tensor(meta_of(conv_state)),
                    nd_tensor(meta_of(checkpoint_state)), nd_out_tensor(meta_of(replay_u)),
                    nd_out_tensor(meta_of(replay_k)), nd_out_tensor(meta_of(replay_decay)),
                    nd_tensor(meta_of(state_indices)), nd_tensor(meta_of(conv_num_accepted_tokens)),
                    nd_tensor(meta_of(a_log)), nd_tensor(meta_of(dt_bias)), nd_optional_tensor(cu_seqlens),
                    nd_optional_tensor(plan), nd_tensor(sw), nd_tensor(meta_of(source_conv)), nd_tensor(meta_of(starts)),
                    nd_out_tensor(meta_of(flag)), scalar(scale), scalar(lower_bound), scalar(rms_norm_eps),
                    nd_out_tensor(meta_of(result)));
    return result;
}
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> run_mega_recurrent_kda_int8(
    Tensor hidden_states, Tensor qkv_projection_weight, Tensor decay_projection_a_weight,
    Tensor decay_projection_b_weight, Tensor beta_projection_weight, Tensor output_gate_projection_weight,
    Tensor output_norm_weight, Tensor output_projection_weight, Tensor conv1d_weight, Tensor conv_state,
    Tensor checkpoint_state, Tensor replay_u, Tensor replay_k, Tensor replay_decay, Tensor state_indices,
    Tensor conv_num_accepted_tokens, Tensor a_log, Tensor dt_bias, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> plan, Tensor weight_scale, double scale, double lower_bound, double rms_norm_eps, int64_t stream)
{
    auto w = meta_of(qkv_projection_weight);
    const auto sw = meta_of(weight_scale);
    // Explicit packed weights from QuantMatmulWeightNz have four storage axes.
    // Normalise their ACL view to the same logical matrix as an NZ-format Tensor.
    if (w.ndim == 4 && w.sizes[0] > 0 && w.sizes[0] <= INT32_MAX / 32 &&
        w.sizes[1] == 448 && w.sizes[2] == 16 && w.sizes[3] == 32) {
        w.ndim = 2;
        w.sizes = {7168, w.sizes[0] * 32};
        w.strides = {w.sizes[1], 1};
    }
    if (w.scalar_type != 1 || sw.scalar_type != 15 || w.ndim != 2 || w.sizes[0] != 7168 || w.sizes[1] <= 0 ||
        w.sizes[1] % 384 || sw.sizes != std::vector<int64_t>{w.sizes[1]} || !w.contiguous || !sw.contiguous ||
        w.storage_offset || sw.device_index != w.device_index) {
        throw std::invalid_argument("expected packed INT8 NZ QKV weights and one BF16 scale per column");
    }
    TensorArg qkv(w, 29, acl_dtype(w.scalar_type), {w.sizes[1] / 32, 448, 16, 32});
    return launch_kda(hidden_states, std::move(qkv), decay_projection_a_weight, decay_projection_b_weight,
                      beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                      output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                      replay_decay, nd_tensor(meta_of(state_indices)), nd_tensor(meta_of(conv_num_accepted_tokens)),
                      a_log, dt_bias, nd_optional_tensor(cu_seqlens), scale, lower_bound, rms_norm_eps, stream,
                      "aclnnMegaRecurrentKdaInt8", nd_optional_tensor(plan), nd_tensor(sw));
}
} // namespace
