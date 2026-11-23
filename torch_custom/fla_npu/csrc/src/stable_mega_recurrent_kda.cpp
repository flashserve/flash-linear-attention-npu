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

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
run_mega_recurrent_kda(Tensor hidden_states, Tensor qkv_projection_weight, Tensor decay_projection_a_weight,
                       Tensor decay_projection_b_weight, Tensor beta_projection_weight,
                       Tensor output_gate_projection_weight, Tensor output_norm_weight, Tensor output_projection_weight,
                       Tensor conv1d_weight, Tensor conv_state, Tensor checkpoint_state, Tensor replay_u,
                       Tensor replay_k, Tensor replay_decay, Tensor state_indices, Tensor conv_num_accepted_tokens,
                       Tensor a_log, Tensor dt_bias, std::optional<Tensor> cu_seqlens, double scale, double lower_bound,
                       double rms_norm_eps, int64_t stream)
{
    const auto hidden = meta_of(hidden_states);
    auto out = allocate_like(hidden);
    FLA_STABLE_EXEC(
        "aclnnMegaRecurrentKda", hidden, stream, nd_tensor(hidden), nd_tensor(meta_of(qkv_projection_weight)),
        nd_tensor(meta_of(decay_projection_a_weight)), nd_tensor(meta_of(decay_projection_b_weight)),
        nd_tensor(meta_of(beta_projection_weight)), nd_tensor(meta_of(output_gate_projection_weight)),
        nd_tensor(meta_of(output_norm_weight)), nd_tensor(meta_of(output_projection_weight)),
        nd_tensor(meta_of(conv1d_weight)), nd_out_tensor(meta_of(conv_state)), nd_tensor(meta_of(checkpoint_state)),
        nd_out_tensor(meta_of(replay_u)), nd_out_tensor(meta_of(replay_k)), nd_out_tensor(meta_of(replay_decay)),
        int_array(ReadKdaMetadata(state_indices)), int_array(ReadKdaMetadata(conv_num_accepted_tokens)),
        nd_tensor(meta_of(a_log)), nd_tensor(meta_of(dt_bias)),
        int_array(cu_seqlens ? ReadKdaMetadata(*cu_seqlens) : std::vector<int64_t>{}), scalar(scale),
        scalar(lower_bound), scalar(rms_norm_eps), nd_out_tensor(meta_of(out)));
    return {out, conv_state, replay_u, replay_k, replay_decay};
}
} // namespace
