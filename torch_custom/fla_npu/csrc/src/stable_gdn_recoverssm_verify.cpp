/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace {
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::kAclFormatNd;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::TensorArg;
using fla_npu_stable::stable::TensorMeta;
using torch::stable::Tensor;

constexpr const char *kSchemaGdnRecoverssmVerify =
    "npu_gdn_recoverssm_verify(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, "
    "Tensor q, Tensor k, Tensor v, Tensor checkpoint_state, "
    "Tensor(a!) replay_cache, Tensor query_start_loc, Tensor state_indices, "
    "float scale, float softplus_beta, float softplus_threshold, "
    "bool use_qk_l2norm_in_kernel, Tensor? out, int stream) -> Tensor";

Tensor run_gdn_recoverssm_verify(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, Tensor v,
                                 Tensor checkpoint_state, Tensor replay_cache, Tensor query_start_loc,
                                 Tensor state_indices, double scale, double softplus_beta, double softplus_threshold,
                                 bool use_qk_l2norm_in_kernel, std::optional<Tensor> out, int64_t stream)
{
    const TensorMeta q_meta = meta_of(q);
    const int32_t npu_device_type = aoti_torch_device_type_privateuse1();
    const auto check_device = [&](const TensorMeta &meta, const char *name) {
        if (!meta.defined) {
            return;
        }
        constexpr const char *prefix = "fla_npu(stable): npu_gdn_recoverssm_verify: ";
        if (meta.device_type != npu_device_type) {
            throw std::runtime_error(std::string(prefix) + name +
                                     " must be on NPU (got device_type=" + std::to_string(meta.device_type) + ")");
        }
        if (meta.device_index != q_meta.device_index) {
            throw std::runtime_error(std::string(prefix) + name + " must be on the same device as q");
        }
    };
    check_device(q_meta, "q");
    const TensorMeta a_log_meta = meta_of(A_log);
    const TensorMeta a_meta = meta_of(a);
    const TensorMeta b_meta = meta_of(b);
    const TensorMeta bias_meta = meta_of(dt_bias);
    const TensorMeta k_meta = meta_of(k);
    const TensorMeta v_meta = meta_of(v);
    const TensorMeta checkpoint_meta = meta_of(checkpoint_state);
    const TensorMeta replay_meta = meta_of(replay_cache);
    const TensorMeta starts_meta = meta_of(query_start_loc);
    const TensorMeta indices_meta = meta_of(state_indices);
    const TensorMeta out_meta = out ? meta_of(*out) : TensorMeta();
    for (const auto &item :
         {std::pair<TensorMeta, const char *>{a_log_meta, "A_log"}, std::pair<TensorMeta, const char *>{a_meta, "a"},
          std::pair<TensorMeta, const char *>{b_meta, "b"}, std::pair<TensorMeta, const char *>{bias_meta, "dt_bias"},
          std::pair<TensorMeta, const char *>{k_meta, "k"}, std::pair<TensorMeta, const char *>{v_meta, "v"},
          std::pair<TensorMeta, const char *>{checkpoint_meta, "checkpoint_state"},
          std::pair<TensorMeta, const char *>{replay_meta, "replay_cache"},
          std::pair<TensorMeta, const char *>{starts_meta, "query_start_loc"},
          std::pair<TensorMeta, const char *>{indices_meta, "state_indices"},
          std::pair<TensorMeta, const char *>{out_meta, "out"}}) {
        check_device(item.first, item.second);
    }
    Tensor result = out ? *out : allocate_like(v_meta);
    FLA_STABLE_EXEC(
        "GdnRecoverssmVerifyInner", v_meta, stream, TensorArg(a_log_meta, kAclFormatNd),
        TensorArg(a_meta, kAclFormatNd), TensorArg(b_meta, kAclFormatNd), TensorArg(bias_meta, kAclFormatNd),
        TensorArg(q_meta, kAclFormatNd), TensorArg(k_meta, kAclFormatNd), TensorArg(v_meta, kAclFormatNd),
        TensorArg(checkpoint_meta, kAclFormatNd), TensorArg(replay_meta, kAclFormatNd),
        TensorArg(starts_meta, kAclFormatNd), TensorArg(indices_meta, kAclFormatNd), scalar(static_cast<float>(scale)),
        scalar(static_cast<float>(softplus_beta)), scalar(static_cast<float>(softplus_threshold)),
        scalar(use_qk_l2norm_in_kernel), nd_out_tensor(meta_of(result)));
    return result;
}
} // namespace
