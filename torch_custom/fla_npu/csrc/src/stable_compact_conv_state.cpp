// Stable-ABI adapter for npu_compact_conv_state.
// aclnn: aclnnCompactConvState

#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
using fla_npu_stable::stable::kAclFormatNd;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::TensorArg;
using fla_npu_stable::stable::TensorMeta;
using torch::stable::Tensor;

constexpr const char *kSchemaCompactConvState =
    "npu_compact_conv_state(Tensor(a!) conv_state, Tensor conv_state_base_addrs, "
    "Tensor conv_state_block_strides, Tensor conv_state_dim_strides, "
    "Tensor conv_state_token_strides, Tensor state_indices, Tensor commit_lens, "
    "Tensor final_state_indices, Tensor boundary_state_indices, "
    "Tensor boundary_recovery_lens, int conv_history_len, bool align_mode, "
    "int dim_stride, int null_block_id, int stream) -> Tensor";

Tensor run_compact_conv_state(Tensor conv_state, Tensor base_addrs, Tensor block_strides, Tensor dim_strides,
                              Tensor token_strides, Tensor state_indices, Tensor commit_lens,
                              Tensor final_state_indices, Tensor boundary_state_indices, Tensor boundary_recovery_lens,
                              int64_t conv_history_len, bool align_mode, int64_t dim_stride, int64_t null_block_id,
                              int64_t stream)
{
    const TensorMeta state_meta = meta_of(conv_state);
    const int32_t npu_device_type = aoti_torch_device_type_privateuse1();
    const auto check_device = [&](const TensorMeta &meta, const char *name) {
        constexpr const char *prefix = "fla_npu(stable): npu_compact_conv_state: ";
        if (meta.device_type != npu_device_type) {
            throw std::runtime_error(std::string(prefix) + name + " must be on NPU");
        }
        if (meta.device_index != state_meta.device_index) {
            throw std::runtime_error(std::string(prefix) + name + " must be on the same device as conv_state");
        }
    };
    check_device(state_meta, "conv_state");
    const TensorMeta base_meta = meta_of(base_addrs);
    const TensorMeta block_meta = meta_of(block_strides);
    const TensorMeta dim_meta = meta_of(dim_strides);
    const TensorMeta token_meta = meta_of(token_strides);
    const TensorMeta source_meta = meta_of(state_indices);
    const TensorMeta lengths_meta = meta_of(commit_lens);
    const TensorMeta final_meta = meta_of(final_state_indices);
    const TensorMeta boundary_meta = meta_of(boundary_state_indices);
    const TensorMeta recovery_meta = meta_of(boundary_recovery_lens);
    for (const auto &item : {std::pair<TensorMeta, const char *>{base_meta, "conv_state_base_addrs"},
                             std::pair<TensorMeta, const char *>{block_meta, "conv_state_block_strides"},
                             std::pair<TensorMeta, const char *>{dim_meta, "conv_state_dim_strides"},
                             std::pair<TensorMeta, const char *>{token_meta, "conv_state_token_strides"},
                             std::pair<TensorMeta, const char *>{source_meta, "state_indices"},
                             std::pair<TensorMeta, const char *>{lengths_meta, "commit_lens"},
                             std::pair<TensorMeta, const char *>{final_meta, "final_state_indices"},
                             std::pair<TensorMeta, const char *>{boundary_meta, "boundary_state_indices"},
                             std::pair<TensorMeta, const char *>{recovery_meta, "boundary_recovery_lens"}}) {
        check_device(item.first, item.second);
    }
    FLA_STABLE_EXEC("aclnnCompactConvState", state_meta, stream, TensorArg(state_meta, kAclFormatNd),
                    TensorArg(base_meta, kAclFormatNd), TensorArg(block_meta, kAclFormatNd),
                    TensorArg(dim_meta, kAclFormatNd), TensorArg(token_meta, kAclFormatNd),
                    TensorArg(source_meta, kAclFormatNd), TensorArg(lengths_meta, kAclFormatNd),
                    TensorArg(final_meta, kAclFormatNd), TensorArg(boundary_meta, kAclFormatNd),
                    TensorArg(recovery_meta, kAclFormatNd), scalar(conv_history_len), scalar(align_mode),
                    scalar(dim_stride), scalar(null_block_id), nd_tensor(state_meta));
    return conv_state;
}
} // namespace
