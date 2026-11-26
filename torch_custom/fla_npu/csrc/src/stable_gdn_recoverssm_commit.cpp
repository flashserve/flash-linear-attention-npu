// Stable-ABI adapter for npu_gdn_recoverssm_commit.
// aclnn: aclnnInnerGdnRecoverssmCommit

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

constexpr const char *kSchemaGdnRecoverssmCommit =
    "npu_gdn_recoverssm_commit(Tensor(a!) state, Tensor replay, "
    "Tensor state_commit_cache_address_stride_table, Tensor state_indices, "
    "Tensor commit_lens, Tensor final_state_indices, "
    "Tensor boundary_state_indices, Tensor boundary_recovery_lens, "
    "int spec_query_len, bool align_mode, int stream) -> Tensor";

Tensor run_gdn_recoverssm_commit(Tensor state, Tensor replay, Tensor state_commit_cache_address_stride_table,
                                 Tensor state_indices, Tensor commit_lens, Tensor final_state_indices,
                                 Tensor boundary_state_indices, Tensor boundary_recovery_lens, int64_t spec_query_len,
                                 bool align_mode, int64_t stream)
{
    const TensorMeta state_meta = meta_of(state);
    const int32_t npu_device_type = aoti_torch_device_type_privateuse1();
    const auto check_device = [&](const TensorMeta &meta, const char *name) {
        constexpr const char *prefix = "fla_npu(stable): npu_gdn_recoverssm_commit: ";
        if (meta.device_type != npu_device_type) {
            throw std::runtime_error(std::string(prefix) + name + " must be on NPU");
        }
        if (meta.device_index != state_meta.device_index) {
            throw std::runtime_error(std::string(prefix) + name + " must be on the same device as state");
        }
    };
    check_device(state_meta, "state");
    const TensorMeta replay_meta = meta_of(replay);
    const TensorMeta table_meta = meta_of(state_commit_cache_address_stride_table);
    const TensorMeta source_meta = meta_of(state_indices);
    const TensorMeta lengths_meta = meta_of(commit_lens);
    const TensorMeta final_meta = meta_of(final_state_indices);
    const TensorMeta boundary_meta = meta_of(boundary_state_indices);
    const TensorMeta recovery_meta = meta_of(boundary_recovery_lens);
    for (const auto &item : {std::pair<TensorMeta, const char *>{replay_meta, "replay"},
                             std::pair<TensorMeta, const char *>{table_meta, "state_commit_cache_address_stride_table"},
                             std::pair<TensorMeta, const char *>{source_meta, "state_indices"},
                             std::pair<TensorMeta, const char *>{lengths_meta, "commit_lens"},
                             std::pair<TensorMeta, const char *>{final_meta, "final_state_indices"},
                             std::pair<TensorMeta, const char *>{boundary_meta, "boundary_state_indices"},
                             std::pair<TensorMeta, const char *>{recovery_meta, "boundary_recovery_lens"}}) {
        check_device(item.first, item.second);
    }
    FLA_STABLE_EXEC("aclnnInnerGdnRecoverssmCommit", state_meta, stream, TensorArg(state_meta, kAclFormatNd),
                    TensorArg(replay_meta, kAclFormatNd), TensorArg(table_meta, kAclFormatNd),
                    TensorArg(source_meta, kAclFormatNd), TensorArg(lengths_meta, kAclFormatNd),
                    TensorArg(final_meta, kAclFormatNd), TensorArg(boundary_meta, kAclFormatNd),
                    TensorArg(recovery_meta, kAclFormatNd), scalar(spec_query_len), scalar(align_mode),
                    nd_tensor(state_meta));
    return state;
}
} // namespace
