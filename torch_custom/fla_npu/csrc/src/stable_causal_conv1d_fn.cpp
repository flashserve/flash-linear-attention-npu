// Stable-ABI adapter for npu_causal_conv1d_fn.
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

// Prefill: runMode 0, headNum forwarded, and the device tensor metadata
// travels together with the host twins the wrapper also passes.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::CStrArg;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::enum_name;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// prefill
// ---------------------------------------------------------------------------

constexpr const char* kSchema_causal_conv1d_fn =
    "npu_causal_conv1d_fn(Tensor x, Tensor weight, Tensor? bias, "
    "Tensor? conv_states, Tensor? query_start_loc, Tensor? cache_indices, "
    "Tensor? has_initial_state, Tensor? query_start_loc_cpu, "
    "Tensor? cache_indices_cpu, Tensor? has_initial_state_cpu, int activation, "
    "int pad_slot_id, int null_block_id, int head_num, int stream) -> Tensor";

Tensor run_npu_causal_conv1d_fn(
    Tensor x, Tensor weight, std::optional<Tensor> bias,
    std::optional<Tensor> conv_states, std::optional<Tensor> query_start_loc,
    std::optional<Tensor> cache_indices,
    std::optional<Tensor> has_initial_state,
    std::optional<Tensor> query_start_loc_cpu,
    std::optional<Tensor> cache_indices_cpu,
    std::optional<Tensor> has_initial_state_cpu, int64_t activation,
    int64_t pad_slot_id, int64_t null_block_id, int64_t head_num,
    int64_t stream) {
  return launch(x, weight, bias, conv_states, query_start_loc, cache_indices,
                has_initial_state, /*num_accepted_tokens=*/std::nullopt,
                query_start_loc_cpu, cache_indices_cpu, has_initial_state_cpu,
                /*num_accepted_tokens_cpu=*/std::nullopt, activation,
                pad_slot_id, null_block_id, /*run_mode=*/0, head_num,
                /*max_query_len=*/kNoQueryLenBound, stream);
}

}  // namespace
