// Stable-ABI adapter for npu_causal_conv1d_update.
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

// Decode: runMode 1, no padding (the reference passes INT64_MIN for
// padSlotId).

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
// decode
// ---------------------------------------------------------------------------

constexpr const char* kSchema_causal_conv1d_update =
    "npu_causal_conv1d_update(Tensor x, Tensor conv_state, Tensor weight, "
    "Tensor? bias, int activation, Tensor? conv_state_indices, "
    "Tensor? num_accepted_tokens, Tensor? query_start_loc, int max_query_len, "
    "int null_block_id, Tensor? conv_state_indices_cpu, "
    "Tensor? num_accepted_tokens_cpu, Tensor? query_start_loc_cpu, Tensor? out, "
    "int stream) "
    "-> Tensor";

Tensor run_npu_causal_conv1d_update(
    Tensor x, Tensor conv_state, Tensor weight, std::optional<Tensor> bias,
    int64_t activation, std::optional<Tensor> conv_state_indices,
    std::optional<Tensor> num_accepted_tokens,
    std::optional<Tensor> query_start_loc, int64_t max_query_len,
    int64_t null_block_id, std::optional<Tensor> conv_state_indices_cpu,
    std::optional<Tensor> num_accepted_tokens_cpu,
    std::optional<Tensor> query_start_loc_cpu, std::optional<Tensor> out,
    int64_t stream) {
  // UPDATE never pads, which the reference spells as INT64_MIN rather than -1.
  constexpr int64_t kNoPadding = -9223372036854775807LL - 1;
  return launch(x, weight, bias, conv_state, query_start_loc,
                conv_state_indices, /*has_initial_state=*/std::nullopt,
                num_accepted_tokens, query_start_loc_cpu,
                conv_state_indices_cpu, /*has_initial_state_cpu=*/std::nullopt,
                num_accepted_tokens_cpu, activation, kNoPadding, null_block_id,
                /*run_mode=*/1, /*head_num=*/0, max_query_len, stream, out);
}

}  // namespace
