// Stable-ABI adapters for the causal conv1d family.
//
// aclnn exposes one forward entry point for the three forward APIs; what differs is
// `runMode` (0 = prefill/FN, 1 = decode/UPDATE) plus how the caller's metadata
// is shaped.  The launcher therefore has one C++ implementation and three stable
// entry points:
//
//   * npu_causal_conv1d_fn      -- prefill; runMode 0, headNum forwarded, the
//                                  device tensor metadata plus its host twins.
//   * npu_causal_conv1d_update  -- decode; runMode 1, no padding (the reference
//                                  passes INT64_MIN for padSlotId).
//   * npu_causal_conv1d         -- deprecated compatibility API; runMode and
//                                  headNum come from the caller and every
//                                  metadata slot arrives as a host array.
//   * npu_causal_conv1d_bwd     -- the backward, a different aclnn op that sizes
//                                  its d(initial_state) output from the segment
//                                  count.
//
// The scheduling parameters the reference refuses (block cache / APC /
// metadata objects) are refused by the Python wrappers, so nothing here has to
// describe them.
//
// Included by stable_ops.cpp (single TU); registration lives there.

// Owns the conv1d family: npu_causal_conv1d, npu_causal_conv1d_fn,
// npu_causal_conv1d_update, npu_causal_conv1d_bwd.

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

constexpr const char* kActivationNames[] = {"none", "silu", "swish"};
constexpr const char* kCausalConv1dBwdInputLayoutNames[] = {"BSND", "BNSD",
                                                            "TND", "NTD"};

// maxQueryLen is only meaningful for the varlen update form; the other two
// entry points pass the reference's -1.
constexpr int64_t kNoQueryLenBound = -1;

// Mirrors _infer_causal_conv1d_y: the prefill path with a positive headNum
// returns the head-split view instead of a same-shape copy.
Tensor infer_out(const TensorMeta& x_meta, int64_t head_num, int64_t run_mode) {
  if (run_mode == 0 && head_num > 0) {
    if (x_meta.ndim == 3) {
      return allocate_sizes({size_of(x_meta, 0), head_num, size_of(x_meta, 1),
                             size_of(x_meta, 2) / head_num},
                            x_meta.scalar_type, x_meta);
    }
    if (x_meta.ndim == 2) {
      return allocate_sizes({head_num, size_of(x_meta, 0),
                             size_of(x_meta, 1) / head_num},
                            x_meta.scalar_type, x_meta);
    }
  }
  return allocate_like(x_meta);
}

// The single aclnn argument list, in the order the header declares it.
Tensor launch(Tensor x, Tensor weight, std::optional<Tensor> bias,
              std::optional<Tensor> conv_states,
              std::optional<Tensor> query_start_loc,
              std::optional<Tensor> cache_indices,
              std::optional<Tensor> has_initial_state,
              std::optional<Tensor> num_accepted_tokens,
              std::optional<Tensor> query_start_loc_cpu,
              std::optional<Tensor> cache_indices_cpu,
              std::optional<Tensor> has_initial_state_cpu,
              std::optional<Tensor> num_accepted_tokens_cpu, int64_t activation,
              int64_t pad_slot_id, int64_t null_block_id, int64_t run_mode,
              int64_t head_num, int64_t max_query_len, int64_t stream,
              std::optional<Tensor> out_override = std::nullopt) {
  const TensorMeta x_meta = meta_of(x);
  // The update form may be handed the caller's destination buffer, which is
  // what keeps the result off a second allocation and off the `copy_` back
  // (the caller's `out=` contract).  Only `x`'s own shape is legal there: the
  // prefill form reshapes, and writing a differently-shaped buffer silently
  // would be worse than refusing.
  Tensor out;
  if (out_override.has_value()) {
    if (run_mode != 1 || head_num != 0) {
      throw std::runtime_error(
          "fla_npu(stable): an output buffer is only accepted by the update "
          "form (run_mode=1, head_num=0)");
    }
    const TensorMeta out_meta = meta_of(*out_override);
    if (out_meta.sizes != x_meta.sizes ||
        out_meta.scalar_type != x_meta.scalar_type ||
        out_meta.device_index != x_meta.device_index) {
      throw std::runtime_error(
          "fla_npu(stable): the output buffer must have x's shape, dtype and "
          "device");
    }
    out = *out_override;
  } else {
    out = infer_out(x_meta, head_num, run_mode);
  }
  FLA_STABLE_EXEC("aclnnCausalConv1d", x_meta, stream, tensor(x_meta),
                  tensor(meta_of(weight)), optional_tensor(bias),
                  optional_tensor(conv_states), optional_tensor(query_start_loc),
                  optional_tensor(cache_indices),
                  optional_tensor(has_initial_state),
                  optional_tensor(num_accepted_tokens),
                  int_array(query_start_loc_cpu),
                  int_array(cache_indices_cpu),
                  int_array(has_initial_state_cpu),
                  int_array(num_accepted_tokens_cpu),
                  cstr(kActivationNames, activation), scalar(pad_slot_id),
                  scalar(null_block_id), scalar(run_mode), scalar(head_num),
                  scalar(max_query_len), out_tensor(meta_of(out)));
  return out;
}

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

// ---------------------------------------------------------------------------
// deprecated host-metadata compatibility API
// ---------------------------------------------------------------------------

constexpr const char* kSchema_causal_conv1d =
    "npu_causal_conv1d(Tensor x, Tensor weight, Tensor? bias, "
    "Tensor? conv_states, Tensor? query_start_loc, Tensor? cache_indices, "
    "Tensor? initial_state_mode, Tensor? num_accepted_tokens, int activation, "
    "int pad_slot_id, int run_mode, int head_num, int stream) -> Tensor";

Tensor run_npu_causal_conv1d(
    Tensor x, Tensor weight, std::optional<Tensor> bias,
    std::optional<Tensor> conv_states,
    std::optional<Tensor> query_start_loc,
    std::optional<Tensor> cache_indices,
    std::optional<Tensor> initial_state_mode,
    std::optional<Tensor> num_accepted_tokens, int64_t activation,
    int64_t pad_slot_id, int64_t run_mode, int64_t head_num, int64_t stream) {
  return launch(x, weight, bias, conv_states, /*query_start_loc=*/std::nullopt,
                /*cache_indices=*/std::nullopt,
                /*has_initial_state=*/std::nullopt,
                /*num_accepted_tokens=*/std::nullopt, query_start_loc,
                cache_indices, initial_state_mode, num_accepted_tokens,
                activation, pad_slot_id, /*null_block_id=*/-1, run_mode, head_num,
                /*max_query_len=*/kNoQueryLenBound, stream);
}

// ---------------------------------------------------------------------------
// npu_causal_conv1d_bwd
// ---------------------------------------------------------------------------

constexpr const char* kSchema_causal_conv1d_bwd =
    "npu_causal_conv1d_bwd(Tensor x, Tensor? y, Tensor weight, Tensor dy, "
    "Tensor? initial_state, Tensor? dht, Tensor? query_start_loc, "
    "int activation, int input_layout, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_causal_conv1d_bwd(
    Tensor x, std::optional<Tensor> y, Tensor weight, Tensor dy,
    std::optional<Tensor> initial_state, std::optional<Tensor> dht,
    std::optional<Tensor> query_start_loc, int64_t activation,
    int64_t input_layout, int64_t stream) {
  const TensorMeta x_meta = meta_of(x);
  const TensorMeta weight_meta = meta_of(weight);
  const char* layout =
      enum_name(kCausalConv1dBwdInputLayoutNames, input_layout);
  const std::vector<int64_t> qsl = int_values(query_start_loc);

  // TND/NTD carry one initial state per segment; the other layouts carry one
  // per batch row.
  const bool per_segment =
      std::strcmp(layout, "TND") == 0 || std::strcmp(layout, "NTD") == 0;
  const int64_t state_rows =
      per_segment ? (qsl.empty() ? 0 : static_cast<int64_t>(qsl.size()) - 1)
                  : size_of(x_meta, 0);

  Tensor out_dx = allocate_like(x_meta);
  Tensor out_dw = allocate_sizes(
      {size_of(weight_meta, 0), size_of(weight_meta, 1)},
      weight_meta.scalar_type, weight_meta);
  Tensor out_db = allocate_sizes({size_of(weight_meta, 1)},
                                 weight_meta.scalar_type, weight_meta);
  Tensor out_dinit = allocate_sizes(
      {state_rows, size_of(weight_meta, 0), size_of(weight_meta, 1)},
      x_meta.scalar_type, x_meta);

  FLA_STABLE_EXEC("aclnnCausalConv1dBwd", x_meta, stream, tensor(x_meta),
                  optional_tensor(y), tensor(weight_meta), tensor(meta_of(dy)),
                  optional_tensor(initial_state), optional_tensor(dht),
                  int_array(qsl), scalar(activation), CStrArg(layout),
                  out_tensor(meta_of(out_dx)), out_tensor(meta_of(out_dw)),
                  out_tensor(meta_of(out_db)), out_tensor(meta_of(out_dinit)));
  return std::make_tuple(out_dx, out_dw, out_db, out_dinit);
}

}  // namespace
