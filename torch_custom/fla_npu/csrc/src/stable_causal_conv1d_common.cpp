// Shared helpers for the operators whose adapters are
// causal_conv1d_fn, causal_conv1d_update, causal_conv1d, causal_conv1d_bwd.
//
// Only declarations used by more than one operator live here; everything
// else belongs to the operator's own stable_<op>.cpp.  csrc/src/stable_ops.cpp
// includes this file first, so its names are declared before every user.

// aclnn exposes one forward entry point for the three forward APIs; what
// differs is runMode (0 = prefill/FN, 1 = decode/UPDATE) and how the
// caller's metadata is shaped, so infer_out()/launch() are shared here and
// the three entry points differ only in the arguments they hand over.
// The scheduling parameters the reference refuses (block cache / APC /
// metadata objects) are refused by the Python wrappers, so nothing here
// has to describe them.

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

// maxQueryLen is only meaningful for the varlen update form; the other two
// entry points pass the reference's -1.
constexpr int64_t kNoQueryLenBound = -1;

// Mirrors _infer_causal_conv1d_y: the prefill path with a positive headNum
// returns the head-split view instead of a same-shape copy.
Tensor infer_out(const TensorMeta& x_meta, int64_t head_num, int64_t run_mode) {
  if (run_mode == 0 && head_num > 0) {
    if (x_meta.ndim == 3) {
      return allocate_sizes({SIZE_OF(x_meta, 0), head_num, SIZE_OF(x_meta, 1),
                             SIZE_OF(x_meta, 2) / head_num},
                            x_meta.scalar_type, x_meta);
    }
    if (x_meta.ndim == 2) {
      return allocate_sizes({head_num, SIZE_OF(x_meta, 0),
                             SIZE_OF(x_meta, 1) / head_num},
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

}  // namespace
