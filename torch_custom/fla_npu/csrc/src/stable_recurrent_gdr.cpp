// Stable-ABI adapter: npu_recurrent_gated_delta_rule.
//
// Included by stable_ops.cpp (single TU).  Only torch/csrc/stable/* plus the
// shared acl_meta helper: no ATen/c10, no libtorch C++ ABI.
// Owns: npu_recurrent_gated_delta_rule.  Pre-macro on purpose: its state
// is an in-place argument, and the macro's typed unboxing would steal the
// handle (see stable-abi-macro-design.md).

#include <torch/csrc/stable/library.h>
#ifndef FLA_STABLE_NO_DEBUG_PROBE
#include <torch/csrc/stable/accelerator.h>
#endif
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>

#include "stable/acl_meta.h"

#include <cstdint>
#include <optional>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::AclTensorView;
using fla_npu_stable::stable::LaunchFn;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::aclOpExecutor;
using fla_npu_stable::stable::aclTensor;
using fla_npu_stable::stable::allocate_bytes;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::meta_of_handle;
using fla_npu_stable::stable::meta_optional_handle;
using fla_npu_stable::stable::kAclFormatNd;

// Prefixed per op: everything lives in one TU (see stable_ops.cpp), so shared
// local names would collide.
using GdrGetWorkspaceFn = int (*)(const aclTensor*, const aclTensor*,
                                  const aclTensor*, const aclTensor*,
                                  aclTensor*, const aclTensor*,
                                  const aclTensor*, const aclTensor*,
                                  const aclTensor*, const aclTensor*, float,
                                  aclTensor*, uint64_t*, aclOpExecutor**);

constexpr const char* kSchemaRecurrentGdr =
    "npu_recurrent_gated_delta_rule(Tensor query, Tensor key, Tensor value, "
    "Tensor(a!) state, Tensor beta, Tensor actual_seq_lengths, "
    "Tensor ssm_state_indices, Tensor? num_accepted_tokens, Tensor? g, "
    "Tensor? gk, float scale, int stream) -> Tensor";

// Returns the attn output; `state` is an in/out ref, exactly like ctypes.
Tensor run_recurrent_gated_delta_rule(AtenTensorHandle query,
                                      AtenTensorHandle key,
                                      AtenTensorHandle value,
                                      AtenTensorHandle state,
                                      AtenTensorHandle beta,
                                      AtenTensorHandle actual_seq_lengths,
                                      AtenTensorHandle ssm_state_indices,
                                      std::optional<AtenTensorHandle>
                                          num_accepted_tokens,
                                      std::optional<AtenTensorHandle> g,
                                      std::optional<AtenTensorHandle> gk,
                                      double scale, int64_t stream) {
  auto& rt = fla_npu_stable::Runtime::instance();
  auto get_ws = reinterpret_cast<GdrGetWorkspaceFn>(
      rt.symbol("aclnnRecurrentGatedDeltaRuleGetWorkspaceSize"));
  auto launch =
      reinterpret_cast<LaunchFn>(rt.symbol("aclnnRecurrentGatedDeltaRule"));

  const TensorMeta value_meta = meta_of_handle(value);
  Tensor out = allocate_like(value_meta);

  AclTensorView v_query(meta_of_handle(query), kAclFormatNd);
  AclTensorView v_key(meta_of_handle(key), kAclFormatNd);
  AclTensorView v_value(value_meta, kAclFormatNd);
  AclTensorView v_beta(meta_of_handle(beta), kAclFormatNd);
  AclTensorView v_state(meta_of_handle(state), kAclFormatNd);
  AclTensorView v_seq(meta_of_handle(actual_seq_lengths), kAclFormatNd);
  AclTensorView v_idx(meta_of_handle(ssm_state_indices), kAclFormatNd);
  AclTensorView v_g(meta_optional_handle(g), kAclFormatNd);
  AclTensorView v_gk(meta_optional_handle(gk), kAclFormatNd);
  AclTensorView v_accepted(meta_optional_handle(num_accepted_tokens), kAclFormatNd);
  AclTensorView v_out(meta_of(out), kAclFormatNd);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const int get_ret = get_ws(
      v_query.get(), v_key.get(), v_value.get(), v_beta.get(), v_state.get(),
      v_seq.get(), v_idx.get(), v_g.get(), v_gk.get(), v_accepted.get(),
      static_cast<float>(scale), v_out.get(), &workspace_size, &executor);
  if (get_ret != 0) {
    throw std::runtime_error(
        "fla_npu(stable): aclnnRecurrentGatedDeltaRuleGetWorkspaceSize "
        "failed: " +
        std::to_string(get_ret));
  }

  Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = allocate_bytes(static_cast<int64_t>(workspace_size), value_meta);
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_data_ptr(workspace.get(), &workspace_ptr));
  }
  const int launch_ret = launch(workspace_ptr, workspace_size, executor,
                                reinterpret_cast<void*>(stream));
  if (launch_ret != 0) {
    throw std::runtime_error(
        "fla_npu(stable): aclnnRecurrentGatedDeltaRule failed: " +
        std::to_string(launch_ret));
  }
  return out;
}

// Boxed entry point: required tensors unbox straight to handles (never through
// the ownership-stealing Tensor(AtenTensorHandle) constructor), optionals keep
// the Tensor form so their liveness is explicit.
void boxed_recurrent_gated_delta_rule(StableIValue* stack,
                                      uint64_t num_inputs,
                                      uint64_t num_outputs) {
  (void)num_inputs;
  (void)num_outputs;
  const AtenTensorHandle query = to<AtenTensorHandle>(stack[0]);
  const AtenTensorHandle key = to<AtenTensorHandle>(stack[1]);
  const AtenTensorHandle value = to<AtenTensorHandle>(stack[2]);
  const AtenTensorHandle state = to<AtenTensorHandle>(stack[3]);
  const AtenTensorHandle beta = to<AtenTensorHandle>(stack[4]);
  const AtenTensorHandle actual_seq_lengths = to<AtenTensorHandle>(stack[5]);
  const AtenTensorHandle ssm_state_indices = to<AtenTensorHandle>(stack[6]);
  const auto num_accepted_tokens = to<std::optional<Tensor>>(stack[7]);
  const auto g = to<std::optional<Tensor>>(stack[8]);
  const auto gk = to<std::optional<Tensor>>(stack[9]);
  const double scale = to<double>(stack[10]);
  const int64_t stream = to<int64_t>(stack[11]);
  Tensor out = run_recurrent_gated_delta_rule(
      query, key, value, state, beta, actual_seq_lengths, ssm_state_indices,
      num_accepted_tokens.has_value()
          ? std::optional<AtenTensorHandle>(num_accepted_tokens->get())
          : std::nullopt,
      g.has_value() ? std::optional<AtenTensorHandle>(g->get()) : std::nullopt,
      gk.has_value() ? std::optional<AtenTensorHandle>(gk->get())
                     : std::nullopt,
      scale, stream);
  stack[0] = from(out);
}

#ifndef FLA_STABLE_NO_DEBUG_PROBE
// Reports the current stream for `device_index` two ways so Python can compare
// them against torch_npu's raw accessor (both returned 0 on torch_npu
// 2.9.0.post2, i.e. the stable stream API does not map to the NPU stream yet).
void boxed_stream_probe(StableIValue* stack, uint64_t num_inputs,
                        uint64_t num_outputs) {
  (void)num_inputs;
  (void)num_outputs;
  const int64_t device_index = to<int64_t>(stack[0]);
  int64_t shim_id = -1;
  StreamHandle handle = nullptr;
  if (aoti_torch_get_current_stream(static_cast<int32_t>(device_index),
                                    &handle) == 0 &&
      handle != nullptr) {
    if (aoti_torch_stream_id(handle, &shim_id) != 0) {
      shim_id = -2;
    }
    aoti_torch_delete_stream(handle);
  }
  int64_t stream_id = 0;
  try {
    auto stream = torch::stable::accelerator::getCurrentStream(
        static_cast<int32_t>(device_index));
    stream_id = static_cast<int64_t>(stream.id());
  } catch (...) {
    stream_id = -1;
  }
  stack[0] = from(shim_id);
  stack[1] = from(stream_id);
}
#endif  // FLA_STABLE_NO_DEBUG_PROBE

}  // namespace
