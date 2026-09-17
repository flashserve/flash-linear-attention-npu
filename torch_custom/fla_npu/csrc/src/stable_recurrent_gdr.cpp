// Stable-ABI adapter: npu_recurrent_gated_delta_rule.
//
// Included by stable_ops.cpp (single TU).  Only torch/csrc/stable/* plus the
// shared acl_meta helper: no ATen/c10, no libtorch C++ ABI.
// Owns: npu_recurrent_gated_delta_rule.  Pre-macro on purpose: it builds the
// argument list and submits the launch by hand, so the descriptors travel to
// the queue as one bundle (see detail::enqueue_launch) instead of in the
// macro's tuple.  The boxed entry point below still consumes each required
// argument's stack reference, exactly like the macro's typed unboxing.

#include <torch/csrc/stable/library.h>
#ifndef FLA_STABLE_NO_DEBUG_PROBE
#include <torch/csrc/stable/accelerator.h>
#endif
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>

#include "stable/acl_meta.h"
// For detail::enqueue_launch: the submission the macro applies for free.
#include "stable/exec.h"

#include <dlfcn.h>

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
using fla_npu_stable::stable::note_launch_stream;
using fla_npu_stable::stable::detail::check_async_failure;
using fla_npu_stable::stable::detail::enqueue_launch;

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
  // A queued launch reports its failure where no caller can catch it: the next
  // operator call on this thread raises it instead.
  check_async_failure();

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

  note_launch_stream(stream);
  // Named, not a temporary: a refused enqueue hands the descriptors back so
  // this call can still launch inline (see detail::enqueue_launch).
  auto held = std::make_tuple(
      std::move(v_query), std::move(v_key), std::move(v_value),
      std::move(v_beta), std::move(v_state), std::move(v_seq),
      std::move(v_idx), std::move(v_g), std::move(v_gk),
      std::move(v_accepted), std::move(v_out));
  if (enqueue_launch(rt, "aclnnRecurrentGatedDeltaRule", launch, stream,
                     workspace_ptr, workspace_size, executor, held)) {
    return out;
  }
  const int launch_ret =
      launch(workspace_ptr, workspace_size, executor,
             reinterpret_cast<void*>(stream));
  if (launch_ret != 0) {
    throw std::runtime_error(
        "fla_npu(stable): aclnnRecurrentGatedDeltaRule failed: " +
        std::to_string(launch_ret));
  }
  return out;
}

// Boxed entry point: every required tensor is unboxed into an owning Tensor, so
// the reference the dispatcher handed us is consumed exactly once and released
// when this function returns.  Optionals keep the std::optional<Tensor> form,
// which consumes the inner handle itself.
void boxed_recurrent_gated_delta_rule(StableIValue* stack,
                                      uint64_t num_inputs,
                                      uint64_t num_outputs) {
  // Slots are read positionally: a schema that gained or lost a parameter would
  // shift every following one instead of failing to build.
  if (num_inputs != 12 || num_outputs != 1) {
    throw std::runtime_error(
        "fla_npu(stable): npu_recurrent_gated_delta_rule takes 12 inputs "
        "and 1 output, the stack declares " + std::to_string(num_inputs) +
        " and " + std::to_string(num_outputs));
  }
  // The boxed stack hands the kernel ownership of every argument it reads:
  // library.h says fn is responsible for stealing the memory of the inputs,
  // in effect "popping" them off the stack.  Reading a required slot with
  // to<AtenTensorHandle> consumes nothing, so the reference the dispatcher
  // created for the caller was never released.  Measured on 910B3: 2000
  // decode-shaped calls with a fresh q/k/v grew the caching allocator by
  // 191 MiB (~99 KiB a call: three 32 KiB tensors plus beta/asl/idx at the
  // 512 B block floor), and the conc32 service grew 8.2 GiB until it OOMd.
  // One owning Tensor per required slot consumes exactly that reference and
  // releases it when this function returns; the handle the descriptors see
  // is borrowed from it.  Optional slots keep the to<std::optional<Tensor>>
  // form: it consumes the inner handle and frees the box the dispatcher
  // allocated for it, and it is the only safe reader here.  A *present*
  // optional puts a pointer to that heap box in the slot, so to<Tensor> would
  // wrap the box pointer as an AtenTensorHandle and delete it as a tensor --
  // the heap corruption the first two handle-unboxing attempts hit; a None
  // arrives as a null handle instead.
  const Tensor t_query = to<Tensor>(stack[0]);
  const Tensor t_key = to<Tensor>(stack[1]);
  const Tensor t_value = to<Tensor>(stack[2]);
  const Tensor t_state = to<Tensor>(stack[3]);
  const Tensor t_beta = to<Tensor>(stack[4]);
  const Tensor t_actual_seq_lengths = to<Tensor>(stack[5]);
  const Tensor t_ssm_state_indices = to<Tensor>(stack[6]);
  const AtenTensorHandle query = t_query.get();
  const AtenTensorHandle key = t_key.get();
  const AtenTensorHandle value = t_value.get();
  const AtenTensorHandle state = t_state.get();
  const AtenTensorHandle beta = t_beta.get();
  const AtenTensorHandle actual_seq_lengths = t_actual_seq_lengths.get();
  const AtenTensorHandle ssm_state_indices = t_ssm_state_indices.get();
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
//
// The stream shims this needs -- aoti_torch_get_current_stream,
// aoti_torch_stream_id and aoti_torch_delete_stream -- only exist from torch
// 2.9 on, and torch::stable::accelerator::getCurrentStream() resolves the same
// three.  They are looked up at run time instead of being left to the loader,
// because a probe-enabled build would otherwise refuse to load on torch
// 2.7.1/2.8 with "undefined symbol: aoti_torch_stream_id": the debug flag would
// silently raise the library's torch floor above the >= 2.7.1 the wheel
// metadata declares.  Where the shims are missing the probe reports -3 and
// every operator keeps working.
using GetCurrentStreamFn = int32_t (*)(int32_t, StreamHandle*);
using StreamIdFn = int32_t (*)(StreamHandle, int64_t*);
using DeleteStreamFn = int32_t (*)(StreamHandle);

void* stable_runtime_symbol(const char* name) {
  // The launcher links libtorch_cpu/libc10/libtorch, so those objects are
  // already mapped; RTLD_NOLOAD reaches them without depending on whether
  // torch.ops.load_library() opened the launcher into the global scope.
  static void* const libs[] = {
      dlopen("libtorch_cpu.so", RTLD_NOLOAD | RTLD_LAZY),
      dlopen("libtorch.so", RTLD_NOLOAD | RTLD_LAZY),
      dlopen("libc10.so", RTLD_NOLOAD | RTLD_LAZY),
  };
  for (void* lib : libs) {
    if (lib != nullptr) {
      if (void* symbol = dlsym(lib, name)) {
        return symbol;
      }
    }
  }
  return dlsym(RTLD_DEFAULT, name);
}

GetCurrentStreamFn stable_get_current_stream() {
  static const auto fn = reinterpret_cast<GetCurrentStreamFn>(
      stable_runtime_symbol("aoti_torch_get_current_stream"));
  return fn;
}

StreamIdFn stable_stream_id() {
  static const auto fn = reinterpret_cast<StreamIdFn>(
      stable_runtime_symbol("aoti_torch_stream_id"));
  return fn;
}

DeleteStreamFn stable_delete_stream() {
  static const auto fn = reinterpret_cast<DeleteStreamFn>(
      stable_runtime_symbol("aoti_torch_delete_stream"));
  return fn;
}

void boxed_stream_probe(StableIValue* stack, uint64_t num_inputs,
                        uint64_t num_outputs) {
  (void)num_inputs;
  (void)num_outputs;
  const int64_t device_index = to<int64_t>(stack[0]);
  const auto get_stream = stable_get_current_stream();
  const auto stream_id_of = stable_stream_id();
  const auto drop_stream = stable_delete_stream();
  const bool have_shims = get_stream != nullptr && stream_id_of != nullptr &&
                          drop_stream != nullptr;
  int64_t shim_id = have_shims ? -1 : -3;
  StreamHandle handle = nullptr;
  if (have_shims &&
      get_stream(static_cast<int32_t>(device_index), &handle) == 0 &&
      handle != nullptr) {
    if (stream_id_of(handle, &shim_id) != 0) {
      shim_id = -2;
    }
    drop_stream(handle);
  }
  // Second read: the same shims the header's getCurrentStream() wrapper uses.
  // Two independent reads keep the (raw, stable) shape the Python side expects
  // and would expose a shim that answers differently the second time.
  int64_t stream_id = have_shims ? -1 : -3;
  StreamHandle stable_handle = nullptr;
  if (have_shims &&
      get_stream(static_cast<int32_t>(device_index), &stable_handle) == 0 &&
      stable_handle != nullptr) {
    if (stream_id_of(stable_handle, &stream_id) != 0) {
      stream_id = -2;
    }
    drop_stream(stable_handle);
  }
  stack[0] = from(shim_id);
  stack[1] = from(stream_id);
}
#endif  // FLA_STABLE_NO_DEBUG_PROBE

}  // namespace
