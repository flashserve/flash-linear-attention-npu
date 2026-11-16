// Stable-ABI launcher: npu_recurrent_kda (two outputs, optional inputs).
//
// Only torch/csrc/stable/* plus the shared acl_meta helper: no ATen/c10, no
// libtorch C++ ABI.  `layout` is an int code because the stable value
// conversions have no std::string support (0 = BSND, 1 = TND); Python maps it.
// Pre-macro legacy for the same reason as stable_recurrent_gated_delta_rule.cpp
// (read that header, and do not copy this shape): it submits the same way, and
// the boxed entry point below consumes each required
// argument's stack reference.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>

#include "stable/acl_meta.h"
// Only for the enum-table helper: this adapter builds its argument list by
// hand (see the boxed entry point below), so it does not use FLA_STABLE_EXEC --
// which also means it submits the launch itself.
#include "stable/exec.h"

#include <cstdint>
#include <optional>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::AclTensorView;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::allocate_bytes;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::enum_name;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::meta_of_handle;
using fla_npu_stable::stable::meta_optional_handle;
using fla_npu_stable::stable::kAclFormatNd;
using fla_npu_stable::stable::note_launch_stream;
using fla_npu_stable::stable::detail::check_async_failure;
using fla_npu_stable::stable::detail::enqueue_launch;

using fla_npu_stable::stable::aclOpExecutor;
using fla_npu_stable::stable::aclTensor;
using LaunchFn = fla_npu_stable::stable::LaunchFn;

using KdaGetWorkspaceFn = int (*)(const aclTensor*, const aclTensor*,
                                  const aclTensor*, const aclTensor*,
                                  const aclTensor*, aclTensor*,
                                  const aclTensor*, const aclTensor*,
                                  const aclTensor*, const aclTensor*,
                                  const aclTensor*, const char*, double,
                                  bool, bool, bool, bool, bool, bool, bool,
                                  double, bool, aclTensor*, aclTensor*,
                                  uint64_t*, aclOpExecutor**);

constexpr const char* kSchemaRecurrentKda =
    "npu_recurrent_kda(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
    "Tensor(a!) initial_state, Tensor? cu_seqlens, "
    "Tensor? ssm_state_indices, Tensor? A_log, Tensor? dt_bias, "
    "Tensor? num_accepted_tokens, int layout, float scale, "
    "bool output_final_state, bool inplace_final_state, "
    "bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel, "
    "bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, bool safe_gate, "
    "float lower_bound, bool state_v_first, int stream) -> (Tensor, Tensor?)";

// This operator only implements the two spellings the reference accepts, so its
// table is the (BSND, TND) subset rather than the canonical four; the codes
// must match _stable._ENUM["npu_recurrent_kda"]["layout"].
constexpr const char* kRecurrentKdaLayoutNames[] = {"BSND", "TND"};

// The caller's state is read-only in non-inplace mode. When that mode
// requests a final state, ACLNN writes into a separate output pool.
void run_recurrent_kda(AtenTensorHandle q, AtenTensorHandle k,
                       AtenTensorHandle v, AtenTensorHandle g,
                       AtenTensorHandle beta, AtenTensorHandle initial_state,
                       std::optional<AtenTensorHandle> cu_seqlens,
                       std::optional<AtenTensorHandle> ssm_state_indices,
                       std::optional<AtenTensorHandle> A_log,
                       std::optional<AtenTensorHandle> dt_bias,
                       std::optional<AtenTensorHandle> num_accepted_tokens,
                       int64_t layout, double scale,
                       bool output_final_state, bool inplace_final_state,
                       bool use_qk_l2norm_in_kernel, bool use_gate_in_kernel,
                       bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval,
                       bool safe_gate, double lower_bound,
                       bool state_v_first, int64_t stream, Tensor* out,
                       std::optional<Tensor>* final_state) {
  auto& rt = fla_npu_stable::Runtime::instance();
  auto get_ws = reinterpret_cast<KdaGetWorkspaceFn>(
      rt.symbol("aclnnRecurrentKdaGetWorkspaceSize"));
  auto launch = reinterpret_cast<LaunchFn>(rt.symbol("aclnnRecurrentKda"));
  // A queued launch reports its failure where no caller can catch it: the next
  // operator call on this thread raises it instead.
  check_async_failure();

  const TensorMeta v_meta = meta_of_handle(v);
  *out = allocate_like(v_meta);

  const TensorMeta state_meta = meta_of_handle(initial_state);
  if (output_final_state && !inplace_final_state) {
    final_state->emplace(allocate_like(state_meta));
  }
  AclTensorView v_q(meta_of_handle(q), kAclFormatNd);
  AclTensorView v_k(meta_of_handle(k), kAclFormatNd);
  AclTensorView v_v(v_meta, kAclFormatNd);
  AclTensorView v_g(meta_of_handle(g), kAclFormatNd);
  AclTensorView v_beta(meta_of_handle(beta), kAclFormatNd);
  AclTensorView v_state(state_meta, kAclFormatNd);
  AclTensorView v_cu(meta_optional_handle(cu_seqlens), kAclFormatNd);
  AclTensorView v_idx(meta_optional_handle(ssm_state_indices), kAclFormatNd);
  AclTensorView v_alog(meta_optional_handle(A_log), kAclFormatNd);
  AclTensorView v_dtb(meta_optional_handle(dt_bias), kAclFormatNd);
  AclTensorView v_accepted(meta_optional_handle(num_accepted_tokens), kAclFormatNd);
  AclTensorView v_out(meta_of(*out), kAclFormatNd);
  // ACLNN requires a descriptor even when there is no separate output.
  AclTensorView v_final = final_state->has_value()
      ? AclTensorView(meta_of(**final_state), kAclFormatNd)
      : AclTensorView(state_meta, kAclFormatNd);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const int get_ret = get_ws(
      v_q.get(), v_k.get(), v_v.get(), v_g.get(), v_beta.get(), v_state.get(),
      v_cu.get(), v_idx.get(), v_alog.get(), v_dtb.get(), v_accepted.get(),
      enum_name(kRecurrentKdaLayoutNames, layout), scale, output_final_state,
      inplace_final_state, use_qk_l2norm_in_kernel, use_gate_in_kernel,
      use_beta_sigmoid_in_kernel, allow_neg_eigval, safe_gate, lower_bound,
      state_v_first, v_out.get(), v_final.get(), &workspace_size, &executor);
  if (get_ret != 0) {
    throw std::runtime_error(
        "fla_npu(stable): aclnnRecurrentKdaGetWorkspaceSize failed: " +
        std::to_string(get_ret));
  }

  Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = allocate_bytes(static_cast<int64_t>(workspace_size), v_meta);
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_data_ptr(workspace.get(), &workspace_ptr));
  }

  note_launch_stream(stream);
  // Named, not a temporary: a refused enqueue hands the descriptors back so
  // this call can still launch inline (see detail::enqueue_launch).
  auto held = std::make_tuple(
      std::move(v_q), std::move(v_k), std::move(v_v), std::move(v_g),
      std::move(v_beta), std::move(v_state), std::move(v_cu),
      std::move(v_idx), std::move(v_alog), std::move(v_dtb),
      std::move(v_accepted), std::move(v_out), std::move(v_final));
  if (enqueue_launch(rt, "aclnnRecurrentKda", launch, stream, workspace_ptr,
                     workspace_size, executor, held)) {
    return;
  }
  const int launch_ret =
      launch(workspace_ptr, workspace_size, executor,
             reinterpret_cast<void*>(stream));
  if (launch_ret != 0) {
    throw std::runtime_error(
        "fla_npu(stable): aclnnRecurrentKda failed: " +
        std::to_string(launch_ret));
  }
}

void boxed_recurrent_kda(StableIValue* stack, uint64_t num_inputs,
                         uint64_t num_outputs) {
  // Slots are read positionally: a schema that gained or lost a parameter would
  // shift every following one instead of failing to build.
  if (num_inputs != 23 || num_outputs != 2) {
    throw std::runtime_error(
        "fla_npu(stable): npu_recurrent_kda takes 23 inputs and 2 outputs, "
        "the stack declares " + std::to_string(num_inputs) + " and " +
        std::to_string(num_outputs));
  }
  // Same contract as stable_recurrent_gated_delta_rule.cpp: the stack hands
  // the kernel
  // ownership of every argument it reads, so each required slot is unboxed
  // into an owning Tensor and released when this function returns.  The raw
  // to<AtenTensorHandle> read consumed nothing, which retained every fresh
  // input for the life of the process.
  const Tensor t_q = to<Tensor>(stack[0]);
  const Tensor t_k = to<Tensor>(stack[1]);
  const Tensor t_v = to<Tensor>(stack[2]);
  const Tensor t_g = to<Tensor>(stack[3]);
  const Tensor t_beta = to<Tensor>(stack[4]);
  const Tensor t_initial_state = to<Tensor>(stack[5]);
  const AtenTensorHandle q = t_q.get();
  const AtenTensorHandle k = t_k.get();
  const AtenTensorHandle v = t_v.get();
  const AtenTensorHandle g = t_g.get();
  const AtenTensorHandle beta = t_beta.get();
  const AtenTensorHandle initial_state = t_initial_state.get();
  const auto cu_seqlens = to<std::optional<Tensor>>(stack[6]);
  const auto ssm_state_indices = to<std::optional<Tensor>>(stack[7]);
  const auto a_log = to<std::optional<Tensor>>(stack[8]);
  const auto dt_bias = to<std::optional<Tensor>>(stack[9]);
  const auto num_accepted_tokens = to<std::optional<Tensor>>(stack[10]);
  const int64_t layout = to<int64_t>(stack[11]);
  const double scale = to<double>(stack[12]);
  const bool output_final_state = to<bool>(stack[13]);
  const bool inplace_final_state = to<bool>(stack[14]);
  const bool use_qk_l2norm_in_kernel = to<bool>(stack[15]);
  const bool use_gate_in_kernel = to<bool>(stack[16]);
  const bool use_beta_sigmoid_in_kernel = to<bool>(stack[17]);
  const bool allow_neg_eigval = to<bool>(stack[18]);
  const bool safe_gate = to<bool>(stack[19]);
  const double lower_bound = to<double>(stack[20]);
  const bool state_v_first = to<bool>(stack[21]);
  const int64_t stream = to<int64_t>(stack[22]);

  Tensor out;
  std::optional<Tensor> final_state;
  run_recurrent_kda(
      q, k, v, g, beta, initial_state,
      cu_seqlens.has_value()
          ? std::optional<AtenTensorHandle>(cu_seqlens->get())
          : std::nullopt,
      ssm_state_indices.has_value()
          ? std::optional<AtenTensorHandle>(ssm_state_indices->get())
          : std::nullopt,
      a_log.has_value() ? std::optional<AtenTensorHandle>(a_log->get())
                        : std::nullopt,
      dt_bias.has_value() ? std::optional<AtenTensorHandle>(dt_bias->get())
                          : std::nullopt,
      num_accepted_tokens.has_value()
          ? std::optional<AtenTensorHandle>(num_accepted_tokens->get())
          : std::nullopt,
      layout, scale, output_final_state, inplace_final_state,
      use_qk_l2norm_in_kernel, use_gate_in_kernel,
      use_beta_sigmoid_in_kernel, allow_neg_eigval, safe_gate, lower_bound,
      state_v_first, stream, &out, &final_state);
  stack[0] = from(out);
  // Python returns the original tensor in inplace mode.
  stack[1] = from(final_state);
}

}  // namespace
