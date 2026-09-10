#include <ATen/ATen.h>
#include <torch/extension.h>

// NOTE(M2): this adapter mirrors the upstream-main aclnnCausalConv1d ABI
// (int-array metadata). The service currently runs flash-linear-attention-npu
// PR #390, whose aclnnCausalConv1d ABI differs (device tensor + *_cpu int-array
// dual channels, char* activation, null_block_id/max_query_len). Rewrite this
// adapter against PR #390 before enabling causal_conv1d in the thin launcher.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "thin_launcher/runtime.h"
#include "thin_launcher/tensor_desc.h"

namespace fla_npu_thin {

namespace {

typedef struct aclOpExecutor aclOpExecutor;

using GetWorkspaceFn = int (*)(
    const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    const aclIntArray*, const aclIntArray*, const aclIntArray*,
    const aclIntArray*, int64_t, int64_t, int64_t, int64_t, const aclTensor*,
    uint64_t*, aclOpExecutor**);
using LaunchFn = int (*)(void*, uint64_t, aclOpExecutor*, void*);

at::Tensor infer_output(const at::Tensor& x, int64_t head_num,
                        int64_t run_mode) {
  if (run_mode == 0 && head_num > 0) {
    if (x.dim() == 3) {
      const int64_t b = x.size(0);
      const int64_t s = x.size(1);
      const int64_t d = x.size(2);
      TORCH_CHECK(d % head_num == 0);
      return at::empty({b, head_num, s, d / head_num}, x.options());
    }
    if (x.dim() == 2) {
      const int64_t s = x.size(0);
      const int64_t d = x.size(1);
      TORCH_CHECK(d % head_num == 0);
      return at::empty({head_num, s, d / head_num}, x.options());
    }
  }
  return at::empty_like(x);
}

}  // namespace

at::Tensor npu_causal_conv1d(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& conv_states,
    const std::vector<int64_t>& query_start_loc,
    const std::vector<int64_t>& cache_indices,
    const std::vector<int64_t>& initial_state_mode,
    const std::vector<int64_t>& num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode,
    int64_t head_num,
    uint64_t stream) {
  at::Tensor output = infer_output(x, head_num, run_mode);

  std::vector<std::unique_ptr<AclTensorView>> views;
  views.reserve(5);
  views.push_back(std::make_unique<AclTensorView>(x));
  views.push_back(std::make_unique<AclTensorView>(weight));
  views.push_back(std::make_unique<AclTensorView>(
      bias.has_value() && bias->defined() ? *bias : at::Tensor()));
  views.push_back(std::make_unique<AclTensorView>(conv_states));
  views.push_back(std::make_unique<AclTensorView>(output));

  AclIntArrayView qsl(query_start_loc);
  AclIntArrayView ci(cache_indices);
  AclIntArrayView ism(initial_state_mode);
  AclIntArrayView nat(num_accepted_tokens);

  auto& rt = Runtime::instance();
  auto get_ws = reinterpret_cast<GetWorkspaceFn>(
      rt.symbol("aclnnCausalConv1dGetWorkspaceSize"));
  auto launch =
      reinterpret_cast<LaunchFn>(rt.symbol("aclnnCausalConv1d"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const int get_ret = get_ws(
      views[0]->get(), views[1]->get(), views[2]->get(), views[3]->get(),
      qsl.get(), ci.get(), ism.get(), nat.get(), activation_mode,
      pad_slot_id, run_mode, head_num, views[4]->get(), &workspace_size,
      &executor);
  TORCH_CHECK(get_ret == 0,
              "aclnnCausalConv1dGetWorkspaceSize failed: ", get_ret);

  void* workspace_ptr = nullptr;
  at::Tensor workspace;
  if (workspace_size != 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        at::TensorOptions().dtype(at::kByte).device(x.device()));
    workspace_ptr = workspace.data_ptr();
  }

  const int launch_ret =
      launch(workspace_ptr, workspace_size, executor,
             reinterpret_cast<void*>(stream));
  TORCH_CHECK(launch_ret == 0, "aclnnCausalConv1d failed: ", launch_ret);
  return output;
}

namespace {

// PR #390 aclnnCausalConv1d ABI (update/decode form):
// tensors: x, weight, bias, conv_states, query_start_loc,
//          conv_state_indices, has_initial_state(None), num_accepted_tokens
// arrays:  *_cpu variants of the three metadata inputs
// scalars: activation(char*), pad_slot_id, null_block_id, run_mode(=1),
//          head_num(=0), max_query_len, y
using UpdateGetWorkspaceFn = int (*)(
    const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    const aclIntArray*, const aclIntArray*, const aclIntArray*,
    const aclIntArray*, const char*, int64_t, int64_t, int64_t, int64_t,
    int64_t, const aclTensor*, uint64_t*, aclOpExecutor**);

void update_add(std::vector<std::unique_ptr<AclTensorView>>& views,
                const c10::optional<at::Tensor>& t) {
  views.push_back(std::make_unique<AclTensorView>(
      t.has_value() && t->defined() ? *t : at::Tensor()));
}

void update_add(std::vector<std::unique_ptr<AclTensorView>>& views,
                const at::Tensor& t) {
  views.push_back(std::make_unique<AclTensorView>(t));
}

}  // namespace

at::Tensor npu_causal_conv1d_update(
    const at::Tensor& x,
    const at::Tensor& conv_state,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const std::string& activation,
    const c10::optional<at::Tensor>& conv_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& query_start_loc,
    const std::vector<int64_t>& conv_state_indices_cpu,
    const std::vector<int64_t>& num_accepted_tokens_cpu,
    const std::vector<int64_t>& query_start_loc_cpu,
    int64_t max_query_len,
    int64_t null_block_id,
    c10::optional<at::Tensor> out,
    uint64_t stream) {
  at::Tensor y = out.has_value() && out->defined() ? *out
                                                   : at::empty_like(x);

  std::vector<std::unique_ptr<AclTensorView>> views;
  views.reserve(9);
  update_add(views, x);
  update_add(views, weight);
  update_add(views, bias);
  update_add(views, conv_state);
  update_add(views, query_start_loc);
  update_add(views, conv_state_indices);
  update_add(views, c10::optional<at::Tensor>());  // has_initial_state
  update_add(views, num_accepted_tokens);
  update_add(views, y);

  AclIntArrayView qsl_cpu(query_start_loc_cpu);
  AclIntArrayView cidx_cpu(conv_state_indices_cpu);
  AclIntArrayView his_cpu({});
  AclIntArrayView nat_cpu(num_accepted_tokens_cpu);

  const std::string act = activation.empty() ? "none" : activation;
  const int64_t pad_slot_id = -(1LL << 63);

  auto& rt = Runtime::instance();
  auto get_ws = reinterpret_cast<UpdateGetWorkspaceFn>(
      rt.symbol("aclnnCausalConv1dGetWorkspaceSize"));
  auto launch =
      reinterpret_cast<LaunchFn>(rt.symbol("aclnnCausalConv1d"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const int get_ret = get_ws(
      views[0]->get(), views[1]->get(), views[2]->get(), views[3]->get(),
      views[4]->get(), views[5]->get(), views[6]->get(), views[7]->get(),
      qsl_cpu.get(), cidx_cpu.get(), his_cpu.get(), nat_cpu.get(), act.c_str(),
      pad_slot_id, null_block_id, 1, 0, max_query_len, views[8]->get(),
      &workspace_size, &executor);
  TORCH_CHECK(get_ret == 0,
              "aclnnCausalConv1dGetWorkspaceSize failed: ", get_ret);

  void* workspace_ptr = nullptr;
  at::Tensor workspace;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().dtype(at::kByte).device(x.device()));
    workspace_ptr = workspace.data_ptr();
  }
  const int launch_ret =
      launch(workspace_ptr, workspace_size, executor,
             reinterpret_cast<void*>(stream));
  TORCH_CHECK(launch_ret == 0, "aclnnCausalConv1d failed: ", launch_ret);
  return y;
}

}  // namespace fla_npu_thin
