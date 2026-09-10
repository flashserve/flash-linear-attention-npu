#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "thin_launcher/runtime.h"
#include "thin_launcher/tensor_desc.h"

namespace fla_npu_thin {

namespace {

typedef struct aclOpExecutor aclOpExecutor;

// aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
//   query, key, value, beta, state, actual_seq_lengths, ssm_state_indices,
//   g, gk, num_accepted_tokens, float scale, out, &ws_size, &executor)
using GetWorkspaceFn = int (*)(const aclTensor*, const aclTensor*,
                               const aclTensor*, const aclTensor*,
                               const aclTensor*, const aclTensor*,
                               const aclTensor*, const aclTensor*,
                               const aclTensor*, const aclTensor*, float,
                               const aclTensor*, uint64_t*, aclOpExecutor**);
using LaunchFn = int (*)(void*, uint64_t, aclOpExecutor*, void*);

void add_view(std::vector<std::unique_ptr<AclTensorView>>& views,
              const at::Tensor& t) {
  views.push_back(std::make_unique<AclTensorView>(t));
}

void add_view(std::vector<std::unique_ptr<AclTensorView>>& views,
              const c10::optional<at::Tensor>& t) {
  add_view(views, t.has_value() && t->defined() ? *t : at::Tensor());
}

}  // namespace

at::Tensor npu_recurrent_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    double scale,
    const c10::optional<at::Tensor>& actual_seq_lengths,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    uint64_t stream) {
  auto output = at::empty(value.sizes(),
                          value.options().dtype(at::kBFloat16));

  std::vector<std::unique_ptr<AclTensorView>> views;
  views.reserve(12);
  add_view(views, query);
  add_view(views, key);
  add_view(views, value);
  add_view(views, beta);
  add_view(views, state);
  add_view(views, actual_seq_lengths);
  add_view(views, ssm_state_indices);
  add_view(views, g);
  add_view(views, gk);
  add_view(views, num_accepted_tokens);
  add_view(views, output);

  auto& rt = Runtime::instance();
  auto get_ws = reinterpret_cast<GetWorkspaceFn>(
      rt.symbol("aclnnRecurrentGatedDeltaRuleGetWorkspaceSize"));
  auto launch = reinterpret_cast<LaunchFn>(
      rt.symbol("aclnnRecurrentGatedDeltaRule"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const int get_ret = get_ws(
      views[0]->get(), views[1]->get(), views[2]->get(), views[3]->get(),
      views[4]->get(), views[5]->get(), views[6]->get(), views[7]->get(),
      views[8]->get(), views[9]->get(), static_cast<float>(scale),
      views[10]->get(), &workspace_size, &executor);
  TORCH_CHECK(get_ret == 0,
              "aclnnRecurrentGatedDeltaRuleGetWorkspaceSize failed: ",
              get_ret);

  void* workspace_ptr = nullptr;
  at::Tensor workspace;
  if (workspace_size != 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        at::TensorOptions().dtype(at::kByte).device(value.device()));
    workspace_ptr = workspace.data_ptr();
  }

  const int launch_ret =
      launch(workspace_ptr, workspace_size, executor,
             reinterpret_cast<void*>(stream));
  TORCH_CHECK(launch_ret == 0,
              "aclnnRecurrentGatedDeltaRule failed: ", launch_ret);
  return output;
}

}  // namespace fla_npu_thin
