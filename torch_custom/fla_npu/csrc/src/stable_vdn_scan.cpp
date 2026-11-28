// Stable-ABI adapter for aclnnVdnScan.
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/vdn.h"

namespace {
using torch::stable::Tensor;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::tensor;

constexpr const char* kSchema_vdn_scan =
    "npu_vdn_scan(Tensor transition, Tensor(a!) prefix, Tensor(b!) suffix, "
    "int stream) -> (Tensor(a!), Tensor(b!))";

std::tuple<Tensor, Tensor> run_vdn_scan(Tensor transition, Tensor prefix,
                                         Tensor suffix, int64_t stream) {
  const auto anchor = meta_of(transition);
  const auto prefix_meta = meta_of(prefix);
  fla_npu_stable::vdn::check_same_npu(anchor, prefix_meta);
  const auto suffix_meta = meta_of(suffix);
  fla_npu_stable::vdn::check_same_npu(anchor, suffix_meta);
  FLA_STABLE_EXEC("aclnnVdnScan", anchor, stream,
                  tensor(anchor), tensor(prefix_meta), tensor(suffix_meta));
  return {prefix, suffix};
}
}  // namespace
