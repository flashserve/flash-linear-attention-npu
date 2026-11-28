// Stable-ABI adapter for aclnnVdnBridge.
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/vdn.h"

namespace {
using torch::stable::Tensor;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;

constexpr const char* kSchema_vdn_bridge =
    "npu_vdn_bridge(Tensor prefix, Tensor suffix, Tensor bounds, Tensor logs, "
    "int start, Tensor(a!) out, int stream) -> Tensor(a!)";

Tensor run_vdn_bridge(Tensor prefix, Tensor suffix, Tensor bounds, Tensor logs,
                          int64_t start, Tensor out, int64_t stream) {
  const auto anchor = meta_of(prefix);
  const auto suffix_meta = meta_of(suffix);
  fla_npu_stable::vdn::check_same_npu(anchor, suffix_meta);
  const auto bounds_meta = meta_of(bounds);
  fla_npu_stable::vdn::check_same_npu(anchor, bounds_meta);
  const auto logs_meta = meta_of(logs);
  fla_npu_stable::vdn::check_same_npu(anchor, logs_meta);
  const auto out_meta = meta_of(out);
  fla_npu_stable::vdn::check_same_npu(anchor, out_meta);
  FLA_STABLE_EXEC("aclnnVdnBridge", anchor, stream,
                  tensor(anchor), tensor(suffix_meta), tensor(bounds_meta), tensor(logs_meta),
                  scalar(start), out_tensor(out_meta));
  return out;
}
}  // namespace
