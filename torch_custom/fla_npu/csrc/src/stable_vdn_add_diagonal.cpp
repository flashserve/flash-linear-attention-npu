// Stable-ABI adapter for aclnnVdnAddDiagonal.
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/vdn.h"

namespace {
using torch::stable::Tensor;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::tensor;

constexpr const char* kSchema_vdn_add_diagonal =
    "npu_vdn_add_diagonal(Tensor(a!) correction, Tensor two_eye, int stream) -> Tensor(a!)";

Tensor run_vdn_add_diagonal(Tensor correction, Tensor two_eye, int64_t stream) {
  const auto anchor = meta_of(correction);
  const auto two_eye_meta = meta_of(two_eye);
  fla_npu_stable::vdn::check_same_npu(anchor, two_eye_meta);
  FLA_STABLE_EXEC("aclnnVdnAddDiagonal", anchor, stream,
                  tensor(anchor), tensor(two_eye_meta));
  return correction;
}
}  // namespace
