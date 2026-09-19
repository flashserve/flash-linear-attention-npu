// Stable-ABI adapter for npu_fast_gelu_custom_backward.
// aclnn: aclnnFastGeluBackward
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::tensor;

constexpr const char* kSchema_npu_fast_gelu_custom_backward =
    "npu_fast_gelu_custom_backward(Tensor grad, Tensor self, int stream) -> Tensor";

Tensor run_npu_fast_gelu_custom_backward(Tensor grad, Tensor self,
                                         int64_t stream) {
  const TensorMeta grad_meta = meta_of(grad);
  Tensor out = allocate_like(grad_meta);
  FLA_STABLE_EXEC("aclnnFastGeluBackward", grad_meta, stream,
                  tensor(meta_of(grad)), tensor(meta_of(self)),
                  out_tensor(meta_of(out)));
  return out;
}

}  // namespace
