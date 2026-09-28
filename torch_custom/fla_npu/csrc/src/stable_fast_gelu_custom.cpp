// Stable-ABI adapter for npu_fast_gelu_custom.
// aclnn: aclnnFastGelu
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

constexpr const char* kSchema_npu_fast_gelu_custom =
    "npu_fast_gelu_custom(Tensor self, int stream) -> Tensor";

Tensor run_npu_fast_gelu_custom(Tensor self, int64_t stream) {
  const TensorMeta self_meta = meta_of(self);
  Tensor out = allocate_like(self_meta);
  FLA_STABLE_EXEC("aclnnFastGelu", self_meta, stream, tensor(self_meta),
                  out_tensor(meta_of(out)));
  return out;
}

}  // namespace
