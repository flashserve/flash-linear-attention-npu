// Stable-ABI adapters: npu_fast_gelu_custom, npu_fast_gelu_custom_backward.
//
// Included by stable_ops.cpp (single TU).  Hand-written on purpose: the
// generator that produced the neighbouring adapters has been removed, so this
// is the shape every operator follows now.
//
//   * `kSchema_<op>` is the dispatcher signature -- inputs in the order the
//     boxed adapter unboxes them, then the caller's stream.
//   * `run_<op>` is a plain typed function: it allocates its own output (the
//     shape rule belongs to the operator, not to the helper) and hands the
//     aclnn argument list to FLA_STABLE_EXEC in exactly aclnn order.
//   * Registration lives in stable_ops.cpp, after this file is included.
//
// tools/op_abi_parity.py cross-checks the three orders (schema, this call, the
// ctypes reference), so a slip here fails offline instead of silently handing a
// kernel the wrong tensor.

// Owns: npu_fast_gelu_custom, npu_fast_gelu_custom_backward.

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
