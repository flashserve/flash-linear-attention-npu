// Stable-ABI adapter for npu_merge_fwd_bwd_kernel.
// aclnn: aclnnMergeFwdBwdKernel
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.
//
// h is the only state buffer, matching FLA merge_fwd_bwd_kernel. The kernel
// stores into it. state_v_first selects [K, V] or [V, K].

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::scalar;

constexpr const char* kSchema_merge_fwd_bwd_kernel =
    "npu_merge_fwd_bwd_kernel(Tensor(a!) h, Tensor ag_hm, "
    "int pre_or_post_num_ranks, int rank, bool forward, bool state_v_first, "
    "int stream) -> Tensor";

Tensor run_npu_merge_fwd_bwd_kernel(Tensor h, Tensor ag_hm,
                                    int64_t pre_or_post_num_ranks, int64_t rank,
                                    bool forward, bool state_v_first,
                                    int64_t stream) {
  const TensorMeta h_meta = meta_of(h);
  const TensorMeta ag_meta = meta_of(ag_hm);
  FLA_STABLE_EXEC("aclnnMergeFwdBwdKernel", h_meta, stream, nd_tensor(h_meta),
                  nd_tensor(ag_meta), scalar(pre_or_post_num_ranks), scalar(rank),
                  scalar(forward), scalar(state_v_first));
  return h;
}

}  // namespace
