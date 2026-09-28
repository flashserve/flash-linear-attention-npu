// Stable-ABI adapter for npu_solve_tri.
// aclnn: aclnnSolveTri
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>
#include <tuple>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kBFloat16;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_solve_tri
// ---------------------------------------------------------------------------

// The kernel takes the layout as a `char*`.  Its lowercase spelling is what
// the reference sends, so the table keeps it; a caller passing anything else
// gets an error from `cstr` rather than a string handed straight to aclnn.
constexpr const char* kSolveTriLayoutNames[] = {"bsnd", "bnsd", "tnd", "ntd"};

constexpr const char* kSchema_solve_tri =
    "npu_solve_tri(Tensor x, Tensor? cu_seqlens, Tensor? chunk_indices, "
    "int layout, int stream) -> Tensor";

Tensor run_npu_solve_tri(Tensor x, std::optional<Tensor> cu_seqlens,
                         std::optional<Tensor> chunk_indices, int64_t layout,
                         int64_t stream) {
  const TensorMeta x_meta = meta_of(x);
  Tensor out = allocate_sizes(x_meta.sizes, x_meta.scalar_type, x_meta);
  FLA_STABLE_EXEC("aclnnSolveTri", x_meta, stream, tensor(x_meta),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  cstr(kSolveTriLayoutNames, layout), out_tensor(meta_of(out)));
  return out;
}

}  // namespace
