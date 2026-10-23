// Stable-ABI adapter for npu_chunk_local_cumsum.
// aclnn: aclnnChunkLocalCumsum
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
#include <stdexcept>
#include <tuple>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kBFloat16;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::at_shim::kHalf;
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
// npu_chunk_local_cumsum
// ---------------------------------------------------------------------------

constexpr const char* kChunkLocalCumsumOutputDtypeNames[] = {
    "float32", "bfloat16", "float16", "same"};

constexpr const char* kSchema_chunk_local_cumsum =
    "npu_chunk_local_cumsum(Tensor g, Tensor? cu_seqlens, "
    "Tensor? chunk_indices_out, int chunk_size, bool reverse, float scale, "
    "bool head_first, int output_dtype, int stream) -> Tensor";

Tensor run_npu_chunk_local_cumsum(Tensor g, std::optional<Tensor> cu_seqlens,
                                  std::optional<Tensor> chunk_indices_out,
                                  int64_t chunk_size, bool reverse, double scale,
                                  bool head_first, int64_t output_dtype,
                                  int64_t stream) {
  const TensorMeta g_meta = meta_of(g);
  int32_t out_dtype;
  switch (output_dtype) {
    case 0: out_dtype = kFloat; break;
    case 1: out_dtype = kBFloat16; break;
    case 2: out_dtype = kHalf; break;
    case 3: out_dtype = g_meta.scalar_type; break;
    default:
      throw std::runtime_error("chunk_local_cumsum: invalid output_dtype code");
  }
  Tensor out = allocate_sizes(g_meta.sizes, out_dtype, g_meta);
  FLA_STABLE_EXEC("aclnnChunkLocalCumsum", g_meta, stream, tensor(g_meta),
                  int_array(cu_seqlens), int_array(chunk_indices_out),
                  scalar(chunk_size), scalar(reverse), scalar(scale),
                  scalar(head_first),
                  cstr(kChunkLocalCumsumOutputDtypeNames, output_dtype),
                  out_tensor(meta_of(out)));
  return out;
}

}  // namespace
