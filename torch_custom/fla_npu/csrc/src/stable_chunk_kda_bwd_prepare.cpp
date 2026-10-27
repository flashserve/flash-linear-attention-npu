// Stable-ABI adapter for npu_chunk_kda_bwd_prepare.
// aclnn: aclnnChunkKdaBwdPrepare
#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/layout_math.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace {
using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kBFloat16;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::nd_out_tensor;
using fla_npu_stable::stable::nd_tensor;
using fla_npu_stable::stable::scalar;

constexpr const char* kChunkKdaBwdPrepareLayoutNames[] = {
    "BSND", "BNSD", "TND", "NTD"};

constexpr const char* kSchema_chunk_kda_bwd_prepare =
    "npu_chunk_kda_bwd_prepare(Tensor aqk, Tensor v_new, Tensor d_o, Tensor h, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int layout, float scale, "
    "int chunk_size, bool state_v_first, int stream) -> (Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor> run_npu_chunk_kda_bwd_prepare(
    Tensor aqk, Tensor v_new, Tensor d_o, Tensor h,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t layout, double scale, int64_t chunk_size, bool state_v_first,
    int64_t stream) {
  namespace layout_math = fla_npu_stable::stable::layout_math;
  const TensorMeta aqk_meta = meta_of(aqk);
  std::vector<int64_t> output;
  if (!layout_math::packed(layout)) {
    output.push_back(SIZE_OF(aqk_meta, 0));
  }
  output.push_back(SIZE_OF(aqk_meta, layout_math::head_axis(layout)));
  output.push_back(SIZE_OF(aqk_meta, layout_math::token_axis(layout)));
  output.push_back(64);
  Tensor d_aqk = allocate_sizes(output, kFloat, aqk_meta);
  output.back() = 128;
  Tensor dv = allocate_sizes(output, kBFloat16, aqk_meta);
  Tensor dq_raw = allocate_sizes(output, kFloat, aqk_meta);

  FLA_STABLE_EXEC(
      "aclnnChunkKdaBwdPrepare", aqk_meta, stream, nd_tensor(aqk_meta),
      nd_tensor(meta_of(v_new)), nd_tensor(meta_of(d_o)), nd_tensor(meta_of(h)),
      int_array(cu_seqlens), int_array(chunk_indices),
      cstr(kChunkKdaBwdPrepareLayoutNames, layout), scalar(scale),
      scalar(chunk_size), scalar(state_v_first), nd_out_tensor(meta_of(d_aqk)),
      nd_out_tensor(meta_of(dv)), nd_out_tensor(meta_of(dq_raw)));
  return std::make_tuple(d_aqk, dv, dq_raw);
}
}  // namespace
