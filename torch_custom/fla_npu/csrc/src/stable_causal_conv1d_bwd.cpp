// Stable-ABI adapter for npu_causal_conv1d_bwd.
// aclnn: aclnnCausalConv1dBwd
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

// A different aclnn op from the forward: its d(initial_state) output is
// sized from the segment count.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::CStrArg;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::enum_name;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

constexpr const char* kCausalConv1dBwdInputLayoutNames[] = {"BSND", "BNSD",
                                                            "TND", "NTD"};

// ---------------------------------------------------------------------------
// npu_causal_conv1d_bwd
// ---------------------------------------------------------------------------

constexpr const char* kSchema_causal_conv1d_bwd =
    "npu_causal_conv1d_bwd(Tensor x, Tensor? y, Tensor weight, Tensor dy, "
    "Tensor? initial_state, Tensor? dht, Tensor? query_start_loc, "
    "int activation, int input_layout, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_causal_conv1d_bwd(
    Tensor x, std::optional<Tensor> y, Tensor weight, Tensor dy,
    std::optional<Tensor> initial_state, std::optional<Tensor> dht,
    std::optional<Tensor> query_start_loc, int64_t activation,
    int64_t input_layout, int64_t stream) {
  const TensorMeta x_meta = meta_of(x);
  const TensorMeta weight_meta = meta_of(weight);
  const char* layout =
      enum_name(kCausalConv1dBwdInputLayoutNames, input_layout);
  const std::vector<int64_t> qsl = int_values(query_start_loc);

  // TND/NTD carry one initial state per segment; the other layouts carry one
  // per batch row.
  const bool per_segment =
      std::strcmp(layout, "TND") == 0 || std::strcmp(layout, "NTD") == 0;
  const int64_t state_rows =
      per_segment ? (qsl.empty() ? 0 : static_cast<int64_t>(qsl.size()) - 1)
                  : size_of(x_meta, 0);

  Tensor out_dx = allocate_like(x_meta);
  Tensor out_dw = allocate_sizes(
      {size_of(weight_meta, 0), size_of(weight_meta, 1)},
      weight_meta.scalar_type, weight_meta);
  Tensor out_db = allocate_sizes({size_of(weight_meta, 1)},
                                 weight_meta.scalar_type, weight_meta);
  Tensor out_dinit = allocate_sizes(
      {state_rows, size_of(weight_meta, 0), size_of(weight_meta, 1)},
      x_meta.scalar_type, x_meta);

  FLA_STABLE_EXEC("aclnnCausalConv1dBwd", x_meta, stream, tensor(x_meta),
                  optional_tensor(y), tensor(weight_meta), tensor(meta_of(dy)),
                  optional_tensor(initial_state), optional_tensor(dht),
                  int_array(qsl), scalar(activation), CStrArg(layout),
                  out_tensor(meta_of(out_dx)), out_tensor(meta_of(out_dw)),
                  out_tensor(meta_of(out_db)), out_tensor(meta_of(out_dinit)));
  return std::make_tuple(out_dx, out_dw, out_db, out_dinit);
}

}  // namespace
