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

// ---------------------------------------------------------------------------
// npu_causal_conv1d_bwd
// ---------------------------------------------------------------------------

constexpr const char* kCausalConv1dBwdInputLayoutNames[] = {"BSND", "BNSD", "TND", "NTD", "BSH"};

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
  const char* layout = enum_name(kCausalConv1dBwdInputLayoutNames, input_layout);
  auto [dx, dw, db, dh0] = launch_conv1d_backward(
      x, y, weight, dy, initial_state, dht, query_start_loc, activation,
      layout, stream, true);
  return std::make_tuple(dx, dw, db.value(), dh0);
}

}  // namespace
