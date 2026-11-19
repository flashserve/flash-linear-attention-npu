// Stable-ABI entry for callers that only require dx, dw and d(initial_state).
// The existing dispatcher schema keeps its four gradient outputs.
namespace {

constexpr const char* kSchema_causal_conv1d_bwd_no_bias =
    "npu_causal_conv1d_bwd_no_bias(Tensor x, Tensor? y, Tensor weight, Tensor dy, "
    "Tensor? initial_state, Tensor? dht, Tensor? query_start_loc, "
    "int activation, int input_layout, int stream) -> (Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor> run_npu_causal_conv1d_bwd_no_bias(
    Tensor x, std::optional<Tensor> y, Tensor weight, Tensor dy,
    std::optional<Tensor> initial_state, std::optional<Tensor> dht,
    std::optional<Tensor> query_start_loc, int64_t activation,
    int64_t input_layout, int64_t stream) {
  const char* layout = enum_name(kCausalConv1dBwdInputLayoutNames, input_layout);
  auto [dx, dw, db, dh0] = launch_conv1d_backward(
      x, y, weight, dy, initial_state, dht, query_start_loc, activation,
      layout, stream, false);
  return std::make_tuple(dx, dw, dh0);
}

}  // namespace
