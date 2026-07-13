// Copyright (c) 2024 Tianjin University, Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace op_api {

at::Tensor npu_fast_gelu_custom(const at::Tensor &self);
at::Tensor npu_fast_gelu_custom_backward(const at::Tensor &grad, const at::Tensor &self);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_prepare_wy_repr_bwd_full(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    const at::Tensor &dA,
    const at::Tensor &dw,
    const at::Tensor &du,
    const at::Tensor &g,
    int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices);

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_chunk_gated_delta_rule_bwd_dhu(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &w,
    const at::Tensor &d_o,
    const at::Tensor &dv,
    double scale,
    int64_t chunk_size,
    const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &gK,
    const c10::optional<at::Tensor> &h0,
    const c10::optional<at::Tensor> &dht,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout);

at::Tensor npu_chunk_bwd_dv_local(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &d_o,
    const at::Tensor &g,
    double scale,
    int64_t chunk_size,
    const c10::optional<at::Tensor> &g_gamma,
    const c10::optional<at::Tensor> &A,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices);

at::Tensor npu_prepare_wy_repr_bwd_da(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    const at::Tensor &dw,
    const at::Tensor &du,
    const at::Tensor &g,
    int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_chunk_bwd_dqkwg(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &g,
    const at::Tensor &h,
    const at::Tensor &dox,
    const at::Tensor &dh,
    const at::Tensor &dv,
    int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    const c10::optional<at::Tensor> &w,
    const c10::optional<at::Tensor> &g_gamma,
    c10::optional<double> scale,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout);

at::Tensor npu_chunk_fwd_o(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &h,
    double scale,
    const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &g_gamma,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> transpose_state_layout);

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_chunk_gated_delta_rule_fwd_h(
    const at::Tensor &k,
    const at::Tensor &w,
    const at::Tensor &u,
    const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &gk,
    const c10::optional<at::Tensor> &initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> save_new_value,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout);

std::tuple<at::Tensor, at::Tensor> npu_recompute_w_u_fwd(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    int64_t chunk_size,
    const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &gK,
    c10::OptionalIntArrayRef cu_seqlens,
    c10::OptionalIntArrayRef chunk_indices);

at::Tensor npu_causal_conv1d(
    const at::Tensor &x,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias,
    const at::Tensor &conv_states,
    at::OptionalIntArrayRef query_start_loc,
    at::OptionalIntArrayRef cache_indices,
    at::OptionalIntArrayRef initial_state_mode,
    at::OptionalIntArrayRef num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode,
    int64_t head_num);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_causal_conv1d_bwd(
    const at::Tensor &x,
    const c10::optional<at::Tensor> &y,
    const at::Tensor &weight,
    const at::Tensor &dy,
    const c10::optional<at::Tensor> &initial_state,
    const c10::optional<at::Tensor> &dht,
    at::OptionalIntArrayRef query_start_loc,
    int64_t activation,
    c10::string_view input_layout);

at::Tensor npu_solve_tri(
    const at::Tensor &x,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    c10::string_view layout);

}  // namespace op_api

namespace {

using OptionalIntVector = c10::optional<std::vector<int64_t>>;

OptionalIntVector optional_int_array(const py::object &value)
{
    if (value.is_none()) {
        return c10::nullopt;
    }
    return value.cast<std::vector<int64_t>>();
}

at::OptionalIntArrayRef optional_int_array_ref(const OptionalIntVector &value)
{
    if (!value.has_value()) {
        return c10::nullopt;
    }
    return at::IntArrayRef(value.value());
}

c10::optional<at::Tensor> optional_tensor(const py::object &value)
{
    if (value.is_none()) {
        return c10::nullopt;
    }
    return value.cast<at::Tensor>();
}

at::Tensor tensor_or_undefined(const py::object &value)
{
    if (value.is_none()) {
        return at::Tensor();
    }
    return value.cast<at::Tensor>();
}

template <typename T>
c10::optional<T> optional_value(const py::object &value)
{
    if (value.is_none()) {
        return c10::nullopt;
    }
    return value.cast<T>();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> py_npu_prepare_wy_repr_bwd_full(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    const at::Tensor &dA,
    const at::Tensor &dw,
    const at::Tensor &du,
    const at::Tensor &g,
    int64_t chunk_size,
    const py::object &cu_seqlens,
    const py::object &chunk_indices)
{
    const auto cu_seqlens_vec = optional_int_array(cu_seqlens);
    const auto chunk_indices_vec = optional_int_array(chunk_indices);
    return op_api::npu_prepare_wy_repr_bwd_full(
        k, v, beta, A, dA, dw, du, g, chunk_size,
        optional_int_array_ref(cu_seqlens_vec),
        optional_int_array_ref(chunk_indices_vec));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> py_npu_chunk_gated_delta_rule_bwd_dhu(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &w,
    const at::Tensor &d_o,
    const at::Tensor &dv,
    double scale,
    int64_t chunk_size,
    const py::object &g,
    const py::object &gK,
    const py::object &h0,
    const py::object &dht,
    const py::object &cu_seqlens,
    const py::object &chunk_indices,
    const py::object &use_exp2,
    const py::object &transpose_state_layout)
{
    const auto cu_seqlens_vec = optional_int_array(cu_seqlens);
    const auto chunk_indices_vec = optional_int_array(chunk_indices);
    return op_api::npu_chunk_gated_delta_rule_bwd_dhu(
        q, k, w, d_o, dv, scale, chunk_size,
        optional_tensor(g),
        optional_tensor(gK),
        optional_tensor(h0),
        optional_tensor(dht),
        optional_int_array_ref(cu_seqlens_vec),
        optional_int_array_ref(chunk_indices_vec),
        optional_value<bool>(use_exp2),
        optional_value<bool>(transpose_state_layout));
}

at::Tensor py_npu_chunk_bwd_dv_local(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &d_o,
    const at::Tensor &g,
    double scale,
    int64_t chunk_size,
    const py::object &g_gamma,
    const py::object &A,
    const py::object &cu_seqlens,
    const py::object &chunk_indices)
{
    const auto cu_seqlens_vec = optional_int_array(cu_seqlens);
    const auto chunk_indices_vec = optional_int_array(chunk_indices);
    return op_api::npu_chunk_bwd_dv_local(
        q, k, d_o, g, scale, chunk_size,
        optional_tensor(g_gamma),
        optional_tensor(A),
        optional_int_array_ref(cu_seqlens_vec),
        optional_int_array_ref(chunk_indices_vec));
}

at::Tensor py_npu_prepare_wy_repr_bwd_da(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    const at::Tensor &dw,
    const at::Tensor &du,
    const at::Tensor &g,
    int64_t chunk_size,
    const py::object &cu_seqlens,
    const py::object &chunk_indices)
{
    const auto cu_seqlens_vec = optional_int_array(cu_seqlens);
    const auto chunk_indices_vec = optional_int_array(chunk_indices);
    return op_api::npu_prepare_wy_repr_bwd_da(
        k, v, beta, A, dw, du, g, chunk_size,
        optional_int_array_ref(cu_seqlens_vec),
        optional_int_array_ref(chunk_indices_vec));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> py_npu_chunk_bwd_dqkwg(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &g,
    const at::Tensor &h,
    const at::Tensor &dox,
    const at::Tensor &dh,
    const at::Tensor &dv,
    int64_t chunk_size,
    const py::object &cu_seqlens,
    const py::object &chunk_indices,
    const py::object &w,
    const py::object &g_gamma,
    const py::object &scale,
    const py::object &use_exp2,
    const py::object &transpose_state_layout)
{
    const auto cu_seqlens_vec = optional_int_array(cu_seqlens);
    const auto chunk_indices_vec = optional_int_array(chunk_indices);
    return op_api::npu_chunk_bwd_dqkwg(
        q, k, v, g, h, dox, dh, dv, chunk_size,
        optional_int_array_ref(cu_seqlens_vec),
        optional_int_array_ref(chunk_indices_vec),
        optional_tensor(w),
        optional_tensor(g_gamma),
        optional_value<double>(scale),
        optional_value<bool>(use_exp2),
        optional_value<bool>(transpose_state_layout));
}

at::Tensor py_npu_chunk_fwd_o(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &h,
    double scale,
    const py::object &g,
    const py::object &g_gamma,
    const py::object &cu_seqlens,
    const py::object &chunk_indices,
    const py::object &chunk_size,
    const py::object &transpose_state_layout)
{
    const auto cu_seqlens_vec = optional_int_array(cu_seqlens);
    const auto chunk_indices_vec = optional_int_array(chunk_indices);
    return op_api::npu_chunk_fwd_o(
        q, k, v, h, scale,
        optional_tensor(g),
        optional_tensor(g_gamma),
        optional_int_array_ref(cu_seqlens_vec),
        optional_int_array_ref(chunk_indices_vec),
        optional_value<int64_t>(chunk_size),
        optional_value<bool>(transpose_state_layout));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> py_npu_chunk_gated_delta_rule_fwd_h(
    const at::Tensor &k,
    const at::Tensor &w,
    const at::Tensor &u,
    const py::object &g,
    const py::object &gk,
    const py::object &initial_state,
    const py::object &output_final_state,
    const py::object &chunk_size,
    const py::object &save_new_value,
    const py::object &cu_seqlens,
    const py::object &chunk_indices,
    const py::object &use_exp2,
    const py::object &transpose_state_layout)
{
    const auto cu_seqlens_vec = optional_int_array(cu_seqlens);
    const auto chunk_indices_vec = optional_int_array(chunk_indices);
    return op_api::npu_chunk_gated_delta_rule_fwd_h(
        k, w, u,
        optional_tensor(g),
        optional_tensor(gk),
        optional_tensor(initial_state),
        optional_value<bool>(output_final_state),
        optional_value<int64_t>(chunk_size),
        optional_value<bool>(save_new_value),
        optional_int_array_ref(cu_seqlens_vec),
        optional_int_array_ref(chunk_indices_vec),
        optional_value<bool>(use_exp2),
        optional_value<bool>(transpose_state_layout));
}

std::tuple<at::Tensor, at::Tensor> py_npu_recompute_w_u_fwd(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    int64_t chunk_size,
    const py::object &g,
    const py::object &gk,
    const py::object &cu_seqlens,
    const py::object &chunk_indices)
{
    const auto cu_seqlens_vec = optional_int_array(cu_seqlens);
    const auto chunk_indices_vec = optional_int_array(chunk_indices);
    return op_api::npu_recompute_w_u_fwd(
        k, v, beta, A, chunk_size,
        optional_tensor(g),
        optional_tensor(gk),
        optional_int_array_ref(cu_seqlens_vec),
        optional_int_array_ref(chunk_indices_vec));
}

at::Tensor py_npu_causal_conv1d(
    const at::Tensor &x,
    const at::Tensor &weight,
    const py::object &bias,
    const py::object &conv_states,
    const py::object &query_start_loc,
    const py::object &cache_indices,
    const py::object &initial_state_mode,
    const py::object &num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode,
    int64_t head_num)
{
    const auto query_start_loc_vec = optional_int_array(query_start_loc);
    const auto cache_indices_vec = optional_int_array(cache_indices);
    const auto initial_state_mode_vec = optional_int_array(initial_state_mode);
    const auto num_accepted_tokens_vec = optional_int_array(num_accepted_tokens);
    return op_api::npu_causal_conv1d(
        x, weight,
        optional_tensor(bias),
        tensor_or_undefined(conv_states),
        optional_int_array_ref(query_start_loc_vec),
        optional_int_array_ref(cache_indices_vec),
        optional_int_array_ref(initial_state_mode_vec),
        optional_int_array_ref(num_accepted_tokens_vec),
        activation_mode, pad_slot_id, run_mode, head_num);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> py_npu_causal_conv1d_bwd(
    const at::Tensor &x,
    const py::object &y,
    const at::Tensor &weight,
    const at::Tensor &dy,
    const py::object &initial_state,
    const py::object &dht,
    const py::object &query_start_loc,
    int64_t activation,
    const std::string &input_layout)
{
    const auto query_start_loc_vec = optional_int_array(query_start_loc);
    return op_api::npu_causal_conv1d_bwd(
        x,
        optional_tensor(y),
        weight,
        dy,
        optional_tensor(initial_state),
        optional_tensor(dht),
        optional_int_array_ref(query_start_loc_vec),
        activation,
        c10::string_view(input_layout.data(), input_layout.size()));
}

at::Tensor py_npu_solve_tri(
    const at::Tensor &x,
    const py::object &cu_seqlens,
    const py::object &chunk_indices,
    const std::string &layout)
{
    const auto cu_seqlens_vec = optional_int_array(cu_seqlens);
    const auto chunk_indices_vec = optional_int_array(chunk_indices);
    return op_api::npu_solve_tri(
        x,
        optional_int_array_ref(cu_seqlens_vec),
        optional_int_array_ref(chunk_indices_vec),
        c10::string_view(layout.data(), layout.size()));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Direct FLA NPU Ascend C bindings that bypass torch.ops dispatcher.";

    m.def("npu_fast_gelu_custom", &op_api::npu_fast_gelu_custom, py::arg("self"));
    m.def("npu_fast_gelu_custom_backward", &op_api::npu_fast_gelu_custom_backward, py::arg("grad"), py::arg("self"));
    m.def(
        "npu_causal_conv1d",
        &py_npu_causal_conv1d,
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("conv_states") = py::none(),
        py::kw_only(),
        py::arg("query_start_loc") = py::none(),
        py::arg("cache_indices") = py::none(),
        py::arg("initial_state_mode") = py::none(),
        py::arg("num_accepted_tokens") = py::none(),
        py::arg("activation_mode") = 0,
        py::arg("pad_slot_id") = -1,
        py::arg("run_mode") = 0,
        py::arg("head_num") = 0);
    m.def(
        "npu_causal_conv1d_bwd",
        &py_npu_causal_conv1d_bwd,
        py::arg("x"),
        py::arg("y"),
        py::arg("weight"),
        py::arg("dy"),
        py::arg("initial_state") = py::none(),
        py::arg("dht") = py::none(),
        py::kw_only(),
        py::arg("query_start_loc") = py::none(),
        py::arg("activation") = 0,
        py::arg("input_layout") = "BSND");
    m.def(
        "npu_prepare_wy_repr_bwd_full",
        &py_npu_prepare_wy_repr_bwd_full,
        py::arg("k"),
        py::arg("v"),
        py::arg("beta"),
        py::arg("A"),
        py::arg("dA"),
        py::arg("dw"),
        py::arg("du"),
        py::arg("g"),
        py::arg("chunk_size"),
        py::kw_only(),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none());
    m.def(
        "npu_chunk_gated_delta_rule_bwd_dhu",
        &py_npu_chunk_gated_delta_rule_bwd_dhu,
        py::arg("q"),
        py::arg("k"),
        py::arg("w"),
        py::arg("d_o"),
        py::arg("dv"),
        py::arg("scale"),
        py::arg("chunk_size"),
        py::kw_only(),
        py::arg("g") = py::none(),
        py::arg("gK") = py::none(),
        py::arg("h0") = py::none(),
        py::arg("dht") = py::none(),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(),
        py::arg("use_exp2") = false,
        py::arg("transpose_state_layout") = false);
    m.def(
        "npu_chunk_bwd_dv_local",
        &py_npu_chunk_bwd_dv_local,
        py::arg("q"),
        py::arg("k"),
        py::arg("d_o"),
        py::arg("g"),
        py::arg("scale"),
        py::arg("chunk_size"),
        py::kw_only(),
        py::arg("g_gamma") = py::none(),
        py::arg("A") = py::none(),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none());
    m.def(
        "npu_prepare_wy_repr_bwd_da",
        &py_npu_prepare_wy_repr_bwd_da,
        py::arg("k"),
        py::arg("v"),
        py::arg("beta"),
        py::arg("A"),
        py::arg("dw"),
        py::arg("du"),
        py::arg("g"),
        py::kw_only(),
        py::arg("chunk_size"),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none());
    m.def(
        "npu_chunk_bwd_dqkwg",
        &py_npu_chunk_bwd_dqkwg,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("g"),
        py::arg("h"),
        py::arg("dox"),
        py::arg("dh"),
        py::arg("dv"),
        py::arg("chunk_size"),
        py::kw_only(),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(),
        py::arg("w") = py::none(),
        py::arg("g_gamma") = py::none(),
        py::arg("scale") = py::none(),
        py::arg("use_exp2") = py::none(),
        py::arg("transpose_state_layout") = py::none());
    m.def(
        "npu_chunk_fwd_o",
        &py_npu_chunk_fwd_o,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("h"),
        py::arg("scale"),
        py::kw_only(),
        py::arg("g") = py::none(),
        py::arg("g_gamma") = py::none(),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(),
        py::arg("chunk_size") = py::none(),
        py::arg("transpose_state_layout") = false);
    m.def(
        "npu_chunk_gated_delta_rule_fwd_h",
        &py_npu_chunk_gated_delta_rule_fwd_h,
        py::arg("k"),
        py::arg("w"),
        py::arg("u"),
        py::arg("g") = py::none(),
        py::kw_only(),
        py::arg("gk") = py::none(),
        py::arg("initial_state") = py::none(),
        py::arg("output_final_state") = false,
        py::arg("chunk_size") = py::none(),
        py::arg("save_new_value") = true,
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(),
        py::arg("use_exp2") = false,
        py::arg("transpose_state_layout") = false);
    m.def(
        "npu_recompute_w_u_fwd",
        &py_npu_recompute_w_u_fwd,
        py::arg("k"),
        py::arg("v"),
        py::arg("beta"),
        py::arg("A"),
        py::arg("chunk_size"),
        py::kw_only(),
        py::arg("g") = py::none(),
        py::arg("gk") = py::none(),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none());
    m.def(
        "npu_solve_tri",
        &py_npu_solve_tri,
        py::arg("x"),
        py::kw_only(),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(),
        py::arg("layout") = "bsnd");
}
