#include <vector>
#include <torch/extension.h>

#include "thin_launcher/runtime.h"

namespace fla_npu_thin {

at::Tensor npu_recurrent_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    double scale,
    const c10::optional<at::Tensor>& actual_seq_lengths,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    uint64_t stream);

at::Tensor npu_causal_conv1d(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& conv_states,
    const std::vector<int64_t>& query_start_loc,
    const std::vector<int64_t>& cache_indices,
    const std::vector<int64_t>& initial_state_mode,
    const std::vector<int64_t>& num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode,
    int64_t head_num,
    uint64_t stream);

at::Tensor npu_kda_gate_cumsum(
    const at::Tensor& g,
    const c10::optional<at::Tensor>& A_log,
    const c10::optional<at::Tensor>& dt_bias,
    const std::vector<int64_t>& cu_seqlens,
    int64_t chunk_size,
    bool use_gate_in_kernel,
    bool safe_gate,
    double lower_bound,
    uint64_t stream);


at::Tensor npu_chunk_local_cumsum(
    const at::Tensor& g,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    int64_t chunk_size,
    bool reverse,
    double scale,
    bool head_first,
    const std::string& output_dtype,
    uint64_t stream);


at::Tensor npu_chunk_scaled_dot_kkt(
    const at::Tensor& k,
    const at::Tensor& g,
    const at::Tensor& beta,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    int64_t chunk_size,
    uint64_t stream);





std::vector<at::Tensor> npu_recompute_w_u_fwd(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& beta,
    const at::Tensor& A,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    int64_t chunk_size,
    uint64_t stream);


std::vector<at::Tensor> npu_prepare_wy_repr_bwd_full(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& beta,
    const at::Tensor& A,
    const at::Tensor& dA,
    const at::Tensor& dw,
    const at::Tensor& du,
    const at::Tensor& g,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    int64_t chunk_size,
    uint64_t stream);


std::vector<at::Tensor> npu_prepare_wy_repr_bwd(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& beta,
    const at::Tensor& A,
    const at::Tensor& dw,
    const at::Tensor& du,
    const at::Tensor& g,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    int64_t chunk_size,
    uint64_t stream);


at::Tensor npu_chunk_bwd_dv_local(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& d_o,
    const at::Tensor& g,
    const c10::optional<at::Tensor>& g_gamma,
    const c10::optional<at::Tensor>& A,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    double scale,
    int64_t chunk_size,
    uint64_t stream);


at::Tensor npu_prepare_wy_repr_bwd_da(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& beta,
    const at::Tensor& A,
    const at::Tensor& dw,
    const at::Tensor& du,
    const at::Tensor& g,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    int64_t chunk_size,
    uint64_t stream);


at::Tensor npu_fast_gelu_custom(
    const at::Tensor& self,
    uint64_t stream);


at::Tensor npu_fast_gelu_custom_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    uint64_t stream);


std::vector<at::Tensor> npu_chunk_bwd_dqkwg(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    const at::Tensor& h,
    const at::Tensor& dox,
    const at::Tensor& dh,
    const at::Tensor& dv,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    const c10::optional<at::Tensor>& w,
    const c10::optional<at::Tensor>& g_gamma,
    float scale,
    int64_t chunk_size,
    bool use_exp2,
    bool transpose_state_layout,
    uint64_t stream);


std::vector<at::Tensor> npu_chunk_gated_delta_rule_fwd_h(
    const at::Tensor& k,
    const at::Tensor& w,
    const at::Tensor& u,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    const c10::optional<at::Tensor>& initial_state,
    bool output_final_state,
    int64_t chunk_size,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    bool state_v_first,
    uint64_t stream);


std::vector<at::Tensor> npu_chunk_fwd_h(
    const at::Tensor& k,
    const at::Tensor& w,
    const at::Tensor& u,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    const c10::optional<at::Tensor>& initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool save_new_value,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    bool use_exp2,
    bool state_v_first,
    uint64_t stream);


at::Tensor npu_chunk_fwd_o(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& h,
    const c10::optional<at::Tensor>& g,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    double scale,
    int64_t chunk_size,
    bool use_exp2,
    bool transpose_state_layout,
    const std::string& output_layout,
    uint64_t stream);


std::vector<at::Tensor> npu_chunk_gated_delta_rule_bwd_dhu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& w,
    const at::Tensor& d_o,
    const at::Tensor& dv,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gK,
    const c10::optional<at::Tensor>& h0,
    const c10::optional<at::Tensor>& dht,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    double scale,
    int64_t chunk_size,
    bool use_exp2,
    uint64_t stream);


std::vector<at::Tensor> npu_recurrent_kda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    const at::Tensor& beta,
    const c10::optional<at::Tensor>& initial_state,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& A_log,
    const c10::optional<at::Tensor>& dt_bias,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const std::string& layout,
    double scale,
    bool output_final_state,
    bool inplace_final_state,
    bool use_qk_l2norm_in_kernel,
    bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel,
    bool allow_neg_eigval,
    bool safe_gate,
    double lower_bound,
    bool state_v_first,
    uint64_t stream);





std::vector<at::Tensor> npu_causal_conv1d_bwd(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& y,
    const at::Tensor& weight,
    const at::Tensor& dy,
    const c10::optional<at::Tensor>& initial_state,
    const c10::optional<at::Tensor>& dht,
    const std::vector<int64_t>& query_start_loc,
    int64_t activation,
    const std::string& input_layout,
    uint64_t stream);





std::vector<at::Tensor> npu_chunk_kda_bwd_intra(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& gk,
    const at::Tensor& beta,
    const at::Tensor& dAqk,
    const at::Tensor& dAkk,
    const at::Tensor& dq,
    const at::Tensor& dk,
    const at::Tensor& db,
    const at::Tensor& dg,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    int64_t chunk_size,
    bool safe_gate,
    const std::string& layout,
    uint64_t stream);


std::vector<at::Tensor> npu_chunk_kda_bwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& beta,
    const at::Tensor& gk,
    const at::Tensor& Aqk,
    const at::Tensor& Akk,
    const c10::optional<at::Tensor>& w,
    const c10::optional<at::Tensor>& qg,
    const c10::optional<at::Tensor>& kg,
    const c10::optional<at::Tensor>& v_new,
    const c10::optional<at::Tensor>& h,
    const at::Tensor& d_o,
    const c10::optional<at::Tensor>& raw_g,
    const c10::optional<at::Tensor>& A_log,
    const c10::optional<at::Tensor>& dt_bias,
    const c10::optional<at::Tensor>& initial_state,
    const c10::optional<at::Tensor>& dht,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    double scale,
    int64_t chunk_size,
    bool safe_gate,
    bool use_gate_in_kernel,
    double lower_bound,
    bool disable_recompute,
    bool use_exp2,
    bool state_v_first,
    uint64_t stream);
















std::vector<at::Tensor> npu_chunk_gated_delta_rule_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    const at::Tensor& beta,
    const c10::optional<at::Tensor>& a_log,
    const c10::optional<at::Tensor>& dt_bias,
    const c10::optional<at::Tensor>& initial_state,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    const std::string& layout,
    double scale,
    int64_t chunk_size,
    bool use_exp2,
    bool use_qk_l2norm_in_kernel,
    bool allow_neg_eigval,
    bool state_v_first,
    bool output_final_state,
    bool disable_recompute,
    bool return_intermediate_states,
    bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel,
    uint64_t stream);


std::vector<at::Tensor> npu_chunk_gated_delta_rule_bwd_finalize(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& v_new,
    const at::Tensor& d_o,
    const at::Tensor& du,
    const at::Tensor& g,
    const at::Tensor& beta,
    const at::Tensor& h,
    const at::Tensor& dh,
    const at::Tensor& a,
    const c10::optional<at::Tensor>& q_rstd,
    const c10::optional<at::Tensor>& k_rstd,
    const c10::optional<at::Tensor>& beta_raw,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    double scale,
    int64_t chunk_size,
    bool use_qk_l2_norm_in_kernel,
    bool use_beta_sigmoid_in_kernel,
    bool use_gate_in_kernel,
    bool state_v_first,
    bool use_exp2,
    uint64_t stream);








at::Tensor npu_solve_tri(
    const at::Tensor& x,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    const std::string& layout,
    uint64_t stream);





std::vector<at::Tensor> npu_chunk_kda_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    const at::Tensor& beta,
    const c10::optional<at::Tensor>& A_log,
    const c10::optional<at::Tensor>& dt_bias,
    const c10::optional<at::Tensor>& initial_state,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    const std::string& layout,
    double scale,
    int64_t chunk_size,
    bool safe_gate,
    double lower_bound,
    bool use_gate_in_kernel,
    bool state_v_first,
    bool output_final_state,
    bool disable_recompute,
    bool return_intermediate_states,
    uint64_t stream);





std::vector<at::Tensor> npu_chunk_gated_delta_rule_fwd_prepare(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    const at::Tensor& beta,
    const c10::optional<at::Tensor>& a_log,
    const c10::optional<at::Tensor>& dt_bias,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    int64_t chunk_size,
    bool allow_neg_eigval,
    bool use_exp2,
    bool output_a,
    bool use_beta_sigmoid_in_kernel,
    bool use_gate_in_kernel,
    uint64_t stream);

}  // namespace fla_npu_thin

PYBIND11_MODULE(_C_thin, m) {
  using namespace fla_npu_thin;
  m.def("init", [](const std::string& path) {
    Runtime::instance().init(path);
  });
  m.def(
      "npu_recurrent_gated_delta_rule",
      &npu_recurrent_gated_delta_rule, py::arg("query"), py::arg("key"),
      py::arg("value"), py::arg("state"), py::arg("beta"),
      py::arg("scale"), py::arg("actual_seq_lengths"),
      py::arg("ssm_state_indices"), py::arg("num_accepted_tokens"),
      py::arg("g"), py::arg("gk"), py::arg("stream"));
  m.def(
      "npu_causal_conv1d",
      &npu_causal_conv1d, py::arg("x"), py::arg("weight"),
      py::arg("bias"), py::arg("conv_states"),
      py::arg("query_start_loc"), py::arg("cache_indices"),
      py::arg("initial_state_mode"), py::arg("num_accepted_tokens"),
      py::arg("activation_mode"), py::arg("pad_slot_id"),
      py::arg("run_mode"), py::arg("head_num"),
      py::arg("stream"));
  m.def(
      "npu_kda_gate_cumsum",
      &npu_kda_gate_cumsum, py::arg("g"), py::arg("A_log"),
      py::arg("dt_bias"), py::arg("cu_seqlens"),
      py::arg("chunk_size"), py::arg("use_gate_in_kernel"),
      py::arg("safe_gate"), py::arg("lower_bound"), py::arg("stream"));
  m.def(
      "npu_chunk_local_cumsum",
      &npu_chunk_local_cumsum, py::arg("g"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("chunk_size"),
      py::arg("reverse"),
      py::arg("scale"),
      py::arg("head_first"),
      py::arg("output_dtype"),
      py::arg("stream"));
  m.def(
      "npu_chunk_scaled_dot_kkt",
      &npu_chunk_scaled_dot_kkt, py::arg("k"),
      py::arg("g"),
      py::arg("beta"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("chunk_size"),
      py::arg("stream"));
  
  m.def(
      "npu_recompute_w_u_fwd",
      &npu_recompute_w_u_fwd, py::arg("k"),
      py::arg("v"),
      py::arg("beta"),
      py::arg("A"),
      py::arg("g"),
      py::arg("gk"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("chunk_size"),
      py::arg("stream"));
  m.def(
      "npu_prepare_wy_repr_bwd_full",
      &npu_prepare_wy_repr_bwd_full, py::arg("k"),
      py::arg("v"),
      py::arg("beta"),
      py::arg("A"),
      py::arg("dA"),
      py::arg("dw"),
      py::arg("du"),
      py::arg("g"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("chunk_size"),
      py::arg("stream"));
  m.def(
      "npu_prepare_wy_repr_bwd",
      &npu_prepare_wy_repr_bwd, py::arg("k"),
      py::arg("v"),
      py::arg("beta"),
      py::arg("A"),
      py::arg("dw"),
      py::arg("du"),
      py::arg("g"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("chunk_size"),
      py::arg("stream"));
  m.def(
      "npu_chunk_bwd_dv_local",
      &npu_chunk_bwd_dv_local, py::arg("q"),
      py::arg("k"),
      py::arg("d_o"),
      py::arg("g"),
      py::arg("g_gamma"),
      py::arg("A"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("scale"),
      py::arg("chunk_size"),
      py::arg("stream"));
  m.def(
      "npu_prepare_wy_repr_bwd_da",
      &npu_prepare_wy_repr_bwd_da, py::arg("k"),
      py::arg("v"),
      py::arg("beta"),
      py::arg("A"),
      py::arg("dw"),
      py::arg("du"),
      py::arg("g"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("chunk_size"),
      py::arg("stream"));
  m.def(
      "npu_fast_gelu_custom",
      &npu_fast_gelu_custom, py::arg("self"),
      py::arg("stream"));
  m.def(
      "npu_fast_gelu_custom_backward",
      &npu_fast_gelu_custom_backward, py::arg("grad"),
      py::arg("self"),
      py::arg("stream"));
  m.def(
      "npu_chunk_bwd_dqkwg",
      &npu_chunk_bwd_dqkwg, py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("g"),
      py::arg("h"),
      py::arg("dox"),
      py::arg("dh"),
      py::arg("dv"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("w"),
      py::arg("g_gamma"),
      py::arg("scale"),
      py::arg("chunk_size"),
      py::arg("use_exp2"),
      py::arg("transpose_state_layout"),
      py::arg("stream"));
  m.def(
      "npu_chunk_gated_delta_rule_fwd_h",
      &npu_chunk_gated_delta_rule_fwd_h, py::arg("k"),
      py::arg("w"),
      py::arg("u"),
      py::arg("g"),
      py::arg("gk"),
      py::arg("initial_state"),
      py::arg("output_final_state"),
      py::arg("chunk_size"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("state_v_first"),
      py::arg("stream"));
  m.def(
      "npu_chunk_fwd_h",
      &npu_chunk_fwd_h, py::arg("k"),
      py::arg("w"),
      py::arg("u"),
      py::arg("g"),
      py::arg("gk"),
      py::arg("initial_state"),
      py::arg("output_final_state"),
      py::arg("chunk_size"),
      py::arg("save_new_value"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("use_exp2"),
      py::arg("state_v_first"),
      py::arg("stream"));
  m.def(
      "npu_chunk_fwd_o",
      &npu_chunk_fwd_o, py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("h"),
      py::arg("g"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("scale"),
      py::arg("chunk_size"),
      py::arg("use_exp2"),
      py::arg("transpose_state_layout"),
      py::arg("output_layout"),
      py::arg("stream"));
  m.def(
      "npu_chunk_gated_delta_rule_bwd_dhu",
      &npu_chunk_gated_delta_rule_bwd_dhu, py::arg("q"),
      py::arg("k"),
      py::arg("w"),
      py::arg("d_o"),
      py::arg("dv"),
      py::arg("g"),
      py::arg("gK"),
      py::arg("h0"),
      py::arg("dht"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("scale"),
      py::arg("chunk_size"),
      py::arg("use_exp2"),
      py::arg("stream"));
  m.def(
      "npu_recurrent_kda",
      &npu_recurrent_kda, py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("g"),
      py::arg("beta"),
      py::arg("initial_state"),
      py::arg("cu_seqlens"),
      py::arg("ssm_state_indices"),
      py::arg("A_log"),
      py::arg("dt_bias"),
      py::arg("num_accepted_tokens"),
      py::arg("layout"),
      py::arg("scale"),
      py::arg("output_final_state"),
      py::arg("inplace_final_state"),
      py::arg("use_qk_l2norm_in_kernel"),
      py::arg("use_gate_in_kernel"),
      py::arg("use_beta_sigmoid_in_kernel"),
      py::arg("allow_neg_eigval"),
      py::arg("safe_gate"),
      py::arg("lower_bound"),
      py::arg("state_v_first"),
      py::arg("stream"));
  
  m.def(
      "npu_causal_conv1d_bwd",
      &npu_causal_conv1d_bwd, py::arg("x"),
      py::arg("y"),
      py::arg("weight"),
      py::arg("dy"),
      py::arg("initial_state"),
      py::arg("dht"),
      py::arg("query_start_loc"),
      py::arg("activation"),
      py::arg("input_layout"),
      py::arg("stream"));
  
  m.def(
      "npu_chunk_kda_bwd_intra",
      &npu_chunk_kda_bwd_intra, py::arg("q"),
      py::arg("k"),
      py::arg("gk"),
      py::arg("beta"),
      py::arg("dAqk"),
      py::arg("dAkk"),
      py::arg("dq"),
      py::arg("dk"),
      py::arg("db"),
      py::arg("dg"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("chunk_size"),
      py::arg("safe_gate"),
      py::arg("layout"),
      py::arg("stream"));
  m.def(
      "npu_chunk_kda_bwd",
      &npu_chunk_kda_bwd, py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("beta"),
      py::arg("gk"),
      py::arg("Aqk"),
      py::arg("Akk"),
      py::arg("w"),
      py::arg("qg"),
      py::arg("kg"),
      py::arg("v_new"),
      py::arg("h"),
      py::arg("d_o"),
      py::arg("raw_g"),
      py::arg("A_log"),
      py::arg("dt_bias"),
      py::arg("initial_state"),
      py::arg("dht"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("scale"),
      py::arg("chunk_size"),
      py::arg("safe_gate"),
      py::arg("use_gate_in_kernel"),
      py::arg("lower_bound"),
      py::arg("disable_recompute"),
      py::arg("use_exp2"),
      py::arg("state_v_first"),
      py::arg("stream"));
    
  
  
  m.def(
      "npu_chunk_gated_delta_rule_fwd",
      &npu_chunk_gated_delta_rule_fwd, py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("g"),
      py::arg("beta"),
      py::arg("a_log"),
      py::arg("dt_bias"),
      py::arg("initial_state"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("layout"),
      py::arg("scale"),
      py::arg("chunk_size"),
      py::arg("use_exp2"),
      py::arg("use_qk_l2norm_in_kernel"),
      py::arg("allow_neg_eigval"),
      py::arg("state_v_first"),
      py::arg("output_final_state"),
      py::arg("disable_recompute"),
      py::arg("return_intermediate_states"),
      py::arg("use_gate_in_kernel"),
      py::arg("use_beta_sigmoid_in_kernel"),
      py::arg("stream"));
  m.def(
      "npu_chunk_gated_delta_rule_bwd_finalize",
      &npu_chunk_gated_delta_rule_bwd_finalize, py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("v_new"),
      py::arg("do"),
      py::arg("du"),
      py::arg("g"),
      py::arg("beta"),
      py::arg("h"),
      py::arg("dh"),
      py::arg("a"),
      py::arg("q_rstd"),
      py::arg("k_rstd"),
      py::arg("beta_raw"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("scale"),
      py::arg("chunk_size"),
      py::arg("use_qk_l2_norm_in_kernel"),
      py::arg("use_beta_sigmoid_in_kernel"),
      py::arg("use_gate_in_kernel"),
      py::arg("state_v_first"),
      py::arg("use_exp2"),
      py::arg("stream"));
  
  
  m.def(
      "npu_solve_tri",
      &npu_solve_tri, py::arg("x"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("layout"),
      py::arg("stream"));
  
  m.def(
      "npu_chunk_kda_fwd",
      &npu_chunk_kda_fwd, py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("g"),
      py::arg("beta"),
      py::arg("A_log"),
      py::arg("dt_bias"),
      py::arg("initial_state"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("layout"),
      py::arg("scale"),
      py::arg("chunk_size"),
      py::arg("safe_gate"),
      py::arg("lower_bound"),
      py::arg("use_gate_in_kernel"),
      py::arg("state_v_first"),
      py::arg("output_final_state"),
      py::arg("disable_recompute"),
      py::arg("return_intermediate_states"),
      py::arg("stream"));
  
  m.def(
      "npu_chunk_gated_delta_rule_fwd_prepare",
      &npu_chunk_gated_delta_rule_fwd_prepare, py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("g"),
      py::arg("beta"),
      py::arg("a_log"),
      py::arg("dt_bias"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("chunk_size"),
      py::arg("allow_neg_eigval"),
      py::arg("use_exp2"),
      py::arg("output_a"),
      py::arg("use_beta_sigmoid_in_kernel"),
      py::arg("use_gate_in_kernel"),
      py::arg("stream"));
}