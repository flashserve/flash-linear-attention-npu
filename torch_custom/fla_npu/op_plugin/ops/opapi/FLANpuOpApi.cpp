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
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include "op_plugin/include/npu_cpp_extension.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using namespace op_plugin::utils;
using namespace op_infer;

namespace {
int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

int64_t GetKdaSeqNum(int64_t batch, at::OptionalIntArrayRef cu_seqlens)
{
    if (!cu_seqlens.has_value()) {
        return batch;
    }
    auto cu = cu_seqlens.value();
    return static_cast<int64_t>(cu.size()) - 1;
}

int64_t GetKdaTotalChunks(int64_t batch, int64_t seqlen, int64_t chunk_size,
                          at::OptionalIntArrayRef cu_seqlens,
                          at::OptionalIntArrayRef chunk_indices)
{
    if (chunk_indices.has_value()) {
        return static_cast<int64_t>(chunk_indices.value().size()) / 2;
    }
    if (!cu_seqlens.has_value()) {
        return CeilDiv(seqlen, chunk_size);
    }
    (void)batch;
    int64_t total = 0;
    auto cu = cu_seqlens.value();
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
        total += CeilDiv(cu[i + 1] - cu[i], chunk_size);
    }
    return total;
}
} // namespace


::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_prepare_wy_repr_bwd_full(const at::Tensor & k, const at::Tensor & v, const at::Tensor & beta, const at::Tensor & A, const at::Tensor & dA, const at::Tensor & dw, const at::Tensor & du, const at::Tensor & g, int64_t chunk_size, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    at::Tensor dk = at::empty_like(k);
    at::Tensor dv = at::empty_like(v);
    at::Tensor dbeta = at::empty_like(beta);
    at::Tensor dg = at::empty_like(g);
    EXEC_NPU_CMD_EXT(
        aclnnPrepareWyReprBwdFull,
        k, v, beta, A, dA, dw, du, g,
        cu_seqlens, chunk_indices, chunk_size,
        dk, dv, dbeta, dg
    );
    return std::make_tuple(std::move(dk), std::move(dv), std::move(dbeta), std::move(dg));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_chunk_gated_delta_rule_bwd_dhu(
    const at::Tensor & q, 
    const at::Tensor & k, 
    const at::Tensor & w, 
    const at::Tensor & d_o, 
    const at::Tensor & dv, 
    double scale, 
    int64_t chunk_size, 
    const c10::optional<at::Tensor> & g, 
    const c10::optional<at::Tensor> & gK, 
    const c10::optional<at::Tensor> & h0, 
    const c10::optional<at::Tensor> & dht, 
    at::OptionalIntArrayRef cu_seqlens, 
    at::OptionalIntArrayRef chunk_indices, 
    c10::optional<bool> use_exp2, 
    c10::optional<bool> transpose_state_layout)
{
    TORCH_CHECK(
        (g.has_value() && g->defined()) || (gK.has_value() && gK->defined()),
        "npu_chunk_gated_delta_rule_bwd_dhu: either g or gK must be defined.");
    TORCH_CHECK(
        !transpose_state_layout.value_or(false),
        "npu_chunk_gated_delta_rule_bwd_dhu: transpose_state_layout=True is not supported.");

    auto q_size = q.sizes();
    auto dv_size = dv.sizes();
    int64_t B = q_size[0];
    int64_t H = q_size[1];
    int64_t T = q_size[2];
    int64_t K = q_size[3];
    int64_t V = dv_size[3];
    int64_t chunk_num = (T + chunk_size -1) / chunk_size; 

    if (chunk_indices.has_value()) {
        auto chunk_indices_ref = chunk_indices.value();
        chunk_num = chunk_indices_ref.size() / 2;
    }

    // 创建输出tensor（PTA推荐）
    at::Tensor dv2 = at::empty_like(dv);
    at::Tensor dh = at::empty({B, H, chunk_num, K, V}, q.options());
    at::Tensor dh0;
    if (h0.has_value() && h0->defined()) {
        dh0 = at::empty({B, H, K, V}, q.options());
    } else {
        dh0 = at::Tensor();
    }

    // optional tensor处理
    const at::Tensor &g_  = c10::value_or_else(g,  [] { return at::Tensor(); });
    const at::Tensor &gK_ = c10::value_or_else(gK, [] { return at::Tensor(); });
    const at::Tensor &h0_ = c10::value_or_else(h0, [] { return at::Tensor(); });
    const at::Tensor &dht_ = c10::value_or_else(dht, [] { return at::Tensor(); });

    // 调用ACLNN算子
    EXEC_NPU_CMD_EXT(
        aclnnChunkGatedDeltaRuleBwdDhu,
        q, k, w, d_o, dv,
        g_, gK_, h0_, dht_,
        cu_seqlens, chunk_indices,
        scale, chunk_size,
        dh, dh0, dv2
    );
    return std::make_tuple(dh, dh0, dv2);
}

at::Tensor npu_chunk_bwd_dv_local(const at::Tensor & q, const at::Tensor & k, const at::Tensor & d_o, const at::Tensor & g, double scale, int64_t chunk_size, const c10::optional<at::Tensor> & g_gamma, const c10::optional<at::Tensor> & A, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    at::Tensor dv = at::empty_like(d_o);
    const at::Tensor &g_gamma_ = c10::value_or_else(g_gamma, [] { return at::Tensor(); });
    const at::Tensor &A_ = c10::value_or_else(A, [] { return at::Tensor(); });

    EXEC_NPU_CMD_EXT(
        aclnnChunkBwdDvLocal,
        q, k, d_o, g, g_gamma_, A_,
        cu_seqlens, chunk_indices, scale, chunk_size,
        dv
    );
    return dv;
}

at::Tensor npu_prepare_wy_repr_bwd_da(const at::Tensor & k, const at::Tensor & v, const at::Tensor & beta, const at::Tensor & A, const at::Tensor & dw, const at::Tensor & du, const at::Tensor & g, int64_t chunk_size, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    at::Tensor dA = at::empty_like(A);
    EXEC_NPU_CMD_EXT(
        aclnnPrepareWyReprBwdDa,
        k, v, beta, A, dw, du, g,
        cu_seqlens, chunk_indices, chunk_size,
        dA
    );
    return dA;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_chunk_bwd_dqkwg(const at::Tensor & q, const at::Tensor & k, const at::Tensor & v, const at::Tensor & g, const at::Tensor & h, const at::Tensor & dox, const at::Tensor & dh, const at::Tensor & dv, int64_t chunk_size, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices, const c10::optional<at::Tensor> & w, const c10::optional<at::Tensor> & g_gamma, c10::optional<double> scale, c10::optional<bool> use_exp2, c10::optional<bool> transpose_state_layout)
{
    // 创建输出tensor
    at::Tensor dq = at::empty_like(q);
    at::Tensor dk = at::empty_like(k);
    at::Tensor dw = at::empty_like(k);
    at::Tensor dg = at::empty_like(g);

    // scale处理
    float scale_real = static_cast<float>(scale.value_or(1.0));
    const at::Tensor &w_ = c10::value_or_else(w, [] { return at::Tensor(); });
    const at::Tensor &g_gamma_ = c10::value_or_else(g_gamma, [] { return at::Tensor(); });
    bool use_exp2_ = static_cast<bool>(use_exp2.value_or(0));
    bool transpose_state_layout_ = static_cast<bool>(transpose_state_layout.value_or(0));

    // 调用ACLNN算子
    EXEC_NPU_CMD_EXT(
        aclnnChunkBwdDqkwg,
        q, k, v, g, h,
        dox, dh, dv,
        cu_seqlens, chunk_indices, w_, g_gamma_, scale_real, chunk_size, use_exp2_, transpose_state_layout_,
        dq, dk, dw, dg
    );
    return std::make_tuple(dq, dk, dw, dg);
}

at::Tensor npu_chunk_fwd_o(
    const at::Tensor & q, 
    const at::Tensor & k, 
    const at::Tensor & v, 
    const at::Tensor & h, 
    double scale, 
    const c10::optional<at::Tensor> & g,
    const c10::optional<at::Tensor> & g_gamma,
    at::OptionalIntArrayRef cu_seqlens, 
    at::OptionalIntArrayRef chunk_indices, 
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> transpose_state_layout)
{
    // 创建输出tensor
    at::Tensor o = at::empty_like(v);

    // chunk_size默认值
    int64_t chunk_size_ = chunk_size.value_or(64);
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    (void)g_gamma;
    (void)transpose_state_layout;

    // 调用ACLNN算子
    EXEC_NPU_CMD_EXT(
        aclnnChunkFwdO,
        q, k, v, h, g_,
        cu_seqlens, chunk_indices, scale, chunk_size_,
        o
    );
    return o;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_chunk_gated_delta_rule_fwd_h(
    const at::Tensor & k, 
    const at::Tensor & w, 
    const at::Tensor & u, 
    const c10::optional<at::Tensor> & g, 
    const c10::optional<at::Tensor> & gk,
    const c10::optional<at::Tensor> & initial_state, 
    c10::optional<bool> output_final_state, 
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> save_new_value,
    at::OptionalIntArrayRef cu_seqlens, 
    at::OptionalIntArrayRef chunk_indices, 
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout)
{
    TORCH_CHECK(
        (g.has_value() && g->defined()) || (gk.has_value() && gk->defined()),
        "npu_chunk_gated_delta_rule_fwd_h: either g or gk must be defined.");
    TORCH_CHECK(
        save_new_value.value_or(true),
        "npu_chunk_gated_delta_rule_fwd_h: save_new_value is reserved and only true is supported.");
    TORCH_CHECK(
        !use_exp2.value_or(false),
        "npu_chunk_gated_delta_rule_fwd_h: use_exp2 is reserved and only false is supported.");
    TORCH_CHECK(
        !transpose_state_layout.value_or(false),
        "npu_chunk_gated_delta_rule_fwd_h: transpose_state_layout is reserved and only false is supported.");

    // optional 参数处理
    bool output_final_state_ = output_final_state.value_or(false);
    int64_t chunk_size_ = chunk_size.value_or(64);
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    const at::Tensor &gk_ = c10::value_or_else(gk, [] { return at::Tensor(); });
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });

    // 计算shape
    auto k_sizes = k.sizes();
    auto u_sizes = u.sizes();
    int64_t B = k_sizes[0];
    int64_t T = k_sizes[2];
    int64_t K = k_sizes[3];
    int64_t V = u_sizes[3];
    int64_t HV = u_sizes[1];

    int64_t NT = 0;
    if (chunk_indices.has_value()) {
        auto chunk_indices_ref = chunk_indices.value();
        NT = chunk_indices_ref.size() / 2;
    } else {
        NT = (T + chunk_size_ - 1) / chunk_size_;
    }

    // 创建输出 tensor
    at::Tensor h_out = at::zeros({B, HV, NT, K, V}, k.options());
    at::Tensor v_new_out = at::empty_like(u);
    at::Tensor final_state_out;
    if (output_final_state_) {
        int N = cu_seqlens.has_value() ? cu_seqlens->size() - 1 : B;
        auto state_options = initial_state.has_value() ? initial_state->options() : h_out.options();
        final_state_out = at::empty({N, HV, K, V}, state_options);
    } else {
        final_state_out = at::empty({1}, k.options());
    }

    // 调用ACLNN算子（两阶段调用：先获取工作空间大小，再执行）
    bool save_new_value_ = save_new_value.value_or(true);
    bool use_exp2_ = use_exp2.value_or(false);
    bool transpose_state_layout_ = transpose_state_layout.value_or(false);

    EXEC_NPU_CMD_EXT(
        aclnnChunkGatedDeltaRuleFwdH,
        k, w, u, g_,
        gk_, initial_state_, output_final_state_, chunk_size_, save_new_value_,
        cu_seqlens, chunk_indices, use_exp2_, transpose_state_layout_,
        h_out, v_new_out, final_state_out
    );
    if (output_final_state_) {
        return std::make_tuple(h_out, v_new_out, final_state_out);
    } else {
        return std::make_tuple(h_out, v_new_out, at::Tensor());
    }
}

::std::tuple<at::Tensor, at::Tensor> npu_recompute_w_u_fwd(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    int64_t chunk_size,
    const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &gK,
    c10::OptionalIntArrayRef cu_seqlens,
    c10::OptionalIntArrayRef chunk_indices
)
{
    at::Tensor w = at::empty_like(k);
    at::Tensor u = at::empty_like(v);
    const at::Tensor &gK_ = c10::value_or_else(gK, [] { return at::Tensor(); });
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });

    EXEC_NPU_CMD_EXT(aclnnRecomputeWUFwd,
        k, v, beta, A, g_, gK_,cu_seqlens, chunk_indices, chunk_size, w, u);
    return std::tie(w,u);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
             at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_chunk_kda_fwd(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &gk,
    const at::Tensor &beta,
    double scale,
    int64_t chunk_size,
    const c10::optional<at::Tensor> &initial_state,
    c10::optional<bool> output_final_state,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    c10::optional<bool> return_intermediate,
    c10::optional<bool> safe_gate,
    c10::optional<bool> transpose_state_layout)
{
    TORCH_CHECK(!safe_gate.value_or(false), "npu_chunk_kda_fwd: safe_gate=True is not supported.");
    TORCH_CHECK(!transpose_state_layout.value_or(false),
                "npu_chunk_kda_fwd: transpose_state_layout=True is not supported.");
    TORCH_CHECK(chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
                "npu_chunk_kda_fwd: chunk_size must be 32, 64 or 128.");
    bool is_tnd = q.dim() == 3;
    TORCH_CHECK((!is_tnd && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && gk.dim() == 4 && beta.dim() == 3) ||
                    (is_tnd && k.dim() == 3 && v.dim() == 3 && gk.dim() == 3 && beta.dim() == 2),
                "npu_chunk_kda_fwd: expected q/k/v/gk in BSND rank4 with beta rank3, or TND rank3 with beta rank2.");
    TORCH_CHECK(q.sizes() == k.sizes(), "npu_chunk_kda_fwd: q and k must have identical shape.");
    auto is_gate_dtype = [](at::ScalarType dtype) {
        return dtype == at::kFloat || dtype == at::kBFloat16;
    };
    TORCH_CHECK(is_gate_dtype(gk.scalar_type()) && is_gate_dtype(beta.scalar_type()),
                "npu_chunk_kda_fwd: gk and beta must be float32 or bfloat16.");

    auto q_sizes = q.sizes();
    auto v_sizes = v.sizes();
    int64_t B = is_tnd ? 1 : q_sizes[0];
    int64_t T = is_tnd ? q_sizes[0] : q_sizes[1];
    int64_t H = is_tnd ? q_sizes[1] : q_sizes[2];
    int64_t K = is_tnd ? q_sizes[2] : q_sizes[3];
    int64_t HV = is_tnd ? v_sizes[1] : v_sizes[2];
    int64_t V = is_tnd ? v_sizes[2] : v_sizes[3];
    TORCH_CHECK((is_tnd && v_sizes[0] == T) || (!is_tnd && v_sizes[0] == B && v_sizes[1] == T),
                "npu_chunk_kda_fwd: v shape prefix must match q.");
    TORCH_CHECK((is_tnd && gk.sizes()[0] == T && gk.sizes()[1] == HV && gk.sizes()[2] == K) ||
                    (!is_tnd && gk.sizes()[0] == B && gk.sizes()[1] == T && gk.sizes()[2] == HV &&
                        gk.sizes()[3] == K),
                "npu_chunk_kda_fwd: gk shape mismatch.");
    TORCH_CHECK((is_tnd && beta.sizes()[0] == T && beta.sizes()[1] == HV) ||
                    (!is_tnd && beta.sizes()[0] == B && beta.sizes()[1] == T && beta.sizes()[2] == HV),
                "npu_chunk_kda_fwd: beta shape mismatch.");
    TORCH_CHECK(HV % H == 0, "npu_chunk_kda_fwd: HV must be divisible by H.");

    int64_t seq_num = GetKdaSeqNum(B, cu_seqlens);
    int64_t total_chunks = GetKdaTotalChunks(B, T, chunk_size, cu_seqlens, chunk_indices);
    bool output_final_state_ = output_final_state.value_or(false);
    bool return_intermediate_ = return_intermediate.value_or(false);
    double scale_ = scale;
    int64_t chunk_size_ = chunk_size;
    bool recompute_output_final_state = true;
    int64_t total_chunks_ = total_chunks;

    at::Tensor o = at::empty_like(v);
    at::Tensor final_state_work = at::empty({seq_num, HV, K, V},
        (initial_state.has_value() && initial_state->defined()) ? initial_state->options() : q.options());
    at::Tensor aqk = is_tnd ? at::empty({T, HV, chunk_size}, q.options()) :
        at::empty({B, T, HV, chunk_size}, q.options());
    at::Tensor akk = is_tnd ? at::empty({T, HV, chunk_size}, q.options()) :
        at::empty({B, T, HV, chunk_size}, q.options());
    at::Tensor w = is_tnd ? at::empty({T, HV, K}, q.options()) : at::empty({B, T, HV, K}, q.options());
    at::Tensor u = at::empty_like(v);
    at::Tensor qg = is_tnd ? at::empty({T, HV, K}, q.options()) : at::empty({B, T, HV, K}, q.options());
    at::Tensor kg = is_tnd ? at::empty({T, HV, K}, q.options()) : at::empty({B, T, HV, K}, q.options());
    at::Tensor v_new = at::empty_like(v);
    at::Tensor h = is_tnd ? at::empty({total_chunks, HV, K, V}, q.options()) :
        at::empty({B, total_chunks, HV, K, V}, q.options());
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });

    EXEC_NPU_CMD_EXT(
        aclnnChunkKdaFwd,
        q, k, v, gk, beta, initial_state_,
        cu_seqlens, chunk_indices,
        scale_, chunk_size_, recompute_output_final_state, total_chunks_,
        o, final_state_work, aqk, akk, w, u, qg, kg, v_new, h
    );

    at::Tensor final_state = output_final_state_ ? final_state_work : at::empty({0}, q.options());
    if (return_intermediate_) {
        return std::make_tuple(o, final_state, aqk, akk, w, u, qg, kg, v_new, h);
    }
    at::Tensor empty = at::empty({0}, q.options());
    return std::make_tuple(o, final_state, aqk, akk, empty, empty, empty, empty, empty, h);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_chunk_kda_bwd(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &gk,
    const at::Tensor &beta,
    const at::Tensor &aqk,
    const at::Tensor &akk,
    const at::Tensor &d_o,
    double scale,
    int64_t chunk_size,
    const c10::optional<at::Tensor> &initial_state,
    const c10::optional<at::Tensor> &dht,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    c10::optional<bool> safe_gate,
    c10::optional<bool> transpose_state_layout)
{
    TORCH_CHECK(!safe_gate.value_or(false), "npu_chunk_kda_bwd: safe_gate=True is not supported.");
    TORCH_CHECK(!transpose_state_layout.value_or(false),
                "npu_chunk_kda_bwd: transpose_state_layout=True is not supported.");
    TORCH_CHECK(chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
                "npu_chunk_kda_bwd: chunk_size must be 32, 64 or 128.");
    auto is_gate_dtype = [](at::ScalarType dtype) {
        return dtype == at::kFloat || dtype == at::kBFloat16;
    };
    TORCH_CHECK(is_gate_dtype(gk.scalar_type()) && is_gate_dtype(beta.scalar_type()),
                "npu_chunk_kda_bwd: gk and beta must be float32 or bfloat16.");
    bool is_tnd = q.dim() == 3;
    TORCH_CHECK((!is_tnd && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && gk.dim() == 4 && beta.dim() == 3 &&
                    d_o.dim() == 4 && aqk.dim() == 4 && akk.dim() == 4) ||
                    (is_tnd && k.dim() == 3 && v.dim() == 3 && gk.dim() == 3 && beta.dim() == 2 &&
                        d_o.dim() == 3 && aqk.dim() == 3 && akk.dim() == 3),
                "npu_chunk_kda_bwd: expected tensors in BSND rank4 layout or TND rank3 layout.");
    TORCH_CHECK(q.sizes() == k.sizes(), "npu_chunk_kda_bwd: q and k must have identical shape.");

    auto q_sizes = q.sizes();
    auto v_sizes = v.sizes();
    int64_t B = is_tnd ? 1 : q_sizes[0];
    int64_t T = is_tnd ? q_sizes[0] : q_sizes[1];
    int64_t H = is_tnd ? q_sizes[1] : q_sizes[2];
    int64_t K = is_tnd ? q_sizes[2] : q_sizes[3];
    int64_t HV = is_tnd ? v_sizes[1] : v_sizes[2];
    int64_t V = is_tnd ? v_sizes[2] : v_sizes[3];
    TORCH_CHECK((is_tnd && v_sizes[0] == T) || (!is_tnd && v_sizes[0] == B && v_sizes[1] == T),
                "npu_chunk_kda_bwd: v shape prefix must match q.");
    TORCH_CHECK(d_o.sizes() == v.sizes(), "npu_chunk_kda_bwd: d_o shape must match v.");
    TORCH_CHECK((is_tnd && gk.sizes()[0] == T && gk.sizes()[1] == HV && gk.sizes()[2] == K) ||
                    (!is_tnd && gk.sizes()[0] == B && gk.sizes()[1] == T && gk.sizes()[2] == HV &&
                        gk.sizes()[3] == K),
                "npu_chunk_kda_bwd: gk shape mismatch.");
    TORCH_CHECK((is_tnd && beta.sizes()[0] == T && beta.sizes()[1] == HV) ||
                    (!is_tnd && beta.sizes()[0] == B && beta.sizes()[1] == T && beta.sizes()[2] == HV),
                "npu_chunk_kda_bwd: beta shape mismatch.");
    TORCH_CHECK(((is_tnd && aqk.sizes()[0] == T && aqk.sizes()[1] == HV && aqk.sizes()[2] == chunk_size) ||
                    (!is_tnd && aqk.sizes()[0] == B && aqk.sizes()[1] == T && aqk.sizes()[2] == HV &&
                        aqk.sizes()[3] == chunk_size)) && akk.sizes() == aqk.sizes(),
                "npu_chunk_kda_bwd: Aqk/Akk shape mismatch.");
    TORCH_CHECK(HV % H == 0, "npu_chunk_kda_bwd: HV must be divisible by H.");
    int64_t seq_num = GetKdaSeqNum(B, cu_seqlens);
    int64_t total_chunks = GetKdaTotalChunks(B, T, chunk_size, cu_seqlens, chunk_indices);
    double scale_ = scale;
    int64_t chunk_size_ = chunk_size;
    int64_t seq_num_ = seq_num;
    int64_t total_chunks_ = total_chunks;

    at::Tensor dq = at::empty_like(q);
    at::Tensor dk = at::empty_like(k);
    at::Tensor dv = at::empty_like(v);
    at::Tensor dbeta = at::empty(beta.sizes(), beta.options().dtype(at::kFloat));
    at::Tensor dgk = at::empty(gk.sizes(), gk.options().dtype(at::kFloat));
    at::Tensor dh0_work = at::empty({seq_num, HV, K, V},
        (initial_state.has_value() && initial_state->defined()) ? initial_state->options() :
        ((dht.has_value() && dht->defined()) ? dht->options() : q.options()));
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    const at::Tensor &dht_ = c10::value_or_else(dht, [] { return at::Tensor(); });

    EXEC_NPU_CMD_EXT(
        aclnnChunkKdaBwd,
        q, k, v, gk, beta, aqk, akk, d_o,
        initial_state_, dht_,
        cu_seqlens, chunk_indices,
        scale_, chunk_size_, seq_num_, total_chunks_,
        dq, dk, dv, dbeta, dgk, dh0_work
    );

    at::Tensor dh0 = (initial_state.has_value() && initial_state->defined()) ? dh0_work : at::empty({0}, q.options());
    return std::make_tuple(dq, dk, dv, dbeta, dgk, dh0);
}

at::Tensor npu_kda_gate_cumsum(
    const at::Tensor &g,
    int64_t chunk_size,
    const c10::optional<at::Tensor> &A_log,
    const c10::optional<at::Tensor> &dt_bias,
    at::OptionalIntArrayRef cu_seqlens,
    c10::optional<bool> use_gate_in_kernel,
    c10::optional<bool> safe_gate,
    c10::optional<double> lower_bound)
{
    TORCH_CHECK(g.dim() == 3 || g.dim() == 4, "npu_kda_gate_cumsum: g must be BSND rank4 or TND rank3.");
    TORCH_CHECK(chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
                "npu_kda_gate_cumsum: chunk_size must be 32, 64 or 128.");
    auto gate_dtype = g.scalar_type();
    TORCH_CHECK(gate_dtype == at::kFloat || gate_dtype == at::kBFloat16 || gate_dtype == at::kHalf,
                "npu_kda_gate_cumsum: g must be float32, bfloat16 or float16.");
    int64_t K = g.dim() == 4 ? g.sizes()[3] : g.sizes()[2];
    int64_t HV = g.dim() == 4 ? g.sizes()[2] : g.sizes()[1];
    TORCH_CHECK(K <= 256, "npu_kda_gate_cumsum: K must be <= 256.");

    bool use_gate = use_gate_in_kernel.value_or(false);
    bool safe = safe_gate.value_or(false);
    double lower = lower_bound.value_or(-5.0);
    if (use_gate) {
        TORCH_CHECK(A_log.has_value() && A_log->defined(),
                    "npu_kda_gate_cumsum: A_log is required when use_gate_in_kernel=True.");
        TORCH_CHECK(A_log->scalar_type() == at::kFloat && A_log->dim() == 1 && A_log->sizes()[0] == HV,
                    "npu_kda_gate_cumsum: A_log must be float32 with shape [HV].");
        TORCH_CHECK(safe, "npu_kda_gate_cumsum: raw gate path currently requires safe_gate=True.");
        TORCH_CHECK(lower >= -5.0 && lower < 0.0, "npu_kda_gate_cumsum: lower_bound must be in [-5, 0).");
        if (dt_bias.has_value() && dt_bias->defined()) {
            bool valid_bias = dt_bias->scalar_type() == at::kFloat &&
                ((dt_bias->dim() == 1 && dt_bias->sizes()[0] == HV * K) ||
                 (dt_bias->dim() == 2 && dt_bias->sizes()[0] == HV && dt_bias->sizes()[1] == K));
            TORCH_CHECK(valid_bias, "npu_kda_gate_cumsum: dt_bias must be float32 with shape [HV*K] or [HV, K].");
        }
    } else {
        TORCH_CHECK(!safe, "npu_kda_gate_cumsum: safe_gate only applies when use_gate_in_kernel=True.");
    }

    at::Tensor gk = at::empty(g.sizes(), g.options().dtype(at::kFloat));
    const at::Tensor &A_log_ = c10::value_or_else(A_log, [] { return at::Tensor(); });
    const at::Tensor &dt_bias_ = c10::value_or_else(dt_bias, [] { return at::Tensor(); });
    EXEC_NPU_CMD_EXT(
        aclnnKdaGateCumsum,
        g, A_log_, dt_bias_, cu_seqlens,
        chunk_size, use_gate, safe, lower, gk
    );
    return gk;
}

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
    int64_t run_mode)
{
    at::Tensor y = at::empty_like(x);

    const at::Tensor &bias_ = c10::value_or_else(bias, [] { return at::Tensor(); });

    c10::optional<at::IntArrayRef> query_start_loc_ = query_start_loc.has_value()
        ? c10::optional<at::IntArrayRef>(query_start_loc.value()) : c10::nullopt;
    c10::optional<at::IntArrayRef> cache_indices_ = cache_indices.has_value()
        ? c10::optional<at::IntArrayRef>(cache_indices.value()) : c10::nullopt;
    c10::optional<at::IntArrayRef> initial_state_mode_ = initial_state_mode.has_value()
        ? c10::optional<at::IntArrayRef>(initial_state_mode.value()) : c10::nullopt;
    c10::optional<at::IntArrayRef> num_accepted_tokens_ = num_accepted_tokens.has_value()
        ? c10::optional<at::IntArrayRef>(num_accepted_tokens.value()) : c10::nullopt;

    EXEC_NPU_CMD_EXT(
        aclnnCausalConv1d,
        x, weight, bias_, conv_states,
        query_start_loc_, cache_indices_, initial_state_mode_, num_accepted_tokens_,
        activation_mode, pad_slot_id, run_mode,
        y
    );
    return y;
}

}  // namespace op_api
