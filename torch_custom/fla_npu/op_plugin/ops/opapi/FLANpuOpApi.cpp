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
#include <vector>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include "op_plugin/include/npu_cpp_extension.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using namespace op_plugin::utils;
using namespace op_infer;


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
    // GVA：q/k 为 [B,Hk,T,K]；w、d_o、dv、g 等为 [B,Hv,T,·]，且 Hv % Hk == 0（与 device tiling 一致）
    auto q_size = q.sizes();
    auto k_size = k.sizes();
    auto w_size = w.sizes();
    auto do_size = d_o.sizes();
    auto dv_size = dv.sizes();
    int64_t B = q_size[0];
    int64_t Hk = q_size[1];
    int64_t T = q_size[2];
    int64_t K = q_size[3];
    int64_t Hv = dv_size[1];
    int64_t V = dv_size[3];
    int64_t chunk_num = (T + chunk_size -1) / chunk_size;

    TORCH_CHECK(
        k_size[0] == B && k_size[1] == Hk && k_size[2] == T && k_size[3] == K,
        "npu_chunk_gated_delta_rule_bwd_dhu: k must match q as [B,Hk,T,K]; q=", q_size, " k=", k_size);
    TORCH_CHECK(
        w_size[0] == B && w_size[1] == Hv && w_size[2] == T && w_size[3] == K,
        "npu_chunk_gated_delta_rule_bwd_dhu: w must be [B,Hv,T,K] with Hv=dv.dim1; w=", w_size, " dv=", dv_size);
    TORCH_CHECK(
        do_size[0] == B && do_size[1] == Hv && do_size[2] == T && do_size[3] == V,
        "npu_chunk_gated_delta_rule_bwd_dhu: d_o must be [B,Hv,T,V]; d_o=", do_size, " dv=", dv_size);
    TORCH_CHECK(
        dv_size[0] == B && dv_size[1] == Hv && dv_size[2] == T && dv_size[3] == V,
        "npu_chunk_gated_delta_rule_bwd_dhu: dv must be [B,Hv,T,V]; dv=", dv_size);
    TORCH_CHECK(
        Hk > 0 && Hv > 0 && Hv % Hk == 0,
        "npu_chunk_gated_delta_rule_bwd_dhu: GVA requires Hv divisible by Hk; Hk=",
        Hk,
        " Hv=",
        Hv);

    if (chunk_indices.has_value()) {
        auto chunk_indices_ref = chunk_indices.value();
        chunk_num = chunk_indices_ref.size() / 2;
    }

    // 创建输出 tensor：dh/dh0 与 value 头维 Hv 对齐（device 输出 [B,Hv,chunk_num,K,V]）
    at::Tensor dv2 = at::empty_like(dv);
    at::Tensor dh = at::empty({B, Hv, chunk_num, K, V}, q.options());
    at::Tensor dh0;
    if (h0.has_value()) {
        dh0 = at::empty({B, Hv, chunk_num, K, V}, q.options());
    } else {
        dh0 = at::Tensor();
    }

    // optional tensor处理
    const at::Tensor &g_  = c10::value_or_else(g,  [] { return at::Tensor(); });
    const at::Tensor &gK_ = c10::value_or_else(gK, [] { return at::Tensor(); });
    const at::Tensor &h0_ = c10::value_or_else(h0, [] { return at::Tensor(); });
    const at::Tensor &dht_ = c10::value_or_else(dht, [] { return at::Tensor(); });

    if (g_.defined()) {
        TORCH_CHECK(
            g_.dim() == 3 && g_.size(0) == B && g_.size(1) == Hv && g_.size(2) == T,
            "npu_chunk_gated_delta_rule_bwd_dhu: g must be [B,Hv,T]; g=", g_.sizes());
    }
    if (h0_.defined()) {
        TORCH_CHECK(
            h0_.dim() == 5 && h0_.size(0) == B && h0_.size(1) == Hv && h0_.size(2) == chunk_num,
            "npu_chunk_gated_delta_rule_bwd_dhu: h0 must be [B,Hv,chunk_num,K,V]; h0=", h0_.sizes(),
            " chunk_num=", chunk_num);
    }
    if (dht_.defined()) {
        TORCH_CHECK(
            dht_.dim() == 5 && dht_.size(0) == B && dht_.size(1) == Hv && dht_.size(2) == chunk_num,
            "npu_chunk_gated_delta_rule_bwd_dhu: dht must be [B,Hv,chunk_num,K,V]; dht=", dht_.sizes(),
            " chunk_num=", chunk_num);
    }

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
    // 获取输入 q 的维度信息
    auto q_sizes = q.sizes();
    auto v_sizes = v.sizes();
    int64_t batch_size = q_sizes[0];
    int64_t num_heads = v_sizes[1];
    int64_t seq_len = q_sizes[2];
    int64_t head_dim = q_sizes[3];
    
    // 使用具体 shape 创建，但保持相同的 dtype/device
    at::Tensor dq = at::empty({batch_size, num_heads, seq_len, head_dim}, q.options());
    at::Tensor dk = at::empty({batch_size, num_heads, seq_len, head_dim}, q.options());
    at::Tensor dw = at::empty({batch_size, num_heads, seq_len, head_dim}, q.options());
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
        g.has_value() && g->defined(),
        "npu_chunk_gated_delta_rule_fwd_h: g cannot be None or undefined; pass a real gate tensor until NPU supports g=None.");
    TORCH_CHECK(
        !gk.has_value() || !gk->defined(),
        "npu_chunk_gated_delta_rule_fwd_h: gk is reserved and only None is supported.");
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
    auto w_shape = v.sizes().vec();
    w_shape[3] = k.size(3);
    at::Tensor w = at::empty(w_shape, v.options().dtype(k.scalar_type()));
    at::Tensor u = at::empty_like(v);
    const at::Tensor &gK_ = c10::value_or_else(gK, [] { return at::Tensor(); });
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });

    EXEC_NPU_CMD_EXT(aclnnRecomputeWUFwd,
        k, v, beta, A, g_, gK_,cu_seqlens, chunk_indices, chunk_size, w, u);
    return std::tie(w,u);
}

at::Tensor infer_y_tensor(
    const at::Tensor& x,
    int64_t head_num,
    int64_t run_mode)
{
    at::Tensor y;
    int64_t x_dim = x.dim();

    if (run_mode == 0 && head_num > 0) {
        if (x_dim == 3) {
            auto sizes = x.sizes();
            int64_t b = sizes[0];
            int64_t s = sizes[1];
            int64_t d = sizes[2] / head_num;
            y = at::empty({b, head_num, s, d}, x.options());
        } else if (x_dim == 2) {
            auto sizes = x.sizes();
            int64_t s = sizes[0];
            int64_t d = sizes[1] / head_num;
            y = at::empty({head_num, s, d}, x.options());
        } else {
            y = at::empty_like(x);
        }
    } else {
        y = at::empty_like(x);
    }

    return y;
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
    int64_t run_mode,
    int64_t head_num = 0)
{
    at::Tensor y = infer_y_tensor(x, head_num, run_mode);

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

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_causal_conv1d_bwd(
    const at::Tensor &x,
    const c10::optional<at::Tensor> &y,
    const at::Tensor &weight,
    const at::Tensor &dy,
    const c10::optional<at::Tensor> &initial_state,
    const c10::optional<at::Tensor> &dht,
    at::OptionalIntArrayRef query_start_loc,
    int64_t activation,
    c10::string_view input_layout)
{
    const std::string input_layout_str(input_layout);
    const int64_t width = weight.size(0);
    const int64_t dim = weight.size(1);

    std::vector<int64_t> dx_shape;
    int64_t batch = 0;
    if (input_layout_str == "BNSD") {
        TORCH_CHECK(x.dim() == 3, "BNSD x must be logical [B, T, D]");
        TORCH_CHECK(dy.dim() == 4, "BNSD dy must be [B, N, T, Dh]");
        batch = x.size(0);
        TORCH_CHECK(
            dy.size(0) == x.size(0) &&
                dy.size(2) == x.size(1) &&
                dy.size(1) * dy.size(3) == x.size(2),
            "BNSD dy must be [B, N, T, Dh] with D=N*Dh for x [B, T, D]");
        dx_shape = x.sizes().vec();
    } else if (input_layout_str == "NTD") {
        TORCH_CHECK(x.dim() == 2, "NTD x must be logical [total_tokens, D]");
        TORCH_CHECK(dy.dim() == 3, "NTD dy must be [N, total_tokens, Dh]");
        TORCH_CHECK(query_start_loc.has_value(), "query_start_loc is required for NTD input");
        TORCH_CHECK(
            dy.size(1) == x.size(0) &&
                dy.size(0) * dy.size(2) == x.size(1),
            "NTD dy must be [N, total_tokens, Dh] with D=N*Dh for x [total_tokens, D]");
        batch = static_cast<int64_t>(query_start_loc.value().size()) - 1;
        dx_shape = x.sizes().vec();
    } else if (input_layout_str == "TND") {
        TORCH_CHECK(x.dim() == 2, "TND input must be [total_tokens, D]");
        TORCH_CHECK(dy.sizes() == x.sizes(), "TND dy shape must match x");
        TORCH_CHECK(query_start_loc.has_value(), "query_start_loc is required for TND input");
        batch = static_cast<int64_t>(query_start_loc.value().size()) - 1;
        dx_shape = x.sizes().vec();
    } else {
        TORCH_CHECK(
            input_layout_str == "BSND" || input_layout_str == "BSH",
            "input_layout must be one of BSND, BSH, TND, BNSD, or NTD");
        TORCH_CHECK(x.dim() == 3, "BSND/BSH input must be [B, T, D]");
        TORCH_CHECK(dy.sizes() == x.sizes(), "BSND/BSH dy shape must match x");
        batch = x.size(0);
        dx_shape = x.sizes().vec();
    }
    if (y.has_value()) {
        TORCH_CHECK(y->sizes() == dy.sizes(), "y shape must match dy");
    } else {
        TORCH_CHECK(activation == 0, "y is required when activation is enabled");
    }

    at::Tensor dx = at::empty(dx_shape, x.options());
    at::Tensor dw = at::empty({width, dim}, weight.options());
    at::Tensor db = at::empty({dim}, weight.options());
    const auto check_state_shape = [batch, width, dim](
                                       const c10::optional<at::Tensor> &state,
                                       const char *name) {
        if (!state.has_value()) {
            return;
        }
        TORCH_CHECK(
            state->dim() == 3 &&
                state->size(0) == batch &&
                state->size(1) == width &&
                state->size(2) == dim,
            name, " must be [B, W, D]=[", batch, ", ", width, ", ", dim,
            "], got ", state->sizes());
    };
    check_state_shape(initial_state, "initial_state");
    check_state_shape(dht, "dht");

    at::Tensor dh0 = at::empty({batch, width, dim}, x.options());

    const at::Tensor &y_ = c10::value_or_else(y, [] { return at::Tensor(); });
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    const at::Tensor &dht_ = c10::value_or_else(dht, [] { return at::Tensor(); });
    c10::optional<at::IntArrayRef> query_start_loc_ = query_start_loc.has_value()
        ? c10::optional<at::IntArrayRef>(query_start_loc.value()) : c10::nullopt;

    EXEC_NPU_CMD_EXT(
        aclnnCausalConv1dBwd,
        x, y_, weight, dy, initial_state_, dht_, query_start_loc_,
        activation, input_layout_str,
        dx, dw, db, dh0
    );
    return std::make_tuple(std::move(dx), std::move(dw), std::move(db), std::move(dh0));
}

at::Tensor npu_solve_tri(
    const at::Tensor &x,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    c10::string_view layout)
{
    at::Tensor x_contig = x.contiguous();
    at::Tensor x_out = at::empty_like(x_contig);

    std::string layout_str(layout.data(), layout.size());
    const char *layout_cstr = layout_str.c_str();

    EXEC_NPU_CMD_EXT(
        aclnnSolveTri,
        x_contig, cu_seqlens, chunk_indices, layout_cstr,
        x_out
    );
    return x_out;
}

}  // namespace op_api
