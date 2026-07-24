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

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
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

void CheckKdaCuSeqlens(at::OptionalIntArrayRef cu_seqlens, int64_t total_tokens, const char *op_name)
{
    if (!cu_seqlens.has_value()) {
        return;
    }
    auto cu = cu_seqlens.value();
    TORCH_CHECK(cu.size() >= 2, op_name, ": cu_seqlens must contain at least [0, total_tokens].");
    TORCH_CHECK(cu[0] == 0, op_name, ": cu_seqlens[0] must be 0, but got ", cu[0], ".");
    TORCH_CHECK(cu[cu.size() - 1] == total_tokens,
                op_name,
                ": cu_seqlens[-1] must equal sequence length ",
                total_tokens,
                ", but got ",
                cu[cu.size() - 1],
                ".");
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
        TORCH_CHECK(cu[i] <= cu[i + 1],
                    op_name,
                    ": cu_seqlens must be nondecreasing, but cu_seqlens[",
                    i,
                    "]=",
                    cu[i],
                    " > cu_seqlens[",
                    i + 1,
                    "]=",
                    cu[i + 1],
                    ".");
    }
}

void CheckKdaRecurrentCuSeqlens(at::OptionalIntArrayRef cu_seqlens,
                                int64_t total_tokens,
                                int64_t max_seq_len,
                                const char *op_name)
{
    CheckKdaCuSeqlens(cu_seqlens, total_tokens, op_name);
    if (!cu_seqlens.has_value()) {
        return;
    }
    auto cu = cu_seqlens.value();
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
        TORCH_CHECK(cu[i + 1] - cu[i] <= max_seq_len,
                    op_name,
                    ": each recurrent sequence length must be <= ",
                    max_seq_len,
                    ", but got ",
                    cu[i + 1] - cu[i],
                    ".");
    }
}

void CheckKdaChunkIndices(at::OptionalIntArrayRef chunk_indices,
                          at::OptionalIntArrayRef cu_seqlens,
                          int64_t chunk_size,
                          const char *op_name)
{
    if (!chunk_indices.has_value()) {
        return;
    }
    auto indices = chunk_indices.value();
    TORCH_CHECK(indices.size() % 2 == 0,
                op_name,
                ": chunk_indices must contain (seq_id, chunk_id) pairs, but got ",
                indices.size(),
                " elements.");
    TORCH_CHECK(cu_seqlens.has_value(), op_name, ": chunk_indices requires cu_seqlens.");
    auto cu = cu_seqlens.value();
    int64_t expected_chunks = 0;
    for (size_t seq = 0; seq + 1 < cu.size(); ++seq) {
        expected_chunks += CeilDiv(cu[seq + 1] - cu[seq], chunk_size);
    }
    TORCH_CHECK(static_cast<int64_t>(indices.size() / 2) == expected_chunks,
                op_name, ": chunk_indices must contain exactly one pair per chunk.");
    for (size_t idx = 0; idx < indices.size(); idx += 2) {
        int64_t seq = indices[idx];
        int64_t chunk = indices[idx + 1];
        TORCH_CHECK(seq >= 0 && seq + 1 < static_cast<int64_t>(cu.size()),
                    op_name, ": chunk_indices seq_id is out of range.");
        int64_t chunks = CeilDiv(cu[seq + 1] - cu[seq], chunk_size);
        TORCH_CHECK(chunk >= 0 && chunk < chunks,
                    op_name, ": chunk_indices chunk_id is out of range.");
    }
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

std::vector<int64_t> BuildKdaChunkIndices(at::IntArrayRef cu_seqlens, int64_t chunk_size)
{
    std::vector<int64_t> indices;
    int64_t total_chunks = 0;
    for (size_t i = 0; i + 1 < cu_seqlens.size(); ++i) {
        total_chunks += CeilDiv(cu_seqlens[i + 1] - cu_seqlens[i], chunk_size);
    }
    indices.reserve(static_cast<size_t>(total_chunks * 2));
    for (size_t seq = 0; seq + 1 < cu_seqlens.size(); ++seq) {
        int64_t seq_len = cu_seqlens[seq + 1] - cu_seqlens[seq];
        int64_t chunks = CeilDiv(seq_len, chunk_size);
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            indices.push_back(static_cast<int64_t>(seq));
            indices.push_back(chunk);
        }
    }
    return indices;
}

bool ResolveChunkLocalCumsumOutputDtype(
    const std::string &output_dtype_str,
    c10::ScalarType input_dtype,
    c10::ScalarType &resolved_dtype)
{
    if (output_dtype_str.empty() || output_dtype_str == "float" || output_dtype_str == "float32" ||
        output_dtype_str == "fp32" || output_dtype_str == "torch.float" ||
        output_dtype_str == "torch.float32") {
        resolved_dtype = c10::ScalarType::Float;
        return true;
    }
    if (output_dtype_str == "same" || output_dtype_str == "same_as_input" || output_dtype_str == "input" ||
        output_dtype_str == "none" || output_dtype_str == "None" || output_dtype_str == "null") {
        resolved_dtype = input_dtype;
        return true;
    }
    if (output_dtype_str == "float16" || output_dtype_str == "fp16" || output_dtype_str == "half" ||
        output_dtype_str == "torch.float16" || output_dtype_str == "torch.half") {
        resolved_dtype = c10::ScalarType::Half;
        return true;
    }
    if (output_dtype_str == "bfloat16" || output_dtype_str == "bf16" ||
        output_dtype_str == "torch.bfloat16") {
        resolved_dtype = c10::ScalarType::BFloat16;
        return true;
    }
    return false;
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
    const int64_t key_num_heads = q.size(1);
    const int64_t value_num_heads = v.size(1);
    TORCH_CHECK(
        key_num_heads > 0 && value_num_heads > 0 && value_num_heads % key_num_heads == 0,
        "npu_chunk_bwd_dqkwg: GVA requires value heads divisible by key heads; Hk=",
        key_num_heads,
        " Hv=",
        value_num_heads);

    at::Tensor dq = at::empty_like(q);
    at::Tensor dk = at::empty_like(k);
    at::Tensor dw = at::empty({q.size(0), value_num_heads, q.size(2), q.size(3)}, q.options());
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
        auto state_options = initial_state.has_value() ? initial_state->options() : k.options().dtype(at::kFloat);
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

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
             at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_chunk_kda_fwd(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &gk,
    const at::Tensor &beta,
    double scale,
    int64_t chunk_size,
    c10::string_view layout,
    const c10::optional<at::Tensor> &initial_state,
    c10::optional<bool> output_final_state,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    c10::optional<bool> return_intermediate,
    c10::optional<bool> safe_gate,
    c10::optional<bool> transpose_state_layout)
{
    std::string layout_str(layout.data(), layout.size());
    TORCH_CHECK(layout_str == "BSND" || layout_str == "BNSD" || layout_str == "TND" || layout_str == "NTD",
                "npu_chunk_kda_fwd: layout must be one of BSND, BNSD, TND, NTD and must be uppercase.");
    TORCH_CHECK(!safe_gate.value_or(false), "npu_chunk_kda_fwd: safe_gate=True is not supported.");
    TORCH_CHECK(!transpose_state_layout.value_or(false),
                "npu_chunk_kda_fwd: transpose_state_layout=True is not supported.");
    TORCH_CHECK(chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
                "npu_chunk_kda_fwd: chunk_size must be 32, 64 or 128.");
    bool is_tnd = layout_str == "TND";
    bool is_ntd = layout_str == "NTD";
    bool is_bsnd = layout_str == "BSND";
    bool is_bnsd = layout_str == "BNSD";
    bool is_rank3 = is_tnd || is_ntd;
    TORCH_CHECK((is_rank3 && q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && gk.dim() == 3 && beta.dim() == 2) ||
                    (!is_rank3 && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && gk.dim() == 4 && beta.dim() == 3),
                "npu_chunk_kda_fwd: layout/rank mismatch. TND/NTD require q/k/v/gk rank3 with beta rank2; "
                "BSND/BNSD require q/k/v/gk rank4 with beta rank3.");
    TORCH_CHECK(q.sizes() == k.sizes(), "npu_chunk_kda_fwd: q and k must have identical shape.");
    auto is_gate_dtype = [](at::ScalarType dtype) {
        return dtype == at::kFloat || dtype == at::kBFloat16;
    };
    TORCH_CHECK(is_gate_dtype(gk.scalar_type()) && is_gate_dtype(beta.scalar_type()),
                "npu_chunk_kda_fwd: gk and beta must be float32 or bfloat16.");

    auto q_sizes = q.sizes();
    auto v_sizes = v.sizes();

    bool is_internal_layout = is_bnsd || is_ntd;
    int64_t B = is_rank3 ? 1 : q_sizes[0];
    int64_t T = is_tnd ? q_sizes[0] : (is_ntd ? q_sizes[1] : (is_bnsd ? q_sizes[2] : q_sizes[1]));
    int64_t H = is_tnd ? q_sizes[1] : (is_ntd ? q_sizes[0] : (is_bnsd ? q_sizes[1] : q_sizes[2]));
    int64_t K = is_rank3 ? q_sizes[2] : q_sizes[3];
    int64_t HV = is_tnd ? v_sizes[1] : (is_ntd ? v_sizes[0] : (is_bnsd ? v_sizes[1] : v_sizes[2]));
    int64_t V = is_rank3 ? v_sizes[2] : v_sizes[3];
    TORCH_CHECK(H > 0 && HV >= H,
                "npu_chunk_kda_fwd: H and HV must be positive and H must be <= HV.");
    TORCH_CHECK(H <= 128 && HV <= 128,
                "npu_chunk_kda_fwd: H and HV must be <= 128.");
    TORCH_CHECK(!is_tnd || H == 1,
                "npu_chunk_kda_fwd: TND layout with H > 1 is not supported; use NTD [H,T,D] layout "
                "for multi-head rank3 input.");
    CheckKdaCuSeqlens(cu_seqlens, T, "npu_chunk_kda_fwd");
    CheckKdaChunkIndices(chunk_indices, cu_seqlens, chunk_size, "npu_chunk_kda_fwd");
    TORCH_CHECK(!cu_seqlens.has_value() || is_rank3 || B == 1,
                "npu_chunk_kda_fwd: rank4 varlen input with cu_seqlens currently requires B=1.");
    TORCH_CHECK((is_tnd && v_sizes[0] == T) || (is_ntd && v_sizes[1] == T) ||
                    (is_bsnd && v_sizes[0] == B && v_sizes[1] == T) ||
                    (is_bnsd && v_sizes[0] == B && v_sizes[2] == T),
                "npu_chunk_kda_fwd: v shape prefix must match q layout.");
    TORCH_CHECK((is_tnd && gk.sizes()[0] == T && gk.sizes()[1] == HV && gk.sizes()[2] == K) ||
                    (is_ntd && gk.sizes()[0] == HV && gk.sizes()[1] == T && gk.sizes()[2] == K) ||
                    (is_bsnd && gk.sizes()[0] == B && gk.sizes()[1] == T && gk.sizes()[2] == HV &&
                        gk.sizes()[3] == K) ||
                    (is_bnsd && gk.sizes()[0] == B && gk.sizes()[1] == HV && gk.sizes()[2] == T &&
                        gk.sizes()[3] == K),
                "npu_chunk_kda_fwd: gk shape mismatch.");
    TORCH_CHECK((is_tnd && beta.sizes()[0] == T && beta.sizes()[1] == HV) ||
                    (is_ntd && beta.sizes()[0] == HV && beta.sizes()[1] == T) ||
                    (is_bsnd && beta.sizes()[0] == B && beta.sizes()[1] == T && beta.sizes()[2] == HV) ||
                    (is_bnsd && beta.sizes()[0] == B && beta.sizes()[1] == HV && beta.sizes()[2] == T),
                "npu_chunk_kda_fwd: beta shape mismatch.");
    TORCH_CHECK(HV % H == 0, "npu_chunk_kda_fwd: HV must be divisible by H.");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "npu_chunk_kda_fwd: q/k/v must use float16 or bfloat16.");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "npu_chunk_kda_fwd: q/k/v dtype must match.");
    TORCH_CHECK(chunk_size == 64 && K >= 16 && V >= 16 && K % 16 == 0 && V % 16 == 0 && V <= 128 &&
                    K * V >= 4 * 64 * 64 && K * V >= chunk_size * (K + V),
                "npu_chunk_kda_fwd: shape is outside the supported split cube/vector template.");
    int64_t seq_num = GetKdaSeqNum(B, cu_seqlens);
    if (initial_state.has_value() && initial_state->defined()) {
        TORCH_CHECK(initial_state->scalar_type() == at::kFloat,
                    "npu_chunk_kda_fwd: initial_state must be float32 when provided.");
        TORCH_CHECK(initial_state->dim() == 4 && initial_state->size(0) == seq_num &&
                        initial_state->size(1) == HV && initial_state->size(2) == K &&
                        initial_state->size(3) == V,
                    "npu_chunk_kda_fwd: initial_state must be [seq_num,Hv,K,V], where seq_num is batch "
                    "for dense input or len(cu_seqlens)-1 for varlen input; got initial_state=",
                    initial_state->sizes(),
                    ", expected [",
                    seq_num,
                    ",",
                    HV,
                    ",",
                    K,
                    ",",
                    V,
                    "].");
    }

    std::vector<int64_t> generated_chunk_indices;
    c10::optional<at::IntArrayRef> chunk_indices_for_call;
    if (chunk_indices.has_value()) {
        chunk_indices_for_call = chunk_indices.value();
    } else if (cu_seqlens.has_value()) {
        generated_chunk_indices = BuildKdaChunkIndices(cu_seqlens.value(), chunk_size);
        chunk_indices_for_call = at::IntArrayRef(generated_chunk_indices);
    } else {
        chunk_indices_for_call = c10::nullopt;
    }

    int64_t total_chunks = GetKdaTotalChunks(B, T, chunk_size, cu_seqlens, chunk_indices_for_call);
    bool output_final_state_ = output_final_state.value_or(false);
    bool return_intermediate_ = return_intermediate.value_or(false);
    double scale_ = scale;
    int64_t chunk_size_ = chunk_size;
    bool recompute_output_final_state = true;
    int64_t total_chunks_ = total_chunks;

    at::Tensor o = at::empty_like(v);
    at::Tensor final_state_work = at::empty({seq_num, HV, K, V}, q.options().dtype(at::kFloat));
    at::Tensor aqk = is_rank3 ? (is_internal_layout ? at::empty({HV, T, chunk_size}, q.options()) :
        at::empty({T, HV, chunk_size}, q.options())) : (is_internal_layout ?
        at::empty({B, HV, T, chunk_size}, q.options()) : at::empty({B, T, HV, chunk_size}, q.options()));
    at::Tensor akk = is_rank3 ? (is_internal_layout ? at::empty({HV, T, chunk_size}, q.options()) :
        at::empty({T, HV, chunk_size}, q.options())) : (is_internal_layout ?
        at::empty({B, HV, T, chunk_size}, q.options()) : at::empty({B, T, HV, chunk_size}, q.options()));
    at::Tensor w = is_rank3 ? (is_internal_layout ? at::empty({HV, T, K}, q.options()) :
        at::empty({T, HV, K}, q.options())) : (is_internal_layout ?
        at::empty({B, HV, T, K}, q.options()) : at::empty({B, T, HV, K}, q.options()));
    at::Tensor u = at::empty_like(v);
    at::Tensor qg = is_rank3 ? (is_internal_layout ? at::empty({HV, T, K}, q.options()) :
        at::empty({T, HV, K}, q.options())) : (is_internal_layout ?
        at::empty({B, HV, T, K}, q.options()) : at::empty({B, T, HV, K}, q.options()));
    at::Tensor kg = is_rank3 ? (is_internal_layout ? at::empty({HV, T, K}, q.options()) :
        at::empty({T, HV, K}, q.options())) : (is_internal_layout ?
        at::empty({B, HV, T, K}, q.options()) : at::empty({B, T, HV, K}, q.options()));
    at::Tensor v_new = at::empty_like(v);
    at::Tensor h = is_rank3 ? (is_internal_layout ? at::empty({HV, total_chunks, K, V}, q.options()) :
        at::empty({total_chunks, HV, K, V}, q.options())) : (is_internal_layout ?
        at::empty({B, HV, total_chunks, K, V}, q.options()) : at::empty({B, total_chunks, HV, K, V}, q.options()));
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    const char *layout_cstr = layout_str.c_str();

    EXEC_NPU_CMD_EXT(
        aclnnChunkKdaFwd,
        q, k, v, gk, beta, initial_state_,
        cu_seqlens, chunk_indices_for_call,
        layout_cstr, scale_, chunk_size_, recompute_output_final_state, total_chunks_,
        o, final_state_work, aqk, akk, w, u, qg, kg, v_new, h
    );

    at::Tensor final_state = output_final_state_ ? final_state_work : at::empty({0}, q.options().dtype(at::kFloat));
    at::Tensor empty = at::empty({0}, q.options());
    at::Tensor g = gk.scalar_type() == at::kFloat ? gk : gk.to(at::kFloat);
    at::Tensor initial_state_out =
        (initial_state.has_value() && initial_state->defined()) ? initial_state_ : empty;
    c10_npu::npuSynchronizeDevice();
    if (return_intermediate_) {
        return std::make_tuple(o, final_state, g, aqk, akk, w, u, qg, kg, v_new, h, initial_state_out);
    }
    return std::make_tuple(o, final_state, g, aqk, akk, w, u, qg, kg, v_new, h, initial_state_out);
}

at::Tensor npu_kda_gate_cumsum(
    const at::Tensor &g,
    int64_t chunk_size,
    const c10::optional<at::Tensor> &A_log,
    const c10::optional<at::Tensor> &dt_bias,
    at::OptionalIntArrayRef cu_seqlens,
    c10::optional<bool> use_gate_in_kernel,
    c10::optional<bool> safe_gate,
    c10::optional<double> lower_bound,
    c10::string_view layout)
{
    TORCH_CHECK(g.dim() == 3 || g.dim() == 4,
                "npu_kda_gate_cumsum: g must be BSND/BNSD rank4 or TND/NTD rank3.");
    TORCH_CHECK(chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
                "npu_kda_gate_cumsum: chunk_size must be 32, 64 or 128.");
    auto gate_dtype = g.scalar_type();
    TORCH_CHECK(gate_dtype == at::kFloat || gate_dtype == at::kBFloat16 || gate_dtype == at::kHalf,
                "npu_kda_gate_cumsum: g must be float32, bfloat16 or float16.");
    const std::string layout_str(layout.data(), layout.size());
    TORCH_CHECK(layout_str == "BSND" || layout_str == "BNSD" || layout_str == "TND" || layout_str == "NTD",
                "npu_kda_gate_cumsum: layout must be uppercase and one of BSND, BNSD, TND or NTD.");
    TORCH_CHECK((g.dim() == 4 && (layout_str == "BSND" || layout_str == "BNSD")) ||
                    (g.dim() == 3 && (layout_str == "TND" || layout_str == "NTD")),
                "npu_kda_gate_cumsum: layout rank does not match g rank.");
    bool is_bnsd = layout_str == "BNSD";
    bool is_ntd = layout_str == "NTD";
    int64_t T = is_bnsd ? g.sizes()[2] : (is_ntd ? g.sizes()[1] : (g.dim() == 4 ? g.sizes()[1] : g.sizes()[0]));
    int64_t K = g.dim() == 4 ? g.sizes()[3] : g.sizes()[2];
    int64_t HV = is_bnsd ? g.sizes()[1] : (is_ntd ? g.sizes()[0] : (g.dim() == 4 ? g.sizes()[2] : g.sizes()[1]));
    TORCH_CHECK(K <= 256, "npu_kda_gate_cumsum: K must be <= 256.");
    CheckKdaCuSeqlens(cu_seqlens, T, "npu_kda_gate_cumsum");
    TORCH_CHECK(!cu_seqlens.has_value() || g.dim() == 3 || g.sizes()[0] == 1,
                "npu_kda_gate_cumsum: rank4 varlen input with cu_seqlens currently requires B=1.");

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
    const char *layout_cstr = layout_str.c_str();
    EXEC_NPU_CMD_EXT(
        aclnnKdaGateCumsum,
        g, A_log_, dt_bias_, cu_seqlens,
        chunk_size, use_gate, safe, lower, layout_cstr, gk
    );
    return gk;
}

at::Tensor npu_recurrent_kda(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &g,
    const at::Tensor &beta,
    at::Tensor &initial_state,
    const at::Tensor &cu_seqlens,
    const c10::optional<at::Tensor> &ssm_state_indices,
    const c10::optional<at::Tensor> &A_log,
    const c10::optional<at::Tensor> &dt_bias,
    const c10::optional<at::Tensor> &num_accepted_tokens,
    c10::string_view layout,
    c10::optional<double> scale,
    c10::optional<bool> use_qk_l2norm_in_kernel,
    c10::optional<bool> use_gate_in_kernel,
    c10::optional<bool> use_beta_sigmoid_in_kernel,
    c10::optional<bool> allow_neg_eigval,
    c10::optional<bool> safe_gate,
    c10::optional<double> lower_bound,
    c10::optional<bool> state_v_first)
{
    const std::string layout_str(layout.data(), layout.size());
    TORCH_CHECK(layout_str == "BSND" || layout_str == "TND",
                "npu_recurrent_kda: layout must be BSND or TND.");
    bool is_tnd = layout_str == "TND";
    TORCH_CHECK((is_tnd && q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && g.dim() == 3 && beta.dim() == 2) ||
                    (!is_tnd && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && g.dim() == 4 && beta.dim() == 3),
                "npu_recurrent_kda: layout/rank mismatch. TND expects q/k [T,H,K], v [T,HV,V], "
                "g [T,HV,K], beta [T,HV]; BSND expects q/k [B,T,H,K], v [B,T,HV,V], g [B,T,HV,K], "
                "beta [B,T,HV].");
    TORCH_CHECK(q.sizes() == k.sizes(), "npu_recurrent_kda: q and k must have identical shape.");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "npu_recurrent_kda: q/k/v currently support bfloat16 only.");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device() && g.device() == q.device() &&
                    beta.device() == q.device(),
                "npu_recurrent_kda: q/k/v/g/beta must be on the same device.");
    TORCH_CHECK((g.scalar_type() == at::kFloat || g.scalar_type() == at::kBFloat16 || g.scalar_type() == at::kHalf) &&
                    (beta.scalar_type() == at::kFloat || beta.scalar_type() == at::kBFloat16 ||
                     beta.scalar_type() == at::kHalf),
                "npu_recurrent_kda: g and beta must be float32, bfloat16 or float16.");

    int64_t batch = is_tnd ? 1 : q.size(0);
    int64_t total_tokens = is_tnd ? q.size(0) : q.size(0) * q.size(1);
    int64_t dense_seq_len = is_tnd ? q.size(0) : q.size(1);
    int64_t h = is_tnd ? q.size(1) : q.size(2);
    int64_t k_dim = is_tnd ? q.size(2) : q.size(3);
    int64_t hv = is_tnd ? v.size(1) : v.size(2);
    int64_t v_dim = is_tnd ? v.size(2) : v.size(3);
    TORCH_CHECK(h > 0 && hv > 0 && k_dim > 0 && v_dim > 0 && total_tokens > 0,
                "npu_recurrent_kda: all shape dimensions must be positive.");
    TORCH_CHECK(hv % h == 0, "npu_recurrent_kda: HV must be divisible by H.");
    TORCH_CHECK(k_dim == 128 && (v_dim == 128 || v_dim == 256),
                "npu_recurrent_kda: K/V currently support only K=128,V=128 or K=128,V=256.");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.size(0) >= 2,
                "npu_recurrent_kda: cu_seqlens must be a 1D device tensor with at least two elements.");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt || cu_seqlens.scalar_type() == at::kLong,
                "npu_recurrent_kda: cu_seqlens must be int32 or int64.");
    TORCH_CHECK(cu_seqlens.device() == q.device(),
                "npu_recurrent_kda: cu_seqlens must be on the same device as q.");
    TORCH_CHECK((is_tnd && v.size(0) == total_tokens && g.size(0) == total_tokens &&
                 beta.size(0) == total_tokens && g.size(1) == hv && beta.size(1) == hv &&
                 g.size(2) == k_dim) ||
                    (!is_tnd && v.size(0) == batch && v.size(1) == dense_seq_len &&
                     g.size(0) == batch && g.size(1) == dense_seq_len && g.size(2) == hv &&
                     g.size(3) == k_dim && beta.size(0) == batch && beta.size(1) == dense_seq_len &&
                     beta.size(2) == hv),
                "npu_recurrent_kda: v/g/beta shape mismatch.");

    int64_t seq_num = cu_seqlens.size(0) - 1;
    bool state_v_first_ = state_v_first.value_or(true);
    TORCH_CHECK(state_v_first_, "npu_recurrent_kda: state_v_first=false is not supported.");
    bool has_state_indices = ssm_state_indices.has_value() && ssm_state_indices->defined();
    TORCH_CHECK(initial_state.scalar_type() == at::kFloat || initial_state.scalar_type() == at::kBFloat16,
                "npu_recurrent_kda: initial_state must be float32 or bfloat16.");
    TORCH_CHECK(initial_state.device() == q.device(),
                "npu_recurrent_kda: initial_state must be on the same device as q.");
    TORCH_CHECK(initial_state.dim() == 4 && initial_state.size(0) > 0 &&
                    initial_state.size(1) == hv && initial_state.size(2) == v_dim &&
                    initial_state.size(3) == k_dim && (has_state_indices || initial_state.size(0) == seq_num),
                "npu_recurrent_kda: initial_state must be [state_capacity,HV,V,K]; without "
                "ssm_state_indices, state_capacity must equal seq_num.");
    if (has_state_indices) {
        bool packed_1d = ssm_state_indices->dim() == 1 && ssm_state_indices->size(0) >= total_tokens;
        bool speculative_2d = ssm_state_indices->dim() == 2 && ssm_state_indices->size(0) == seq_num &&
                              ssm_state_indices->size(1) > 0;
        TORCH_CHECK(packed_1d || speculative_2d,
                    "npu_recurrent_kda: ssm_state_indices must be packed [T] or "
                    "speculative [seq_num,max_step].");
        TORCH_CHECK(ssm_state_indices->scalar_type() == at::kInt || ssm_state_indices->scalar_type() == at::kLong,
                    "npu_recurrent_kda: ssm_state_indices must be int32 or int64.");
        TORCH_CHECK(ssm_state_indices->device() == q.device(),
                    "npu_recurrent_kda: ssm_state_indices must be on the same device as q.");
    }
    if (num_accepted_tokens.has_value() && num_accepted_tokens->defined()) {
        TORCH_CHECK(has_state_indices,
                    "npu_recurrent_kda: num_accepted_tokens requires ssm_state_indices.");
        TORCH_CHECK(num_accepted_tokens->dim() == 1 && num_accepted_tokens->size(0) == seq_num,
                    "npu_recurrent_kda: num_accepted_tokens length must equal sequence number.");
        TORCH_CHECK(num_accepted_tokens->scalar_type() == at::kInt ||
                        num_accepted_tokens->scalar_type() == at::kLong,
                    "npu_recurrent_kda: num_accepted_tokens must be int32 or int64.");
        TORCH_CHECK(num_accepted_tokens->device() == q.device(),
                    "npu_recurrent_kda: num_accepted_tokens must be on the same device as q.");
    }

    bool use_gate = use_gate_in_kernel.value_or(false);
    bool safe = safe_gate.value_or(false);
    double lower = lower_bound.value_or(-5.0);
    if (use_gate) {
        TORCH_CHECK(A_log.has_value() && A_log->defined(),
                    "npu_recurrent_kda: A_log is required when use_gate_in_kernel=True.");
        TORCH_CHECK(A_log->scalar_type() == at::kFloat && A_log->dim() == 1 && A_log->size(0) == hv,
                    "npu_recurrent_kda: A_log must be float32 with shape [HV].");
        TORCH_CHECK(A_log->device() == q.device(),
                    "npu_recurrent_kda: A_log must be on the same device as q.");
        TORCH_CHECK(!safe || (lower >= -5.0 && lower < 0.0),
                    "npu_recurrent_kda: lower_bound must be in [-5, 0) when safe_gate=True.");
        if (dt_bias.has_value() && dt_bias->defined()) {
            bool valid_bias = dt_bias->scalar_type() == at::kFloat &&
                ((dt_bias->dim() == 1 && dt_bias->size(0) == hv * k_dim) ||
                 (dt_bias->dim() == 2 && dt_bias->size(0) == hv && dt_bias->size(1) == k_dim));
            TORCH_CHECK(valid_bias, "npu_recurrent_kda: dt_bias must be float32 with shape [HV*K] or [HV,K].");
            TORCH_CHECK(dt_bias->device() == q.device(),
                        "npu_recurrent_kda: dt_bias must be on the same device as q.");
        }
    } else {
        TORCH_CHECK(!safe, "npu_recurrent_kda: safe_gate only applies when use_gate_in_kernel=True.");
        TORCH_CHECK(!A_log.has_value() || !A_log->defined(),
                    "npu_recurrent_kda: A_log must be omitted when use_gate_in_kernel=False.");
        TORCH_CHECK(!dt_bias.has_value() || !dt_bias->defined(),
                    "npu_recurrent_kda: dt_bias must be omitted when use_gate_in_kernel=False.");
    }

    at::Tensor out = at::empty_like(v);
    bool output_final_state_ = true;
    double scale_ = scale.value_or(std::pow(static_cast<double>(k_dim), -0.5));
    bool use_qk_l2norm_ = use_qk_l2norm_in_kernel.value_or(false);
    bool use_beta_sigmoid_ = use_beta_sigmoid_in_kernel.value_or(false);
    bool allow_neg_eigval_ = allow_neg_eigval.value_or(false);
    const at::Tensor &ssm_state_indices_ = c10::value_or_else(ssm_state_indices, [] { return at::Tensor(); });
    const at::Tensor &A_log_ = c10::value_or_else(A_log, [] { return at::Tensor(); });
    const at::Tensor &dt_bias_ = c10::value_or_else(dt_bias, [] { return at::Tensor(); });
    const at::Tensor &num_accepted_tokens_ = c10::value_or_else(num_accepted_tokens, [] { return at::Tensor(); });
    const char *layout_cstr = layout_str.c_str();

    EXEC_NPU_CMD_EXT(
        aclnnRecurrentKda,
        q, k, v, g, beta, initial_state, cu_seqlens,
        ssm_state_indices_, A_log_, dt_bias_, num_accepted_tokens_,
        layout_cstr, scale_, output_final_state_, use_qk_l2norm_, use_gate,
        use_beta_sigmoid_, allow_neg_eigval_, safe, lower, state_v_first_,
        out
    );

    return out;
}

at::Tensor npu_chunk_scaled_dot_kkt(
    const at::Tensor &k,
    const at::Tensor &g,
    const at::Tensor &beta,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    int64_t chunk_size)
{
    TORCH_CHECK(k.dim() == 4, "npu_chunk_scaled_dot_kkt: k must be [B,Hk,T,K], got ", k.sizes());
    TORCH_CHECK(g.dim() == 3, "npu_chunk_scaled_dot_kkt: g must be [B,Hv,T], got ", g.sizes());
    TORCH_CHECK(beta.dim() == 3, "npu_chunk_scaled_dot_kkt: beta must be [B,Hv,T], got ", beta.sizes());
    TORCH_CHECK(
        k.scalar_type() == c10::ScalarType::Half || k.scalar_type() == c10::ScalarType::BFloat16,
        "npu_chunk_scaled_dot_kkt: k dtype must be float16 or bfloat16, got ", k.scalar_type());
    TORCH_CHECK(g.scalar_type() == c10::ScalarType::Float,
        "npu_chunk_scaled_dot_kkt: g dtype must be float32, got ", g.scalar_type());
    TORCH_CHECK(beta.scalar_type() == c10::ScalarType::Float,
        "npu_chunk_scaled_dot_kkt: beta dtype must be float32, got ", beta.scalar_type());
    TORCH_CHECK(
        chunk_size == 16 || chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
        "npu_chunk_scaled_dot_kkt: chunk_size must be one of 16, 32, 64, 128, got ", chunk_size);
    TORCH_CHECK(
        cu_seqlens.has_value() == chunk_indices.has_value(),
        "npu_chunk_scaled_dot_kkt: cu_seqlens and chunk_indices must be both provided or both omitted.");
    if (cu_seqlens.has_value()) {
        const auto cu = cu_seqlens.value();
        const auto chunks = chunk_indices.value();
        TORCH_CHECK(cu.size() >= 2,
            "npu_chunk_scaled_dot_kkt: cu_seqlens must have at least 2 elements, got ", cu.size());
        TORCH_CHECK(chunks.size() > 0 && chunks.size() % 2 == 0,
            "npu_chunk_scaled_dot_kkt: chunk_indices must be a non-empty flat [seq, chunk] list, got ",
            chunks.size(), " elements.");
    }

    const int64_t B = k.size(0);
    const int64_t Hk = k.size(1);
    const int64_t T = k.size(2);
    const int64_t Hv = g.size(1);
    TORCH_CHECK(
        g.size(0) == B && g.size(2) == T,
        "npu_chunk_scaled_dot_kkt: g must match k B/T and provide value heads [B,Hv,T]; k=", k.sizes(), " g=", g.sizes());
    TORCH_CHECK(
        beta.size(0) == B && beta.size(1) == Hv && beta.size(2) == T,
        "npu_chunk_scaled_dot_kkt: beta must match g as [B,Hv,T]; g=", g.sizes(), " beta=", beta.sizes());
    TORCH_CHECK(
        Hk > 0 && Hv > 0 && Hv % Hk == 0,
        "npu_chunk_scaled_dot_kkt: GVA requires Hv divisible by Hk; Hk=", Hk, " Hv=", Hv);

    at::Tensor k_contig = k.contiguous();
    at::Tensor g_contig = g.contiguous();
    at::Tensor beta_contig = beta.contiguous();
    at::Tensor A = at::empty({B, Hk, T, chunk_size}, k.options().dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD_EXT(
        aclnnChunkScaledDotKkt,
        k_contig, g_contig, beta_contig, cu_seqlens, chunk_indices, chunk_size,
        A
    );
    return A;
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
        activation_mode, pad_slot_id, run_mode, head_num,
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
    const char *input_layout_cstr = nullptr;
    const int64_t width = weight.size(0);
    const int64_t dim = weight.size(1);

    std::vector<int64_t> dx_shape;
    int64_t batch = 0;
    if (input_layout_str == "BNSD") {
        input_layout_cstr = "BNSD";
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
        input_layout_cstr = "NTD";
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
        input_layout_cstr = "TND";
        TORCH_CHECK(x.dim() == 2, "TND input must be [total_tokens, D]");
        TORCH_CHECK(dy.sizes() == x.sizes(), "TND dy shape must match x");
        TORCH_CHECK(query_start_loc.has_value(), "query_start_loc is required for TND input");
        batch = static_cast<int64_t>(query_start_loc.value().size()) - 1;
        dx_shape = x.sizes().vec();
    } else {
        TORCH_CHECK(
            input_layout_str == "BSND" || input_layout_str == "BSH",
            "input_layout must be one of BSND, BSH, TND, BNSD, or NTD");
        input_layout_cstr = "BSND";
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
        activation, input_layout_cstr,
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

at::Tensor npu_chunk_local_cumsum(
    const at::Tensor &g,
    int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices_out,
    bool reverse,
    double scale,
    bool head_first,
    c10::string_view output_dtype)
{
    TORCH_CHECK(g.dim() == 3, "npu_chunk_local_cumsum: g must be rank-3 [B, H, T], got ", g.sizes());
    TORCH_CHECK(g.scalar_type() == at::kFloat || g.scalar_type() == at::kHalf ||
                    g.scalar_type() == at::kBFloat16,
                "npu_chunk_local_cumsum: g dtype must be float32, float16, or bfloat16, got ",
                g.scalar_type());
    TORCH_CHECK(chunk_size > 0 && (chunk_size & (chunk_size - 1)) == 0,
                "npu_chunk_local_cumsum: chunk_size must be a positive power of two, got ", chunk_size);
    TORCH_CHECK(head_first, "npu_chunk_local_cumsum: only head_first=true / [B, H, T] layout is supported.");

    std::string output_dtype_str(output_dtype.data(), output_dtype.size());
    c10::ScalarType out_scalar_type = c10::ScalarType::Float;
    TORCH_CHECK(ResolveChunkLocalCumsumOutputDtype(output_dtype_str, g.scalar_type(), out_scalar_type),
                "npu_chunk_local_cumsum: output_dtype must be float32/float16/bfloat16 or same/input/none, got ",
                output_dtype_str);

    at::Tensor g_contig = g.contiguous();
    at::Tensor out = at::empty(g_contig.sizes(), g_contig.options().dtype(out_scalar_type));

    const bool has_cu_seqlens = cu_seqlens.has_value() && cu_seqlens->size() > 0;
    const bool has_chunk_indices = chunk_indices_out.has_value() && chunk_indices_out->size() > 0;
    TORCH_CHECK(has_cu_seqlens == has_chunk_indices,
                "npu_chunk_local_cumsum: cu_seqlens and chunk_indices_out must be both provided or both omitted.");
    if (has_cu_seqlens) {
        TORCH_CHECK(cu_seqlens->size() >= 2,
                    "npu_chunk_local_cumsum: cu_seqlens must have at least 2 elements, got ",
                    cu_seqlens->size());
        TORCH_CHECK((chunk_indices_out->size() % 2) == 0,
                    "npu_chunk_local_cumsum: chunk_indices_out element count must be even, got ",
                    chunk_indices_out->size());
        TORCH_CHECK(g_contig.size(0) == 1,
                    "npu_chunk_local_cumsum: B must be 1 when cu_seqlens is provided, got ", g_contig.size(0));
    }

    char *output_dtype_cstr = const_cast<char *>(output_dtype_str.c_str());
    EXEC_NPU_CMD_EXT(
        aclnnChunkLocalCumsum,
        g_contig, cu_seqlens, chunk_indices_out,
        chunk_size, reverse, scale, head_first, output_dtype_cstr,
        out
    );
    return out;
}

}  // namespace op_api
