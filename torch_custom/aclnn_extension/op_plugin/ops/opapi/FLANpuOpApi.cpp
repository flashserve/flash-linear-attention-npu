// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
    if (h0.has_value()) {
        dh0 = at::empty({B, H, chunk_num, K, V}, q.options());
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

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_chunk_bwd_dqkwg(const at::Tensor & q, const at::Tensor & k, const at::Tensor & v, const at::Tensor & g, const at::Tensor & h, const at::Tensor & dox, const at::Tensor & dh, const at::Tensor & dv, int64_t chunk_size, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices, c10::optional<double> scale)
{
    // 创建输出tensor
    at::Tensor dq = at::empty_like(q);
    at::Tensor dk = at::empty_like(k);
    at::Tensor dw = at::empty_like(k);
    at::Tensor dg = at::empty_like(g);

    // scale处理
    float scale_real = static_cast<float>(scale.value_or(1.0));

    // 调用ACLNN算子
    EXEC_NPU_CMD_EXT(
        aclnnChunkBwdDqkwg,
        q, k, v, g, h,
        dox, dh, dv,
        cu_seqlens, chunk_indices, scale_real, chunk_size,
        dq, dk, dw, dg
    );
    return std::make_tuple(dq, dk, dw, dg);
}

at::Tensor npu_chunk_fwd_o(const at::Tensor & q, const at::Tensor & k, const at::Tensor & v, const at::Tensor & h, const at::Tensor & g, double scale, const c10::optional<at::Tensor> & cu_seqlens, const c10::optional<at::Tensor> & chunk_indices, c10::optional<int64_t> chunk_size)
{
    // 创建输出tensor
    at::Tensor o = at::empty_like(v);

    // optional tensor处理
    const at::Tensor &cu_seqlens_ = c10::value_or_else(cu_seqlens, [] { return at::Tensor(); });
    const at::Tensor &chunk_indices_ = c10::value_or_else(chunk_indices, [] { return at::Tensor(); });

    // chunk_size默认值
    int64_t chunk_size_ = chunk_size.value_or(64);

    // 调用ACLNN算子
    EXEC_NPU_CMD_EXT(
        aclnnChunkFwdO,
        q, k, v, h, g,
        cu_seqlens_, chunk_indices_, scale, chunk_size_,
        o
    );
    return o;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_chunk_gated_delta_rule_fwd_h(const at::Tensor & k, const at::Tensor & w, const at::Tensor & u, const at::Tensor & g, const c10::optional<at::Tensor> & initial_state, const c10::optional<at::Tensor> & cu_seqlens, const c10::optional<at::Tensor> & chunk_indices, c10::optional<bool> output_final_state, c10::optional<int64_t> chunk_size)
{
    // optional 参数处理
    bool output_final_state_ = output_final_state.value_or(false);
    int64_t chunk_size_ = chunk_size.value_or(64);
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    const at::Tensor &cu_seqlens_ = c10::value_or_else(cu_seqlens, [] { return at::Tensor(); });
    const at::Tensor &chunk_indices_ = c10::value_or_else(chunk_indices, [] { return at::Tensor(); });

    // 计算shape
    auto k_sizes = k.sizes();
    auto u_sizes = u.sizes();
    int64_t B = k_sizes[0];
    int64_t T = k_sizes[2];
    int64_t K = k_sizes[3];
    int64_t V = k_sizes[3];
    int64_t HV = u_sizes[1];
    int64_t NT = 0;

    if (cu_seqlens.has_value()) {
        NT = cu_seqlens_.sizes()[0];
    } else {
        NT = (T + chunk_size_ - 1) / chunk_size_;
    }

    // 创建输出tensor
    at::Tensor h_out = at::zeros({B, HV, NT, K, V}, k.options());
    at::Tensor v_new_out = at::empty_like(u);
    at::Tensor final_state_out;
    if (output_final_state_) {
        auto initial_state_sizes = initial_state_.sizes();
        int64_t BT = initial_state_sizes[2];

        final_state_out = at::empty({B, HV, BT, K, V}, initial_state_.options());
    } else {
        final_state_out = at::Tensor();
    }

    // 调用ACLNN算子
    EXEC_NPU_CMD_EXT(
        aclnnChunkGatedDeltaRuleFwdH,
        k, w, u, g,
        initial_state_, cu_seqlens_, chunk_indices_,output_final_state_, chunk_size_,
        h_out, v_new_out, final_state_out
    );
    return std::make_tuple(h_out, v_new_out, final_state_out);
}

}  // namespace op_api
