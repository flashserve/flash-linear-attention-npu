// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// npu_chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, Tensor g, Tensor? initial_state, Tensor? cu_seqlens, Tensor? chunk_indices, bool output_final_state, int chunk_size) -> (Tensor, Tensor, Tensor)
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_chunk_gated_delta_rule_fwd_h(
    const at::Tensor &k,
    const at::Tensor &w,
    const at::Tensor &u,
    const at::Tensor &g,
    const c10::optional<at::Tensor> &initial_state,
    const c10::optional<at::Tensor> &cu_seqlens,
    const c10::optional<at::Tensor> &chunk_indices,
    c10::optional<bool> output_final_state, 
    c10::optional<int64_t> chunk_size)
{
    bool output_final_state_ = output_final_state.has_value() ? output_final_state.value() : false;
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    const at::Tensor &cu_seqlens_ = c10::value_or_else(cu_seqlens, [] { return at::Tensor(); });
    const at::Tensor &chunk_indices_ = c10::value_or_else(chunk_indices, [] { return at::Tensor(); });
    int64_t chunk_size_ = chunk_size.has_value() ? chunk_size.value() : 64;

    //cu_seqlens 存在时，NT 为 cu_seqlens的第一维度，不存在时为 T / chunk_size
    auto k_sizes = k.sizes();
    auto u_sizes = u.sizes();
    int K = k_sizes[3];
    int B = k_sizes[0];
    int T = k_sizes[2];
    int HV = u_sizes[1];
    int V = k_sizes[3];
    int NT = 0;
    if(cu_seqlens.has_value()) {
        auto cu_seqlens_sizes = chunk_indices_.sizes();
        NT = cu_seqlens_sizes[0];
    } else {
        NT = (T / chunk_size_ - 1) / chunk_size_;
    }

    at::Tensor h_out = at::zeros({B, HV, NT, K, V}, k.options().dtype());
    at::Tensor v_new_out = npu_preparation::apply_tensor_without_format(u.sizes(), u.options().dtype());
    at::Tensor final_state_out;
    if(output_final_state_) {
        auto initial_state_sizes = initial_state_.sizes();
        int BT = initial_state_sizes[2];
        final_state_out = npu_preparation::apply_tensor_without_format({B, HV, BT, K, V}, initial_state_.options().dtype());    
    }

    EXEC_NPU_CMD(aclnnChunkGatedDeltaRuleFwdH,
        k, w, u, g, initial_state_, cu_seqlens_, chunk_indices_, output_final_state_, chunk_size_, h_out, v_new_out, final_state_out);
    return std::make_tuple(h_out, v_new_out, final_state_out);
}

}  // namespace op_api
