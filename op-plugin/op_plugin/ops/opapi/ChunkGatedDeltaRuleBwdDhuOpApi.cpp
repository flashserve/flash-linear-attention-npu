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
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_chunk_gated_delta_rule_bwd_dhu(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &w,
    const at::Tensor &d_o,
    const at::Tensor &dv,
    const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &gK,
    const c10::optional<at::Tensor> &h0,
    const c10::optional<at::Tensor> &dht,
    c10::OptionalIntArrayRef cu_seqlens,
    c10::OptionalIntArrayRef chunk_indices,
    double scale, 
    int64_t chunk_size
)
{
    auto q_size = q.sizes();
    auto dv_size = dv.sizes();
    int B = q_size[0];
    int H = q_size[1];
    int T = q_size[2];
    int K = q_size[3];
    int V = dv_size[3];
    int chunk_num = T / chunk_size; 

    if (chunk_indices.has_value()) {
        auto chunk_indices_ref = chunk_indices.value();  // 获取ArrayRef
        chunk_num = int(chunk_indices_ref.size() / 2);
    }

    at::Tensor dv2 = npu_preparation::apply_tensor_without_format(dv.sizes(), dv.options().dtype());
    at::Tensor dh = npu_preparation::apply_tensor_without_format({B,H,chunk_num,K,V}, q.options().dtype());
    at::Tensor dh0 = npu_preparation::apply_tensor_without_format({B,H,chunk_num,K,V}, q.options().dtype());
    const at::Tensor &gK_ = c10::value_or_else(gK, [] { return at::Tensor(); });
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    const at::Tensor &h0_ = c10::value_or_else(h0, [] { return at::Tensor(); });
    const at::Tensor &dht_ = c10::value_or_else(dht, [] { return at::Tensor(); });
    EXEC_NPU_CMD(aclnnChunkGatedDeltaRuleBwdDhu,
        q, k, w, d_o, dv, g_, gK_, h0_, dht_, cu_seqlens, chunk_indices, scale, chunk_size, dh, dh0, dv2);
    return std::make_tuple(dh, dh0, dv2); 
}

}  // namespace op_api
