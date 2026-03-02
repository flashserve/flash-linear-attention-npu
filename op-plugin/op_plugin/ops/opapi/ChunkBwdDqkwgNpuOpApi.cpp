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

// Tensor q, Tensor k, Tensor d_o, Tensor g, Tensor? upper_tri_matrix, Tensor? g_gamma, Tensor? A, Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size) -> (Tensor)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_chunk_bwd_dqkwg(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &g,
    const at::Tensor &h,
    const at::Tensor &dox,
    const at::Tensor &dh,
    const at::Tensor &dv,
    const c10::optional<at::Tensor> &cu_seqlens,
    const c10::optional<at::Tensor> &chunk_indices,
    c10::optional<double> scale, 
    int64_t chunk_size)
{

    at::Tensor dq = npu_preparation::apply_tensor_without_format(q.sizes(), q.options().dtype());
    at::Tensor dk = npu_preparation::apply_tensor_without_format(k.sizes(), k.options().dtype());
    at::Tensor dw = npu_preparation::apply_tensor_without_format(k.sizes(), k.options().dtype());
    at::Tensor dg = npu_preparation::apply_tensor_without_format(g.sizes(), g.options().dtype());
    const at::Tensor &cu_seqlens_ = c10::value_or_else(cu_seqlens, [] { return at::Tensor(); });
    const at::Tensor &chunk_indices_ = c10::value_or_else(chunk_indices, [] { return at::Tensor(); });
    float scale_real = static_cast<float>(scale.value_or(1.0));
    EXEC_NPU_CMD(aclnnChunkBwdDqkwg,
        q, k, v, g, h, dox, dh, dv, cu_seqlens_, chunk_indices_, scale_real, chunk_size, dq, dk, dw, dg);
    return std::tie(dq, dk, dw, dg);
}

}  // namespace op_api
