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
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;


// Tensor k, Tensor v, Tensor beta, Tensor a, Tensor dA, Tensor dW, Tensor du, Tensor g, Tensor? cu_seqlens, Tensor? chunk_indices, int chunk_sizes -> (Tensor, Tensor, Tensor, Tensor)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_prepare_wy_repr_bwd_full(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    const at::Tensor &dA,
    const at::Tensor &dw,
    const at::Tensor &du,
    const at::Tensor &g,
    c10::OptionalIntArrayRef cu_seqlens,
    c10::OptionalIntArrayRef chunk_indices,
    int64_t chunk_sizes)
{
    auto cu_seqlens_ = cu_seqlens.value_or(at::IntArrayRef{});
    auto chunk_indices_ = chunk_indices.value_or(at::IntArrayRef{});
    
    at::Tensor dk = npu_preparation::apply_tensor_without_format(k.sizes(), k.options().dtype());
    at::Tensor dv = npu_preparation::apply_tensor_without_format(v.sizes(), v.options().dtype());
    at::Tensor dg = npu_preparation::apply_tensor_without_format(g.sizes(), g.options().dtype());
    at::Tensor dbeta = npu_preparation::apply_tensor_without_format(beta.sizes(), beta.options().dtype());

    EXEC_NPU_CMD(aclnnPrepareWyReprBwdFull,
        k, v, beta, A, dA, dw, du, g, cu_seqlens_, chunk_indices_, chunk_sizes, dk, dv, dbeta, dg);
    return std::tie(dk, dv, dbeta, dg);
}

}  // namespace op_api
