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

at::Tensor npu_prepare_wy_repr_bwd_da(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    const at::Tensor &dw,
    const at::Tensor &du,
    const at::Tensor &g,
    c10::OptionalIntArrayRef cu_seqlens,
    c10::OptionalIntArrayRef chunk_indices,
    int64_t chunk_size
)
{
    at::Tensor dA = npu_preparation::apply_tensor_without_format(A.sizes(), A.options().dtype());

    EXEC_NPU_CMD(aclnnPrepareWyReprBwdDa, k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices, chunk_size, dA);

    return dA;
}

}  // namespace op_api
