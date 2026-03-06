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

// npu_chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, Tensor g, Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size) -> Tensor
at::Tensor npu_chunk_fwd_o(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &h,
    const at::Tensor &g,
    const c10::optional<at::Tensor> &cu_seqlens,
    const c10::optional<at::Tensor> &chunk_indices,
    double scale, 
    c10::optional<int64_t> chunk_size)
{
    at::Tensor o = npu_preparation::apply_tensor_without_format(v.sizes(), v.options().dtype());
    const at::Tensor &cu_seqlens_ = c10::value_or_else(cu_seqlens, [] { return at::Tensor(); });
    const at::Tensor &chunk_indices_ = c10::value_or_else(chunk_indices, [] { return at::Tensor(); });
    int64_t chunk_size_ = chunk_size.has_value() ? chunk_size.value() : 64;
    EXEC_NPU_CMD(aclnnChunkFwdO,
        q, k, v, h, g, cu_seqlens_, chunk_indices_, scale, chunk_size_, o);
    return o;
}

}  // namespace op_api
