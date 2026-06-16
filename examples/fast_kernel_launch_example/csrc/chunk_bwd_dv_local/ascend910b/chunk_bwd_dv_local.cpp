/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dv_local.cpp
 * \brief Chunk backward dv local operator with kernel launch
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <type_traits>

#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/op_host/chunk_bwd_dv_local_tiling_processor.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/op_kernel/chunk_bwd_dv_local_common.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/op_kernel/chunk_bwd_dv_local.cpp"

using TilingData = GDN::ChunkBwdDvLocalTilingData;


namespace ascend_ops {
namespace ChunkBwdDvLocal {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("chunk_bwd_dv_local(Tensor q, Tensor k, Tensor d_o, Tensor g, float scale, int chunk_size, *, Tensor? g_gamma=None, Tensor? A=None, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> Tensor");
}

at::Tensor chunk_bwd_dv_local_meta(const at::Tensor & q, const at::Tensor & k, const at::Tensor & d_o, const at::Tensor & g, double scale, int64_t chunk_size, const c10::optional<at::Tensor> & g_gamma, const c10::optional<at::Tensor> & A, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    at::Tensor dv = at::empty_like(d_o);
    return dv;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("chunk_bwd_dv_local", chunk_bwd_dv_local_meta);
}

TilingData calc_tiling_params(const at::Tensor &q, const at::Tensor &k, const at::Tensor &d_o, const at::Tensor &g, double scale, int64_t chunk_size, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    auto q_sizes = q.sizes();
    auto k_sizes = k.sizes();
    auto d_o_sizes = d_o.sizes();
    auto g_sizes = g.sizes();

    gert::StorageShape qShape({q_sizes[0], q_sizes[1], q_sizes[2], q_sizes[3]}, {q_sizes[0], q_sizes[1], q_sizes[2], q_sizes[3]});
    gert::StorageShape kShape({k_sizes[0], k_sizes[1], k_sizes[2], k_sizes[3]}, {k_sizes[0], k_sizes[1], k_sizes[2], k_sizes[3]});
    gert::StorageShape dOShape({d_o_sizes[0], d_o_sizes[1], d_o_sizes[2], d_o_sizes[3]}, {d_o_sizes[0], d_o_sizes[1], d_o_sizes[2], d_o_sizes[3]});
    gert::StorageShape gShape({g_sizes[0], g_sizes[1], g_sizes[2]}, {g_sizes[0], g_sizes[1], g_sizes[2]});

    gert::StorageShape *cuSeqlensShapePtr = nullptr;
    gert::StorageShape *chunkIndicesShapePtr = nullptr;
    gert::StorageShape cuSeqlensShape;
    gert::StorageShape chunkIndicesShape;

    if (cu_seqlens.has_value()) {
        auto cu_sizes = cu_seqlens.value();
        int64_t cu_dim0 = cu_sizes.size();
        cuSeqlensShape = gert::StorageShape({cu_dim0}, {cu_dim0});
        cuSeqlensShapePtr = &cuSeqlensShape;
    }
    if (chunk_indices.has_value()) {
        auto ci_sizes = chunk_indices.value();
        int64_t ci_dim0 = ci_sizes.size();
        chunkIndicesShape = gert::StorageShape({ci_dim0}, {ci_dim0});
        chunkIndicesShapePtr = &chunkIndicesShape;
    }

    optiling::ChunkBwdDvLocalTilingContext ctx{
        "chunk_bwd_dv_local",
        &qShape,
        &kShape,
        &dOShape,
        &gShape,
        cuSeqlensShapePtr,
        chunkIndicesShapePtr,
        scale,
        static_cast<int32_t>(chunk_size),
    };

    GDN::ChunkBwdDvLocalTilingData tilingData;
    optiling::ChunkBwdDvLocalTilingProcessor processor(ctx, tilingData);

    processor.Process();

    return tilingData;
}

template <typename QKVT, typename GT, int V>
template <typename QKVT, typename GT, int V>
__global__ __aicore__ void chunk_bwd_dv_local_kernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR d_o, GM_ADDR g,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR d_v, GM_ADDR workspace,
    const GDN::ChunkBwdDvLocalTilingData tilingData)
{
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (cu_seqlens == nullptr) {
        GDN::FixedLengthStrategy fixedStrategy{tilingData.chunkSize, tilingData.t, tilingData.chunkNumForT};
        GDN::ChunkBwdDvLocalKernelImpl<QKVT, GT, V>(q, k, d_o, g, nullptr, nullptr, cu_seqlens, chunk_indices, d_v,
                                                     userWS, &tilingData, fixedStrategy);
    } else {
        GDN::VariableLengthStrategy variableStrategy{tilingData.chunkSize, tilingData.t, tilingData.chunkNumForT,
                                                     cu_seqlens, chunk_indices};
        GDN::ChunkBwdDvLocalKernelImpl<QKVT, GT, V>(q, k, d_o, g, nullptr, nullptr, cu_seqlens, chunk_indices, d_v,
                                                     userWS, &tilingData, variableStrategy);
    }
}

at::Tensor chunk_bwd_dv_local_npu(const at::Tensor & q, const at::Tensor & k, const at::Tensor & d_o, const at::Tensor & g, double scale, int64_t chunk_size, const c10::optional<at::Tensor> & g_gamma, const c10::optional<at::Tensor> & A, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    const c10::OptionalDeviceGuard guard(q.device());

    auto dv = chunk_bwd_dv_local_meta(q, k, d_o, g, scale, chunk_size, g_gamma, A, cu_seqlens, chunk_indices);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto tiling = calc_tiling_params(q, k, d_o, g, scale, chunk_size, cu_seqlens, chunk_indices);

    auto q_ptr = (GM_ADDR)q.data_ptr();
    auto k_ptr = (GM_ADDR)k.data_ptr();
    auto d_o_ptr = (GM_ADDR)d_o.data_ptr();
    auto g_ptr = (GM_ADDR)g.data_ptr();
    auto dv_ptr = (GM_ADDR)dv.data_ptr();

    GM_ADDR cu_seqlens_ptr = nullptr;
    GM_ADDR chunk_indices_ptr = nullptr;

    std::vector<int64_t> cu_seqlens_vec;
    std::vector<int64_t> chunk_indices_vec;
    at::Tensor cu_seqlens_tensor;
    at::Tensor chunk_indices_tensor;

    if (cu_seqlens.has_value()) {
        cu_seqlens_vec = std::vector<int64_t>(cu_seqlens.value().begin(), cu_seqlens.value().end());
        cu_seqlens_tensor = at::tensor(cu_seqlens_vec, at::dtype(at::kLong).device(q.device()));
        cu_seqlens_ptr = (GM_ADDR)cu_seqlens_tensor.data_ptr();
    }
    if (chunk_indices.has_value()) {
        chunk_indices_vec = std::vector<int64_t>(chunk_indices.value().begin(), chunk_indices.value().end());
        chunk_indices_tensor = at::tensor(chunk_indices_vec, at::dtype(at::kLong).device(q.device()));
        chunk_indices_ptr = (GM_ADDR)chunk_indices_tensor.data_ptr();
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAic();
    uint32_t blockDim = std::min(tiling.chunkNumForT * tiling.b, coreNum);

    uint64_t sysWorkspaceSize = 16U * 1024U * 1024U;
    uint64_t userWorkspaceSize =
        optiling::QKV_DTYPE_SIZE * blockDim * tiling.headBufNum * tiling.chunkSize * tiling.chunkSize;
    uint64_t workspaceSize = sysWorkspaceSize + userWorkspaceSize;
    void *workspace_ptr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d", ret);
    }

    auto q_dtype = q.scalar_type();
    auto g_dtype = g.scalar_type();

    auto workspace_gm = (GM_ADDR)workspace_ptr;

    auto acl_call = [=]() -> int {
        if (q_dtype == at::kBFloat16 && g_dtype == at::kBFloat16) {
            using QKVT = bfloat16_t;
            using GT = bfloat16_t;
            if (tiling.v == 128) {
                chunk_bwd_dv_local_kernel<QKVT, GT, 128><<<blockDim, nullptr, stream>>>(
                    q_ptr, k_ptr, d_o_ptr, g_ptr, cu_seqlens_ptr, chunk_indices_ptr, dv_ptr, workspace_gm, tiling);
            } else if (tiling.v == 256) {
                chunk_bwd_dv_local_kernel<QKVT, GT, 256><<<blockDim, nullptr, stream>>>(
                    q_ptr, k_ptr, d_o_ptr, g_ptr, cu_seqlens_ptr, chunk_indices_ptr, dv_ptr, workspace_gm, tiling);
            }
        } else if (q_dtype == at::kHalf && g_dtype == at::kHalf) {
            using QKVT = half;
            using GT = half;
            if (tiling.v == 128) {
                chunk_bwd_dv_local_kernel<QKVT, GT, 128><<<blockDim, nullptr, stream>>>(
                    q_ptr, k_ptr, d_o_ptr, g_ptr, cu_seqlens_ptr, chunk_indices_ptr, dv_ptr, workspace_gm, tiling);
            } else if (tiling.v == 256) {
                chunk_bwd_dv_local_kernel<QKVT, GT, 256><<<blockDim, nullptr, stream>>>(
                    q_ptr, k_ptr, d_o_ptr, g_ptr, cu_seqlens_ptr, chunk_indices_ptr, dv_ptr, workspace_gm, tiling);
            }
        } else if (q_dtype == at::kBFloat16 && g_dtype == at::kFloat) {
            using QKVT = bfloat16_t;
            using GT = float;
            if (tiling.v == 128) {
                chunk_bwd_dv_local_kernel<QKVT, GT, 128><<<blockDim, nullptr, stream>>>(
                    q_ptr, k_ptr, d_o_ptr, g_ptr, cu_seqlens_ptr, chunk_indices_ptr, dv_ptr, workspace_gm, tiling);
            } else if (tiling.v == 256) {
                chunk_bwd_dv_local_kernel<QKVT, GT, 256><<<blockDim, nullptr, stream>>>(
                    q_ptr, k_ptr, d_o_ptr, g_ptr, cu_seqlens_ptr, chunk_indices_ptr, dv_ptr, workspace_gm, tiling);
            }
        } else if (q_dtype == at::kHalf && g_dtype == at::kFloat) {
            using QKVT = half;
            using GT = float;
            if (tiling.v == 128) {
                chunk_bwd_dv_local_kernel<QKVT, GT, 128><<<blockDim, nullptr, stream>>>(
                    q_ptr, k_ptr, d_o_ptr, g_ptr, cu_seqlens_ptr, chunk_indices_ptr, dv_ptr, workspace_gm, tiling);
            } else if (tiling.v == 256) {
                chunk_bwd_dv_local_kernel<QKVT, GT, 256><<<blockDim, nullptr, stream>>>(
                    q_ptr, k_ptr, d_o_ptr, g_ptr, cu_seqlens_ptr, chunk_indices_ptr, dv_ptr, workspace_gm, tiling);
            }
        } else {
            TORCH_CHECK(false, "Unsupported dtype combination: q=", q_dtype, ", g=", g_dtype);
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("ChunkBwdDvLocal", acl_call);
    auto sync_ret = aclrtSynchronizeStream(stream);
    TORCH_CHECK(sync_ret == ACL_SUCCESS, "aclrtSynchronizeStream failed. ERROR: ", sync_ret);

    if (workspaceSize > 0 && workspace_ptr != nullptr) {
        aclrtFree(workspace_ptr);
        workspace_ptr = nullptr;
    }

    return dv;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("chunk_bwd_dv_local", chunk_bwd_dv_local_npu);
}

} // namespace ChunkBwdDvLocal
} // namespace ascend_ops
