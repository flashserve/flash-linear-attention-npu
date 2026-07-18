/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg.cpp
 * \brief Chunk backward dqkwg operator with kernel launch
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <type_traits>

#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_host/op_tiling/chunk_bwd_dqkwg_tiling_processor.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_kernel/chunk_bwd_dqkwg_common.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_kernel/chunk_bwd_dqkwg_cube.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_kernel/chunk_bwd_dqkwg_vector.h"

using TilingData = GDN::ChunkBwdDqkwgTilingData;


namespace ascend_ops {
namespace ChunkBwdDqkwg {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("chunk_bwd_dqkwg(Tensor q, Tensor k, Tensor v, Tensor g, Tensor h, Tensor dox, Tensor dh, Tensor dv, float scale, int chunk_size, *, Tensor? w=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> (Tensor, Tensor, Tensor, Tensor)");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_bwd_dqkwg_meta(
    const at::Tensor & q, const at::Tensor & k, const at::Tensor & v, const at::Tensor & g,
    const at::Tensor & h, const at::Tensor & dox, const at::Tensor & dh, const at::Tensor & dv,
    double scale, int64_t chunk_size,
    const c10::optional<at::Tensor> & w, const c10::optional<at::Tensor> & g_gamma,
    at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    auto B = v.size(0);
    auto HV = v.size(1);
    auto T = v.size(2);
    auto K = k.size(3);

    auto dq = at::empty({B, q.size(1), T, K}, q.options());
    auto dk = at::empty({B, k.size(1), T, K}, k.options());
    auto dw = at::empty({B, HV, T, K}, k.options());
    auto dg_opts = g.options();
    if (g.scalar_type() == at::kFloat) {
        dg_opts = dg_opts.dtype(at::kFloat);
    }
    auto dg = at::empty({B, HV, T}, dg_opts);
    return std::make_tuple(dq, dk, dw, dg);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("chunk_bwd_dqkwg", chunk_bwd_dqkwg_meta);
}

TilingData calc_tiling_params(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                               const at::Tensor &g, const at::Tensor &h,
                               const at::Tensor &dox, const at::Tensor &dh, const at::Tensor &dv,
                               double scale, int64_t chunk_size,
                               at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    auto q_sizes = q.sizes();
    auto k_sizes = k.sizes();
    auto v_sizes = v.sizes();
    auto g_sizes = g.sizes();
    auto h_sizes = h.sizes();
    auto dox_sizes = dox.sizes();
    auto dh_sizes = dh.sizes();
    auto dv_sizes = dv.sizes();

    gert::StorageShape qShape({q_sizes[0], q_sizes[1], q_sizes[2], q_sizes[3]}, {q_sizes[0], q_sizes[1], q_sizes[2], q_sizes[3]});
    gert::StorageShape kShape({k_sizes[0], k_sizes[1], k_sizes[2], k_sizes[3]}, {k_sizes[0], k_sizes[1], k_sizes[2], k_sizes[3]});
    gert::StorageShape vShape({v_sizes[0], v_sizes[1], v_sizes[2], v_sizes[3]}, {v_sizes[0], v_sizes[1], v_sizes[2], v_sizes[3]});
    gert::StorageShape gShape({g_sizes[0], g_sizes[1], g_sizes[2]}, {g_sizes[0], g_sizes[1], g_sizes[2]});
    gert::StorageShape hShape({h_sizes[0], h_sizes[1], h_sizes[2], h_sizes[3], h_sizes[4]}, {h_sizes[0], h_sizes[1], h_sizes[2], h_sizes[3], h_sizes[4]});
    gert::StorageShape doxShape({dox_sizes[0], dox_sizes[1], dox_sizes[2], dox_sizes[3]}, {dox_sizes[0], dox_sizes[1], dox_sizes[2], dox_sizes[3]});
    gert::StorageShape dhShape({dh_sizes[0], dh_sizes[1], dh_sizes[2], dh_sizes[3], dh_sizes[4]}, {dh_sizes[0], dh_sizes[1], dh_sizes[2], dh_sizes[3], dh_sizes[4]});
    gert::StorageShape dvShape({dv_sizes[0], dv_sizes[1], dv_sizes[2], dv_sizes[3]}, {dv_sizes[0], dv_sizes[1], dv_sizes[2], dv_sizes[3]});

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

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    optiling::ChunkBwdDqkwgTilingContext ctx{
        "chunk_bwd_dqkwg",
        &qShape,
        &kShape,
        &vShape,
        &gShape,
        &hShape,
        &doxShape,
        &dhShape,
        &dvShape,
        cuSeqlensShapePtr,
        chunkIndicesShapePtr,
        static_cast<float>(scale),
        static_cast<int32_t>(chunk_size),
        ascendcPlatform->GetCoreNumAic(),
    };

    GDN::ChunkBwdDqkwgTilingData tilingData;
    optiling::ChunkBwdDqkwgTilingProcessor processor(ctx, tilingData);
    TORCH_CHECK(processor.Process() == ge::GRAPH_SUCCESS, "chunk_bwd_dqkwg tiling failed");

    return tilingData;
}

template <typename QKVT, typename GT>
__global__ __aicore__ void chunk_bwd_dqkwg_kernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h,
    GM_ADDR do_, GM_ADDR dh, GM_ADDR dv,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR w, GM_ADDR g_gamma,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg,
    GM_ADDR workspace,
    const GDN::ChunkBwdDqkwgTilingData tilingData)
{
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendCUtils::SetOverflow(1);
    if ASCEND_IS_AIC {
        ChunkBwdDqkwgCubeProcess<QKVT, GT> cubeProcess(
            q, k, v, g, h, do_, dh, dv, cu_seqlens, chunk_indices, dq, dk, dw, dg, userWS);
        cubeProcess.Init(tilingData);
        cubeProcess.Process();
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        ChunkBwdDqkwgVectorProcess<QKVT, GT> vectorProcess(
            q, k, v, g, h, do_, dh, dv, cu_seqlens, chunk_indices, nullptr, dq, dk, dw, dg, userWS);
        vectorProcess.Init(tilingData, &pipe);
        vectorProcess.Process();
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_bwd_dqkwg_npu(
    const at::Tensor & q, const at::Tensor & k, const at::Tensor & v, const at::Tensor & g,
    const at::Tensor & h, const at::Tensor & dox, const at::Tensor & dh, const at::Tensor & dv,
    double scale, int64_t chunk_size,
    const c10::optional<at::Tensor> & w, const c10::optional<at::Tensor> & g_gamma,
    at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    const c10::OptionalDeviceGuard guard(q.device());

    auto [dq, dk, dw, dg] = chunk_bwd_dqkwg_meta(q, k, v, g, h, dox, dh, dv, scale, chunk_size, w, g_gamma, cu_seqlens, chunk_indices);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto tiling = calc_tiling_params(q, k, v, g, h, dox, dh, dv, scale, chunk_size, cu_seqlens, chunk_indices);

    auto q_ptr = (GM_ADDR)q.data_ptr();
    auto k_ptr = (GM_ADDR)k.data_ptr();
    auto v_ptr = (GM_ADDR)v.data_ptr();
    auto g_ptr = (GM_ADDR)g.data_ptr();
    auto h_ptr = (GM_ADDR)h.data_ptr();
    auto do_ptr = (GM_ADDR)dox.data_ptr();
    auto dh_ptr = (GM_ADDR)dh.data_ptr();
    auto dv_ptr = (GM_ADDR)dv.data_ptr();
    auto dq_ptr = (GM_ADDR)dq.data_ptr();
    auto dk_ptr = (GM_ADDR)dk.data_ptr();
    auto dw_ptr = (GM_ADDR)dw.data_ptr();
    auto dg_ptr = (GM_ADDR)dg.data_ptr();

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
    uint32_t blockDim = std::max<uint32_t>(ascendcPlatform->GetCoreNumAic(), 1);

    uint64_t sysWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();
    uint64_t userWorkspaceSize = static_cast<uint64_t>(tiling.totalWorkspaceSize);
    uint64_t workspaceSize = sysWorkspaceSize + userWorkspaceSize;
    void *workspace_ptr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d", ret);
        ret = aclrtMemsetAsync(workspace_ptr, workspaceSize, 0, workspaceSize, stream);
        TORCH_CHECK(ret == ACL_SUCCESS, "memset workspace failed. ERROR: %d", ret);
    }

    auto q_dtype = q.scalar_type();
    auto g_dtype = g.scalar_type();

    auto workspace_gm = (GM_ADDR)workspace_ptr;

    auto acl_call = [=]() -> int {
        if (q_dtype == at::kBFloat16 && g_dtype == at::kBFloat16) {
            using QKVT = bfloat16_t;
            using GT = bfloat16_t;
            chunk_bwd_dqkwg_kernel<QKVT, GT><<<blockDim, nullptr, stream>>>(
                q_ptr, k_ptr, v_ptr, g_ptr, h_ptr, do_ptr, dh_ptr, dv_ptr,
                cu_seqlens_ptr, chunk_indices_ptr, nullptr, nullptr,
                dq_ptr, dk_ptr, dw_ptr, dg_ptr, workspace_gm, tiling);
        } else if (q_dtype == at::kHalf && g_dtype == at::kHalf) {
            using QKVT = half;
            using GT = half;
            chunk_bwd_dqkwg_kernel<QKVT, GT><<<blockDim, nullptr, stream>>>(
                q_ptr, k_ptr, v_ptr, g_ptr, h_ptr, do_ptr, dh_ptr, dv_ptr,
                cu_seqlens_ptr, chunk_indices_ptr, nullptr, nullptr,
                dq_ptr, dk_ptr, dw_ptr, dg_ptr, workspace_gm, tiling);
        } else if (q_dtype == at::kBFloat16 && g_dtype == at::kFloat) {
            using QKVT = bfloat16_t;
            using GT = float;
            chunk_bwd_dqkwg_kernel<QKVT, GT><<<blockDim, nullptr, stream>>>(
                q_ptr, k_ptr, v_ptr, g_ptr, h_ptr, do_ptr, dh_ptr, dv_ptr,
                cu_seqlens_ptr, chunk_indices_ptr, nullptr, nullptr,
                dq_ptr, dk_ptr, dw_ptr, dg_ptr, workspace_gm, tiling);
        } else if (q_dtype == at::kHalf && g_dtype == at::kFloat) {
            using QKVT = half;
            using GT = float;
            chunk_bwd_dqkwg_kernel<QKVT, GT><<<blockDim, nullptr, stream>>>(
                q_ptr, k_ptr, v_ptr, g_ptr, h_ptr, do_ptr, dh_ptr, dv_ptr,
                cu_seqlens_ptr, chunk_indices_ptr, nullptr, nullptr,
                dq_ptr, dk_ptr, dw_ptr, dg_ptr, workspace_gm, tiling);
        } else {
            TORCH_CHECK(false, "Unsupported dtype combination: q=", q_dtype, ", g=", g_dtype);
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("ChunkBwdDqkwg", acl_call);
    c10_npu::getCurrentNPUStream().synchronize();

    if (workspaceSize > 0 && workspace_ptr != nullptr) {
        aclrtFree(workspace_ptr);
        workspace_ptr = nullptr;
    }

    return std::make_tuple(dq, dk, dw, dg);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("chunk_bwd_dqkwg", chunk_bwd_dqkwg_npu);
}

} // namespace ChunkBwdDqkwg
} // namespace ascend_ops
