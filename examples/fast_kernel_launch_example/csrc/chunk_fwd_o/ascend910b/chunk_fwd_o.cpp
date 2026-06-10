/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_o.cpp
 * \brief Chunk forward o operator with fast kernel launch.
 */

#include <algorithm>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_host/chunk_fwd_o_tiling_processor.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/chunk_fwd_o.cpp"

namespace ascend_ops {
namespace ChunkFwdO {

using TilingData = GDN::ChunkFwdOTilingData;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, Tensor g, float scale, int chunk_size, *, int[]? cu_seqlens=None, int[]? chunk_offsets=None) -> Tensor");
}

at::Tensor chunk_fwd_o_meta(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &h,
                            const at::Tensor &g, double scale, int64_t chunk_size,
                            at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_offsets)
{
    (void)q;
    (void)k;
    (void)h;
    (void)g;
    (void)scale;
    (void)chunk_size;
    (void)cu_seqlens;
    (void)chunk_offsets;
    return at::empty_like(v);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("chunk_fwd_o", chunk_fwd_o_meta);
}

struct ChunkFwdOTilingResult {
    TilingData tiling;
    uint32_t blockDim;
    size_t workspaceSize;
};

int64_t ToChunkFwdODtype(at::ScalarType dtype)
{
    if (dtype == at::kBFloat16) {
        return optiling::CHUNK_FWD_O_DTYPE_BF16;
    }
    if (dtype == at::kHalf) {
        return optiling::CHUNK_FWD_O_DTYPE_FP16;
    }
    if (dtype == at::kFloat) {
        return optiling::CHUNK_FWD_O_DTYPE_FP32;
    }
    TORCH_CHECK(false, "Unsupported dtype: ", dtype);
    return optiling::CHUNK_FWD_O_DTYPE_FP16;
}

ChunkFwdOTilingResult CalcTilingParams(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                       const at::Tensor &h, const at::Tensor &g, double scale,
                                       int64_t chunk_size, at::OptionalIntArrayRef cu_seqlens,
                                       at::OptionalIntArrayRef chunk_offsets)
{
    auto qSizes = q.sizes();
    auto kSizes = k.sizes();
    auto vSizes = v.sizes();
    auto hSizes = h.sizes();
    auto gSizes = g.sizes();

    gert::StorageShape qShape({qSizes[0], qSizes[1], qSizes[2], qSizes[3]},
                              {qSizes[0], qSizes[1], qSizes[2], qSizes[3]});
    gert::StorageShape kShape({kSizes[0], kSizes[1], kSizes[2], kSizes[3]},
                              {kSizes[0], kSizes[1], kSizes[2], kSizes[3]});
    gert::StorageShape vShape({vSizes[0], vSizes[1], vSizes[2], vSizes[3]},
                              {vSizes[0], vSizes[1], vSizes[2], vSizes[3]});
    gert::StorageShape hShape({hSizes[0], hSizes[1], hSizes[2], hSizes[3], hSizes[4]},
                              {hSizes[0], hSizes[1], hSizes[2], hSizes[3], hSizes[4]});
    gert::StorageShape gShape({gSizes[0], gSizes[1], gSizes[2]}, {gSizes[0], gSizes[1], gSizes[2]});

    gert::StorageShape *cuSeqlensShapePtr = nullptr;
    gert::StorageShape *chunkOffsetsShapePtr = nullptr;
    gert::StorageShape cuSeqlensShape;
    gert::StorageShape chunkOffsetsShape;

    if (cu_seqlens.has_value()) {
        int64_t cuDim0 = static_cast<int64_t>(cu_seqlens.value().size());
        cuSeqlensShape = gert::StorageShape({cuDim0}, {cuDim0});
        cuSeqlensShapePtr = &cuSeqlensShape;
    }
    if (chunk_offsets.has_value()) {
        int64_t offsetsDim0 = static_cast<int64_t>(chunk_offsets.value().size());
        chunkOffsetsShape = gert::StorageShape({offsetsDim0}, {offsetsDim0});
        chunkOffsetsShapePtr = &chunkOffsetsShape;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendcPlatform != nullptr, "PlatformAscendCManager is null.");
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform->GetCoreNumAic());
    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());

    optiling::ChunkFwdOTilingContext ctx{
        "chunk_fwd_o",
        &qShape,
        &kShape,
        &vShape,
        &hShape,
        &gShape,
        cuSeqlensShapePtr,
        chunkOffsetsShapePtr,
        scale,
        chunk_size,
        ToChunkFwdODtype(q.scalar_type()),
        ToChunkFwdODtype(g.scalar_type()),
        coreNum,
        sysWorkspaceSize,
    };

    TilingData tiling{};
    optiling::ChunkFwdOTilingProcessor processor(ctx, tiling);
    TORCH_CHECK(processor.Process() == ge::GRAPH_SUCCESS, "chunk_fwd_o tiling failed.");
    return ChunkFwdOTilingResult{tiling, coreNum, processor.GetWorkspaceSize()};
}

template <typename InputT, typename GT>
__global__ __aicore__ void chunk_fwd_o_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                               GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o,
                                               GM_ADDR workspace, const GDN::ChunkFwdOTilingData tilingData)
{
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GDN::ChunkFwdOKernelImpl<InputT, GT, float>(q, k, v, h, g, cu_seqlens, chunk_offsets, o, user, &tilingData);
}

template <typename InputT, typename GT>
void LaunchChunkFwdO(uint32_t blockDim, aclrtStream stream, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                     GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, GM_ADDR workspace,
                     const GDN::ChunkFwdOTilingData &tiling)
{
    chunk_fwd_o_kernel<InputT, GT><<<blockDim, nullptr, stream>>>(
        q, k, v, h, g, cu_seqlens, chunk_offsets, o, workspace, tiling);
}

at::Tensor chunk_fwd_o_npu(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &h,
                           const at::Tensor &g, double scale, int64_t chunk_size,
                           at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_offsets)
{
    const c10::OptionalDeviceGuard guard(q.device());
    auto output = chunk_fwd_o_meta(q, k, v, h, g, scale, chunk_size, cu_seqlens, chunk_offsets);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    ChunkFwdOTilingResult tilingResult =
        CalcTilingParams(q, k, v, h, g, scale, chunk_size, cu_seqlens, chunk_offsets);

    GM_ADDR qPtr = (GM_ADDR)q.data_ptr();
    GM_ADDR kPtr = (GM_ADDR)k.data_ptr();
    GM_ADDR vPtr = (GM_ADDR)v.data_ptr();
    GM_ADDR hPtr = (GM_ADDR)h.data_ptr();
    GM_ADDR gPtr = (GM_ADDR)g.data_ptr();
    GM_ADDR oPtr = (GM_ADDR)output.data_ptr();

    GM_ADDR cuSeqlensPtr = nullptr;
    GM_ADDR chunkOffsetsPtr = nullptr;
    std::vector<int64_t> cuSeqlensVec;
    std::vector<int64_t> chunkOffsetsVec;
    at::Tensor cuSeqlensTensor;
    at::Tensor chunkOffsetsTensor;

    if (cu_seqlens.has_value()) {
        cuSeqlensVec = std::vector<int64_t>(cu_seqlens.value().begin(), cu_seqlens.value().end());
        cuSeqlensTensor = at::tensor(cuSeqlensVec, at::dtype(at::kLong).device(q.device()));
        cuSeqlensPtr = (GM_ADDR)cuSeqlensTensor.data_ptr();
    }
    if (chunk_offsets.has_value()) {
        chunkOffsetsVec = std::vector<int64_t>(chunk_offsets.value().begin(), chunk_offsets.value().end());
        chunkOffsetsTensor = at::tensor(chunkOffsetsVec, at::dtype(at::kLong).device(q.device()));
        chunkOffsetsPtr = (GM_ADDR)chunkOffsetsTensor.data_ptr();
    }

    void *workspacePtr = nullptr;
    if (tilingResult.workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspacePtr, tilingResult.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: ", ret);
    }
    GM_ADDR workspaceGm = (GM_ADDR)workspacePtr;

    auto qDtype = q.scalar_type();
    auto gDtype = g.scalar_type();
    auto tiling = tilingResult.tiling;
    auto blockDim = tilingResult.blockDim;

    auto aclCall = [=]() -> int {
        if (qDtype == at::kBFloat16 && gDtype == at::kBFloat16) {
            LaunchChunkFwdO<bfloat16_t, bfloat16_t>(blockDim, stream, qPtr, kPtr, vPtr, hPtr, gPtr, cuSeqlensPtr,
                                                    chunkOffsetsPtr, oPtr, workspaceGm, tiling);
        } else if (qDtype == at::kHalf && gDtype == at::kHalf) {
            LaunchChunkFwdO<half, half>(blockDim, stream, qPtr, kPtr, vPtr, hPtr, gPtr, cuSeqlensPtr,
                                        chunkOffsetsPtr, oPtr, workspaceGm, tiling);
        } else if (qDtype == at::kBFloat16 && gDtype == at::kFloat) {
            LaunchChunkFwdO<bfloat16_t, float>(blockDim, stream, qPtr, kPtr, vPtr, hPtr, gPtr, cuSeqlensPtr,
                                               chunkOffsetsPtr, oPtr, workspaceGm, tiling);
        } else if (qDtype == at::kHalf && gDtype == at::kFloat) {
            LaunchChunkFwdO<half, float>(blockDim, stream, qPtr, kPtr, vPtr, hPtr, gPtr, cuSeqlensPtr,
                                         chunkOffsetsPtr, oPtr, workspaceGm, tiling);
        } else {
            TORCH_CHECK(false, "Unsupported dtype combination: q=", qDtype, ", g=", gDtype);
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("ChunkFwdO", aclCall);

    if (workspacePtr != nullptr) {
        aclrtFree(workspacePtr);
    }

    return output;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("chunk_fwd_o", chunk_fwd_o_npu);
}

} // namespace ChunkFwdO
} // namespace ascend_ops
