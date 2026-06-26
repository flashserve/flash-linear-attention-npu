/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu.cpp
 * \brief Chunk gated delta rule backward dhu operator with fast kernel launch.
 */

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_host/op_tiling/chunk_gated_delta_rule_bwd_dhu_tiling_processor.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_kernel/chunk_gated_delta_rule_bwd_dhu.cpp"

namespace ascend_ops {
namespace ChunkGatedDeltaRuleBwdDhu {

using TilingData = GDN::ChunkGatedDeltaRuleBwdDhuTilingData;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("chunk_gated_delta_rule_bwd_dhu(Tensor q, Tensor k, Tensor w, Tensor d_o, Tensor dv, float scale, int chunk_size, *, Tensor? g=None, Tensor? gK=None, Tensor? h0=None, Tensor? dht=None, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> (Tensor, Tensor, Tensor)");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_bwd_dhu_meta(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, const at::Tensor &d_o, const at::Tensor &dv,
    double scale, int64_t chunk_size, const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &gK,
    const c10::optional<at::Tensor> &h0, const c10::optional<at::Tensor> &dht, at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices)
{
    (void)scale;
    (void)gK;
    (void)dht;

    int64_t B = q.size(0);
    int64_t Hk = q.size(1);
    int64_t T = q.size(2);
    int64_t K = q.size(3);
    int64_t Hv = dv.size(1);
    int64_t V = dv.size(3);
    int64_t chunkNum = (T + chunk_size - 1) / chunk_size;
    if (chunk_indices.has_value()) {
        chunkNum = static_cast<int64_t>(chunk_indices.value().size()) / 2;
    }

    at::Tensor dh = at::empty({B, Hv, chunkNum, K, V}, q.options());
    at::Tensor dv2 = at::empty_like(dv);
    at::Tensor dh0;
    if (h0.has_value()) {
        dh0 = at::empty({B, Hv, chunkNum, K, V}, q.options());
    } else {
        dh0 = at::empty({0}, q.options());
    }
    return std::make_tuple(dh, dh0, dv2);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("chunk_gated_delta_rule_bwd_dhu", chunk_gated_delta_rule_bwd_dhu_meta);
}

struct ChunkGatedDeltaRuleBwdDhuTilingResult {
    TilingData tiling;
    uint32_t blockDim;
    size_t workspaceSize;
};

void CheckInputRank(const at::Tensor &tensor, int64_t rank, const char *name)
{
    TORCH_CHECK(tensor.dim() == rank, name, " should be ", rank, "D, but got ", tensor.dim(), "D.");
}

void CheckSameDevice(const at::Tensor &base, const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device() == base.device(), name, " should be on the same device as q.");
}

void CheckInputs(const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, const at::Tensor &d_o,
                 const at::Tensor &dv, const at::Tensor &g, at::OptionalIntArrayRef cu_seqlens,
                 at::OptionalIntArrayRef chunk_indices)
{
    CheckInputRank(q, 4, "q");
    CheckInputRank(k, 4, "k");
    CheckInputRank(w, 4, "w");
    CheckInputRank(d_o, 4, "d_o");
    CheckInputRank(dv, 4, "dv");
    CheckInputRank(g, 3, "g");

    CheckSameDevice(q, k, "k");
    CheckSameDevice(q, w, "w");
    CheckSameDevice(q, d_o, "d_o");
    CheckSameDevice(q, dv, "dv");
    CheckSameDevice(q, g, "g");

    int64_t B = q.size(0);
    int64_t Hk = q.size(1);
    int64_t T = q.size(2);
    int64_t K = q.size(3);
    int64_t Hv = dv.size(1);
    int64_t V = dv.size(3);

    TORCH_CHECK(k.size(0) == B && k.size(1) == Hk && k.size(2) == T && k.size(3) == K,
              "k must match q as [B,Hk,T,K]; q=", q.sizes(), " k=", k.sizes());
    TORCH_CHECK(w.size(0) == B && w.size(1) == Hv && w.size(2) == T && w.size(3) == K,
              "w must be [B,Hv,T,K]; w=", w.sizes(), " dv=", dv.sizes());
    TORCH_CHECK(d_o.size(0) == B && d_o.size(1) == Hv && d_o.size(2) == T && d_o.size(3) == V,
              "d_o must be [B,Hv,T,V]; d_o=", d_o.sizes());
    TORCH_CHECK(dv.size(0) == B && dv.size(1) == Hv && dv.size(2) == T && dv.size(3) == V,
              "dv must be [B,Hv,T,V]; dv=", dv.sizes());
    TORCH_CHECK(Hk > 0 && Hv > 0 && Hv % Hk == 0,
              "GVA requires Hv divisible by Hk; Hk=", Hk, " Hv=", Hv);
    TORCH_CHECK(g.size(0) == B && g.size(1) == Hv && g.size(2) == T,
              "g must be [B,Hv,T]; g=", g.sizes());

    auto qDtype = q.scalar_type();
    auto gDtype = g.scalar_type();
    TORCH_CHECK(qDtype == at::kHalf || qDtype == at::kBFloat16,
              "q/k/w/d_o/dv only support float16 or bfloat16, but got ", qDtype);
    TORCH_CHECK(k.scalar_type() == qDtype && w.scalar_type() == qDtype && d_o.scalar_type() == qDtype &&
                    dv.scalar_type() == qDtype,
              "k/w/d_o/dv should have the same dtype as q.");
    TORCH_CHECK(gDtype == qDtype || gDtype == at::kFloat,
              "g should have the same dtype as q or float32, but g is ", gDtype, " and q is ", qDtype);
    TORCH_CHECK(cu_seqlens.has_value() == chunk_indices.has_value(),
              "cu_seqlens and chunk_indices should be both provided or both None.");
}

ge::DataType ToGeDtype(at::ScalarType dtype)
{
    if (dtype == at::kBFloat16) {
        return ge::DT_BF16;
    }
    if (dtype == at::kHalf) {
        return ge::DT_FLOAT16;
    }
    if (dtype == at::kFloat) {
        return ge::DT_FLOAT;
    }
    TORCH_CHECK(false, "Unsupported dtype: ", dtype);
    return ge::DT_FLOAT16;
}

ChunkGatedDeltaRuleBwdDhuTilingResult CalcTilingParams(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, const at::Tensor &d_o, const at::Tensor &dv,
    const at::Tensor &g, double scale, int64_t chunk_size, at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices)
{
    auto qSizes = q.sizes();
    auto kSizes = k.sizes();
    auto wSizes = w.sizes();
    auto doSizes = d_o.sizes();
    auto dvSizes = dv.sizes();
    auto gSizes = g.sizes();

    gert::StorageShape qShape({qSizes[0], qSizes[1], qSizes[2], qSizes[3]},
                              {qSizes[0], qSizes[1], qSizes[2], qSizes[3]});
    gert::StorageShape kShape({kSizes[0], kSizes[1], kSizes[2], kSizes[3]},
                              {kSizes[0], kSizes[1], kSizes[2], kSizes[3]});
    gert::StorageShape wShape({wSizes[0], wSizes[1], wSizes[2], wSizes[3]},
                              {wSizes[0], wSizes[1], wSizes[2], wSizes[3]});
    gert::StorageShape doShape({doSizes[0], doSizes[1], doSizes[2], doSizes[3]},
                               {doSizes[0], doSizes[1], doSizes[2], doSizes[3]});
    gert::StorageShape dvShape({dvSizes[0], dvSizes[1], dvSizes[2], dvSizes[3]},
                                {dvSizes[0], dvSizes[1], dvSizes[2], dvSizes[3]});
    gert::StorageShape gShape({gSizes[0], gSizes[1], gSizes[2]}, {gSizes[0], gSizes[1], gSizes[2]});

    gert::StorageShape *cuSeqlensShapePtr = nullptr;
    gert::StorageShape *chunkIndicesShapePtr = nullptr;
    gert::StorageShape cuSeqlensShape;
    gert::StorageShape chunkIndicesShape;

    if (cu_seqlens.has_value()) {
        int64_t cuDim0 = static_cast<int64_t>(cu_seqlens.value().size());
        cuSeqlensShape = gert::StorageShape({cuDim0}, {cuDim0});
        cuSeqlensShapePtr = &cuSeqlensShape;
    }
    if (chunk_indices.has_value()) {
        int64_t ciDim0 = static_cast<int64_t>(chunk_indices.value().size());
        chunkIndicesShape = gert::StorageShape({ciDim0}, {ciDim0});
        chunkIndicesShapePtr = &chunkIndicesShape;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendcPlatform != nullptr, "PlatformAscendCManager is null.");
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform->GetCoreNumAic());
    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    TilingData tiling{};
    optiling::ChunkGatedDeltaRuleBwdDhuTilingContext ctx{
        "chunk_gated_delta_rule_bwd_dhu",
        &qShape,
        &kShape,
        &wShape,
        &doShape,
        &dvShape,
        &gShape,
        cuSeqlensShapePtr,
        chunkIndicesShapePtr,
        ToGeDtype(q.scalar_type()),
        ToGeDtype(g.scalar_type()),
        true,
        true,
        scale,
        static_cast<int32_t>(chunk_size),
        ubSize,
        coreNum,
        sysWorkspaceSize,
    };

    optiling::ChunkGatedDeltaRuleBwdDhuTilingProcessor processor(ctx, tiling);
    TORCH_CHECK(processor.Process() == ge::GRAPH_SUCCESS, "chunk_gated_delta_rule_bwd_dhu tiling failed.");
    return ChunkGatedDeltaRuleBwdDhuTilingResult{tiling, processor.GetBlockDim(), processor.GetWorkspaceSize()};
}

template <typename DT, typename GT>
__global__ __aicore__ void chunk_gated_delta_rule_bwd_dhu_kernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR h0, GM_ADDR dht,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace,
    const GDN::ChunkGatedDeltaRuleBwdDhuTilingData tilingData)
{
    (void)gk;
    (void)h0;
    (void)dht;
    AscendC::AscendCUtils::SetOverflow(1);
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    if constexpr (std::is_same_v<GT, float>) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    }
    GDN::ChunkGatedDeltaRuleBwdDhuKernelImpl<DT, GT>(
        q, k, w, d_o, dv, g, cu_seqlens, chunk_indices, dh, dh0, dv2, userWS, &tilingData);
}

template <typename DT, typename GT>
void LaunchChunkGatedDeltaRuleBwdDhu(uint32_t blockDim, aclrtStream stream, GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o,
                                     GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dh,
                                     GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace,
                                     const GDN::ChunkGatedDeltaRuleBwdDhuTilingData &tiling)
{
    chunk_gated_delta_rule_bwd_dhu_kernel<DT, GT><<<blockDim, nullptr, stream>>>(
        q, k, w, d_o, dv, g, nullptr, nullptr, nullptr, cu_seqlens, chunk_indices, dh, dh0, dv2, workspace, tiling);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_bwd_dhu_npu(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, const at::Tensor &d_o, const at::Tensor &dv,
    double scale, int64_t chunk_size, const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &gK,
    const c10::optional<at::Tensor> &h0, const c10::optional<at::Tensor> &dht, at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices)
{
    TORCH_CHECK(g.has_value(), "chunk_gated_delta_rule_bwd_dhu requires g to be provided.");
    TORCH_CHECK(!gK.has_value(), "gK is not supported and must be None.");
    TORCH_CHECK(!h0.has_value(), "h0 is not supported in fast kernel launch and must be None.");
    TORCH_CHECK(!dht.has_value(), "dht is not supported in fast kernel launch and must be None.");

    const c10::OptionalDeviceGuard guard(q.device());
    const at::Tensor &gTensor = g.value();
    CheckInputs(q, k, w, d_o, dv, gTensor, cu_seqlens, chunk_indices);

    auto outputs = chunk_gated_delta_rule_bwd_dhu_meta(q, k, w, d_o, dv, scale, chunk_size, g, gK, h0, dht,
                                                       cu_seqlens, chunk_indices);
    auto dh = std::get<0>(outputs);
    auto dh0 = std::get<1>(outputs);
    auto dv2 = std::get<2>(outputs);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto tilingResult = CalcTilingParams(q, k, w, d_o, dv, gTensor, scale, chunk_size, cu_seqlens, chunk_indices);

    GM_ADDR qPtr = (GM_ADDR)q.data_ptr();
    GM_ADDR kPtr = (GM_ADDR)k.data_ptr();
    GM_ADDR wPtr = (GM_ADDR)w.data_ptr();
    GM_ADDR doPtr = (GM_ADDR)d_o.data_ptr();
    GM_ADDR dvPtr = (GM_ADDR)dv.data_ptr();
    GM_ADDR gPtr = (GM_ADDR)gTensor.data_ptr();
    GM_ADDR dhPtr = (GM_ADDR)dh.data_ptr();
    GM_ADDR dh0Ptr = dh0.numel() > 0 ? (GM_ADDR)dh0.data_ptr() : nullptr;
    GM_ADDR dv2Ptr = (GM_ADDR)dv2.data_ptr();

    GM_ADDR cuSeqlensPtr = nullptr;
    GM_ADDR chunkIndicesPtr = nullptr;
    std::vector<int64_t> cuSeqlensVec;
    std::vector<int64_t> chunkIndicesVec;
    at::Tensor cuSeqlensTensor;
    at::Tensor chunkIndicesTensor;

    if (cu_seqlens.has_value()) {
        cuSeqlensVec = std::vector<int64_t>(cu_seqlens.value().begin(), cu_seqlens.value().end());
        cuSeqlensTensor = at::tensor(cuSeqlensVec, at::dtype(at::kLong).device(q.device()));
        cuSeqlensPtr = (GM_ADDR)cuSeqlensTensor.data_ptr();
    }
    if (chunk_indices.has_value()) {
        chunkIndicesVec = std::vector<int64_t>(chunk_indices.value().begin(), chunk_indices.value().end());
        chunkIndicesTensor = at::tensor(chunkIndicesVec, at::dtype(at::kLong).device(q.device()));
        chunkIndicesPtr = (GM_ADDR)chunkIndicesTensor.data_ptr();
    }

    void *workspacePtr = nullptr;
    if (tilingResult.workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspacePtr, tilingResult.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: ", ret);
    }
    GM_ADDR workspaceGm = (GM_ADDR)workspacePtr;

    auto qDtype = q.scalar_type();
    auto gDtype = gTensor.scalar_type();
    auto tiling = tilingResult.tiling;
    auto blockDim = tilingResult.blockDim;

    auto aclCall = [=]() -> int {
        if (qDtype == at::kBFloat16 && gDtype == at::kBFloat16) {
            LaunchChunkGatedDeltaRuleBwdDhu<bfloat16_t, bfloat16_t>(
                blockDim, stream, qPtr, kPtr, wPtr, doPtr, dvPtr, gPtr, cuSeqlensPtr, chunkIndicesPtr, dhPtr, dh0Ptr,
                dv2Ptr, workspaceGm, tiling);
        } else if (qDtype == at::kHalf && gDtype == at::kHalf) {
            LaunchChunkGatedDeltaRuleBwdDhu<half, half>(
                blockDim, stream, qPtr, kPtr, wPtr, doPtr, dvPtr, gPtr, cuSeqlensPtr, chunkIndicesPtr, dhPtr, dh0Ptr,
                dv2Ptr, workspaceGm, tiling);
        } else if (qDtype == at::kBFloat16 && gDtype == at::kFloat) {
            LaunchChunkGatedDeltaRuleBwdDhu<bfloat16_t, float>(
                blockDim, stream, qPtr, kPtr, wPtr, doPtr, dvPtr, gPtr, cuSeqlensPtr, chunkIndicesPtr, dhPtr, dh0Ptr,
                dv2Ptr, workspaceGm, tiling);
        } else if (qDtype == at::kHalf && gDtype == at::kFloat) {
            LaunchChunkGatedDeltaRuleBwdDhu<half, float>(
                blockDim, stream, qPtr, kPtr, wPtr, doPtr, dvPtr, gPtr, cuSeqlensPtr, chunkIndicesPtr, dhPtr, dh0Ptr,
                dv2Ptr, workspaceGm, tiling);
        } else {
            TORCH_CHECK(false, "Unsupported dtype combination: q=", qDtype, ", g=", gDtype);
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("ChunkGatedDeltaRuleBwdDhu", aclCall);

    if (workspacePtr != nullptr) {
        aclrtFree(workspacePtr);
    }

    return std::make_tuple(dh, dh0, dv2);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("chunk_gated_delta_rule_bwd_dhu", chunk_gated_delta_rule_bwd_dhu_npu);
}

} // namespace ChunkGatedDeltaRuleBwdDhu
} // namespace ascend_ops
