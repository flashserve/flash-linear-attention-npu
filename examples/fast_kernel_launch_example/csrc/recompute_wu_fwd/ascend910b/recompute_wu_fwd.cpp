/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd.cpp
 * \brief Recompute w/u forward operator with fast kernel launch.
 */

#include <algorithm>
#include <tuple>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd/op_host/op_tiling/recompute_wu_fwd_tiling_processor.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd/op_kernel/recompute_wu_fwd.cpp"

namespace ascend_ops {
namespace RecomputeWUFwd {

using TilingData = GDN::RecomputeWUFwdTilingData;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("recompute_wu_fwd(Tensor k, Tensor v, Tensor beta, Tensor A, int chunk_size, *, Tensor? g=None, Tensor? gk=None, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> (Tensor, Tensor)");
}

std::tuple<at::Tensor, at::Tensor> recompute_wu_fwd_meta(
    const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta, const at::Tensor &A, int64_t chunk_size,
    const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &gk, at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices)
{
    (void)beta;
    (void)A;
    (void)chunk_size;
    (void)g;
    (void)gk;
    (void)cu_seqlens;
    (void)chunk_indices;
    auto wShape = v.sizes().vec();
    wShape[3] = k.size(3);
    at::Tensor w = at::empty(wShape, v.options().dtype(k.scalar_type()));
    at::Tensor u = at::empty_like(v);
    return std::make_tuple(w, u);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("recompute_wu_fwd", recompute_wu_fwd_meta);
}

struct RecomputeWUFwdTilingResult {
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
    TORCH_CHECK(tensor.device() == base.device(), name, " should be on the same device as k.");
}

void CheckInputs(const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta, const at::Tensor &A,
                 const at::Tensor &g, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    CheckInputRank(k, 4, "k");
    CheckInputRank(v, 4, "v");
    CheckInputRank(beta, 3, "beta");
    CheckInputRank(A, 4, "A");
    CheckInputRank(g, 3, "g");

    CheckSameDevice(k, v, "v");
    CheckSameDevice(k, beta, "beta");
    CheckSameDevice(k, A, "A");
    CheckSameDevice(k, g, "g");

    auto kDtype = k.scalar_type();
    auto betaDtype = beta.scalar_type();
    TORCH_CHECK(kDtype == at::kHalf || kDtype == at::kBFloat16,
              "k/v/A only support float16 or bfloat16, but got ", kDtype);
    TORCH_CHECK(v.scalar_type() == kDtype && A.scalar_type() == kDtype,
              "v/A should have the same dtype as k.");
    TORCH_CHECK(betaDtype == kDtype || betaDtype == at::kFloat,
              "beta/g should have the same dtype as k or float32, but beta is ", betaDtype, " and k is ", kDtype);
    TORCH_CHECK(g.scalar_type() == betaDtype, "g should have the same dtype as beta.");
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

RecomputeWUFwdTilingResult CalcTilingParams(
    const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta, const at::Tensor &A, const at::Tensor &g,
    int64_t chunk_size, at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    auto kSizes = k.sizes();
    auto vSizes = v.sizes();
    auto betaSizes = beta.sizes();
    auto aSizes = A.sizes();
    auto gSizes = g.sizes();

    gert::StorageShape kShape({kSizes[0], kSizes[1], kSizes[2], kSizes[3]},
                              {kSizes[0], kSizes[1], kSizes[2], kSizes[3]});
    gert::StorageShape vShape({vSizes[0], vSizes[1], vSizes[2], vSizes[3]},
                              {vSizes[0], vSizes[1], vSizes[2], vSizes[3]});
    gert::StorageShape betaShape({betaSizes[0], betaSizes[1], betaSizes[2]},
                                 {betaSizes[0], betaSizes[1], betaSizes[2]});
    gert::StorageShape aShape({aSizes[0], aSizes[1], aSizes[2], aSizes[3]},
                              {aSizes[0], aSizes[1], aSizes[2], aSizes[3]});
    gert::StorageShape gShape({gSizes[0], gSizes[1], gSizes[2]}, {gSizes[0], gSizes[1], gSizes[2]});

    gert::StorageShape *cuSeqlensShapePtr = nullptr;
    gert::StorageShape *chunkIndicesShapePtr = nullptr;
    gert::StorageShape cuSeqlensShape;
    gert::StorageShape chunkIndicesShape;

    std::vector<int64_t> cuSeqlensVec;
    std::vector<int64_t> chunkIndicesVec;
    const int64_t *cuSeqlensData = nullptr;
    const int64_t *chunkIndicesData = nullptr;

    if (cu_seqlens.has_value()) {
        cuSeqlensVec = std::vector<int64_t>(cu_seqlens.value().begin(), cu_seqlens.value().end());
        int64_t cuDim0 = static_cast<int64_t>(cuSeqlensVec.size());
        cuSeqlensShape = gert::StorageShape({cuDim0}, {cuDim0});
        cuSeqlensShapePtr = &cuSeqlensShape;
        cuSeqlensData = cuSeqlensVec.data();
    }
    if (chunk_indices.has_value()) {
        chunkIndicesVec = std::vector<int64_t>(chunk_indices.value().begin(), chunk_indices.value().end());
        int64_t ciDim0 = static_cast<int64_t>(chunkIndicesVec.size());
        chunkIndicesShape = gert::StorageShape({ciDim0}, {ciDim0});
        chunkIndicesShapePtr = &chunkIndicesShape;
        chunkIndicesData = chunkIndicesVec.data();
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendcPlatform != nullptr, "PlatformAscendCManager is null.");
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform->GetCoreNumAic());
    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    TilingData tiling{};
    optiling::RecomputeWUFwdTilingContext ctx{
        "recompute_wu_fwd",
        &kShape,
        &vShape,
        &betaShape,
        &aShape,
        &gShape,
        cuSeqlensShapePtr,
        chunkIndicesShapePtr,
        cuSeqlensData,
        chunkIndicesData,
        static_cast<int32_t>(chunk_size),
        ToGeDtype(k.scalar_type()),
        ToGeDtype(beta.scalar_type()),
        ubSize,
        sysWorkspaceSize,
    };

    optiling::RecomputeWUFwdTilingProcessor processor(ctx, tiling);
    TORCH_CHECK(processor.Process() == ge::GRAPH_SUCCESS, "recompute_wu_fwd tiling failed.");
    return RecomputeWUFwdTilingResult{tiling, coreNum, processor.GetWorkspaceSize()};
}

template <typename KType, typename BetaType, int VDim>
__global__ __aicore__ void recompute_wu_fwd_kernel(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g,
                                                   GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR u,
                                                   GM_ADDR workspace, const GDN::RecomputeWUFwdTilingData tilingData)
{
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    if (tilingData.V == optiling::RECOMPUTE_WU_FWD_V_DIM_256) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    }
    AscendC::AscendCUtils::SetOverflow(1);
    GDN::RecomputeWUFwdKernelImpl<KType, BetaType, VDim>(
        k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, user, &tilingData);
}

template <typename KType, typename BetaType, int VDim>
void LaunchRecomputeWUFwd(uint32_t blockDim, aclrtStream stream, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A,
                          GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace,
                          const GDN::RecomputeWUFwdTilingData &tiling)
{
    recompute_wu_fwd_kernel<KType, BetaType, VDim><<<blockDim, nullptr, stream>>>(
        k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, workspace, tiling);
}

template <typename KType, typename BetaType>
void DispatchByV(uint32_t blockDim, aclrtStream stream, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g,
                 GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace,
                 const GDN::RecomputeWUFwdTilingData &tiling)
{
    if (tiling.V == 256) {
        LaunchRecomputeWUFwd<KType, BetaType, 256>(blockDim, stream, k, v, beta, A, g, cu_seqlens, chunk_indices, w, u,
                                                   workspace, tiling);
    } else {
        LaunchRecomputeWUFwd<KType, BetaType, 128>(blockDim, stream, k, v, beta, A, g, cu_seqlens, chunk_indices, w, u,
                                                 workspace, tiling);
    }
}

std::tuple<at::Tensor, at::Tensor> recompute_wu_fwd_npu(
    const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta, const at::Tensor &A, int64_t chunk_size,
    const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &gk, at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices)
{
    TORCH_CHECK(g.has_value(), "recompute_wu_fwd requires g to be provided.");
    TORCH_CHECK(!gk.has_value(), "gk is not supported and must be None.");

    const c10::OptionalDeviceGuard guard(k.device());
    const at::Tensor &gTensor = g.value();
    CheckInputs(k, v, beta, A, gTensor, cu_seqlens, chunk_indices);
    auto outputs = recompute_wu_fwd_meta(k, v, beta, A, chunk_size, g, gk, cu_seqlens, chunk_indices);
    auto w = std::get<0>(outputs);
    auto u = std::get<1>(outputs);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    RecomputeWUFwdTilingResult tilingResult = CalcTilingParams(k, v, beta, A, gTensor, chunk_size, cu_seqlens, chunk_indices);

    GM_ADDR kPtr = (GM_ADDR)k.data_ptr();
    GM_ADDR vPtr = (GM_ADDR)v.data_ptr();
    GM_ADDR betaPtr = (GM_ADDR)beta.data_ptr();
    GM_ADDR aPtr = (GM_ADDR)A.data_ptr();
    GM_ADDR gPtr = (GM_ADDR)gTensor.data_ptr();
    GM_ADDR wPtr = (GM_ADDR)w.data_ptr();
    GM_ADDR uPtr = (GM_ADDR)u.data_ptr();

    GM_ADDR cuSeqlensPtr = nullptr;
    GM_ADDR chunkIndicesPtr = nullptr;
    std::vector<int64_t> cuSeqlensVec;
    std::vector<int64_t> chunkIndicesVec;
    at::Tensor cuSeqlensTensor;
    at::Tensor chunkIndicesTensor;

    if (cu_seqlens.has_value()) {
        cuSeqlensVec = std::vector<int64_t>(cu_seqlens.value().begin(), cu_seqlens.value().end());
        cuSeqlensTensor = at::tensor(cuSeqlensVec, at::dtype(at::kLong).device(k.device()));
        cuSeqlensPtr = (GM_ADDR)cuSeqlensTensor.data_ptr();
    }
    if (chunk_indices.has_value()) {
        chunkIndicesVec = std::vector<int64_t>(chunk_indices.value().begin(), chunk_indices.value().end());
        chunkIndicesTensor = at::tensor(chunkIndicesVec, at::dtype(at::kLong).device(k.device()));
        chunkIndicesPtr = (GM_ADDR)chunkIndicesTensor.data_ptr();
    }

    void *workspacePtr = nullptr;
    if (tilingResult.workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspacePtr, tilingResult.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: ", ret);
    }
    GM_ADDR workspaceGm = (GM_ADDR)workspacePtr;

    auto kDtype = k.scalar_type();
    auto betaDtype = beta.scalar_type();
    auto tiling = tilingResult.tiling;
    auto blockDim = tilingResult.blockDim;

    auto aclCall = [=]() -> int {
        if (kDtype == at::kBFloat16 && betaDtype == at::kBFloat16) {
            DispatchByV<bfloat16_t, bfloat16_t>(blockDim, stream, kPtr, vPtr, betaPtr, aPtr, gPtr, cuSeqlensPtr,
                                                chunkIndicesPtr, wPtr, uPtr, workspaceGm, tiling);
        } else if (kDtype == at::kHalf && betaDtype == at::kHalf) {
            DispatchByV<half, half>(blockDim, stream, kPtr, vPtr, betaPtr, aPtr, gPtr, cuSeqlensPtr, chunkIndicesPtr,
                                    wPtr, uPtr, workspaceGm, tiling);
        } else if (kDtype == at::kBFloat16 && betaDtype == at::kFloat) {
            DispatchByV<bfloat16_t, float>(blockDim, stream, kPtr, vPtr, betaPtr, aPtr, gPtr, cuSeqlensPtr,
                                           chunkIndicesPtr, wPtr, uPtr, workspaceGm, tiling);
        } else if (kDtype == at::kHalf && betaDtype == at::kFloat) {
            DispatchByV<half, float>(blockDim, stream, kPtr, vPtr, betaPtr, aPtr, gPtr, cuSeqlensPtr, chunkIndicesPtr,
                                     wPtr, uPtr, workspaceGm, tiling);
        } else {
            TORCH_CHECK(false, "Unsupported dtype combination: k=", kDtype, ", beta=", betaDtype);
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("RecomputeWUFwd", aclCall);
    c10_npu::getCurrentNPUStream().synchronize();

    if (workspacePtr != nullptr) {
        aclrtFree(workspacePtr);
    }

    return std::make_tuple(w, u);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("recompute_wu_fwd", recompute_wu_fwd_npu);
}

} // namespace RecomputeWUFwd
} // namespace ascend_ops
