/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_full.cpp
 * \brief Prepare WY representation backward full operator with fast kernel launch.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/op_host/prepare_wy_repr_bwd_full_tiling_processor.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/op_kernel/prepare_wy_repr_bwd_full.cpp"

namespace ascend_ops {
namespace PrepareWyReprBwdFull {

using TilingData = GDN::PrepareWyReprBwdFullTilingData;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("prepare_wy_repr_bwd_full(Tensor k, Tensor v, Tensor beta, Tensor A, Tensor dA, Tensor dw, Tensor du, Tensor g, int chunk_size, *, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> (Tensor, Tensor, Tensor, Tensor)");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> prepare_wy_repr_bwd_full_meta(
    const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta, const at::Tensor &A, const at::Tensor &dA,
    const at::Tensor &dw, const at::Tensor &du, const at::Tensor &g, int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    (void)A;
    (void)dA;
    (void)dw;
    (void)du;
    (void)chunk_size;
    (void)cu_seqlens;
    (void)chunk_indices;
    return std::make_tuple(at::empty_like(k), at::empty_like(v), at::empty_like(beta), at::empty_like(g));
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("prepare_wy_repr_bwd_full", prepare_wy_repr_bwd_full_meta);
}

struct PrepareWyReprBwdFullTilingResult {
    TilingData tiling;
    uint32_t blockDim;
    size_t workspaceSize;
};

int64_t ToPrepareWyReprBwdFullDtype(at::ScalarType dtype)
{
    if (dtype == at::kBFloat16) {
        return optiling::PREPARE_WY_REPR_BWD_FULL_DTYPE_BF16;
    }
    if (dtype == at::kFloat) {
        return optiling::PREPARE_WY_REPR_BWD_FULL_DTYPE_FP32;
    }
    if (dtype == at::kHalf) {
        return optiling::PREPARE_WY_REPR_BWD_FULL_DTYPE_FP16;
    }
    TORCH_CHECK(false, "Unsupported dtype: ", dtype);
    return optiling::PREPARE_WY_REPR_BWD_FULL_DTYPE_FP16;
}

void CheckInputRank(const at::Tensor &tensor, int64_t rank, const char *name)
{
    TORCH_CHECK(tensor.dim() == rank, name, " should be ", rank, "D, but got ", tensor.dim(), "D.");
}

void CheckSameDevice(const at::Tensor &base, const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device() == base.device(), name, " should be on the same device as k.");
}

void CheckInputs(const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta, const at::Tensor &A,
                 const at::Tensor &dA, const at::Tensor &dw, const at::Tensor &du, const at::Tensor &g,
                 at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    CheckInputRank(k, 4, "k");
    CheckInputRank(v, 4, "v");
    CheckInputRank(beta, 3, "beta");
    CheckInputRank(A, 4, "A");
    CheckInputRank(dA, 4, "dA");
    CheckInputRank(dw, 4, "dw");
    CheckInputRank(du, 4, "du");
    CheckInputRank(g, 3, "g");

    CheckSameDevice(k, v, "v");
    CheckSameDevice(k, beta, "beta");
    CheckSameDevice(k, A, "A");
    CheckSameDevice(k, dA, "dA");
    CheckSameDevice(k, dw, "dw");
    CheckSameDevice(k, du, "du");
    CheckSameDevice(k, g, "g");

    auto kDtype = k.scalar_type();
    auto betaDtype = beta.scalar_type();
    TORCH_CHECK(kDtype == at::kHalf || kDtype == at::kBFloat16,
                "k/v/A/dA/dw/du only support float16 or bfloat16, but got ", kDtype);
    TORCH_CHECK(v.scalar_type() == kDtype && A.scalar_type() == kDtype && dA.scalar_type() == kDtype &&
                    dw.scalar_type() == kDtype && du.scalar_type() == kDtype,
                "v/A/dA/dw/du should have the same dtype as k.");
    TORCH_CHECK(betaDtype == kDtype || betaDtype == at::kFloat,
                "beta/g should have the same dtype as k or float32, but beta is ", betaDtype, " and k is ", kDtype);
    TORCH_CHECK(g.scalar_type() == betaDtype, "g should have the same dtype as beta.");
    TORCH_CHECK(cu_seqlens.has_value() == chunk_indices.has_value(),
                "cu_seqlens and chunk_indices should be both provided or both None.");
}

PrepareWyReprBwdFullTilingResult CalcTilingParams(
    const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta, const at::Tensor &A, const at::Tensor &dA,
    const at::Tensor &dw, const at::Tensor &du, const at::Tensor &g, int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    auto kSizes = k.sizes();
    auto vSizes = v.sizes();
    auto betaSizes = beta.sizes();
    auto aSizes = A.sizes();
    auto dASizes = dA.sizes();
    auto dwSizes = dw.sizes();
    auto duSizes = du.sizes();
    auto gSizes = g.sizes();

    gert::StorageShape kShape({kSizes[0], kSizes[1], kSizes[2], kSizes[3]},
                              {kSizes[0], kSizes[1], kSizes[2], kSizes[3]});
    gert::StorageShape vShape({vSizes[0], vSizes[1], vSizes[2], vSizes[3]},
                              {vSizes[0], vSizes[1], vSizes[2], vSizes[3]});
    gert::StorageShape betaShape({betaSizes[0], betaSizes[1], betaSizes[2]},
                                 {betaSizes[0], betaSizes[1], betaSizes[2]});
    gert::StorageShape aShape({aSizes[0], aSizes[1], aSizes[2], aSizes[3]},
                              {aSizes[0], aSizes[1], aSizes[2], aSizes[3]});
    gert::StorageShape dAShape({dASizes[0], dASizes[1], dASizes[2], dASizes[3]},
                               {dASizes[0], dASizes[1], dASizes[2], dASizes[3]});
    gert::StorageShape dwShape({dwSizes[0], dwSizes[1], dwSizes[2], dwSizes[3]},
                               {dwSizes[0], dwSizes[1], dwSizes[2], dwSizes[3]});
    gert::StorageShape duShape({duSizes[0], duSizes[1], duSizes[2], duSizes[3]},
                               {duSizes[0], duSizes[1], duSizes[2], duSizes[3]});
    gert::StorageShape gShape({gSizes[0], gSizes[1], gSizes[2]}, {gSizes[0], gSizes[1], gSizes[2]});

    gert::StorageShape *cuSeqlensShapePtr = nullptr;
    gert::StorageShape *chunkIndicesShapePtr = nullptr;
    gert::StorageShape cuSeqlensShape;
    gert::StorageShape chunkIndicesShape;
    std::vector<int64_t> cuSeqlensHost;
    std::vector<int64_t> chunkIndicesHost;

    if (cu_seqlens.has_value()) {
        cuSeqlensHost = std::vector<int64_t>(cu_seqlens.value().begin(), cu_seqlens.value().end());
        int64_t dim0 = static_cast<int64_t>(cuSeqlensHost.size());
        cuSeqlensShape = gert::StorageShape({dim0}, {dim0});
        cuSeqlensShapePtr = &cuSeqlensShape;
    }
    if (chunk_indices.has_value()) {
        chunkIndicesHost = std::vector<int64_t>(chunk_indices.value().begin(), chunk_indices.value().end());
        int64_t dim0 = static_cast<int64_t>(chunkIndicesHost.size());
        chunkIndicesShape = gert::StorageShape({dim0}, {dim0});
        chunkIndicesShapePtr = &chunkIndicesShape;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendcPlatform != nullptr, "PlatformAscendCManager is null.");
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform->GetCoreNumAic());
    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());

    optiling::PrepareWyReprBwdFullTilingContext ctx{
        "prepare_wy_repr_bwd_full",
        &kShape,
        &vShape,
        &betaShape,
        &aShape,
        &dAShape,
        &dwShape,
        &duShape,
        &gShape,
        cuSeqlensShapePtr,
        chunkIndicesShapePtr,
        cuSeqlensHost.empty() ? nullptr : cuSeqlensHost.data(),
        chunkIndicesHost.empty() ? nullptr : chunkIndicesHost.data(),
        chunk_size,
        ToPrepareWyReprBwdFullDtype(k.scalar_type()),
        ToPrepareWyReprBwdFullDtype(beta.scalar_type()),
        ubSize,
        coreNum,
        sysWorkspaceSize,
    };

    TilingData tiling{};
    optiling::PrepareWyReprBwdFullTilingProcessor processor(ctx, tiling);
    TORCH_CHECK(processor.Process() == ge::GRAPH_SUCCESS, "prepare_wy_repr_bwd_full tiling failed.");
    return PrepareWyReprBwdFullTilingResult{tiling, processor.GetBlockDim(), processor.GetWorkspaceSize()};
}

template <typename KType, typename BetaType>
__global__ __aicore__ void prepare_wy_repr_bwd_full_kernel(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dA, GM_ADDR dw, GM_ADDR du, GM_ADDR g,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg,
    GM_ADDR workspace, const GDN::PrepareWyReprBwdFullTilingData tilingData)
{
    AscendC::AscendCUtils::SetOverflow(1);
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    if (tilingData.V == optiling::V_DIM_256) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    }
    GDN::PrepareWyReprBwdFullKernelImpl<KType, BetaType>(
        k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg, userWorkspace, &tilingData);
}

template <typename KType, typename BetaType>
void LaunchPrepareWyReprBwdFull(uint32_t blockDim, aclrtStream stream, GM_ADDR k, GM_ADDR v, GM_ADDR beta,
                                GM_ADDR A, GM_ADDR dA, GM_ADDR dw, GM_ADDR du, GM_ADDR g, GM_ADDR cu_seqlens,
                                GM_ADDR chunk_indices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg,
                                GM_ADDR workspace, const GDN::PrepareWyReprBwdFullTilingData &tiling)
{
    prepare_wy_repr_bwd_full_kernel<KType, BetaType><<<blockDim, nullptr, stream>>>(
        k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg, workspace, tiling);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> prepare_wy_repr_bwd_full_npu(
    const at::Tensor &k, const at::Tensor &v, const at::Tensor &beta, const at::Tensor &A, const at::Tensor &dA,
    const at::Tensor &dw, const at::Tensor &du, const at::Tensor &g, int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    const c10::OptionalDeviceGuard guard(k.device());
    CheckInputs(k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices);

    auto outputs = prepare_wy_repr_bwd_full_meta(k, v, beta, A, dA, dw, du, g, chunk_size, cu_seqlens, chunk_indices);
    auto dk = std::get<0>(outputs);
    auto dv = std::get<1>(outputs);
    auto dbeta = std::get<2>(outputs);
    auto dg = std::get<3>(outputs);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    PrepareWyReprBwdFullTilingResult tilingResult =
        CalcTilingParams(k, v, beta, A, dA, dw, du, g, chunk_size, cu_seqlens, chunk_indices);

    GM_ADDR kPtr = (GM_ADDR)k.data_ptr();
    GM_ADDR vPtr = (GM_ADDR)v.data_ptr();
    GM_ADDR betaPtr = (GM_ADDR)beta.data_ptr();
    GM_ADDR APtr = (GM_ADDR)A.data_ptr();
    GM_ADDR dAPtr = (GM_ADDR)dA.data_ptr();
    GM_ADDR dwPtr = (GM_ADDR)dw.data_ptr();
    GM_ADDR duPtr = (GM_ADDR)du.data_ptr();
    GM_ADDR gPtr = (GM_ADDR)g.data_ptr();
    GM_ADDR dkPtr = (GM_ADDR)dk.data_ptr();
    GM_ADDR dvPtr = (GM_ADDR)dv.data_ptr();
    GM_ADDR dbetaPtr = (GM_ADDR)dbeta.data_ptr();
    GM_ADDR dgPtr = (GM_ADDR)dg.data_ptr();

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
            LaunchPrepareWyReprBwdFull<bfloat16_t, bfloat16_t>(
                blockDim, stream, kPtr, vPtr, betaPtr, APtr, dAPtr, dwPtr, duPtr, gPtr, cuSeqlensPtr,
                chunkIndicesPtr, dkPtr, dvPtr, dbetaPtr, dgPtr, workspaceGm, tiling);
        } else if (kDtype == at::kHalf && betaDtype == at::kHalf) {
            LaunchPrepareWyReprBwdFull<half, half>(
                blockDim, stream, kPtr, vPtr, betaPtr, APtr, dAPtr, dwPtr, duPtr, gPtr, cuSeqlensPtr,
                chunkIndicesPtr, dkPtr, dvPtr, dbetaPtr, dgPtr, workspaceGm, tiling);
        } else if (kDtype == at::kBFloat16 && betaDtype == at::kFloat) {
            LaunchPrepareWyReprBwdFull<bfloat16_t, float>(
                blockDim, stream, kPtr, vPtr, betaPtr, APtr, dAPtr, dwPtr, duPtr, gPtr, cuSeqlensPtr,
                chunkIndicesPtr, dkPtr, dvPtr, dbetaPtr, dgPtr, workspaceGm, tiling);
        } else if (kDtype == at::kHalf && betaDtype == at::kFloat) {
            LaunchPrepareWyReprBwdFull<half, float>(
                blockDim, stream, kPtr, vPtr, betaPtr, APtr, dAPtr, dwPtr, duPtr, gPtr, cuSeqlensPtr,
                chunkIndicesPtr, dkPtr, dvPtr, dbetaPtr, dgPtr, workspaceGm, tiling);
        } else {
            TORCH_CHECK(false, "Unsupported dtype combination: k=", kDtype, ", beta=", betaDtype);
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("PrepareWyReprBwdFull", aclCall);
    auto syncRet = aclrtSynchronizeStream(stream);
    TORCH_CHECK(syncRet == ACL_SUCCESS, "aclrtSynchronizeStream failed. ERROR: ", syncRet);

    if (workspacePtr != nullptr) {
        aclrtFree(workspacePtr);
    }

    return std::make_tuple(std::move(dk), std::move(dv), std::move(dbeta), std::move(dg));
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("prepare_wy_repr_bwd_full", prepare_wy_repr_bwd_full_npu);
}

} // namespace PrepareWyReprBwdFull
} // namespace ascend_ops
