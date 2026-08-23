/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#include "lib/matmul_intf.h"

using namespace Catlass;

namespace ascend_ops {
namespace ChunkGatedDeltaRuleFwdHStage0Debug {

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) : size_(size)
    {
        if (size_ == 0) {
            return;
        }
        auto ret = aclrtMalloc(&ptr_, size_, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "stage0 debug device allocation failed. ERROR: ", ret);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) {
            aclrtFree(ptr_);
        }
    }

    void *Get() const
    {
        return ptr_;
    }

private:
    void *ptr_ = nullptr;
    size_t size_ = 0;
};

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "chunk_gated_delta_rule_fwd_h_stage0_debug(Tensor w, Tensor h_entry, *, int chunk_size=64, "
        "int[]? cu_seqlens=None) -> Tensor");
}

static int64_t DtypeToEnum(at::ScalarType dtype)
{
    if (dtype == at::kBFloat16) {
        return optiling::GDN_FWD_H_DTYPE_BF16;
    }
    if (dtype == at::kHalf) {
        return optiling::GDN_FWD_H_DTYPE_FP16;
    }
    return optiling::GDN_FWD_H_DTYPE_FP32;
}

static int64_t ValidateChunkLengths(
    int64_t totalTokens, int64_t chunkSize, at::OptionalIntArrayRef cuSeqlens)
{
    int64_t totalChunks = 0;
    auto validateLength = [&](int64_t length) {
        TORCH_CHECK(length > 0, "stage0 debug does not accept empty sequences");
        int64_t tail = length % chunkSize;
        TORCH_CHECK(tail == 0 || tail >= 16,
                    "stage0 debug only covers the Cube path; every tail must contain at least 16 tokens");
        totalChunks += (length + chunkSize - 1) / chunkSize;
    };

    if (!cuSeqlens.has_value()) {
        validateLength(totalTokens);
        return totalChunks;
    }

    auto boundaries = cuSeqlens.value();
    TORCH_CHECK(boundaries.size() >= 2, "cu_seqlens must contain at least [0, T]");
    TORCH_CHECK(boundaries.front() == 0 && boundaries.back() == totalTokens,
                "cu_seqlens must start at 0 and end at T");
    for (size_t i = 1; i < boundaries.size(); ++i) {
        TORCH_CHECK(boundaries[i] > boundaries[i - 1], "cu_seqlens must be strictly increasing");
        validateLength(boundaries[i] - boundaries[i - 1]);
    }
    return totalChunks;
}

static at::Tensor Stage0DebugMeta(
    const at::Tensor &w, const at::Tensor &hEntry, int64_t chunkSize,
    at::OptionalIntArrayRef cuSeqlens)
{
    TORCH_CHECK(w.dim() == 4, "w must be [B, HV, T, K]");
    TORCH_CHECK(hEntry.dim() == 5, "h_entry must be [B, HV, NT, K, V]");
    TORCH_CHECK(chunkSize > 0, "chunk_size must be positive");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 || w.scalar_type() == at::kHalf,
                "w must use bfloat16 or float16");
    TORCH_CHECK(w.size(0) > 0 && w.size(1) > 0, "w batch and head dimensions must be positive");
    TORCH_CHECK(hEntry.scalar_type() == w.scalar_type(), "h_entry and w must have the same dtype");
    TORCH_CHECK(hEntry.size(0) == w.size(0), "h_entry and w batch dimensions must match");
    TORCH_CHECK(hEntry.size(1) == w.size(1), "h_entry and w head dimensions must match");
    TORCH_CHECK(hEntry.size(3) == w.size(3), "h_entry K dimension must match w");
    if (cuSeqlens.has_value()) {
        TORCH_CHECK(w.size(0) == 1, "varlen stage0 debug requires B == 1");
    }
    int64_t expectedChunks = ValidateChunkLengths(w.size(2), chunkSize, cuSeqlens);
    TORCH_CHECK(hEntry.size(2) == expectedChunks,
                "h_entry NT does not match the number of chunks");
    return at::empty(
        {w.size(0), w.size(1), w.size(2), hEntry.size(4)},
        w.options().dtype(at::kFloat));
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("chunk_gated_delta_rule_fwd_h_stage0_debug", Stage0DebugMeta);
}

static ::ChunkGatedDeltaRuleFwdHTilingData CalcStage0DebugTiling(
    const at::Tensor &w, const at::Tensor &hEntry, int64_t chunkSize,
    at::OptionalIntArrayRef cuSeqlens, uint32_t &blockDim, size_t &workspaceSize)
{
    optiling::ChunkGatedDeltaRuleFwdHTilingContext ctx{};
    ctx.seqlen = w.size(2);
    ctx.kNumHead = w.size(1);
    ctx.kHeadDim = w.size(3);
    ctx.vNumHead = w.size(1);
    ctx.vHeadDim = hEntry.size(4);
    ctx.shapeBatchDim = w.size(0);
    ctx.hasCuSeqlens = cuSeqlens.has_value();
    ctx.cuSeqlensDim0 = cuSeqlens.has_value() ? static_cast<int64_t>(cuSeqlens.value().size()) : 0;
    ctx.dataType = DtypeToEnum(w.scalar_type());
    ctx.gDataType = optiling::GDN_FWD_H_DTYPE_FP32;
    ctx.useInitialState = false;
    ctx.stateDataType = optiling::GDN_FWD_H_DTYPE_FP32;
    // Match the scalar-gated production specialization. Stage0 does not read g.
    ctx.useG = true;
    ctx.useGk = false;
    ctx.storeFinalState = false;
    ctx.chunkSize = chunkSize;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ctx.aicCoreNum = platform->GetCoreNumAic();
    ctx.libApiWorkSpaceSize = platform->GetLibApiWorkSpaceSize();

    ::ChunkGatedDeltaRuleFwdHTilingData tiling{};
    optiling::ChunkGatedDeltaRuleFwdHTilingProcessor processor(ctx);
    processor.Process(tiling, blockDim, workspaceSize);
    return tiling;
}

template <typename InputT>
__global__ __aicore__ void Stage0DebugKernel(
    GM_ADDR w, GM_ADDR hEntry, GM_ADDR cuSeqlens, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }

    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<InputT, float, float, float>;
    Kernel kernel;
    kernel.Init(
        w, w, nullptr, nullptr, nullptr, nullptr, cuSeqlens, nullptr,
        hEntry, nullptr, nullptr, tiling, user);
    kernel.ProcessStage0Only(output);
}

static at::Tensor Stage0DebugNpu(
    const at::Tensor &w, const at::Tensor &hEntry, int64_t chunkSize,
    at::OptionalIntArrayRef cuSeqlens)
{
    const c10::OptionalDeviceGuard guard(w.device());
    TORCH_CHECK(w.device() == hEntry.device(), "w and h_entry must be on the same device");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous BNSD [B, HV, T, K]");
    TORCH_CHECK(hEntry.is_contiguous(), "h_entry must be contiguous [B, HV, NT, K, V]");
    TORCH_CHECK(w.size(3) == 128 && hEntry.size(4) == 128,
                "stage0 debug currently supports K == V == 128");
    TORCH_CHECK(chunkSize == 64, "stage0 debug currently supports chunk_size == 64");

    at::Tensor output = Stage0DebugMeta(w, hEntry, chunkSize, cuSeqlens);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    auto tiling = CalcStage0DebugTiling(w, hEntry, chunkSize, cuSeqlens, blockDim, workspaceSize);

    std::vector<int64_t> cuSeqlensVector;
    at::Tensor cuSeqlensTensor;
    GM_ADDR cuSeqlensPtr = nullptr;
    if (cuSeqlens.has_value()) {
        cuSeqlensVector.assign(cuSeqlens.value().begin(), cuSeqlens.value().end());
        cuSeqlensTensor = at::tensor(cuSeqlensVector, at::dtype(at::kLong).device(w.device()));
        cuSeqlensPtr = (GM_ADDR)cuSeqlensTensor.data_ptr();
    }

    DeviceBuffer workspace(workspaceSize);
    auto ret = ACL_SUCCESS;
    if (workspaceSize > 0) {
        ret = aclrtMemsetAsync(workspace.Get(), workspaceSize, 0, workspaceSize, stream);
        TORCH_CHECK(ret == ACL_SUCCESS, "memset stage0 debug workspace failed. ERROR: ", ret);
    }

    const size_t tilingBytes = sizeof(::ChunkGatedDeltaRuleFwdHTilingData);
    DeviceBuffer tilingBuffer(tilingBytes);
    ret = aclrtMemcpy(
        tilingBuffer.Get(), tilingBytes, &tiling, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(ret == ACL_SUCCESS, "copy stage0 debug tiling failed. ERROR: ", ret);

    auto wPtr = (GM_ADDR)w.data_ptr();
    auto hPtr = (GM_ADDR)hEntry.data_ptr();
    auto outputPtr = (GM_ADDR)output.data_ptr();
    auto workspaceGm = (GM_ADDR)workspace.Get();
    auto tilingGm = (GM_ADDR)tilingBuffer.Get();
    auto dtype = w.scalar_type();
    auto aclCall = [=]() -> int {
        if (dtype == at::kBFloat16) {
            Stage0DebugKernel<bfloat16_t><<<blockDim, nullptr, stream>>>(
                wPtr, hPtr, cuSeqlensPtr, outputPtr, workspaceGm, tilingGm);
        } else if (dtype == at::kHalf) {
            Stage0DebugKernel<half><<<blockDim, nullptr, stream>>>(
                wPtr, hPtr, cuSeqlensPtr, outputPtr, workspaceGm, tilingGm);
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("ChunkGatedDeltaRuleFwdHStage0Debug", aclCall);
    c10_npu::getCurrentNPUStream().synchronize();
    return output;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("chunk_gated_delta_rule_fwd_h_stage0_debug", Stage0DebugNpu);
}

} // namespace ChunkGatedDeltaRuleFwdHStage0Debug
} // namespace ascend_ops
