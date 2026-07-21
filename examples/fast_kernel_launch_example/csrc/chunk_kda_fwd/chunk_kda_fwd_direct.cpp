/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif

#include "fla/ops/ascendc/common/kda/chunk_kda_fwd_kernel.hpp"

namespace ascend_ops::ChunkKdaFwdDirect {
namespace {

constexpr uint64_t WORKSPACE_ALIGN = 512;
constexpr uint64_t SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t SCORE_QUEUE_SLOTS = 2;
constexpr uint64_t SCORE_SCRATCH_PLANES = 3;

struct DirectKdaTilingData {
    int64_t batch;
    int64_t seqNum;
    int64_t qHeadNum;
    int64_t vHeadNum;
    int64_t seqlen;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t totalChunks;
    float scale;
    bool hasInitialState;
    bool outputFinalState;
    bool isVarLen;
    bool safeGate;
    int64_t prepareUsedCoreNum;
    int64_t prepareQgScaledOffset;
    int64_t prepareWSeedOffset;
    int64_t prepareUSeedOffset;
    int64_t prepareAqkFp32Offset;
    int64_t prepareAkkFp32Offset;
    int64_t prepareScratchOffset;
    int64_t postWuUsedCoreNum;
    int64_t postWuQgScaledOffset;
    int64_t postWuWSeedOffset;
    int64_t postWuUSeedOffset;
    int64_t postWuScratchOffset;
    int64_t fwdHBatch;
    int64_t fwdHSeqlen;
    int64_t fwdHKNumHead;
    int64_t fwdHVNumHead;
    int64_t fwdHKHeadDim;
    int64_t fwdHVHeadDim;
    int64_t fwdHChunkSize;
    bool fwdHUseInitialState;
    bool fwdHStoreFinalState;
    int64_t fwdHIsVariedLen;
    int64_t fwdHShapeBatch;
    int64_t fwdHTokenBatch;
    int64_t fwdHVWorkspaceOffset;
    int64_t fwdHVUpdateWorkspaceOffset;
    int64_t fwdHKDecayWorkspaceOffset;
    int64_t fwdHHWorkspaceOffset;
    int64_t fwdHNumSeqWorkspaceOffset;
    int64_t fwdHNumChunksWorkspaceOffset;
    int64_t fwdHWorkspaceBaseOffset;
    int64_t outputUsedCoreNum;
    int64_t outputQgScaledOffset;
    int64_t outputScratchOffset;
    bool outputSequenceMajor;
};

struct DirectOutputs {
    at::Tensor o;
    at::Tensor finalState;
    at::Tensor aqk;
    at::Tensor akk;
    at::Tensor w;
    at::Tensor u;
    at::Tensor qg;
    at::Tensor kg;
    at::Tensor vNew;
    at::Tensor h;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t size) : size_(size)
    {
        if (size_ == 0) {
            return;
        }
        auto ret = aclrtMalloc(&ptr_, size_, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "chunk_kda_fwd_direct: device allocation failed: ", ret);
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

    size_t Size() const
    {
        return size_;
    }

private:
    void *ptr_ = nullptr;
    size_t size_ = 0;
};

uint64_t AlignUp(uint64_t value)
{
    return (value + WORKSPACE_ALIGN - 1) / WORKSPACE_ALIGN * WORKSPACE_ALIGN;
}

template <bool SAFE_GATE, typename T>
__global__ __aicore__ void ChunkKdaPrepareDirectKernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR aqk, GM_ADDR akk, GM_ADDR qg, GM_ADDR workspace, DirectKdaTilingData tiling)
{
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    AscendC::TPipe pipe;
    RunChunkKdaPrepare<SAFE_GATE, T, float, float>(
        q, k, v, gk, beta, initialState, nullptr, nullptr, aqk, akk, qg,
        userWorkspace + tiling.prepareQgScaledOffset,
        userWorkspace + tiling.prepareWSeedOffset,
        userWorkspace + tiling.prepareUSeedOffset,
        userWorkspace, tiling, pipe);
}

template <typename T>
__global__ __aicore__ void ChunkKdaPostWuDirectKernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR kg, GM_ADDR vNew,
    GM_ADDR workspace, DirectKdaTilingData tiling)
{
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    AscendC::TPipe pipe;
    RunChunkKdaPostWu<T, float, float>(
        q, k, v, gk, beta, initialState, nullptr, nullptr,
        userWorkspace + tiling.postWuWSeedOffset, akk,
        userWorkspace + tiling.postWuUSeedOffset,
        w, u, kg, vNew, userWorkspace, tiling, pipe);
}

template <typename T, typename TileShapes>
__aicore__ inline void RunChunkKdaFwdHDirect(
    GM_ADDR kg, GM_ADDR w, GM_ADDR u, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR userWorkspace,
    const DirectKdaTilingData &tiling)
{
    ChunkGatedDeltaRuleFwdHTilingData stateTiling{};
    stateTiling.batch = tiling.fwdHBatch;
    stateTiling.seqlen = tiling.fwdHSeqlen;
    stateTiling.kNumHead = tiling.fwdHKNumHead;
    stateTiling.vNumHead = tiling.fwdHVNumHead;
    stateTiling.kHeadDim = tiling.fwdHKHeadDim;
    stateTiling.vHeadDim = tiling.fwdHVHeadDim;
    stateTiling.chunkSize = tiling.fwdHChunkSize;
    stateTiling.useInitialState = tiling.fwdHUseInitialState;
    stateTiling.storeFinalState = tiling.fwdHStoreFinalState;
    stateTiling.isVariedLen = tiling.fwdHIsVariedLen;
    stateTiling.shapeBatch = tiling.fwdHShapeBatch;
    stateTiling.tokenBatch = tiling.fwdHTokenBatch;
    stateTiling.vWorkspaceOffset = tiling.fwdHVWorkspaceOffset;
    stateTiling.vUpdateWorkspaceOffset = tiling.fwdHVUpdateWorkspaceOffset;
    stateTiling.kDecayWorkspaceOffset = tiling.fwdHKDecayWorkspaceOffset;
    stateTiling.hWorkspaceOffset = tiling.fwdHHWorkspaceOffset;
    stateTiling.numSeqWorkspaceOffset = tiling.fwdHNumSeqWorkspaceOffset;
    stateTiling.numChunksWorkspaceOffset = tiling.fwdHNumChunksWorkspaceOffset;
    stateTiling.useG = false;
    stateTiling.useGk = true;
    stateTiling.useExp2 = true;

    using FwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        T, float, float, float, TileShapes, true, false, true>;
    FwdHKernel stateOp;
    stateOp.InitFromData(kg, w, u, gk, gk, initialState, nullptr, nullptr,
                         h, vNew, finalState, stateTiling,
                         userWorkspace + tiling.fwdHWorkspaceBaseOffset);
    stateOp.Process();
}

template <typename T>
__global__ __aicore__ void ChunkKdaFwdHDirectKernel(
    GM_ADDR kg, GM_ADDR w, GM_ADDR u, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR workspace,
    DirectKdaTilingData tiling)
{
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    if (tiling.vHeadDim > 128) {
        RunChunkKdaFwdHDirect<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            kg, w, u, gk, initialState, h, vNew, finalState, userWorkspace, tiling);
    } else {
        RunChunkKdaFwdHDirect<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            kg, w, u, gk, initialState, h, vNew, finalState, userWorkspace, tiling);
    }
}

template <typename T>
__global__ __aicore__ void ChunkKdaOutputDirectKernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR aqk, GM_ADDR vNew, GM_ADDR h, GM_ADDR o, GM_ADDR workspace,
    DirectKdaTilingData tiling)
{
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    AscendC::TPipe pipe;
    RunChunkKdaOutput<T, float, float>(
        q, k, v, gk, beta, initialState, nullptr, nullptr,
        userWorkspace + tiling.outputQgScaledOffset, aqk, vNew, h, o,
        userWorkspace, tiling, pipe);
}

template <bool SAFE_GATE, typename T>
void LaunchStages(
    uint32_t blockDim, aclrtStream stream, const DirectKdaTilingData &tiling,
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &gk,
    const at::Tensor &beta, const c10::optional<at::Tensor> &initialState,
    DirectOutputs &outputs, const DeviceBuffer &workspace)
{
    auto ptr = [](const at::Tensor &tensor) -> GM_ADDR {
        return tensor.defined() ? (GM_ADDR)tensor.data_ptr() : nullptr;
    };
    GM_ADDR initialPtr = initialState.has_value() ?
        (GM_ADDR)initialState.value().data_ptr() : nullptr;
    ChunkKdaPrepareDirectKernel<SAFE_GATE, T><<<blockDim, nullptr, stream>>>(
        ptr(q), ptr(k), ptr(v), ptr(gk), ptr(beta), initialPtr,
        ptr(outputs.aqk), ptr(outputs.akk), ptr(outputs.qg),
        (GM_ADDR)workspace.Get(), tiling);
    ChunkKdaPostWuDirectKernel<T><<<blockDim, nullptr, stream>>>(
        ptr(q), ptr(k), ptr(v), ptr(gk), ptr(beta), initialPtr,
        ptr(outputs.akk), ptr(outputs.w), ptr(outputs.u), ptr(outputs.kg),
        ptr(outputs.vNew), (GM_ADDR)workspace.Get(), tiling);
    ChunkKdaFwdHDirectKernel<T><<<blockDim, nullptr, stream>>>(
        ptr(outputs.kg), ptr(outputs.w), ptr(outputs.u), ptr(gk), initialPtr,
        ptr(outputs.h), ptr(outputs.vNew), ptr(outputs.finalState),
        (GM_ADDR)workspace.Get(), tiling);
    ChunkKdaOutputDirectKernel<T><<<blockDim, nullptr, stream>>>(
        ptr(q), ptr(k), ptr(v), ptr(gk), ptr(beta), initialPtr,
        ptr(outputs.aqk), ptr(outputs.vNew), ptr(outputs.h), ptr(outputs.o),
        (GM_ADDR)workspace.Get(), tiling);
}

DirectOutputs MakeOutputs(
    const at::Tensor &q, const at::Tensor &v, int64_t chunkSize, bool outputFinalState)
{
    int64_t batch = q.size(0);
    int64_t hv = v.size(1);
    int64_t seqlen = q.size(2);
    int64_t kdim = q.size(3);
    int64_t vdim = v.size(3);
    int64_t totalChunks = (seqlen + chunkSize - 1) / chunkSize;
    auto fp32 = q.options().dtype(at::kFloat);
    return {
        at::empty({batch, hv, seqlen, vdim}, q.options()),
        outputFinalState ? at::empty({batch, hv, kdim, vdim}, fp32) : at::empty({1}, fp32),
        at::empty({batch, hv, seqlen, chunkSize}, q.options()),
        at::empty({batch, hv, seqlen, chunkSize}, q.options()),
        at::empty({batch, hv, seqlen, kdim}, q.options()),
        at::empty({batch, hv, seqlen, vdim}, q.options()),
        at::empty({batch, hv, seqlen, kdim}, q.options()),
        at::empty({batch, hv, seqlen, kdim}, q.options()),
        at::empty({batch, hv, seqlen, vdim}, q.options()),
        at::empty({batch, hv, totalChunks, kdim, vdim}, q.options()),
    };
}

void CheckInputs(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &gk,
    const at::Tensor &beta, const c10::optional<at::Tensor> &initialState, int64_t chunkSize)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1,
                "chunk_kda_fwd_direct: inputs must be NPU tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                    gk.is_contiguous() && beta.is_contiguous(),
                "chunk_kda_fwd_direct: BNSD inputs must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && gk.dim() == 4 && beta.dim() == 3,
                "chunk_kda_fwd_direct: only dense BNSD rank-4 inputs are supported by this route");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "chunk_kda_fwd_direct: q/k/v must be float16 or bfloat16");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "chunk_kda_fwd_direct: q/k/v dtype must match");
    TORCH_CHECK(gk.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat,
                "chunk_kda_fwd_direct: the direct route currently requires FP32 gk/beta");
    TORCH_CHECK(q.sizes() == k.sizes(), "chunk_kda_fwd_direct: q and k shapes must match");
    TORCH_CHECK(chunkSize == 64 || chunkSize == 128,
                "chunk_kda_fwd_direct: chunk_size must be 64 or 128");
    TORCH_CHECK(q.size(3) == 128 && (v.size(3) == 128 || v.size(3) == 256),
                "chunk_kda_fwd_direct: K must be 128 and V must be 128 or 256");
    TORCH_CHECK(v.size(0) == q.size(0) && v.size(2) == q.size(2) &&
                    gk.size(0) == q.size(0) && gk.size(1) == v.size(1) &&
                    gk.size(2) == q.size(2) && gk.size(3) == q.size(3) &&
                    beta.size(0) == q.size(0) && beta.size(1) == v.size(1) &&
                    beta.size(2) == q.size(2),
                "chunk_kda_fwd_direct: v/gk/beta shape mismatch");
    TORCH_CHECK(v.size(1) >= q.size(1) && v.size(1) % q.size(1) == 0,
                "chunk_kda_fwd_direct: H_v must be divisible by H_k");
    if (initialState.has_value()) {
        TORCH_CHECK(initialState.value().scalar_type() == at::kFloat && initialState.value().is_contiguous(),
                    "chunk_kda_fwd_direct: initial_state must be contiguous FP32");
        TORCH_CHECK(initialState.value().sizes() ==
                        at::IntArrayRef({q.size(0), v.size(1), q.size(3), v.size(3)}),
                    "chunk_kda_fwd_direct: initial_state shape mismatch");
    }
}

using DirectReturn = std::tuple<
    at::Tensor, c10::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>;

DirectReturn
ChunkKdaFwdDirectMeta(
    const at::Tensor &q, const at::Tensor &, const at::Tensor &v, const at::Tensor &gk,
    const at::Tensor &, double, int64_t chunkSize,
    const c10::optional<at::Tensor> &initialState, bool outputFinalState, bool safeGate)
{
    (void)safeGate;
    auto outputs = MakeOutputs(q, v, chunkSize, outputFinalState);
    c10::optional<at::Tensor> finalState = outputFinalState ?
        c10::optional<at::Tensor>(outputs.finalState) : c10::nullopt;
    return {outputs.o, finalState, gk, outputs.aqk, outputs.akk, outputs.w,
            outputs.u, outputs.qg, outputs.kg, outputs.vNew, outputs.h, initialState};
}


DirectReturn
ChunkKdaFwdDirectNpu(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &gk,
    const at::Tensor &beta, double scale, int64_t chunkSize,
    const c10::optional<at::Tensor> &initialState, bool outputFinalState, bool safeGate)
{
    const c10::OptionalDeviceGuard guard(q.device());
    CheckInputs(q, k, v, gk, beta, initialState, chunkSize);
    DirectOutputs outputs = MakeOutputs(q, v, chunkSize, outputFinalState);

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "chunk_kda_fwd_direct: PlatformAscendCManager is null");
    uint32_t blockDim = static_cast<uint32_t>(platform->GetCoreNumAic());
    size_t sysWorkspace = static_cast<size_t>(platform->GetLibApiWorkSpaceSize());
    int64_t totalChunks = (q.size(2) + chunkSize - 1) / chunkSize;
    const uint64_t dataBytes = q.element_size();
    const uint64_t tokenHeadCount =
        static_cast<uint64_t>(q.size(0)) * v.size(1) * q.size(2);
    const uint64_t qgScaledOffset = 0;
    const uint64_t qgScaledBytes = tokenHeadCount * q.size(3) * dataBytes;
    const uint64_t phaseBaseOffset = AlignUp(qgScaledBytes);
    const uint64_t wSeedOffset = phaseBaseOffset;
    const uint64_t uSeedOffset = AlignUp(wSeedOffset + qgScaledBytes);
    const uint64_t uSeedBytes = tokenHeadCount * v.size(3) * dataBytes;
    const uint64_t aqkFp32Offset = AlignUp(uSeedOffset + uSeedBytes);
    const uint64_t matrixFp32Bytes = tokenHeadCount * chunkSize * sizeof(float);
    const uint64_t akkFp32Offset = AlignUp(aqkFp32Offset + matrixFp32Bytes);
    const uint64_t prepareScratchOffset = AlignUp(akkFp32Offset + matrixFp32Bytes);
    const uint64_t solveScratch = static_cast<uint64_t>(blockDim) * SOLVE_SCRATCH_SLOTS *
                                  chunkSize * chunkSize * sizeof(float);
    const uint64_t scoreScratch = static_cast<uint64_t>(blockDim) * SCORE_QUEUE_SLOTS *
                                  SCORE_SCRATCH_PLANES * chunkSize * q.size(3) * dataBytes;
    const uint64_t prepareEnd = prepareScratchOffset + AlignUp(solveScratch) + scoreScratch;
    const uint64_t postScratchOffset = AlignUp(uSeedOffset + uSeedBytes);
    const uint64_t postScratchBytes = static_cast<uint64_t>(q.size(0)) * v.size(1) *
                                      totalChunks * chunkSize * q.size(3) * sizeof(float);
    const uint64_t postEnd = postScratchOffset + postScratchBytes;

    optiling::ChunkGatedDeltaRuleFwdHTilingContext stateContext{};
    stateContext.seqlen = q.size(2);
    stateContext.kNumHead = v.size(1);
    stateContext.kHeadDim = q.size(3);
    stateContext.vNumHead = v.size(1);
    stateContext.vHeadDim = v.size(3);
    stateContext.shapeBatchDim = q.size(0);
    stateContext.hasCuSeqlens = false;
    stateContext.cuSeqlensDim0 = 0;
    stateContext.dataType = q.scalar_type() == at::kBFloat16 ?
        optiling::GDN_FWD_H_DTYPE_BF16 : optiling::GDN_FWD_H_DTYPE_FP16;
    stateContext.gDataType = optiling::GDN_FWD_H_DTYPE_FP32;
    stateContext.useInitialState = initialState.has_value();
    stateContext.stateDataType = optiling::GDN_FWD_H_DTYPE_FP32;
    stateContext.useG = false;
    stateContext.useGk = true;
    stateContext.useExp2 = true;
    stateContext.storeFinalState = outputFinalState;
    stateContext.chunkSize = chunkSize;
    stateContext.aicCoreNum = blockDim;
    stateContext.libApiWorkSpaceSize = 0;
    ChunkGatedDeltaRuleFwdHTilingData stateTilingHost{};
    size_t stateWorkspaceSize = 0;
    uint32_t stateBlockDim = 0;
    optiling::ChunkGatedDeltaRuleFwdHTilingProcessor(stateContext).Process(
        stateTilingHost, stateBlockDim, stateWorkspaceSize);
    TORCH_CHECK(stateBlockDim == blockDim, "chunk_kda_fwd_direct: inconsistent state block dim");

    const uint64_t stateEnd = phaseBaseOffset + stateWorkspaceSize;
    const uint64_t outputScratchOffset = phaseBaseOffset;
    const uint64_t outputElements = tokenHeadCount * v.size(3);
    const uint64_t outputEnd = outputScratchOffset + 2 * outputElements * sizeof(float);
    const uint64_t userWorkspaceBytes = AlignUp(std::max({prepareEnd, postEnd, stateEnd, outputEnd}));
    DeviceBuffer workspace(sysWorkspace + userWorkspaceBytes);

    DirectKdaTilingData tiling{};
    tiling.batch = q.size(0);
    tiling.seqNum = q.size(0);
    tiling.qHeadNum = q.size(1);
    tiling.vHeadNum = v.size(1);
    tiling.seqlen = q.size(2);
    tiling.kHeadDim = q.size(3);
    tiling.vHeadDim = v.size(3);
    tiling.chunkSize = chunkSize;
    tiling.totalChunks = totalChunks;
    tiling.scale = static_cast<float>(scale);
    tiling.hasInitialState = initialState.has_value();
    tiling.outputFinalState = outputFinalState;
    tiling.isVarLen = false;
    tiling.safeGate = safeGate;
    tiling.prepareUsedCoreNum = static_cast<int64_t>(blockDim);
    tiling.prepareQgScaledOffset = static_cast<int64_t>(qgScaledOffset);
    tiling.prepareWSeedOffset = static_cast<int64_t>(wSeedOffset);
    tiling.prepareUSeedOffset = static_cast<int64_t>(uSeedOffset);
    tiling.prepareAqkFp32Offset = static_cast<int64_t>(aqkFp32Offset);
    tiling.prepareAkkFp32Offset = static_cast<int64_t>(akkFp32Offset);
    tiling.prepareScratchOffset = static_cast<int64_t>(prepareScratchOffset);
    tiling.postWuUsedCoreNum = static_cast<int64_t>(blockDim);
    tiling.postWuQgScaledOffset = static_cast<int64_t>(qgScaledOffset);
    tiling.postWuWSeedOffset = static_cast<int64_t>(wSeedOffset);
    tiling.postWuUSeedOffset = static_cast<int64_t>(uSeedOffset);
    tiling.postWuScratchOffset = static_cast<int64_t>(postScratchOffset);
    tiling.fwdHBatch = stateTilingHost.batch;
    tiling.fwdHSeqlen = stateTilingHost.seqlen;
    tiling.fwdHKNumHead = stateTilingHost.kNumHead;
    tiling.fwdHVNumHead = stateTilingHost.vNumHead;
    tiling.fwdHKHeadDim = stateTilingHost.kHeadDim;
    tiling.fwdHVHeadDim = stateTilingHost.vHeadDim;
    tiling.fwdHChunkSize = stateTilingHost.chunkSize;
    tiling.fwdHUseInitialState = stateTilingHost.useInitialState;
    tiling.fwdHStoreFinalState = stateTilingHost.storeFinalState;
    tiling.fwdHIsVariedLen = stateTilingHost.isVariedLen;
    tiling.fwdHShapeBatch = stateTilingHost.shapeBatch;
    tiling.fwdHTokenBatch = stateTilingHost.tokenBatch;
    tiling.fwdHVWorkspaceOffset = stateTilingHost.vWorkspaceOffset;
    tiling.fwdHVUpdateWorkspaceOffset = stateTilingHost.vUpdateWorkspaceOffset;
    tiling.fwdHKDecayWorkspaceOffset = stateTilingHost.kDecayWorkspaceOffset;
    tiling.fwdHHWorkspaceOffset = stateTilingHost.hWorkspaceOffset;
    tiling.fwdHNumSeqWorkspaceOffset = stateTilingHost.numSeqWorkspaceOffset;
    tiling.fwdHNumChunksWorkspaceOffset = stateTilingHost.numChunksWorkspaceOffset;
    tiling.fwdHWorkspaceBaseOffset = static_cast<int64_t>(phaseBaseOffset);
    tiling.outputUsedCoreNum = static_cast<int64_t>(blockDim);
    tiling.outputQgScaledOffset = static_cast<int64_t>(qgScaledOffset);
    tiling.outputScratchOffset = static_cast<int64_t>(outputScratchOffset);
    tiling.outputSequenceMajor = false;

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto ret = aclrtMemsetAsync(workspace.Get(), workspace.Size(), 0, workspace.Size(), stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "chunk_kda_fwd_direct: workspace memset failed: ", ret);

    auto launch = [&]() -> int {
        if (q.scalar_type() == at::kBFloat16) {
            if (safeGate) {
                LaunchStages<true, bfloat16_t>(
                    blockDim, stream, tiling, q, k, v, gk, beta, initialState, outputs, workspace);
            } else {
                LaunchStages<false, bfloat16_t>(
                    blockDim, stream, tiling, q, k, v, gk, beta, initialState, outputs, workspace);
            }
        } else if (safeGate) {
            LaunchStages<true, half>(
                blockDim, stream, tiling, q, k, v, gk, beta, initialState, outputs, workspace);
        } else {
            LaunchStages<false, half>(
                blockDim, stream, tiling, q, k, v, gk, beta, initialState, outputs, workspace);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ChunkKdaFwdDirect", launch);
    c10_npu::getCurrentNPUStream().synchronize();

    c10::optional<at::Tensor> finalState = outputFinalState ?
        c10::optional<at::Tensor>(outputs.finalState) : c10::nullopt;
    return {outputs.o, finalState, gk, outputs.aqk, outputs.akk, outputs.w, outputs.u,
            outputs.qg, outputs.kg, outputs.vNew, outputs.h, initialState};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("chunk_kda_fwd_direct(Tensor q, Tensor k, Tensor v, Tensor gk, Tensor beta, "
          "float scale, int chunk_size, *, Tensor? initial_state=None, bool output_final_state=True, "
          "bool safe_gate=False) "
          "-> (Tensor, Tensor?, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?)");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("chunk_kda_fwd_direct", ChunkKdaFwdDirectMeta);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("chunk_kda_fwd_direct", ChunkKdaFwdDirectNpu);
}

} // namespace ascend_ops::ChunkKdaFwdDirect
