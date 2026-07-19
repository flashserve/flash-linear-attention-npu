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

#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif

#define KDA_FAST_KERNEL_LAUNCH
#include "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp"

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
    int64_t usedCoreNum;
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

enum class DirectLaunchPhase : uint8_t {
    PREPARE,
    POST_WU,
    STATE,
    OUTPUT,
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

template <KdaPhase PHASE, typename T>
__global__ __aicore__ void ChunkKdaPhaseKernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR preparedQG, GM_ADDR preparedAqk, GM_ADDR propagatedVNew, GM_ADDR propagatedH,
    GM_ADDR o, GM_ADDR finalState, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u,
    GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h, GM_ADDR workspace,
    DirectKdaTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    AscendC::TPipe pipe;
    if constexpr (PHASE == KdaPhase::PREPARE) {
        if ASCEND_IS_AIC {
            ChunkKdaFwdKernel<KdaPhase::PREPARE, T, T, float, float, float> op;
            op.Init(q, k, v, gk, beta, initialState, nullptr, nullptr, preparedQG, preparedAqk,
                    propagatedVNew, propagatedH, o, finalState, aqk, akk, w, u, qg, kg, vNew, h,
                    userWorkspace, tiling, &pipe, false);
            op.ProcessAic();
        }
        if ASCEND_IS_AIV {
            ChunkKdaFwdKernel<KdaPhase::PREPARE, T, T, float, float, float> op;
            op.Init(q, k, v, gk, beta, initialState, nullptr, nullptr, preparedQG, preparedAqk,
                    propagatedVNew, propagatedH, o, finalState, aqk, akk, w, u, qg, kg, vNew, h,
                    userWorkspace, tiling, &pipe);
            op.ProcessAiv();
        }
    } else if constexpr (PHASE == KdaPhase::POST_WU) {
        if ASCEND_IS_AIC {
            ChunkKdaFwdKernel<KdaPhase::POST_WU, T, T, T, float, float> op;
            op.Init(q, k, v, gk, beta, initialState, nullptr, nullptr, preparedQG, preparedAqk,
                    propagatedVNew, propagatedH, o, finalState, aqk, akk, w, u, qg, kg, vNew, h,
                    userWorkspace, tiling, &pipe, false);
            op.ProcessAic();
        }
        if ASCEND_IS_AIV {
            ChunkKdaFwdKernel<KdaPhase::POST_WU, T, T, T, float, float> op;
            op.Init(q, k, v, gk, beta, initialState, nullptr, nullptr, preparedQG, preparedAqk,
                    propagatedVNew, propagatedH, o, finalState, aqk, akk, w, u, qg, kg, vNew, h,
                    userWorkspace, tiling, &pipe);
            op.ProcessAiv();
        }
    } else {
        if ASCEND_IS_AIC {
            ChunkKdaFwdKernel<KdaPhase::OUTPUT, T, float, float, float, float> op;
            op.Init(q, k, v, gk, beta, initialState, nullptr, nullptr, preparedQG, preparedAqk,
                    propagatedVNew, propagatedH, o, finalState, aqk, akk, w, u, qg, kg, vNew, h,
                    userWorkspace, tiling, &pipe, false);
            op.ProcessAic();
        }
        if ASCEND_IS_AIV {
            ChunkKdaFwdKernel<KdaPhase::OUTPUT, T, float, float, float, float> op;
            op.Init(q, k, v, gk, beta, initialState, nullptr, nullptr, preparedQG, preparedAqk,
                    propagatedVNew, propagatedH, o, finalState, aqk, akk, w, u, qg, kg, vNew, h,
                    userWorkspace, tiling, &pipe);
            op.ProcessAiv();
        }
    }
}

template <typename T, typename TileShapes>
__aicore__ inline void RunChunkKdaState(
    GM_ADDR kg, GM_ADDR w, GM_ADDR u, GM_ADDR neutralG, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<T, float, float, float, TileShapes, true>;
    Kernel kernel;
    kernel.Init(kg, w, u, neutralG, gk, initialState, nullptr, nullptr,
                h, vNew, finalState, tiling, userWorkspace);
    kernel.Process();
}

#define DEFINE_KDA_STATE_KERNEL(kernelName, ElementType, TileType)                                      \
    __global__ __aicore__ void kernelName(                                                              \
        GM_ADDR kg, GM_ADDR w, GM_ADDR u, GM_ADDR neutralG, GM_ADDR gk, GM_ADDR initialState,           \
        GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling)                  \
    {                                                                                                    \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                              \
        RunChunkKdaState<ElementType, TileType>(kg, w, u, neutralG, gk, initialState, h, vNew,          \
                                                 finalState, workspace, tiling);                          \
    }

DEFINE_KDA_STATE_KERNEL(ChunkKdaStateFp16V128, half, Catlass::Gemm::Kernel::GDNFwdHTileShapes128)
DEFINE_KDA_STATE_KERNEL(ChunkKdaStateFp16V256, half, Catlass::Gemm::Kernel::GDNFwdHTileShapes256)
DEFINE_KDA_STATE_KERNEL(ChunkKdaStateBf16V128, bfloat16_t, Catlass::Gemm::Kernel::GDNFwdHTileShapes128)
DEFINE_KDA_STATE_KERNEL(ChunkKdaStateBf16V256, bfloat16_t, Catlass::Gemm::Kernel::GDNFwdHTileShapes256)

#undef DEFINE_KDA_STATE_KERNEL

template <KdaPhase PHASE, typename T>
void LaunchKdaPhase(
    uint32_t blockDim, aclrtStream stream, const DirectKdaTilingData &tiling,
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &gk,
    const at::Tensor &beta, const c10::optional<at::Tensor> &initialState,
    const at::Tensor &preparedQG, const at::Tensor &preparedAqk,
    const at::Tensor &propagatedVNew, const at::Tensor &propagatedH,
    const at::Tensor &o, const at::Tensor &finalState, const at::Tensor &aqk,
    const at::Tensor &akk, const at::Tensor &w, const at::Tensor &u,
    const at::Tensor &qg, const at::Tensor &kg, const at::Tensor &vNew,
    const at::Tensor &h, const DeviceBuffer &workspace)
{
    auto ptr = [](const at::Tensor &tensor) -> GM_ADDR {
        return tensor.defined() ? (GM_ADDR)tensor.data_ptr() : nullptr;
    };
    GM_ADDR initialPtr = initialState.has_value() ?
        (GM_ADDR)initialState.value().data_ptr() : nullptr;
    ChunkKdaPhaseKernel<PHASE, T><<<blockDim, nullptr, stream>>>(
        ptr(q), ptr(k), ptr(v), ptr(gk), ptr(beta), initialPtr,
        ptr(preparedQG), ptr(preparedAqk), ptr(propagatedVNew), ptr(propagatedH),
        ptr(o), ptr(finalState), ptr(aqk), ptr(akk), ptr(w), ptr(u), ptr(qg), ptr(kg),
        ptr(vNew), ptr(h), (GM_ADDR)workspace.Get(), tiling);
}

template <typename T, typename TileShapes>
void LaunchState(
    uint32_t blockDim, aclrtStream stream, const at::Tensor &kg, const at::Tensor &w,
    const at::Tensor &u, const at::Tensor &neutralG, const at::Tensor &gk,
    const c10::optional<at::Tensor> &initialState, const at::Tensor &h,
    const at::Tensor &vNew, const at::Tensor &finalState, const DeviceBuffer &workspace,
    const DeviceBuffer &tiling)
{
    GM_ADDR initialPtr = initialState.has_value() ?
        (GM_ADDR)initialState.value().data_ptr() : nullptr;
    GM_ADDR kgPtr = (GM_ADDR)kg.data_ptr();
    GM_ADDR wPtr = (GM_ADDR)w.data_ptr();
    GM_ADDR uPtr = (GM_ADDR)u.data_ptr();
    GM_ADDR neutralGPtr = (GM_ADDR)neutralG.data_ptr();
    GM_ADDR gkPtr = (GM_ADDR)gk.data_ptr();
    GM_ADDR hPtr = (GM_ADDR)h.data_ptr();
    GM_ADDR vNewPtr = (GM_ADDR)vNew.data_ptr();
    GM_ADDR finalStatePtr = (GM_ADDR)finalState.data_ptr();
    GM_ADDR workspacePtr = (GM_ADDR)workspace.Get();
    GM_ADDR tilingPtr = (GM_ADDR)tiling.Get();
    if constexpr (std::is_same_v<T, half>) {
        if constexpr (std::is_same_v<TileShapes, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>) {
            ChunkKdaStateFp16V256<<<blockDim, nullptr, stream>>>(
                kgPtr, wPtr, uPtr, neutralGPtr, gkPtr, initialPtr, hPtr, vNewPtr,
                finalStatePtr, workspacePtr, tilingPtr);
        } else {
            ChunkKdaStateFp16V128<<<blockDim, nullptr, stream>>>(
                kgPtr, wPtr, uPtr, neutralGPtr, gkPtr, initialPtr, hPtr, vNewPtr,
                finalStatePtr, workspacePtr, tilingPtr);
        }
    } else {
        if constexpr (std::is_same_v<TileShapes, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>) {
            ChunkKdaStateBf16V256<<<blockDim, nullptr, stream>>>(
                kgPtr, wPtr, uPtr, neutralGPtr, gkPtr, initialPtr, hPtr, vNewPtr,
                finalStatePtr, workspacePtr, tilingPtr);
        } else {
            ChunkKdaStateBf16V128<<<blockDim, nullptr, stream>>>(
                kgPtr, wPtr, uPtr, neutralGPtr, gkPtr, initialPtr, hPtr, vNewPtr,
                finalStatePtr, workspacePtr, tilingPtr);
        }
    }
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
    const c10::optional<at::Tensor> &initialState, bool outputFinalState)
{
    auto outputs = MakeOutputs(q, v, chunkSize, outputFinalState);
    c10::optional<at::Tensor> finalState = outputFinalState ?
        c10::optional<at::Tensor>(outputs.finalState) : c10::nullopt;
    return {outputs.o, finalState, gk, outputs.aqk, outputs.akk, outputs.w,
            outputs.u, outputs.qg, outputs.kg, outputs.vNew, outputs.h, initialState};
}

template <typename T>
void LaunchOne(
    DirectLaunchPhase phase,
    uint32_t blockDim, aclrtStream stream, const DirectKdaTilingData &tiling,
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &gk,
    const at::Tensor &beta, const c10::optional<at::Tensor> &initialState,
    DirectOutputs &outputs, const at::Tensor &aqkFp32, const at::Tensor &akkFp32,
    const at::Tensor &wPre, const at::Tensor &akkPost, const at::Tensor &qgScaled,
    const at::Tensor &uSeed, const at::Tensor &wScratch, const at::Tensor &neutralG,
    const DeviceBuffer &prepareWorkspace, const DeviceBuffer &postWorkspace,
    const DeviceBuffer &stateWorkspace, const DeviceBuffer &stateTiling,
    const DeviceBuffer &outputWorkspace)
{
    auto dummyT = at::empty({1}, q.options());
    auto dummyFloat = at::empty({1}, q.options().dtype(at::kFloat));

    switch (phase) {
        case DirectLaunchPhase::PREPARE:
            LaunchKdaPhase<KdaPhase::PREPARE, T>(
                blockDim, stream, tiling, q, k, v, gk, beta, initialState,
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), outputs.aqk,
                dummyFloat, aqkFp32, akkFp32, wPre, akkPost, outputs.qg, qgScaled,
                uSeed, dummyFloat, prepareWorkspace);
            break;
        case DirectLaunchPhase::POST_WU:
            LaunchKdaPhase<KdaPhase::POST_WU, T>(
                blockDim, stream, tiling, q, k, v, gk, beta, initialState,
                wPre, akkPost, uSeed, at::Tensor(), dummyT, dummyFloat, dummyFloat,
                dummyT, outputs.w, outputs.u, dummyT, outputs.kg, dummyT, wScratch,
                postWorkspace);
            break;
        case DirectLaunchPhase::STATE:
            if (outputs.vNew.size(3) == 256) {
                LaunchState<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
                    blockDim, stream, outputs.kg, outputs.w, outputs.u, neutralG, gk,
                    initialState, outputs.h, outputs.vNew, outputs.finalState,
                    stateWorkspace, stateTiling);
            } else {
                LaunchState<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
                    blockDim, stream, outputs.kg, outputs.w, outputs.u, neutralG, gk,
                    initialState, outputs.h, outputs.vNew, outputs.finalState,
                    stateWorkspace, stateTiling);
            }
            break;
        case DirectLaunchPhase::OUTPUT:
            LaunchKdaPhase<KdaPhase::OUTPUT, T>(
                blockDim, stream, tiling, q, k, v, gk, beta, initialState,
                qgScaled, outputs.aqk, outputs.vNew, outputs.h, dummyFloat, dummyFloat,
                dummyFloat, dummyFloat, dummyT, dummyFloat, dummyT, dummyT, outputs.o,
                dummyFloat, outputWorkspace);
            break;
    }
}

DirectReturn
ChunkKdaFwdDirectNpu(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &gk,
    const at::Tensor &beta, double scale, int64_t chunkSize,
    const c10::optional<at::Tensor> &initialState, bool outputFinalState)
{
    const c10::OptionalDeviceGuard guard(q.device());
    CheckInputs(q, k, v, gk, beta, initialState, chunkSize);
    DirectOutputs outputs = MakeOutputs(q, v, chunkSize, outputFinalState);

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "chunk_kda_fwd_direct: PlatformAscendCManager is null");
    uint32_t blockDim = static_cast<uint32_t>(platform->GetCoreNumAic());
    size_t sysWorkspace = static_cast<size_t>(platform->GetLibApiWorkSpaceSize());
    int64_t totalChunks = (q.size(2) + chunkSize - 1) / chunkSize;
    DirectKdaTilingData tiling{
        q.size(0), q.size(0), q.size(1), v.size(1), q.size(2), q.size(3), v.size(3),
        chunkSize, totalChunks, static_cast<float>(scale), initialState.has_value(),
        outputFinalState, false, static_cast<int64_t>(blockDim)};

    uint64_t solveScratch = static_cast<uint64_t>(blockDim) * SOLVE_SCRATCH_SLOTS *
                            chunkSize * chunkSize * sizeof(float);
    uint64_t scoreScratch = static_cast<uint64_t>(blockDim) * SCORE_QUEUE_SLOTS *
                            SCORE_SCRATCH_PLANES * chunkSize * q.size(3) * q.element_size();
    uint64_t prepareScratch = AlignUp(solveScratch) + scoreScratch;
    uint64_t outputElements = static_cast<uint64_t>(q.size(0)) * v.size(1) * q.size(2) * v.size(3);
    DeviceBuffer prepareWorkspace(sysWorkspace + std::max<uint64_t>(AlignUp(prepareScratch), WORKSPACE_ALIGN));
    DeviceBuffer postWorkspace(sysWorkspace + WORKSPACE_ALIGN);
    DeviceBuffer outputWorkspace(sysWorkspace + AlignUp(2 * outputElements * sizeof(float)));

    auto qOptions = q.options();
    auto fp32Options = qOptions.dtype(at::kFloat);
    at::Tensor aqkFp32 = at::empty({q.size(0), v.size(1), q.size(2), chunkSize}, fp32Options);
    at::Tensor akkFp32 = at::empty_like(aqkFp32);
    at::Tensor wPre = at::empty({q.size(0), v.size(1), q.size(2), q.size(3)}, qOptions);
    at::Tensor akkPost = outputs.akk;
    at::Tensor qgScaled = at::empty_like(wPre);
    at::Tensor uSeed = at::empty({q.size(0), v.size(1), q.size(2), v.size(3)}, qOptions);
    at::Tensor wScratch = at::empty({q.size(0), v.size(1), totalChunks, chunkSize, q.size(3)}, fp32Options);
    at::Tensor neutralG = at::zeros({q.size(0), v.size(1), q.size(2)}, fp32Options);

    optiling::ChunkGatedDeltaRuleFwdHTilingContext stateContext{};
    stateContext.seqlen = q.size(2);
    // The state kernel consumes post-WU kg, whose head axis has already been
    // expanded from H_k to H_v.
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
    stateContext.useGk = true;
    stateContext.storeFinalState = outputFinalState;
    stateContext.chunkSize = chunkSize;
    stateContext.aicCoreNum = blockDim;
    stateContext.libApiWorkSpaceSize = sysWorkspace;
    ChunkGatedDeltaRuleFwdHTilingData stateTilingHost{};
    size_t stateWorkspaceSize = 0;
    uint32_t stateBlockDim = 0;
    optiling::ChunkGatedDeltaRuleFwdHTilingProcessor(stateContext).Process(
        stateTilingHost, stateBlockDim, stateWorkspaceSize);
    TORCH_CHECK(stateBlockDim == blockDim, "chunk_kda_fwd_direct: inconsistent state block dim");
    DeviceBuffer stateWorkspace(stateWorkspaceSize);
    DeviceBuffer stateTiling(sizeof(stateTilingHost));

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    for (const DeviceBuffer *buffer : {&prepareWorkspace, &postWorkspace, &outputWorkspace, &stateWorkspace}) {
        auto ret = aclrtMemsetAsync(buffer->Get(), buffer->Size(), 0, buffer->Size(), stream);
        TORCH_CHECK(ret == ACL_SUCCESS, "chunk_kda_fwd_direct: workspace memset failed: ", ret);
    }
    auto ret = aclrtMemcpy(stateTiling.Get(), stateTiling.Size(), &stateTilingHost,
                           sizeof(stateTilingHost), ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(ret == ACL_SUCCESS, "chunk_kda_fwd_direct: state tiling copy failed: ", ret);

    auto launchPhase = [&](DirectLaunchPhase phase) -> int {
        if (q.scalar_type() == at::kBFloat16) {
            LaunchOne<bfloat16_t>(phase, blockDim, stream, tiling, q, k, v, gk, beta, initialState,
                                  outputs, aqkFp32, akkFp32, wPre, akkPost, qgScaled, uSeed, wScratch,
                                  neutralG, prepareWorkspace, postWorkspace, stateWorkspace, stateTiling,
                                  outputWorkspace);
        } else {
            LaunchOne<half>(phase, blockDim, stream, tiling, q, k, v, gk, beta, initialState,
                            outputs, aqkFp32, akkFp32, wPre, akkPost, qgScaled, uSeed, wScratch,
                            neutralG, prepareWorkspace, postWorkspace, stateWorkspace, stateTiling,
                            outputWorkspace);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "ChunkKdaFwdDirectPrepare", [&]() -> int { return launchPhase(DirectLaunchPhase::PREPARE); });
    at_npu::native::OpCommand::RunOpApi(
        "ChunkKdaFwdDirectPostWu", [&]() -> int { return launchPhase(DirectLaunchPhase::POST_WU); });
    at_npu::native::OpCommand::RunOpApi(
        "ChunkKdaFwdDirectState", [&]() -> int { return launchPhase(DirectLaunchPhase::STATE); });
    at_npu::native::OpCommand::RunOpApi(
        "ChunkKdaFwdDirectOutput", [&]() -> int { return launchPhase(DirectLaunchPhase::OUTPUT); });
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
          "float scale, int chunk_size, *, Tensor? initial_state=None, bool output_final_state=True) "
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
