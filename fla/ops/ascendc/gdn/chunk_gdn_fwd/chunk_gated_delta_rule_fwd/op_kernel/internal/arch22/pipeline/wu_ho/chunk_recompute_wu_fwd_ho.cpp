/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "../../stages/fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#include "../../stages/fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#undef CATLASS_ARCH
#include "../../stages/fwd_o/op_kernel/chunk_fwd_o_struct.h"
#include "../../stages/fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp"
#include "gdn_ho_owner_local_kernel.hpp"
#include "../../stages/recompute_wu/op_kernel/recompute_w_u_fwd_common.h"
#include "../../stages/recompute_wu/op_kernel/recompute_w_u_fwd_cube.h"
#include "../../stages/recompute_wu/op_kernel/recompute_w_u_fwd_vector.h"
#include "../chunk_gated_delta_rule_ho_pipeline.h"
#include "../../common/gdn_raw_front_resources.h"
#include "chunk_recompute_wu_fwd_ho_struct.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace GDN {
namespace {

template <class... Dims>
using GemmCubeTileShape = tla::Shape<Dims...>;

template <typename kType, typename betaType>
struct RecomputeWUFwdTileShapes128 {
    using L1TileShape = GemmCubeTileShape<_128, _128, _256>;
    using L0TileShape = GemmCubeTileShape<_128, _128, _128>;
};

template <typename kType, typename betaType>
struct RecomputeWUFwdTileShapes256 {
    using L1TileShape = GemmCubeTileShape<_128, _256, _256>;
    using L0TileShape = GemmCubeTileShape<_128, _256, _64>;
};

constexpr uint64_t TILING_ALIGNMENT = 8;
__aicore__ inline uint64_t AlignTilingSize(uint64_t value)
{
    return (value + TILING_ALIGNMENT - 1) / TILING_ALIGNMENT * TILING_ALIGNMENT;
}

template <typename InputT, typename GT, typename StateT, typename TileShapes, bool kGated,
          int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1>
__aicore__ inline void RunFwdH(GdnRawFront::Resource &resource, GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                               GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                               GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                               GM_ADDR userWorkspace,
                               const GdnHoPipeline::HoPipelineConfig *idleConfig = nullptr,
                               GM_ADDR hoReadyAddr = nullptr, bool inputSequenceMajor = false,
                               GM_ADDR gLastExp = nullptr, GM_ADDR gDecay = nullptr,
                                const HoOwnerLocal::Config *ownerConfig = nullptr)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if constexpr (kStaticV == 128 && kStaticBT == 64 && kQkvLayout >= 0 && kQkvLayout <= 1 && !kGated &&
                  sizeof(InputT) == 2 && std::is_same_v<GT, float> && std::is_same_v<StateT, float>) {
        if (ownerConfig != nullptr && ownerConfig->enabled) {
            using OwnerKernel = Catlass::Gemm::Kernel::GDNFwdHOOwnerLocalKernel<InputT, TileShapes, kQkvLayout>;
            OwnerKernel kernel(resource);
            kernel.Init(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                        tiling, userWorkspace);
            kernel.ConfigureInputLayout(inputSequenceMajor);
            if (gLastExp != nullptr) { kernel.ConfigureGLastExp(gLastExp); }
            kernel.ConfigureGDecay(gDecay);
            kernel.ProcessOwnerLocal(*ownerConfig);
            return;
        }
    }
#endif
    // Keep the same H implementation mode as the established FwdHO kernel.
    // The final boolean enables the H/O fused scheduling path; using the
    // standalone-H mode here changes synchronization and precision behavior.
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        InputT, GT, StateT, float, TileShapes, kGated, true, false, true, kStaticV, kStaticBT, kQkvLayout>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // The private H implementation retains its original AtlasA2 resource type.
    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> hResource;
    Kernel kernel(hResource);
#else
    Kernel kernel(resource);
#endif
    kernel.Init(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
    if (idleConfig != nullptr) {
        // 统一入口在 Init 后、Process 前配置；关闭态在 kernel 内部直接返回。
        kernel.ConfigureIdlePipeline(*idleConfig, hoReadyAddr);
    }
    kernel.ConfigureInputLayout(inputSequenceMajor);
    if (gLastExp != nullptr) {
        kernel.ConfigureGLastExp(gLastExp);
    }
    kernel.ConfigureGDecay(gDecay);
    kernel.Process();
}

template <typename InputT, typename TileShapes,
          int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1>
__aicore__ inline void DispatchFwdH(GdnRawFront::Resource &resource, GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                                    GM_ADDR userWorkspace,
                                    const GdnHoPipeline::HoPipelineConfig *idleConfig = nullptr,
                                    GM_ADDR hoReadyAddr = nullptr, bool inputSequenceMajor = false,
                                    GM_ADDR gLastExp = nullptr, GM_ADDR gDecay = nullptr,
                                const HoOwnerLocal::Config *ownerConfig = nullptr)
{
    const __gm__ GdnMegaArch22FwdHTilingData *hTiling =
        reinterpret_cast<const __gm__ GdnMegaArch22FwdHTilingData *>(tiling);
    // Mega's input dtype is fixed by the generated DTYPE_Q variant, and its
    // cumsum/gk contract is FP32. State remains runtime-selected: a disabled
    // final-state output is an FP32 placeholder, not the initial-state dtype.
    if (hTiling->stateDataType == 2) {
        if (hTiling->useGk) {
            RunFwdH<InputT, float, float, TileShapes, true, kStaticV, kStaticBT, kQkvLayout>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay, ownerConfig);
        } else {
            RunFwdH<InputT, float, float, TileShapes, false, kStaticV, kStaticBT, kQkvLayout>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay, ownerConfig);
        }
    } else if (hTiling->useGk) {
        RunFwdH<InputT, float, InputT, TileShapes, true, kStaticV, kStaticBT, kQkvLayout>(
            resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay, ownerConfig);
    } else {
        RunFwdH<InputT, float, InputT, TileShapes, false, kStaticV, kStaticBT, kQkvLayout>(
            resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay, ownerConfig);
    }
}

#ifndef GDN_CHUNK_RECOMPUTE_WU_FWD_HO_IMPL_ONLY
template <typename TileShapes>
__aicore__ inline void DispatchFwdH(GdnRawFront::Resource &resource, GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                                    GM_ADDR userWorkspace)
{
    const __gm__ GdnMegaArch22FwdHTilingData *hTiling =
        reinterpret_cast<const __gm__ GdnMegaArch22FwdHTilingData *>(tiling);
    const bool useGk = hTiling->useGk;
    if (hTiling->dataType == 1) {
        if (hTiling->stateDataType == 2) {
            if (hTiling->gDataType == 2) {
                if (useGk) {
                    RunFwdH<bfloat16_t, float, float, TileShapes, true>(
                        resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                        tiling, userWorkspace);
                } else {
                    RunFwdH<bfloat16_t, float, float, TileShapes, false>(
                        resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                        tiling, userWorkspace);
                }
            } else if (useGk) {
                RunFwdH<bfloat16_t, bfloat16_t, float, TileShapes, true>(
                    resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            } else {
                RunFwdH<bfloat16_t, bfloat16_t, float, TileShapes, false>(
                    resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            }
        } else if (hTiling->gDataType == 2) {
            if (useGk) {
                RunFwdH<bfloat16_t, float, bfloat16_t, TileShapes, true>(
                    resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            } else {
                RunFwdH<bfloat16_t, float, bfloat16_t, TileShapes, false>(
                    resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            }
        } else if (useGk) {
            RunFwdH<bfloat16_t, bfloat16_t, bfloat16_t, TileShapes, true>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        } else {
            RunFwdH<bfloat16_t, bfloat16_t, bfloat16_t, TileShapes, false>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        }
    } else if (hTiling->stateDataType == 2) {
        if (hTiling->gDataType == 2) {
            if (useGk) {
                RunFwdH<half, float, float, TileShapes, true>(
                    resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            } else {
                RunFwdH<half, float, float, TileShapes, false>(
                    resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            }
        } else if (useGk) {
            RunFwdH<half, half, float, TileShapes, true>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        } else {
            RunFwdH<half, half, float, TileShapes, false>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        }
    } else if (hTiling->gDataType == 2) {
        if (useGk) {
            RunFwdH<half, float, half, TileShapes, true>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        } else {
            RunFwdH<half, float, half, TileShapes, false>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        }
    } else if (useGk) {
        RunFwdH<half, half, half, TileShapes, true>(
            resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace);
    } else {
        RunFwdH<half, half, half, TileShapes, false>(
            resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace);
    }
}

#endif

__aicore__ inline void CopyOTiling(const __gm__ GdnMegaArch22FwdOTilingData *src, GdnMegaArch22FwdOTilingData &dst)
{
    dst.shapeBatch = src->shapeBatch;
    dst.seqlen = src->seqlen;
    dst.kNumHead = src->kNumHead;
    dst.vNumHead = src->vNumHead;
    dst.kHeadDim = src->kHeadDim;
    dst.vHeadDim = src->vHeadDim;
    dst.chunkSize = src->chunkSize;
    dst.isVariedLen = src->isVariedLen;
    dst.tokenBatch = src->tokenBatch;
    dst.dataType = src->dataType;
    dst.gDataType = src->gDataType;
    dst.vWorkspaceOffset = src->vWorkspaceOffset;
    dst.hWorkspaceOffset = src->hWorkspaceOffset;
    dst.attnWorkspaceOffset = src->attnWorkspaceOffset;
    dst.aftermaskWorkspaceOffset = src->aftermaskWorkspaceOffset;
    dst.maskWorkspaceOffset = src->maskWorkspaceOffset;
    dst.scale = src->scale;
}

__aicore__ inline void CopyRecomputeTiling(const __gm__ GdnMegaArch22RecomputeWUTilingData *src,
                                            GdnMegaArch22RecomputeWUTilingData &dst)
{
    // Tiling data is serialized in GM; the recompute process consumes a local-memory copy.
    dst.B = src->B;
    dst.Hk = src->Hk;
    dst.Hv = src->Hv;
    dst.hvPerHk = src->hvPerHk;
    dst.T = src->T;
    dst.K = src->K;
    dst.V = src->V;
    dst.chunkNum = src->chunkNum;
    dst.chunkSize = src->chunkSize;
    dst.interleavedMembaseVecRow = src->interleavedMembaseVecRow;
    dst.isVariable = src->isVariable;
}

template <typename InputT, typename GT, int kStaticV = 0, int kStaticBT = 0,
          int kQkvLayout = -1, int kOLayout = -1>
__aicore__ inline void RunFwdO(GdnRawFront::Resource &resource, GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
                               GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
                               GM_ADDR userWorkspace, const GdnMegaArch22FwdOTilingData *tiling,
                               const GdnHoPipeline::HoPipelineConfig *idleConfig = nullptr,
                               GM_ADDR hoReadyAddr = nullptr, bool inputSequenceMajor = false, bool outputSequenceMajor = false)
{
    using Kernel = Catlass::Gemm::Kernel::GDNFwdOKernel<InputT, GT, float, true, kStaticV, kStaticBT, kQkvLayout, kOLayout>;
    Kernel kernel(resource);
    kernel.Init(q, k, vNew, h, g, cuSeqlens, chunkIndices, o, tiling, userWorkspace);
    if (idleConfig != nullptr) {
        kernel.ConfigureIdlePipeline(*idleConfig, hoReadyAddr);
    }
    kernel.ConfigureInputLayout(inputSequenceMajor);
    kernel.ConfigureOutputLayout(outputSequenceMajor);
    kernel.Process();
}

template <typename kType, typename betaType, int VDim, typename TileShapes,
          bool kAbcTaskOrder = false, int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1>
__aicore__ inline void RunRecompute(
    GdnRawFront::Resource &resource, GdnRawFront::VectorResources *resources,
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace,
    const GdnMegaArch22RecomputeWUTilingData *tiling,
    const GDN::RecomputeTaskRange *taskRange = nullptr, bool inputSequenceMajor = false)
{
    if ASCEND_IS_AIC {
        RecomputeWUFwdProcess<kType, betaType, typename TileShapes::L1TileShape,
                              typename TileShapes::L0TileShape, true, kAbcTaskOrder, kStaticV, kStaticBT>
            process(k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace);
        process.Init(*tiling);
        process.Process(resource, taskRange);
    }
    if ASCEND_IS_AIV {
        RecomputeWUFwdVectorProcess<kType, betaType, true, kAbcTaskOrder, true,
                                    kStaticV, kStaticBT, kQkvLayout> process(
            k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace);
        process.Init(*tiling, resources);
        process.ConfigureInputLayout(inputSequenceMajor);
        process.Process(taskRange);
    }
}

template <typename kType, typename betaType, int VDim, bool kAbcTaskOrder = false,
          int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1>
__aicore__ inline void DispatchRecompute(
    GdnRawFront::Resource &resource, GdnRawFront::VectorResources *resources,
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace,
    const GdnMegaArch22RecomputeWUTilingData *tiling,
    const GDN::RecomputeTaskRange *taskRange = nullptr, bool inputSequenceMajor = false)
{
    if constexpr (VDim == 256) {
        RunRecompute<kType, betaType, VDim,
                     GDN::RecomputeWUFwdTileShapes256<kType, betaType>, kAbcTaskOrder,
                     kStaticV, kStaticBT, kQkvLayout>(
            resource, resources, k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace, tiling, taskRange, inputSequenceMajor);
    } else {
        RunRecompute<kType, betaType, VDim,
                     GDN::RecomputeWUFwdTileShapes128<kType, betaType>, kAbcTaskOrder,
                     kStaticV, kStaticBT, kQkvLayout>(
            resource, resources, k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace, tiling, taskRange, inputSequenceMajor);
    }
}

template <typename InputT, int kStaticV = 0, int kStaticBT = 0,
          int kQkvLayout = -1, int kOLayout = -1>
__aicore__ inline void DispatchFwdO(GdnRawFront::Resource &resource, GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
                                    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
                                    GM_ADDR userWorkspace, const GdnMegaArch22FwdOTilingData *tiling,
                                    const GdnHoPipeline::HoPipelineConfig *idleConfig = nullptr,
                                    GM_ADDR hoReadyAddr = nullptr, bool inputSequenceMajor = false, bool outputSequenceMajor = false)
{
    RunFwdO<InputT, float, kStaticV, kStaticBT, kQkvLayout, kOLayout>(resource, q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, tiling,
                           idleConfig, hoReadyAddr, inputSequenceMajor, outputSequenceMajor);
}

#ifndef GDN_CHUNK_RECOMPUTE_WU_FWD_HO_IMPL_ONLY
__aicore__ inline void DispatchFwdO(GdnRawFront::Resource &resource, GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
                                    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
                                    GM_ADDR userWorkspace, const GdnMegaArch22FwdOTilingData *tiling)
{
    if (tiling->dataType == 1) {
        if (tiling->gDataType == 2) {
            RunFwdO<bfloat16_t, float>(resource, q, k, vNew, h, g, cuSeqlens, chunkIndices, o,
                                       userWorkspace, tiling);
        } else {
            RunFwdO<bfloat16_t, bfloat16_t>(resource, q, k, vNew, h, g, cuSeqlens, chunkIndices, o,
                                            userWorkspace, tiling);
        }
    } else if (tiling->gDataType == 2) {
        RunFwdO<half, float>(resource, q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, tiling);
    } else {
        RunFwdO<half, half>(resource, q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, tiling);
    }
}

template <typename TileShapes>
__aicore__ inline void RunFused(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling)
{
    GdnRawFront::Resource resource;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const uint64_t oTilingOffset = AlignTilingSize(sizeof(GdnMegaArch22FwdHTilingData));
    const __gm__ GdnMegaArch22FwdOTilingData *gmOTiling =
        reinterpret_cast<const __gm__ GdnMegaArch22FwdOTilingData *>(tiling + oTilingOffset);
    const __gm__ ChunkRecomputeWUFwdHOTrailer *trailer =
        reinterpret_cast<const __gm__ ChunkRecomputeWUFwdHOTrailer *>(
            tiling + oTilingOffset + sizeof(GdnMegaArch22FwdOTilingData));
    GM_ADDR w = userWorkspace + trailer->wIntermediateOffset;
    GM_ADDR u = userWorkspace + trailer->uIntermediateOffset;
    GM_ADDR h = userWorkspace + trailer->hIntermediateOffset;
    GM_ADDR vNew = userWorkspace + trailer->vNewIntermediateOffset;
    GdnMegaArch22RecomputeWUTilingData recomputeTiling{};
    CopyRecomputeTiling(&trailer->recompute, recomputeTiling);
    GdnRawFront::VectorResources vectorResources;
    if ASCEND_IS_AIV {
        vectorResources.Init(resource, recomputeTiling.chunkSize, recomputeTiling.chunkSize,
            recomputeTiling.Hv, sizeof(uint16_t));
    }

    if (trailer->qDataType == 1) {
        if (trailer->recompute.V == 256) {
            DispatchRecompute<bfloat16_t, float, 256>(
                resource, &vectorResources, k, v, beta, A, g, cuSeqlens, chunkIndices, w, u,
                userWorkspace + trailer->recomputeWorkspaceOffset, &recomputeTiling);
        } else {
            DispatchRecompute<bfloat16_t, float, 128>(
                resource, &vectorResources, k, v, beta, A, g, cuSeqlens, chunkIndices, w, u,
                userWorkspace + trailer->recomputeWorkspaceOffset, &recomputeTiling);
        }
    } else if (trailer->recompute.V == 256) {
        DispatchRecompute<half, float, 256>(
            resource, &vectorResources, k, v, beta, A, g, cuSeqlens, chunkIndices, w, u,
            userWorkspace + trailer->recomputeWorkspaceOffset, &recomputeTiling);
    } else {
        DispatchRecompute<half, float, 128>(
            resource, &vectorResources, k, v, beta, A, g, cuSeqlens, chunkIndices, w, u,
            userWorkspace + trailer->recomputeWorkspaceOffset, &recomputeTiling);
    }
    // FwdH performs a full MIX barrier after state initialization and before
    // either AIC or AIV consumes w/u, so a second stage barrier here is redundant.
    DispatchFwdH<TileShapes>(resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
                             h, vNew, finalState, tiling, userWorkspace);

    GdnMegaArch22FwdOTilingData oTiling{};
    CopyOTiling(gmOTiling, oTiling);
    DispatchFwdO(resource, q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, &oTiling);
}

#endif // GDN_CHUNK_RECOMPUTE_WU_FWD_HO_IMPL_ONLY

} // namespace
} // namespace GDN

#ifndef GDN_CHUNK_RECOMPUTE_WU_FWD_HO_IMPL_ONLY
extern "C" __global__ __aicore__ void chunk_recompute_wu_fwd_ho(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR o, GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    // The host packs H/O/recompute metadata into one opaque tiling blob. Register
    // the trailer type so the AscendC compiler emits the kernel tiling metadata;
    // the entry still reads the packed sections by their explicit offsets above.
    REGISTER_TILING_DEFAULT(GDN::ChunkRecomputeWUFwdHOTrailer);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunFused<Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            q, k, v, beta, A, g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, workspace, tiling);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunFused<Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            q, k, v, beta, A, g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, workspace, tiling);
    }
}
#endif
