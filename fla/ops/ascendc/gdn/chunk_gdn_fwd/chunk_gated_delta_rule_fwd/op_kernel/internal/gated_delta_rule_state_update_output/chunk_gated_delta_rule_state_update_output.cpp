/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "../operators/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "../operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "../operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif
#undef CATLASS_ARCH
#include "../operators/chunk_fwd_o/op_kernel/chunk_fwd_o_struct.h"
#include "../operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp"
#include "../operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_common.h"
#include "../operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h"
#include "../operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_vector.h"
#include "chunk_gated_delta_rule_state_update_output_struct.h"
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

template <typename InputT, typename GT, typename StateT, typename TileShapes, bool kGated,
          Arch35GdnSyncVariant Variant = Arch35GdnSyncVariant::B0>
__aicore__ inline void RunFwdH(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                               GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                               GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                               GM_ADDR userWorkspace)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    constexpr bool kB30 = Arch35GdnSyncTraits<Variant>::kB30;
    constexpr uint32_t rowTile = kB30 && std::is_same_v<StateT, bfloat16_t> ? 64 : 16;
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        InputT, GT, StateT, float, TileShapes, kGated, true, false, true, kB30, rowTile>;
#else
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        InputT, GT, StateT, float, TileShapes, kGated, true, false, false>;
#endif
    Kernel kernel;
    kernel.Init(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
    kernel.Process();
}

template <typename InputT, typename TileShapes, Arch35GdnSyncVariant Variant, typename StateT>
__aicore__ inline void DispatchFwdH(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                                    GM_ADDR userWorkspace)
{
    const __gm__ GdnMegaArch35FwdHTilingData *hTiling =
        reinterpret_cast<const __gm__ GdnMegaArch35FwdHTilingData *>(tiling);
    // Mega's input dtype is fixed by the generated DTYPE_Q variant, and its
    // cumsum/gk contract is FP32. State remains runtime-selected: a disabled
    // final-state output is an FP32 placeholder, not the initial-state dtype.
    if constexpr (Arch35GdnSyncTraits<Variant>::kB30) {
        // B30 requires an initial state and final-state output; host tiling
        // therefore uses the generated initial-state dtype without a placeholder.
        static_assert(std::is_same_v<StateT, float> || std::is_same_v<StateT, bfloat16_t>,
                      "B30 requires a supported generated initial-state dtype.");
        RunFwdH<InputT, float, StateT, TileShapes, false, Variant>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace);
    } else {
        if (hTiling->stateDataType == 2) {
            if (hTiling->useGk) {
                RunFwdH<InputT, float, float, TileShapes, true>(
                    k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            } else {
                RunFwdH<InputT, float, float, TileShapes, false>(
                    k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            }
        } else if (hTiling->useGk) {
            RunFwdH<InputT, float, InputT, TileShapes, true>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        } else {
            RunFwdH<InputT, float, InputT, TileShapes, false>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        }
    }
}

__aicore__ inline void CopyOTiling(const __gm__ GdnMegaArch35FwdOTilingData *src, GdnMegaArch35FwdOTilingData &dst)
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

__aicore__ inline void CopyRecomputeTiling(const __gm__ GdnMegaArch35RecomputeWUTilingData *src,
                                            GdnMegaArch35RecomputeWUTilingData &dst)
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
    dst.vbVecRow = src->vbVecRow;
    dst.kbgExpVecRow = src->kbgExpVecRow;
    dst.isVariable = src->isVariable;
}

template <typename InputT, typename GT, Arch35GdnSyncVariant Variant>
__aicore__ inline void RunFwdO(GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
                               GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
                               GM_ADDR userWorkspace, const GdnMegaArch35FwdOTilingData *tiling)
{
    using Sync = Arch35GdnSyncTraits<Variant>;
    using Kernel = Catlass::Gemm::Kernel::GDNFwdOKernel<
        InputT, GT, float, true, Sync::kAggregateQkMask, Sync::kAggregateOutput>;
    Kernel kernel;
    kernel.Init(q, k, vNew, h, g, cuSeqlens, chunkIndices, o, tiling, userWorkspace);
    kernel.Process();
}

template <typename kType, typename betaType, int VDim, typename TileShapes,
          bool kCoefficientGenerationTaskOrder = false>
__aicore__ inline void RunRecompute(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace,
    const GdnMegaArch35RecomputeWUTilingData *tiling)
{
    if ASCEND_IS_AIC {
        RecomputeWUFwdProcess<kType, betaType, typename TileShapes::L1TileShape,
                              typename TileShapes::L0TileShape, true, kCoefficientGenerationTaskOrder>
            process(k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace);
        process.Init(*tiling);
        process.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        RecomputeWUFwdVectorProcess<kType, betaType, true, kCoefficientGenerationTaskOrder> process(
            k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace);
        process.Init(*tiling, &pipe);
        process.Process();
    }
}

template <typename kType, typename betaType, int VDim, bool kCoefficientGenerationTaskOrder = false>
__aicore__ inline void DispatchRecompute(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace,
    const GdnMegaArch35RecomputeWUTilingData *tiling)
{
    if constexpr (VDim == 256) {
        RunRecompute<kType, betaType, VDim,
                     GDN::RecomputeWUFwdTileShapes256<kType, betaType>, kCoefficientGenerationTaskOrder>(
            k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace, tiling);
    } else {
        RunRecompute<kType, betaType, VDim,
                     GDN::RecomputeWUFwdTileShapes128<kType, betaType>, kCoefficientGenerationTaskOrder>(
            k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace, tiling);
    }
}

template <typename InputT, Arch35GdnSyncVariant Variant>
__aicore__ inline void DispatchFwdO(GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
                                    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
                                    GM_ADDR userWorkspace, const GdnMegaArch35FwdOTilingData *tiling)
{
    RunFwdO<InputT, float, Variant>(q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, tiling);
}

} // namespace
} // namespace GDN
