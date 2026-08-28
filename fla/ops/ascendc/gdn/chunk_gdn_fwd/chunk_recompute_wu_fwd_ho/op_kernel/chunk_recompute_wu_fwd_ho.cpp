/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif
#undef CATLASS_ARCH
#include "../../chunk_fwd_o/op_kernel/chunk_fwd_o_struct.h"
#include "../../chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp"
#include "../../recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_common.h"
#include "../../recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h"
#include "../../recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_vector.h"
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

template <typename InputT, typename GT, typename StateT, typename TileShapes, bool kGated>
__aicore__ inline void RunFwdH(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                               GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                               GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                               GM_ADDR userWorkspace)
{
    // Keep the same H implementation mode as the established FwdHO kernel.
    // The final boolean enables the H/O fused scheduling path; using the
    // standalone-H mode here changes synchronization and precision behavior.
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        InputT, GT, StateT, float, TileShapes, kGated, true, false, true>;
    Kernel kernel;
    kernel.Init(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
    kernel.Process();
}

template <typename TileShapes>
__aicore__ inline void DispatchFwdH(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                                    GM_ADDR userWorkspace)
{
    const __gm__ ChunkGatedDeltaRuleFwdHTilingData *hTiling =
        reinterpret_cast<const __gm__ ChunkGatedDeltaRuleFwdHTilingData *>(tiling);
    const bool useGk = hTiling->useGk;
    if (hTiling->dataType == 1) {
        if (hTiling->stateDataType == 2) {
            if (hTiling->gDataType == 2) {
                if (useGk) {
                    RunFwdH<bfloat16_t, float, float, TileShapes, true>(
                        k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                        tiling, userWorkspace);
                } else {
                    RunFwdH<bfloat16_t, float, float, TileShapes, false>(
                        k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                        tiling, userWorkspace);
                }
            } else if (useGk) {
                RunFwdH<bfloat16_t, bfloat16_t, float, TileShapes, true>(
                    k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            } else {
                RunFwdH<bfloat16_t, bfloat16_t, float, TileShapes, false>(
                    k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            }
        } else if (hTiling->gDataType == 2) {
            if (useGk) {
                RunFwdH<bfloat16_t, float, bfloat16_t, TileShapes, true>(
                    k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            } else {
                RunFwdH<bfloat16_t, float, bfloat16_t, TileShapes, false>(
                    k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            }
        } else if (useGk) {
            RunFwdH<bfloat16_t, bfloat16_t, bfloat16_t, TileShapes, true>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        } else {
            RunFwdH<bfloat16_t, bfloat16_t, bfloat16_t, TileShapes, false>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        }
    } else if (hTiling->stateDataType == 2) {
        if (hTiling->gDataType == 2) {
            if (useGk) {
                RunFwdH<half, float, float, TileShapes, true>(
                    k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            } else {
                RunFwdH<half, float, float, TileShapes, false>(
                    k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                    tiling, userWorkspace);
            }
        } else if (useGk) {
            RunFwdH<half, half, float, TileShapes, true>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        } else {
            RunFwdH<half, half, float, TileShapes, false>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        }
    } else if (hTiling->gDataType == 2) {
        if (useGk) {
            RunFwdH<half, float, half, TileShapes, true>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        } else {
            RunFwdH<half, float, half, TileShapes, false>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
        }
    } else if (useGk) {
        RunFwdH<half, half, half, TileShapes, true>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace);
    } else {
        RunFwdH<half, half, half, TileShapes, false>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace);
    }
}

__aicore__ inline void CopyOTiling(const __gm__ ChunkFwdOTilingData *src, ChunkFwdOTilingData &dst)
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

__aicore__ inline void CopyRecomputeTiling(const __gm__ RecomputeWUFwdTilingData *src,
                                            RecomputeWUFwdTilingData &dst)
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

template <typename InputT, typename GT>
__aicore__ inline void RunFwdO(GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
                               GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
                               GM_ADDR userWorkspace, const ChunkFwdOTilingData *tiling)
{
    using Kernel = Catlass::Gemm::Kernel::GDNFwdOKernel<InputT, GT, float, true>;
    Kernel kernel;
    kernel.Init(q, k, vNew, h, g, cuSeqlens, chunkIndices, o, tiling, userWorkspace);
    kernel.Process();
}

template <typename kType, typename betaType, int VDim, typename TileShapes,
          bool kAbcTaskOrder = false>
__aicore__ inline void RunRecompute(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace,
    const RecomputeWUFwdTilingData *tiling)
{
    if ASCEND_IS_AIC {
        RecomputeWUFwdProcess<kType, betaType, typename TileShapes::L1TileShape,
                              typename TileShapes::L0TileShape, true, kAbcTaskOrder>
            process(k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace);
        process.Init(*tiling);
        process.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        RecomputeWUFwdVectorProcess<kType, betaType, true, kAbcTaskOrder> process(
            k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace);
        process.Init(*tiling, &pipe);
        process.Process();
    }
}

template <typename kType, typename betaType, int VDim, bool kAbcTaskOrder = false>
__aicore__ inline void DispatchRecompute(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace,
    const RecomputeWUFwdTilingData *tiling)
{
    if constexpr (VDim == 256) {
        RunRecompute<kType, betaType, VDim,
                     GDN::RecomputeWUFwdTileShapes256<kType, betaType>, kAbcTaskOrder>(
            k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace, tiling);
    } else {
        RunRecompute<kType, betaType, VDim,
                     GDN::RecomputeWUFwdTileShapes128<kType, betaType>, kAbcTaskOrder>(
            k, v, beta, A, g, cuSeqlens, chunkIndices, w, u, workspace, tiling);
    }
}

__aicore__ inline void DispatchFwdO(GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
                                    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
                                    GM_ADDR userWorkspace, const ChunkFwdOTilingData *tiling)
{
    if (tiling->dataType == 1) {
        if (tiling->gDataType == 2) {
            RunFwdO<bfloat16_t, float>(q, k, vNew, h, g, cuSeqlens, chunkIndices, o,
                                       userWorkspace, tiling);
        } else {
            RunFwdO<bfloat16_t, bfloat16_t>(q, k, vNew, h, g, cuSeqlens, chunkIndices, o,
                                            userWorkspace, tiling);
        }
    } else if (tiling->gDataType == 2) {
        RunFwdO<half, float>(q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, tiling);
    } else {
        RunFwdO<half, half>(q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, tiling);
    }
}

template <typename TileShapes>
__aicore__ inline void RunFused(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const uint64_t oTilingOffset = AlignTilingSize(sizeof(ChunkGatedDeltaRuleFwdHTilingData));
    const __gm__ ChunkFwdOTilingData *gmOTiling =
        reinterpret_cast<const __gm__ ChunkFwdOTilingData *>(tiling + oTilingOffset);
    const __gm__ ChunkRecomputeWUFwdHOTrailer *trailer =
        reinterpret_cast<const __gm__ ChunkRecomputeWUFwdHOTrailer *>(
            tiling + oTilingOffset + sizeof(ChunkFwdOTilingData));
    GM_ADDR w = userWorkspace + trailer->wIntermediateOffset;
    GM_ADDR u = userWorkspace + trailer->uIntermediateOffset;
    GM_ADDR h = userWorkspace + trailer->hIntermediateOffset;
    GM_ADDR vNew = userWorkspace + trailer->vNewIntermediateOffset;
    RecomputeWUFwdTilingData recomputeTiling{};
    CopyRecomputeTiling(&trailer->recompute, recomputeTiling);

    if (trailer->qDataType == 1) {
        if (trailer->recompute.V == 256) {
            DispatchRecompute<bfloat16_t, float, 256>(
                k, v, beta, A, g, cuSeqlens, chunkIndices, w, u,
                userWorkspace + trailer->recomputeWorkspaceOffset, &recomputeTiling);
        } else {
            DispatchRecompute<bfloat16_t, float, 128>(
                k, v, beta, A, g, cuSeqlens, chunkIndices, w, u,
                userWorkspace + trailer->recomputeWorkspaceOffset, &recomputeTiling);
        }
    } else if (trailer->recompute.V == 256) {
        DispatchRecompute<half, float, 256>(
            k, v, beta, A, g, cuSeqlens, chunkIndices, w, u,
            userWorkspace + trailer->recomputeWorkspaceOffset, &recomputeTiling);
    } else {
        DispatchRecompute<half, float, 128>(
            k, v, beta, A, g, cuSeqlens, chunkIndices, w, u,
            userWorkspace + trailer->recomputeWorkspaceOffset, &recomputeTiling);
    }
    // FwdH performs a full MIX barrier after state initialization and before
    // either AIC or AIV consumes w/u, so a second stage barrier here is redundant.
    DispatchFwdH<TileShapes>(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
                             h, vNew, finalState, tiling, userWorkspace);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // FwdO consumes h/vNew across all FwdH task groups.  Drain the A5 H phase
    // globally before any group starts the dependent O phase.
    AscendC::SyncAll<false>();
#endif

    ChunkFwdOTilingData oTiling{};
    CopyOTiling(gmOTiling, oTiling);
    DispatchFwdO(q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, &oTiling);
}

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
