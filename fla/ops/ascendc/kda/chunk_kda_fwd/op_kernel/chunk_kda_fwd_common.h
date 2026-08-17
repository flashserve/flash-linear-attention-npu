#pragma once

#include "kernel_operator.h"
#include "chunk_kda_fwd_plan.h"
#include "../../kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_fwd_prepare.h"
#include "arch35/chunk_kda_fwd_post_wu.h"
#include "arch35/chunk_kda_fwd_finalize.h"
#else
#include "chunk_kda_fwd_prepare.h"
#include "chunk_kda_fwd_post_wu.h"
#include "chunk_kda_fwd_finalize.h"
#endif

#if __has_include("../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h")
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif
#else
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif
#endif

namespace KdaForward {

using namespace AscendC;

__aicore__ inline void ReleaseAicPipeReservedMmadEvents(TPipe &pipe)
{
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201) || \
    (defined(__CCE_AICORE__) && __CCE_AICORE__ == 2201)
    if ASCEND_IS_AIC {
        // A2/A3 TPipe::Init reserves M_MTE1 events 0/1/2. Manual MMAD
        // pipelines reuse those IDs and must consume the reserved credits first.
        pipe.DestroyWithoutPipeAll();
    }
#elif (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || \
      (defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
    if ASCEND_IS_AIC {
        // CANN 9.0 on c310 exposes only Destroy().
        pipe.Destroy();
    }
#else
    (void)pipe;
#endif
}

struct GateRuntimeTiling {
    int64_t batch;
    int64_t t;
    int64_t hv;
    int64_t k;
    int64_t rank;
    int64_t chunkSize;
    int64_t seqNum;
    int64_t hasCuSeqlens;
    int64_t hasALog;
    int64_t hasDtBias;
    int64_t dataType;
    int64_t useGateInKernel;
    int64_t safeGate;
    int64_t inputSequenceMajor;
    float lowerBound;
    int64_t usedCoreNum;
};

struct ChunkKdaFwdAddresses {
    GM_ADDR gk;
    GM_ADDR finalState;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR qg;
    GM_ADDR kg;
    GM_ADDR vNew;
    GM_ADDR h;
    GM_ADDR qgScaled;
    GM_ADDR uSeed;
};

struct FwdHTilingView {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t numSeqWorkspaceOffset;
    int64_t numChunksWorkspaceOffset;
    bool useCompactSequencePlan;
    uint32_t sequenceCount;
    uint32_t compactTotalChunks;
    uint32_t fwdUsedCoreNum;
    GM_ADDR compactPlan;
    uint32_t seqChunkOffsetsOffset;
};

template <typename TilingData>
__aicore__ inline FwdHTilingView MakeFwdHTiling(
    const TilingData &tiling, GM_ADDR compactPlan, bool tailOnly = false)
{
    const CompactSequencePlanView plan(compactPlan);
    const bool useCompactSequencePlan = !tailOnly && plan.IsValid();
    return {
        tiling.isVarLen ? tiling.seqNum : tiling.batch,
        tiling.seqlen,
        tiling.vHeadNum,
        tiling.vHeadNum,
        tiling.kHeadDim,
        tiling.vHeadDim,
        tiling.chunkSize,
        tailOnly || tiling.hasInitialState,
        tiling.storeFinalState,
        tailOnly ? 2 : (tiling.isVarLen ? 1 : 0),
        tiling.isVarLen ? 1 : tiling.batch,
        tiling.isVarLen ? tiling.seqNum : 1,
        tiling.vWorkspaceOffset,
        tiling.vUpdateWorkspaceOffset,
        tiling.kDecayWorkspaceOffset,
        tiling.hWorkspaceOffset,
        tiling.numSeqWorkspaceOffset,
        tiling.numChunksWorkspaceOffset,
        useCompactSequencePlan,
        useCompactSequencePlan ? plan.SequenceCount() : 0,
        static_cast<uint32_t>(tiling.totalChunks),
        useCompactSequencePlan ? plan.FwdUsedCoreNum() : 0,
        compactPlan,
        useCompactSequencePlan && tiling.isVarLen
            ? plan.SequenceChunkOffsetsOffset()
            : 0,
    };
}

__aicore__ inline GM_ADDR ResolveStorage(
    GM_ADDR output, GM_ADDR userWorkspace, int64_t offset, bool storeOutput)
{
    return storeOutput ? output : userWorkspace + offset;
}

template <typename TilingData>
__aicore__ inline ChunkKdaFwdAddresses ResolveAddresses(
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR w, GM_ADDR u, GM_ADDR qg,
    GM_ADDR kg, GM_ADDR vNew, GM_ADDR h, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    return {
        ResolveStorage(gk, userWorkspace, tiling.gkStorageOffset, tiling.storeGk),
        ResolveStorage(finalState, userWorkspace, tiling.finalStateStorageOffset,
                       tiling.storeFinalState),
        ResolveStorage(w, userWorkspace, tiling.wStorageOffset, tiling.storeW),
        ResolveStorage(u, userWorkspace, tiling.uStorageOffset, tiling.storeU),
        ResolveStorage(qg, userWorkspace, tiling.qgStorageOffset, tiling.storeQG),
        ResolveStorage(kg, userWorkspace, tiling.kgStorageOffset, tiling.storeKg),
        ResolveStorage(vNew, userWorkspace, tiling.vNewStorageOffset, tiling.storeVNew),
        ResolveStorage(h, userWorkspace, tiling.hStorageOffset, tiling.storeH),
        userWorkspace + tiling.qgScaledOffset,
        userWorkspace + tiling.outputScratchOffset,
    };
}

template <typename TilingData>
__aicore__ inline GateRuntimeTiling MakeGateTiling(const TilingData &tiling)
{
    return {
        tiling.batch,
        tiling.seqlen,
        tiling.vHeadNum,
        tiling.kHeadDim,
        tiling.inputRank,
        tiling.chunkSize,
        tiling.seqNum,
        tiling.isVarLen ? 1 : 0,
        tiling.hasALog ? 1 : 0,
        tiling.hasDtBias ? 1 : 0,
        tiling.gateDataType,
        tiling.useGateInKernel ? 1 : 0,
        tiling.safeGate ? 1 : 0,
        tiling.inputSequenceMajor ? 1 : 0,
        tiling.lowerBound,
        tiling.gateUsedCoreNum,
    };
}

template <bool USE_GATE_IN_KERNEL, bool SAFE_GATE, typename G_T,
          typename A_LOG_T, typename DT_BIAS_T, typename TilingData>
__aicore__ inline void RunGateCumsum(
    GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens,
    GM_ADDR gk, const TilingData &tiling, TPipe &pipe)
{
    if (tiling.computeGateInPrepare) {
        return;
    }
    if ASCEND_IS_AIV {
        static_assert(IsSameType<G_T, float>::value ||
                      IsSameType<G_T, bfloat16_t>::value,
                      "chunk_kda_fwd gate dtype must be FP32 or BF16");
        GateRuntimeTiling gateTiling = MakeGateTiling(tiling);
        KdaGateCumsum::RunKdaGateCumsum<G_T, USE_GATE_IN_KERNEL,
            SAFE_GATE, GateRuntimeTiling, A_LOG_T, DT_BIAS_T>(
            g, aLog, dtBias, cuSeqlens, gk, gateTiling, &pipe);
    }
}

template <bool SAFE_GATE, typename T, typename GK_T,
          typename BETA_T, typename A_LOG_T, typename DT_BIAS_T,
          typename TilingData, uint32_t COMPILE_BT, uint32_t COMPILE_K,
          uint32_t COMPILE_V>
__aicore__ inline void RunPrepareStage(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling, TPipe &pipe)
{
    GM_ADDR uSeed = (tiling.fusePostWu || tiling.fusePostWuIntoFwdH)
        ? addresses.u
        : addresses.uSeed;

    KdaPrepare::RunChunkKdaPrepare<SAFE_GATE, T, GK_T,
        BETA_T, A_LOG_T, DT_BIAS_T,
        TilingData, COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, addresses.gk, g, aLog, dtBias, beta, initialState,
        cuSeqlens, chunkIndices, compactPlan, aqk, akk, addresses.qg,
        addresses.qgScaled, addresses.w, uSeed, addresses.kg,
        userWorkspace, tiling, pipe, tiling.storeQG);
}

template <typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunPostWuStage(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling, TPipe &pipe)
{
    SyncAll<false>();
    pipe.Reset();

    GM_ADDR uSeed = (tiling.fusePostWu || tiling.fusePostWuIntoFwdH)
        ? addresses.u
        : addresses.uSeed;
    if (!tiling.fusePostWu && !tiling.fusePostWuIntoFwdH) {
        KdaPostWu::RunChunkKdaPostWu<T, GK_T, BETA_T>(
            q, k, v, addresses.gk, beta, initialState, cuSeqlens,
            chunkIndices, compactPlan, addresses.w, akk, uSeed,
            addresses.w, addresses.u, addresses.kg, addresses.vNew,
            userWorkspace, tiling, pipe);
        SyncAll<false>();
        pipe.Reset();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
        if (tiling.hasVarlenTail && tiling.chunkSize == 64 &&
            tiling.kHeadDim == 128 && tiling.vHeadDim == 128) {
            KdaPostWu::RunChunkKdaPostWuTailSeedCopy<T, GK_T, BETA_T>(
                cuSeqlens, chunkIndices, compactPlan, addresses.w,
                userWorkspace, tiling, pipe);
            SyncAll<false>();
            pipe.Reset();
            KdaPostWu::RunChunkKdaPostWuTail<T, GK_T, BETA_T>(
                q, k, v, addresses.gk, beta, initialState, cuSeqlens,
                chunkIndices, compactPlan, akk, uSeed, addresses.w, addresses.u,
                addresses.kg, userWorkspace, tiling, pipe);
            SyncAll<false>();
            pipe.Reset();
        }
#endif
    }
}

template <typename T, typename TileShapes, typename TilingData>
__aicore__ inline void RunFwdH(
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling, bool tailOnly = false)
{
    using FwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        T, float, float, float, TileShapes, true, false, true>;
    const auto fwdHTiling = MakeFwdHTiling(tiling, compactPlan, tailOnly);
    GM_ADDR stateInput = tailOnly ? addresses.finalState : initialState;
    FwdHKernel stateOp;
    stateOp.InitFromData(
        addresses.kg, addresses.w, addresses.u, addresses.gk, addresses.gk,
        stateInput, cuSeqlens, chunkIndices, addresses.h, addresses.vNew,
        addresses.finalState, fwdHTiling,
        userWorkspace + tiling.fwdHWorkspaceBaseOffset);
    stateOp.Process();
}

template <typename T, typename BETA_T, typename TilingData>
__aicore__ inline void RunGenericTailBackEnd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR attnOut,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling)
{
    if (tiling.vHeadDim > 128) {
        RunFwdH<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            addresses.finalState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, compactPlan, tiling, true);
    } else {
        RunFwdH<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            addresses.finalState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, compactPlan, tiling, true);
    }
    SyncAll<false>();
    TPipe pipe;
    ReleaseAicPipeReservedMmadEvents(pipe);
    KdaFinalize::RunChunkKdaOutput<T, float, BETA_T>(
        q, k, v, addresses.gk, beta, addresses.finalState, cuSeqlens,
        chunkIndices, compactPlan, addresses.qgScaled, aqk,
        addresses.vNew, addresses.h, attnOut, userWorkspace, tiling, pipe);
}

template <typename T, typename BETA_T, typename TilingData>
__aicore__ inline void RunGenericBackEnd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR attnOut,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling)
{
    if (tiling.vHeadDim > 128) {
        RunFwdH<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            initialState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, compactPlan, tiling);
    } else {
        RunFwdH<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            initialState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, compactPlan, tiling);
    }
    SyncAll<false>();
    TPipe pipe;
    ReleaseAicPipeReservedMmadEvents(pipe);
    KdaFinalize::RunChunkKdaOutput<T, float, BETA_T>(
        q, k, v, addresses.gk, beta, initialState, cuSeqlens,
        chunkIndices, compactPlan, addresses.qgScaled, aqk,
        addresses.vNew, addresses.h, attnOut, userWorkspace, tiling, pipe);
}

template <typename T, typename BETA_T, typename TilingData>
__aicore__ inline void RunGenericBackEnd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR attnOut,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling, TPipe &pipe)
{
    if (tiling.vHeadDim > 128) {
        RunFwdH<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            initialState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, compactPlan, tiling);
    } else {
        RunFwdH<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            initialState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, compactPlan, tiling);
    }
    SyncAll<false>();
    KdaFinalize::RunChunkKdaOutput<T, float, BETA_T>(
        q, k, v, addresses.gk, beta, initialState, cuSeqlens,
        chunkIndices, compactPlan, addresses.qgScaled, aqk,
        addresses.vNew, addresses.h, attnOut, userWorkspace, tiling, pipe);
}

} // namespace KdaForward
