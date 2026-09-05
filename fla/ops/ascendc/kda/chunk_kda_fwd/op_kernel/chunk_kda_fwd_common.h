#pragma once

#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "./kernel_utils/vector/regbase.hpp"
#endif
#include "../../kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_fwd_prepare.h"
#include "arch35/chunk_kda_fwd_post_wu.h"
#include "arch35/chunk_kda_fwd_finalize.h"
#include "./kernel_utils/block/block_mmad_pingpong_tla_preloadA_l1B.hpp"
#else
#include "chunk_kda_fwd_prepare.h"
#include "chunk_kda_fwd_post_wu.h"
#include "chunk_kda_fwd_finalize.h"
#endif

#include "fwd_h/chunk_kda_fwd_h_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "fwd_h/arch35/gemm/kernel/kda_fwd_h_kernel.hpp"
#else
#include "fwd_h/gemm/kernel/kda_fwd_h_kernel.hpp"
#endif

namespace KdaForward {

using namespace AscendC;

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
};

template <bool SAFE_GATE, typename T, uint32_t COMPILE_BT,
          uint32_t COMPILE_K, uint32_t COMPILE_V>
struct KgResidualPolicy {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr bool compileEnabled =
        !SAFE_GATE && IsSameType<T, bfloat16_t>::value &&
        COMPILE_BT == 0 && COMPILE_K == 0 && COMPILE_V == 0;
#else
    static constexpr bool compileEnabled = false;
#endif

    template <typename TilingData>
    __aicore__ static inline bool IsEnabled(const TilingData &tiling)
    {
        if constexpr (!compileEnabled) {
            return false;
        }
        return tiling.kHeadDim == 128 &&
               tiling.vHeadDim >= tiling.kHeadDim &&
               !tiling.fusePostWu && !tiling.fusePostWuIntoFwdH;
    }
};

template <typename TilingData>
__aicore__ inline FwdHTilingView MakeFwdHTiling(const TilingData &tiling)
{
    return {
        tiling.isVarLen ? tiling.seqNum : tiling.batch,
        tiling.seqlen,
        tiling.vHeadNum,
        tiling.vHeadNum,
        tiling.kHeadDim,
        tiling.vHeadDim,
        tiling.chunkSize,
        tiling.hasInitialState,
        tiling.storeFinalState,
        tiling.isVarLen ? 1 : 0,
        tiling.isVarLen ? 1 : tiling.batch,
        tiling.isVarLen ? tiling.seqNum : 1,
        tiling.vWorkspaceOffset,
        tiling.vUpdateWorkspaceOffset,
        tiling.kDecayWorkspaceOffset,
        tiling.hWorkspaceOffset,
        tiling.numSeqWorkspaceOffset,
        tiling.numChunksWorkspaceOffset,
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

template <typename TilingData>
__aicore__ inline void RunGateCumsum(
    GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens,
    GM_ADDR gk, const TilingData &tiling)
{
    if (tiling.computeGateInPrepare) {
        return;
    }
    if ASCEND_IS_AIV {
        GateRuntimeTiling gateTiling = MakeGateTiling(tiling);
        TPipe gatePipe;
        if (gateTiling.dataType == 2) {
            KdaGateCumsum::DispatchKdaGateCumsum<float>(
                g, aLog, dtBias, cuSeqlens, gk, gateTiling, &gatePipe);
        } else if (gateTiling.dataType == 1) {
            KdaGateCumsum::DispatchKdaGateCumsum<bfloat16_t>(
                g, aLog, dtBias, cuSeqlens, gk, gateTiling, &gatePipe);
        } else {
            KdaGateCumsum::DispatchKdaGateCumsum<half>(
                g, aLog, dtBias, cuSeqlens, gk, gateTiling, &gatePipe);
        }
    }
}

template <bool SAFE_GATE, typename T, typename GK_T, typename BETA_T,
          typename TilingData, uint32_t COMPILE_BT, uint32_t COMPILE_K,
          uint32_t COMPILE_V>
__aicore__ inline void RunPostWu(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR wSeed, GM_ADDR akk, GM_ADDR uSeed, GM_ADDR w, GM_ADDR u,
    GM_ADDR kg, GM_ADDR vNew, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ResidualPolicy = KgResidualPolicy<
        SAFE_GATE, T, COMPILE_BT, COMPILE_K, COMPILE_V>;
    if constexpr (ResidualPolicy::compileEnabled) {
        if (ResidualPolicy::IsEnabled(tiling)) {
            KdaPostWu::RunChunkKdaPostWu<true, T, GK_T, BETA_T>(
                q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                wSeed, akk, uSeed, w, u, kg, vNew, userWorkspace, tiling,
                pipe);
            return;
        }
    }
    KdaPostWu::RunChunkKdaPostWu<false, T, GK_T, BETA_T>(
        q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices, wSeed,
        akk, uSeed, w, u, kg, vNew, userWorkspace, tiling, pipe);
#else
    KdaPostWu::RunChunkKdaPostWu<T, GK_T, BETA_T>(
        q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices, wSeed,
        akk, uSeed, w, u, kg, vNew, userWorkspace, tiling, pipe);
#endif
}

template <bool SAFE_GATE, typename T, typename GK_T, typename BETA_T,
          typename TilingData, uint32_t COMPILE_BT, uint32_t COMPILE_K,
          uint32_t COMPILE_V>
__aicore__ inline void RunFrontEnd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
    RunGateCumsum(g, aLog, dtBias, cuSeqlens, addresses.gk, tiling);
    if (!tiling.computeGateInPrepare) {
        SyncAll<false>();
    }
    GM_ADDR uSeed = (tiling.fusePostWu || tiling.fusePostWuIntoFwdH)
        ? addresses.u
        : addresses.uSeed;

    KdaPrepare::RunChunkKdaPrepare<SAFE_GATE, T, GK_T, BETA_T,
        TilingData, COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, addresses.gk, g, aLog, dtBias, beta, initialState,
        cuSeqlens, chunkIndices, aqk, akk, addresses.qg,
        addresses.qgScaled, addresses.w, uSeed, addresses.kg,
        userWorkspace, tiling, pipe, tiling.storeQG);
    SyncAll<false>();
    pipe.Reset();

    if (!tiling.fusePostWu && !tiling.fusePostWuIntoFwdH) {
        RunPostWu<SAFE_GATE, T, GK_T, BETA_T, TilingData, COMPILE_BT,
                  COMPILE_K, COMPILE_V>(
            q, k, v, addresses.gk, beta, initialState, cuSeqlens,
            chunkIndices, addresses.w, akk, uSeed,
            addresses.w, addresses.u, addresses.kg, addresses.vNew,
            userWorkspace, tiling, pipe);
        SyncAll<false>();
        pipe.Reset();
    }
}

template <bool HI_LO_C2, typename T, typename TileShapes,
          typename TilingData>
__aicore__ inline void RunFwdHImpl(
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using FwdHKernel = Catlass::Gemm::Kernel::KDAFwdHKernel<
        T, float, float, float, TileShapes, true, false, true, HI_LO_C2>;
#else
    static_assert(!HI_LO_C2, "HI_LO_C2 is only supported on Ascend950");
    using FwdHKernel = Catlass::Gemm::Kernel::KDAFwdHKernel<
        T, float, float, float, TileShapes, true, false, true>;
#endif
    const auto fwdHTiling = MakeFwdHTiling(tiling);
    FwdHKernel stateOp;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    stateOp.InitFromData(
        addresses.kg, addresses.w, addresses.u, addresses.gk, addresses.gk,
        initialState, cuSeqlens, chunkIndices, addresses.h, addresses.vNew,
        addresses.finalState, fwdHTiling,
        userWorkspace + tiling.fwdHWorkspaceBaseOffset, addresses.uSeed);
#else
    stateOp.InitFromData(
        addresses.kg, addresses.w, addresses.u, addresses.gk, addresses.gk,
        initialState, cuSeqlens, chunkIndices, addresses.h, addresses.vNew,
        addresses.finalState, fwdHTiling,
        userWorkspace + tiling.fwdHWorkspaceBaseOffset);
#endif
    stateOp.Process();
}

template <bool SAFE_GATE, typename T, typename TileShapes,
          typename TilingData, uint32_t COMPILE_BT, uint32_t COMPILE_K,
          uint32_t COMPILE_V>
__aicore__ inline void RunFwdH(
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    using ResidualPolicy = KgResidualPolicy<
        SAFE_GATE, T, COMPILE_BT, COMPILE_K, COMPILE_V>;
    if constexpr (ResidualPolicy::compileEnabled) {
        if (ResidualPolicy::IsEnabled(tiling)) {
            RunFwdHImpl<true, T, TileShapes>(
                initialState, cuSeqlens, chunkIndices, addresses,
                userWorkspace, tiling);
            return;
        }
    }
    RunFwdHImpl<false, T, TileShapes>(
        initialState, cuSeqlens, chunkIndices, addresses, userWorkspace,
        tiling);
}

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void RunGenericBackEnd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR attnOut,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    if (tiling.vHeadDim > 128) {
        RunFwdH<SAFE_GATE, T,
                Catlass::Gemm::Kernel::KDAFwdHTileShapes256, TilingData,
                COMPILE_BT, COMPILE_K, COMPILE_V>(
            initialState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, tiling);
    } else {
        RunFwdH<SAFE_GATE, T,
                Catlass::Gemm::Kernel::KDAFwdHTileShapes128, TilingData,
                COMPILE_BT, COMPILE_K, COMPILE_V>(
            initialState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, tiling);
    }
    SyncAll<false>();
    TPipe pipe;
    KdaFinalize::RunChunkKdaOutput<T, float, BETA_T>(
        q, k, v, addresses.gk, beta, initialState, cuSeqlens,
        chunkIndices, addresses.qgScaled, aqk,
        addresses.vNew, addresses.h, attnOut, userWorkspace, tiling, pipe);
}

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void RunGenericBackEnd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR attnOut,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
    if (tiling.vHeadDim > 128) {
        RunFwdH<SAFE_GATE, T,
                Catlass::Gemm::Kernel::KDAFwdHTileShapes256, TilingData,
                COMPILE_BT, COMPILE_K, COMPILE_V>(
            initialState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, tiling);
    } else {
        RunFwdH<SAFE_GATE, T,
                Catlass::Gemm::Kernel::KDAFwdHTileShapes128, TilingData,
                COMPILE_BT, COMPILE_K, COMPILE_V>(
            initialState, cuSeqlens, chunkIndices, addresses,
            userWorkspace, tiling);
    }
    SyncAll<false>();
    KdaFinalize::RunChunkKdaOutput<T, float, BETA_T>(
        q, k, v, addresses.gk, beta, initialState, cuSeqlens,
        chunkIndices, addresses.qgScaled, aqk,
        addresses.vNew, addresses.h, attnOut, userWorkspace, tiling, pipe);
}

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void RunGeneric(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    const auto addresses = ResolveAddresses(
        finalState, gk, w, u, qg, kg, vNew, h, userWorkspace, tiling);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    TPipe pipe;
    RunFrontEnd<SAFE_GATE, T, float, BETA_T, TilingData,
        COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, aqk, akk, addresses, userWorkspace, tiling, pipe);
    if (!tiling.isVarLen && tiling.seqlen % tiling.chunkSize == 0) {
        pipe.Destroy();
        RunGenericBackEnd<SAFE_GATE, T, BETA_T, TilingData,
                          COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
            attnOut, addresses, userWorkspace, tiling);
    } else {
        RunGenericBackEnd<SAFE_GATE, T, BETA_T, TilingData,
                          COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
            attnOut, addresses, userWorkspace, tiling, pipe);
    }
#else
    {
        TPipe pipe;
        RunFrontEnd<SAFE_GATE, T, float, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, aqk, akk, addresses, userWorkspace, tiling, pipe);
    }
    RunGenericBackEnd<SAFE_GATE, T, BETA_T, TilingData,
                      COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
        attnOut, addresses, userWorkspace, tiling);
#endif
}

} // namespace KdaForward
