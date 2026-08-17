#pragma once

#include "../chunk_kda_fwd_common.h"
#include "chunk_kda_fwd_fwd_h.h"

namespace KdaForward::arch35 {

template <typename T, typename BETA_T, typename TilingData>
__aicore__ inline void RunBackEnd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut, GM_ADDR aqk,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling, AscendC::TPipe &pipe)
{
    if (tiling.useDenseFwdH) {
        ChunkKdaFwdFwdH<T, float, TilingData> fwdH;
        fwdH.Init(
            addresses.gk, initialState, addresses.finalState,
            addresses.w, addresses.u, addresses.kg,
            addresses.vNew, addresses.h, cuSeqlens,
            compactPlan, tiling);
        fwdH.Process();
        SyncAll<false>();
        pipe.Reset();
        KdaFinalize::RunChunkKdaOutput<T, float, BETA_T>(
            q, k, v, addresses.gk, beta, initialState, cuSeqlens,
            chunkIndices, compactPlan, addresses.qgScaled, aqk,
            addresses.vNew, addresses.h, attnOut, userWorkspace, tiling, pipe);
        return;
    }

    const int64_t fwdHTaskCount =
        (tiling.isVarLen ? tiling.seqNum : tiling.batch) * tiling.vHeadNum;
    const bool isolateGenericBackEnd =
        (!tiling.isVarLen && tiling.seqlen % tiling.chunkSize == 0) ||
        fwdHTaskCount > tiling.prepareUsedCoreNum;
    if (isolateGenericBackEnd) {
        if ASCEND_IS_AIV {
            pipe.Destroy();
        }
        RunGenericBackEnd<T, BETA_T, TilingData>(
            q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
            attnOut, addresses, userWorkspace, compactPlan, tiling);
    } else {
        RunGenericBackEnd<T, BETA_T, TilingData>(
            q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
            attnOut, addresses, userWorkspace, compactPlan, tiling, pipe);
    }
}

} // namespace KdaForward::arch35
