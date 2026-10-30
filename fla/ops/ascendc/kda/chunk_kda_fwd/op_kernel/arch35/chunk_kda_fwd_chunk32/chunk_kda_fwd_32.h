#pragma once

#include "chunk_kda_fwd_32_common.h"

#if defined(__DAV_C310_CUBE__)
#include "chunk_kda_fwd_32_matmul.h"
#elif defined(__DAV_C310_VEC__)
#include "chunk_kda_fwd_32_vf.h"
#endif

namespace KdaForward {
namespace Chunk32 {

__aicore__ inline void Run(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR qgScaled, GM_ADDR uSeed, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)cuSeqlens; (void)chunkIndices; (void)gk; (void)aqk; (void)akk;
    (void)w; (void)u; (void)qg; (void)kg; (void)vNew; (void)h;
    (void)qgScaled; (void)uSeed;
    const __gm__ KdaFd32TilingData *td = reinterpret_cast<const __gm__ KdaFd32TilingData *>(tiling);
#if defined(__DAV_C310_CUBE__)
    (void)q; (void)k; (void)v; (void)g; (void)beta; (void)aLog; (void)dtBias;
    (void)attnOut; (void)initialState; (void)finalState;
    Kernel<bfloat16_t> kda;
    kda.kdaImp.tmp_gm = workspace;
    kda.task_count = td->taskCount;
    kda.block_num = td->blockNum;
    kda.is_varlen = td->varlenMode;
    kda.hv_real = td->hvNumReal;
    kda.cu_base = (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData));
    kda.kdaImp.Init();
    kda.Process((int)td->nChunks);
#elif defined(__DAV_C310_VEC__)
    if (td->varlenMode == 0 && td->seqLen % tokens_per_group == 0) {
        Kernel<bfloat16_t, false, false> kda;
        kda.kdaImp.Init(td->nChunks, td->scaleBf16, td->gateScaleBits,
                        aLog, g, q, k, v, beta, dtBias, workspace, attnOut,
                        initialState, finalState, td->hasInitialState, td->outputFinalState,
                        td->seqLen, td->qkGroup, td->taskCount, td->hvNumReal, td->blockNum,
                        td->varlenMode,
                        (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData)),
                        td->epsilonBits, td->gateFlags);
        kda.Process((int)td->nChunks);
    } else if (td->varlenMode == 0) {
        Kernel<bfloat16_t, false, true> kda;
        kda.kdaImp.Init(td->nChunks, td->scaleBf16, td->gateScaleBits,
                        aLog, g, q, k, v, beta, dtBias, workspace, attnOut,
                        initialState, finalState, td->hasInitialState, td->outputFinalState,
                        td->seqLen, td->qkGroup, td->taskCount, td->hvNumReal, td->blockNum,
                        td->varlenMode,
                        (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData)),
                        td->epsilonBits, td->gateFlags);
        kda.Process((int)td->nChunks);
    } else {
        Kernel<bfloat16_t, true, true> kda;
        kda.kdaImp.Init(td->nChunks, td->scaleBf16, td->gateScaleBits,
                        aLog, g, q, k, v, beta, dtBias, workspace, attnOut,
                        initialState, finalState, td->hasInitialState, td->outputFinalState,
                        td->seqLen, td->qkGroup, td->taskCount, td->hvNumReal, td->blockNum,
                        td->varlenMode,
                        (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData)),
                        td->epsilonBits, td->gateFlags);
        kda.Process((int)td->nChunks);
    }
#endif
}

} // namespace Chunk32
} // namespace KdaForward
