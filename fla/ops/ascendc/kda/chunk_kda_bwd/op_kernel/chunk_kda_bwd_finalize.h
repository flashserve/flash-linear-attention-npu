#ifndef CHUNK_KDA_BWD_FINALIZE_H
#define CHUNK_KDA_BWD_FINALIZE_H

#include "chunk_kda_bwd_finalize_wy.h"
#include "chunk_kda_bwd_finalize_gate.h"
#include "chunk_kda_bwd_finalize_intra.h"

namespace KDA {

template <typename DataT, uint32_t V_DIM, typename BetaT,
          bool SAFE_GATE, bool VARLEN_TND>
__aicore__ inline void RunChunkKdaBwdC(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR gk,
    GM_ADDR beta, GM_ADDR akk, GM_ADDR h, GM_ADDR dh, GM_ADDR dvScan,
    GM_ADDR dqRaw, GM_ADDR dAqk, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR db, GM_ADDR dg,
    GM_ADDR dAkk, GM_ADDR dA, GM_ADDR dBias, GM_ADDR workspace,
    const ChunkKdaBwdCTilingData *tiling, bool active)
{
    if ASCEND_IS_AIC {
        if (active) {
            ChunkKdaBwdCCubeProcess<DataT, V_DIM, SAFE_GATE> process(
                v, vNew, akk, h, dh, dvScan, dq, dk, dg, dAkk,
                cuSeqlens, chunkIndices, workspace);
            process.Init(*tiling);
            process.Process();
        }
    }
    if ASCEND_IS_AIV {
        if (active) {
            AscendC::TPipe pipe;
            ChunkKdaBwdCVectorProcess<DataT, V_DIM, BetaT> process(
                q, k, v, gk, beta, h, dh, dqRaw,
                dq, dk, dv, db, dg, dAkk, cuSeqlens, chunkIndices,
                workspace);
            process.Init(*tiling, &pipe);
            process.Process();
        }
    }

    // WY materializes the base gradients in GM and Intra immediately reads
    // them back for the local correction.  Complete every WY store before
    // remapping the same buffers to the Intra owner.
    AscendC::SyncAll<false>();

    if ASCEND_IS_AIC {
        if (active) {
            ChunkKdaBwdCIntraCubeProcess process(
                cuSeqlens, chunkIndices, workspace);
            process.Init(*tiling);
            process.Process();
        }
    }
    if ASCEND_IS_AIV {
        if (active) {
            AscendC::TPipe pipe;
            ChunkKdaBwdCIntraVectorProcess<
                128, 64, SAFE_GATE, false, VARLEN_TND,
                DataT, BetaT, DTYPE_RAW_G> process(
                    q, k, gk, beta, dAqk, dAkk,
                    dqRaw, dq, dk, db, dg, dq, dk, db, dg,
                    cuSeqlens, chunkIndices, rawG, aLog, dtBias,
                    dA, dBias, workspace);
            process.Init(*tiling, &pipe);
            process.Process();
        }
    }

    // Intra writes dg by half rows on both AIV sub-blocks, whereas Gate
    // remaps the work to complete heads and updates dg in place.  Finish the
    // whole Intra phase before any core starts the Gate read/scan/write phase.
    AscendC::SyncAll<false>();

    if ASCEND_IS_AIV {
        if (active) {
            AscendC::TPipe pipe;
            ChunkKdaBwdCGateProcess<SAFE_GATE, DTYPE_RAW_G> process(
                dg, rawG, aLog, dtBias, dA, dBias,
                cuSeqlens, chunkIndices);
            if (tiling->deferGatePost == 0) {
                process.Init(*tiling, &pipe);
                process.Process();
            }
        }
    }

    // Gate is the final phase of the fused MIX kernel.  Do not return while
    // either AIV sub-block still has a queued dg/dA/dbias store: a following
    // invocation can reuse the same output allocation and observe that late
    // write as a stale cross-shape generation.
    if (tiling->seqlen < static_cast<int32_t>(kA5SharedGateMinSeqlen)) {
        AscendC::SyncAll<false>();
    }
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_H
