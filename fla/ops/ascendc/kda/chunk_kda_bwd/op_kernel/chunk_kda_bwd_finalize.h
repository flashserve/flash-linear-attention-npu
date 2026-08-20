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
    const ChunkKdaBwdCTilingData *tiling)
{
    if ASCEND_IS_AIC {
        {
            ChunkKdaBwdCCubeProcess<DataT, V_DIM> process(
                v, vNew, akk, h, dh, dvScan, dq, dk, dg, dAkk,
                cuSeqlens, chunkIndices, workspace);
            process.Init(*tiling);
            process.Process();
        }
        {
            ChunkKdaBwdCIntraCubeProcess process(
                cuSeqlens, chunkIndices, workspace);
            process.Init(*tiling);
            process.Process();
        }
    }
    if ASCEND_IS_AIV {
        {
            AscendC::TPipe pipe;
            ChunkKdaBwdCVectorProcess<DataT, V_DIM, BetaT> process(
                q, k, v, gk, beta, h, dh, dqRaw,
                dq, dk, dv, db, dg, dAkk, cuSeqlens, chunkIndices,
                workspace);
            process.Init(*tiling, &pipe);
            process.Process();
        }
        {
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
        {
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
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_H
