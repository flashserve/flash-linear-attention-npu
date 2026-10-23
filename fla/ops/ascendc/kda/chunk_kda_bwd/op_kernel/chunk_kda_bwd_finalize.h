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
    bool localBatches = false;
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
    if constexpr (SAFE_GATE && !VARLEN_TND && V_DIM == 128 && AscendC::IsSameType<DataT, bfloat16_t>::value &&
                  AscendC::IsSameType<BetaT, bfloat16_t>::value &&
                  AscendC::IsSameType<DTYPE_RAW_G, float>::value) {
        localBatches = tiling->isVarLen == 0 && tiling->batch == 1 &&
            tiling->headNum == 32 && tiling->keyDim == 128 &&
            tiling->valueDim == 128 && tiling->deferGatePost != 0;
    }
    if (localBatches && active) {
        uint32_t core = 0;
        if ASCEND_IS_AIC { core = AscendC::GetBlockIdx(); }
        if ASCEND_IS_AIV { core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(); }
        const uint32_t cores = static_cast<uint32_t>(tiling->usedCoreNum);
        const uint64_t groups = static_cast<uint64_t>(tiling->chunkNum) * 16U;
        const uint64_t rounds = core < groups ? (groups - 1U - core) / cores + 1U : 0U;
        for (uint64_t first = 0; first < rounds; first += 8U) {
            const uint64_t count = rounds - first < 8U ? rounds - first : 8U;
            if ASCEND_IS_AIC {
                ChunkKdaBwdCCubeProcess<DataT, V_DIM, SAFE_GATE> process(
                    v, vNew, akk, h, dh, dvScan, dq, dk, dg, dAkk,
                    cuSeqlens, chunkIndices, workspace);
                process.Init(*tiling);
                process.Process(first, count);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(1);
            }
            if ASCEND_IS_AIV {
                {
                    AscendC::TPipe pipe;
                    ChunkKdaBwdCVectorProcess<DataT, V_DIM, BetaT> process(
                        q, k, v, gk, beta, h, dh, dqRaw, dq, dk, dv,
                        db, dg, dAkk, cuSeqlens, chunkIndices, workspace);
                    process.Init(*tiling, &pipe);
                    process.Process(first, count);
                }
                AscendC::CrossCoreWaitFlag(1);
            }
            if ASCEND_IS_AIC {
                ChunkKdaBwdCIntraCubeProcess process(cuSeqlens, chunkIndices, workspace);
                process.Init(*tiling);
                process.Process(first, count);
                AscendC::CrossCoreWaitFlag(1);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0);
            }
            if ASCEND_IS_AIV {
                {
                    AscendC::TPipe pipe;
                    ChunkKdaBwdCIntraVectorProcess<
                        128, 64, SAFE_GATE, false, VARLEN_TND, DataT, BetaT, DTYPE_RAW_G, 3, 1, 7, 1> process(
                        q, k, gk, beta, dAqk, dAkk, dqRaw, dq, dk, db, dg,
                        dq, dk, db, dg, cuSeqlens, chunkIndices, rawG, aLog, dtBias,
                        dA, dBias, workspace);
                    process.Init(*tiling, &pipe);
                    process.Process(first, count);
                    Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
                    AscendC::CrossCoreWaitFlag(0);
                }
            }
        }
    }
#endif
    if (!localBatches) {
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
    bool needsFinalSync =
        tiling->seqlen < static_cast<int32_t>(kA5SharedGateMinSeqlen);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    needsFinalSync = needsFinalSync ||
        (tiling->useGateInKernel != 0 &&
         !CanUseA5SharedGateAnchor(*tiling));
#endif
    if (needsFinalSync) {
        AscendC::SyncAll<false>();
    }
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_H
