#ifndef CHUNK_KDA_BWD_FINALIZE_H
#define CHUNK_KDA_BWD_FINALIZE_H

#include "chunk_kda_bwd_finalize_wy.h"
#include "chunk_kda_bwd_finalize_gate.h"
#include "chunk_kda_bwd_finalize_intra.h"

// FC-S1: compile-time switch for the PhaseC batch=2 lockstep orchestration.
#ifndef KDA_FC_BATCH_LOCKSTEP
#define KDA_FC_BATCH_LOCKSTEP 1
#endif

namespace KDA {

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310 && KDA_FC_BATCH_LOCKSTEP
// Batch credits live in the intra_block (0x4) flag space on IDs measured free
// of WY/Intra traffic: credit A (AIC->AIV) on 7, credit B (AIV->AIC) on 10.
// AIV1 is addressed with the +16 sub-block stride.
constexpr uint32_t kFcBatchCreditA = 7;
constexpr uint32_t kFcBatchCreditB = 10;
constexpr uint32_t kFcBatchSubBlockStride = 16;
constexpr uint64_t kFcBatchSize = 2;
#endif

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
    // Runtime gate: the A5 dense16 fused form only (useGateInKernel set, dense,
    // seqlen a multiple of the chunk).  Every other shape keeps the phased
    // structure below byte for byte.  The predicate is tiling-derived, so all
    // launched blocks pick the same path and still meet at the phase barriers.
    bool fcBatchLockstep = false;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310 && KDA_FC_BATCH_LOCKSTEP
    fcBatchLockstep = CanUseA5DenseRow16Intra(*tiling);
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310 && KDA_FC_BATCH_LOCKSTEP
    if (fcBatchLockstep && active) {
        // Batch = up to two (chunk, two-head window) tasks of this core.  Per
        // batch the owner order is WY -> Intra; batch generations of the
        // shared workspace slots are serialized by credits A/B:
        //   AIC: WY-Cube(b) -> Set(A) -> Intra-Cube(b) -> Wait(B) -> Set(A)
        //   AIV: WY-Vector(b) -> Wait(A) -> [MTE3->MTE2 fence] ->
        //        Intra-Vector(b) -> Barrier+Set(B) -> Wait(A)
        uint32_t core = 0;
        if ASCEND_IS_AIC {
            core = AscendC::GetBlockIdx();
        }
        if ASCEND_IS_AIV {
            core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        }
        const uint32_t cores = static_cast<uint32_t>(tiling->usedCoreNum);
        const uint32_t headWindows =
            (static_cast<uint32_t>(tiling->headNum) + 1U) / 2U;
        const uint64_t groups =
            static_cast<uint64_t>(tiling->chunkNum) * headWindows;
        const uint64_t rounds =
            core < groups ? (groups - 1U - core) / cores + 1U : 0U;
        for (uint64_t first = 0; first < rounds; first += kFcBatchSize) {
            const uint64_t count = rounds - first < kFcBatchSize ?
                rounds - first : kFcBatchSize;
            if ASCEND_IS_AIC {
                ChunkKdaBwdCCubeProcess<DataT, V_DIM, SAFE_GATE> process(
                    v, vNew, akk, h, dh, dvScan, dq, dk, dg, dAkk,
                    cuSeqlens, chunkIndices, workspace);
                process.Init(*tiling);
                process.Process(first, count);
                // Credit A: the batch's WY generations are fully retired (the
                // cube drain consumed every AIV taskDone), so Intra may touch
                // the shared workspace slots.
                AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(kFcBatchCreditA);
                AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
                    kFcBatchCreditA + kFcBatchSubBlockStride);
            }
            if ASCEND_IS_AIV {
                const uint32_t subBlock = AscendC::GetSubBlockIdx();
                AscendC::TPipe pipe;
                ChunkKdaBwdCVectorProcess<DataT, V_DIM, BetaT> process(
                    q, k, v, gk, beta, h, dh, dqRaw,
                    dq, dk, dv, db, dg, dAkk, cuSeqlens, chunkIndices,
                    workspace);
                process.Init(*tiling, &pipe);
                process.Process(first, count);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    kFcBatchCreditA + kFcBatchSubBlockStride * subBlock);
                // Batch-end MTE3->MTE2 fence: this AIV's WY dq/dk/dg GM
                // stores must be ordered before its own Intra loads.  The
                // phased form took this ordering from SyncAll; shrink the
                // write-read distance to this batch.
                const event_t wyStoreFence = static_cast<event_t>(
                    pipe.AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(wyStoreFence);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(wyStoreFence);
                pipe.ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(
                    wyStoreFence);
            }
            if ASCEND_IS_AIC {
                ChunkKdaBwdCIntraCubeProcess process(
                    cuSeqlens, chunkIndices, workspace);
                process.Init(*tiling);
                process.Process(first, count);
                // Credit B: collect both AIV sub-blocks' Intra-Vector
                // completion; the next batch may not overwrite the workspace
                // slots they still read.
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(kFcBatchCreditB);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(
                    kFcBatchCreditB + kFcBatchSubBlockStride);
                // The ready flags stall FIX only.  Make the next batch's MTE2
                // loads depend on the wait, mirroring WaitVectorStage.
                constexpr int32_t kFixToMte2Event = 0;
                AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(
                    kFixToMte2Event);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(
                    kFixToMte2Event);
                AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(kFcBatchCreditA);
                AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
                    kFcBatchCreditA + kFcBatchSubBlockStride);
            }
            if ASCEND_IS_AIV {
                const uint32_t subBlock = AscendC::GetSubBlockIdx();
                AscendC::TPipe pipe;
                ChunkKdaBwdCIntraVectorProcess<
                    128, 64, SAFE_GATE, false, VARLEN_TND,
                    DataT, BetaT, DTYPE_RAW_G> process(
                        q, k, gk, beta, dAqk, dAkk,
                        dqRaw, dq, dk, db, dg, dq, dk, db, dg,
                        cuSeqlens, chunkIndices, rawG, aLog, dtBias,
                        dA, dBias, workspace);
                process.Init(*tiling, &pipe);
                process.Process(first, count);
                // Publish B on MTE3 after joining the paired sub-block, so
                // the credit cannot overtake the batch's final GM stores.
                Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                    kFcBatchCreditB + kFcBatchSubBlockStride * subBlock);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    kFcBatchCreditA + kFcBatchSubBlockStride * subBlock);
            }
        }
    }
#endif
    if (!fcBatchLockstep) {
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
