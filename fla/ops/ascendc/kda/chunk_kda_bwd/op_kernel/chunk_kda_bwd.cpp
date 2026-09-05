#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#include "chunk_kda_bwd_common.h"
#include "chunk_kda_bwd_prepare.h"
#include "chunk_kda_bwd_state_scan.h"
#include "chunk_kda_bwd_finalize.h"

namespace KDA {

template <typename DataT, uint32_t V_DIM, typename BetaT,
          bool SAFE_GATE, bool VARLEN_TND>
__aicore__ inline void RunChunkKdaBwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR gk,
    GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR qg, GM_ADDR kg,
    GM_ADDR vNew, GM_ADDR h, GM_ADDR dO, GM_ADDR rawG, GM_ADDR aLog,
    GM_ADDR dtBias, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dv0, GM_ADDR dqRaw, GM_ADDR dAqk, GM_ADDR dh,
    GM_ADDR dvScan, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR db,
    GM_ADDR dg, GM_ADDR dAkk, GM_ADDR dA, GM_ADDR dBias,
    GM_ADDR userWorkspace, const ChunkKdaBwdTilingData &tiling)
{
    const ChunkKdaBwdATilingData aTiling{
        tiling.kernelC.headNum,
        tiling.kernelC.seqlen,
        tiling.kernelC.chunkSize,
        tiling.kernelC.chunkNum,
        tiling.kernelC.chunkNumPerBatch,
        tiling.kernelC.isVarLen,
        tiling.kernelC.usedCoreNum};
    // Phase A is a true MIX stage on A5: AIC publishes raw dAqk into four
    // per-core slots and the paired AIVs finish scale+tril into final dAqk.
    bool aActive = false;
    if ASCEND_IS_AIC {
        aActive = AscendC::GetBlockIdx() <
            static_cast<uint32_t>(aTiling.usedCoreNum);
    }
    if ASCEND_IS_AIV {
        aActive = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum() <
            static_cast<uint32_t>(aTiling.usedCoreNum);
    }
    if (aActive) {
        RunChunkKdaBwdA<DataT, V_DIM>(
            aqk, vNew, h, dO, cuSeqlens, chunkIndices,
            dv0, dqRaw, dAqk,
            userWorkspace + tiling.kernelCWorkspaceOffset -
                static_cast<uint64_t>(tiling.kernelC.usedCoreNum) *
                    KDA_BWD_A_WORKSPACE_CORE_BYTES,
            aTiling, tiling.kernelC);
    }
    AscendC::SyncAll<false>();

    // Phase B is the PR291 sequence/head-window owner and retains its reverse
    // chunk recurrence.  Only the outer device entry and workspace base move.
    RunChunkKdaBwdB<DataT, V_DIM>(
        qg, kg, w, dO, dv0, gk, cuSeqlens, chunkIndices,
        dh, dvScan, userWorkspace + tiling.kernelBWorkspaceOffset,
        tiling.kernelB);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if (tiling.kernelC.useGateInKernel != 0 &&
        tiling.kernelC.deferGatePost != 0) {
        if ASCEND_IS_AIV {
            // The in-kernel Gate path overwrites every dA/dbias head from its
            // private UB accumulators.  Only a deferred post kernel needs the
            // outputs pre-zeroed here.
            RunChunkKdaBwdCInitGateOutputsA5(
                dA, dBias, tiling.kernelC);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
#endif
    AscendC::SyncAll<false>();

    // Phase C remaps blockIdx back to its original chunk/head owner.
    bool cActive = false;
    if ASCEND_IS_AIC {
        cActive = AscendC::GetBlockIdx() <
            static_cast<uint32_t>(tiling.kernelC.usedCoreNum);
    }
    if ASCEND_IS_AIV {
        cActive = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum() <
            static_cast<uint32_t>(tiling.kernelC.usedCoreNum);
    }
    // Every launched MIX block must reach the phase-C global barriers.  Keep
    // inactive blocks out of the computation, but not out of RunChunkKdaBwdC:
    // otherwise a short task list leaves active blocks waiting forever in
    // SyncAll.
    RunChunkKdaBwdC<DataT, V_DIM, BetaT, SAFE_GATE, VARLEN_TND>(
        q, k, v, vNew, gk, beta, akk, h, dh, dvScan,
        dqRaw, dAqk, cuSeqlens, chunkIndices, rawG, aLog, dtBias,
        dq, dk, dv, db, dg, dAkk, dA, dBias,
        userWorkspace + tiling.kernelCWorkspaceOffset, &tiling.kernelC,
        cActive);
}

} // namespace KDA

#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void chunk_kda_bwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR gk,
    GM_ADDR Aqk, GM_ADDR Akk, GM_ADDR w, GM_ADDR qg, GM_ADDR kg,
    GM_ADDR v_new, GM_ADDR h, GM_ADDR d_o, GM_ADDR raw_g, GM_ADDR a_log,
    GM_ADDR dt_bias, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR db,
    GM_ADDR dg, GM_ADDR dA, GM_ADDR dbias,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(6, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::AscendCUtils::SetOverflow(1);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        KDA::ChunkKdaBwdTilingData, tilingData, tiling);

    GM_ADDR dv0 = userWorkspace + tilingData.dv0Offset;
    GM_ADDR dq_raw = userWorkspace + tilingData.dqRawOffset;
    GM_ADDR dAqk = userWorkspace + tilingData.dAqkOffset;
    GM_ADDR dh = userWorkspace + tilingData.dhOffset;
    GM_ADDR dv_scan = userWorkspace + tilingData.dvScanOffset;
    GM_ADDR dAkk = userWorkspace + tilingData.dAkkOffset;

#define RUN_KDA_BWD(V_DIM, SAFE, VARLEN)                                      \
    KDA::RunChunkKdaBwd<DTYPE_Q, V_DIM, DTYPE_BETA, SAFE, VARLEN>(            \
        q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o, raw_g,        \
        a_log, dt_bias, cu_seqlens, chunk_indices, dv0, dq_raw, dAqk, dh,     \
        dv_scan, dq, dk, dv, db, dg, dAkk, dA, dbias, userWorkspace,          \
        tilingData)

    if (TILING_KEY_IS(1)) {
        RUN_KDA_BWD(128, true, false);
    } else if (TILING_KEY_IS(2)) {
        RUN_KDA_BWD(128, true, true);
    } else if (TILING_KEY_IS(3)) {
        RUN_KDA_BWD(128, false, false);
    } else if (TILING_KEY_IS(4)) {
        RUN_KDA_BWD(128, false, true);
    } else if (TILING_KEY_IS(5)) {
        RUN_KDA_BWD(256, true, false);
    } else if (TILING_KEY_IS(6)) {
        RUN_KDA_BWD(256, true, true);
    } else if (TILING_KEY_IS(7)) {
        RUN_KDA_BWD(256, false, false);
    } else if (TILING_KEY_IS(8)) {
        RUN_KDA_BWD(256, false, true);
    }

#undef RUN_KDA_BWD
}
#endif
