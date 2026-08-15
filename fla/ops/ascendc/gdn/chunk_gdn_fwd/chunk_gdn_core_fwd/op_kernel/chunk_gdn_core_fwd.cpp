/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gdn_core_fwd_struct.h"

#define GDN_CHUNK_RECOMPUTE_WU_FWD_HO_IMPL_ONLY
#include "../../chunk_recompute_wu_fwd_ho/op_kernel/chunk_recompute_wu_fwd_ho.cpp"
#undef GDN_CHUNK_RECOMPUTE_WU_FWD_HO_IMPL_ONLY

#define GDN_CHUNK_CUMSUM_KKT_SOLVE_IMPL_ONLY
#include "../../chunk_kkt_solve_tri/op_kernel/chunk_cumsum_kkt_solve_tri.cpp"
#undef GDN_CHUNK_CUMSUM_KKT_SOLVE_IMPL_ONLY

namespace GDN {
namespace {

constexpr uint32_t OWNER_MTE2_TO_V_EVENT = 0;
constexpr uint32_t OWNER_V_TO_MTE3_EVENT = 1;
constexpr uint32_t OWNER_MTE3_TO_V_EVENT = 2;
constexpr uint32_t UB_ALIGNMENT = 32;
constexpr uint32_t PHASE6_TILING_ALIGNMENT = 8;

__aicore__ inline uint64_t AlignPhase6(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline const __gm__ ChunkRecomputeWUFwdHOTrailer *GetPhase5Trailer(GM_ADDR tiling)
{
    const uint64_t oTilingOffset = AlignPhase6(
        sizeof(ChunkGatedDeltaRuleFwdHTilingData), PHASE6_TILING_ALIGNMENT);
    return reinterpret_cast<const __gm__ ChunkRecomputeWUFwdHOTrailer *>(
        tiling + oTilingOffset + sizeof(ChunkFwdOTilingData));
}

__aicore__ inline const __gm__ ChunkGdnCoreFwdTrailer *GetPhase6Trailer(GM_ADDR tiling)
{
    const uint64_t oTilingOffset = AlignPhase6(
        sizeof(ChunkGatedDeltaRuleFwdHTilingData), PHASE6_TILING_ALIGNMENT);
    const uint64_t phase5End = oTilingOffset + sizeof(ChunkFwdOTilingData) +
                               sizeof(ChunkRecomputeWUFwdHOTrailer);
    return reinterpret_cast<const __gm__ ChunkGdnCoreFwdTrailer *>(
        tiling + AlignPhase6(phase5End, PHASE6_TILING_ALIGNMENT));
}

__aicore__ inline void CopyAbcTiling(
    const __gm__ ChunkGdnCoreFwdAbcTiling *src, ChunkGdnCoreFwdAbcTiling &dst)
{
    dst.B = src->B;
    dst.Hk = src->Hk;
    dst.Hv = src->Hv;
    dst.hvPerHk = src->hvPerHk;
    dst.T = src->T;
    dst.K = src->K;
    dst.BT = src->BT;
    dst.NT = src->NT;
    dst.taskNum = src->taskNum;
    dst.usedAicNum = src->usedAicNum;
    dst.usedAivNum = src->usedAivNum;
    dst.btAlign = src->btAlign;
    dst.isVarlen = src->isVarlen;
    dst.scoreWorkspaceBytes = src->scoreWorkspaceBytes;
    dst.aWorkspaceBytes = src->aWorkspaceBytes;
    dst.solveWorkspacePerCoreBytes = src->solveWorkspacePerCoreBytes;
    dst.totalTiles = src->totalTiles;
    dst.matrixSize = src->matrixSize;
    dst.numHeads = src->numHeads;
    dst.seqLen = src->seqLen;
    dst.batchSize = src->batchSize;
    dst.isLower = src->isLower;
    dst.hasCuSeqlens = src->hasCuSeqlens;
    dst.tilesPerCore = src->tilesPerCore;
    dst.chunkSize = src->chunkSize;
    dst.numChunks = src->numChunks;
    dst.lastChunkValidSize = src->lastChunkValidSize;
    dst.totalChunks = src->totalChunks;
    dst.layoutMode = src->layoutMode;
    dst.dtypeMode = src->dtypeMode;
}

__aicore__ inline void WritePublicCumsumRows(
    GM_ADDR gCumsumBht, GM_ADDR gCumsumBth, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkGdnCoreFwdAbcTiling &tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (GetSubBlockIdx() != 0) {
        return;
    }

    const uint64_t coreGroup = static_cast<uint64_t>(GetBlockIdx()) /
                               static_cast<uint64_t>(GetSubBlockNum());
    const uint64_t taskBegin = coreGroup * static_cast<uint64_t>(tiling.tilesPerCore);
    const uint64_t taskEnd =
        (taskBegin + static_cast<uint64_t>(tiling.tilesPerCore)) < tiling.taskNum
            ? taskBegin + static_cast<uint64_t>(tiling.tilesPerCore)
            : tiling.taskNum;
    bool ownsChunk = false;
    for (uint64_t task = taskBegin; task < taskEnd; ++task) {
        const uint64_t head = (task / tiling.NT) % tiling.Hv;
        if (head == 0) {
            ownsChunk = true;
            break;
        }
    }
    if (!ownsChunk) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dataBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf;
    const uint32_t tileElements = static_cast<uint32_t>(tiling.BT * tiling.Hv);
    const uint32_t tileBytes = static_cast<uint32_t>(
        AlignPhase6(static_cast<uint64_t>(tileElements) * sizeof(float), UB_ALIGNMENT));
    pipe.InitBuffer(dataBuf, 2 * tileBytes);
    pipe.InitBuffer(offsetBuf, tileBytes);

    AscendC::LocalTensor<float> dataLocal = dataBuf.Get<float>();
    AscendC::LocalTensor<float> srcLocal = dataLocal;
    AscendC::LocalTensor<float> dstLocal = dataLocal[tileBytes / sizeof(float)];
    AscendC::LocalTensor<uint32_t> offsets = offsetBuf.Get<uint32_t>();
    for (uint32_t row = 0; row < static_cast<uint32_t>(tiling.BT); ++row) {
        for (uint32_t head = 0; head < static_cast<uint32_t>(tiling.Hv); ++head) {
            offsets.SetValue(row * static_cast<uint32_t>(tiling.Hv) + head,
                             (head * static_cast<uint32_t>(tiling.BT) + row) * sizeof(float));
        }
    }
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::GlobalTensor<float> input;
    AscendC::GlobalTensor<float> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gCumsumBht),
                          tiling.B * tiling.Hv * tiling.T);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gCumsumBth),
                           tiling.B * tiling.T * tiling.Hv);
    AscendC::GlobalTensor<int64_t> cu;
    AscendC::GlobalTensor<int64_t> indices;
    if (tiling.isVarlen != 0) {
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
    }
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

    for (uint64_t task = taskBegin; task < taskEnd; ++task) {
        const uint64_t chunk = task % tiling.NT;
        const uint64_t head = (task / tiling.NT) % tiling.Hv;
        const uint64_t batch = task / (tiling.Hv * tiling.NT);
        if (head != 0) {
            continue;
        }
        uint64_t rowStart = chunk * tiling.BT;
        uint64_t currentRows = tiling.BT;
        if (tiling.isVarlen != 0) {
            const int64_t sequence = indices.GetValue(chunk * 2);
            const int64_t localChunk = indices.GetValue(chunk * 2 + 1);
            const int64_t bos = cu.GetValue(sequence);
            const int64_t eos = cu.GetValue(sequence + 1);
            const int64_t varlenRowStart = bos + localChunk * static_cast<int64_t>(tiling.BT);
            const int64_t valid = eos - varlenRowStart;
            if (valid <= 0) {
                continue;
            }
            rowStart = static_cast<uint64_t>(varlenRowStart);
            currentRows = static_cast<uint64_t>(valid) < tiling.BT
                              ? static_cast<uint64_t>(valid) : tiling.BT;
        } else {
            const uint64_t remaining = tiling.T - rowStart;
            currentRows = remaining < tiling.BT ? remaining : tiling.BT;
        }
        AscendC::DataCopyExtParams headParams{
            1, static_cast<uint32_t>(currentRows * sizeof(float)), 0, 0, 0};
        for (uint64_t sourceHead = 0; sourceHead < tiling.Hv; ++sourceHead) {
            const uint64_t sourceOffset =
                ((batch * tiling.Hv + sourceHead) * tiling.T + rowStart);
            AscendC::DataCopyPad(srcLocal[sourceHead * tiling.BT], input[sourceOffset],
                                headParams, padParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(OWNER_MTE2_TO_V_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(OWNER_MTE2_TO_V_EVENT);

        const uint32_t elementCount = static_cast<uint32_t>(currentRows * tiling.Hv);
        AscendC::Gather(dstLocal, srcLocal, offsets, 0, elementCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(OWNER_V_TO_MTE3_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(OWNER_V_TO_MTE3_EVENT);
        AscendC::DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(elementCount * sizeof(float)), 0, 0, 0};
        const uint64_t outputOffset = (batch * tiling.T + rowStart) * tiling.Hv;
        AscendC::DataCopyPad(output[outputOffset], dstLocal, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OWNER_MTE3_TO_V_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OWNER_MTE3_TO_V_EVENT);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <typename InputT, typename TileShapes>
__aicore__ inline void RunPhase6(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR rawG, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR finalState, GM_ADDR gCumsumBth, GM_ADDR A, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const __gm__ ChunkRecomputeWUFwdHOTrailer *phase5 = GetPhase5Trailer(tiling);
    const __gm__ ChunkGdnCoreFwdTrailer *phase6 = GetPhase6Trailer(tiling);
    ChunkGdnCoreFwdAbcTiling abc{};
    CopyAbcTiling(&phase6->abc, abc);

    GM_ADDR scoreWorkspace = userWorkspace + phase6->scoreWorkspaceOffset;
    GM_ADDR aWorkspace = userWorkspace + phase6->aWorkspaceOffset;
    GM_ADDR solveWorkspaceBase = userWorkspace + phase6->solveWorkspaceOffset;
    GM_ADDR gCumsumBht = userWorkspace + phase6->gCumsumBhtOffset;
    uint64_t coreGroup = static_cast<uint64_t>(AscendC::GetBlockIdx());
    if ASCEND_IS_AIV {
        coreGroup /= static_cast<uint64_t>(AscendC::GetSubBlockNum());
    }
    GM_ADDR solveWorkspace =
        solveWorkspaceBase + coreGroup * abc.solveWorkspacePerCoreBytes;

    if ASCEND_IS_AIC {
        NsChunkKktCube::ChunkKktCube<InputT> kktCube;
        kktCube.Process(k, cuSeqlens, chunkIndices, scoreWorkspace, &abc);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SCORE_READY_FLAG);
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe kktPipe;
        NsChunkScaledDotKkt::ChunkScaledDotKkt<InputT, InputT> kkt;
        kkt.InitFusedCumsum(
            k, rawG, beta, cuSeqlens, chunkIndices, gCumsumBht, aWorkspace,
            scoreWorkspace, abc.B, abc.Hk, abc.Hv, abc.hvPerHk, abc.T, abc.K,
            abc.BT, abc.NT, abc.taskNum, abc.usedAicNum, abc.usedAivNum,
            abc.btAlign, abc.isVarlen, &kktPipe);
        AscendC::CrossCoreWaitFlag(SCORE_READY_FLAG);
        kkt.ProcessEpilogueForSolve(abc.tilesPerCore);
        kktPipe.Reset();
    }
    if (abc.BT == 64) {
        RunSolvePhase<InputT, 64>(aWorkspace, cuSeqlens, chunkIndices, A,
                                  solveWorkspace, &abc);
    } else {
        RunSolvePhase<InputT, 128>(aWorkspace, cuSeqlens, chunkIndices, A,
                                   solveWorkspace, &abc);
    }
    GM_ADDR w = userWorkspace + phase5->wIntermediateOffset;
    GM_ADDR u = userWorkspace + phase5->uIntermediateOffset;
    GM_ADDR h = userWorkspace + phase5->hIntermediateOffset;
    GM_ADDR vNew = userWorkspace + phase5->vNewIntermediateOffset;
    RecomputeWUFwdTilingData recomputeTiling{};
    CopyRecomputeTiling(&phase5->recompute, recomputeTiling);
    if (phase5->recompute.V == 256) {
        DispatchRecompute<InputT, float, 256, true>(
            k, v, beta, A, gCumsumBht, cuSeqlens, chunkIndices, w, u,
            userWorkspace + phase5->recomputeWorkspaceOffset, &recomputeTiling);
    } else {
        DispatchRecompute<InputT, float, 128, true>(
            k, v, beta, A, gCumsumBht, cuSeqlens, chunkIndices, w, u,
            userWorkspace + phase5->recomputeWorkspaceOffset, &recomputeTiling);
    }

    WritePublicCumsumRows(gCumsumBht, gCumsumBth, cuSeqlens, chunkIndices, abc);
    DispatchFwdH<TileShapes>(k, w, u, gCumsumBht, gk, initialState, cuSeqlens,
                             chunkIndices, h, vNew, finalState, tiling, userWorkspace);

    const uint64_t oTilingOffset =
        AlignPhase6(sizeof(ChunkGatedDeltaRuleFwdHTilingData), PHASE6_TILING_ALIGNMENT);
    const __gm__ ChunkFwdOTilingData *gmOTiling =
        reinterpret_cast<const __gm__ ChunkFwdOTilingData *>(tiling + oTilingOffset);
    ChunkFwdOTilingData oTiling{};
    CopyOTiling(gmOTiling, oTiling);
    DispatchFwdO(q, k, vNew, h, gCumsumBht, cuSeqlens, chunkIndices, o,
                 userWorkspace, &oTiling);
}

} // namespace
} // namespace GDN

extern "C" __global__ __aicore__ void chunk_gdn_core_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR a_storage, GM_ADDR raw_g,
    GM_ADDR gk, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR o, GM_ADDR final_state, GM_ADDR g_cumsum_bth, GM_ADDR A,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)a_storage;
    REGISTER_TILING_DEFAULT(GDN::ChunkGdnCoreFwdTrailer);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        const __gm__ GDN::ChunkGdnCoreFwdTrailer *phase6 = GDN::GetPhase6Trailer(tiling);
        if (phase6->abc.dtypeMode == 1) {
            GDN::RunPhase6<bfloat16_t, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
                q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
                o, final_state, g_cumsum_bth, A, workspace, tiling);
        } else {
            GDN::RunPhase6<half, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
                q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
                o, final_state, g_cumsum_bth, A, workspace, tiling);
        }
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        const __gm__ GDN::ChunkGdnCoreFwdTrailer *phase6 = GDN::GetPhase6Trailer(tiling);
        if (phase6->abc.dtypeMode == 1) {
            GDN::RunPhase6<bfloat16_t, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
                q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
                o, final_state, g_cumsum_bth, A, workspace, tiling);
        } else {
            GDN::RunPhase6<half, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
                q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
                o, final_state, g_cumsum_bth, A, workspace, tiling);
        }
    }
}
