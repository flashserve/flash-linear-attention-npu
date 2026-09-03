/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_arch35_struct.h"

#include "../gated_delta_rule_state_update_output/chunk_gated_delta_rule_state_update_output.cpp"
#include "../coefficient_generation/chunk_gated_delta_rule_coefficient_generation.cpp"

#define GDN_CHUNK_LOCAL_CUMSUM_IMPL_ONLY
#include "../operators/chunk_local_cumsum/op_kernel/chunk_local_cumsum.cpp"
#undef GDN_CHUNK_LOCAL_CUMSUM_IMPL_ONLY

namespace GDN {
namespace {

constexpr uint32_t OWNER_MTE2_TO_V_EVENT = 0;
constexpr uint32_t OWNER_V_TO_MTE3_EVENT = 1;
constexpr uint32_t OWNER_MTE3_TO_V_EVENT = 2;
constexpr uint32_t UB_ALIGNMENT = 32;
constexpr uint32_t PHASE6_TILING_ALIGNMENT = 8;
constexpr uint64_t PHASE6_SOLVE_DONE_FLAG = 5;
constexpr int64_t PHASE6_CUMSUM_FAST_BUFFER_LIMIT = 160 * 1024;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
constexpr AscendC::SyncAllConfig PHASE6_HO_SYNC_CONFIG = {PIPE_MTE3, PIPE_MTE2};
#endif

__aicore__ inline uint64_t AlignPhase6(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline const __gm__ ChunkGatedDeltaRuleStateOutputTrailer *GetStateOutputTrailer(GM_ADDR tiling)
{
    const uint64_t oTilingOffset = AlignPhase6(
        sizeof(ChunkGatedDeltaRuleFwdHTilingData), PHASE6_TILING_ALIGNMENT);
    return reinterpret_cast<const __gm__ ChunkGatedDeltaRuleStateOutputTrailer *>(
        tiling + oTilingOffset + sizeof(ChunkFwdOTilingData));
}

__aicore__ inline const __gm__ Arch35ChunkGatedDeltaRuleFwdTrailer *GetPhase6Trailer(GM_ADDR tiling)
{
    const uint64_t oTilingOffset = AlignPhase6(
        sizeof(ChunkGatedDeltaRuleFwdHTilingData), PHASE6_TILING_ALIGNMENT);
    const uint64_t stateOutputTilingEnd = oTilingOffset + sizeof(ChunkFwdOTilingData) +
                                          sizeof(ChunkGatedDeltaRuleStateOutputTrailer);
    return reinterpret_cast<const __gm__ Arch35ChunkGatedDeltaRuleFwdTrailer *>(
        tiling + AlignPhase6(stateOutputTilingEnd, PHASE6_TILING_ALIGNMENT));
}

__aicore__ inline void CopyCoefficientTiling(
    const __gm__ Arch35ChunkGatedDeltaRuleCoefficientTiling *src, Arch35ChunkGatedDeltaRuleCoefficientTiling &dst)
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
    dst.totalTokens = src->totalTokens;

    static_assert(sizeof(TCubeTiling) % sizeof(uint32_t) == 0,
                  "TCubeTiling must be copied as complete 32-bit words");
    const __gm__ uint32_t *cubeSrc =
        reinterpret_cast<const __gm__ uint32_t *>(&src->cubeTilingData);
    uint32_t *cubeDst = reinterpret_cast<uint32_t *>(&dst.cubeTilingData);
    constexpr uint32_t cubeWordCount = sizeof(TCubeTiling) / sizeof(uint32_t);
    for (uint32_t index = 0; index < cubeWordCount; ++index) {
        cubeDst[index] = cubeSrc[index];
    }
}

__aicore__ inline void WritePublicCumsumRows(
    GM_ADDR gCumsumBht, GM_ADDR gCumsumBth, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const Arch35ChunkGatedDeltaRuleCoefficientTiling &tiling)
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

__aicore__ inline void RunPhase6Cumsum(
    GM_ADDR rawG, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR gCumsumBht,
    const Arch35ChunkGatedDeltaRuleCoefficientTiling &tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    // In a MIX launch, the AIV block index is already expanded across both
    // subblocks. Keep both subblocks so the standalone scheduler covers every
    // vector task exactly once.

    ChunkLocalCumsumTilingData cumsumTiling{};
    cumsumTiling.b = static_cast<int64_t>(tiling.B);
    cumsumTiling.t = static_cast<int64_t>(tiling.T);
    cumsumTiling.h = static_cast<int64_t>(tiling.Hv);
    cumsumTiling.chunkSize = static_cast<int64_t>(tiling.BT);
    cumsumTiling.blockT = static_cast<int64_t>(tiling.BT);
    cumsumTiling.numBlocks = static_cast<int64_t>(tiling.NT);
    cumsumTiling.seqNum = 0;
    cumsumTiling.totalElements = static_cast<int64_t>(tiling.B * tiling.Hv * tiling.T);
    cumsumTiling.isVarlen = static_cast<int64_t>(tiling.isVarlen);
    cumsumTiling.reverse = 0;
    cumsumTiling.headFirst = 1;
    cumsumTiling.optimizedHeadFirst = 1;
    cumsumTiling.varlenSeqTask = 0;
    cumsumTiling.enableCumSumFastPath = 1;
    cumsumTiling.fastBufferLimit = PHASE6_CUMSUM_FAST_BUFFER_LIMIT;
    cumsumTiling.inputDtype = 0;
    cumsumTiling.outputDtype = 0;
    cumsumTiling.scale = 1.0f;

    ChunkLocalCumsumKernel<float, float> cumsum;
    cumsum.Init(rawG, cuSeqlens, chunkIndices, gCumsumBht, &cumsumTiling);
    cumsum.Process();
}

template <typename InputT, typename TileShapes>
__aicore__ inline void RunPhase6(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR rawG, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR finalState, GM_ADDR gCumsumBth, GM_ADDR A, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const __gm__ ChunkGatedDeltaRuleStateOutputTrailer *stateOutputTiling = GetStateOutputTrailer(tiling);
    const __gm__ Arch35ChunkGatedDeltaRuleFwdTrailer *phase6 = GetPhase6Trailer(tiling);
    Arch35ChunkGatedDeltaRuleCoefficientTiling coefficient{};
    CopyCoefficientTiling(&phase6->coefficient, coefficient);

    GM_ADDR scoreWorkspace = userWorkspace + phase6->scoreWorkspaceOffset;
    GM_ADDR aWorkspace = userWorkspace + phase6->aWorkspaceOffset;
    GM_ADDR solveWorkspaceBase = userWorkspace + phase6->solveWorkspaceOffset;
    GM_ADDR gCumsumBht = userWorkspace + phase6->gCumsumBhtOffset;
    uint64_t coreGroup = static_cast<uint64_t>(AscendC::GetBlockIdx());
    if ASCEND_IS_AIV {
        coreGroup /= static_cast<uint64_t>(AscendC::GetSubBlockNum());
    }
    GM_ADDR solveWorkspace =
        solveWorkspaceBase + coreGroup * coefficient.solveWorkspacePerCoreBytes;

    if ASCEND_IS_AIC {
        NsChunkKktCube::ChunkKktCube<InputT> kktCube;
        kktCube.Process(k, cuSeqlens, chunkIndices, scoreWorkspace, &coefficient);
    }
    if ASCEND_IS_AIV {
        RunPhase6Cumsum(rawG, cuSeqlens, chunkIndices, gCumsumBht, coefficient);
    }

    // Cumsum and coefficient epilogue use different AIV task mappings.  An
    // epilogue can therefore consume gCumsumBht written by another core, not
    // merely by its paired AIV.  Publish every cumsum tile together with all
    // AIC score tiles before any epilogue starts reading either workspace.
    AscendC::SyncAll<false>();

    if ASCEND_IS_AIV {
        AscendC::TPipe kktPipe;
        NsChunkScaledDotKktFusedCumsum::ChunkScaledDotKktFusedCumsum<InputT, InputT> kkt;
        kkt.Init(
            k, gCumsumBht, beta, cuSeqlens, chunkIndices, aWorkspace, scoreWorkspace,
            coefficient.B, coefficient.Hk, coefficient.Hv, coefficient.hvPerHk, coefficient.T, coefficient.K, coefficient.BT, coefficient.NT,
            coefficient.taskNum, coefficient.usedAicNum, coefficient.usedAivNum, coefficient.btAlign, coefficient.isVarlen, &kktPipe);
        kkt.ProcessEpilogueForSolve(coefficient.tilesPerCore);
        kktPipe.Reset();
    }

    if (coefficient.BT == 64) {
        RunSolvePhase<InputT, 64>(aWorkspace, cuSeqlens, chunkIndices, A,
                                  solveWorkspace, &coefficient);
    } else {
        RunSolvePhase<InputT, 128>(aWorkspace, cuSeqlens, chunkIndices, A,
                                   solveWorkspace, &coefficient);
    }
    // Solve and recompute share a contiguous task range. Publish solved A
    // before either paired AIV enters its local consumer range.
    if ASCEND_IS_AIC {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(PHASE6_SOLVE_DONE_FLAG);
    }
    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(PHASE6_SOLVE_DONE_FLAG);
    }
    GM_ADDR w = userWorkspace + stateOutputTiling->wIntermediateOffset;
    GM_ADDR u = userWorkspace + stateOutputTiling->uIntermediateOffset;
    GM_ADDR h = userWorkspace + stateOutputTiling->hIntermediateOffset;
    // Diagnostic build: expose the fused FwdH vNew through the public O buffer
    // so case-level evidence can determine whether the first mismatch precedes
    // FwdO.  This alias is valid for the target case because O and vNew share
    // the same [B, Hv, T, V] physical layout.
    GM_ADDR vNew = o;
    RecomputeWUFwdTilingData recomputeTiling{};
    CopyRecomputeTiling(&stateOutputTiling->recompute, recomputeTiling);
    if (stateOutputTiling->recompute.V == 256) {
        DispatchRecompute<InputT, float, 256, true>(
            k, v, beta, A, gCumsumBht, cuSeqlens, chunkIndices, w, u,
            userWorkspace + stateOutputTiling->recomputeWorkspaceOffset, &recomputeTiling);
    } else {
        DispatchRecompute<InputT, float, 128, true>(
            k, v, beta, A, gCumsumBht, cuSeqlens, chunkIndices, w, u,
            userWorkspace + stateOutputTiling->recomputeWorkspaceOffset, &recomputeTiling);
    }

    WritePublicCumsumRows(gCumsumBht, gCumsumBth, cuSeqlens, chunkIndices, coefficient);
    DispatchFwdH<TileShapes>(k, w, u, gCumsumBht, gk, initialState, cuSeqlens,
                             chunkIndices, h, vNew, finalState, tiling, userWorkspace);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // H publishes h/vNew through MTE3 and O first consumes them through MTE2.
    // Limit the global hand-off to those pipelines instead of draining PIPE_ALL.
    AscendC::SyncAll<false, PHASE6_HO_SYNC_CONFIG>();
#else
    // Ascend910B supports only the full-pipeline SyncAll overload.
    AscendC::SyncAll<false>();
#endif

    // Diagnostic build: vNew is already stored in O.  Stop before FwdO can
    // overwrite the captured intermediate.
    return;

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

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR a_storage, GM_ADDR raw_g,
    GM_ADDR gk, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR o, GM_ADDR final_state, GM_ADDR g_cumsum_bth, GM_ADDR A,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)a_storage;
    REGISTER_TILING_DEFAULT(GDN::Arch35ChunkGatedDeltaRuleFwdTrailer);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        const __gm__ GDN::Arch35ChunkGatedDeltaRuleFwdTrailer *phase6 = GDN::GetPhase6Trailer(tiling);
        if (phase6->coefficient.dtypeMode == 1) {
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
        const __gm__ GDN::Arch35ChunkGatedDeltaRuleFwdTrailer *phase6 = GDN::GetPhase6Trailer(tiling);
        if (phase6->coefficient.dtypeMode == 1) {
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
