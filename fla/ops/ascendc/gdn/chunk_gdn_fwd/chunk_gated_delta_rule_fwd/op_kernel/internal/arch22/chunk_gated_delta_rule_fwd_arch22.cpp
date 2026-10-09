/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_arch22_struct.h"
#include "gdn_cumsum_prepare.hpp"

#define GDN_CHUNK_RECOMPUTE_WU_FWD_HO_IMPL_ONLY
#include "operators/chunk_recompute_wu_fwd_ho/op_kernel/chunk_recompute_wu_fwd_ho.cpp"
#undef GDN_CHUNK_RECOMPUTE_WU_FWD_HO_IMPL_ONLY

#define GDN_CHUNK_CUMSUM_KKT_SOLVE_IMPL_ONLY
#include "operators/chunk_kkt_solve_tri/op_kernel/chunk_cumsum_kkt_solve_tri.cpp"
#include "operators/chunk_kkt_solve_tri/op_kernel/solve_layout_staging.h"
#undef GDN_CHUNK_CUMSUM_KKT_SOLVE_IMPL_ONLY

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "operators/solve_tri_fp32/solve_tri_pipeline.h"
#endif

namespace GDN {
namespace {

constexpr uint32_t OWNER_MTE2_TO_V_EVENT = 0;
constexpr uint32_t OWNER_V_TO_MTE3_EVENT = 1;
constexpr uint32_t OWNER_MTE3_TO_V_EVENT = 2;
constexpr uint32_t UB_ALIGNMENT = 32;
constexpr uint32_t PHASE6_TILING_ALIGNMENT = 8;
constexpr uint64_t PHASE6_SOLVE_DONE_FLAG = 5;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
constexpr AscendC::SyncAllConfig PHASE6_HO_SYNC_CONFIG = {PIPE_MTE3, PIPE_MTE2};
#endif

__aicore__ inline uint64_t AlignPhase6(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline const __gm__ ChunkRecomputeWUFwdHOTrailer *GetPhase5Trailer(GM_ADDR tiling)
{
    const uint64_t oTilingOffset = AlignPhase6(
        sizeof(GdnMegaArch22FwdHTilingData), PHASE6_TILING_ALIGNMENT);
    return reinterpret_cast<const __gm__ ChunkRecomputeWUFwdHOTrailer *>(
        tiling + oTilingOffset + sizeof(GdnMegaArch22FwdOTilingData));
}

__aicore__ inline const __gm__ Arch22ChunkGatedDeltaRuleFwdTrailer *GetPhase6Trailer(GM_ADDR tiling)
{
    const uint64_t oTilingOffset = AlignPhase6(
        sizeof(GdnMegaArch22FwdHTilingData), PHASE6_TILING_ALIGNMENT);
    const uint64_t phase5End = oTilingOffset + sizeof(GdnMegaArch22FwdOTilingData) +
                               sizeof(ChunkRecomputeWUFwdHOTrailer);
    return reinterpret_cast<const __gm__ Arch22ChunkGatedDeltaRuleFwdTrailer *>(
        tiling + AlignPhase6(phase5End, PHASE6_TILING_ALIGNMENT));
}

__aicore__ inline void CopyAbcTiling(
    const __gm__ Arch22ChunkGatedDeltaRuleFwdAbcTiling *src, Arch22ChunkGatedDeltaRuleFwdAbcTiling &dst)
{
    dst.qkvLayout = src->qkvLayout;
    dst.oLayout = src->oLayout;
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
    const Arch22ChunkGatedDeltaRuleFwdAbcTiling &tiling)
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
    // offsets由Scalar写入，Gather读取前建立S到V的依赖。
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
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

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
// HO 空闲流水统一准入（仅 220）：host 的 hoPipelineAvailable 只是 GM 预留信号，
// 设备侧用与 H 一致的任务轴独立复核全部结构条件。所有 AIC/AIV 基于相同只读
// 输入得出同一配置，不新增全核同步；任一条件不满足即保持默认关闭。
__aicore__ inline void ResolveHoIdlePipeline(
    GM_ADDR tiling, GM_ADDR cuSeqlens, GM_ADDR userWorkspace,
    const __gm__ Arch22ChunkGatedDeltaRuleFwdTrailer *phase6,
    GdnHoPipeline::HoPipelineConfig &cfg, GM_ADDR &readyAddr)
{
    cfg = GdnHoPipeline::HoPipelineConfig{};
    readyAddr = nullptr;
    // host 无预留时立即关闭，不额外扫描 varlen。
    if (phase6->hoPipelineAvailable == 0) {
        return;
    }
    const __gm__ GdnMegaArch22FwdHTilingData *hTiling =
        reinterpret_cast<const __gm__ GdnMegaArch22FwdHTilingData *>(tiling);
    // 协议前置：无 gk、K128、V128/256、BT64/128（BT 即 H tiling chunkSize）。
    if (hTiling->useGk || hTiling->kHeadDim != 128) {
        return;
    }
    const int64_t vHeadDim = hTiling->vHeadDim;
    const int64_t bt = hTiling->chunkSize;
    if (vHeadDim != 128 && vHeadDim != 256) {
        return;
    }
    if (bt != 64 && bt != 128) {
        return;
    }
    // bank 数须可转协议侧 uint32 且非零。
    const uint64_t readyBankCount = phase6->hoReadyBankCount;
    if (readyBankCount == 0 || readyBankCount > 0xFFFFFFFFull) {
        return;
    }
    const uint32_t cubeCoreNum = AscendC::GetBlockNum();
    if (cubeCoreNum == 0) {
        return;
    }
    // 与 H scheduler 一致的 vBlock 切分：VB = ceil(V/128)，V 命中 128/256 时为 1/2。
    const uint64_t vBlockNum = static_cast<uint64_t>(vHeadDim) / 128u;

    // S 与最大 local chunk 数：dense 取 H tiling 实际 batch（与原 H 任务轴一致）；
    // varlen 扫 tokenBatch 个原序列，按 H InitRuntime 的正序列压缩规则计数。
    uint64_t sequenceNum = 0;
    uint64_t maxLocalChunks = 0;
    if (hTiling->isVariedLen != 0) {
        AscendC::GlobalTensor<int64_t> cu;
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        int64_t prevSeq = 0;
        for (int64_t seq = 1; seq <= hTiling->tokenBatch; ++seq) {
            const int64_t currSeq = cu.GetValue(seq);
            // 同原 H 从 prev=0 开始；非单调边界视为非法输入，直接关闭。
            if (currSeq < prevSeq) {
                return;
            }
            const int64_t seqLen = currSeq - prevSeq;
            if (seqLen > 0) {
                ++sequenceNum;
                // ceil 用除法+取余表达，避免 int64 加法回绕。
                const int64_t localChunks = seqLen / bt + (seqLen % bt != 0 ? 1 : 0);
                if (localChunks > static_cast<int64_t>(maxLocalChunks)) {
                    maxLocalChunks = static_cast<uint64_t>(localChunks);
                }
            }
            prevSeq = currSeq;
        }
    } else {
        const int64_t seqLen = hTiling->seqlen;
        if (hTiling->batch <= 0 || seqLen < 0) {
            return;
        }
        sequenceNum = static_cast<uint64_t>(hTiling->batch);
        maxLocalChunks = static_cast<uint64_t>(seqLen / bt + (seqLen % bt != 0 ? 1 : 0));
    }

    // P = S*Hv*VB 须严格落在 (0, C)：先以 C 约束各因子再分步相乘，避免
    // uint64 乘法回绕与 uint32 截断。
    if (sequenceNum == 0 || hTiling->vNumHead <= 0) {
        return;
    }
    if (sequenceNum >= cubeCoreNum || static_cast<uint64_t>(hTiling->vNumHead) >= cubeCoreNum) {
        return;
    }
    const uint64_t partial = sequenceNum * static_cast<uint64_t>(hTiling->vNumHead);
    if (partial >= cubeCoreNum) {
        return;
    }
    const uint64_t producerCount = partial * vBlockNum;
    if (producerCount == 0 || producerCount >= cubeCoreNum) {
        return;
    }
    // 比例准入：O 消费者 (C-P) 至少与 H 生产者等量，P > C-P 即回退默认关闭。
    // 此时 P < C 已成立，cubeCoreNum - producerCount 不会无符号下溢。
    if (producerCount > cubeCoreNum - producerCount) {
        return;
    }
    // 每序列 chunk 数上界须被 GM bank 数覆盖；至少一条序列长度 > BT（多 chunk），
    // 不能以 totalChunks > 0 代替。
    if (maxLocalChunks < 2 || maxLocalChunks > readyBankCount) {
        return;
    }

    GdnHoPipeline::HoPipelineConfig resolved{};
    resolved.cubeCoreNum = cubeCoreNum;
    resolved.sequenceNum = static_cast<uint32_t>(sequenceNum);
    resolved.headNum = static_cast<uint32_t>(hTiling->vNumHead);
    resolved.vBlockNum = static_cast<uint32_t>(vBlockNum);
    resolved.producerCount = static_cast<uint32_t>(producerCount);
    resolved.readyBankCount = static_cast<uint32_t>(readyBankCount);
    resolved.enabled = true;
    if (!GdnHoPipeline::HoPipelineValid(resolved)) {
        return;
    }
    cfg = resolved;
    readyAddr = userWorkspace + phase6->hoReadyWorkspaceOffset;
}
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
// 每次调用只处理同一物理组的一批parent。保留原KKT后处理和低精度舍入点。
template <typename InputT, bool kPreparedCumsum>
__aicore__ inline void RunKktEpilogueBatch(
    GM_ADDR k, GM_ADDR beta, GM_ADDR rawG, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR gCumsumBht, GM_ADDR userWorkspace,
    const __gm__ Arch22ChunkGatedDeltaRuleFwdTrailer *phase6,
    const Arch22ChunkGatedDeltaRuleFwdAbcTiling &abc, int64_t begin, int64_t end)
{
    AscendC::TPipe pipe;
    NsChunkScaledDotKkt::ChunkScaledDotKkt<InputT, InputT, true> kkt;
    GM_ADDR aWorkspace = userWorkspace + phase6->aWorkspaceOffset;
    GM_ADDR scoreWorkspace = userWorkspace + phase6->scoreWorkspaceOffset;
    if constexpr (kPreparedCumsum) {
        kkt.Init(
            k, gCumsumBht, beta, cuSeqlens, chunkIndices, aWorkspace,
            scoreWorkspace, abc.B, abc.Hk, abc.Hv, abc.hvPerHk, abc.T, abc.K,
            abc.BT, abc.NT, abc.taskNum, abc.usedAicNum, abc.usedAivNum,
            abc.btAlign, abc.isVarlen, &pipe);
    } else {
        kkt.InitFusedCumsum(
            k, rawG, beta, cuSeqlens, chunkIndices, gCumsumBht, aWorkspace,
            scoreWorkspace, abc.B, abc.Hk, abc.Hv, abc.hvPerHk, abc.T, abc.K,
            abc.BT, abc.NT, abc.taskNum, abc.usedAicNum, abc.usedAivNum,
            abc.btAlign, abc.isVarlen, &pipe);
    }
    kkt.InitSolveFp32Input(userWorkspace + phase6->solveFp32InputOffset);
    kkt.ProcessEpilogueRange(begin, end);
    AscendC::PipeBarrier<PIPE_ALL>();
    pipe.Reset();
}

template <typename InputT, bool kPreparedCumsum>
__aicore__ inline void RunFrontBatch(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR rawG, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR gCumsumBht, GM_ADDR A, GM_ADDR w, GM_ADDR u,
    GM_ADDR userWorkspace, const __gm__ Arch22ChunkGatedDeltaRuleFwdTrailer *phase6,
    const Arch22ChunkGatedDeltaRuleFwdAbcTiling &abc,
    const GdnMegaArch22RecomputeWUTilingData &recomputeTiling)
{
    constexpr uint64_t FRONT_READY_FLAG = 6;
    constexpr uint64_t FRONT_ACK_DONE_FLAG = 7;
    const uint64_t core = static_cast<uint64_t>(GdnFp32Solve::CoreGroup());
    const uint64_t perCore = static_cast<uint64_t>(abc.tilesPerCore);
    const uint64_t capacity = perCore < FRONT_PARENT_BATCH_SIZE
        ? perCore : FRONT_PARENT_BATCH_SIZE;
    const uint64_t ownerBegin = core * perCore;
    const uint64_t ownerEnd = ownerBegin + perCore < abc.taskNum
        ? ownerBegin + perCore : abc.taskNum;
    GM_ADDR wuWorkspace = userWorkspace + phase6->frontWuWorkspaceOffset +
        core * capacity * abc.BT * (recomputeTiling.V + recomputeTiling.K) * sizeof(InputT);
    GM_ADDR x = userWorkspace + phase6->solveFp32InputOffset;
    GdnFp32Solve::FullProblem problem{
        static_cast<int64_t>(abc.B), static_cast<int64_t>(abc.T),
        static_cast<int64_t>(abc.Hv), static_cast<int64_t>(abc.BT),
        1, static_cast<int64_t>(phase6->solveSequenceCount), 0, 0, 0};
    if (problem.sequences == 0) {
        problem.tasks32 = (problem.tokens + 31) / 32 * problem.batch * problem.heads;
        problem.tasks64 = (problem.tokens + 63) / 64 * problem.batch * problem.heads;
        problem.tasks128 = (problem.tokens + 127) / 128 * problem.batch * problem.heads;
    } else {
        AscendC::GlobalTensor<int64_t> cu;
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        for (int64_t sequence = 0; sequence < problem.sequences; ++sequence) {
            const int64_t length = cu.GetValue(sequence + 1) - cu.GetValue(sequence);
            problem.tasks32 += (length + 31) / 32 * problem.heads;
            problem.tasks64 += (length + 63) / 64 * problem.heads;
            problem.tasks128 += (length + 127) / 128 * problem.heads;
        }
    }
    if ASCEND_IS_AIV {
        // 零任务组也消费Stage P的通知，再直接到最终全核会合。
        AscendC::CrossCoreWaitFlag(SCORE_READY_FLAG);
        if (ownerBegin < ownerEnd) {
            const uint64_t firstEnd = ownerBegin + capacity < ownerEnd
                ? ownerBegin + capacity : ownerEnd;
            RunKktEpilogueBatch<InputT, kPreparedCumsum>(
                k, beta, rawG, cuSeqlens, chunkIndices, gCumsumBht,
                userWorkspace, phase6, abc, ownerBegin, firstEnd);
        }
    }
    for (uint64_t begin = ownerBegin; begin < ownerEnd; begin += capacity) {
        const uint64_t end = begin + capacity < ownerEnd ? begin + capacity : ownerEnd;
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(FRONT_READY_FLAG);
            AscendC::CrossCoreWaitFlag(FRONT_ACK_DONE_FLAG);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(FRONT_READY_FLAG);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(FRONT_ACK_DONE_FLAG);
        }
        problem.localParentBegin = static_cast<int64_t>(begin - ownerBegin);
        problem.localParentCount = static_cast<int64_t>(end - begin);
        GdnFp32Solve::RunOwnedBatch<InputT>(
            x, userWorkspace + phase6->solveD16Offset, userWorkspace + phase6->solveD32Offset,
            userWorkspace + phase6->solveD64Offset, A,
            userWorkspace + phase6->solveWorkspaceOffset, cuSeqlens, problem);
        const RecomputeTaskRange range{begin, end, capacity};
        if (recomputeTiling.V == 256) {
            DispatchRecompute<InputT, float, 256, true>(
                k, v, beta, A, gCumsumBht, cuSeqlens, chunkIndices, w, u,
                wuWorkspace, &recomputeTiling, &range, abc.qkvLayout == 1);
        } else {
            DispatchRecompute<InputT, float, 128, true>(
                k, v, beta, A, gCumsumBht, cuSeqlens, chunkIndices, w, u,
                wuWorkspace, &recomputeTiling, &range, abc.qkvLayout == 1);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if ASCEND_IS_AIC {
            // 必须在WU作用域析构且FIX完成后才能允许下批复用两个临时段。
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(FRONT_ACK_DONE_FLAG);
        }
        if ASCEND_IS_AIV {
            if (end < ownerEnd) {
                const uint64_t nextEnd = end + capacity < ownerEnd ? end + capacity : ownerEnd;
                RunKktEpilogueBatch<InputT, kPreparedCumsum>(
                    k, beta, rawG, cuSeqlens, chunkIndices, gCumsumBht,
                    userWorkspace, phase6, abc, end, nextEnd);
            }
            // 下一批KKT可与本批Cube WU重叠，但Solve和WU临时区复用须等done。
            AscendC::CrossCoreWaitFlag(FRONT_ACK_DONE_FLAG);
        }
    }
    // 发布所有W/U/g，并覆盖零任务组；随后沿用原公共g输出及H/O路径。
    AscendC::SyncAll<false>();
}
#endif

template <typename InputT, typename TileShapes, bool kPreparedCumsum = false>
__aicore__ inline void RunPhase6(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR rawG, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR finalState, GM_ADDR gCumsumBth, GM_ADDR A, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const __gm__ ChunkRecomputeWUFwdHOTrailer *phase5 = GetPhase5Trailer(tiling);
    const __gm__ Arch22ChunkGatedDeltaRuleFwdTrailer *phase6 = GetPhase6Trailer(tiling);
    Arch22ChunkGatedDeltaRuleFwdAbcTiling abc{};
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
        kktCube.ConfigureInputLayout(abc.qkvLayout == 1);
        kktCube.Process(k, cuSeqlens, chunkIndices, scoreWorkspace, &abc);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SCORE_READY_FLAG);
    }
    if ASCEND_IS_AIV {
        if constexpr (kPreparedCumsum) {
            // Stage P consumes the explicit BTH raw_g contract.  Its task
            // queue is one task per batch/chunk; each task writes all value
            // heads, unlike the ABC KKT queue which is B*Hv*NT.
            GdnCumsumPrepare::PrepareArgs prepareArgs{
                rawG, gCumsumBht,
                phase6->outputGCumsum != 0 ? gCumsumBth : nullptr,
                cuSeqlens, chunkIndices, abc.B, abc.Hv, abc.T, abc.BT, abc.NT,
                abc.B * abc.NT, abc.isVarlen, phase6->outputGCumsum};
            GdnCumsumPrepare::Kernel prepare;
            prepare.Init(prepareArgs);
            prepare.ProcessMixed();
        }
    }
    // Every AIC/AIV participant reaches the hand-off.  This publishes all
    // BHT prefixes before KKT, Solve, or recompute can consume them.
    if constexpr (kPreparedCumsum) {
        AscendC::SyncAll<false>();
    }
    GM_ADDR w = userWorkspace + phase5->wIntermediateOffset;
    GM_ADDR u = userWorkspace + phase5->uIntermediateOffset;
    GM_ADDR h = userWorkspace + phase5->hIntermediateOffset;
    GM_ADDR vNew = userWorkspace + phase5->vNewIntermediateOffset;
    GdnMegaArch22RecomputeWUTilingData recomputeTiling{};
    CopyRecomputeTiling(&phase5->recompute, recomputeTiling);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    RunFrontBatch<InputT, kPreparedCumsum>(
        k, v, beta, rawG, cuSeqlens, chunkIndices, gCumsumBht, A, w, u,
        userWorkspace, phase6, abc, recomputeTiling);
#else
    constexpr bool kStoreFp32SolveInput = false;
    if ASCEND_IS_AIV {
        AscendC::TPipe kktPipe;
        NsChunkScaledDotKkt::ChunkScaledDotKkt<InputT, InputT, kStoreFp32SolveInput> kkt;
        if constexpr (kPreparedCumsum) {
            kkt.Init(
                k, gCumsumBht, beta, cuSeqlens, chunkIndices, aWorkspace,
                scoreWorkspace, abc.B, abc.Hk, abc.Hv, abc.hvPerHk, abc.T, abc.K,
                abc.BT, abc.NT, abc.taskNum, abc.usedAicNum, abc.usedAivNum,
                abc.btAlign, abc.isVarlen, &kktPipe);
        } else {
            kkt.InitFusedCumsum(
                k, rawG, beta, cuSeqlens, chunkIndices, gCumsumBht, aWorkspace,
                scoreWorkspace, abc.B, abc.Hk, abc.Hv, abc.hvPerHk, abc.T, abc.K,
                abc.BT, abc.NT, abc.taskNum, abc.usedAicNum, abc.usedAivNum,
                abc.btAlign, abc.isVarlen, &kktPipe);
        }
        if constexpr (kStoreFp32SolveInput) {
            kkt.InitSolveFp32Input(userWorkspace + phase6->solveFp32InputOffset);
        }
        AscendC::CrossCoreWaitFlag(SCORE_READY_FLAG);
        kkt.ProcessEpilogueForSolve(abc.tilesPerCore);
        kktPipe.Reset();
    }

    GM_ADDR tndInput = scoreWorkspace;
    GM_ADDR tndOutput = scoreWorkspace + abc.aWorkspaceBytes;
    if (abc.BT == 64 && abc.isVarlen != 0) {
        // Match the public BT64 SolveTri path exactly: physical TND layout,
        // chunk-to-head task order, and the native FP32 implementation.
        AscendC::SyncAll<false>();
        NsPhase6SolveLayoutStaging::TransposeBhtTnd<InputT>(
            aWorkspace, tndInput, &abc, true);
        AscendC::SyncAll<false>();

        Arch22ChunkGatedDeltaRuleFwdAbcTiling solveTiling = abc;
        solveTiling.layoutMode = 2;
        RunSolvePhase<InputT, 64>(tndInput, cuSeqlens, chunkIndices,
                                  tndOutput, solveWorkspaceBase, &solveTiling);

        AscendC::SyncAll<false>();
        NsPhase6SolveLayoutStaging::TransposeBhtTnd<InputT>(
            tndOutput, A, &abc, false);
        AscendC::SyncAll<false>();
    } else if (abc.BT == 64) {
        // The public SolveTri runs after a kernel boundary.  Recreate that
        // visibility point before its FP32 AIC/AIV protocol consumes the
        // fused KKT epilogue written by all participating vector cores.
        AscendC::SyncAll<false>();
        RunSolvePhase<InputT, 64>(aWorkspace, cuSeqlens, chunkIndices, A,
                                  solveWorkspaceBase, &abc);
    } else {
        RunSolvePhase<InputT, 128>(aWorkspace, cuSeqlens, chunkIndices, A,
                                   solveWorkspace, &abc);
    }
    // Solve and recompute share a contiguous task range. Publish solved A
    // before either paired AIV enters its local consumer range.
    if ASCEND_IS_AIC {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(PHASE6_SOLVE_DONE_FLAG);
    }
    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(PHASE6_SOLVE_DONE_FLAG);
    }
    if (phase5->recompute.V == 256) {
        DispatchRecompute<InputT, float, 256, true>(
            k, v, beta, A, gCumsumBht, cuSeqlens, chunkIndices, w, u,
            userWorkspace + phase5->recomputeWorkspaceOffset, &recomputeTiling);
    } else {
        DispatchRecompute<InputT, float, 128, true>(
            k, v, beta, A, gCumsumBht, cuSeqlens, chunkIndices, w, u,
            userWorkspace + phase5->recomputeWorkspaceOffset, &recomputeTiling);
    }

#endif

    if (phase6->outputGCumsum != 0) {
        if constexpr (kPreparedCumsum) {
            // Stage P already wrote the public BTH output in its producer
            // pass.  Do not transpose or write it a second time.
        } else {
            WritePublicCumsumRows(gCumsumBht, gCumsumBth, cuSeqlens, chunkIndices, abc);
        }
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    GdnHoPipeline::HoPipelineConfig hoIdleConfig{};
    GM_ADDR hoIdleReadyAddr = nullptr;
    ResolveHoIdlePipeline(tiling, cuSeqlens, userWorkspace, phase6, hoIdleConfig,
                          hoIdleReadyAddr);
    const bool hoIdleEnabled = hoIdleConfig.enabled;
    const GdnHoPipeline::HoPipelineConfig *hoIdleConfigPtr =
        hoIdleEnabled ? &hoIdleConfig : nullptr;
#else
    constexpr bool hoIdleEnabled = false;
    const GdnHoPipeline::HoPipelineConfig *hoIdleConfigPtr = nullptr;
    GM_ADDR hoIdleReadyAddr = nullptr;
#endif
    // 全部实际核仍调用 H，保留原 entry 及唯一 wave 的 SyncAll。
    DispatchFwdH<InputT, TileShapes>(k, w, u, gCumsumBht, gk, initialState, cuSeqlens,
                             chunkIndices, h, vNew, finalState, tiling, userWorkspace,
                             hoIdleConfigPtr, hoIdleReadyAddr, abc.qkvLayout == 1);

    if (!hoIdleEnabled) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // H publishes h/vNew through MTE3 and O first consumes them through MTE2.
        // Limit the global hand-off to those pipelines instead of draining PIPE_ALL.
        AscendC::SyncAll<false, PHASE6_HO_SYNC_CONFIG>();
#else
        // DAV_2201 supports only the full-pipeline SyncAll overload.
        AscendC::SyncAll<false>();
#endif
    }

    const uint64_t oTilingOffset =
        AlignPhase6(sizeof(GdnMegaArch22FwdHTilingData), PHASE6_TILING_ALIGNMENT);
    const __gm__ GdnMegaArch22FwdOTilingData *gmOTiling =
        reinterpret_cast<const __gm__ GdnMegaArch22FwdOTilingData *>(tiling + oTilingOffset);
    GdnMegaArch22FwdOTilingData oTiling{};
    CopyOTiling(gmOTiling, oTiling);
    if (!hoIdleEnabled) {
        // fallback：原 H -> arch310 定制/220 全 PIPE SyncAll -> 全核 O，
        // 不额外增加收尾屏障。
        DispatchFwdO<InputT>(q, k, vNew, h, gCumsumBht, cuSeqlens, chunkIndices, o,
                     userWorkspace, &oTiling, nullptr, nullptr, abc.qkvLayout == 1, abc.oLayout == 1);
    } else {
        // 新路径：H/O 之间不再执行全核 SyncAll（否则不会重叠）。生产者前缀
        // [0, P) 完成 H 后直达最终会合；仅空闲物理核组 [P, C) 执行 O，O 内部
        // 无 SyncAll、保持已有局部握手。AIC 物理组是 GetBlockIdx()，AIV 物理组
        // 是 GetBlockIdx()/GetSubBlockNum()（即上方 coreGroup）。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if (coreGroup >= hoIdleConfig.producerCount) {
            DispatchFwdO<InputT>(q, k, vNew, h, gCumsumBht, cuSeqlens, chunkIndices, o,
                         userWorkspace, &oTiling, &hoIdleConfig, hoIdleReadyAddr, abc.qkvLayout == 1, abc.oLayout == 1);
        }
#endif
        // 所有核在条件 O 之后共同执行一次最终全核会合。
        AscendC::SyncAll<false>();
    }
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
    REGISTER_TILING_DEFAULT(GDN::Arch22ChunkGatedDeltaRuleFwdTrailer);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, workspace, tiling);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, workspace, tiling);
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128, true>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, workspace, tiling);
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256, true>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, workspace, tiling);
    }
}
