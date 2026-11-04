/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Shared constants, tiling, walk state, GM BNSD offsets, on-chip byte
 * offsets, UB/L1/L0 views, ping-pong, and CrossCore Notify/Wait helpers.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_PREPARE_COMMON_ARCH92_H
#define CHUNK_GATED_DELTA_RULE_FWD_PREPARE_COMMON_ARCH92_H

#include "kernel_operator.h"

struct HardwareInfo {
    static uint32_t const l1Size = 512 * 1024;
    static uint32_t const l0ASize = 64 * 1024;
    static uint32_t const l0BSize = 64 * 1024;
    static uint32_t const l0CSize = 256 * 1024;
    static uint32_t const l2Size = 128 * 1024 * 1024;
    static uint32_t const biasSize = 4 * 1024;
    static uint32_t const fixBufSize = 6 * 1024;
    static uint32_t const ubSize = 376 * 1024;
};

enum class BufferType : uint32_t {
    ASCEND_UB,
    ASCEND_CB,
    ASCEND_L0A,
    ASCEND_L0B,
    ASCEND_L0C,
    ASCEND_MAX
};

struct OnChipBuffer {
    template <typename T>
    using Tensor = AscendC::LocalTensor<T>;

public:
    __aicore__ inline OnChipBuffer()
    {
        constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {
            HardwareInfo::ubSize, HardwareInfo::l1Size, HardwareInfo::l0ASize,
            HardwareInfo::l0BSize, HardwareInfo::l0CSize};

        buffer_[(uint32_t)BufferType::ASCEND_UB] =
            Tensor<uint8_t>(AscendC::TPosition::VECIN, 0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        buffer_[(uint32_t)BufferType::ASCEND_CB] =
            Tensor<uint8_t>(AscendC::TPosition::A1, 0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        buffer_[(uint32_t)BufferType::ASCEND_L0A] =
            Tensor<uint8_t>(AscendC::TPosition::A2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        buffer_[(uint32_t)BufferType::ASCEND_L0B] =
            Tensor<uint8_t>(AscendC::TPosition::B2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        buffer_[(uint32_t)BufferType::ASCEND_L0C] =
            Tensor<uint8_t>(AscendC::TPosition::CO1, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
    }

    template <BufferType bufferType, typename Dtype = float>
    __aicore__ inline Tensor<Dtype> GetBuffer(const uint32_t offset) const
    {
        return buffer_[(uint32_t)bufferType][offset].template ReinterpretCast<Dtype>();
    }

private:
    AscendC::LocalTensor<uint8_t> buffer_[(uint32_t)BufferType::ASCEND_MAX];
};

constexpr int64_t kGdnChunkSize = 64;
constexpr int64_t kGdnHeadDimK = 128;
constexpr float kGdnRcpLn2 = 1.4426950216f;
constexpr float kGdnLn2 = 0.6931471825f;
constexpr float kGdnL2NormEps = 1e-6f;
constexpr float kGdnGateClip = 50.0f;

struct ChunkGatedDeltaRuleFwdStageTilingData {
    // B. Number of independent sequences packed as dim-0 of BNSD.
    // Varlen packed layout uses 1: all sequences sit on dim T via cu_seqlens.
    int64_t inputBatchSize;

    // Hk. Query/key head count. q/k GM dim-1. Independent of Hv except Hv % Hk == 0.
    int64_t queryKeyHeadCount;

    // Hv. Value / gate / beta / A head count. GM dim-1 of v, g, beta, A, w, u.
    int64_t valueHeadCount;

    // T. Tokens along BNSD dim-2. Fixed-length: tokens per batch item.
    // Varlen: packed token total = cu_seqlens[-1].
    int64_t sequenceTokenLength;

    // K. Channel of q/k/q_hat/k_hat/w. First version is 128.
    int64_t queryKeyHeadDim;

    // V. Channel of v/u. First version is 128 or 256.
    int64_t valueHeadDim;

    // G = Hv / Hk in {1,2,3,4}. hk = hv / G. Pack-local owner is slot 0 or the first HV of an HK.
    int64_t valueHeadsPerQueryKeyHead;

    // BT. Tokens in one chunk tile. First version is 64. Last chunk may be shorter.
    int64_t tokensPerChunk;

    // N. Sequence count used with cu_seqlens: B if fixed-length, else cu_seqlens.numel()-1.
    int64_t packedSequenceCount;

    // 1: T is packed varlen and cu_seqlens is present. 0: dense BNSD.
    int64_t isVariableLengthPacked;

    // 1: chunk_indices GM is [2 * numChunks] pairs (seqId, localChunk).
    int64_t hasChunkIndexTable;

    // 1: q' = q/||q||, k' = k/||k||; write q_hat, k_hat, rstd.
    // 0: skip L2Norm; kkt/kbg use k; q_hat/k_hat/rstd outputs omitted.
    int64_t enableQueryKeyL2NormInKernel;

    // 1: g_raw = -exp(a_log) * softplus(g + dt_bias) before chunk-local cumsum.
    int64_t enableFusedGateSoftplus;

    // 1: beta_eff = sigmoid(beta). 0: fp32 copy of beta.
    int64_t enableBetaSigmoid;

    // 1: dt_bias[Hv] is present. Only used with fused gate.
    int64_t hasGateDtBias;

    // Prepare-only. 1: q_hat GM output is allocated. Must match host tiling.
    int64_t hasQueryHatGmOutput;

    // Prepare-only. 1: k_hat GM output is allocated. Must match host tiling.
    int64_t hasKeyHatGmOutput;

    // 1 and sigmoid on: beta_eff = 2 * sigmoid(beta) (allow_neg_eigval).
    int64_t scaleBetaByTwoWhenNegEigval;

    // 1: g' = RCP_LN2 * chunk_cumsum(g_raw); later exp uses exp2(g').
    int64_t useExp2ForGateCumsum;

    // ge::DataType of q/k storage (bf16 == 27).
    int64_t queryKeyStorageDtype;

    // ge::DataType of raw gate g.
    int64_t gateStorageDtype;

    // ge::DataType of raw beta.
    int64_t betaStorageDtype;

    // How many (batch, seq-chunk, hv) tiles this kernel walks.
    // Fixed-length: Ceil(T / BT) * B * Hv. Varlen: (chunk_indices.numel()/2) * Hv.
    int64_t totalChunkTileCount;

    // 1: A is written to GM. 0: A remains resident in L1 only.
    int64_t hasAOutput;
};


namespace ChunkGatedDeltaRuleFwdPrepare {
using namespace AscendC;

// arch92 is selected by the kernel entry for A6 MIX 1:1.
constexpr uint32_t PREPARE_AIV_COUNT = 1;
constexpr int64_t kTasksPerRound = 3;

// Mode 0x4, AIC <-> AIV0. t is the active task's slot (0..2).
// ID      Producer/set pipe     Consumer/wait pipe   Condition and reuse
// t       S1 AIV/MTE3           S2 AIC/MTE1          Once per pack-local K owner.
// t       S2 AIC/FIX            S3 AIV/V             After S2 takes the owner's ready flag,
//                                                    signal each sharing task on its own ID.
// 3+t     S3 AIV/MTE3           S4 AIC/MTE1          -L and leaves ready for this task.
// 6+t     init or S4 AIC/FIX    next S1 AIV/default  One credit per slot. Take before reuse,
//                                                    return after S4 dump, drain all at exit.
// 9+t     S6 AIV/MTE3           S7 AIC/MTE1          Kbg and Vb ready for this task.
// 12..14  Unused.
// 15      S7 AIC/FIX            AIV/default          Once per pack. Take before next S6,
//                                                    or at exit, then drain the pack's W/U.
// "default" is the wait pipe of CrossCoreWaitFlag<0x4> (PIPE_S).
// Each row has at most one outstanding signal per ID and direction (4-bit count).
// A partial pack uses only active task IDs; unused S4 credits carry over.
constexpr uint16_t kFlagS3DoneBase = kTasksPerRound;
constexpr uint16_t kFlagS4DumpBase = kFlagS3DoneBase + kTasksPerRound;
constexpr uint16_t kFlagS6DoneBase = kFlagS4DumpBase + kTasksPerRound;
constexpr uint16_t kFlagS7Done = 0xF; // mode 0x4, pack output completion
static_assert(kFlagS6DoneBase + kTasksPerRound - 1 < kFlagS7Done, "S6 flags overlap S7");
static_assert(kFlagS7Done < 16, "S7 exceeds the A6 flag ID range");

// Intra-AIV HardEvent ids. Same numeric id is reused after Wait.
constexpr uint16_t kHeUb0 = 0;     // S0 I, S1 q/k, S3 L-ready, S6 v
constexpr uint16_t kHeUb1 = 1;     // S1 tail pad, S6 k'
constexpr uint16_t kHeS3Pack = 4;  // S3 pack leaves vs -L UB→L1

constexpr uint32_t kChunk64 = 64;
constexpr int32_t kNumMFracs64 = 4;

constexpr AscendC::FixpipeConfig CFG_NZ_L1 = {AscendC::CO2Layout::NZ, false};
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr int32_t kFracLen8 = 16 * 8;

constexpr uint32_t kVcs32 = 32;
constexpr uint32_t kVcsPack32 = 64;
constexpr uint32_t kVcsPackedElems32 = kVcs32 * kVcsPack32;
constexpr uint32_t kLeavesPerVec32 = 2;

__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline int32_t TaskSlot(int64_t taskIdx)
{
    return static_cast<int32_t>(taskIdx);
}

// Mode 4 intra-block flags target the only AIV; there is no +16 bank.
__aicore__ inline void NotifyAicStage1Done(int64_t taskIdx)
{
    CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx));
}
__aicore__ inline void WaitAivStage1Done(int64_t taskIdx)
{
    CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(taskIdx));
}
__aicore__ inline void NotifyAivKktDone(int64_t taskIdx)
{
    CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(taskIdx));
}
__aicore__ inline void WaitCubeKktDone(int64_t taskIdx)
{
    CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(taskIdx));
}
__aicore__ inline void NotifyAicStage3Done(int64_t taskIdx)
{
    CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx + kFlagS3DoneBase));
}
__aicore__ inline void WaitAivStage3Done(int64_t taskIdx)
{
    CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(taskIdx + kFlagS3DoneBase));
}
__aicore__ inline void NotifyAivStage4Done(int64_t taskIdx)
{
    CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(taskIdx + kFlagS4DumpBase));
}
__aicore__ inline void WaitAicStage4Done(int64_t taskIdx)
{
    CrossCoreWaitFlag<0x4>(static_cast<uint16_t>(taskIdx + kFlagS4DumpBase));
}
__aicore__ inline void NotifyAicStage6Done(int64_t taskIdx)
{
    CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx + kFlagS6DoneBase));
}
__aicore__ inline void WaitAivStage6Done(int64_t taskIdx)
{
    CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(taskIdx + kFlagS6DoneBase));
}
__aicore__ inline void NotifyAivStage7Done()
{
    CrossCoreSetFlag<0x4, PIPE_FIX>(kFlagS7Done);
}
__aicore__ inline void WaitAicStage7Done()
{
    CrossCoreWaitFlag<0x4>(kFlagS7Done);
}

struct PrepareState {
    template <typename Td>
    __aicore__ inline void LoadTiling(const Td &td)
    {
        B = td.inputBatchSize;
        HK = td.queryKeyHeadCount;
        HV = td.valueHeadCount;
        T = td.sequenceTokenLength;
        K = td.queryKeyHeadDim;
        V = td.valueHeadDim;
        HRatio = td.valueHeadsPerQueryKeyHead;
        chunkSize = td.tokensPerChunk;
        isVariable = td.isVariableLengthPacked;
        hasChunkIndices = td.hasChunkIndexTable;
        useQkL2norm = td.enableQueryKeyL2NormInKernel;
        useGateInKernel = td.enableFusedGateSoftplus;
        useBetaSigmoid = td.enableBetaSigmoid;
        hasDtBias = td.hasGateDtBias;
        allowNegEigval = td.scaleBetaByTwoWhenNegEigval;
        useExp2 = td.useExp2ForGateCumsum;
        totalChunks = td.totalChunkTileCount;
        outputA = td.hasAOutput;
        if (chunkSize <= 0) {
            chunkSize = kGdnChunkSize;
        }
        if (K <= 0) {
            K = kGdnHeadDimK;
        }
        if (HRatio <= 0) {
            HRatio = 1;
        }
    }

    __aicore__ inline int64_t TasksPerPack() const { return kTasksPerRound; }

    __aicore__ inline int64_t PackCount() const
    {
        return (totalChunks / HV) * CeilDiv(HV, kTasksPerRound);
    }

    __aicore__ inline int64_t PackWorkId(int64_t base, int64_t, int64_t taskIdx) const
    {
        return base + taskIdx;
    }

    struct PackWalk {
        int64_t base;
        int64_t nThis;
    };

    // Never carry a partial head pack across a sequence-chunk boundary.
    __aicore__ inline PackWalk MakePackWalk(int64_t pack, int64_t packSize) const
    {
        const int64_t packsPerChunk = CeilDiv(HV, packSize);
        const int64_t headStart = (pack % packsPerChunk) * packSize;
        const int64_t remain = HV - headStart;
        return { (pack / packsPerChunk) * HV + headStart,
                 remain < packSize ? remain : packSize };
    }

    // A split HK group is recomputed independently in each pack.
    __aicore__ inline bool OwnsHk(int64_t hv, int64_t taskIdx) const
    {
        return taskIdx == 0 || hv % HRatio == 0;
    }

    __aicore__ inline int64_t OwnerTaskIdx(int64_t hv, int64_t taskIdx) const
    {
        const int64_t owner = taskIdx - hv % HRatio;
        return owner < 0 ? 0 : owner;
    }

    // Only this HV writes q_hat/k_hat/rstd, even when another pack
    // recomputes the same HK. No cross-core dependency or duplicate stores.
    __aicore__ inline bool WritesHkOutput(int64_t hv) const
    {
        return hv % HRatio == 0;
    }

    AscendC::GlobalTensor<int64_t> gmCu;
    AscendC::GlobalTensor<int64_t> gmIdx;

    int64_t B, HK, HV, T, K, V, HRatio, chunkSize;
    int64_t isVariable, hasChunkIndices;
    int64_t useQkL2norm, useGateInKernel, useBetaSigmoid, hasDtBias, allowNegEigval, useExp2;
    int64_t totalChunks, outputA;
    int64_t coreIdx, numCore, subBlock, auxReady;
};

constexpr uint32_t kPrepareKb = 1024;
constexpr uint32_t kVecFp32 = 64 * static_cast<uint32_t>(sizeof(float));

// Three 112 KiB compute slots, followed by three 12 KiB state slots.
// S1 Khat [C+64,C+80) remains live through every sibling's S6.
// W/U [C+80,C+112) remain live across next-pack S1/S3 until drained.
constexpr uint32_t kUbComputeBytes = 112 * kPrepareKb;
constexpr uint32_t kUbStateBytes = 12 * kPrepareKb;
constexpr uint32_t kUbStateBase = kTasksPerRound * kUbComputeBytes;
constexpr uint32_t kUbUsedBytes = kUbStateBase + kTasksPerRound * kUbStateBytes;
constexpr uint32_t kUbVcsIdx = kUbStateBase;
constexpr uint32_t kUbIVcs = kUbStateBase + 512;
constexpr uint32_t kUbS0Zero = 0; // released before C2

// Slot bases: compute slots are 112 KiB apart; state slots are 12 KiB apart.
constexpr uint32_t PREPARE_UB_COMPUTE_SLOT_BASE[kTasksPerRound] = {
    0, kUbComputeBytes, 2 * kUbComputeBytes};
constexpr uint32_t PREPARE_UB_STATE_SLOT_BASE[kTasksPerRound] = {
    kUbStateBase, kUbStateBase + kUbStateBytes, kUbStateBase + 2 * kUbStateBytes};
// g/beta/rstd start 9 KiB into each state slot; slot 0 also stores I/indices.
constexpr uint32_t PREPARE_UB_STATE_DATA_OFFSET = 9 * kPrepareKb;

constexpr uint32_t kUbGPrime[kTasksPerRound] = {
    PREPARE_UB_STATE_SLOT_BASE[0] + PREPARE_UB_STATE_DATA_OFFSET,
    PREPARE_UB_STATE_SLOT_BASE[1] + PREPARE_UB_STATE_DATA_OFFSET,
    PREPARE_UB_STATE_SLOT_BASE[2] + PREPARE_UB_STATE_DATA_OFFSET};
constexpr uint32_t kUbBetaEff[kTasksPerRound] = {
    PREPARE_UB_STATE_SLOT_BASE[0] + PREPARE_UB_STATE_DATA_OFFSET + kVecFp32,
    PREPARE_UB_STATE_SLOT_BASE[1] + PREPARE_UB_STATE_DATA_OFFSET + kVecFp32,
    PREPARE_UB_STATE_SLOT_BASE[2] + PREPARE_UB_STATE_DATA_OFFSET + kVecFp32};
constexpr uint32_t kUbS1KRstd[kTasksPerRound] = {
    PREPARE_UB_STATE_SLOT_BASE[0] + PREPARE_UB_STATE_DATA_OFFSET + 2 * kVecFp32,
    PREPARE_UB_STATE_SLOT_BASE[1] + PREPARE_UB_STATE_DATA_OFFSET + 2 * kVecFp32,
    PREPARE_UB_STATE_SLOT_BASE[2] + PREPARE_UB_STATE_DATA_OFFSET + 2 * kVecFp32};
constexpr uint32_t kUbS1QRstd[kTasksPerRound] = {
    PREPARE_UB_STATE_SLOT_BASE[0] + PREPARE_UB_STATE_DATA_OFFSET + 3 * kVecFp32,
    PREPARE_UB_STATE_SLOT_BASE[1] + PREPARE_UB_STATE_DATA_OFFSET + 3 * kVecFp32,
    PREPARE_UB_STATE_SLOT_BASE[2] + PREPARE_UB_STATE_DATA_OFFSET + 3 * kVecFp32};

// Compute-slot offsets retain the A6 layout; aliases below share storage across stages.
constexpr uint32_t kUbS1Q[kTasksPerRound] = {
    PREPARE_UB_COMPUTE_SLOT_BASE[0] + 16 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[1] + 16 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[2] + 16 * kPrepareKb};
constexpr uint32_t kUbS1K[kTasksPerRound] = {
    PREPARE_UB_COMPUTE_SLOT_BASE[0] + 32 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[1] + 32 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[2] + 32 * kPrepareKb};
constexpr uint32_t kUbS1QHat[kTasksPerRound] = {
    PREPARE_UB_COMPUTE_SLOT_BASE[0] + 48 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[1] + 48 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[2] + 48 * kPrepareKb};
constexpr uint32_t kUbS1KHat[kTasksPerRound] = {
    PREPARE_UB_COMPUTE_SLOT_BASE[0] + 64 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[1] + 64 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[2] + 64 * kPrepareKb};
constexpr uint32_t kUbS2Kkt[kTasksPerRound] = {
    PREPARE_UB_COMPUTE_SLOT_BASE[0],
    PREPARE_UB_COMPUTE_SLOT_BASE[1],
    PREPARE_UB_COMPUTE_SLOT_BASE[2]};
constexpr uint32_t kUbS3LPacked[kTasksPerRound] = {kUbS1Q[0], kUbS1Q[1], kUbS1Q[2]};
constexpr uint32_t kUbS3ResVcs[kTasksPerRound] = {
    PREPARE_UB_COMPUTE_SLOT_BASE[0] + 24 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[1] + 24 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[2] + 24 * kPrepareKb};
constexpr uint32_t kUbS3LFull[kTasksPerRound] = {
    PREPARE_UB_COMPUTE_SLOT_BASE[0] + 40 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[1] + 40 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[2] + 40 * kPrepareKb};
constexpr uint32_t kUbS6K[kTasksPerRound] = {kUbS1Q[0], kUbS1Q[1], kUbS1Q[2]};
constexpr uint32_t kUbS6V[kTasksPerRound] = {kUbS1K[0], kUbS1K[1], kUbS1K[2]};
constexpr uint32_t kUbS7W[kTasksPerRound] = {
    PREPARE_UB_COMPUTE_SLOT_BASE[0] + 80 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[1] + 80 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[2] + 80 * kPrepareKb};
constexpr uint32_t kUbS7U[kTasksPerRound] = {
    PREPARE_UB_COMPUTE_SLOT_BASE[0] + 96 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[1] + 96 * kPrepareKb,
    PREPARE_UB_COMPUTE_SLOT_BASE[2] + 96 * kPrepareKb};

// L1
constexpr uint32_t kBytesA64 = 64 * 64 * 2;
constexpr uint32_t kBytesK128 = 64 * 128 * 2;
constexpr uint32_t kBytesVb256 = 64 * 256 * 2;
constexpr uint32_t kBytesFp32Nz64 = 64 * 64 * 4;
constexpr uint32_t kWsYBytes = 16 * kPrepareKb;
// Three Y slots so Acc/Dump pipeline Dump(t) Fixpipe does not overwrite
// Dump(t-1) MTE2 still reading gmWsY. 64x64 fp32 NZ is 16 KiB.
constexpr uint32_t kWsYSlots = kTasksPerRound;
constexpr uint32_t kWsYElems = kWsYBytes / sizeof(float);
constexpr uint32_t kWsYTotalBytes = kWsYSlots * kWsYBytes;
// User workspace is Y only. Tail A goes L0C→L1 (same Fixpipe as outputA=0).
constexpr uint32_t kWsPerCoreBytes = kWsYTotalBytes;

__aicore__ inline int64_t WsYOffset(int64_t taskIdx)
{
    return taskIdx * static_cast<int64_t>(kWsYElems);
}

// Active Y slots occupy [0,48) KiB; k' aliases NegL at [64,112) KiB: 64x128 bf16 == 64x64 fp32 NZ.
// Intra-pack: Stage2 consumes k', Stage3 overwrites the same slots with -L.
// Cross-pack: pack N Stage5 reads Y while pack N+1 Stage1 writes k' (NegL
// already consumed in Stage4). Gate is Wait Stage4, not Stage5.
constexpr uint32_t kL1Y0 = 0;
constexpr uint32_t kL1NegL0 = 64 * kPrepareKb;
constexpr uint32_t kL1KHat0 = kL1NegL0;
constexpr uint32_t kL1LeafRight0 = 128 * kPrepareKb;
constexpr uint32_t kL1LeafLeft0 = 192 * kPrepareKb;
constexpr uint32_t kL1ResidentA0 = 256 * kPrepareKb;
constexpr uint32_t kL1ResidentKbg0 = 288 * kPrepareKb;
constexpr uint32_t kL1ResidentVb0 = 352 * kPrepareKb;
constexpr uint32_t kL1ResidentI = 480 * kPrepareKb;
// [496, 512) unused; layout kept so later L1 map stays put.

// L0 (same physical banks, typed views)
constexpr uint32_t kL0Bf16Pair = 16 * kPrepareKb;
constexpr uint32_t kL0Fp32Pair = 16 * kPrepareKb;
// L0A/L0B 16 KiB slots. Stage4/5 second MMAD (fp32) and Stage7 (bf16)
// share even [32, 48) / odd [48, 64). Stage4/5 drain before Stage7.
// SetSize 16 KiB so a view at 32 does not cover the slot at 48.
constexpr uint32_t kL0S7Slot = 16 * kPrepareKb;
constexpr uint32_t kL0S7Ping = 32 * kPrepareKb;
constexpr uint32_t kL0S7Pong = 48 * kPrepareKb;
constexpr uint32_t kL0C1 = 64 * kPrepareKb;
// Stage4/5: one 64x64 fp32 C tile per pack task. 64 KiB stride so the three
// views sit on L0C [0,192) KiB without sharing a 64 KiB bank. Tile itself is 16 KiB.
constexpr uint32_t kL0CTaskStride = 64 * kPrepareKb;
constexpr uint32_t kL0CTaskElems = kChunk64 * kChunk64;
constexpr uint32_t kL0PairOff[2] = {0, kL0Bf16Pair};
constexpr uint32_t kL0S7Off[2] = {kL0S7Ping, kL0S7Pong};
constexpr uint32_t kL0CS7Off[2] = {0, kL0C1};

__aicore__ inline uint32_t L1KHat(uint32_t taskIdx)
{
    return kL1KHat0 + taskIdx * kBytesK128;
}

__aicore__ inline uint32_t L1Y(uint32_t taskIdx)
{
    return kL1Y0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1NegL(uint32_t taskIdx)
{
    return kL1NegL0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1LeafRight(uint32_t taskIdx)
{
    return kL1LeafRight0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1LeafLeft(uint32_t taskIdx)
{
    return kL1LeafLeft0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1ResidentA(uint32_t taskIdx)
{
    return kL1ResidentA0 + taskIdx * kBytesA64;
}

__aicore__ inline uint32_t L1ResidentKbg(uint32_t taskIdx)
{
    return kL1ResidentKbg0 + taskIdx * kBytesK128;
}

__aicore__ inline uint32_t L1ResidentVb(uint32_t taskIdx)
{
    return kL1ResidentVb0 + taskIdx * kBytesVb256;
}

// One BT-token window on a single sequence (not including hv).
// chunkIdx in GetChunkRange is workId / HV.
struct ChunkRange {
    // BNSD dim-0. Varlen packed layout is always 0: sequences sit on T.
    int64_t batch;
    // Which sequence this window belongs to. Dense: same as batch.
    // Varlen: chunk_indices[2 * chunkIdx].
    int64_t seqId;
    // First token on T (or packed T). Dense: localChunk * BT.
    // Varlen: cu_seqlens[seqId] + localChunk * BT.
    int64_t tokenStart;
    // Valid tokens in this window, in 1..BT. The last window of a sequence
    // may be shorter than BT.
    int64_t M;
    // Window index within the sequence. Dense: chunkIdx % Ceil(T / BT).
    // Varlen: chunk_indices[2 * chunkIdx + 1].
    int64_t localChunk;
};

// Maps sequence-chunk index (workId / HV) to a BT-token window. No hv.
template <typename St>
__aicore__ inline ChunkRange GetChunkRange(const St &st,
                                           AscendC::GlobalTensor<int64_t> gmCuSeqlens,
                                           AscendC::GlobalTensor<int64_t> gmChunkIndices,
                                           int64_t chunkIdx)
{
    ChunkRange chunk{};
    const int64_t bt = st.chunkSize;
    if (st.isVariable != 0 && st.hasChunkIndices != 0) {
        chunk.seqId = gmChunkIndices.GetValue(chunkIdx * 2);
        chunk.localChunk = gmChunkIndices.GetValue(chunkIdx * 2 + 1);
        const int64_t seqStart = gmCuSeqlens.GetValue(chunk.seqId);
        const int64_t seqEnd = gmCuSeqlens.GetValue(chunk.seqId + 1);
        chunk.tokenStart = seqStart + chunk.localChunk * bt;
        int64_t remain = seqEnd - chunk.tokenStart;
        chunk.M = remain > bt ? bt : remain;
        chunk.batch = 0;
        return chunk;
    }
    const int64_t nt = CeilDiv(st.T, bt);
    chunk.batch = chunkIdx / nt;
    chunk.localChunk = chunkIdx % nt;
    chunk.seqId = chunk.batch;
    chunk.tokenStart = chunk.localChunk * bt;
    int64_t remain = st.T - chunk.tokenStart;
    chunk.M = remain > bt ? bt : remain;
    if (chunk.M < 0) {
        chunk.M = 0;
    }
    return chunk;
}

// Linear index of [b, h, t, 0] in GM [B, H, T, D].
__aicore__ inline int64_t OffsetBHTD(int64_t b, int64_t h, int64_t t, int64_t heads, int64_t seq, int64_t dim)
{
    return ((b * heads + h) * seq + t) * dim;
}

// Linear index of [b, h, t] in GM [B, H, T].
__aicore__ inline int64_t OffsetBHT(int64_t b, int64_t h, int64_t t, int64_t heads, int64_t seq)
{
    return (b * heads + h) * seq + t;
}

} // namespace ChunkGatedDeltaRuleFwdPrepare

#endif
