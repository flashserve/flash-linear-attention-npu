/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Shared constants, tiling, walk state, GM BNSD offsets, on-chip byte
 * offsets, UB/L1/L0 views, ping-pong, and CrossCore Notify/Wait helpers.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_PREPARE_COMMON_H
#define CHUNK_GATED_DELTA_RULE_FWD_PREPARE_COMMON_H

#include "kernel_operator.h"

struct HardwareInfo {
    static uint32_t const l1Size = 512 * 1024;
    static uint32_t const l0ASize = 64 * 1024;
    static uint32_t const l0BSize = 64 * 1024;
    static uint32_t const l0CSize = 256 * 1024;
    static uint32_t const l2Size = 128 * 1024 * 1024;
    static uint32_t const biasSize = 4 * 1024;
    static uint32_t const fixBufSize = 6 * 1024;
    static uint32_t const ubSize = 248 * 1024;
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

    // G = Hv / Hk in {1,2,3,4}. hk = hv / G. OwnsHk(hv) is (hv % G) == 0.
    int64_t valueHeadsPerQueryKeyHead;

    // BT. Tokens in one chunk tile. First version is 64. Last chunk may be shorter.
    int64_t tokensPerChunk;

    // N. Sequence count used with cu_seqlens: B if fixed-length, else cu_seqlens.numel()-1.
    int64_t packedSequenceCount;

    // 1: T is packed varlen and cu_seqlens is present. 0: dense BNSD.
    int64_t isVariableLengthPacked;

    // 1: chunk_indices GM is [2 * numChunks] pairs (seqId, localChunk).
    int64_t hasChunkIndexTable;

    // Always 1 this version: q' = q/||q||, k' = k/||k||; write q_hat, k_hat, rstd.
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

    // Host GetDataSize+8 pad. Layout must match ChunkGatedDeltaRuleFwdPrepareTilingData.
    int64_t reservedEightByteAlignPad;
};


namespace ChunkGatedDeltaRuleFwdPrepare {
using namespace AscendC;

constexpr int64_t kTasksPerRound = 4;
// dav_3510 MIX 1:2: AIV1 set_intra_block(id) == AIC wait_intra_block(id+16).
constexpr uint16_t kAiv1IntraFlagOff = 16;
// Stage3→Stage4 per-task ids 4,5,6,7 (AIC sees AIV1 as 20,21,22,23).
// Separate from Stage1/2/3 ids 0..3 so AIC's Stage4 Wait cannot steal the
// Stage2→Stage3 notify.
constexpr uint16_t kFlagS3DoneBase = 4;
// Pack N Stage4 dumped Y / freed NegL → pack N+1 Stage1 may write k'.
// Ids 8..11 (AIV1 24..27). Same ids V3 used as kFlagS4DumpBase.
constexpr uint16_t kFlagS4DumpBase = 8;
// Stage6→Stage7 reuses Stage3 ids 4..7 (AIC sees AIV1 as 20,21,22,23).
// Those are consumed after Stage4. Ids 0..3 still carry the Stage2 kkt
// notify on PIPE_FIX; a PIPE_MTE1 Wait(0) can steal that leftover and
// Stage7 runs before L1 vb/kbg exist. Ids 12..15 are ignored by intra-block.
constexpr uint16_t kFlagS6DoneBase = 4;
// Pack N Stage7 finished with L1 kbg/vb. Pack N+1 Stage6 may then overwrite
// those slots. Mode 2 id 0xF. Stage1 still uses the Stage4 pack gate so it
// can overlap pack N Stage5/7.
constexpr uint16_t kFlagS7Done = 0xF;

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

// AIV0: task 0 ping, 2 pong. AIV1: task 1 ping, 3 pong.
__aicore__ inline int32_t PingPongSlot(int64_t taskIdx)
{
    return static_cast<int32_t>((taskIdx >> 1) & 1);
}

// MIX 1:2 intra-block: AIC Wait of an AIV1 Set(id) uses id+16.
__aicore__ inline uint16_t CubeWaitFlagForTask(int64_t taskIdx)
{
    uint16_t flag = static_cast<uint16_t>(taskIdx);
    if ((taskIdx & 1) != 0) {
        flag += kAiv1IntraFlagOff;
    }
    return flag;
}

// Stage1 k' on L1 → Stage2. AIV Sets raw taskIdx; AIC Waits +16 on AIV1.
__aicore__ inline void NotifyAicStage1Done(int64_t taskIdx)
{
    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx));
}

__aicore__ inline void WaitAivStage1Done(int64_t taskIdx)
{
    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlagForTask(taskIdx));
}

// Same ids as Stage1 (0,1,2,3). AIC already Waited this id for k' ready,
// then Sets it again after kkt Fixpipe so that AIV can Stage3.
// AIV waits the raw id; AIC Sets id+16 for AIV1.
__aicore__ inline void NotifyAivKktDone(int64_t taskIdx)
{
    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(CubeWaitFlagForTask(taskIdx));
}

__aicore__ inline void WaitCubeKktDone(int64_t taskIdx)
{
    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(taskIdx));
}

// Stage3 L1 ready → Stage4. AIV Sets raw taskIdx+4; AIC Waits +16 on AIV1.
__aicore__ inline void NotifyAicStage3Done(int64_t taskIdx)
{
    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx + kFlagS3DoneBase));
}

__aicore__ inline void WaitAivStage3Done(int64_t taskIdx)
{
    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlagForTask(taskIdx + kFlagS3DoneBase));
}

// Stage4 Y on L1 / NegL free. Pack N+1 Stage1 waits this before k' L1 write.
__aicore__ inline void NotifyAivStage4Done(int64_t taskIdx)
{
    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(CubeWaitFlagForTask(taskIdx + kFlagS4DumpBase));
}

__aicore__ inline void WaitAicStage4Done(int64_t taskIdx)
{
    AscendC::CrossCoreWaitFlag<0x4>(static_cast<uint16_t>(taskIdx + kFlagS4DumpBase));
}

// Stage6 L1 vb/kbg ready → Stage7. Reuse Stage3 ids 4..7 (consumed
// after Stage4). AIV Sets raw taskIdx+4; AIC Waits +16 on AIV1.
// Intra-block 12..15 is ignored on this SoC (Wait returns immediately).
__aicore__ inline void NotifyAicStage6Done(int64_t taskIdx)
{
    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx + kFlagS6DoneBase));
}

__aicore__ inline void WaitAivStage6Done(int64_t taskIdx)
{
    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlagForTask(taskIdx + kFlagS6DoneBase));
}

__aicore__ inline void NotifyAivStage7Done()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagS7Done);
}

__aicore__ inline void WaitAicStage7Done()
{
    AscendC::CrossCoreWaitFlag<0x2>(kFlagS7Done);
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

    // G=3 cannot fill 4 consecutive HV of one HK, so a pack is one HK group
    // (nThis=3). G=1/2/4 fill a pack of 4: 4 HK, 2 HK, or 1 HK.
    __aicore__ inline int64_t TasksPerPack() const
    {
        return (HRatio == 3) ? 3 : kTasksPerRound;
    }

    // First HV of an HK group owns L2Norm(q/k) and Cube kkt.
    __aicore__ inline bool OwnsHk(int64_t hv) const
    {
        return (HRatio <= 1) || ((hv % HRatio) == 0);
    }

    // Pack-local task that produced kkt for this hv. Packing keeps the owner
    // at taskIdx - (hv % G); never negative when TasksPerPack() is used.
    __aicore__ inline int64_t OwnerTaskIdx(int64_t hv, int64_t taskIdx) const
    {
        if (HRatio <= 1) {
            return taskIdx;
        }
        const int64_t owner = taskIdx - (hv % HRatio);
        return (owner < 0) ? taskIdx : owner;
    }

    AscendC::GlobalTensor<int64_t> gmCu;
    AscendC::GlobalTensor<int64_t> gmIdx;

    int64_t B, HK, HV, T, K, V, HRatio, chunkSize;
    int64_t isVariable, hasChunkIndices;
    int64_t useQkL2norm, useGateInKernel, useBetaSigmoid, hasDtBias, allowNegEigval, useExp2;
    int64_t totalChunks;
    int64_t coreIdx, numCore, subBlock, auxReady;
};

constexpr uint32_t kPrepareKb = 1024;
constexpr uint32_t kVecFp32 = 64 * static_cast<uint32_t>(sizeof(float));

// UB resident. [0, 0.5) KiB is an unused hole so later offsets stay put.
constexpr uint32_t kUbVcsIdx = 512;
constexpr uint32_t kUbIVcs = 1 * kPrepareKb;
constexpr uint32_t kUbGPrime0 = 9 * kPrepareKb;
constexpr uint32_t kUbBetaEff0 = 9 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbGPrime1 = 9 * kPrepareKb + 2 * kVecFp32;
constexpr uint32_t kUbBetaEff1 = 9 * kPrepareKb + 3 * kVecFp32;

// UB S0 temps (released before S1 writes g'). [9, 25) unused so later map stays put.
constexpr uint32_t kUbS0Zero = 25 * kPrepareKb;

// UB S1 input db [10, 74). One AIV, two tasks: task 0/1 → ping, task 2/3 → pong.
constexpr uint32_t kUbS1QPing = 10 * kPrepareKb;
constexpr uint32_t kUbS1KPing = 26 * kPrepareKb;
constexpr uint32_t kUbS1QPong = 42 * kPrepareKb;
constexpr uint32_t kUbS1KPong = 58 * kPrepareKb;
// Hat ping [75, 107), pong in the hole before kkt [108, 140).
constexpr uint32_t kUbS1QHatPing = 75 * kPrepareKb;
constexpr uint32_t kUbS1KHatPing = 91 * kPrepareKb;
constexpr uint32_t kUbS1QHatPong = 108 * kPrepareKb;
constexpr uint32_t kUbS1KHatPong = 124 * kPrepareKb;
// Rstd ping next to ping k'; pong reuses [74, 75).
constexpr uint32_t kUbS1KRstdPing = 107 * kPrepareKb;
constexpr uint32_t kUbS1QRstdPing = 107 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbS1KRstdPong = 74 * kPrepareKb;
constexpr uint32_t kUbS1QRstdPong = 74 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbS1QHat[2] = {kUbS1QHatPing, kUbS1QHatPong};
constexpr uint32_t kUbS1KHat[2] = {kUbS1KHatPing, kUbS1KHatPong};
constexpr uint32_t kUbS1KRstd[2] = {kUbS1KRstdPing, kUbS1KRstdPong};
constexpr uint32_t kUbS1QRstd[2] = {kUbS1QRstdPing, kUbS1QRstdPong};

constexpr uint32_t kUbGPrime[2] = {kUbGPrime0, kUbGPrime1};
constexpr uint32_t kUbBetaEff[2] = {kUbBetaEff0, kUbBetaEff1};
constexpr uint32_t kUbS1Q[2] = {kUbS1QPing, kUbS1QPong};
constexpr uint32_t kUbS1K[2] = {kUbS1KPing, kUbS1KPong};

// UB S2
constexpr uint32_t kUbS2KktPing = 140 * kPrepareKb;
constexpr uint32_t kUbS2KktPong = 156 * kPrepareKb;
constexpr uint32_t kUbS2Kkt[2] = {kUbS2KktPing, kUbS2KktPong};

// UB S3 (overlaps S1 input after k' is on L1/GM). Ping db=0 / pong db=1.
// Pong reuses the old Nz16/Nz8/LeafTmp hole [50, 90).
constexpr uint32_t kUbS3LPacked[2] = {10 * kPrepareKb, 50 * kPrepareKb};
constexpr uint32_t kUbS3ResVcs[2] = {18 * kPrepareKb, 58 * kPrepareKb};
constexpr uint32_t kUbS3LFull[2] = {34 * kPrepareKb, 74 * kPrepareKb};
constexpr uint32_t kUbMaskFp32 = 172 * kPrepareKb;

// UB S6 (after S3; overlaps S1/S3). K' 16 KiB. V=128 tile is 16 KiB,
// V=256 tile is 32 KiB. After Stage3, rstd/hat at [107,140) are free.
// V=128: ping [64,80) pong [80,96). V=256: ping [64,96) pong [96,128).
constexpr uint32_t kUbS6KPing = 32 * kPrepareKb;
constexpr uint32_t kUbS6KPong = 48 * kPrepareKb;
constexpr uint32_t kUbS6VPing = 64 * kPrepareKb;
constexpr uint32_t kUbS6VPong = 80 * kPrepareKb;
constexpr uint32_t kUbS6VPong256 = 96 * kPrepareKb;
constexpr uint32_t kUbS6K[2] = {kUbS6KPing, kUbS6KPong};
constexpr uint32_t kUbS6V[2] = {kUbS6VPing, kUbS6VPong};

// L1
constexpr uint32_t kBytesA64 = 64 * 64 * 2;
constexpr uint32_t kBytesK128 = 64 * 128 * 2;
constexpr uint32_t kBytesVb256 = 64 * 256 * 2;
constexpr uint32_t kBytesFp32Nz64 = 64 * 64 * 4;
constexpr uint32_t kWsYBytes = 16 * kPrepareKb;
// One 64x64 bf16 ND tile is 8 KiB; slot kept at 16 KiB. Four slots so pack
// tasks 0..3 do not share gmWsA (Stage5 Fixpipe vs in-flight MTE2 Copy).
constexpr uint32_t kWsABytes = 16 * kPrepareKb;
constexpr uint32_t kWsAElems = kWsABytes / 2;
constexpr uint32_t kWsASlots = 4;
constexpr uint32_t kWsATotalBytes = kWsASlots * kWsABytes;
constexpr uint32_t kWsPerCoreBytes = 128 * kPrepareKb;

__aicore__ inline int64_t WsAOffset(int64_t taskIdx)
{
    return taskIdx * static_cast<int64_t>(kWsAElems);
}

// Y owns [0, 64). k' aliases NegL at [64, 128): 64x128 bf16 == 64x64 fp32 NZ.
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
