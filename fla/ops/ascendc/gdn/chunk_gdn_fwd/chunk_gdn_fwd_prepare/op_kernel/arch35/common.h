/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Shared constants, tiling, and walk state for fused chunk_gdn_fwd_prepare.
 */

#ifndef CHUNK_GDN_FWD_PREPARE_COMMON_H
#define CHUNK_GDN_FWD_PREPARE_COMMON_H

#include "kernel_operator.h"

constexpr int64_t kGdnChunkSize = 64;
constexpr int64_t kGdnHeadDimK = 128;
constexpr float kGdnRcpLn2 = 1.4426950216f;
constexpr float kGdnL2NormEps = 1e-6f;
constexpr float kGdnGateClip = 50.0f;

struct ChunkGdnFwdStageTilingData {
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

    // Host GetDataSize+8 pad. Layout must match ChunkGdnFwdPrepareTilingData.
    int64_t reservedEightByteAlignPad;
};


namespace GdnStage {

constexpr int64_t kTasksPerRound = 4;
// dav_3510 MIX 1:2: AIV1 set_intra_block(id) == AIC wait_intra_block(id+16).
constexpr uint16_t kAiv1IntraFlagOff = 16;
// Stage3→Stage4 per-task ids 4,5,6,7 (AIC sees AIV1 as 20,21,22,23).
// Separate from Stage1/2/3 ids 0..3 so AIC's Stage4 Wait cannot steal the
// Stage2→Stage3 notify.
constexpr uint16_t kFlagS3DoneBase = 4;
// Stage4 done (NegL consumed). Pack N+1 Stage1 may then write k' onto the
// NegL slots. AIC Sets taskIdx+8 (AIV1 observed as 24, 25, 26, 27).
constexpr uint16_t kFlagS4DumpBase = 8;
constexpr uint16_t kFlagS4Ready = 0x8;
constexpr uint16_t kFlagS5Done = 0x9;
// Stage6→Stage7 reuses Stage3 ids 4..7 (AIC sees AIV1 as 20,21,22,23).
// Those are consumed after Stage4. Ids 0..3 still carry the Stage2 kkt
// notify on PIPE_FIX; a PIPE_MTE1 Wait(0) can steal that leftover and
// Stage7 runs before L1 vb/kbg exist. Ids 12..15 are ignored by intra-block.
constexpr uint16_t kFlagS6DoneBase = 4;
constexpr uint16_t kFlagS6Ready = 0xC;
constexpr uint16_t kFlagS7Done = 0xF;

constexpr uint32_t kChunk64 = 64;
constexpr int32_t kNumMFracs64 = 4;
constexpr int32_t kNumNFracs64 = 8;

constexpr AscendC::FixpipeConfig CFG_NZ_L1 = {AscendC::CO2Layout::NZ, false};
constexpr AscendC::FixpipeConfig CFG_NZ_UB = {AscendC::CO2Layout::NZ, true};
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr int64_t DATA_BLOCK_COUNT = 16;
constexpr int64_t DATA_BLOCK_COUNT_HALF = 8;
constexpr int32_t kFracLen = 16 * 16;
constexpr int32_t kFracLen8 = 16 * 8;
constexpr int32_t CONST_TWO = 2;

constexpr uint32_t kVcs32 = 32;
constexpr uint32_t kVcsPack32 = 64;
constexpr uint32_t kVcsPackedElems32 = kVcs32 * kVcsPack32;
constexpr uint32_t kLeavesPerVec32 = 2;
constexpr uint32_t kVcs32NzElems = kVcs32 * kVcs32;

__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
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
    AscendC::GlobalTensor<int64_t> gmCu;
    AscendC::GlobalTensor<int64_t> gmIdx;

    int64_t B, HK, HV, T, K, V, HRatio, chunkSize;
    int64_t isVariable, hasChunkIndices;
    int64_t useQkL2norm, useGateInKernel, useBetaSigmoid, hasDtBias, allowNegEigval, useExp2;
    int64_t totalChunks;
    int64_t coreIdx, numCore, subBlock, auxReady;
};

} // namespace GdnStage

#endif
