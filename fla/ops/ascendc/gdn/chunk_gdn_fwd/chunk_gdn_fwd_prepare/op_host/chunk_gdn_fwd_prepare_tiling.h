/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Host tiling for fused ChunkGdnFwdPrepare. Field names, types, and order
 * (except the three prepare-only fields) must match ChunkGdnFwdStageTilingData.
 */

#ifndef CHUNK_GDN_FWD_PREPARE_TILING_H
#define CHUNK_GDN_FWD_PREPARE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkGdnFwdPrepareTilingData)
    // B. BNSD dim-0. Varlen packed layout is 1.
    TILING_DATA_FIELD_DEF(int64_t, inputBatchSize);
    // Hk. Query/key head count. q/k GM [B, Hk, T, K].
    TILING_DATA_FIELD_DEF(int64_t, queryKeyHeadCount);
    // Hv. Value/gate/beta/A head count. Must satisfy Hv % Hk == 0.
    TILING_DATA_FIELD_DEF(int64_t, valueHeadCount);
    // T. Tokens per batch, or packed token total if varlen (cu_seqlens[-1]).
    TILING_DATA_FIELD_DEF(int64_t, sequenceTokenLength);
    // K. q/k channel. First version is 128.
    TILING_DATA_FIELD_DEF(int64_t, queryKeyHeadDim);
    // V. v/u channel. First version is 128 or 256.
    TILING_DATA_FIELD_DEF(int64_t, valueHeadDim);
    // G = Hv / Hk in {1,2,3,4}. One k' kkt is reused by G value heads.
    TILING_DATA_FIELD_DEF(int64_t, valueHeadsPerQueryKeyHead);
    // BT. Tokens per chunk tile. First version is 64.
    TILING_DATA_FIELD_DEF(int64_t, tokensPerChunk);
    // N. Sequence count: B, or cu_seqlens.numel()-1 when packed.
    TILING_DATA_FIELD_DEF(int64_t, packedSequenceCount);
    // 1 if cu_seqlens is present (packed varlen). 0 if dense BNSD.
    TILING_DATA_FIELD_DEF(int64_t, isVariableLengthPacked);
    // 1 if chunk_indices [2*NT] (seqId, localChunk) pairs are present.
    TILING_DATA_FIELD_DEF(int64_t, hasChunkIndexTable);
    // Always 1 this version: in-kernel L2norm of q/k, write q_hat / k_hat / rstd.
    TILING_DATA_FIELD_DEF(int64_t, enableQueryKeyL2NormInKernel);
    // 1: g_raw = -exp(a_log)*softplus(g+dt_bias) before chunk-local cumsum.
    TILING_DATA_FIELD_DEF(int64_t, enableFusedGateSoftplus);
    // 1: beta_eff = sigmoid(beta). 0: fp32 copy of beta.
    TILING_DATA_FIELD_DEF(int64_t, enableBetaSigmoid);
    // 1: dt_bias[Hv] is present (only with fused gate).
    TILING_DATA_FIELD_DEF(int64_t, hasGateDtBias);
    // Prepare-only. 1: q_hat GM output is allocated.
    TILING_DATA_FIELD_DEF(int64_t, hasQueryHatGmOutput);
    // Prepare-only. 1: k_hat GM output is allocated.
    TILING_DATA_FIELD_DEF(int64_t, hasKeyHatGmOutput);
    // 1 and sigmoid on: beta_eff = 2 * sigmoid(beta).
    TILING_DATA_FIELD_DEF(int64_t, scaleBetaByTwoWhenNegEigval);
    // 1: g' = RCP_LN2 * chunk_cumsum(g_raw); later exp uses exp2.
    TILING_DATA_FIELD_DEF(int64_t, useExp2ForGateCumsum);
    // ge::DataType of q/k (bf16 == 27).
    TILING_DATA_FIELD_DEF(int64_t, queryKeyStorageDtype);
    // ge::DataType of raw gate g.
    TILING_DATA_FIELD_DEF(int64_t, gateStorageDtype);
    // ge::DataType of raw beta.
    TILING_DATA_FIELD_DEF(int64_t, betaStorageDtype);
    // Tile count this kernel walks: Ceil(T/BT)*B, or NT if varlen indices.
    TILING_DATA_FIELD_DEF(int64_t, totalChunkTileCount);
    // Prepare-only. Unused pad so host GetDataSize matches device opParaSize+8.
    TILING_DATA_FIELD_DEF(int64_t, reservedEightByteAlignPad);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkGdnFwdPrepare, ChunkGdnFwdPrepareTilingData)

constexpr size_t PREPARE_INPUT_Q = 0;
constexpr size_t PREPARE_INPUT_K = 1;
constexpr size_t PREPARE_INPUT_V = 2;
constexpr size_t PREPARE_INPUT_G = 3;
constexpr size_t PREPARE_INPUT_BETA = 4;
constexpr size_t PREPARE_INPUT_A_LOG = 5;
constexpr size_t PREPARE_INPUT_DT_BIAS = 6;
constexpr size_t PREPARE_INPUT_CU_SEQLENS = 7;
constexpr size_t PREPARE_INPUT_CHUNK_INDICES = 8;

constexpr size_t PREPARE_OUTPUT_G = 0;
constexpr size_t PREPARE_OUTPUT_W = 1;
constexpr size_t PREPARE_OUTPUT_U = 2;
constexpr size_t PREPARE_OUTPUT_A = 3;
constexpr size_t PREPARE_OUTPUT_Q_HAT = 4;
constexpr size_t PREPARE_OUTPUT_K_HAT = 5;
constexpr size_t PREPARE_OUTPUT_Q_RSTD = 6;
constexpr size_t PREPARE_OUTPUT_K_RSTD = 7;
constexpr size_t PREPARE_OUTPUT_BETA_EFF = 8;

constexpr size_t PREPARE_ATTR_CHUNK_SIZE = 0;
constexpr size_t PREPARE_ATTR_ALLOW_NEG_EIGVAL = 1;
constexpr size_t PREPARE_ATTR_USE_EXP2 = 2;
constexpr size_t PREPARE_ATTR_USE_QK_L2NORM = 3;
constexpr size_t PREPARE_ATTR_USE_GATE = 4;
constexpr size_t PREPARE_ATTR_USE_BETA_SIGMOID = 5;

} // namespace optiling

#endif
