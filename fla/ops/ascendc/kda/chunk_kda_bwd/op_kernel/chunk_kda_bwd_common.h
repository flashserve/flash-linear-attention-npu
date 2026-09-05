#ifndef CHUNK_KDA_BWD_COMMON_H
#define CHUNK_KDA_BWD_COMMON_H

#include <cstdint>

#ifndef CHUNK_KDA_BWD_A_STRUCT_H
#define CHUNK_KDA_BWD_A_STRUCT_H

#include <cstdint>

namespace KDA {

// Private L0 contract for the first fused backward kernel.  The public L2
// wrapper normalizes BSND/TND inputs to the head-major storage used here.
struct ChunkKdaBwdATilingData {
    // These are shape/count scalars, not byte offsets.  The supported KDA
    // envelope keeps each value below INT32_MAX.  Compact fields leave room
    // for the unchanged PR291 Kernel-B tiling in the unified 512-byte ABI.
    int32_t headNum;
    int32_t seqlen;
    int32_t chunkSize;
    int32_t chunkNum;
    int32_t chunkNumPerBatch;
    int32_t isVarLen;
    int32_t usedCoreNum;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_A_STRUCT_H

#ifndef CHUNK_KDA_BWD_C_STRUCT_H
#define CHUNK_KDA_BWD_C_STRUCT_H

#include <cstdint>

namespace KDA {

struct ChunkKdaBwdCTilingData {
    // P0 bounds (C=64, K=128, V<=256 and practical B/H/T) keep every
    // scalar and per-core byte offset below INT32_MAX.  Using int32_t keeps
    // the complete MIX kernel argument block below the 512-byte device ABI
    // boundary; int64_t fields made the otherwise no-op kernel exceed it.
    int32_t seqlen;
    int32_t chunkNum;
    int32_t workspaceSlotSize;
    int32_t workspaceCoreSize;
    int16_t batch;
    int16_t headNum;
    int16_t chunkNumPerBatch;
    int16_t chunkSize;
    int16_t keyDim;
    int16_t valueDim;
    int16_t workspaceSlotCount;
    int16_t usedCoreNum;
    int8_t isVarLen;
    // PR291 Kernel B writes dh as [B,H,NT,K,V] for dense and
    // [1,H,totalChunks,K,V] for varlen. Saved h remains chunk-major.
    int8_t dhHeadMajor;
    int8_t useGateInKernel;
    int8_t hasDtBias;
    int8_t deferGatePost;
    // raw_g is optional, so the generated A5 binary key does not specialize
    // DTYPE_RAW_G. Carry its runtime storage type explicitly instead.
    int8_t rawGateIsBf16;
    int32_t kEOffset;
    int32_t dqRawOffset;
    int32_t dkRawOffset;
    int32_t dWOffset;
    int32_t zVOffset;
    int32_t dVbOffset;
    int32_t zWOffset;
    int32_t dKgbOffset;
    int32_t zaInputOffset;
    int32_t zaOutputOffset;
    int32_t intraALowerOffset;
    int32_t intraBLowerOffset;
    int32_t intraAUpperOffset;
    int32_t intraBUpperOffset;
    int32_t intraResultRegionOffset;
    int32_t intraResultDqOffset;
    int32_t intraResultDkLowerOffset;
    int32_t intraResultDkUpperOffset;
    float scale;
    float lowerBound;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_STRUCT_H

#define KDA_BWD_EMBEDDED_DHU 1
/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_kda_bwd_common.h
 * \brief Shared tiling data for chunk_gated_delta_rule_bwd_dhu.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_STRUCT_H

#include <cstdint>

#if !defined(TORCH_MODE) && !defined(KDA_BWD_EMBEDDED_DHU)
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace GDN {

#define TPL_BF16 10
#define TPL_FP16 20
#define TPL_FP32 30

#if !defined(TORCH_MODE) && !defined(KDA_BWD_EMBEDDED_DHU)
ASCENDC_TPL_ARGS_DECL(ChunkGatedDeltaRuleBwdDhu,
    ASCENDC_TPL_DTYPE_DECL(D_T_Q, TPL_BF16,
        TPL_FP16),
    ASCENDC_TPL_DTYPE_DECL(D_T_G, TPL_BF16,
        TPL_FP16,
        TPL_FP32),
    ASCENDC_TPL_UINT_DECL(V, 1, ASCENDC_TPL_UI_LIST, 128),
    ASCENDC_TPL_UINT_DECL(USE_GK, 1, ASCENDC_TPL_UI_LIST, 0, 1),
);

#define TPL_SEL_ONE(Q_TYPE, G_TYPE, V_VALUE, USE_GK_VALUE) \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_DTYPE_SEL(D_T_Q, Q_TYPE), \
        ASCENDC_TPL_DTYPE_SEL(D_T_G, G_TYPE), \
        ASCENDC_TPL_UINT_SEL(V, ASCENDC_TPL_UI_LIST, V_VALUE), \
        ASCENDC_TPL_UINT_SEL(USE_GK, ASCENDC_TPL_UI_LIST, USE_GK_VALUE), \
    )

#define TPL_SEL_FOR_PAIR(Q_TYPE, G_TYPE, USE_GK_VALUE) \
    TPL_SEL_ONE(Q_TYPE, G_TYPE, 128, USE_GK_VALUE), \
    TPL_SEL_ONE(Q_TYPE, G_TYPE, 256, USE_GK_VALUE)

#define TPL_SEL_FOR_GATE(USE_GK_VALUE) \
    TPL_SEL_FOR_PAIR(TPL_BF16, TPL_FP32, USE_GK_VALUE), \
    TPL_SEL_FOR_PAIR(TPL_FP16, TPL_FP32, USE_GK_VALUE), \
    TPL_SEL_FOR_PAIR(TPL_BF16, TPL_BF16, USE_GK_VALUE), \
    TPL_SEL_FOR_PAIR(TPL_FP16, TPL_FP16, USE_GK_VALUE)

ASCENDC_TPL_SEL(
    TPL_SEL_FOR_GATE(0),
    TPL_SEL_FOR_GATE(1),
);
#undef TPL_SEL_FOR_GATE
#undef TPL_SEL_FOR_PAIR
#undef TPL_SEL_ONE
#endif

#if defined(KDA_BWD_EMBEDDED_DHU)
// The standalone PR291 contract keeps int64_t metadata.  In the fused KDA
// entry all supported P0 dimensions, counters and per-owner offsets are
// bounded by int32_t.  A compact embedded view keeps the combined tiling
// payload well below A5's 512-byte kernel-parameter boundary.
using ChunkGatedDeltaRuleBwdDhuTilingInt = int32_t;
using ChunkGatedDeltaRuleBwdDhuTilingSmallInt = int16_t;
#else
using ChunkGatedDeltaRuleBwdDhuTilingInt = int64_t;
using ChunkGatedDeltaRuleBwdDhuTilingSmallInt = int64_t;
#endif

struct ChunkGatedDeltaRuleBwdDhuTilingData {
    ChunkGatedDeltaRuleBwdDhuTilingInt B;
    ChunkGatedDeltaRuleBwdDhuTilingInt HK;
    ChunkGatedDeltaRuleBwdDhuTilingInt HV;
    ChunkGatedDeltaRuleBwdDhuTilingInt T;
    ChunkGatedDeltaRuleBwdDhuTilingInt K;
    ChunkGatedDeltaRuleBwdDhuTilingInt V;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt HRatio;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt chunkSize;
    ChunkGatedDeltaRuleBwdDhuTilingInt chunkNumForT;
    ChunkGatedDeltaRuleBwdDhuTilingInt totalChunkNum;
    ChunkGatedDeltaRuleBwdDhuTilingInt chunkTaskNum;
    ChunkGatedDeltaRuleBwdDhuTilingInt seqNum;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt headWindowNum;
    ChunkGatedDeltaRuleBwdDhuTilingInt taskNum;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt isVariable;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt hasDh0;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt dh0ClearCoreNum;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt dh0ClearElemsPerCore;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt dh0ClearTailElems;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt hasGk;
    ChunkGatedDeltaRuleBwdDhuTilingInt workspaceElemsPerSubBlock;
    ChunkGatedDeltaRuleBwdDhuTilingInt qgWorkspaceOffset;
    ChunkGatedDeltaRuleBwdDhuTilingInt stateWorkspaceOffset;
    ChunkGatedDeltaRuleBwdDhuTilingInt dvStateWorkspaceOffset;
    ChunkGatedDeltaRuleBwdDhuTilingInt termQWorkspaceOffset;
    ChunkGatedDeltaRuleBwdDhuTilingInt dv2WorkspaceOffset;
    ChunkGatedDeltaRuleBwdDhuTilingInt termWWorkspaceOffset;
    ChunkGatedDeltaRuleBwdDhuTilingInt qgWorkspaceElems;
    ChunkGatedDeltaRuleBwdDhuTilingInt stateWorkspaceElems;
    ChunkGatedDeltaRuleBwdDhuTilingInt dvStateWorkspaceElems;
    ChunkGatedDeltaRuleBwdDhuTilingInt termQWorkspaceElems;
    ChunkGatedDeltaRuleBwdDhuTilingInt dv2WorkspaceElems;
    ChunkGatedDeltaRuleBwdDhuTilingInt termWWorkspaceElems;
    ChunkGatedDeltaRuleBwdDhuTilingSmallInt vecRow;
    float scale;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_STRUCT_H

#undef KDA_BWD_EMBEDDED_DHU

namespace KDA {

// CATLASS exposes the L0C epilogue under different names on the two
// architecture families used by this operator.  Keep that API difference in
// one place so the shared A/C kernels can use the same implementation.
template <class TileCopy, class TensorC>
struct KdaBwdCopyL0CToDstSelector {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using Type = typename TileCopy::template CopyL0CToDst<TensorC>;
#else
    using Type = typename TileCopy::template CopyL0CToGm<TensorC>;
#endif
};

template <class TileCopy, class TensorC>
using KdaBwdCopyL0CToDst =
    typename KdaBwdCopyL0CToDstSelector<TileCopy, TensorC>::Type;

struct ChunkKdaBwdTilingData {
    GDN::ChunkGatedDeltaRuleBwdDhuTilingData kernelB;
    ChunkKdaBwdCTilingData kernelC;
    // Cross-phase tensors are private workspace regions, not public kernel
    // parameters.  This keeps the single device entry comfortably below the
    // A5 kernel-argument limit and mirrors chunk_kda_fwd's address plan.
    uint32_t dv0Offset;
    uint32_t dqRawOffset;
    uint32_t dAqkOffset;
    uint32_t dhOffset;
    uint32_t dvScanOffset;
    uint32_t dAkkOffset;
    uint32_t kernelBWorkspaceOffset;
    uint32_t kernelCWorkspaceOffset;
};

// Keep the fused raw-struct contract no larger than the proven standalone
// PR291 tiling envelope. The generated Ascend950 payload is 280 bytes.
static_assert(sizeof(ChunkKdaBwdTilingData) <= 288,
              "ChunkKdaBwdTilingData exceeds the fused A5 budget");

} // namespace KDA

#endif // CHUNK_KDA_BWD_COMMON_H
