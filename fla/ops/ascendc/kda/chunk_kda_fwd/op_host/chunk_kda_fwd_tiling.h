/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

struct ChunkKdaPrepareTilingData {
    int64_t usedCoreNum;
    int64_t qgScaledOffset;
    int64_t wSeedOffset;
    int64_t uSeedOffset;
    int64_t aqkFp32Offset;
    int64_t akkFp32Offset;
    int64_t scratchOffset;
};

struct ChunkKdaPostWuTilingData {
    int64_t usedCoreNum;
    int64_t qgScaledOffset;
    int64_t wSeedOffset;
    int64_t uSeedOffset;
    int64_t scratchOffset;
};

struct ChunkKdaFwdHStageTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t numSeqWorkspaceOffset;
    int64_t numChunksWorkspaceOffset;
    int64_t workspaceBaseOffset;
};

struct ChunkKdaOutputTilingData {
    int64_t usedCoreNum;
    int64_t qgScaledOffset;
    int64_t scratchOffset;
};

BEGIN_TILING_DATA_DEF(ChunkKdaFwdTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seqNum);
TILING_DATA_FIELD_DEF(int64_t, qHeadNum);
TILING_DATA_FIELD_DEF(int64_t, vHeadNum);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, totalChunks);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(bool, hasInitialState);
TILING_DATA_FIELD_DEF(bool, outputFinalState);
TILING_DATA_FIELD_DEF(bool, isVarLen);
TILING_DATA_FIELD_DEF(bool, safeGate);
TILING_DATA_FIELD_DEF(int64_t, prepareUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, prepareQgScaledOffset);
TILING_DATA_FIELD_DEF(int64_t, prepareWSeedOffset);
TILING_DATA_FIELD_DEF(int64_t, prepareUSeedOffset);
TILING_DATA_FIELD_DEF(int64_t, prepareAqkFp32Offset);
TILING_DATA_FIELD_DEF(int64_t, prepareAkkFp32Offset);
TILING_DATA_FIELD_DEF(int64_t, prepareScratchOffset);
TILING_DATA_FIELD_DEF(int64_t, postWuUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, postWuQgScaledOffset);
TILING_DATA_FIELD_DEF(int64_t, postWuWSeedOffset);
TILING_DATA_FIELD_DEF(int64_t, postWuUSeedOffset);
TILING_DATA_FIELD_DEF(int64_t, postWuScratchOffset);
TILING_DATA_FIELD_DEF(int64_t, fwdHBatch);
TILING_DATA_FIELD_DEF(int64_t, fwdHSeqlen);
TILING_DATA_FIELD_DEF(int64_t, fwdHKNumHead);
TILING_DATA_FIELD_DEF(int64_t, fwdHVNumHead);
TILING_DATA_FIELD_DEF(int64_t, fwdHKHeadDim);
TILING_DATA_FIELD_DEF(int64_t, fwdHVHeadDim);
TILING_DATA_FIELD_DEF(int64_t, fwdHChunkSize);
TILING_DATA_FIELD_DEF(bool, fwdHUseInitialState);
TILING_DATA_FIELD_DEF(bool, fwdHStoreFinalState);
TILING_DATA_FIELD_DEF(int64_t, fwdHIsVariedLen);
TILING_DATA_FIELD_DEF(int64_t, fwdHShapeBatch);
TILING_DATA_FIELD_DEF(int64_t, fwdHTokenBatch);
TILING_DATA_FIELD_DEF(int64_t, fwdHVWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, fwdHVUpdateWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, fwdHKDecayWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, fwdHHWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, fwdHNumSeqWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, fwdHNumChunksWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, fwdHWorkspaceBaseOffset);
TILING_DATA_FIELD_DEF(int64_t, outputUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, outputQgScaledOffset);
TILING_DATA_FIELD_DEF(int64_t, outputScratchOffset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaFwd, ChunkKdaFwdTilingData)

struct ChunkKdaFwdCompileInfo {};
} // namespace optiling
