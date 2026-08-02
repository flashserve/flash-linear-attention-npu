/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details.
 */

#ifndef CHUNK_KDA_FWD_INTRA_SUB_CHUNK_TILING_H
#define CHUNK_KDA_FWD_INTRA_SUB_CHUNK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkKdaFwdIntraSubChunkTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, t);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, hv);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, subChunkSize);
TILING_DATA_FIELD_DEF(int64_t, numChunks);
TILING_DATA_FIELD_DEF(int64_t, numSubChunks);
// Cube: USE_HEAD_WINDOW → flat(B, NT), HV in-core; else flat(B, HV, NT).
TILING_DATA_FIELD_DEF(int64_t, totalTasks);
TILING_DATA_FIELD_DEF(int64_t, hasCuSeqlens);
TILING_DATA_FIELD_DEF(int64_t, hasChunkIndices);
TILING_DATA_FIELD_DEF(int64_t, seqNum);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
// Aligned byte size of score scratch region (before cmat). Kernel: cWs = userWS + scoreScratchBytes.
TILING_DATA_FIELD_DEF(int64_t, scoreScratchBytes);
TILING_DATA_FIELD_DEF(int64_t, scoreQueueDepth);
TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaFwdIntraSubChunk, ChunkKdaFwdIntraSubChunkTilingData)

struct ChunkKdaFwdIntraSubChunkCompileInfo {};
} // namespace optiling

#endif
