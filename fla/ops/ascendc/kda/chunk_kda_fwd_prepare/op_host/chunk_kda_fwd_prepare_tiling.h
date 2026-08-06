/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkKdaFwdPrepareTilingData)
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
TILING_DATA_FIELD_DEF(bool, isVarLen);
TILING_DATA_FIELD_DEF(bool, safeGate);
TILING_DATA_FIELD_DEF(int64_t, prepareUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, prepareAqkFp32Offset);
TILING_DATA_FIELD_DEF(int64_t, prepareAkkFp32Offset);
TILING_DATA_FIELD_DEF(int64_t, prepareScratchOffset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaFwdPrepare, ChunkKdaFwdPrepareTilingData)

struct ChunkKdaFwdPrepareCompileInfo {};
} // namespace optiling
