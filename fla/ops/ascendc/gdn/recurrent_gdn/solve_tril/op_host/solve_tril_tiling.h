/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef SOLVE_TRIL_TILING_H
#define SOLVE_TRIL_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SolveTrilTilingData)
    TILING_DATA_FIELD_DEF(int64_t, totalTiles);
    TILING_DATA_FIELD_DEF(int64_t, matrixSize);
    TILING_DATA_FIELD_DEF(int64_t, numHeads);
    TILING_DATA_FIELD_DEF(int64_t, seqLen);
    TILING_DATA_FIELD_DEF(int64_t, batchSize);
    TILING_DATA_FIELD_DEF(int64_t, isLower);
    TILING_DATA_FIELD_DEF(int64_t, hasCuSeqlens);
    TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
    TILING_DATA_FIELD_DEF(int64_t, chunkSize);
    TILING_DATA_FIELD_DEF(int64_t, numChunks);
    TILING_DATA_FIELD_DEF(int64_t, lastChunkValidSize);
    TILING_DATA_FIELD_DEF(int64_t, isVarlen);
    TILING_DATA_FIELD_DEF(int64_t, totalChunks);
    TILING_DATA_FIELD_DEF(int64_t, layoutMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SolveTril, SolveTrilTilingData)

}  // namespace optiling

#endif
