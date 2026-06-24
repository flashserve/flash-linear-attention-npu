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
    TILING_DATA_FIELD_DEF(int64_t, batchSize);
    TILING_DATA_FIELD_DEF(int64_t, seqLength);
    TILING_DATA_FIELD_DEF(int64_t, numHead);
    TILING_DATA_FIELD_DEF(int64_t, chunkSize);
    TILING_DATA_FIELD_DEF(int64_t, chunkNumInSeq);
    TILING_DATA_FIELD_DEF(int64_t, chunkNumTotal);
    TILING_DATA_FIELD_DEF(int64_t, mode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SolveTril, SolveTrilTilingData)

}  // namespace optiling

#endif

