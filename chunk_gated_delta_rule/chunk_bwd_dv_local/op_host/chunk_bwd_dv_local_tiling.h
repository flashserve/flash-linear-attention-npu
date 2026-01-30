/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_bwd_dv_local_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkBwdDvLocalTilingData)
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, t);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, v);
TILING_DATA_FIELD_DEF(int64_t, chunkNumForT);
TILING_DATA_FIELD_DEF(int64_t, chunkNumPreCore);
TILING_DATA_FIELD_DEF(int64_t, chunkNumTailCore);
TILING_DATA_FIELD_DEF(int64_t, preCoreNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(bool, isVariable);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkBwdDvLocal, ChunkBwdDvLocalTilingData)

struct ChunkBwdDvLocalCompileInfo {};
} // namespace optiling
