/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd_tiling.h
 * \brief
 */
#ifndef RECOMPUTE_WU_FWD_TILING_H
#define RECOMPUTE_WU_FWD_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RecomputeWUFwdTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, H);
TILING_DATA_FIELD_DEF(int64_t, T);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, V);
TILING_DATA_FIELD_DEF(int64_t, chunkNum);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, vbVecRow); // kBeta计算流程时vector单次处理的行数
TILING_DATA_FIELD_DEF(int64_t, kbgExpVecRow);   // dk计算流程时vector单次处理的行数
TILING_DATA_FIELD_DEF(int64_t, isVariable);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RecomputeWUFwd, RecomputeWUFwdTilingData)

} // namespace optiling

#endif // RECOMPUTE_WU_FWD_TILING_H