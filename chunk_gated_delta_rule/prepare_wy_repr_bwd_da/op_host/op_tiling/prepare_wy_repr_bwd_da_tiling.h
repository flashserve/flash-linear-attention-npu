/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prepare_wy_repr_bwd_da_tiling.h
 * \brief
 */
#ifndef PREPARE_WY_REPR_BWD_DA_TILING_H
#define PREPARE_WY_REPR_BWD_DA_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PrepareWyReprBwdDaTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, H);
TILING_DATA_FIELD_DEF(int64_t, T);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, V);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, chunkNum);
TILING_DATA_FIELD_DEF(int64_t, rowNumKBetaG);
TILING_DATA_FIELD_DEF(int64_t, rowNumVBeta);
TILING_DATA_FIELD_DEF(int64_t, rowNumMDuDw);
TILING_DATA_FIELD_DEF(int64_t, rowNumG);
TILING_DATA_FIELD_DEF(int64_t, isVariable);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PrepareWyReprBwdDa, PrepareWyReprBwdDaTilingData)

}  // namespace optiling

#endif  // PREPARE_WY_REPR_BWD_DA_TILING_H