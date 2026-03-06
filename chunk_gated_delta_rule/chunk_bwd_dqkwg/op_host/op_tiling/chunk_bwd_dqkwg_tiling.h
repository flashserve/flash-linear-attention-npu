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
 * \file chunk_bwd_dqkwg_tiling.h
 * \brief ChunkBwdDqkwg Tiling 数据结构定义
 */

#ifndef CHUNK_BWD_DQKWG_TILING_H
#define CHUNK_BWD_DQKWG_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkBwdDqkwgTilingData)
    // 基本形状参数
    TILING_DATA_FIELD_DEF(uint64_t, B);              // batch size
    TILING_DATA_FIELD_DEF(uint64_t, H);              // number of heads
    TILING_DATA_FIELD_DEF(uint64_t, T);              // sequence length
    TILING_DATA_FIELD_DEF(uint64_t, K);              // key/query dimension
    TILING_DATA_FIELD_DEF(uint64_t, V);              // value dimension
    TILING_DATA_FIELD_DEF(uint64_t, BT);             // chunk size (tile size in T dimension)
    TILING_DATA_FIELD_DEF(uint64_t, numChunks);      // T / BT
    
    // scale 参数
    TILING_DATA_FIELD_DEF(float, scale);             // 1.0 / sqrt(K)
    
    // Workspace 偏移量 (按字节)
    TILING_DATA_FIELD_DEF(uint64_t, wsDwOffset);         // Part 1: b_dw 偏移
    TILING_DATA_FIELD_DEF(uint64_t, wsDgLastOffset);     // Part 1: b_dg_last 偏移
    TILING_DATA_FIELD_DEF(uint64_t, dgLastSize);     // Part 1: b_dg_last 偏移
    TILING_DATA_FIELD_DEF(uint64_t, wsMm5Offset);        // Part 2: mm5 (q @ k^T) 偏移
    TILING_DATA_FIELD_DEF(uint64_t, wsDsTempOffset);     // Part 3: b_ds_temp 偏移
    TILING_DATA_FIELD_DEF(uint64_t, wsMm6Offset);        // Part 6: mm6 偏移
    TILING_DATA_FIELD_DEF(uint64_t, wsMm7Offset);        // Part 7: mm7 偏移
    TILING_DATA_FIELD_DEF(uint64_t, wsMul1Offset);        // Part 2: mul1 偏移
    
    // 其他偏移
    TILING_DATA_FIELD_DEF(uint64_t, totalWorkspaceSize); // 总 workspace 大小
    
    // IS_VARLEN 相关
    TILING_DATA_FIELD_DEF(uint64_t, isVarLen);           // 是否变长序列
    
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkBwdDqkwg, ChunkBwdDqkwgTilingData)


}  // namespace optiling

#endif  // CHUNK_BWD_DQKWG_TILING_H
