/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
    TILING_DATA_FIELD_DEF(uint64_t, HV);             // value 侧 head 数 (v/g/h/do/dh/dv/dw/dg)
    TILING_DATA_FIELD_DEF(uint64_t, HK);             // key/query 侧 head 数 (q/k), HV = n_ratio * HK
    TILING_DATA_FIELD_DEF(uint64_t, T);              // sequence length
    TILING_DATA_FIELD_DEF(uint64_t, K);              // key/query dimension
    TILING_DATA_FIELD_DEF(uint64_t, V);              // value dimension
    TILING_DATA_FIELD_DEF(uint64_t, BT);             // chunk size (tile size in T dimension)
    TILING_DATA_FIELD_DEF(uint64_t, numChunks);      // T / BT

    // scale 参数
    TILING_DATA_FIELD_DEF(float, scale);             // 1.0 / sqrt(K)
    TILING_DATA_FIELD_DEF(uint32_t, mul0RowNum);
    TILING_DATA_FIELD_DEF(uint32_t, aicCoreNum);     // CV 深融合使用的 AIC blockDim (cube/vector 共用)

    // Workspace 偏移量 (按字节)
    TILING_DATA_FIELD_DEF(uint64_t, wsDwOffset);         // PartA: b_dw / 之后 PartC mm6 复用
    TILING_DATA_FIELD_DEF(uint64_t, wsBtxKSyncSlotsPerHead); // cross-stage group ring depth per core
    TILING_DATA_FIELD_DEF(uint64_t, wsDgLastOffset);     // PartA: b_dg_last 偏移
    TILING_DATA_FIELD_DEF(uint64_t, dgLastSize);         // PartA: b_dg_last 大小, 32B 对齐
    TILING_DATA_FIELD_DEF(uint64_t, wsMm5Offset);        // PartA: mm5 / GVA C: dq_inner / PartD: mm7
    TILING_DATA_FIELD_DEF(uint64_t, wsDsTempOffset);     // PartB: b_ds_temp 偏移
    TILING_DATA_FIELD_DEF(uint64_t, wsMm6Offset);        // PartC: mm6 / GVA D: dk_inner
    TILING_DATA_FIELD_DEF(uint64_t, wsMm7Offset);        // PartD: mm7 复用已释放的 wsMm5
    TILING_DATA_FIELD_DEF(uint64_t, wsMul1Offset);       // independent short BT x BT ring for mul1

    // 其他偏移
    TILING_DATA_FIELD_DEF(uint64_t, totalWorkspaceSize); // 总 workspace 大小

    // IS_VARLEN 相关
    TILING_DATA_FIELD_DEF(uint64_t, isVarLen);           // 是否变长序列

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkBwdDqkwg, ChunkBwdDqkwgTilingData)


}  // namespace optiling

#endif  // CHUNK_BWD_DQKWG_TILING_H
