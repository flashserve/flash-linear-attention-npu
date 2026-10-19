/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_h_tiling.h
 * \brief
 */

#ifndef CHUNK_FWD_H_TILING_H
#define CHUNK_FWD_H_TILING_H

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkFwdHTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(bool, useInitialState);
TILING_DATA_FIELD_DEF(bool, storeFinalState);
TILING_DATA_FIELD_DEF(int64_t, isVariedLen);
TILING_DATA_FIELD_DEF(int64_t, shapeBatch);
TILING_DATA_FIELD_DEF(int64_t, tokenBatch);
TILING_DATA_FIELD_DEF(int64_t, vWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, vUpdateWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, kDecayWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, coreBoundCount);
// 容量与 CHUNK_FWD_H_MAX_CORE_BOUNDS 同步（宏体需整型字面量位置，保持互指注释）
TILING_DATA_FIELD_DEF_ARR(int64_t, 32, coreTaskBounds);
// ===== gated 复用路径（GATED_REUSE）追加段 =====
// gated kernel（chunk_gated_delta_rule_fwd_h preload）运行时分支复用：kernel
// 入口在 workspace 现场构造 ChunkGatedDeltaRuleFwdHTilingData 后原样调用。
// gateDataType：gated tiling 的 gDataType 字段值（1=bf16, 2=fp32，与上游
// aclnn 约定一致）；gatedStateDataType：stateDataType（fp32 路径=2）。
// 三个 offset 为 workspace 内的段偏移（8B 对齐）。
TILING_DATA_FIELD_DEF(int64_t, gatedGateDataType);
TILING_DATA_FIELD_DEF(int64_t, gatedStateDataType);
TILING_DATA_FIELD_DEF(int64_t, gatedTilingOffset);
TILING_DATA_FIELD_DEF(int64_t, gatedNumSeqOffset);
TILING_DATA_FIELD_DEF(int64_t, gatedNumChunksOffset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkFwdH, ChunkFwdHTilingData)

struct ChunkFwdHCompileInfo {};
} // namespace optiling

#endif // CHUNK_FWD_H_TILING_H
