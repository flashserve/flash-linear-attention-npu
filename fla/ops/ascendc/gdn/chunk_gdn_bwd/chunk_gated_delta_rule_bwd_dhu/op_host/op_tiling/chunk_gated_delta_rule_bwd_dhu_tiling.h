/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_tiling.h
 * \brief Compatibility include for chunk_gated_delta_rule_bwd_dhu tiling data.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

#include "chunk_gated_delta_rule_bwd_dhu_tiling_processor.h"

namespace optiling {

struct ChunkGatedDeltaRuleBwdDhuCompileInfo {};

static constexpr size_t CGDR_BWD_DHU_INPUT_Q_IDX = 0;
static constexpr size_t CGDR_BWD_DHU_INPUT_K_IDX = 1;
static constexpr size_t CGDR_BWD_DHU_INPUT_W_IDX = 2;
static constexpr size_t CGDR_BWD_DHU_INPUT_DO_IDX = 3;
static constexpr size_t CGDR_BWD_DHU_INPUT_DV_IDX = 4;
static constexpr size_t CGDR_BWD_DHU_INPUT_G_IDX = 5;
static constexpr size_t CGDR_BWD_DHU_INPUT_GK_IDX = 6;
static constexpr size_t CGDR_BWD_DHU_INPUT_H0_IDX = 7;
static constexpr size_t CGDR_BWD_DHU_INPUT_CU_SEQLENS_IDX = 9;
static constexpr size_t CGDR_BWD_DHU_INPUT_CHUNK_INDICES_IDX = 10;
static constexpr size_t CGDR_BWD_DHU_ATTR_SCALE_IDX = 0;
static constexpr size_t CGDR_BWD_DHU_ATTR_CHUNK_SIZE_IDX = 1;
static constexpr size_t CGDR_BWD_DHU_ATTR_USE_EXP2_IDX = 2;

} // namespace optiling

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_H
