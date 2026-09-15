/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_TILING_H
#define CHUNK_KDA_BWD_PREPARE_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

#include "chunk_kda_bwd_prepare_tiling_processor.h"

namespace optiling {

struct ChunkKdaBwdPrepareCompileInfo {};

constexpr size_t KDA_PREPARE_INPUT_AQK = 0;
constexpr size_t KDA_PREPARE_INPUT_V_NEW = 1;
constexpr size_t KDA_PREPARE_INPUT_D_O = 2;
constexpr size_t KDA_PREPARE_INPUT_H = 3;
constexpr size_t KDA_PREPARE_INPUT_CU_SEQLENS = 4;
constexpr size_t KDA_PREPARE_INPUT_CHUNK_INDICES = 5;
constexpr size_t KDA_PREPARE_ATTR_SCALE = 0;
constexpr size_t KDA_PREPARE_ATTR_CHUNK_SIZE = 1;
constexpr size_t KDA_PREPARE_ATTR_STATE_V_FIRST = 2;

} // namespace optiling

#endif // CHUNK_KDA_BWD_PREPARE_TILING_H
