/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_TILING_H
#define CHUNK_KDA_BWD_RECOMPUTE_TILING_H

#include "register/op_impl_registry.h"

namespace optiling {
struct ChunkKdaBwdRecomputeCompileInfo {
    uint32_t aicNum = 28;
    uint32_t aivNum = 56;
};
ge::graphStatus Tiling4ChunkKdaBwdRecompute(gert::TilingContext *context);
ge::graphStatus TilingRecomputeForChunkKdaBwdRecompute(gert::TilingParseContext *context);
} // namespace optiling

#endif // CHUNK_KDA_BWD_RECOMPUTE_TILING_H
