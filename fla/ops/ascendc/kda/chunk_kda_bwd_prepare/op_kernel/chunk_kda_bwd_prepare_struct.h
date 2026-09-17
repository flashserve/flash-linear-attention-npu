/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_STRUCT_H
#define CHUNK_KDA_BWD_PREPARE_STRUCT_H

#include <cstdint>

namespace KDA {

struct ChunkKdaBwdPrepareTilingData {
    int64_t B;
    int64_t NV;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t denseChunkNum;
    int64_t totalChunkNum;
    int64_t chunkTaskNum;
    int64_t headWindowNum;
    int64_t workTaskNum;
    int64_t seqNum;
    int64_t chunkSize;
    uint32_t stateVFirst;
    uint32_t isVariable;
    float scale;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_PREPARE_STRUCT_H
