/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_INTRA_STRUCT_H
#define CHUNK_KDA_BWD_INTRA_STRUCT_H

#include <cstdint>

namespace KDA {

struct ChunkKdaBwdIntraTilingData {
    int64_t batch;
    int64_t headNum;
    int64_t seqlen;
    int64_t headDim;
    int64_t chunkSize;
    int64_t chunkNum;
    int64_t chunkNumPerBatch;

    int64_t workspaceSlotSize;
    int64_t workspaceCoreSize;
    int64_t resultRegionOffset;

    int64_t aLowerOffset;
    int64_t bLowerOffset;
    int64_t aUpperOffset;
    int64_t bUpperOffset;
    int64_t resultDqOffset;
    int64_t resultDkLowerOffset;
    int64_t resultDkUpperOffset;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_INTRA_STRUCT_H
