/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_STRUCT_H
#define CHUNK_GDN_BWD_INTRA_STRUCT_H

#include <cstdint>

namespace GDN {

struct ChunkGdnBwdIntraTilingData {
    int64_t batch;
    int64_t qkHeads;
    int64_t valueHeads;
    int64_t seqlen;
    int64_t keyDim;
    int64_t valueDim;
    int64_t chunkSize;
    int64_t chunksPerBatch;
    int64_t chunkCount;
    int64_t headRatio;
    int64_t cg;
    int64_t hvSliceCount;
    int64_t workCount;
    int64_t blockDim;
    int64_t isVarlen;
    int64_t useExp2;
    int64_t stage;
    float scale;
    uint32_t reserved;
    uint64_t matrixStrideBytes;
    uint64_t aBgWorkspaceOffset;
    uint64_t aBetaWorkspaceOffset;
    uint64_t dWorkspaceOffset;
    uint64_t userWorkspaceBytes;
};

struct ChunkGdnBwdIntraWorkMeta {
    int64_t batch;
    int64_t tokenStart;
    int64_t validTokens;
    int64_t hvBegin;
    int64_t validHeads;
};

} // namespace GDN

#endif // CHUNK_GDN_BWD_INTRA_STRUCT_H
