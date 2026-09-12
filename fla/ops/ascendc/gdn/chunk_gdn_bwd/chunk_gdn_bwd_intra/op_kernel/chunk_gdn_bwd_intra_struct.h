/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_STRUCT_H
#define CHUNK_GDN_BWD_INTRA_STRUCT_H

#include <cstdint>

namespace GDN {

struct ChunkGdnBwdIntraTilingData {
    // 输入逻辑 shape。
    int64_t batch;
    int64_t qkHeads;
    int64_t valueHeads;
    int64_t seqlen;
    int64_t keyDim;
    int64_t valueDim;
    int64_t chunkSize;

    // chunk 与 HV 切片共同决定全局 work 数量和分核方式。
    int64_t chunksPerBatch;
    int64_t chunkCount;
    int64_t cg;
    int64_t hvSliceCount;
    int64_t workCount;
    int64_t blockDim;

    // 运行时路径和算子属性。
    int64_t isVarlen;
    int64_t useExp2;
    float scale;
    uint32_t reserved;
};

struct ChunkGdnBwdIntraWorkMeta {
    // 当前 work 对应的 batch、token 区间和 HV 切片。
    int64_t batch;
    int64_t tokenStart;
    int64_t validTokens;
    int64_t hvBegin;
    int64_t validHeads;
};

} // namespace GDN

#endif // CHUNK_GDN_BWD_INTRA_STRUCT_H
