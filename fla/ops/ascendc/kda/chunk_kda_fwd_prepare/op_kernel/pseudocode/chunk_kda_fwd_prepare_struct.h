/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_STRUCT_H
#define PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_STRUCT_H

#include "kernel_operator.h"

#include "chunk_kda_fwd_prepare_tiling_key.h"

namespace KdaPrepare {

// 一个实际 chunk 的直接索引。Stage 只接收这些标量，不再传递通用 StageArgs。
struct ChunkRange {
    // dense: batchIndex=sequence，tokenBegin 为序列内位置；
    // varlen: batchIndex=0，tokenBegin 为压平后的全局 token 位置。
    uint32_t batchIndex = 0;
    uint32_t sequence = 0;
    uint32_t globalChunk = 0;
    uint32_t tokenBegin = 0;
    uint32_t validRows = 0;
};

// 仅保存真实 GM 参数和运行时 tiling；本结构不承担 buffer、公式或同步记录功能。
struct PrepareKernelArgs {
    GM_ADDR q = nullptr;
    GM_ADDR k = nullptr;
    GM_ADDR v = nullptr;
    GM_ADDR rawGate = nullptr;
    GM_ADDR beta = nullptr;
    GM_ADDR dtBias = nullptr;
    GM_ADDR aLog = nullptr;
    GM_ADDR cuSeqlens = nullptr;
    GM_ADDR chunkIndices = nullptr;

    GM_ADDR qg = nullptr;
    GM_ADDR kg = nullptr;
    GM_ADDR gk = nullptr;
    GM_ADDR aqk = nullptr;
    GM_ADDR akk = nullptr;
    GM_ADDR w = nullptr;
    GM_ADDR u = nullptr;
    GM_ADDR workspace = nullptr;

    ChunkKdaFwdPrepareTilingData tiling{};
};

} // namespace KdaPrepare

#endif // PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_STRUCT_H
