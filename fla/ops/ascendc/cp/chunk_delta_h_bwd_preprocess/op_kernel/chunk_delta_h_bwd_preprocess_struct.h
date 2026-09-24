/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess_struct.h
 * \brief Tiling data struct and tiling key declarations for chunk_delta_h_bwd_preprocess.
 */

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_STRUCT_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_STRUCT_H

#include <cstdint>

namespace CP {

// tilingKey：gate 模式 + g 的 dtype
constexpr uint32_t CHUNK_DELTA_H_BWD_PREPROCESS_TILING_KEY_NONE = 1;      // 无门控（g / gk 均未传）
constexpr uint32_t CHUNK_DELTA_H_BWD_PREPROCESS_TILING_KEY_G = 2;         // USE_G，g 与 q 同 dtype
constexpr uint32_t CHUNK_DELTA_H_BWD_PREPROCESS_TILING_KEY_G_FP32 = 3;    // USE_G，g 为 FP32
constexpr uint32_t CHUNK_DELTA_H_BWD_PREPROCESS_TILING_KEY_GK = 4;        // USE_GK，gk 与 q 同 dtype

// 分核模式
constexpr uint64_t CHUNK_DELTA_H_BWD_PREPROCESS_SPLIT_BY_HEAD = 0;  // 默认：仅按 head 连续分核
constexpr uint64_t CHUNK_DELTA_H_BWD_PREPROCESS_SPLIT_BY_TILE = 1;  // Hv < 核数：按 (hv, 列 tile) 展平

constexpr uint32_t CHUNK_DELTA_H_BWD_PREPROCESS_MAX_K = 256;
constexpr uint32_t CHUNK_DELTA_H_BWD_PREPROCESS_K_GROUP_ROWS = 64;
constexpr uint32_t CHUNK_DELTA_H_BWD_PREPROCESS_CHUNK_SIZE = 64;

struct ChunkDeltaHBwdPreprocessTilingData {
    // shape
    uint64_t B;
    uint64_t T;
    uint64_t Hk;
    uint64_t Hv;
    uint64_t K;
    uint64_t V;
    // chunk 与 tile
    uint64_t chunkSize;
    uint64_t chunkNum;
    uint64_t seqNum;
    uint64_t blockSize;   // 列 tile 宽度：K <= 64 时为 32，否则 64
    uint64_t tileV;
    uint64_t tileK;
    uint64_t tileNum;     // tileV + tileK
    uint64_t kGroupNum;   // ceil(K / 64)，最多 4
    // segment 与模式
    uint64_t bos;
    uint64_t eos;
    uint64_t isVarLen;
    uint64_t isScale;
    uint64_t useGateG;
    uint64_t useGateGk;
    // 分核
    uint64_t usedCoreNum;
    uint64_t blockDim;
    uint64_t splitMode;
    uint64_t groupHeads;  // splitMode == BY_HEAD 时每核连续 head 数
    // 用户 workspace 规划（相对 user workspace 起始的字节偏移）
    uint64_t slotNum;
    uint64_t slotBytes;
    uint64_t slotWsOffset;
    uint64_t dhWsOffset;
    uint64_t dhBfWsOffset;
    uint64_t dvPreWsOffset;
    uint64_t dvHatWsOffset;
    uint64_t qtermWsOffset;
    uint64_t wtermWsOffset;
    uint64_t t1WsOffset;
    uint64_t pcWsOffset;
    uint64_t pWsOffset;
    uint64_t pBfWsOffset;
    uint64_t metaWsOffset;
    uint64_t dhWsBytes;
    uint64_t dhBfWsBytes;
    uint64_t dvPreWsBytes;
    uint64_t dvHatWsBytes;
    uint64_t qtermWsBytes;
    uint64_t wtermWsBytes;
    uint64_t t1WsBytes;
    uint64_t pcWsBytes;
    uint64_t pWsBytes;
    uint64_t pBfWsBytes;
    uint64_t metaWsBytes;
    uint64_t totalWsBytes;
    float scale;
};

} // namespace CP

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_STRUCT_H
