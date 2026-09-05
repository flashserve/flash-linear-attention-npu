/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_o_struct.h
 * \brief Shared tiling data for chunk_fwd_o.
 */

#ifndef CHUNK_FWD_O_STRUCT_H
#define CHUNK_FWD_O_STRUCT_H

#include <cstdint>

namespace GDN {

static constexpr uint64_t CHUNK_FWD_O_TILING_KEY_LEGACY = 1;
static constexpr uint64_t CHUNK_FWD_O_TILING_KEY_A5 = 2;

enum ChunkFwdOOutputLayout : int64_t {
    CHUNK_FWD_O_LAYOUT_BNSD = 0,
    CHUNK_FWD_O_LAYOUT_BSND = 1,
    CHUNK_FWD_O_LAYOUT_TND = 2,
    CHUNK_FWD_O_LAYOUT_NTD = 3,
};

struct ChunkFwdOTilingData {
    int64_t shapeBatch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t isVariedLen;
    int64_t tokenBatch;
    int64_t dataType;
    int64_t gDataType;
    int64_t vWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t attnWorkspaceOffset;
    int64_t aftermaskWorkspaceOffset;
    int64_t maskWorkspaceOffset;
    int64_t useExp2;
    int64_t outputLayout;
    float scale;
    int64_t chunkNum;
    int64_t hvPerHk;
    int64_t taskGroupSize;
    int64_t numChunksPerBatch;
    int64_t aPrimeWorkspaceOffset;
};

} // namespace GDN

#endif // CHUNK_FWD_O_STRUCT_H
