/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_kda_fwd_h_struct.h
 * \brief Private tiling view used by the ChunkKdaFwd FwdH stage.
 *
 * The field order and types mirror KdaForward::FwdHTilingView. ChunkKdaFwd
 * constructs that view in its private dispatcher and passes it to this stage without
 * depending on another operator's generated tiling definition.
 */

#ifndef CHUNK_KDA_FWD_H_STRUCT_H
#define CHUNK_KDA_FWD_H_STRUCT_H

#include <cstdint>

struct ChunkKdaFwdHTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    int64_t dataType;
    int64_t gDataType;
    int64_t stateDataType;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    bool useG;
    bool useGk;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t numSeqWorkspaceOffset;
    int64_t numChunksWorkspaceOffset;
};

#endif // CHUNK_KDA_FWD_H_STRUCT_H
