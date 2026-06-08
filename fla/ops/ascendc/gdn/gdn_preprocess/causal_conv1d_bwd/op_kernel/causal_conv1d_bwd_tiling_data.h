/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_bwd_tiling_data.h
 * \brief Causal Conv1D backward tiling data structure
 */

#ifndef CAUSAL_CONV1D_BWD_TILING_DATA_H_
#define CAUSAL_CONV1D_BWD_TILING_DATA_H_

#include <cstdint>

struct CausalConv1dBwdTilingData {
    int64_t B;
    int64_t T;
    int64_t D;
    int64_t W;
    int64_t activation;
    int64_t hasWeight;
    int64_t hasBias;
    int64_t useInitialState;
    int64_t useFinalState;
    int64_t inputMode;
    int64_t inputLayout;
    int64_t inputN;
    int64_t inputHeadDim;
    int64_t totalTokens;
    int64_t blockNum;
    int64_t BT;
    int64_t BD;
    int64_t numBlksD;
    int64_t numChunks;
    int64_t batchPerCore;
    int64_t tailBatch;
    int64_t chunkPerCore;
    int64_t tailChunk;
    int64_t workspaceSize;
};

#endif // CAUSAL_CONV1D_BWD_TILING_DATA_H_
