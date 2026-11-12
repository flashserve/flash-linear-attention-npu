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
    // 追加字段（保持前序字段偏移不变）：1 表示 arch35 直接路径按 i_d 切分核间任务
    // （每个核独占若干 i_d 的全部 chunk，无需 partial 归约）；0 表示沿用按 chunk 切分 + partial 归约。
    int64_t useIdSplit;
    // 追加字段（保持前序字段偏移不变）：两级归约的中间行数 R。
    // R >= 1 表示 SILU/SWISH 分支把 dw/db 的行归约拆成两级：chunk 内把 BT 行只归约到 R 行
    // 并累积到 stage buffer，i_d 末再把 R 行归约到 1 行。0 表示不启用（沿用逐 chunk 全量归约）。
    int64_t dwStageRows;
    // 追加字段（保持前序字段偏移不变）：chunk × i_d 二维切分。1 表示把 numBlksD 个 i_d 与
    // numChunks 个 chunk 一起铺到核上，每个核只处理 1 个 i_d 的 1 个 chunk 组（blockNum =
    // numBlksD * numChunks），消除"每核串行遍历全部 i_d"的固定开销；0 表示沿用按 chunk 切分。
    // 仅 regbase 平台（A5/A6）且全部 (i_d, chunk) 工作单元数不超过 AIV 核数时置 1。
    int64_t use2DSplit;
    int64_t inputStages;
};

#endif // CAUSAL_CONV1D_BWD_TILING_DATA_H_
