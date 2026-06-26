/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_struct.h
 * \brief Tiling data struct and tiling key declarations for chunk_gated_delta_rule_bwd_dhu operator.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_STRUCT_H

#include <cstdint>

namespace GDN {

constexpr uint32_t CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_KEY = 1;
constexpr uint32_t CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_KEY_G_FP32 = 2;

struct ChunkGatedDeltaRuleBwdDhuTilingData {
    uint64_t B;
    uint64_t Hv;
    uint64_t Hk;
    uint64_t T;
    uint64_t K;
    uint64_t V;
    uint64_t chunkSize;
    uint64_t chunkNum;
    uint64_t seqNum;
    uint64_t gBufSize;
    uint64_t dvBufSize;
    uint64_t qBufSize;
    uint64_t dhBufSize;
    uint64_t totalTbufByte;
    uint64_t bdvWs;
    uint64_t qWs;
    uint64_t wDv2Ws;
    uint64_t qDoWs;
    uint64_t isVarLen;
    uint64_t isScale;
    uint32_t usedCoreNum;
    float scale;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_STRUCT_H
