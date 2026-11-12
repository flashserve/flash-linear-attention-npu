/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda_struct.h
 * \brief Plain tiling struct shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECURRENT_KDA_STRUCT_H
#define RECURRENT_KDA_STRUCT_H

#include <cstdint>

namespace RecurrentKda {

constexpr uint32_t RKDA_A5_WINDOW_TOKENS = 8;

// A6（DAV_9201）静态 UB 引擎的布局常量。kernel 与 host tiling 共用这一份定义，
// 避免两侧各自维护同一组数字后漂移。
constexpr int64_t RKDA_A6_DIM = 128;
constexpr int64_t RKDA_A6_TOKEN_TILE = 32;
constexpr int64_t RKDA_A6_INPUT_BUFFERS = 2;
constexpr int64_t RKDA_A6_PARAMETER_BUFFERS = 2;
// State bank 固定放一片 [DIM, DIM]：FP32 64 KiB、BF16 32 KiB。A6 的 384 KiB 向量 UB
// 允许两种 dtype 都留 4 份（FP32 共 364 544 B），把跨任务/跨 V 片的初态预取深度拉满。
constexpr int64_t RKDA_A6_FP32_STATE_BANKS = 4;
constexpr int64_t RKDA_A6_BF16_STATE_BANKS = 4;

// A6 静态 UB 用量：State bank + 输入/输出双槽 + dt_bias 双槽 + VF scratch。
constexpr int64_t RkdaA6UbBytes(int64_t stateBanks, int64_t tokenTile, int64_t stateBytes)
{
    const int64_t bankBytes = RKDA_A6_DIM * RKDA_A6_DIM * stateBytes;
    const int64_t slotBytes = 4 * tokenTile * RKDA_A6_DIM * 2 + // q/k/v/out bf16
                              tokenTile * RKDA_A6_DIM * 4 +     // gate fp32
                              tokenTile * 32;                   // beta 与标量
    return stateBanks * bankBytes + RKDA_A6_INPUT_BUFFERS * slotBytes +
           RKDA_A6_PARAMETER_BUFFERS * RKDA_A6_DIM * 4 +
           RKDA_A6_DIM * 4 +  // prediction scratch
           RKDA_A6_DIM * 4;   // V-first out 的 FP32 staging
}


// A5 static VF configuration, built once by host tiling. Runtime bank/slot
// ownership remains in the kernel and is never encoded by the host.
enum RecurrentKdaVfFlags : uint32_t {
    RKDA_VF_NORM = 1U << 0,
    RKDA_VF_RAW_GATE = 1U << 1,
    RKDA_VF_BIAS = 1U << 2,
    RKDA_VF_SIGMOID = 1U << 3,
    RKDA_VF_NEGATIVE = 1U << 4,
    RKDA_VF_SAFE = 1U << 5,
    RKDA_VF_WRITE_STATE = 1U << 6,
    RKDA_VF_INDEXED = 1U << 7,
    RKDA_VF_FUSED = 1U << 8,
};

#pragma pack(push, 8)
struct alignas(8) RecurrentKdaTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t t;
    uint32_t seqLen;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t ssmStateStride;
    uint32_t b;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
    float lowerBound;
    uint32_t layout;
    uint32_t hasSsmStateIndices;
    uint32_t hasALog;
    uint32_t hasDtBias;
    uint32_t hasAcceptedTokens;
    uint32_t useQkL2norm;
    uint32_t useGateInKernel;
    uint32_t useBetaSigmoid;
    uint32_t allowNegEigval;
    uint32_t safeGate;
    uint32_t stateVFirst;
    uint32_t outputFinalState;
    uint32_t inplaceFinalState;
    uint32_t hasCuSeqlens;
    uint32_t gateDtype;
    uint32_t betaDtype;
    uint32_t cuSeqlensDtype;
    uint32_t ssmStateIndicesDtype;
    uint32_t acceptedTokensDtype;
    uint64_t stateInStride0;
    uint64_t stateInStride1;
    uint64_t stateInStride2;
    uint64_t stateInStride3;
    uint64_t stateOutStride0;
    uint64_t stateOutStride1;
    uint64_t stateOutStride2;
    uint64_t stateOutStride3;
    uint64_t queryTokenStride;
    uint64_t queryHeadStride;
    uint64_t keyTokenStride;
    uint64_t keyHeadStride;
    uint64_t valueTokenStride;
    uint64_t valueHeadStride;
    uint32_t windowSize; // A5 token capacity selected by host tiling; A2/A3 ignore it.
    uint32_t a5VfFlags; // A2/A3 keep their existing flags and resource paths.
    uint32_t a5HeadRatio;
};
#pragma pack(pop)

} // namespace RecurrentKda

#endif // RECURRENT_KDA_STRUCT_H
