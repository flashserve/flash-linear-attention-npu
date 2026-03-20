/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_bwd_dqkwg_common.h
 * \brief ChunkBwdDqkwg 通用常量和定义
 */

#ifndef CHUNK_BWD_DQKWG_COMMON_H
#define CHUNK_BWD_DQKWG_COMMON_H

#include "kernel_operator.h"

constexpr uint64_t CONST_B = 1;
constexpr uint64_t CONST_H = 4;
constexpr uint64_t CONST_T = 2816;
constexpr uint64_t CONST_K = 128;
constexpr uint64_t CONST_V = 128;
constexpr uint64_t CONST_BT = 64;
constexpr uint64_t CONST_NUM_CHUNKS = 44;//CONST_T / CONST_BT;  // 32

// Part 1: dw 和 dg_last 计算 (C-V 融合)
constexpr uint64_t SYNC_PART1_AIC_AIV = 10;  // AIC -> AIV
constexpr uint64_t SYNC_PART1_AIV_AIC = 11;  // AIV -> AIC

// Part 2: mm5 = q @ k^T (纯 Cube)
constexpr uint64_t SYNC_PART2_AIC_AIV = 20;  // AIC -> AIV
constexpr uint64_t SYNC_PART2_AIV_AIC = 21;  // AIV -> AIC

// Part 3: ds 计算 (C-V 融合)
constexpr uint64_t SYNC_PART3_AIC_AIV = 30;  // AIC -> AIV
constexpr uint64_t SYNC_PART3_AIV_AIC = 31;  // AIV -> AIC

// Part 4: dq 计算 (C-V 融合)
constexpr uint64_t SYNC_PART4_AIC_AIV = 40;  // AIC -> AIV
constexpr uint64_t SYNC_PART4_AIV_AIC = 41;  // AIV -> AIC

// Part 5: dk 计算 (C-V 融合)
constexpr uint64_t SYNC_PART5_AIC_AIV = 50;  // AIC -> AIV
constexpr uint64_t SYNC_PART5_AIV_AIC = 51;  // AIV -> AIC

// Part 6: dq += ds @ k (C-V 融合)
constexpr uint64_t SYNC_PART6_AIC_AIV = 60;  // AIC -> AIV
constexpr uint64_t SYNC_PART6_AIV_AIC = 61;  // AIV -> AIC

// Part 7: dk += ds^T @ q (C-V 融合)
constexpr uint64_t SYNC_PART7_AIC_AIV = 70;  // AIC -> AIV
constexpr uint64_t SYNC_PART7_AIV_AIC = 71;  // AIV -> AIC

// 通用同步 Flag (用于简化)
constexpr uint64_t SYNC_AIV_AIC_FLAG_0 = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG_0 = 5;

constexpr uint32_t UB_SIZE = 192 * 1024;  // 192KB
constexpr uint32_t ONE_BLOCK_32 = 32;
constexpr uint32_t FP32_PER_REPEAT = 64;
constexpr uint32_t FP16_PER_BLOCK = 16;  // 32 bytes / 2 bytes per fp16

constexpr uint32_t FP16_SIZE = 2;
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BF16_SIZE = 2;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ALIGN_UP(x, align) (((x) + (align) - 1) / (align) * (align))

template<typename T>
struct TypeTraits {
    using ComputeType = float;  // 默认计算类型为 fp32
    static constexpr bool needsCast = true;
};

template<>
struct TypeTraits<half> {
    using ComputeType = float;
    static constexpr bool needsCast = true;
};

__aicore__ void inline GetChunkOffset(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, uint64_t B, uint64_t H, uint64_t T,
                                      uint64_t chunkSize, uint32_t loopIdx, uint32_t &bos, uint32_t &eos)
{
    if (cu_seqlens == nullptr) {
        // AscendC::printf("111\n");
        uint32_t coreLoopsInB = CEIL_DIV(T, chunkSize);
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        uint32_t bIdx = loopIdx / coreLoopsInB;
        bos = chunkIdx * chunkSize;
        eos = bos + chunkSize > T ? T : bos + chunkSize;
        bos += (bIdx * H * T);
        eos += (bIdx * H * T);
    } else {
        AscendC::GlobalTensor<uint64_t> cuSeqlensTensor;
        AscendC::GlobalTensor<uint64_t> chunkIndicesTensor;
        cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)cu_seqlens);
        chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)chunk_indices);

        uint32_t seqIdx = chunkIndicesTensor.GetValue(2 * loopIdx);
        uint32_t chunkIdx = chunkIndicesTensor.GetValue(2 * loopIdx + 1);
        uint32_t curSeqBegin = cuSeqlensTensor.GetValue(seqIdx);
        uint32_t curSeqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
        bos = curSeqBegin + chunkIdx * chunkSize;
        eos = bos + chunkSize > curSeqEnd ? curSeqEnd : bos + chunkSize;
    }

    return;
}

#endif  // CHUNK_BWD_DQKWG_COMMON_H
