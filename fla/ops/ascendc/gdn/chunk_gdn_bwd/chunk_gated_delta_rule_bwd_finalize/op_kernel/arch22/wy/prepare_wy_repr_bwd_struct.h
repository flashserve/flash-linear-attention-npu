// Finalize A2 private implementation; arithmetic/scheduling source: 07d1b942.
// Kept local so --ops=chunk_gated_delta_rule_bwd_finalize builds independently.
/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_struct.h
 * \brief Shared tiling data for fused prepare_wy_repr_bwd.
 */

#ifndef PREPARE_WY_REPR_BWD_STRUCT_H
#define PREPARE_WY_REPR_BWD_STRUCT_H

#include <cstdint>

namespace GDN {

// QK 批尾仅在 Output/DG 完成并经双方 publish 后借用这些 WY buffer。
// GetWithOffset 的长度单位是元素，以下偏移和容量均以字节表示，起点按 32B 对齐。
//
// buffer       切片组成                         生命周期
// inputPing    qInput | dqInput                 输入搬入至最终输出搬运完成
// calcA        gradQ（含连续的 Q/K 两组行）      DK 合并后开始，norm 与输出 Cast 完成后释放
//              gradK 位于后半区，先暂存 DK 合并，随后由 Q/K grad 视图接管
// calcB        x（Q/K 两组行）                  输入 Cast 至 norm 完成
// calcC        dkInput | dkWyInput | kInput     三个输入 Cast 完成后由 product 整区接管
// inputPong    sum | reduceWork | broadcast    每行两个 partial 归约后广播，rstd 单独预载
//              | qRstdInput | kRstdInput
// 每个 QK panel 末的 PIPE_ALL 完成借用区读写，批尾 release 后恢复 WY 使用。
template <typename InputType, uint32_t KEY_DIM>
struct PrepareWyReprBwdQkTailUbLayout {
    static constexpr uint32_t ROWS_32 = 32;
    static constexpr uint32_t DATA_BLOCK_BYTES_32 = 32;
    static constexpr uint32_t INPUT_BUFFER_BYTES_16K = 16 * 1024;
    static constexpr uint32_t CALC_BUFFER_BYTES_32K = 2 * INPUT_BUFFER_BYTES_16K;
    static constexpr uint32_t TILE_ELEMENTS = ROWS_32 * KEY_DIM;
    static constexpr uint32_t NORM_ROWS = 2 * ROWS_32;
    static constexpr uint32_t NORM_ELEMENTS = 2 * TILE_ELEMENTS;
    static constexpr uint32_t INPUT_TILE_BYTES = TILE_ELEMENTS * sizeof(InputType);
    static constexpr uint32_t FP32_TILE_BYTES = TILE_ELEMENTS * sizeof(float);
    static constexpr uint32_t NORM_TILE_BYTES = NORM_ELEMENTS * sizeof(float);
    static constexpr uint32_t FP32_PER_BLOCK = DATA_BLOCK_BYTES_32 / sizeof(float);
    static constexpr uint32_t REDUCE_ELEMENTS = NORM_ROWS * FP32_PER_BLOCK;
    static constexpr uint32_t SUM_BYTES = NORM_ROWS * sizeof(float);
    static constexpr uint32_t REDUCE_BYTES = REDUCE_ELEMENTS * sizeof(float);
    static constexpr uint32_t RSTD_BYTES = ROWS_32 * sizeof(float);

    static constexpr uint32_t Q_INPUT_OFFSET_BYTES = 0;
    static constexpr uint32_t DQ_INPUT_OFFSET_BYTES = Q_INPUT_OFFSET_BYTES + INPUT_TILE_BYTES;
    static constexpr uint32_t DK_INPUT_OFFSET_BYTES = 0;
    static constexpr uint32_t DK_WY_INPUT_OFFSET_BYTES = DK_INPUT_OFFSET_BYTES + INPUT_TILE_BYTES;
    static constexpr uint32_t K_INPUT_OFFSET_BYTES = DK_WY_INPUT_OFFSET_BYTES + INPUT_TILE_BYTES;
    static constexpr uint32_t GRAD_Q_OFFSET_BYTES = 0;
    static constexpr uint32_t GRAD_K_OFFSET_BYTES = GRAD_Q_OFFSET_BYTES + FP32_TILE_BYTES;
    static constexpr uint32_t X_OFFSET_BYTES = 0;
    static constexpr uint32_t PRODUCT_OFFSET_BYTES = 0;
    static constexpr uint32_t SUM_OFFSET_BYTES = 0;
    static constexpr uint32_t REDUCE_WORK_OFFSET_BYTES = SUM_OFFSET_BYTES + SUM_BYTES;
    static constexpr uint32_t BROADCAST_OFFSET_BYTES = REDUCE_WORK_OFFSET_BYTES + REDUCE_BYTES;
    static constexpr uint32_t Q_RSTD_OFFSET_BYTES = BROADCAST_OFFSET_BYTES + REDUCE_BYTES;
    static constexpr uint32_t K_RSTD_OFFSET_BYTES = Q_RSTD_OFFSET_BYTES + RSTD_BYTES;

    static_assert(Q_INPUT_OFFSET_BYTES + INPUT_TILE_BYTES <= INPUT_BUFFER_BYTES_16K &&
                  DQ_INPUT_OFFSET_BYTES + INPUT_TILE_BYTES <= INPUT_BUFFER_BYTES_16K,
                  "QK inputPing slices exceed the allocated buffer");
    static_assert(DK_INPUT_OFFSET_BYTES + INPUT_TILE_BYTES <= CALC_BUFFER_BYTES_32K &&
                  DK_WY_INPUT_OFFSET_BYTES + INPUT_TILE_BYTES <= CALC_BUFFER_BYTES_32K &&
                  K_INPUT_OFFSET_BYTES + INPUT_TILE_BYTES <= CALC_BUFFER_BYTES_32K &&
                  PRODUCT_OFFSET_BYTES + NORM_TILE_BYTES <= CALC_BUFFER_BYTES_32K,
                  "QK calcC slices exceed the allocated buffer");
    static_assert(GRAD_Q_OFFSET_BYTES + NORM_TILE_BYTES <= CALC_BUFFER_BYTES_32K &&
                  GRAD_K_OFFSET_BYTES + FP32_TILE_BYTES <= CALC_BUFFER_BYTES_32K &&
                  X_OFFSET_BYTES + NORM_TILE_BYTES <= CALC_BUFFER_BYTES_32K,
                  "QK calcA/calcB slices exceed the allocated buffer");
    static_assert(SUM_OFFSET_BYTES + SUM_BYTES <= INPUT_BUFFER_BYTES_16K &&
                  REDUCE_WORK_OFFSET_BYTES + REDUCE_BYTES <= INPUT_BUFFER_BYTES_16K &&
                  BROADCAST_OFFSET_BYTES + REDUCE_BYTES <= INPUT_BUFFER_BYTES_16K &&
                  Q_RSTD_OFFSET_BYTES + RSTD_BYTES <= INPUT_BUFFER_BYTES_16K &&
                  K_RSTD_OFFSET_BYTES + RSTD_BYTES <= INPUT_BUFFER_BYTES_16K,
                  "QK inputPong slices exceed the allocated buffer");
    static_assert(DQ_INPUT_OFFSET_BYTES + INPUT_TILE_BYTES == INPUT_BUFFER_BYTES_16K &&
                  GRAD_K_OFFSET_BYTES + FP32_TILE_BYTES == NORM_TILE_BYTES &&
                  NORM_TILE_BYTES == CALC_BUFFER_BYTES_32K,
                  "QK input and fused gradient panels must retain their buffer composition");
    static_assert(INPUT_TILE_BYTES % DATA_BLOCK_BYTES_32 == 0 &&
                  FP32_TILE_BYTES % DATA_BLOCK_BYTES_32 == 0 &&
                  SUM_BYTES % DATA_BLOCK_BYTES_32 == 0 &&
                  REDUCE_BYTES % DATA_BLOCK_BYTES_32 == 0 &&
                  RSTD_BYTES % DATA_BLOCK_BYTES_32 == 0,
                  "QK slice offsets must remain 32-byte aligned");
    static_assert(ROWS_32 % FP32_PER_BLOCK == 0 && NORM_ROWS % FP32_PER_BLOCK == 0,
                  "QK row groups must contain complete reduction/broadcast blocks");
};

struct PrepareWyReprBwdTilingData {
    int64_t layout;
    int64_t B;
    int64_t HV;
    int64_t HK;
    int64_t groupSize;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkSize;
    int64_t chunkNum;
    int64_t chunkNumPerB;
    int64_t isVariable;

    int64_t workspaceBufferCount;
    int64_t workspaceSlotSize;
    int64_t workspaceCoreSize;

    int64_t kBytes;
    int64_t vBytes;
    int64_t mBytes;

    int64_t kbgOffset;
    int64_t vbOffset;
    int64_t kbetaOffset;
    int64_t dkbgOffset;
    int64_t dvbOffset;
    int64_t da1Offset;
    int64_t da2Offset;
    int64_t da4Offset;
    int64_t da5Offset;
    int64_t da6Offset;
    int64_t dOffset;
    int64_t dkbOffset;
    int64_t dkOffset;
    int64_t kktOffset;

    int64_t kVecRow;
    int64_t vVecRow;
    int64_t mVecRow;
    int64_t kktVecRow;
};

} // namespace GDN

#endif // PREPARE_WY_REPR_BWD_STRUCT_H
