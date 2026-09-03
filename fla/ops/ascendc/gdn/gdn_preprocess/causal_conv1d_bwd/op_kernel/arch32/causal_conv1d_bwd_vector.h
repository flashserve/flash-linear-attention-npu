/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ASCENDC_CAUSAL_CONV1D_BWD_ARCH32_VECTOR_H_
#define ASCENDC_CAUSAL_CONV1D_BWD_ARCH32_VECTOR_H_

#include "kernel_operator.h"

namespace NsCausalConv1dBwdArch32 {

using namespace AscendC;

constexpr uint32_t DIRECT_BD = 64;
constexpr uint32_t DIRECT_WIDE_BD = 128;
constexpr uint32_t DIRECT_W = 4;
constexpr uint32_t FP32_BLOCK_ELEMENTS = 8;
constexpr uint32_t FP32_VECTOR_ELEMENTS = 64;

__aicore__ inline void ReduceRowsInplace(LocalTensor<float> tensor, uint32_t rows, uint32_t cols)
{
    uint32_t remainRows = rows;
    while (remainRows > 1) {
        uint32_t upperRows = (remainRows + 1) >> 1;
        uint32_t pairRows = remainRows >> 1;
        Add(tensor, tensor, tensor[upperRows * cols], pairRows * cols);
        PipeBarrier<PIPE_V>();
        remainRows = upperRows;
    }
}

__aicore__ inline void ApplySiluBackward(
    LocalTensor<float> dy, LocalTensor<float> y, LocalTensor<float> sigmoidScratch,
    LocalTensor<float> oneBlock, uint32_t count)
{
    Muls(sigmoidScratch, y, float(-1.0), count);
    PipeBarrier<PIPE_V>();
    Exp(sigmoidScratch, sigmoidScratch, count);
    PipeBarrier<PIPE_V>();
    Adds(sigmoidScratch, sigmoidScratch, float(1.0), count);
    PipeBarrier<PIPE_V>();

    BinaryRepeatParams divParams;
    divParams.dstBlkStride = 1;
    divParams.src0BlkStride = 1;
    divParams.src1BlkStride = 1;
    divParams.dstRepStride = static_cast<uint16_t>(FP32_VECTOR_ELEMENTS / FP32_BLOCK_ELEMENTS);
    divParams.src0RepStride = 0;
    divParams.src1RepStride = static_cast<uint16_t>(FP32_VECTOR_ELEMENTS / FP32_BLOCK_ELEMENTS);
    Div(sigmoidScratch, oneBlock, sigmoidScratch, static_cast<uint64_t>(FP32_VECTOR_ELEMENTS),
        static_cast<uint8_t>(count / FP32_VECTOR_ELEMENTS), divParams);
    PipeBarrier<PIPE_V>();

    Mul(dy, dy, sigmoidScratch, count);
    PipeBarrier<PIPE_V>();
    Muls(sigmoidScratch, sigmoidScratch, float(-1.0), count);
    PipeBarrier<PIPE_V>();
    Adds(sigmoidScratch, sigmoidScratch, float(1.0), count);
    PipeBarrier<PIPE_V>();
    Mul(y, y, sigmoidScratch, count);
    PipeBarrier<PIPE_V>();
    Adds(y, y, float(1.0), count);
    PipeBarrier<PIPE_V>();
    Mul(dy, dy, y, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComputeTile(
    LocalTensor<float> x, LocalTensor<float> dy, LocalTensor<float> y,
    LocalTensor<float> weight, LocalTensor<float> dx, LocalTensor<float> dw,
    LocalTensor<float> db, LocalTensor<float> oneBlock,
    uint32_t bt, uint32_t bd, uint32_t width, uint32_t dyRows, uint32_t hasBias,
    bool reuseHalo)
{
    uint32_t btCount = bt * bd;
    uint32_t activationOffset = reuseHalo ? (width - 1) * bd : 0;
    uint32_t activationCount = reuseHalo ? btCount : dyRows * bd;
    ApplySiluBackward(dy[activationOffset], y[activationOffset], dx, oneBlock, activationCount);

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = static_cast<uint16_t>(bd / FP32_BLOCK_ELEMENTS);
    repeatParams.src0RepStride = static_cast<uint16_t>(bd / FP32_BLOCK_ELEMENTS);
    repeatParams.src1RepStride = 0;

    for (uint32_t iW = 0; iW < width; iW++) {
        uint32_t wIdx = width - iW - 1;
        LocalTensor<float> shiftedDy = dy[iW * bd];
        for (uint32_t dOffset = 0; dOffset < bd; dOffset += FP32_VECTOR_ELEMENTS) {
            if (iW == 0) {
                Mul(dx[dOffset], shiftedDy[dOffset], weight[wIdx * bd + dOffset],
                    static_cast<uint64_t>(FP32_VECTOR_ELEMENTS),
                    static_cast<uint8_t>(bt), repeatParams);
            } else {
                MulAddDst(dx[dOffset], shiftedDy[dOffset], weight[wIdx * bd + dOffset],
                          static_cast<uint64_t>(FP32_VECTOR_ELEMENTS),
                          static_cast<uint8_t>(bt), repeatParams);
            }
        }
        PipeBarrier<PIPE_V>();

        Mul(y, shiftedDy, x, btCount);
        PipeBarrier<PIPE_V>();
        ReduceRowsInplace(y, bt, bd);
        Add(dw[wIdx * bd], dw[wIdx * bd], y, bd);
        PipeBarrier<PIPE_V>();
    }

    if (hasBias != 0) {
        ReduceRowsInplace(dy, bt, bd);
        Add(db, db, dy, bd);
        PipeBarrier<PIPE_V>();
    }
}

} // namespace NsCausalConv1dBwdArch32

#endif // ASCENDC_CAUSAL_CONV1D_BWD_ARCH32_VECTOR_H_
