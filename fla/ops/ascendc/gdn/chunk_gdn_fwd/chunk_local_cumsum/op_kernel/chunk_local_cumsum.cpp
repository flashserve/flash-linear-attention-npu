/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_local_cumsum.cpp
 * \brief
 */

#include "kernel_operator.h"

#if defined(CHUNK_LOCAL_CUMSUM_RUNTIME_KEYS)
#include "chunk_local_cumsum_kernel.h"
extern "C" __global__ __aicore__ void chunk_local_cumsum(GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                                          GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ChunkLocalCumsumTilingData);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(ChunkLocalCumsumTilingData, tilingData, tiling);
        ChunkLocalCumsumKernel<float, float, CHUNK_128_SIZE> op;
        op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(ChunkLocalCumsumTilingData, tilingData, tiling);
        if (tilingData.inputDtype == DTYPE_FP16 && tilingData.outputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<half, half> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_FP16 && tilingData.outputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<half, bfloat16_t> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<half, float> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_BF16 && tilingData.outputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<bfloat16_t, half> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_BF16 && tilingData.outputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<bfloat16_t, bfloat16_t> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<bfloat16_t, float> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.outputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<float, half> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.outputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<float, bfloat16_t> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else {
            ChunkLocalCumsumKernel<float, float> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        }
    }
}
#else
#include "chunk_local_cumsum_tiling_key.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
#include "arch92/chunk_local_cumsum_kernel.h"
#else
#include "chunk_local_cumsum_kernel.h"
#endif

namespace {
template <uint32_t SEG128>
__aicore__ inline void RunChunkLocalCumsum(GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR out,
                                           const ChunkLocalCumsumTilingData *tilingData)
{
    if constexpr (SEG128 == 1) {
        ChunkLocalCumsumKernel<float, float, CHUNK_128_SIZE> op;
        op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
        op.Process();
    } else {
        if (tilingData->inputDtype == DTYPE_FP16 && tilingData->outputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<half, half> op;
            op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
            op.Process();
        } else if (tilingData->inputDtype == DTYPE_FP16 && tilingData->outputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<half, bfloat16_t> op;
            op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
            op.Process();
        } else if (tilingData->inputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<half, float> op;
            op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
            op.Process();
        } else if (tilingData->inputDtype == DTYPE_BF16 && tilingData->outputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<bfloat16_t, half> op;
            op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
            op.Process();
        } else if (tilingData->inputDtype == DTYPE_BF16 && tilingData->outputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<bfloat16_t, bfloat16_t> op;
            op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
            op.Process();
        } else if (tilingData->inputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<bfloat16_t, float> op;
            op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
            op.Process();
        } else if (tilingData->outputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<float, half> op;
            op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
            op.Process();
        } else if (tilingData->outputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<float, bfloat16_t> op;
            op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
            op.Process();
        } else {
            ChunkLocalCumsumKernel<float, float> op;
            op.Init(g, cuSeqlens, chunkIndices, out, tilingData);
            op.Process();
        }
    }
}
}

template <uint32_t SEG128>
__global__ __aicore__ void chunk_local_cumsum(GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                              GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ChunkLocalCumsumTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkLocalCumsumTilingData, tilingData, tiling);
    RunChunkLocalCumsum<SEG128>(g, cuSeqlens, chunkIndices, out, &tilingData);
}
#endif
