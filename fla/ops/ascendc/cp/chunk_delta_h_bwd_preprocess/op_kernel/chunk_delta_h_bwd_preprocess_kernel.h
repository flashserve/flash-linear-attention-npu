/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess_kernel.h
 * \brief Kernel 主体与模板分派：dtype / gate 模式由模板参数给定，arch35 与 arch22 各自实现 Cube/Vector。
 */

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_KERNEL_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_KERNEL_H

#include "kernel_operator.h"
#include "chunk_delta_h_bwd_preprocess_struct.h"
#include "chunk_delta_h_bwd_preprocess_policy.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
// arch35（Ascend950 / A5）
#include "arch35/chunk_delta_h_bwd_preprocess_cube.h"
#include "arch35/chunk_delta_h_bwd_preprocess_vector.h"
#else
// arch22（Ascend910b / Ascend910_93，即 A2/A3）
#include "arch22/chunk_delta_h_bwd_preprocess_cube.h"
#include "arch22/chunk_delta_h_bwd_preprocess_vector.h"
#endif

namespace CP {

// 模板 dtype 码 → 具体类型
template <int D_TYPE>
struct ChunkDeltaHBwdPreprocessDType;

template <>
struct ChunkDeltaHBwdPreprocessDType<CHUNK_DELTA_H_BWD_PREPROCESS_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct ChunkDeltaHBwdPreprocessDType<CHUNK_DELTA_H_BWD_PREPROCESS_TPL_FP16> {
    using type = half;
};

template <>
struct ChunkDeltaHBwdPreprocessDType<CHUNK_DELTA_H_BWD_PREPROCESS_TPL_FP32> {
    using type = float;
};

template <typename DT, typename GT>
__aicore__ inline void ChunkDeltaHBwdPreprocessImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR cu_seqlens,
    GM_ADDR dhm, GM_ADDR userWS, const ChunkDeltaHBwdPreprocessTilingData *tilingData)
{
    if ASCEND_IS_AIC {
        ChunkDeltaHBwdPreprocessCube<DT> cubeOp;
        cubeOp.Init(q, k, w, d_o, dv, cu_seqlens, dhm, userWS, *tilingData);
        cubeOp.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkDeltaHBwdPreprocessVector<DT, GT> vecOp;
        vecOp.Init(q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS, *tilingData);
        vecOp.InitBuffer(&pipe);
        vecOp.Process();
    }
}

// 按模板 key 分派：D_T_Q/D_T_G 取自 tiling 的 GET_TPL_TILING_KEY，GATE_MODE 决定门控分支
template <int D_T_Q, int D_T_G, uint32_t GATE_MODE>
struct ChunkDeltaHBwdPreprocessDispatch {
    using DTypeQ = typename ChunkDeltaHBwdPreprocessDType<D_T_Q>::type;
    using DTypeG = typename ChunkDeltaHBwdPreprocessDType<D_T_G>::type;

    __aicore__ inline static void Invoke(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g,
                                         GM_ADDR gk, GM_ADDR cu_seqlens, GM_ADDR dhm, GM_ADDR userWS,
                                         const ChunkDeltaHBwdPreprocessTilingData *tilingData)
    {
        ChunkDeltaHBwdPreprocessImpl<DTypeQ, DTypeG>(q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS, tilingData);
    }
};

} // namespace CP

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_KERNEL_H
