/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess.cpp
 * \brief Device entry：模板参数直接给出 dtype / gate 模式，由 tilingKey 选择实例（不使用 TILING_KEY_IS）。
 */

#include "chunk_delta_h_bwd_preprocess_kernel.h"

#ifndef TORCH_MODE
#include "lib/matmul_intf.h"

template <int D_T_Q, int D_T_G, uint32_t GATE_MODE>
__global__ __aicore__ void chunk_delta_h_bwd_preprocess(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR cu_seqlens,
    GM_ADDR dhm, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(CP::ChunkDeltaHBwdPreprocessTilingData);
    GET_TILING_DATA_WITH_STRUCT(CP::ChunkDeltaHBwdPreprocessTilingData, tilingData, tiling);
    // A5（arch35）：1 AIC : 2 AIV，两个 AIV 各承包一个 head 的整条逆序链（参考 ChunkFwdH）。
    // A2/A3（arch22）：仍 1:1 —— 910B 的 0x2 集合同步与当前协议组合会死锁（见 op_host 的 aivPerBlock 注释）。
    // 统一 1 AIC : 1 AIV。1:2 核型（参考 ChunkFwdH）实测能让 A5 gva 档从 3.35×H20 降到 1.78×，
    // 但尾块（T 不是 chunkSize 整数倍）的 P 面仍不对（E 面正常），修好前不启用；
    // 打开时需与 policy.h 的 CDHP_CROSS_CORE_MODE、op_host 的 aivPerBlock 一起切。
    // 统一 1 AIC : 1 AIV（1:2 的框架在 policy/common 里已就位，切换方式见 policy.h 注释）。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#endif

    CP::ChunkDeltaHBwdPreprocessDispatch<D_T_Q, D_T_G, GATE_MODE>::Invoke(
        q, k, w, d_o, dv, g, gk, cu_seqlens, dhm, userWS, &tilingData);
}
#endif
