/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged_tiling.cpp
 * \brief Host tiling：校验契约 + 填 tiling 结构 + 选 TilingKey（gate 模式）。
 *
 * 契约（docs/api.md）：
 *   k [1,HK,T,K] / w [1,HV,T,K] / u [1,HV,T,V]；v / bg 是 DPLR 专用输入位，本版本不支持 DPLR，
 *   必须为空（传非空直接拒绝，见 api.md §3.4）
 *   g 或 gk 二选一；cu_seqlens 必给（host int 数组 → INT64 tensor），严格递增、0<=cu[0]<cu[-1]<=T
 *   K = V = 128、chunk_size = 64；hm [Nseq,HV,K,V+K] FP32
 */

#include "pre_process_fwd_kernel_merged_tiling.h"

#include <register/op_impl_registry.h>
#include "platform/soc_spec.h"
#include "tiling_base/tiling_templates_registry.h"
#include <cstdlib>   // std::getenv / std::atoll（PPFM_FORCE_COLSPLIT 测试钩子）

namespace optiling {

namespace {
constexpr size_t INPUT_K_IDX = 0;
constexpr size_t INPUT_W_IDX = 1;
constexpr size_t INPUT_U_IDX = 2;
constexpr size_t INPUT_G_IDX = 3;
constexpr size_t INPUT_GK_IDX = 4;
constexpr size_t INPUT_BG_IDX = 5;
constexpr size_t INPUT_V_IDX = 6;
constexpr size_t INPUT_SEQLENS_IDX = 7;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;

constexpr int64_t PPFM_FIXED_K = 128;
constexpr int64_t PPFM_FIXED_V = 128;
constexpr int64_t PPFM_FIXED_CHUNK = 64;

int64_t DtypeToEnum(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return GDN::PPFM_DTYPE_BF16;
    }
    if (dtype == ge::DT_FLOAT16) {
        return GDN::PPFM_DTYPE_FP16;
    }
    return GDN::PPFM_DTYPE_FP32;
}

void PrintTiling(gert::TilingContext *context, const PreProcessFwdKernelMergedTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>> PreProcessFwdKernelMerged tiling <<<<<<<<<<<");
    OP_LOGD(nodeName, "= B:%ld Hk:%ld Hv:%ld hvPerHk:%ld T:%ld K:%ld V:%ld chunkSize:%ld",
            tiling.B, tiling.Hk, tiling.Hv, tiling.hvPerHk, tiling.T, tiling.K, tiling.V, tiling.chunkSize);
    OP_LOGD(nodeName, "= nSeq:%ld gateMode:%ld gateDtype:%ld usedAicNum:%ld taskNum:%ld",
            tiling.nSeq, tiling.gateMode, tiling.gateDtype, tiling.usedAicNum, tiling.taskNum);
}

// Cube(Matmul) 自己的 tiling：v2 的四个矩阵乘只有两种形状
//   ① vTmp = W_c @ bf16(h)、③ T1 = W_c @ bf16(m)：M=BT, N=128, K=128，A 不转置
//   ② dH   = k_c^T @ bf16(v_new)、④ T2 = left^T @ bf16(T1)：M=128, N=128, K=BT，A 转置
bool BuildCubeTiling(const platform_ascendc::PlatformAscendC &platform,
                     GDN::PreProcessFwdKernelMergedTilingData *tiling)
{
    constexpr uint64_t kBT = 64ULL;
    constexpr uint64_t kKv = 128ULL;
    {
        matmul_tiling::MatmulApiTiling mm(platform);
        if (mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_BFLOAT16, false) != 0 ||
            mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_BFLOAT16, false) != 0 ||
            mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT) != 0 ||
            mm.EnableBias(false) != 0 ||
            mm.SetShape(static_cast<int32_t>(kBT), static_cast<int32_t>(kKv),
                        static_cast<int32_t>(kKv)) != 0 ||
            mm.SetOrgShape(static_cast<int32_t>(kBT), static_cast<int32_t>(kKv),
                           static_cast<int32_t>(kKv)) != 0 ||
            mm.SetFixSplit(static_cast<int32_t>(kBT), static_cast<int32_t>(kKv), -1) != 0 ||
            mm.SetBufferSpace(-1, -1, 0, -1) != 0 ||
            mm.GetTiling(tiling->cubeNoTrans) == -1) {
            return false;
        }
    }
    {
        matmul_tiling::MatmulApiTiling mm(platform);
        if (mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_BFLOAT16, true) != 0 ||
            mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_BFLOAT16, false) != 0 ||
            mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT) != 0 ||
            mm.EnableBias(false) != 0 ||
            mm.SetShape(static_cast<int32_t>(kKv), static_cast<int32_t>(kKv),
                        static_cast<int32_t>(kBT)) != 0 ||
            mm.SetOrgShape(static_cast<int32_t>(kKv), static_cast<int32_t>(kKv),
                           static_cast<int32_t>(kBT)) != 0 ||
            mm.SetFixSplit(static_cast<int32_t>(kKv), static_cast<int32_t>(kKv), -1) != 0 ||
            mm.SetBufferSpace(-1, -1, 0, -1) != 0 ||
            mm.GetTiling(tiling->cubeTransA) == -1) {
            return false;
        }
    }
    return true;
}
} // namespace

#include "pre_process_fwd_kernel_merged_tiling_processor.h"

ge::graphStatus Tiling4PreProcessFwdKernelMerged(gert::TilingContext *context)
{
    // tiling 计算主体在 *_tiling_processor.h（header-only，§2）
    return PreProcessFwdTilingProcessor(context);
}

ge::graphStatus TilingParse4PreProcessFwdKernelMerged(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PreProcessFwdKernelMerged)
    .Tiling(Tiling4PreProcessFwdKernelMerged)
    .TilingParse<PreProcessFwdKernelMergedCompileInfo>(TilingParse4PreProcessFwdKernelMerged);

} // namespace optiling
