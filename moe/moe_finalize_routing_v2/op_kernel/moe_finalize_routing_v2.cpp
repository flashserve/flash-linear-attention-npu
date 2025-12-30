/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_finalize_routing_v2.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_fp_db.h"
#include "moe_finalize_routing_v2_bf16.h"
#include "moe_finalize_routing_v2_fp_cuth.h"
#include "moe_finalize_routing_v2_fp_cuth_k2.h"
#include "moe_finalize_routing_v2_fp_cuth_k4.h"
#include "moe_finalize_routing_v2_fp_cutk.h"
#include "moe_finalize_routing_v2_bf16_cutk.h"
#include "moe_finalize_routing_v2_bf16_cuth.h"
#include "moe_finalize_routing_v2_bf16_cuth_k2.h"
#include "moe_finalize_routing_v2_bf16_cuth_k4.h"
#include "moe_finalize_routing_v2_fp_db_all_bias.h"
#include "moe_finalize_routing_v2_bf16_all_bias.h"
#include "moe_finalize_routing_v2_fp_cuth_k2_optimized.h"

#define TILING_KEY_DTYPE_FLOAT_BIG_K_V2 20000
#define TILING_KEY_DTYPE_FLOAT16_BIG_K_V2 20001
#define TILING_KEY_DTYPE_BF16_BIG_K_V2 20002
#define TILING_KEY_DTYPE_BF16_V2 20003
#define TILING_KEY_DTYPE_FLOAT_DB_V2 20004
#define TILING_KEY_DTYPE_FLOAT16_DB_V2 20005
#define TILING_KEY_DTYPE_FLOAT_CUTH_K2_V2 20006
#define TILING_KEY_DTYPE_FLOAT16_CUTH_K2_V2 20007
#define TILING_KEY_DTYPE_BF16_CUTH_K2_V2 20008
#define TILING_KEY_DTYPE_FLOAT_CUTH_K4_V2 20009
#define TILING_KEY_DTYPE_FLOAT16_CUTH_K4_V2 20010
#define TILING_KEY_DTYPE_BF16_CUTH_K4_V2 20011
#define TILING_KEY_DTYPE_FLOAT_CUTH_V2 20012
#define TILING_KEY_DTYPE_FLOAT16_CUTH_V2 20013
#define TILING_KEY_DTYPE_BF16_CUTH_V2 20014
#define TILING_KEY_DTYPE_FLOAT_DB_ALL_BIAS_V2 20015
#define TILING_KEY_DTYPE_FLOAT16_DB_ALL_BIAS_V2 20016
#define TILING_KEY_DTYPE_BF16_ALL_BIAS_V2 20017
#define TILING_KEY_DTYPE_FLOAT_CUTH_NETWORK_V2 20018
#define TILING_KEY_DTYPE_FLOAT_BIG_K_V2_WITHOUT_BIAS 20019
#define TILING_KEY_DTYPE_FLOAT16_BIG_K_V2_WITHOUT_BIAS 20020
#define TILING_KEY_DTYPE_BF16_BIG_K_V2_WITHOUT_BIAS 20021
#define TILING_KEY_DTYPE_BF16_V2_WITHOUT_BIAS 20022
#define TILING_KEY_DTYPE_FLOAT_DB_V2_WITHOUT_BIAS 20023
#define TILING_KEY_DTYPE_FLOAT16_DB_V2_WITHOUT_BIAS 20024
#define TILING_KEY_DTYPE_FLOAT_CUTH_K2_V2_WITHOUT_BIAS 20025
#define TILING_KEY_DTYPE_FLOAT16_CUTH_K2_V2_WITHOUT_BIAS 20026
#define TILING_KEY_DTYPE_BF16_CUTH_K2_V2_WITHOUT_BIAS 20027
#define TILING_KEY_DTYPE_FLOAT_CUTH_K4_V2_WITHOUT_BIAS 20028
#define TILING_KEY_DTYPE_FLOAT16_CUTH_K4_V2_WITHOUT_BIAS 20029
#define TILING_KEY_DTYPE_BF16_CUTH_K4_V2_WITHOUT_BIAS 20030
#define TILING_KEY_DTYPE_FLOAT_CUTH_V2_WITHOUT_BIAS 20031
#define TILING_KEY_DTYPE_FLOAT16_CUTH_V2_WITHOUT_BIAS 20032
#define TILING_KEY_DTYPE_BF16_CUTH_V2_WITHOUT_BIAS 20033
#define TILING_KEY_DTYPE_FLOAT_DB_ALL_BIAS_V2_WITHOUT_BIAS 20034
#define TILING_KEY_DTYPE_FLOAT16_DB_ALL_BIAS_V2_WITHOUT_BIAS 20035
#define TILING_KEY_DTYPE_BF16_ALL_BIAS_V2_WITHOUT_BIAS 20036
#define TILING_KEY_DTYPE_BF16_BIG_K_V2_MIX_PRECISION 20038
#define TILING_KEY_DTYPE_BF16_V2_MIX_PRECISION 20039
#define TILING_KEY_DTYPE_BF16_CUTH_K2_V2_MIX_PRECISION 20040
#define TILING_KEY_DTYPE_BF16_CUTH_K4_V2_MIX_PRECISION 20041
#define TILING_KEY_DTYPE_BF16_CUTH_V2_MIX_PRECISION 20042
#define TILING_KEY_DTYPE_BF16_ALL_BIAS_V2_MIX_PRECISION 20043
#define TILING_KEY_DTYPE_BF16_BIG_K_V2_WITHOUT_BIAS_MIX_PRECISION 20044
#define TILING_KEY_DTYPE_BF16_V2_WITHOUT_BIAS_MIX_PRECISION 20045
#define TILING_KEY_DTYPE_BF16_CUTH_K2_V2_WITHOUT_BIAS_MIX_PRECISION 20046
#define TILING_KEY_DTYPE_BF16_CUTH_K4_V2_WITHOUT_BIAS_MIX_PRECISION 20047
#define TILING_KEY_DTYPE_BF16_CUTH_V2_WITHOUT_BIAS_MIX_PRECISION 20048
#define TILING_KEY_DTYPE_BF16_ALL_BIAS_V2_WITHOUT_BIAS_MIX_PRECISION 20049

using namespace MoeFinalizeRoutingV2;

extern "C" __global__ __aicore__ void moe_finalize_routing_v2(
    GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
    GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
#if (ORIG_DTYPE_EXPANDED_X == DT_FLOAT)
    if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_BIG_K_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCutK<float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_DB_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpDb<float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_CUTH_K2_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuthK2<float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_CUTH_K4_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuthK4<float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_CUTH_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuth<float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_DB_ALL_BIAS_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpDbAllBias<float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_CUTH_NETWORK_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuthK2Optimized<float> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_BIG_K_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCutK<float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_CUTH_K4_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuthK4<float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_CUTH_K2_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuthK2<float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_DB_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpDb<float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_CUTH_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuth<float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT_DB_ALL_BIAS_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpDbAllBias<float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    }
#endif

#if (ORIG_DTYPE_EXPANDED_X == DT_FLOAT16)
    if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_BIG_K_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCutK<half, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_DB_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpDb<half, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_CUTH_K2_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuthK2<half, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_CUTH_K4_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuthK4<half, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_CUTH_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuth<half, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_DB_ALL_BIAS_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpDbAllBias<half, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_BIG_K_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCutK<half, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_DB_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpDb<half, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_CUTH_K2_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuthK2<half, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_DB_ALL_BIAS_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpDbAllBias<half, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_CUTH_K4_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuthK4<half, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_FLOAT16_CUTH_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2FpCuth<half, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    }
#endif

#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003) && (ORIG_DTYPE_EXPANDED_X == DT_BF16)
    if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_BIG_K_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CutK<bfloat16_t, bfloat16_t, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2BF16<bfloat16_t, bfloat16_t, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_K2_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CuthK2<bfloat16_t, bfloat16_t, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_K4_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CuthK4<bfloat16_t, bfloat16_t, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16Cuth<bfloat16_t, bfloat16_t, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_ALL_BIAS_V2)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2BF16AllBias<bfloat16_t, bfloat16_t, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_BIG_K_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CutK<bfloat16_t, bfloat16_t, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2BF16<bfloat16_t, bfloat16_t, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_K2_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CuthK2<bfloat16_t, bfloat16_t, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_K4_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CuthK4<bfloat16_t, bfloat16_t, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16Cuth<bfloat16_t, bfloat16_t, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_ALL_BIAS_V2_WITHOUT_BIAS)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2BF16AllBias<bfloat16_t, bfloat16_t, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_BIG_K_V2_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CutK<bfloat16_t, float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_V2_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2BF16<bfloat16_t, float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_K2_V2_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CuthK2<bfloat16_t, float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_K4_V2_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CuthK4<bfloat16_t, float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_V2_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16Cuth<bfloat16_t, float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_ALL_BIAS_V2_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2BF16AllBias<bfloat16_t, float, true> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_BIG_K_V2_WITHOUT_BIAS_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CutK<bfloat16_t, float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_V2_WITHOUT_BIAS_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2BF16<bfloat16_t, float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_K2_V2_WITHOUT_BIAS_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CuthK2<bfloat16_t, float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_K4_V2_WITHOUT_BIAS_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16CuthK4<bfloat16_t, float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_CUTH_V2_WITHOUT_BIAS_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2Bf16Cuth<bfloat16_t, float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DTYPE_BF16_ALL_BIAS_V2_WITHOUT_BIAS_MIX_PRECISION)) {
        MoeFinalizeRoutingV2::MoeFinalizeRoutingV2BF16AllBias<bfloat16_t, float, false> op(tilingData, pipe);
        op.Init(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow, out, userWS);
        op.Process();
    }
#endif
}
