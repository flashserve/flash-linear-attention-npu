/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/moe_distribute_combine_v2.h"
#include "moe_distribute_combine_v2_entry.h"

using namespace MoeDistributeCombineV2Impl;
using namespace Mc2Kernel;
using namespace AscendC;

/*
* A3 tilingkey说明
* 3位的十进制数
* 第1位（个位）：quantMode:
*     0: 不量化, 2: 使能量化
* 第2位（十位）：x输入类型:
*     0: float16, 1: bfloat16
* 第3位（百位）：无实际含义
*/

template<typename ExpandXType, bool IsInt8Quant>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_combine_v2(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeCombineV2Info tilingData)
{
    TPipe pipe;
    MoeDistributeCombineV2<ExpandXType, IsInt8Quant> op;
    op.Init(mc2Context, expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales,
    xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo,
    XOut, workspaceGM, tilingData, &pipe);
        
    op.Process();

    pipe.DestroyWithoutPipeAll();
    return;
}

__aicore__ inline void moe_combine_switch(
    int32_t tilingKey,
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
    GM_ADDR performanceInfo, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR mc2Context,
    MoeDistributeCombineV2Info tilingData)
{
    switch (tilingKey) {

    case 100:
        moe_distribute_combine_v2<float16_t, false>(
            expandX, expertIds, expandIdx, epSendCount, residualX,
            gamma, expertScales, xActiveMask,
            sharedExpertX, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, XOut,
            workspaceGM, mc2Context, tilingData);
        break;

    case 102:
        moe_distribute_combine_v2<float16_t, true>(
            expandX, expertIds, expandIdx, epSendCount, residualX,
            gamma, expertScales, xActiveMask,
            sharedExpertX, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, XOut,
            workspaceGM, mc2Context, tilingData);
        break;

    case 110:
        moe_distribute_combine_v2<bfloat16_t, false>(
            expandX, expertIds, expandIdx, epSendCount, residualX,
            gamma, expertScales, xActiveMask,
            sharedExpertX, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, XOut,
            workspaceGM, mc2Context, tilingData);
        break;

    case 112:
        moe_distribute_combine_v2<bfloat16_t, true>(
            expandX, expertIds, expandIdx, epSendCount, residualX,
            gamma, expertScales, xActiveMask,
            sharedExpertX, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, XOut,
            workspaceGM, mc2Context, tilingData);
        break;

    default:
        AscendC::PRINTF("moe_distribute_combine_v2 Error: invalid tilingKey = %d\n", tilingKey);
        break;
    }
}

extern "C" __global__ __aicore__ void moe_distribute_combine_v2_generic(
    int32_t tilingKey,
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeCombineV2Info tilingData)
{
    // 根据不同的数据类型调用不同的模板
    moe_combine_switch(
        tilingKey,
        expandX, expertIds, expandIdx, epSendCount, residualX,
        gamma, expertScales, xActiveMask,
        sharedExpertX, oriX,
        constExpertAlpha1, constExpertAlpha2, constExpertV,
        performanceInfo, XOut,
        workspaceGM, mc2Context,
        tilingData);
}

template <int32_t split_num>
__sk__ __vector__ void moe_distribute_combine_v2_generic_sk(MoeDistributeCombineV2SkArgs *args)
{
    int32_t tilingKey = args->tilingKey;
    GM_ADDR expandX = args->expandX;
    GM_ADDR expertIds = args->expertIds;
    GM_ADDR expandIdx = args->expandIdx;
    GM_ADDR epSendCount = args->epSendCount;
    GM_ADDR residualX = args->residualX;
    GM_ADDR gamma = args->gamma;
    GM_ADDR expertScales = args->expertScales;
    GM_ADDR xActiveMask = args->xActiveMask;
    GM_ADDR sharedExpertX = args->sharedExpertX;
    GM_ADDR oriX = args->oriX;
    GM_ADDR constExpertAlpha1 = args->constExpertAlpha1;
    GM_ADDR constExpertAlpha2 = args->constExpertAlpha2;
    GM_ADDR constExpertV = args->constExpertV;
    GM_ADDR performanceInfo = args->performanceInfo;
    GM_ADDR XOut = args->XOut;
    GM_ADDR workspaceGM = args->workspaceGM;
    GM_ADDR mc2Context = args->mc2Context;
    MoeDistributeCombineV2Info tilingData = args->tilingData;

    // 根据不同的数据类型调用不同的模板
    moe_combine_switch(
        tilingKey,
        expandX, expertIds, expandIdx, epSendCount, residualX,
        gamma, expertScales, xActiveMask,
        sharedExpertX, oriX,
        constExpertAlpha1, constExpertAlpha2, constExpertV,
        performanceInfo, XOut,
        workspaceGM, mc2Context,
        tilingData);
}

SK_BIND(moe_distribute_combine_v2_generic, 4, moe_distribute_combine_v2_generic_sk<0>,
    moe_distribute_combine_v2_generic_sk<1>, moe_distribute_combine_v2_generic_sk<2>,
    moe_distribute_combine_v2_generic_sk<3>);

// <<<>>>调用函数
void moe_distribute_combine_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeCombineV2Info tilingData)
{
    moe_distribute_combine_v2_generic<<<blockDim, nullptr, stream>>>(
        tilingKey,
        expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales, xActiveMask,
        sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut,
        workspaceGM, mc2Context, tilingData);
}
