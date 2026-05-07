/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_ENTRY_H
#define MOE_DISTRIBUTE_COMBINE_V2_ENTRY_H

#include "op_kernel/moe_distribute_combine_v2_tiling.h"

struct MoeDistributeCombineV2SkArgs {
    int32_t tilingKey;
    GM_ADDR expandX;
    GM_ADDR expertIds;
    GM_ADDR expandIdx;
    GM_ADDR epSendCount;
    GM_ADDR residualX;
    GM_ADDR gamma;
    GM_ADDR expertScales;
    GM_ADDR xActiveMask;
    GM_ADDR sharedExpertX;
    GM_ADDR oriX;
    GM_ADDR constExpertAlpha1;
    GM_ADDR constExpertAlpha2;
    GM_ADDR constExpertV;
    GM_ADDR performanceInfo;
    GM_ADDR XOut;
    GM_ADDR workspaceGM;
    GM_ADDR mc2Context;
    MoeDistributeCombineV2Info tilingData;
};

// <<<>>>调用函数声明
void moe_distribute_combine_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR yOut,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeCombineV2Info tilingData);

#endif //MOE_DISTRIBUTE_COMBINE_V2_ENTRY_H
