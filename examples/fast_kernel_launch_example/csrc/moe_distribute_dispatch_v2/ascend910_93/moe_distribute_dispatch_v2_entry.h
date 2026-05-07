/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_ENTRY_H
#define MOE_DISTRIBUTE_DISPATCH_V2_ENTRY_H
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"

struct MoeDistributeDispatchV2SkArgs {
    int32_t tilingKey;
    GM_ADDR x;
    GM_ADDR expertIds;
    GM_ADDR scales;
    GM_ADDR xActiveMask;
    GM_ADDR expertScales; 
    GM_ADDR performanceInfo;
    GM_ADDR expandXOut;
    GM_ADDR dynamicScalesOut; 
    GM_ADDR assistInfoOut;
    GM_ADDR expertTokenNumsOut;
    GM_ADDR epSendCountsOut; 
    GM_ADDR expandScalesOut;
    GM_ADDR workspaceGM;
    GM_ADDR mc2Context;
    MoeDistributeDispatchV2Info tilingData;
};


// <<<>>>调用函数声明
void moe_distribute_dispatch_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR performanceInfo, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context,
    MoeDistributeDispatchV2Info tilingData);

#endif //MOE_DISTRIBUTE_DISPATCH_V2_ENTRY_H
