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
 * \file moe_finalize_routing_v2_tiling_arch35.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_TILING_ARCH35_H_
#define MOE_FINALIZE_ROUTING_V2_TILING_ARCH35_H_

#include <cstdint>

struct MoeFinalizeRoutingV2RegbaseTilingData {
    int64_t row = 0;                        // BS
    int64_t e = 0;                          // E
    int64_t c = 0;                          // C
    int64_t h = 0;                          // H
    int64_t hAligned = 0;                   // H按block对齐后的大小
    int64_t k = 0;                          // K
    int64_t rowOfFormerBlock = 0;           // 整核处理的BS数
    int64_t rowOfTailBlock = 0;             // 尾核处理的BS数
    int64_t rowLoopOfFormerBlock = 0;       // 整核BS的循环次数
    int64_t rowLoopOfTailBlock = 0;         // 尾核BS的循环次数
    int64_t rowFactor = 0;                  // 每次循环处理的BS数
    int64_t tailRowFactorOfFormerBlock = 0; // 整核处理的BS尾块数
    int64_t tailRowFactorOfTailBlock = 0;   // 尾核处理的BS尾块数
    int64_t hLoop = 0;                      // H循环次数
    int64_t hFactor = 0;                    // 每次循环处理的H数
    int64_t tailHFactor = 0;                // H尾块数
    int64_t kLoop = 0;                      // K循环次数
    int64_t kFactor = 0;                    // 每次循环处理的K数
    int64_t tailKFactor = 0;                // K尾块数
    int64_t activeNum = 0;                  // 实际传入的expandedX的第一维的大小
};

#endif