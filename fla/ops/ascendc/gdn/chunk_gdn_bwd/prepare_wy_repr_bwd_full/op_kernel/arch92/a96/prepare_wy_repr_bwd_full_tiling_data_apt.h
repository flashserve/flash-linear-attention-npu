/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PREPARE_WY_REPR_BWD_FULL_TILING_DATA_H
#define PREPARE_WY_REPR_BWD_FULL_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct PrepareWyReprBwdFullTilingDataA5 {
    int64_t B = 0;
    int64_t HV = 0;
    int64_t HK = 0;
    int64_t T = 0;
    int64_t K = 0;
    int64_t V = 0;
    int64_t chunkNum = 0;
    int64_t chunkSize = 0;
    int64_t kBeteVecRow = 0;
    int64_t dkbVecRow = 0;
    int64_t dkbgVecRow = 0;
    int64_t dkGatherVecRow = 0;
    int64_t dvbVecRow = 0;
    int64_t kktVecRow = 0;
    int64_t isVariable = 0;
    int64_t kBetaCVNum = 0;
    int64_t dkbCVNum = 0;
    int64_t dkbgCVNum = 0;
    int64_t dvbCVNum = 0;
    int64_t kktCVNum = 0;
};
#pragma pack(pop)

#endif
