/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file repare_wy_repr_bwd_full_tiling_data_apt.h
 * \brief
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
    int64_t kBeteVecRow = 0; // kBeta计算流程时vector单次处理的行数
    int64_t dkbVecRow = 0;   // dk计算流程时vector单次处理的行数
    int64_t dkbgVecRow = 0;  // dkbg计算流程时vector单次处理的行数
    int64_t dvbVecRow = 0;   // dvb计算流程时vector单次处理的行数
    int64_t kktVecRow = 0;   // kkt计算流程时vector单次处理的行数
    int64_t isVariable = 0;
    int64_t kBetaCVNum = 0; // vector计算时UB可存放kBeta的个数
    int64_t dkbCVNum = 0;   // vector计算时UB可存放dk的个数
    int64_t dkbgCVNum = 0;  // vector计算时UB可存放dkbg的个数
    int64_t dvbCVNum = 0;   // vector计算时UB可存放dvb的个数
    int64_t kktCVNum = 0;   // vector计算时UB可存放kkt的个数
};
#pragma pack(pop)

#endif // PREPARE_WY_REPR_BWD_FULL_COMMON_H
