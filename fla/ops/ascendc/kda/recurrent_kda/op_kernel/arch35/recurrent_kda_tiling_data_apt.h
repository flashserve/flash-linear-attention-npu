/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*!
 * \file recurrent_kda_tiling_data_apt.h
 * \brief Ascend 950 tiling data for the mixed AIC/AIV RegBase kernel.
 */

#ifndef RECURRENT_KDA_TILING_DATA_APT_H
#define RECURRENT_KDA_TILING_DATA_APT_H

#include "../recurrent_kda_struct.h"
#include "kernel_tiling/kernel_tiling.h"

namespace RecurrentKda {

#pragma pack(push, 8)
struct alignas(8) RecurrentKdaTilingDataA5 {
    RecurrentKdaTilingData base;
    uint32_t aicCoreNum;
    uint32_t workspaceStride;
    uint32_t cubeVecRows;
    uint32_t cubeVecBufferNum;
};
#pragma pack(pop)

} // namespace RecurrentKda

#endif // RECURRENT_KDA_TILING_DATA_APT_H
