/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*!
 * \file recurrent_kda_tiling_a5.h
 * \brief Ascend 950 workspace planning for RecurrentKda.
 */

#ifndef RECURRENT_KDA_TILING_A5_H
#define RECURRENT_KDA_TILING_A5_H

#include <cstddef>
#include <cstdint>

namespace optiling {

struct RecurrentKdaA5Plan {
    uint32_t workspaceStride = 0;
    uint32_t cubeVecRows = 0;
    uint32_t cubeVecBufferNum = 1;
    size_t userWorkspaceSize = 0;
};

RecurrentKdaA5Plan BuildRecurrentKdaA5Plan(uint32_t blockDim, uint32_t dk, uint32_t vStep);

} // namespace optiling

#endif // RECURRENT_KDA_TILING_A5_H
