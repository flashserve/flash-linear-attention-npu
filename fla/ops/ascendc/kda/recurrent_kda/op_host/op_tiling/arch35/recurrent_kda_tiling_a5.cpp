/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*!
 * \file recurrent_kda_tiling_a5.cpp
 * \brief Ascend 950 workspace planning for RecurrentKda.
 */

#include "recurrent_kda_tiling_a5.h"

namespace optiling {
namespace {

constexpr uint32_t A5_BF16_BYTES = 2;
constexpr uint32_t A5_K_ALIGN = 16;
constexpr uint32_t A5_WORKSPACE_ALIGN = 512;
constexpr uint32_t A5_CUBE_M = 16;

uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

RecurrentKdaA5Plan BuildRecurrentKdaA5Plan(uint32_t blockDim, uint32_t dk, uint32_t vStep)
{
    RecurrentKdaA5Plan plan;
    const uint32_t alignedK = AlignUp(dk, A5_K_ALIGN);
    const uint32_t stateBytes = 2 * vStep * alignedK * A5_BF16_BYTES;
    const uint32_t qkBytes = A5_CUBE_M * alignedK * A5_BF16_BYTES;
    plan.workspaceStride = AlignUp(stateBytes + qkBytes, A5_WORKSPACE_ALIGN);
    plan.cubeVecRows = vStep;
    plan.userWorkspaceSize = static_cast<size_t>(blockDim) * plan.workspaceStride;
    return plan;
}

} // namespace optiling
