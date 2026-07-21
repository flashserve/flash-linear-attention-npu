/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef BLOCK_EPILOGUE_GDN_FWDH_REGBASE_HPP
#define BLOCK_EPILOGUE_GDN_FWDH_REGBASE_HPP

#include "kernel_operator.h"

namespace Catlass::Epilogue::Block::detail {

static __simd_vf__ inline void ApplyRowScaleDualIssue(
    __ubuf__ float *matrix, __ubuf__ float *rowScale, uint32_t rowScaleOffset,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    RegTensor<float> matrixReg0;
    RegTensor<float> matrixReg1;
    RegTensor<float> scaleReg0;
    RegTensor<float> scaleReg1;
    MaskReg mask0;
    MaskReg mask1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + rowScaleOffset + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg1, rowScale + rowScaleOffset + row + 1);
        uint32_t remaining0 = cols;
        uint32_t remaining1 = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining0);
            mask1 = UpdateMask<float>(remaining1);
            LoadAlign(matrixReg0, matrix + static_cast<uint32_t>(row) * cols + col);
            LoadAlign(matrixReg1, matrix + static_cast<uint32_t>(row + 1) * cols + col);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask0);
            Mul(matrixReg1, matrixReg1, scaleReg1, mask1);
            StoreAlign(matrix + static_cast<uint32_t>(row) * cols + col, matrixReg0, mask0);
            StoreAlign(matrix + static_cast<uint32_t>(row + 1) * cols + col, matrixReg1, mask1);
        }
    }

    if (row < rows) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + rowScaleOffset + row);
        uint32_t remaining = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining);
            LoadAlign(matrixReg0, matrix + static_cast<uint32_t>(row) * cols + col);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask0);
            StoreAlign(matrix + static_cast<uint32_t>(row) * cols + col, matrixReg0, mask0);
        }
    }
}

} // namespace Catlass::Epilogue::Block::detail

#endif
