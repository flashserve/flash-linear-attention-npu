/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_A_ARCH35_PREPARE_H
#define CHUNK_KDA_BWD_A_ARCH35_PREPARE_H

#include "kernel_utils/vector/regbase.hpp"

namespace KDA {

using namespace AscendC;
using namespace AscendC::MicroAPI;

// Kernel A's AIV post stage consumes one FP32 CxC result half.  It applies
// the public scale and materializes the lower triangle (including diagonal)
// before the final dAqk write.  Rows outside validC are never published.
static __simd_vf__ inline void KdaBwdAPostDAqkA5(
    __ubuf__ float *data, float scale, uint16_t rowStart,
    uint16_t rows, uint16_t rowStride)
{
    RegTensor<float> value;
    RegTensor<float> zero;
    RegTensor<float> result;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zero, 0.0f, fullMask);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t validCols = rowStart + row + 1;
        uint32_t activeCount = validCols;
        MaskReg lowerMask = UpdateMask<float>(activeCount);
        LoadAlign(value, data + row * rowStride);
        Muls(value, value, scale, fullMask);
        Select(result, value, zero, lowerMask);
        StoreAlign(data + row * rowStride, result, fullMask);
    }
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_A_ARCH35_PREPARE_H
