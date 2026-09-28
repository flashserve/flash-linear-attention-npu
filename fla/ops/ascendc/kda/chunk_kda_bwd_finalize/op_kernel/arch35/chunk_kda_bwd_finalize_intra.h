/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_INTRA_H
#define CHUNK_KDA_BWD_FINALIZE_INTRA_H

// Included inside namespace KDA after the register load/store helpers.
// A feature-wise power-of-two shift cancels between the two GEMM operands.

// The supported safe-gate contract supplies a chunk cumsum of nonpositive
// gate steps. Each feature is nonincreasing, so its endpoints are exactly
// the max/min previously obtained by scanning every row of this band.
template <bool FULL_TILE>
__simd_vf__ inline void FinalizeIntraCenterVF(
    __ubuf__ float *g, __ubuf__ float *center, uint16_t rows)
{
    const uint16_t bandRows = FULL_TILE ? static_cast<uint16_t>(KDA_FINALIZE_INTRA_ROWS) : rows;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t half = 0; half < 2; ++half) {
        RegTensor<float> first, last, value;
        LoadAlign(first, g + half * 64);
        LoadAlign(last, g + (bandRows - 1U) * 128 + half * 64);
        Add(value, first, last, mask);
        Muls(value, value, 0.5f, mask);
        StoreAlign(center + half * 64, value, mask);
    }
}

#endif
