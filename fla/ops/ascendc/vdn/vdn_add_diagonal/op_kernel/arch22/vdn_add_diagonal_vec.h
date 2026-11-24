// SPDX-License-Identifier: BSD-3-Clause
#ifndef VDN_ADD_DIAGONAL_VEC_H
#define VDN_ADD_DIAGONAL_VEC_H

#include "kernel_operator.h"
#include "../vdn_add_diagonal_struct.h"

namespace Vdn {
template <uint32_t SectorWidth>
__aicore__ inline void AddDiagonal(__gm__ float *correction, __gm__ float *twoEye,
                                   const VdnAddDiagonalTilingData &tiling)
{
    using namespace AscendC;
    constexpr uint32_t headDim = 128;
    constexpr uint32_t sectors = headDim / SectorWidth;
    constexpr uint32_t tileSize = SectorWidth * SectorWidth;
    constexpr uint32_t floatsPerBlock = 32 / sizeof(float);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 2 * tileSize * sizeof(float));
    auto values = buffer.Get<float>();
    auto diagonal = values[tileSize];
    GlobalTensor<float> matrix, identity;
    matrix.SetGlobalBuffer(correction);
    identity.SetGlobalBuffer(twoEye);
    DataCopyParams read{SectorWidth, SectorWidth / floatsPerBlock, (headDim - SectorWidth) / floatsPerBlock, 0};
    DataCopyParams write{SectorWidth, SectorWidth / floatsPerBlock, 0, (headDim - SectorWidth) / floatsPerBlock};
    // Both diagonal tiles of 2I are identical. Load one per core, then reuse it.
    DataCopy(diagonal, identity, read);
    for (int64_t job = GetBlockIdx(); job < tiling.count * sectors; job += GetBlockNum()) {
        const int64_t offset = (job / sectors) * headDim * headDim + (job % sectors) * SectorWidth * (headDim + 1);
        DataCopy(values, matrix[offset], read);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Add(values, values, diagonal, tileSize);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(matrix[offset], values, write);
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace Vdn

#endif // VDN_ADD_DIAGONAL_VEC_H
