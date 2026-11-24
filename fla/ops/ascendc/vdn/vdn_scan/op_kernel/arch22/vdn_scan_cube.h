// SPDX-License-Identifier: BSD-3-Clause
#ifndef VDN_SCAN_CUBE_H
#define VDN_SCAN_CUBE_H

#include "kernel_operator.h"
#include "../vdn_scan_struct.h"

#if defined(__DAV_C220_CUBE__)
#define MEMORY_BASE
#include <pto/pto-inst.hpp>
#undef MEMORY_BASE

namespace Vdn {
template <uint32_t HeadDim>
__aicore__ inline void Scan(__gm__ float *transition, __gm__ float *prefix, __gm__ float *suffix,
                           const VdnScanTilingData &tiling)
{
    using namespace pto;
    using Matrix = pto::GlobalTensor<float, pto::Shape<1, 1, 1, HeadDim, HeadDim>,
                                pto::Stride<HeadDim * HeadDim, HeadDim * HeadDim, HeadDim * HeadDim, HeadDim, 1>,
                                pto::Layout::ND>;
    using L1Matrix = Tile<TileType::Mat, float, HeadDim, HeadDim, BLayout::ColMajor,
                          HeadDim, HeadDim, SLayout::RowMajor>;
    AscendC::SetHF32Mode(false);
    L1Matrix state, matrix;
    TileLeft<float, HeadDim, HeadDim> lhs;
    TileRight<float, HeadDim, HeadDim> rhs;
    TileAcc<float, HeadDim, HeadDim> product;
    TASSIGN(state, 0);
    TASSIGN(matrix, HeadDim * HeadDim * sizeof(float));
    TASSIGN(lhs, 0);
    TASSIGN(rhs, 0);
    TASSIGN(product, 0);

    const int64_t stride = int64_t(tiling.heads) * HeadDim * HeadDim;
    // A bank slice has one writer. Its original contents are the injection
    // for this frame, so FIXPIPE can accumulate without another GM round-trip.
    for (int32_t job = AscendC::GetBlockIdx(); job < 2 * tiling.heads; job += AscendC::GetBlockNum()) {
        const bool reverse = job >= tiling.heads;
        const int64_t offset = int64_t(job % tiling.heads) * HeadDim * HeadDim;
        auto *bank = reverse ? suffix : prefix;
        for (int32_t i = 0; i < tiling.frames; ++i) {
            const int32_t frame = reverse ? tiling.frames - 1 - i : i;
            Matrix src(bank + (frame + int32_t(reverse)) * stride + offset);
            Matrix mat(transition + frame * stride + offset);
            Matrix dst(bank + (frame + 1 - int32_t(reverse)) * stride + offset);
            TLOAD(matrix, mat);
            TLOAD(state, src);
            pipe_barrier(PIPE_ALL);
            TMOV(lhs, state);
            TMOV(rhs, matrix);
            pipe_barrier(PIPE_ALL);
            TMATMUL(product, lhs, rhs);
            pipe_barrier(PIPE_ALL);
            AscendC::SetAtomicAdd<float>();
            TSTORE(dst, product);
            pipe_barrier(PIPE_ALL);
            AscendC::SetAtomicNone();
            pipe_barrier(PIPE_ALL);
        }
    }
}
} // namespace Vdn
#endif

#endif // VDN_SCAN_CUBE_H
