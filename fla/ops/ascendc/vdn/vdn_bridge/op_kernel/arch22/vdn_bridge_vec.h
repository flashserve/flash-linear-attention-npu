// SPDX-License-Identifier: BSD-3-Clause
#ifndef VDN_BRIDGE_VEC_H
#define VDN_BRIDGE_VEC_H

#include "kernel_operator.h"
#include "../vdn_bridge_struct.h"
#include <type_traits>

namespace VdnBridgeKernel {
using namespace AscendC;

template <typename Out>
__aicore__ inline void Gather(GM_ADDR prefix, GM_ADDR suffix, GM_ADDR bounds, GM_ADDR logs, GM_ADDR out,
                              const VdnBridgeTilingData &tile)
{
    constexpr int32_t elements = TILE_ROWS * HEAD_DIM;
    constexpr int32_t fp32Lanes = 64;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, (2 * elements + 4 * HEAD_DIM) * sizeof(float) + elements * sizeof(Out));
    auto prefixTile = buf.Get<float>();
    auto suffixTile = prefixTile[elements];
    auto leftBoundary = prefixTile[2 * elements];
    auto rightBoundary = leftBoundary[HEAD_DIM];
    auto leftWeight = rightBoundary[HEAD_DIM];
    auto rightWeight = leftWeight[HEAD_DIM];
    auto converted = rightWeight[HEAD_DIM].template ReinterpretCast<Out>();

    GlobalTensor<float> prefixBank, suffixBank, logBank;
    GlobalTensor<Out> output;
    GlobalTensor<int64_t> indices;
    prefixBank.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(prefix));
    suffixBank.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(suffix));
    logBank.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logs));
    indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(bounds));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ Out *>(out));

    const int64_t frameStride = tile.heads * HEAD_DIM * HEAD_DIM;
    const int64_t jobs = tile.count * tile.heads;
    BinaryRepeatParams broadcast{1, 1, 1, 16, 16, 0};
    for (int64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum()) {
        const int64_t frame = tile.start + job / tile.heads;
        const int64_t head = job % tile.heads;
        const int64_t lo = indices.GetValue(2 * frame);
        const int64_t hi = indices.GetValue(2 * frame + 1);
        // Direct device calls can supply unchecked indices. Limit reads to the two state banks.
        const int64_t before = lo < 0 ? 0 : (lo > frame ? frame : lo);
        const int64_t after = (hi < frame ? frame : (hi >= tile.frames ? tile.frames - 1 : hi)) + 1;
        DataCopy(leftBoundary, logBank[(before * tile.heads + head) * HEAD_DIM], HEAD_DIM);
        DataCopy(rightBoundary, logBank[(after * tile.heads + head) * HEAD_DIM], HEAD_DIM);
        DataCopy(leftWeight, logBank[((frame + 1) * tile.heads + head) * HEAD_DIM], HEAD_DIM);
        DataCopy(rightWeight, logBank[(frame * tile.heads + head) * HEAD_DIM], HEAD_DIM);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Sub(leftWeight, leftWeight, leftBoundary, HEAD_DIM);
        Sub(rightWeight, rightBoundary, rightWeight, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Exp(leftWeight, leftWeight, HEAD_DIM);
        Exp(rightWeight, rightWeight, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < HEAD_DIM; row += TILE_ROWS) {
            const int64_t offset = head * HEAD_DIM * HEAD_DIM + row * HEAD_DIM;
            DataCopy(prefixTile, prefixBank[before * frameStride + offset], elements);
            DataCopy(suffixTile, suffixBank[after * frameStride + offset], elements);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            for (int32_t lane = 0; lane < HEAD_DIM; lane += fp32Lanes) {
                Mul(prefixTile[lane], prefixTile[lane], leftWeight[lane], fp32Lanes, TILE_ROWS, broadcast);
                Mul(suffixTile[lane], suffixTile[lane], rightWeight[lane], fp32Lanes, TILE_ROWS, broadcast);
            }
            PipeBarrier<PIPE_V>();
            Add(prefixTile, prefixTile, suffixTile, elements);
            if constexpr (!std::is_same_v<Out, float>) {
                PipeBarrier<PIPE_V>();
                Cast(converted, prefixTile, RoundMode::CAST_RINT, elements);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            const int64_t outOffset = job * HEAD_DIM * HEAD_DIM + row * HEAD_DIM;
            if constexpr (std::is_same_v<Out, float>) {
                DataCopy(output[outOffset], prefixTile, elements);
            } else {
                DataCopy(output[outOffset], converted, elements);
            }
            // Both DMA buffers and the cast scratch are reused by the next row tile.
            PipeBarrier<PIPE_ALL>();
        }
    }
}
} // namespace VdnBridgeKernel
#endif
