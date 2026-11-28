#pragma once
#include "mega_recurrent_kda_v2_cube.h"

namespace MegaRecurrentKdaV2 {
namespace Int8Qkv {
using namespace AscendC;
using Arch = Catlass::Arch::AtlasA2;
using Row = Catlass::layout::RowMajor;
using Nz = Catlass::layout::zN;
using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, int8_t, Row, int8_t, Nz, int32_t, Row>;
// Ping-pong into separate 32 KiB L0 banks for both INT8 operands.
using Policy = Catlass::Gemm::MmadPingpong<Arch, false, false, 1, false, 2, 2, 2, 2>;
using Mmad = Catlass::Gemm::Block::BlockMmadTla<Policy,
    tla::Shape<tla::_128, tla::_128, tla::_512>, tla::Shape<tla::_128, tla::_128, tla::_256>,
    int8_t, int8_t, int32_t, void, Copy>;

// out is private until Stage O. Each token lends H bytes to INT8 activations,
// one DMA block to its scale, and the remaining bytes to a wave of INT32 sums.
// When the unused tail of the QKV buffer fits all sums, one wave suffices.
static_assert(kArch22UbBytes == Arch::UB_SIZE, "Atlas A2 UB size must match host tiling");
template <typename T> CATLASS_DEVICE GlobalTensor<T> Global(GM_ADDR ptr)
{
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));
    return tensor;
}

// Distinct buffers at every level avoid in-place scratch reduction hazards.
CATLASS_DEVICE void RowMaxLevel(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                               uint32_t count)
{
    uint32_t full = count / 64, tail = count % 64;
    if (full) WholeReduceMax(dst, src, 64, full, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    if (tail) WholeReduceMax(dst[full], src[full * 64], tail, 1, 1, 1, 8,
                             ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
}

CATLASS_DEVICE void QuantizeRows(const V2Params &p, const LocalTensor<uint8_t> &ub, GM_ADDR scales)
{
    auto input = Global<bfloat16_t>(p.hiddenStates);
    auto codes = Global<int8_t>(p.out);
    auto tokenScale = Global<float>(scales);
    auto bf = ub.ReinterpretCast<bfloat16_t>();
    constexpr uint32_t maxChunk = kArch22UbBytes / 16;
    const uint32_t chunk = p.hidden < maxChunk ? p.hidden : maxChunk;
    auto x = ub[chunk * 2].ReinterpretCast<float>();
    auto work = ub[chunk * 6].ReinterpretCast<float>();
    auto reduced = ub[chunk * 10].ReinterpretCast<float>();
    auto ints = work.ReinterpretCast<int32_t>();
    auto fp16 = bf.ReinterpretCast<half>();
    auto bytes = reduced.ReinterpretCast<int8_t>();
    for (uint32_t row = GetBlockIdx(); row < p.tokens; row += p.aivNum) {
        const uint64_t offset = uint64_t(row) * p.hidden;
        float maximum = 0.f;
        for (uint32_t col = 0; col < p.hidden; col += chunk) {
            const uint32_t count = p.hidden - col < chunk ? p.hidden - col : chunk;
            DataCopy(bf, input[offset + col], count);
            PipeBarrier<PIPE_ALL>();
            Cast(x, bf, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Abs(work, x, count);
            PipeBarrier<PIPE_V>();
            auto firstMax = ub[chunk * 12].ReinterpretCast<float>();
            auto secondMax = ub[chunk * 14].ReinterpretCast<float>();
            const uint32_t firstCount = (count + 63) / 64;
            RowMaxLevel(firstMax, work, count);
            RowMaxLevel(secondMax, firstMax, firstCount);
            RowMaxLevel(reduced, secondMax, (firstCount + 63) / 64);
            PipeBarrier<PIPE_ALL>();
            const float localMax = reduced.GetValue(0);
            maximum = maximum < localMax ? localMax : maximum;
        }
        if (chunk < p.hidden) {
            reduced.SetValue(0, maximum);
            PipeBarrier<PIPE_ALL>();
        }
        // DynamicQuantV2 uses Vector Div for the multiplier and a separately
        // rounded multiplication by 1/127 for the returned scale.
        Duplicate(work, 127.0f, 8);
        PipeBarrier<PIPE_V>();
        Div(work, work, reduced, 1);
        PipeBarrier<PIPE_ALL>();
        const float multiplier = maximum == 0.0f ? 0.0f : work.GetValue(0);
        Duplicate(work, maximum * (1.0f / 127.0f), 8);
        PipeBarrier<PIPE_ALL>();
        DataCopy(tokenScale[uint64_t(row) * 8], work, 8);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t col = 0; col < p.hidden; col += chunk) {
            const uint32_t count = p.hidden - col < chunk ? p.hidden - col : chunk;
            if (chunk < p.hidden) {
                DataCopy(bf, input[offset + col], count);
                PipeBarrier<PIPE_ALL>();
                Cast(x, bf, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
            }
            Muls(x, x, multiplier, count);
            PipeBarrier<PIPE_V>();
            Cast(ints, x, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            Cast(x, ints, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Cast(fp16, x, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Cast(bytes, fp16, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_ALL>();
            DataCopy(codes[offset + col], bytes, count);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

template <bool splitSums, bool bypassQkvL2>
CATLASS_DEVICE void Multiply(Catlass::Arch::Resource<Arch> &resource, const V2Params &p, GM_ADDR sums,
                             uint32_t colBase, uint32_t width, GM_ADDR extraSums)
{
    Mmad multiply(resource);
    uint32_t panels = width / kNW;
    for (uint32_t job = GetBlockIdx(); job < p.tiles * panels; job += p.aicNum) {
        uint32_t row = job / panels * kTileRows, col = job % panels * kNW;
        uint32_t rows = p.tokens - row < kTileRows ? p.tokens - row : kTileRows;
        auto a = Global<int8_t>(p.out + uint64_t(row) * p.hidden);
        auto b = Global<int8_t>(p.qkvWeight + uint64_t(colBase + col) * p.hidden);
        if constexpr (bypassQkvL2) {
            b.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        GM_ADDR buffer = sums;
        uint32_t stride = width;
        uint32_t sumCol = col;
        if constexpr (splitSums) {
            const bool second = col >= Int8SumColumns(p.hidden);
            buffer = second ? extraSums : sums;
            stride = second ? width - Int8SumColumns(p.hidden) : Int8SumColumns(p.hidden);
            sumCol = second ? col - Int8SumColumns(p.hidden) : col;
        }
        auto c = Global<int32_t>(buffer + (uint64_t(row) * stride + sumCol) * sizeof(int32_t));
        auto ta = tla::MakeTensor(a, tla::MakeLayout<int8_t, Row>(rows, p.hidden), Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(b, tla::MakeLayout<int8_t, Nz>(p.hidden, kNW), Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(c, KdaGemm::MakeLayoutC(rows, kNW, stride), Catlass::Arch::PositionGM{});
        multiply(ta, tb, tc, Catlass::GemmCoord(rows, kNW, p.hidden));
    }
    PipeBarrier<PIPE_ALL>();
}

template <bool splitSums>
CATLASS_DEVICE void Dequantize(const V2Params &p, const LocalTensor<uint8_t> &ub, GM_ADDR scales,
                              GM_ADDR sums, uint32_t colBase, uint32_t width, GM_ADDR extraSums)
{
    auto tokenScale = Global<float>(scales);
    auto weightScale = Global<bfloat16_t>(p.weightScale);
    auto accum = Global<int32_t>(sums);
    auto result = Global<bfloat16_t>(p.wQkv);
    auto i32 = ub.ReinterpretCast<int32_t>();
    auto fp = ub[4 * width].ReinterpretCast<float>();
    auto channel = ub[8 * width].ReinterpretCast<float>();
    auto bf = ub[12 * width].ReinterpretCast<bfloat16_t>();
    DataCopy(bf, weightScale[colBase], width);
    PipeBarrier<PIPE_ALL>();
    Cast(channel, bf, RoundMode::CAST_NONE, width);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = GetBlockIdx(); row < p.tokens; row += p.aivNum) {
        if constexpr (splitSums) {
            const uint32_t rest = width - Int8SumColumns(p.hidden);
            auto second = Global<int32_t>(extraSums);
            DataCopy(i32, accum[uint64_t(row) * Int8SumColumns(p.hidden)], Int8SumColumns(p.hidden));
            DataCopy(i32[Int8SumColumns(p.hidden)], second[uint64_t(row) * rest], rest);
        } else {
            DataCopy(i32, accum[uint64_t(row) * width], width);
        }
        PipeBarrier<PIPE_ALL>();
        Cast(fp, i32, RoundMode::CAST_RINT, width);
        PipeBarrier<PIPE_V>();
        const float sx = tokenScale.GetValue(uint64_t(row) * 8);
        if (p.tiles == 1) {
            // The short-row native projection combines scales before rounding the dot to BF16.
            auto rowScale = i32.template ReinterpretCast<float>();
            Muls(rowScale, channel, sx, width);
            PipeBarrier<PIPE_V>();
            Mul(fp, fp, rowScale, width);
        } else {
            Mul(fp, fp, channel, width);
            PipeBarrier<PIPE_V>();
            Muls(fp, fp, sx, width);
        }
        PipeBarrier<PIPE_V>();
        Cast(bf, fp, RoundMode::CAST_RINT, width);
        PipeBarrier<PIPE_ALL>();
        DataCopy(result[uint64_t(row) * 3 * p.p + colBase], bf, width);
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace Int8Qkv

template <bool splitSums, bool bypassQkvL2, bool NarrowAux = false>
CATLASS_DEVICE void ProjectInt8Qkv(const V2Params &p, bool overlapAuxiliary, GM_ADDR extraSums)
{
    using namespace AscendC;
    GM_ADDR scales = p.out + uint64_t(p.tokens) * p.hidden;
    GM_ADDR sums = scales + uint64_t(p.tokens) * kInt8ScaleBytes;
    const uint32_t columns = 3 * p.p;
    uint32_t wave = columns;
    if constexpr (!splitSums) {
        const uint64_t qkvRows = uint64_t(p.slotsPR) * kTileRows;
        const uint64_t available = (qkvRows - p.tokens) * columns * sizeof(bfloat16_t);
        const uint64_t required = uint64_t(p.tokens) * columns * sizeof(int32_t);
        wave = Int8SumColumns(p.hidden);
        if (!wave) {
            // Before auxiliary projection the adjacent decay and gate slabs hold one INT32 panel.
            wave = kNW;
            sums = p.wDecayRaw;
        }
        const uint32_t ubColumns = Int8Qkv::Arch::UB_SIZE / kInt8DequantBytesPerColumn / kNW * kNW;
        wave = wave < ubColumns ? wave : ubColumns;
        if (columns > wave && columns <= Int8Qkv::Arch::UB_SIZE / kInt8DequantBytesPerColumn && required <= available) {
            sums = p.wQkv + uint64_t(p.tokens) * columns * sizeof(bfloat16_t);
            wave = columns;
        }
    }
    Catlass::Arch::Resource<Int8Qkv::Arch> resource;
    auto scratch = resource.ubBuf.GetBufferByByte<uint8_t>(0);
    if ASCEND_IS_AIC {
        if (overlapAuxiliary) {
            RunStageP<ProjectionPhase::Independent, NarrowAux>(resource, p, 0);
        }
    }
    if ASCEND_IS_AIV {
        Int8Qkv::QuantizeRows(p, scratch, scales);
    }
    SyncAll<false>();
    for (uint32_t col = 0; col < 3 * p.p; col += wave) {
        uint32_t left = 3 * p.p - col;
        uint32_t width = left < wave ? left : wave;
        if ASCEND_IS_AIC {
            Int8Qkv::Multiply<splitSums, bypassQkvL2>(resource, p, sums, col, width, extraSums);
        }
        SyncAll<false>();
        if ASCEND_IS_AIC {
            if (overlapAuxiliary && width == left) {
                RunStageP<ProjectionPhase::DecayB>(resource, p, 0);
            }
        }
        if ASCEND_IS_AIV {
            Int8Qkv::Dequantize<splitSums>(p, scratch, scales, sums, col, width, extraSums);
        }
        SyncAll<false>();
    }
}
} // namespace MegaRecurrentKdaV2
