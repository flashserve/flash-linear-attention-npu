#pragma once
#include "projection_types.h"
#include "catlass/gemm/block/block_mmad_mx_tla.hpp"
#include "simt_api/device_functions.h"

namespace kda {
// BF16 magnitudes use one exponent and seven fraction bits. Round a shifted integer to nearest even.
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void QuantizeHidden(
    __gm__ const uint16_t *hidden, __gm__ uint8_t *codes, __gm__ uint8_t *scales, uint32_t rows,
    uint32_t core, uint32_t cores, uint32_t algorithm)
{
    const uint32_t lane = AscendC::Simt::GetThreadIdx();
    // Distribute aligned groups across all Vector workers, including the last partial row wave.
    const uint64_t chunks = uint64_t(rows) * (7168 / 256);
    const uint64_t first = chunks * core / cores * 256;
    const uint64_t end = chunks * (core + 1) / cores * 256;
    for (uint64_t index = first + lane; index < end; index += 1024) {
        const uint32_t raw = hidden[index], magnitude = raw & 0x7fff;
        int32_t maximum = static_cast<int32_t>(magnitude);
        for (int mask = 16; mask; mask >>= 1) {
            int32_t other = asc_shfl_xor(maximum, mask, 32);
            maximum = maximum < other ? other : maximum;
        }
        int scale = (maximum >> 7) - 8 + int(algorithm && (maximum & 127) > 96);
        scale = maximum >= 0x7f80 ? 255 : (scale < 0 ? 0 : scale);
        if ((lane & 31) == 0) {
            scales[index / 32] = scale;
        }
        int exponent = magnitude >> 7;
        int mantissa = (magnitude & 127) + (exponent ? 128 : 0);
        if (!exponent && mantissa) {
            exponent = 1;
            while (mantissa < 128) {
                mantissa <<= 1;
                --exponent;
            }
        }
        const int fp8Exp = exponent - scale + 7;
        const int shift = fp8Exp > 0 ? 4 : 5 - fp8Exp;
        int value = 0;
        if (mantissa && shift < 9 && scale != 255 && (scale || algorithm)) {
            const int whole = mantissa >> shift;
            const int rest = mantissa - (whole << shift);
            const int halfStep = 1 << (shift - 1);
            const int rounded = whole + int(rest > halfStep || (rest == halfStep && (whole & 1)));
            value = (fp8Exp > 0 ? (fp8Exp - 1) * 8 : 0) + rounded;
        }
        codes[index] = (raw >> 8 & 128) | (value > 126 ? 126 : value);
    }
}

CATLASS_DEVICE void ProjectMxQkv(Catlass::Arch::Resource<Arch> &resource, GM_ADDR codes, GM_ADDR scales,
                                 GM_ADDR weights, GM_ADDR weightScales, GM_ADDR projected, uint32_t rows,
                                 uint32_t channels)
{
    using F8 = float8_e4m3_t;
    using S8 = float8_e8m0_t;
    using NZ = Catlass::layout::zN;
    constexpr uint32_t K = 7168, groups = K / 32;
    auto layoutA = tla::MakeLayout<F8, Row>(rows, K);
    auto layoutB = tla::MakeLayout<F8, NZ>(K, channels);
    auto layoutSA = tla::MakeMxScaleLayout<S8, Row, false>(rows, groups);
    auto layoutSB = tla::MakeMxScaleLayout<S8, Row, true>(groups, channels);
    auto layoutC = tla::MakeLayout<Bf, Row>(rows, channels);
    using MxCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<Arch, F8, Row, F8, NZ, S8, decltype(layoutSA),
                                                            S8, decltype(layoutSB), Bf, Row>;
    using MxTile = Catlass::Gemm::Tile::TileMmadTla<Arch, F8, typename MxCopy::LayoutTagL1A>;
    constexpr uint32_t tileRows = 128, tileCols = 192;
    using MxBlock = Catlass::Gemm::Block::BlockMmadTla<Catlass::Gemm::MmadMx<Arch, true, 2>,
        tla::Shape<tla::Int<tileRows>, tla::Int<tileCols>, tla::Int<512>>,
        tla::Shape<tla::Int<tileRows>, tla::Int<tileCols>, tla::Int<64>>, F8, F8, Bf, void, MxCopy, MxTile>;
    AscendC::GlobalTensor<F8> x, w;
    AscendC::GlobalTensor<S8> sx, sw;
    AscendC::GlobalTensor<Bf> y;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ F8 *>(codes));
    w.SetGlobalBuffer(reinterpret_cast<__gm__ F8 *>(weights));
    sx.SetGlobalBuffer(reinterpret_cast<__gm__ S8 *>(scales));
    sw.SetGlobalBuffer(reinterpret_cast<__gm__ S8 *>(weightScales));
    y.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(projected));
    if (rows <= tileRows) {
        w.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        sw.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    auto a = tla::MakeTensor(x, layoutA, Catlass::Arch::PositionGM{});
    auto b = tla::MakeTensor(w, layoutB, Catlass::Arch::PositionGM{});
    auto sa = tla::MakeTensor(sx, layoutSA, Catlass::Arch::PositionGM{});
    auto sb = tla::MakeTensor(sw, layoutSB, Catlass::Arch::PositionGM{});
    auto c = tla::MakeTensor(y, layoutC, Catlass::Arch::PositionGM{});
    const uint32_t panels = (uint64_t(channels) + tileCols - 1) / tileCols;
    const uint64_t rowTiles = (uint64_t(rows) + tileRows - 1) / tileRows;
    MxBlock op(resource);
    for (uint64_t tile = AscendC::GetBlockIdx(); tile < rowTiles * panels;
         tile += AscendC::GetBlockNum()) {
        const uint32_t row = tile / panels * tileRows, col = tile % panels * tileCols;
        const uint32_t m = rows - row < tileRows ? rows - row : tileRows;
        const uint32_t n = channels - col < tileCols ? channels - col : tileCols;
        auto aa = GetTile(a, tla::MakeCoord(row, 0), tla::MakeShape(m, K));
        auto bb = GetTile(b, tla::MakeCoord(0, col), tla::MakeShape(K, n));
        auto cc = GetTile(c, tla::MakeCoord(row, col), tla::MakeShape(m, n));
        auto as = GetTile(sa, tla::MakeCoord(row, 0), tla::MakeShape(m, groups));
        auto bs = GetTile(sb, tla::MakeCoord(0, col), tla::MakeShape(groups, n));
        op(aa, bb, cc, Catlass::GemmCoord(m, n, K), as, bs);
    }
}
} // namespace kda
