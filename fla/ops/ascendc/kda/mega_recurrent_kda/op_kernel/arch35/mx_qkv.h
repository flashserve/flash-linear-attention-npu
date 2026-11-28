#pragma once
#include "projection_types.h"
#include "catlass/gemm/block/block_mmad_mx_tla.hpp"
#include "simt_api/device_functions.h"

namespace kda {
// BF16 magnitudes use one exponent and seven fraction bits. Round a shifted integer to nearest even.
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void QuantizeHidden(
    __gm__ const uint16_t *hidden, __gm__ uint8_t *codes, __gm__ uint8_t *scales, uint32_t rows,
    uint32_t hiddenWidth, uint32_t core, uint32_t cores, uint32_t algorithm)
{
    const uint32_t lane = AscendC::Simt::GetThreadIdx();
    // Distribute aligned groups across all Vector workers, including the last partial row wave.
    const uint64_t elements = uint64_t(rows) * hiddenWidth;
    const uint64_t chunks = (elements + 255) / 256;
    const uint64_t first = chunks * core / cores * 256;
    const uint64_t end = chunks * (core + 1) / cores * 256;
    for (uint64_t index = first + lane; index < end && index < elements; index += 1024) {
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

template <uint32_t maxRows>
CATLASS_DEVICE void ProjectMxQkvTile(Catlass::Arch::Resource<Arch> &resource, GM_ADDR codes, GM_ADDR scales,
                                 GM_ADDR weights, GM_ADDR weightScales, GM_ADDR projected, uint32_t rows,
                                 uint32_t channels, uint32_t tileRows, uint32_t K,
                                 uint32_t firstHead = 0, uint32_t headCount = 0)
{
    using F8 = float8_e4m3_t;
    using S8 = float8_e8m0_t;
    using NZ = Catlass::layout::zN;
    const uint32_t groups = K / 32;
    auto layoutA = tla::MakeLayout<F8, Row>(rows, K);
    auto layoutB = tla::MakeLayout<F8, NZ>(K, channels);
    auto layoutSA = tla::MakeMxScaleLayout<S8, Row, false>(rows, groups);
    auto layoutSB = tla::MakeMxScaleLayout<S8, Row, true>(groups, channels);
    auto layoutC = tla::MakeLayout<Bf, Row>(rows, channels);
    using MxCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<Arch, F8, Row, F8, NZ, S8, decltype(layoutSA),
                                                            S8, decltype(layoutSB), Bf, Row>;
    using MxTile = Catlass::Gemm::Tile::TileMmadTla<Arch, F8, typename MxCopy::LayoutTagL1A>;
    constexpr uint32_t tileCols = maxRows <= 128 ? 128 : 192;
    // Short row panels leave room for a deeper pair of K tiles and their MX scales in L1.
    constexpr uint32_t tileK = maxRows <= 128 ? 768 : 512;
    using MxBlock = Catlass::Gemm::Block::BlockMmadTla<Catlass::Gemm::MmadMx<Arch, true, 2>,
        tla::Shape<tla::Int<maxRows>, tla::Int<tileCols>, tla::Int<tileK>>,
        tla::Shape<tla::Int<maxRows>, tla::Int<tileCols>, tla::Int<(maxRows <= 128 ? 256 : 64)>>,
        F8, F8, Bf, void, MxCopy, MxTile>;
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
    const uint32_t regularPanels = (uint64_t(channels) + tileCols - 1) / tileCols;
    const uint64_t rowTiles = (uint64_t(rows) + tileRows - 1) / tileRows;
    const uint32_t workers = AscendC::GetBlockNum();
    const uint32_t nzBlocks = channels / 32;
    const bool distribute = rowTiles == 1 && regularPanels < workers && nzBlocks >= workers;
    const uint32_t panels = headCount || distribute ? workers : regularPanels;
    MxBlock op(resource);
    for (uint64_t tile = AscendC::GetBlockIdx(); tile < rowTiles * panels;
         tile += workers) {
        const uint32_t row = tile / panels * tileRows;
        const uint32_t panel = tile % panels;
        uint32_t col = distribute ? uint64_t(nzBlocks) * panel / panels * 32 : panel * tileCols;
        const uint32_t m = rows - row < tileRows ? rows - row : tileRows;
        uint32_t n = distribute ? uint64_t(nzBlocks) * (panel + 1) / panels * 32 - col
                                      : (channels - col < tileCols ? channels - col : tileCols);
        if (headCount) {
            const uint32_t plane = panel % 3, rank = panel / 3, team = (workers + 2 - plane) / 3;
            const uint32_t blocks = headCount * 4;
            const uint32_t start = uint64_t(blocks) * rank / team * 32;
            n = uint64_t(blocks) * (rank + 1) / team * 32 - start;
            col = plane * (channels / 3) + firstHead * 128 + start;
            if (!n) {
                continue;
            }
        }
        for (uint32_t subCol = col; subCol < col + n; subCol += tileCols) {
            const uint32_t subN = col + n - subCol < tileCols ? col + n - subCol : tileCols;
            auto aa = GetTile(a, tla::MakeCoord(row, 0), tla::MakeShape(m, K));
            auto bb = GetTile(b, tla::MakeCoord(0, subCol), tla::MakeShape(K, subN));
            auto cc = GetTile(c, tla::MakeCoord(row, subCol), tla::MakeShape(m, subN));
            auto as = GetTile(sa, tla::MakeCoord(row, 0), tla::MakeShape(m, groups));
            auto bs = GetTile(sb, tla::MakeCoord(0, subCol), tla::MakeShape(groups, subN));
            op(aa, bb, cc, Catlass::GemmCoord(m, subN, K), as, bs);
        }
    }
}
CATLASS_DEVICE void ProjectMxHeads(Catlass::Arch::Resource<Arch> &resource, GM_ADDR codes, GM_ADDR weights,
                                  GM_ADDR weightScales, GM_ADDR result, uint32_t rows, uint32_t heads,
                                  uint32_t hidden, uint32_t first, uint32_t count)
{
    GM_ADDR scales = codes + uint64_t(rows) * hidden;
    if (rows <= 128) {
        ProjectMxQkvTile<128>(resource, codes, scales, weights, weightScales, result, rows, heads * 384, 128,
                              hidden, first, count);
    } else {
        ProjectMxQkvTile<288>(resource, codes, scales, weights, weightScales, result, rows, heads * 384, 256,
                              hidden, first, count);
    }
}
CATLASS_DEVICE void ProjectMxQkv(Catlass::Arch::Resource<Arch> &resource, GM_ADDR codes, GM_ADDR scales,
                                 GM_ADDR weights, GM_ADDR weightScales, GM_ADDR projected, uint32_t rows,
                                 uint32_t channels, uint32_t hidden)
{
    const uint32_t panels = (channels + 127) / 128, workers = AscendC::GetBlockNum();
    const uint32_t largePanels = (channels + 191) / 192;
    const uint64_t defaultTiles = (uint64_t(rows) + 127) / 128;
    const uint64_t defaultWaves = (defaultTiles * panels + workers - 1) / workers;
    constexpr uint32_t maxRows = 288;
    const uint64_t minTiles = (uint64_t(rows) + maxRows - 1) / maxRows;
    const uint64_t minWaves = (minTiles * largePanels + workers - 1) / workers;
    if (minWaves < defaultWaves) {
        const uint64_t tiles = minWaves * workers / largePanels;
        const uint32_t tileRows = ((rows + tiles - 1) / tiles + 15) / 16 * 16;
        ProjectMxQkvTile<maxRows>(resource, codes, scales, weights, weightScales, projected, rows, channels, tileRows, hidden);
    } else {
        ProjectMxQkvTile<128>(resource, codes, scales, weights, weightScales, projected, rows, channels, 128, hidden);
    }
}
} // namespace kda
