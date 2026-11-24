#pragma once
#include "short_decay_projection.h"
#include "catlass/arch/cross_core_sync.hpp"

namespace kda {
// One BF16 tile in the paired AIV's UB. Beta has one value per row and a padded stride of 16.
constexpr uint32_t kSlotBytes = 20992;
constexpr uint32_t kSliceRows = 16;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kGlobalRows = 64;
constexpr uint32_t kHiddenWidth = 7168;
constexpr uint32_t kFreeSlotFlagBase = 2;
constexpr uint32_t kReadySlotFlagBase = 4;
enum class Plane : uint32_t {
    Q,
    K,
    V,
    Decay,
    Gate,
    Beta
};

struct MegaParams {
    GM_ADDR hiddenStates;
    GM_ADDR qkvWeight;
    GM_ADDR decayAWeight;
    GM_ADDR decayBWeight;
    GM_ADDR gateWeight;
    GM_ADDR betaWeight;
    GM_ADDR decayRing;
    uint32_t rows, heads;
    RecurrenceArgs recurrence{};
};

// The Vector side publishes the second head group while Cube is computing the first.
using OutputCopy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, Bf, Row, Bf, Col, Bf, Row>;
struct StreamingOutputCopy : OutputCopy {
    template <class Source>
    struct CopyGmToL1A : OutputCopy::template CopyGmToL1A<Source> {
        template <class Destination, class Tile>
        CATLASS_DEVICE void operator()(Destination &dst, Tile &src)
        {
            uint32_t width = tla::get<0>(src.layout().stride());
            uint32_t second = (width / 128 / 2) * 128;
            if (uint32_t(tla::get<1>(src.coord())) == second) {
                AscendC::SyncAll<false>();
            }
            OutputCopy::template CopyGmToL1A<Source>::operator()(dst, src);
        }
    };
};
template <bool SpansTile, bool AllowColumnMajor = SpansTile>
class MegaKernel {
public:
    using Params = MegaParams;
    CATLASS_DEVICE void operator()(const Params &p)
    {
        if constexpr (!SpansTile) {
            RunDoubleBuffered(p);
            return;
        }
        if (p.heads >= 2 && TryRunSequenceFrames(p, AscendC::GetBlockIdx(), AscendC::GetBlockNum())) {
            return;
        }
        RunTilePipeline(p);
    }

private:
    Catlass::Arch::Resource<Arch> resource_;

    CATLASS_DEVICE void RunTilePipeline(const Params &p)
    {
        uint32_t group = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            if (AscendC::GetSubBlockIdx() != 0) {
                return;
            }
            group /= AscendC::GetTaskRation();
            PublishFreeSlots();
        }
        uint32_t groups = AscendC::GetBlockNum();
        Recurrence rec;
        const auto &a = p.recurrence;
        const uint32_t capacity = ((32768 + uint64_t(p.heads) * 98304) / (256 + uint64_t(p.heads) * 256)) / 16 * 16;
        bool wholeSequences = a.batch >= 2 && p.rows >= 2 * kGlobalRows;
        if (a.sequence) {
            wholeSequences = wholeSequences && a.sequence <= capacity / 2;
        } else if (a.plan) {
            wholeSequences =
                wholeSequences && uint32_t(reinterpret_cast<__gm__ const int32_t *>(a.plan)[1]) <= capacity / 2;
        } else {
            for (uint32_t b = 0; wholeSequences && b < a.batch; ++b) {
                wholeSequences = rec.prefix(a, b + 1) - rec.prefix(a, b) <= capacity / 2;
            }
        }
        uint32_t phase = 0, firstSequence = 0;
        GM_ADDR gated = p.decayRing + (wholeSequences ? capacity * 256 : 32768);
        GM_ADDR carry = wholeSequences ? nullptr : gated + uint64_t(p.heads) * 32768;
        for (uint32_t first = 0; first < p.rows;) {
            uint32_t rows = p.rows - first < kGlobalRows ? p.rows - first : kGlobalRows;
            uint32_t endSequence = firstSequence;
            if (wholeSequences) {
                endSequence = p.rows - first <= capacity ? a.batch : rec.owner(a, first + capacity);
                rows = rec.prefix(a, endSequence) - first;
            }
            ProjectSharedDecay(p, first, rows, group, groups, phase);
            SyncCores();
            const uint32_t cuts = wholeSequences ? endSequence - firstSequence : 4;
            for (uint64_t work = group; work < uint64_t(cuts) * p.heads; work += groups) {
                uint32_t part = work / p.heads, head = work % p.heads;
                uint32_t offset = wholeSequences ? rec.prefix(a, firstSequence + part) - first :
                                                   (part ? FindSequenceCut(p, first, rows, part * 16) : 0);
                uint32_t end = wholeSequences ?
                                   rec.prefix(a, firstSequence + part + 1) - first :
                                   (part + 1 < cuts ? FindSequenceCut(p, first, rows, (part + 1) * 16) : rows);
                for (uint32_t r = offset; r < end; r += kSliceRows) {
                    uint32_t valid = end - r < kSliceRows ? end - r : kSliceRows;
                    if ASCEND_IS_AIC {
                        WaitFree(phase);
                        uint32_t col = head * kHeadDim;
                        uint32_t width = p.heads * kHeadDim;
                        for (uint32_t i = 0; i < 2; ++i) {
                            Matmul(p.hiddenStates, p.qkvWeight, first + r, col + i * width, valid, kHeadDim,
                                   kHiddenWidth, LocalTile(phase, static_cast<Plane>(i)));
                        }
                    }
                    uint32_t col = head * kHeadDim;
                    if ASCEND_IS_AIC {
                        Matmul(p.hiddenStates, p.qkvWeight, first + r, col + 2 * p.heads * kHeadDim, valid, kHeadDim,
                               kHiddenWidth, LocalTile(phase, Plane::V));
                        Matmul(p.decayRing, p.decayBWeight, r, col, valid, kHeadDim, 128,
                               LocalTile(phase, Plane::Decay));
                        Matmul(p.hiddenStates, p.gateWeight, first + r, col, valid, kHeadDim, kHiddenWidth,
                               LocalTile(phase, Plane::Gate));
                        Matmul(p.hiddenStates, p.betaWeight, first + r, head, valid, 1, kHiddenWidth,
                               LocalTile(phase, Plane::Beta));
                        NotifyReady(phase);
                    }
                    if ASCEND_IS_AIV {
                        WaitReady(phase);
                        Recurrence{}.consume<AllowColumnMajor>(resource_, p.recurrence, gated, carry, p.heads, head,
                                                               first + r, r, valid, LocalTile(phase, Plane::Q));
                        NotifyFree(phase);
                    }
                    phase ^= 1;
                }
            }
            SyncCores();
            if (wholeSequences) {
                if ASCEND_IS_AIC {
                    for (uint32_t panel = group; panel < kHiddenWidth / 256; panel += groups) {
                        MatmulToGm<256, 256, 64, 128>(gated, p.recurrence.outWeight, p.recurrence.output, 0,
                                                      panel * 256, first, panel * 256, kHiddenWidth, rows,
                                                      p.heads * 128, 256);
                    }
                }
            } else {
                ProjectOutputTiles(p, gated, first, rows, group, groups, phase);
            }
            SyncCores();
            first += rows;
            firstSequence = endSequence;
        }
        if ASCEND_IS_AIC {
            WaitFree(0);
            WaitFree(1);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE AscendC::LocalTensor<Bf> LocalTile(uint32_t phase, Plane plane)
    {
        uint32_t offset = phase * kSlotBytes + static_cast<uint32_t>(plane) * kSliceRows * kHeadDim * sizeof(Bf);
        return resource_.ubBuf.template GetBufferByByte<Bf>(offset);
    }
    template <uint32_t TileM = 16>
    CATLASS_DEVICE void Matmul(GM_ADDR aPtr, GM_ADDR wPtr, uint32_t row, uint32_t col, uint32_t m, uint32_t n,
                               uint32_t k, AscendC::LocalTensor<Bf> buffer)
    {
        AscendC::GlobalTensor<Bf> a, w;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(aPtr) + uint64_t(row) * k);
        w.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(wPtr) + uint64_t(col) * k);
        auto x = tla::MakeTensor(a, tla::MakeLayout<Bf, Row>(m, k), Catlass::Arch::PositionGM{});
        auto weight = tla::MakeTensor(w, tla::MakeLayout<Bf, Col>(k, n), Catlass::Arch::PositionGM{});
        uint32_t stride = (n + 15) / 16 * 16;
        auto ub = tla::MakeTensor(buffer, tla::MakeLayout<Bf, Row>(m, stride), Catlass::Arch::PositionUB{});
        using WideProjection =
            Catlass::Gemm::Block::BlockMmadTla<Policy, tla::Shape<tla::Int<TileM>, tla::Int<128>, tla::Int<512>>,
                                               tla::Shape<tla::Int<TileM>, tla::Int<128>, tla::Int<128>>, Bf, Bf, Bf,
                                               void, Copy, TileMmad>;
        WideProjection op(resource_);
        op(x, weight, ub, Catlass::GemmCoord(m, n, k));
    }
    // Lead with 64 rows when the final 128-row wave would contain at most 64.
    // This releases the first group of sequences to Vector one projection earlier.
    struct Waves {
        uint32_t firstRows, firstCapacity, storageRows, count, span, decayRows;
        CATLASS_DEVICE uint32_t offset(uint32_t i) const
        {
            return i == 0 ? 0 : firstRows + (i - 1) * span;
        }
    };
    CATLASS_DEVICE Waves PlanWaves(uint32_t rows)
    {
        uint32_t span = rows > 128 && rows <= 256 ? 256 : 128;
        uint32_t tail = rows % span;
        uint32_t first = rows <= span ? rows : ((tail && tail <= 64) ? 64 : span);
        uint32_t count = 1 + (rows - first + span - 1) / span;
        uint32_t firstCap = count > 2 ? span : (first + 15) / 16 * 16;
        uint32_t left = rows - first;
        uint32_t secondCap = left >= span ? span : (left + 15) / 16 * 16;
        return {first, firstCap, firstCap + secondCap, count, span, span == 256 ? firstCap : 128};
    }
    CATLASS_DEVICE void ConsumeStagedProjections(const Params &p, uint32_t first, uint32_t rows, uint32_t worker,
                                                 uint32_t workers, uint32_t firstHead = 0, uint32_t count = 0,
                                                 bool reuseQ = false, GM_ADDR framePlanes = nullptr)
    {
        uint32_t width = p.heads * kHeadDim;
        uint32_t span = p.recurrence.staged == 3 ? 128 : 64;
        uint32_t slot = p.recurrence.staged >= 2 ? (first / span) % 2 : 0;
        uint32_t betaStride = p.recurrence.staged >= 2 ? (p.heads + 15) / 16 * 16 : p.heads;
        GM_ADDR gatedBase = p.decayRing + (p.recurrence.staged >= 2 ? 16384 : 32768);
        uint32_t planeRows = 64, totalGatedRows = 128;
        if (reuseQ) {
            slot = 0;
            planeRows = (rows + 15) / 16 * 16;
            gatedBase = p.decayRing + planeRows * 256;
        } else if (p.recurrence.staged == 3) {
            auto shape = PlanWaves(p.rows);
            slot = first ? (1 + (first - shape.firstRows) / shape.span) % 2 : 0;
            span = shape.firstCapacity;
            totalGatedRows = shape.storageRows;
            planeRows = slot ? shape.storageRows - span : span;
            gatedBase = p.decayRing + shape.decayRows * 256;
        }
        GM_ADDR gated = gatedBase + uint64_t(slot) * span * width * 2;
        GM_ADDR planes = gatedBase + uint64_t(totalGatedRows) * width * 2 +
                         uint64_t(slot) * span * (5 * uint64_t(width) + betaStride) * 2;
        if (SpansTile && framePlanes) {
            planes = framePlanes;
        } else if (reuseQ) {
            planes = gatedBase;
        }
        Recurrence rec;
        uint32_t firstSeq = rec.owner(p.recurrence, first);
        uint32_t lastSeq = rec.owner(p.recurrence, first + rows - 1);
        if (!count) {
            count = p.heads;
        }
        for (uint64_t task = worker; task < uint64_t(lastSeq - firstSeq + 1) * count; task += workers) {
            uint32_t head = firstHead + task % count, b = firstSeq + task / count;
            uint32_t start = rec.prefix(p.recurrence, b), end = rec.prefix(p.recurrence, b + 1);
            for (uint32_t row = start; row < end; row += kSliceRows) {
                uint32_t m = end - row < kSliceRows ? end - row : kSliceRows;
                for (uint32_t i = 0; i < 6; ++i) {
                    AscendC::GlobalTensor<Bf> src;
                    src.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(planes) + uint64_t(i) * planeRows * width);
                    auto tile = LocalTile(0, static_cast<Plane>(i));
                    uint64_t row0 = row - first;
                    if (i == 5) {
                        AscendC::DataCopyPad(
                            tile, src[row0 * betaStride + head],
                            AscendC::DataCopyExtParams{static_cast<uint16_t>(m), 2, (betaStride - 1) * 2, 0, 0},
                            AscendC::DataCopyPadExtParams<Bf>{});
                    } else {
                        AscendC::DataCopyPad(
                            tile, src[row0 * width + head * 128],
                            AscendC::DataCopyExtParams{static_cast<uint16_t>(m), 256, (width - 128) * 2, 0, 0},
                            AscendC::DataCopyPadExtParams<Bf>{});
                    }
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                rec.consume<AllowColumnMajor>(resource_, p.recurrence, gated, nullptr, p.heads, head, row, row - first,
                                              m, LocalTile(0, Plane::Q));
            }
        }
    }
    CATLASS_DEVICE void ProjectSharedDecay(const Params &p, uint32_t first, uint32_t rows, uint32_t group,
                                           uint32_t groups, uint32_t &phase)
    {
        for (uint32_t start = group * kSliceRows; start < rows; start += groups * kSliceRows) {
            uint32_t valid = rows - start < kSliceRows ? rows - start : kSliceRows;
            auto local = LocalTile(phase, Plane::Q);
            if ASCEND_IS_AIC {
                WaitFree(phase);
            }
            if ASCEND_IS_AIC {
                Matmul(p.hiddenStates, p.decayAWeight, first + start, 0, valid, 128, kHiddenWidth, local);
            }
            if ASCEND_IS_AIC {
                NotifyReady(phase);
            }
            if ASCEND_IS_AIV {
                WaitReady(phase);
                StoreTile(p.decayRing, start, 0, kHeadDim, valid, kHeadDim, local);
                NotifyFree(phase);
            }
            phase ^= 1;
        }
    }
    CATLASS_DEVICE void ProjectOutputTiles(const Params &p, GM_ADDR gated, uint32_t first, uint32_t rows,
                                           uint32_t group, uint32_t groups, uint32_t &phase)
    {
        constexpr uint32_t cols = kHiddenWidth / kHeadDim;
        constexpr uint32_t outputRows = 64;
        uint32_t slices = (rows + outputRows - 1) / outputRows;
        for (uint32_t task = group; task < slices * cols; task += groups) {
            uint32_t r = task / cols * outputRows, col = task % cols * kHeadDim;
            uint32_t valid = rows - r < outputRows ? rows - r : outputRows;
            auto tile = LocalTile(phase, Plane::Q);
            if ASCEND_IS_AIC {
                WaitFree(phase);
                Matmul<64>(gated, p.recurrence.outWeight, r, col, valid, kHeadDim, p.heads * kHeadDim, tile);
                NotifyReady(phase);
            }
            if ASCEND_IS_AIV {
                WaitReady(phase);
                StoreTile(p.recurrence.output, first + r, col, kHiddenWidth, valid, kHeadDim, tile);
                NotifyFree(phase);
            }
            phase ^= 1;
        }
    }
    template <int TileM, int TileN, int L0K>
    CATLASS_DEVICE void StreamOutput(const Params &p, GM_ADDR gated, uint32_t id, uint32_t groups)
    {
        constexpr uint32_t panels = kHiddenWidth / TileN;
        const uint32_t width = p.heads * kHeadDim;
        if (id < panels) {
            MatmulToGm<TileN, 128, L0K, TileM, true>(gated, p.recurrence.outWeight, p.recurrence.output, 0, id * TileN,
                                                     0, id * TileN, kHiddenWidth, p.rows, width, TileN);
        } else {
            AscendC::SyncAll<false>();
        }
        for (uint32_t j = id + groups; j < panels; j += groups) {
            MatmulToGm<TileN, 256, L0K, TileM>(gated, p.recurrence.outWeight, p.recurrence.output, 0, j * TileN, 0,
                                               j * TileN, kHiddenWidth, p.rows, width, TileN);
        }
    }
    // Each worker produces and consumes the same row range, so decay-B need not wait for Q/K/V.
    CATLASS_DEVICE void ProjectDecayRows(const Params &p, GM_ADDR output, uint32_t worker, uint32_t workers)
    {
        const uint32_t blocks = (p.rows + 15) / 16;
        const uint32_t first = 16 * (blocks * worker / workers);
        const uint32_t boundary = 16 * (blocks * (worker + 1) / workers);
        const uint32_t end = boundary < p.rows ? boundary : p.rows;
        if (first >= end) {
            return;
        }
        const uint32_t width = p.heads * kHeadDim;
        MatmulToGm<128, 384, 64, 80>(p.hiddenStates, p.decayAWeight, p.decayRing, first, 0, first, 0, 128, end - first,
                                     kHiddenWidth);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t col = 0; col < width; col += 256) {
            const uint32_t count = width - col < 256 ? width - col : 256;
            MatmulToGm<256, 128, 64, 80>(p.decayRing, p.decayBWeight, output, first, col, first, col, width,
                                         end - first, 128, count);
        }
    }
    CATLASS_DEVICE void RunHeadPipeline(const Params &p, const Waves &shape, uint32_t id, uint32_t groups,
                                        bool decayAReady, uint32_t globalBase = 0, bool reuseQ = false,
                                        GM_ADDR framePlanes = nullptr)
    {
        const uint32_t width = p.heads * 128, betaStride = (p.heads + 15) / 16 * 16, planeRows = shape.firstCapacity;
        GM_ADDR gated = p.decayRing + shape.decayRows * 256;
        GM_ADDR planes = reuseQ ? gated : gated + uint64_t(shape.storageRows) * width * 2;
        if (SpansTile && framePlanes) {
            planes = framePlanes;
        }
        constexpr uint32_t waveLimit = 2;
        const uint32_t waves = p.heads < waveLimit ? p.heads : waveLimit;
        bool overlap = p.rows <= 288 && waves == 2;
        for (uint32_t tick = 0; tick < waves + 2; ++tick) {
            if ASCEND_IS_AIC {
                if (tick < waves) {
                    uint32_t firstHead = uint64_t(p.heads) * tick / waves;
                    uint32_t count = uint64_t(p.heads) * (tick + 1) / waves - firstHead;
                    uint64_t part = uint64_t(count) * 128, betaBase = 4 * part + 128;
                    uint64_t total = tick ? 4 * part : betaBase + p.heads, blocks = (total + 15) / 16;
                    constexpr uint32_t workers = 4;
                    // Reserve one core for beta and divide the rest across Q, K, V and gate.
                    // A third 128-column panel would force another complete read of hidden_states.
                    const bool localDecay = !tick && !SpansTile && !reuseQ && !decayAReady && p.rows >= 64 &&
                                            p.rows <= 160 && groups >= workers + 6 &&
                                            count <= 2 * ((groups - workers - 1) / 4);
                    const uint32_t team = localDecay ? groups - workers : groups;
                    uint64_t cursor = (blocks * id / groups) * 16, stop = (blocks * (id + 1) / groups) * 16;
                    if (stop > total) {
                        stop = total;
                    }
                    // Keep each Cube core on one projection to avoid a second matrix pipeline setup.
                    if (localDecay && id >= team) {
                        cursor = stop;
                        ProjectDecayRows(p, planes + uint64_t(3) * planeRows * width * 2, id - team, workers);
                    }
                    const uint32_t kinds = tick ? 4 : 6;
                    if (id < team && team >= kinds) {
                        uint32_t groupStart = 0;
                        uint64_t columnStart = 0;
                        for (uint32_t kind = 0; kind < kinds; ++kind) {
                            const uint64_t columns = kind < 4 ? part : (kind == 4 ? 128 : p.heads);
                            const uint64_t end = columnStart + columns;
                            if (localDecay && kind == 4) {
                                columnStart = end;
                                continue;
                            }
                            const uint64_t grouped = localDecay ? total - 128 : total;
                            const uint64_t at = localDecay && kind == 5 ? end - 128 : end;
                            uint32_t groupEnd = static_cast<uint32_t>((at * team + grouped / 2) / grouped);
                            const uint32_t limit = team - (kinds - kind - 1) + (localDecay && kind < 4 ? 1 : 0);
                            if (groupEnd > limit) {
                                groupEnd = limit;
                            }
                            if (groupEnd <= groupStart) {
                                groupEnd = groupStart + 1;
                            }
                            if (id < groupEnd) {
                                const uint32_t rank = id - groupStart, team = groupEnd - groupStart;
                                const uint64_t units = (columns + 15) / 16;
                                cursor = columnStart + 16 * (units * rank / team);
                                stop = columnStart + 16 * (units * (rank + 1) / team);
                                if (stop > end) {
                                    stop = end;
                                }
                                break;
                            }
                            groupStart = groupEnd;
                            columnStart = end;
                        }
                    }
                    while (cursor < stop) {
                        uint32_t kind = cursor < 4 * part ? cursor / part : (cursor < betaBase ? 4 : 5);
                        uint64_t start = kind < 4 ? uint64_t(kind) * part : (kind == 4 ? 4 * part : betaBase);
                        uint32_t stride = kind < 4 ? width : (kind == 4 ? 128 : betaStride);
                        uint32_t local = cursor - start, extent = kind < 4 ? part : (kind == 4 ? 128 : p.heads);
                        uint32_t col = local + (kind < 4 ? firstHead * 128 : 0);
                        uint64_t n = stop - cursor;
                        if (n > extent - local) {
                            n = extent - local;
                        }
                        if (n > 128) {
                            n = 128;
                        }
                        if (kind == 4 && decayAReady) {
                            cursor += n;
                            continue;
                        }
                        GM_ADDR weight = kind < 3 ?
                                             p.qkvWeight :
                                             (kind == 3 ? p.gateWeight : (kind == 4 ? p.decayAWeight : p.betaWeight));
                        uint32_t weightCol = kind < 3 ? kind * width + col : col;
                        GM_ADDR dest =
                            kind == 4 ? p.decayRing : planes + uint64_t(kind == 3 ? 4 : kind) * planeRows * width * 2;
                        if (p.rows > 256 && p.rows <= 288) {
                            MatmulToGm<128, 256, 32, 288>(p.hiddenStates, weight, dest, 0, weightCol, 0, col, stride,
                                                          p.rows, kHiddenWidth, n);
                        } else if (p.rows <= 160 && n <= 96) {
                            MatmulToGm<96, 384, 96, 160>(p.hiddenStates, weight, dest, 0, weightCol, 0, col, stride,
                                                         p.rows, kHiddenWidth, n);
                        } else {
                            MatmulToGm<128, 256, 64>(p.hiddenStates, weight, dest, 0, weightCol, 0, col, stride, p.rows,
                                                     kHiddenWidth, n);
                        }
                        cursor += n;
                    }
                    if (!tick && !localDecay) {
                        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
                        AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
                        for (uint32_t h = id; h < p.heads; h += groups) {
                            if (p.rows > 256 && p.rows <= 288) {
                                MatmulToGm<128, 128, 32, 288>(p.decayRing, p.decayBWeight,
                                                              planes + uint64_t(3) * planeRows * width * 2, 0, h * 128,
                                                              0, h * 128, width, p.rows, 128);
                            } else {
                                MatmulToGm(p.decayRing, p.decayBWeight, planes + uint64_t(3) * planeRows * width * 2, 0,
                                           h * 128, 0, h * 128, width, p.rows, 128);
                            }
                        }
                    }
                } else if (overlap && tick == waves) {
                    if (p.rows <= 256) {
                        StreamOutput<256, 256, 64>(p, gated, id, groups);
                    } else {
                        StreamOutput<288, 128, 32>(p, gated, id, groups);
                    }
                } else if (!overlap && tick == waves + 1) {
                    if (p.rows > 256 && p.rows <= 288) {
                        for (uint32_t j = id; j < kHiddenWidth / 128; j += groups) {
                            MatmulToGm<128, 256, 32, 288>(gated, p.recurrence.outWeight, p.recurrence.output, 0,
                                                          j * 128, 0, j * 128, kHiddenWidth, p.rows, width, 128);
                        }
                    } else {
                        for (uint32_t panel = id; panel < kHiddenWidth / 256; panel += groups) {
                            MatmulToGm<256, 256, 64>(gated, p.recurrence.outWeight, p.recurrence.output, 0, panel * 256,
                                                     0, panel * 256, kHiddenWidth, p.rows, width, 256);
                        }
                    }
                }
            }
            if ASCEND_IS_AIV {
                if (tick >= 1 && tick <= waves) {
                    uint32_t left = uint64_t(p.heads) * (tick - 1) / waves;
                    uint32_t count = uint64_t(p.heads) * tick / waves - left;
                    ConsumeStagedProjections(p, globalBase, p.rows, id, 2 * groups, left, count, reuseQ, framePlanes);
                }
            }
            if ASCEND_IS_AIV {
                AscendC::SyncAll<false>();
            }
            if ASCEND_IS_AIC {
                if (!overlap || tick != waves) {
                    AscendC::SyncAll<false>();
                }
            }
        }
    }
    CATLASS_DEVICE bool TryRunSequenceFrames(const Params &p, uint32_t id, uint32_t groups)
    {
        const uint64_t width = uint64_t(p.heads) * kHeadDim;
        const uint64_t betaStride = (uint64_t(p.heads) + 15) / 16 * 16;
        // A complete frame reuses Q for its gated result after each sequence/head reads Q.
        // Include BF16 decay-A, the five projection planes, and the padded beta columns.
        const uint64_t bytes = SpansTile ? 32768 + uint64_t(p.heads) * 98304 : 32768 + 512 * (6 * width + betaStride);
        const uint64_t planeColumns = 5 * width + betaStride;
        const uint32_t workspaceRows = bytes / (2 * (128 + planeColumns)) / 16 * 16;
        uint32_t capacity = workspaceRows;
        bool borrowOutput = false;
        const auto &a = p.recurrence;
        Recurrence rec;
        if constexpr (SpansTile) {
            uint32_t longest = a.sequence;
            if (a.plan) {
                longest = reinterpret_cast<__gm__ const int32_t *>(a.plan)[1];
            } else if (!a.sequence) {
                for (uint32_t b = 0; b < a.batch; ++b) {
                    const uint32_t length = rec.prefix(a, b + 1) - rec.prefix(a, b);
                    longest = length > longest ? length : longest;
                }
            }
            if (longest > workspaceRows / 4) {
                // A frame keeps gated and decay-A in workspace. Its future output can hold the six projection planes.
                // Account for 16-row projection padding. A shorter tail can always use the original workspace.
                if (a.output == p.hiddenStates ||
                    planeColumns * (workspaceRows + 15) > kHiddenWidth * uint64_t(workspaceRows)) {
                    return false;
                }
                const uint32_t available = bytes / (2 * (128 + width)) / 16 * 16;
                capacity = available < 256 ? available : 256;
                // Each Vector task owns one sequence and one head in the current projection wave.
                const uint32_t sequencesPerFrame = capacity / longest;
                if (capacity <= workspaceRows || sequencesPerFrame < 4 || sequencesPerFrame * (p.heads / 2) < groups) {
                    return false;
                }
                borrowOutput = true;
            }
        }
        for (uint32_t first = 0; first < p.rows;) {
            uint32_t rows = p.rows - first;
            if (rows > capacity) {
                if constexpr (SpansTile) {
                    rows = rec.prefix(a, rec.owner(a, first + capacity)) - first;
                } else {
                    rows = capacity / kGlobalRows * kGlobalRows;
                }
            }
            const uint32_t padded = (rows + 15) / 16 * 16;
            Params frame = p;
            frame.hiddenStates = p.hiddenStates + uint64_t(first) * kHiddenWidth * sizeof(Bf);
            frame.recurrence.output = a.output + uint64_t(first) * kHiddenWidth * sizeof(Bf);
            frame.rows = rows;
            frame.recurrence.staged = 3;
            Waves single{rows, padded, padded, 1, padded, padded};
            const bool planesFit = uint64_t(padded) * planeColumns <= uint64_t(rows) * kHiddenWidth;
            RunHeadPipeline(frame, single, id, groups, false, first, true,
                            borrowOutput && planesFit ? frame.recurrence.output : nullptr);
            first += rows;
        }
        return true;
    }
    CATLASS_DEVICE void RunDoubleBuffered(const Params &p)
    {
        uint32_t id = AscendC::GetBlockIdx(), groups = AscendC::GetBlockNum();
        uint32_t span = p.recurrence.staged == 3 ? 128 : 64;
        auto shape = PlanWaves(p.rows);
        if (useDecayGemv(p.rows)) {
            if ASCEND_IS_AIV {
                shortDecayA(resource_, p.hiddenStates, p.decayAWeight, p.decayRing, p.rows, id, 2 * groups);
            }
            AscendC::SyncAll<false>();
        }
        if (p.recurrence.staged == 3 && p.heads >= 2 && p.rows > 256 &&
            uint64_t(p.rows) <= uint64_t(p.recurrence.batch) * 4) {
            TryRunSequenceFrames(p, id, groups);
            return;
        }
        if (p.recurrence.staged == 3 && shape.count == 1 && p.heads >= 2) {
            RunHeadPipeline(p, shape, id, groups, useDecayGemv(p.rows));
            return;
        }
        uint32_t tiles = span == 128 ? shape.count : (p.rows - 1) / span + 1, width = p.heads * 128,
                 betaStride = (p.heads + 15) / 16 * 16;
        uint32_t storageRows = span == 128 ? shape.storageRows : 128;
        uint32_t firstCap = span == 128 ? shape.firstCapacity : 64;
        GM_ADDR gated = p.decayRing + (span == 128 ? shape.decayRows : span) * 256;
        GM_ADDR scratch = gated + uint64_t(storageRows) * width * 2;
        uint64_t slotBytes = uint64_t(firstCap) * 2 * (5 * uint64_t(width) + betaStride);
        // The two projection slabs and the two gated slabs have independent consumers.
        for (uint32_t tick = 0; tick < tiles + 2; ++tick) {
            if ASCEND_IS_AIC {
                if (tick < tiles) {
                    uint32_t first = span == 128 ? shape.offset(tick) : span * tick;
                    uint32_t limit = span == 128 && !tick ? shape.firstRows : span,
                             m = p.rows - first < limit ? p.rows - first : limit;
                    uint32_t planeRows = tick % 2 ? storageRows - firstCap : firstCap;
                    GM_ADDR planes = scratch + (tick % 2) * slotBytes;
                    // Shared decay-A and the independent input projections share one dispatch.
                    // Assign equal numbers of aligned output columns to resident Cube cores.
                    const uint64_t betaBase = 4 * uint64_t(width) + 128;
                    const uint64_t total = betaBase + p.heads, blocks = (total + 15) / 16;
                    uint64_t cursor = (blocks * id / groups) * 16, stop = (blocks * (id + 1) / groups) * 16;
                    if (stop > total) {
                        stop = total;
                    }
                    while (cursor < stop) {
                        uint32_t kind = cursor < 4 * uint64_t(width) ? cursor / width : (cursor < betaBase ? 4 : 5);
                        uint64_t start =
                            kind < 4 ? uint64_t(kind) * width : (kind == 4 ? 4 * uint64_t(width) : betaBase);
                        uint32_t stride = kind < 4 ? width : (kind == 4 ? 128 : betaStride);
                        uint32_t physical = kind == 5 ? p.heads : stride;
                        uint32_t col = cursor - start;
                        uint64_t n = stop - cursor;
                        if (n > physical - col) {
                            n = physical - col;
                        }
                        if (n > 160) {
                            n = 160;
                        }
                        if (kind == 4 && useDecayGemv(p.rows)) {
                            cursor += n;
                            continue;
                        }
                        GM_ADDR weight = kind < 3 ?
                                             p.qkvWeight :
                                             (kind == 3 ? p.gateWeight : (kind == 4 ? p.decayAWeight : p.betaWeight));
                        uint32_t weightCol = kind < 3 ? kind * width + col : col;
                        GM_ADDR dest =
                            kind == 4 ? p.decayRing : planes + uint64_t(kind == 3 ? 4 : kind) * planeRows * width * 2;
                        MatmulToGm<160, 256, 64>(p.hiddenStates, weight, dest, first, weightCol, 0, col, stride, m,
                                                 kHiddenWidth, n);
                        cursor += n;
                    }
                    // Only Cube workers consume decay-A. The following loads are issued by MTE2.
                    AscendC::CrossCoreSetFlag<0, PIPE_FIX>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
                    AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
                    for (uint32_t h = id; h < p.heads; h += groups) {
                        MatmulToGm(p.decayRing, p.decayBWeight, planes + uint64_t(3) * planeRows * width * 2, 0,
                                   h * 128, 0, h * 128, width, m, 128);
                    }
                }
                if (tick >= 2) {
                    uint32_t tile = tick - 2, first = span == 128 ? shape.offset(tile) : tile * span;
                    uint32_t limit = span == 128 && !tile ? shape.firstRows : span,
                             m = p.rows - first < limit ? p.rows - first : limit;
                    GM_ADDR ready = gated + uint64_t(tile % 2) * firstCap * width * 2;
                    for (uint32_t panel = id; panel < kHiddenWidth / 256; panel += groups) {
                        MatmulToGm<256, 256, 64>(ready, p.recurrence.outWeight, p.recurrence.output, 0, panel * 256,
                                                 first, panel * 256, kHiddenWidth, m, width, 256);
                    }
                }
            }
            if ASCEND_IS_AIV {
                if (tick >= 1 && tick <= tiles) {
                    uint32_t tile = tick - 1, first = span == 128 ? shape.offset(tile) : tile * span;
                    uint32_t limit = span == 128 && !tile ? shape.firstRows : span,
                             m = p.rows - first < limit ? p.rows - first : limit;
                    ConsumeStagedProjections(p, first, m, id, 2 * groups);
                }
            }
            AscendC::SyncAll<false>();
        }
    }
    template <int TileN = 128, int L1K = 256, int L0K = 64, int TileM = 256, bool Stream = false>
    CATLASS_DEVICE void MatmulToGm(GM_ADDR input, GM_ADDR weight, GM_ADDR out, uint32_t xRow, uint32_t wCol,
                                   uint32_t outRow, uint32_t outCol, uint32_t stride, uint32_t m, uint32_t k,
                                   uint32_t n = 128)
    {
        AscendC::GlobalTensor<Bf> a, w, c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(input) + uint64_t(xRow) * k);
        w.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(weight) + uint64_t(wCol) * k);
        c.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(out) + uint64_t(outRow) * stride);
        auto x = tla::MakeTensor(a, tla::MakeLayout<Bf, Row>(m, k), Catlass::Arch::PositionGM{});
        auto wt = tla::MakeTensor(w, tla::MakeLayout<Bf, Col>(k, n), Catlass::Arch::PositionGM{});
        auto base = tla::MakeTensor(c, tla::MakeLayout<Bf, Row>(m, stride), Catlass::Arch::PositionGM{});
        using GmCopy = std::conditional_t<Stream, StreamingOutputCopy, OutputCopy>;
        using Mmad = Catlass::Gemm::Tile::TileMmadTla<Arch, Bf, typename GmCopy::LayoutTagL1A>;
        using Op =
            Catlass::Gemm::Block::BlockMmadTla<Policy, tla::Shape<tla::Int<TileM>, tla::Int<TileN>, tla::Int<L1K>>,
                                               tla::Shape<tla::Int<TileM>, tla::Int<TileN>, tla::Int<L0K>>, Bf, Bf, Bf,
                                               void, GmCopy, Mmad>;
        Op op(resource_);
        for (uint32_t first = 0; first < m; first += TileM) {
            uint32_t active = m - first < TileM ? m - first : TileM;
            auto slab = GetTile(x, tla::MakeCoord(first, 0), tla::MakeShape(active, k));
            auto result = GetTile(base, tla::MakeCoord(first, outCol), tla::MakeShape(active, n));
            op(slab, wt, result, Catlass::GemmCoord(active, n, k));
        }
    }
    CATLASS_DEVICE uint32_t FindSequenceCut(Params const &p, uint32_t first, uint32_t tileEnd, uint32_t cut)
    {
        if (cut >= tileEnd) {
            return tileEnd;
        }
        const auto &a = p.recurrence;
        uint32_t target = first + cut;
        if (a.sequence) {
            uint64_t boundary = (uint64_t(target) + a.sequence - 1) / a.sequence * a.sequence;
            return boundary - first < tileEnd ? boundary - first : tileEnd;
        }
        Recurrence rec;
        const uint32_t seq = rec.owner(a, target);
        const uint32_t start = rec.prefix(a, seq);
        const uint32_t boundary = start == target ? start : rec.prefix(a, seq + 1);
        return boundary - first < tileEnd ? boundary - first : tileEnd;
    }
    CATLASS_DEVICE void StoreTile(GM_ADDR output, uint32_t row, uint32_t col, uint32_t stride, uint32_t m, uint32_t n,
                                  AscendC::LocalTensor<Bf> buffer)
    {
        AscendC::GlobalTensor<Bf> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(output) + uint64_t(row) * stride);
        auto full = tla::MakeTensor(gm, tla::MakeLayout<Bf, Row>(m, stride), Catlass::Arch::PositionGM{});
        auto dest = GetTile(full, tla::MakeCoord(0, col), tla::MakeShape(m, n));
        auto ub = tla::MakeTensor(buffer, tla::MakeLayout<Bf, Row>(m, (n + 15) / 16 * 16), Catlass::Arch::PositionUB{});
        ObserveProjection observe;
        observe(dest, ub);
    }
    CATLASS_DEVICE void NotifyFree(uint32_t i)
    {
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(kFreeSlotFlagBase + i);
    }
    CATLASS_DEVICE void WaitFree(uint32_t i)
    {
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFreeSlotFlagBase + i);
    }
    CATLASS_DEVICE void NotifyReady(uint32_t i)
    {
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kReadySlotFlagBase + i);
    }
    CATLASS_DEVICE void WaitReady(uint32_t i)
    {
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(kReadySlotFlagBase + i);
    }
    CATLASS_DEVICE void PublishFreeSlots()
    {
        NotifyFree(0);
        NotifyFree(1);
    }
    CATLASS_DEVICE void SyncCores()
    {
        if ASCEND_IS_AIC {
            WaitFree(0);
            WaitFree(1);
            // Every resident Cube participates after its paired AIV completed its GM writes.
            Catlass::Arch::CrossCoreBarrier<0, PIPE_FIX>();
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(6);
        }
        if ASCEND_IS_AIV {
            // A new tile reads its predecessor's conv history and carry through MTE2.
            // Delay both these reads and the MTE3 free credit until the grid handoff.
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(6);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            PublishFreeSlots();
        }
    }
};
} // namespace kda
