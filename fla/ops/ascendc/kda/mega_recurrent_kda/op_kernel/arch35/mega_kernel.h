#pragma once
#include "short_decay_projection.h"
#include "mx_qkv.h"
#include "catlass/arch/cross_core_sync.hpp"

namespace kda {
// One BF16 tile in the paired AIV's UB. Beta has one value per row and a padded stride of 16.
constexpr uint32_t kSlotBytes = 20992;
constexpr uint32_t kSliceRows = 16;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kGlobalRows = 64;
CATLASS_DEVICE uint32_t OutputPanels(uint32_t hidden, uint32_t tile)
{
    return (hidden - 1) / tile + 1;
}
CATLASS_DEVICE uint32_t PanelWidth(uint32_t hidden, uint32_t start, uint32_t tile)
{
    return hidden - start < tile ? hidden - start : tile;
}
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
    GM_ADDR projectedQkv;
    bool firstWaveReady = false;
    uint32_t mxFirstHeads = 0;
    GM_ADDR mxWeightScale = nullptr;
    GM_ADDR decayAWeight;
    GM_ADDR decayBWeight;
    GM_ADDR gateWeight;
    GM_ADDR betaWeight;
    GM_ADDR decayRing;
    uint32_t rows, heads, hidden;
    RecurrenceArgs recurrence{};
};

// The Vector side publishes the second head group while Cube is computing the first.
using OutputCopy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, Bf, Row, Bf, Col, Bf, Row>;
template <uint32_t ExtraHead, uint32_t LoadK>
struct StreamingOutputCopy : OutputCopy {
    template <class Source>
    struct CopyGmToL1A : OutputCopy::template CopyGmToL1A<Source> {
        template <class Destination, class Tile>
        CATLASS_DEVICE void operator()(Destination &dst, Tile &src)
        {
            uint32_t width = tla::get<0>(src.layout().stride());
            uint32_t second = (width / kHeadDim / 2 + ExtraHead) * kHeadDim / LoadK * LoadK;
            if (uint32_t(tla::get<1>(src.coord())) == second) {
                AscendC::SyncAll<false>();
            }
            OutputCopy::template CopyGmToL1A<Source>::operator()(dst, src);
        }
    };
};
template <bool SpansTile, bool AllowColumnMajor = SpansTile, bool MxQkv = false>
class MegaKernel {
public:
    using Params = MegaParams;

    // The first BF16 head projections consume the original input while Vector quantizes it.
    CATLASS_DEVICE void PrepareFirstMxWave(Params &p)
    {
        const auto shape = PlanWaves(p.rows);
        const bool stagedFrame = p.rows > 256 && uint64_t(p.rows) <= uint64_t(p.recurrence.batch) * 4;
        p.firstWaveReady = p.recurrence.staged == 3 && p.rows >= kSliceRows && p.heads >= 2 &&
                           (shape.count == 1 || stagedFrame);
        const uint32_t cores = AscendC::GetBlockNum();
        if (p.firstWaveReady && p.rows <= 256 && cores >= 3) {
            const uint32_t left = FirstWaveHeads(p, cores);
            const uint32_t larger = left > p.heads - left ? left : p.heads - left;
            const uint32_t smaller = p.heads - larger;
            // Each head supplies twelve 32-column Q/K/V panels across the three matrices.
            if (uint64_t(larger) * 4 <= (cores / 3) * 6 && uint64_t(smaller) * 12 >= cores) {
                p.mxFirstHeads = left;
            }
        }
        if ASCEND_IS_AIC {
            if (p.firstWaveReady) {
                Params frame = p;
                uint32_t planeRows = shape.firstCapacity;
                GM_ADDR planes = p.decayRing + shape.decayRows * 256 + uint64_t(shape.storageRows) * p.heads * 256;
                if (stagedFrame) {
                    const uint32_t capacity = T01FrameCapacity(p.heads);
                    frame.rows = p.rows <= capacity ? p.rows : capacity / kGlobalRows * kGlobalRows;
                    planeRows = (frame.rows + 15) / 16 * 16;
                    planes = p.decayRing + planeRows * 256;
                }
                ProjectHeadWave(frame, planes, planeRows, 0, 2, AscendC::GetBlockIdx(),
                                AscendC::GetBlockNum(), false, stagedFrame);
            }
        }
    }

    CATLASS_DEVICE void operator()(const Params &p)
    {
        if constexpr (!SpansTile) {
            RunDoubleBuffered(p);
            return;
        }
        if constexpr (MxQkv && SpansTile) {
            const uint64_t planeColumns = 4u * p.heads * 128 + (p.heads + 15) / 16 * 16;
            const uint32_t tail = p.rows % 64;
            if (p.rows >= 64 && uint64_t(p.rows) >= uint64_t(p.recurrence.batch) * 16 && planeColumns <= p.hidden &&
                p.recurrence.output != p.hiddenStates) {
                const uint32_t capacity = TileCapacity(p.heads);
                Recurrence rec;
                uint32_t longest = p.recurrence.sequence;
                if (!longest && p.recurrence.plan) {
                    longest = reinterpret_cast<__gm__ const int32_t *>(p.recurrence.plan)[1];
                } else if (!longest) {
                    for (uint32_t b = 0; b < p.recurrence.batch; ++b) {
                        const uint32_t n = rec.prefix(p.recurrence, b + 1) - rec.prefix(p.recurrence, b);
                        longest = n > longest ? n : longest;
                    }
                }
                bool fits = p.recurrence.batch == 1 || longest > capacity / 2;
                // Each visible request needs nine history vectors in the existing 128-row gated slab.
                if (p.recurrence.batch > 14) {
                    for (uint32_t row = 0; fits && row < p.rows; row += 64) {
                        const uint32_t last = p.rows - row <= 64 ? p.rows - 1 : row + 63;
                        fits = rec.owner(p.recurrence, last) - rec.owner(p.recurrence, row) < 14;
                    }
                }
                if (fits && tail) {
                    const uint32_t padded = (tail + 15) / 16 * 16, width = p.heads * kHeadDim;
                    const uint64_t projectedWords = uint64_t(padded) * planeColumns;
                    if (projectedWords > uint64_t(tail) * p.hidden) {
                        const uint32_t active =
                            rec.owner(p.recurrence, p.rows - 1) - rec.owner(p.recurrence, p.rows - tail) + 1;
                        const uint64_t liveRows = tail > active * 9 ? tail : active * 9;
                        fits = projectedWords + liveRows * width <= 128u * uint64_t(width);
                    }
                }
                if (fits) {
                    RunValueShards(p);
                    return;
                }
            }
        }
        if (p.heads >= 2 && TryRunSequenceFrames(p, AscendC::GetBlockIdx(), AscendC::GetBlockNum())) {
            return;
        }
        RunTilePipeline(p);
    }

private:
    Catlass::Arch::Resource<Arch> resource_;

    CATLASS_DEVICE uint32_t TileCapacity(uint32_t heads)
    {
        return ((32768 + uint64_t(heads) * 98304) / (256 + uint64_t(heads) * 256)) / 16 * 16;
    }
    template <class T>
    CATLASS_DEVICE AscendC::GlobalTensor<T> mem(GM_ADDR ptr, uint64_t offset = 0)
    {
        AscendC::GlobalTensor<T> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr) + offset);
        return g;
    }
    CATLASS_DEVICE void RunValueShards(const Params &p)
    {
        const uint32_t id = AscendC::GetBlockIdx(), groups = AscendC::GetBlockNum(), width = p.heads * 128;
        const auto &a = p.recurrence;
        Recurrence rec;
        GM_ADDR gated = p.decayRing + 32768;
        // History is consumed before the same workspace is filled with gated rows.
        GM_ADDR history = gated;
        GM_ADDR carry = gated + uint64_t(p.heads) * 32768;
        for (uint32_t first = 0; first < p.rows; first += 64) {
            const uint32_t rows = p.rows - first < 64 ? p.rows - first : 64, padded = (rows + 15) / 16 * 16;
            Params frame = p;
            frame.hiddenStates += uint64_t(first) * p.hidden * 2;
            frame.recurrence.output += uint64_t(first) * p.hidden * 2;
            frame.rows = rows;
            frame.recurrence.staged = 3;
            frame.firstWaveReady = false;
            // The future output holds FP32 y, BF16 decay/gate and the padded beta projection.
            const uint64_t planeWords = uint64_t(padded) * (4 * width + (p.heads + 15) / 16 * 16);
            // A short tail can place its padded projection planes above gated/history in the same workspace slab.
            GM_ADDR planes =
                planeWords <= uint64_t(rows) * p.hidden ? frame.recurrence.output : carry - planeWords * sizeof(Bf);
            if ASCEND_IS_AIC {
                ProjectHeadWave<1>(frame, planes, padded, 0, 1, id, groups, false, true);
            }
            if ASCEND_IS_AIV {
                const uint32_t beginSeq = rec.owner(a, first);
                const uint32_t endSeq = rec.owner(a, first + rows - 1);
                for (uint32_t task = id; task < (endSeq - beginSeq + 1) * p.heads; task += 2 * groups) {
                    const uint32_t b = beginSeq + task / p.heads, h = task % p.heads;
                    const uint32_t slot = mem<int32_t>(a.indices).GetValue(b);
                    const uint32_t accepted = mem<int32_t>(a.accepted).GetValue(b);
                    const uint32_t start = rec.prefix(a, b);
                    const uint32_t hs = first > start ? first - start - 1 : (accepted ? accepted - 1 : 0);
                    const uint64_t base = uint64_t(slot) * a.convStride + uint64_t(hs) * 3 * width + h * 128;
                    auto local = resource_.ubBuf.template GetBufferByByte<Bf>(kMxUbOffset);
                    AscendC::DataCopyPad(local, mem<Bf>(a.conv, base),
                                         AscendC::DataCopyExtParams{9, 256, (width - 128) * 2, 0, 0},
                                         AscendC::DataCopyPadExtParams<Bf>{});
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyPad(mem<Bf>(history, uint64_t(b - beginSeq) * 9 * width + h * 128), local,
                                         AscendC::DataCopyExtParams{9, 256, 0, (width - 128) * 2, 0});
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            AscendC::SyncAll<false>();
            if ASCEND_IS_AIV {
                if (p.heads * 8 <= 2 * groups) {
                    ConsumeStagedProjections<16>(frame, first, rows, id, 2 * groups, 0, p.heads, true, planes, carry,
                                                 history);
                } else {
                    ConsumeStagedProjections<32>(frame, first, rows, id, 2 * groups, 0, p.heads, true, planes, carry,
                                                 history);
                }
            }
            AscendC::SyncAll<false>();
            if ASCEND_IS_AIV {
                auto y = resource_.ubBuf.template GetBufferByByte<float>(kMxUbOffset);
                auto gate = resource_.ubBuf.template GetBufferByByte<Bf>(kMxUbOffset + 512);
                auto gamma = resource_.ubBuf.template GetBufferByByte<float>(kMxUbOffset + 768);
                auto out = resource_.ubBuf.template GetBufferByByte<Bf>(kMxUbOffset + 1280);
                AscendC::DataCopy(gamma, mem<float>(a.gamma), 128);
                for (uint32_t i = id; i < rows * p.heads; i += 2 * groups) {
                    AscendC::DataCopy(y, mem<float>(planes, uint64_t(i) * 128), 128);
                    AscendC::DataCopy(gate, mem<Bf>(planes, 3 * uint64_t(padded) * width + i * 128), 128);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    vec::finish((__ubuf__ float *)y.GetPhyAddr(), (__ubuf__ Bf *)gate.GetPhyAddr(),
                                (__ubuf__ float *)gamma.GetPhyAddr(), a.eps, (__ubuf__ Bf *)out.GetPhyAddr());
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopy(mem<Bf>(gated, i * 128), out, 128);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            AscendC::SyncAll<false>();
            if ASCEND_IS_AIC {
                for (uint32_t panel = id; panel < OutputPanels(p.hidden, 256); panel += groups) {
                    MatmulToGm<256, 256, 64>(gated, a.outWeight, frame.recurrence.output, 0, panel * 256, 0,
                                             panel * 256, p.hidden, rows, width, PanelWidth(p.hidden, panel * 256, 256));
                }
            }
            AscendC::SyncAll<false>();
        }
    }
    CATLASS_DEVICE bool ReusesCarry(Recurrence &rec, const RecurrenceArgs &a, uint32_t first, uint32_t rows,
                                   uint32_t firstSeq, uint32_t lastSeq)
    {
        return firstSeq != lastSeq && rec.prefix(a, firstSeq) < first && rec.prefix(a, lastSeq + 1) > first + rows;
    }
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
        const uint32_t capacity = TileCapacity(p.heads);
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
            uint32_t stride = p.heads;
            if constexpr (MxQkv) {
                if (!wholeSequences && first && first + rows < p.rows &&
                    ReusesCarry(rec, a, first, rows, rec.owner(a, first), rec.owner(a, first + rows - 1))) {
                    // Consecutive cuts must consume and replace a head's single carry in order.
                    stride = (p.heads + groups - 1) / groups * groups;
                }
            }
            for (uint64_t work = group; work < uint64_t(cuts) * stride; work += groups) {
                uint32_t part = work / stride, head = work % stride;
                if (head >= p.heads) {
                    continue;
                }
                uint32_t offset = wholeSequences ? rec.prefix(a, firstSequence + part) - first :
                                                   (part ? FindSequenceCut(p, first, rows, part * 16) : 0);
                uint32_t end = wholeSequences ?
                                   rec.prefix(a, firstSequence + part + 1) - first :
                                   (part + 1 < cuts ? FindSequenceCut(p, first, rows, (part + 1) * 16) : rows);
                for (uint32_t r = offset; r < end; r += kSliceRows) {
                    uint32_t valid = end - r < kSliceRows ? end - r : kSliceRows;
                    if ASCEND_IS_AIC {
                        WaitFree(phase);
                        if constexpr (!MxQkv) {
                            for (uint32_t i = 0; i < 3; ++i) {
                                Matmul(p.hiddenStates, p.qkvWeight, first + r, (head + i * p.heads) * kHeadDim,
                                       valid, kHeadDim, p.hidden, LocalTile(phase, static_cast<Plane>(i)));
                            }
                        }
                    }
                    uint32_t col = head * kHeadDim;
                    if ASCEND_IS_AIC {
                        Matmul(p.decayRing, p.decayBWeight, r, col, valid, kHeadDim, 128,
                               LocalTile(phase, Plane::Decay));
                        Matmul(p.hiddenStates, p.gateWeight, first + r, col, valid, kHeadDim, p.hidden,
                               LocalTile(phase, Plane::Gate));
                        Matmul(p.hiddenStates, p.betaWeight, first + r, head, valid, 1, p.hidden,
                               LocalTile(phase, Plane::Beta));
                        NotifyReady(phase);
                    }
                    if ASCEND_IS_AIV {
                        WaitReady(phase);
                        if constexpr (MxQkv) {
                            LoadProjected(p, first + r, head, valid, phase);
                            AscendC::PipeBarrier<PIPE_ALL>();
                        }
                        Recurrence{}.consume<AllowColumnMajor, MxQkv>(resource_, p.recurrence, gated, carry, p.heads,
                                                                      head, first + r, r, valid,
                                                                      LocalTile(phase, Plane::Q));
                        NotifyFree(phase);
                    }
                    phase ^= 1;
                }
            }
            SyncCores();
            if (wholeSequences) {
                if ASCEND_IS_AIC {
                    for (uint32_t panel = group; panel < OutputPanels(p.hidden, 256); panel += groups) {
                        MatmulToGm<256, 256, 64, 128>(gated, p.recurrence.outWeight, p.recurrence.output, 0,
                                                      panel * 256, first, panel * 256, p.hidden, rows,
                                                      p.heads * 128, PanelWidth(p.hidden, panel * 256, 256));
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
        uint32_t offset = (MxQkv ? kMxUbOffset : 0) + phase * kSlotBytes +
                          static_cast<uint32_t>(plane) * kSliceRows * kHeadDim * sizeof(Bf);
        return resource_.ubBuf.template GetBufferByByte<Bf>(offset);
    }
    CATLASS_DEVICE void LoadProjected(const Params &p, uint32_t row, uint32_t head, uint32_t m, uint32_t phase)
    {
        uint32_t width = p.heads * kHeadDim;
        AscendC::GlobalTensor<Bf> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(p.projectedQkv));
        for (uint32_t i = 0; i < 3; ++i) {
            auto offset = uint64_t(row) * (3 * width) + i * width + head * kHeadDim;
            AscendC::DataCopyPad(LocalTile(phase, static_cast<Plane>(i)), src[offset],
                AscendC::DataCopyExtParams{static_cast<uint16_t>(m), 256, (3 * width - 128) * 2, 0, 0},
                AscendC::DataCopyPadExtParams<Bf>{});
        }
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
    template <uint16_t ValueRows = 128, uint32_t SkippedPlanes = (ValueRows < 128 ? 1 : 0)>
    CATLASS_DEVICE void ConsumeStagedProjections(const Params &p, uint32_t first, uint32_t rows, uint32_t worker,
                                                 uint32_t workers, uint32_t firstHead = 0, uint32_t count = 0,
                                                 bool reuseQ = false, GM_ADDR framePlanes = nullptr,
                                                 GM_ADDR carry = nullptr, GM_ADDR historySnapshot = nullptr)
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
        constexpr uint32_t shards = 128 / ValueRows;
        const uint32_t lanes = count * shards;
        uint32_t stride = lanes;
        if constexpr (ValueRows < 128) {
            // Keep each carry stripe on its reader until a different request may overwrite it in this frame.
            if (ReusesCarry(rec, p.recurrence, first, rows, firstSeq, lastSeq)) {
                stride = (lanes + workers - 1) / workers * workers;
            }
        }
        const uint32_t requests = lastSeq - firstSeq + 1;
        const uint64_t tasks = uint64_t(requests) * stride;
        const bool headMajor = ValueRows == 128 && !carry && requests > 1 && tasks > workers;
        const uint64_t begin = headMajor ? tasks * worker / workers : worker;
        const uint64_t end = headMajor ? tasks * (worker + 1) / workers : tasks;
        uint32_t previousHead = UINT32_MAX;
        for (uint64_t task = begin; task < end; task += headMajor ? 1 : workers) {
            const uint32_t lane = headMajor ? task / requests : task % stride;
            if (lane >= lanes) {
                continue;
            }
            uint32_t head = firstHead + lane / shards, b = firstSeq + (headMajor ? task % requests : task / stride);
            uint32_t start = rec.prefix(p.recurrence, b), end = rec.prefix(p.recurrence, b + 1);
            const uint32_t beginRow = carry && start < first ? first : start;
            const uint32_t endRow = carry && end > first + rows ? first + rows : end;
            for (uint32_t row = beginRow; row < endRow; row += kSliceRows) {
                uint32_t m = endRow - row < kSliceRows ? endRow - row : kSliceRows;
                if constexpr (MxQkv) {
                    LoadProjected(p, row, head, m, 0);
                }
                for (uint32_t i = MxQkv ? 3 : 0; i < 6; ++i) {
                    AscendC::GlobalTensor<Bf> src;
                    src.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(planes) +
                                        uint64_t(i - SkippedPlanes) * planeRows * width);
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
                // consume waits for its own loads before it reads the projected inputs.
                rec.consume<AllowColumnMajor, MxQkv, ValueRows>(
                    resource_, p.recurrence, ValueRows == 128 ? gated : planes, carry, p.heads, head, row, row - first,
                    m, LocalTile(0, Plane::Q), (row == start && row + m == end) ? b : UINT32_MAX,
                    (lane % shards) * ValueRows,
                    ValueRows == 128 ? nullptr : historySnapshot + uint64_t(b - firstSeq) * 9 * width * 2,
                    headMajor && previousHead == head);
                previousHead = head;
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
                Matmul(p.hiddenStates, p.decayAWeight, first + start, 0, valid, 128, p.hidden, local);
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
        const uint32_t cols = OutputPanels(p.hidden, kHeadDim);
        constexpr uint32_t outputRows = 64;
        uint32_t slices = (rows + outputRows - 1) / outputRows;
        for (uint32_t task = group; task < slices * cols; task += groups) {
            uint32_t r = task / cols * outputRows, col = task % cols * kHeadDim;
            uint32_t valid = rows - r < outputRows ? rows - r : outputRows;
            auto tile = LocalTile(phase, Plane::Q);
            if ASCEND_IS_AIC {
                WaitFree(phase);
                Matmul<64>(gated, p.recurrence.outWeight, r, col, valid, PanelWidth(p.hidden, col, kHeadDim), p.heads * kHeadDim, tile);
                NotifyReady(phase);
            }
            if ASCEND_IS_AIV {
                WaitReady(phase);
                StoreTile(p.recurrence.output, first + r, col, p.hidden, valid, PanelWidth(p.hidden, col, kHeadDim), tile);
                NotifyFree(phase);
            }
            phase ^= 1;
        }
    }
    template <int TileM, int TileN, int L0K, uint32_t ExtraHead = 0>
    CATLASS_DEVICE void StreamOutput(const Params &p, GM_ADDR gated, uint32_t id, uint32_t groups)
    {
        const uint32_t panels = OutputPanels(p.hidden, TileN);
        const uint32_t width = p.heads * kHeadDim;
        if (!MxQkv && p.rows <= kSliceRows && width >= 512 && p.hidden % 32 == 0 &&
            panels < groups && p.hidden / 32 >= groups) {
            const uint32_t blocks = p.hidden / 32;
            const uint32_t first = blocks * id / groups * 32;
            const uint32_t end = blocks * (id + 1) / groups * 32;
            MatmulToGm<TileN, 256, L0K, TileM, true, ExtraHead>(
                gated, p.recurrence.outWeight, p.recurrence.output, 0, first, 0, first,
                p.hidden, p.rows, width, end - first);
            return;
        }
        if (id < panels) {
            if (!MxQkv && p.rows <= 16 && width >= 512) {
                MatmulToGm<TileN, 256, L0K, TileM, true, ExtraHead>(
                    gated, p.recurrence.outWeight, p.recurrence.output, 0, id * TileN, 0, id * TileN,
                    p.hidden, p.rows, width, PanelWidth(p.hidden, id * TileN, TileN));
            } else {
                MatmulToGm<TileN, 128, L0K, TileM, true, ExtraHead>(
                    gated, p.recurrence.outWeight, p.recurrence.output, 0, id * TileN, 0, id * TileN,
                    p.hidden, p.rows, width, PanelWidth(p.hidden, id * TileN, TileN));
            }
        } else {
            AscendC::SyncAll<false>();
        }
        for (uint32_t j = id + groups; j < panels; j += groups) {
            MatmulToGm<TileN, 256, L0K, TileM>(gated, p.recurrence.outWeight, p.recurrence.output, 0, j * TileN, 0,
                                               j * TileN, p.hidden, p.rows, width, PanelWidth(p.hidden, j * TileN, TileN));
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
                                     p.hidden);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t col = 0; col < width; col += 256) {
            const uint32_t count = width - col < 256 ? width - col : 256;
            MatmulToGm<256, 128, 64, 80>(p.decayRing, p.decayBWeight, output, first, col, first, col, width,
                                         end - first, 128, count);
        }
    }
    CATLASS_DEVICE uint32_t FirstWaveHeads(const Params &p, uint32_t groups)
    {
        const uint32_t half = p.heads / 2;
        if (!MxQkv || !p.firstWaveReady || p.rows > 256 || half + 1 >= p.heads) {
            return half;
        }
        // Move one head only when it eliminates a scheduling round across both Vector waves.
        const uint64_t requests = p.recurrence.batch, workers = uint64_t(groups) * 2;
        const uint64_t left = requests * half, right = requests * (p.heads - half);
        const uint64_t rounds = (left + workers - 1) / workers + (right + workers - 1) / workers;
        const uint64_t shifted =
            (left + requests + workers - 1) / workers + (right - requests + workers - 1) / workers;
        return shifted < rounds ? half + 1 : half;
    }
    template <uint32_t SkippedPlanes = 0>
    CATLASS_DEVICE void ProjectHeadWave(const Params &p, GM_ADDR planes, uint32_t planeRows, uint32_t tick,
                                        uint32_t waves, uint32_t id, uint32_t groups, bool decayAReady, bool reuseQ)
    {
        const uint32_t width = p.heads * 128, betaStride = (p.heads + 15) / 16 * 16;
        constexpr uint32_t firstKind = MxQkv ? 3 : 0;  // Q/K/V already reside in projectedQkv.
        constexpr uint32_t wideProjections = 4 - firstKind;
        uint32_t split = waves == 2 ? FirstWaveHeads(p, groups) : p.heads;
        uint32_t firstHead = tick ? split : 0;
        uint32_t count = tick ? p.heads - split : split;
        uint64_t part = uint64_t(count) * 128, betaBase = wideProjections * part + 128;
        uint64_t total = tick ? wideProjections * part : betaBase + p.heads, blocks = (total + 15) / 16;
        // Each MX worker owns one 16-row decay tile, ready for its dependent decay-B projection.
        const uint32_t workers = MxQkv ? (p.rows + kSliceRows - 1) / kSliceRows : 4;
        // Reserve one core for beta and divide the rest across Q, K, V and gate.
        // A third 128-column panel would force another complete read of hidden_states.
        const bool localDecay = !tick && !SpansTile && !reuseQ && !decayAReady && p.rows >= 64 &&
                                p.rows <= 160 && groups >= workers + 6 &&
                                count <= 2 * ((groups - workers - 1) / wideProjections);
        const uint32_t team = localDecay ? groups - workers : groups;
        uint64_t cursor = (blocks * id / groups) * 16, stop = (blocks * (id + 1) / groups) * 16;
        if (stop > total) {
            stop = total;
        }
        // Keep each Cube core on one projection to avoid a second matrix pipeline setup.
        if (localDecay && id >= team) {
            cursor = stop;
            ProjectDecayRows(p, planes + uint64_t(3 - SkippedPlanes) * planeRows * width * 2, id - team, workers);
        }
        const uint32_t kinds = tick ? 4 : 6;
        if (id < team && team >= kinds - firstKind) {
            uint32_t groupStart = 0;
            uint64_t columnStart = 0;
            for (uint32_t kind = firstKind; kind < kinds; ++kind) {
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
            uint32_t kind = cursor < wideProjections * part ? firstKind + cursor / part : (cursor < betaBase ? 4 : 5);
            uint64_t start = kind < 4 ? uint64_t(kind - firstKind) * part : (kind == 4 ? wideProjections * part : betaBase);
            uint32_t stride = kind < 4 ? width : (kind == 4 ? 128 : betaStride);
            uint32_t local = cursor - start, extent = kind < 4 ? part : (kind == 4 ? 128 : p.heads);
            uint32_t col = local + (kind < 4 ? firstHead * 128 : 0);
            uint64_t n = stop - cursor;
            if (n > extent - local) {
                n = extent - local;
            }
            const uint32_t panel = !MxQkv && p.rows <= 16 ? 256 : 128;
            if (n > panel) {
                n = panel;
            }
            if (kind == 4 && decayAReady) {
                cursor += n;
                continue;
            }
            GM_ADDR weight = kind < 3 ?
                                 p.qkvWeight :
                                 (kind == 3 ? p.gateWeight : (kind == 4 ? p.decayAWeight : p.betaWeight));
            uint32_t weightCol = kind < 3 ? kind * width + col : col;
            GM_ADDR dest = kind == 4 ?
                               p.decayRing :
                               planes + uint64_t((kind == 3 ? 4 : kind) - SkippedPlanes) * planeRows * width * 2;
            if (!MxQkv && p.rows <= 16 && n > 128) {
                MatmulToGm<256, 384, 64, 16>(p.hiddenStates, weight, dest, 0, weightCol, 0, col, stride,
                                               p.rows, p.hidden, n);
            } else if (p.rows > 256 && p.rows <= 288) {
                MatmulToGm<128, 256, 32, 288>(p.hiddenStates, weight, dest, 0, weightCol, 0, col, stride,
                                              p.rows, p.hidden, n);
            } else if (p.rows <= 160 && n <= 96) {
                MatmulToGm<96, 384, 96, 160>(p.hiddenStates, weight, dest, 0, weightCol, 0, col, stride,
                                             p.rows, p.hidden, n, MxQkv && p.mxFirstHeads && kind == 3);
            } else {
                MatmulToGm<128, 256, 64>(p.hiddenStates, weight, dest, 0, weightCol, 0, col, stride, p.rows,
                                         p.hidden, n, MxQkv && p.mxFirstHeads && kind == 3);
            }
            cursor += n;
        }
        if (!tick && !localDecay) {
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
            AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
            for (uint32_t h = id; h < p.heads; h += groups) {
                if (p.rows > 256 && p.rows <= 288) {
                    MatmulToGm<128, 128, 32, 288>(p.decayRing, p.decayBWeight,
                                                  planes + uint64_t(3 - SkippedPlanes) * planeRows * width * 2, 0,
                                                  h * 128, 0, h * 128, width, p.rows, 128);
                } else {
                    MatmulToGm(p.decayRing, p.decayBWeight,
                               planes + uint64_t(3 - SkippedPlanes) * planeRows * width * 2, 0, h * 128, 0, h * 128,
                               width, p.rows, 128);
                }
            }
        }
    }
    template <uint32_t SkippedPlanes = 0>
    CATLASS_DEVICE void RunHeadPipeline(const Params &p, const Waves &shape, uint32_t id, uint32_t groups,
                                        bool decayAReady, uint32_t globalBase = 0, bool reuseQ = false,
                                        GM_ADDR framePlanes = nullptr)
    {
        const uint32_t width = p.heads * 128, planeRows = shape.firstCapacity;
        GM_ADDR gated = p.decayRing + shape.decayRows * 256;
        GM_ADDR planes = reuseQ ? gated : gated + uint64_t(shape.storageRows) * width * 2;
        if (SpansTile && framePlanes) {
            planes = framePlanes;
        }
        constexpr uint32_t waveLimit = 2;
        const uint32_t waves = p.heads < waveLimit ? p.heads : waveLimit;
        bool overlap = p.rows <= 288 && waves == 2;
        for (uint32_t tick = (MxQkv && p.firstWaveReady) ? 1 : 0; tick < waves + 2; ++tick) {
            if ASCEND_IS_AIC {
                if (tick < waves) {
                    if constexpr (MxQkv) {
                        if (tick == 1 && p.mxFirstHeads) {
                            ProjectMxHeads(resource_, p.recurrence.output, p.qkvWeight, p.mxWeightScale,
                                           p.projectedQkv, p.rows, p.heads, p.hidden, p.mxFirstHeads,
                                           p.heads - p.mxFirstHeads);
                        }
                    }
                    ProjectHeadWave<SkippedPlanes>(p, planes, planeRows, tick, waves, id, groups, decayAReady, reuseQ);
                } else if (overlap && tick == waves) {
                    if (MxQkv && FirstWaveHeads(p, groups) > p.heads / 2) {
                        StreamOutput<256, 256, 64, 1>(p, gated, id, groups);
                    } else if (p.rows <= 256) {
                        StreamOutput<256, 256, 64>(p, gated, id, groups);
                    } else {
                        StreamOutput<288, 128, 32>(p, gated, id, groups);
                    }
                } else if (!overlap && tick == waves + 1) {
                    if (p.rows > 256 && p.rows <= 288) {
                        for (uint32_t j = id; j < OutputPanels(p.hidden, 128); j += groups) {
                            MatmulToGm<128, 256, 32, 288>(gated, p.recurrence.outWeight, p.recurrence.output, 0,
                                                          j * 128, 0, j * 128, p.hidden, p.rows, width, PanelWidth(p.hidden, j * 128, 128));
                        }
                    } else {
                        for (uint32_t panel = id; panel < OutputPanels(p.hidden, 256); panel += groups) {
                            MatmulToGm<256, 256, 64>(gated, p.recurrence.outWeight, p.recurrence.output, 0, panel * 256,
                                                     0, panel * 256, p.hidden, p.rows, width, PanelWidth(p.hidden, panel * 256, 256));
                        }
                    }
                }
            }
            if ASCEND_IS_AIV {
                if (tick >= 1 && tick <= waves) {
                    uint32_t split = waves == 2 ? FirstWaveHeads(p, groups) : p.heads;
                    uint32_t left = tick == 1 ? 0 : split;
                    uint32_t count = tick == 1 ? split : p.heads - split;
                    ConsumeStagedProjections<128, SkippedPlanes>(p, globalBase, p.rows, id, 2 * groups, left, count, reuseQ, framePlanes);
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
    CATLASS_DEVICE uint32_t T01FrameCapacity(uint32_t heads)
    {
        const uint64_t width = uint64_t(heads) * kHeadDim, beta = (uint64_t(heads) + 15) / 16 * 16;
        const uint64_t bytes = 32768 + 512 * (6 * width + beta);
        return bytes / (2 * (128 + 5 * width + beta)) / 16 * 16;
    }
    CATLASS_DEVICE bool TryRunSequenceFrames(const Params &p, uint32_t id, uint32_t groups)
    {
        const uint64_t width = uint64_t(p.heads) * kHeadDim;
        const uint64_t betaStride = (uint64_t(p.heads) + 15) / 16 * 16;
        // A complete frame reuses Q for its gated result after each sequence/head reads Q.
        // Include BF16 decay-A, the five projection planes, and the padded beta columns.
        const uint64_t bytes = 32768 + uint64_t(p.heads) * 98304;
        const uint64_t planeColumns = 5 * width + betaStride;
        const uint32_t workspaceRows = SpansTile ? bytes / (2 * (128 + planeColumns)) / 16 * 16 :
                                                  T01FrameCapacity(p.heads);
        uint32_t capacity = workspaceRows;
        bool borrowOutput = false, compact = false;
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
            if constexpr (MxQkv) {
                // QKV lives outside this frame. Use three planes for gated, decay and gate.
                const uint32_t packedRows = bytes / (2 * (128 + 3 * width + betaStride)) / 16 * 16;
                // Repeated frames must offer at least one independent request/head task per Cube group in each wave.
                const bool enoughWork = longest && uint64_t(packedRows / longest) * (p.heads / 2) >= groups;
                compact = p.rows > workspaceRows && uint64_t(longest) * 2 <= p.rows &&
                          (p.rows <= packedRows || enoughWork);
                if (compact) {
                    capacity = packedRows;
                }
            }
            if (!compact && longest > workspaceRows / 4) {
                // A frame keeps gated and decay-A in workspace. Its future output can hold the six projection planes.
                // Account for 16-row projection padding. A shorter tail can always use the original workspace.
                if (a.output == p.hiddenStates ||
                    planeColumns * (workspaceRows + 15) > p.hidden * uint64_t(workspaceRows)) {
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
            frame.hiddenStates = p.hiddenStates + uint64_t(first) * p.hidden * sizeof(Bf);
            frame.recurrence.output = a.output + uint64_t(first) * p.hidden * sizeof(Bf);
            frame.rows = rows;
            frame.recurrence.staged = 3;
            frame.firstWaveReady = first == 0 && p.firstWaveReady;
            Waves single{rows, padded, padded, 1, padded, padded};
            const bool planesFit = uint64_t(padded) * planeColumns <= uint64_t(rows) * p.hidden;
            if constexpr (MxQkv && SpansTile) {
                if (compact) {
                    RunHeadPipeline<2>(frame, single, id, groups, false, first, true);
                    first += rows;
                    continue;
                }
            }
            RunHeadPipeline(frame, single, id, groups, false, first, true,
                            borrowOutput && planesFit ? frame.recurrence.output : nullptr);
            first += rows;
        }
        return true;
    }
    CATLASS_DEVICE void RunDoubleBuffered(const Params &p)
    {
        uint32_t id = AscendC::GetBlockIdx(), groups = AscendC::GetBlockNum();
        constexpr uint32_t firstKind = MxQkv ? 3 : 0;
        uint32_t span = p.recurrence.staged == 3 ? 128 : 64;
        auto shape = PlanWaves(p.rows);
        if (useDecayGemv(p.rows, p.hidden)) {
            if ASCEND_IS_AIV {
                shortDecayA(resource_, p.hiddenStates, p.decayAWeight, p.decayRing, p.rows, id, 2 * groups, p.hidden);
            }
            AscendC::SyncAll<false>();
        }
        if (p.recurrence.staged == 3 && p.heads >= 2 && p.rows > 256 &&
            uint64_t(p.rows) <= uint64_t(p.recurrence.batch) * 4) {
            TryRunSequenceFrames(p, id, groups);
            return;
        }
        if (p.recurrence.staged == 3 && shape.count == 1 && p.heads >= 2) {
            RunHeadPipeline(p, shape, id, groups, useDecayGemv(p.rows, p.hidden));
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
                    const uint64_t wideColumns = uint64_t(4 - firstKind) * width;
                    const uint64_t betaBase = wideColumns + 128;
                    const uint64_t total = betaBase + p.heads, blocks = (total + 15) / 16;
                    uint64_t cursor = (blocks * id / groups) * 16, stop = (blocks * (id + 1) / groups) * 16;
                    if (stop > total) {
                        stop = total;
                    }
                    while (cursor < stop) {
                        uint32_t kind = cursor < wideColumns ? firstKind + cursor / width : (cursor < betaBase ? 4 : 5);
                        uint64_t start =
                            kind < 4 ? uint64_t(kind - firstKind) * width : (kind == 4 ? wideColumns : betaBase);
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
                        if (kind == 4 && useDecayGemv(p.rows, p.hidden)) {
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
                                                 p.hidden, n);
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
                    for (uint32_t panel = id; panel < OutputPanels(p.hidden, 256); panel += groups) {
                        MatmulToGm<256, 256, 64>(ready, p.recurrence.outWeight, p.recurrence.output, 0, panel * 256,
                                                 first, panel * 256, p.hidden, m, width, PanelWidth(p.hidden, panel * 256, 256));
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
    template <int TileN = 128, int L1K = 256, int L0K = 64, int TileM = 256, bool Stream = false,
              uint32_t ExtraHead = 0>
    CATLASS_DEVICE void MatmulToGm(GM_ADDR input, GM_ADDR weight, GM_ADDR out, uint32_t xRow, uint32_t wCol,
                                   uint32_t outRow, uint32_t outCol, uint32_t stride, uint32_t m, uint32_t k,
                                   uint32_t n = 128, bool oneUseWeight = false)
    {
        AscendC::GlobalTensor<Bf> a, w, c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(input) + uint64_t(xRow) * k);
        w.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(weight) + uint64_t(wCol) * k);
        c.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(out) + uint64_t(outRow) * stride);
        if constexpr (MxQkv) {
            if ((Stream || oneUseWeight) && m <= TileM) {
                w.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
        }
        auto x = tla::MakeTensor(a, tla::MakeLayout<Bf, Row>(m, k), Catlass::Arch::PositionGM{});
        auto wt = tla::MakeTensor(w, tla::MakeLayout<Bf, Col>(k, n), Catlass::Arch::PositionGM{});
        auto base = tla::MakeTensor(c, tla::MakeLayout<Bf, Row>(m, stride), Catlass::Arch::PositionGM{});
        using GmCopy = std::conditional_t<Stream, StreamingOutputCopy<ExtraHead, L1K>, OutputCopy>;
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
