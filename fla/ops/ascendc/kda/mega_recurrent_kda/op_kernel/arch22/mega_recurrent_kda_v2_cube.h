/**
 * MegaRecurrentKDA v2 Cube-side (AIC) stages.
 *
 * Stage P executes five GEMM chains for one row block (<= 3 tiles = 192 rows):
 *   qkv      [rows,3P] = hidden[rows,H]  @ qkv_w[3P,H]^T          (BF16 out)
 *   gate_raw [rows,P]  = hidden[rows,H]  @ gate_w[P,H]^T          (BF16 out)
 *   beta_raw [rows,N]  = hidden[rows,H]  @ beta_w[N,H]^T          (BF16 out)
 *   decay_a  [rows,D]  = hidden[rows,H]  @ decay_a_w[D,H]^T       (BF16 out)
 *   decay_raw[rows,P]  = decay_a[rows,D] @ decay_b_w[P,D]^T       (BF16 out)
 * One AIC produces decay_a while other AICs execute independent projections.
 * The decay_b column panels join the regular round-robin work queue.
 *
 * Every projection block of 128 output columns is one work unit.  Blocks are
 * handed out round-robin over the active AIC cores; after a core finished all
 * of its blocks for the row block it publishes its block count to every tile
 * counter covered by the row block, so the per-tile counter total is exactly
 * nProj for every tile.
 *
 * Stage O executes out = gated @ Wout^T for one row block of 2 tiles (128 rows)
 * after the corresponding R tiles are ready.
 *
 * The regular projections use a 192 x 128 x 256 L1 tile and one
 * Catlass resource per launch; the runtime (M,N,K) are passed per call, so no
 * host-side repacking is needed:
 *   A: RowMajor with leading dimension ldA
 *   B: the weight [cols, K] is consumed as [K, cols] ColumnMajor (ld = K)
 *   C: RowMajor with leading dimension ldC
 */
#ifndef MEGA_RECURRENT_KDA_V2_CUBE_H
#define MEGA_RECURRENT_KDA_V2_CUBE_H

#include "catlass/catlass.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "mega_recurrent_kda_v2_sync.h"
#include "mega_recurrent_kda_v2_tiling.h"

namespace MegaRecurrentKdaV2 {

/// One GEMM assembly for every projection and for the output projection.
struct KdaGemm {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementA = bfloat16_t;
    using ElementB = bfloat16_t;
    using ElementC = bfloat16_t;
    using LayoutTagA = Catlass::layout::RowMajor;
    using LayoutTagB = Catlass::layout::ColumnMajor;
    using LayoutTagC = Catlass::layout::RowMajor;

    static constexpr bool kEnableUnitFlag = true;
    static constexpr bool kUseHF32 = false;
    static constexpr uint32_t kL1M = 192;
    static constexpr uint32_t kL1N = 128;
    static constexpr uint32_t kL1K = 256;
    static constexpr uint32_t kL1AStages = 2;
    static constexpr uint32_t kL1BStages = 2;
    static constexpr uint32_t kL0AStages = 2;
    static constexpr uint32_t kL0BStages = 2;

    using DispatchPolicy =
        Catlass::Gemm::MmadPingpong<ArchTag, kEnableUnitFlag, kUseHF32, 1 /*L0C stages*/, false /*L1 resident*/,
                                    kL1AStages, kL1BStages, kL0AStages, kL0BStages>;
    using L1TileShape = tla::Shape<tla::Int<kL1M>, tla::Int<kL1N>, tla::Int<kL1K>>;
    using L0TileShape = tla::Shape<tla::Int<kL1M>, tla::Int<kL1N>, tla::_64>;

    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
                                                            ElementC, LayoutTagC>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB,
                                                         ElementC, void, TileCopy>;

    using LayoutA = typename TileCopy::LayoutA;
    using LayoutB = typename TileCopy::LayoutB;
    using LayoutC = typename TileCopy::LayoutC;

    CATLASS_DEVICE static LayoutA MakeLayoutA(uint32_t rows, uint32_t k, int64_t ldA)
    {
        return tla::MakeLayout(tla::MakeShape(rows, k), tla::MakeStride(ldA, tla::Int<1>{}));
    }
    CATLASS_DEVICE static LayoutB MakeLayoutB(uint32_t k, uint32_t cols, int64_t ldB)
    {
        return tla::MakeLayout(tla::MakeShape(k, cols), tla::MakeStride(tla::Int<1>{}, ldB));
    }
    CATLASS_DEVICE static LayoutC MakeLayoutC(uint32_t rows, uint32_t cols, int64_t ldC)
    {
        return tla::MakeLayout(tla::MakeShape(rows, cols), tla::MakeStride(ldC, tla::Int<1>{}));
    }
};

/// One GEMM: C[rows, cols) = A[rows, K) @ B[K, cols) with A/B/C already resolved
/// to GM addresses and element leading dimensions.
CATLASS_DEVICE void RunGemm(Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource, GM_ADDR ptrA, uint32_t rows,
                            int64_t ldA, GM_ADDR ptrB, uint32_t cols, int64_t ldB, GM_ADDR ptrC, int64_t ldC,
                            uint32_t k)
{
    typename KdaGemm::BlockMmad blockMmad(resource);

    AscendC::GlobalTensor<bfloat16_t> gmA;
    gmA.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ptrA));
    AscendC::GlobalTensor<bfloat16_t> gmB;
    gmB.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ptrB));
    AscendC::GlobalTensor<bfloat16_t> gmC;
    gmC.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ptrC));

    auto tensorA = tla::MakeTensor(gmA, KdaGemm::MakeLayoutA(rows, k, ldA), Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(gmB, KdaGemm::MakeLayoutB(k, cols, ldB), Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(gmC, KdaGemm::MakeLayoutC(rows, cols, ldC), Catlass::Arch::PositionGM{});

    blockMmad(tensorA, tensorB, tensorC, Catlass::GemmCoord(rows, cols, k));
}

/// GEMM with the block-mmad object kept alive across the whole core's work of a
/// row block (no L1/L0 reconfiguration between projections).
template <typename Gemm>
CATLASS_DEVICE void RunGemmReuse(Gemm &blockMmad, GM_ADDR ptrA, uint32_t rows, int64_t ldA,
                                 GM_ADDR ptrB, uint32_t cols, int64_t ldB, GM_ADDR ptrC, int64_t ldC, uint32_t k, bool bypassB = false)
{
    AscendC::GlobalTensor<bfloat16_t> gmA;
    gmA.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ptrA));
    AscendC::GlobalTensor<bfloat16_t> gmB;
    gmB.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ptrB));
    if (bypassB) gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    AscendC::GlobalTensor<bfloat16_t> gmC;
    gmC.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ptrC));

    auto tensorA = tla::MakeTensor(gmA, KdaGemm::MakeLayoutA(rows, k, ldA), Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(gmB, KdaGemm::MakeLayoutB(k, cols, ldB), Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(gmC, KdaGemm::MakeLayoutC(rows, cols, ldC), Catlass::Arch::PositionGM{});

    blockMmad(tensorA, tensorB, tensorC, Catlass::GemmCoord(rows, cols, k));
}

/// Parameters shared by the whole fused v2 launch; the stages read what they
/// need.  All addresses are raw GM addresses (API tensors or workspace).
struct V2Params {
    uint64_t convStride, checkpointStride, uStride, keyStride, decayStride;
    uint32_t convLayout;
    // problem geometry
    uint32_t tokens;
    uint32_t tiles;
    uint32_t n;      // N
    uint32_t p;      // P = N*D
    uint32_t hidden; // H
    uint32_t batch;
    uint32_t sequence; // S (BSH), 0 for TH
    uint32_t blockNum;
    uint32_t convLength;
    uint32_t replayCapacity;
    uint32_t pBlocks;
    uint32_t pSplit;
    uint32_t oBlocks;
    uint32_t nQkv;
    uint32_t nGate;
    uint32_t nBeta;
    uint32_t nDecayRaw;
    uint32_t nProj;
    uint32_t nOut;
    uint32_t slotsPR;
    uint32_t slotsGated;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t templateId; // kTemplateT01 / kTemplateT02 (R stage ownership)
    // API inputs
    GM_ADDR hiddenStates;
    GM_ADDR qkvWeight;
    GM_ADDR weightScale;
    GM_ADDR decayAWeight;
    GM_ADDR decayBWeight;
    GM_ADDR betaWeight;
    GM_ADDR gateWeight;
    GM_ADDR wOut;
    GM_ADDR convWeight;
    GM_ADDR convState;
    GM_ADDR checkpoint;
    GM_ADDR replayU;
    GM_ADDR replayK;
    GM_ADDR replayDecay;
    GM_ADDR stateIndices;
    GM_ADDR accepted;
    GM_ADDR prefix;
    uint32_t prefix64;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR gamma;
    // API output
    GM_ADDR out;
    // user workspace
    GM_ADDR wQkv;
    GM_ADDR wDecayRaw;
    GM_ADDR wGateRaw;
    GM_ADDR wBetaRaw;
    GM_ADDR wGated;
    GM_ADDR counts;
    GM_ADDR syncConst;
    GM_ADDR expect; // number of completed sequence/head slices required for each tile
    // scalars
    float scale;
    float lowerBound;
    float rmsNormEps;
    // profiling knobs (MK_PROFILE stage mask + MK_SKIP_SYNC), default all/sync
    uint32_t stageMask;
    uint32_t skipSync;
    bool wideInt8Panels = false;
    uint32_t rowGateOffset = 0;
};

CATLASS_DEVICE uint32_t SequenceStart(const V2Params &p, uint32_t b)
{
    if (p.sequence) return b * p.sequence;
    if (p.prefix64) return static_cast<uint32_t>(reinterpret_cast<__gm__ const int64_t *>(p.prefix)[b]);
    return static_cast<uint32_t>(reinterpret_cast<__gm__ const int32_t *>(p.prefix)[b]);
}

constexpr uint32_t kStageP = 1u << 0;
constexpr uint32_t kStageR = 1u << 1;
constexpr uint32_t kStageO = 1u << 2;
constexpr uint32_t kStageAll = kStageP | kStageR | kStageO;

// ---------------------------------------------------------------------------
// Stage P
// ---------------------------------------------------------------------------
enum class ProjectionPhase { All, Independent, DecayB };

template <ProjectionPhase phase, typename Gemm, uint32_t panel, bool NarrowAux = false>
CATLASS_DEVICE void RunProjectionPanels(Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource, V2Params const &p,
                                       uint32_t blockIndex)
{
    const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t coreNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    const uint32_t qkvPanels = CeilDiv(p.nQkv, panel / kNW), gatePanels = CeilDiv(p.nGate, panel / kNW);
    const uint32_t regularPanels = qkvPanels + gatePanels + p.nBeta;
    const uint32_t first = phase == ProjectionPhase::DecayB ? regularPanels : 0;
    const uint32_t end = phase == ProjectionPhase::Independent ? regularPanels : regularPanels + p.nDecayRaw;
    const uint32_t jobs = end - first + (phase == ProjectionPhase::Independent ? 1u : 0u);
    const uint32_t active = coreNum < jobs ? coreNum : jobs;

    KdaTiling t{};
    t.tiles = p.tiles;
    t.tokens = p.tokens;
    t.pSplit = p.pSplit;
    const RowBlockRange range = PBlockRange(t, blockIndex);
    if (range.rowBegin >= range.rowEnd) {
        return;
    }
    const uint32_t rows = range.rowEnd - range.rowBegin;
    const uint32_t localRow0 = range.tileBegin * kTileRows;

    AscendC::LocalTensor<int32_t> constBase = resource.l1Buf.GetBufferByByte<int32_t>(0);

    if constexpr (phase == ProjectionPhase::Independent) {
        if (jobs < coreNum && coreNum >= 3 && p.n <= kNW) {
            // Use the idle Cube cores on independent, aligned columns of gate and decay-A.
            const uint32_t share = (coreNum - 1) * kHeadDim / (p.p + kHeadDim);
            const uint32_t decayCores = share ? share : 1u;
            const uint32_t gateCores = coreNum - decayCores - 1;
            const bool gate = core < gateCores;
            const bool beta = core == coreNum - 1;
            const uint32_t columns = gate ? p.p : kHeadDim;
            const uint32_t team = gate ? gateCores : decayCores;
            const uint32_t rank = gate ? core : core - gateCores;
            const uint32_t col = beta ? 0 : uint64_t(columns / 16) * rank / team * 16;
            const uint32_t limit = beta ? p.n : uint64_t(columns / 16) * (rank + 1) / team * 16;
            const uint32_t stride = beta ? p.n : columns;
            GM_ADDR weight = beta ? p.betaWeight : (gate ? p.gateWeight : p.decayAWeight);
            GM_ADDR dest = beta ? p.wBetaRaw : (gate ? p.wGateRaw : p.wGated);
            if (col < limit) {
                if (NarrowAux && rows <= 64 && limit - col <= 64) {
                    using Narrow = Catlass::Gemm::Block::BlockMmadTla<typename KdaGemm::DispatchPolicy,
                        tla::Shape<tla::_64, tla::_64, tla::_768>, tla::Shape<tla::_64, tla::_64, tla::_256>,
                        bfloat16_t, bfloat16_t, bfloat16_t, void, typename KdaGemm::TileCopy>;
                    Narrow multiply(resource);
                    RunGemmReuse(multiply, p.hiddenStates + uint64_t(range.rowBegin) * p.hidden * 2, rows, p.hidden,
                                 weight + uint64_t(col) * p.hidden * 2, limit - col, p.hidden,
                                 dest + (uint64_t(localRow0) * stride + col) * 2, stride, p.hidden);
                } else {
                typename KdaGemm::BlockMmad multiply(resource);
                RunGemmReuse(multiply, p.hiddenStates + uint64_t(range.rowBegin) * p.hidden * 2, rows, p.hidden,
                             weight + uint64_t(col) * p.hidden * 2, limit - col, p.hidden,
                             dest + (uint64_t(localRow0) * stride + col) * 2, stride, p.hidden);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
    }
    if (core >= active) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    uint32_t credits = 0u;
    Gemm blockMmad(resource);
    const uint32_t threeP = 3u * p.p;
    GM_ADDR aPtr = p.hiddenStates + static_cast<size_t>(range.rowBegin) * p.hidden * 2u;

    // The gated buffer belongs to this row block until R consumes its P credits.
    GM_ADDR decayA = p.wGated + static_cast<size_t>(localRow0) * p.p * 2u;
    const uint32_t decayReadyOffset = CounterByteOffset(2u, blockIndex, p.tiles);
    if (phase != ProjectionPhase::DecayB && core == regularPanels % active) {
        RunGemmReuse(blockMmad, aPtr, rows, p.hidden, p.decayAWeight, kHeadDim, p.hidden, decayA, kHeadDim, p.hidden);
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (phase == ProjectionPhase::All) {
            Sync::AicAdd(constBase, p.syncConst, p.counts, decayReadyOffset, 1u);
        }
    }
    bool decayReady = false;
    for (uint32_t block = first + core; block < end; block += active) {
        ++credits;
        if (block < qkvPanels) {
            const uint32_t col = block * panel;
            const uint32_t width = (col + panel <= threeP) ? panel : (threeP - col);
            credits += CeilDiv(width, kNW) - 1;
            RunGemmReuse(blockMmad, aPtr, rows, p.hidden, p.qkvWeight + static_cast<size_t>(col) * p.hidden * 2u, width,
                         p.hidden, p.wQkv + (static_cast<size_t>(localRow0) * threeP + col) * 2u, threeP, p.hidden);
        } else if (block < qkvPanels + gatePanels) {
            const uint32_t col = (block - qkvPanels) * panel;
            const uint32_t width = (col + panel <= p.p) ? panel : (p.p - col);
            credits += CeilDiv(width, kNW) - 1;
            RunGemmReuse(blockMmad, aPtr, rows, p.hidden, p.gateWeight + static_cast<size_t>(col) * p.hidden * 2u,
                         width, p.hidden, p.wGateRaw + (static_cast<size_t>(localRow0) * p.p + col) * 2u, p.p,
                         p.hidden);
        } else if (block < regularPanels) {
            const uint32_t i = block - (qkvPanels + gatePanels);
            const uint32_t col = i * kNW;
            const uint32_t width = (col + kNW <= p.n) ? kNW : (p.n - col);
            RunGemmReuse(blockMmad, aPtr, rows, p.hidden, p.betaWeight + static_cast<size_t>(col) * p.hidden * 2u,
                         width, p.hidden, p.wBetaRaw + (static_cast<size_t>(localRow0) * p.n + col) * 2u, p.n,
                         p.hidden);
        } else {
            if (phase == ProjectionPhase::All && !decayReady) {
                Sync::WaitEq(p.counts, decayReadyOffset, 1);
                decayReady = true;
            }
            const uint32_t col = (block - regularPanels) * kNW;
            RunGemmReuse(blockMmad, decayA, rows, kHeadDim, p.decayBWeight + static_cast<size_t>(col) * kHeadDim * 2u,
                         kNW, kHeadDim, p.wDecayRaw + (static_cast<size_t>(localRow0) * p.p + col) * 2u, p.p, kHeadDim);
        }
    }

    AscendC::PipeBarrier<PIPE_ALL>();
    if constexpr (phase != ProjectionPhase::All) {
        return;
    }
    for (uint32_t tile = range.tileBegin; tile < range.tileEnd; ++tile) {
        Sync::AicAdd(constBase, p.syncConst, p.counts, CounterByteOffset(0u, tile, p.tiles), credits);
    }
}

template <ProjectionPhase phase = ProjectionPhase::All, bool NarrowAux = false>
CATLASS_DEVICE void RunStageP(Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource, V2Params const &p,
                              uint32_t blockIndex)
{
    if constexpr (phase == ProjectionPhase::All) {
        if (!p.weightScale && p.tokens <= 128 && p.nQkv >= 2 * p.aicNum) {
            // There are enough output columns to double reuse of each loaded hidden tile.
            using Wide = Catlass::Gemm::Block::BlockMmadTla<typename KdaGemm::DispatchPolicy,
                tla::Shape<tla::_128, tla::_256, tla::_256>, tla::Shape<tla::_128, tla::_256, tla::_64>,
                bfloat16_t, bfloat16_t, bfloat16_t, void, typename KdaGemm::TileCopy>;
            RunProjectionPanels<phase, Wide, 256>(resource, p, blockIndex);
            return;
        }
    }
    RunProjectionPanels<phase, KdaGemm::BlockMmad, kNW, NarrowAux>(resource, p, blockIndex);
}

// ---------------------------------------------------------------------------
// Stage O
// ---------------------------------------------------------------------------
template <typename Gemm, uint32_t panel>
CATLASS_DEVICE void RunOutputPanels(Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
                                   const V2Params &p, GM_ADDR gated, GM_ADDR out, uint32_t rows,
                                   uint32_t core, uint32_t coreNum)
{
    Gemm multiply(resource);
    for (uint32_t col = core * panel; col < p.hidden; col += panel * coreNum) {
        const uint32_t width = p.hidden - col < panel ? p.hidden - col : panel;
        RunGemmReuse(multiply, gated, rows, p.p, p.wOut + uint64_t(col) * p.p * 2, width, p.p,
                     out + col * 2, p.hidden, p.p);
    }
}

CATLASS_DEVICE void RunStageO(Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource, V2Params const &p,
                              uint32_t blockIndex)
{
    const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t coreNum = static_cast<uint32_t>(AscendC::GetBlockNum());

    KdaTiling t{};
    t.tiles = p.tiles;
    t.tokens = p.tokens;
    const RowBlockRange range = OBlockRange(t, blockIndex);
    if (range.rowBegin >= range.rowEnd) {
        return;
    }
    const uint32_t rows = range.rowEnd - range.rowBegin;

    if ((p.skipSync & 2u) == 0u) {
        AscendC::GlobalTensor<uint32_t> gmUnits;
        gmUnits.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(p.expect));
        const uint32_t bail = (p.skipSync & 4u) != 0u ? 2000000u : 0u;
        for (uint32_t tile = range.tileBegin; tile < range.tileEnd; ++tile) {
            const uint32_t want = gmUnits.GetValue(tile);
            Sync::WaitEq(p.counts, CounterByteOffset(1u, tile, p.tiles), static_cast<int32_t>(want), bail);
        }
    }

    // gated ring: the two tiles of the block are always adjacent
    const uint32_t slotBase = (range.tileBegin % p.slotsGated);
    GM_ADDR gatedPtr = p.wGated + static_cast<size_t>(slotBase) * kTileRows * p.p * 2u;
    GM_ADDR outPtr = p.out + static_cast<size_t>(range.rowBegin) * p.hidden * 2u;

    if (p.weightScale) {
        constexpr uint32_t panel = 256;
        using Gemm = Catlass::Gemm::Block::BlockMmadTla<typename KdaGemm::DispatchPolicy,
            tla::Shape<tla::_128, tla::Int<panel>, tla::_256>, tla::Shape<tla::_128, tla::Int<panel>, tla::_64>,
            bfloat16_t, bfloat16_t, bfloat16_t, void, typename KdaGemm::TileCopy>;
        RunOutputPanels<Gemm, panel>(resource, p, gatedPtr, outPtr, rows, core, coreNum);
    } else {
        RunOutputPanels<KdaGemm::BlockMmad, kNW>(resource, p, gatedPtr, outPtr, rows, core, coreNum);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace MegaRecurrentKdaV2

#endif // MEGA_RECURRENT_KDA_V2_CUBE_H
