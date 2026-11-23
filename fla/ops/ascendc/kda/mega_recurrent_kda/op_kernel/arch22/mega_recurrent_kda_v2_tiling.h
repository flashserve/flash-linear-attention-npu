/**
 * MegaRecurrentKDA v2 tiling, task mapping and user-workspace layout.
 *
 * This header is shared by the host translation unit and the device kernel, so
 * the workspace layout the host allocates is byte-identical to the layout the
 * kernel addresses.  Nothing here is AscendC specific.
 *
 * v2 three-stage decomposition (see docs/design/T01.md):
 *   Stage P (AIC): five GEMMs -- qkv, gate_raw, beta_raw, decay_a, and the
 *                  chained decay_raw = decay_a(bf16) @ Wb^T -- for one row
 *                  block of up to 3 tiles (BP = 192 rows).
 *   Stage R (AIV): one (b, head) group per work unit: conv1d + silu for the
 *                  head's 384 components, L2 normalisation of q/k, the KDA
 *                  recursion (H resident in UB), RMSNorm + output gate, the
 *                  in-place conv_state update and the replay_* stores.
 *   Stage O (AIC): out = gated @ Wout^T for one row block of 2 tiles.
 *
 * Ring slots: the P->R intermediate rings use 3 slots (the main case has 3
 * tiles, so no wrap happens); the R->O `gated` ring uses 4 slots so that the
 * two consecutive tiles of an O row block are always adjacent in the ring and
 * a single GEMM can read them with one leading dimension.
 */
#ifndef MEGA_RECURRENT_KDA_V2_TILING_H
#define MEGA_RECURRENT_KDA_V2_TILING_H

#include <cstdint>

#ifdef __CCE_AICORE__
#define KDA_INLINE __aicore__ inline
#else
#define KDA_INLINE inline
#endif

namespace MegaRecurrentKdaV2 {

constexpr uint32_t kHeadDim = 128;      // D = K = V
constexpr uint32_t kHidden = 7168;      // H
constexpr uint32_t kTileRows = 64;      // tile = one ring slot = 64 tokens
constexpr uint32_t kBlockTiles = 3;     // P row block spans at most 3 tiles
constexpr uint32_t kBlockRows = kTileRows * kBlockTiles;  // BP = 192
constexpr uint32_t kPRSlotsMin = 3;     // P->R ring depth for the main case (3 tiles)
constexpr uint32_t kGatedSlotsMin = 4;  // gated ring depth for a 2-tile O block
constexpr uint32_t kObTiles = 2;        // O row block spans 2 tiles
constexpr uint32_t kNW = 128;           // logical N tile width of every GEMM
constexpr uint32_t kAlign = 64;
constexpr uint32_t kCounterWords = 8;   // 32 B atomic DMA unit per counter
constexpr uint32_t kStages = 3;         // P, R, O
constexpr uint32_t kConvWin = 3;        // 3 history rows fed into conv1d

// Template identifiers (docs/design.md 2 and 3).
//   T01: every sequence stays inside one 64-row tile.
//   T02: at least one sequence straddles a 64-row tile; the R stage owns whole
//        (b, head) groups and carries H across tiles.
constexpr uint32_t kTemplateT01 = 1;
constexpr uint32_t kTemplateT02 = 2;

KDA_INLINE uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1u) / align) * align;
}

KDA_INLINE uint64_t AlignUp64(uint64_t value, uint64_t align)
{
    return ((value + align - 1u) / align) * align;
}

KDA_INLINE uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

/// Problem description straight from the frozen API arguments.
struct KdaProblem {
    uint32_t batch;            // B
    uint32_t sequence;         // S (BSH). TH -> 0, the prefix array is used.
    uint32_t heads;            // N
    uint32_t blockNum;         // state pool size
    uint32_t convLength;       // L
    uint32_t replayCapacity;   // S capacity
    uint32_t tokens;           // T = token_count
    uint32_t tileRows;         // kTileRows

    uint32_t projectWidth() const { return heads * kHeadDim; }
};

/// Stage P's work unit: one 128-column projection block of one row block.
struct ProjectionBlock {
    // 0 .. nQkv-1        : qkv columns [128*t, +128)
    // nQkv .. nQkv+nGate-1: gate_raw columns
    // nQkv+nGate         : beta_raw (N columns)
    // nProj-1            : decay_a (D columns) then the decay_raw chain
    uint32_t index;
};

struct KdaTiling {
    uint32_t tokens;
    uint32_t tiles;
    uint32_t p;                 // P = N * D
    uint32_t hidden;            // H
    uint32_t n;                 // N
    uint32_t d;                 // D
    uint32_t pBlocks;           // S0 row blocks
    uint32_t pSplit;            // 1: one tile per P row block (R needs tile 0 early)
    uint32_t oBlocks;           // S2 row blocks
    uint32_t nQkv;
    uint32_t nGate;
    uint32_t nBeta;
    uint32_t nDecayRaw;
    uint32_t nProj;             // S0 publication expectation per tile
    uint32_t nOut;              // S2 publication expectation per tile
    uint32_t units;             // total S1 units = sum_k units(k)
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t templateId;        // kTemplateT01 or kTemplateT02
    uint32_t quota;             // T02: ceil(B*N/aivNum) groups per AIV
    uint32_t carryElems;        // B*N*D*D when sequences straddle tiles else 0
    uint32_t slotsPR;           // qkv/decay_raw/gate_raw/beta_raw ring depth
    uint32_t slotsGated;        // gated ring depth
    uint32_t overflow;          // 1 when the byte layout does not fit 32-bit offsets
    // Workspace layout (byte offsets).
    uint32_t offQkv;
    uint32_t offDecayRaw;
    uint32_t offGateRaw;
    uint32_t offBetaRaw;
    uint32_t offGated;
    uint32_t offDecayA;
    uint32_t offCounts;
    uint32_t offSyncAux;
    uint32_t offExpect;
    uint32_t offCarry;
    uint32_t workspaceBytes;
};

/// Byte sizes of the per-slot regions (single slot).
KDA_INLINE uint32_t QkvSlotBytes(uint32_t p) { return kTileRows * p * 3u * 2u; }
KDA_INLINE uint32_t GateSlotBytes(uint32_t p) { return kTileRows * p * 2u; }
KDA_INLINE uint32_t BetaSlotBytes(uint32_t n) { return kTileRows * n * 2u; }
KDA_INLINE uint32_t GatedSlotBytes(uint32_t p) { return kTileRows * p * 2u; }

/// Number of int32 words of the counter block: 3 stages x tiles x 8 words.
KDA_INLINE uint32_t CounterWords(uint32_t tiles) { return kStages * tiles * kCounterWords; }

/// Byte offset of counter (stage, tile) inside the counts region.
KDA_INLINE uint32_t CounterByteOffset(uint32_t stage, uint32_t tile, uint32_t tiles)
{
    return (stage * tiles + tile) * kCounterWords * static_cast<uint32_t>(sizeof(int32_t));
}

/// Expectations region: units[tiles] (uint32) + unitsPrefix[tiles+1] (uint32)
/// + firstSeq[tiles] (uint32).
KDA_INLINE uint32_t ExpectWords(uint32_t tiles) { return tiles + (tiles + 1u) + tiles; }

/// S1 units of one tile (device side, from the host-written arrays).
KDA_INLINE uint32_t UnitsOf(const uint32_t *units, uint32_t tile) { return units[tile]; }

/// O row block geometry: block j covers tiles [k0, k1) with k0 = j*kObTiles and
/// k1 = min(tiles, k0 + kObTiles); the corresponding row range is
/// [k0*kTileRows, min(tokens, k1*kTileRows)).
struct RowBlockRange {
    uint32_t tileBegin;
    uint32_t tileEnd;
    uint32_t rowBegin;
    uint32_t rowEnd;
};

KDA_INLINE RowBlockRange OBlockRange(const KdaTiling &t, uint32_t block)
{
    RowBlockRange r{};
    r.tileBegin = block * kObTiles;
    r.tileEnd = r.tileBegin + kObTiles < t.tiles ? r.tileBegin + kObTiles : t.tiles;
    r.rowBegin = r.tileBegin * kTileRows;
    const uint32_t rowEnd = r.tileEnd * kTileRows;
    r.rowEnd = rowEnd < t.tokens ? rowEnd : t.tokens;
    return r;
}

KDA_INLINE RowBlockRange PBlockRange(const KdaTiling &t, uint32_t block)
{
    RowBlockRange r{};
    if (t.pSplit != 0u) {
        r.tileBegin = block;
        r.tileEnd = block + 1u < t.tiles ? block + 1u : t.tiles;
    } else {
        r.tileBegin = block * kBlockTiles;
        r.tileEnd = r.tileBegin + kBlockTiles < t.tiles ? r.tileBegin + kBlockTiles : t.tiles;
    }
    r.rowBegin = r.tileBegin * kTileRows;
    const uint32_t rowEnd = r.tileEnd * kTileRows;
    r.rowEnd = rowEnd < t.tokens ? rowEnd : t.tokens;
    return r;
}

#ifndef __CCE_AICORE__
inline KdaTiling ComputeTiling(const KdaProblem &problem, uint32_t aicNum, uint32_t aivNum,
    bool straddle)
{
    KdaTiling t{};
    t.tokens = problem.tokens;
    t.tiles = CeilDiv(problem.tokens, kTileRows);
    t.n = problem.heads;
    t.d = kHeadDim;
    t.p = problem.projectWidth();
    t.hidden = kHidden;
    t.aicNum = aicNum;
    t.aivNum = aivNum;
    // Template selection (docs/design.md 3).  T02 owns every input in which a
    // sequence crosses a 64-row tile; T01 owns the rest.  The choice is a pure
    // function of (S, cu_seqlens) and is recorded in the tiling for the host
    // evidence line.

    t.templateId = straddle ? kTemplateT02 : kTemplateT01;
    t.quota = CeilDiv(problem.batch * problem.heads, aivNum);
    // P publishes one counter per tile.  For T01, if the three tiles of a
    // 192-row block are published together, stage R cannot start until the whole
    // block is done; with many R units that wait dominates, so the main case
    // splits P per tile.  T02 keeps the designed BP = 192 row block (the R groups
    // are self-paced and the P weight traffic is what matters at long S).
    if (straddle) {
        t.pSplit = 0u;
    } else {
        t.pSplit = (problem.batch * problem.heads > aivNum) ? 1u : 0u;
    }
    t.pBlocks = t.pSplit != 0u ? t.tiles : CeilDiv(t.tiles, kBlockTiles);
    t.oBlocks = CeilDiv(t.tiles, kObTiles);
    t.nQkv = CeilDiv(t.p * 3u, kNW);
    t.nGate = CeilDiv(t.p, kNW);
    t.nBeta = CeilDiv(t.n, kNW);
    t.nDecayRaw = CeilDiv(t.p, kNW);
    t.nProj = t.nQkv + t.nGate + t.nBeta;
    t.nOut = CeilDiv(kHidden, kNW);
    t.carryElems = straddle
        ? static_cast<uint32_t>(problem.batch) * t.n * kHeadDim * kHeadDim
        : 0u;
    // Ring depth: 3 (resp. 4) is enough whenever the whole tile set fits, which is
    // the main case.  For more tiles the rings grow with the tile count so that a
    // producer never has to wait for a consumer (the cross-stage path stays a
    // DAG and cannot deadlock); the extra workspace is documented in 5.
    t.slotsPR = t.tiles > kPRSlotsMin ? t.tiles : kPRSlotsMin;
    t.slotsGated = t.tiles > kGatedSlotsMin ? t.tiles : kGatedSlotsMin;

    // The layout is addressed with 32-bit byte offsets, so it has to stay below
    // 4 GiB; the offsets are accumulated in 64 bits and the overflow flag lets the
    // host reject inputs beyond the supported S instead of wrapping silently
    // (docs/design/T02.md 5, T02-scaling.md 6).
    uint64_t off = 0;
    auto take = [&off](uint64_t bytes, uint32_t &slot) {
        slot = static_cast<uint32_t>(off);
        off += AlignUp64(bytes, kAlign);
    };
    take(static_cast<uint64_t>(t.slotsPR) * QkvSlotBytes(t.p), t.offQkv);
    take(static_cast<uint64_t>(t.slotsPR) * GateSlotBytes(t.p), t.offDecayRaw);
    take(static_cast<uint64_t>(t.slotsPR) * GateSlotBytes(t.p), t.offGateRaw);
    take(static_cast<uint64_t>(t.slotsPR) * BetaSlotBytes(t.n), t.offBetaRaw);
    take(static_cast<uint64_t>(t.slotsGated) * GatedSlotBytes(t.p), t.offGated);
    take(static_cast<uint64_t>(kBlockRows) * kHeadDim * 2u, t.offDecayA);
    take(AlignUp64(static_cast<uint64_t>(CounterWords(t.tiles)) * 4u, 1024u), t.offCounts);
    take(1024u, t.offSyncAux);
    take(static_cast<uint64_t>(ExpectWords(t.tiles)) * 4u, t.offExpect);
    take(static_cast<uint64_t>(t.carryElems) * 4u, t.offCarry);
    t.overflow = (off > 0xFFFFFFFFull - kAlign) ? 1u : 0u;
    t.workspaceBytes = static_cast<uint32_t>(off);
    // units is filled by FillExpectations on the host; keep a placeholder here.
    t.units = 0;
    return t;
}

#endif
#undef KDA_INLINE
}  // namespace MegaRecurrentKdaV2

#endif  // MEGA_RECURRENT_KDA_V2_TILING_H
