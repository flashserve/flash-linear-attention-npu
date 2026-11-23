/**
 * MegaRecurrentKDA v2 T02 (long sequence) Stage R driver.
 *
 * T02 owns every input in which at least one sequence crosses a 64-row tile
 * (docs/design.md section 3).  The stage structure is the T01 P -> R -> O chain;
 * what changes is the ownership of the R work units:
 *
 *   group  g = b*N + head  is owned by AIV  g % aivNum  for the whole call.
 *
 * An AIV therefore walks its `quota = ceil(B*N/aivNum)` groups through the
 * global tiles, always advancing the group whose next tile has the smallest
 * global tile number (the tile frontier).  A group's tiles are consecutive, so
 * the same AIV performs the whole recursion of one (b, head) sequence and the
 * H hand-off between two tiles of a group is an intra-core program-order edge
 * (write back to the workspace carry region, reload on the next tile) instead of
 * a cross-AIV notification.
 *
 * Small-S branch (docs/design/T02.md 2.2): when a sequence is at most one tile
 * long (S <= 64, so it spans at most two tiles) the group instead waits for the
 * P counters of *all* tiles covering the sequence and then runs the whole
 * sequence in one pass with H loaded once and kept in UB; the carry region is
 * never touched.  Only S > 64 uses the per-tile carry hand-off.  The branch is
 * what removes the per-boundary round trip for the decode-heavy S = 3/5/63
 * inputs; MK_SKIP_SYNC bit 26 forces the per-tile path for A/B measurement.
 *
 * Synchronisation (docs/design/T02.md section 6):
 *   - before a group consumes tile k, wait for count[P][k] == nProj;
 *   - after a group finished tile k, publish count[R][k] += 1; the host wrote
 *     units(k) = (number of sequences intersecting tile k) * N into the
 *     expectation region, so stage O starts a row block when both its tiles
 *     reached units(k).
 * Because P publication order per core is row-block order, tile readiness is
 * monotone in k, so waiting for the frontier tile never leaves a ready tile
 * unprocessed behind it (the design's "smallest ready group, else wait for the
 * smallest group" rule degenerates to "wait for the frontier tile").
 */
#ifndef MEGA_RECURRENT_KDA_V2_T02_H
#define MEGA_RECURRENT_KDA_V2_T02_H

#include "catlass/catlass.hpp"

#include "mega_recurrent_kda_v2_sync.h"
#include "mega_recurrent_kda_v2_tiling.h"
#include "mega_recurrent_kda_v2_vector.h"

namespace MegaRecurrentKdaV2 {

/// Diagnostic ablation bit (MK_SKIP_SYNC, never set in production): force every
/// group through the per-tile carry path, i.e. disable the S <= 64 branch, so its
/// cost can be compared against the default one-pass path.
constexpr uint32_t kSkipSmallS = 1u << 26;

/// T02 Stage R driver: fixed (b, head) ownership, forward tile cursor per group.
CATLASS_DEVICE void RunStageR02(V2Params const &p, AscendC::LocalTensor<uint8_t> const &ub)
{
    using namespace RDetail;
    const uint32_t rank = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum()) * 2u;
    AscendC::LocalTensor<int32_t> stg = ub[kUbStg].ReinterpretCast<int32_t>();
    // Scheduler state lives in UB (the AIV scalar stack is small): [cursor |
    // endTile] for up to kMaxGroupsPerAiv groups of this AIV.  Stored as int32
    // because the UB LocalTensor element type must be int32; -1 means "finished".
    AscendC::LocalTensor<int32_t> gt = ub[kUbGroup].ReinterpretCast<int32_t>();

    AscendC::GlobalTensor<int32_t> gmPrefix;
    gmPrefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.prefix));

    const uint32_t groups = p.batch * p.n;
    const uint32_t quota = CeilDiv(groups, aivNum);
    const uint32_t bail = (p.skipSync & 4u) != 0u ? 2000000u : 0u;
    const bool smallSEnabled = (p.skipSync & kSkipSmallS) == 0u;
    HeadConstants cached;
    uint32_t knownTile = 0u;      // highest tile already known to be P-ready
    bool haveKnown = false;

    // quota > kMaxGroupsPerAiv is handled by processing the owned groups in
    // chunks; each chunk runs the same frontier loop to completion.
    for (uint32_t base = 0u; base < quota; base += kMaxGroupsPerAiv) {
        const uint32_t remain = quota - base;
        const uint32_t cnt = remain < kMaxGroupsPerAiv ? remain : kMaxGroupsPerAiv;
        uint32_t live = 0u;
        for (uint32_t i = 0u; i < cnt; ++i) {
            const uint32_t g = rank + (base + i) * aivNum;
            if (g >= groups) {
                gt.SetValue(i, -1);  // not owned by this AIV
                gt.SetValue(kMaxGroupsPerAiv + i, 0);
                gt.SetValue(2u * kMaxGroupsPerAiv + i, 0);
                continue;
            }
            const uint32_t b = g / p.n;
            const uint32_t seqStart = static_cast<uint32_t>(gmPrefix.GetValue(b));
            const uint32_t seqEnd = static_cast<uint32_t>(gmPrefix.GetValue(b + 1u));
            gt.SetValue(i, static_cast<int32_t>(seqStart / kTileRows));  // cursor
            gt.SetValue(kMaxGroupsPerAiv + i,
                static_cast<int32_t>((seqEnd - 1u) / kTileRows));       // last tile
            gt.SetValue(2u * kMaxGroupsPerAiv + i,
                static_cast<int32_t>(seqStart / kTileRows));            // first tile
            ++live;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        while (live != 0u) {
            // Frontier: the owned group whose next tile has the smallest global
            // tile index.  Finished groups carry cursor -1 (=> 0xFFFFFFFFu).
            uint32_t best = 0xFFFFFFFFu;
            uint32_t tile = 0xFFFFFFFFu;
            for (uint32_t i = 0u; i < cnt; ++i) {
                const uint32_t cursor = static_cast<uint32_t>(gt.GetValue(i));
                if (cursor < tile) {
                    tile = cursor;
                    best = i;
                }
            }
            if (best == 0xFFFFFFFFu) {
                break;
            }
            const uint32_t g = rank + (base + best) * aivNum;
            const uint32_t b = g / p.n;
            const uint32_t head = g % p.n;
            const uint32_t lastTile = static_cast<uint32_t>(gt.GetValue(kMaxGroupsPerAiv + best));
            const uint32_t seqStart = static_cast<uint32_t>(gmPrefix.GetValue(b));
            const uint32_t seqEnd = static_cast<uint32_t>(gmPrefix.GetValue(b + 1u));

            // ---- small S branch (docs/design/T02.md 2.2, 6) -----------------
            // A sequence of at most one tile's worth of tokens spans at most two
            // tiles.  Wait for every covered tile, then run the whole sequence in
            // one pass: H is loaded once from checkpoint, stays in UB and is
            // never written to the carry region.  R credits are published once
            // for all covered tiles afterwards.
            if (smallSEnabled && (seqEnd - seqStart <= kTileRows)) {
                if (!haveKnown || lastTile > knownTile) {
                    if ((p.skipSync & 1u) == 0u) {
                        Sync::WaitEq(p.counts, CounterByteOffset(0u, lastTile, p.tiles),
                            static_cast<int32_t>(p.nProj), bail);
                    }
                    knownTile = lastTile;
                    haveKnown = true;
                }
                ProcessRows(p, ub, b, head, seqStart, seqEnd, seqStart, seqEnd, false, cached);
                const uint32_t firstTile =
                    static_cast<uint32_t>(gt.GetValue(2u * kMaxGroupsPerAiv + best));
                for (uint32_t k = firstTile; k <= lastTile; ++k) {
                    Sync::AivAdd(stg, p.counts, CounterByteOffset(1u, k, p.tiles), 1u);
                }
                gt.SetValue(best, -1);
                --live;
                continue;
            }

            // ---- S > 64: per-tile path with the H carry hand-off ------------
            // count[P][tile] == nProj; readiness is monotone in the tile index so
            // only a frontier advance needs a new wait.
            if (!haveKnown || tile > knownTile) {
                if ((p.skipSync & 1u) == 0u) {
                    Sync::WaitEq(p.counts, CounterByteOffset(0u, tile, p.tiles),
                        static_cast<int32_t>(p.nProj), bail);
                }
                knownTile = tile;
                haveKnown = true;
            }
            // One credit per group per tile: units(tile) = (#sequences in the
            // tile) * N exactly.  The sequence bounds were read once above.
            const uint32_t tileRow0 = tile * kTileRows;
            const uint32_t lo = seqStart > tileRow0 ? seqStart : tileRow0;
            const uint32_t tileRow1 = tileRow0 + kTileRows;
            const uint32_t hi = seqEnd < tileRow1 ? seqEnd : tileRow1;
            ProcessRows(p, ub, b, head, seqStart, seqEnd, lo, hi, true, cached);
            Sync::AivAdd(stg, p.counts, CounterByteOffset(1u, tile, p.tiles), 1u);
            if (tile == lastTile) {
                gt.SetValue(best, -1);
                --live;
            } else {
                gt.SetValue(best, static_cast<int32_t>(tile + 1u));
            }
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace MegaRecurrentKdaV2

#endif  // MEGA_RECURRENT_KDA_V2_T02_H
