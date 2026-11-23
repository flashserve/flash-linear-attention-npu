/**
 * MegaRecurrentKDA v2 cross-stage synchronisation primitives (Atlas A2).
 *
 * Only GM atomic counters are used; there is no global barrier and no
 * per-slot free handshake beyond the ring-depth guards.  Protocol:
 *
 *   publication: 32 B DMA with SetAtomicAdd<int32_t>()
 *     AIC: L1 -> GM (AIC has no UB and no scalar GM atomic; the L1->GM
 *          copy_cbuf_to_gm path is its only coherent GM write)
 *     AIV: UB -> GM
 *   observation: compiler barrier -> DataCacheCleanAndInvalid (dcci) ->
 *     direct __gm__ load -> compiler barrier.  A plain scalar load loop gets
 *     hoisted/cached and never observes the update, so the dcci read is
 *     mandatory (same protocol as AscendC's own cross-core GM sync).
 *
 * The flag region layout is documented in mega_recurrent_kda_v2_tiling.h:
 *   counts[stage][tile] = one 8-word (32 B) counter, stage 0=P 1=R 2=O.
 * A stage tile is ready exactly when its counter reaches the expected total
 * that the host derived from the shape.
 */
#ifndef MEGA_RECURRENT_KDA_V2_SYNC_H
#define MEGA_RECURRENT_KDA_V2_SYNC_H

#include "catlass/catlass.hpp"
#include "mega_recurrent_kda_v2_tiling.h"

namespace MegaRecurrentKdaV2 {
namespace Sync {

constexpr uint32_t kMaxAddPerDma = 8;  // aux constant blocks 1..8
// Poll backoff: one dcci read, then a growing number of cheap reads.
constexpr uint32_t kFirstSpin = 8u;
constexpr uint32_t kMaxSpin = 512u;

__aicore__ inline __gm__ int32_t *Word(GM_ADDR base, uint32_t byteOffset)
{
    return reinterpret_cast<__gm__ int32_t *>(base + byteOffset);
}

/// Read one counter with cross-core visibility.
__aicore__ inline int32_t Load(GM_ADDR base, uint32_t byteOffset)
{
    __gm__ int32_t *p = Word(base, byteOffset);
    __asm__ __volatile__("");
    AscendC::GlobalTensor<int32_t> t;
    t.SetGlobalBuffer(p);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(t);
    __asm__ __volatile__("");
    return *reinterpret_cast<__gm__ int32_t *>(p);
}

/// Cheap re-read with compiler barriers only (may be stale; used between dcci).
__aicore__ inline int32_t LoadLoose(GM_ADDR base, uint32_t byteOffset)
{
    __asm__ __volatile__("");
    const int32_t value = *reinterpret_cast<__gm__ volatile int32_t *>(Word(base, byteOffset));
    __asm__ __volatile__("");
    return value;
}

/// Spin until the counter reaches exactly `want` (counters only grow).
/// `bailAfter` > 0 bounds the poll rounds: it is only used by the diagnostic
/// MK_SKIP_SYNC=4 mode so a missed notification shows up as a wrong value in the
/// host-side counter dump instead of a kernel timeout.
__aicore__ inline void WaitEq(GM_ADDR base, uint32_t byteOffset, int32_t want,
    uint32_t bailAfter = 0u)
{
    uint32_t spin = 0u;
    uint32_t rounds = 0u;
    for (;;) {
        if (Load(base, byteOffset) == want) {
            return;
        }
        for (uint32_t i = 0; i < spin; ++i) {
            if (LoadLoose(base, byteOffset) == want) {
                return;
            }
        }
        if (bailAfter != 0u) {
            if (++rounds >= bailAfter) {
                return;
            }
        }
        if (spin < kMaxSpin) {
            spin = (spin == 0u) ? kFirstSpin : (spin * 2u > kMaxSpin ? kMaxSpin : spin * 2u);
        }
    }
}

// ---------------------------------------------------------------------------
// AIC publication: 32 B DMA L1 -> GM with atomic add.
// `l1` is a scratch LocalTensor inside the AIC L1 (only touched between GEMMs).
// The value is loaded from the aux GM constant block n = {n, 0 x7}; AIC cannot
// scalar-store into L1.
// ---------------------------------------------------------------------------
__aicore__ inline void AicLoadConst32(AscendC::LocalTensor<int32_t> const &l1, __gm__ int32_t *src)
{
    AscendC::GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(src);
    AscendC::DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = 1;  // one 32 B block
    params.srcStride = 0;
    params.dstStride = 0;
    AscendC::DataCopy(l1, g, params);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AicDma32(AscendC::LocalTensor<int32_t> const &l1, __gm__ int32_t *dst)
{
    AscendC::GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(dst);
    AscendC::DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    AscendC::SetAtomicAdd<int32_t>();
    AscendC::DataCopy(g, l1, params);
    AscendC::SetAtomicNone();
    AscendC::PipeBarrier<PIPE_ALL>();
}

/// counter += n via the aux constant blocks (n = 0 is a no-op).
__aicore__ inline void AicAdd(AscendC::LocalTensor<int32_t> const &l1, GM_ADDR constBase,
    GM_ADDR countBase, uint32_t byteOffset, uint32_t n)
{
    __gm__ int32_t *src = Word(constBase, 0);
    __gm__ int32_t *dst = Word(countBase, byteOffset);
    while (n > 0u) {
        const uint32_t chunk = n > kMaxAddPerDma ? kMaxAddPerDma : n;
        AicLoadConst32(l1, src + (chunk - 1u) * kCounterWords);
        AicDma32(l1, dst);
        n -= chunk;
    }
}

// ---------------------------------------------------------------------------
// AIV publication: 32 B DMA UB -> GM with atomic add.
// ---------------------------------------------------------------------------
__aicore__ inline void AivStageValue(AscendC::LocalTensor<int32_t> const &stg, int32_t value)
{
    for (uint32_t i = 0; i < kCounterWords; ++i) {
        stg.SetValue(i, i == 0u ? value : 0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AivDma32(AscendC::LocalTensor<int32_t> const &stg, __gm__ int32_t *dst)
{
    AscendC::GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(dst);
    AscendC::DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    AscendC::SetAtomicAdd<int32_t>();
    AscendC::DataCopy(g, stg, params);
    AscendC::SetAtomicNone();
    AscendC::PipeBarrier<PIPE_ALL>();
}

/// counter += n (n = 0 is a no-op).
__aicore__ inline void AivAdd(AscendC::LocalTensor<int32_t> const &stg, GM_ADDR countBase,
    uint32_t byteOffset, uint32_t n)
{
    __gm__ int32_t *dst = Word(countBase, byteOffset);
    while (n > 0u) {
        const uint32_t chunk = n > kMaxAddPerDma ? kMaxAddPerDma : n;
        AivStageValue(stg, static_cast<int32_t>(chunk));
        AivDma32(stg, dst);
        n -= chunk;
    }
}

}  // namespace Sync
}  // namespace MegaRecurrentKdaV2

#endif  // MEGA_RECURRENT_KDA_V2_SYNC_H
