/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH92_COMMON_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH92_COMMON_H

#include "../chunk_kda_bwd_recompute_common.h"
#include "../chunk_kda_bwd_recompute_struct.h"

namespace KdaBwdRecomputeArch92 {
using namespace AscendC;

struct WorkInfo {
    uint64_t id;
    uint64_t head;
    uint64_t qkOffset;
    uint64_t valueOffset;
    uint64_t betaOffset;
    uint64_t aOffset;
    uint32_t rows;
    uint32_t slot;
};

// Slot follows the LOCAL iteration, not id%2: an even grid stride must still
// alternate IN0/IN1 and A0/A1. Both engines use the same ordinal.
__aicore__ inline WorkInfo GetWork(
    const KDA::ChunkKdaBwdRecomputeTilingData &td, GM_ADDR cu, GM_ADDR indices,
    uint64_t id, uint64_t ordinal)
{
    WorkInfo work;
    work.id = id;
    work.head = id % td.Hv;
    work.slot = static_cast<uint32_t>(ordinal % kSlots);
    const uint64_t chunk = id / td.Hv;
    const uint64_t tokens = static_cast<uint64_t>(td.T);
    uint64_t batch = 0;
    uint64_t begin = 0;
    uint64_t end = tokens;
    if (td.isVariable) {
        GlobalTensor<int64_t> gmCu, gmIndices;
        gmCu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu));
        gmIndices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(indices));
        const uint64_t seq = gmIndices.GetValue(2 * chunk);
        const uint64_t seqChunk = gmIndices.GetValue(2 * chunk + 1);
        begin = gmCu.GetValue(seq) + seqChunk * kRows;
        end = gmCu.GetValue(seq + 1);
    } else {
        const uint64_t chunksPerBatch = (tokens + kRows - 1) / kRows;
        batch = chunk / chunksPerBatch;
        begin = (chunk % chunksPerBatch) * kRows;
    }
    work.rows = static_cast<uint32_t>(end - begin < kRows ? end - begin : kRows);
    work.betaOffset = (batch * td.Hv + work.head) * tokens + begin;
    work.valueOffset = work.betaOffset * kChannels;
    work.qkOffset = ((batch * td.Hk + work.head / td.hvPerHk) * tokens + begin) * kChannels;
    work.aOffset = work.betaOffset * kRows;
    return work;
}

__aicore__ inline uint32_t AOffset(uint32_t slot) { return slot * kL1ABytes; }
__aicore__ inline uint32_t KbgOffset(uint32_t slot) { return kL1RhsBase + slot * kL1RhsBytes; }
__aicore__ inline uint32_t VbOffset(uint32_t slot) { return KbgOffset(slot) + kL1VbRelative; }

template <pipe_t PIPE>
__aicore__ inline void WaitRhsReady(uint32_t slot)
{
    CrossCoreWaitFlag<0x4, PIPE>(static_cast<uint16_t>(kReadyFlag + 2 * slot));
}
template <pipe_t PIPE>
__aicore__ inline void SetRhsReady(uint32_t slot)
{
    CrossCoreSetFlag<0x4, PIPE>(static_cast<uint16_t>(kReadyFlag + 2 * slot));
}
template <pipe_t PIPE>
__aicore__ inline void WaitRhsFree(uint32_t slot)
{
    CrossCoreWaitFlag<0x4, PIPE>(static_cast<uint16_t>(kFreeFlag + 2 * slot));
}
template <pipe_t PIPE>
__aicore__ inline void SetRhsFree(uint32_t slot)
{
    CrossCoreSetFlag<0x4, PIPE>(static_cast<uint16_t>(kFreeFlag + 2 * slot));
}

template <typename T>
__aicore__ inline void CopyGmToUb(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(dst, src, p, pad);
}

// Full 64-row, BF16 ND -> zN. V0 explicitly writes zero to invalid RHS rows,
// so Cube always reads a completely initialized 64x128 tile, including tails.
template <typename T>
__aicore__ inline void CopyRhsToL1(LocalTensor<T> dst, LocalTensor<T> src)
{
    static_assert(sizeof(T) == 2, "RHS requires BF16");
    for (uint32_t col = 0; col < kChannels; col += 16) {
        DataCopy(dst[(col / 16) * kRows * 16], src[col], DataCopyParams(kRows, 1, 7, 0));
    }
}

} // namespace KdaBwdRecomputeArch92

#endif
