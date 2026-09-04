/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * GM BNSD offsets, chunk walk, and on-chip byte offsets.
 */

#ifndef CHUNK_GDN_FWD_PREPARE_OFFSET_H
#define CHUNK_GDN_FWD_PREPARE_OFFSET_H

#include "common.h"
#include "mem.h"

namespace GdnStage {

constexpr uint32_t kPrepareKb = 1024;
constexpr uint32_t kSlotBf16_64 = 64 * 64 * static_cast<uint32_t>(sizeof(bfloat16_t));
constexpr uint32_t kSlotFp32_64 = 64 * 64 * static_cast<uint32_t>(sizeof(float));
constexpr uint32_t kVecFp32 = 64 * static_cast<uint32_t>(sizeof(float));
constexpr uint32_t kMaskBitsBytes = 64 * 64 / 8;

// UB resident
constexpr uint32_t kUbMaskBits = 0;
constexpr uint32_t kUbVcsIdx = 512;
constexpr uint32_t kUbIVcs = 1 * kPrepareKb;
constexpr uint32_t kUbGPrime0 = 9 * kPrepareKb;
constexpr uint32_t kUbBetaEff0 = 9 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbGPrime1 = 9 * kPrepareKb + 2 * kVecFp32;
constexpr uint32_t kUbBetaEff1 = 9 * kPrepareKb + 3 * kVecFp32;

// UB S0 temps (released before S1 writes g')
constexpr uint32_t kUbS0INz8 = 9 * kPrepareKb; // unused; [9,25) kept so later UB map stays put
constexpr uint32_t kUbS0Zero = 25 * kPrepareKb;

// UB S1 input db [10, 75). One AIV, two tasks: task 0/1 → ping, task 2/3 → pong.
constexpr uint32_t kUbS1QPing = 10 * kPrepareKb;
constexpr uint32_t kUbS1KPing = 26 * kPrepareKb;
constexpr uint32_t kUbS1QPong = 42 * kPrepareKb;
constexpr uint32_t kUbS1KPong = 58 * kPrepareKb;
constexpr uint32_t kUbS1GPing = 74 * kPrepareKb;
constexpr uint32_t kUbS1BetaPing = 74 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbS1GPong = 74 * kPrepareKb + 2 * kVecFp32;
constexpr uint32_t kUbS1BetaPong = 74 * kPrepareKb + 3 * kVecFp32;
// Hat ping [75, 107), pong in the hole before kkt [108, 140).
constexpr uint32_t kUbS1QHatPing = 75 * kPrepareKb;
constexpr uint32_t kUbS1KHatPing = 91 * kPrepareKb;
constexpr uint32_t kUbS1QHatPong = 108 * kPrepareKb;
constexpr uint32_t kUbS1KHatPong = 124 * kPrepareKb;
// Rstd ping next to ping k'; pong reuses leftover S1 g/beta [74, 74.5).
constexpr uint32_t kUbS1KRstdPing = 107 * kPrepareKb;
constexpr uint32_t kUbS1QRstdPing = 107 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbS1KRstdPong = 74 * kPrepareKb;
constexpr uint32_t kUbS1QRstdPong = 74 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbS1GateTmp = 107 * kPrepareKb + 2 * kVecFp32;
constexpr uint32_t kUbS1QHat[2] = {kUbS1QHatPing, kUbS1QHatPong};
constexpr uint32_t kUbS1KHat[2] = {kUbS1KHatPing, kUbS1KHatPong};
constexpr uint32_t kUbS1KRstd[2] = {kUbS1KRstdPing, kUbS1KRstdPong};
constexpr uint32_t kUbS1QRstd[2] = {kUbS1QRstdPing, kUbS1QRstdPong};

constexpr uint32_t kUbGPrime[2] = {kUbGPrime0, kUbGPrime1};
constexpr uint32_t kUbBetaEff[2] = {kUbBetaEff0, kUbBetaEff1};
constexpr uint32_t kUbS1Q[2] = {kUbS1QPing, kUbS1QPong};
constexpr uint32_t kUbS1K[2] = {kUbS1KPing, kUbS1KPong};
constexpr uint32_t kUbS1G[2] = {kUbS1GPing, kUbS1GPong};
constexpr uint32_t kUbS1Beta[2] = {kUbS1BetaPing, kUbS1BetaPong};

// UB S2
constexpr uint32_t kUbS2KktPing = 140 * kPrepareKb;
constexpr uint32_t kUbS2KktPong = 156 * kPrepareKb;
constexpr uint32_t kUbS2Kkt[2] = {kUbS2KktPing, kUbS2KktPong};

// UB S3 (overlaps S1 input after k' is on L1/GM). Ping db=0 / pong db=1.
// Pong reuses the old Nz16/Nz8/LeafTmp hole [50, 90).
constexpr uint32_t kUbS3LPacked[2] = {10 * kPrepareKb, 50 * kPrepareKb};
constexpr uint32_t kUbS3ResVcs[2] = {18 * kPrepareKb, 58 * kPrepareKb};
constexpr uint32_t kUbS3LFull[2] = {34 * kPrepareKb, 74 * kPrepareKb};
constexpr uint32_t kUbMaskFp32 = 172 * kPrepareKb;

// UB S6 (after S3; overlaps S1/S3). K' 16 KiB. V=128 tile is 16 KiB,
// V=256 tile is 32 KiB. After Stage3, rstd/hat at [107,140) are free.
// V=128: ping [64,80) pong [80,96). V=256: ping [64,96) pong [96,128).
constexpr uint32_t kUbS6KPing = 32 * kPrepareKb;
constexpr uint32_t kUbS6KPong = 48 * kPrepareKb;
constexpr uint32_t kUbS6VPing = 64 * kPrepareKb;
constexpr uint32_t kUbS6VPong = 80 * kPrepareKb;
constexpr uint32_t kUbS6VPong256 = 96 * kPrepareKb;
constexpr uint32_t kUbS6K[2] = {kUbS6KPing, kUbS6KPong};
constexpr uint32_t kUbS6V[2] = {kUbS6VPing, kUbS6VPong};

// L1
constexpr uint32_t kBytesA64 = 64 * 64 * 2;
constexpr uint32_t kBytesK128 = 64 * 128 * 2;
constexpr uint32_t kBytesVb256 = 64 * 256 * 2;
constexpr uint32_t kBytesFp32Nz64 = 64 * 64 * 4;
constexpr uint32_t kWsPerCoreBytes = 128 * kPrepareKb;
constexpr uint32_t kWsYElems = kChunk64 * kChunk64;
constexpr uint32_t kWsYBytes = 16 * kPrepareKb;
// One 64x64 bf16 ND tile is 8 KiB; slot is 16 KiB. Four slots so pack
// tasks 0..3 do not share gmWsA (Stage5 Fixpipe vs in-flight MTE2 Copy).
constexpr uint32_t kWsABytes = 16 * kPrepareKb;
constexpr uint32_t kWsAElems = kWsABytes / 2;
constexpr uint32_t kWsASlots = 4;
constexpr uint32_t kWsATotalBytes = kWsASlots * kWsABytes;
constexpr uint32_t kWsYPerCoreBytes = 64 * kPrepareKb;

__aicore__ inline int64_t WsAOffset(int64_t taskIdx)
{
    return taskIdx * static_cast<int64_t>(kWsAElems);
}

// Y owns [0, 64). k' aliases NegL at [64, 128): 64x128 bf16 == 64x64 fp32 NZ.
// Intra-pack: Stage2 consumes k', Stage3 overwrites the same slots with -L.
// Cross-pack: pack N Stage5 reads Y while pack N+1 Stage1 writes k' (NegL
// already consumed in Stage4). Gate is Wait Stage4, not Stage5.
constexpr uint32_t kL1Y0 = 0;
constexpr uint32_t kL1NegL0 = 64 * kPrepareKb;
constexpr uint32_t kL1KHat0 = kL1NegL0;
constexpr uint32_t kL1LeafRight0 = 128 * kPrepareKb;
constexpr uint32_t kL1LeafLeft0 = 192 * kPrepareKb;
constexpr uint32_t kL1ResidentA0 = 256 * kPrepareKb;
constexpr uint32_t kL1ResidentKbg0 = 288 * kPrepareKb;
constexpr uint32_t kL1ResidentVb0 = 352 * kPrepareKb;
constexpr uint32_t kL1ResidentI = 480 * kPrepareKb;
// [496, 512) was Zero@Zero L1. Unused; layout kept so later L1 map stays put.
constexpr uint32_t kL1ResidentZero = 496 * kPrepareKb;

// L0 (same physical banks, typed views)
constexpr uint32_t kL0Bf16Pair = 16 * kPrepareKb;
constexpr uint32_t kL0Fp32Pair = 16 * kPrepareKb;
// L0A/L0B 16 KiB slots. Stage4/5 second MMAD (fp32) and Stage7 (bf16)
// share even [32, 48) / odd [48, 64). Stage4/5 drain before Stage7.
// SetSize 16 KiB so a view at 32 does not cover the slot at 48.
constexpr uint32_t kL0S7Slot = 16 * kPrepareKb;
constexpr uint32_t kL0S7Ping = 32 * kPrepareKb;
constexpr uint32_t kL0S7Pong = 48 * kPrepareKb;
constexpr uint32_t kL0C1 = 64 * kPrepareKb;

__aicore__ inline uint32_t L1KHat(uint32_t taskIdx)
{
    return kL1KHat0 + taskIdx * kBytesK128;
}

__aicore__ inline uint32_t L1Y(uint32_t taskIdx)
{
    return kL1Y0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1NegL(uint32_t taskIdx)
{
    return kL1NegL0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1LeafRight(uint32_t taskIdx)
{
    return kL1LeafRight0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1LeafLeft(uint32_t taskIdx)
{
    return kL1LeafLeft0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1ResidentA(uint32_t taskIdx)
{
    return kL1ResidentA0 + taskIdx * kBytesA64;
}

__aicore__ inline uint32_t L1ResidentKbg(uint32_t taskIdx)
{
    return kL1ResidentKbg0 + taskIdx * kBytesK128;
}

__aicore__ inline uint32_t L1ResidentVb(uint32_t taskIdx)
{
    return kL1ResidentVb0 + taskIdx * kBytesVb256;
}

// One BT-token window on a single sequence (not including hv).
// chunkIdx in GetChunkRange is workId / HV.
struct ChunkRange {
    // BNSD dim-0. Varlen packed layout is always 0: sequences sit on T.
    int64_t batch;
    // Which sequence this window belongs to. Dense: same as batch.
    // Varlen: chunk_indices[2 * chunkIdx].
    int64_t seqId;
    // First token on T (or packed T). Dense: localChunk * BT.
    // Varlen: cu_seqlens[seqId] + localChunk * BT.
    int64_t tokenStart;
    // Valid tokens in this window, in 1..BT. The last window of a sequence
    // may be shorter than BT.
    int64_t M;
    // Window index within the sequence. Dense: chunkIdx % Ceil(T / BT).
    // Varlen: chunk_indices[2 * chunkIdx + 1].
    int64_t localChunk;
};

// Maps sequence-chunk index (workId / HV) to a BT-token window. No hv.
template <typename St>
__aicore__ inline ChunkRange GetChunkRange(const St &st,
                                           AscendC::GlobalTensor<int64_t> gmCuSeqlens,
                                           AscendC::GlobalTensor<int64_t> gmChunkIndices,
                                           int64_t chunkIdx)
{
    ChunkRange chunk{};
    const int64_t bt = st.chunkSize;
    if (st.isVariable != 0 && st.hasChunkIndices != 0) {
        chunk.seqId = gmChunkIndices.GetValue(chunkIdx * 2);
        chunk.localChunk = gmChunkIndices.GetValue(chunkIdx * 2 + 1);
        const int64_t seqStart = gmCuSeqlens.GetValue(chunk.seqId);
        const int64_t seqEnd = gmCuSeqlens.GetValue(chunk.seqId + 1);
        chunk.tokenStart = seqStart + chunk.localChunk * bt;
        int64_t remain = seqEnd - chunk.tokenStart;
        chunk.M = remain > bt ? bt : remain;
        chunk.batch = 0;
        return chunk;
    }
    const int64_t nt = CeilDiv(st.T, bt);
    chunk.batch = chunkIdx / nt;
    chunk.localChunk = chunkIdx % nt;
    chunk.seqId = chunk.batch;
    chunk.tokenStart = chunk.localChunk * bt;
    int64_t remain = st.T - chunk.tokenStart;
    chunk.M = remain > bt ? bt : remain;
    if (chunk.M < 0) {
        chunk.M = 0;
    }
    return chunk;
}

// Linear index of [b, h, t, 0] in GM [B, H, T, D].
__aicore__ inline int64_t OffsetBHTD(int64_t b, int64_t h, int64_t t, int64_t heads, int64_t seq, int64_t dim)
{
    return ((b * heads + h) * seq + t) * dim;
}

// Linear index of [b, h, t] in GM [B, H, T].
__aicore__ inline int64_t OffsetBHT(int64_t b, int64_t h, int64_t t, int64_t heads, int64_t seq)
{
    return (b * heads + h) * seq + t;
}

} // namespace GdnStage

#endif
