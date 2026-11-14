/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_STRUCT_H
#define CHUNK_KDA_BWD_RECOMPUTE_STRUCT_H

#include <cstdint>

namespace KDA {

// All int64_t so host and CCE layouts match (no 4-byte float padding).
// sizeof must stay a multiple of 16 to equal kernel opParaSize.
struct ChunkKdaBwdRecomputeTilingData {
    int64_t B;
    int64_t Hk;
    int64_t Hv;
    int64_t hvPerHk;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkNum;
    int64_t chunkSize;
    int64_t isVariable;
    int64_t useGateInKernel;
    int64_t useExp2;
    int64_t hasALog;
    int64_t hasDtBias;
    int64_t lowerBoundBits;
    // Vector row tile selected from the target UB budget.
    int64_t vecRow;
    // Logical AIC count selected by host tiling (used by the arch35 task mapping).
    int64_t usedAic;
    int64_t reserved;
};

static_assert(sizeof(ChunkKdaBwdRecomputeTilingData) == 144,
              "TilingData is 18 int64 fields");

} // namespace KDA

namespace KdaBwdRecomputeArch92 {

constexpr uint32_t kRows = 64;
constexpr uint32_t kChannels = 128;
constexpr uint32_t kSlots = 2;
constexpr uint32_t kInputBytes = 0x14000; // 80 KiB per IN slot
constexpr uint32_t kOutputBytes = 0x18000; // 96 KiB per OUT slot
constexpr uint32_t kMetaBytes = 0x800; // 2 KiB per META slot
constexpr uint32_t kInputBase = 0; // 0 KiB offset in UB
constexpr uint32_t kOutputBase = 0x28000; // 160 KiB offset in UB
constexpr uint32_t kMetaBase = 0x58000; // 352 KiB offset in UB
constexpr uint32_t kUbBytes = 0x59000; // 356 KiB total UB allocation
constexpr uint32_t kUsableUbBytes = 376 * 1024; // 376 KiB usable UB budget

// Byte offsets within an IN / OUT slot. g and gk each reserve FP32 capacity.
constexpr uint32_t kG = 0; // 0 KiB offset in IN
constexpr uint32_t kQ = 0x8000; // 32 KiB offset in IN
constexpr uint32_t kK = 0xC000; // 48 KiB offset in IN
constexpr uint32_t kV = 0x10000; // 64 KiB offset in IN
constexpr uint32_t kGk = 0; // 0 KiB offset in OUT
constexpr uint32_t kQg = 0x8000; // 32 KiB offset in OUT
constexpr uint32_t kKbg = 0xC000; // 48 KiB offset in OUT
constexpr uint32_t kVb = 0x10000; // 64 KiB offset in OUT
constexpr uint32_t kKg = 0x14000; // 80 KiB offset in OUT
constexpr uint32_t kBetaRaw = 0; // 0 KiB offset in META
constexpr uint32_t kBetaFp32 = 0x80; // 0.125 KiB (128 B) offset in META
constexpr uint32_t kDtBias = 0x180; // 0.375 KiB (384 B) offset in META
constexpr uint32_t kALog = 0x380; // 0.875 KiB (896 B) offset in META

constexpr uint32_t kL1ABytes = 0x2000; // 8 KiB per L1 A slot
constexpr uint32_t kL1RhsBase = 0x4000; // 16 KiB offset in L1
constexpr uint32_t kL1RhsBytes = 0x10000; // 64 KiB reserved per L1 RHS slot
constexpr uint32_t kL1VbRelative = 0x8000; // 32 KiB offset within an L1 RHS slot
constexpr uint32_t kL1Bytes = 0x24000; // 144 KiB total L1 layout
constexpr uint32_t kL0ABytes = kRows * kRows * sizeof(uint16_t); // 8 KiB
constexpr uint32_t kL0BBytes = kRows * kChannels * sizeof(uint16_t); // 16 KiB
constexpr uint32_t kL0CBytes = kRows * kChannels * sizeof(float); // 32 KiB
constexpr uint16_t kReadyFlag = 6;
constexpr uint16_t kFreeFlag = 7;

} // namespace KdaBwdRecomputeArch92

#endif // CHUNK_KDA_BWD_RECOMPUTE_STRUCT_H
