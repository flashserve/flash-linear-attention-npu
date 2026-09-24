/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_KERNEL_STRUCT_H
#define MERGE_FWD_BWD_KERNEL_STRUCT_H

#include <cstdint>

namespace MergeFwBwd {

// Single Mix launch: Vector packs/adds, Cube GEMMs. Mode-4 per-head CV/VC
// so add(h) overlaps GEMM(h+1) and GEMM step i+1(h) starts when add(h) of
// step i finishes — no chip-wide SyncAll in the rank loop.
struct MergeFwdBwdKernelTilingData {
    int64_t S;
    int64_t Hv;
    int64_t K;
    int64_t V;
    int64_t rank;
    int64_t N;
    int64_t forward;
    int64_t usedAic;
    int64_t sysWorkspaceSize;
};

static_assert(sizeof(MergeFwdBwdKernelTilingData) == 72, "TilingData is 9 int64 fields");

constexpr uint32_t kKDim = 128;
constexpr uint32_t kVDim = 128;
constexpr uint32_t kTileM = 64;
constexpr uint32_t kRowStride = kVDim + kKDim; // 256
constexpr uint32_t kKvElems = kKDim * kVDim;
constexpr uint32_t kKkElems = kKDim * kKDim;

constexpr uint32_t kHFp32Bytes = kKvElems * 4;
constexpr uint32_t kMFp32Bytes = kKkElems * 4;

// FP32 scratch [3, HV, 128, 128]: bank0=H, bank1/2=M ping-pong so Vector can
// pack M_{i+1} while Cube GEMMs M_i.
constexpr uint32_t kScratchBanks = 3;
constexpr uint32_t kScratchBankH = 0;
constexpr uint32_t kScratchBankM0 = 1;
constexpr uint32_t kScratchBankM1 = 2;

constexpr uint32_t kCTileElems = kTileM * kVDim;

// Mode 4: AIC↔AIV0/AIV1 independent. Flag IDs 0..7; AIV1 uses +16.
// CV/VC are per-AIV (flag = base + (h&1)*16), so all heads on the same AIV
// share one C UB (fp32Buf). Cube must Wait CUbFree before any Fixpipe to that AIV.
constexpr uint8_t kCrossCoreModeAiv = 0x4;
constexpr uint8_t kCvReadyFlag = 2;    // Cube C0 Fixpipe → Vector add top 64 rows
constexpr uint8_t kCvC1ReadyFlag = 3;  // Cube C1 Fixpipe → Vector add bottom 64 rows
constexpr uint8_t kVcReadyFlag = 4;    // Vector H/M ready → Cube GEMM
constexpr uint8_t kCUbFreeFlag = 5;    // Vector drained fp32Buf → Cube may Fixpipe C
constexpr uint8_t kSubBlockFlagOffset = 16;

// 910b/910_93 leftover Mode-2 constants (unused on the fused Mix path).
constexpr uint8_t kCrossCoreModeIndep = 0x2;
constexpr uint8_t kChunkReadyFlag = 2;
constexpr uint8_t kChunkFreeFlag = 4;
constexpr uint8_t kChunkC1Flag = 3;
constexpr uint8_t kChunkRoundFlag = 5;
constexpr uint32_t kAivHeadSlots = 2;
constexpr uint32_t kMaxTaskPerAic = 4;
constexpr uint32_t kCoreStride = kMaxTaskPerAic * (kHFp32Bytes + kMFp32Bytes);

} // namespace MergeFwBwd

#endif
