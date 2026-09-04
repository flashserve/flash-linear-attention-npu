/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_POLICY_H
#define FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_POLICY_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace kda_prepare_pseudocode {

using Offset = std::uint32_t;

struct Region {
    Offset offset;
    Offset size;

    constexpr Offset End() const
    {
        return offset + size;
    }
};

struct CopyRegion {
    Offset source;
    Offset destination;
    Offset size;
};

constexpr bool IsAligned(Offset value, Offset alignment)
{
    return alignment != 0U && value % alignment == 0U;
}

constexpr bool Fits(const Region &region, Offset capacity)
{
    return region.offset <= capacity && region.size <= capacity - region.offset;
}

constexpr bool Disjoint(const Region &lhs, const Region &rhs)
{
    return lhs.End() <= rhs.offset || rhs.End() <= lhs.offset;
}

constexpr bool ForwardRowCompressionIsSafe(Offset rows,
                                           Offset inputRowBytes,
                                           Offset outputRowBytes)
{
    if (inputRowBytes == 0U || outputRowBytes == 0U) {
        return false;
    }
    for (Offset row = 0U; row < rows; ++row) {
        const Offset lastOverwrittenInputRow =
            ((row + 1U) * outputRowBytes - 1U) / inputRowBytes;
        if (lastOverwrittenInputRow > row) {
            return false;
        }
    }
    return true;
}

template <std::size_t N>
constexpr bool AllFit(const std::array<Region, N> &regions, Offset capacity)
{
    for (const Region &region : regions) {
        if (!Fits(region, capacity)) {
            return false;
        }
    }
    return true;
}

template <std::size_t N>
constexpr bool PairwiseDisjoint(const std::array<Region, N> &regions)
{
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            if (!Disjoint(regions[i], regions[j])) {
                return false;
            }
        }
    }
    return true;
}

template <std::size_t N>
constexpr Offset SumCopySizes(const std::array<CopyRegion, N> &copies)
{
    Offset result = 0U;
    for (const CopyRegion &copy : copies) {
        result += copy.size;
    }
    return result;
}

template <std::size_t N>
constexpr bool DestinationsAreContiguous(
    const std::array<CopyRegion, N> &copies, Offset expectedEnd)
{
    Offset cursor = 0U;
    for (const CopyRegion &copy : copies) {
        if (copy.destination != cursor) {
            return false;
        }
        cursor += copy.size;
    }
    return cursor == expectedEnd;
}

enum class GateStorageLayout : std::uint8_t {
    GATE_2B,
    GATE_FP32,
};

enum class AkkStorageLayout : std::uint8_t {
    AKK_2B_ABI,
    AKK_FP32_INTERNAL,
};

struct ShapePolicy {
    static constexpr Offset kBt = 64U;
    static constexpr Offset kK = 128U;
    static constexpr Offset kV = 128U;
    static constexpr Offset kScoreBlockRows = 16U;
    static constexpr Offset kScoreBlockCount = 4U;
    static constexpr Offset kStorageBytes = 2U;
    static constexpr Offset kFp32Bytes = 4U;

    static constexpr Offset kQBytes = kBt * kK * kStorageBytes;
    static constexpr Offset kKBytes = kBt * kK * kStorageBytes;
    static constexpr Offset kGBytes = kBt * kK * kFp32Bytes;
    static constexpr Offset kBetaInputBytes = kBt * kFp32Bytes;
    static constexpr Offset kBetaEffBytes = kBt * kFp32Bytes;
    static constexpr Offset kRawScoreBytes = kBt * kBt * kFp32Bytes;

    static constexpr std::array<Offset, kScoreBlockCount> kPrefixRows = {
        16U, 32U, 48U, 64U};
    static constexpr std::array<Offset, kScoreBlockCount> kKMinusBytes = {
        0x1000U, 0x2000U, 0x3000U, 0x4000U};

    static constexpr Offset LogicalPrefixRows(Offset block, Offset validRows)
    {
        return block < kScoreBlockCount
                   ? (kPrefixRows[block] < validRows ? kPrefixRows[block]
                                                     : validRows)
                   : 0U;
    }

    static constexpr Offset kQPlusBytes = kQBytes;
    static constexpr Offset kKPlusBytes = kKBytes;
    static constexpr Offset kKMinusTotalBytes =
        kKMinusBytes[0] + kKMinusBytes[1] + kKMinusBytes[2] + kKMinusBytes[3];
    static constexpr Offset kScorePayloadBytes =
        kQPlusBytes + kKPlusBytes + kKMinusTotalBytes;
};

struct UbPolicy {
    static constexpr Offset kCapacity = 0x3E000U;  // 248 KiB per AIV.
    static constexpr Offset kMainBytes = 0x1C000U; // 112 KiB per local head.
    static constexpr Offset kAuxBytes = 0x03000U;  // 12 KiB per local head.
    static constexpr std::array<Offset, 2> kMainBase = {0x00000U, 0x1C000U};
    static constexpr std::array<Offset, 2> kAuxBase = {0x38000U, 0x3B000U};
};

struct AuxLayout {
    static constexpr Region kBetaRaw{0x0000U, 0x0200U};
    static constexpr Region kBetaEff{0x0200U, 0x0200U};
    static constexpr std::array<Region, 4> kGRef = {{{0x0400U, 0x0200U},
                                                       {0x0600U, 0x0200U},
                                                       {0x0800U, 0x0200U},
                                                       {0x0A00U, 0x0200U}}};
    static constexpr Region kScanCarry{0x0C00U, 0x0200U};
    static constexpr Region kGLast{0x0E00U, 0x0200U};

    // V0 consumes the selective-gate inputs during the token scan, before it
    // materializes GRef[0:2]. This exact-address overlay preserves the full
    // 8 KiB VF scratch without any UB movement.
    static constexpr Region kDtBias{0x0400U, 0x0200U};
    static constexpr Region kALogOrGateAttrs{0x0600U, 0x0200U};
    static constexpr Region kVfScratch{0x1000U, 0x2000U};
};

struct V0Gate2BLayout {
    static constexpr Region kQHat{0x0000U, 0x4000U};
    static constexpr Region kKHat{0x4000U, 0x4000U};
    static constexpr Region kGateRaw{0x8000U, 0x4000U};
    static constexpr Region kG{0xC000U, 0x8000U};
    static constexpr Region kWork{0x14000U, 0x8000U};
};

struct V0GateFp32Layout {
    static constexpr Region kQHat{0x0000U, 0x4000U};
    static constexpr Region kKHat{0x4000U, 0x4000U};
    static constexpr Region kG{0x8000U, 0x8000U};
    static constexpr Region kWork{0x10000U, 0x8000U};
    static constexpr Region kReserve{0x18000U, 0x4000U};
};

struct V1Gate2BLayout {
    static constexpr Region kQPlus{0x0000U, 0x4000U};
    static constexpr Region kKPlus{0x4000U, 0x4000U};
    static constexpr std::array<Region, 4> kKMinus = {{{0x8000U, 0x1000U},
                                                         {0x9000U, 0x2000U},
                                                         {0x14000U, 0x3000U},
                                                         {0x17000U, 0x4000U}}};
    static constexpr Region kLiveG{0xC000U, 0x8000U};
    static constexpr std::array<CopyRegion, 2> kScoreWriteback = {{{0x0000U, 0x0000U, 0xB000U},
                                                                     {0x14000U, 0xB000U, 0x7000U}}};
};

struct V1GateFp32Layout {
    static constexpr Region kQPlus{0x0000U, 0x4000U};
    static constexpr Region kKPlus{0x4000U, 0x4000U};
    static constexpr std::array<Region, 4> kKMinus = {{{0x10000U, 0x1000U},
                                                         {0x11000U, 0x2000U},
                                                         {0x13000U, 0x3000U},
                                                         {0x16000U, 0x4000U}}};
    static constexpr Region kLiveG{0x8000U, 0x8000U};
    static constexpr std::array<CopyRegion, 2> kScoreWriteback = {{{0x0000U, 0x0000U, 0x8000U},
                                                                     {0x10000U, 0x8000U, 0xA000U}}};
};

struct V3Layout {
    static constexpr Region kRawAqk{0x0000U, 0x4000U};
    static constexpr Region kRawAkk{0x4000U, 0x4000U};
    static constexpr Region kLeaf0{0x8000U, 0x1000U};
    static constexpr Region kLeaf1{0x9000U, 0x1000U};
    static constexpr Region kB{0xA000U, 0x1000U};
    static constexpr Region kX0{0xB000U, 0x1000U};
    static constexpr Region kX1{0xC000U, 0x1000U};
    static constexpr Region kWork{0xD000U, 0x1000U};
    static constexpr Region kOptionalPack{0xE000U, 0x2000U};
    static constexpr Region kX0Tau{0xE000U, 0x0800U};
    static constexpr Region kX1Tau{0xE800U, 0x0800U};
    static constexpr Region kQ01Zero{0xF000U, 0x0800U};
};

struct V6Layout {
    static constexpr Region kQHatToQg{0x0000U, 0x4000U};
    static constexpr Region kKHatToKg{0x4000U, 0x4000U};
    static constexpr Region kVToVBeta{0x8000U, 0x4000U};
    static constexpr Region kGInput{0xC000U, 0x8000U};
    static constexpr Region kQgScaled{0xC000U, 0x4000U};
    static constexpr Region kKBetaG{0x14000U, 0x4000U};
    static constexpr Region kVfScratch{0x18000U, 0x4000U};
};

struct ScorePayloadLayout {
    static constexpr Region kQPlus{0x0000U, 0x4000U};
    static constexpr Region kKPlus{0x4000U, 0x4000U};
    static constexpr std::array<Region, 4> kKMinus = {{{0x8000U, 0x1000U},
                                                         {0x9000U, 0x2000U},
                                                         {0xB000U, 0x3000U},
                                                         {0xE000U, 0x4000U}}};
};

struct L1Policy {
    static constexpr Offset kCapacity = 0x80000U;   // 512 KiB.
    static constexpr Offset kLaneBytes = 0x12000U;  // 72 KiB.
    static constexpr std::array<Offset, 4> kLaneBase = {0x00000U, 0x12000U, 0x24000U, 0x36000U};
    static constexpr Offset kCurrentEnd = 0x48000U;
    static constexpr Offset kResidentBase = 0x48000U;
    static constexpr Offset kResidentBytes = 0x14000U;
    static constexpr Offset kPeakEnd = 0x5C000U;    // 368 KiB.
    static constexpr Region kHardReserve{0x5C000U, 0x24000U};

    struct AkkFp32Resident {
        static constexpr Region kAkk{0x48000U, 0x10000U};
        static constexpr Offset kAkkStride = 0x4000U;
        static constexpr Region kT{0x58000U, 0x4000U};
        static constexpr Offset kTStride = 0x1000U;
    };

    struct Akk2BResident {
        static constexpr Region kX0{0x48000U, 0x4000U};
        static constexpr Region kX1{0x4C000U, 0x4000U};
        static constexpr Region kT{0x50000U, 0x4000U};
        static constexpr Region kAkkTau{0x54000U, 0x8000U};
        static constexpr Offset kMatrixStride = 0x1000U;
        static constexpr Offset kAkkStride = 0x2000U;
    };
};

struct WorkspacePolicy {
    static constexpr Region kQHatContext{0x00000U, 0x04000U};
    static constexpr Region kKHatContext{0x04000U, 0x04000U};
    static constexpr Region kContextHardPad{0x08000U, 0x00200U};
    static constexpr Region kGContext{0x08200U, 0x08000U};
    static constexpr Region kAlignmentPad{0x10200U, 0x00200U};
    static constexpr Region kStagePayload{0x10400U, 0x12000U};

    static constexpr Offset kSlotStride = 0x22400U;       // 137 KiB.
    static constexpr Offset kSlotCount = 8U;
    static constexpr Offset kSlotsEnd = 0x112000U;        // 1096 KiB.
    static constexpr Region kControl{0x112000U, 0x01000U};
    static constexpr Offset kWorkgroupStride = 0x113000U;  // 1100 KiB.

    static constexpr Region kVcsX0{0x0000U, 0x1000U};
    static constexpr Region kVcsX1{0x1000U, 0x1000U};
    static constexpr Region kVcsB{0x2000U, 0x1000U};
    static constexpr Region kVcsX0Tau{0x3000U, 0x0800U};
    static constexpr Region kVcsX1Tau{0x3800U, 0x0800U};
    static constexpr Region kVcsQ01Zero{0x4000U, 0x0800U};
    static constexpr Region kVcsHardUnused{0x4800U, 0xD800U};

    static constexpr Region kPostKBetaG{0x0000U, 0x4000U};
    static constexpr Region kPostVBeta{0x4000U, 0x4000U};
    static constexpr Region kPostHardUnused{0x8000U, 0xA000U};

    static constexpr Region kRelayRawAqk{0x0000U, 0x4000U};
    static constexpr Region kRelayRawAkk{0x4000U, 0x4000U};
};

struct C2Policy {
    static constexpr Offset kM = 16U;
    static constexpr Offset kK = 128U;
    static constexpr std::array<Offset, 4> kN = ShapePolicy::kPrefixRows;
    static constexpr Offset kRawLeadingDimension = 64U;
    static constexpr Offset kRawRowBytes = kRawLeadingDimension * ShapePolicy::kFp32Bytes;
    static constexpr std::array<Offset, 4> kQBandOffset = {0x0000U, 0x1000U, 0x2000U, 0x3000U};
    static constexpr std::array<Offset, 4> kKBandOffset = {0x4000U, 0x5000U, 0x6000U, 0x7000U};
    static constexpr std::array<Offset, 4> kKMinusOffset = {0x8000U, 0x9000U, 0xB000U, 0xE000U};
};

struct Akk2BPackPolicy {
    // One logical [64,64] matrix is packed as four tight [32,32]
    // quadrants. C7 must use the dedicated quadrant-packed MTE1/MMAD path;
    // treating this 8 KiB payload as a row-major matrix is invalid.
    static constexpr Offset kQuadrantRows = 32U;
    static constexpr Offset kQuadrantColumns = 32U;
    static constexpr Offset kQuadrantBytes = 0x0800U;
    static constexpr Region kQ00{0x0000U, kQuadrantBytes};
    static constexpr Region kQ01{0x0800U, kQuadrantBytes};
    static constexpr Region kQ10{0x1000U, kQuadrantBytes};
    static constexpr Region kQ11{0x1800U, kQuadrantBytes};
    static constexpr Offset kPackedBytes = 0x2000U;
};

struct L0cPolicy {
    // PROPOSED API gate: four Cube-pipelined heads require four disjoint 64
    // KiB L0C lanes. Target A5/CANN must confirm the 256 KiB physical budget
    // and exact MMAD/Fixpipe forms; host syntax checks cannot prove either.
    // C2/C4/C5/C7 overlay only within one head lane across completed stages;
    // an independent L0cBankFree ticket protects cross-group lane reuse.
    static constexpr Offset kHeadCount = 4U;
    static constexpr Offset kHeadLaneBytes = 0x10000U;
    static constexpr Offset kRequiredBytes = kHeadCount * kHeadLaneBytes;
    static constexpr std::array<Offset, 4> kC2PerHeadResultBytes = {
        0x0400U, 0x0800U, 0x0C00U, 0x1000U};
    static constexpr std::array<Offset, 4> kC2AqkOffset = {
        0x0000U, 0x0400U, 0x0C00U, 0x1800U};
    static constexpr std::array<Offset, 4> kC2AkkOffset = {
        0x2800U, 0x2C00U, 0x3400U, 0x4000U};
    static constexpr Offset kC2PerHeadEnd = 0x5000U;
    static constexpr Region kC4T{0x0000U, 0x1000U};
    static constexpr Region kC5Y{0x1000U, 0x1000U};
    static constexpr Region kC7W{0x0000U, 0x8000U};
    static constexpr Region kC7U{0x8000U, 0x8000U};

    static constexpr Offset HeadLaneBase(Offset groupLocalHead)
    {
        return groupLocalHead * kHeadLaneBytes;
    }
};

static_assert(ShapePolicy::kBt == 64U && ShapePolicy::kK == 128U && ShapePolicy::kV == 128U,
              "The fixed Prepare shape changed without a new layout proof");
static_assert(ShapePolicy::kScoreBlockRows * ShapePolicy::kScoreBlockCount == ShapePolicy::kBt,
              "Four 16-row score bands must cover the chunk");
static_assert(ShapePolicy::kQBytes == 0x4000U && ShapePolicy::kKBytes == 0x4000U &&
                  ShapePolicy::kGBytes == 0x8000U &&
                  ShapePolicy::kBetaInputBytes == 0x0100U &&
                  ShapePolicy::kBetaEffBytes == 0x0100U,
              "Context byte sizes do not match the fixed address contract");
static_assert(ShapePolicy::kBetaInputBytes <= AuxLayout::kBetaRaw.size &&
                  ShapePolicy::kBetaEffBytes <= AuxLayout::kBetaEff.size,
              "aligned AUX regions must contain FP32 token-scalar beta arrays");
static_assert(ShapePolicy::kKMinusTotalBytes == 0xA000U,
              "Causal Kminus prefixes must occupy 4+8+12+16 KiB");
static_assert(ShapePolicy::kKMinusBytes[0] ==
                      ShapePolicy::kPrefixRows[0] * ShapePolicy::kK *
                          ShapePolicy::kStorageBytes &&
                  ShapePolicy::kKMinusBytes[1] ==
                      ShapePolicy::kPrefixRows[1] * ShapePolicy::kK *
                          ShapePolicy::kStorageBytes &&
                  ShapePolicy::kKMinusBytes[2] ==
                      ShapePolicy::kPrefixRows[2] * ShapePolicy::kK *
                          ShapePolicy::kStorageBytes &&
                  ShapePolicy::kKMinusBytes[3] ==
                      ShapePolicy::kPrefixRows[3] * ShapePolicy::kK *
                          ShapePolicy::kStorageBytes,
              "each Kminus segment must match its physical prefix extent");
static_assert(ShapePolicy::kScorePayloadBytes == 0x12000U,
              "The S=4 causal-prefix score payload must be exactly 72 KiB");

static_assert(2U * UbPolicy::kMainBytes + 2U * UbPolicy::kAuxBytes == UbPolicy::kCapacity,
              "Two fixed MAIN/AUX head slots must exactly cover 248 KiB UB");
static_assert(UbPolicy::kMainBase[1] + UbPolicy::kMainBytes == UbPolicy::kAuxBase[0] &&
                  UbPolicy::kAuxBase[1] + UbPolicy::kAuxBytes == UbPolicy::kCapacity,
              "MAIN/AUX absolute offsets overlap or leave an unowned tail");
static_assert(AuxLayout::kVfScratch.size == 0x2000U &&
                  AuxLayout::kVfScratch.End() == UbPolicy::kAuxBytes,
              "the full 8 KiB AUX scratch must end at the 12 KiB boundary");
static_assert(AuxLayout::kDtBias.offset == AuxLayout::kGRef[0].offset &&
                  AuxLayout::kDtBias.size == AuxLayout::kGRef[0].size &&
                  AuxLayout::kALogOrGateAttrs.offset ==
                      AuxLayout::kGRef[1].offset &&
                  AuxLayout::kALogOrGateAttrs.size ==
                      AuxLayout::kGRef[1].size,
              "selective-gate temporaries must exactly overlay late GRef outputs");
static_assert(Disjoint(AuxLayout::kBetaRaw, AuxLayout::kDtBias) &&
                  Disjoint(AuxLayout::kBetaEff,
                           AuxLayout::kALogOrGateAttrs) &&
                  Disjoint(AuxLayout::kGLast, AuxLayout::kVfScratch),
              "AUX overlay must not consume beta or the 8 KiB VF scratch");

static_assert(V0Gate2BLayout::kWork.End() == UbPolicy::kMainBytes &&
                  V0GateFp32Layout::kReserve.End() == UbPolicy::kMainBytes,
              "V0 layouts must fit one 112 KiB MAIN");
static_assert(V1Gate2BLayout::kKMinus[3].End() <= UbPolicy::kMainBytes &&
                  V1GateFp32Layout::kKMinus[3].End() <= UbPolicy::kMainBytes,
              "V1 Kminus output exceeds MAIN");
static_assert(Disjoint(V1Gate2BLayout::kQPlus, V1Gate2BLayout::kLiveG) &&
                  Disjoint(V1Gate2BLayout::kKPlus, V1Gate2BLayout::kLiveG) &&
                  Disjoint(V1Gate2BLayout::kKMinus[0], V1Gate2BLayout::kLiveG) &&
                  Disjoint(V1Gate2BLayout::kKMinus[1], V1Gate2BLayout::kLiveG) &&
                  Disjoint(V1Gate2BLayout::kKMinus[2], V1Gate2BLayout::kLiveG) &&
                  Disjoint(V1Gate2BLayout::kKMinus[3], V1Gate2BLayout::kLiveG),
              "GATE_2B V1 output must not overwrite unread G");
static_assert(Disjoint(V1GateFp32Layout::kQPlus, V1GateFp32Layout::kLiveG) &&
                  Disjoint(V1GateFp32Layout::kKPlus, V1GateFp32Layout::kLiveG) &&
                  Disjoint(V1GateFp32Layout::kKMinus[0], V1GateFp32Layout::kLiveG) &&
                  Disjoint(V1GateFp32Layout::kKMinus[1], V1GateFp32Layout::kLiveG) &&
                  Disjoint(V1GateFp32Layout::kKMinus[2], V1GateFp32Layout::kLiveG) &&
                  Disjoint(V1GateFp32Layout::kKMinus[3], V1GateFp32Layout::kLiveG),
              "GATE_FP32 V1 output must not overwrite unread G");
static_assert(ShapePolicy::kScorePayloadBytes + ShapePolicy::kGBytes == 0x1A000U,
              "V1 live MAIN occupancy must remain 104 KiB");
static_assert(SumCopySizes(V1Gate2BLayout::kScoreWriteback) == ShapePolicy::kScorePayloadBytes &&
                  DestinationsAreContiguous(V1Gate2BLayout::kScoreWriteback,
                                            ShapePolicy::kScorePayloadBytes),
              "GATE_2B writeback must cover the 72 KiB GM payload exactly once");
static_assert(SumCopySizes(V1GateFp32Layout::kScoreWriteback) == ShapePolicy::kScorePayloadBytes &&
                  DestinationsAreContiguous(V1GateFp32Layout::kScoreWriteback,
                                            ShapePolicy::kScorePayloadBytes),
              "GATE_FP32 writeback must cover the 72 KiB GM payload exactly once");
static_assert(V3Layout::kOptionalPack.End() == 0x10000U && V6Layout::kVfScratch.End() == 0x1C000U,
              "V3/V6 MAIN high-water marks changed");
static_assert(ForwardRowCompressionIsSafe(
                  ShapePolicy::kBt,
                  ShapePolicy::kK * ShapePolicy::kFp32Bytes,
                  ShapePolicy::kK * ShapePolicy::kStorageBytes),
              "V6 G-to-Qg compression must overwrite only current/past rows");

static_assert(ScorePayloadLayout::kKMinus[3].End() == ShapePolicy::kScorePayloadBytes,
              "Packed score payload offsets must end at 72 KiB");
static_assert(4U * L1Policy::kLaneBytes == L1Policy::kCurrentEnd &&
                  L1Policy::kCurrentEnd == L1Policy::kResidentBase,
              "Four 72 KiB L1 lanes must be contiguous");
static_assert(L1Policy::kResidentBase + L1Policy::kResidentBytes == L1Policy::kPeakEnd &&
                  L1Policy::kPeakEnd <= L1Policy::kCapacity &&
                  L1Policy::kHardReserve.End() == L1Policy::kCapacity,
              "L1 current lanes, residents, and hard reserve do not close");
static_assert(L1Policy::AkkFp32Resident::kT.End() == L1Policy::kPeakEnd &&
                  L1Policy::Akk2BResident::kAkkTau.End() == L1Policy::kPeakEnd,
              "Both Akk templates must consume the same 80 KiB resident tail");
static_assert(Akk2BPackPolicy::kQ00.End() == Akk2BPackPolicy::kQ01.offset &&
                  Akk2BPackPolicy::kQ01.End() ==
                      Akk2BPackPolicy::kQ10.offset &&
                  Akk2BPackPolicy::kQ10.End() ==
                      Akk2BPackPolicy::kQ11.offset &&
                  Akk2BPackPolicy::kQ11.End() ==
                      Akk2BPackPolicy::kPackedBytes &&
                  Akk2BPackPolicy::kPackedBytes ==
                      L1Policy::Akk2BResident::kAkkStride,
              "2-byte Akk must remain a tight q00/q01/q10/q11 quadrant-major pack");
static_assert(V3Layout::kX0Tau.size == Akk2BPackPolicy::kQuadrantBytes &&
                  V3Layout::kX1Tau.size ==
                      Akk2BPackPolicy::kQuadrantBytes &&
                  V3Layout::kQ01Zero.size ==
                      Akk2BPackPolicy::kQuadrantBytes,
              "V3 q00/q11/q01 staging regions must each hold one rounded 32x32 quadrant");

static_assert(WorkspacePolicy::kStagePayload.End() == WorkspacePolicy::kSlotStride,
              "Context plus 72 KiB payload must exactly fill a workspace slot");
static_assert(WorkspacePolicy::kQHatContext.End() ==
                      WorkspacePolicy::kKHatContext.offset &&
                  WorkspacePolicy::kKHatContext.End() ==
                      WorkspacePolicy::kContextHardPad.offset &&
                  WorkspacePolicy::kContextHardPad.End() ==
                      WorkspacePolicy::kGContext.offset &&
                  WorkspacePolicy::kGContext.End() ==
                      WorkspacePolicy::kAlignmentPad.offset &&
                  WorkspacePolicy::kAlignmentPad.End() ==
                      WorkspacePolicy::kStagePayload.offset,
              "workspace context/pad/G/payload offsets must stay contiguous");
static_assert(WorkspacePolicy::kSlotCount * WorkspacePolicy::kSlotStride == WorkspacePolicy::kSlotsEnd &&
                  WorkspacePolicy::kControl.offset == WorkspacePolicy::kSlotsEnd &&
                  WorkspacePolicy::kControl.End() == WorkspacePolicy::kWorkgroupStride,
              "Eight slots and the control region must exactly fill a workgroup");
static_assert(WorkspacePolicy::kVcsHardUnused.End() == ShapePolicy::kScorePayloadBytes &&
                  WorkspacePolicy::kPostHardUnused.End() == ShapePolicy::kScorePayloadBytes,
              "Every payload generation must retain a fixed 72 KiB owner");
static_assert(WorkspacePolicy::kRelayRawAkk.End() <= ShapePolicy::kScorePayloadBytes,
              "The optional raw-score relay must fit the stage payload");
static_assert(IsAligned(WorkspacePolicy::kStagePayload.offset, 0x200U) &&
                  IsAligned(WorkspacePolicy::kSlotStride, 0x200U) &&
                  IsAligned(WorkspacePolicy::kWorkgroupStride, 0x1000U),
              "Workspace offsets violate 512-byte/4-KiB alignment");

static_assert(C2Policy::kN[0] == 16U && C2Policy::kN[1] == 32U &&
                  C2Policy::kN[2] == 48U && C2Policy::kN[3] == 64U,
              "C2 N must match the four causal prefixes");
static_assert(C2Policy::kRawRowBytes == 0x100U,
              "C2 Fixpipe destinations must retain a raw-score leading dimension of 64 FP32 elements");
static_assert(L0cPolicy::kC2PerHeadResultBytes[0] ==
                      C2Policy::kM * C2Policy::kN[0] *
                          ShapePolicy::kFp32Bytes &&
                  L0cPolicy::kC2PerHeadResultBytes[1] ==
                      C2Policy::kM * C2Policy::kN[1] *
                          ShapePolicy::kFp32Bytes &&
                  L0cPolicy::kC2PerHeadResultBytes[2] ==
                      C2Policy::kM * C2Policy::kN[2] *
                          ShapePolicy::kFp32Bytes &&
                  L0cPolicy::kC2PerHeadResultBytes[3] ==
                      C2Policy::kM * C2Policy::kN[3] *
                          ShapePolicy::kFp32Bytes,
              "C2 per-head L0C bytes must match tight 16xN FP32 results");
static_assert(L0cPolicy::kC2AqkOffset[0] == 0U &&
                  L0cPolicy::kC2AqkOffset[1] ==
                      L0cPolicy::kC2AqkOffset[0] +
                          L0cPolicy::kC2PerHeadResultBytes[0] &&
                  L0cPolicy::kC2AqkOffset[2] ==
                      L0cPolicy::kC2AqkOffset[1] +
                          L0cPolicy::kC2PerHeadResultBytes[1] &&
                  L0cPolicy::kC2AqkOffset[3] ==
                      L0cPolicy::kC2AqkOffset[2] +
                          L0cPolicy::kC2PerHeadResultBytes[2] &&
                  L0cPolicy::kC2AkkOffset[0] ==
                      L0cPolicy::kC2AqkOffset[3] +
                          L0cPolicy::kC2PerHeadResultBytes[3] &&
                  L0cPolicy::kC2AkkOffset[1] ==
                      L0cPolicy::kC2AkkOffset[0] +
                          L0cPolicy::kC2PerHeadResultBytes[0] &&
                  L0cPolicy::kC2AkkOffset[2] ==
                      L0cPolicy::kC2AkkOffset[1] +
                          L0cPolicy::kC2PerHeadResultBytes[1] &&
                  L0cPolicy::kC2AkkOffset[3] ==
                      L0cPolicy::kC2AkkOffset[2] +
                          L0cPolicy::kC2PerHeadResultBytes[2] &&
                  L0cPolicy::kC2AkkOffset[3] +
                          L0cPolicy::kC2PerHeadResultBytes[3] ==
                      L0cPolicy::kC2PerHeadEnd &&
                  L0cPolicy::kC2PerHeadEnd <=
                      L0cPolicy::kHeadLaneBytes &&
                  Disjoint(L0cPolicy::kC4T, L0cPolicy::kC5Y) &&
                  L0cPolicy::kC5Y.End() <=
                      L0cPolicy::kHeadLaneBytes &&
                  L0cPolicy::kC7W.End() == L0cPolicy::kC7U.offset &&
                  L0cPolicy::kC7U.End() ==
                      L0cPolicy::kHeadLaneBytes &&
                  L0cPolicy::HeadLaneBase(
                      L0cPolicy::kHeadCount - 1U) +
                          L0cPolicy::kHeadLaneBytes ==
                      L0cPolicy::kRequiredBytes,
              "each head must own one disjoint 64 KiB L0C lane within the 256 KiB proposal");

} // namespace kda_prepare_pseudocode

#endif // FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_POLICY_H
