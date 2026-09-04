/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
#define PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace kda_prepare_pseudocode {

// This directory is non-building design pseudocode. The actual Ascend C tiling
// key declaration and encoding remain PROPOSED until the ABI is frozen.
enum class Architecture : std::uint8_t {
    Arch22,
    Arch35,
};

constexpr bool IsKnownArchitecture(Architecture architecture) noexcept
{
    return architecture == Architecture::Arch22 ||
           architecture == Architecture::Arch35;
}

constexpr std::uint32_t kChunkRows = 64;
constexpr std::uint32_t kHeadDimension = 128;
constexpr std::uint32_t kScoreBlockRows = 16;
constexpr std::uint32_t kScoreBlockCount = 4;
constexpr std::uint32_t kHeadsPerGroup = 4;
constexpr std::uint32_t kAivPerWorkgroup = 2;
constexpr std::uint32_t kHeadsPerAiv = 2;
constexpr std::uint32_t kArch22PairCount =
    kHeadsPerGroup / kAivPerWorkgroup;
constexpr std::uint32_t kArch22HeadGroupsInFlight = 1;
constexpr std::uint32_t kArch35HeadGroupsInFlight = 2;
constexpr std::uint32_t kHeadGroupsInFlight = kArch35HeadGroupsInFlight;
constexpr std::uint32_t kArch22WorkspaceSlotCount = kHeadsPerGroup;
constexpr std::uint32_t kArch35WorkspaceSlotCount =
    kHeadsPerGroup * kArch35HeadGroupsInFlight;
constexpr std::uint32_t kWorkspaceSlotCount =
    kArch35WorkspaceSlotCount;
constexpr std::uint32_t kMaxWorkspaceSlotCount =
    kArch35WorkspaceSlotCount;

constexpr std::size_t kKiB = 1024;
constexpr std::size_t kArch22UbBytes = 184 * kKiB;
constexpr std::size_t kArch22PrivateBankBytes = 72 * kKiB;
constexpr std::size_t kArch22SharedArenaBytes = 40 * kKiB;
constexpr std::size_t kArch35UbBytes = 248 * kKiB;
constexpr std::size_t kUbBytes = 248 * kKiB;
constexpr std::size_t kL1Bytes = 512 * kKiB;
constexpr std::size_t kMainBytesPerLocalHead = 112 * kKiB;
constexpr std::size_t kAuxBytesPerLocalHead = 12 * kKiB;
constexpr std::uint64_t kWorkspaceSlotStrideBytes = 0x22400ULL;
constexpr std::uint64_t kWorkspaceControlBytes = 0x1000ULL;
constexpr std::uint64_t kArch22WorkspaceSlotsEndBytes = 0x89000ULL;
constexpr std::uint64_t kArch22WorkspaceWorkgroupStrideBytes = 0x8A000ULL;
constexpr std::uint64_t kArch35WorkspaceSlotsEndBytes = 0x112000ULL;
constexpr std::uint64_t kArch35WorkspaceWorkgroupStrideBytes = 0x113000ULL;
constexpr std::uint64_t kWorkspaceSlotsEndBytes =
    kArch35WorkspaceSlotsEndBytes;
constexpr std::uint64_t kWorkspaceWorkgroupStrideBytes =
    kArch35WorkspaceWorkgroupStrideBytes;

constexpr std::uint32_t WorkspaceSlotCountFor(
    Architecture architecture) noexcept
{
    return architecture == Architecture::Arch22
               ? kArch22WorkspaceSlotCount
               : kArch35WorkspaceSlotCount;
}

constexpr std::uint32_t HeadGroupsInFlightFor(
    Architecture architecture) noexcept
{
    return architecture == Architecture::Arch22
               ? kArch22HeadGroupsInFlight
               : kArch35HeadGroupsInFlight;
}

constexpr std::uint64_t WorkspaceWorkgroupStrideFor(
    Architecture architecture) noexcept
{
    return architecture == Architecture::Arch22
               ? kArch22WorkspaceWorkgroupStrideBytes
               : kArch35WorkspaceWorkgroupStrideBytes;
}

// Base-2 exponent bounds copied from the existing split Prepare semantics.
// SCORE_T controls the score path; direct gate-derived values retain the
// narrower range even when the score payload is BF16.
constexpr float kFp16ScoreExp2InputMin = -80.0F;
constexpr float kFp16ScoreExp2InputMax = 80.0F;
constexpr float kBf16ScoreExp2InputMin = -126.0F;
constexpr float kBf16ScoreExp2InputMax = 120.0F;
constexpr float kDirectExp2InputMin = -80.0F;
constexpr float kDirectExp2InputMax = 80.0F;
constexpr float kFp16FiniteMin = -65504.0F;
constexpr float kFp16FiniteMax = 65504.0F;

constexpr std::array<std::uint32_t, kScoreBlockCount> kPrefixRows = {
    16, 32, 48, 64};
constexpr std::size_t kElementBytes = 2;
constexpr std::size_t kQplusBytes =
    kChunkRows * kHeadDimension * kElementBytes;
constexpr std::size_t kKplusBytes = kQplusBytes;
constexpr std::size_t kKminusPrefixBytes =
    (16 + 32 + 48 + 64) * kHeadDimension * kElementBytes;
constexpr std::size_t kPackedScoreBytes =
    kQplusBytes + kKplusBytes + kKminusPrefixBytes;

static_assert(kQplusBytes == 16 * kKiB, "Qplus must occupy 16 KiB");
static_assert(kKplusBytes == 16 * kKiB, "Kplus must occupy 16 KiB");
static_assert(kKminusPrefixBytes == 40 * kKiB,
              "four causal Kminus prefixes must occupy 40 KiB");
static_assert(kPackedScoreBytes == 72 * kKiB,
              "the frozen S=4 causal-prefix score payload is 72 KiB");
static_assert(kAivPerWorkgroup * kHeadsPerAiv == kHeadsPerGroup,
              "two AIVs must cover one four-head group");
static_assert(kArch22PairCount == 2,
              "Arch22 four-head groups must execute two pair waves");
static_assert(kWorkspaceSlotCount == 8,
              "two head groups in flight require eight workspace slots");
static_assert(kArch22WorkspaceSlotCount * kWorkspaceSlotStrideBytes ==
                  kArch22WorkspaceSlotsEndBytes &&
                  kArch22WorkspaceSlotsEndBytes + kWorkspaceControlBytes ==
                      kArch22WorkspaceWorkgroupStrideBytes,
              "Arch22 four-slot workspace must end at 0x8A000");
static_assert(kArch35WorkspaceSlotCount * kWorkspaceSlotStrideBytes ==
                  kArch35WorkspaceSlotsEndBytes &&
                  kArch35WorkspaceSlotsEndBytes + kWorkspaceControlBytes ==
                      kArch35WorkspaceWorkgroupStrideBytes,
              "Arch35 eight-slot workspace must end at 0x113000");
static_assert(kWorkspaceSlotStrideBytes * kWorkspaceSlotCount ==
                  kWorkspaceSlotsEndBytes,
              "eight workspace slots must exactly precede control storage");
static_assert(kWorkspaceSlotsEndBytes + kWorkspaceControlBytes ==
                  kWorkspaceWorkgroupStrideBytes,
              "workspace workgroup stride must include the control region");
static_assert(kHeadsPerAiv *
                      (kMainBytesPerLocalHead + kAuxBytesPerLocalHead) ==
                  kUbBytes,
              "each AIV owns two 112 KiB MAIN + 12 KiB AUX banks");
static_assert(2 * kArch22PrivateBankBytes + kArch22SharedArenaBytes ==
                  kArch22UbBytes,
              "Arch22 must exactly use two 72 KiB banks plus one 40 KiB arena");
static_assert(kArch35UbBytes == kUbBytes,
              "legacy UB constants remain Arch35 aliases");

enum class PartitionMode : std::uint8_t {
    ChunkOnly,
    ChunkHeadGroup,
};

enum class GateStorage : std::uint8_t {
    Fp16,
    Bf16,
    Fp32,
};

constexpr bool IsTwoByteGateStorage(GateStorage storage) noexcept
{
    return storage == GateStorage::Fp16 || storage == GateStorage::Bf16;
}

enum class InputStorage : std::uint8_t {
    Fp16,
    Bf16,
};

enum class ScoreStorage : std::uint8_t {
    Fp16,
    Bf16,
};

enum class MatrixStorage : std::uint8_t {
    Fp16,
    Bf16,
    Fp32,
};

enum class AkkStorage : std::uint8_t {
    TwoByteAbi,
    Fp32Internal,
};

enum class PrepareAbi : std::uint8_t {
    Current,
    Fused,
};

// V3's coordinate-level ownership contract is shared by both VF pseudocode
// paths and the host contract test. Physical C2 writes include 16-row tail
// padding, while the two raw readers are predicated to valid causal elements.
constexpr bool V3C2RawDefined(std::uint32_t validRows, std::uint32_t row,
                              std::uint32_t column) noexcept
{
    const std::uint32_t activeRows =
        ((validRows + kScoreBlockRows - 1U) / kScoreBlockRows) *
        kScoreBlockRows;
    return validRows <= kChunkRows && row < activeRows &&
           row < kChunkRows && column < kChunkRows &&
           column < kPrefixRows[row / kScoreBlockRows];
}

constexpr bool V3AqkRawReadRequired(std::uint32_t validRows,
                                    std::uint32_t row,
                                    std::uint32_t column) noexcept
{
    return V3C2RawDefined(validRows, row, column) && row < validRows &&
           column < validRows && column <= row;
}

constexpr bool V3AkkRawReadRequired(std::uint32_t validRows,
                                    std::uint32_t row,
                                    std::uint32_t column) noexcept
{
    return V3C2RawDefined(validRows, row, column) && row < validRows &&
           column < validRows && column < row;
}

constexpr bool V3C5Q10WriteRequired(Architecture architecture,
                                    std::uint32_t validRows,
                                    std::uint32_t row,
                                    std::uint32_t column) noexcept
{
    if (validRows <= 32U || validRows > kChunkRows || row < 32U ||
        row >= kChunkRows || column >= 32U) {
        return false;
    }
    // Arch35's final L1 operand is a fixed quadrant; Arch22 writes only the
    // valid bottom rows to its row-major relay and lets C7 fill the tail.
    return architecture == Architecture::Arch35 || row < validRows;
}

constexpr bool V3StableAkkWriteRequired(Architecture architecture,
                                        PrepareAbi abi,
                                        std::uint32_t validRows,
                                        std::uint32_t row,
                                        std::uint32_t column) noexcept
{
    if (validRows == 0U || validRows > kChunkRows || row >= kChunkRows ||
        column >= kChunkRows ||
        V3C5Q10WriteRequired(architecture, validRows, row, column)) {
        return false;
    }
    if (architecture == Architecture::Arch22) {
        if (row >= validRows) {
            return false;
        }
        const bool q01 = row < 32U && column >= 32U;
        return !(abi == PrepareAbi::Fused && validRows <= 32U && q01);
    }

    const bool q00 = row < 32U && column < 32U;
    const bool q01 = row < 32U && column >= 32U;
    const bool q11 = row >= 32U && column >= 32U;
    if (abi == PrepareAbi::Current) {
        // Current drains only valid public Akk rows; C4 fills tail rows before
        // its strided-to-tight reload.
        return row < validRows && (q00 || q01 || q11);
    }
    // Fused feeds fixed tight Cube operands directly, including tail padding.
    return q00 || (validRows > 32U && q11);
}

enum class QkNormMode : std::uint8_t {
    Identity,
    L2,
};

enum class BetaMode : std::uint8_t {
    Raw,
    Sigmoid,
    TwoSigmoid,
};

enum class GateMode : std::uint8_t {
    PrecomputedStep,
    Softplus,
    SafeSigmoid,
};

enum class Pow2Primitive : std::uint8_t {
    ExpLn2,
    Exp2,
};

constexpr Pow2Primitive ResolvePow2Primitive(bool useExp2) noexcept
{
    return useExp2 ? Pow2Primitive::Exp2 : Pow2Primitive::ExpLn2;
}

struct ProposedTilingKey {
    // q/k and q/k-derived public outputs.
    InputStorage inputStorage = InputStorage::Bf16;
    // v/u are an independent public dtype axis with the same 2-byte width.
    InputStorage valueStorage = InputStorage::Bf16;
    GateStorage gateStorage = GateStorage::Bf16;
    ScoreStorage scoreStorage = ScoreStorage::Bf16;
    AkkStorage akkStorage = AkkStorage::TwoByteAbi;
    PrepareAbi abi = PrepareAbi::Current;
    QkNormMode qkNormMode = QkNormMode::Identity;
    BetaMode betaMode = BetaMode::Raw;
    GateMode gateMode = GateMode::PrecomputedStep;
    // Public gk remains log2-valued in both specializations. This axis only
    // selects exp2(x) versus exp(x * ln(2)) when materializing 2^x.
    bool useExp2 = true;
    bool safeGate = false;
};

static_assert(IsTwoByteGateStorage(GateStorage::Fp16) &&
                  IsTwoByteGateStorage(GateStorage::Bf16) &&
                  !IsTwoByteGateStorage(GateStorage::Fp32) &&
                  ProposedTilingKey{}.inputStorage == InputStorage::Bf16 &&
                  ProposedTilingKey{}.valueStorage == InputStorage::Bf16 &&
                  ProposedTilingKey{}.gateStorage == GateStorage::Bf16 &&
                  ProposedTilingKey{}.qkNormMode == QkNormMode::Identity &&
                  ProposedTilingKey{}.useExp2 &&
                  ResolvePow2Primitive(true) == Pow2Primitive::Exp2 &&
                  ResolvePow2Primitive(false) == Pow2Primitive::ExpLn2,
              "Prepare public defaults and both pow2 implementations drifted");

constexpr bool IsSupportedStorageMapping(InputStorage inputStorage,
                                         ScoreStorage scoreStorage,
                                         bool safeGate) noexcept
{
    const bool sameStorage =
        (inputStorage == InputStorage::Fp16 &&
         scoreStorage == ScoreStorage::Fp16) ||
        (inputStorage == InputStorage::Bf16 &&
         scoreStorage == ScoreStorage::Bf16);
    if (safeGate && inputStorage == InputStorage::Fp16) {
        return scoreStorage == ScoreStorage::Bf16;
    }
    return sameStorage;
}

constexpr bool IsSupportedStorageMapping(
    const ProposedTilingKey &key) noexcept
{
    return IsSupportedStorageMapping(key.inputStorage, key.scoreStorage,
                                     key.safeGate);
}

constexpr bool IsSupportedGateMapping(GateMode gateMode,
                                      bool safeGate) noexcept
{
    return gateMode == GateMode::PrecomputedStep ||
           (gateMode == GateMode::Softplus && !safeGate) ||
           (gateMode == GateMode::SafeSigmoid && safeGate);
}

constexpr bool IsSupportedTilingKey(const ProposedTilingKey &key) noexcept
{
    return key.akkStorage == AkkStorage::TwoByteAbi &&
           IsSupportedStorageMapping(key) &&
           IsSupportedGateMapping(key.gateMode, key.safeGate);
}

constexpr MatrixStorage FromInputStorage(InputStorage storage) noexcept
{
    return storage == InputStorage::Fp16 ? MatrixStorage::Fp16
                                         : MatrixStorage::Bf16;
}

constexpr MatrixStorage FromScoreStorage(ScoreStorage storage) noexcept
{
    return storage == ScoreStorage::Fp16 ? MatrixStorage::Fp16
                                         : MatrixStorage::Bf16;
}

namespace storage_mapping_contract {

constexpr ProposedTilingKey MakeKey(InputStorage inputStorage,
                                    ScoreStorage scoreStorage,
                                    GateMode gateMode,
                                    bool safeGate) noexcept
{
    ProposedTilingKey key{};
    key.inputStorage = inputStorage;
    key.scoreStorage = scoreStorage;
    key.gateMode = gateMode;
    key.safeGate = safeGate;
    return key;
}

} // namespace storage_mapping_contract

static_assert(
    IsSupportedStorageMapping(storage_mapping_contract::MakeKey(
        InputStorage::Fp16, ScoreStorage::Bf16,
        GateMode::PrecomputedStep, true)),
    "SAFE_GATE FP16 must promote score to BF16 independently of GateMode");
static_assert(
    !IsSupportedStorageMapping(storage_mapping_contract::MakeKey(
        InputStorage::Fp16, ScoreStorage::Fp16,
        GateMode::SafeSigmoid, true)),
    "SAFE_GATE FP16 must reject an FP16 score even in SafeSigmoid mode");
static_assert(
    IsSupportedStorageMapping(storage_mapping_contract::MakeKey(
        InputStorage::Fp16, ScoreStorage::Fp16,
        GateMode::PrecomputedStep, false)) &&
        IsSupportedStorageMapping(storage_mapping_contract::MakeKey(
            InputStorage::Bf16, ScoreStorage::Bf16,
            GateMode::Softplus, false)),
    "non-SAFE_GATE mappings must preserve the input storage dtype");
static_assert(
    !IsSupportedStorageMapping(storage_mapping_contract::MakeKey(
        InputStorage::Bf16, ScoreStorage::Fp16,
        GateMode::SafeSigmoid, false)),
    "every BF16-to-FP16 score mapping must be rejected");
static_assert(IsSupportedGateMapping(GateMode::PrecomputedStep, false) &&
                  IsSupportedGateMapping(GateMode::PrecomputedStep, true) &&
                  IsSupportedGateMapping(GateMode::Softplus, false) &&
                  IsSupportedGateMapping(GateMode::SafeSigmoid, true) &&
                  !IsSupportedGateMapping(GateMode::Softplus, true) &&
                  !IsSupportedGateMapping(GateMode::SafeSigmoid, false),
              "GateMode must be derivable from public use_gate/safe_gate");
static_assert(FromInputStorage(InputStorage::Fp16) == MatrixStorage::Fp16 &&
                  FromInputStorage(InputStorage::Bf16) == MatrixStorage::Bf16 &&
                  FromScoreStorage(ScoreStorage::Fp16) == MatrixStorage::Fp16 &&
                  FromScoreStorage(ScoreStorage::Bf16) == MatrixStorage::Bf16,
              "Cube operand storage must preserve the selected two-byte dtype");

constexpr float ScoreExp2InputMin(ScoreStorage storage) noexcept
{
    return storage == ScoreStorage::Bf16 ? kBf16ScoreExp2InputMin
                                         : kFp16ScoreExp2InputMin;
}

constexpr float ScoreExp2InputMax(ScoreStorage storage) noexcept
{
    return storage == ScoreStorage::Bf16 ? kBf16ScoreExp2InputMax
                                         : kFp16ScoreExp2InputMax;
}

constexpr bool RequiresFiniteSaturation(InputStorage storage) noexcept
{
    return storage == InputStorage::Fp16;
}

constexpr bool RequiresFiniteSaturation(ScoreStorage storage) noexcept
{
    return storage == ScoreStorage::Fp16;
}

static_assert(ScoreExp2InputMin(ScoreStorage::Bf16) == -126.0F &&
                  ScoreExp2InputMax(ScoreStorage::Bf16) == 120.0F,
              "BF16 score Exp2 must preserve the split Prepare range");
static_assert(ScoreExp2InputMin(ScoreStorage::Fp16) == -80.0F &&
                  ScoreExp2InputMax(ScoreStorage::Fp16) == 80.0F,
              "FP16 score Exp2 must preserve the direct range");
static_assert(RequiresFiniteSaturation(InputStorage::Fp16) &&
                  !RequiresFiniteSaturation(InputStorage::Bf16) &&
                  RequiresFiniteSaturation(ScoreStorage::Fp16) &&
                  !RequiresFiniteSaturation(ScoreStorage::Bf16),
              "only FP16 storage applies the finite +/-65504 saturation");
static_assert(kFp16FiniteMin == -kFp16FiniteMax,
              "FP16 finite saturation must be symmetric");

struct RuntimeTiling {
    std::uint32_t sequenceCount = 0;
    std::uint32_t totalChunks = 0;
    // headCount is HV. qkHeadCount is HK; zero is a host-test shorthand for
    // HK=HV, while production Host tiling must write the shape-derived HK.
    std::uint32_t headCount = 0;
    std::uint32_t qkHeadCount = 0;
    std::uint32_t aicWorkgroupCount = 0;
    Architecture architecture = Architecture::Arch35;
    ProposedTilingKey key{};
    float epsilon = 1.0e-6F;
    float lowerBound = -5.0F;
    float scale = 1.0F;
    bool hasDtBias = false;
};

static_assert(RuntimeTiling{}.epsilon == 1.0e-6F &&
                  RuntimeTiling{}.lowerBound == -5.0F,
              "runtime scalar defaults must match the public Prepare API");

struct WorkspaceSizing {
    bool valid = false;
    std::uint64_t workgroupBase = 0;
    std::uint64_t totalBytes = 0;
};

constexpr WorkspaceSizing CheckedWorkspaceSizing(
    Architecture architecture, std::uint64_t workgroupCount,
    std::uint64_t workgroupId) noexcept
{
    constexpr std::uint64_t limit =
        std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t stride = WorkspaceWorkgroupStrideFor(architecture);
    if (!IsKnownArchitecture(architecture) || workgroupCount == 0 ||
        workgroupId >= workgroupCount ||
        stride == 0 || workgroupCount > limit / stride) {
        return {};
    }
    return {true,
            workgroupId * stride,
            workgroupCount * stride};
}

constexpr WorkspaceSizing CheckedWorkspaceSizing(
    std::uint64_t workgroupCount, std::uint64_t workgroupId) noexcept
{
    return CheckedWorkspaceSizing(Architecture::Arch35, workgroupCount,
                                  workgroupId);
}

static_assert(CheckedWorkspaceSizing(Architecture::Arch35, 8, 7).valid &&
                  CheckedWorkspaceSizing(Architecture::Arch35, 8, 7)
                          .workgroupBase ==
                      7 * kWorkspaceWorkgroupStrideBytes &&
                  CheckedWorkspaceSizing(Architecture::Arch35, 8, 7)
                          .totalBytes ==
                      8 * kWorkspaceWorkgroupStrideBytes,
              "checked workspace sizing must preserve valid u64 products");
static_assert(
    CheckedWorkspaceSizing(Architecture::Arch22, 8, 7).valid &&
        CheckedWorkspaceSizing(Architecture::Arch22, 8, 7).workgroupBase ==
            7 * kArch22WorkspaceWorkgroupStrideBytes &&
        CheckedWorkspaceSizing(Architecture::Arch22, 8, 7).totalBytes ==
            8 * kArch22WorkspaceWorkgroupStrideBytes,
    "Arch22 checked workspace sizing must use its four-slot stride");
static_assert(
    !CheckedWorkspaceSizing(
         Architecture::Arch35,
         std::numeric_limits<std::uint64_t>::max() /
                  kWorkspaceWorkgroupStrideBytes +
              1,
         0)
         .valid,
    "checked workspace sizing must reject u64 multiplication overflow");
static_assert(
    !CheckedWorkspaceSizing(
         Architecture::Arch22,
         std::numeric_limits<std::uint64_t>::max() /
                 kArch22WorkspaceWorkgroupStrideBytes +
             1,
         0)
         .valid,
    "Arch22 checked sizing must reject u64 multiplication overflow");

// PROPOSED Host contract: beta is one FP32 scalar per token at this split-kernel
// boundary. L2 may accept BF16/FP32 publicly but must cast to FP32 before launch;
// beta storage is therefore not a kernel template axis. Reject non-finite
// scalars before launch; L2 norm requires a positive epsilon. Selective gate
// modes require A_log and use dt_bias only when hasDtBias is true; false means
// exact zero and no GM read. InputStorage selects q/k and their derived
// outputs; valueStorage independently selects v/u. ScoreStorage is a distinct
// compile-time semantic because a generic two-byte label cannot choose the
// FP16/BF16 Exp2 clamp. Host must reject unimplemented input/score and
// gate/safeGate mappings. epsilon/lowerBound/scale/hasDtBias remain runtime data;
// the enum axes are compile-time semantics whose numeric key encoding is not
// frozen by this pseudocode. Every FP16 write first saturates to
// [kFp16FiniteMin,kFp16FiniteMax] and then uses RINT; BF16 writes use RINT
// without that finite-magnitude saturation.

} // namespace kda_prepare_pseudocode

#endif // PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
