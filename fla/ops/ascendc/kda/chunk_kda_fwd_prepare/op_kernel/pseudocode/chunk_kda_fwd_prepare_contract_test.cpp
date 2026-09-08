/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "chunk_kda_fwd_prepare_policy.h"
#include "chunk_kda_fwd_prepare_struct.h"
#include "chunk_kda_fwd_prepare_tiling_key.h"
#include "chunk_kda_fwd_prepare_utils.h"
#include "arch22/chunk_kda_fwd_prepare_vec.h"
#include "arch35/chunk_kda_fwd_prepare_vec.h"

namespace kda_prepare_pseudocode {

void RunChunkKdaFwdPrepareArch22Contract(
    const RuntimeTiling &tiling, std::uint32_t workgroupId, CoreRole role,
    std::uint32_t aivId, WorkspaceView &workspace, SyncLedger &sync,
    VectorOps &vectorOps, CubeOps &cubeOps,
    ChunkTask (*resolveChunk)(std::uint32_t)) noexcept;

void RunChunkKdaFwdPrepareArch35Contract(
    const RuntimeTiling &tiling, std::uint32_t workgroupId, CoreRole role,
    std::uint32_t aivId, WorkspaceView &workspace, SyncLedger &sync,
    VectorOps &vectorOps, CubeOps &cubeOps,
    ChunkTask (*resolveChunk)(std::uint32_t)) noexcept;

namespace {

using ResolveChunk = ChunkTask (*)(std::uint32_t);

void RunArchitectureContract(
    Architecture architecture, const RuntimeTiling &tiling,
    std::uint32_t workgroupId, CoreRole role, std::uint32_t aivId,
    WorkspaceView &workspace, SyncLedger &sync, VectorOps &vectorOps,
    CubeOps &cubeOps, ResolveChunk resolveChunk) noexcept
{
    if (architecture == Architecture::Arch22) {
        RunChunkKdaFwdPrepareArch22Contract(
            tiling, workgroupId, role, aivId, workspace, sync, vectorOps,
            cubeOps, resolveChunk);
        return;
    }
    RunChunkKdaFwdPrepareArch35Contract(
        tiling, workgroupId, role, aivId, workspace, sync, vectorOps,
        cubeOps, resolveChunk);
}

constexpr std::uint32_t kNoAivId =
    std::numeric_limits<std::uint32_t>::max();

struct Pow2Probe {
    bool exp2Called = false;
    bool clampCalled = false;
    bool multiplyCalled = false;
    bool expCalled = false;
    float clampMinimum = 0.0F;
    float clampMaximum = 0.0F;
    float naturalExponent = 0.0F;

    float Exp2Clamped(float value, float minimum, float maximum) noexcept
    {
        exp2Called = true;
        clampMinimum = minimum;
        clampMaximum = maximum;
        naturalExponent = value;
        return 11.0F;
    }

    float Clamp(float value, float minimum, float maximum) noexcept
    {
        clampCalled = true;
        clampMinimum = minimum;
        clampMaximum = maximum;
        return value < minimum ? minimum : (value > maximum ? maximum
                                                            : value);
    }

    float Ln2() const noexcept
    {
        // 使用哨兵值使乘法路径可观测，而不依赖 libm。
        return 0.5F;
    }

    float Mul(float lhs, float rhs) noexcept
    {
        multiplyCalled = true;
        return lhs * rhs;
    }

    float Exp(float value) noexcept
    {
        expCalled = true;
        naturalExponent = value;
        return 22.0F;
    }
};

constexpr float kRuntimeScaleProbeValue = 0.375F;

struct VfSemanticProbe {
    explicit VfSemanticProbe(float runtimeScaleValue) noexcept
        : runtimeScale(runtimeScaleValue)
    {}

    float runtimeScale = 0.0F;
    float lastRuntimeScaleProduct = 0.0F;
    std::uint32_t runtimeScaleMultiplyCount = 0U;
    std::uint32_t exp2CallCount = 0U;
    std::uint32_t expCallCount = 0U;
    std::uint32_t l2NormalizeCount = 0U;
    std::uint32_t normSourceLoadCount = 0U;
    std::uint32_t normDestinationStoreCount = 0U;
    std::uint32_t normWorkDependencyCount = 0U;

    float Exp2Clamped(float, float, float) noexcept
    {
        ++exp2CallCount;
        return 7.0F;
    }

    float Clamp(float value, float minimum, float maximum) const noexcept
    {
        return value < minimum ? minimum : (value > maximum ? maximum
                                                            : value);
    }

    float Ln2() const noexcept
    {
        return 0.5F;
    }

    float Exp(float) noexcept
    {
        ++expCallCount;
        return 7.0F;
    }

    float Mul(float lhs, float rhs) noexcept
    {
        const float result = lhs * rhs;
        if (lhs == runtimeScale) {
            ++runtimeScaleMultiplyCount;
            lastRuntimeScaleProduct = result;
        }
        return result;
    }

    float Add(float lhs, float rhs) const noexcept
    {
        return lhs + rhs;
    }

    float Sub(float lhs, float rhs) const noexcept
    {
        return lhs - rhs;
    }

    float Div(float lhs, float rhs) const noexcept
    {
        return lhs / rhs;
    }

    float Max(float lhs, float rhs) const noexcept
    {
        return lhs > rhs ? lhs : rhs;
    }

    float Neg(float value) const noexcept
    {
        return -value;
    }

    float Abs(float value) const noexcept
    {
        return value < 0.0F ? -value : value;
    }

    float Log1p(float value) const noexcept
    {
        return value;
    }

    float Sigmoid(float) const noexcept
    {
        return 0.5F;
    }

    float OneFp32() const noexcept
    {
        return 1.0F;
    }

    float ZeroFp32() const noexcept
    {
        return 0.0F;
    }

    float ZeroFp32Row(std::uint32_t) const noexcept
    {
        return 0.0F;
    }

    template <typename... Args>
    float LoadALogScalarOnce(Args...) const noexcept
    {
        return 2.0F;
    }

    template <typename... Args>
    float LoadDtBiasRow(Args...) const noexcept
    {
        return 2.0F;
    }

    template <typename... Args>
    float LoadGateRow(Args...) const noexcept
    {
        return 2.0F;
    }

    template <typename... Args>
    float LoadBetaFp32Scalar(Args...) const noexcept
    {
        return 2.0F;
    }

    template <typename... Args>
    float LoadFp32Row(Args...) const noexcept
    {
        return 3.0F;
    }

    template <typename... Args>
    float LoadBetaEff(Args...) const noexcept
    {
        return 5.0F;
    }

    template <typename... Args>
    float LoadStorageRow(Args...) noexcept
    {
        ++normSourceLoadCount;
        return 2.0F;
    }

    template <typename... Args>
    float LoadQStorageRow(Args...) noexcept
    {
        ++normSourceLoadCount;
        return 2.0F;
    }

    template <typename... Args>
    float LoadKStorageRow(Args...) noexcept
    {
        ++normSourceLoadCount;
        return 2.0F;
    }

    float ToFp32(float value) const noexcept
    {
        return value;
    }

    float ClampForInputStorage(float value, InputStorage) const noexcept
    {
        return value;
    }

    float RoundToInputStorage(float value, InputStorage) const noexcept
    {
        return value;
    }

    template <typename... Args>
    float L2NormalizeRsqrtSumPlusEpsilon(Args...) noexcept
    {
        ++l2NormalizeCount;
        return 2.0F;
    }

    template <typename... Args>
    float L2NormalizeRsqrtSumPlusEpsilonWithWork(Args...) noexcept
    {
        ++l2NormalizeCount;
        return 2.0F;
    }

    void DependNormWorkOnGateLastReader() noexcept
    {
        ++normWorkDependencyCount;
    }

    template <typename... Args>
    void StoreStorageRow(Args...) noexcept
    {
        ++normDestinationStoreCount;
    }

    template <typename... Args>
    void StoreFp32Row(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreBetaEffScalar(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreGLast(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreZeroQHatKHatAndGPadding(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreZeroV6PrivateRows(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreZeroV6OutputRows(Args...) const noexcept
    {}

    template <typename... Args>
    float LoadCompactRawAqk(Args...) const noexcept
    {
        return 11.0F;
    }

    template <typename... Args>
    float LoadCompactRawAkk(Args...) const noexcept
    {
        return 13.0F;
    }

    template <typename... Args>
    float LoadRawAqk(Args...) const noexcept
    {
        return 11.0F;
    }

    template <typename... Args>
    float LoadRawAkk(Args...) const noexcept
    {
        return 13.0F;
    }

    template <typename... Args>
    float LoadAqk(Args...) const noexcept
    {
        return 17.0F;
    }

    template <typename... Args>
    float LoadStableAkkOrZeroColumnPadding(Args...) const noexcept
    {
        return 17.0F;
    }

    template <typename... Args>
    float LoadStableAkkQ00OrZeroColumnPadding(Args...) const noexcept
    {
        return 17.0F;
    }

    template <typename... Args>
    float LoadStableAkkQ11OrZeroColumnPadding(Args...) const noexcept
    {
        return 17.0F;
    }

    template <typename... Args>
    void StoreStorageMatrix(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreInputStorageMatrix(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreAqk(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreAqkStorageInPlace(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreLkkOrIdentityPaddingAt(Args...) const noexcept
    {}

    template <typename... Args>
    void StoreLkkOrIdentityPadding(Args...) const noexcept
    {}

    void DependLateOutputsOnCompactRawLastReader() const noexcept
    {}

    template <typename... Args>
    void InvertTwo32By32LeavesWithFixedColumnScan(Args...) const noexcept
    {}

    template <typename... Args>
    void MaterializeX0X1AndBAtFinalOffsets(Args...) const noexcept
    {}
};

bool CheckPow2EvaluationContract() noexcept
{
    Pow2Probe native{};
    Pow2Probe natural{};
    const float nativeResult =
        EvaluatePow2<true>(native, 4.0F, -2.0F, 3.0F);
    const float naturalResult =
        EvaluatePow2<false>(natural, 4.0F, -2.0F, 3.0F);
    return ResolvePow2Primitive(true) == Pow2Primitive::Exp2 &&
           ResolvePow2Primitive(false) == Pow2Primitive::ExpLn2 &&
           nativeResult == 11.0F && native.exp2Called &&
           !native.clampCalled && !native.multiplyCalled &&
           !native.expCalled && native.clampMinimum == -2.0F &&
           native.clampMaximum == 3.0F && naturalResult == 22.0F &&
           !natural.exp2Called && natural.clampCalled &&
           natural.multiplyCalled && natural.expCalled &&
           natural.clampMinimum == -2.0F &&
           natural.clampMaximum == 3.0F &&
           natural.naturalExponent == 1.5F;
}

template <bool UseExp2>
bool CheckV6RuntimeScaleSemanticContract() noexcept
{
    HeadTask head{};
    head.active = true;
    head.qkOwner = true;

    VfSemanticProbe arch22Current{kRuntimeScaleProbeValue};
    arch22::detail::V6OneVf<UseExp2>(
        arch22Current, head, 1U, PrepareAbi::Current, InputStorage::Bf16,
        InputStorage::Fp16, kRuntimeScaleProbeValue);
    VfSemanticProbe arch22Fused{kRuntimeScaleProbeValue};
    arch22::detail::V6OneVf<UseExp2>(
        arch22Fused, head, 1U, PrepareAbi::Fused, InputStorage::Bf16,
        InputStorage::Fp16, kRuntimeScaleProbeValue);
    VfSemanticProbe arch35Current{kRuntimeScaleProbeValue};
    arch35::detail::V6OneVf<UseExp2>(
        arch35Current, head, 1U, PrepareAbi::Current, InputStorage::Bf16,
        InputStorage::Fp16, kRuntimeScaleProbeValue);
    VfSemanticProbe arch35Fused{kRuntimeScaleProbeValue};
    arch35::detail::V6OneVf<UseExp2>(
        arch35Fused, head, 1U, PrepareAbi::Fused, InputStorage::Bf16,
        InputStorage::Fp16, kRuntimeScaleProbeValue);

    const auto pow2Count = [](const VfSemanticProbe &probe) noexcept {
        return UseExp2 ? probe.exp2CallCount : probe.expCallCount;
    };
    return arch22Current.runtimeScaleMultiplyCount == 0U &&
           arch35Current.runtimeScaleMultiplyCount == 0U &&
           arch22Fused.runtimeScaleMultiplyCount == 1U &&
           arch35Fused.runtimeScaleMultiplyCount == 1U &&
           arch22Fused.lastRuntimeScaleProduct ==
               kRuntimeScaleProbeValue * 14.0F &&
           arch35Fused.lastRuntimeScaleProduct ==
               kRuntimeScaleProbeValue * 14.0F &&
           pow2Count(arch22Current) == 2U &&
           pow2Count(arch22Fused) == 2U &&
           pow2Count(arch35Current) == 2U &&
           pow2Count(arch35Fused) == 2U &&
           (UseExp2 ? arch22Current.expCallCount == 0U
                    : arch22Current.exp2CallCount == 0U) &&
           (UseExp2 ? arch35Current.expCallCount == 0U
                    : arch35Current.exp2CallCount == 0U);
}

bool CheckRuntimeScaleSemanticContract() noexcept
{
    HeadTask head{};
    head.active = true;
    head.qkOwner = true;

    VfSemanticProbe arch22V3{kRuntimeScaleProbeValue};
    arch22::detail::V3OneVf(
        arch22V3, head, 1U, PrepareAbi::Current, InputStorage::Bf16,
        kRuntimeScaleProbeValue);
    VfSemanticProbe arch35V3{kRuntimeScaleProbeValue};
    arch35::detail::V3OneVf(
        arch35V3, head, 1U, PrepareAbi::Current, AkkStorage::TwoByteAbi,
        InputStorage::Bf16, kRuntimeScaleProbeValue);
    return arch22V3.runtimeScaleMultiplyCount == 1U &&
           arch35V3.runtimeScaleMultiplyCount == 1U &&
           arch22V3.lastRuntimeScaleProduct ==
               kRuntimeScaleProbeValue * 11.0F &&
           arch35V3.lastRuntimeScaleProduct ==
               kRuntimeScaleProbeValue * 11.0F &&
           CheckV6RuntimeScaleSemanticContract<true>() &&
           CheckV6RuntimeScaleSemanticContract<false>();
}

bool CheckV0QkNormalizeSemanticContract() noexcept
{
    constexpr std::uint32_t kValidRows = kChunkRows;
    ProposedTilingKey identityKey{};
    identityKey.inputStorage = InputStorage::Bf16;
    identityKey.gateStorage = GateStorage::Bf16;
    identityKey.qkNormMode = QkNormMode::Identity;
    ProposedTilingKey l2Key = identityKey;
    l2Key.qkNormMode = QkNormMode::L2;

    HeadTask owner{};
    owner.active = true;
    owner.qkOwner = true;
    HeadTask nonOwner = owner;
    nonOwner.qkOwner = false;

    VfSemanticProbe arch22L2Owner{kRuntimeScaleProbeValue};
    arch22::detail::V0OneVf(arch22L2Owner, owner, kValidRows, l2Key,
                            1.0e-6F, -5.0F);
    VfSemanticProbe arch22IdentityOwner{kRuntimeScaleProbeValue};
    arch22::detail::V0OneVf(arch22IdentityOwner, owner, kValidRows,
                            identityKey, 1.0e-6F, -5.0F);
    VfSemanticProbe arch22L2NonOwner{kRuntimeScaleProbeValue};
    arch22::detail::V0OneVf(arch22L2NonOwner, nonOwner, kValidRows, l2Key,
                            1.0e-6F, -5.0F);
    VfSemanticProbe arch35L2Owner{kRuntimeScaleProbeValue};
    arch35::detail::V0OneVf(arch35L2Owner, owner, kValidRows, l2Key,
                            1.0e-6F, -5.0F);
    VfSemanticProbe arch35IdentityOwner{kRuntimeScaleProbeValue};
    arch35::detail::V0OneVf(arch35IdentityOwner, owner, kValidRows,
                            identityKey, 1.0e-6F, -5.0F);
    VfSemanticProbe arch35L2NonOwner{kRuntimeScaleProbeValue};
    arch35::detail::V0OneVf(arch35L2NonOwner, nonOwner, kValidRows, l2Key,
                            1.0e-6F, -5.0F);

    constexpr std::uint32_t kExpectedQkRows = 2U * kValidRows;
    return arch22L2Owner.l2NormalizeCount == kExpectedQkRows &&
           arch22L2Owner.normSourceLoadCount == kExpectedQkRows &&
           arch22L2Owner.normDestinationStoreCount == kExpectedQkRows &&
           arch22L2Owner.normWorkDependencyCount == 1U &&
           arch35L2Owner.l2NormalizeCount == kExpectedQkRows &&
           arch35L2Owner.normSourceLoadCount == kExpectedQkRows &&
           arch35L2Owner.normDestinationStoreCount == kExpectedQkRows &&
           arch35L2Owner.normWorkDependencyCount == 0U &&
           arch22IdentityOwner.l2NormalizeCount == 0U &&
           arch22IdentityOwner.normSourceLoadCount == 0U &&
           arch22IdentityOwner.normDestinationStoreCount == 0U &&
           arch22IdentityOwner.normWorkDependencyCount == 0U &&
           arch35IdentityOwner.l2NormalizeCount == 0U &&
           arch35IdentityOwner.normSourceLoadCount == 0U &&
           arch35IdentityOwner.normDestinationStoreCount == 0U &&
           arch22L2NonOwner.l2NormalizeCount == 0U &&
           arch22L2NonOwner.normSourceLoadCount == 0U &&
           arch22L2NonOwner.normDestinationStoreCount == 0U &&
           arch22L2NonOwner.normWorkDependencyCount == 0U &&
           arch35L2NonOwner.l2NormalizeCount == 0U &&
           arch35L2NonOwner.normSourceLoadCount == 0U &&
           arch35L2NonOwner.normDestinationStoreCount == 0U;
}

ChunkTask ResolveTail16(std::uint32_t chunkOrdinal) noexcept
{
    return {0U, chunkOrdinal, chunkOrdinal, 16U};
}

ChunkTask ResolveTail1(std::uint32_t chunkOrdinal) noexcept
{
    return {0U, chunkOrdinal, chunkOrdinal, 1U};
}

ChunkTask ResolveTail17(std::uint32_t chunkOrdinal) noexcept
{
    return {0U, chunkOrdinal, chunkOrdinal, 17U};
}

ChunkTask ResolveTail32(std::uint32_t chunkOrdinal) noexcept
{
    return {0U, chunkOrdinal, chunkOrdinal, 32U};
}

ChunkTask ResolveTail33(std::uint32_t chunkOrdinal) noexcept
{
    return {0U, chunkOrdinal, chunkOrdinal, 33U};
}

ChunkTask ResolveTail49(std::uint32_t chunkOrdinal) noexcept
{
    return {0U, chunkOrdinal, chunkOrdinal, 49U};
}

ChunkTask ResolveTail63(std::uint32_t chunkOrdinal) noexcept
{
    return {0U, chunkOrdinal, chunkOrdinal, 63U};
}

ChunkTask ResolveDense(std::uint32_t chunkOrdinal) noexcept
{
    return {0U, chunkOrdinal, chunkOrdinal, kChunkRows};
}

constexpr bool ExpectedV3C2RawDefined(std::uint32_t validRows,
                                      std::uint32_t row,
                                      std::uint32_t column) noexcept
{
    const std::uint32_t activeRows =
        ((validRows + kScoreBlockRows - 1U) / kScoreBlockRows) *
        kScoreBlockRows;
    return row < activeRows &&
           column < ((row / kScoreBlockRows) + 1U) * kScoreBlockRows;
}

constexpr bool ExpectedV3C5Q10Write(Architecture architecture,
                                    std::uint32_t validRows,
                                    std::uint32_t row,
                                    std::uint32_t column) noexcept
{
    return validRows > 32U && row >= 32U && column < 32U &&
           (architecture == Architecture::Arch35 || row < validRows);
}

constexpr bool ExpectedV3StableWrite(Architecture architecture,
                                     PrepareAbi abi,
                                     std::uint32_t validRows,
                                     std::uint32_t row,
                                     std::uint32_t column) noexcept
{
    if (ExpectedV3C5Q10Write(architecture, validRows, row, column)) {
        return false;
    }
    if (architecture == Architecture::Arch22) {
        const bool fusedTopQ01 =
            abi == PrepareAbi::Fused && validRows <= 32U && row < 32U &&
            column >= 32U;
        return row < validRows && !fusedTopQ01;
    }
    const bool q00 = row < 32U && column < 32U;
    const bool q01 = row < 32U && column >= 32U;
    const bool q11 = row >= 32U && column >= 32U;
    if (abi == PrepareAbi::Current) {
        return row < validRows && (q00 || q01 || q11);
    }
    return q00 || (validRows > 32U && q11);
}

bool CheckV3SemanticContracts() noexcept
{
    constexpr std::array<Architecture, 2U> kArchitectures = {{
        Architecture::Arch22,
        Architecture::Arch35,
    }};
    constexpr std::array<PrepareAbi, 2U> kAbis = {{
        PrepareAbi::Current,
        PrepareAbi::Fused,
    }};
    for (std::uint32_t validRows = 1U; validRows <= kChunkRows;
         ++validRows) {
        for (std::uint32_t row = 0U; row < kChunkRows; ++row) {
            for (std::uint32_t column = 0U; column < kChunkRows; ++column) {
                if (V3C2RawDefined(validRows, row, column) !=
                    ExpectedV3C2RawDefined(validRows, row, column)) {
                    return false;
                }
                const bool expectedAqkRead =
                    row < validRows && column < validRows && column <= row;
                const bool expectedAkkRead =
                    row < validRows && column < validRows && column < row;
                if (V3AqkRawReadRequired(validRows, row, column) !=
                        expectedAqkRead ||
                    V3AkkRawReadRequired(validRows, row, column) !=
                        expectedAkkRead) {
                    return false;
                }
                for (Architecture architecture : kArchitectures) {
                    const bool q10 = V3C5Q10WriteRequired(
                        architecture, validRows, row, column);
                    if (q10 != ExpectedV3C5Q10Write(
                                   architecture, validRows, row, column)) {
                        return false;
                    }
                    for (PrepareAbi abi : kAbis) {
                        const bool stable = V3StableAkkWriteRequired(
                            architecture, abi, validRows, row, column);
                        if (stable != ExpectedV3StableWrite(
                                          architecture, abi, validRows, row,
                                          column) ||
                            (stable && q10)) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return !V3C2RawDefined(65U, 0U, 0U) &&
           !V3StableAkkWriteRequired(Architecture::Arch22,
                                     PrepareAbi::Current, 65U, 0U, 0U) &&
           !V3C5Q10WriteRequired(Architecture::Arch35, 65U, 32U, 0U);
}

struct TraceCursor {
    const SyncTrace &trace;
    std::size_t index = 0U;

    bool Match(SyncAction action, SyncPoint point, std::uint32_t ownerId,
               std::uint64_t generation, std::uint32_t aivId,
               bool hasActiveHead, Stage stage, Pipe pipe) noexcept
    {
        if (index >= trace.size) {
            return false;
        }
        const SyncRecord &record = trace.records[index];
        if (record.action != action || record.point != point ||
            record.ownerId != ownerId || record.generation != generation ||
            record.aivId != aivId ||
            record.hasActiveHead != hasActiveHead ||
            record.stage != stage || record.pipe != pipe) {
            return false;
        }
        ++index;
        return true;
    }

    bool Complete() const noexcept
    {
        return !trace.overflow && index == trace.size;
    }
};

struct LocalTraceCursor {
    const LocalSyncTrace &trace;
    std::size_t index = 0U;

    bool Match(LocalDependency dependency, Stage stage) noexcept
    {
        if (index >= trace.size) {
            return false;
        }
        const LocalSyncRecord &record = trace.records[index];
        if (record.dependency != dependency || record.stage != stage) {
            return false;
        }
        ++index;
        return true;
    }

    bool Complete() const noexcept
    {
        return !trace.overflow && index == trace.size;
    }
};

bool SameName(const char *actual, const char *expected) noexcept
{
    return expected == nullptr ||
           (actual != nullptr && std::strcmp(actual, expected) == 0);
}

bool ContainsName(const char *actual, const char *needle) noexcept
{
    return actual != nullptr && needle != nullptr &&
           std::strstr(actual, needle) != nullptr;
}

bool IsWriteOperation(OperationKind kind) noexcept
{
    return kind == OperationKind::Store ||
           kind == OperationKind::StoreRounded ||
           kind == OperationKind::Fill || kind == OperationKind::Zero ||
           kind == OperationKind::ZeroUndefined;
}

std::size_t CountWritesContaining(const OperationTrace &trace, Stage stage,
                                  const char *nameFragment) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.stage == stage && IsWriteOperation(record.kind) &&
            ContainsName(record.destination.name, nameFragment)) {
            ++count;
        }
    }
    return count;
}

std::size_t CountOperations(const OperationTrace &trace, OperationKind kind,
                            Stage stage, const char *sourceName = nullptr,
                            const char *destinationName = nullptr) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.kind == kind && record.stage == stage &&
            SameName(record.source.name, sourceName) &&
            SameName(record.destination.name, destinationName)) {
            ++count;
        }
    }
    return count;
}

const OperationRecord *FindUniqueOperation(
    const OperationTrace &trace, OperationKind kind, Stage stage,
    const char *sourceName = nullptr,
    const char *destinationName = nullptr) noexcept
{
    const OperationRecord *match = nullptr;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.kind != kind || record.stage != stage ||
            !SameName(record.source.name, sourceName) ||
            !SameName(record.destination.name, destinationName)) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = &record;
    }
    return match;
}

const OperationRecord *FindUniqueOperationForSlots(
    const OperationTrace &trace, OperationKind kind, Stage stage,
    const char *sourceName, const char *destinationName,
    std::uint32_t sourceSlot, std::uint32_t destinationSlot) noexcept
{
    const OperationRecord *match = nullptr;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.kind != kind || record.stage != stage ||
            !SameName(record.source.name, sourceName) ||
            !SameName(record.destination.name, destinationName) ||
            record.source.slot != sourceSlot ||
            record.destination.slot != destinationSlot) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = &record;
    }
    return match;
}

std::size_t CountOperationsForSpanGenerations(
    const OperationTrace &trace, OperationKind kind, Stage stage,
    const char *sourceName, const char *destinationName,
    std::uint32_t sourceSlot, std::uint64_t sourceGeneration,
    std::uint32_t destinationSlot,
    std::uint64_t destinationGeneration) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.kind != kind || record.stage != stage ||
            !SameName(record.source.name, sourceName) ||
            !SameName(record.destination.name, destinationName) ||
            record.source.slot != sourceSlot ||
            record.source.generation != sourceGeneration ||
            record.destination.slot != destinationSlot ||
            record.destination.generation != destinationGeneration) {
            continue;
        }
        ++count;
    }
    return count;
}

const OperationRecord *FindUniqueOperationForSpanGenerations(
    const OperationTrace &trace, OperationKind kind, Stage stage,
    const char *sourceName, const char *destinationName,
    std::uint32_t sourceSlot, std::uint64_t sourceGeneration,
    std::uint32_t destinationSlot,
    std::uint64_t destinationGeneration) noexcept
{
    if (CountOperationsForSpanGenerations(
            trace, kind, stage, sourceName, destinationName, sourceSlot,
            sourceGeneration, destinationSlot, destinationGeneration) != 1U) {
        return nullptr;
    }
    const OperationRecord *match = nullptr;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.kind == kind && record.stage == stage &&
            SameName(record.source.name, sourceName) &&
            SameName(record.destination.name, destinationName) &&
            record.source.slot == sourceSlot &&
            record.source.generation == sourceGeneration &&
            record.destination.slot == destinationSlot &&
            record.destination.generation == destinationGeneration) {
            match = &record;
        }
    }
    return match;
}

bool CheckSpanIdentity(const BufferSpan &span, const char *name,
                       MemorySpace space, std::uint64_t byteOffset,
                       std::size_t byteSize, std::uint32_t slot,
                       std::uint64_t generation, CoreRole ownerRole,
                       std::uint32_t ownerId) noexcept
{
    return SameName(span.name, name) && span.space == space &&
           span.byteOffset == byteOffset && span.byteSize == byteSize &&
           span.slot == slot && span.generation == generation &&
           span.ownerRole == ownerRole && span.ownerId == ownerId;
}

bool SamePhysicalAllocation(const BufferSpan &lhs,
                            const BufferSpan &rhs) noexcept
{
    return lhs.space == rhs.space && lhs.slot == rhs.slot &&
           lhs.generation == rhs.generation &&
           lhs.ownerRole == rhs.ownerRole && lhs.ownerId == rhs.ownerId;
}

bool PhysicalByteRangesOverlap(const BufferSpan &lhs,
                               const BufferSpan &rhs) noexcept
{
    if (!SamePhysicalAllocation(lhs, rhs) || lhs.byteSize == 0U ||
        rhs.byteSize == 0U) {
        return false;
    }
    const std::uint64_t lhsEnd = lhs.byteOffset + lhs.byteSize;
    const std::uint64_t rhsEnd = rhs.byteOffset + rhs.byteSize;
    return lhs.byteOffset < rhsEnd && rhs.byteOffset < lhsEnd;
}

bool SamePhysicalSpan(const BufferSpan &lhs, const BufferSpan &rhs) noexcept
{
    return SamePhysicalAllocation(lhs, rhs) &&
           lhs.byteOffset == rhs.byteOffset && lhs.byteSize == rhs.byteSize;
}

bool CheckMatrixSpan(const BufferSpan &span, std::uint32_t rows,
                     std::uint32_t columns,
                     std::uint32_t leadingDimension,
                     std::uint32_t elementBytes) noexcept
{
    if (span.rows != rows || span.columns != columns ||
        span.leadingDimension != leadingDimension ||
        span.elementBytes != elementBytes || rows == 0U || columns == 0U) {
        return false;
    }
    const std::size_t expectedBytes =
        (static_cast<std::size_t>(rows - 1U) * leadingDimension + columns) *
        elementBytes;
    return span.byteSize == expectedBytes;
}

bool CheckNativeMatrixView(const BufferSpan &span,
                           NativeMatrixLayout layout,
                           std::uint32_t parentRows,
                           std::uint32_t parentColumns,
                           std::uint32_t rowOffset,
                           std::uint32_t columnOffset,
                           std::uint32_t rows,
                           std::uint32_t columns,
                           std::uint32_t elementBytes) noexcept
{
    return span.nativeLayout == layout && span.parentRows == parentRows &&
           span.parentColumns == parentColumns &&
           span.logicalRowOffset == rowOffset &&
           span.logicalColumnOffset == columnOffset && span.rows == rows &&
           span.columns == columns && span.leadingDimension == 0U &&
           span.elementBytes == elementBytes;
}

bool FindUniqueSyncOrder(const SyncTrace &trace, SyncAction action,
                         SyncPoint point, std::uint32_t ownerId,
                         std::uint64_t generation, Stage stage, Pipe pipe,
                         std::uint64_t &order) noexcept
{
    bool found = false;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const SyncRecord &record = trace.records[index];
        if (record.action != action || record.point != point ||
            record.ownerId != ownerId || record.generation != generation ||
            record.aivId != kNoAivId || !record.hasActiveHead ||
            record.stage != stage || record.pipe != pipe) {
            continue;
        }
        if (found) {
            return false;
        }
        found = true;
        order = record.order;
    }
    return found;
}

std::size_t CountLocalDependencies(const LocalSyncTrace &trace,
                                   LocalDependency dependency,
                                   Stage stage) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const LocalSyncRecord &record = trace.records[index];
        if (record.dependency == dependency && record.stage == stage) {
            ++count;
        }
    }
    return count;
}

bool FindUniqueLocalOrder(const LocalSyncTrace &trace,
                          LocalDependency dependency, Stage stage,
                          std::uint64_t &order) noexcept
{
    bool found = false;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const LocalSyncRecord &record = trace.records[index];
        if (record.dependency != dependency || record.stage != stage) {
            continue;
        }
        if (found) {
            return false;
        }
        found = true;
        order = record.order;
    }
    return found;
}

bool FindOperationOrderRange(const OperationTrace &trace,
                             OperationKind kind, Stage stage,
                             std::uint64_t &firstOrder,
                             std::uint64_t &lastOrder) noexcept
{
    bool found = false;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.kind != kind || record.stage != stage) {
            continue;
        }
        if (!found) {
            firstOrder = record.order;
            found = true;
        }
        lastOrder = record.order;
    }
    return found;
}

bool FindNthOperationOrder(const OperationTrace &trace,
                           OperationKind kind, Stage stage,
                           std::size_t ordinal,
                           std::uint64_t &order) noexcept
{
    std::size_t match = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.kind != kind || record.stage != stage) {
            continue;
        }
        if (match == ordinal) {
            order = record.order;
            return true;
        }
        ++match;
    }
    return false;
}

const OperationRecord *FindNthOperation(const OperationTrace &trace,
                                        OperationKind kind, Stage stage,
                                        std::size_t ordinal) noexcept
{
    std::size_t match = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.kind != kind || record.stage != stage) {
            continue;
        }
        if (match == ordinal) {
            return &record;
        }
        ++match;
    }
    return nullptr;
}

bool FindNthMmadOrder(const OperationTrace &trace, Stage stage,
                      std::size_t ordinal,
                      std::uint64_t &order) noexcept
{
    std::size_t match = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        const bool mmad = record.kind == OperationKind::Mmad ||
                           record.kind ==
                               OperationKind::MmadRowStackedLhs ||
                           record.kind ==
                               OperationKind::MmadQuadrantPackedLhs;
        if (!mmad || record.stage != stage) {
            continue;
        }
        if (match == ordinal) {
            order = record.order;
            return true;
        }
        ++match;
    }
    return false;
}

bool FindNthLocalOrder(const LocalSyncTrace &trace,
                       LocalDependency dependency, Stage stage,
                       std::size_t ordinal,
                       std::uint64_t &order) noexcept
{
    std::size_t match = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const LocalSyncRecord &record = trace.records[index];
        if (record.dependency != dependency || record.stage != stage) {
            continue;
        }
        if (match == ordinal) {
            order = record.order;
            return true;
        }
        ++match;
    }
    return false;
}

bool EmptyMutexTrace(const MutexTrace &trace) noexcept
{
    return !trace.overflow && trace.size == 0U;
}

bool ClaimMutexId(
    std::array<bool, kArch35StaticMutexIdLimit> &claimed,
    SymbolicMutexId mutexId) noexcept
{
    if (mutexId >= kArch35StaticMutexIdLimit || claimed[mutexId]) {
        return false;
    }
    claimed[mutexId] = true;
    return true;
}

bool CheckArch35StaticMutexIdTable() noexcept
{
    if (Arch35VectorMutexIds::kCount != 2U ||
        Arch35CubeMutexIds::kCount != 13U) {
        return false;
    }

    // AIV 和 AIC 位于不同的物理核，各自在独立 MutexID 命名空间内验重。
    std::array<bool, kArch35StaticMutexIdLimit> aivClaimed{};
    for (std::uint32_t localSlot = 0U; localSlot < kHeadsPerAiv;
         ++localSlot) {
        const SymbolicMutexId mutexId =
            Arch35VectorMutexIds::UbBank(localSlot);
        if (mutexId != localSlot || !ClaimMutexId(aivClaimed, mutexId)) {
            return false;
        }
    }

    std::array<bool, kArch35StaticMutexIdLimit> aicClaimed{};
    for (std::uint32_t bank = 0U; bank < kHeadsPerGroup; ++bank) {
        const SymbolicMutexId mutexId =
            Arch35CubeMutexIds::L1Bank(bank);
        if (mutexId != bank || !ClaimMutexId(aicClaimed, mutexId)) {
            return false;
        }
    }
    const SymbolicMutexId l0Operand =
        Arch35CubeMutexIds::L0OperandBank(0U);
    if (l0Operand != 4U || !ClaimMutexId(aicClaimed, l0Operand)) {
        return false;
    }
    for (std::uint32_t bank = 0U; bank < kHeadsPerGroup; ++bank) {
        const SymbolicMutexId mutexId =
            Arch35CubeMutexIds::L0cLowerHalf(bank);
        if (mutexId != 5U + bank || !ClaimMutexId(aicClaimed, mutexId)) {
            return false;
        }
    }
    for (std::uint32_t bank = 0U; bank < kHeadsPerGroup; ++bank) {
        const SymbolicMutexId mutexId =
            Arch35CubeMutexIds::L0cUpperHalf(bank);
        if (mutexId != 9U + bank || !ClaimMutexId(aicClaimed, mutexId)) {
            return false;
        }
    }
    return true;
}

std::size_t CountMutexRecords(const MutexTrace &trace,
                              MutexAction action,
                              MutexResource resource,
                              SymbolicMutexId mutexId,
                              std::uint32_t ownerId, Stage stage,
                              Pipe pipe) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const MutexRecord &record = trace.records[index];
        if (record.action == action && record.resource == resource &&
            record.mutexId == mutexId && record.ownerId == ownerId &&
            record.stage == stage && record.pipe == pipe) {
            ++count;
        }
    }
    return count;
}

std::size_t CountMutexStageOwner(const MutexTrace &trace, Stage stage,
                                 std::uint32_t ownerId) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const MutexRecord &record = trace.records[index];
        if (record.stage == stage && record.ownerId == ownerId) {
            ++count;
        }
    }
    return count;
}

bool FindNthMutexOrder(const MutexTrace &trace, MutexAction action,
                       MutexResource resource, SymbolicMutexId mutexId,
                       std::uint32_t ownerId, Stage stage, Pipe pipe,
                       std::size_t ordinal,
                       std::uint64_t &order) noexcept
{
    std::size_t match = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const MutexRecord &record = trace.records[index];
        if (record.action != action || record.resource != resource ||
            record.mutexId != mutexId || record.ownerId != ownerId ||
            record.stage != stage || record.pipe != pipe) {
            continue;
        }
        if (match == ordinal) {
            order = record.order;
            return true;
        }
        ++match;
    }
    return false;
}

bool FindUniqueMutexOrder(const MutexTrace &trace, MutexAction action,
                          MutexResource resource,
                          SymbolicMutexId mutexId,
                          std::uint32_t ownerId, Stage stage, Pipe pipe,
                          std::uint64_t &order) noexcept
{
    return CountMutexRecords(trace, action, resource, mutexId, ownerId,
                             stage, pipe) == 1U &&
           FindNthMutexOrder(trace, action, resource, mutexId, ownerId,
                             stage, pipe, 0U, order);
}

bool IsOrderInsideMutexInterval(
    const MutexTrace &trace, MutexResource resource,
    SymbolicMutexId mutexId, std::uint32_t ownerId, Stage stage, Pipe pipe,
    std::uint64_t operationOrder) noexcept
{
    bool held = false;
    bool found = false;
    std::uint64_t lockOrder = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const MutexRecord &record = trace.records[index];
        if (record.resource != resource || record.mutexId != mutexId ||
            record.ownerId != ownerId || record.stage != stage ||
            record.pipe != pipe ||
            record.action == MutexAction::PipeBarrier) {
            continue;
        }
        if (record.action == MutexAction::Lock) {
            if (held) {
                return false;
            }
            held = true;
            lockOrder = record.order;
            continue;
        }
        if (!held) {
            return false;
        }
        if (lockOrder < operationOrder && operationOrder < record.order) {
            if (found) {
                return false;
            }
            found = true;
        }
        held = false;
    }
    return !held && found;
}

bool FindUniqueOwnerSyncOrderOnPipe(
    const SyncTrace &trace, SyncAction action, SyncPoint point,
    std::uint32_t ownerId, std::uint64_t generation, Stage stage, Pipe pipe,
    std::uint64_t &order) noexcept
{
    bool found = false;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const SyncRecord &record = trace.records[index];
        if (record.action != action || record.point != point ||
            record.ownerId != ownerId ||
            record.generation != generation ||
            record.aivId != kNoAivId || !record.hasActiveHead ||
            record.stage != stage) {
            continue;
        }
        if (found || record.pipe != pipe) {
            return false;
        }
        found = true;
        order = record.order;
    }
    return found;
}

bool HasStageOperationBetween(const OperationTrace &trace, Stage stage,
                              std::uint64_t begin,
                              std::uint64_t end) noexcept
{
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const OperationRecord &record = trace.records[index];
        if (record.stage == stage && record.order > begin &&
            record.order < end) {
            return true;
        }
    }
    return false;
}

bool IsAllowedVectorMutexPipe(Pipe pipe) noexcept
{
    return pipe == Pipe::Mte2 || pipe == Pipe::Vector ||
           pipe == Pipe::Mte3;
}

bool CheckArch35MutexRecordMapping(const MutexRecord &record,
                                   bool vectorTrace,
                                   std::uint32_t aivId) noexcept
{
    if (record.action == MutexAction::PipeBarrier) {
        return !vectorTrace &&
               record.resource == MutexResource::Mte2Overwrite &&
               record.mutexId == kInvalidMutexId &&
               record.ownerId < kHeadsPerGroup &&
               record.stage == Stage::C4 && record.pipe == Pipe::Mte2;
    }
    if (record.mutexId >= kArch35StaticMutexIdLimit ||
        record.resource == MutexResource::Mte2Overwrite) {
        return false;
    }
    if (vectorTrace) {
        return record.resource == MutexResource::AivUbBank &&
               record.ownerId < kHeadsPerGroup &&
               record.ownerId / kHeadsPerAiv == aivId &&
               record.mutexId == Arch35VectorMutexIds::UbBank(
                                     record.ownerId % kHeadsPerAiv) &&
               IsAllowedVectorMutexPipe(record.pipe);
    }

    switch (record.resource) {
        case MutexResource::AicL1Bank:
            return record.ownerId < kHeadsPerGroup &&
                   record.mutexId ==
                       Arch35CubeMutexIds::L1Bank(record.ownerId) &&
                   (record.pipe == Pipe::Mte2 ||
                    record.pipe == Pipe::Mte1 ||
                    record.pipe == Pipe::Fixpipe);
        case MutexResource::AicL0OperandBank:
            return record.ownerId == 0U &&
                   record.mutexId ==
                       Arch35CubeMutexIds::L0OperandBank(0U) &&
                   (record.pipe == Pipe::Mte1 ||
                    record.pipe == Pipe::Cube);
        case MutexResource::AicL0cLowerHalf:
            return record.ownerId < kHeadsPerGroup &&
                   record.mutexId ==
                       Arch35CubeMutexIds::L0cLowerHalf(record.ownerId) &&
                   (record.pipe == Pipe::Cube ||
                    record.pipe == Pipe::Fixpipe);
        case MutexResource::AicL0cUpperHalf:
            return record.ownerId < kHeadsPerGroup &&
                   record.mutexId ==
                       Arch35CubeMutexIds::L0cUpperHalf(record.ownerId) &&
                   (record.pipe == Pipe::Cube ||
                    record.pipe == Pipe::Fixpipe);
        default:
            return false;
    }
}

bool CheckArch35MutexTrace(const MutexTrace &trace, bool vectorTrace,
                           std::uint32_t aivId = 0U) noexcept
{
    if (trace.overflow) {
        return false;
    }
    std::array<bool, kArch35StaticMutexIdLimit> held{};
    std::array<MutexResource, kArch35StaticMutexIdLimit> heldResource{};
    std::array<std::uint32_t, kArch35StaticMutexIdLimit> heldOwner{};
    std::array<Pipe, kArch35StaticMutexIdLimit> heldPipe{};
    std::uint64_t previousOrder = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const MutexRecord &record = trace.records[index];
        if ((index != 0U && record.order <= previousOrder) ||
            !CheckArch35MutexRecordMapping(record, vectorTrace, aivId)) {
            return false;
        }
        previousOrder = record.order;
        if (record.action == MutexAction::PipeBarrier) {
            continue;
        }
        const std::size_t mutexId = record.mutexId;
        if (record.action == MutexAction::Lock) {
            if (held[mutexId]) {
                return false;
            }
            held[mutexId] = true;
            heldResource[mutexId] = record.resource;
            heldOwner[mutexId] = record.ownerId;
            heldPipe[mutexId] = record.pipe;
            continue;
        }
        if (!held[mutexId] || heldResource[mutexId] != record.resource ||
            heldOwner[mutexId] != record.ownerId ||
            heldPipe[mutexId] != record.pipe) {
            return false;
        }
        held[mutexId] = false;
    }
    for (bool isHeld : held) {
        if (isHeld) {
            return false;
        }
    }
    return true;
}

bool CheckArch35RetiredLocalSyncPoints(const SyncTrace &trace) noexcept
{
    for (std::size_t index = 0U; index < trace.size; ++index) {
        switch (trace.records[index].point) {
            case SyncPoint::LocalBankFree:
            case SyncPoint::V0ExportDone:
            case SyncPoint::V0BetaReady:
            case SyncPoint::V0ContextReady:
            case SyncPoint::V3LocalSourceFree:
            case SyncPoint::L1BankFree:
            case SyncPoint::L0cBankFree:
            case SyncPoint::C2ScoreL1Free:
            case SyncPoint::C4TReady:
            case SyncPoint::C4AkkPrepReady:
            case SyncPoint::C5AkkReady:
                return false;
            default:
                break;
        }
    }
    return true;
}

bool IsWaitAction(SyncAction action) noexcept
{
    return action == SyncAction::Wait || action == SyncAction::AicWait ||
           action == SyncAction::AivWait;
}

std::size_t CountStageWaitActions(const SyncTrace &trace,
                                  Stage stage) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        if (trace.records[index].stage == stage &&
            IsWaitAction(trace.records[index].action)) {
            ++count;
        }
    }
    return count;
}

bool CheckStageWaitsBeforeOrder(const SyncTrace &trace, Stage stage,
                                std::size_t expectedCount,
                                std::uint64_t beforeOrder) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const SyncRecord &record = trace.records[index];
        if (record.stage != stage || !IsWaitAction(record.action)) {
            continue;
        }
        if (record.order >= beforeOrder) {
            return false;
        }
        ++count;
    }
    return count == expectedCount;
}

bool FindUniqueActivePairSyncOrder(const SyncTrace &trace,
                                   SyncPoint point, Stage stage,
                                   Pipe pipe,
                                   std::uint64_t &order) noexcept
{
    bool found = false;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const SyncRecord &record = trace.records[index];
        if (record.action != SyncAction::AivArrive ||
            record.point != point || record.ownerId != 0U ||
            record.generation != 0U || record.aivId != 0U ||
            !record.hasActiveHead || record.stage != stage ||
            record.pipe != pipe) {
            continue;
        }
        if (found) {
            return false;
        }
        found = true;
        order = record.order;
    }
    return found;
}

bool CheckVectorStagePipeline(const OperationTrace &operations,
                              const SyncTrace &synchronization,
                              const LocalSyncTrace &localDependencies,
                              Stage stage, bool hasInputLoads,
                              SyncPoint finalPoint,
                              bool pairFinalPoint) noexcept
{
    const OperationRecord *runVf = FindUniqueOperation(
        operations, OperationKind::RunVf, stage);
    std::uint64_t firstStore = 0U;
    std::uint64_t lastStore = 0U;
    std::uint64_t outputReady = 0U;
    std::uint64_t finalReady = 0U;
    if (runVf == nullptr ||
        !FindOperationOrderRange(operations, OperationKind::Store, stage,
                                 firstStore, lastStore) ||
        CountLocalDependencies(
            localDependencies, LocalDependency::VectorToMte3Outputs,
            stage) != 1U ||
        !FindUniqueLocalOrder(
            localDependencies, LocalDependency::VectorToMte3Outputs,
            stage, outputReady)) {
        return false;
    }
    const bool foundFinal = pairFinalPoint
        ? FindUniqueActivePairSyncOrder(synchronization, finalPoint, stage,
                                        Pipe::Mte3, finalReady)
        : FindUniqueSyncOrder(synchronization, SyncAction::Set, finalPoint,
                              0U, 0U, stage, Pipe::Mte3, finalReady);
    if (!foundFinal || runVf->order >= outputReady ||
        outputReady >= firstStore || lastStore >= finalReady) {
        return false;
    }

    std::uint64_t firstLoad = 0U;
    std::uint64_t lastLoad = 0U;
    const bool foundLoads = FindOperationOrderRange(
        operations, OperationKind::Load, stage, firstLoad, lastLoad);
    const std::size_t inputEdgeCount = CountLocalDependencies(
        localDependencies, LocalDependency::Mte2ToVectorInputs, stage);
    if (!hasInputLoads) {
        return !foundLoads && inputEdgeCount == 0U;
    }
    std::uint64_t inputReady = 0U;
    return foundLoads && inputEdgeCount == 1U &&
           FindUniqueLocalOrder(localDependencies,
                                LocalDependency::Mte2ToVectorInputs,
                                stage, inputReady) &&
           firstLoad <= lastLoad && lastLoad < inputReady &&
           inputReady < runVf->order;
}

bool CheckArch35VectorStageMutex(
    const OperationTrace &operations, const SyncTrace &synchronization,
    const LocalSyncTrace &localDependencies, const MutexTrace &mutexes,
    Stage stage, bool hasInputLoads, std::size_t expectedWaits,
    SyncPoint requiredReady) noexcept
{
    constexpr std::uint32_t kOwner = 0U;
    const SymbolicMutexId mutexId = Arch35VectorMutexIds::UbBank(0U);
    const OperationRecord *runVf = FindUniqueOperation(
        operations, OperationKind::RunVf, stage);
    std::uint64_t vectorLock = 0U;
    std::uint64_t vectorUnlock = 0U;
    std::uint64_t mte3Lock = 0U;
    std::uint64_t mte3Unlock = 0U;
    std::uint64_t outputReady = 0U;
    std::uint64_t firstStore = 0U;
    std::uint64_t lastStore = 0U;
    std::uint64_t requiredReadyOrder = 0U;
    if (runVf == nullptr ||
        CountMutexStageOwner(mutexes, stage, kOwner) !=
            (hasInputLoads ? 6U : 4U) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AivUbBank,
            mutexId, kOwner, stage, Pipe::Vector, vectorLock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock, MutexResource::AivUbBank,
            mutexId, kOwner, stage, Pipe::Vector, vectorUnlock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AivUbBank,
            mutexId, kOwner, stage, Pipe::Mte3, mte3Lock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock, MutexResource::AivUbBank,
            mutexId, kOwner, stage, Pipe::Mte3, mte3Unlock) ||
        CountLocalDependencies(
            localDependencies, LocalDependency::VectorToMte3Outputs,
            stage) != 1U ||
        !FindUniqueLocalOrder(
            localDependencies, LocalDependency::VectorToMte3Outputs,
            stage, outputReady) ||
        !FindOperationOrderRange(operations, OperationKind::Store, stage,
                                 firstStore, lastStore) ||
        !FindUniqueSyncOrder(synchronization, SyncAction::Set,
                             requiredReady, kOwner, 0U, stage, Pipe::Mte3,
                             requiredReadyOrder) ||
        !(vectorLock < runVf->order &&
          runVf->order < vectorUnlock && vectorUnlock < outputReady &&
          outputReady < mte3Lock && mte3Lock < firstStore &&
          firstStore <= lastStore && lastStore < mte3Unlock &&
          mte3Unlock < requiredReadyOrder)) {
        return false;
    }

    // 同阶段的跨核发布都必须晚于该 head 的最后一次 MTE3 Unlock。
    for (std::size_t index = 0U; index < synchronization.size; ++index) {
        const SyncRecord &record = synchronization.records[index];
        if (record.action == SyncAction::Set && record.stage == stage &&
            record.order <= mte3Unlock) {
            return false;
        }
    }

    std::uint64_t firstLoad = 0U;
    std::uint64_t lastLoad = 0U;
    const bool foundLoads = FindOperationOrderRange(
        operations, OperationKind::Load, stage, firstLoad, lastLoad);
    const std::size_t inputEdges = CountLocalDependencies(
        localDependencies, LocalDependency::Mte2ToVectorInputs, stage);
    if (!hasInputLoads) {
        return !foundLoads && inputEdges == 0U &&
               CheckStageWaitsBeforeOrder(synchronization, stage,
                                          expectedWaits, vectorLock);
    }

    std::uint64_t mte2Lock = 0U;
    std::uint64_t mte2Unlock = 0U;
    std::uint64_t inputReady = 0U;
    return foundLoads && inputEdges == 1U &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock, MutexResource::AivUbBank,
               mutexId, kOwner, stage, Pipe::Mte2, mte2Lock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock, MutexResource::AivUbBank,
               mutexId, kOwner, stage, Pipe::Mte2, mte2Unlock) &&
           FindUniqueLocalOrder(localDependencies,
                                LocalDependency::Mte2ToVectorInputs,
                                stage, inputReady) &&
           mte2Lock < firstLoad && firstLoad <= lastLoad &&
           lastLoad < mte2Unlock && mte2Unlock < inputReady &&
           inputReady < vectorLock &&
           CheckStageWaitsBeforeOrder(synchronization, stage,
                                      expectedWaits, mte2Lock);
}

bool CheckArch35VectorMutexPipelines(
    const OperationTrace &operations, const SyncTrace &synchronization,
    const LocalSyncTrace &localDependencies,
    const MutexTrace &mutexes) noexcept
{
    return CheckArch35MutexTrace(mutexes, true, 0U) &&
           CheckArch35RetiredLocalSyncPoints(synchronization) &&
           CheckArch35VectorStageMutex(
               operations, synchronization, localDependencies, mutexes,
               Stage::V0, true, 2U, SyncPoint::QkCacheReady) &&
           CheckArch35VectorStageMutex(
               operations, synchronization, localDependencies, mutexes,
               Stage::V1, false, 0U, SyncPoint::V1ScoreReady) &&
           CheckArch35VectorStageMutex(
               operations, synchronization, localDependencies, mutexes,
               Stage::V3, false, 2U, SyncPoint::V3VcsReady) &&
           CheckArch35VectorStageMutex(
               operations, synchronization, localDependencies, mutexes,
               Stage::V6, true, 1U, SyncPoint::V6RhsReady);
}

bool MatchOwner(TraceCursor &cursor, SyncAction action, SyncPoint point,
                std::uint32_t ownerId, std::uint64_t generation, Stage stage,
                Pipe pipe) noexcept
{
    return cursor.Match(action, point, ownerId, generation, kNoAivId, true,
                        stage, pipe);
}

bool MatchAivPair(TraceCursor &cursor, SyncAction action, SyncPoint point,
                  std::uint32_t pair, std::uint64_t generation,
                  std::uint32_t aivId, bool hasActiveHead, Stage stage,
                  Pipe pipe) noexcept
{
    return cursor.Match(action, point, pair, generation, aivId,
                        hasActiveHead, stage, pipe);
}

bool MatchAicPair(TraceCursor &cursor, SyncAction action, SyncPoint point,
                  std::uint32_t pair, std::uint64_t generation, Stage stage,
                  Pipe pipe) noexcept
{
    return cursor.Match(action, point, pair, generation, kNoAivId, true,
                        stage, pipe);
}

bool CheckAivItem(TraceCursor &cursor, const WorkItem &item,
                  std::uint32_t aivId) noexcept
{
    const std::uint32_t pairWaves = static_cast<std::uint32_t>(
        CeilDiv(item.group.activeHeads, kAivPerWorkgroup));

    for (std::uint32_t pair = 0U; pair < pairWaves; ++pair) {
        const std::uint64_t pairGeneration =
            PairCollectiveGenerationFor(item.group, pair);
        const std::uint32_t local = pair * kAivPerWorkgroup + aivId;
        const bool active = local < item.group.activeHeads;
        if (!MatchAivPair(cursor, SyncAction::AivWait,
                          SyncPoint::SlotFree, pair, pairGeneration, aivId,
                          true, Stage::V0, Pipe::Control)) {
            return false;
        }
        if (active) {
            const HeadTask &head = item.group.heads[local];
            const std::uint64_t sharedGeneration =
                SharedGenerationFor(head, SharedArenaUse::V01);
            if (!MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::LocalBankFree, head.localBankId,
                            head.localGeneration, Stage::V0, Pipe::Mte2) ||
                !MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::SharedArenaFree, head.sharedArenaId,
                            sharedGeneration, Stage::V0, Pipe::Mte2)) {
                return false;
            }
            const SyncPoint qkPoint = head.qkOwner
                                          ? SyncPoint::QkCacheFree
                                          : SyncPoint::QkCacheReady;
            if (!MatchOwner(cursor, SyncAction::Wait, qkPoint,
                            head.qkCacheSlot, head.qkCacheGeneration,
                            Stage::V0, Pipe::Mte2) ||
                (head.qkOwner &&
                 !MatchOwner(cursor, SyncAction::Set,
                             SyncPoint::QkCacheReady, head.qkCacheSlot,
                             head.qkCacheGeneration, Stage::V0,
                             Pipe::Mte3)) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::V0ContextReady, head.workspaceSlot,
                            head.workspaceGeneration, Stage::V0, Pipe::Mte3) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::V0BetaReady, head.workspaceSlot,
                            head.workspaceGeneration, Stage::V0, Pipe::Mte3) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::V0ExportDone, head.localBankId,
                            head.localGeneration, Stage::V0, Pipe::Mte3) ||
                !MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::V0ExportDone, head.localBankId,
                            head.localGeneration, Stage::V1, Pipe::Vector) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::SharedArenaFree, head.sharedArenaId,
                            sharedGeneration + 1U, Stage::V1,
                            Pipe::Vector)) {
                return false;
            }
        }
        if (!MatchAivPair(cursor, SyncAction::AivArrive,
                          SyncPoint::V1ScoreReady, pair, pairGeneration,
                          aivId, active, Stage::V1, Pipe::Mte3)) {
            return false;
        }
    }

    for (std::uint32_t pair = 0U; pair < pairWaves; ++pair) {
        const std::uint64_t pairGeneration =
            PairCollectiveGenerationFor(item.group, pair);
        const std::uint32_t local = pair * kAivPerWorkgroup + aivId;
        const bool active = local < item.group.activeHeads;
        if (!MatchAivPair(cursor, SyncAction::AivWait,
                          SyncPoint::C2RawReady, pair, pairGeneration, aivId,
                          true, Stage::V3, Pipe::Mte2)) {
            return false;
        }
        if (active) {
            const HeadTask &head = item.group.heads[local];
            if (!MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::V0BetaReady, head.workspaceSlot,
                            head.workspaceGeneration, Stage::V3, Pipe::Mte2) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::V3LocalSourceFree, head.localBankId,
                            head.localGeneration, Stage::V3, Pipe::Mte3)) {
                return false;
            }
        }
        if (!MatchAivPair(cursor, SyncAction::AivArrive,
                          SyncPoint::V3VcsReady, pair, pairGeneration, aivId,
                          active, Stage::V3, Pipe::Mte3)) {
            return false;
        }
    }

    for (std::uint32_t pair = 0U; pair < pairWaves; ++pair) {
        const std::uint64_t pairGeneration =
            PairCollectiveGenerationFor(item.group, pair);
        const std::uint32_t local = pair * kAivPerWorkgroup + aivId;
        const bool active = local < item.group.activeHeads;
        if (!MatchAivPair(cursor, SyncAction::AivWait,
                          SyncPoint::C4PayloadFree, pair, pairGeneration,
                          aivId, true, Stage::V6, Pipe::Mte3)) {
            return false;
        }
        if (active) {
            const HeadTask &head = item.group.heads[local];
            const std::uint64_t sharedGeneration =
                SharedGenerationFor(head, SharedArenaUse::V6);
            if (!MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::V0ContextReady, head.workspaceSlot,
                            head.workspaceGeneration, Stage::V6, Pipe::Mte2) ||
                !MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::V3LocalSourceFree, head.localBankId,
                            head.localGeneration, Stage::V6, Pipe::Mte2) ||
                !MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::SharedArenaFree, head.sharedArenaId,
                            sharedGeneration, Stage::V6, Pipe::Mte2)) {
                return false;
            }
            if (!MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::SharedArenaFree, head.sharedArenaId,
                            sharedGeneration + 1U, Stage::V6, Pipe::Vector) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::LocalBankFree, head.localBankId,
                            head.localGeneration + 1U, Stage::V6,
                            Pipe::Mte3)) {
                return false;
            }
        }
        if (!MatchAivPair(cursor, SyncAction::AivArrive,
                          SyncPoint::V6RhsReady, pair, pairGeneration, aivId,
                          active, Stage::V6, Pipe::Mte3)) {
            return false;
        }
    }
    return true;
}

bool CheckAicItem(TraceCursor &cursor, const WorkItem &item) noexcept
{
    const bool hasQ10 = item.group.chunk.validRows > 32U;
    const Pipe c4LoadPipe = hasQ10 ? Pipe::Mte2 : Pipe::Control;
    const Pipe c4CubePipe = hasQ10 ? Pipe::Cube : Pipe::Control;
    const Pipe c4PublishPipe = hasQ10 ? Pipe::Mte2 : Pipe::Control;

    for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
        if (!PairHasActiveHead(item.group, pair)) {
            continue;
        }
        const std::uint64_t pairGeneration =
            PairCollectiveGenerationFor(item.group, pair);
        if (!MatchAicPair(cursor, SyncAction::AicWait,
                          SyncPoint::V1ScoreReady, pair, pairGeneration,
                          Stage::C2, Pipe::Mte2)) {
            return false;
        }
        for (std::uint32_t lane = 0U; lane < kAivPerWorkgroup; ++lane) {
            const std::uint32_t local = pair * kAivPerWorkgroup + lane;
            if (local >= item.group.activeHeads) {
                continue;
            }
            const HeadTask &head = item.group.heads[local];
            const std::uint64_t l0cGeneration = L0cGenerationFor(
                head, L0cStageUse::C2, Architecture::Arch22);
            if (!MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::L1BankFree, head.l1BankId,
                            head.l1Generation, Stage::C2, Pipe::Mte2) ||
                !MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration, Stage::C2, Pipe::Cube) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::C2ScoreL1Free, head.l1BankId,
                            head.l1Generation, Stage::C2, Pipe::Mte1) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration + 1U, Stage::C2,
                            Pipe::Fixpipe)) {
                return false;
            }
        }
        if (!MatchAicPair(cursor, SyncAction::AicPublish,
                          SyncPoint::C2RawReady, pair, pairGeneration,
                          Stage::C2, Pipe::Fixpipe)) {
            return false;
        }
    }

    for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
        if (!PairHasActiveHead(item.group, pair)) {
            continue;
        }
        const std::uint64_t pairGeneration =
            PairCollectiveGenerationFor(item.group, pair);
        if (!MatchAicPair(cursor, SyncAction::AicWait,
                          SyncPoint::V3VcsReady, pair, pairGeneration,
                          Stage::C4, c4LoadPipe)) {
            return false;
        }
        for (std::uint32_t lane = 0U; lane < kAivPerWorkgroup; ++lane) {
            const std::uint32_t local = pair * kAivPerWorkgroup + lane;
            if (local >= item.group.activeHeads) {
                continue;
            }
            const HeadTask &head = item.group.heads[local];
            const std::uint64_t l0cGeneration = L0cGenerationFor(
                head, L0cStageUse::C4, Architecture::Arch22);
            if (!MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::C2ScoreL1Free, head.l1BankId,
                            head.l1Generation, Stage::C4, c4LoadPipe) ||
                !MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration, Stage::C4, c4CubePipe) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::C4AkkPrepReady, head.l1BankId,
                            head.l1Generation, Stage::C4, c4LoadPipe)) {
                return false;
            }
            if (hasQ10 &&
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::C4TReady, head.workspaceSlot,
                            head.workspaceGeneration, Stage::C4,
                            Pipe::Fixpipe)) {
                return false;
            }
            if (!MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration + 1U, Stage::C4,
                            hasQ10 ? Pipe::Fixpipe : Pipe::Control)) {
                return false;
            }
        }
        if (!MatchAicPair(cursor, SyncAction::AicPublish,
                          SyncPoint::C4PayloadFree, pair, pairGeneration,
                          Stage::C4, c4PublishPipe)) {
            return false;
        }
    }

    for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
        if (!PairHasActiveHead(item.group, pair)) {
            continue;
        }
        for (std::uint32_t lane = 0U; lane < kAivPerWorkgroup; ++lane) {
            const std::uint32_t local = pair * kAivPerWorkgroup + lane;
            if (local >= item.group.activeHeads) {
                continue;
            }
            const HeadTask &head = item.group.heads[local];
            const std::uint64_t l0cGeneration = L0cGenerationFor(
                head, L0cStageUse::C5, Architecture::Arch22);
            if (hasQ10) {
                if (!MatchOwner(cursor, SyncAction::Wait,
                                SyncPoint::C4TReady, head.workspaceSlot,
                                head.workspaceGeneration, Stage::C5,
                                Pipe::Mte2) ||
                    !MatchOwner(cursor, SyncAction::Wait,
                                SyncPoint::C4AkkPrepReady, head.l1BankId,
                                head.l1Generation, Stage::C5, Pipe::Mte1) ||
                    !MatchOwner(cursor, SyncAction::Wait,
                                SyncPoint::L0cBankFree, head.l0cBankId,
                                l0cGeneration, Stage::C5, Pipe::Cube) ||
                    !MatchOwner(cursor, SyncAction::Set,
                                SyncPoint::C5AkkReady, head.workspaceSlot,
                                head.workspaceGeneration, Stage::C5,
                                Pipe::Fixpipe) ||
                    !MatchOwner(cursor, SyncAction::Set,
                                SyncPoint::L0cBankFree, head.l0cBankId,
                                l0cGeneration + 1U, Stage::C5,
                                Pipe::Fixpipe)) {
                    return false;
                }
            } else if (!MatchOwner(cursor, SyncAction::Wait,
                                   SyncPoint::C4AkkPrepReady, head.l1BankId,
                                   head.l1Generation, Stage::C5,
                                   Pipe::Control) ||
                       !MatchOwner(cursor, SyncAction::Wait,
                                   SyncPoint::L0cBankFree, head.l0cBankId,
                                   l0cGeneration, Stage::C5,
                                   Pipe::Control) ||
                       !MatchOwner(cursor, SyncAction::Set,
                                   SyncPoint::C5AkkReady,
                                   head.workspaceSlot,
                                   head.workspaceGeneration, Stage::C5,
                                   Pipe::Control) ||
                       !MatchOwner(cursor, SyncAction::Set,
                                   SyncPoint::L0cBankFree, head.l0cBankId,
                                   l0cGeneration + 1U, Stage::C5,
                                   Pipe::Control)) {
                return false;
            }
        }
    }

    for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
        if (!PairHasActiveHead(item.group, pair)) {
            continue;
        }
        const std::uint64_t pairGeneration =
            PairCollectiveGenerationFor(item.group, pair);
        if (!MatchAicPair(cursor, SyncAction::AicWait,
                          SyncPoint::V6RhsReady, pair, pairGeneration,
                          Stage::C7, Pipe::Mte2)) {
            return false;
        }
        for (const HeadTask &head : item.group.heads) {
            if (head.active && head.qkLastConsumer &&
                arch22_policy::PairWave(head.groupLocalHead) == pair &&
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::QkCacheFree, head.qkCacheSlot,
                            head.qkCacheGeneration + 1U, Stage::C7,
                            Pipe::Control)) {
                return false;
            }
        }
        for (std::uint32_t lane = 0U; lane < kAivPerWorkgroup; ++lane) {
            const std::uint32_t local = pair * kAivPerWorkgroup + lane;
            if (local >= item.group.activeHeads) {
                continue;
            }
            const HeadTask &head = item.group.heads[local];
            const std::uint64_t l0cGeneration = L0cGenerationFor(
                head, L0cStageUse::C7, Architecture::Arch22);
            if (!MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::C5AkkReady, head.workspaceSlot,
                            head.workspaceGeneration, Stage::C7, Pipe::Mte2) ||
                !MatchOwner(cursor, SyncAction::Wait,
                            SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration, Stage::C7, Pipe::Cube) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::L1BankFree, head.l1BankId,
                            head.l1Generation + 1U, Stage::C7, Pipe::Mte1) ||
                !MatchOwner(cursor, SyncAction::Set,
                            SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration + 1U, Stage::C7,
                            Pipe::Fixpipe)) {
                return false;
            }
        }
        if (!MatchAicPair(cursor, SyncAction::AicPublish,
                          SyncPoint::SlotFree, pair, pairGeneration + 1U,
                          Stage::C7, Pipe::Fixpipe)) {
            return false;
        }
    }
    return true;
}

bool CheckAivLocalItem(LocalTraceCursor &cursor, const WorkItem &item,
                       std::uint32_t aivId) noexcept
{
    const std::uint32_t pairWaves = static_cast<std::uint32_t>(
        CeilDiv(item.group.activeHeads, kAivPerWorkgroup));
    for (std::uint32_t pair = 0U; pair < pairWaves; ++pair) {
        const std::uint32_t local = pair * kAivPerWorkgroup + aivId;
        if (local >= item.group.activeHeads) {
            continue;
        }
        if (!cursor.Match(LocalDependency::Mte2ToVectorInputs, Stage::V0) ||
            !cursor.Match(LocalDependency::VectorToMte3Outputs, Stage::V0) ||
            !cursor.Match(LocalDependency::VectorToMte3Outputs, Stage::V1)) {
            return false;
        }
    }
    for (std::uint32_t pair = 0U; pair < pairWaves; ++pair) {
        const std::uint32_t local = pair * kAivPerWorkgroup + aivId;
        if (local >= item.group.activeHeads) {
            continue;
        }
        if (!cursor.Match(LocalDependency::Mte2ToVectorInputs, Stage::V3) ||
            !cursor.Match(LocalDependency::Mte2ToMte3SourceFree,
                          Stage::V3) ||
            !cursor.Match(LocalDependency::VectorToMte3Outputs, Stage::V3)) {
            return false;
        }
    }
    for (std::uint32_t pair = 0U; pair < pairWaves; ++pair) {
        const std::uint32_t local = pair * kAivPerWorkgroup + aivId;
        if (local >= item.group.activeHeads) {
            continue;
        }
        if (!cursor.Match(LocalDependency::Mte2ToVectorInputs, Stage::V6) ||
            !cursor.Match(LocalDependency::VectorToMte3Outputs, Stage::V6)) {
            return false;
        }
    }
    return true;
}

bool CheckAicLocalItem(LocalTraceCursor &cursor,
                       const WorkItem &item) noexcept
{
    const std::uint32_t activeBlocks = static_cast<std::uint32_t>(
        CeilDiv(item.group.chunk.validRows, ShapePolicy::kScoreBlockRows));
    const bool hasQ10 = item.group.chunk.validRows > 32U;

    for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
        if (!PairHasActiveHead(item.group, pair)) {
            continue;
        }
        for (std::uint32_t lane = 0U; lane < kAivPerWorkgroup; ++lane) {
            const std::uint32_t local = pair * kAivPerWorkgroup + lane;
            if (local >= item.group.activeHeads) {
                continue;
            }
            if (!cursor.Match(
                    LocalDependency::Mte2ToFixpipePayloadReuse, Stage::C2) ||
                !cursor.Match(LocalDependency::Mte2ToMte1Inputs,
                              Stage::C2)) {
                return false;
            }
            for (std::uint32_t block = 0U; block < activeBlocks; ++block) {
                (void)block;
                if (!cursor.Match(LocalDependency::CubeToMte1OperandReuse,
                                  Stage::C2) ||
                    !cursor.Match(LocalDependency::CubeToFixpipeOutput,
                                  Stage::C2)) {
                    return false;
                }
            }
        }
    }

    if (hasQ10) {
        for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
            if (!PairHasActiveHead(item.group, pair)) {
                continue;
            }
            for (std::uint32_t lane = 0U; lane < kAivPerWorkgroup; ++lane) {
                const std::uint32_t local = pair * kAivPerWorkgroup + lane;
                if (local < item.group.activeHeads &&
                    (!cursor.Match(LocalDependency::Mte2ToMte1Inputs,
                                   Stage::C4) ||
                     !cursor.Match(LocalDependency::CubeToMte1OperandReuse,
                                   Stage::C4) ||
                     !cursor.Match(LocalDependency::CubeToFixpipeOutput,
                                   Stage::C4))) {
                    return false;
                }
            }
        }
        for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
            if (!PairHasActiveHead(item.group, pair)) {
                continue;
            }
            for (std::uint32_t lane = 0U; lane < kAivPerWorkgroup; ++lane) {
                const std::uint32_t local = pair * kAivPerWorkgroup + lane;
                if (local < item.group.activeHeads &&
                    (!cursor.Match(LocalDependency::FixpipeToMte2Relay,
                                   Stage::C5) ||
                     !cursor.Match(LocalDependency::Mte2ToMte1Inputs,
                                   Stage::C5) ||
                     !cursor.Match(LocalDependency::CubeToMte1OperandReuse,
                                   Stage::C5) ||
                     !cursor.Match(LocalDependency::CubeToFixpipeOutput,
                                   Stage::C5))) {
                    return false;
                }
            }
        }
    }

    for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
        if (!PairHasActiveHead(item.group, pair)) {
            continue;
        }
        for (std::uint32_t lane = 0U; lane < kAivPerWorkgroup; ++lane) {
            const std::uint32_t local = pair * kAivPerWorkgroup + lane;
            if (local >= item.group.activeHeads) {
                continue;
            }
            if (!cursor.Match(LocalDependency::Mte2FillToLoadWaw,
                              Stage::C7)) {
                return false;
            }
            if (hasQ10 &&
                !cursor.Match(LocalDependency::FixpipeToMte2Relay,
                              Stage::C7)) {
                return false;
            }
            if (!cursor.Match(LocalDependency::Mte2ToMte1Inputs,
                              Stage::C7) ||
                !cursor.Match(LocalDependency::CubeToMte1OperandReuse,
                              Stage::C7) ||
                !cursor.Match(LocalDependency::CubeToFixpipeOutput,
                              Stage::C7)) {
                return false;
            }
        }
    }
    return true;
}

template <typename CheckItem>
bool ReplayItems(Architecture architecture, const RuntimeTiling &tiling,
                 std::uint32_t workgroupId, ResolveChunk resolveChunk,
                 CheckItem checkItem) noexcept
{
    const CorePlan plan = BuildCorePlan(tiling, workgroupId, architecture);
    OwnerTicketState tickets{};
    const auto replayPartition =
        [&plan, &tiling, &tickets, &checkItem](
            std::uint64_t ordinal, const ChunkTask &chunk,
            std::uint32_t partitionOrdinal) noexcept {
            const std::uint32_t headBegin = PartitionValueHeadBegin(
                partitionOrdinal, tiling.headCount, tiling.qkHeadCount);
            const std::uint32_t headEnd = PartitionValueHeadEnd(
                partitionOrdinal, tiling.headCount, tiling.qkHeadCount);
            std::uint32_t groupOrdinal =
                partitionOrdinal * HeadGroupsPerPartition(
                                       tiling.headCount,
                                       tiling.qkHeadCount);
            for (std::uint32_t groupBegin = headBegin; groupBegin < headEnd;
                 groupBegin += kHeadsPerGroup, ++groupOrdinal) {
                const WorkItem item = BuildWorkItemRange(
                    plan, ordinal, chunk, groupOrdinal, groupBegin, headEnd,
                    tiling.headCount, tickets, tiling.qkHeadCount);
                if (!checkItem(item)) {
                    return false;
                }
            }
            return true;
        };
    for (std::uint64_t ordinal = plan.begin; ordinal < plan.end; ++ordinal) {
        std::uint32_t chunkOrdinal = 0U;
        std::uint32_t partitionOrdinal = 0U;
        DecodeChunkHeadPartitionOrdinal(plan, ordinal, chunkOrdinal,
                                        partitionOrdinal);
        const ChunkTask chunk = resolveChunk(chunkOrdinal);
        if (chunk.validRows == 0U || chunk.validRows > kChunkRows) {
            continue;
        }
        if (plan.mode == PartitionMode::ChunkOnly) {
            for (partitionOrdinal = 0U;
                 partitionOrdinal < plan.headPartitionCount;
                 ++partitionOrdinal) {
                if (!replayPartition(ordinal, chunk, partitionOrdinal)) {
                    return false;
                }
            }
        } else if (!replayPartition(ordinal, chunk, partitionOrdinal)) {
            return false;
        }
    }
    return true;
}

bool CheckAivTrace(Architecture architecture, const SyncTrace &trace,
                   const RuntimeTiling &tiling, std::uint32_t workgroupId,
                   std::uint32_t aivId, ResolveChunk resolveChunk) noexcept
{
    TraceCursor cursor{trace};
    const bool valid = ReplayItems(
        architecture, tiling, workgroupId, resolveChunk,
        [&cursor, aivId](const WorkItem &item) noexcept {
            return CheckAivItem(cursor, item, aivId);
        });
    return valid && cursor.Complete();
}

bool CheckAicTrace(Architecture architecture, const SyncTrace &trace,
                   const RuntimeTiling &tiling, std::uint32_t workgroupId,
                   ResolveChunk resolveChunk) noexcept
{
    TraceCursor cursor{trace};
    const bool valid = ReplayItems(
        architecture, tiling, workgroupId, resolveChunk,
        [&cursor](const WorkItem &item) noexcept {
            return CheckAicItem(cursor, item);
        });
    return valid && cursor.Complete();
}

bool CheckAivLocalTrace(Architecture architecture,
                        const LocalSyncTrace &trace,
                        const RuntimeTiling &tiling,
                        std::uint32_t workgroupId, std::uint32_t aivId,
                        ResolveChunk resolveChunk) noexcept
{
    LocalTraceCursor cursor{trace};
    const bool valid = ReplayItems(
        architecture, tiling, workgroupId, resolveChunk,
        [&cursor, aivId](const WorkItem &item) noexcept {
            return CheckAivLocalItem(cursor, item, aivId);
        });
    return valid && cursor.Complete();
}

bool CheckAicLocalTrace(Architecture architecture,
                        const LocalSyncTrace &trace,
                        const RuntimeTiling &tiling,
                        std::uint32_t workgroupId,
                        ResolveChunk resolveChunk) noexcept
{
    LocalTraceCursor cursor{trace};
    const bool valid = ReplayItems(
        architecture, tiling, workgroupId, resolveChunk,
        [&cursor](const WorkItem &item) noexcept {
            return CheckAicLocalItem(cursor, item);
        });
    return valid && cursor.Complete();
}

bool RunTraceCase(std::uint32_t heads, std::uint32_t totalChunks,
                  std::uint32_t workgroupCount,
                  std::uint32_t workgroupId,
                  ResolveChunk resolveChunk, PrepareAbi abi,
                  std::uint32_t qkHeads = 0U) noexcept
{
    RuntimeTiling tiling{};
    tiling.totalChunks = totalChunks;
    tiling.headCount = heads;
    tiling.qkHeadCount = qkHeads;
    tiling.aicWorkgroupCount = workgroupCount;
    tiling.epsilon = 1.0e-6F;
    tiling.key.abi = abi;

    const WorkspaceSizing sizing = CheckedWorkspaceSizing(
        Architecture::Arch22, workgroupCount, workgroupId);
    if (!sizing.valid) {
        return false;
    }
    if (resolveChunk == nullptr) {
        resolveChunk = ResolveDense;
    }

    std::array<SyncTrace, kAivPerWorkgroup> aivTraces{};
    std::array<LocalSyncTrace, kAivPerWorkgroup> aivLocalTraces{};
    std::array<MutexTrace, kAivPerWorkgroup> aivMutexTraces{};
    SyncTrace aicTrace{};
    LocalSyncTrace aicLocalTrace{};
    MutexTrace aicMutexTrace{};
    VectorOps vectorOps{};
    CubeOps cubeOps{};
    for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
        WorkspaceView workspace{};
        workspace.backingBytes = sizing.totalBytes;
        SyncLedger sync{&aivTraces[aiv], &aivLocalTraces[aiv], nullptr,
                        &aivMutexTraces[aiv]};
        RunArchitectureContract(
            Architecture::Arch22, tiling, workgroupId, CoreRole::Aiv, aiv,
            workspace, sync, vectorOps, cubeOps, resolveChunk);
    }
    WorkspaceView workspace{};
    workspace.backingBytes = sizing.totalBytes;
    SyncLedger sync{&aicTrace, &aicLocalTrace, nullptr, &aicMutexTrace};
    RunArchitectureContract(
        Architecture::Arch22, tiling, workgroupId, CoreRole::Aic, 0U,
        workspace, sync, vectorOps, cubeOps, resolveChunk);

    for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
        if (!CheckAivTrace(Architecture::Arch22, aivTraces[aiv], tiling,
                           workgroupId, aiv, resolveChunk) ||
            !CheckAivLocalTrace(Architecture::Arch22,
                                aivLocalTraces[aiv], tiling, workgroupId,
                                aiv, resolveChunk) ||
            !EmptyMutexTrace(aivMutexTraces[aiv])) {
            return false;
        }
    }
    return CheckAicTrace(Architecture::Arch22, aicTrace, tiling,
                         workgroupId, resolveChunk) &&
           CheckAicLocalTrace(Architecture::Arch22, aicLocalTrace, tiling,
                              workgroupId, resolveChunk) &&
           EmptyMutexTrace(aicMutexTrace);
}

bool RunTraceCaseForBothAbis(std::uint32_t heads,
                             std::uint32_t totalChunks,
                             std::uint32_t workgroupCount,
                             std::uint32_t workgroupId,
                             ResolveChunk resolveChunk,
                             std::uint32_t qkHeads = 0U) noexcept
{
    return RunTraceCase(heads, totalChunks, workgroupCount, workgroupId,
                        resolveChunk, PrepareAbi::Current, qkHeads) &&
           RunTraceCase(heads, totalChunks, workgroupCount, workgroupId,
                        resolveChunk, PrepareAbi::Fused, qkHeads);
}

bool CheckPow2VfDispatch(const OperationTrace &operations,
                         bool useExp2) noexcept
{
    const Pow2Primitive expected = ResolvePow2Primitive(useExp2);
    std::size_t pow2Stages = 0U;
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (record.kind != OperationKind::RunVf) {
            continue;
        }
        const bool materializesPow2 =
            record.stage == Stage::V1 || record.stage == Stage::V6;
        if (record.hasPow2Primitive != materializesPow2) {
            return false;
        }
        if (materializesPow2) {
            if (record.pow2Primitive != expected) {
                return false;
            }
            ++pow2Stages;
        }
    }
    return pow2Stages == 2U;
}

bool CheckRuntimeScaleVfDispatch(const OperationTrace &operations,
                                 PrepareAbi abi,
                                 float runtimeScale) noexcept
{
    std::size_t scaledStages = 0U;
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (record.kind != OperationKind::RunVf) {
            continue;
        }
        const bool carriesScale = record.stage == Stage::V3 ||
                                  record.stage == Stage::V6;
        if (record.hasRuntimeScale != carriesScale) {
            return false;
        }
        if (!carriesScale) {
            continue;
        }
        ++scaledStages;
        const RuntimeScaleUse expectedUse =
            record.stage == Stage::V3
                ? RuntimeScaleUse::Aqk
                : (abi == PrepareAbi::Fused
                       ? RuntimeScaleUse::FusedQg
                       : RuntimeScaleUse::None);
        const std::uint32_t expectedMultiplyCount =
            record.stage == Stage::V3 || abi == PrepareAbi::Fused ? 1U : 0U;
        if (record.runtimeScale != runtimeScale ||
            record.runtimeScaleUse != expectedUse ||
            record.runtimeScaleMultiplyCount != expectedMultiplyCount) {
            return false;
        }
    }
    return scaledStages == 2U;
}

bool CheckArch35VectorOperations(const OperationTrace &operations,
                                 const SyncTrace &synchronization,
                                 const LocalSyncTrace &localDependencies,
                                 const MutexTrace &mutexes,
                                 std::uint32_t validRows,
                                 PrepareAbi abi, bool useExp2,
                                 GateStorage gateStorage,
                                 float runtimeScale) noexcept
{
    if (operations.overflow || synchronization.overflow ||
        localDependencies.overflow ||
        !CheckPow2VfDispatch(operations, useExp2) ||
        !CheckRuntimeScaleVfDispatch(operations, abi, runtimeScale) ||
        !CheckArch35VectorMutexPipelines(
            operations, synchronization, localDependencies, mutexes)) {
        return false;
    }
    for (Stage stage : {Stage::V0, Stage::V1, Stage::V3, Stage::V6}) {
        if (CountOperations(operations, OperationKind::RunVf, stage) != 1U) {
            return false;
        }
    }

    const bool current = abi == PrepareAbi::Current;
    const bool twoByteGate = IsTwoByteGateStorage(gateStorage);
    const std::size_t gateBytes =
        static_cast<std::size_t>(validRows) * kHeadDimension *
        (twoByteGate ? kElementBytes : sizeof(float));
    const OperationRecord *gateLoad = FindUniqueOperation(
        operations, OperationKind::Load, Stage::V0, "gate", "gate-to-G");
    const std::uint64_t expectedGateOffset =
        UbPolicy::kMainBase[0] +
        (twoByteGate ? V0Gate2BLayout::kGateRaw.offset
                     : V0GateFp32Layout::kG.offset);
    if (gateLoad == nullptr || gateLoad->source.byteSize != gateBytes ||
        gateLoad->destination.byteSize != gateBytes ||
        gateLoad->destination.byteOffset != expectedGateOffset ||
        CountOperations(operations, OperationKind::Store, Stage::V1,
                        "score-source-2b", "packed-score") !=
            (twoByteGate ? 2U : 0U) ||
        CountOperations(operations, OperationKind::Store, Stage::V1,
                        "score-source-fp32", "packed-score") !=
            (twoByteGate ? 0U : 2U)) {
        return false;
    }
    if (CountOperations(operations, OperationKind::Store, Stage::V0,
                        nullptr, "gk-output") != (current ? 1U : 0U) ||
        CountOperations(operations, OperationKind::Store, Stage::V0,
                        nullptr, "G-context") != (current ? 0U : 1U) ||
        CountOperations(operations, OperationKind::Load, Stage::V6,
                        "gk-output-reuse") != (current ? 1U : 0U) ||
        CountOperations(operations, OperationKind::Load, Stage::V6,
                        "G-context") != (current ? 0U : 1U)) {
        return false;
    }
    const std::size_t gBytes = static_cast<std::size_t>(validRows) *
                               kHeadDimension * sizeof(float);
    const OperationRecord *gExport = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V0,
        current ? "G-output" : "G-context-source",
        current ? "gk-output" : "G-context");
    const OperationRecord *gReload = FindUniqueOperation(
        operations, OperationKind::Load, Stage::V6,
        current ? "gk-output-reuse" : "G-context", "G");
    if (gExport == nullptr || gReload == nullptr ||
        gExport->source.byteSize != gBytes ||
        gExport->destination.byteSize != gBytes ||
        gReload->source.byteSize != gBytes ||
        gReload->destination.byteSize != gBytes ||
        gExport->source.byteOffset !=
            UbPolicy::kMainBase[0] +
                (twoByteGate ? V0Gate2BLayout::kG.offset
                             : V0GateFp32Layout::kG.offset)) {
        return false;
    }
    const std::size_t qkBytes = static_cast<std::size_t>(validRows) *
                                kHeadDimension * kElementBytes;
    const OperationRecord *qhatStore = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V0, nullptr,
        "qhat-HK-cache");
    const OperationRecord *khatStore = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V0, nullptr,
        "khat-HK-cache");
    const OperationRecord *qhatLoad = FindUniqueOperation(
        operations, OperationKind::Load, Stage::V6, "qhat-HK-cache",
        "qhat-to-qg");
    const OperationRecord *khatLoad = FindUniqueOperation(
        operations, OperationKind::Load, Stage::V6, "khat-HK-cache",
        "khat-to-kg");
    if (qhatStore == nullptr || khatStore == nullptr || qhatLoad == nullptr ||
        khatLoad == nullptr || qhatStore->source.byteSize != qkBytes ||
        khatStore->source.byteSize != qkBytes ||
        qhatStore->destination.byteSize != qkBytes ||
        khatStore->destination.byteSize != qkBytes ||
        qhatLoad->source.byteSize != qkBytes ||
        khatLoad->source.byteSize != qkBytes ||
        qhatLoad->destination.byteSize != qkBytes ||
        khatLoad->destination.byteSize != qkBytes) {
        return false;
    }

    const std::uint32_t rhsRows = validRows > 32U ? 64U : 32U;
    const std::size_t rhsPlaneBytes =
        static_cast<std::size_t>(rhsRows) * kHeadDimension * kElementBytes;
    const OperationRecord *kBetaStore = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V6, "K-beta-g",
        "K-beta-g");
    const OperationRecord *vBetaStore = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V6, "V-beta", "V-beta");
    if (kBetaStore == nullptr || vBetaStore == nullptr ||
        kBetaStore->source.byteSize != rhsPlaneBytes ||
        kBetaStore->destination.byteSize != rhsPlaneBytes ||
        vBetaStore->source.byteSize != rhsPlaneBytes ||
        vBetaStore->destination.byteSize != rhsPlaneBytes ||
        vBetaStore->destination.byteOffset !=
            kBetaStore->destination.byteOffset + 0x4000U) {
        return false;
    }

    constexpr std::uint32_t kQuadrant = 32U;
    const std::uint32_t top = validRows < kQuadrant ? validRows : kQuadrant;
    const std::uint32_t bottom =
        validRows > kQuadrant ? validRows - kQuadrant : 0U;
    std::uint64_t lastStableWrite = 0U;
    if (current) {
        if (CountOperations(operations, OperationKind::Store, Stage::V3,
                            nullptr, "X0-tau") != 0U ||
            CountOperations(operations, OperationKind::Store, Stage::V3,
                            nullptr, "X1-tau") != 0U ||
            CountOperations(operations, OperationKind::Store, Stage::V3,
                            nullptr, "q01-zero") != 0U) {
            return false;
        }
        const OperationRecord *q00 = FindUniqueOperation(
            operations, OperationKind::Store, Stage::V3,
            "Akk-q00-valid-ld32", "AkkOut-q00-valid-ld64");
        const OperationRecord *q01 = FindUniqueOperation(
            operations, OperationKind::Store, Stage::V3,
            "Akk-q01-valid-ld32", "AkkOut-q01-valid-ld64");
        if (q00 == nullptr || q01 == nullptr ||
            !CheckMatrixSpan(q00->source, top, kQuadrant, kQuadrant, 2U) ||
            !CheckMatrixSpan(q00->destination, top, kQuadrant, 64U, 2U) ||
            !CheckMatrixSpan(q01->source, top, kQuadrant, kQuadrant, 2U) ||
            !CheckMatrixSpan(q01->destination, top, kQuadrant, 64U, 2U)) {
            return false;
        }
        lastStableWrite = q00->order > q01->order ? q00->order : q01->order;
        const std::size_t q11Count = CountOperations(
            operations, OperationKind::Store, Stage::V3,
            "Akk-q11-valid-ld32", "AkkOut-q11-valid-ld64");
        if (q11Count != (bottom == 0U ? 0U : 1U)) {
            return false;
        }
        if (bottom != 0U) {
            const OperationRecord *q11 = FindUniqueOperation(
                operations, OperationKind::Store, Stage::V3,
                "Akk-q11-valid-ld32", "AkkOut-q11-valid-ld64");
            if (q11 == nullptr ||
                !CheckMatrixSpan(q11->source, bottom, kQuadrant,
                                 kQuadrant, 2U) ||
                !CheckMatrixSpan(q11->destination, bottom, kQuadrant,
                                 64U, 2U)) {
                return false;
            }
            if (q11->order > lastStableWrite) {
                lastStableWrite = q11->order;
            }
        }
    } else {
        if (CountOperations(operations, OperationKind::Store, Stage::V3,
                            nullptr, "AkkOut-q00-valid-ld64") != 0U ||
            CountOperations(operations, OperationKind::Store, Stage::V3,
                            nullptr, "AkkOut-q01-valid-ld64") != 0U ||
            CountOperations(operations, OperationKind::Store, Stage::V3,
                            nullptr, "AkkOut-q11-valid-ld64") != 0U) {
            return false;
        }
        const OperationRecord *x0 = FindUniqueOperation(
            operations, OperationKind::Store, Stage::V3, nullptr,
            "X0-tau");
        if (x0 == nullptr) {
            return false;
        }
        lastStableWrite = x0->order;
        const std::size_t expectedBottomStores = bottom == 0U ? 0U : 1U;
        if (CountOperations(operations, OperationKind::Store, Stage::V3,
                            nullptr, "X1-tau") != expectedBottomStores ||
            CountOperations(operations, OperationKind::Store, Stage::V3,
                            nullptr, "q01-zero") != 0U) {
            return false;
        }
        if (bottom != 0U) {
            const OperationRecord *x1 = FindUniqueOperation(
                operations, OperationKind::Store, Stage::V3, nullptr,
                "X1-tau");
            if (x1 == nullptr) {
                return false;
            }
            if (x1->order > lastStableWrite) {
                lastStableWrite = x1->order;
            }
        }
    }

    std::uint64_t readyOrder = 0U;
    return lastStableWrite != 0U &&
           FindUniqueSyncOrder(synchronization, SyncAction::Set,
                               SyncPoint::V3VcsReady, 0U, 0U, Stage::V3,
                               Pipe::Mte3,
                               readyOrder) &&
           readyOrder > lastStableWrite;
}

std::size_t CountC4PhysicalFillLoadOverlaps(
    const OperationTrace &operations) noexcept
{
    std::size_t count = 0U;
    for (std::size_t fillIndex = 0U; fillIndex < operations.size;
         ++fillIndex) {
        const OperationRecord &fill = operations.records[fillIndex];
        if (fill.stage != Stage::C4 || fill.kind != OperationKind::Fill) {
            continue;
        }
        for (std::size_t loadIndex = 0U; loadIndex < operations.size;
             ++loadIndex) {
            const OperationRecord &load = operations.records[loadIndex];
            if (load.stage == Stage::C4 &&
                load.kind == OperationKind::Load &&
                fill.order < load.order &&
                PhysicalByteRangesOverlap(fill.destination,
                                          load.destination)) {
                ++count;
            }
        }
    }
    return count;
}

bool CheckArch35C4FillOrder(const OperationTrace &operations,
                            const LocalSyncTrace &localDependencies,
                            const MutexTrace &mutexes,
                            std::uint32_t validRows,
                            PrepareAbi abi) noexcept
{
    const bool hasQ10 = validRows > 32U;
    const bool needsFill = abi == PrepareAbi::Current &&
                           validRows != 32U && validRows != 64U;
    const std::size_t edgeCount = CountLocalDependencies(
        localDependencies, LocalDependency::Mte2FillToLoadWaw, Stage::C4);
    const std::size_t barrierCount = CountMutexRecords(
        mutexes, MutexAction::PipeBarrier, MutexResource::Mte2Overwrite,
        kInvalidMutexId, 0U, Stage::C4, Pipe::Mte2);
    const std::size_t overlapCount =
        CountC4PhysicalFillLoadOverlaps(operations);
    if ((overlapCount == 1U) != needsFill || edgeCount != overlapCount ||
        barrierCount != overlapCount) {
        return false;
    }
    if (!needsFill) {
        return true;
    }
    const char *fillName = hasQ10 ? "Akk-q11" : "Akk-q00";
    const char *loadSource = hasQ10 ? "AkkOut-q11-valid-ld64"
                                    : "AkkOut-q00-valid-ld64";
    const char *loadDestination = hasQ10 ? "Akk-q11-valid-ld32"
                                         : "Akk-q00-valid-ld32";
    const OperationRecord *fill = FindUniqueOperation(
        operations, OperationKind::Fill, Stage::C4, nullptr, fillName);
    const OperationRecord *load = FindUniqueOperation(
        operations, OperationKind::Load, Stage::C4, loadSource,
        loadDestination);
    std::uint64_t edgeOrder = 0U;
    std::uint64_t barrierOrder = 0U;
    return fill != nullptr && load != nullptr &&
           PhysicalByteRangesOverlap(fill->destination, load->destination) &&
           FindUniqueLocalOrder(localDependencies,
                                LocalDependency::Mte2FillToLoadWaw,
                                Stage::C4, edgeOrder) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::PipeBarrier,
               MutexResource::Mte2Overwrite, kInvalidMutexId, 0U,
               Stage::C4, Pipe::Mte2, barrierOrder) &&
           fill->order < edgeOrder && edgeOrder < barrierOrder &&
           barrierOrder < load->order &&
           !HasStageOperationBetween(operations, Stage::C4, fill->order,
                                     barrierOrder);
}

bool CheckOperandReleaseOrder(
    const OperationTrace &operations,
    const LocalSyncTrace &localDependencies, Stage stage,
    std::size_t expectedReleases) noexcept
{
    std::size_t releaseIndex = 0U;
    for (std::size_t localIndex = 0U;
         localIndex < localDependencies.size; ++localIndex) {
        const LocalSyncRecord &release =
            localDependencies.records[localIndex];
        if (release.dependency !=
                LocalDependency::CubeToMte1OperandReuse ||
            release.stage != stage) {
            continue;
        }
        ++releaseIndex;
        std::size_t precedingMmad = 0U;
        for (std::size_t operationIndex = 0U;
             operationIndex < operations.size; ++operationIndex) {
            const OperationRecord &operation =
                operations.records[operationIndex];
            const bool isMmad =
                operation.kind == OperationKind::Mmad ||
                operation.kind == OperationKind::MmadRowStackedLhs ||
                operation.kind == OperationKind::MmadQuadrantPackedLhs;
            if (isMmad && operation.stage == stage &&
                operation.order < release.order) {
                ++precedingMmad;
            }
        }
        const std::size_t expectedPreceding =
            stage == Stage::C2 ? releaseIndex
                               : (stage == Stage::C7 ? 2U : 1U);
        if (precedingMmad != expectedPreceding) {
            return false;
        }
    }
    return releaseIndex == expectedReleases;
}

std::size_t CountStageMmad(const OperationTrace &operations,
                           Stage stage) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        const bool isMmad =
            record.kind == OperationKind::Mmad ||
            record.kind == OperationKind::MmadRowStackedLhs ||
            record.kind == OperationKind::MmadQuadrantPackedLhs;
        if (record.stage == stage && isMmad) {
            ++count;
        }
    }
    return count;
}

std::size_t CountStageComputeWrites(const OperationTrace &operations,
                                    Stage stage) noexcept
{
    return CountOperations(operations, OperationKind::Store, stage) +
           CountOperations(operations, OperationKind::StoreRounded, stage);
}

bool FindFirstStageComputeWriteOrder(const OperationTrace &operations,
                                     Stage stage,
                                     std::uint64_t &order) noexcept
{
    bool found = false;
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (record.stage != stage ||
            (record.kind != OperationKind::Store &&
             record.kind != OperationKind::StoreRounded)) {
            continue;
        }
        if (!found || record.order < order) {
            order = record.order;
            found = true;
        }
    }
    return found;
}

bool FindLastStageComputeWriteOrder(const OperationTrace &operations,
                                    Stage stage,
                                    std::uint64_t &order) noexcept
{
    bool found = false;
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (record.stage != stage ||
            (record.kind != OperationKind::Store &&
             record.kind != OperationKind::StoreRounded)) {
            continue;
        }
        if (!found || record.order > order) {
            order = record.order;
            found = true;
        }
    }
    return found;
}

bool FindLastStageWaitOrder(const SyncTrace &synchronization, Stage stage,
                            std::uint64_t &order) noexcept
{
    bool found = false;
    for (std::size_t index = 0U; index < synchronization.size; ++index) {
        const SyncRecord &record = synchronization.records[index];
        const bool wait = record.action == SyncAction::Wait ||
                          record.action == SyncAction::AicWait;
        if (record.stage != stage || !wait) {
            continue;
        }
        if (!found || record.order > order) {
            order = record.order;
            found = true;
        }
    }
    return found;
}

bool CheckSingleMmadPhase(const OperationTrace &operations,
                          const SyncTrace &synchronization,
                          const LocalSyncTrace &localDependencies,
                          Stage stage, bool active,
                          bool sameStageLoad) noexcept
{
    const std::size_t mmads = CountStageMmad(operations, stage);
    const std::size_t releases = CountLocalDependencies(
        localDependencies, LocalDependency::CubeToMte1OperandReuse, stage);
    const std::size_t fixpipeEdges = CountLocalDependencies(
        localDependencies, LocalDependency::CubeToFixpipeOutput, stage);
    const std::size_t inputEdges = CountLocalDependencies(
        localDependencies, LocalDependency::Mte2ToMte1Inputs, stage);
    const std::size_t writes = CountStageComputeWrites(operations, stage);
    if (!active) {
        return mmads == 0U && releases == 0U && fixpipeEdges == 0U &&
               inputEdges == 0U && writes == 0U;
    }
    if (mmads != 1U || releases != 1U || fixpipeEdges != 1U ||
        writes == 0U || inputEdges != (sameStageLoad ? 1U : 0U)) {
        return false;
    }

    std::uint64_t producerReady = 0U;
    std::uint64_t inputReady = 0U;
    std::uint64_t mmad = 0U;
    std::uint64_t release = 0U;
    std::uint64_t fixpipe = 0U;
    std::uint64_t firstWrite = 0U;
    if (sameStageLoad) {
        std::uint64_t firstLoad = 0U;
        if (!FindOperationOrderRange(operations, OperationKind::Load, stage,
                                     firstLoad, producerReady) ||
            !FindUniqueLocalOrder(localDependencies,
                                  LocalDependency::Mte2ToMte1Inputs, stage,
                                  inputReady)) {
            return false;
        }
    } else if (!FindLastStageWaitOrder(synchronization, stage,
                                       producerReady)) {
        return false;
    }
    const bool producerReadyBeforeMmad =
        sameStageLoad ? producerReady < inputReady : true;
    return FindNthMmadOrder(operations, stage, 0U, mmad) &&
           FindUniqueLocalOrder(localDependencies,
                                LocalDependency::CubeToMte1OperandReuse,
                                stage, release) &&
           FindUniqueLocalOrder(localDependencies,
                                LocalDependency::CubeToFixpipeOutput, stage,
                                fixpipe) &&
           FindFirstStageComputeWriteOrder(operations, stage, firstWrite) &&
           producerReadyBeforeMmad &&
           (sameStageLoad ? inputReady < mmad : producerReady < mmad) &&
           mmad < release && release < fixpipe && fixpipe < firstWrite;
}

bool CheckC2BandPipeline(const OperationTrace &operations,
                         const LocalSyncTrace &localDependencies,
                         std::size_t activeBlocks,
                         bool directAivUbOutput) noexcept
{
    const std::size_t expectedStores = 2U * activeBlocks;
    if (CountOperations(operations, OperationKind::Load, Stage::C2) != 1U ||
        CountOperations(operations, OperationKind::Mmad, Stage::C2) != 0U ||
        CountOperations(operations, OperationKind::MmadRowStackedLhs,
                        Stage::C2) != activeBlocks ||
        CountOperations(operations, OperationKind::Store, Stage::C2) !=
            expectedStores ||
        CountLocalDependencies(
            localDependencies, LocalDependency::Mte2ToMte1Inputs,
            Stage::C2) != 1U ||
        CountLocalDependencies(
            localDependencies,
            LocalDependency::Mte2ToFixpipePayloadReuse,
            Stage::C2) != (directAivUbOutput ? 0U : 1U) ||
        CountLocalDependencies(
            localDependencies, LocalDependency::CubeToFixpipeOutput,
            Stage::C2) != activeBlocks) {
        return false;
    }

    std::uint64_t firstLoad = 0U;
    std::uint64_t lastLoad = 0U;
    std::uint64_t inputReady = 0U;
    std::uint64_t firstMmad = 0U;
    if (!FindOperationOrderRange(operations, OperationKind::Load, Stage::C2,
                                 firstLoad, lastLoad) ||
        !FindUniqueLocalOrder(localDependencies,
                              LocalDependency::Mte2ToMte1Inputs,
                              Stage::C2, inputReady) ||
        !FindNthOperationOrder(operations,
                               OperationKind::MmadRowStackedLhs,
                               Stage::C2, 0U, firstMmad) ||
        firstLoad > lastLoad || lastLoad >= inputReady ||
        inputReady >= firstMmad) {
        return false;
    }
    if (!directAivUbOutput) {
        std::uint64_t payloadReuse = 0U;
        if (!FindUniqueLocalOrder(
                localDependencies,
                LocalDependency::Mte2ToFixpipePayloadReuse, Stage::C2,
                payloadReuse) ||
            lastLoad >= payloadReuse || payloadReuse >= inputReady) {
            return false;
        }
    }

    for (std::size_t block = 0U; block < activeBlocks; ++block) {
        std::uint64_t bandMmad = 0U;
        std::uint64_t release = 0U;
        std::uint64_t fixpipe = 0U;
        std::uint64_t firstStore = 0U;
        std::uint64_t secondStore = 0U;
        if (!FindNthOperationOrder(
                operations, OperationKind::MmadRowStackedLhs,
                Stage::C2, block, bandMmad) ||
            !FindNthLocalOrder(
                localDependencies,
                LocalDependency::CubeToMte1OperandReuse, Stage::C2,
                block, release) ||
            !FindNthLocalOrder(
                localDependencies, LocalDependency::CubeToFixpipeOutput,
                Stage::C2, block, fixpipe) ||
            !FindNthOperationOrder(operations, OperationKind::Store,
                                   Stage::C2, 2U * block, firstStore) ||
            !FindNthOperationOrder(operations, OperationKind::Store,
                                   Stage::C2, 2U * block + 1U,
                                   secondStore)) {
            return false;
        }
        if (bandMmad >= release || release >= fixpipe ||
            fixpipe >= firstStore || firstStore >= secondStore) {
            return false;
        }

        if (block + 1U < activeBlocks) {
            std::uint64_t nextBandMmad = 0U;
            if (!FindNthOperationOrder(
                    operations, OperationKind::MmadRowStackedLhs,
                    Stage::C2, block + 1U, nextBandMmad) ||
                secondStore >= nextBandMmad) {
                return false;
            }
        }
    }
    return true;
}

bool CheckC7ComputePipeline(const OperationTrace &operations,
                            const LocalSyncTrace &localDependencies,
                            bool splitFixpipeEdges) noexcept
{
    const std::size_t expectedFixpipe = splitFixpipeEdges ? 2U : 1U;
    if (CountOperations(operations, OperationKind::StoreRounded,
                        Stage::C7) != 2U ||
        CountLocalDependencies(
            localDependencies, LocalDependency::Mte2ToMte1Inputs,
            Stage::C7) != 1U ||
        CountLocalDependencies(
            localDependencies, LocalDependency::CubeToFixpipeOutput,
            Stage::C7) != expectedFixpipe) {
        return false;
    }

    std::uint64_t firstLoad = 0U;
    std::uint64_t lastLoad = 0U;
    std::uint64_t inputReady = 0U;
    std::uint64_t firstMmad = 0U;
    std::uint64_t secondMmad = 0U;
    std::uint64_t firstStore = 0U;
    std::uint64_t secondStore = 0U;
    if (!FindOperationOrderRange(operations, OperationKind::Load, Stage::C7,
                                 firstLoad, lastLoad) ||
        !FindUniqueLocalOrder(localDependencies,
                              LocalDependency::Mte2ToMte1Inputs,
                              Stage::C7, inputReady) ||
        !FindNthMmadOrder(operations, Stage::C7, 0U, firstMmad) ||
        !FindNthMmadOrder(operations, Stage::C7, 1U, secondMmad) ||
        !FindNthOperationOrder(operations, OperationKind::StoreRounded,
                               Stage::C7, 0U, firstStore) ||
        !FindNthOperationOrder(operations, OperationKind::StoreRounded,
                               Stage::C7, 1U, secondStore) ||
        firstLoad > lastLoad || lastLoad >= inputReady ||
        inputReady >= firstMmad) {
        return false;
    }

    std::uint64_t firstFixpipe = 0U;
    if (!FindNthLocalOrder(localDependencies,
                           LocalDependency::CubeToFixpipeOutput,
                           Stage::C7, 0U, firstFixpipe)) {
        return false;
    }
    if (!splitFixpipeEdges) {
        return firstMmad < secondMmad && secondMmad < firstFixpipe &&
               firstFixpipe < firstStore && firstStore < secondStore;
    }

    std::uint64_t secondFixpipe = 0U;
    return FindNthLocalOrder(localDependencies,
                             LocalDependency::CubeToFixpipeOutput,
                             Stage::C7, 1U, secondFixpipe) &&
           firstMmad < firstFixpipe && firstFixpipe < firstStore &&
           firstStore < secondMmad && secondMmad < secondFixpipe &&
           secondFixpipe < secondStore;
}

bool CheckArch35C4Transfers(const OperationTrace &operations,
                            const LocalSyncTrace &localDependencies,
                            const MutexTrace &mutexes,
                            std::uint32_t validRows,
                            PrepareAbi abi) noexcept
{
    constexpr std::uint32_t kQuadrant = 32U;
    const bool current = abi == PrepareAbi::Current;
    const bool hasQ10 = validRows > kQuadrant;
    const std::uint32_t bottom =
        hasQ10 ? validRows - kQuadrant : 0U;
    const std::size_t expectedFills =
        hasQ10 ? 1U + (current && bottom < kQuadrant ? 1U : 0U)
               : (current && validRows < kQuadrant ? 1U : 0U);
    if (CountOperations(operations, OperationKind::Fill, Stage::C4) !=
            expectedFills ||
        !CheckArch35C4FillOrder(operations, localDependencies, mutexes,
                                validRows, abi)) {
        return false;
    }
    if (expectedFills != 0U) {
        const OperationRecord *q01Fill = hasQ10
            ? FindUniqueOperation(operations, OperationKind::Fill,
                                  Stage::C4, nullptr, "Akk-q01")
            : nullptr;
        if (hasQ10 && (q01Fill == nullptr || q01Fill->value != 0U)) {
            return false;
        }
        const bool needsPartialFill = current &&
                                      validRows != kQuadrant &&
                                      validRows != 64U;
        const char *partialFillName = hasQ10 ? "Akk-q11" : "Akk-q00";
        const OperationRecord *partialFill = needsPartialFill
            ? FindUniqueOperation(operations, OperationKind::Fill,
                                  Stage::C4, nullptr, partialFillName)
            : nullptr;
        if (needsPartialFill &&
            (partialFill == nullptr || partialFill->value != 0U)) {
            return false;
        }
    }

    if (!current) {
        if (CountOperations(operations, OperationKind::Load, Stage::C4,
                            "AkkOut-q00-valid-ld64") != 0U ||
            CountOperations(operations, OperationKind::Load, Stage::C4,
                            "AkkOut-q00-ld64") != 0U ||
            CountOperations(operations, OperationKind::Load, Stage::C4,
                            "AkkOut-q01-ld64") != 0U ||
            CountOperations(operations, OperationKind::Load, Stage::C4,
                            "AkkOut-q11-valid-ld64") != 0U) {
            return false;
        }
        const std::size_t expectedBottomLoads = hasQ10 ? 1U : 0U;
        return CountOperations(operations, OperationKind::Load, Stage::C4,
                               "X0-tau", "Akk-q00") == 1U &&
               CountOperations(operations, OperationKind::Load, Stage::C4,
                               "X1-tau", "Akk-q11") ==
                   expectedBottomLoads &&
               CountOperations(operations, OperationKind::Load, Stage::C4,
                               "q01-zero", "Akk-q01") == 0U;
    }

    if (!hasQ10) {
        const OperationRecord *q00 = FindUniqueOperation(
            operations, OperationKind::Load, Stage::C4,
            "AkkOut-q00-valid-ld64", "Akk-q00-valid-ld32");
        return q00 != nullptr &&
               CheckMatrixSpan(q00->source, validRows, kQuadrant, 64U, 2U) &&
               CheckMatrixSpan(q00->destination, validRows, kQuadrant,
                               kQuadrant, 2U);
    }

    const OperationRecord *q00 = FindUniqueOperation(
        operations, OperationKind::Load, Stage::C4, "AkkOut-q00-ld64",
        "Akk-q00");
    const OperationRecord *q11 = FindUniqueOperation(
        operations, OperationKind::Load, Stage::C4,
        "AkkOut-q11-valid-ld64", "Akk-q11-valid-ld32");
    return q00 != nullptr && q11 != nullptr &&
           CheckMatrixSpan(q00->source, kQuadrant, kQuadrant, 64U, 2U) &&
           CheckMatrixSpan(q00->destination, kQuadrant, kQuadrant,
                           kQuadrant, 2U) &&
           CheckMatrixSpan(q11->source, bottom, kQuadrant, 64U, 2U) &&
           CheckMatrixSpan(q11->destination, bottom, kQuadrant,
                           kQuadrant, 2U);
}

bool CheckStandardMmad(const OperationRecord &record, Stage stage,
                       const char *lhs, const char *rhs,
                       const char *destination, MatrixStorage lhsStorage,
                       MatrixStorage rhsStorage, std::uint32_t m,
                       std::uint32_t n, std::uint32_t k,
                       bool transposeRhs, bool negate) noexcept
{
    return record.kind == OperationKind::Mmad && record.stage == stage &&
           SameName(record.source.name, lhs) &&
           SameName(record.rhsOperand.name, rhs) &&
           SameName(record.destination.name, destination) &&
           record.lhsStorage == lhsStorage &&
           record.rhsStorage == rhsStorage && record.m == m &&
           record.n == n && record.k == k &&
           record.transposeRhs == transposeRhs && record.negate == negate;
}

bool CheckRowStackedMmad(const OperationRecord &record,
                         const char *destination,
                         std::uint32_t n,
                         NativeMatrixLayout l0aLayout,
                         MatrixStorage expectedStorage) noexcept
{
    constexpr std::uint32_t kBandRows = 16U;
    constexpr std::uint32_t kInner = 128U;
    constexpr std::size_t kBandBytes = 0x1000U;
    constexpr std::size_t kStackedL0aBytes = 0x2000U;
    return record.kind == OperationKind::MmadRowStackedLhs &&
           record.stage == Stage::C2 &&
           SameName(record.source.name, "Qplus-band") &&
           SameName(record.secondarySource.name, "Kplus-band") &&
           SameName(record.rhsOperand.name, "Kminus-prefix") &&
           SameName(record.destination.name, destination) &&
           SameName(record.l0aOperand.name, "C2-QK-stacked-L0A") &&
           SameName(record.l0bOperand.name, "C2-Kminus-L0B") &&
           record.lhsStorage == expectedStorage &&
           record.rhsStorage == expectedStorage &&
           record.m == 2U * kBandRows && record.n == n &&
           record.k == kInner && record.lhsTopRows == kBandRows &&
           record.lhsBottomRows == kBandRows && record.transposeRhs &&
           !record.negate && record.source.byteSize == kBandBytes &&
           record.secondarySource.byteSize == kBandBytes &&
           record.rhsOperand.byteSize ==
               static_cast<std::size_t>(n) * kInner * 2U &&
           record.destination.byteSize ==
               static_cast<std::size_t>(2U * kBandRows) * n * 4U &&
           record.l0aOperand.byteSize == kStackedL0aBytes &&
           record.l0bOperand.byteSize ==
               static_cast<std::size_t>(n) * kInner * 2U &&
           record.l0aOperand.space == MemorySpace::L0A &&
           record.lhsTopL0aTile.space == MemorySpace::L0A &&
           record.lhsBottomL0aTile.space == MemorySpace::L0A &&
           record.destination.space == MemorySpace::L0 &&
           CheckNativeMatrixView(record.l0aOperand, l0aLayout,
                                 2U * kBandRows, kInner, 0U, 0U,
                                 2U * kBandRows, kInner, 2U) &&
           CheckNativeMatrixView(record.lhsTopL0aTile, l0aLayout,
                                 2U * kBandRows, kInner, 0U, 0U,
                                 kBandRows, kInner, 2U) &&
           CheckNativeMatrixView(record.lhsBottomL0aTile, l0aLayout,
                                 2U * kBandRows, kInner, kBandRows, 0U,
                                 kBandRows, kInner, 2U) &&
           SamePhysicalSpan(record.l0aOperand,
                            record.lhsTopL0aTile) &&
           SamePhysicalSpan(record.l0aOperand,
                            record.lhsBottomL0aTile) &&
           CheckNativeMatrixView(record.destination,
                                 NativeMatrixLayout::L0cFractal,
                                 2U * kBandRows, n, 0U, 0U,
                                 2U * kBandRows, n, 4U);
}

bool CheckC2StoreDescriptors(const OperationTrace &operations,
                             std::size_t activeBlocks,
                             Architecture architecture) noexcept
{
    constexpr std::uint32_t kBandRows = 16U;
    constexpr std::uint32_t kRawLeadingDimension = 64U;
    constexpr std::uint32_t kFp32Bytes = 4U;
    const bool arch22 = architecture == Architecture::Arch22;
    const OperationRecord *payloadLoad =
        arch22 ? FindUniqueOperation(
                     operations, OperationKind::Load, Stage::C2,
                     "shared-payload", "packed-score-L1")
               : nullptr;
    const OperationRecord *firstAqk = FindNthOperation(
        operations, OperationKind::Store, Stage::C2, 0U);
    const OperationRecord *firstAkk = FindNthOperation(
        operations, OperationKind::Store, Stage::C2, 1U);
    if (firstAqk == nullptr || firstAkk == nullptr ||
        (arch22 && payloadLoad == nullptr)) {
        return false;
    }

    for (std::size_t block = 0U; block < activeBlocks; ++block) {
        const OperationRecord *mmad = FindNthOperation(
            operations, OperationKind::MmadRowStackedLhs,
            Stage::C2, block);
        const OperationRecord *aqk = FindNthOperation(
            operations, OperationKind::Store, Stage::C2, 2U * block);
        const OperationRecord *akk = FindNthOperation(
            operations, OperationKind::Store, Stage::C2,
            2U * block + 1U);
        if (mmad == nullptr || aqk == nullptr || akk == nullptr) {
            return false;
        }
        const std::uint32_t n = kPrefixRows[block];
        const char *aqkSource =
            arch22 ? "Aqk-compact-L0C" : "raw-Aqk-L0C";
        const char *akkSource =
            arch22 ? "Akk-compact-L0C" : "raw-Akk-L0C";
        if (!SameName(aqk->source.name, aqkSource) ||
            !SameName(akk->source.name, akkSource) ||
            !CheckNativeMatrixView(
                aqk->source, NativeMatrixLayout::L0cFractal,
                2U * kBandRows, n, 0U, 0U, kBandRows, n,
                kFp32Bytes) ||
            !CheckNativeMatrixView(
                akk->source, NativeMatrixLayout::L0cFractal,
                2U * kBandRows, n, kBandRows, 0U, kBandRows, n,
                kFp32Bytes) ||
            !SamePhysicalSpan(aqk->source, akk->source) ||
            !SamePhysicalSpan(aqk->source, mmad->destination)) {
            return false;
        }

        if (!arch22) {
            const std::uint64_t rowOffset =
                block * kBandRows * kRawLeadingDimension * kFp32Bytes;
            const std::uint64_t rawPlaneDistance =
                V3Layout::kRawAkk.offset - V3Layout::kRawAqk.offset;
            if (!SameName(aqk->destination.name,
                          "raw-Aqk-band-ld64") ||
                !SameName(akk->destination.name,
                          "raw-Akk-band-ld64") ||
                aqk->destination.space != MemorySpace::Ub ||
                akk->destination.space != MemorySpace::Ub ||
                !CheckMatrixSpan(aqk->destination, kBandRows, n,
                                 kRawLeadingDimension, kFp32Bytes) ||
                !CheckMatrixSpan(akk->destination, kBandRows, n,
                                 kRawLeadingDimension, kFp32Bytes) ||
                aqk->destination.byteOffset !=
                    firstAqk->destination.byteOffset + rowOffset ||
                akk->destination.byteOffset !=
                    firstAkk->destination.byteOffset + rowOffset ||
                firstAkk->destination.byteOffset !=
                    firstAqk->destination.byteOffset + rawPlaneDistance) {
                return false;
            }
            continue;
        }

        const Region &aqkRegion =
            arch22_policy::WorkspacePolicy::kRelayRawAqk[block];
        const Region &akkRegion =
            arch22_policy::WorkspacePolicy::kRelayRawAkk[block];
        if (!SameName(aqk->destination.name, "Aqk-compact-relay") ||
            !SameName(akk->destination.name, "Akk-compact-relay") ||
            aqk->destination.space != MemorySpace::Workspace ||
            akk->destination.space != MemorySpace::Workspace ||
            !CheckMatrixSpan(aqk->destination, kBandRows, n, n,
                             kFp32Bytes) ||
            !CheckMatrixSpan(akk->destination, kBandRows, n, n,
                             kFp32Bytes) ||
            aqk->destination.byteOffset !=
                payloadLoad->source.byteOffset + aqkRegion.offset ||
            akk->destination.byteOffset !=
                payloadLoad->source.byteOffset + akkRegion.offset ||
            aqk->destination.byteSize != aqkRegion.size ||
            akk->destination.byteSize != akkRegion.size ||
            aqk->destination.slot != payloadLoad->source.slot ||
            akk->destination.slot != payloadLoad->source.slot ||
            aqk->destination.generation !=
                payloadLoad->source.generation ||
            akk->destination.generation !=
                payloadLoad->source.generation ||
            aqk->destination.ownerRole !=
                payloadLoad->source.ownerRole ||
            akk->destination.ownerRole !=
                payloadLoad->source.ownerRole ||
            aqk->destination.ownerId != payloadLoad->source.ownerId ||
            akk->destination.ownerId != payloadLoad->source.ownerId) {
            return false;
        }
    }
    return true;
}

bool CheckC2RowStackedMmadDescriptors(
    const OperationTrace &operations, std::uint32_t validRows,
    const char *destination, NativeMatrixLayout l0aLayout,
    Architecture architecture, MatrixStorage expectedStorage) noexcept
{
    const std::size_t activeBlocks =
        (static_cast<std::size_t>(validRows) + 15U) / 16U;
    std::size_t c2Index = 0U;
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (record.kind != OperationKind::MmadRowStackedLhs ||
            record.stage != Stage::C2) {
            continue;
        }
        if (c2Index >= activeBlocks ||
            !CheckRowStackedMmad(record, destination,
                                 kPrefixRows[c2Index], l0aLayout,
                                 expectedStorage)) {
            return false;
        }
        ++c2Index;
    }
    return c2Index == activeBlocks &&
           CountOperations(operations, OperationKind::Mmad,
                           Stage::C2) == 0U &&
           CheckC2StoreDescriptors(operations, activeBlocks,
                                   architecture);
}

bool CheckQuadrantMmad(const OperationRecord &record, const char *rhs,
                       const char *destination,
                       MatrixStorage lhsStorage,
                       MatrixStorage rhsStorage) noexcept
{
    return record.kind == OperationKind::MmadQuadrantPackedLhs &&
           record.stage == Stage::C7 &&
           SameName(record.source.name, "Akk-resident") &&
           SameName(record.rhsOperand.name, rhs) &&
           SameName(record.destination.name, destination) &&
           record.lhsStorage == lhsStorage &&
           record.rhsStorage == rhsStorage &&
           record.quadrantRows == 32U && record.quadrantColumns == 32U &&
           record.n == 128U && record.k == 64U;
}

bool CheckArch35MmadDescriptors(const OperationTrace &operations,
                                std::uint32_t validRows,
                                const ProposedTilingKey &key) noexcept
{
    const MatrixStorage inputStorage =
        FromInputStorage(key.inputStorage);
    const MatrixStorage valueStorage =
        FromInputStorage(key.valueStorage);
    const MatrixStorage scoreStorage =
        FromScoreStorage(key.scoreStorage);
    if (!CheckC2RowStackedMmadDescriptors(
            operations, validRows, "raw-Aqk-Akk-packed-L0C",
            NativeMatrixLayout::L0aZN, Architecture::Arch35,
            scoreStorage)) {
        return false;
    }

    const bool hasQ10 = validRows > 32U;
    if (hasQ10) {
        const OperationRecord *c4 = FindUniqueOperation(
            operations, OperationKind::Mmad, Stage::C4, "B-current",
            "T-L0C");
        const OperationRecord *c5 = FindUniqueOperation(
            operations, OperationKind::Mmad, Stage::C5, "X1-resident",
            "Y-fp32-L0C");
        if (c4 == nullptr || c5 == nullptr ||
            !CheckStandardMmad(*c4, Stage::C4, "B-current",
                               "X0-resident", "T-L0C",
                               MatrixStorage::Fp32, MatrixStorage::Fp32,
                               32U, 32U, 32U, false, false) ||
            !CheckStandardMmad(*c5, Stage::C5, "X1-resident",
                               "T-resident", "Y-fp32-L0C",
                               MatrixStorage::Fp32, MatrixStorage::Fp32,
                               32U, 32U, 32U, false, true)) {
            return false;
        }
        const OperationRecord *w = FindUniqueOperation(
            operations, OperationKind::MmadQuadrantPackedLhs, Stage::C7,
            "Akk-resident", "W-fp32-L0C");
        const OperationRecord *u = FindUniqueOperation(
            operations, OperationKind::MmadQuadrantPackedLhs, Stage::C7,
            "Akk-resident", "U-fp32-L0C");
        return w != nullptr && u != nullptr &&
               CheckQuadrantMmad(*w, "K-beta-g-RHS", "W-fp32-L0C",
                                 inputStorage, inputStorage) &&
               CheckQuadrantMmad(*u, "V-beta-RHS", "U-fp32-L0C",
                                 inputStorage, valueStorage);
    }

    const OperationRecord *w = FindUniqueOperation(
        operations, OperationKind::Mmad, Stage::C7, "Akk-q00",
        "W-fp32-L0C");
    const OperationRecord *u = FindUniqueOperation(
        operations, OperationKind::Mmad, Stage::C7, "Akk-q00",
        "U-fp32-L0C");
    return w != nullptr && u != nullptr &&
           CheckStandardMmad(*w, Stage::C7, "Akk-q00",
                              "K-beta-g-top32", "W-fp32-L0C",
                              inputStorage, inputStorage,
                              32U, 128U, 32U, false, false) &&
           CheckStandardMmad(*u, Stage::C7, "Akk-q00", "V-beta-top32",
                              "U-fp32-L0C", inputStorage,
                              valueStorage, 32U, 128U, 32U, false,
                              false);
}

bool CheckArch35C2MutexPipeline(
    const OperationTrace &operations, const SyncTrace &synchronization,
    const LocalSyncTrace &localDependencies, const MutexTrace &mutexes,
    std::size_t activeBlocks) noexcept
{
    constexpr std::uint32_t kOwner = 0U;
    const SymbolicMutexId l1 = Arch35CubeMutexIds::L1Bank(kOwner);
    const SymbolicMutexId l0 = Arch35CubeMutexIds::L0OperandBank(0U);
    const SymbolicMutexId lower =
        Arch35CubeMutexIds::L0cLowerHalf(kOwner);
    std::uint64_t l1Mte2Lock = 0U;
    std::uint64_t l1Mte2Unlock = 0U;
    std::uint64_t l1Mte1Lock = 0U;
    std::uint64_t l1Mte1Unlock = 0U;
    std::uint64_t firstLoad = 0U;
    std::uint64_t lastLoad = 0U;
    std::uint64_t inputReady = 0U;
    std::uint64_t payloadFree = 0U;
    std::uint64_t rawReady = 0U;
    if (CountMutexStageOwner(mutexes, Stage::C2, kOwner) !=
            4U + 8U * activeBlocks ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C2, Pipe::Mte2, l1Mte2Lock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C2, Pipe::Mte2, l1Mte2Unlock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C2, Pipe::Mte1, l1Mte1Lock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C2, Pipe::Mte1, l1Mte1Unlock) ||
        !FindOperationOrderRange(operations, OperationKind::Load,
                                 Stage::C2, firstLoad, lastLoad) ||
        !FindUniqueLocalOrder(localDependencies,
                              LocalDependency::Mte2ToMte1Inputs,
                              Stage::C2, inputReady) ||
        !FindUniqueSyncOrder(synchronization, SyncAction::Set,
                             SyncPoint::C2ScorePayloadFree, kOwner, 0U,
                             Stage::C2, Pipe::Mte2, payloadFree) ||
        !FindUniqueSyncOrder(synchronization, SyncAction::Set,
                             SyncPoint::C2RawReady, kOwner, 0U,
                             Stage::C2, Pipe::Fixpipe, rawReady) ||
        !(l1Mte2Lock < firstLoad && firstLoad <= lastLoad &&
          lastLoad < l1Mte2Unlock && l1Mte2Unlock < inputReady &&
          l1Mte2Unlock < payloadFree && inputReady < l1Mte1Lock) ||
        !CheckStageWaitsBeforeOrder(synchronization, Stage::C2, 3U,
                                    l1Mte2Lock)) {
        return false;
    }

    std::uint64_t finalFixpipeUnlock = 0U;
    for (std::size_t block = 0U; block < activeBlocks; ++block) {
        std::uint64_t l0Mte1Lock = 0U;
        std::uint64_t l0Mte1Unlock = 0U;
        std::uint64_t lowerCubeLock = 0U;
        std::uint64_t lowerCubeUnlock = 0U;
        std::uint64_t l0CubeLock = 0U;
        std::uint64_t l0CubeUnlock = 0U;
        std::uint64_t lowerFixpipeLock = 0U;
        std::uint64_t lowerFixpipeUnlock = 0U;
        std::uint64_t mmad = 0U;
        std::uint64_t operandReleased = 0U;
        std::uint64_t cubeReady = 0U;
        std::uint64_t firstStore = 0U;
        std::uint64_t secondStore = 0U;
        if (!FindNthMutexOrder(
                mutexes, MutexAction::Lock,
                MutexResource::AicL0OperandBank, l0, 0U, Stage::C2,
                Pipe::Mte1, block, l0Mte1Lock) ||
            !FindNthMutexOrder(
                mutexes, MutexAction::Unlock,
                MutexResource::AicL0OperandBank, l0, 0U, Stage::C2,
                Pipe::Mte1, block, l0Mte1Unlock) ||
            !FindNthMutexOrder(
                mutexes, MutexAction::Lock,
                MutexResource::AicL0cLowerHalf, lower, kOwner,
                Stage::C2, Pipe::Cube, block, lowerCubeLock) ||
            !FindNthMutexOrder(
                mutexes, MutexAction::Unlock,
                MutexResource::AicL0cLowerHalf, lower, kOwner,
                Stage::C2, Pipe::Cube, block, lowerCubeUnlock) ||
            !FindNthMutexOrder(
                mutexes, MutexAction::Lock,
                MutexResource::AicL0OperandBank, l0, 0U, Stage::C2,
                Pipe::Cube, block, l0CubeLock) ||
            !FindNthMutexOrder(
                mutexes, MutexAction::Unlock,
                MutexResource::AicL0OperandBank, l0, 0U, Stage::C2,
                Pipe::Cube, block, l0CubeUnlock) ||
            !FindNthMutexOrder(
                mutexes, MutexAction::Lock,
                MutexResource::AicL0cLowerHalf, lower, kOwner,
                Stage::C2, Pipe::Fixpipe, block, lowerFixpipeLock) ||
            !FindNthMutexOrder(
                mutexes, MutexAction::Unlock,
                MutexResource::AicL0cLowerHalf, lower, kOwner,
                Stage::C2, Pipe::Fixpipe, block, lowerFixpipeUnlock) ||
            !FindNthOperationOrder(
                operations, OperationKind::MmadRowStackedLhs,
                Stage::C2, block, mmad) ||
            !FindNthLocalOrder(
                localDependencies,
                LocalDependency::CubeToMte1OperandReuse, Stage::C2,
                block, operandReleased) ||
            !FindNthLocalOrder(
                localDependencies, LocalDependency::CubeToFixpipeOutput,
                Stage::C2, block, cubeReady) ||
            !FindNthOperationOrder(operations, OperationKind::Store,
                                   Stage::C2, 2U * block, firstStore) ||
            !FindNthOperationOrder(operations, OperationKind::Store,
                                   Stage::C2, 2U * block + 1U,
                                   secondStore) ||
            !(l1Mte1Lock < l0Mte1Lock &&
              l0Mte1Lock < l0Mte1Unlock &&
              l0Mte1Unlock < lowerCubeLock &&
              lowerCubeLock < l0CubeLock && l0CubeLock < mmad &&
              mmad < l0CubeUnlock && l0CubeUnlock < lowerCubeUnlock &&
              lowerCubeUnlock < operandReleased &&
              operandReleased < cubeReady &&
              cubeReady < lowerFixpipeLock &&
              lowerFixpipeLock < firstStore && firstStore < secondStore &&
              secondStore < lowerFixpipeUnlock)) {
            return false;
        }
        finalFixpipeUnlock = lowerFixpipeUnlock;
    }
    return finalFixpipeUnlock < l1Mte1Unlock &&
           l1Mte1Unlock < rawReady;
}

bool CheckArch35C4MutexPipeline(
    const OperationTrace &operations, const SyncTrace &synchronization,
    const LocalSyncTrace &localDependencies, const MutexTrace &mutexes,
    bool hasQ10, bool hasBarrier, PrepareAbi abi) noexcept
{
    constexpr std::uint32_t kOwner = 0U;
    const SymbolicMutexId l1 = Arch35CubeMutexIds::L1Bank(kOwner);
    const SymbolicMutexId l0 = Arch35CubeMutexIds::L0OperandBank(0U);
    const SymbolicMutexId lower =
        Arch35CubeMutexIds::L0cLowerHalf(kOwner);
    std::uint64_t l1Mte2Lock = 0U;
    std::uint64_t l1Mte2Unlock = 0U;
    std::uint64_t firstLoad = 0U;
    std::uint64_t lastLoad = 0U;
    std::uint64_t payloadFree = 0U;
    const Pipe payloadFreePipe =
        !hasQ10 && abi == PrepareAbi::Current ? Pipe::Control : Pipe::Mte2;
    const std::size_t expectedRecords =
        (hasQ10 ? 14U : 2U) + (hasBarrier ? 1U : 0U);
    if (CountMutexStageOwner(mutexes, Stage::C4, kOwner) !=
            expectedRecords ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C4, Pipe::Mte2, l1Mte2Lock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C4, Pipe::Mte2, l1Mte2Unlock) ||
        !FindOperationOrderRange(operations, OperationKind::Load,
                                 Stage::C4, firstLoad, lastLoad) ||
        !FindUniqueOwnerSyncOrderOnPipe(
            synchronization, SyncAction::Set, SyncPoint::C4PayloadFree,
            kOwner, 0U, Stage::C4, payloadFreePipe, payloadFree) ||
        !(l1Mte2Lock < firstLoad && firstLoad <= lastLoad &&
          lastLoad < l1Mte2Unlock && l1Mte2Unlock < payloadFree) ||
        !CheckStageWaitsBeforeOrder(synchronization, Stage::C4, 1U,
                                    l1Mte2Lock)) {
        return false;
    }
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (record.stage == Stage::C4 &&
            (record.kind == OperationKind::Load ||
             record.kind == OperationKind::Fill) &&
            !(l1Mte2Lock < record.order &&
              record.order < l1Mte2Unlock)) {
            return false;
        }
    }
    if (!hasQ10) {
        return true;
    }

    std::uint64_t inputReady = 0U;
    std::uint64_t l1Mte1Lock = 0U;
    std::uint64_t l1Mte1Unlock = 0U;
    std::uint64_t l0Mte1Lock = 0U;
    std::uint64_t l0Mte1Unlock = 0U;
    std::uint64_t lowerCubeLock = 0U;
    std::uint64_t lowerCubeUnlock = 0U;
    std::uint64_t l0CubeLock = 0U;
    std::uint64_t l0CubeUnlock = 0U;
    std::uint64_t operandReleased = 0U;
    std::uint64_t cubeReady = 0U;
    std::uint64_t lowerFixpipeLock = 0U;
    std::uint64_t lowerFixpipeUnlock = 0U;
    std::uint64_t l1FixpipeLock = 0U;
    std::uint64_t l1FixpipeUnlock = 0U;
    std::uint64_t mmad = 0U;
    std::uint64_t firstWrite = 0U;
    std::uint64_t lastWrite = 0U;
    return FindUniqueLocalOrder(
               localDependencies, LocalDependency::Mte2ToMte1Inputs,
               Stage::C4, inputReady) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
               kOwner, Stage::C4, Pipe::Mte1, l1Mte1Lock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL1Bank, l1, kOwner, Stage::C4,
               Pipe::Mte1, l1Mte1Unlock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock,
               MutexResource::AicL0OperandBank, l0, 0U, Stage::C4,
               Pipe::Mte1, l0Mte1Lock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL0OperandBank, l0, 0U, Stage::C4,
               Pipe::Mte1, l0Mte1Unlock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock,
               MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C4,
               Pipe::Cube, lowerCubeLock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C4,
               Pipe::Cube, lowerCubeUnlock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock,
               MutexResource::AicL0OperandBank, l0, 0U, Stage::C4,
               Pipe::Cube, l0CubeLock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL0OperandBank, l0, 0U, Stage::C4,
               Pipe::Cube, l0CubeUnlock) &&
           FindUniqueLocalOrder(
               localDependencies,
               LocalDependency::CubeToMte1OperandReuse, Stage::C4,
               operandReleased) &&
           FindUniqueLocalOrder(
               localDependencies, LocalDependency::CubeToFixpipeOutput,
               Stage::C4, cubeReady) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock,
               MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C4,
               Pipe::Fixpipe, lowerFixpipeLock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C4,
               Pipe::Fixpipe, lowerFixpipeUnlock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
               kOwner, Stage::C4, Pipe::Fixpipe, l1FixpipeLock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL1Bank, l1, kOwner, Stage::C4,
               Pipe::Fixpipe, l1FixpipeUnlock) &&
           FindNthMmadOrder(operations, Stage::C4, 0U, mmad) &&
           FindFirstStageComputeWriteOrder(operations, Stage::C4,
                                           firstWrite) &&
           FindLastStageComputeWriteOrder(operations, Stage::C4,
                                          lastWrite) &&
           l1Mte2Unlock < inputReady && inputReady < l1Mte1Lock &&
           l1Mte1Lock < l0Mte1Lock && l0Mte1Lock < l0Mte1Unlock &&
           l0Mte1Unlock < l1Mte1Unlock &&
           l1Mte1Unlock < lowerCubeLock &&
           lowerCubeLock < l0CubeLock && l0CubeLock < mmad &&
           mmad < l0CubeUnlock && l0CubeUnlock < lowerCubeUnlock &&
           lowerCubeUnlock < operandReleased &&
           operandReleased < cubeReady && cubeReady < lowerFixpipeLock &&
           lowerFixpipeLock < l1FixpipeLock &&
           l1FixpipeLock < firstWrite && firstWrite <= lastWrite &&
           lastWrite < l1FixpipeUnlock &&
           l1FixpipeUnlock < lowerFixpipeUnlock;
}

bool CheckArch35C5MutexPipeline(
    const OperationTrace &operations, const SyncTrace &synchronization,
    const LocalSyncTrace &localDependencies, const MutexTrace &mutexes,
    bool hasQ10) noexcept
{
    constexpr std::uint32_t kOwner = 0U;
    if (!hasQ10) {
        return CountMutexStageOwner(mutexes, Stage::C5, kOwner) == 0U &&
               CountStageMmad(operations, Stage::C5) == 0U &&
               CountStageComputeWrites(operations, Stage::C5) == 0U;
    }
    for (std::size_t index = 0U; index < synchronization.size; ++index) {
        if (synchronization.records[index].stage == Stage::C5) {
            return false;
        }
    }
    const SymbolicMutexId l1 = Arch35CubeMutexIds::L1Bank(kOwner);
    const SymbolicMutexId l0 = Arch35CubeMutexIds::L0OperandBank(0U);
    const SymbolicMutexId lower =
        Arch35CubeMutexIds::L0cLowerHalf(kOwner);
    std::uint64_t l1Mte1Lock = 0U;
    std::uint64_t l1Mte1Unlock = 0U;
    std::uint64_t l0Mte1Lock = 0U;
    std::uint64_t l0Mte1Unlock = 0U;
    std::uint64_t lowerCubeLock = 0U;
    std::uint64_t lowerCubeUnlock = 0U;
    std::uint64_t l0CubeLock = 0U;
    std::uint64_t l0CubeUnlock = 0U;
    std::uint64_t operandReleased = 0U;
    std::uint64_t cubeReady = 0U;
    std::uint64_t lowerFixpipeLock = 0U;
    std::uint64_t lowerFixpipeUnlock = 0U;
    std::uint64_t l1FixpipeLock = 0U;
    std::uint64_t l1FixpipeUnlock = 0U;
    std::uint64_t mmad = 0U;
    std::uint64_t firstWrite = 0U;
    std::uint64_t lastWrite = 0U;
    return CountMutexStageOwner(mutexes, Stage::C5, kOwner) == 12U &&
           CountLocalDependencies(
               localDependencies, LocalDependency::Mte2ToMte1Inputs,
               Stage::C5) == 0U &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
               kOwner, Stage::C5, Pipe::Mte1, l1Mte1Lock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL1Bank, l1, kOwner, Stage::C5,
               Pipe::Mte1, l1Mte1Unlock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock,
               MutexResource::AicL0OperandBank, l0, 0U, Stage::C5,
               Pipe::Mte1, l0Mte1Lock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL0OperandBank, l0, 0U, Stage::C5,
               Pipe::Mte1, l0Mte1Unlock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock,
               MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C5,
               Pipe::Cube, lowerCubeLock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C5,
               Pipe::Cube, lowerCubeUnlock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock,
               MutexResource::AicL0OperandBank, l0, 0U, Stage::C5,
               Pipe::Cube, l0CubeLock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL0OperandBank, l0, 0U, Stage::C5,
               Pipe::Cube, l0CubeUnlock) &&
           FindUniqueLocalOrder(
               localDependencies,
               LocalDependency::CubeToMte1OperandReuse, Stage::C5,
               operandReleased) &&
           FindUniqueLocalOrder(
               localDependencies, LocalDependency::CubeToFixpipeOutput,
               Stage::C5, cubeReady) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock,
               MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C5,
               Pipe::Fixpipe, lowerFixpipeLock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C5,
               Pipe::Fixpipe, lowerFixpipeUnlock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
               kOwner, Stage::C5, Pipe::Fixpipe, l1FixpipeLock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Unlock,
               MutexResource::AicL1Bank, l1, kOwner, Stage::C5,
               Pipe::Fixpipe, l1FixpipeUnlock) &&
           FindNthMmadOrder(operations, Stage::C5, 0U, mmad) &&
           FindFirstStageComputeWriteOrder(operations, Stage::C5,
                                           firstWrite) &&
           FindLastStageComputeWriteOrder(operations, Stage::C5,
                                          lastWrite) &&
           l1Mte1Lock < l0Mte1Lock && l0Mte1Lock < l0Mte1Unlock &&
           l0Mte1Unlock < l1Mte1Unlock &&
           l1Mte1Unlock < lowerCubeLock &&
           lowerCubeLock < l0CubeLock && l0CubeLock < mmad &&
           mmad < l0CubeUnlock && l0CubeUnlock < lowerCubeUnlock &&
           lowerCubeUnlock < operandReleased &&
           operandReleased < cubeReady && cubeReady < lowerFixpipeLock &&
           lowerFixpipeLock < l1FixpipeLock &&
           l1FixpipeLock < firstWrite && firstWrite <= lastWrite &&
           lastWrite < l1FixpipeUnlock &&
           l1FixpipeUnlock < lowerFixpipeUnlock;
}

bool CheckArch35C7MutexPipeline(
    const OperationTrace &operations, const SyncTrace &synchronization,
    const LocalSyncTrace &localDependencies,
    const MutexTrace &mutexes) noexcept
{
    constexpr std::uint32_t kOwner = 0U;
    const SymbolicMutexId l1 = Arch35CubeMutexIds::L1Bank(kOwner);
    const SymbolicMutexId l0 = Arch35CubeMutexIds::L0OperandBank(0U);
    const SymbolicMutexId lower =
        Arch35CubeMutexIds::L0cLowerHalf(kOwner);
    const SymbolicMutexId upper =
        Arch35CubeMutexIds::L0cUpperHalf(kOwner);
    std::uint64_t rhsReady = 0U;
    std::uint64_t l1Mte2Lock = 0U;
    std::uint64_t l1Mte2Unlock = 0U;
    std::uint64_t firstLoad = 0U;
    std::uint64_t lastLoad = 0U;
    std::uint64_t inputReady = 0U;
    std::uint64_t l1Mte1Lock = 0U;
    std::uint64_t l1Mte1Unlock = 0U;
    std::uint64_t l0Mte1Lock = 0U;
    std::uint64_t l0Mte1Unlock = 0U;
    std::uint64_t l0CubeLock = 0U;
    std::uint64_t l0CubeUnlock = 0U;
    std::uint64_t lowerCubeLock = 0U;
    std::uint64_t lowerCubeUnlock = 0U;
    std::uint64_t lowerFixpipeLock = 0U;
    std::uint64_t lowerFixpipeUnlock = 0U;
    std::uint64_t upperCubeLock = 0U;
    std::uint64_t upperCubeUnlock = 0U;
    std::uint64_t upperFixpipeLock = 0U;
    std::uint64_t upperFixpipeUnlock = 0U;
    std::uint64_t firstMmad = 0U;
    std::uint64_t secondMmad = 0U;
    std::uint64_t firstStore = 0U;
    std::uint64_t secondStore = 0U;
    std::uint64_t firstCubeReady = 0U;
    std::uint64_t secondCubeReady = 0U;
    std::uint64_t operandReleased = 0U;
    std::uint64_t slotFree = 0U;
    if (CountMutexStageOwner(mutexes, Stage::C7, kOwner) != 16U ||
        !FindUniqueSyncOrder(synchronization, SyncAction::Wait,
                             SyncPoint::V6RhsReady, kOwner, 0U,
                             Stage::C7, Pipe::Mte2, rhsReady) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C7, Pipe::Mte2, l1Mte2Lock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C7, Pipe::Mte2, l1Mte2Unlock) ||
        !FindOperationOrderRange(operations, OperationKind::Load,
                                 Stage::C7, firstLoad, lastLoad) ||
        !FindUniqueLocalOrder(localDependencies,
                              LocalDependency::Mte2ToMte1Inputs,
                              Stage::C7, inputReady) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C7, Pipe::Mte1, l1Mte1Lock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock, MutexResource::AicL1Bank, l1,
            kOwner, Stage::C7, Pipe::Mte1, l1Mte1Unlock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock,
            MutexResource::AicL0OperandBank, l0, 0U, Stage::C7,
            Pipe::Mte1, l0Mte1Lock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock,
            MutexResource::AicL0OperandBank, l0, 0U, Stage::C7,
            Pipe::Mte1, l0Mte1Unlock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock,
            MutexResource::AicL0OperandBank, l0, 0U, Stage::C7,
            Pipe::Cube, l0CubeLock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock,
            MutexResource::AicL0OperandBank, l0, 0U, Stage::C7,
            Pipe::Cube, l0CubeUnlock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock,
            MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C7,
            Pipe::Cube, lowerCubeLock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock,
            MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C7,
            Pipe::Cube, lowerCubeUnlock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock,
            MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C7,
            Pipe::Fixpipe, lowerFixpipeLock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock,
            MutexResource::AicL0cLowerHalf, lower, kOwner, Stage::C7,
            Pipe::Fixpipe, lowerFixpipeUnlock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock,
            MutexResource::AicL0cUpperHalf, upper, kOwner, Stage::C7,
            Pipe::Cube, upperCubeLock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock,
            MutexResource::AicL0cUpperHalf, upper, kOwner, Stage::C7,
            Pipe::Cube, upperCubeUnlock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock,
            MutexResource::AicL0cUpperHalf, upper, kOwner, Stage::C7,
            Pipe::Fixpipe, upperFixpipeLock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Unlock,
            MutexResource::AicL0cUpperHalf, upper, kOwner, Stage::C7,
            Pipe::Fixpipe, upperFixpipeUnlock) ||
        !FindNthMmadOrder(operations, Stage::C7, 0U, firstMmad) ||
        !FindNthMmadOrder(operations, Stage::C7, 1U, secondMmad) ||
        !FindNthOperationOrder(operations, OperationKind::StoreRounded,
                               Stage::C7, 0U, firstStore) ||
        !FindNthOperationOrder(operations, OperationKind::StoreRounded,
                               Stage::C7, 1U, secondStore) ||
        !FindNthLocalOrder(
            localDependencies, LocalDependency::CubeToFixpipeOutput,
            Stage::C7, 0U, firstCubeReady) ||
        !FindNthLocalOrder(
            localDependencies, LocalDependency::CubeToFixpipeOutput,
            Stage::C7, 1U, secondCubeReady) ||
        !FindUniqueLocalOrder(
            localDependencies,
            LocalDependency::CubeToMte1OperandReuse, Stage::C7,
            operandReleased) ||
        !FindUniqueSyncOrder(synchronization, SyncAction::Set,
                             SyncPoint::SlotFree, kOwner, 1U,
                             Stage::C7, Pipe::Fixpipe, slotFree)) {
        return false;
    }
    return CheckStageWaitsBeforeOrder(synchronization, Stage::C7, 1U,
                                      l1Mte2Lock) &&
           rhsReady < l1Mte2Lock && l1Mte2Lock < firstLoad &&
           firstLoad <= lastLoad && lastLoad < l1Mte2Unlock &&
           l1Mte2Unlock < inputReady && inputReady < l1Mte1Lock &&
           l1Mte1Lock < l0Mte1Lock && l0Mte1Lock < l0Mte1Unlock &&
           l0Mte1Unlock < l1Mte1Unlock && l1Mte1Unlock < l0CubeLock &&
           l0CubeLock < lowerCubeLock && lowerCubeLock < firstMmad &&
           firstMmad < lowerCubeUnlock &&
           lowerCubeUnlock < firstCubeReady &&
           firstCubeReady < lowerFixpipeLock &&
           lowerFixpipeLock < firstStore && firstStore < lowerFixpipeUnlock &&
           lowerFixpipeUnlock < upperCubeLock &&
           upperCubeLock < secondMmad && secondMmad < upperCubeUnlock &&
           upperCubeUnlock < l0CubeUnlock &&
           l0CubeUnlock < operandReleased &&
           operandReleased < secondCubeReady &&
           secondCubeReady < upperFixpipeLock &&
           upperFixpipeLock < secondStore &&
           secondStore < upperFixpipeUnlock &&
           upperFixpipeUnlock < slotFree;
}

bool CheckArch35CubeMutexPipelines(
    const OperationTrace &operations, const SyncTrace &synchronization,
    const LocalSyncTrace &localDependencies, const MutexTrace &mutexes,
    std::uint32_t validRows, PrepareAbi abi) noexcept
{
    const bool hasQ10 = validRows > 32U;
    const bool hasBarrier = abi == PrepareAbi::Current &&
                            validRows != 32U && validRows != 64U;
    const std::size_t activeBlocks =
        (static_cast<std::size_t>(validRows) + 15U) / 16U;
    return CheckArch35MutexTrace(mutexes, false) &&
           CheckArch35RetiredLocalSyncPoints(synchronization) &&
           CheckArch35C2MutexPipeline(
               operations, synchronization, localDependencies, mutexes,
               activeBlocks) &&
           CheckArch35C4MutexPipeline(
               operations, synchronization, localDependencies, mutexes,
               hasQ10, hasBarrier, abi) &&
           CheckArch35C5MutexPipeline(
               operations, synchronization, localDependencies, mutexes,
               hasQ10) &&
           CheckArch35C7MutexPipeline(
               operations, synchronization, localDependencies, mutexes);
}

bool CheckArch35CubeOperations(const OperationTrace &operations,
                               const SyncTrace &synchronization,
                               const LocalSyncTrace &localDependencies,
                               const MutexTrace &mutexes,
                               std::uint32_t validRows,
                               const ProposedTilingKey &key) noexcept
{
    const PrepareAbi abi = key.abi;
    if (operations.overflow || synchronization.overflow ||
        localDependencies.overflow ||
        !CheckArch35C4Transfers(operations, localDependencies, mutexes,
                                validRows, abi) ||
        !CheckArch35CubeMutexPipelines(
            operations, synchronization, localDependencies, mutexes,
            validRows, abi) ||
        !CheckArch35MmadDescriptors(operations, validRows, key)) {
        return false;
    }
    const bool hasQ10 = validRows > 32U;
    const std::size_t expectedDependentMmad = hasQ10 ? 1U : 0U;
    const std::size_t activeBlocks =
        (static_cast<std::size_t>(validRows) + 15U) / 16U;
    if (!CheckC2BandPipeline(operations, localDependencies, activeBlocks,
                             true) ||
        !CheckSingleMmadPhase(operations, synchronization,
                              localDependencies, Stage::C4, hasQ10, true) ||
        !CheckC7ComputePipeline(operations, localDependencies, true) ||
        !CheckOperandReleaseOrder(operations, localDependencies,
                                  Stage::C2, activeBlocks) ||
        !CheckOperandReleaseOrder(
            operations, localDependencies, Stage::C4,
            expectedDependentMmad) ||
        !CheckOperandReleaseOrder(
            operations, localDependencies, Stage::C5,
            expectedDependentMmad) ||
        !CheckOperandReleaseOrder(operations, localDependencies, Stage::C7,
                                  1U) ||
        CountOperations(operations, OperationKind::Mmad, Stage::C2) != 0U ||
        CountOperations(operations, OperationKind::MmadRowStackedLhs,
                        Stage::C2) != activeBlocks ||
        CountOperations(operations, OperationKind::Mmad, Stage::C4) !=
            expectedDependentMmad ||
        CountOperations(operations, OperationKind::Mmad, Stage::C5) !=
            expectedDependentMmad ||
        CountOperations(operations, OperationKind::Mmad, Stage::C7) !=
            (hasQ10 ? 0U : 2U) ||
        CountOperations(operations,
                        OperationKind::MmadQuadrantPackedLhs,
                        Stage::C7) != (hasQ10 ? 2U : 0U)) {
        return false;
    }
    if (hasQ10) {
        const OperationRecord *combined = FindUniqueOperation(
            operations, OperationKind::Load, Stage::C7,
            "Kbeta-and-Vbeta", "C7-planar-RHS");
        if (combined == nullptr || combined->source.byteSize != 0x8000U ||
            combined->destination.byteSize != 0x8000U ||
            CountOperations(operations, OperationKind::Load, Stage::C7,
                            "K-beta-g-top32") != 0U ||
            CountOperations(operations, OperationKind::Load, Stage::C7,
                            "V-beta-top32") != 0U) {
            return false;
        }
    } else {
        const OperationRecord *kTop = FindUniqueOperation(
            operations, OperationKind::Load, Stage::C7,
            "K-beta-g-top32", "K-beta-g-top32");
        const OperationRecord *vTop = FindUniqueOperation(
            operations, OperationKind::Load, Stage::C7, "V-beta-top32",
            "V-beta-top32");
        if (kTop == nullptr || vTop == nullptr ||
            kTop->source.byteSize != 0x2000U ||
            kTop->destination.byteSize != 0x2000U ||
            vTop->source.byteSize != 0x2000U ||
            vTop->destination.byteSize != 0x2000U ||
            vTop->source.byteOffset != kTop->source.byteOffset + 0x4000U ||
            vTop->destination.byteOffset !=
                kTop->destination.byteOffset + 0x4000U ||
            CountOperations(operations, OperationKind::Load, Stage::C7,
                            "Kbeta-and-Vbeta") != 0U) {
            return false;
        }
    }
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (record.kind == OperationKind::Mmad &&
            (record.stage == Stage::C4 || record.stage == Stage::C5) &&
            (record.m != 32U || record.n != 32U || record.k != 32U)) {
            return false;
        }
        if (record.kind == OperationKind::Mmad &&
            record.stage == Stage::C7 &&
            (record.m != 32U || record.n != 128U || record.k != 32U)) {
            return false;
        }
        if (record.kind == OperationKind::MmadQuadrantPackedLhs &&
            record.stage == Stage::C7 &&
            (record.quadrantRows != 32U ||
             record.quadrantColumns != 32U || record.n != 128U ||
             record.k != 64U)) {
            return false;
        }
    }

    const std::size_t expectedQ10 = hasQ10 ? 1U : 0U;
    const OperationRecord *residentQ10 = FindUniqueOperation(
        operations, OperationKind::StoreRounded, Stage::C5, nullptr,
        "Akk-q10-cast");
    if ((residentQ10 != nullptr ? 1U : 0U) != expectedQ10 ||
        CountOperations(operations, OperationKind::StoreRounded,
                        Stage::C5, "Y-q10-valid-ld32",
                        "AkkOut-q10-valid-ld64") !=
            (abi == PrepareAbi::Current ? expectedQ10 : 0U)) {
        return false;
    }
    if (hasQ10 && abi == PrepareAbi::Current) {
        const OperationRecord *publicQ10 = FindUniqueOperation(
            operations, OperationKind::StoreRounded, Stage::C5,
            "Y-q10-valid-ld32", "AkkOut-q10-valid-ld64");
        const std::uint32_t bottom = validRows - 32U;
        if (publicQ10 == nullptr ||
            !CheckMatrixSpan(publicQ10->source, bottom, 32U, 32U, 4U) ||
            !CheckMatrixSpan(publicQ10->destination, bottom, 32U, 64U,
                             2U)) {
            return false;
        }
    }

    return true;
}

bool CheckTransferBytes(const OperationRecord *operation,
                        std::size_t expectedBytes) noexcept
{
    return operation != nullptr &&
           operation->source.byteSize == expectedBytes &&
           operation->destination.byteSize == expectedBytes;
}

bool CheckArch22VectorOperations(const OperationTrace &operations,
                                 const SyncTrace &synchronization,
                                 const LocalSyncTrace &localDependencies,
                                 std::uint32_t validRows,
                                 PrepareAbi abi, bool useExp2,
                                 GateStorage gateStorage,
                                 float runtimeScale) noexcept
{
    if (operations.overflow || synchronization.overflow ||
        localDependencies.overflow ||
        !CheckPow2VfDispatch(operations, useExp2) ||
        !CheckRuntimeScaleVfDispatch(operations, abi, runtimeScale) ||
        !CheckVectorStagePipeline(
            operations, synchronization, localDependencies, Stage::V0,
            true, SyncPoint::V0ContextReady, false) ||
        !CheckVectorStagePipeline(
            operations, synchronization, localDependencies, Stage::V1,
            false, SyncPoint::V1ScoreReady, true) ||
        !CheckVectorStagePipeline(
            operations, synchronization, localDependencies, Stage::V3,
            true, SyncPoint::V3LocalSourceFree, false) ||
        !CheckVectorStagePipeline(
            operations, synchronization, localDependencies, Stage::V6,
            true, SyncPoint::V6RhsReady, true)) {
        return false;
    }
    for (Stage stage : {Stage::V0, Stage::V1, Stage::V3, Stage::V6}) {
        if (CountOperations(operations, OperationKind::RunVf, stage) != 1U) {
            return false;
        }
    }

    const bool current = abi == PrepareAbi::Current;
    const bool twoByteGate = IsTwoByteGateStorage(gateStorage);
    const std::size_t gateBytes =
        static_cast<std::size_t>(validRows) * kHeadDimension *
        (twoByteGate ? kElementBytes : sizeof(float));
    const OperationRecord *gateLoad = FindUniqueOperation(
        operations, OperationKind::Load, Stage::V0,
        twoByteGate ? "gate-2b" : "gate-fp32",
        twoByteGate ? "gate-2b" : "gate-fp32-to-G");
    const std::uint64_t expectedGateOffset =
        twoByteGate
            ? arch22_policy::UbPolicy::kPrivate0.offset +
                  arch22_policy::V01PrivateLayout::kGateRaw2B.offset
            : arch22_policy::UbPolicy::kShared.offset +
                  arch22_policy::V01SharedLayout::kGateRawFp32.offset;
    if (gateLoad == nullptr || gateLoad->source.byteSize != gateBytes ||
        gateLoad->destination.byteSize != gateBytes ||
        gateLoad->destination.byteOffset != expectedGateOffset) {
        return false;
    }
    const std::size_t gBytes = static_cast<std::size_t>(validRows) *
                               kHeadDimension * sizeof(float);
    const OperationRecord *gExport = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V0,
        current ? "G-output" : "G-context-source",
        current ? "gk-output" : "G-context");
    const OperationRecord *gReload = FindUniqueOperation(
        operations, OperationKind::Load, Stage::V6,
        current ? "gk-output-reuse" : "G-context", "G");
    if (!CheckTransferBytes(gExport, gBytes) ||
        gExport->source.byteOffset !=
            arch22_policy::UbPolicy::kShared.offset +
                arch22_policy::V01SharedLayout::kG.offset ||
        !CheckTransferBytes(gReload, gBytes)) {
        return false;
    }

    const std::size_t qkBytes = static_cast<std::size_t>(validRows) *
                                kHeadDimension * kElementBytes;
    const std::array<const OperationRecord *, 6U> qkTransfers = {{
        FindUniqueOperation(operations, OperationKind::Load, Stage::V0,
                            "q", "q-to-qhat"),
        FindUniqueOperation(operations, OperationKind::Load, Stage::V0,
                            "k", "k-to-khat"),
        FindUniqueOperation(operations, OperationKind::Store, Stage::V0,
                            "qhat", "qhat-HK-cache"),
        FindUniqueOperation(operations, OperationKind::Store, Stage::V0,
                            "khat", "khat-HK-cache"),
        FindUniqueOperation(operations, OperationKind::Load, Stage::V6,
                            "qhat-HK-cache", "qhat-to-qg"),
        FindUniqueOperation(operations, OperationKind::Load, Stage::V6,
                            "khat-HK-cache", "khat-to-kg"),
    }};
    for (const OperationRecord *transfer : qkTransfers) {
        if (!CheckTransferBytes(transfer, qkBytes)) {
            return false;
        }
    }

    const bool hasQ10 = validRows > 32U;
    const std::uint32_t rhsRows = hasQ10 ? 64U : 32U;
    const std::size_t rhsPlaneBytes =
        static_cast<std::size_t>(rhsRows) * kHeadDimension * kElementBytes;
    const OperationRecord *kBetaStore = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V6, "K-beta-g",
        "K-beta-g-relay");
    const OperationRecord *vBetaStore = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V6, "V-beta",
        "V-beta-relay");
    if (!CheckTransferBytes(kBetaStore, rhsPlaneBytes) ||
        !CheckTransferBytes(vBetaStore, rhsPlaneBytes) ||
        vBetaStore->destination.byteOffset !=
            kBetaStore->destination.byteOffset + 0x4000U) {
        return false;
    }

    constexpr std::uint32_t kQuadrant = 32U;
    const std::uint32_t top =
        validRows < kQuadrant ? validRows : kQuadrant;
    const std::uint32_t bottom =
        hasQ10 ? validRows - kQuadrant : 0U;
    const OperationRecord *q00 = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V3,
        "Akk-q00-row-major", "Akk-q00-relay");
    const OperationRecord *q01 = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V3,
        "Akk-q01-row-major", "Akk-q01-relay");
    const OperationRecord *q11 = FindUniqueOperation(
        operations, OperationKind::Store, Stage::V3,
        "Akk-q11-row-major", "Akk-q11-relay");
    const bool needsQ01 = abi == PrepareAbi::Current || hasQ10;
    if (q00 == nullptr ||
        !CheckMatrixSpan(q00->source, top, kQuadrant, 64U, 2U) ||
        !CheckMatrixSpan(q00->destination, top, kQuadrant, 64U, 2U) ||
        CountOperations(operations, OperationKind::Store, Stage::V3,
                        "Akk-q01-row-major", "Akk-q01-relay") !=
            (needsQ01 ? 1U : 0U) ||
        CountOperations(operations, OperationKind::Store, Stage::V3,
                        "Akk-q11-row-major", "Akk-q11-relay") !=
            (hasQ10 ? 1U : 0U) ||
        (q01 != nullptr) != needsQ01 ||
        (q11 != nullptr) != hasQ10 ||
        CountWritesContaining(operations, Stage::V3, "q10") != 0U) {
        return false;
    }
    if (q01 != nullptr &&
        (!CheckMatrixSpan(q01->source, top, kQuadrant, 64U, 2U) ||
         !CheckMatrixSpan(q01->destination, top, kQuadrant, 64U, 2U))) {
        return false;
    }
    if (q11 != nullptr &&
        (!CheckMatrixSpan(q11->source, bottom, kQuadrant, 64U, 2U) ||
         !CheckMatrixSpan(q11->destination, bottom, kQuadrant, 64U, 2U))) {
        return false;
    }
    const MemorySpace expectedRelaySpace =
        abi == PrepareAbi::Current ? MemorySpace::Gm
                                   : MemorySpace::Workspace;
    return q00->destination.space == expectedRelaySpace &&
           (q01 == nullptr ||
            q01->destination.space == expectedRelaySpace) &&
           (q11 == nullptr ||
            q11->destination.space == expectedRelaySpace);
}

bool CheckArch22C5Contract(const OperationTrace &operations,
                           const SyncTrace &synchronization,
                           const LocalSyncTrace &localDependencies,
                           std::uint32_t validRows,
                           PrepareAbi abi) noexcept
{
    const bool hasQ10 = validRows > 32U;
    const OperationRecord *tLoad = FindUniqueOperation(
        operations, OperationKind::Load, Stage::C5, "T-fp32-relay",
        "T-resident");
    if ((tLoad != nullptr) != hasQ10 ||
        CountLocalDependencies(
            localDependencies, LocalDependency::FixpipeToMte2Relay,
            Stage::C5) != (hasQ10 ? 1U : 0U)) {
        return false;
    }
    if (hasQ10) {
        std::uint64_t relayReady = 0U;
        if (!FindUniqueLocalOrder(
                localDependencies, LocalDependency::FixpipeToMte2Relay,
                Stage::C5, relayReady) ||
            relayReady >= tLoad->order) {
            return false;
        }
    }

    std::size_t q10WriterCount = 0U;
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (!IsWriteOperation(record.kind) ||
            !ContainsName(record.destination.name, "q10")) {
            continue;
        }
        if (record.kind != OperationKind::StoreRounded ||
            record.stage != Stage::C5) {
            return false;
        }
        ++q10WriterCount;
    }
    if (q10WriterCount != (hasQ10 ? 1U : 0U)) {
        return false;
    }

    const OperationRecord *q10 = FindUniqueOperation(
        operations, OperationKind::StoreRounded, Stage::C5,
        "Akk-q10-L0C", "Akk-q10-row-major");
    if ((q10 != nullptr) != hasQ10) {
        return false;
    }
    if (q10 != nullptr) {
        const std::uint32_t bottom = validRows - 32U;
        const MemorySpace expectedRelaySpace =
            abi == PrepareAbi::Current ? MemorySpace::Gm
                                       : MemorySpace::Workspace;
        if (!CheckMatrixSpan(q10->source, bottom, 32U, 32U, 4U) ||
            !CheckMatrixSpan(q10->destination, bottom, 32U, 64U, 2U) ||
            q10->destination.space != expectedRelaySpace) {
            return false;
        }
    }

    std::uint64_t readyOrder = 0U;
    if (!FindUniqueSyncOrder(
            synchronization, SyncAction::Set, SyncPoint::C5AkkReady, 0U,
            0U, Stage::C5, hasQ10 ? Pipe::Fixpipe : Pipe::Control,
            readyOrder)) {
        return false;
    }
    return q10 == nullptr || readyOrder > q10->order;
}

bool CheckArch22C7Contract(const OperationTrace &operations,
                           const LocalSyncTrace &localDependencies,
                           std::uint32_t validRows,
                           const ProposedTilingKey &key) noexcept
{
    const PrepareAbi abi = key.abi;
    const MatrixStorage inputStorage =
        FromInputStorage(key.inputStorage);
    const MatrixStorage valueStorage =
        FromInputStorage(key.valueStorage);
    const bool hasQ10 = validRows > 32U;
    const std::size_t rhsPlaneBytes =
        static_cast<std::size_t>(hasQ10 ? 64U : 32U) *
        kHeadDimension * kElementBytes;
    const OperationRecord *kBetaLoad = FindUniqueOperation(
        operations, OperationKind::Load, Stage::C7, "K-beta-g-relay",
        "K-beta-g-L1");
    const OperationRecord *vBetaLoad = FindUniqueOperation(
        operations, OperationKind::Load, Stage::C7, "V-beta-relay",
        "V-beta-L1");
    if (!CheckTransferBytes(kBetaLoad, rhsPlaneBytes) ||
        !CheckTransferBytes(vBetaLoad, rhsPlaneBytes) ||
        kBetaLoad->source.space != MemorySpace::Workspace ||
        vBetaLoad->source.space != MemorySpace::Workspace ||
        kBetaLoad->destination.space != MemorySpace::L1 ||
        vBetaLoad->destination.space != MemorySpace::L1 ||
        vBetaLoad->source.byteOffset !=
            kBetaLoad->source.byteOffset + 0x4000U ||
        vBetaLoad->destination.byteOffset !=
            kBetaLoad->destination.byteOffset + 0x4000U) {
        return false;
    }
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (record.kind == OperationKind::Load &&
            record.stage == Stage::C7 &&
            ContainsName(record.source.name, "q01")) {
            return false;
        }
    }

    const OperationRecord *akkFill = nullptr;
    const OperationRecord *akkLoad = nullptr;
    const char *lhsName = nullptr;
    const MemorySpace expectedRelaySpace =
        abi == PrepareAbi::Current ? MemorySpace::Gm
                                   : MemorySpace::Workspace;
    if (hasQ10) {
        akkFill = FindUniqueOperation(
            operations, OperationKind::Fill, Stage::C7, nullptr,
            "Akk-cube-ready-resident");
        akkLoad = FindUniqueOperation(
            operations, OperationKind::Load, Stage::C7, nullptr,
            "Akk-cube-ready-resident");
        lhsName = "Akk-cube-ready-resident";
        const char *expectedSourceName =
            abi == PrepareAbi::Current ? "Akk-output-and-C7-relay"
                                       : "Akk-row-major-relay-valid";
        if (akkFill == nullptr || akkLoad == nullptr ||
            akkFill->destination.byteSize != 0x2000U ||
            !SameName(akkLoad->source.name, expectedSourceName) ||
            !CheckMatrixSpan(akkLoad->source, validRows, 64U, 64U, 2U) ||
            akkLoad->source.space != expectedRelaySpace ||
            akkLoad->destination.byteSize != 0x2000U ||
            akkLoad->destination.space != MemorySpace::L1 ||
            CountOperations(operations, OperationKind::Load, Stage::C7,
                            nullptr, "Akk-q00-valid-tight") != 0U) {
            return false;
        }
    } else {
        akkFill = FindUniqueOperation(
            operations, OperationKind::Fill, Stage::C7, nullptr,
            "Akk-q00-tight-cube-ready");
        akkLoad = FindUniqueOperation(
            operations, OperationKind::Load, Stage::C7,
            "Akk-q00-relay-ld64", "Akk-q00-valid-tight");
        lhsName = "Akk-q00-tight-cube-ready";
        if (akkFill == nullptr || akkLoad == nullptr ||
            !CheckMatrixSpan(akkFill->destination, 32U, 32U, 32U, 2U) ||
            !CheckMatrixSpan(akkLoad->source, validRows, 32U, 64U, 2U) ||
            !CheckMatrixSpan(akkLoad->destination, validRows, 32U, 32U,
                             2U) ||
            akkLoad->source.space != expectedRelaySpace ||
            akkLoad->destination.space != MemorySpace::L1 ||
            CountOperations(operations, OperationKind::Load, Stage::C7,
                            nullptr, "Akk-cube-ready-resident") != 0U) {
            return false;
        }
    }

    std::uint64_t fillToLoadOrder = 0U;
    if (CountLocalDependencies(localDependencies,
                               LocalDependency::Mte2FillToLoadWaw,
                               Stage::C7) != 1U ||
        !FindUniqueLocalOrder(localDependencies,
                              LocalDependency::Mte2FillToLoadWaw,
                              Stage::C7, fillToLoadOrder) ||
        akkFill->order >= fillToLoadOrder ||
        fillToLoadOrder >= akkLoad->order) {
        return false;
    }
    std::uint64_t fixpipeToLoadOrder = 0U;
    const std::size_t expectedFixpipeRelay = hasQ10 ? 1U : 0U;
    if (CountLocalDependencies(localDependencies,
                               LocalDependency::FixpipeToMte2Relay,
                               Stage::C7) != expectedFixpipeRelay) {
        return false;
    }
    if (hasQ10 &&
        (!FindUniqueLocalOrder(localDependencies,
                               LocalDependency::FixpipeToMte2Relay,
                               Stage::C7, fixpipeToLoadOrder) ||
         fillToLoadOrder >= fixpipeToLoadOrder ||
         fixpipeToLoadOrder >= akkLoad->order)) {
        return false;
    }

    const char *kRhsName = hasQ10 ? "K-beta-g-L1" : "K-beta-g-top32";
    const char *vRhsName = hasQ10 ? "V-beta-L1" : "V-beta-top32";
    const OperationRecord *w = FindUniqueOperation(
        operations, OperationKind::Mmad, Stage::C7, lhsName, "W-L0C");
    const OperationRecord *u = FindUniqueOperation(
        operations, OperationKind::Mmad, Stage::C7, lhsName, "U-L0C");
    const std::uint32_t m = hasQ10 ? 64U : 32U;
    const std::uint32_t k = hasQ10 ? 64U : 32U;
    return w != nullptr && u != nullptr &&
           CheckStandardMmad(*w, Stage::C7, lhsName, kRhsName, "W-L0C",
                             inputStorage, inputStorage, m,
                             128U, k, false, false) &&
           CheckStandardMmad(*u, Stage::C7, lhsName, vRhsName, "U-L0C",
                             inputStorage, valueStorage, m,
                             128U, k, false, false) &&
           akkLoad->order < w->order && akkLoad->order < u->order &&
           kBetaLoad->order < w->order && vBetaLoad->order < u->order;
}

bool CheckArch22CubeOperations(const OperationTrace &operations,
                               const SyncTrace &synchronization,
                               const LocalSyncTrace &localDependencies,
                               std::uint32_t validRows,
                               const ProposedTilingKey &key) noexcept
{
    const PrepareAbi abi = key.abi;
    if (operations.overflow || synchronization.overflow ||
        localDependencies.overflow ||
        !CheckC2RowStackedMmadDescriptors(
            operations, validRows, "Aqk-Akk-stacked-L0C",
            NativeMatrixLayout::L0aZZ, Architecture::Arch22,
            FromScoreStorage(key.scoreStorage))) {
        return false;
    }
    const bool hasQ10 = validRows > 32U;
    const std::size_t activeBlocks =
        (static_cast<std::size_t>(validRows) + 15U) / 16U;
    const std::size_t dependentMmad = hasQ10 ? 1U : 0U;
    if (!CheckC2BandPipeline(operations, localDependencies, activeBlocks,
                             false) ||
        !CheckSingleMmadPhase(operations, synchronization,
                              localDependencies, Stage::C4, hasQ10, true) ||
        !CheckSingleMmadPhase(operations, synchronization,
                              localDependencies, Stage::C5, hasQ10, true) ||
        !CheckC7ComputePipeline(operations, localDependencies, false) ||
        CountOperations(operations, OperationKind::Mmad, Stage::C2) != 0U ||
        CountOperations(operations, OperationKind::MmadRowStackedLhs,
                        Stage::C2) != activeBlocks ||
        CountOperations(operations, OperationKind::Mmad, Stage::C4) !=
            dependentMmad ||
        CountOperations(operations, OperationKind::Mmad, Stage::C5) !=
            dependentMmad ||
        CountOperations(operations, OperationKind::Mmad, Stage::C7) != 2U ||
        CountOperations(operations,
                        OperationKind::MmadQuadrantPackedLhs,
                        Stage::C7) != 0U ||
        !CheckOperandReleaseOrder(operations, localDependencies, Stage::C2,
                                  activeBlocks) ||
        !CheckOperandReleaseOrder(operations, localDependencies, Stage::C4,
                                  dependentMmad) ||
        !CheckOperandReleaseOrder(operations, localDependencies, Stage::C5,
                                  dependentMmad) ||
        !CheckOperandReleaseOrder(operations, localDependencies, Stage::C7,
                                  1U) ||
        !CheckArch22C5Contract(operations, synchronization,
                               localDependencies, validRows, abi) ||
        !CheckArch22C7Contract(operations, localDependencies, validRows,
                               key)) {
        return false;
    }
    return true;
}

struct MultiHeadAddressTrace {
    std::array<OperationTrace, kAivPerWorkgroup> aivOperations{};
    std::array<SyncTrace, kAivPerWorkgroup> aivSynchronization{};
    std::array<MutexTrace, kAivPerWorkgroup> aivMutexes{};
    OperationTrace aicOperations{};
    SyncTrace aicSynchronization{};
    LocalSyncTrace aicLocalDependencies{};
    MutexTrace aicMutexes{};
};

std::size_t CountSyncRecords(const SyncTrace &trace, SyncAction action,
                             SyncPoint point, std::uint32_t ownerId,
                             std::uint64_t generation, Stage stage,
                             Pipe pipe) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const SyncRecord &record = trace.records[index];
        if (record.action == action && record.point == point &&
            record.ownerId == ownerId && record.generation == generation &&
            record.aivId == kNoAivId && record.hasActiveHead &&
            record.stage == stage && record.pipe == pipe) {
            ++count;
        }
    }
    return count;
}

std::size_t CountSyncRecordsBefore(const SyncTrace &trace,
                                   SyncAction action, SyncPoint point,
                                   std::uint32_t ownerId,
                                   std::uint64_t generation, Stage stage,
                                   Pipe pipe,
                                   std::uint64_t beforeOrder) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const SyncRecord &record = trace.records[index];
        if (record.action == action && record.point == point &&
            record.ownerId == ownerId && record.generation == generation &&
            record.aivId == kNoAivId && record.hasActiveHead &&
            record.stage == stage && record.pipe == pipe &&
            record.order < beforeOrder) {
            ++count;
        }
    }
    return count;
}

std::size_t CountAivSyncRecords(const MultiHeadAddressTrace &trace,
                                SyncAction action, SyncPoint point,
                                std::uint32_t ownerId,
                                std::uint64_t generation, Stage stage,
                                Pipe pipe) noexcept
{
    std::size_t count = 0U;
    for (const SyncTrace &aivTrace : trace.aivSynchronization) {
        count += CountSyncRecords(aivTrace, action, point, ownerId,
                                  generation, stage, pipe);
    }
    return count;
}

std::size_t CountAivPairSyncRecords(
    const MultiHeadAddressTrace &trace, SyncAction action, SyncPoint point,
    std::uint32_t pair, std::uint64_t generation, std::uint32_t aivId,
    Stage stage, Pipe pipe) noexcept
{
    if (aivId >= kAivPerWorkgroup) {
        return 0U;
    }
    std::size_t count = 0U;
    const SyncTrace &aivTrace = trace.aivSynchronization[aivId];
    for (std::size_t index = 0U; index < aivTrace.size; ++index) {
        const SyncRecord &record = aivTrace.records[index];
        if (record.action == action && record.point == point &&
            record.ownerId == pair && record.generation == generation &&
            record.aivId == aivId && record.stage == stage &&
            record.pipe == pipe) {
            ++count;
        }
    }
    return count;
}

std::size_t CountC7ReadyWaitsBefore(const SyncTrace &trace,
                                    Architecture architecture,
                                    std::uint64_t beforeOrder) noexcept
{
    const SyncAction expectedAction =
        architecture == Architecture::Arch22 ? SyncAction::AicWait
                                             : SyncAction::Wait;
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < trace.size; ++index) {
        const SyncRecord &record = trace.records[index];
        if (record.action == expectedAction &&
            record.point == SyncPoint::V6RhsReady &&
            record.aivId == kNoAivId && record.hasActiveHead &&
            record.stage == Stage::C7 && record.pipe == Pipe::Mte2 &&
            record.order < beforeOrder) {
            ++count;
        }
    }
    return count;
}

bool CheckC7ReadyWaitSources(const MultiHeadAddressTrace &trace,
                             Architecture architecture) noexcept
{
    for (std::size_t index = 0U; index < trace.aicSynchronization.size;
         ++index) {
        const SyncRecord &wait = trace.aicSynchronization.records[index];
        const SyncAction expectedAction =
            architecture == Architecture::Arch22 ? SyncAction::AicWait
                                                 : SyncAction::Wait;
        if (wait.action != expectedAction ||
            wait.point != SyncPoint::V6RhsReady ||
            wait.stage != Stage::C7 || wait.pipe != Pipe::Mte2) {
            continue;
        }
        if (architecture == Architecture::Arch22) {
            for (std::uint32_t aivId = 0U; aivId < kAivPerWorkgroup;
                 ++aivId) {
                if (CountAivPairSyncRecords(
                        trace, SyncAction::AivArrive,
                        SyncPoint::V6RhsReady, wait.ownerId,
                        wait.generation, aivId, Stage::V6,
                        Pipe::Mte3) != 1U) {
                    return false;
                }
            }
        } else if (CountAivSyncRecords(
                       trace, SyncAction::Set, SyncPoint::V6RhsReady,
                       wait.ownerId, wait.generation, Stage::V6,
                       Pipe::Mte3) != 1U) {
            return false;
        }
    }
    return true;
}

template <std::size_t N>
bool CollectExpectedHeads(Architecture architecture,
                          const RuntimeTiling &tiling,
                          std::array<HeadTask, N> &heads,
                          std::array<bool, N> &seen) noexcept
{
    return ReplayItems(
        architecture, tiling, 0U, ResolveDense,
        [&heads, &seen](const WorkItem &item) noexcept {
            for (const HeadTask &head : item.group.heads) {
                if (!head.active) {
                    continue;
                }
                if (head.headId >= N || seen[head.headId]) {
                    return false;
                }
                heads[head.headId] = head;
                seen[head.headId] = true;
            }
            return true;
        });
}

bool CaptureMultiHeadAddressTrace(Architecture architecture,
                                  std::uint32_t heads,
                                  MultiHeadAddressTrace &trace,
                                  std::uint32_t qkHeads = 0U,
                                  ProposedTilingKey key = {}) noexcept
{
    RuntimeTiling tiling{};
    tiling.totalChunks = 1U;
    tiling.headCount = heads;
    tiling.qkHeadCount = qkHeads;
    tiling.aicWorkgroupCount = 1U;
    tiling.epsilon = 1.0e-6F;
    tiling.key = key;
    const WorkspaceSizing sizing =
        CheckedWorkspaceSizing(architecture, 1U, 0U);
    if (!sizing.valid) {
        return false;
    }

    CubeOps unusedCubeOps{};
    for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
        TraceClock clock{};
        VectorOps vectorOps{};
        vectorOps.trace = &trace.aivOperations[aiv];
        vectorOps.clock = &clock;
        SyncLedger sync{&trace.aivSynchronization[aiv], nullptr, &clock,
                        &trace.aivMutexes[aiv]};
        WorkspaceView workspace{};
        workspace.backingBytes = sizing.totalBytes;
        RunArchitectureContract(
            architecture, tiling, 0U, CoreRole::Aiv, aiv, workspace, sync,
            vectorOps, unusedCubeOps, ResolveDense);
    }

    TraceClock clock{};
    CubeOps cubeOps{};
    cubeOps.trace = &trace.aicOperations;
    cubeOps.clock = &clock;
    VectorOps unusedVectorOps{};
    SyncLedger sync{&trace.aicSynchronization,
                    &trace.aicLocalDependencies, &clock,
                    &trace.aicMutexes};
    WorkspaceView workspace{};
    workspace.backingBytes = sizing.totalBytes;
    RunArchitectureContract(
        architecture, tiling, 0U, CoreRole::Aic, 0U, workspace, sync,
        unusedVectorOps, cubeOps, ResolveDense);

    if (trace.aicOperations.overflow || trace.aicSynchronization.overflow ||
        trace.aicLocalDependencies.overflow || trace.aicMutexes.overflow) {
        return false;
    }
    for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
        if (trace.aivOperations[aiv].overflow ||
            trace.aivSynchronization[aiv].overflow ||
            trace.aivMutexes[aiv].overflow) {
            return false;
        }
    }
    if (architecture == Architecture::Arch22) {
        if (!EmptyMutexTrace(trace.aicMutexes)) {
            return false;
        }
        for (const MutexTrace &mutexes : trace.aivMutexes) {
            if (!EmptyMutexTrace(mutexes)) {
                return false;
            }
        }
        return true;
    }
    if (!CheckArch35MutexTrace(trace.aicMutexes, false) ||
        !CheckArch35RetiredLocalSyncPoints(trace.aicSynchronization)) {
        return false;
    }
    for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
        if (!CheckArch35MutexTrace(trace.aivMutexes[aiv], true, aiv) ||
            !CheckArch35RetiredLocalSyncPoints(
                trace.aivSynchronization[aiv])) {
            return false;
        }
    }
    return true;
}

bool IsMmadOperation(const OperationRecord &record) noexcept
{
    return record.kind == OperationKind::Mmad ||
           record.kind == OperationKind::MmadRowStackedLhs ||
           record.kind == OperationKind::MmadQuadrantPackedLhs;
}

bool AreDisjointPhysicalSpans(const BufferSpan &lhs,
                              const BufferSpan &rhs) noexcept
{
    if (lhs.space != rhs.space || lhs.slot != rhs.slot ||
        lhs.generation != rhs.generation ||
        lhs.ownerRole != rhs.ownerRole || lhs.ownerId != rhs.ownerId) {
        return false;
    }
    const std::uint64_t lhsEnd = lhs.byteOffset + lhs.byteSize;
    const std::uint64_t rhsEnd = rhs.byteOffset + rhs.byteSize;
    return lhsEnd <= rhs.byteOffset || rhsEnd <= lhs.byteOffset;
}

Stage StageForL0OperandUse(L0OperandUse use) noexcept
{
    if (use <= L0OperandUse::C2Band3) {
        return Stage::C2;
    }
    if (use == L0OperandUse::C4) {
        return Stage::C4;
    }
    if (use == L0OperandUse::C5) {
        return Stage::C5;
    }
    return Stage::C7;
}

bool CheckL0OperandUseConsumers(const OperationTrace &operations,
                                const HeadTask &head,
                                L0OperandUse use) noexcept
{
    const Stage stage = StageForL0OperandUse(use);
    const std::uint64_t generation = L0OperandGenerationFor(head, use);
    std::array<const OperationRecord *, 2U> consumers{};
    std::size_t consumerCount = 0U;
    for (std::size_t index = 0U; index < operations.size; ++index) {
        const OperationRecord &record = operations.records[index];
        if (!IsMmadOperation(record) || record.stage != stage ||
            record.headId != head.headId ||
            record.l0aOperand.slot != head.l0OperandBankId ||
            record.l0bOperand.slot != head.l0OperandBankId ||
            record.l0aOperand.generation != generation ||
            record.l0bOperand.generation != generation) {
            continue;
        }
        if (consumerCount == consumers.size() ||
            record.l0aOperand.space != MemorySpace::L0A ||
            record.l0bOperand.space != MemorySpace::L0B ||
            record.l0aOperand.logicalHeadId != head.headId ||
            record.l0bOperand.logicalHeadId != head.headId ||
            record.l0aOperand.ownerRole != CoreRole::Aic ||
            record.l0bOperand.ownerRole != CoreRole::Aic ||
            record.l0aOperand.ownerId != 0U ||
            record.l0bOperand.ownerId != 0U) {
            return false;
        }
        consumers[consumerCount++] = &record;
    }

    const bool twoConsumers = stage == Stage::C7;
    if (consumerCount != (twoConsumers ? 2U : 1U)) {
        return false;
    }
    if (stage == Stage::C2) {
        const OperationRecord &packed = *consumers[0];
        return packed.kind == OperationKind::MmadRowStackedLhs &&
               SameName(packed.l0aOperand.name,
                        "C2-QK-stacked-L0A") &&
               packed.l0aOperand.byteSize == 0x2000U &&
               SameName(packed.l0bOperand.name, "C2-Kminus-L0B");
    }
    if (stage == Stage::C4) {
        return SameName(consumers[0]->l0aOperand.name, "C4-B-L0A") &&
               SameName(consumers[0]->l0bOperand.name, "C4-X0-L0B");
    }
    if (stage == Stage::C5) {
        return SameName(consumers[0]->l0aOperand.name, "C5-X1-L0A") &&
               SameName(consumers[0]->l0bOperand.name, "C5-T-L0B");
    }

    const OperationRecord *w = nullptr;
    const OperationRecord *u = nullptr;
    for (std::size_t index = 0U; index < consumerCount; ++index) {
        if (SameName(consumers[index]->l0bOperand.name,
                     "C7-Kbeta-L0B")) {
            w = consumers[index];
        } else if (SameName(consumers[index]->l0bOperand.name,
                            "C7-Vbeta-L0B")) {
            u = consumers[index];
        }
    }
    return w != nullptr && u != nullptr &&
           SameName(w->l0aOperand.name, "C7-Akk-L0A") &&
           SameName(u->l0aOperand.name, "C7-Akk-L0A") &&
           SamePhysicalSpan(w->l0aOperand, u->l0aOperand) &&
           AreDisjointPhysicalSpans(w->l0bOperand, u->l0bOperand);
}

struct L0OperandEpochObservation {
    Stage stage = Stage::C2;
    std::uint32_t headId = 0U;
    std::uint32_t bank = 0U;
    std::uint64_t generation = 0U;
    std::uint64_t firstConsumerOrder = 0U;
    std::uint64_t lastConsumerOrder = 0U;
    std::size_t consumerCount = 0U;
};

bool CheckL0OperandEpochContract(Architecture architecture) noexcept
{
    constexpr std::uint32_t kValueHeads = 8U;
    constexpr std::array<L0OperandUse, kL0OperandUseCount> kUses = {{
        L0OperandUse::C2Band0,
        L0OperandUse::C2Band1,
        L0OperandUse::C2Band2,
        L0OperandUse::C2Band3,
        L0OperandUse::C4,
        L0OperandUse::C5,
        L0OperandUse::C7,
    }};
    MultiHeadAddressTrace trace{};
    if (!CaptureMultiHeadAddressTrace(architecture, kValueHeads, trace)) {
        return false;
    }

    RuntimeTiling tiling{};
    tiling.totalChunks = 1U;
    tiling.headCount = kValueHeads;
    tiling.aicWorkgroupCount = 1U;
    tiling.epsilon = 1.0e-6F;
    std::array<HeadTask, kValueHeads> heads{};
    std::array<bool, kValueHeads> seen{};
    if (!CollectExpectedHeads(architecture, tiling, heads, seen)) {
        return false;
    }
    for (std::uint32_t headId = 0U; headId < kValueHeads; ++headId) {
        const HeadTask &head = heads[headId];
        const std::uint32_t expectedBank =
            architecture == Architecture::Arch22
                ? head.groupLocalHead % kAivPerWorkgroup
                : 0U;
        if (!seen[headId] || head.l0OperandBankId != expectedBank) {
            return false;
        }
        for (L0OperandUse use : kUses) {
            if (!CheckL0OperandUseConsumers(trace.aicOperations, head,
                                            use)) {
                return false;
            }
        }
    }

    constexpr std::size_t kMaxEpochs =
        kValueHeads * kL0OperandUseCount;
    std::array<L0OperandEpochObservation, kMaxEpochs> epochs{};
    std::array<std::uint64_t, kMaxL0OperandBankCount> nextGeneration{};
    std::array<bool, kMaxL0OperandBankCount> bankSeen{};
    std::size_t epochCount = 0U;
    for (std::size_t index = 0U; index < trace.aicOperations.size; ++index) {
        const OperationRecord &record = trace.aicOperations.records[index];
        if (!IsMmadOperation(record)) {
            continue;
        }
        if (record.l0aOperand.slot != record.l0bOperand.slot ||
            record.l0aOperand.generation != record.l0bOperand.generation ||
            record.l0aOperand.slot >= kMaxL0OperandBankCount) {
            return false;
        }
        const std::uint32_t bank = record.l0aOperand.slot;
        const std::uint64_t generation = record.l0aOperand.generation;
        if (epochCount != 0U) {
            L0OperandEpochObservation &previous = epochs[epochCount - 1U];
            if (previous.stage == record.stage &&
                previous.headId == record.headId &&
                previous.bank == bank &&
                previous.generation == generation) {
                previous.lastConsumerOrder = record.order;
                ++previous.consumerCount;
                continue;
            }
        }
        if (epochCount == epochs.size() ||
            generation != nextGeneration[bank]) {
            return false;
        }
        ++nextGeneration[bank];
        bankSeen[bank] = true;
        epochs[epochCount++] = {record.stage, record.headId, bank,
                                generation, record.order, record.order, 1U};
    }

    std::array<const LocalSyncRecord *, kMaxEpochs> releases{};
    std::size_t releaseCount = 0U;
    for (std::size_t index = 0U; index < trace.aicLocalDependencies.size;
         ++index) {
        const LocalSyncRecord &record =
            trace.aicLocalDependencies.records[index];
        if (record.dependency !=
            LocalDependency::CubeToMte1OperandReuse) {
            continue;
        }
        if (releaseCount == releases.size()) {
            return false;
        }
        releases[releaseCount++] = &record;
    }
    if (epochCount != kMaxEpochs || releaseCount != epochCount ||
        !bankSeen[0] ||
        (architecture == Architecture::Arch22
             ? !bankSeen[1] || nextGeneration[0] != kMaxEpochs / 2U ||
                   nextGeneration[1] != kMaxEpochs / 2U
             : bankSeen[1] || nextGeneration[0] != kMaxEpochs)) {
        return false;
    }
    for (std::size_t index = 0U; index < epochCount; ++index) {
        const std::size_t expectedConsumers =
            epochs[index].stage == Stage::C7 ? 2U : 1U;
        if (epochs[index].consumerCount != expectedConsumers ||
            releases[index]->stage != epochs[index].stage ||
            releases[index]->order <= epochs[index].lastConsumerOrder ||
            (index + 1U < epochCount &&
             releases[index]->order >=
                 epochs[index + 1U].firstConsumerOrder)) {
            return false;
        }
    }
    return true;
}

std::size_t CountAivOperations(
    const MultiHeadAddressTrace &trace, OperationKind kind, Stage stage,
    const char *sourceName = nullptr,
    const char *destinationName = nullptr) noexcept
{
    std::size_t count = 0U;
    for (const OperationTrace &operations : trace.aivOperations) {
        count += CountOperations(operations, kind, stage, sourceName,
                                 destinationName);
    }
    return count;
}

std::size_t CountAivRunVfForHead(const MultiHeadAddressTrace &trace,
                                 Stage stage,
                                 std::uint32_t headId) noexcept
{
    std::size_t count = 0U;
    for (const OperationTrace &operations : trace.aivOperations) {
        for (std::size_t index = 0U; index < operations.size; ++index) {
            const OperationRecord &record = operations.records[index];
            if (record.kind == OperationKind::RunVf &&
                record.stage == stage && record.headId == headId) {
                ++count;
            }
        }
    }
    return count;
}

bool CheckCompleteQkCohortPartition(Architecture architecture,
                                    std::uint32_t valueHeads,
                                    std::uint32_t qkHeads,
                                    std::uint32_t ratio) noexcept
{
    constexpr std::uint32_t kWorkgroups = 8U;
    constexpr std::size_t kMaxValueHeads = 16U;
    if (valueHeads > kMaxValueHeads || qkHeads > kMaxValueHeads ||
        valueHeads != qkHeads * ratio ||
        (ratio != 1U && ratio != 2U && ratio != 3U && ratio != 4U &&
         ratio != 8U)) {
        return false;
    }

    RuntimeTiling tiling{};
    tiling.totalChunks = 1U;
    tiling.headCount = valueHeads;
    tiling.qkHeadCount = qkHeads;
    tiling.aicWorkgroupCount = kWorkgroups;
    tiling.epsilon = 1.0e-6F;
    const std::uint32_t expectedPartitions =
        HeadPartitionCount(valueHeads, qkHeads);
    const std::uint32_t expectedGroups =
        EmittedHeadGroupCount(valueHeads, qkHeads);

    std::array<std::uint32_t, kMaxValueHeads> qkWorkgroup{};
    std::array<std::uint32_t, kMaxValueHeads> runtimeWorkgroup{};
    std::array<std::uint32_t, kMaxValueHeads> planVisits{};
    std::array<std::uint32_t, kMaxValueHeads> runtimeVisits{};
    std::array<std::uint32_t, kMaxValueHeads> groupVisits{};
    for (std::uint32_t &workgroup : qkWorkgroup) {
        workgroup = kNoAivId;
    }
    for (std::uint32_t &workgroup : runtimeWorkgroup) {
        workgroup = kNoAivId;
    }

    for (std::uint32_t workgroup = 0U; workgroup < kWorkgroups;
         ++workgroup) {
        const CorePlan plan =
            BuildCorePlan(tiling, workgroup, architecture);
        if (plan.mode != PartitionMode::ChunkHeadGroup ||
            plan.headPartitionCount != expectedPartitions ||
            plan.headGroupCount != expectedGroups ||
            plan.primaryWorkItemCount != expectedPartitions) {
            return false;
        }
        for (std::uint64_t ordinal = plan.begin; ordinal < plan.end;
             ++ordinal) {
            std::uint32_t chunkOrdinal = 0U;
            std::uint32_t partitionOrdinal = 0U;
            DecodeChunkHeadPartitionOrdinal(plan, ordinal, chunkOrdinal,
                                            partitionOrdinal);
            if (chunkOrdinal != 0U ||
                partitionOrdinal >= expectedPartitions) {
                return false;
            }
            const std::uint32_t headBegin = PartitionValueHeadBegin(
                partitionOrdinal, valueHeads, qkHeads);
            const std::uint32_t headEnd = PartitionValueHeadEnd(
                partitionOrdinal, valueHeads, qkHeads);
            if (headBegin >= headEnd || headEnd > valueHeads ||
                headBegin % ratio != 0U || headEnd % ratio != 0U) {
                return false;
            }
            for (std::uint32_t headId = headBegin; headId < headEnd;
                 ++headId) {
                const std::uint32_t qkHeadId = headId / ratio;
                if (++planVisits[headId] != 1U) {
                    return false;
                }
                if (qkWorkgroup[qkHeadId] == kNoAivId) {
                    qkWorkgroup[qkHeadId] = workgroup;
                } else if (qkWorkgroup[qkHeadId] != workgroup) {
                    return false;
                }
            }
        }

        if (!ReplayItems(
                architecture, tiling, workgroup, ResolveDense,
                [ratio, expectedGroups, &qkWorkgroup, &groupVisits,
                 workgroup](const WorkItem &item) noexcept {
                    if (item.group.headGroupId >= expectedGroups ||
                        ++groupVisits[item.group.headGroupId] != 1U) {
                        return false;
                    }
                    for (const HeadTask &head : item.group.heads) {
                        if (!head.active) {
                            continue;
                        }
                        const std::uint32_t expectedOwner =
                            head.qkHeadId * ratio;
                        if (head.qkHeadId != head.headId / ratio ||
                            head.qkOwner != (head.headId == expectedOwner) ||
                            head.qkLastConsumer !=
                                (head.headId + 1U == expectedOwner + ratio) ||
                            head.qkCacheGeneration ==
                                std::numeric_limits<std::uint64_t>::max() ||
                            qkWorkgroup[head.qkHeadId] != workgroup) {
                            return false;
                        }
                    }
                    return true;
                })) {
            return false;
        }

        const WorkspaceSizing sizing = CheckedWorkspaceSizing(
            architecture, kWorkgroups, workgroup);
        if (!sizing.valid) {
            return false;
        }
        CubeOps unusedCubeOps{};
        for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
            OperationTrace operations{};
            MutexTrace mutexes{};
            TraceClock clock{};
            VectorOps vectorOps{};
            vectorOps.trace = &operations;
            vectorOps.clock = &clock;
            SyncLedger sync{nullptr, nullptr, &clock, &mutexes};
            WorkspaceView workspace{};
            workspace.backingBytes = sizing.totalBytes;
            RunArchitectureContract(
                architecture, tiling, workgroup, CoreRole::Aiv, aiv,
                workspace, sync, vectorOps, unusedCubeOps, ResolveDense);
            if (operations.overflow || mutexes.overflow ||
                (architecture == Architecture::Arch22
                     ? !EmptyMutexTrace(mutexes)
                     : !CheckArch35MutexTrace(mutexes, true, aiv))) {
                return false;
            }
            for (std::size_t index = 0U; index < operations.size; ++index) {
                const OperationRecord &record = operations.records[index];
                if (record.kind != OperationKind::RunVf ||
                    record.stage != Stage::V0) {
                    continue;
                }
                if (record.headId >= valueHeads ||
                    ++runtimeVisits[record.headId] != 1U) {
                    return false;
                }
                runtimeWorkgroup[record.headId] = workgroup;
            }
        }
    }

    for (std::uint32_t qkHeadId = 0U; qkHeadId < qkHeads; ++qkHeadId) {
        if (qkWorkgroup[qkHeadId] == kNoAivId) {
            return false;
        }
    }
    for (std::uint32_t groupId = 0U; groupId < expectedGroups; ++groupId) {
        if (groupVisits[groupId] != 1U) {
            return false;
        }
    }
    for (std::uint32_t headId = 0U; headId < valueHeads; ++headId) {
        if (planVisits[headId] != 1U || runtimeVisits[headId] != 1U ||
            runtimeWorkgroup[headId] != qkWorkgroup[headId / ratio]) {
            return false;
        }
    }
    return true;
}

bool CheckCompleteQkCohortPartitions() noexcept
{
    struct CohortCase {
        std::uint32_t valueHeads;
        std::uint32_t qkHeads;
        std::uint32_t ratio;
    };
    constexpr std::array<CohortCase, 5U> kCases = {{{8U, 8U, 1U},
                                                    {8U, 4U, 2U},
                                                    {6U, 2U, 3U},
                                                    {8U, 2U, 4U},
                                                    {16U, 2U, 8U}}};
    for (Architecture architecture : {Architecture::Arch22,
                                      Architecture::Arch35}) {
        for (const CohortCase &cohort : kCases) {
            if (!CheckCompleteQkCohortPartition(
                    architecture, cohort.valueHeads, cohort.qkHeads,
                    cohort.ratio)) {
                return false;
            }
        }
    }
    return true;
}

bool TouchesInactiveArch22Head(const MultiHeadAddressTrace &trace,
                               const HeadTask &inactiveHead) noexcept
{
    for (const OperationTrace &operations : trace.aivOperations) {
        for (std::size_t index = 0U; index < operations.size; ++index) {
            const OperationRecord &record = operations.records[index];
            if (record.kind == OperationKind::RunVf &&
                record.headId == inactiveHead.headId) {
                return true;
            }
            const std::array<const BufferSpan *, 3U> spans = {{
                &record.source, &record.rhsOperand, &record.destination,
            }};
            for (const BufferSpan *span : spans) {
                if (span->name == nullptr) {
                    continue;
                }
                if ((span->space == MemorySpace::Ub &&
                     span->slot == inactiveHead.localBankId) ||
                    ((span->space == MemorySpace::Gm ||
                      span->space == MemorySpace::Workspace) &&
                     span->slot == inactiveHead.workspaceSlot)) {
                    return true;
                }
            }
        }
    }
    for (std::size_t index = 0U; index < trace.aicOperations.size; ++index) {
        const OperationRecord &record = trace.aicOperations.records[index];
        const std::array<const BufferSpan *, 3U> spans = {{
            &record.source, &record.rhsOperand, &record.destination,
        }};
        for (const BufferSpan *span : spans) {
            if (span->name == nullptr) {
                continue;
            }
            if ((span->space == MemorySpace::L1 &&
                 span->slot == inactiveHead.l1BankId) ||
                ((span->space == MemorySpace::Gm ||
                  span->space == MemorySpace::Workspace) &&
                 span->slot == inactiveHead.workspaceSlot)) {
                return true;
            }
        }
    }
    return false;
}

bool CheckArch22MultiHeadAddresses(std::uint32_t heads) noexcept
{
    if (heads != 3U && heads != 4U) {
        return false;
    }
    MultiHeadAddressTrace trace{};
    if (!CaptureMultiHeadAddressTrace(Architecture::Arch22, heads, trace)) {
        return false;
    }
    OwnerTicketState tickets{};
    const HeadGroup group = BuildHeadGroup(
        ResolveDense(0U), 0U, heads, Architecture::Arch22, tickets);
    constexpr std::array<std::uint32_t, 4U> kExpectedAiv = {{
        0U, 1U, 0U, 1U,
    }};
    constexpr std::array<std::uint32_t, 4U> kExpectedLocalBank = {{
        0U, 2U, 1U, 3U,
    }};
    constexpr std::array<std::uint32_t, 4U> kExpectedPrivateSlot = {{
        0U, 0U, 1U, 1U,
    }};
    constexpr std::array<std::uint32_t, 4U> kExpectedPhysicalLane = {{
        0U, 1U, 0U, 1U,
    }};
    for (std::uint32_t local = 0U; local < heads; ++local) {
        const HeadTask &head = group.heads[local];
        const std::uint32_t aiv = kExpectedAiv[local];
        if (!head.active || head.headId != local || head.aivId != aiv ||
            head.localBankId != kExpectedLocalBank[local] ||
            arch22_policy::PairWave(local) != kExpectedPrivateSlot[local] ||
            arch22_policy::PhysicalLane(local) !=
                kExpectedPhysicalLane[local]) {
            return false;
        }

        const OperationRecord *qLoad = FindUniqueOperationForSlots(
            trace.aivOperations[aiv], OperationKind::Load, Stage::V0, "q",
            "q-to-qhat", head.workspaceSlot, head.localBankId);
        const std::uint64_t privateOffset =
            arch22_policy::UbPolicy::PrivateBase(kExpectedPrivateSlot[local]) +
            arch22_policy::V01PrivateLayout::kQToQPlus.offset;
        if (qLoad == nullptr ||
            !CheckSpanIdentity(qLoad->source, "q", MemorySpace::Gm, 0U,
                               0x4000U, head.workspaceSlot,
                               head.workspaceGeneration, CoreRole::Shared,
                               0U) ||
            !CheckSpanIdentity(qLoad->destination, "q-to-qhat",
                               MemorySpace::Ub, privateOffset, 0x4000U,
                               head.localBankId, head.localGeneration,
                               CoreRole::Aiv, aiv)) {
            return false;
        }

        const OperationRecord *scoreLoad = FindUniqueOperationForSlots(
            trace.aicOperations, OperationKind::Load, Stage::C2, nullptr,
            "packed-score-L1", head.workspaceSlot, head.l1BankId);
        if (scoreLoad == nullptr ||
            !CheckSpanIdentity(
                scoreLoad->destination, "packed-score-L1", MemorySpace::L1,
                arch22_policy::L1Policy::kLaneBase[local], 0x12000U,
                head.l1BankId, head.l1Generation, CoreRole::Aic, 0U)) {
            return false;
        }

        const OperationRecord *w = FindUniqueOperationForSlots(
            trace.aicOperations, OperationKind::Mmad, Stage::C7,
            "Akk-cube-ready-resident", "W-L0C", head.l1BankId,
            head.l0cBankId);
        const std::uint64_t akkOffset =
            arch22_policy::L1Policy::Akk2BResident::kAkkTau.offset +
            local * arch22_policy::L1Policy::Akk2BResident::kAkkStride;
        const std::uint64_t l0cOffset =
            arch22_policy::L0cPolicy::HeadLaneBase(
                kExpectedPhysicalLane[local]) +
            arch22_policy::L0cPolicy::kC7W.offset;
        if (w == nullptr ||
            !CheckSpanIdentity(w->source, "Akk-cube-ready-resident",
                               MemorySpace::L1, akkOffset, 0x2000U,
                               head.l1BankId, head.l1Generation,
                               CoreRole::Aic, 0U) ||
            !CheckSpanIdentity(
                w->destination, "W-L0C", MemorySpace::L0, l0cOffset,
                0x8000U, kExpectedPhysicalLane[local],
                L0cGenerationFor(head, L0cStageUse::C7,
                                 Architecture::Arch22),
                CoreRole::Aic, 0U)) {
            return false;
        }

        for (Stage stage : {Stage::V0, Stage::V1, Stage::V3, Stage::V6}) {
            if (CountAivRunVfForHead(trace, stage, local) != 1U) {
                return false;
            }
        }
    }

    if (CountAivOperations(trace, OperationKind::Load, Stage::V0, "q",
                           "q-to-qhat") != heads ||
        CountOperations(trace.aicOperations, OperationKind::Load, Stage::C2,
                        nullptr, "packed-score-L1") != heads ||
        CountOperations(trace.aicOperations, OperationKind::Mmad, Stage::C7,
                        "Akk-cube-ready-resident", "W-L0C") != heads) {
        return false;
    }

    if (heads == 3U) {
        RuntimeTiling tiling{};
        tiling.totalChunks = 1U;
        tiling.headCount = heads;
        tiling.aicWorkgroupCount = 1U;
        tiling.epsilon = 1.0e-6F;
        tiling.key.abi = PrepareAbi::Current;
        for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
            if (!CheckAivTrace(Architecture::Arch22,
                               trace.aivSynchronization[aiv], tiling, 0U,
                               aiv, ResolveDense)) {
                return false;
            }
        }
        const HeadTask &inactiveHead = group.heads[3U];
        if (inactiveHead.active || TouchesInactiveArch22Head(trace,
                                                             inactiveHead)) {
            return false;
        }
        for (Stage stage : {Stage::V0, Stage::V1, Stage::V3, Stage::V6}) {
            if (CountAivRunVfForHead(trace, stage, 3U) != 0U) {
                return false;
            }
        }
    }
    return true;
}

bool MatchesAivLocalSpan(const BufferSpan &span,
                         const HeadTask &head) noexcept
{
    return span.name != nullptr && span.space == MemorySpace::Ub &&
           span.ownerRole == CoreRole::Aiv && span.ownerId == head.aivId &&
           span.slot == head.localBankId &&
           span.generation == head.localGeneration;
}

bool IsAivOperationForHead(const OperationRecord &record,
                           const HeadTask &head) noexcept
{
    switch (record.kind) {
        case OperationKind::RunVf:
            return record.headId == head.headId;
        case OperationKind::Load:
        case OperationKind::Zero:
        case OperationKind::ZeroUndefined:
            return MatchesAivLocalSpan(record.destination, head);
        case OperationKind::Store:
            return MatchesAivLocalSpan(record.source, head);
        default:
            return false;
    }
}

bool IsAivOperationInsideMutex(const OperationRecord &record,
                               const MutexTrace &mutexes,
                               const HeadTask &head) noexcept
{
    Pipe pipe = Pipe::Control;
    switch (record.kind) {
        case OperationKind::Load:
        case OperationKind::Zero:
        case OperationKind::ZeroUndefined:
            pipe = Pipe::Mte2;
            break;
        case OperationKind::RunVf:
            pipe = Pipe::Vector;
            break;
        case OperationKind::Store:
            pipe = Pipe::Mte3;
            break;
        default:
            return false;
    }
    return IsOrderInsideMutexInterval(
        mutexes, MutexResource::AivUbBank,
        Arch35VectorMutexIds::UbBank(head.aivLocalSlot), head.localBankId,
        record.stage, pipe, record.order);
}

bool MatchesAicL1Span(const BufferSpan &span,
                      const HeadTask &head) noexcept
{
    return span.name != nullptr && span.space == MemorySpace::L1 &&
           span.ownerRole == CoreRole::Aic && span.ownerId == 0U &&
           span.slot == head.l1BankId &&
           span.generation == head.l1Generation;
}

bool MatchesAicL0cSpan(const BufferSpan &span,
                       const HeadTask &head) noexcept
{
    return span.name != nullptr && span.space == MemorySpace::L0 &&
           span.ownerRole == CoreRole::Aic && span.ownerId == 0U &&
           span.slot == head.l0cBankId &&
           span.generation == head.l0cGeneration;
}

bool IsAicOperationForHead(const OperationRecord &record,
                           const HeadTask &head) noexcept
{
    switch (record.kind) {
        case OperationKind::Load:
        case OperationKind::Fill:
            return MatchesAicL1Span(record.destination, head);
        case OperationKind::Mmad:
        case OperationKind::MmadRowStackedLhs:
        case OperationKind::MmadQuadrantPackedLhs:
            return record.headId == head.headId;
        case OperationKind::Store:
        case OperationKind::StoreRounded:
            return MatchesAicL0cSpan(record.source, head);
        default:
            return false;
    }
}

bool UsesUpperC7L0c(const OperationRecord &record,
                    const HeadTask &head) noexcept
{
    if (record.stage != Stage::C7) {
        return false;
    }
    const std::uint64_t upperOffset =
        L0cPolicy::HeadLaneBase(head.groupLocalHead) +
        L0cPolicy::kC7U.offset;
    const BufferSpan &l0c =
        record.kind == OperationKind::StoreRounded ? record.source
                                                   : record.destination;
    return l0c.byteOffset == upperOffset;
}

bool IsAicOperationInsideMutex(const OperationRecord &record,
                               const MutexTrace &mutexes,
                               const HeadTask &head) noexcept
{
    const SymbolicMutexId l1 = Arch35CubeMutexIds::L1Bank(head.l1BankId);
    if (record.kind == OperationKind::Load ||
        record.kind == OperationKind::Fill) {
        return IsOrderInsideMutexInterval(
            mutexes, MutexResource::AicL1Bank, l1, head.l1BankId,
            record.stage, Pipe::Mte2, record.order);
    }

    const bool compute = record.kind == OperationKind::Mmad ||
                         record.kind == OperationKind::MmadRowStackedLhs ||
                         record.kind ==
                             OperationKind::MmadQuadrantPackedLhs;
    const bool fixpipe = record.kind == OperationKind::Store ||
                         record.kind == OperationKind::StoreRounded;
    if (!compute && !fixpipe) {
        return false;
    }
    const bool upper = UsesUpperC7L0c(record, head);
    const MutexResource l0cResource =
        upper ? MutexResource::AicL0cUpperHalf
              : MutexResource::AicL0cLowerHalf;
    const SymbolicMutexId l0cMutex =
        upper ? Arch35CubeMutexIds::L0cUpperHalf(head.l0cBankId)
              : Arch35CubeMutexIds::L0cLowerHalf(head.l0cBankId);
    if (!IsOrderInsideMutexInterval(
            mutexes, l0cResource, l0cMutex, head.l0cBankId,
            record.stage, compute ? Pipe::Cube : Pipe::Fixpipe,
            record.order)) {
        return false;
    }
    if (compute &&
        !IsOrderInsideMutexInterval(
            mutexes, MutexResource::AicL0OperandBank,
            Arch35CubeMutexIds::L0OperandBank(head.l0OperandBankId),
            head.l0OperandBankId, record.stage, Pipe::Cube,
            record.order)) {
        return false;
    }
    if (fixpipe &&
        (record.stage == Stage::C4 || record.stage == Stage::C5) &&
        !IsOrderInsideMutexInterval(
            mutexes, MutexResource::AicL1Bank, l1, head.l1BankId,
            record.stage, Pipe::Fixpipe, record.order)) {
        return false;
    }
    return true;
}

bool CheckUniqueWaitBefore(const SyncTrace &synchronization,
                           SyncPoint point, std::uint32_t ownerId,
                           std::uint64_t generation, Stage stage, Pipe pipe,
                           std::uint64_t firstLock) noexcept
{
    std::uint64_t waitOrder = 0U;
    return FindUniqueSyncOrder(synchronization, SyncAction::Wait, point,
                               ownerId, generation, stage, pipe, waitOrder) &&
           waitOrder < firstLock;
}

bool CheckArch35AivHeadWaits(const SyncTrace &synchronization,
                             const MutexTrace &mutexes,
                             const HeadTask &head) noexcept
{
    const SymbolicMutexId ub =
        Arch35VectorMutexIds::UbBank(head.aivLocalSlot);
    std::uint64_t v0Lock = 0U;
    std::uint64_t v3Lock = 0U;
    std::uint64_t v6Lock = 0U;
    if (!FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AivUbBank, ub,
            head.localBankId, Stage::V0, Pipe::Mte2, v0Lock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AivUbBank, ub,
            head.localBankId, Stage::V3, Pipe::Vector, v3Lock) ||
        !FindUniqueMutexOrder(
            mutexes, MutexAction::Lock, MutexResource::AivUbBank, ub,
            head.localBankId, Stage::V6, Pipe::Mte2, v6Lock) ||
        !CheckUniqueWaitBefore(
            synchronization, SyncPoint::SlotFree, head.workspaceSlot,
            head.workspaceGeneration, Stage::V0, Pipe::Control, v0Lock) ||
        !CheckUniqueWaitBefore(
            synchronization,
            head.qkOwner ? SyncPoint::QkCacheFree
                         : SyncPoint::QkCacheReady,
            head.qkCacheSlot, head.qkCacheGeneration, Stage::V0,
            Pipe::Mte2, v0Lock) ||
        !CheckUniqueWaitBefore(
            synchronization, SyncPoint::C2ScorePayloadFree,
            head.workspaceSlot, head.workspaceGeneration, Stage::V3,
            Pipe::Mte3, v3Lock) ||
        !CheckUniqueWaitBefore(
            synchronization, SyncPoint::C2RawReady, head.localBankId,
            head.localGeneration, Stage::V3, Pipe::Vector, v3Lock) ||
        !CheckUniqueWaitBefore(
            synchronization, SyncPoint::C4PayloadFree,
            head.workspaceSlot, head.workspaceGeneration, Stage::V6,
            Pipe::Mte2, v6Lock)) {
        return false;
    }
    return CountMutexStageOwner(mutexes, Stage::V0, head.localBankId) == 6U &&
           CountMutexStageOwner(mutexes, Stage::V1, head.localBankId) == 4U &&
           CountMutexStageOwner(mutexes, Stage::V3, head.localBankId) == 4U &&
           CountMutexStageOwner(mutexes, Stage::V6, head.localBankId) == 6U;
}

bool CheckArch35AicHeadWaits(const SyncTrace &synchronization,
                             const MutexTrace &mutexes,
                             const HeadTask &head) noexcept
{
    const SymbolicMutexId l1 = Arch35CubeMutexIds::L1Bank(head.l1BankId);
    std::uint64_t c2Lock = 0U;
    std::uint64_t c4Lock = 0U;
    std::uint64_t c7Lock = 0U;
    return FindUniqueMutexOrder(
               mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
               head.l1BankId, Stage::C2, Pipe::Mte2, c2Lock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
               head.l1BankId, Stage::C4, Pipe::Mte2, c4Lock) &&
           FindUniqueMutexOrder(
               mutexes, MutexAction::Lock, MutexResource::AicL1Bank, l1,
               head.l1BankId, Stage::C7, Pipe::Mte2, c7Lock) &&
           CheckUniqueWaitBefore(
               synchronization, SyncPoint::V1ScoreReady,
               head.workspaceSlot, head.workspaceGeneration, Stage::C2,
               Pipe::Mte2, c2Lock) &&
           CheckUniqueWaitBefore(
               synchronization, SyncPoint::V1MainSourceFree,
               head.localBankId, head.localGeneration, Stage::C2,
               Pipe::Fixpipe, c2Lock) &&
           CheckUniqueWaitBefore(
               synchronization, SyncPoint::C2RawDstFree,
               head.localBankId, head.localGeneration, Stage::C2,
               Pipe::Fixpipe, c2Lock) &&
           CheckUniqueWaitBefore(
               synchronization, SyncPoint::V3VcsReady,
               head.workspaceSlot, head.workspaceGeneration, Stage::C4,
               Pipe::Mte2, c4Lock) &&
           CheckUniqueWaitBefore(
               synchronization, SyncPoint::V6RhsReady,
               head.workspaceSlot, head.workspaceGeneration, Stage::C7,
               Pipe::Mte2, c7Lock);
}

bool CheckArch35MultiHeadMutexWindows(
    const MultiHeadAddressTrace &trace, const HeadGroup &group) noexcept
{
    for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
        std::size_t ownedHeads = 0U;
        for (const HeadTask &head : group.heads) {
            if (!head.active || head.aivId != aiv) {
                continue;
            }
            ++ownedHeads;
            if (!CheckArch35AivHeadWaits(trace.aivSynchronization[aiv],
                                         trace.aivMutexes[aiv], head)) {
                return false;
            }
        }
        if (CountStageWaitActions(trace.aivSynchronization[aiv], Stage::V0) !=
                2U * ownedHeads ||
            CountStageWaitActions(trace.aivSynchronization[aiv], Stage::V1) !=
                0U ||
            CountStageWaitActions(trace.aivSynchronization[aiv], Stage::V3) !=
                2U * ownedHeads ||
            CountStageWaitActions(trace.aivSynchronization[aiv], Stage::V6) !=
                ownedHeads) {
            return false;
        }
        const OperationTrace &operations = trace.aivOperations[aiv];
        for (std::size_t index = 0U; index < operations.size; ++index) {
            const OperationRecord &record = operations.records[index];
            std::size_t claims = 0U;
            for (const HeadTask &head : group.heads) {
                if (head.active && head.aivId == aiv &&
                    IsAivOperationForHead(record, head)) {
                    ++claims;
                    if (!IsAivOperationInsideMutex(record,
                                                   trace.aivMutexes[aiv],
                                                   head)) {
                        return false;
                    }
                }
            }
            if (claims != 1U) {
                return false;
            }
        }
    }

    for (const HeadTask &head : group.heads) {
        if (head.active &&
            !CheckArch35AicHeadWaits(trace.aicSynchronization,
                                     trace.aicMutexes, head)) {
            return false;
        }
    }
    const std::size_t activeHeads = group.activeHeads;
    if (CountStageWaitActions(trace.aicSynchronization, Stage::C2) !=
            3U * activeHeads ||
        CountStageWaitActions(trace.aicSynchronization, Stage::C4) !=
            activeHeads ||
        CountStageWaitActions(trace.aicSynchronization, Stage::C5) != 0U ||
        CountStageWaitActions(trace.aicSynchronization, Stage::C7) !=
            activeHeads) {
        return false;
    }
    for (std::size_t index = 0U; index < trace.aicOperations.size; ++index) {
        const OperationRecord &record = trace.aicOperations.records[index];
        std::size_t claims = 0U;
        for (const HeadTask &head : group.heads) {
            if (head.active && IsAicOperationForHead(record, head)) {
                ++claims;
                if (!IsAicOperationInsideMutex(record, trace.aicMutexes,
                                               head)) {
                    return false;
                }
            }
        }
        if (claims != 1U) {
            return false;
        }
    }
    return true;
}

bool CheckArch35MultiHeadAddresses() noexcept
{
    constexpr std::uint32_t kHeads = 4U;
    MultiHeadAddressTrace trace{};
    if (!CaptureMultiHeadAddressTrace(Architecture::Arch35, kHeads, trace)) {
        return false;
    }
    OwnerTicketState tickets{};
    const HeadGroup group = BuildHeadGroup(
        ResolveDense(0U), 0U, kHeads, Architecture::Arch35, tickets);
    if (!CheckArch35MultiHeadMutexWindows(trace, group)) {
        return false;
    }
    constexpr std::array<std::uint32_t, 4U> kExpectedAiv = {{
        0U, 0U, 1U, 1U,
    }};
    constexpr std::array<std::uint32_t, 4U> kExpectedAivLocalSlot = {{
        0U, 1U, 0U, 1U,
    }};
    for (std::uint32_t local = 0U; local < kHeads; ++local) {
        const HeadTask &head = group.heads[local];
        const std::uint32_t aiv = kExpectedAiv[local];
        const std::uint32_t aivLocalSlot = kExpectedAivLocalSlot[local];
        if (!head.active || head.headId != local || head.aivId != aiv ||
            head.aivLocalSlot != aivLocalSlot ||
            head.localBankId != local || head.l1BankId != local ||
            head.l0cBankId != local ||
            CountMutexRecords(
                trace.aivMutexes[aiv], MutexAction::Lock,
                MutexResource::AivUbBank,
                Arch35VectorMutexIds::UbBank(aivLocalSlot),
                head.localBankId, Stage::V0, Pipe::Mte2) != 1U ||
            CountMutexRecords(
                trace.aicMutexes, MutexAction::Lock,
                MutexResource::AicL1Bank,
                Arch35CubeMutexIds::L1Bank(head.l1BankId),
                head.l1BankId, Stage::C2, Pipe::Mte2) != 1U ||
            CountMutexRecords(
                trace.aicMutexes, MutexAction::Lock,
                MutexResource::AicL0cLowerHalf,
                Arch35CubeMutexIds::L0cLowerHalf(head.l0cBankId),
                head.l0cBankId, Stage::C2, Pipe::Cube) != 4U ||
            CountMutexRecords(
                trace.aicMutexes, MutexAction::Lock,
                MutexResource::AicL0cUpperHalf,
                Arch35CubeMutexIds::L0cUpperHalf(head.l0cBankId),
                head.l0cBankId, Stage::C7, Pipe::Cube) != 1U) {
            return false;
        }

        const OperationRecord *qLoad = FindUniqueOperationForSlots(
            trace.aivOperations[aiv], OperationKind::Load, Stage::V0, "q",
            "q-to-qhat", head.workspaceSlot, head.localBankId);
        const std::uint64_t mainOffset =
            UbPolicy::kMainBase[aivLocalSlot] + V0Gate2BLayout::kQHat.offset;
        if (qLoad == nullptr ||
            !CheckSpanIdentity(qLoad->destination, "q-to-qhat",
                               MemorySpace::Ub, mainOffset, 0x4000U,
                               head.localBankId, head.localGeneration,
                               CoreRole::Aiv, aiv)) {
            return false;
        }

        const OperationRecord *betaLoad = FindUniqueOperationForSlots(
            trace.aivOperations[aiv], OperationKind::Load, Stage::V0, "beta",
            "beta-raw", head.workspaceSlot, head.localBankId);
        const std::uint64_t vectorStateOffset =
            UbPolicy::kVectorStateBase[aivLocalSlot] +
            VectorStateLayout::kBetaRaw.offset;
        if (betaLoad == nullptr ||
            !CheckSpanIdentity(betaLoad->destination, "beta-raw",
                               MemorySpace::Ub, vectorStateOffset, 0x0100U,
                               head.localBankId, head.localGeneration,
                               CoreRole::Aiv, aiv)) {
            return false;
        }

        const OperationRecord *scoreLoad = FindUniqueOperationForSlots(
            trace.aicOperations, OperationKind::Load, Stage::C2,
            "packed-score", "C2-current-lane", head.workspaceSlot,
            head.l1BankId);
        if (scoreLoad == nullptr ||
            !CheckSpanIdentity(scoreLoad->destination, "C2-current-lane",
                               MemorySpace::L1,
                               L1Policy::kLaneBase[local], 0x12000U,
                               head.l1BankId, head.l1Generation,
                               CoreRole::Aic, 0U)) {
            return false;
        }

        const OperationRecord *w = FindUniqueOperationForSlots(
            trace.aicOperations, OperationKind::MmadQuadrantPackedLhs,
            Stage::C7, "Akk-resident", "W-fp32-L0C", head.l1BankId,
            head.l0cBankId);
        const std::uint64_t akkOffset =
            L1Policy::Akk2BResident::kAkkTau.offset +
            local * L1Policy::Akk2BResident::kAkkStride;
        const std::uint64_t l0cOffset =
            L0cPolicy::HeadLaneBase(local) + L0cPolicy::kC7W.offset;
        if (w == nullptr ||
            !CheckSpanIdentity(w->source, "Akk-resident", MemorySpace::L1,
                               akkOffset, 0x2000U, head.l1BankId,
                               head.l1Generation, CoreRole::Aic, 0U) ||
            !CheckSpanIdentity(w->destination, "W-fp32-L0C",
                               MemorySpace::L0, l0cOffset, 0x8000U,
                               head.l0cBankId, head.l0cGeneration,
                               CoreRole::Aic, 0U)) {
            return false;
        }

        for (Stage stage : {Stage::V0, Stage::V1, Stage::V3, Stage::V6}) {
            if (CountAivRunVfForHead(trace, stage, local) != 1U) {
                return false;
            }
        }
    }

    return CountMutexRecords(
               trace.aicMutexes, MutexAction::Lock,
               MutexResource::AicL0OperandBank,
               Arch35CubeMutexIds::L0OperandBank(0U), 0U, Stage::C7,
               Pipe::Cube) == kHeads &&
           CountAivOperations(trace, OperationKind::Load, Stage::V0, "q",
                              "q-to-qhat") == kHeads &&
           CountAivOperations(trace, OperationKind::Load, Stage::V0, "beta",
                              "beta-raw") == kHeads &&
           CountOperations(trace.aicOperations, OperationKind::Load,
                           Stage::C2, "packed-score", "C2-current-lane") ==
               kHeads &&
           CountOperations(trace.aicOperations,
                           OperationKind::MmadQuadrantPackedLhs, Stage::C7,
                           "Akk-resident", "W-fp32-L0C") == kHeads;
}

bool CheckGroupedQkHeadMapping(Architecture architecture) noexcept
{
    constexpr std::uint32_t kValueHeads = 4U;
    constexpr std::uint32_t kQkHeads = 2U;
    constexpr std::array<std::uint32_t, kValueHeads> kExpectedQkHead = {{
        0U, 0U, 1U, 1U,
    }};
    ProposedTilingKey key{};
    key.inputStorage = InputStorage::Bf16;
    key.qkNormMode = QkNormMode::Identity;
    MultiHeadAddressTrace trace{};
    if (!CaptureMultiHeadAddressTrace(architecture, kValueHeads, trace,
                                      kQkHeads, key)) {
        return false;
    }
    OwnerTicketState tickets{};
    const HeadGroup group = BuildHeadGroup(
        ResolveDense(0U), 0U, kValueHeads, architecture, tickets, kQkHeads);
    for (std::uint32_t local = 0U; local < kValueHeads; ++local) {
        const HeadTask &head = group.heads[local];
        const std::uint32_t aiv =
            AivForGroupLocalHead(architecture, local);
        const bool owner = local % 2U == 0U;
        if (!head.active || head.headId != local ||
            head.qkHeadId != kExpectedQkHead[local] ||
            head.qkOwner != owner || head.qkLastConsumer == owner ||
            head.qkCacheSlot != (local / 2U) * 2U) {
            return false;
        }
        const char *qSourceName = owner ? "q" : "qhat-HK-cache";
        const char *kSourceName = owner ? "k" : "khat-HK-cache";
        const std::uint64_t sourceGeneration =
            owner ? head.workspaceGeneration : head.qkCacheGeneration;
        const OperationRecord *qLoad =
            FindUniqueOperationForSpanGenerations(
                trace.aivOperations[aiv], OperationKind::Load, Stage::V0,
                qSourceName, "q-to-qhat", head.qkCacheSlot,
                sourceGeneration, head.localBankId, head.localGeneration);
        const OperationRecord *kLoad =
            FindUniqueOperationForSpanGenerations(
                trace.aivOperations[aiv], OperationKind::Load, Stage::V0,
                kSourceName, "k-to-khat", head.qkCacheSlot,
                sourceGeneration, head.localBankId, head.localGeneration);
        const char *gateName =
            architecture == Architecture::Arch22 ? "gate-2b" : "gate";
        const char *gateDestination = architecture == Architecture::Arch22
                                          ? "gate-2b"
                                          : "gate-to-G";
        const OperationRecord *gateLoad =
            FindUniqueOperationForSpanGenerations(
                trace.aivOperations[aiv], OperationKind::Load, Stage::V0,
                gateName, gateDestination, head.workspaceSlot,
                head.workspaceGeneration, head.localBankId,
                head.localGeneration);
        const OperationRecord *valueLoad =
            FindUniqueOperationForSpanGenerations(
                trace.aivOperations[aiv], OperationKind::Load, Stage::V6,
                "V", "V-to-Vbeta", head.workspaceSlot,
                head.workspaceGeneration, head.localBankId,
                head.localGeneration);
        const OperationRecord *qV6 =
            FindUniqueOperationForSpanGenerations(
                trace.aivOperations[aiv], OperationKind::Load, Stage::V6,
                "qhat-HK-cache", "qhat-to-qg", head.qkCacheSlot,
                head.qkCacheGeneration, head.localBankId,
                head.localGeneration);
        const OperationRecord *kV6 =
            FindUniqueOperationForSpanGenerations(
                trace.aivOperations[aiv], OperationKind::Load, Stage::V6,
                "khat-HK-cache", "khat-to-kg", head.qkCacheSlot,
                head.qkCacheGeneration, head.localBankId,
                head.localGeneration);
        const OperationRecord *qStore =
            FindUniqueOperationForSpanGenerations(
                trace.aivOperations[aiv], OperationKind::Store, Stage::V0,
                "qhat", "qhat-HK-cache", head.localBankId,
                head.localGeneration, head.qkCacheSlot,
                head.qkCacheGeneration);
        const OperationRecord *kStore =
            FindUniqueOperationForSpanGenerations(
                trace.aivOperations[aiv], OperationKind::Store, Stage::V0,
                "khat", "khat-HK-cache", head.localBankId,
                head.localGeneration, head.qkCacheSlot,
                head.qkCacheGeneration);
        const std::uint32_t gSourceSlot =
            architecture == Architecture::Arch22 ? head.sharedArenaId
                                                 : head.localBankId;
        const OperationRecord *gOutput = FindUniqueOperationForSlots(
            trace.aivOperations[aiv], OperationKind::Store, Stage::V0,
            "G-output", "gk-output", gSourceSlot, head.workspaceSlot);
        if (qLoad == nullptr || kLoad == nullptr || gateLoad == nullptr ||
            valueLoad == nullptr || qV6 == nullptr || kV6 == nullptr ||
            gOutput == nullptr ||
            qLoad->source.byteSize != 0x4000U ||
            kLoad->source.byteSize != 0x4000U ||
            qLoad->destination.byteSize != 0x4000U ||
            kLoad->destination.byteSize != 0x4000U ||
            (owner &&
             (qStore == nullptr || kStore == nullptr ||
              qStore->source.byteSize != 0x4000U ||
              kStore->source.byteSize != 0x4000U ||
              qStore->destination.byteSize != 0x4000U ||
              kStore->destination.byteSize != 0x4000U)) ||
            (!owner && (qStore != nullptr || kStore != nullptr)) ||
            (owner &&
             (qLoad->source.space != MemorySpace::Gm ||
              kLoad->source.space != MemorySpace::Gm ||
              qLoad->source.logicalHeadId != kExpectedQkHead[local] ||
              kLoad->source.logicalHeadId != kExpectedQkHead[local])) ||
            (!owner &&
             (qLoad->source.space != MemorySpace::Workspace ||
              kLoad->source.space != MemorySpace::Workspace ||
              qLoad->source.byteOffset !=
                  static_cast<std::uint64_t>(head.qkCacheSlot) *
                      kWorkspaceSlotStrideBytes ||
              kLoad->source.byteOffset !=
                  static_cast<std::uint64_t>(head.qkCacheSlot) *
                          kWorkspaceSlotStrideBytes +
                      0x4000U)) ||
            gateLoad->source.logicalHeadId != local ||
            valueLoad->source.logicalHeadId != local ||
            gOutput->destination.logicalHeadId != local) {
            return false;
        }

        const char *wSource = architecture == Architecture::Arch22
                                  ? "W-L0C-valid"
                                  : "W-valid-fp32-ld128";
        const char *wDestination = architecture == Architecture::Arch22
                                       ? "W-output"
                                       : "W-valid-output-ld128";
        const char *uSource = architecture == Architecture::Arch22
                                  ? "U-L0C-valid"
                                  : "U-valid-fp32-ld128";
        const char *uDestination = architecture == Architecture::Arch22
                                       ? "U-output"
                                       : "U-valid-output-ld128";
        const OperationRecord *wOutput = FindUniqueOperationForSlots(
            trace.aicOperations, OperationKind::StoreRounded, Stage::C7,
            wSource, wDestination, head.l0cBankId, head.workspaceSlot);
        const OperationRecord *uOutput = FindUniqueOperationForSlots(
            trace.aicOperations, OperationKind::StoreRounded, Stage::C7,
            uSource, uDestination, head.l0cBankId, head.workspaceSlot);
        if (wOutput == nullptr || uOutput == nullptr ||
            wOutput->destination.logicalHeadId != local ||
            uOutput->destination.logicalHeadId != local) {
            return false;
        }
        if (CountAivRunVfForHead(trace, Stage::V0, local) != 1U) {
            return false;
        }
    }
    return CountAivOperations(trace, OperationKind::Load, Stage::V0, "q",
                              "q-to-qhat") == kQkHeads &&
           CountAivOperations(trace, OperationKind::Load, Stage::V0, "k",
                              "k-to-khat") == kQkHeads &&
           CountAivOperations(trace, OperationKind::Load, Stage::V0,
                              "qhat-HK-cache", "q-to-qhat") ==
               kValueHeads - kQkHeads &&
           CountAivOperations(trace, OperationKind::Load, Stage::V0,
                              "khat-HK-cache", "k-to-khat") ==
               kValueHeads - kQkHeads &&
           CountAivOperations(trace, OperationKind::Store, Stage::V0,
                              "qhat", "qhat-HK-cache") == kQkHeads &&
           CountAivOperations(trace, OperationKind::Store, Stage::V0,
                              "khat", "khat-HK-cache") == kQkHeads &&
           CountAivOperations(trace, OperationKind::Store, Stage::V0,
                              "qhat", "qhat-context") == 0U &&
           CountAivOperations(trace, OperationKind::Store, Stage::V0,
                              "khat", "khat-context") == 0U &&
           CountAivOperations(trace, OperationKind::Load, Stage::V6,
                              "qhat-HK-cache", "qhat-to-qg") ==
               kValueHeads &&
           CountAivOperations(trace, OperationKind::Load, Stage::V6,
                              "khat-HK-cache", "khat-to-kg") ==
               kValueHeads &&
           CountAivOperations(trace, OperationKind::Load, Stage::V6, "V",
                              "V-to-Vbeta") == kValueHeads;
}

bool CheckQkContextAddressCase(Architecture architecture,
                               std::uint32_t valueHeads,
                               std::uint32_t qkHeads,
                               std::uint32_t headId, bool expectOwner,
                               bool expectAlias) noexcept
{
    constexpr std::size_t kMaxHeads = 16U;
    if (valueHeads > kMaxHeads || headId >= valueHeads) {
        return false;
    }
    MultiHeadAddressTrace trace{};
    if (!CaptureMultiHeadAddressTrace(architecture, valueHeads, trace,
                                      qkHeads)) {
        return false;
    }
    RuntimeTiling tiling{};
    tiling.totalChunks = 1U;
    tiling.headCount = valueHeads;
    tiling.qkHeadCount = qkHeads;
    tiling.aicWorkgroupCount = 1U;
    tiling.epsilon = 1.0e-6F;
    std::array<HeadTask, kMaxHeads> heads{};
    std::array<bool, kMaxHeads> seen{};
    if (!CollectExpectedHeads(architecture, tiling, heads, seen)) {
        return false;
    }
    for (std::uint32_t current = 0U; current < valueHeads; ++current) {
        if (!seen[current]) {
            return false;
        }
    }

    const HeadTask &head = heads[headId];
    const bool aliasesCache = head.qkCacheSlot == head.workspaceSlot;
    if (head.qkOwner != expectOwner || aliasesCache != expectAlias ||
        (!expectOwner && expectAlias &&
         head.qkCacheGeneration == head.workspaceGeneration)) {
        return false;
    }
    const OperationTrace &operations = trace.aivOperations[head.aivId];
    const std::uint32_t contextSlot = head.qkCacheSlot;
    const std::uint64_t contextGeneration = head.qkCacheGeneration;
    const std::uint64_t contextBase =
        static_cast<std::uint64_t>(contextSlot) *
        kWorkspaceSlotStrideBytes;

    const OperationRecord *qV6 = FindUniqueOperationForSpanGenerations(
        operations, OperationKind::Load, Stage::V6, "qhat-HK-cache",
        "qhat-to-qg", contextSlot, contextGeneration, head.localBankId,
        head.localGeneration);
    const OperationRecord *kV6 = FindUniqueOperationForSpanGenerations(
        operations, OperationKind::Load, Stage::V6, "khat-HK-cache",
        "khat-to-kg", contextSlot, contextGeneration, head.localBankId,
        head.localGeneration);
    if (qV6 == nullptr || kV6 == nullptr ||
        !CheckSpanIdentity(qV6->source, nullptr, MemorySpace::Workspace,
                           contextBase, 0x4000U, contextSlot,
                           contextGeneration, CoreRole::Shared, 0U) ||
        !CheckSpanIdentity(kV6->source, nullptr, MemorySpace::Workspace,
                           contextBase + 0x4000U, 0x4000U, contextSlot,
                           contextGeneration, CoreRole::Shared, 0U)) {
        return false;
    }

    if (!expectOwner) {
        const std::uint32_t ownerId = QkOwnerValueHead(
            head.qkHeadId, valueHeads, qkHeads);
        const HeadTask &owner = heads[ownerId];
        const OperationTrace &ownerOperations =
            trace.aivOperations[owner.aivId];
        const OperationRecord *ownerQStore =
            FindUniqueOperationForSpanGenerations(
                ownerOperations, OperationKind::Store, Stage::V0, "qhat",
                "qhat-HK-cache", owner.localBankId, owner.localGeneration,
                owner.qkCacheSlot, owner.qkCacheGeneration);
        const OperationRecord *ownerKStore =
            FindUniqueOperationForSpanGenerations(
                ownerOperations, OperationKind::Store, Stage::V0, "khat",
                "khat-HK-cache", owner.localBankId, owner.localGeneration,
                owner.qkCacheSlot, owner.qkCacheGeneration);
        return ownerQStore != nullptr && ownerKStore != nullptr &&
               SamePhysicalSpan(ownerQStore->destination, qV6->source) &&
               SamePhysicalSpan(ownerKStore->destination, kV6->source) &&
               CountOperationsForSpanGenerations(
                   operations, OperationKind::Store, Stage::V0, "qhat",
                   "qhat-context", head.localBankId, head.localGeneration,
                   head.workspaceSlot, head.workspaceGeneration) == 0U &&
               CountOperationsForSpanGenerations(
                   operations, OperationKind::Store, Stage::V0, "khat",
                   "khat-context", head.localBankId, head.localGeneration,
                   head.workspaceSlot, head.workspaceGeneration) == 0U;
    }

    const OperationRecord *qStore = FindUniqueOperationForSpanGenerations(
        operations, OperationKind::Store, Stage::V0, "qhat",
        "qhat-HK-cache", head.localBankId, head.localGeneration, contextSlot,
        contextGeneration);
    const OperationRecord *kStore = FindUniqueOperationForSpanGenerations(
        operations, OperationKind::Store, Stage::V0, "khat",
        "khat-HK-cache", head.localBankId, head.localGeneration, contextSlot,
        contextGeneration);
    return qStore != nullptr && kStore != nullptr &&
           SamePhysicalSpan(qStore->destination, qV6->source) &&
           SamePhysicalSpan(kStore->destination, kV6->source);
}

bool CheckQkContextAddressModes(Architecture architecture) noexcept
{
    const std::uint32_t slotCount = WorkspaceSlotCountFor(architecture);
    return CheckQkContextAddressCase(architecture, 4U, 2U, 0U, true, true) &&
           CheckQkContextAddressCase(architecture, 4U, 2U, 1U, false,
                                     false) &&
           CheckQkContextAddressCase(architecture, slotCount * 2U, 1U,
                                     slotCount, false, true);
}

bool CheckQkCacheProtocolCase(Architecture architecture,
                              std::uint32_t valueHeads,
                              std::uint32_t qkHeads) noexcept
{
    constexpr std::size_t kMaxValueHeads = 16U;
    if (valueHeads == 0U || valueHeads > kMaxValueHeads || qkHeads == 0U ||
        valueHeads % qkHeads != 0U) {
        return false;
    }
    const std::uint32_t ratio = valueHeads / qkHeads;
    MultiHeadAddressTrace trace{};
    if (!CaptureMultiHeadAddressTrace(architecture, valueHeads, trace,
                                      qkHeads)) {
        return false;
    }
    RuntimeTiling tiling{};
    tiling.totalChunks = 1U;
    tiling.headCount = valueHeads;
    tiling.qkHeadCount = qkHeads;
    tiling.aicWorkgroupCount = 1U;
    tiling.epsilon = 1.0e-6F;
    std::array<HeadTask, kMaxValueHeads> heads{};
    std::array<bool, kMaxValueHeads> seen{};
    if (!CollectExpectedHeads(architecture, tiling, heads, seen)) {
        return false;
    }
    std::array<std::size_t, kMaxValueHeads>
        expectedReadyWaitsBeforeFree{};
    std::array<bool, kMaxValueHeads> expectedFreeSeen{};
    std::size_t emittedReadyWaits = 0U;
    if (!ReplayItems(
            architecture, tiling, 0U, ResolveDense,
            [architecture, qkHeads, &expectedReadyWaitsBeforeFree,
             &expectedFreeSeen,
             &emittedReadyWaits](const WorkItem &item) noexcept {
                if (architecture == Architecture::Arch22) {
                    for (std::uint32_t pair = 0U;
                         pair < kArch22PairCount; ++pair) {
                        if (!PairHasActiveHead(item.group, pair)) {
                            continue;
                        }
                        ++emittedReadyWaits;
                        for (const HeadTask &head : item.group.heads) {
                            if (!head.active || !head.qkLastConsumer ||
                                arch22_policy::PairWave(
                                    head.groupLocalHead) != pair) {
                                continue;
                            }
                            if (head.qkHeadId >= qkHeads ||
                                expectedFreeSeen[head.qkHeadId]) {
                                return false;
                            }
                            expectedFreeSeen[head.qkHeadId] = true;
                            expectedReadyWaitsBeforeFree[head.qkHeadId] =
                                emittedReadyWaits;
                        }
                    }
                    return true;
                }
                for (const HeadTask &head : item.group.heads) {
                    if (!head.active) {
                        continue;
                    }
                    ++emittedReadyWaits;
                    if (!head.qkLastConsumer) {
                        continue;
                    }
                    if (head.qkHeadId >= qkHeads ||
                        expectedFreeSeen[head.qkHeadId]) {
                        return false;
                    }
                    expectedFreeSeen[head.qkHeadId] = true;
                    expectedReadyWaitsBeforeFree[head.qkHeadId] =
                        emittedReadyWaits;
                }
                return true;
            })) {
        return false;
    }
    for (std::uint32_t qkHeadId = 0U; qkHeadId < qkHeads; ++qkHeadId) {
        if (!expectedFreeSeen[qkHeadId] ||
            expectedReadyWaitsBeforeFree[qkHeadId] == 0U) {
            return false;
        }
    }

    for (const SyncTrace &aivTrace : trace.aivSynchronization) {
        for (std::size_t index = 0U; index < aivTrace.size; ++index) {
            const SyncRecord &record = aivTrace.records[index];
            if (record.action == SyncAction::Set &&
                record.point == SyncPoint::QkCacheFree) {
                return false;
            }
        }
    }
    for (std::size_t index = 0U; index < trace.aicSynchronization.size;
         ++index) {
        const SyncRecord &record =
            trace.aicSynchronization.records[index];
        if (record.point == SyncPoint::QkCacheFree &&
            (record.action != SyncAction::Set ||
             record.stage != Stage::C7 || record.pipe != Pipe::Control)) {
            return false;
        }
    }
    if (!CheckC7ReadyWaitSources(trace, architecture)) {
        return false;
    }
    for (std::uint32_t qkHeadId = 0U; qkHeadId < qkHeads; ++qkHeadId) {
        const std::uint32_t ownerId = qkHeadId * ratio;
        const std::uint32_t lastId = ownerId + ratio - 1U;
        const HeadTask &owner = heads[ownerId];
        const HeadTask &last = heads[lastId];
        std::uint32_t ownerCount = 0U;
        std::uint32_t lastCount = 0U;
        for (std::uint32_t headId = ownerId; headId <= lastId; ++headId) {
            const HeadTask &head = heads[headId];
            if (!seen[headId] || head.qkHeadId != qkHeadId ||
                head.qkCacheSlot != owner.qkCacheSlot ||
                head.qkCacheGeneration != owner.qkCacheGeneration) {
                return false;
            }
            ownerCount += head.qkOwner ? 1U : 0U;
            lastCount += head.qkLastConsumer ? 1U : 0U;
            if (CountAivRunVfForHead(trace, Stage::V0, headId) != 1U) {
                return false;
            }
            const OperationRecord *qV6 =
                FindUniqueOperationForSpanGenerations(
                    trace.aivOperations[head.aivId], OperationKind::Load,
                    Stage::V6, "qhat-HK-cache", "qhat-to-qg",
                    head.qkCacheSlot, head.qkCacheGeneration,
                    head.localBankId, head.localGeneration);
            const OperationRecord *kV6 =
                FindUniqueOperationForSpanGenerations(
                    trace.aivOperations[head.aivId], OperationKind::Load,
                    Stage::V6, "khat-HK-cache", "khat-to-kg",
                    head.qkCacheSlot, head.qkCacheGeneration,
                    head.localBankId, head.localGeneration);
            if (qV6 == nullptr || kV6 == nullptr) {
                return false;
            }
            if (head.qkOwner) {
                continue;
            }
            const OperationRecord *qLoad =
                FindUniqueOperationForSpanGenerations(
                    trace.aivOperations[head.aivId], OperationKind::Load,
                    Stage::V0, "qhat-HK-cache", "q-to-qhat",
                    head.qkCacheSlot, head.qkCacheGeneration,
                    head.localBankId, head.localGeneration);
            const OperationRecord *kLoad =
                FindUniqueOperationForSpanGenerations(
                    trace.aivOperations[head.aivId], OperationKind::Load,
                    Stage::V0, "khat-HK-cache", "k-to-khat",
                    head.qkCacheSlot, head.qkCacheGeneration,
                    head.localBankId, head.localGeneration);
            if (qLoad == nullptr || kLoad == nullptr ||
                CountSyncRecordsBefore(
                    trace.aivSynchronization[head.aivId], SyncAction::Wait,
                    SyncPoint::QkCacheReady, head.qkCacheSlot,
                    head.qkCacheGeneration, Stage::V0, Pipe::Mte2,
                    std::min(qLoad->order, kLoad->order)) == 0U) {
                return false;
            }
        }
        if (ownerCount != 1U || lastCount != 1U || !owner.qkOwner ||
            !last.qkLastConsumer ||
            CountAivSyncRecords(
                trace, SyncAction::Wait, SyncPoint::QkCacheFree,
                owner.qkCacheSlot, owner.qkCacheGeneration, Stage::V0,
                Pipe::Mte2) != 1U ||
            CountAivSyncRecords(
                trace, SyncAction::Set, SyncPoint::QkCacheReady,
                owner.qkCacheSlot, owner.qkCacheGeneration, Stage::V0,
                Pipe::Mte3) != 1U ||
            CountAivSyncRecords(
                trace, SyncAction::Wait, SyncPoint::QkCacheReady,
                owner.qkCacheSlot, owner.qkCacheGeneration, Stage::V0,
                Pipe::Mte2) != ratio - 1U ||
            CountSyncRecords(
                trace.aicSynchronization, SyncAction::Set,
                SyncPoint::QkCacheFree, owner.qkCacheSlot,
                owner.qkCacheGeneration + 1U, Stage::C7,
                Pipe::Control) != 1U) {
            return false;
        }

        std::uint64_t ownerWaitOrder = 0U;
        std::uint64_t readyOrder = 0U;
        std::uint64_t freeOrder = 0U;
        if (!FindUniqueSyncOrder(
                trace.aivSynchronization[owner.aivId], SyncAction::Wait,
                SyncPoint::QkCacheFree, owner.qkCacheSlot,
                owner.qkCacheGeneration, Stage::V0, Pipe::Mte2,
                ownerWaitOrder) ||
            !FindUniqueSyncOrder(
                trace.aivSynchronization[owner.aivId], SyncAction::Set,
                SyncPoint::QkCacheReady, owner.qkCacheSlot,
                owner.qkCacheGeneration, Stage::V0, Pipe::Mte3,
                readyOrder) ||
            !FindUniqueSyncOrder(
                trace.aicSynchronization, SyncAction::Set,
                SyncPoint::QkCacheFree, owner.qkCacheSlot,
                owner.qkCacheGeneration + 1U, Stage::C7, Pipe::Control,
                freeOrder)) {
            return false;
        }
        const OperationRecord *rawQ = FindUniqueOperationForSpanGenerations(
            trace.aivOperations[owner.aivId], OperationKind::Load, Stage::V0,
            "q", "q-to-qhat", owner.workspaceSlot,
            owner.workspaceGeneration, owner.localBankId,
            owner.localGeneration);
        const OperationRecord *rawK = FindUniqueOperationForSpanGenerations(
            trace.aivOperations[owner.aivId], OperationKind::Load, Stage::V0,
            "k", "k-to-khat", owner.workspaceSlot,
            owner.workspaceGeneration, owner.localBankId,
            owner.localGeneration);
        const OperationRecord *qStore = FindUniqueOperationForSpanGenerations(
            trace.aivOperations[owner.aivId], OperationKind::Store, Stage::V0,
            "qhat", "qhat-HK-cache", owner.localBankId,
            owner.localGeneration, owner.qkCacheSlot,
            owner.qkCacheGeneration);
        const OperationRecord *kStore = FindUniqueOperationForSpanGenerations(
            trace.aivOperations[owner.aivId], OperationKind::Store, Stage::V0,
            "khat", "khat-HK-cache", owner.localBankId,
            owner.localGeneration, owner.qkCacheSlot,
            owner.qkCacheGeneration);
        if (rawQ == nullptr || rawK == nullptr || qStore == nullptr ||
            kStore == nullptr ||
            ownerWaitOrder >= rawQ->order || ownerWaitOrder >= rawK->order ||
            qStore->order >= readyOrder || kStore->order >= readyOrder) {
            return false;
        }
        if (CountC7ReadyWaitsBefore(trace.aicSynchronization,
                                    architecture, freeOrder) !=
            expectedReadyWaitsBeforeFree[qkHeadId]) {
            return false;
        }
    }
    if (ratio == 8U) {
        if (qkHeads != 2U || valueHeads != 16U ||
            EmittedHeadGroupCount(valueHeads, qkHeads) != 4U) {
            return false;
        }
        const HeadTask &firstOwner = heads[0U];
        const HeadTask &firstSecondGroup = heads[kHeadsPerGroup];
        const HeadTask &secondOwner = heads[ratio];
        const HeadTask &secondSecondGroup =
            heads[ratio + kHeadsPerGroup];
        if (firstSecondGroup.groupLocalHead != 0U ||
            secondSecondGroup.groupLocalHead != 0U ||
            firstSecondGroup.qkCacheSlot != firstOwner.qkCacheSlot ||
            firstSecondGroup.qkCacheGeneration !=
                firstOwner.qkCacheGeneration ||
            secondSecondGroup.qkCacheSlot != secondOwner.qkCacheSlot ||
            secondSecondGroup.qkCacheGeneration !=
                secondOwner.qkCacheGeneration ||
            firstOwner.qkCacheSlot != 0U ||
            firstOwner.qkCacheGeneration != 0U ||
            secondOwner.qkCacheGeneration != 1U ||
            firstOwner.qkCacheSlot != secondOwner.qkCacheSlot ||
            secondOwner.qkCacheGeneration !=
                firstOwner.qkCacheGeneration + 1U ||
            CountSyncRecords(
                trace.aicSynchronization, SyncAction::Set,
                SyncPoint::QkCacheFree,
                firstOwner.qkCacheSlot,
                firstOwner.qkCacheGeneration + 1U, Stage::C7,
                Pipe::Control) != 1U ||
            CountAivSyncRecords(
                trace, SyncAction::Wait, SyncPoint::QkCacheFree,
                secondOwner.qkCacheSlot, secondOwner.qkCacheGeneration,
                Stage::V0, Pipe::Mte2) != 1U) {
            return false;
        }
    }
    return true;
}

bool CheckQkCacheProtocol(Architecture architecture) noexcept
{
    return CheckQkCacheProtocolCase(architecture, 8U, 8U) &&
           CheckQkCacheProtocolCase(architecture, 8U, 4U) &&
           CheckQkCacheProtocolCase(architecture, 6U, 2U) &&
           CheckQkCacheProtocolCase(architecture, 8U, 2U) &&
           CheckQkCacheProtocolCase(architecture, 16U, 2U);
}

bool CheckMixedValueStorage(Architecture architecture) noexcept
{
    ProposedTilingKey key{};
    key.inputStorage = InputStorage::Bf16;
    key.valueStorage = InputStorage::Fp16;
    MultiHeadAddressTrace trace{};
    if (!CaptureMultiHeadAddressTrace(architecture, 1U, trace, 0U, key)) {
        return false;
    }

    const bool arch22 = architecture == Architecture::Arch22;
    const OperationKind computeKind =
        arch22 ? OperationKind::Mmad
               : OperationKind::MmadQuadrantPackedLhs;
    const char *lhs = arch22 ? "Akk-cube-ready-resident" : "Akk-resident";
    const char *wRhs = arch22 ? "K-beta-g-L1" : "K-beta-g-RHS";
    const char *uRhs = arch22 ? "V-beta-L1" : "V-beta-RHS";
    const char *wL0c = arch22 ? "W-L0C" : "W-fp32-L0C";
    const char *uL0c = arch22 ? "U-L0C" : "U-fp32-L0C";
    const OperationRecord *w = FindUniqueOperation(
        trace.aicOperations, computeKind, Stage::C7, lhs, wL0c);
    const OperationRecord *u = FindUniqueOperation(
        trace.aicOperations, computeKind, Stage::C7, lhs, uL0c);
    const OperationRecord *wOutput = FindUniqueOperation(
        trace.aicOperations, OperationKind::StoreRounded, Stage::C7,
        arch22 ? "W-L0C-valid" : "W-valid-fp32-ld128",
        arch22 ? "W-output" : "W-valid-output-ld128");
    const OperationRecord *uOutput = FindUniqueOperation(
        trace.aicOperations, OperationKind::StoreRounded, Stage::C7,
        arch22 ? "U-L0C-valid" : "U-valid-fp32-ld128",
        arch22 ? "U-output" : "U-valid-output-ld128");
    return w != nullptr && u != nullptr && wOutput != nullptr &&
           uOutput != nullptr && SameName(w->rhsOperand.name, wRhs) &&
           SameName(u->rhsOperand.name, uRhs) &&
           w->lhsStorage == MatrixStorage::Bf16 &&
           u->lhsStorage == MatrixStorage::Bf16 &&
           w->rhsStorage == MatrixStorage::Bf16 &&
           u->rhsStorage == MatrixStorage::Fp16 &&
           wOutput->inputStorage == InputStorage::Bf16 &&
           uOutput->inputStorage == InputStorage::Fp16;
}

bool RunMultiHeadAddressContracts() noexcept
{
    return CheckArch35StaticMutexIdTable() &&
           CheckCompleteQkCohortPartitions() &&
           CheckArch22MultiHeadAddresses(4U) &&
           CheckArch22MultiHeadAddresses(3U) &&
           CheckArch35MultiHeadAddresses() &&
           CheckGroupedQkHeadMapping(Architecture::Arch22) &&
           CheckGroupedQkHeadMapping(Architecture::Arch35) &&
           CheckQkContextAddressModes(Architecture::Arch22) &&
           CheckQkContextAddressModes(Architecture::Arch35) &&
           CheckQkCacheProtocol(Architecture::Arch22) &&
           CheckQkCacheProtocol(Architecture::Arch35) &&
           CheckL0OperandEpochContract(Architecture::Arch22) &&
           CheckL0OperandEpochContract(Architecture::Arch35) &&
           RunTraceCaseForBothAbis(4U, 1U, 1U, 0U, ResolveDense, 2U) &&
           RunTraceCaseForBothAbis(8U, 1U, 1U, 0U, ResolveDense, 2U) &&
           CheckMixedValueStorage(Architecture::Arch22) &&
           CheckMixedValueStorage(Architecture::Arch35);
}

bool RunArch22OperationCase(ResolveChunk resolveChunk,
                            std::uint32_t validRows,
                            PrepareAbi abi, bool useExp2,
                            GateStorage gateStorage,
                            InputStorage inputStorage = InputStorage::Bf16,
                            ScoreStorage scoreStorage = ScoreStorage::Bf16,
                            bool safeGate = false) noexcept
{
    RuntimeTiling tiling{};
    tiling.totalChunks = 1U;
    tiling.headCount = 1U;
    tiling.aicWorkgroupCount = 1U;
    tiling.epsilon = 1.0e-6F;
    tiling.scale = kRuntimeScaleProbeValue;
    tiling.key.abi = abi;
    tiling.key.useExp2 = useExp2;
    tiling.key.gateStorage = gateStorage;
    tiling.key.inputStorage = inputStorage;
    tiling.key.scoreStorage = scoreStorage;
    tiling.key.safeGate = safeGate;
    const WorkspaceSizing sizing = CheckedWorkspaceSizing(
        Architecture::Arch22, tiling.aicWorkgroupCount, 0U);
    if (!sizing.valid) {
        return false;
    }

    OperationTrace vectorOperations{};
    SyncTrace vectorSynchronization{};
    LocalSyncTrace vectorLocalDependencies{};
    MutexTrace vectorMutexes{};
    TraceClock vectorClock{};
    VectorOps vectorOps{};
    vectorOps.trace = &vectorOperations;
    vectorOps.clock = &vectorClock;
    CubeOps unusedCubeOps{};
    SyncLedger vectorSync{&vectorSynchronization, &vectorLocalDependencies,
                          &vectorClock, &vectorMutexes};
    WorkspaceView vectorWorkspace{};
    vectorWorkspace.backingBytes = sizing.totalBytes;
    for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
        RunArchitectureContract(
            Architecture::Arch22, tiling, 0U, CoreRole::Aiv, aiv,
            vectorWorkspace, vectorSync, vectorOps, unusedCubeOps,
            resolveChunk);
    }

    OperationTrace cubeOperations{};
    SyncTrace cubeSynchronization{};
    LocalSyncTrace cubeLocalDependencies{};
    MutexTrace cubeMutexes{};
    TraceClock cubeClock{};
    CubeOps cubeOps{};
    cubeOps.trace = &cubeOperations;
    cubeOps.clock = &cubeClock;
    VectorOps unusedVectorOps{};
    SyncLedger cubeSync{&cubeSynchronization, &cubeLocalDependencies,
                        &cubeClock, &cubeMutexes};
    WorkspaceView cubeWorkspace{};
    cubeWorkspace.backingBytes = sizing.totalBytes;
    RunArchitectureContract(
        Architecture::Arch22, tiling, 0U, CoreRole::Aic, 0U,
        cubeWorkspace, cubeSync, unusedVectorOps, cubeOps, resolveChunk);

    return EmptyMutexTrace(vectorMutexes) && EmptyMutexTrace(cubeMutexes) &&
           CheckArch22VectorOperations(
               vectorOperations, vectorSynchronization,
               vectorLocalDependencies, validRows, abi, useExp2,
               gateStorage, tiling.scale) &&
           CheckArch22CubeOperations(cubeOperations, cubeSynchronization,
                                     cubeLocalDependencies, validRows,
                                     tiling.key);
}

bool RunArch22OperationCasesForBothAbis(ResolveChunk resolveChunk,
                                        std::uint32_t validRows) noexcept
{
    constexpr std::array<GateStorage, 3U> kGateStorages = {{
        GateStorage::Fp16,
        GateStorage::Bf16,
        GateStorage::Fp32,
    }};
    for (GateStorage gateStorage : kGateStorages) {
        if (!RunArch22OperationCase(resolveChunk, validRows,
                                    PrepareAbi::Current, false,
                                    gateStorage) ||
            !RunArch22OperationCase(resolveChunk, validRows,
                                    PrepareAbi::Fused, false, gateStorage) ||
            !RunArch22OperationCase(resolveChunk, validRows,
                                    PrepareAbi::Current, true, gateStorage) ||
            !RunArch22OperationCase(resolveChunk, validRows,
                                    PrepareAbi::Fused, true, gateStorage)) {
            return false;
        }
    }
    return true;
}

bool RunArch35OperationCase(ResolveChunk resolveChunk,
                            std::uint32_t validRows,
                            PrepareAbi abi, bool useExp2,
                            GateStorage gateStorage,
                            InputStorage inputStorage = InputStorage::Bf16,
                            ScoreStorage scoreStorage = ScoreStorage::Bf16,
                            bool safeGate = false) noexcept
{
    RuntimeTiling tiling{};
    tiling.totalChunks = 1U;
    tiling.headCount = 1U;
    tiling.aicWorkgroupCount = 1U;
    tiling.epsilon = 1.0e-6F;
    tiling.scale = kRuntimeScaleProbeValue;
    tiling.key.abi = abi;
    tiling.key.useExp2 = useExp2;
    tiling.key.gateStorage = gateStorage;
    tiling.key.inputStorage = inputStorage;
    tiling.key.scoreStorage = scoreStorage;
    tiling.key.safeGate = safeGate;
    const WorkspaceSizing sizing = CheckedWorkspaceSizing(
        Architecture::Arch35, tiling.aicWorkgroupCount, 0U);
    if (!sizing.valid) {
        return false;
    }

    OperationTrace vectorOperations{};
    SyncTrace vectorSynchronization{};
    LocalSyncTrace vectorLocalDependencies{};
    MutexTrace vectorMutexes{};
    TraceClock vectorClock{};
    VectorOps vectorOps{};
    vectorOps.trace = &vectorOperations;
    vectorOps.clock = &vectorClock;
    CubeOps unusedCubeOps{};
    SyncLedger vectorSync{&vectorSynchronization, &vectorLocalDependencies,
                          &vectorClock, &vectorMutexes};
    WorkspaceView vectorWorkspace{};
    vectorWorkspace.backingBytes = sizing.totalBytes;
    RunArchitectureContract(
        Architecture::Arch35, tiling, 0U, CoreRole::Aiv, 0U,
        vectorWorkspace, vectorSync, vectorOps, unusedCubeOps,
        resolveChunk);

    OperationTrace cubeOperations{};
    SyncTrace cubeSynchronization{};
    LocalSyncTrace cubeLocalDependencies{};
    MutexTrace cubeMutexes{};
    TraceClock cubeClock{};
    CubeOps cubeOps{};
    cubeOps.trace = &cubeOperations;
    cubeOps.clock = &cubeClock;
    VectorOps unusedVectorOps{};
    SyncLedger cubeSync{&cubeSynchronization, &cubeLocalDependencies,
                        &cubeClock, &cubeMutexes};
    WorkspaceView cubeWorkspace{};
    cubeWorkspace.backingBytes = sizing.totalBytes;
    RunArchitectureContract(
        Architecture::Arch35, tiling, 0U, CoreRole::Aic, 0U,
        cubeWorkspace, cubeSync, unusedVectorOps, cubeOps, resolveChunk);

    return CheckArch35VectorOperations(
               vectorOperations, vectorSynchronization,
               vectorLocalDependencies, vectorMutexes, validRows, abi,
               useExp2, gateStorage, tiling.scale) &&
           CheckArch35CubeOperations(cubeOperations, cubeSynchronization,
                                     cubeLocalDependencies, cubeMutexes,
                                     validRows, tiling.key);
}

bool RunArch35OperationCasesForBothAbis(ResolveChunk resolveChunk,
                                        std::uint32_t validRows) noexcept
{
    constexpr std::array<GateStorage, 3U> kGateStorages = {{
        GateStorage::Fp16,
        GateStorage::Bf16,
        GateStorage::Fp32,
    }};
    for (GateStorage gateStorage : kGateStorages) {
        if (!RunArch35OperationCase(resolveChunk, validRows,
                                    PrepareAbi::Current, false,
                                    gateStorage) ||
            !RunArch35OperationCase(resolveChunk, validRows,
                                    PrepareAbi::Fused, false, gateStorage) ||
            !RunArch35OperationCase(resolveChunk, validRows,
                                    PrepareAbi::Current, true, gateStorage) ||
            !RunArch35OperationCase(resolveChunk, validRows,
                                    PrepareAbi::Fused, true, gateStorage)) {
            return false;
        }
    }
    return true;
}

bool RunScoreStorageDtypeCases() noexcept
{
    constexpr PrepareAbi kAbi = PrepareAbi::Current;
    constexpr bool kUseExp2 = true;
    constexpr GateStorage kGateStorage = GateStorage::Bf16;
    return IsSupportedStorageMapping(InputStorage::Fp16,
                                     ScoreStorage::Fp16, false) &&
           IsSupportedStorageMapping(InputStorage::Fp16,
                                     ScoreStorage::Bf16, true) &&
           RunArch22OperationCase(
               ResolveDense, 64U, kAbi, kUseExp2, kGateStorage,
               InputStorage::Fp16, ScoreStorage::Fp16, false) &&
           RunArch22OperationCase(
               ResolveDense, 64U, kAbi, kUseExp2, kGateStorage,
               InputStorage::Fp16, ScoreStorage::Bf16, true) &&
           RunArch35OperationCase(
               ResolveDense, 64U, kAbi, kUseExp2, kGateStorage,
               InputStorage::Fp16, ScoreStorage::Fp16, false) &&
           RunArch35OperationCase(
               ResolveDense, 64U, kAbi, kUseExp2, kGateStorage,
               InputStorage::Fp16, ScoreStorage::Bf16, true);
}

} // namespace
} // namespace kda_prepare_pseudocode

int main()
{
    using namespace kda_prepare_pseudocode;
    if (!CheckPow2EvaluationContract()) {
        return 5;
    }
    if (!CheckRuntimeScaleSemanticContract() ||
        !CheckV0QkNormalizeSemanticContract()) {
        return 7;
    }
    if (!CheckV3SemanticContracts()) {
        return 4;
    }
    if (!RunMultiHeadAddressContracts()) {
        return 6;
    }
    for (std::uint32_t heads = 1U; heads <= 17U; ++heads) {
        if (!RunTraceCaseForBothAbis(heads, 3U, 1U, 0U, nullptr)) {
            return 1;
        }
    }
    if (!RunTraceCaseForBothAbis(3U, 1U, 1U, 0U, ResolveTail16) ||
        !RunTraceCaseForBothAbis(3U, 1U, 1U, 0U, ResolveTail32) ||
        !RunTraceCaseForBothAbis(3U, 1U, 1U, 0U, ResolveTail33)) {
        return 1;
    }
    for (std::uint32_t workgroup = 0U; workgroup < 3U; ++workgroup) {
        if (!RunTraceCaseForBothAbis(9U, 1U, 3U, workgroup, nullptr) ||
            !RunTraceCaseForBothAbis(5U, 3U, 3U, workgroup, nullptr)) {
            return 1;
        }
    }
    if (!RunArch22OperationCasesForBothAbis(ResolveTail1, 1U) ||
        !RunArch22OperationCasesForBothAbis(ResolveTail16, 16U) ||
        !RunArch22OperationCasesForBothAbis(ResolveTail17, 17U) ||
        !RunArch22OperationCasesForBothAbis(ResolveTail32, 32U) ||
        !RunArch22OperationCasesForBothAbis(ResolveTail33, 33U) ||
        !RunArch22OperationCasesForBothAbis(ResolveTail49, 49U) ||
        !RunArch22OperationCasesForBothAbis(ResolveTail63, 63U) ||
        !RunArch22OperationCasesForBothAbis(ResolveDense, 64U)) {
        return 2;
    }
    if (!RunArch35OperationCasesForBothAbis(ResolveTail1, 1U) ||
        !RunArch35OperationCasesForBothAbis(ResolveTail16, 16U) ||
        !RunArch35OperationCasesForBothAbis(ResolveTail17, 17U) ||
        !RunArch35OperationCasesForBothAbis(ResolveTail32, 32U) ||
        !RunArch35OperationCasesForBothAbis(ResolveTail33, 33U) ||
        !RunArch35OperationCasesForBothAbis(ResolveTail49, 49U) ||
        !RunArch35OperationCasesForBothAbis(ResolveTail63, 63U) ||
        !RunArch35OperationCasesForBothAbis(ResolveDense, 64U)) {
        return 3;
    }
    if (!RunScoreStorageDtypeCases()) {
        return 8;
    }
    return 0;
}
