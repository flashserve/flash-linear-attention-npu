/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H
#define PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_tiling_key.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace kda_prepare_pseudocode::arch22 {

inline constexpr bool kVectorDesignCovered = true;

namespace detail {

using Policy = arch22_policy::UbPolicy;
using Private = arch22_policy::V01PrivateLayout;
using Shared = arch22_policy::V01SharedLayout;

inline BufferSpan Subspan(const BufferSpan &parent, const char *name,
                          Offset relativeOffset, Offset bytes)
{
    BufferSpan span = parent;
    span.name = name;
    span.byteOffset += relativeOffset;
    span.byteSize = bytes;
    span.rows = 0U;
    span.columns = 0U;
    span.leadingDimension = 0U;
    span.elementBytes = 0U;
    return span;
}

inline BufferSpan PrivateSpan(const HeadTask &head, const char *name,
                              const Region &region,
                              std::uint64_t generation)
{
    const Offset privateSlot = arch22_policy::PairWave(head.groupLocalHead);
    return {name,
            MemorySpace::Ub,
            static_cast<std::uint64_t>(Policy::PrivateBase(privateSlot)) +
                region.offset,
            region.size,
            head.localBankId,
            generation,
            CoreRole::Aiv,
            head.aivId};
}

inline BufferSpan SharedSpan(const HeadTask &head, const char *name,
                             const Region &region,
                             std::uint64_t generation)
{
    return {name,
            MemorySpace::Ub,
            static_cast<std::uint64_t>(Policy::kShared.offset) + region.offset,
            region.size,
            head.sharedArenaId,
            generation,
            CoreRole::Aiv,
            head.aivId};
}

inline BufferSpan SymbolicGmSpan(const HeadTask &head, const char *name,
                                 std::size_t bytes,
                                 std::uint64_t generation,
                                 std::uint32_t logicalHeadId =
                                     kAllGroupLocalHeads)
{
    BufferSpan span{name, MemorySpace::Gm, 0U, bytes, head.workspaceSlot,
                    generation, CoreRole::Shared, 0U};
    span.logicalHeadId = logicalHeadId == kAllGroupLocalHeads
                             ? head.headId
                             : logicalHeadId;
    return span;
}

constexpr Offset MatrixFootprintBytes(Offset rows, Offset columns,
                                      Offset leadingDimension,
                                      Offset elementBytes)
{
    return rows == 0U || columns == 0U
               ? 0U
               : ((rows - 1U) * leadingDimension + columns) * elementBytes;
}

inline BufferSpan SymbolicGmRows(const HeadTask &head, const char *name,
                                 Offset rows, Offset columns,
                                 Offset leadingDimension,
                                 Offset elementBytes,
                                 std::uint64_t generation)
{
    // PROPOSED 2-D descriptor. byteSize encloses the complete strided view;
    // the matrix fields retain its logical payload and physical row stride.
    BufferSpan span{name,
                    MemorySpace::Gm,
                    0U,
                    MatrixFootprintBytes(rows, columns, leadingDimension,
                                         elementBytes),
                    head.workspaceSlot,
                    generation,
                    CoreRole::Shared,
                    0U,
                    rows,
                    columns,
                    leadingDimension,
                    elementBytes};
    span.logicalHeadId = head.headId;
    return span;
}

inline BufferSpan MatrixRect(const BufferSpan &parent, const char *name,
                             Offset row, Offset column, Offset rows,
                             Offset columns, Offset leadingDimension,
                             Offset elementBytes)
{
    // PROPOSED 2-D descriptor. byteSize encloses the complete strided view;
    // rows/columns retain its logical payload.
    BufferSpan span = parent;
    span.name = name;
    span.byteOffset +=
        (static_cast<std::uint64_t>(row) * leadingDimension + column) *
        elementBytes;
    span.byteSize = MatrixFootprintBytes(rows, columns, leadingDimension,
                                         elementBytes);
    span.rows = rows;
    span.columns = columns;
    span.leadingDimension = leadingDimension;
    span.elementBytes = elementBytes;
    return span;
}

inline bool IsOwnedSelectedHead(const HeadTask &head,
                                const VectorStageArgs &args)
{
    return head.active && head.aivId == args.aivId &&
           (args.selectedGroupLocalHead == kAllGroupLocalHeads ||
           head.groupLocalHead == args.selectedGroupLocalHead);
}

constexpr std::uint32_t ActiveScoreBlocks(std::uint32_t validRows) noexcept
{
    return std::min<std::uint32_t>(
        ShapePolicy::kScoreBlockCount,
        (validRows + ShapePolicy::kScoreBlockRows - 1U) /
            ShapePolicy::kScoreBlockRows);
}

inline bool IsSupportedKey(const ProposedTilingKey &key)
{
    return IsSupportedTilingKey(key);
}

inline bool ValidArgs(const VectorStageArgs &args)
{
    return args.work != nullptr && args.workspace != nullptr &&
           args.sync != nullptr && args.ops != nullptr &&
           args.architecture == Architecture::Arch22 &&
           IsSupportedKey(args.key);
}

inline bool IsPairInvocation(const VectorStageArgs &args) noexcept
{
    return args.selectedGroupLocalHead < kHeadsPerGroup;
}

inline std::uint32_t SelectedPair(const VectorStageArgs &args) noexcept
{
    return arch22_policy::PairWave(args.selectedGroupLocalHead);
}

inline BufferSpan Context(const VectorStageArgs &args, const HeadTask &head)
{
    return args.workspace->Span(WorkspaceRegion::Context, head.workspaceSlot,
                                head.workspaceGeneration);
}

inline BufferSpan Payload(const VectorStageArgs &args, const HeadTask &head)
{
    return args.workspace->Span(WorkspaceRegion::SharedPayload,
                                head.workspaceSlot,
                                head.workspaceGeneration);
}

inline BufferSpan AkkRelay(const VectorStageArgs &args, const HeadTask &head,
                           Offset rows)
{
    if (args.key.abi == PrepareAbi::Current) {
        return SymbolicGmRows(head, "Akk-output-and-C7-relay", rows,
                              ShapePolicy::kBt, ShapePolicy::kBt,
                              ShapePolicy::kStorageBytes,
                              head.workspaceGeneration);
    }
    const BufferSpan relay = Subspan(
        Payload(args, head), "Akk-row-major-relay",
        arch22_policy::WorkspacePolicy::kAkkRowMajor.offset,
        arch22_policy::WorkspacePolicy::kAkkRowMajor.size);
    return MatrixRect(relay, "Akk-row-major-relay-valid", 0U, 0U, rows,
                      ShapePolicy::kBt, ShapePolicy::kBt,
                      ShapePolicy::kStorageBytes);
}

inline void RequireMte2ToMte3SourceFree(const SyncLedger &sync,
                                        Stage stage) noexcept
{
    // PROPOSED local event contract. On the target c220 headers this must be
    // implemented with the verified MTE2_MTE3 HardEvent pair. C2RawReady is a
    // cross-core visibility edge and cannot replace this same-core source-free
    // edge before V3 overwrites payload[0,0x4800) with VCS.
    sync.Local(LocalDependency::Mte2ToMte3SourceFree, stage);
}

inline void RequireMte2ToVectorInputs(const SyncLedger &sync,
                                      Stage stage) noexcept
{
    // PROPOSED local input-ready edge. GM/workspace -> UB MTE2 loads must
    // complete before the stage VF reads those destinations. The concrete
    // c220 event remains a target-version compile gate.
    sync.Local(LocalDependency::Mte2ToVectorInputs, stage);
}

inline void RequireVectorToMte3Outputs(const SyncLedger &sync,
                                       Stage stage) noexcept
{
    // PROPOSED local output-ready edge. The stage VF must finish writing every
    // MTE3 source before any corresponding GM/workspace drain starts. Pair
    // ready tokens only order other cores and cannot replace this edge.
    sync.Local(LocalDependency::VectorToMte3Outputs, stage);
}

// Dependent pseudo-interfaces are not instantiated by the host syntax build.
// They freeze one-VF ordering without claiming concrete c220 intrinsic names.
template <typename Vf>
inline void V0OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    const ProposedTilingKey &key, float epsilon,
                    float lowerBound)
{
    const Offset privateBase = Policy::PrivateBase(
        arch22_policy::PairWave(head.groupLocalHead));
    const Offset sharedBase = Policy::kShared.offset;
    const bool selectiveGate = key.gateMode != GateMode::PrecomputedStep;
    auto gateCoefficient = vf.OneFp32();
    auto dtBias = vf.ZeroFp32Row(ShapePolicy::kK);
    if (selectiveGate) {
        gateCoefficient = vf.Exp(vf.LoadALogScalarOnce(
            privateBase + Private::kALogOrGateAttrs.offset));
        dtBias = vf.LoadDtBiasRow(privateBase + Private::kDtBias.offset);
    }

    auto carry = vf.ZeroFp32Row(ShapePolicy::kK);
    // Phase 1 consumes every gate row and completes the true token scan. For
    // Gate2B the raw source is private [0x8000,0xC000); Q/K norm work cannot
    // claim that range until this loop's last reader finishes.
    for (std::uint32_t row = 0U; row < ShapePolicy::kBt; ++row) {
        if (row >= validRows) {
            vf.StoreFp32Row(sharedBase + Shared::kG.offset, row,
                            vf.ZeroFp32Row(ShapePolicy::kK));
            vf.StoreBetaEffScalar(privateBase + Private::kBetaEff.offset,
                                  row, 0.0F);
            continue;
        }
        auto gateRaw = IsTwoByteGateStorage(key.gateStorage)
                           ? vf.LoadGateRow(
                                 privateBase + Private::kGateRaw2B.offset,
                                 row, key.gateStorage)
                           : vf.LoadGateRow(
                                 sharedBase + Shared::kGateRawFp32.offset, row,
                                 key.gateStorage);
        auto betaRaw = vf.LoadBetaFp32Scalar(
            privateBase + Private::kBetaRaw.offset, row);
        auto betaEff = betaRaw;
        if (key.betaMode == BetaMode::Sigmoid) {
            betaEff = vf.Sigmoid(betaRaw);
        } else if (key.betaMode == BetaMode::TwoSigmoid) {
            betaEff = vf.Mul(2.0F, vf.Sigmoid(betaRaw));
        }

        auto gateStep = gateRaw;
        if (key.gateMode == GateMode::PrecomputedStep) {
            gateStep = vf.Div(gateRaw, vf.Ln2());
        } else {
            const auto x = vf.Add(gateRaw, dtBias);
            if (key.gateMode == GateMode::Softplus) {
                const auto stableSoftplus = vf.Add(
                    vf.Max(x, vf.ZeroFp32()),
                    vf.Log1p(vf.Exp(vf.Neg(vf.Abs(x)))));
                gateStep = vf.Div(
                    vf.Neg(vf.Mul(gateCoefficient, stableSoftplus)),
                    vf.Ln2());
            } else {
                gateStep = vf.Div(
                    vf.Mul(lowerBound,
                           vf.Sigmoid(vf.Mul(gateCoefficient, x))),
                    vf.Ln2());
            }
        }

        carry = vf.Add(carry, gateStep);
        vf.StoreFp32Row(sharedBase + Shared::kG.offset, row, carry);
        vf.StoreBetaEffScalar(privateBase + Private::kBetaEff.offset, row,
                              betaEff);
    }
    vf.StoreGLast(privateBase + Private::kGLast.offset, carry);

    const bool normalizeQk =
        head.qkOwner && key.qkNormMode == QkNormMode::L2;
    // Only the HK owner with L2 enabled claims the aliased norm work region.
    // Identity owners and cache readers retain the 2-byte MTE2 result in place.
    if (normalizeQk) {
        // A concrete implementation needs a true same-V dependency here.
        // PIPE_V is a candidate after c220 verification; PIPE_ALL is forbidden.
        vf.DependNormWorkOnGateLastReader();
    }
    const auto zeroStorage = vf.RoundToInputStorage(
        vf.ClampForInputStorage(vf.ZeroFp32(), key.inputStorage),
        key.inputStorage);
    // Phase 2 changes [0x8000,0x10000) from gate/work to norm scratch and
    // normalizes every Q/K row. It is still part of this one VF invocation.
    for (std::uint32_t row = 0U; row < ShapePolicy::kBt; ++row) {
        if (row >= validRows) {
            vf.StoreStorageRow(privateBase + Private::kQToQPlus.offset, row,
                               zeroStorage, key.inputStorage);
            vf.StoreStorageRow(privateBase + Private::kKToKPlus.offset, row,
                               zeroStorage, key.inputStorage);
            continue;
        }
        if (!normalizeQk) {
            continue;
        }
        auto q = vf.ToFp32(vf.LoadStorageRow(
            privateBase + Private::kQToQPlus.offset, row, key.inputStorage));
        auto k = vf.ToFp32(vf.LoadStorageRow(
            privateBase + Private::kKToKPlus.offset, row, key.inputStorage));
        // Same frozen semantics as Arch35; the work address changes only the
        // reduction implementation, never the denominator formula.
        const auto qHat = vf.L2NormalizeRsqrtSumPlusEpsilonWithWork(
            q, epsilon, privateBase + Private::kV0NormWork.offset);
        const auto kHat = vf.L2NormalizeRsqrtSumPlusEpsilonWithWork(
            k, epsilon, privateBase + Private::kV0NormWork.offset);
        vf.StoreStorageRow(
            privateBase + Private::kQToQPlus.offset, row,
            vf.RoundToInputStorage(
                vf.ClampForInputStorage(qHat, key.inputStorage),
                key.inputStorage),
            key.inputStorage);
        vf.StoreStorageRow(
            privateBase + Private::kKToKPlus.offset, row,
            vf.RoundToInputStorage(
                vf.ClampForInputStorage(kHat, key.inputStorage),
                key.inputStorage),
            key.inputStorage);
    }
}

template <bool UseExp2, typename Vf>
inline void V1OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    InputStorage inputStorage, ScoreStorage scoreStorage)
{
    const Offset privateBase = Policy::PrivateBase(
        arch22_policy::PairWave(head.groupLocalHead));
    const Offset sharedBase = Policy::kShared.offset;
    const std::uint32_t activeBlocks = ActiveScoreBlocks(validRows);
    const float exp2Min = ScoreExp2InputMin(scoreStorage);
    const float exp2Max = ScoreExp2InputMax(scoreStorage);
    const auto zeroScore = vf.RoundToScoreStorage(
        vf.ClampForScoreStorage(vf.ZeroFp32(), scoreStorage), scoreStorage);

    for (std::uint32_t row = 0U; row < ShapePolicy::kBt; ++row) {
        if (row < validRows) {
            const auto qHat = vf.ToFp32(vf.LoadStorageRow(
                privateBase + Private::kQToQPlus.offset, row, inputStorage));
            const auto kHat = vf.ToFp32(vf.LoadStorageRow(
                privateBase + Private::kKToKPlus.offset, row, inputStorage));
            const auto g = vf.LoadFp32Row(sharedBase + Shared::kG.offset, row);
            const std::uint32_t owner = row / ShapePolicy::kScoreBlockRows;
            const std::uint32_t ownerBegin =
                owner * ShapePolicy::kScoreBlockRows;
            const std::uint32_t ownerEnd = std::min(
                ownerBegin + ShapePolicy::kScoreBlockRows, validRows);
            const std::uint32_t ownerReference =
                ownerBegin + (ownerEnd - ownerBegin) / 2U;
            // Direct G row views avoid a separate 2 KiB Gref allocation.
            const auto ownerRef = vf.LoadFp32Row(
                sharedBase + Shared::kG.offset, ownerReference);
            const auto plusFactor = EvaluatePow2<UseExp2>(
                vf, vf.Sub(g, ownerRef), exp2Min, exp2Max);

            for (std::uint32_t s = 0U;
                 s < ShapePolicy::kScoreBlockCount; ++s) {
                const std::uint32_t physicalEnd =
                    ShapePolicy::kPrefixRows[s];
                const std::uint32_t logicalEnd =
                    ShapePolicy::LogicalPrefixRows(s, validRows);
                if (row >= physicalEnd) {
                    continue;
                }
                auto value = zeroScore;
                if (s < activeBlocks && row < logicalEnd) {
                    const std::uint32_t begin =
                        s * ShapePolicy::kScoreBlockRows;
                    const std::uint32_t end = std::min(
                        begin + ShapePolicy::kScoreBlockRows, validRows);
                    const std::uint32_t referenceRow =
                        begin + (end - begin) / 2U;
                    const auto reference = vf.LoadFp32Row(
                        sharedBase + Shared::kG.offset, referenceRow);
                    const auto factor = EvaluatePow2<UseExp2>(
                        vf, vf.Sub(reference, g), exp2Min, exp2Max);
                    value = vf.RoundToScoreStorage(
                        vf.ClampForScoreStorage(vf.Mul(kHat, factor),
                                                scoreStorage),
                        scoreStorage);
                }
                vf.StoreStorageRow(
                    privateBase + Private::kKMinus[s].offset, row, value,
                    scoreStorage);
            }

            vf.StoreStorageRow(
                privateBase + Private::kQToQPlus.offset, row,
                vf.RoundToScoreStorage(
                    vf.ClampForScoreStorage(vf.Mul(qHat, plusFactor),
                                            scoreStorage),
                    scoreStorage),
                scoreStorage);
            vf.StoreStorageRow(
                privateBase + Private::kKToKPlus.offset, row,
                vf.RoundToScoreStorage(
                    vf.ClampForScoreStorage(vf.Mul(kHat, plusFactor),
                                            scoreStorage),
                    scoreStorage),
                scoreStorage);
        } else {
            vf.StoreStorageRow(privateBase + Private::kQToQPlus.offset, row,
                               zeroScore, scoreStorage);
            vf.StoreStorageRow(privateBase + Private::kKToKPlus.offset, row,
                               zeroScore, scoreStorage);
            for (std::uint32_t s = 0U;
                 s < ShapePolicy::kScoreBlockCount; ++s) {
                if (row < ShapePolicy::kPrefixRows[s]) {
                    vf.StoreStorageRow(
                        privateBase + Private::kKMinus[s].offset, row,
                        zeroScore, scoreStorage);
                }
            }
        }
    }
}

template <typename Vf>
inline void V3OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    PrepareAbi abi, InputStorage inputStorage, float scale)
{
    const Offset privateBase = Policy::PrivateBase(
        arch22_policy::PairWave(head.groupLocalHead));
    for (std::uint32_t row = 0U; row < ShapePolicy::kBt; ++row) {
        const std::uint32_t band = row / ShapePolicy::kScoreBlockRows;
        const std::uint32_t rowInBand =
            row % ShapePolicy::kScoreBlockRows;
        for (std::uint32_t col = 0U; col < ShapePolicy::kBt; ++col) {
            const bool readAqk =
                V3AqkRawReadRequired(validRows, row, col);
            const bool readAkk =
                V3AkkRawReadRequired(validRows, row, col);
            const auto rawAqk = readAqk
                                    ? vf.LoadCompactRawAqk(
                                          privateBase + arch22_policy::
                                                            V3PrivateLayout::
                                                                kCompactRaw.offset,
                                          band, rowInBand, col)
                                    : vf.ZeroFp32();
            const auto rawAkk = readAkk
                                    ? vf.LoadCompactRawAkk(
                                          privateBase + arch22_policy::
                                                            V3PrivateLayout::
                                                                kCompactRaw.offset,
                                          band, rowInBand, col)
                                    : vf.ZeroFp32();
            const auto aqk = readAqk ? vf.Mul(scale, rawAqk)
                                     : vf.ZeroFp32();
            const auto lkk = readAkk
                                 ? vf.Mul(vf.LoadBetaEff(
                                               privateBase +
                                                   arch22_policy::
                                                       V3PrivateLayout::
                                                           kBetaEff.offset,
                                               row),
                                           rawAkk)
                                 : vf.ZeroFp32();
            const auto aqkStorage = vf.RoundToInputStorage(
                vf.ClampForInputStorage(aqk, inputStorage), inputStorage);
            vf.StoreStorageMatrix(
                privateBase +
                    arch22_policy::V3PrivateLayout::kAqkStorage.offset,
                row, col, ShapePolicy::kBt, aqkStorage, inputStorage);
            vf.StoreLkkOrIdentityPaddingAt(
                privateBase + arch22_policy::V3PrivateLayout::kLkk.offset,
                row, col, lkk, validRows);
        }
    }

    // All compact source rows have now been consumed. Only after this exact
    // V-pipe dependency may [0xD000,0x12000) change to work/Akk/reserve.
    vf.DependLateOutputsOnCompactRawLastReader();
    vf.InvertTwo32By32LeavesWithFixedColumnScan(privateBase);
    vf.MaterializeX0X1AndBAtFinalOffsets(privateBase);
    // Materialize only the stable q00/q01/q11 rectangles that V3 drains.
    // The q10 rectangle is neither initialized nor read here; C5 is its sole
    // producer. Rows outside validRows are supplied by C7's final L1 fill.
    for (std::uint32_t row = 0U; row < validRows; ++row) {
        for (std::uint32_t col = 0U; col < ShapePolicy::kBt; ++col) {
            if (!V3StableAkkWriteRequired(Architecture::Arch22, abi,
                                          validRows, row, col)) {
                continue;
            }
            const auto fp32 = vf.LoadStableAkkOrZeroColumnPadding(
                privateBase, row, col, validRows);
            const auto value = vf.RoundToInputStorage(
                vf.ClampForInputStorage(fp32, inputStorage), inputStorage);
            vf.StoreStorageMatrix(
                privateBase +
                    arch22_policy::V3PrivateLayout::kAkkRowMajor.offset,
                row, col, ShapePolicy::kBt, value, inputStorage);
        }
    }
}

template <bool UseExp2, typename Vf>
inline void V6OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    PrepareAbi abi, InputStorage inputStorage,
                    InputStorage valueStorage, float scale)
{
    const Offset privateBase = Policy::PrivateBase(
        arch22_policy::PairWave(head.groupLocalHead));
    const Offset sharedBase = Policy::kShared.offset;
    const std::uint32_t rhsRows =
        validRows > 32U ? ShapePolicy::kBt : 32U;
    const auto zeroQkStorage = vf.RoundToInputStorage(
        vf.ClampForInputStorage(vf.ZeroFp32(), inputStorage), inputStorage);
    const auto zeroValueStorage = vf.RoundToInputStorage(
        vf.ClampForInputStorage(vf.ZeroFp32(), valueStorage), valueStorage);
    for (std::uint32_t row = 0U; row < rhsRows; ++row) {
        if (row >= validRows) {
            vf.StoreZeroV6PrivateRows(privateBase, row, zeroQkStorage,
                                      zeroValueStorage, inputStorage,
                                      valueStorage);
            continue;
        }
        const auto qHat = vf.ToFp32(vf.LoadStorageRow(
            privateBase +
                arch22_policy::V6PrivateLayout::kQHatToQg.offset,
            row, inputStorage));
        const auto kHat = vf.ToFp32(vf.LoadStorageRow(
            privateBase +
                arch22_policy::V6PrivateLayout::kKHatToKg.offset,
            row, inputStorage));
        const auto v = vf.ToFp32(vf.LoadStorageRow(
            privateBase +
                arch22_policy::V6PrivateLayout::kVToVBeta.offset,
            row, valueStorage));
        const auto g = vf.LoadFp32Row(
            sharedBase + arch22_policy::V6SharedLayout::kG.offset, row);
        const auto gLast = vf.LoadFp32Row(
            sharedBase + arch22_policy::V6SharedLayout::kG.offset,
            validRows - 1U);
        const auto beta = vf.LoadBetaEff(
            privateBase + arch22_policy::V6PrivateLayout::kBetaEff.offset,
            row);
        const auto expG = EvaluatePow2<UseExp2>(
            vf, g, kDirectExp2InputMin, kDirectExp2InputMax);
        const auto expGLastMinusG = EvaluatePow2<UseExp2>(
            vf, vf.Sub(gLast, g), kDirectExp2InputMin,
            kDirectExp2InputMax);

        const auto qgStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.Mul(qHat, expG), inputStorage),
            inputStorage);
        const auto kgStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.Mul(kHat, expGLastMinusG),
                                    inputStorage),
            inputStorage);
        const auto kGateStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.Mul(kHat, expG), inputStorage),
            inputStorage);
        // K_beta_g has two required storage boundaries.
        const auto kBetaStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(
                vf.Mul(beta, vf.ToFp32(kGateStorage)), inputStorage),
            inputStorage);
        const auto vBetaStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.Mul(beta, v), valueStorage),
            valueStorage);

        auto qOutput = qgStorage;
        if (abi == PrepareAbi::Fused) {
            // QgScaled consumes the already rounded qgStorage value and
            // replaces it at the same private address.
            qOutput = vf.RoundToInputStorage(
                vf.ClampForInputStorage(
                    vf.Mul(scale, vf.ToFp32(qgStorage)), inputStorage),
                inputStorage);
        }
        vf.StoreStorageRow(
            privateBase +
                arch22_policy::V6PrivateLayout::kQHatToQg.offset,
            row, qOutput, inputStorage);
        vf.StoreStorageRow(
            privateBase +
                arch22_policy::V6PrivateLayout::kKHatToKg.offset,
            row, kgStorage, inputStorage);
        vf.StoreStorageRow(
            privateBase +
                arch22_policy::V6PrivateLayout::kVToVBeta.offset,
            row, vBetaStorage, valueStorage);
        vf.StoreStorageRow(
            privateBase + arch22_policy::V6PrivateLayout::kKBetaG.offset,
            row, kBetaStorage, inputStorage);
    }
}

} // namespace detail

inline void RunV0(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args) || !detail::IsPairInvocation(args)) {
        return;
    }
    const std::uint32_t pair = detail::SelectedPair(args);
    const std::uint64_t collectiveGeneration =
        PairCollectiveGenerationFor(args.work->group, pair);
    // mode=0x2: both AIVs consume the same pair credit exactly once. An AIV
    // with no active partner head still performs this wait but touches no GM.
    args.sync->AivWaitPair(SyncPoint::SlotFree, pair,
                           collectiveGeneration, args.aivId, Stage::V0,
                           Pipe::Control);
    bool selected = false;
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedSelectedHead(head, args)) {
            continue;
        }
        selected = true;
        const std::uint64_t sharedGeneration =
            SharedGenerationFor(head, SharedArenaUse::V01);
        args.sync->Wait(SyncPoint::LocalBankFree, head.localBankId,
                        head.localGeneration, Stage::V0, Pipe::Mte2);
        args.sync->Wait(SyncPoint::SharedArenaFree, head.sharedArenaId,
                        sharedGeneration, Stage::V0, Pipe::Mte2);
        if (head.qkOwner) {
            args.sync->Wait(SyncPoint::QkCacheFree, head.qkCacheSlot,
                            head.qkCacheGeneration, Stage::V0,
                            Pipe::Mte2);
        } else {
            // A workspace generation state, not a consumptive one-shot flag:
            // every mapped HV may acquire the same ready publication.
            args.sync->Wait(SyncPoint::QkCacheReady, head.qkCacheSlot,
                            head.qkCacheGeneration, Stage::V0,
                            Pipe::Mte2);
        }

        const Offset validRows = args.work->group.chunk.validRows;
        const Offset tokenBytes = validRows * ShapePolicy::kK *
                                  ShapePolicy::kStorageBytes;
        const Offset gateBytes =
            validRows * ShapePolicy::kK *
            (IsTwoByteGateStorage(args.key.gateStorage)
                 ? ShapePolicy::kStorageBytes
                 : ShapePolicy::kFp32Bytes);
        const BufferSpan qkCache = args.workspace->Span(
            WorkspaceRegion::Context, head.qkCacheSlot,
            head.qkCacheGeneration);
        const BufferSpan qSource =
            head.qkOwner
                ? detail::SymbolicGmSpan(
                      head, "q", tokenBytes, head.workspaceGeneration,
                      head.qkHeadId)
                : detail::Subspan(
                      qkCache, "qhat-HK-cache",
                      arch22_policy::WorkspacePolicy::kQHatContext.offset,
                      tokenBytes);
        const BufferSpan kSource =
            head.qkOwner
                ? detail::SymbolicGmSpan(
                      head, "k", tokenBytes, head.workspaceGeneration,
                      head.qkHeadId)
                : detail::Subspan(
                      qkCache, "khat-HK-cache",
                      arch22_policy::WorkspacePolicy::kKHatContext.offset,
                      tokenBytes);
        args.ops->Load(
            Stage::V0,
            qSource,
            detail::PrivateSpan(
                head, "q-to-qhat",
                {arch22_policy::V01PrivateLayout::kQToQPlus.offset,
                 tokenBytes},
                head.localGeneration));
        args.ops->Load(
            Stage::V0,
            kSource,
            detail::PrivateSpan(
                head, "k-to-khat",
                {arch22_policy::V01PrivateLayout::kKToKPlus.offset,
                 tokenBytes},
                head.localGeneration));
        if (IsTwoByteGateStorage(args.key.gateStorage)) {
            args.ops->Load(
                Stage::V0,
                detail::SymbolicGmSpan(head, "gate-2b", gateBytes,
                                       head.workspaceGeneration),
                detail::PrivateSpan(
                    head, "gate-2b",
                    {arch22_policy::V01PrivateLayout::kGateRaw2B.offset,
                     gateBytes},
                    head.localGeneration));
        } else {
            args.ops->Load(
                Stage::V0,
                detail::SymbolicGmSpan(head, "gate-fp32", gateBytes,
                                       head.workspaceGeneration),
                detail::SharedSpan(
                    head, "gate-fp32-to-G",
                    {arch22_policy::V01SharedLayout::kGateRawFp32.offset,
                     gateBytes},
                    sharedGeneration));
        }
        const Offset betaBytes = validRows * ShapePolicy::kFp32Bytes;
        args.ops->Load(
            Stage::V0,
            detail::SymbolicGmSpan(head, "beta-fp32", betaBytes,
                                   head.workspaceGeneration),
            detail::PrivateSpan(
                head, "beta-raw",
                {arch22_policy::V01PrivateLayout::kBetaRaw.offset,
                 betaBytes},
                head.localGeneration));
        if (args.key.gateMode != GateMode::PrecomputedStep) {
            const BufferSpan dtBias = detail::PrivateSpan(
                head, "dt-bias", arch22_policy::V01PrivateLayout::kDtBias,
                head.localGeneration);
            if (args.hasDtBias) {
                args.ops->Load(
                    Stage::V0,
                    detail::SymbolicGmSpan(
                        head, "dt-bias",
                        ShapePolicy::kK * ShapePolicy::kFp32Bytes,
                        head.workspaceGeneration),
                    dtBias);
            } else {
                args.ops->Zero(Stage::V0, dtBias);
            }
            args.ops->Load(
                Stage::V0,
                detail::SymbolicGmSpan(head, "A-log",
                                       ShapePolicy::kFp32Bytes,
                                       head.workspaceGeneration),
                detail::PrivateSpan(
                    head, "A-log",
                    {arch22_policy::V01PrivateLayout::kALogOrGateAttrs.offset,
                     ShapePolicy::kFp32Bytes},
                    head.localGeneration));
        }

        detail::RequireMte2ToVectorInputs(*args.sync, Stage::V0);
        // Exactly one VF. Gate/cumsum is phase 1; when this head owns an L2
        // cohort, a verified V-pipe dependency precedes phase-2 norm work at
        // the aliased address. Other paths retain their MTE2 Q/K rows in place.
        args.ops->RunVf(Stage::V0, head);
        detail::RequireVectorToMte3Outputs(*args.sync, Stage::V0);

        const BufferSpan context = detail::Context(args, head);
        if (head.qkOwner) {
            args.ops->Store(
                Stage::V0,
                detail::PrivateSpan(
                    head, "qhat",
                    {arch22_policy::V01PrivateLayout::kQToQPlus.offset,
                     tokenBytes},
                    head.localGeneration),
                detail::Subspan(
                    qkCache, "qhat-HK-cache",
                    arch22_policy::WorkspacePolicy::kQHatContext.offset,
                    tokenBytes));
            args.ops->Store(
                Stage::V0,
                detail::PrivateSpan(
                    head, "khat",
                    {arch22_policy::V01PrivateLayout::kKToKPlus.offset,
                     tokenBytes},
                    head.localGeneration),
                detail::Subspan(
                    qkCache, "khat-HK-cache",
                    arch22_policy::WorkspacePolicy::kKHatContext.offset,
                    tokenBytes));
        }
        args.ops->Store(
            Stage::V0,
            detail::PrivateSpan(
                head, "beta-eff",
                {arch22_policy::V01PrivateLayout::kBetaEff.offset,
                 ShapePolicy::kBetaEffBytes},
                head.localGeneration),
            detail::Subspan(
                context, "beta-eff-context",
                arch22_policy::WorkspacePolicy::kBetaEffContext.offset,
                ShapePolicy::kBetaEffBytes));
        const Offset gBytes =
            validRows * ShapePolicy::kK * ShapePolicy::kFp32Bytes;
        if (args.key.abi == PrepareAbi::Current) {
            // Current already exposes G as gk. Reuse that one GM copy in V6
            // instead of also materializing the same Vector data in context.
            args.ops->Store(
                Stage::V0,
                detail::SharedSpan(
                    head, "G-output",
                    {arch22_policy::V01SharedLayout::kG.offset, gBytes},
                    sharedGeneration),
                detail::SymbolicGmRows(
                    head, "gk-output", validRows, ShapePolicy::kK,
                    ShapePolicy::kK, ShapePolicy::kFp32Bytes,
                    head.workspaceGeneration));
        } else {
            args.ops->Store(
                Stage::V0,
                detail::SharedSpan(
                    head, "G-context-source",
                    {arch22_policy::V01SharedLayout::kG.offset, gBytes},
                    sharedGeneration),
                detail::Subspan(
                    context, "G-context",
                    arch22_policy::WorkspacePolicy::kGContext.offset,
                    gBytes));
        }
        if (head.qkOwner) {
            args.sync->Set(SyncPoint::QkCacheReady, head.qkCacheSlot,
                           head.qkCacheGeneration, Stage::V0, Pipe::Mte3);
        }
        args.sync->Set(SyncPoint::V0ContextReady, head.workspaceSlot,
                       head.workspaceGeneration, Stage::V0, Pipe::Mte3);
        args.sync->Set(SyncPoint::V0BetaReady, head.workspaceSlot,
                       head.workspaceGeneration, Stage::V0, Pipe::Mte3);
        args.sync->Set(SyncPoint::V0ExportDone, head.localBankId,
                       head.localGeneration, Stage::V0, Pipe::Mte3);
    }
    (void)selected;
}

inline void RunV1(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args) || !detail::IsPairInvocation(args)) {
        return;
    }
    const std::uint32_t pair = detail::SelectedPair(args);
    const std::uint64_t collectiveGeneration =
        PairCollectiveGenerationFor(args.work->group, pair);
    bool selected = false;
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedSelectedHead(head, args)) {
            continue;
        }
        selected = true;
        const std::uint64_t sharedGeneration =
            SharedGenerationFor(head, SharedArenaUse::V01);
        args.sync->Wait(SyncPoint::V0ExportDone, head.localBankId,
                        head.localGeneration, Stage::V1, Pipe::Vector);
        // Exactly one VF. Host dispatch specializes V1OneVf<useExp2>; Gref is
        // a SHARED G row view, and V0 work is renamed to all four Kminus
        // prefixes without any UB position move.
        args.ops->RunVf(Stage::V1, head,
                        ResolvePow2Primitive(args.key.useExp2));
        detail::RequireVectorToMte3Outputs(*args.sync, Stage::V1);
        args.sync->Set(SyncPoint::SharedArenaFree, head.sharedArenaId,
                       sharedGeneration + 1U, Stage::V1, Pipe::Vector);
        args.ops->Store(
            Stage::V1,
            detail::PrivateSpan(
                head, "packed-score",
                arch22_policy::V01PrivateLayout::kPackedScore,
                head.localGeneration),
            detail::Subspan(detail::Payload(args, head), "packed-score", 0U,
                            ShapePolicy::kScorePayloadBytes));
    }
    // Each AIV arrives once per pair. selected=false is the mandatory dummy
    // participant for an odd tail pair and performs no address calculation.
    args.sync->AivArrivePair(SyncPoint::V1ScoreReady, pair,
                             collectiveGeneration, args.aivId, selected,
                             Stage::V1, Pipe::Mte3);
}

inline void RunV3(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args) || !detail::IsPairInvocation(args)) {
        return;
    }
    const std::uint32_t pair = detail::SelectedPair(args);
    const std::uint64_t collectiveGeneration =
        PairCollectiveGenerationFor(args.work->group, pair);
    args.sync->AivWaitPair(SyncPoint::C2RawReady, pair,
                           collectiveGeneration, args.aivId, Stage::V3,
                           Pipe::Mte2);
    bool selected = false;
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedSelectedHead(head, args)) {
            continue;
        }
        selected = true;
        const BufferSpan payload = detail::Payload(args, head);
        const Offset validRows = args.work->group.chunk.validRows;
        const std::uint32_t activeBlocks =
            detail::ActiveScoreBlocks(validRows);
        for (std::uint32_t s = 0U; s < activeBlocks; ++s) {
            const Region aqk =
                arch22_policy::WorkspacePolicy::kRelayRawAqk[s];
            const Region akk =
                arch22_policy::WorkspacePolicy::kRelayRawAkk[s];
            args.ops->Load(
                Stage::V3,
                detail::Subspan(payload, "compact-Aqk-band", aqk.offset,
                                aqk.size),
                detail::PrivateSpan(
                    head, "compact-Aqk-band",
                    {arch22_policy::V3PrivateLayout::kCompactRaw.offset +
                         aqk.offset,
                     aqk.size},
                    head.localGeneration));
            args.ops->Load(
                Stage::V3,
                detail::Subspan(payload, "compact-Akk-band", akk.offset,
                                akk.size),
                detail::PrivateSpan(
                    head, "compact-Akk-band",
                    {arch22_policy::V3PrivateLayout::kCompactRaw.offset +
                         akk.offset,
                     akk.size},
                    head.localGeneration));
        }
        const BufferSpan context = detail::Context(args, head);
        args.sync->Wait(SyncPoint::V0BetaReady, head.workspaceSlot,
                        head.workspaceGeneration, Stage::V3, Pipe::Mte2);
        args.ops->Load(
            Stage::V3,
            detail::Subspan(
                context, "beta-eff-context",
                arch22_policy::WorkspacePolicy::kBetaEffContext.offset,
                ShapePolicy::kBetaEffBytes),
            detail::PrivateSpan(
                head, "beta-eff",
                {arch22_policy::V3PrivateLayout::kBetaEff.offset,
                 ShapePolicy::kBetaEffBytes},
                head.localGeneration));

        detail::RequireMte2ToVectorInputs(*args.sync, Stage::V3);
        // Exactly one VF. Runtime scale is an explicit semantic input and is
        // applied once to Aqk; the high compact source is consumed before its
        // address changes to late work/Akk storage.
        args.ops->RunVf(Stage::V3, head, args.scale, RuntimeScaleUse::Aqk,
                        1U);
        detail::RequireMte2ToMte3SourceFree(*args.sync, Stage::V3);
        detail::RequireVectorToMte3Outputs(*args.sync, Stage::V3);
        if (validRows > 32U) {
            args.ops->Store(
                Stage::V3,
                detail::PrivateSpan(head, "X0",
                                    arch22_policy::V3PrivateLayout::kX0,
                                    head.localGeneration),
                detail::Subspan(payload, "VCS-X0",
                                arch22_policy::WorkspacePolicy::kVcsX0.offset,
                                arch22_policy::WorkspacePolicy::kVcsX0.size));
            args.ops->Store(
                Stage::V3,
                detail::PrivateSpan(head, "X1",
                                    arch22_policy::V3PrivateLayout::kX1,
                                    head.localGeneration),
                detail::Subspan(payload, "VCS-X1",
                                arch22_policy::WorkspacePolicy::kVcsX1.offset,
                                arch22_policy::WorkspacePolicy::kVcsX1.size));
            args.ops->Store(
                Stage::V3,
                detail::PrivateSpan(head, "B",
                                    arch22_policy::V3PrivateLayout::kB,
                                    head.localGeneration),
                detail::Subspan(payload, "VCS-B",
                                arch22_policy::WorkspacePolicy::kVcsB.offset,
                                arch22_policy::WorkspacePolicy::kVcsB.size));
        }
        const BufferSpan akkPrivate = detail::PrivateSpan(
            head, "Akk-row-major",
            arch22_policy::V3PrivateLayout::kAkkRowMajor,
            head.localGeneration);

        args.ops->Store(
            Stage::V3,
            detail::PrivateSpan(
                head, "Aqk-storage",
                {arch22_policy::V3PrivateLayout::kAqkStorage.offset,
                 validRows * ShapePolicy::kBt *
                     ShapePolicy::kStorageBytes},
                head.localGeneration),
            detail::SymbolicGmRows(
                head, "Aqk-output", validRows, ShapePolicy::kBt,
                ShapePolicy::kBt, ShapePolicy::kStorageBytes,
                head.workspaceGeneration));
        // V3 owns only the stable quadrants. Do not write a zero q10 that C5
        // would immediately overwrite; each GM cell has exactly one producer.
        constexpr Offset kQuadrant = 32U;
        const Offset top = std::min(validRows, kQuadrant);
        const Offset bottom =
            validRows > kQuadrant ? validRows - kQuadrant : 0U;
        const BufferSpan akkRelay = detail::AkkRelay(args, head, validRows);
        args.ops->Store(
            Stage::V3,
            detail::MatrixRect(akkPrivate, "Akk-q00-row-major", 0U, 0U,
                               top, kQuadrant, ShapePolicy::kBt,
                               ShapePolicy::kStorageBytes),
            detail::MatrixRect(akkRelay, "Akk-q00-relay", 0U, 0U, top,
                               kQuadrant, ShapePolicy::kBt,
                               ShapePolicy::kStorageBytes));
        if (args.key.abi == PrepareAbi::Current || validRows > kQuadrant) {
            // Current needs q01 as a public Akk output. Fused relays this
            // known zero quadrant only for the full-matrix conversion; its
            // top-only path does not materialize an unconsumed q01.
            args.ops->Store(
                Stage::V3,
                detail::MatrixRect(
                    akkPrivate, "Akk-q01-row-major", 0U, kQuadrant, top,
                    kQuadrant, ShapePolicy::kBt,
                    ShapePolicy::kStorageBytes),
                detail::MatrixRect(
                    akkRelay, "Akk-q01-relay", 0U, kQuadrant, top,
                    kQuadrant, ShapePolicy::kBt,
                    ShapePolicy::kStorageBytes));
        }
        if (bottom != 0U) {
            args.ops->Store(
                Stage::V3,
                detail::MatrixRect(
                    akkPrivate, "Akk-q11-row-major", kQuadrant, kQuadrant,
                    bottom, kQuadrant, ShapePolicy::kBt,
                    ShapePolicy::kStorageBytes),
                detail::MatrixRect(
                    akkRelay, "Akk-q11-relay", kQuadrant, kQuadrant,
                    bottom, kQuadrant, ShapePolicy::kBt,
                    ShapePolicy::kStorageBytes));
        }
        args.sync->Set(SyncPoint::V3LocalSourceFree, head.localBankId,
                       head.localGeneration, Stage::V3, Pipe::Mte3);
    }
    args.sync->AivArrivePair(SyncPoint::V3VcsReady, pair,
                             collectiveGeneration, args.aivId, selected,
                             Stage::V3, Pipe::Mte3);
}

inline void RunV6(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args) || !detail::IsPairInvocation(args)) {
        return;
    }
    const std::uint32_t pair = detail::SelectedPair(args);
    const std::uint64_t collectiveGeneration =
        PairCollectiveGenerationFor(args.work->group, pair);
    args.sync->AivWaitPair(SyncPoint::C4PayloadFree, pair,
                           collectiveGeneration, args.aivId, Stage::V6,
                           Pipe::Mte3);
    bool selected = false;
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedSelectedHead(head, args)) {
            continue;
        }
        selected = true;
        const std::uint64_t sharedGeneration =
            SharedGenerationFor(head, SharedArenaUse::V6);
        args.sync->Wait(SyncPoint::V0ContextReady, head.workspaceSlot,
                        head.workspaceGeneration, Stage::V6, Pipe::Mte2);
        args.sync->Wait(SyncPoint::V3LocalSourceFree, head.localBankId,
                        head.localGeneration, Stage::V6, Pipe::Mte2);
        args.sync->Wait(SyncPoint::SharedArenaFree, head.sharedArenaId,
                        sharedGeneration, Stage::V6, Pipe::Mte2);

        const BufferSpan context = detail::Context(args, head);
        // Every mapped HV reloads Qhat/Khat from the owner copy. C7 aggregates
        // the pair V6RhsReady publications before it releases the cache;
        // non-owner context Q/K ranges remain unused.
        const BufferSpan qkContext = args.workspace->Span(
            WorkspaceRegion::Context, head.qkCacheSlot,
            head.qkCacheGeneration);
        const Offset validRows = args.work->group.chunk.validRows;
        const Offset qkBytes = validRows * ShapePolicy::kK *
                               ShapePolicy::kStorageBytes;
        const Offset gBytes =
            validRows * ShapePolicy::kK * ShapePolicy::kFp32Bytes;
        args.ops->Load(
            Stage::V6,
            detail::Subspan(
                qkContext, "qhat-HK-cache",
                arch22_policy::WorkspacePolicy::kQHatContext.offset,
                qkBytes),
            detail::PrivateSpan(
                head, "qhat-to-qg",
                {arch22_policy::V6PrivateLayout::kQHatToQg.offset, qkBytes},
                head.localGeneration));
        args.ops->Load(
            Stage::V6,
            detail::Subspan(
                qkContext, "khat-HK-cache",
                arch22_policy::WorkspacePolicy::kKHatContext.offset,
                qkBytes),
            detail::PrivateSpan(
                head, "khat-to-kg",
                {arch22_policy::V6PrivateLayout::kKHatToKg.offset, qkBytes},
                head.localGeneration));
        const BufferSpan gSource =
            args.key.abi == PrepareAbi::Current
                ? detail::SymbolicGmRows(
                      head, "gk-output-reuse", validRows, ShapePolicy::kK,
                      ShapePolicy::kK, ShapePolicy::kFp32Bytes,
                      head.workspaceGeneration)
                : detail::Subspan(
                      context, "G-context",
                      arch22_policy::WorkspacePolicy::kGContext.offset,
                      gBytes);
        args.ops->Load(
            Stage::V6, gSource,
            detail::SharedSpan(
                head, "G",
                {arch22_policy::V6SharedLayout::kG.offset, gBytes},
                sharedGeneration));
        args.ops->Load(
            Stage::V6,
            detail::Subspan(
                context, "beta-eff-context",
                arch22_policy::WorkspacePolicy::kBetaEffContext.offset,
                ShapePolicy::kBetaEffBytes),
            detail::PrivateSpan(
                head, "beta-eff",
                {arch22_policy::V6PrivateLayout::kBetaEff.offset,
                 ShapePolicy::kBetaEffBytes},
                head.localGeneration));
        const Offset tokenBytes = validRows * ShapePolicy::kV *
                                  ShapePolicy::kStorageBytes;
        args.ops->Load(
            Stage::V6,
            detail::SymbolicGmSpan(head, "V", tokenBytes,
                                   head.workspaceGeneration),
            detail::PrivateSpan(
                head, "V-to-Vbeta",
                {arch22_policy::V6PrivateLayout::kVToVBeta.offset,
                 tokenBytes},
                head.localGeneration));

        detail::RequireMte2ToVectorInputs(*args.sync, Stage::V6);
        // Exactly one VF, specialized as V6OneVf<useExp2> with independent
        // q/k and value storage. Current forwards the runtime scale without a
        // V6 multiply; Fused applies it exactly once to the rounded qg value.
        // The support gate requires <=8 KiB scratch.
        args.ops->RunVf(Stage::V6, head,
                        ResolvePow2Primitive(args.key.useExp2), args.scale,
                        args.key.abi == PrepareAbi::Fused
                            ? RuntimeScaleUse::FusedQg
                            : RuntimeScaleUse::None,
                        args.key.abi == PrepareAbi::Fused ? 1U : 0U);
        detail::RequireVectorToMte3Outputs(*args.sync, Stage::V6);
        args.sync->Set(SyncPoint::SharedArenaFree, head.sharedArenaId,
                       sharedGeneration + 1U, Stage::V6, Pipe::Vector);

        const BufferSpan payload = detail::Payload(args, head);
        const Offset rhsRows = validRows > 32U ? ShapePolicy::kBt : 32U;
        const Offset rhsPlaneBytes =
            rhsRows * ShapePolicy::kK * ShapePolicy::kStorageBytes;
        args.ops->Store(
            Stage::V6,
            detail::PrivateSpan(
                head, "K-beta-g",
                {arch22_policy::V6PrivateLayout::kKBetaG.offset,
                 rhsPlaneBytes},
                head.localGeneration),
            detail::Subspan(
                payload, "K-beta-g-relay",
                arch22_policy::WorkspacePolicy::kPostKBetaG.offset,
                rhsPlaneBytes));
        args.ops->Store(
            Stage::V6,
            detail::PrivateSpan(
                head, "V-beta",
                {arch22_policy::V6PrivateLayout::kVToVBeta.offset,
                 rhsPlaneBytes},
                head.localGeneration),
            detail::Subspan(
                payload, "V-beta-relay",
                arch22_policy::WorkspacePolicy::kPostVBeta.offset,
                rhsPlaneBytes));
        const BufferSpan qResult = detail::PrivateSpan(
            head,
            args.key.abi == PrepareAbi::Current ? "qg" : "Qg-scaled",
            {arch22_policy::V6PrivateLayout::kQHatToQg.offset, tokenBytes},
            head.localGeneration);
        args.ops->Store(
            Stage::V6, qResult,
            detail::SymbolicGmRows(
                head,
                args.key.abi == PrepareAbi::Current ? "qg-output"
                                                     : "Qg-scaled-output",
                validRows, ShapePolicy::kK, ShapePolicy::kK,
                ShapePolicy::kStorageBytes, head.workspaceGeneration));
        args.ops->Store(
            Stage::V6,
            detail::PrivateSpan(
                head, "kg",
                {arch22_policy::V6PrivateLayout::kKHatToKg.offset,
                 tokenBytes},
                head.localGeneration),
            detail::SymbolicGmRows(
                head, "kg-output-or-handoff", validRows, ShapePolicy::kK,
                ShapePolicy::kK, ShapePolicy::kStorageBytes,
                head.workspaceGeneration));
        args.sync->Set(SyncPoint::LocalBankFree, head.localBankId,
                       head.localGeneration + 1U, Stage::V6, Pipe::Mte3);
    }
    args.sync->AivArrivePair(SyncPoint::V6RhsReady, pair,
                             collectiveGeneration, args.aivId, selected,
                             Stage::V6, Pipe::Mte3);
}

} // namespace kda_prepare_pseudocode::arch22

#endif // PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H
