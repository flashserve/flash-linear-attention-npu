/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_ARCH35_VEC_H
#define FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_ARCH35_VEC_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_tiling_key.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace kda_prepare_pseudocode {
namespace arch35 {
namespace detail {

inline BufferSpan Subspan(const BufferSpan &parent, const char *name,
                          Offset relativeOffset, Offset bytes)
{
    return {name, parent.space, parent.byteOffset + relativeOffset, bytes,
            parent.slot, parent.generation, parent.ownerRole, parent.ownerId};
}

inline BufferSpan UbMainSpan(const HeadTask &head, const char *name,
                             const Region &region, std::uint64_t generation)
{
    return {name, MemorySpace::Ub,
            static_cast<std::uint64_t>(UbPolicy::kMainBase[head.aivLocalSlot]) +
                region.offset,
            region.size, head.localBankId, generation, CoreRole::Aiv,
            head.aivId};
}

inline BufferSpan UbAuxSpan(const HeadTask &head, const char *name,
                            const Region &region, std::uint64_t generation)
{
    return {name, MemorySpace::Ub,
            static_cast<std::uint64_t>(UbPolicy::kAuxBase[head.aivLocalSlot]) +
                region.offset,
            region.size, head.localBankId, generation, CoreRole::Aiv,
            head.aivId};
}

inline BufferSpan SymbolicGmSpan(const HeadTask &head, const char *name,
                                 Offset bytes, std::uint64_t generation)
{
    // Public strides and canonical GM offsets are an ABI/tiling gate. Zero is
    // deliberately symbolic and must not be copied into the real kernel.
    return {name, MemorySpace::Gm, 0U, bytes, head.workspaceSlot, generation,
            CoreRole::Shared, 0U};
}

constexpr Offset MatrixFootprintBytes(Offset rows, Offset columns,
                                      Offset leadingDimension,
                                      Offset elementBytes)
{
    return rows == 0U || columns == 0U
               ? 0U
               : ((rows - 1U) * leadingDimension + columns) * elementBytes;
}

inline BufferSpan UbMainMatrixSpan(
    const HeadTask &head, const char *name, Offset base, Offset row,
    Offset column, Offset rows, Offset columns, Offset leadingDimension,
    Offset elementBytes, std::uint64_t generation)
{
    const Offset relativeOffset =
        base + (row * leadingDimension + column) * elementBytes;
    return UbMainSpan(
        head, name,
        {relativeOffset, MatrixFootprintBytes(
                             rows, columns, leadingDimension, elementBytes)},
        generation);
}

inline BufferSpan SymbolicGmMatrixSpan(
    const HeadTask &head, const char *name, Offset row, Offset column,
    Offset rows, Offset columns, Offset leadingDimension,
    Offset elementBytes, std::uint64_t generation)
{
    // row/column/rows/columns/leadingDimension are deliberately present in
    // this pseudo-interface. BufferSpan retains the exact strided footprint;
    // the real ABI must freeze and encode the corresponding 2-D DataCopy.
    const Offset relativeOffset =
        (row * leadingDimension + column) * elementBytes;
    const Offset footprint = MatrixFootprintBytes(
        rows, columns, leadingDimension, elementBytes);
    return {name, MemorySpace::Gm, relativeOffset, footprint,
            head.workspaceSlot, generation, CoreRole::Shared, 0U};
}

inline bool IsOwnedActiveHead(const HeadTask &head,
                              const VectorStageArgs &args)
{
    return head.active && head.aivId == args.aivId;
}

inline std::uint32_t ActiveScoreBlocks(std::uint32_t validRows)
{
    return std::min<std::uint32_t>(ShapePolicy::kScoreBlockCount,
                                   (validRows + ShapePolicy::kScoreBlockRows - 1U) /
                                       ShapePolicy::kScoreBlockRows);
}

inline bool IsSupportedKey(const ProposedTilingKey &key)
{
    // FP32Internal remains a capacity-proven layout candidate, but the target
    // mixed FP32-Akk / 2-byte-RHS Cube operand contract is not yet proven.
    return key.akkStorage == AkkStorage::TwoByteAbi &&
           IsSupportedStorageMapping(key);
}

// These dependent pseudo-interfaces are never instantiated by the host-only
// design build. They freeze operation order without claiming an Ascend C API.
template <typename Vf>
inline void V0OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    const ProposedTilingKey &key, float epsilon,
                    float lowerBound)
{
    const Offset gOffset = key.gateStorage == GateStorage::TwoByte
                               ? V0Gate2BLayout::kG.offset
                               : V0GateFp32Layout::kG.offset;
    const bool usesSelectiveGate =
        key.gateMode != GateMode::PrecomputedStep;
    auto gateCoefficient = vf.OneFp32();
    auto dtBias = vf.ZeroFp32Row(ShapePolicy::kK);
    if (usesSelectiveGate) {
        // The per-head coefficient and K-vector bias are loaded/computed once,
        // outside the token scan. Their AUX addresses overlay GRef[0:2], which
        // is materialized only after this loop has consumed both values.
        gateCoefficient = vf.Exp(vf.LoadALogScalarOnce(
            UbPolicy::kAuxBase[head.aivLocalSlot] +
            AuxLayout::kALogOrGateAttrs.offset));
        dtBias = vf.LoadDtBiasRow(
            UbPolicy::kAuxBase[head.aivLocalSlot] +
            AuxLayout::kDtBias.offset);
    }
    auto carry = vf.ZeroFp32Row(ShapePolicy::kK);
    auto zeroInputStorage = vf.RoundToInputStorage(
        vf.ClampForInputStorage(vf.ZeroFp32(), key.inputStorage),
        key.inputStorage);
    for (std::uint32_t row = 0; row < ShapePolicy::kBt; ++row) {
        if (row >= validRows) {
            vf.StoreZeroQHatKHatAndGPadding(
                row, zeroInputStorage, key.inputStorage);
            vf.StoreBetaEffScalar(row, 0.0F);
            continue;
        }
        auto q = vf.ToFp32(vf.LoadQStorageRow(row, key.inputStorage));
        auto k = vf.ToFp32(vf.LoadKStorageRow(row, key.inputStorage));
        auto gateRaw = vf.LoadGateRow(row, key.gateStorage);
        auto betaRaw = vf.LoadBetaFp32Scalar(
            UbPolicy::kAuxBase[head.aivLocalSlot] +
                AuxLayout::kBetaRaw.offset,
            row);
        auto betaEff = betaRaw;
        if (key.betaMode == BetaMode::Sigmoid) {
            betaEff = vf.Sigmoid(betaRaw);
        } else if (key.betaMode == BetaMode::TwoSigmoid) {
            betaEff = vf.Mul(2.0F, vf.Sigmoid(betaRaw));
        }

        auto qHat = q;
        auto kHat = k;
        if (key.qkNormMode == QkNormMode::L2) {
            qHat = vf.L2Normalize(q, epsilon);
            kHat = vf.L2Normalize(k, epsilon);
        }

        auto gateStep = gateRaw;
        if (key.gateMode == GateMode::PrecomputedStep) {
            gateStep = vf.Div(gateRaw, vf.Ln2());
        } else {
            auto x = vf.Add(gateRaw, dtBias);
            if (key.gateMode == GateMode::Softplus) {
                auto stableSoftplus = vf.Add(
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
        carry = vf.Add(carry, gateStep); // Token order is a true scan dependency.
        auto qHatStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(qHat, key.inputStorage),
            key.inputStorage);
        auto kHatStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(kHat, key.inputStorage),
            key.inputStorage);
        vf.StoreStorageRow(V0Gate2BLayout::kQHat.offset, row, qHatStorage,
                           key.inputStorage);
        vf.StoreStorageRow(V0Gate2BLayout::kKHat.offset, row, kHatStorage,
                           key.inputStorage);
        vf.StoreFp32Row(gOffset, row, carry);
        vf.StoreBetaEffScalar(row, betaEff);
    }

    for (std::uint32_t s = 0; s < ShapePolicy::kScoreBlockCount; ++s) {
        const std::uint32_t begin = s * ShapePolicy::kScoreBlockRows;
        const std::uint32_t end = std::min(begin + ShapePolicy::kScoreBlockRows,
                                           validRows);
        if (begin >= end) {
            vf.ZeroFp32Row(UbPolicy::kAuxBase[head.aivLocalSlot] +
                           AuxLayout::kGRef[s].offset);
            continue;
        }
        const std::uint32_t referenceRow = begin + (end - begin) / 2U;
        auto reference = vf.LoadFp32Row(gOffset, referenceRow);
        vf.StoreFp32Row(UbPolicy::kAuxBase[head.aivLocalSlot] +
                            AuxLayout::kGRef[s].offset,
                        reference);
    }
    vf.StoreGLast(carry);
}

template <typename Vf>
inline void V1OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    GateStorage gateStorage, InputStorage inputStorage,
                    ScoreStorage scoreStorage)
{
    const Offset gOffset = gateStorage == GateStorage::TwoByte
                               ? V1Gate2BLayout::kLiveG.offset
                               : V1GateFp32Layout::kLiveG.offset;
    const auto &kMinus = gateStorage == GateStorage::TwoByte
                             ? V1Gate2BLayout::kKMinus
                             : V1GateFp32Layout::kKMinus;
    const std::uint32_t activeBlocks = ActiveScoreBlocks(validRows);
    const float exp2InputMin = ScoreExp2InputMin(scoreStorage);
    const float exp2InputMax = ScoreExp2InputMax(scoreStorage);
    auto zeroScoreStorage = vf.RoundToScoreStorage(
        vf.ClampForScoreStorage(vf.ZeroFp32(), scoreStorage), scoreStorage);

    for (std::uint32_t row = 0; row < ShapePolicy::kBt; ++row) {
        if (row < validRows) {
            // All three source rows are captured before any aliased Q/K store.
            auto qHat = vf.ToFp32(vf.LoadStorageRow(
                V1Gate2BLayout::kQPlus.offset, row, inputStorage));
            auto kHat = vf.ToFp32(vf.LoadStorageRow(
                V1Gate2BLayout::kKPlus.offset, row, inputStorage));
            auto g = vf.LoadFp32Row(gOffset, row);
            const std::uint32_t owner = row / ShapePolicy::kScoreBlockRows;
            auto ownerRef = vf.LoadFp32Row(
                UbPolicy::kAuxBase[head.aivLocalSlot] +
                AuxLayout::kGRef[owner].offset);
            auto plusFactor = vf.Exp2Clamped(
                vf.Sub(g, ownerRef), exp2InputMin, exp2InputMax);
            auto qPlus = vf.Mul(qHat, plusFactor);
            auto kPlus = vf.Mul(kHat, plusFactor);

            // Khat remains in a register until every required prefix has been
            // produced. No Kminus destination overlaps the still-live G.
            for (std::uint32_t s = 0; s < ShapePolicy::kScoreBlockCount; ++s) {
                const std::uint32_t physicalEnd =
                    ShapePolicy::kPrefixRows[s];
                const std::uint32_t logicalEnd =
                    ShapePolicy::LogicalPrefixRows(s, validRows);
                if (row >= physicalEnd) {
                    continue;
                }
                if (s < activeBlocks && row < logicalEnd) {
                    auto reference = vf.LoadFp32Row(
                        UbPolicy::kAuxBase[head.aivLocalSlot] +
                        AuxLayout::kGRef[s].offset);
                    auto minusFactor = vf.Exp2Clamped(
                        vf.Sub(reference, g), exp2InputMin, exp2InputMax);
                    auto kMinusStorage = vf.RoundToScoreStorage(
                        vf.ClampForScoreStorage(
                            vf.Mul(kHat, minusFactor), scoreStorage),
                        scoreStorage);
                    vf.StoreStorageRow(kMinus[s].offset, row, kMinusStorage,
                                       scoreStorage);
                } else {
                    vf.StoreStorageRow(kMinus[s].offset, row,
                                       zeroScoreStorage, scoreStorage);
                }
            }
            auto qPlusStorage = vf.RoundToScoreStorage(
                vf.ClampForScoreStorage(qPlus, scoreStorage), scoreStorage);
            auto kPlusStorage = vf.RoundToScoreStorage(
                vf.ClampForScoreStorage(kPlus, scoreStorage), scoreStorage);
            vf.StoreStorageRow(V1Gate2BLayout::kQPlus.offset, row,
                               qPlusStorage, scoreStorage);
            vf.StoreStorageRow(V1Gate2BLayout::kKPlus.offset, row,
                               kPlusStorage, scoreStorage);
        } else {
            vf.StoreStorageRow(V1Gate2BLayout::kQPlus.offset, row,
                               zeroScoreStorage, scoreStorage);
            vf.StoreStorageRow(V1Gate2BLayout::kKPlus.offset, row,
                               zeroScoreStorage, scoreStorage);
            for (std::uint32_t s = 0; s < ShapePolicy::kScoreBlockCount; ++s) {
                // Tail keeps the fixed B_s-byte segment and explicitly zeros
                // [b_s=min(B_s,M), B_s); it never shrinks the physical slot.
                if (row < ShapePolicy::kPrefixRows[s]) {
                    vf.StoreStorageRow(kMinus[s].offset, row,
                                       zeroScoreStorage, scoreStorage);
                }
            }
        }
    }
}

template <typename Vf>
inline void V3OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    AkkStorage akkStorage, InputStorage inputStorage)
{
    (void)head;
    for (std::uint32_t row = 0; row < ShapePolicy::kBt; ++row) {
        const std::uint32_t block = row / ShapePolicy::kScoreBlockRows;
        for (std::uint32_t col = 0; col < ShapePolicy::kBt; ++col) {
            const bool c2Wrote = row < validRows &&
                                 col < ShapePolicy::kPrefixRows[block];
            if (!c2Wrote) {
                // Source-free zero: never read a stale raw-score generation.
                vf.StoreRawAqk(row, col, 0.0F);
                vf.StoreRawAkk(row, col, 0.0F);
            }
        }
    }

    // This ordering is one VF: source-free initialization precedes every raw
    // reader; a real same-V dependency may use PIPE_V only after target-CANN
    // verification. All-pipe barriers are forbidden; MTE/Fixpipe need events.
    for (std::uint32_t row = 0; row < ShapePolicy::kBt; ++row) {
        for (std::uint32_t col = 0; col < ShapePolicy::kBt; ++col) {
            const bool valid = row < validRows && col < validRows;
            auto aqk = valid && col <= row
                           ? vf.Mul(vf.Scale(), vf.LoadRawAqk(row, col))
                           : vf.ZeroFp32();
            auto lkk = valid && col < row
                           ? vf.Mul(vf.LoadBetaEff(row),
                                    vf.LoadRawAkk(row, col))
                           : vf.ZeroFp32();
            vf.StoreAqk(row, col, aqk);
            vf.StoreLkkOrIdentityPadding(row, col, lkk, validRows);
        }
    }
    vf.InvertTwo32By32LeavesWithFixedColumnScan();
    vf.MaterializeX0X1AndBAtFinalOffsets();
    if (akkStorage == AkkStorage::TwoByteAbi) {
        // q00/q01/q11 are stored at their final tight quadrant-major
        // addresses. ClampForInputStorage applies finite +/-65504 saturation
        // only for FP16; both FP16 and BF16 then use their own rounding.
        auto zeroStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.ZeroFp32(), inputStorage),
            inputStorage);
        for (std::uint32_t row = 0U;
             row < Akk2BPackPolicy::kQuadrantRows; ++row) {
            for (std::uint32_t col = 0U;
                 col < Akk2BPackPolicy::kQuadrantColumns; ++col) {
                auto q00 = vf.LoadStableAkkQ00OrZeroColumnPadding(
                    row, col, validRows);
                auto q11 = vf.LoadStableAkkQ11OrZeroColumnPadding(
                    row, col, validRows);
                auto q00Storage = vf.RoundToInputStorage(
                    vf.ClampForInputStorage(q00, inputStorage),
                    inputStorage);
                auto q11Storage = vf.RoundToInputStorage(
                    vf.ClampForInputStorage(q11, inputStorage),
                    inputStorage);
                // V3 stages q00/q11/q01 in three final UB regions. C4 later
                // loads them directly into the q00/q11/q01 positions of the
                // L1 quadrant-major resident; no UB or L1 relocation occurs.
                vf.StoreInputStorageMatrix(
                    V3Layout::kX0Tau.offset, row, col, q00Storage,
                    inputStorage);
                vf.StoreInputStorageMatrix(
                    V3Layout::kQ01Zero.offset, row, col, zeroStorage,
                    inputStorage);
                vf.StoreInputStorageMatrix(
                    V3Layout::kX1Tau.offset, row, col, q11Storage,
                    inputStorage);
            }
        }
    }
    // Aqk has no later FP32 reader. Cast it in place toward lower addresses at
    // the same base; forward row/column order overwrites only values already
    // read. This is not a UB relocation or a second VF invocation.
    for (std::uint32_t row = 0U; row < ShapePolicy::kBt; ++row) {
        for (std::uint32_t col = 0U; col < ShapePolicy::kBt; ++col) {
            auto aqkStorage = vf.RoundToInputStorage(
                vf.ClampForInputStorage(vf.LoadAqk(row, col), inputStorage),
                inputStorage);
            vf.StoreAqkStorageInPlace(row, col, aqkStorage, inputStorage);
        }
    }
}

template <typename Vf>
inline void V6OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    PrepareAbi abi, InputStorage inputStorage)
{
    (void)head;
    auto zeroInputStorage = vf.RoundToInputStorage(
        vf.ClampForInputStorage(vf.ZeroFp32(), inputStorage), inputStorage);
    if (validRows == 0U) {
        for (std::uint32_t row = 0; row < ShapePolicy::kBt; ++row) {
            vf.StoreZeroV6OutputRows(row, abi, zeroInputStorage,
                                     inputStorage);
        }
        return;
    }
    auto gLast = vf.LoadFp32Row(V6Layout::kGInput.offset, validRows - 1U);
    for (std::uint32_t row = 0; row < ShapePolicy::kBt; ++row) {
        if (row >= validRows) {
            vf.StoreZeroV6OutputRows(row, abi, zeroInputStorage,
                                     inputStorage);
            continue;
        }
        // Capture every reader before Q/K/V in-place stores or the optional
        // 2-byte QgScaled compression into the low half of FP32 G.
        auto qHat = vf.ToFp32(vf.LoadStorageRow(
            V6Layout::kQHatToQg.offset, row, inputStorage));
        auto kHat = vf.ToFp32(vf.LoadStorageRow(
            V6Layout::kKHatToKg.offset, row, inputStorage));
        auto v = vf.ToFp32(vf.LoadStorageRow(
            V6Layout::kVToVBeta.offset, row, inputStorage));
        auto g = vf.LoadFp32Row(V6Layout::kGInput.offset, row);
        auto beta = vf.LoadBetaEff(row);
        auto expG = vf.Exp2Clamped(
            g, kDirectExp2InputMin, kDirectExp2InputMax);
        auto qgFp32 = vf.Mul(qHat, expG);
        auto kgFp32 = vf.Mul(
            kHat,
            vf.Exp2Clamped(vf.Sub(gLast, g), kDirectExp2InputMin,
                           kDirectExp2InputMax));
        auto qgStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(qgFp32, inputStorage), inputStorage);
        auto kgStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(kgFp32, inputStorage), inputStorage);

        // Preserve the split Prepare double-rounding point. kHat*exp2(G) is
        // first materialized in InputStorage, promoted back to FP32 for beta,
        // then saturated/rounded again for K_beta_g.
        auto kGateStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.Mul(kHat, expG), inputStorage),
            inputStorage);
        auto kBetaGFp32 = vf.Mul(beta, vf.ToFp32(kGateStorage));
        auto kBetaGStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(kBetaGFp32, inputStorage), inputStorage);
        auto vBetaStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.Mul(beta, v), inputStorage),
            inputStorage);

        vf.StoreStorageRow(V6Layout::kQHatToQg.offset, row, qgStorage,
                           inputStorage);
        vf.StoreStorageRow(V6Layout::kKHatToKg.offset, row, kgStorage,
                           inputStorage);
        vf.StoreStorageRow(V6Layout::kKBetaG.offset, row, kBetaGStorage,
                           inputStorage);
        vf.StoreStorageRow(V6Layout::kVToVBeta.offset, row, vBetaStorage,
                           inputStorage);
        if (abi == PrepareAbi::Fused) {
            // The 2-byte destination overlaps FP32 G row floor(row/2), which
            // is no later than row. The current G row was fully captured above.
            // Scaling consumes the first rounded qg value, then applies the
            // second InputStorage saturation/rounding required by FUSED ABI.
            auto qgScaledStorage = vf.RoundToInputStorage(
                vf.ClampForInputStorage(
                    vf.Mul(vf.Scale(), vf.ToFp32(qgStorage)), inputStorage),
                inputStorage);
            vf.StoreStorageRow(V6Layout::kQgScaled.offset, row,
                               qgScaledStorage, inputStorage);
        }
    }
}

inline bool ValidArgs(const VectorStageArgs &args)
{
    return args.work != nullptr && args.workspace != nullptr &&
           args.sync != nullptr && args.ops != nullptr &&
           IsSupportedKey(args.key);
}

} // namespace detail

inline void RunV0(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedActiveHead(head, args)) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t localGeneration = head.localGeneration;
        args.sync->Wait(SyncPoint::SlotFree, head.workspaceSlot,
                        workspaceGeneration,
                        Stage::V0, Pipe::Control);
        args.sync->Wait(SyncPoint::LocalBankFree, head.localBankId,
                        localGeneration, Stage::V0, Pipe::Mte2);

        const bool gate2B = args.key.gateStorage == GateStorage::TwoByte;
        const Offset validRows = args.work->group.chunk.validRows;
        const Offset qkInputBytes =
            validRows * ShapePolicy::kK * ShapePolicy::kStorageBytes;
        const Offset gateInputBytes =
            validRows * ShapePolicy::kK *
            (gate2B ? ShapePolicy::kStorageBytes : ShapePolicy::kFp32Bytes);
        const Offset gOutputBytes =
            validRows * ShapePolicy::kK * ShapePolicy::kFp32Bytes;
        const Offset betaInputBytes =
            validRows * ShapePolicy::kFp32Bytes;
        const Region q = gate2B ? V0Gate2BLayout::kQHat : V0GateFp32Layout::kQHat;
        const Region k = gate2B ? V0Gate2BLayout::kKHat : V0GateFp32Layout::kKHat;
        const Region g = gate2B ? V0Gate2BLayout::kGateRaw : V0GateFp32Layout::kG;
        args.ops->Load(Stage::V0,
                       detail::SymbolicGmSpan(head, "q", qkInputBytes,
                                              workspaceGeneration),
                       detail::UbMainSpan(
                           head, "q-to-qhat", {q.offset, qkInputBytes},
                           localGeneration));
        args.ops->Load(Stage::V0,
                       detail::SymbolicGmSpan(head, "k", qkInputBytes,
                                              workspaceGeneration),
                       detail::UbMainSpan(
                           head, "k-to-khat", {k.offset, qkInputBytes},
                           localGeneration));
        args.ops->Load(Stage::V0,
                       detail::SymbolicGmSpan(head, "gate", gateInputBytes,
                                              workspaceGeneration),
                       detail::UbMainSpan(
                           head, "gate-to-G", {g.offset, gateInputBytes},
                           localGeneration));
        // The split-forward kernel boundary is FP32 beta. Public 2-byte beta
        // is cast by op_api/L2 before this load, so V0 reads exactly M scalars.
        args.ops->Load(Stage::V0,
                       detail::SymbolicGmSpan(head, "beta", betaInputBytes,
                                              workspaceGeneration),
                       detail::UbAuxSpan(
                           head, "beta-raw",
                           {AuxLayout::kBetaRaw.offset, betaInputBytes},
                           localGeneration));
        if (args.key.gateMode != GateMode::PrecomputedStep) {
            // Each selective-gate input is loaded once. An absent optional
            // dt_bias must be zero-filled at this final AUX address, not read
            // past a null/short GM tensor.
            const BufferSpan dtBias = detail::UbAuxSpan(
                head, "dt-bias", AuxLayout::kDtBias, localGeneration);
            if (args.hasDtBias) {
                args.ops->Load(
                    Stage::V0,
                    detail::SymbolicGmSpan(
                        head, "dt-bias",
                        ShapePolicy::kK * ShapePolicy::kFp32Bytes,
                        workspaceGeneration),
                    dtBias);
            } else {
                args.ops->Zero(Stage::V0, dtBias);
            }
            args.ops->Load(
                Stage::V0,
                detail::SymbolicGmSpan(head, "A-log", ShapePolicy::kFp32Bytes,
                                       workspaceGeneration),
                detail::UbAuxSpan(
                    head, "A-log",
                    {AuxLayout::kALogOrGateAttrs.offset,
                     ShapePolicy::kFp32Bytes},
                    localGeneration));
        }

        // Exactly one invocation; its required body is detail::V0OneVf with
        // args.key and the frozen epsilon/lowerBound scalar attributes.
        args.ops->RunVf(Stage::V0, head);
        args.sync->Set(SyncPoint::V0BetaReady, head.localBankId,
                       localGeneration,
                       Stage::V0, Pipe::Vector);

        const BufferSpan context = args.workspace->Span(
            WorkspaceRegion::Context, head.workspaceSlot,
            workspaceGeneration);
        const Region gResult = gate2B ? V0Gate2BLayout::kG : V0GateFp32Layout::kG;
        args.ops->Store(Stage::V0,
                        detail::UbMainSpan(head, "qhat", q, localGeneration),
                        detail::Subspan(context, "qhat-context", 0x0000U, 0x4000U));
        args.ops->Store(Stage::V0,
                        detail::UbMainSpan(head, "khat", k, localGeneration),
                        detail::Subspan(context, "khat-context", 0x4000U, 0x4000U));
        args.ops->Store(Stage::V0,
                        detail::UbMainSpan(head, "G", gResult,
                                           localGeneration),
                        detail::Subspan(context, "G-context", 0x8200U, 0x8000U));

        if (args.key.abi == PrepareAbi::Current) {
            args.ops->Store(
                Stage::V0,
                detail::UbMainSpan(
                    head, "G-output", {gResult.offset, gOutputBytes},
                    localGeneration),
                detail::SymbolicGmSpan(head, "gk-output", gOutputBytes,
                                       workspaceGeneration));
        } else {
            args.ops->Store(
                Stage::V0,
                detail::UbAuxSpan(head, "G-last", AuxLayout::kGLast,
                                  localGeneration),
                detail::SymbolicGmSpan(head, "G-last-output", 0x0200U,
                                       workspaceGeneration));
        }

        // Both points become visible only after the last enabled MTE3/output
        // drain. Concrete HardEvent and CrossCore IDs remain an API gate.
        args.sync->Set(SyncPoint::V0ContextReady, head.workspaceSlot,
                       workspaceGeneration, Stage::V0, Pipe::Mte3);
        args.sync->Set(SyncPoint::V0ExportDone, head.localBankId,
                       localGeneration, Stage::V0, Pipe::Mte3);
    }
}

inline void RunV1(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedActiveHead(head, args)) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t localGeneration = head.localGeneration;
        args.sync->Wait(SyncPoint::V0ExportDone, head.localBankId,
                        localGeneration,
                        Stage::V1, Pipe::Vector);
        // Exactly one invocation. Its required body is detail::V1OneVf with
        // input/gate/score storage from args.key; both Exp2 sites and every
        // score saturation/rounding point remain inside this same VF.
        args.ops->RunVf(Stage::V1, head);

        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        if (args.key.gateStorage == GateStorage::TwoByte) {
            for (const CopyRegion &copy : V1Gate2BLayout::kScoreWriteback) {
                const Region source{copy.source, copy.size};
                args.ops->Store(
                    Stage::V1,
                    detail::UbMainSpan(head, "score-source-2b", source,
                                       localGeneration),
                    detail::Subspan(payload, "packed-score", copy.destination,
                                    copy.size));
            }
        } else {
            for (const CopyRegion &copy : V1GateFp32Layout::kScoreWriteback) {
                const Region source{copy.source, copy.size};
                args.ops->Store(
                    Stage::V1,
                    detail::UbMainSpan(head, "score-source-fp32", source,
                                       localGeneration),
                    detail::Subspan(payload, "packed-score", copy.destination,
                                    copy.size));
            }
        }
        // One writeback phase contains exactly two fixed MTE3 regions. It is
        // not one contiguous DataCopy. These three states all follow the last
        // source reader; each is consumed once by its named owner.
        args.sync->Set(SyncPoint::V1MainSourceFree, head.localBankId,
                       localGeneration, Stage::V1, Pipe::Mte3);
        args.sync->Set(SyncPoint::C2RawDstFree, head.localBankId,
                       localGeneration, Stage::V1, Pipe::Mte3);
        args.sync->Set(SyncPoint::V1ScoreReady, head.workspaceSlot,
                       workspaceGeneration, Stage::V1, Pipe::Mte3);
    }
}

inline void RunV3(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedActiveHead(head, args)) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t localGeneration = head.localGeneration;
        const Offset validRows = args.work->group.chunk.validRows;
        args.sync->Wait(SyncPoint::V0BetaReady, head.localBankId,
                        localGeneration,
                        Stage::V3, Pipe::Vector);
        args.sync->Wait(SyncPoint::C2ScorePayloadFree, head.workspaceSlot,
                        workspaceGeneration, Stage::V3, Pipe::Mte3);
        args.sync->Wait(SyncPoint::C2RawReady, head.localBankId,
                        localGeneration,
                        Stage::V3, Pipe::Vector);
        // Exactly one invocation; detail::V3OneVf also receives inputStorage.
        // Its first operations source-free-zero every C2-unwritten raw cell
        // before any raw-score reader, then round every 2-byte V3 destination.
        args.ops->RunVf(Stage::V3, head);
        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        args.ops->Store(Stage::V3,
                        detail::UbMainSpan(head, "X0", V3Layout::kX0,
                                           localGeneration),
                        detail::Subspan(payload, "X0-fp32", 0x0000U, 0x1000U));
        args.ops->Store(Stage::V3,
                        detail::UbMainSpan(head, "X1", V3Layout::kX1,
                                           localGeneration),
                        detail::Subspan(payload, "X1-fp32", 0x1000U, 0x1000U));
        args.ops->Store(Stage::V3,
                        detail::UbMainSpan(head, "B", V3Layout::kB,
                                           localGeneration),
                        detail::Subspan(payload, "B-fp32", 0x2000U, 0x1000U));
        if (args.key.akkStorage == AkkStorage::TwoByteAbi) {
            args.ops->Store(Stage::V3,
                            detail::UbMainSpan(head, "X0-tau", V3Layout::kX0Tau,
                                               localGeneration),
                            detail::Subspan(payload, "X0-tau", 0x3000U, 0x0800U));
            args.ops->Store(Stage::V3,
                            detail::UbMainSpan(head, "X1-tau", V3Layout::kX1Tau,
                                               localGeneration),
                            detail::Subspan(payload, "X1-tau", 0x3800U, 0x0800U));
            args.ops->Store(Stage::V3,
                            detail::UbMainSpan(head, "q01-zero", V3Layout::kQ01Zero,
                                               localGeneration),
                            detail::Subspan(payload, "q01-zero", 0x4000U, 0x0800U));
        }
        args.sync->Set(SyncPoint::V3VcsReady, head.workspaceSlot,
                       workspaceGeneration, Stage::V3, Pipe::Mte3);

        // Aqk and optional AkkOut use independent public GM addresses. Their
        // ABI offsets/casts are intentionally symbolic, but their MTE3 drains
        // are included before local MAIN/AUX ownership is returned.
        if (validRows != 0U) {
            args.ops->Store(
                Stage::V3,
                detail::UbMainMatrixSpan(
                    head, "Aqk-valid-ld64", V3Layout::kRawAqk.offset, 0U,
                    0U, validRows, ShapePolicy::kBt, ShapePolicy::kBt,
                    ShapePolicy::kStorageBytes, localGeneration),
                detail::SymbolicGmMatrixSpan(
                    head, "Aqk-output-valid-ld64", 0U, 0U, validRows,
                    ShapePolicy::kBt, ShapePolicy::kBt,
                    ShapePolicy::kStorageBytes, workspaceGeneration));
        }
        if (args.key.abi == PrepareAbi::Current &&
            args.key.akkStorage == AkkStorage::TwoByteAbi) {
            constexpr Offset kQuadrant = 32U;
            const Offset top = std::min(validRows, kQuadrant);
            const Offset bottom = validRows > kQuadrant
                                      ? validRows - kQuadrant
                                      : 0U;
            if (top != 0U) {
                args.ops->Store(
                    Stage::V3,
                    detail::UbMainMatrixSpan(
                        head, "Akk-q00-valid-ld32",
                        V3Layout::kX0Tau.offset, 0U, 0U, top, kQuadrant,
                        kQuadrant, ShapePolicy::kStorageBytes,
                        localGeneration),
                    detail::SymbolicGmMatrixSpan(
                        head, "AkkOut-q00-valid-ld64", 0U, 0U, top,
                        kQuadrant, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes, workspaceGeneration));
                args.ops->Store(
                    Stage::V3,
                    detail::UbMainMatrixSpan(
                        head, "Akk-q01-valid-ld32",
                        V3Layout::kQ01Zero.offset, 0U, 0U, top, kQuadrant,
                        kQuadrant, ShapePolicy::kStorageBytes,
                        localGeneration),
                    detail::SymbolicGmMatrixSpan(
                        head, "AkkOut-q01-valid-ld64", 0U, kQuadrant, top,
                        kQuadrant, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes, workspaceGeneration));
            }
            if (bottom != 0U) {
                args.ops->Store(
                    Stage::V3,
                    detail::UbMainMatrixSpan(
                        head, "Akk-q11-valid-ld32",
                        V3Layout::kX1Tau.offset, 0U, 0U, bottom, kQuadrant,
                        kQuadrant, ShapePolicy::kStorageBytes,
                        localGeneration),
                    detail::SymbolicGmMatrixSpan(
                        head, "AkkOut-q11-valid-ld64", kQuadrant,
                        kQuadrant, bottom, kQuadrant, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes, workspaceGeneration));
            }
        }
        args.sync->Set(SyncPoint::V3LocalSourceFree, head.localBankId,
                       localGeneration, Stage::V3, Pipe::Mte3);
    }
}

inline void RunV6(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedActiveHead(head, args)) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t localGeneration = head.localGeneration;
        const Offset validRows = args.work->group.chunk.validRows;
        const Offset tokenStorageBytes =
            validRows * ShapePolicy::kV * ShapePolicy::kStorageBytes;
        // This is a symbolic three-input join, not a claim that target CANN
        // provides one instruction with these semantics.
        args.sync->Wait(SyncPoint::V0ContextReady, head.workspaceSlot,
                        workspaceGeneration, Stage::V6, Pipe::Mte2);
        args.sync->Wait(SyncPoint::V3LocalSourceFree, head.localBankId,
                        localGeneration, Stage::V6, Pipe::Mte2);
        args.sync->Wait(SyncPoint::C4PayloadFree, head.workspaceSlot,
                        workspaceGeneration, Stage::V6, Pipe::Mte2);

        const BufferSpan context = args.workspace->Span(
            WorkspaceRegion::Context, head.workspaceSlot,
            workspaceGeneration);
        args.ops->Load(Stage::V6,
                       detail::Subspan(context, "qhat-context", 0x0000U, 0x4000U),
                       detail::UbMainSpan(head, "qhat-to-qg", V6Layout::kQHatToQg,
                                              localGeneration));
        args.ops->Load(Stage::V6,
                       detail::Subspan(context, "khat-context", 0x4000U, 0x4000U),
                       detail::UbMainSpan(head, "khat-to-kg", V6Layout::kKHatToKg,
                                              localGeneration));
        args.ops->Load(Stage::V6,
                       detail::Subspan(context, "G-context", 0x8200U, 0x8000U),
                       detail::UbMainSpan(head, "G", V6Layout::kGInput,
                                              localGeneration));
        args.ops->Load(Stage::V6,
                       detail::SymbolicGmSpan(head, "V", tokenStorageBytes,
                                              workspaceGeneration),
                       detail::UbMainSpan(
                           head, "V-to-Vbeta",
                           {V6Layout::kVToVBeta.offset, tokenStorageBytes},
                           localGeneration));
        // Exactly one invocation; detail::V6OneVf receives inputStorage. Both
        // direct Exp2 sites and all first/second storage round points execute
        // inside this same VF.
        args.ops->RunVf(Stage::V6, head);

        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        args.ops->Store(Stage::V6,
                        detail::UbMainSpan(head, "K-beta-g", V6Layout::kKBetaG,
                                               localGeneration),
                        detail::Subspan(payload, "K-beta-g", 0x0000U, 0x4000U));
        args.ops->Store(Stage::V6,
                        detail::UbMainSpan(head, "V-beta", V6Layout::kVToVBeta,
                                               localGeneration),
                        detail::Subspan(payload, "V-beta", 0x4000U, 0x4000U));

        if (args.key.abi == PrepareAbi::Current) {
            args.ops->Store(
                Stage::V6,
                detail::UbMainSpan(
                    head, "qg",
                    {V6Layout::kQHatToQg.offset, tokenStorageBytes},
                    localGeneration),
                detail::SymbolicGmSpan(head, "qg-output", tokenStorageBytes,
                                       workspaceGeneration));
        } else {
            args.ops->Store(
                Stage::V6,
                detail::UbMainSpan(
                    head, "Qg-scaled",
                    {V6Layout::kQgScaled.offset, tokenStorageBytes},
                    localGeneration),
                detail::SymbolicGmSpan(head, "Qg-scaled-output",
                                       tokenStorageBytes,
                                       workspaceGeneration));
        }
        args.ops->Store(
            Stage::V6,
            detail::UbMainSpan(
                head, "kg",
                {V6Layout::kKHatToKg.offset, tokenStorageBytes},
                localGeneration),
            detail::SymbolicGmSpan(head, "kg-output-or-handoff",
                                   tokenStorageBytes,
                                   workspaceGeneration));

        args.sync->Set(SyncPoint::V6RhsReady, head.workspaceSlot,
                       workspaceGeneration, Stage::V6, Pipe::Mte3);
        args.sync->Set(SyncPoint::LocalBankFree, head.localBankId,
                       localGeneration + 1U, Stage::V6, Pipe::Mte3);
    }
}

} // namespace arch35
} // namespace kda_prepare_pseudocode

#endif // FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_ARCH35_VEC_H
