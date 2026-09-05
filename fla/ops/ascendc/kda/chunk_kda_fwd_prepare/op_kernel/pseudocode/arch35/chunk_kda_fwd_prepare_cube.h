/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_ARCH35_CUBE_H
#define FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_ARCH35_CUBE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_tiling_key.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace kda_prepare_pseudocode {
namespace arch35 {
namespace cube_detail {

inline BufferSpan CubeSubspan(const BufferSpan &parent, const char *name,
                              Offset relativeOffset, Offset bytes)
{
    return {name, parent.space, parent.byteOffset + relativeOffset, bytes,
            parent.slot, parent.generation, parent.ownerRole, parent.ownerId};
}

constexpr Offset MatrixFootprintBytes(Offset rows, Offset columns,
                                      Offset leadingDimension,
                                      Offset elementBytes)
{
    return rows == 0U || columns == 0U
               ? 0U
               : ((rows - 1U) * leadingDimension + columns) * elementBytes;
}

inline BufferSpan CubeMatrixSubspan(
    const BufferSpan &parent, const char *name, Offset row, Offset column,
    Offset rows, Offset columns, Offset leadingDimension, Offset elementBytes)
{
    const Offset relativeOffset =
        (row * leadingDimension + column) * elementBytes;
    return CubeSubspan(
        parent, name, relativeOffset,
        MatrixFootprintBytes(rows, columns, leadingDimension, elementBytes));
}

inline BufferSpan L1Span(const HeadTask &head, const char *name,
                         std::uint32_t aicId, Offset offset, Offset bytes,
                         std::uint64_t generation)
{
    return {name, MemorySpace::L1, offset, bytes, head.l1BankId,
            generation, CoreRole::Aic, aicId};
}

inline BufferSpan PairedAivUbSpan(const HeadTask &head, const char *name,
                                  Offset offset, Offset bytes,
                                  std::uint64_t generation)
{
    return {name, MemorySpace::Ub,
            static_cast<std::uint64_t>(UbPolicy::kMainBase[head.aivLocalSlot]) +
                offset,
            bytes, head.localBankId, generation, CoreRole::Aiv, head.aivId};
}

inline BufferSpan SymbolicL0cSpan(const HeadTask &head, const char *name,
                                  std::uint32_t aicId, Offset bytes,
                                  Offset offset = 0U)
{
    return {name, MemorySpace::L0, offset, bytes, head.l0cBankId,
            head.l0cGeneration, CoreRole::Aic, aicId};
}

inline BufferSpan SymbolicGmMatrixOutput(
    const HeadTask &head, const char *name, Offset row, Offset column,
    Offset rows, Offset columns, Offset leadingDimension,
    Offset elementBytes, std::uint64_t generation)
{
    const Offset relativeOffset =
        (row * leadingDimension + column) * elementBytes;
    const Offset footprint = MatrixFootprintBytes(
        rows, columns, leadingDimension, elementBytes);
    return {name, MemorySpace::Gm, relativeOffset, footprint,
            head.workspaceSlot, generation, CoreRole::Shared, 0U};
}

inline bool ValidArgs(const CubeStageArgs &args)
{
    return args.work != nullptr && args.workspace != nullptr &&
           args.sync != nullptr && args.ops != nullptr &&
           args.key.akkStorage == AkkStorage::TwoByteAbi &&
           IsSupportedStorageMapping(args.key);
}

inline std::uint32_t ActiveScoreBlocks(std::uint32_t validRows)
{
    return std::min<std::uint32_t>(ShapePolicy::kScoreBlockCount,
                                   (validRows + ShapePolicy::kScoreBlockRows - 1U) /
                                       ShapePolicy::kScoreBlockRows);
}

inline Offset LaneBase(const HeadTask &head)
{
    return L1Policy::kLaneBase[head.groupLocalHead];
}

inline Offset X0Resident(const HeadTask &head, AkkStorage storage)
{
    if (storage == AkkStorage::Fp32Internal) {
        return L1Policy::AkkFp32Resident::kAkk.offset +
               head.groupLocalHead * L1Policy::AkkFp32Resident::kAkkStride;
    }
    return L1Policy::Akk2BResident::kX0.offset +
           head.groupLocalHead * L1Policy::Akk2BResident::kMatrixStride;
}

inline Offset X1Resident(const HeadTask &head, AkkStorage storage)
{
    if (storage == AkkStorage::Fp32Internal) {
        return L1Policy::AkkFp32Resident::kAkk.offset +
               head.groupLocalHead * L1Policy::AkkFp32Resident::kAkkStride +
               0x3000U;
    }
    return L1Policy::Akk2BResident::kX1.offset +
           head.groupLocalHead * L1Policy::Akk2BResident::kMatrixStride;
}

inline Offset TResident(const HeadTask &head, AkkStorage storage)
{
    if (storage == AkkStorage::Fp32Internal) {
        return L1Policy::AkkFp32Resident::kT.offset +
               head.groupLocalHead * L1Policy::AkkFp32Resident::kTStride;
    }
    return L1Policy::Akk2BResident::kT.offset +
           head.groupLocalHead * L1Policy::Akk2BResident::kMatrixStride;
}

inline Offset Akk2BResident(const HeadTask &head)
{
    return L1Policy::Akk2BResident::kAkkTau.offset +
           head.groupLocalHead * L1Policy::Akk2BResident::kAkkStride;
}

inline Offset AkkFp32Resident(const HeadTask &head)
{
    return L1Policy::AkkFp32Resident::kAkk.offset +
           head.groupLocalHead * L1Policy::AkkFp32Resident::kAkkStride;
}

} // namespace cube_detail

inline void RunC2(const CubeStageArgs &args)
{
    if (!cube_detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!head.active) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t localGeneration = head.localGeneration;
        const std::uint64_t l1Generation = head.l1Generation;
        const Offset lane = cube_detail::LaneBase(head);
        const Offset l0cLane =
            L0cPolicy::HeadLaneBase(head.groupLocalHead);
        args.sync->Wait(SyncPoint::L0cBankFree, head.l0cBankId,
                        head.l0cGeneration, Stage::C2, Pipe::Cube);
        args.sync->Wait(SyncPoint::L1BankFree, head.l1BankId,
                        l1Generation, Stage::C2, Pipe::Mte2);
        args.sync->Wait(SyncPoint::V1ScoreReady, head.workspaceSlot,
                        workspaceGeneration, Stage::C2, Pipe::Mte2);
        args.sync->Wait(SyncPoint::V1MainSourceFree, head.localBankId,
                        localGeneration, Stage::C2, Pipe::Fixpipe);
        args.sync->Wait(SyncPoint::C2RawDstFree, head.localBankId,
                        localGeneration, Stage::C2, Pipe::Fixpipe);

        // Exactly one 72 KiB GM->L1 load. There is no per-s or per-tile reload.
        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        args.ops->Load(
            Stage::C2,
            cube_detail::CubeSubspan(payload, "packed-score", 0U,
                                     ShapePolicy::kScorePayloadBytes),
            cube_detail::L1Span(head, "C2-current-lane", args.workgroupId,
                                lane, L1Policy::kLaneBytes, l1Generation));
        args.sync->Set(SyncPoint::C2ScorePayloadFree, head.workspaceSlot,
                       workspaceGeneration, Stage::C2, Pipe::Mte2);

        const std::uint32_t validRows = args.work->group.chunk.validRows;
        const std::uint32_t activeBlocks =
            cube_detail::ActiveScoreBlocks(validRows);
        const BufferSpan rawAqkParent = cube_detail::PairedAivUbSpan(
            head, "raw-Aqk-parent", V3Layout::kRawAqk.offset,
            V3Layout::kRawAqk.size, localGeneration);
        const BufferSpan rawAkkParent = cube_detail::PairedAivUbSpan(
            head, "raw-Akk-parent", V3Layout::kRawAkk.offset,
            V3Layout::kRawAkk.size, localGeneration);
        for (std::uint32_t s = 0; s < activeBlocks; ++s) {
            const std::uint32_t physicalN = C2Policy::kN[s];
            // Qplus/Kplus have sixteen physical rows. V1 zeroes invalid query
            // rows and Kminus [min(N,M),N), so every active MMAD
            // uses fixed, aligned N={16,32,48,64} without reading garbage.
            const BufferSpan qBand = cube_detail::L1Span(
                head, "Qplus-band", args.workgroupId,
                lane + C2Policy::kQBandOffset[s], 0x1000U, l1Generation);
            const BufferSpan kBand = cube_detail::L1Span(
                head, "Kplus-band", args.workgroupId,
                lane + C2Policy::kKBandOffset[s], 0x1000U, l1Generation);
            const BufferSpan kMinus = cube_detail::L1Span(
                head, "Kminus-prefix", args.workgroupId,
                lane + C2Policy::kKMinusOffset[s],
                ShapePolicy::kKMinusBytes[s], l1Generation);
            const Offset rawRow = s * C2Policy::kM;
            const Offset resultBytes =
                L0cPolicy::kC2PerHeadResultBytes[s];
            const BufferSpan rawAqkL0c = cube_detail::SymbolicL0cSpan(
                head, "raw-Aqk-L0C", args.workgroupId, resultBytes,
                l0cLane + L0cPolicy::kC2AqkOffset[s]);
            args.ops->Mmad(Stage::C2, qBand, kMinus, rawAqkL0c,
                           FromScoreStorage(args.key.scoreStorage),
                           FromScoreStorage(args.key.scoreStorage),
                           C2Policy::kM, physicalN, C2Policy::kK, true);
            args.ops->Store(
                Stage::C2, rawAqkL0c,
                cube_detail::CubeMatrixSubspan(
                    rawAqkParent, "raw-Aqk-band-ld64", rawRow, 0U,
                    C2Policy::kM, physicalN,
                    C2Policy::kRawLeadingDimension,
                    ShapePolicy::kFp32Bytes));

            const BufferSpan rawAkkL0c = cube_detail::SymbolicL0cSpan(
                head, "raw-Akk-L0C", args.workgroupId, resultBytes,
                l0cLane + L0cPolicy::kC2AkkOffset[s]);
            args.ops->Mmad(Stage::C2, kBand, kMinus, rawAkkL0c,
                           FromScoreStorage(args.key.scoreStorage),
                           FromScoreStorage(args.key.scoreStorage),
                           C2Policy::kM, physicalN, C2Policy::kK, true);
            args.ops->Store(
                Stage::C2, rawAkkL0c,
                cube_detail::CubeMatrixSubspan(
                    rawAkkParent, "raw-Akk-band-ld64", rawRow, 0U,
                    C2Policy::kM, physicalN,
                    C2Policy::kRawLeadingDimension,
                    ShapePolicy::kFp32Bytes));

            // PROPOSED API gate: logical element (r,c) above must land at
            // UBM + rawBase + (16*s+r)*0x100 + c*4, c<physicalN.
            // A tight 16xN Fixpipe store is incorrect. Target CANN must prove
            // direct paired-AIV UB write with destination leading dimension 64.
        }
        args.sync->Set(SyncPoint::C2ScoreL1Free, head.l1BankId, l1Generation,
                       Stage::C2, Pipe::Mte1);
        // Inactive bands are not launched. V3 source-free-zeros their complete
        // raw rows and every active band's columns [physicalN,64) before reads.
        args.sync->Set(SyncPoint::C2RawReady, head.localBankId, localGeneration,
                       Stage::C2, Pipe::Fixpipe);
    }
}

inline void RunC4(const CubeStageArgs &args)
{
    if (!cube_detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!head.active) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t l1Generation = head.l1Generation;
        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        const Offset lane = cube_detail::LaneBase(head);
        const Offset l0cLane =
            L0cPolicy::HeadLaneBase(head.groupLocalHead);
        args.sync->Wait(SyncPoint::C2ScoreL1Free, head.l1BankId,
                        l1Generation, Stage::C4, Pipe::Mte2);
        args.sync->Wait(SyncPoint::V3VcsReady, head.workspaceSlot,
                        workspaceGeneration, Stage::C4, Pipe::Mte2);

        const BufferSpan bCurrent = cube_detail::L1Span(
            head, "B-current", args.workgroupId, lane, 0x1000U,
            l1Generation);
        args.ops->Load(
            Stage::C4,
            cube_detail::CubeSubspan(payload, "B-fp32", 0x2000U, 0x1000U),
            bCurrent);
        const Offset x0 = cube_detail::X0Resident(head, args.key.akkStorage);
        const Offset x1 = cube_detail::X1Resident(head, args.key.akkStorage);
        const BufferSpan x0Resident = cube_detail::L1Span(
            head, "X0-resident", args.workgroupId, x0, 0x1000U,
            l1Generation);
        const BufferSpan x1Resident = cube_detail::L1Span(
            head, "X1-resident", args.workgroupId, x1, 0x1000U,
            l1Generation);
        args.ops->Load(
            Stage::C4,
            cube_detail::CubeSubspan(payload, "X0-fp32", 0x0000U, 0x1000U),
            x0Resident);
        args.ops->Load(
            Stage::C4,
            cube_detail::CubeSubspan(payload, "X1-fp32", 0x1000U, 0x1000U),
            x1Resident);

        if (args.key.akkStorage == AkkStorage::TwoByteAbi) {
            const Offset akk = cube_detail::Akk2BResident(head);
            args.ops->Load(
                Stage::C4,
                cube_detail::CubeSubspan(payload, "X0-tau", 0x3000U, 0x0800U),
                cube_detail::L1Span(
                    head, "Akk-q00", args.workgroupId,
                    akk + Akk2BPackPolicy::kQ00.offset,
                    Akk2BPackPolicy::kQ00.size, l1Generation));
            args.ops->Load(
                Stage::C4,
                cube_detail::CubeSubspan(payload, "q01-zero", 0x4000U,
                                          0x0800U),
                cube_detail::L1Span(
                    head, "Akk-q01", args.workgroupId,
                    akk + Akk2BPackPolicy::kQ01.offset,
                    Akk2BPackPolicy::kQ01.size, l1Generation));
            args.ops->Load(
                Stage::C4,
                cube_detail::CubeSubspan(payload, "X1-tau", 0x3800U, 0x0800U),
                cube_detail::L1Span(
                    head, "Akk-q11", args.workgroupId,
                    akk + Akk2BPackPolicy::kQ11.offset,
                    Akk2BPackPolicy::kQ11.size, l1Generation));
        } else {
            // X0/X1 already occupy q00/q11. q01 must be zero-filled directly
            // at final address by a verified MTE2 fill, never by L1 movement.
            const Offset q01 = cube_detail::AkkFp32Resident(head) + 0x1000U;
            args.ops->Fill(
                Stage::C4,
                cube_detail::L1Span(head, "Akk-q01-zero", args.workgroupId,
                                    q01, 0x1000U, l1Generation),
                0U);
        }

        // Payload can be renamed only after every selected MTE2 above drains.
        args.sync->Set(SyncPoint::C4PayloadFree, head.workspaceSlot,
                       workspaceGeneration, Stage::C4, Pipe::Mte2);
        args.sync->Set(SyncPoint::C4AkkPrepReady, head.l1BankId,
                       l1Generation, Stage::C4, Pipe::Mte2);

        // C4 consumes only Stage-entry B/X0. Its T result is not read in C4.
        const Offset t = cube_detail::TResident(head, args.key.akkStorage);
        const BufferSpan tL0c = cube_detail::SymbolicL0cSpan(
            head, "T-L0C", args.workgroupId, L0cPolicy::kC4T.size,
            l0cLane + L0cPolicy::kC4T.offset);
        args.ops->Mmad(Stage::C4, bCurrent, x0Resident, tL0c,
                       MatrixStorage::Fp32, MatrixStorage::Fp32, 32U, 32U,
                       32U);
        args.ops->Store(
            Stage::C4, tL0c,
            cube_detail::L1Span(head, "T-resident", args.workgroupId, t,
                                0x1000U, l1Generation));
        args.sync->Set(SyncPoint::C4TReady, head.l1BankId, l1Generation,
                       Stage::C4, Pipe::Fixpipe);
    }
}

inline void RunC5(const CubeStageArgs &args)
{
    if (!cube_detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!head.active) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t l1Generation = head.l1Generation;
        const Offset validRows = args.work->group.chunk.validRows;
        const Offset l0cLane =
            L0cPolicy::HeadLaneBase(head.groupLocalHead);
        // Both producers are required: T comes from Fixpipe, while X1 and the
        // stable Akk quadrants come from C4 MTE2.
        args.sync->Wait(SyncPoint::C4TReady, head.l1BankId, l1Generation,
                        Stage::C5, Pipe::Mte1);
        args.sync->Wait(SyncPoint::C4AkkPrepReady, head.l1BankId,
                        l1Generation, Stage::C5, Pipe::Mte1);
        const BufferSpan x1 = cube_detail::L1Span(
            head, "X1-resident", args.workgroupId,
            cube_detail::X1Resident(head, args.key.akkStorage), 0x1000U,
            l1Generation);
        const BufferSpan t = cube_detail::L1Span(
            head, "T-resident", args.workgroupId,
            cube_detail::TResident(head, args.key.akkStorage), 0x1000U,
            l1Generation);
        const BufferSpan y = cube_detail::SymbolicL0cSpan(
            head, "Y-fp32-L0C", args.workgroupId, L0cPolicy::kC5Y.size,
            l0cLane + L0cPolicy::kC5Y.offset);
        args.ops->Mmad(Stage::C5, x1, t, y, MatrixStorage::Fp32,
                       MatrixStorage::Fp32, 32U, 32U, 32U, false, true);
        if (args.key.akkStorage == AkkStorage::TwoByteAbi) {
            const Offset q10 = cube_detail::Akk2BResident(head) +
                               Akk2BPackPolicy::kQ10.offset;
            args.ops->StoreRounded(
                Stage::C5, y,
                cube_detail::L1Span(head, "Akk-q10-cast", args.workgroupId,
                                    q10, Akk2BPackPolicy::kQ10.size,
                                    l1Generation),
                args.key.inputStorage);
            if (args.key.abi == PrepareAbi::Current) {
                // PROPOSED API gate: the same L0C result must support a second
                // Fixpipe cast destination with GM ld=64. No repeated MMAD or
                // temporary L1/UB rearrangement is permitted.
                constexpr Offset kQuadrant = 32U;
                const Offset bottom = validRows > kQuadrant
                                          ? validRows - kQuadrant
                                          : 0U;
                if (bottom != 0U) {
                    args.ops->StoreRounded(
                        Stage::C5,
                        cube_detail::CubeMatrixSubspan(
                            y, "Y-q10-valid-ld32", 0U, 0U, bottom,
                            kQuadrant, kQuadrant,
                            ShapePolicy::kFp32Bytes),
                        cube_detail::SymbolicGmMatrixOutput(
                            head, "AkkOut-q10-valid-ld64", kQuadrant, 0U,
                            bottom, kQuadrant, ShapePolicy::kBt,
                            ShapePolicy::kStorageBytes,
                            workspaceGeneration),
                        args.key.inputStorage);
                }
            }
        } else {
            const Offset q10 = cube_detail::AkkFp32Resident(head) + 0x2000U;
            args.ops->Store(
                Stage::C5, y,
                cube_detail::L1Span(head, "Akk-q10-fp32", args.workgroupId,
                                    q10, 0x1000U, l1Generation));
        }
        args.sync->Set(SyncPoint::C5AkkReady, head.l1BankId, l1Generation,
                       Stage::C5, Pipe::Fixpipe);
    }
}

inline void RunC7(const CubeStageArgs &args)
{
    if (!cube_detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!head.active) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t l1Generation = head.l1Generation;
        const Offset validRows = args.work->group.chunk.validRows;
        const Offset l0cLane =
            L0cPolicy::HeadLaneBase(head.groupLocalHead);
        args.sync->Wait(SyncPoint::C5AkkReady, head.l1BankId, l1Generation,
                        Stage::C7, Pipe::Mte1);
        args.sync->Wait(SyncPoint::V6RhsReady, head.workspaceSlot,
                        workspaceGeneration, Stage::C7, Pipe::Mte2);

        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        const Offset lane = cube_detail::LaneBase(head);
        const BufferSpan rhs = cube_detail::L1Span(
            head, "C7-planar-RHS", args.workgroupId, lane, 0x8000U,
            l1Generation);
        args.ops->Load(
            Stage::C7,
            cube_detail::CubeSubspan(payload, "Kbeta-and-Vbeta", 0x0000U,
                                     0x8000U),
            rhs);
        // One logical [64,64] @ concat(KbetaG,Vbeta)[64,256] product. The GM
        // payload is two planar [64,128] matrices, not row-interleaved
        // [64,256], so it is physically expanded as two independent N=128
        // MMADs after the single 32 KiB MTE2 load.
        const BufferSpan akk = cube_detail::L1Span(
            head, "Akk-resident", args.workgroupId,
            args.key.akkStorage == AkkStorage::TwoByteAbi
                ? cube_detail::Akk2BResident(head)
                : cube_detail::AkkFp32Resident(head),
            args.key.akkStorage == AkkStorage::TwoByteAbi ? 0x2000U : 0x4000U,
            l1Generation);
        const BufferSpan kBetaG = cube_detail::CubeSubspan(
            rhs, "K-beta-g-RHS", 0x0000U, 0x4000U);
        const BufferSpan vBeta = cube_detail::CubeSubspan(
            rhs, "V-beta-RHS", 0x4000U, 0x4000U);
        const BufferSpan wL0c = cube_detail::SymbolicL0cSpan(
            head, "W-fp32-L0C", args.workgroupId, L0cPolicy::kC7W.size,
            l0cLane + L0cPolicy::kC7W.offset);
        const BufferSpan uL0c = cube_detail::SymbolicL0cSpan(
            head, "U-fp32-L0C", args.workgroupId, L0cPolicy::kC7U.size,
            l0cLane + L0cPolicy::kC7U.offset);
        // The first MMAD targets wL0c. U uses a disjoint L0C region below, so
        // its MMAD cannot overwrite a still-running W Fixpipe source.
        // PROPOSED API gate: MTE1 must assemble the four tight q00/q01/q10/q11
        // quadrants directly into L0A. No L1 relocation is allowed.
        args.ops->MmadQuadrantPackedLhs(
            Stage::C7, akk, kBetaG, wL0c,
            FromInputStorage(args.key.inputStorage),
            FromInputStorage(args.key.inputStorage),
            Akk2BPackPolicy::kQuadrantRows,
            Akk2BPackPolicy::kQuadrantColumns, 128U, 64U);
        if (validRows != 0U) {
            args.ops->StoreRounded(
                Stage::C7,
                cube_detail::CubeMatrixSubspan(
                    wL0c, "W-valid-fp32-ld128", 0U, 0U, validRows,
                    ShapePolicy::kK, ShapePolicy::kK,
                    ShapePolicy::kFp32Bytes),
                cube_detail::SymbolicGmMatrixOutput(
                    head, "W-valid-output-ld128", 0U, 0U, validRows,
                    ShapePolicy::kK, ShapePolicy::kK,
                    ShapePolicy::kStorageBytes, workspaceGeneration),
                args.key.inputStorage);
        }
        // The second MMAD targets uL0c at +32 KiB.
        args.ops->MmadQuadrantPackedLhs(
            Stage::C7, akk, vBeta, uL0c,
            FromInputStorage(args.key.inputStorage),
            FromInputStorage(args.key.inputStorage),
            Akk2BPackPolicy::kQuadrantRows,
            Akk2BPackPolicy::kQuadrantColumns, 128U, 64U);
        args.sync->Set(SyncPoint::L1BankFree, head.l1BankId,
                       l1Generation + 1U, Stage::C7, Pipe::Mte1);
        if (validRows != 0U) {
            args.ops->StoreRounded(
                Stage::C7,
                cube_detail::CubeMatrixSubspan(
                    uL0c, "U-valid-fp32-ld128", 0U, 0U, validRows,
                    ShapePolicy::kV, ShapePolicy::kV,
                    ShapePolicy::kFp32Bytes),
                cube_detail::SymbolicGmMatrixOutput(
                    head, "U-valid-output-ld128", 0U, 0U, validRows,
                    ShapePolicy::kV, ShapePolicy::kV,
                    ShapePolicy::kStorageBytes, workspaceGeneration),
                args.key.inputStorage);
        }
        args.sync->Set(SyncPoint::L0cBankFree, head.l0cBankId,
                       head.l0cGeneration + 1U, Stage::C7,
                       Pipe::Fixpipe);
        // V0/V3/V6/C5 readiness transitively includes every enabled earlier
        // public drain. This Fixpipe completion is therefore the last slot
        // user and directly returns the next workspace ticket.
        args.sync->Set(SyncPoint::SlotFree, head.workspaceSlot,
                       workspaceGeneration + 1U, Stage::C7, Pipe::Fixpipe);
    }
}

} // namespace arch35
} // namespace kda_prepare_pseudocode

#endif // FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_ARCH35_CUBE_H
