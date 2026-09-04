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
    BufferSpan span = CubeSubspan(
        parent, name, relativeOffset,
        MatrixFootprintBytes(rows, columns, leadingDimension, elementBytes));
    span.rows = rows;
    span.columns = columns;
    span.leadingDimension = leadingDimension;
    span.elementBytes = elementBytes;
    return span;
}

inline BufferSpan L1Span(const HeadTask &head, const char *name,
                         std::uint32_t aicId, Offset offset, Offset bytes,
                         std::uint64_t generation)
{
    return {name, MemorySpace::L1, offset, bytes, head.l1BankId,
            generation, CoreRole::Aic, aicId};
}

inline BufferSpan L1MatrixSpan(
    const HeadTask &head, const char *name, std::uint32_t aicId,
    Offset offset, Offset rows, Offset columns, Offset leadingDimension,
    Offset elementBytes, std::uint64_t generation)
{
    BufferSpan span = L1Span(
        head, name, aicId, offset,
        MatrixFootprintBytes(rows, columns, leadingDimension, elementBytes),
        generation);
    span.rows = rows;
    span.columns = columns;
    span.leadingDimension = leadingDimension;
    span.elementBytes = elementBytes;
    return span;
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

inline BufferSpan L0OperandSpan(const HeadTask &head, const char *name,
                                MemorySpace space, std::uint32_t aicId,
                                Offset offset, Offset bytes,
                                std::uint64_t generation)
{
    BufferSpan span{name, space, offset, bytes, head.l0OperandBankId,
                    generation, CoreRole::Aic, aicId};
    span.logicalHeadId = head.headId;
    return span;
}

inline BufferSpan SymbolicGmMatrixSpan(
    const HeadTask &head, const char *name, Offset row, Offset column,
    Offset rows, Offset columns, Offset leadingDimension,
    Offset elementBytes, std::uint64_t generation)
{
    const Offset relativeOffset =
        (row * leadingDimension + column) * elementBytes;
    const Offset footprint = MatrixFootprintBytes(
        rows, columns, leadingDimension, elementBytes);
    BufferSpan span{name,
                    MemorySpace::Gm,
                    relativeOffset,
                    footprint,
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

inline bool ValidArgs(const CubeStageArgs &args)
{
    return args.work != nullptr && args.workspace != nullptr &&
           args.sync != nullptr && args.ops != nullptr &&
           IsSupportedTilingKey(args.key);
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

inline void RequireMte2ToMte1Inputs(const SyncLedger &sync,
                                    Stage stage) noexcept
{
    // PROPOSED local input-ready edge. Every GM/workspace -> L1 MTE2 load in
    // this stage must complete before MTE1 transfers an MMAD operand. The
    // exact A5/CANN HardEvent remains a target-version compile gate.
    sync.Local(LocalDependency::Mte2ToMte1Inputs, stage);
}

inline void RequireCubeToMte1OperandReuse(const SyncLedger &sync,
                                          Stage stage) noexcept
{
    // PROPOSED local operand-release edge. Call only after every independent
    // MMAD reader for the current C2 band or Cube stage has consumed its
    // L0A/L0B lane; a later MTE1 transfer may then overwrite that lane.
    sync.Local(LocalDependency::CubeToMte1OperandReuse, stage);
}

inline void RequireMte2FillToLoadWaw(const SyncLedger &sync,
                                     Stage stage) noexcept
{
    // PROPOSED local MTE2 WAW edge. A partial public AkkOut reload first
    // zero-fills its final tight L1 quadrant, then overwrites only valid rows.
    // Source order is not a substitute for the target A5/CANN event/barrier.
    sync.Local(LocalDependency::Mte2FillToLoadWaw, stage);
}

inline void RequireCubeToFixpipeOutput(const SyncLedger &sync,
                                       Stage stage) noexcept
{
    // PROPOSED local result-ready edge. Cube must finish producing the named
    // L0C region before Fixpipe reads it. A Fixpipe-tagged owner token cannot
    // make the preceding Store or StoreRounded safe by itself.
    sync.Local(LocalDependency::CubeToFixpipeOutput, stage);
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
        cube_detail::RequireMte2ToMte1Inputs(*args.sync, Stage::C2);
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
            const std::uint64_t operandGeneration =
                L0OperandGenerationFor(head, C2L0OperandUse(s));
            const BufferSpan rawAqkL0c = cube_detail::SymbolicL0cSpan(
                head, "raw-Aqk-L0C", args.workgroupId, resultBytes,
                l0cLane + L0cPolicy::kC2AqkOffset[s]);
            const BufferSpan qL0a = cube_detail::L0OperandSpan(
                head, "C2-Q-L0A", MemorySpace::L0A, args.workgroupId,
                L0aPolicy::kC2Q.offset, L0aPolicy::kC2Q.size,
                operandGeneration);
            const BufferSpan kL0a = cube_detail::L0OperandSpan(
                head, "C2-K-L0A", MemorySpace::L0A, args.workgroupId,
                L0aPolicy::kC2K.offset, L0aPolicy::kC2K.size,
                operandGeneration);
            const BufferSpan kMinusL0b = cube_detail::L0OperandSpan(
                head, "C2-Kminus-L0B", MemorySpace::L0B,
                args.workgroupId, L0bPolicy::kC2KMinus.offset,
                ShapePolicy::kKMinusBytes[s], operandGeneration);
            // The two independent products use disjoint Q/K L0A spans and the
            // same read-only Kminus L0B prefix. One release follows both MMADs.
            args.ops->Mmad(Stage::C2, qBand, kMinus, rawAqkL0c, qL0a,
                           kMinusL0b,
                           FromScoreStorage(args.key.scoreStorage),
                           FromScoreStorage(args.key.scoreStorage),
                           C2Policy::kM, physicalN, C2Policy::kK, true);
            cube_detail::RequireCubeToFixpipeOutput(*args.sync, Stage::C2);
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
            args.ops->Mmad(Stage::C2, kBand, kMinus, rawAkkL0c, kL0a,
                           kMinusL0b,
                           FromScoreStorage(args.key.scoreStorage),
                           FromScoreStorage(args.key.scoreStorage),
                           C2Policy::kM, physicalN, C2Policy::kK, true);
            cube_detail::RequireCubeToMte1OperandReuse(*args.sync,
                                                       Stage::C2);
            cube_detail::RequireCubeToFixpipeOutput(*args.sync, Stage::C2);
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
        // Inactive bands and active-band suffix columns are not launched. V3
        // predicates raw loads to this physically defined domain.
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
        const Offset validRows = args.work->group.chunk.validRows;
        const Offset quadrantRows = Akk2BPackPolicy::kQuadrantRows;
        const bool hasQ10 = validRows > quadrantRows;
        const bool currentAbi = args.key.abi == PrepareAbi::Current;
        args.sync->Wait(SyncPoint::C2ScoreL1Free, head.l1BankId,
                        l1Generation, Stage::C4, Pipe::Mte2);
        args.sync->Wait(SyncPoint::V3VcsReady, head.workspaceSlot,
                        workspaceGeneration, Stage::C4, Pipe::Mte2);

        if (!hasQ10) {
            const Offset akk = cube_detail::Akk2BResident(head);
            const BufferSpan q00 = cube_detail::L1MatrixSpan(
                head, "Akk-q00", args.workgroupId,
                akk + Akk2BPackPolicy::kQ00.offset, quadrantRows,
                Akk2BPackPolicy::kQuadrantColumns,
                Akk2BPackPolicy::kQuadrantColumns,
                ShapePolicy::kStorageBytes, l1Generation);
            if (currentAbi) {
                if (validRows < quadrantRows) {
                    args.ops->Fill(Stage::C4, q00, 0U);
                    cube_detail::RequireMte2FillToLoadWaw(*args.sync,
                                                          Stage::C4);
                }
                // PROPOSED 2-D DataCopy gate: public AkkOut has ld=64 while
                // the final q00 resident is tight ld=32. No L1 relocation is
                // permitted after this direct strided-to-tight load.
                args.ops->Load(
                    Stage::C4,
                    cube_detail::SymbolicGmMatrixSpan(
                        head, "AkkOut-q00-valid-ld64", 0U, 0U, validRows,
                        Akk2BPackPolicy::kQuadrantColumns, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes, workspaceGeneration),
                    cube_detail::L1MatrixSpan(
                        head, "Akk-q00-valid-ld32", args.workgroupId,
                        akk + Akk2BPackPolicy::kQ00.offset, validRows,
                        Akk2BPackPolicy::kQuadrantColumns,
                        Akk2BPackPolicy::kQuadrantColumns,
                        ShapePolicy::kStorageBytes, l1Generation));
            } else {
                args.ops->Load(
                    Stage::C4,
                    cube_detail::CubeMatrixSubspan(
                        cube_detail::CubeSubspan(
                            payload, "X0-tau-parent", 0x3000U, 0x0800U),
                        "X0-tau", 0U, 0U, quadrantRows,
                        Akk2BPackPolicy::kQuadrantColumns,
                        Akk2BPackPolicy::kQuadrantColumns,
                        ShapePolicy::kStorageBytes),
                    q00);
            }
            args.sync->Set(SyncPoint::C4PayloadFree, head.workspaceSlot,
                           workspaceGeneration, Stage::C4,
                           currentAbi ? Pipe::Control : Pipe::Mte2);
            args.sync->Set(SyncPoint::C4AkkPrepReady, head.l1BankId,
                           l1Generation, Stage::C4, Pipe::Mte2);
            continue;
        }

        const Offset lane = cube_detail::LaneBase(head);
        const Offset l0cLane =
            L0cPolicy::HeadLaneBase(head.groupLocalHead);
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
            const Offset columns = Akk2BPackPolicy::kQuadrantColumns;
            const Offset bottomRows = validRows - quadrantRows;
            const BufferSpan q00 = cube_detail::L1MatrixSpan(
                head, "Akk-q00", args.workgroupId,
                akk + Akk2BPackPolicy::kQ00.offset, quadrantRows, columns,
                columns, ShapePolicy::kStorageBytes, l1Generation);
            const BufferSpan q01 = cube_detail::L1MatrixSpan(
                head, "Akk-q01", args.workgroupId,
                akk + Akk2BPackPolicy::kQ01.offset, quadrantRows, columns,
                columns, ShapePolicy::kStorageBytes, l1Generation);
            const BufferSpan q11 = cube_detail::L1MatrixSpan(
                head, "Akk-q11", args.workgroupId,
                akk + Akk2BPackPolicy::kQ11.offset, quadrantRows, columns,
                columns, ShapePolicy::kStorageBytes, l1Generation);
            if (currentAbi) {
                // PROPOSED 2-D DataCopy gate: each source is a public AkkOut
                // rectangle with ld=64; each destination is its final tight
                // ld=32 quadrant. These loads are the sole Current ABI relay.
                args.ops->Load(
                    Stage::C4,
                    cube_detail::SymbolicGmMatrixSpan(
                        head, "AkkOut-q00-ld64", 0U, 0U, quadrantRows,
                        columns, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes, workspaceGeneration),
                    q00);
                // q01 is a required public output but not a useful GM input:
                // install its known zero value directly at the final L1
                // address instead of reading the public zero tile back.
                args.ops->Fill(Stage::C4, q01, 0U);
                if (bottomRows < quadrantRows) {
                    args.ops->Fill(Stage::C4, q11, 0U);
                    cube_detail::RequireMte2FillToLoadWaw(*args.sync,
                                                          Stage::C4);
                }
                args.ops->Load(
                    Stage::C4,
                    cube_detail::SymbolicGmMatrixSpan(
                        head, "AkkOut-q11-valid-ld64", quadrantRows,
                        quadrantRows, bottomRows, columns, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes, workspaceGeneration),
                    cube_detail::L1MatrixSpan(
                        head, "Akk-q11-valid-ld32", args.workgroupId,
                        akk + Akk2BPackPolicy::kQ11.offset, bottomRows,
                        columns, columns, ShapePolicy::kStorageBytes,
                        l1Generation));
            } else {
                args.ops->Load(
                    Stage::C4,
                    cube_detail::CubeMatrixSubspan(
                        cube_detail::CubeSubspan(
                            payload, "X0-tau-parent", 0x3000U, 0x0800U),
                        "X0-tau", 0U, 0U, quadrantRows, columns, columns,
                            ShapePolicy::kStorageBytes),
                    q00);
                // q01 is mathematically zero. Install it once at the final L1
                // address instead of writing and reloading a zero GM tile.
                args.ops->Fill(Stage::C4, q01, 0U);
                args.ops->Load(
                    Stage::C4,
                    cube_detail::CubeMatrixSubspan(
                        cube_detail::CubeSubspan(
                            payload, "X1-tau-parent", 0x3800U, 0x0800U),
                        "X1-tau", 0U, 0U, quadrantRows, columns, columns,
                        ShapePolicy::kStorageBytes),
                    q11);
            }
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
        const std::uint64_t operandGeneration =
            L0OperandGenerationFor(head, L0OperandUse::C4);
        const BufferSpan bL0a = cube_detail::L0OperandSpan(
            head, "C4-B-L0A", MemorySpace::L0A, args.workgroupId,
            L0aPolicy::kC2Q.offset, L0aPolicy::kC2Q.size,
            operandGeneration);
        const BufferSpan x0L0b = cube_detail::L0OperandSpan(
            head, "C4-X0-L0B", MemorySpace::L0B, args.workgroupId,
            L0bPolicy::kC2KMinus.offset, L0aPolicy::kC2Q.size,
            operandGeneration);
        cube_detail::RequireMte2ToMte1Inputs(*args.sync, Stage::C4);
        args.ops->Mmad(Stage::C4, bCurrent, x0Resident, tL0c, bL0a, x0L0b,
                       MatrixStorage::Fp32, MatrixStorage::Fp32, 32U, 32U,
                       32U);
        cube_detail::RequireCubeToMte1OperandReuse(*args.sync, Stage::C4);
        cube_detail::RequireCubeToFixpipeOutput(*args.sync, Stage::C4);
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
        const std::uint64_t l1Generation = head.l1Generation;
        const Offset validRows = args.work->group.chunk.validRows;
        const bool hasQ10 = validRows > Akk2BPackPolicy::kQuadrantRows;
        if (!hasQ10) {
            // C4's MTE2 completion makes q00 resident. No T or q10 exists for
            // this tail, so C5 is a pure stage/owner-ticket pass-through.
            args.sync->Wait(SyncPoint::C4AkkPrepReady, head.l1BankId,
                            l1Generation, Stage::C5, Pipe::Control);
            args.sync->Set(SyncPoint::C5AkkReady, head.l1BankId,
                           l1Generation, Stage::C5, Pipe::Control);
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
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
        const std::uint64_t operandGeneration =
            L0OperandGenerationFor(head, L0OperandUse::C5);
        const BufferSpan x1L0a = cube_detail::L0OperandSpan(
            head, "C5-X1-L0A", MemorySpace::L0A, args.workgroupId,
            L0aPolicy::kC2Q.offset, L0aPolicy::kC2Q.size,
            operandGeneration);
        const BufferSpan tL0b = cube_detail::L0OperandSpan(
            head, "C5-T-L0B", MemorySpace::L0B, args.workgroupId,
            L0bPolicy::kC2KMinus.offset, L0aPolicy::kC2Q.size,
            operandGeneration);
        args.ops->Mmad(Stage::C5, x1, t, y, x1L0a, tL0b,
                       MatrixStorage::Fp32,
                       MatrixStorage::Fp32, 32U, 32U, 32U, false, true);
        cube_detail::RequireCubeToMte1OperandReuse(*args.sync, Stage::C5);
        cube_detail::RequireCubeToFixpipeOutput(*args.sync, Stage::C5);
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
                        cube_detail::SymbolicGmMatrixSpan(
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
        if (head.qkLastConsumer) {
            // Cohort heads are contiguous and C7 visits them in ascending
            // order, including earlier local groups for R>4. Reaching the
            // last head therefore means this AIC has observed V6RhsReady from
            // every AIV cache reader, independent of AIV completion order.
            args.sync->Set(SyncPoint::QkCacheFree, head.qkCacheSlot,
                           head.qkCacheGeneration + 1U, Stage::C7,
                           Pipe::Control);
        }

        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        const Offset lane = cube_detail::LaneBase(head);
        const bool hasQ10 =
            validRows > Akk2BPackPolicy::kQuadrantRows;
        const Offset rhsRows =
            hasQ10 ? ShapePolicy::kBt : Akk2BPackPolicy::kQuadrantRows;
        const Offset rhsPlaneBytes =
            rhsRows * ShapePolicy::kK * ShapePolicy::kStorageBytes;
        const BufferSpan rhs = cube_detail::L1Span(
            head, "C7-planar-RHS", args.workgroupId, lane, 0x8000U,
            l1Generation);
        if (hasQ10) {
            args.ops->Load(
                Stage::C7,
                cube_detail::CubeSubspan(payload, "Kbeta-and-Vbeta",
                                         0x0000U, 0x8000U),
                rhs);
        } else {
            // Preserve the fixed plane offsets while avoiding both unused
            // bottom halves. No L1 movement or tail-dependent renaming occurs.
            args.ops->Load(
                Stage::C7,
                cube_detail::CubeSubspan(payload, "K-beta-g-top32", 0x0000U,
                                         rhsPlaneBytes),
                cube_detail::CubeSubspan(rhs, "K-beta-g-top32", 0x0000U,
                                         rhsPlaneBytes));
            args.ops->Load(
                Stage::C7,
                cube_detail::CubeSubspan(payload, "V-beta-top32", 0x4000U,
                                         rhsPlaneBytes),
                cube_detail::CubeSubspan(rhs, "V-beta-top32", 0x4000U,
                                         rhsPlaneBytes));
        }
        cube_detail::RequireMte2ToMte1Inputs(*args.sync, Stage::C7);
        // The GM payload is two planar [64,128] matrices, not row-interleaved
        // [64,256]. The full path therefore expands one logical product into
        // two N=128 quadrant-packed MMADs; a top-only tail uses q00 and the
        // first 32 rows of each planar RHS instead.
        const BufferSpan akk = cube_detail::L1Span(
            head, "Akk-resident", args.workgroupId,
            args.key.akkStorage == AkkStorage::TwoByteAbi
                ? cube_detail::Akk2BResident(head)
                : cube_detail::AkkFp32Resident(head),
            args.key.akkStorage == AkkStorage::TwoByteAbi ? 0x2000U : 0x4000U,
            l1Generation);
        const BufferSpan kBetaG = cube_detail::CubeSubspan(
            rhs, "K-beta-g-RHS", 0x0000U, rhsPlaneBytes);
        const BufferSpan vBeta = cube_detail::CubeSubspan(
            rhs, "V-beta-RHS", 0x4000U, rhsPlaneBytes);
        const BufferSpan wL0c = cube_detail::SymbolicL0cSpan(
            head, "W-fp32-L0C", args.workgroupId, L0cPolicy::kC7W.size,
            l0cLane + L0cPolicy::kC7W.offset);
        const BufferSpan uL0c = cube_detail::SymbolicL0cSpan(
            head, "U-fp32-L0C", args.workgroupId, L0cPolicy::kC7U.size,
            l0cLane + L0cPolicy::kC7U.offset);
        const Offset akkL0aBytes =
            hasQ10 ? L0aPolicy::kC7Akk.size
                   : Akk2BPackPolicy::kQuadrantBytes;
        const std::uint64_t operandGeneration =
            L0OperandGenerationFor(head, L0OperandUse::C7);
        const BufferSpan akkL0a = cube_detail::L0OperandSpan(
            head, "C7-Akk-L0A", MemorySpace::L0A, args.workgroupId,
            L0aPolicy::kC7Akk.offset, akkL0aBytes, operandGeneration);
        const BufferSpan kBetaL0b = cube_detail::L0OperandSpan(
            head, "C7-Kbeta-L0B", MemorySpace::L0B, args.workgroupId,
            L0bPolicy::kC7KBeta.offset, rhsPlaneBytes, operandGeneration);
        const BufferSpan vBetaL0b = cube_detail::L0OperandSpan(
            head, "C7-Vbeta-L0B", MemorySpace::L0B, args.workgroupId,
            L0bPolicy::kC7VBeta.offset, rhsPlaneBytes, operandGeneration);
        // W and U have disjoint L0C destinations, so the second MMAD cannot
        // overwrite a still-running Fixpipe source from the first.
        if (hasQ10) {
            // PROPOSED API gate: MTE1 must assemble the four tight
            // q00/q01/q10/q11 quadrants directly into L0A. No L1 relocation
            // is allowed.
            args.ops->MmadQuadrantPackedLhs(
                Stage::C7, akk, kBetaG, wL0c, akkL0a, kBetaL0b,
                FromInputStorage(args.key.inputStorage),
                FromInputStorage(args.key.inputStorage),
                Akk2BPackPolicy::kQuadrantRows,
                Akk2BPackPolicy::kQuadrantColumns, 128U, 64U);
        } else {
            // For M<=32, q01 is mathematically zero and bottom output rows are
            // discarded. Consume only V3's tight q00 and the top 32 RHS rows;
            // q10/q11 are neither materialized nor read.
            const BufferSpan q00 = cube_detail::CubeMatrixSubspan(
                akk, "Akk-q00", 0U, 0U,
                Akk2BPackPolicy::kQuadrantRows,
                Akk2BPackPolicy::kQuadrantColumns,
                Akk2BPackPolicy::kQuadrantColumns,
                ShapePolicy::kStorageBytes);
            const BufferSpan kBetaGTop = cube_detail::CubeMatrixSubspan(
                kBetaG, "K-beta-g-top32", 0U, 0U,
                Akk2BPackPolicy::kQuadrantRows, ShapePolicy::kK,
                ShapePolicy::kK, ShapePolicy::kStorageBytes);
            args.ops->Mmad(
                Stage::C7, q00, kBetaGTop, wL0c, akkL0a, kBetaL0b,
                FromInputStorage(args.key.inputStorage),
                FromInputStorage(args.key.inputStorage),
                Akk2BPackPolicy::kQuadrantRows, ShapePolicy::kK,
                Akk2BPackPolicy::kQuadrantColumns);
        }
        cube_detail::RequireCubeToFixpipeOutput(*args.sync, Stage::C7);
        if (validRows != 0U) {
            args.ops->StoreRounded(
                Stage::C7,
                cube_detail::CubeMatrixSubspan(
                    wL0c, "W-valid-fp32-ld128", 0U, 0U, validRows,
                    ShapePolicy::kK, ShapePolicy::kK,
                    ShapePolicy::kFp32Bytes),
                cube_detail::SymbolicGmMatrixSpan(
                    head, "W-valid-output-ld128", 0U, 0U, validRows,
                    ShapePolicy::kK, ShapePolicy::kK,
                    ShapePolicy::kStorageBytes, workspaceGeneration),
                args.key.inputStorage);
        }
        // C7 keeps Kbeta/Vbeta in disjoint L0B regions and shares the same
        // read-only Akk L0A operand. The second MMAD targets uL0c at +32 KiB;
        // policy.h proves one release after both products cannot race MTE1.
        if (hasQ10) {
            args.ops->MmadQuadrantPackedLhs(
                Stage::C7, akk, vBeta, uL0c, akkL0a, vBetaL0b,
                FromInputStorage(args.key.inputStorage),
                FromInputStorage(args.key.valueStorage),
                Akk2BPackPolicy::kQuadrantRows,
                Akk2BPackPolicy::kQuadrantColumns, 128U, 64U);
        } else {
            const BufferSpan q00 = cube_detail::CubeMatrixSubspan(
                akk, "Akk-q00", 0U, 0U,
                Akk2BPackPolicy::kQuadrantRows,
                Akk2BPackPolicy::kQuadrantColumns,
                Akk2BPackPolicy::kQuadrantColumns,
                ShapePolicy::kStorageBytes);
            const BufferSpan vBetaTop = cube_detail::CubeMatrixSubspan(
                vBeta, "V-beta-top32", 0U, 0U,
                Akk2BPackPolicy::kQuadrantRows, ShapePolicy::kV,
                ShapePolicy::kV, ShapePolicy::kStorageBytes);
            args.ops->Mmad(
                Stage::C7, q00, vBetaTop, uL0c, akkL0a, vBetaL0b,
                FromInputStorage(args.key.inputStorage),
                FromInputStorage(args.key.valueStorage),
                Akk2BPackPolicy::kQuadrantRows, ShapePolicy::kV,
                Akk2BPackPolicy::kQuadrantColumns);
        }
        cube_detail::RequireCubeToMte1OperandReuse(*args.sync, Stage::C7);
        cube_detail::RequireCubeToFixpipeOutput(*args.sync, Stage::C7);
        args.sync->Set(SyncPoint::L1BankFree, head.l1BankId,
                       l1Generation + 1U, Stage::C7, Pipe::Mte1);
        if (validRows != 0U) {
            args.ops->StoreRounded(
                Stage::C7,
                cube_detail::CubeMatrixSubspan(
                    uL0c, "U-valid-fp32-ld128", 0U, 0U, validRows,
                    ShapePolicy::kV, ShapePolicy::kV,
                    ShapePolicy::kFp32Bytes),
                cube_detail::SymbolicGmMatrixSpan(
                    head, "U-valid-output-ld128", 0U, 0U, validRows,
                    ShapePolicy::kV, ShapePolicy::kV,
                    ShapePolicy::kStorageBytes, workspaceGeneration),
                args.key.valueStorage);
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
