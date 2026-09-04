/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H
#define PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_tiling_key.h"

namespace kda_prepare_pseudocode::arch22 {

inline constexpr bool kCubeDesignCovered = true;

namespace cube_detail {

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

constexpr Offset MatrixFootprintBytes(Offset rows, Offset columns,
                                      Offset leadingDimension,
                                      Offset elementBytes)
{
    return rows == 0U || columns == 0U
               ? 0U
               : ((rows - 1U) * leadingDimension + columns) * elementBytes;
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

inline BufferSpan SymbolicGmRows(const HeadTask &head, const char *name,
                                 Offset rows, Offset columns,
                                 Offset leadingDimension,
                                 Offset elementBytes)
{
    BufferSpan span{name,
                    MemorySpace::Gm,
                    0U,
                    MatrixFootprintBytes(rows, columns, leadingDimension,
                                         elementBytes),
                    head.workspaceSlot,
                    head.workspaceGeneration,
                    CoreRole::Shared,
                    0U,
                    rows,
                    columns,
                    leadingDimension,
                    elementBytes};
    span.logicalHeadId = head.headId;
    return span;
}

inline BufferSpan Payload(const CubeStageArgs &args, const HeadTask &head)
{
    return args.workspace->Span(WorkspaceRegion::SharedPayload,
                                head.workspaceSlot,
                                head.workspaceGeneration);
}

inline BufferSpan L1Span(const CubeStageArgs &args, const HeadTask &head,
                         const char *name, const Region &region)
{
    return {name, MemorySpace::L1, region.offset, region.size, head.l1BankId,
            head.l1Generation, CoreRole::Aic, args.workgroupId};
}

inline BufferSpan CurrentLane(const CubeStageArgs &args,
                              const HeadTask &head, const char *name)
{
    return L1Span(
        args, head, name,
        {arch22_policy::L1Policy::kLaneBase[head.groupLocalHead],
         arch22_policy::L1Policy::kLaneBytes});
}

inline BufferSpan Resident32(const CubeStageArgs &args, const HeadTask &head,
                             const char *name, Offset regionBase)
{
    return L1Span(
        args, head, name,
        {regionBase + head.groupLocalHead *
                          arch22_policy::L1Policy::Akk2BResident::kMatrixStride,
         0x1000U});
}

inline BufferSpan X0Resident(const CubeStageArgs &args,
                             const HeadTask &head)
{
    return Resident32(args, head, "X0-resident",
                      arch22_policy::L1Policy::Akk2BResident::kX0.offset);
}

inline BufferSpan X1Resident(const CubeStageArgs &args,
                             const HeadTask &head)
{
    return Resident32(args, head, "X1-resident",
                      arch22_policy::L1Policy::Akk2BResident::kX1.offset);
}

inline BufferSpan TResident(const CubeStageArgs &args,
                            const HeadTask &head)
{
    return Resident32(args, head, "T-resident",
                      arch22_policy::L1Policy::Akk2BResident::kT.offset);
}

inline BufferSpan AkkResident(const CubeStageArgs &args,
                              const HeadTask &head)
{
    const Offset base =
        arch22_policy::L1Policy::Akk2BResident::kAkkTau.offset +
        head.groupLocalHead *
            arch22_policy::L1Policy::Akk2BResident::kAkkStride;
    return L1Span(args, head, "Akk-cube-ready-resident", {base, 0x2000U});
}

inline BufferSpan L0cSpan(const CubeStageArgs &args, const HeadTask &head,
                          L0cStageUse use, const char *name,
                          const Region &withinLane)
{
    const Offset lane = arch22_policy::PhysicalLane(head.groupLocalHead);
    const Offset base = arch22_policy::L0cPolicy::HeadLaneBase(lane);
    return {name,
            MemorySpace::L0,
            static_cast<std::uint64_t>(base) + withinLane.offset,
            withinLane.size,
            head.l0cBankId,
            L0cGenerationFor(head, use, args.architecture),
            CoreRole::Aic,
            args.workgroupId};
}

inline BufferSpan L0OperandSpan(const CubeStageArgs &args,
                                const HeadTask &head, const char *name,
                                MemorySpace space, Offset offset,
                                Offset bytes, std::uint64_t generation)
{
    const Offset physicalLane =
        arch22_policy::PhysicalLane(head.groupLocalHead);
    BufferSpan span{name, space, offset, bytes, physicalLane, generation,
                    CoreRole::Aic, args.workgroupId};
    span.logicalHeadId = head.headId;
    return span;
}

inline BufferSpan AkkRelay(const CubeStageArgs &args, const HeadTask &head,
                           Offset validRows)
{
    if (args.key.abi == PrepareAbi::Current) {
        return SymbolicGmRows(head, "Akk-output-and-C7-relay", validRows,
                              ShapePolicy::kBt, ShapePolicy::kBt,
                              ShapePolicy::kStorageBytes);
    }
    const BufferSpan relay = Subspan(
        Payload(args, head), "Akk-row-major-relay",
        arch22_policy::WorkspacePolicy::kAkkRowMajor.offset,
        arch22_policy::WorkspacePolicy::kAkkRowMajor.size);
    return MatrixRect(relay, "Akk-row-major-relay-valid", 0U, 0U,
                      validRows, ShapePolicy::kBt, ShapePolicy::kBt,
                      ShapePolicy::kStorageBytes);
}

inline bool ValidArgs(const CubeStageArgs &args)
{
    return args.work != nullptr && args.workspace != nullptr &&
           args.sync != nullptr && args.ops != nullptr &&
           args.architecture == Architecture::Arch22 &&
           IsSupportedTilingKey(args.key);
}

constexpr std::uint32_t ActiveScoreBlocks(std::uint32_t validRows) noexcept
{
    return std::min<std::uint32_t>(
        ShapePolicy::kScoreBlockCount,
        (validRows + ShapePolicy::kScoreBlockRows - 1U) /
            ShapePolicy::kScoreBlockRows);
}

inline void RequireMte2ToFixpipePayloadReuse(const SyncLedger &sync,
                                             Stage stage) noexcept
{
    // PROPOSED: score MTE2 must finish before compact Fixpipe writers reuse
    // the same GM payload. A cross-core ready token is not this local event.
    sync.Local(LocalDependency::Mte2ToFixpipePayloadReuse, stage);
}

inline void RequireFixpipeToMte2Relay(const SyncLedger &sync,
                                      Stage stage) noexcept
{
    // PROPOSED c220 FIX->MTE2 event and ND2NZ transfer. Exact API/format are
    // compile gates; PIPE_ALL and a made-up PipeBarrier are forbidden.
    sync.Local(LocalDependency::FixpipeToMte2Relay, stage);
}

inline void RequireMte2FillToLoadWaw(const SyncLedger &sync,
                                     Stage stage) noexcept
{
    // PROPOSED c220 MTE2 WAW edge. The asynchronous full-L1 zero fill must
    // complete before GM->L1 writes valid Akk rows at the same addresses.
    // Source order is insufficient; the concrete implementation must use the
    // target-version MTE2 barrier/event proven by a minimal compile.
    sync.Local(LocalDependency::Mte2FillToLoadWaw, stage);
}

inline void RequireMte2ToMte1Inputs(const SyncLedger &sync,
                                    Stage stage) noexcept
{
    // PROPOSED local input-ready edge. Every GM/workspace -> L1 MTE2 load for
    // this stage must complete before MTE1 transfers its MMAD operands. The
    // exact c220 HardEvent remains a target-version compile gate.
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

inline void RequireCubeToFixpipeOutput(const SyncLedger &sync,
                                       Stage stage) noexcept
{
    // PROPOSED local result-ready edge. Cube must finish producing the named
    // L0C region before Fixpipe reads it. A later Fixpipe-tagged ready/free
    // token cannot make the preceding Store itself safe.
    sync.Local(LocalDependency::CubeToFixpipeOutput, stage);
}

} // namespace cube_detail

inline void RunC2(const CubeStageArgs &args)
{
    if (!cube_detail::ValidArgs(args)) {
        return;
    }
    for (Offset pair = 0U; pair < 2U; ++pair) {
        if (!PairHasActiveHead(args.work->group, pair)) {
            continue;
        }
        const std::uint64_t collectiveGeneration =
            PairCollectiveGenerationFor(args.work->group, pair);
        args.sync->AicWaitPair(SyncPoint::V1ScoreReady, pair,
                               collectiveGeneration, Stage::C2, Pipe::Mte2);
        for (const HeadTask &head : args.work->group.heads) {
            if (!head.active ||
                arch22_policy::PairWave(head.groupLocalHead) != pair) {
                continue;
            }
            const std::uint64_t l0cGeneration = L0cGenerationFor(
                head, L0cStageUse::C2, args.architecture);
            args.sync->Wait(SyncPoint::L1BankFree, head.l1BankId,
                            head.l1Generation, Stage::C2, Pipe::Mte2);
            args.sync->Wait(SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration, Stage::C2, Pipe::Cube);

            const BufferSpan payload = cube_detail::Payload(args, head);
            const BufferSpan scoreL1 =
                cube_detail::CurrentLane(args, head, "packed-score-L1");
            args.ops->Load(Stage::C2, payload, scoreL1);
            cube_detail::RequireMte2ToFixpipePayloadReuse(*args.sync,
                                                          Stage::C2);
            cube_detail::RequireMte2ToMte1Inputs(*args.sync, Stage::C2);

            const Offset activeBlocks =
                cube_detail::ActiveScoreBlocks(
                    args.work->group.chunk.validRows);
            const Offset physicalLane =
                arch22_policy::PhysicalLane(head.groupLocalHead);
            const Offset c2L0aBase =
                arch22_policy::L0aPolicy::C2LaneBase(physicalLane);
            const Offset c2L0bBase =
                arch22_policy::L0bPolicy::C2LaneBase(physicalLane);
            for (Offset s = 0U; s < activeBlocks; ++s) {
                const Offset n = arch22_policy::C2Policy::kN[s];
                const Offset resultBytes =
                    arch22_policy::L0cPolicy::kC2ResultBytes[s];
                const BufferSpan qBand = cube_detail::Subspan(
                    scoreL1, "Qplus-band",
                    arch22_policy::C2Policy::kQBandOffset[s], 0x1000U);
                const BufferSpan kBand = cube_detail::Subspan(
                    scoreL1, "Kplus-band",
                    arch22_policy::C2Policy::kKBandOffset[s], 0x1000U);
                const BufferSpan kMinus = cube_detail::Subspan(
                    scoreL1, "Kminus-prefix",
                    arch22_policy::C2Policy::kKMinusOffset[s],
                    ShapePolicy::kKMinusBytes[s]);
                const BufferSpan aqkL0c = cube_detail::L0cSpan(
                    args, head, L0cStageUse::C2, "Aqk-compact-L0C",
                    {arch22_policy::L0cPolicy::kC2AqkOffset[s], resultBytes});
                const BufferSpan akkL0c = cube_detail::L0cSpan(
                    args, head, L0cStageUse::C2, "Akk-compact-L0C",
                    {arch22_policy::L0cPolicy::kC2AkkOffset[s], resultBytes});
                const std::uint64_t operandGeneration =
                    L0OperandGenerationFor(head, C2L0OperandUse(s));
                const BufferSpan qL0a = cube_detail::L0OperandSpan(
                    args, head, "C2-Q-L0A", MemorySpace::L0A,
                    c2L0aBase + arch22_policy::L0aPolicy::kC2QOffset,
                    0x1000U, operandGeneration);
                const BufferSpan kL0a = cube_detail::L0OperandSpan(
                    args, head, "C2-K-L0A", MemorySpace::L0A,
                    c2L0aBase + arch22_policy::L0aPolicy::kC2KOffset,
                    0x1000U, operandGeneration);
                const BufferSpan kMinusL0b = cube_detail::L0OperandSpan(
                    args, head, "C2-Kminus-L0B", MemorySpace::L0B,
                    c2L0bBase, ShapePolicy::kKMinusBytes[s],
                    operandGeneration);

                args.ops->Mmad(Stage::C2, qBand, kMinus, aqkL0c, qL0a,
                               kMinusL0b,
                               FromScoreStorage(args.key.scoreStorage),
                               FromScoreStorage(args.key.scoreStorage), 16U,
                               n, ShapePolicy::kK, true);
                args.ops->Mmad(Stage::C2, kBand, kMinus, akkL0c, kL0a,
                               kMinusL0b,
                               FromScoreStorage(args.key.scoreStorage),
                               FromScoreStorage(args.key.scoreStorage), 16U,
                               n, ShapePolicy::kK, true);
                cube_detail::RequireCubeToMte1OperandReuse(*args.sync,
                                                           Stage::C2);
                cube_detail::RequireCubeToFixpipeOutput(*args.sync,
                                                        Stage::C2);
                const BufferSpan aqkRelay = cube_detail::Subspan(
                    payload, "Aqk-compact-relay",
                    arch22_policy::WorkspacePolicy::kRelayRawAqk[s].offset,
                    resultBytes);
                const BufferSpan akkRelay = cube_detail::Subspan(
                    payload, "Akk-compact-relay",
                    arch22_policy::WorkspacePolicy::kRelayRawAkk[s].offset,
                    resultBytes);
                args.ops->Store(
                    Stage::C2,
                    cube_detail::MatrixRect(aqkL0c, "Aqk-compact-L0C", 0U,
                                            0U, 16U, n, n,
                                            ShapePolicy::kFp32Bytes),
                    cube_detail::MatrixRect(aqkRelay, "Aqk-compact-relay",
                                            0U, 0U, 16U, n, n,
                                            ShapePolicy::kFp32Bytes));
                args.ops->Store(
                    Stage::C2,
                    cube_detail::MatrixRect(akkL0c, "Akk-compact-L0C", 0U,
                                            0U, 16U, n, n,
                                            ShapePolicy::kFp32Bytes),
                    cube_detail::MatrixRect(akkRelay, "Akk-compact-relay",
                                            0U, 0U, 16U, n, n,
                                            ShapePolicy::kFp32Bytes));
            }
            args.sync->Set(SyncPoint::C2ScoreL1Free, head.l1BankId,
                           head.l1Generation, Stage::C2, Pipe::Mte1);
            args.sync->Set(SyncPoint::L0cBankFree, head.l0cBankId,
                           l0cGeneration + 1U, Stage::C2, Pipe::Fixpipe);
        }
        args.sync->AicPublishPair(SyncPoint::C2RawReady, pair,
                                  collectiveGeneration, Stage::C2,
                                  Pipe::Fixpipe);
    }
}

inline void RunC4(const CubeStageArgs &args)
{
    if (!cube_detail::ValidArgs(args)) {
        return;
    }
    for (Offset pair = 0U; pair < 2U; ++pair) {
        if (!PairHasActiveHead(args.work->group, pair)) {
            continue;
        }
        const std::uint64_t collectiveGeneration =
            PairCollectiveGenerationFor(args.work->group, pair);
        const bool hasQ10 = args.work->group.chunk.validRows > 32U;
        args.sync->AicWaitPair(SyncPoint::V3VcsReady, pair,
                               collectiveGeneration, Stage::C4,
                               hasQ10 ? Pipe::Mte2 : Pipe::Control);
        for (const HeadTask &head : args.work->group.heads) {
            if (!head.active ||
                arch22_policy::PairWave(head.groupLocalHead) != pair) {
                continue;
            }
            const std::uint64_t l0cGeneration = L0cGenerationFor(
                head, L0cStageUse::C4, args.architecture);
            args.sync->Wait(SyncPoint::C2ScoreL1Free, head.l1BankId,
                            head.l1Generation, Stage::C4,
                            hasQ10 ? Pipe::Mte2 : Pipe::Control);
            args.sync->Wait(SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration, Stage::C4,
                            hasQ10 ? Pipe::Cube : Pipe::Control);

            if (!hasQ10) {
                // V3 already completed the only live Akk quadrants. Preserve
                // the L1/L0C ticket chain without loading VCS or producing T.
                args.sync->Set(SyncPoint::C4AkkPrepReady, head.l1BankId,
                               head.l1Generation, Stage::C4, Pipe::Control);
                args.sync->Set(SyncPoint::L0cBankFree, head.l0cBankId,
                               l0cGeneration + 1U, Stage::C4,
                               Pipe::Control);
                continue;
            }

            const BufferSpan payload = cube_detail::Payload(args, head);
            const BufferSpan current =
                cube_detail::CurrentLane(args, head, "C4-current-L1");
            const BufferSpan bL1 = cube_detail::Subspan(
                current, "B-L1", 0U,
                arch22_policy::WorkspacePolicy::kVcsB.size);
            const BufferSpan x0L1 = cube_detail::X0Resident(args, head);
            const BufferSpan x1L1 = cube_detail::X1Resident(args, head);
            args.ops->Load(
                Stage::C4,
                cube_detail::Subspan(
                    payload, "B-relay",
                    arch22_policy::WorkspacePolicy::kVcsB.offset,
                    arch22_policy::WorkspacePolicy::kVcsB.size),
                bL1);
            args.ops->Load(
                Stage::C4,
                cube_detail::Subspan(
                    payload, "X0-relay",
                    arch22_policy::WorkspacePolicy::kVcsX0.offset,
                    arch22_policy::WorkspacePolicy::kVcsX0.size),
                x0L1);
            args.ops->Load(
                Stage::C4,
                cube_detail::Subspan(
                    payload, "X1-relay",
                    arch22_policy::WorkspacePolicy::kVcsX1.offset,
                    arch22_policy::WorkspacePolicy::kVcsX1.size),
                x1L1);
            args.sync->Set(SyncPoint::C4AkkPrepReady, head.l1BankId,
                           head.l1Generation, Stage::C4, Pipe::Mte2);
            cube_detail::RequireMte2ToMte1Inputs(*args.sync, Stage::C4);

            const BufferSpan tL0c = cube_detail::L0cSpan(
                args, head, L0cStageUse::C4, "T-L0C",
                arch22_policy::L0cPolicy::kC4T);
            const Offset physicalLane =
                arch22_policy::PhysicalLane(head.groupLocalHead);
            const std::uint64_t operandGeneration =
                L0OperandGenerationFor(head, L0OperandUse::C4);
            const BufferSpan bL0a = cube_detail::L0OperandSpan(
                args, head, "C4-B-L0A", MemorySpace::L0A,
                arch22_policy::L0aPolicy::C2LaneBase(physicalLane),
                0x1000U, operandGeneration);
            const BufferSpan x0L0b = cube_detail::L0OperandSpan(
                args, head, "C4-X0-L0B", MemorySpace::L0B,
                arch22_policy::L0bPolicy::C2LaneBase(physicalLane),
                0x1000U, operandGeneration);
            args.ops->SetHf32Mode(Stage::C4, false);
            args.ops->Mmad(Stage::C4, bL1, x0L1, tL0c, bL0a, x0L0b,
                           MatrixStorage::Fp32, MatrixStorage::Fp32, 32U, 32U,
                           32U);
            cube_detail::RequireCubeToMte1OperandReuse(*args.sync,
                                                       Stage::C4);
            cube_detail::RequireCubeToFixpipeOutput(*args.sync, Stage::C4);
            const BufferSpan tRelay = cube_detail::Subspan(
                payload, "T-fp32-relay",
                arch22_policy::WorkspacePolicy::kRelayT.offset,
                arch22_policy::WorkspacePolicy::kRelayT.size);
            args.ops->Store(
                Stage::C4,
                cube_detail::MatrixRect(tL0c, "T-L0C", 0U, 0U, 32U, 32U,
                                        32U, ShapePolicy::kFp32Bytes),
                cube_detail::MatrixRect(tRelay, "T-fp32-relay", 0U, 0U,
                                        32U, 32U, 32U,
                                        ShapePolicy::kFp32Bytes));
            args.sync->Set(SyncPoint::C4TReady, head.workspaceSlot,
                           head.workspaceGeneration, Stage::C4,
                           Pipe::Fixpipe);
            args.sync->Set(SyncPoint::L0cBankFree, head.l0cBankId,
                           l0cGeneration + 1U, Stage::C4, Pipe::Fixpipe);
        }
        // T/Akk remain live in disjoint payload suffixes. For a real q10 the
        // pair publish follows both MTE2 readers; a top-only tail has no VCS
        // reader and transfers the phase through Control.
        args.sync->AicPublishPair(
            SyncPoint::C4PayloadFree, pair, collectiveGeneration, Stage::C4,
            hasQ10 ? Pipe::Mte2 : Pipe::Control);
    }
}

inline void RunC5(const CubeStageArgs &args)
{
    if (!cube_detail::ValidArgs(args)) {
        return;
    }
    for (Offset pair = 0U; pair < 2U; ++pair) {
        if (!PairHasActiveHead(args.work->group, pair)) {
            continue;
        }
        for (const HeadTask &head : args.work->group.heads) {
            if (!head.active ||
                arch22_policy::PairWave(head.groupLocalHead) != pair) {
                continue;
            }
            const std::uint64_t l0cGeneration = L0cGenerationFor(
                head, L0cStageUse::C5, args.architecture);
            const Offset validRows = args.work->group.chunk.validRows;
            const Offset bottomRows =
                validRows > 32U ? validRows - 32U : 0U;
            if (bottomRows == 0U) {
                args.sync->Wait(SyncPoint::C4AkkPrepReady, head.l1BankId,
                                head.l1Generation, Stage::C5,
                                Pipe::Control);
                args.sync->Wait(SyncPoint::L0cBankFree, head.l0cBankId,
                                l0cGeneration, Stage::C5, Pipe::Control);
                args.sync->Set(SyncPoint::C5AkkReady, head.workspaceSlot,
                               head.workspaceGeneration, Stage::C5,
                               Pipe::Control);
                args.sync->Set(SyncPoint::L0cBankFree, head.l0cBankId,
                               l0cGeneration + 1U, Stage::C5,
                               Pipe::Control);
                continue;
            }
            args.sync->Wait(SyncPoint::C4TReady, head.workspaceSlot,
                            head.workspaceGeneration, Stage::C5, Pipe::Mte2);
            args.sync->Wait(SyncPoint::C4AkkPrepReady, head.l1BankId,
                            head.l1Generation, Stage::C5, Pipe::Mte1);
            args.sync->Wait(SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration, Stage::C5, Pipe::Cube);

            const BufferSpan payload = cube_detail::Payload(args, head);
            const BufferSpan tRelay = cube_detail::Subspan(
                payload, "T-fp32-relay",
                arch22_policy::WorkspacePolicy::kRelayT.offset,
                arch22_policy::WorkspacePolicy::kRelayT.size);
            cube_detail::RequireFixpipeToMte2Relay(*args.sync, Stage::C5);
            args.ops->Load(
                Stage::C5,
                cube_detail::MatrixRect(tRelay, "T-fp32-relay", 0U, 0U,
                                        32U, 32U, 32U,
                                        ShapePolicy::kFp32Bytes),
                cube_detail::TResident(args, head));
            cube_detail::RequireMte2ToMte1Inputs(*args.sync, Stage::C5);

            const BufferSpan yL0c = cube_detail::L0cSpan(
                args, head, L0cStageUse::C5, "negative-X1T-L0C",
                arch22_policy::L0cPolicy::kC5Y);
            const Offset physicalLane =
                arch22_policy::PhysicalLane(head.groupLocalHead);
            const std::uint64_t operandGeneration =
                L0OperandGenerationFor(head, L0OperandUse::C5);
            const BufferSpan x1L0a = cube_detail::L0OperandSpan(
                args, head, "C5-X1-L0A", MemorySpace::L0A,
                arch22_policy::L0aPolicy::C2LaneBase(physicalLane),
                0x1000U, operandGeneration);
            const BufferSpan tL0b = cube_detail::L0OperandSpan(
                args, head, "C5-T-L0B", MemorySpace::L0B,
                arch22_policy::L0bPolicy::C2LaneBase(physicalLane),
                0x1000U, operandGeneration);
            args.ops->SetHf32Mode(Stage::C5, false);
            args.ops->Mmad(Stage::C5, cube_detail::X1Resident(args, head),
                           cube_detail::TResident(args, head), yL0c, x1L0a,
                           tL0b,
                           MatrixStorage::Fp32, MatrixStorage::Fp32, 32U, 32U,
                           32U, false, true);
            cube_detail::RequireCubeToMte1OperandReuse(*args.sync,
                                                       Stage::C5);
            cube_detail::RequireCubeToFixpipeOutput(*args.sync, Stage::C5);
            const BufferSpan akkRelay =
                cube_detail::AkkRelay(args, head, validRows);
            args.ops->StoreRounded(
                Stage::C5,
                cube_detail::MatrixRect(
                    yL0c, "Akk-q10-L0C", 0U, 0U, bottomRows, 32U, 32U,
                    ShapePolicy::kFp32Bytes),
                cube_detail::MatrixRect(
                    akkRelay, "Akk-q10-row-major", 32U, 0U, bottomRows,
                    32U, ShapePolicy::kBt, ShapePolicy::kStorageBytes),
                args.key.inputStorage);
            args.sync->Set(SyncPoint::C5AkkReady, head.workspaceSlot,
                           head.workspaceGeneration, Stage::C5,
                           Pipe::Fixpipe);
            args.sync->Set(SyncPoint::L0cBankFree, head.l0cBankId,
                           l0cGeneration + 1U, Stage::C5, Pipe::Fixpipe);
        }
    }
}

inline void RunC7(const CubeStageArgs &args)
{
    if (!cube_detail::ValidArgs(args)) {
        return;
    }
    for (Offset pair = 0U; pair < 2U; ++pair) {
        if (!PairHasActiveHead(args.work->group, pair)) {
            continue;
        }
        const std::uint64_t collectiveGeneration =
            PairCollectiveGenerationFor(args.work->group, pair);
        args.sync->AicWaitPair(SyncPoint::V6RhsReady, pair,
                               collectiveGeneration, Stage::C7, Pipe::Mte2);
        // The pair wait aggregates both AIVs. Because C7 visits pair waves and
        // earlier local groups in order, a cohort-last head in this pair means
        // every mapped HV has published V6RhsReady; only the AIC coordinator
        // may then release the shared Q/K cache generation.
        for (const HeadTask &head : args.work->group.heads) {
            if (head.active && head.qkLastConsumer &&
                arch22_policy::PairWave(head.groupLocalHead) == pair) {
                args.sync->Set(SyncPoint::QkCacheFree, head.qkCacheSlot,
                               head.qkCacheGeneration + 1U, Stage::C7,
                               Pipe::Control);
            }
        }
        for (const HeadTask &head : args.work->group.heads) {
            if (!head.active ||
                arch22_policy::PairWave(head.groupLocalHead) != pair) {
                continue;
            }
            const std::uint64_t l0cGeneration = L0cGenerationFor(
                head, L0cStageUse::C7, args.architecture);
            args.sync->Wait(SyncPoint::C5AkkReady, head.workspaceSlot,
                            head.workspaceGeneration, Stage::C7, Pipe::Mte2);
            args.sync->Wait(SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration, Stage::C7, Pipe::Cube);

            const Offset validRows = args.work->group.chunk.validRows;
            const bool hasQ10 = validRows > 32U;
            const BufferSpan akkL1 = cube_detail::AkkResident(args, head);
            const BufferSpan akkRelay =
                cube_detail::AkkRelay(args, head, validRows);
            BufferSpan akkMmad = akkL1;
            if (hasQ10) {
                // PROPOSED one-shot row-major GM -> full Cube-ready transfer.
                // Fill supplies bottom tail rows; the complete row transfer
                // intentionally includes q01 because an Arch22 final-address
                // NZ submatrix fill has not yet passed the compile gate.
                args.ops->Fill(Stage::C7, akkL1, 0U);
                cube_detail::RequireMte2FillToLoadWaw(*args.sync,
                                                      Stage::C7);
                // Only the bottom-left q10 has a C5 Fixpipe producer. For a
                // top-only tail, C5 is a Control pass-through and waiting on
                // a nonexistent Fixpipe event would deadlock.
                cube_detail::RequireFixpipeToMte2Relay(*args.sync,
                                                       Stage::C7);
                args.ops->Load(Stage::C7, akkRelay, akkL1);
            } else {
                // Do not derive q00 from a partially populated 64x64 NZ view.
                // This is a direct valid-row/ld64 GM -> tight 32x32
                // Cube-ready transfer into the final L1 operand address.
                constexpr Offset kQuadrant = 32U;
                const BufferSpan q00Tight = cube_detail::MatrixRect(
                    akkL1, "Akk-q00-tight-cube-ready", 0U, 0U,
                    kQuadrant, kQuadrant, kQuadrant,
                    ShapePolicy::kStorageBytes);
                args.ops->Fill(Stage::C7, q00Tight, 0U);
                cube_detail::RequireMte2FillToLoadWaw(*args.sync,
                                                      Stage::C7);
                args.ops->Load(
                    Stage::C7,
                    cube_detail::MatrixRect(
                        akkRelay, "Akk-q00-relay-ld64", 0U, 0U,
                        validRows, kQuadrant, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes),
                    cube_detail::MatrixRect(
                        q00Tight, "Akk-q00-valid-tight", 0U, 0U,
                        validRows, kQuadrant, kQuadrant,
                        ShapePolicy::kStorageBytes));
                akkMmad = q00Tight;
            }
            const BufferSpan current =
                cube_detail::CurrentLane(args, head, "C7-RHS-L1");
            const Offset rhsRows = hasQ10 ? ShapePolicy::kBt : 32U;
            const Offset rhsPlaneBytes =
                rhsRows * ShapePolicy::kK * ShapePolicy::kStorageBytes;
            const BufferSpan kBetaL1 = cube_detail::Subspan(
                current, "K-beta-g-L1", 0U, rhsPlaneBytes);
            const BufferSpan vBetaL1 = cube_detail::Subspan(
                current, "V-beta-L1", 0x4000U, rhsPlaneBytes);
            const BufferSpan payload = cube_detail::Payload(args, head);
            args.ops->Load(
                Stage::C7,
                cube_detail::Subspan(
                    payload, "K-beta-g-relay",
                    arch22_policy::WorkspacePolicy::kPostKBetaG.offset,
                    rhsPlaneBytes),
                kBetaL1);
            args.ops->Load(
                Stage::C7,
                cube_detail::Subspan(
                    payload, "V-beta-relay",
                    arch22_policy::WorkspacePolicy::kPostVBeta.offset,
                    rhsPlaneBytes),
                vBetaL1);
            cube_detail::RequireMte2ToMte1Inputs(*args.sync, Stage::C7);

            const BufferSpan wL0c = cube_detail::L0cSpan(
                args, head, L0cStageUse::C7, "W-L0C",
                arch22_policy::L0cPolicy::kC7W);
            const BufferSpan uL0c = cube_detail::L0cSpan(
                args, head, L0cStageUse::C7, "U-L0C",
                arch22_policy::L0cPolicy::kC7U);
            const Offset physicalLane =
                arch22_policy::PhysicalLane(head.groupLocalHead);
            const Offset c7L0aBase =
                arch22_policy::L0aPolicy::C7LaneBase(physicalLane);
            const Offset c7L0bBase =
                arch22_policy::L0bPolicy::C7LaneBase(physicalLane);
            const Offset akkL0aBytes = hasQ10
                                                  ? arch22_policy::L0aPolicy::
                                                        kC7AkkLaneBytes
                                                  : 0x0800U;
            const std::uint64_t operandGeneration =
                L0OperandGenerationFor(head, L0OperandUse::C7);
            const BufferSpan akkL0a = cube_detail::L0OperandSpan(
                args, head, "C7-Akk-L0A", MemorySpace::L0A, c7L0aBase,
                akkL0aBytes, operandGeneration);
            const BufferSpan kBetaL0b = cube_detail::L0OperandSpan(
                args, head, "C7-Kbeta-L0B", MemorySpace::L0B,
                c7L0bBase + arch22_policy::L0bPolicy::kC7KbetaOffset,
                rhsPlaneBytes, operandGeneration);
            const BufferSpan vBetaL0b = cube_detail::L0OperandSpan(
                args, head, "C7-Vbeta-L0B", MemorySpace::L0B,
                c7L0bBase + arch22_policy::L0bPolicy::kC7VbetaOffset,
                rhsPlaneBytes, operandGeneration);
            if (hasQ10) {
                args.ops->Mmad(Stage::C7, akkMmad, kBetaL1, wL0c, akkL0a,
                               kBetaL0b,
                               FromInputStorage(args.key.inputStorage),
                               FromInputStorage(args.key.inputStorage),
                               ShapePolicy::kBt, ShapePolicy::kK,
                               ShapePolicy::kBt);
                args.ops->Mmad(Stage::C7, akkMmad, vBetaL1, uL0c, akkL0a,
                               vBetaL0b,
                               FromInputStorage(args.key.inputStorage),
                               FromInputStorage(args.key.valueStorage),
                               ShapePolicy::kBt, ShapePolicy::kV,
                               ShapePolicy::kBt);
            } else {
                const BufferSpan kBetaTop = cube_detail::MatrixRect(
                    kBetaL1, "K-beta-g-top32", 0U, 0U, 32U,
                    ShapePolicy::kK, ShapePolicy::kK,
                    ShapePolicy::kStorageBytes);
                const BufferSpan vBetaTop = cube_detail::MatrixRect(
                    vBetaL1, "V-beta-top32", 0U, 0U, 32U,
                    ShapePolicy::kV, ShapePolicy::kV,
                    ShapePolicy::kStorageBytes);
                args.ops->Mmad(Stage::C7, akkMmad, kBetaTop, wL0c, akkL0a,
                               kBetaL0b,
                               FromInputStorage(args.key.inputStorage),
                               FromInputStorage(args.key.inputStorage), 32U,
                               ShapePolicy::kK, 32U);
                args.ops->Mmad(Stage::C7, akkMmad, vBetaTop, uL0c, akkL0a,
                               vBetaL0b,
                               FromInputStorage(args.key.inputStorage),
                               FromInputStorage(args.key.valueStorage), 32U,
                               ShapePolicy::kV, 32U);
            }
            cube_detail::RequireCubeToMte1OperandReuse(*args.sync,
                                                       Stage::C7);
            cube_detail::RequireCubeToFixpipeOutput(*args.sync, Stage::C7);
            args.sync->Set(SyncPoint::L1BankFree, head.l1BankId,
                           head.l1Generation + 1U, Stage::C7, Pipe::Mte1);
            args.ops->StoreRounded(
                Stage::C7,
                cube_detail::MatrixRect(
                    wL0c, "W-L0C-valid", 0U, 0U, validRows,
                    ShapePolicy::kK, ShapePolicy::kK,
                    ShapePolicy::kFp32Bytes),
                cube_detail::SymbolicGmRows(
                    head, "W-output", validRows, ShapePolicy::kK,
                    ShapePolicy::kK, ShapePolicy::kStorageBytes),
                args.key.inputStorage);
            args.ops->StoreRounded(
                Stage::C7,
                cube_detail::MatrixRect(
                    uL0c, "U-L0C-valid", 0U, 0U, validRows,
                    ShapePolicy::kV, ShapePolicy::kV,
                    ShapePolicy::kFp32Bytes),
                cube_detail::SymbolicGmRows(
                    head, "U-output", validRows, ShapePolicy::kV,
                    ShapePolicy::kV, ShapePolicy::kStorageBytes),
                args.key.valueStorage);
            args.sync->Set(SyncPoint::L0cBankFree, head.l0cBankId,
                           l0cGeneration + 1U, Stage::C7, Pipe::Fixpipe);
        }
        // Both physical lanes have completed their U Fixpipe store before the
        // pair credit is broadcast to both AIVs for the next transaction.
        args.sync->AicPublishPair(SyncPoint::SlotFree, pair,
                                  collectiveGeneration + 1U, Stage::C7,
                                  Pipe::Fixpipe);
    }
}

} // namespace kda_prepare_pseudocode::arch22

#endif // PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H
