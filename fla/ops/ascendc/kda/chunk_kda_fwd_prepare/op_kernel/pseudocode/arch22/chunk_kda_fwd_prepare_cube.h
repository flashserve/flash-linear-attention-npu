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
    // 待实现的二维描述符：byteSize 覆盖完整的跨步视图，rows/columns
    // 保留其逻辑载荷尺寸。
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
            L0cGenerationFor(head, use, Architecture::Arch22),
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
    // 待实现：分数数据的 MTE2 搬运必须先完成，紧凑数据的 Fixpipe 写入方
    // 才能复用同一段 GM 载荷区；跨核就绪令牌不能替代这个核内事件。
    sync.Local(LocalDependency::Mte2ToFixpipePayloadReuse, stage);
}

inline void RequireFixpipeToMte2Relay(const SyncLedger &sync,
                                      Stage stage) noexcept
{
    // 待实现 c220 的 FIX->MTE2 事件与 ND2NZ 搬运。具体 API/格式是编译
    // 门禁；禁止使用 PIPE_ALL 或臆造的 PipeBarrier。
    sync.Local(LocalDependency::FixpipeToMte2Relay, stage);
}

inline void RequireMte2FillToLoadWaw(const SyncLedger &sync,
                                     Stage stage) noexcept
{
    // 待实现 c220 的 MTE2 WAW 依赖。异步全 L1 清零必须先完成，随后 GM->L1
    // 才能向相同地址写入有效 Akk 行。仅靠源码顺序不够；具体实现必须使用
    // 经最小编译验证的目标版本 MTE2 屏障/事件。
    sync.Local(LocalDependency::Mte2FillToLoadWaw, stage);
}

inline void RequireMte2ToMte1Inputs(const SyncLedger &sync,
                                    Stage stage) noexcept
{
    // 待实现的本核输入就绪依赖。本阶段的所有 GM/工作空间 -> L1 MTE2
    // 搬入必须先完成，MTE1 随后才能搬运 MMAD 操作数。具体 c220 HardEvent
    // 仍是目标版本的编译门禁。
    sync.Local(LocalDependency::Mte2ToMte1Inputs, stage);
}

inline void RequireCubeToMte1OperandReuse(const SyncLedger &sync,
                                          Stage stage) noexcept
{
    // 待实现的本核操作数释放依赖。仅当当前 C2 分带或 Cube 阶段的所有
    // 独立 MMAD 读取方均已消费各自 L0A/L0B 通道后，后续 MTE1 搬运
    // 才可覆盖该通道。
    sync.Local(LocalDependency::CubeToMte1OperandReuse, stage);
}

inline void RequireCubeToFixpipeOutput(const SyncLedger &sync,
                                       Stage stage) noexcept
{
    // 待实现的本核结果就绪依赖。Cube 必须先完成指定 L0C 区域的生成，
    // Fixpipe 随后才能读取。后续带 Fixpipe 标签的就绪/释放令牌不能
    // 反向保证前序 Store 本身安全。
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
                head, L0cStageUse::C2, Architecture::Arch22);
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
                const Offset packedResultBytes =
                    arch22_policy::L0cPolicy::kC2PackedResultBytes[s];
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
                const BufferSpan packedL0c = NativeMatrixOwner(
                    cube_detail::L0cSpan(
                        args, head, L0cStageUse::C2,
                        "Aqk-Akk-stacked-L0C",
                        {arch22_policy::L0cPolicy::kC2PackedOffset[s],
                         packedResultBytes}),
                    "Aqk-Akk-stacked-L0C",
                    NativeMatrixLayout::L0cFractal,
                    arch22_policy::C2Policy::kPackedM, n,
                    ShapePolicy::kFp32Bytes);
                const std::uint64_t operandGeneration =
                    L0OperandGenerationFor(head, C2L0OperandUse(s));
                // Qplus/Kplus 分别装入 zZ L0A 的逻辑 rows[0,16) 和
                // rows[16,32) tile；二者不按 row-major 连续半区寻址。
                const BufferSpan qkL0a = NativeMatrixOwner(
                    cube_detail::L0OperandSpan(
                        args, head, "C2-QK-stacked-L0A", MemorySpace::L0A,
                        c2L0aBase, arch22_policy::L0aPolicy::kC2LaneBytes,
                        operandGeneration),
                    "C2-QK-stacked-L0A", NativeMatrixLayout::L0aZZ,
                    arch22_policy::C2Policy::kPackedM,
                    arch22_policy::C2Policy::kK,
                    ShapePolicy::kStorageBytes);
                const BufferSpan qL0aTile = NativeMatrixTile(
                    qkL0a, "C2-Qplus-L0A-tile", 0U, 0U,
                    arch22_policy::C2Policy::kM,
                    arch22_policy::C2Policy::kK);
                const BufferSpan kL0aTile = NativeMatrixTile(
                    qkL0a, "C2-Kplus-L0A-tile",
                    arch22_policy::C2Policy::kM, 0U,
                    arch22_policy::C2Policy::kM,
                    arch22_policy::C2Policy::kK);
                const BufferSpan kMinusL0b = cube_detail::L0OperandSpan(
                    args, head, "C2-Kminus-L0B", MemorySpace::L0B,
                    c2L0bBase, ShapePolicy::kKMinusBytes[s],
                    operandGeneration);

                args.ops->MmadRowStackedLhs(
                    Stage::C2, qBand, kBand, kMinus, packedL0c, qkL0a,
                    qL0aTile, kL0aTile, kMinusL0b,
                    FromScoreStorage(args.key.scoreStorage),
                    FromScoreStorage(args.key.scoreStorage), 16U, 16U, n,
                    ShapePolicy::kK, true);
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
                // 待确认的 Arch22 API 约束：必须从同一个
                // MakeLayoutL0C(32,N) owner 选择上下两个逻辑 row tile，
                // 再分别由 Fixpipe 写入紧凑 GM relay。精确的源布局、步长、
                // 模式和 API 均为 PROPOSED，必须在 CANN 9.1/Arch2201
                // 上完成最小编译与设备验证；禁止按 row-major 字节切半。
                args.ops->Store(
                    Stage::C2,
                    NativeMatrixTile(packedL0c, "Aqk-compact-L0C", 0U,
                                     0U, 16U, n),
                    cube_detail::MatrixRect(aqkRelay, "Aqk-compact-relay",
                                            0U, 0U, 16U, n, n,
                                            ShapePolicy::kFp32Bytes));
                args.ops->Store(
                    Stage::C2,
                    NativeMatrixTile(packedL0c, "Akk-compact-L0C", 16U,
                                     0U, 16U, n),
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
                head, L0cStageUse::C4, Architecture::Arch22);
            args.sync->Wait(SyncPoint::C2ScoreL1Free, head.l1BankId,
                            head.l1Generation, Stage::C4,
                            hasQ10 ? Pipe::Mte2 : Pipe::Control);
            args.sync->Wait(SyncPoint::L0cBankFree, head.l0cBankId,
                            l0cGeneration, Stage::C4,
                            hasQ10 ? Pipe::Cube : Pipe::Control);

            if (!hasQ10) {
                // V3 已经完成所有仍有效的 Akk 象限；不加载 VCS、也不生成 T，
                // 仅维持 L1/L0C 代际许可链。
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
        // T/Akk 驻留在载荷区中互不重叠的后缀。确实存在 q10 时，配对发布
        // 位于两个 MTE2 读取方之后；仅含上半部分的尾块没有 VCS 读取方，
        // 通过 Control 传递阶段状态。
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
                head, L0cStageUse::C5, Architecture::Arch22);
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
        // 配对等待汇聚两个 AIV。C7 按配对波次顺序遍历，并已处理所有更早的
        // 本地分组；若当前配对包含协作组的最后一个头，则所有映射的 HV 均已
        // 发布 V6RhsReady；此后只能由 AIC 协调者释放共享 Q/K 缓存代际。
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
                head, L0cStageUse::C7, Architecture::Arch22);
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
                // 待实现一次性行主序 GM -> 完整 Cube 就绪搬运。Fill 补齐
                // 底部尾行；完整行搬运特意包含 q01，因为 Arch22 在最终地址
                // 执行 NZ 子矩阵填充的方案尚未通过编译门禁。
                args.ops->Fill(Stage::C7, akkL1, 0U);
                cube_detail::RequireMte2FillToLoadWaw(*args.sync,
                                                      Stage::C7);
                // 只有左下角 q10 存在 C5 Fixpipe 生产者。对于仅含顶部的尾块，
                // C5 只经 Control 透传；等待不存在的 Fixpipe 事件会导致死锁。
                cube_detail::RequireFixpipeToMte2Relay(*args.sync,
                                                       Stage::C7);
                args.ops->Load(Stage::C7, akkRelay, akkL1);
            } else {
                // 不要从未完整填充的 64x64 NZ 视图派生 q00。这里直接执行
                // 按有效行数、主维度 64 从 GM -> 紧凑 32x32 Cube 就绪搬运，
                // 并写入最终 L1 操作数地址。
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
        // 两条物理通道均完成 U 的 Fixpipe 写回后，才把配对许可广播给
        // 两个 AIV，供下一笔事务使用。
        args.sync->AicPublishPair(SyncPoint::SlotFree, pair,
                                  collectiveGeneration + 1U, Stage::C7,
                                  Pipe::Fixpipe);
    }
}

} // namespace kda_prepare_pseudocode::arch22

#endif // PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H
