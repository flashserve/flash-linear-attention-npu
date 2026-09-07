/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_kda_fwd_prepare_tiling_key.h"
#include "chunk_kda_fwd_prepare_struct.h"
#include "chunk_kda_fwd_prepare_utils.h"

#include "arch22/chunk_kda_fwd_prepare_cube.h"
#include "arch22/chunk_kda_fwd_prepare_vec.h"
#include "arch35/chunk_kda_fwd_prepare_cube.h"
#include "arch35/chunk_kda_fwd_prepare_vec.h"

namespace kda_prepare_pseudocode {

using ResolveChunk = ChunkTask (*)(std::uint32_t chunkOrdinal);

namespace {

bool BindWorkspaceView(Architecture architecture,
                       const WorkspaceSizing &sizing,
                       WorkspaceView &workspace) noexcept
{
    if (!sizing.valid || workspace.backingBytes < sizing.totalBytes) {
        return false;
    }
    workspace.workgroupBase = sizing.workgroupBase;
    workspace.contextOffsetBytes = 0U;
    if (architecture == Architecture::Arch22) {
        workspace.slotStrideBytes =
            arch22_policy::WorkspacePolicy::kSlotStride;
        workspace.contextBytes =
            arch22_policy::WorkspacePolicy::kStagePayload.offset;
        workspace.payloadOffsetBytes =
            arch22_policy::WorkspacePolicy::kStagePayload.offset;
        workspace.payloadBytes =
            arch22_policy::WorkspacePolicy::kStagePayload.size;
        return true;
    }
    workspace.slotStrideBytes = WorkspacePolicy::kSlotStride;
    workspace.contextBytes = WorkspacePolicy::kStagePayload.offset;
    workspace.payloadOffsetBytes = WorkspacePolicy::kStagePayload.offset;
    workspace.payloadBytes = WorkspacePolicy::kStagePayload.size;
    return true;
}

ChunkTask ResolveDenseChunk(std::uint32_t chunkOrdinal) noexcept
{
    return {0, chunkOrdinal, chunkOrdinal, kChunkRows};
}

VectorStageArgs MakeVectorArgs(const WorkItem &item,
                               Architecture architecture,
                               std::uint32_t workgroupId,
                               std::uint32_t aivId,
                               WorkspaceView &workspace, SyncLedger &sync,
                               VectorOps &ops,
                               const RuntimeTiling &tiling) noexcept
{
    return {&item,
            &workspace,
            &sync,
            &ops,
            architecture,
            tiling.key,
            tiling.epsilon,
            tiling.lowerBound,
            tiling.scale,
            tiling.hasDtBias,
            workgroupId,
            aivId,
            kAllGroupLocalHeads};
}

void RunArch35AivBranch(const WorkItem &item, std::uint32_t workgroupId,
                        std::uint32_t aivId, WorkspaceView &workspace,
                        SyncLedger &sync, VectorOps &ops,
                        const RuntimeTiling &tiling) noexcept
{
    VectorStageArgs args = MakeVectorArgs(
        item, Architecture::Arch35, workgroupId, aivId, workspace, sync, ops,
        tiling);

    // 每个函数对应一个物理 Vector 阶段和一次符号化 VF 调用。
    // V3 和 V6 自行执行等待，因此此处源码顺序不表示 AIV 一定先于对应的 AIC 分支执行。
    arch35::RunV0(args);
    arch35::RunV1(args);
    arch35::RunV3(args);
    arch35::RunV6(args);
}

void RunArch22AivBranch(const WorkItem &item, std::uint32_t workgroupId,
                        std::uint32_t aivId, WorkspaceView &workspace,
                        SyncLedger &sync, VectorOps &ops,
                        const RuntimeTiling &tiling) noexcept
{
    VectorStageArgs args = MakeVectorArgs(
        item, Architecture::Arch22, workgroupId, aivId, workspace, sync, ops,
        tiling);
    const std::uint32_t pairWaves = static_cast<std::uint32_t>(
        CeilDiv(item.group.activeHeads, kAivPerWorkgroup));

    // Arch22 的每个 AIV 只有一块 40 KiB 共享区。先针对配对波次中的一个头完成
    // V0 -> V1，再选择下一块 72 KiB 私有缓冲区，从而保留共享区中的 G。
    // 非活动伙伴仍需进入 Arch22 辅助函数，使 0x2 模式集合操作能发出必需的空令牌。
    for (std::uint32_t pair = 0; pair < pairWaves; ++pair) {
        args.selectedGroupLocalHead = pair * kAivPerWorkgroup + aivId;
        arch22::RunV0(args);
        arch22::RunV1(args);
    }
    for (std::uint32_t pair = 0; pair < pairWaves; ++pair) {
        args.selectedGroupLocalHead = pair * kAivPerWorkgroup + aivId;
        arch22::RunV3(args);
    }
    for (std::uint32_t pair = 0; pair < pairWaves; ++pair) {
        args.selectedGroupLocalHead = pair * kAivPerWorkgroup + aivId;
        arch22::RunV6(args);
    }
}

CubeStageArgs MakeCubeArgs(const WorkItem &item,
                           Architecture architecture,
                           std::uint32_t workgroupId,
                           WorkspaceView &workspace, SyncLedger &sync,
                           CubeOps &ops,
                           const RuntimeTiling &tiling) noexcept
{
    return {&item,
            &workspace,
            &sync,
            &ops,
            architecture,
            tiling.key,
            tiling.epsilon,
            tiling.lowerBound,
            tiling.scale,
            workgroupId};
}

void RunArch35AicBranch(const WorkItem &item, std::uint32_t workgroupId,
                        WorkspaceView &workspace, SyncLedger &sync,
                        CubeOps &ops, const RuntimeTiling &tiling) noexcept
{
    CubeStageArgs args = MakeCubeArgs(item, Architecture::Arch35, workgroupId,
                                      workspace, sync, ops, tiling);

    // 对每个 head，C2 将八个数学分数乘积按 Qplus/Kplus 行堆叠为四次 MMAD
    // 提交。C4、C5 和 C7 各自消费前序阶段的结果，因此必须是独立的物理阶段。
    arch35::RunC2(args);
    arch35::RunC4(args);
    arch35::RunC5(args);
    arch35::RunC7(args);
}

void RunArch22AicBranch(const WorkItem &item, std::uint32_t workgroupId,
                        WorkspaceView &workspace, SyncLedger &sync,
                        CubeOps &ops, const RuntimeTiling &tiling) noexcept
{
    CubeStageArgs args = MakeCubeArgs(item, Architecture::Arch22, workgroupId,
                                      workspace, sync, ops, tiling);
    arch22::RunC2(args);
    arch22::RunC4(args);
    arch22::RunC5(args);
    arch22::RunC7(args);
}

void Dispatch(const WorkItem &item, CoreRole role, std::uint32_t workgroupId,
              std::uint32_t aivId, WorkspaceView &workspace,
              SyncLedger &sync, VectorOps &vectorOps,
              CubeOps &cubeOps, const RuntimeTiling &tiling) noexcept
{
    if (item.group.activeHeads == 0) {
        return;
    }
    if (role == CoreRole::Aiv) {
        if (aivId < kAivPerWorkgroup) {
            if (tiling.architecture == Architecture::Arch22) {
                RunArch22AivBranch(item, workgroupId, aivId, workspace, sync,
                                   vectorOps, tiling);
            } else {
                RunArch35AivBranch(item, workgroupId, aivId, workspace, sync,
                                   vectorOps, tiling);
            }
        }
        return;
    }
    if (role == CoreRole::Aic) {
        if (tiling.architecture == Architecture::Arch22) {
            RunArch22AicBranch(item, workgroupId, workspace, sync, cubeOps,
                               tiling);
        } else {
            RunArch35AicBranch(item, workgroupId, workspace, sync, cubeOps,
                               tiling);
        }
    }
}

void DispatchHeadPartition(
    const CorePlan &plan, std::uint64_t ordinal, const ChunkTask &chunk,
    std::uint32_t partitionOrdinal, const RuntimeTiling &tiling,
    CoreRole role, std::uint32_t workgroupId, std::uint32_t aivId,
    WorkspaceView &workspace, SyncLedger &sync, VectorOps &vectorOps,
    CubeOps &cubeOps, OwnerTicketState &ownerTickets) noexcept
{
    const std::uint32_t headBegin = PartitionValueHeadBegin(
        partitionOrdinal, tiling.headCount, tiling.qkHeadCount);
    const std::uint32_t headEnd = PartitionValueHeadEnd(
        partitionOrdinal, tiling.headCount, tiling.qkHeadCount);
    std::uint32_t groupOrdinal =
        partitionOrdinal * HeadGroupsPerPartition(
                               tiling.headCount, tiling.qkHeadCount);
    for (std::uint32_t groupBegin = headBegin; groupBegin < headEnd;
         groupBegin += kHeadsPerGroup, ++groupOrdinal) {
        const WorkItem item = BuildWorkItemRange(
            plan, ordinal, chunk, groupOrdinal, groupBegin, headEnd,
            tiling.headCount, ownerTickets, tiling.qkHeadCount);
        Dispatch(item, role, workgroupId, aivId, workspace, sync, vectorOps,
                 cubeOps, tiling);
    }
}

} // namespace

// 仅为待实现的控制入口。此处刻意不实现为 __global__ 核函数，也不声明 GM ABI、
// TilingKey 注册、任务混合宏或设备 API。AIC 与两个 AIV 角色在概念上使用同一个
// CorePlan 执行相同调度，并且只通过具名的就绪/空闲账本通信。
void RunChunkKdaFwdPreparePseudocode(
    const RuntimeTiling &tiling, std::uint32_t workgroupId, CoreRole role,
    std::uint32_t aivId, WorkspaceView &workspace, SyncLedger &sync,
    VectorOps &vectorOps, CubeOps &cubeOps,
    ResolveChunk resolveChunk) noexcept
{
    if (!IsKnownArchitecture(tiling.architecture) ||
        tiling.aicWorkgroupCount == 0 || tiling.totalChunks == 0 ||
        !IsValidHeadMapping(tiling.headCount, tiling.qkHeadCount) ||
        workgroupId >= tiling.aicWorkgroupCount) {
        return;
    }
    const WorkspaceSizing sizing = CheckedWorkspaceSizing(
        tiling.architecture, tiling.aicWorkgroupCount, workgroupId);
    if (!BindWorkspaceView(tiling.architecture, sizing, workspace)) {
        return;
    }
    if (resolveChunk == nullptr) {
        resolveChunk = ResolveDenseChunk;
    }

    const CorePlan plan = BuildCorePlan(tiling, workgroupId);
    OwnerTicketState ownerTickets{};
    for (std::uint64_t ordinal = plan.begin; ordinal < plan.end; ++ordinal) {
        std::uint32_t chunkOrdinal = 0;
        std::uint32_t partitionOrdinal = 0;
        DecodeChunkHeadPartitionOrdinal(plan, ordinal, chunkOrdinal,
                                        partitionOrdinal);
        const ChunkTask chunk = resolveChunk(chunkOrdinal);
        // 主机分块配置必须忽略空序列，并拒绝不在 [1,64] 范围内的描述符。
        // 符号化核函数同样拒绝进入阶段，避免 V6 对空分块计算 validRows - 1。
        if (chunk.validRows == 0 || chunk.validRows > kChunkRows) {
            continue;
        }

        if (plan.mode == PartitionMode::ChunkOnly) {
            // 分块优先：一个工作组拥有完整分块。内部顺序遵循完整的 HK 同源头组，
            // 使每个 HK 仅生成一次的 Q/K 缓存在所有映射的 HV 消费者间保持存活。
            for (partitionOrdinal = 0U;
                 partitionOrdinal < plan.headPartitionCount;
                 ++partitionOrdinal) {
                DispatchHeadPartition(
                    plan, ordinal, chunk, partitionOrdinal, tiling, role,
                    workgroupId, aivId, workspace, sync, vectorOps, cubeOps,
                    ownerTickets);
            }
            continue;
        }

        // 只有分块无法填满 AIC 工作组时才回退到头分核。展平单元是由完整 HK 同源头组
        // 组成的不可拆分任务包，而不是任意四个 HV 组成的分组，因此不会重复执行 HK 归一化。
        DispatchHeadPartition(plan, ordinal, chunk, partitionOrdinal, tiling,
                              role, workgroupId, aivId, workspace, sync,
                              vectorOps, cubeOps, ownerTickets);
    }
}

} // namespace kda_prepare_pseudocode
