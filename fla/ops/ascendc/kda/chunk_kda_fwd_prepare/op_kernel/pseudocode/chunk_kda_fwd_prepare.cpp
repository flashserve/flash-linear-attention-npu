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

        // 分块优先时，一个工作组顺序遍历该分块的所有 HK 同源头分区；只有分块数
        // 不足时才处理展平任务指定的单个头分区。两种模式在此后共用同一段主循环。
        const std::uint32_t partitionBegin =
            plan.mode == PartitionMode::ChunkOnly ? 0U : partitionOrdinal;
        const std::uint32_t partitionEnd =
            plan.mode == PartitionMode::ChunkOnly
                ? plan.headPartitionCount
                : partitionOrdinal + 1U;
        for (std::uint32_t partition = partitionBegin;
             partition < partitionEnd; ++partition) {
            const std::uint32_t headBegin = PartitionValueHeadBegin(
                partition, tiling.headCount, tiling.qkHeadCount);
            const std::uint32_t headEnd = PartitionValueHeadEnd(
                partition, tiling.headCount, tiling.qkHeadCount);
            std::uint32_t groupOrdinal =
                partition * HeadGroupsPerPartition(tiling.headCount,
                                                   tiling.qkHeadCount);

            // 分组边界始终与完整 HK 同源头组对齐，避免同一份 Q/K 归一化结果
            // 被不同任务重复生成。
            for (std::uint32_t groupBegin = headBegin;
                 groupBegin < headEnd;
                 groupBegin += kHeadsPerGroup, ++groupOrdinal) {
                const WorkItem item = BuildWorkItemRange(
                    plan, ordinal, chunk, groupOrdinal, groupBegin, headEnd,
                    tiling.headCount, ownerTickets, tiling.qkHeadCount);
                if (item.group.activeHeads == 0U) {
                    continue;
                }

                if (role == CoreRole::Aiv) {
                    if (aivId >= kAivPerWorkgroup) {
                        continue;
                    }
                    VectorStageArgs args = {
                        &item,
                        &workspace,
                        &sync,
                        &vectorOps,
                        tiling.architecture,
                        tiling.key,
                        tiling.epsilon,
                        tiling.lowerBound,
                        tiling.scale,
                        tiling.hasDtBias,
                        workgroupId,
                        aivId,
                        kAllGroupLocalHeads};

                    if (tiling.architecture == Architecture::Arch22) {
                        const std::uint32_t pairWaves =
                            static_cast<std::uint32_t>(CeilDiv(
                                item.group.activeHeads, kAivPerWorkgroup));

                        // Arch22 每个 AIV 只有一块 40 KiB 共享区。每个配对波次
                        // 必须先完成 V0 -> V1，再选择下一块 72 KiB 私有缓冲区，
                        // 从而保留共享区中的 G。非活动伙伴仍进入阶段并发出空令牌。
                        for (std::uint32_t pair = 0U; pair < pairWaves;
                             ++pair) {
                            args.selectedGroupLocalHead =
                                pair * kAivPerWorkgroup + aivId;
                            arch22::RunV0(args);
                            arch22::RunV1(args);
                        }
                        for (std::uint32_t pair = 0U; pair < pairWaves;
                             ++pair) {
                            args.selectedGroupLocalHead =
                                pair * kAivPerWorkgroup + aivId;
                            arch22::RunV3(args);
                        }
                        for (std::uint32_t pair = 0U; pair < pairWaves;
                             ++pair) {
                            args.selectedGroupLocalHead =
                                pair * kAivPerWorkgroup + aivId;
                            arch22::RunV6(args);
                        }
                    } else {
                        // Arch35 的四个调用分别对应一次物理 VF。V3/V6 在阶段
                        // 内等待 AIC，因此源码相邻不代表跨核存在隐式先后关系。
                        arch35::RunV0(args);
                        arch35::RunV1(args);
                        arch35::RunV3(args);
                        arch35::RunV6(args);
                    }
                    continue;
                }

                if (role != CoreRole::Aic) {
                    continue;
                }
                CubeStageArgs args = {
                    &item,          &workspace,    &sync,
                    &cubeOps,       tiling.architecture,
                    tiling.key,     tiling.epsilon,
                    tiling.lowerBound,
                    tiling.scale,   workgroupId};

                if (tiling.architecture == Architecture::Arch22) {
                    arch22::RunC2(args);
                    arch22::RunC4(args);
                    arch22::RunC5(args);
                    arch22::RunC7(args);
                } else {
                    // C2 将八个数学分数乘积按 Qplus/Kplus 行堆叠为四次
                    // MMAD 提交；C4/C5/C7 各自消费前序结果，必须保持独立阶段。
                    arch35::RunC2(args);
                    arch35::RunC4(args);
                    arch35::RunC5(args);
                    arch35::RunC7(args);
                }
            }
        }
    }
}

} // namespace kda_prepare_pseudocode
