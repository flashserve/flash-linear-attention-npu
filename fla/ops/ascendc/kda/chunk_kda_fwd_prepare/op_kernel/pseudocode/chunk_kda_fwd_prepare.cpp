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

    // Each function is one physical Vector stage and one symbolic VF call.
    // V3 and V6 own their waits, so this source order is not an assertion that
    // AIV runs ahead of the corresponding AIC branch.
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

    // Arch22 has one 40 KiB shared arena per AIV. Preserve G in that arena by
    // completing V0 -> V1 for one pair-wave head before selecting the next
    // private 72 KiB bank. An inactive partner still enters the arch22 helper
    // so a mode-0x2 collective can emit its required dummy token.
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

    // C2 contains the eight independent score MMADs. C4, C5, and C7 are
    // separate physical stages because each consumes a prior-stage result.
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

// PROPOSED control entry only. It deliberately is not a __global__ kernel and
// declares no GM ABI, tiling-key registration, task-mix macro, or device API.
// The AIC and both AIV roles conceptually execute this same schedule with the
// same CorePlan and communicate only through the named ready/free ledger.
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
        // Host tiling must omit empty sequences and reject any descriptor
        // outside [1,64]. The symbolic kernel also refuses to enter a stage,
        // preventing V6 from evaluating validRows - 1 for an empty chunk.
        if (chunk.validRows == 0 || chunk.validRows > kChunkRows) {
            continue;
        }

        if (plan.mode == PartitionMode::ChunkOnly) {
            // Chunk-first: one workgroup owns the complete chunk. Its inner
            // order follows complete HK cohorts so the once-per-HK Q/K cache
            // remains live across every mapped HV consumer.
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

        // Head splitting is only the fallback when chunks cannot fill AIC
        // workgroups. A flattened unit is an indivisible HK cohort pack, not
        // an arbitrary four-HV group, so no HK normalization is duplicated.
        DispatchHeadPartition(plan, ordinal, chunk, partitionOrdinal, tiling,
                              role, workgroupId, aivId, workspace, sync,
                              vectorOps, cubeOps, ownerTickets);
    }
}

} // namespace kda_prepare_pseudocode
