/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_kda_fwd_prepare_tiling_key.h"
#include "chunk_kda_fwd_prepare_struct.h"
#include "chunk_kda_fwd_prepare_utils.h"

#include "arch35/chunk_kda_fwd_prepare_cube.h"
#include "arch35/chunk_kda_fwd_prepare_vec.h"

namespace kda_prepare_pseudocode {

using ResolveChunk = ChunkTask (*)(std::uint32_t chunkOrdinal);

namespace {

ChunkTask ResolveDenseChunk(std::uint32_t chunkOrdinal) noexcept
{
    return {0, chunkOrdinal, chunkOrdinal, kChunkRows};
}

void RunAivBranch(const WorkItem &item, std::uint32_t workgroupId,
                  std::uint32_t aivId, WorkspaceView &workspace,
                  SyncLedger &sync, VectorOps &ops,
                  const RuntimeTiling &tiling) noexcept
{
    VectorStageArgs args{&item,
                         &workspace,
                         &sync,
                         &ops,
                         tiling.key,
                         tiling.epsilon,
                         tiling.lowerBound,
                         tiling.scale,
                         tiling.hasDtBias,
                         workgroupId,
                         aivId};

    // Each function is one physical Vector stage and one symbolic VF call.
    // V3 and V6 own their waits, so this source order is not an assertion that
    // AIV runs ahead of the corresponding AIC branch.
    arch35::RunV0(args);
    arch35::RunV1(args);
    arch35::RunV3(args);
    arch35::RunV6(args);
}

void RunAicBranch(const WorkItem &item, std::uint32_t workgroupId,
                  WorkspaceView &workspace, SyncLedger &sync,
                  CubeOps &ops, const RuntimeTiling &tiling) noexcept
{
    CubeStageArgs args{&item,
                       &workspace,
                       &sync,
                       &ops,
                       tiling.key,
                       tiling.epsilon,
                       tiling.lowerBound,
                       tiling.scale,
                       workgroupId};

    // C2 contains the eight independent score MMADs. C4, C5, and C7 are
    // separate physical stages because each consumes a prior-stage result.
    arch35::RunC2(args);
    arch35::RunC4(args);
    arch35::RunC5(args);
    arch35::RunC7(args);
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
            RunAivBranch(item, workgroupId, aivId, workspace, sync, vectorOps,
                         tiling);
        }
        return;
    }
    if (role == CoreRole::Aic) {
        RunAicBranch(item, workgroupId, workspace, sync, cubeOps, tiling);
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
    if (tiling.aicWorkgroupCount == 0 || tiling.totalChunks == 0 ||
        tiling.headCount == 0 || workgroupId >= tiling.aicWorkgroupCount) {
        return;
    }
    if (resolveChunk == nullptr) {
        resolveChunk = ResolveDenseChunk;
    }

    const CorePlan plan = BuildCorePlan(tiling, workgroupId);
    OwnerTicketState ownerTickets{};
    for (std::uint64_t ordinal = plan.begin; ordinal < plan.end; ++ordinal) {
        std::uint32_t chunkOrdinal = 0;
        std::uint32_t groupOrdinal = 0;
        DecodeChunkHeadGroupOrdinal(plan, ordinal, chunkOrdinal, groupOrdinal);
        const ChunkTask chunk = resolveChunk(chunkOrdinal);
        // Host tiling must omit empty sequences and reject any descriptor
        // outside [1,64]. The symbolic kernel also refuses to enter a stage,
        // preventing V6 from evaluating validRows - 1 for an empty chunk.
        if (chunk.validRows == 0 || chunk.validRows > kChunkRows) {
            continue;
        }

        if (plan.mode == PartitionMode::ChunkOnly) {
            // Chunk-first: a workgroup owns the complete chunk, then advances
            // all head groups. Adjacent group pairs form one eight-slot wave.
            for (groupOrdinal = 0; groupOrdinal < plan.headGroupCount;
                 ++groupOrdinal) {
                const WorkItem item = BuildWorkItem(
                    plan, ordinal, chunk, groupOrdinal, tiling.headCount,
                    ownerTickets);
                Dispatch(item, role, workgroupId, aivId, workspace, sync,
                         vectorOps, cubeOps, tiling);
            }
            continue;
        }

        // Head grouping is only the fallback when chunks alone cannot fill the
        // AIC workgroups. The flattened ordinal never changes head ownership.
        const WorkItem item = BuildWorkItem(
            plan, ordinal, chunk, groupOrdinal, tiling.headCount,
            ownerTickets);
        Dispatch(item, role, workgroupId, aivId, workspace, sync, vectorOps,
                 cubeOps, tiling);
    }
}

} // namespace kda_prepare_pseudocode
