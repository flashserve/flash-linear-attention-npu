/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_UTILS_H
#define PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_UTILS_H

#include <algorithm>
#include <cstdint>
#include <limits>

#include "chunk_kda_fwd_prepare_struct.h"

namespace kda_prepare_pseudocode {

constexpr std::uint64_t CeilDiv(std::uint64_t value,
                                std::uint64_t divisor) noexcept
{
    return divisor == 0
               ? 0
               : value / divisor + static_cast<std::uint64_t>(value % divisor != 0);
}

constexpr std::uint64_t BalancedPoint(std::uint64_t total,
                                      std::uint32_t index,
                                      std::uint32_t count) noexcept
{
    if (count == 0) {
        return 0;
    }
    const std::uint64_t quotient = total / count;
    const std::uint64_t remainder = total % count;
    return quotient * index +
           (remainder * static_cast<std::uint64_t>(index)) / count;
}

constexpr std::uint64_t BalancedBegin(std::uint64_t total,
                                      std::uint32_t rank,
                                      std::uint32_t count) noexcept
{
    return BalancedPoint(total, rank, count);
}

constexpr std::uint64_t BalancedEnd(std::uint64_t total, std::uint32_t rank,
                                    std::uint32_t count) noexcept
{
    return BalancedPoint(total, rank + 1, count);
}

constexpr CorePlan BuildCorePlan(const RuntimeTiling &tiling,
                                 std::uint32_t workgroupId) noexcept
{
    CorePlan plan{};
    plan.workgroupId = workgroupId;
    plan.workgroupCount = tiling.aicWorkgroupCount;
    plan.totalChunks = tiling.totalChunks;
    plan.headGroupCount = static_cast<std::uint32_t>(
        CeilDiv(tiling.headCount, kHeadsPerGroup));
    plan.mode = tiling.totalChunks >= tiling.aicWorkgroupCount
                    ? PartitionMode::ChunkOnly
                    : PartitionMode::ChunkHeadGroup;
    plan.primaryWorkItemCount =
        plan.mode == PartitionMode::ChunkOnly
            ? tiling.totalChunks
            : static_cast<std::uint64_t>(tiling.totalChunks) *
                  plan.headGroupCount;
    if (workgroupId >= tiling.aicWorkgroupCount) {
        return plan;
    }
    plan.begin = BalancedBegin(plan.primaryWorkItemCount, workgroupId,
                               tiling.aicWorkgroupCount);
    plan.end = BalancedEnd(plan.primaryWorkItemCount, workgroupId,
                           tiling.aicWorkgroupCount);
    return plan;
}

constexpr std::uint32_t AivForGroupLocalHead(
    std::uint32_t groupLocalHead) noexcept
{
    return groupLocalHead / kHeadsPerAiv;
}

constexpr std::uint32_t AivLocalSlotForGroupLocalHead(
    std::uint32_t groupLocalHead) noexcept
{
    return groupLocalHead % kHeadsPerAiv;
}

constexpr std::uint32_t WorkspaceSlotForHead(std::uint32_t headId) noexcept
{
    return headId % kWorkspaceSlotCount;
}

constexpr HeadGroup BuildHeadGroup(const ChunkTask &chunk,
                                   std::uint32_t headGroupId,
                                   std::uint32_t headCount,
                                   OwnerTicketState &tickets) noexcept
{
    HeadGroup group{};
    group.chunk = chunk;
    group.headGroupId = headGroupId;
    group.wavefront = headGroupId / kHeadGroupsInFlight;
    const std::uint32_t headBegin = headGroupId * kHeadsPerGroup;
    group.activeHeads = headBegin < headCount
                            ? std::min(kHeadsPerGroup, headCount - headBegin)
                            : 0;
    for (std::uint32_t local = 0; local < kHeadsPerGroup; ++local) {
        HeadTask &task = group.heads[local];
        task.headId = headBegin + local;
        task.groupLocalHead = local;
        task.workspaceSlot = WorkspaceSlotForHead(task.headId);
        task.aivId = AivForGroupLocalHead(local);
        task.aivLocalSlot = AivLocalSlotForGroupLocalHead(local);
        task.localBankId = task.aivId * kHeadsPerAiv + task.aivLocalSlot;
        task.l1BankId = local;
        task.l0cBankId = local;
        task.active = local < group.activeHeads;
        if (task.active) {
            task.workspaceGeneration = tickets.workspaceNext[task.workspaceSlot]++;
            task.localGeneration = tickets.localNext[task.localBankId]++;
            task.l1Generation = tickets.l1Next[task.l1BankId]++;
            task.l0cGeneration = tickets.l0cNext[task.l0cBankId]++;
        }
    }
    return group;
}

constexpr WorkItem BuildWorkItem(const CorePlan &plan, std::uint64_t ordinal,
                                 const ChunkTask &chunk,
                                 std::uint32_t headGroupOrdinal,
                                 std::uint32_t headCount,
                                 OwnerTicketState &tickets) noexcept
{
    WorkItem item{};
    item.ordinal = ordinal;
    item.chunkOrdinal = chunk.globalChunk;
    item.headGroupOrdinal = headGroupOrdinal;
    item.group = BuildHeadGroup(chunk, headGroupOrdinal, headCount, tickets);
    (void)plan;
    return item;
}

constexpr void DecodeChunkHeadGroupOrdinal(
    const CorePlan &plan, std::uint64_t ordinal, std::uint32_t &chunkOrdinal,
    std::uint32_t &headGroupOrdinal) noexcept
{
    if (plan.mode == PartitionMode::ChunkOnly) {
        chunkOrdinal = static_cast<std::uint32_t>(ordinal);
        headGroupOrdinal = 0;
        return;
    }
    chunkOrdinal = plan.headGroupCount == 0
                       ? 0
                       : static_cast<std::uint32_t>(ordinal /
                                                    plan.headGroupCount);
    headGroupOrdinal = plan.headGroupCount == 0
                           ? 0
                           : static_cast<std::uint32_t>(ordinal %
                                                        plan.headGroupCount);
}

constexpr bool HasEightStageContract() noexcept
{
    return static_cast<std::uint32_t>(Stage::C7) + 1 == 8;
}

constexpr bool CheckTicketedGroup(const HeadGroup &group,
                                  OwnerTicketState &expected) noexcept
{
    for (const HeadTask &head : group.heads) {
        if (!head.active) {
            continue;
        }
        if (head.l0cBankId != head.groupLocalHead ||
            head.workspaceGeneration !=
                expected.workspaceNext[head.workspaceSlot]++ ||
            head.localGeneration != expected.localNext[head.localBankId]++ ||
            head.l1Generation != expected.l1Next[head.l1BankId]++ ||
            head.l0cGeneration != expected.l0cNext[head.l0cBankId]++) {
            return false;
        }
    }
    return true;
}

constexpr bool CheckOwnerTicketsForPlan(std::uint32_t totalChunks,
                                        std::uint32_t headCount,
                                        std::uint32_t workgroupCount,
                                        std::uint32_t workgroupId) noexcept
{
    RuntimeTiling tiling{};
    tiling.totalChunks = totalChunks;
    tiling.headCount = headCount;
    tiling.aicWorkgroupCount = workgroupCount;
    const CorePlan plan = BuildCorePlan(tiling, workgroupId);
    OwnerTicketState actual{};
    OwnerTicketState expected{};
    for (std::uint64_t ordinal = plan.begin; ordinal < plan.end; ++ordinal) {
        std::uint32_t chunkOrdinal = 0;
        std::uint32_t groupOrdinal = 0;
        DecodeChunkHeadGroupOrdinal(plan, ordinal, chunkOrdinal, groupOrdinal);
        const ChunkTask chunk{0, chunkOrdinal, chunkOrdinal, kChunkRows};
        if (plan.mode == PartitionMode::ChunkOnly) {
            for (groupOrdinal = 0; groupOrdinal < plan.headGroupCount;
                 ++groupOrdinal) {
                const WorkItem item = BuildWorkItem(
                    plan, ordinal, chunk, groupOrdinal, headCount, actual);
                if (!CheckTicketedGroup(item.group, expected)) {
                    return false;
                }
            }
        } else {
            const WorkItem item = BuildWorkItem(
                plan, ordinal, chunk, groupOrdinal, headCount, actual);
            if (!CheckTicketedGroup(item.group, expected)) {
                return false;
            }
        }
    }
    return true;
}

constexpr bool CheckOwnerTicketMatrix() noexcept
{
    for (std::uint32_t heads = 1; heads <= 17; ++heads) {
        for (std::uint32_t workgroup = 0; workgroup < 3; ++workgroup) {
            if (!CheckOwnerTicketsForPlan(7, heads, 3, workgroup)) {
                return false;
            }
        }
        for (std::uint32_t workgroup = 0; workgroup < 4; ++workgroup) {
            if (!CheckOwnerTicketsForPlan(2, heads, 4, workgroup)) {
                return false;
            }
        }
    }
    return true;
}

static_assert(HasEightStageContract(), "Prepare must retain eight physical stages");
static_assert(
    CeilDiv(std::numeric_limits<std::uint64_t>::max(), 2) ==
        std::numeric_limits<std::uint64_t>::max() / 2 + 1,
    "CeilDiv must not overflow at uint64 max");
static_assert(
    BalancedBegin(std::numeric_limits<std::uint64_t>::max(), 0,
                  std::numeric_limits<std::uint32_t>::max()) == 0 &&
        BalancedEnd(std::numeric_limits<std::uint64_t>::max(),
                    std::numeric_limits<std::uint32_t>::max() - 1,
                    std::numeric_limits<std::uint32_t>::max()) ==
            std::numeric_limits<std::uint64_t>::max(),
    "balanced range endpoints must not overflow at integer limits");
static_assert(AivForGroupLocalHead(0) == 0 &&
                  AivForGroupLocalHead(1) == 0 &&
                  AivForGroupLocalHead(2) == 1 &&
                  AivForGroupLocalHead(3) == 1,
              "AIV ownership must be contiguous: {0,1} and {2,3}");
static_assert(AivLocalSlotForGroupLocalHead(0) == 0 &&
                  AivLocalSlotForGroupLocalHead(1) == 1 &&
                  AivLocalSlotForGroupLocalHead(2) == 0 &&
                  AivLocalSlotForGroupLocalHead(3) == 1,
              "each AIV must use local slots zero and one");
static_assert(CheckOwnerTicketMatrix(),
              "owner tickets must be contiguous for H=1..17 in both partition modes");

} // namespace kda_prepare_pseudocode

#endif // PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_UTILS_H
