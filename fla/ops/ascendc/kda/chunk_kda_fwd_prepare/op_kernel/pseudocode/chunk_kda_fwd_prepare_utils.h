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

template <bool UseExp2, typename Vf, typename Value>
inline auto EvaluatePow2(Vf &vf, const Value &base2Exponent,
                         float base2Minimum, float base2Maximum)
{
    if constexpr (UseExp2) {
        return vf.Exp2Clamped(base2Exponent, base2Minimum, base2Maximum);
    }
    // gk and every relative gate remain in log2 units. The alternate public
    // mode clamps that same x first, then evaluates exp(x * ln(2)) in FP32.
    const auto clamped =
        vf.Clamp(base2Exponent, base2Minimum, base2Maximum);
    return vf.Exp(vf.Mul(clamped, vf.Ln2()));
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

constexpr std::uint32_t EffectiveQkHeadCount(
    std::uint32_t valueHeadCount, std::uint32_t qkHeadCount) noexcept
{
    return qkHeadCount == 0U ? valueHeadCount : qkHeadCount;
}

constexpr bool IsValidHeadMapping(std::uint32_t valueHeadCount,
                                  std::uint32_t qkHeadCount) noexcept
{
    const std::uint32_t effectiveQk =
        EffectiveQkHeadCount(valueHeadCount, qkHeadCount);
    return valueHeadCount != 0U && effectiveQk != 0U &&
           effectiveQk <= valueHeadCount &&
           valueHeadCount % effectiveQk == 0U;
}

constexpr std::uint32_t QkHeadForValueHead(
    std::uint32_t valueHeadId, std::uint32_t valueHeadCount,
    std::uint32_t qkHeadCount) noexcept
{
    if (!IsValidHeadMapping(valueHeadCount, qkHeadCount) ||
        valueHeadId >= valueHeadCount) {
        return kAllGroupLocalHeads;
    }
    const std::uint32_t effectiveQk =
        EffectiveQkHeadCount(valueHeadCount, qkHeadCount);
    return valueHeadId / (valueHeadCount / effectiveQk);
}

constexpr std::uint32_t ValueHeadsPerQkHead(
    std::uint32_t valueHeadCount, std::uint32_t qkHeadCount) noexcept
{
    return IsValidHeadMapping(valueHeadCount, qkHeadCount)
               ? valueHeadCount /
                     EffectiveQkHeadCount(valueHeadCount, qkHeadCount)
               : 0U;
}

constexpr std::uint32_t QkOwnerValueHead(
    std::uint32_t qkHeadId, std::uint32_t valueHeadCount,
    std::uint32_t qkHeadCount) noexcept
{
    const std::uint32_t ratio =
        ValueHeadsPerQkHead(valueHeadCount, qkHeadCount);
    return ratio == 0U ? kAllGroupLocalHeads : qkHeadId * ratio;
}

constexpr std::uint64_t QkCacheLogicalKey(const ChunkTask &chunk,
                                          std::uint32_t qkHeadId) noexcept
{
    return (static_cast<std::uint64_t>(chunk.globalChunk) << 32U) |
           qkHeadId;
}

constexpr void BindQkCacheTask(HeadTask &task, const ChunkTask &chunk,
                               std::uint32_t valueHeadCount,
                               std::uint32_t qkHeadCount,
                               Architecture architecture,
                               OwnerTicketState &tickets) noexcept
{
    const std::uint32_t ratio =
        ValueHeadsPerQkHead(valueHeadCount, qkHeadCount);
    const std::uint32_t owner = QkOwnerValueHead(
        task.qkHeadId, valueHeadCount, qkHeadCount);
    task.qkOwner = task.headId == owner;
    task.qkLastConsumer =
        ratio != 0U && task.headId + 1U == owner + ratio;
    task.qkCacheSlot = owner % WorkspaceSlotCountFor(architecture);
    const std::uint64_t logicalKey =
        QkCacheLogicalKey(chunk, task.qkHeadId);
    if (task.qkOwner) {
        task.qkCacheGeneration =
            tickets.qkCacheNext[task.qkCacheSlot]++;
        tickets.qkCacheCurrentGeneration[task.qkCacheSlot] =
            task.qkCacheGeneration;
        tickets.qkCacheCurrentKey[task.qkCacheSlot] = logicalKey;
        tickets.qkCacheValid[task.qkCacheSlot] = true;
        return;
    }
    if (tickets.qkCacheValid[task.qkCacheSlot] &&
        tickets.qkCacheCurrentKey[task.qkCacheSlot] == logicalKey) {
        task.qkCacheGeneration =
            tickets.qkCacheCurrentGeneration[task.qkCacheSlot];
        return;
    }
    task.qkCacheGeneration = std::numeric_limits<std::uint64_t>::max();
}

// Head fallback partitions only at complete HK cohorts. When one cohort fits
// in four lanes, pack as many complete cohorts as possible; a larger cohort
// stays on one workgroup and is emitted as several consecutive four-head
// groups. Thus no second workgroup can repeat the same HK normalization.
constexpr std::uint32_t QkHeadsPerPartition(
    std::uint32_t valueHeadCount, std::uint32_t qkHeadCount) noexcept
{
    const std::uint32_t ratio =
        ValueHeadsPerQkHead(valueHeadCount, qkHeadCount);
    return ratio == 0U || ratio > kHeadsPerGroup
               ? (ratio == 0U ? 0U : 1U)
               : std::max(1U, kHeadsPerGroup / ratio);
}

constexpr std::uint32_t HeadPartitionCount(
    std::uint32_t valueHeadCount, std::uint32_t qkHeadCount) noexcept
{
    const std::uint32_t qkPerPartition =
        QkHeadsPerPartition(valueHeadCount, qkHeadCount);
    return qkPerPartition == 0U
               ? 0U
               : static_cast<std::uint32_t>(CeilDiv(
                     EffectiveQkHeadCount(valueHeadCount, qkHeadCount),
                     qkPerPartition));
}

constexpr std::uint32_t HeadGroupsPerPartition(
    std::uint32_t valueHeadCount, std::uint32_t qkHeadCount) noexcept
{
    const std::uint32_t ratio =
        ValueHeadsPerQkHead(valueHeadCount, qkHeadCount);
    const std::uint32_t qkPerPartition =
        QkHeadsPerPartition(valueHeadCount, qkHeadCount);
    return ratio == 0U || qkPerPartition == 0U
               ? 0U
               : static_cast<std::uint32_t>(
                     CeilDiv(ratio * qkPerPartition, kHeadsPerGroup));
}

constexpr std::uint32_t EmittedHeadGroupCount(
    std::uint32_t valueHeadCount, std::uint32_t qkHeadCount) noexcept
{
    return HeadPartitionCount(valueHeadCount, qkHeadCount) *
           HeadGroupsPerPartition(valueHeadCount, qkHeadCount);
}

constexpr std::uint32_t PartitionValueHeadBegin(
    std::uint32_t partitionOrdinal, std::uint32_t valueHeadCount,
    std::uint32_t qkHeadCount) noexcept
{
    return partitionOrdinal *
           QkHeadsPerPartition(valueHeadCount, qkHeadCount) *
           ValueHeadsPerQkHead(valueHeadCount, qkHeadCount);
}

constexpr std::uint32_t PartitionValueHeadEnd(
    std::uint32_t partitionOrdinal, std::uint32_t valueHeadCount,
    std::uint32_t qkHeadCount) noexcept
{
    return std::min(
        valueHeadCount,
        PartitionValueHeadBegin(partitionOrdinal + 1U, valueHeadCount,
                                qkHeadCount));
}

static_assert(ValueHeadsPerQkHead(8U, 2U) == 4U &&
                  QkHeadsPerPartition(8U, 2U) == 1U &&
                  HeadPartitionCount(8U, 2U) == 2U &&
                  HeadGroupsPerPartition(8U, 2U) == 1U &&
                  EmittedHeadGroupCount(8U, 2U) == 2U &&
                  PartitionValueHeadBegin(1U, 8U, 2U) == 4U &&
                  PartitionValueHeadEnd(1U, 8U, 2U) == 8U,
              "R=4 fallback must keep each complete HK cohort together");
static_assert(ValueHeadsPerQkHead(12U, 4U) == 3U &&
                  QkHeadsPerPartition(12U, 4U) == 1U &&
                  HeadPartitionCount(12U, 4U) == 4U &&
                  HeadGroupsPerPartition(12U, 4U) == 1U &&
                  EmittedHeadGroupCount(12U, 4U) == 4U &&
                  PartitionValueHeadBegin(1U, 12U, 4U) == 3U &&
                  PartitionValueHeadEnd(1U, 12U, 4U) == 6U,
              "non-power-of-two cohorts must not be split at four-head boundaries");
static_assert(ValueHeadsPerQkHead(16U, 2U) == 8U &&
                  QkHeadsPerPartition(16U, 2U) == 1U &&
                  HeadPartitionCount(16U, 2U) == 2U &&
                  HeadGroupsPerPartition(16U, 2U) == 2U &&
                  EmittedHeadGroupCount(16U, 2U) == 4U,
              "R>4 must expand one cohort into consecutive local groups");

constexpr CorePlan BuildCorePlan(const RuntimeTiling &tiling,
                                 std::uint32_t workgroupId) noexcept
{
    CorePlan plan{};
    plan.workgroupId = workgroupId;
    plan.workgroupCount = tiling.aicWorkgroupCount;
    plan.totalChunks = tiling.totalChunks;
    plan.architecture = tiling.architecture;
    plan.headPartitionCount =
        HeadPartitionCount(tiling.headCount, tiling.qkHeadCount);
    plan.headGroupCount =
        EmittedHeadGroupCount(tiling.headCount, tiling.qkHeadCount);
    plan.mode = tiling.totalChunks >= tiling.aicWorkgroupCount
                    ? PartitionMode::ChunkOnly
                    : PartitionMode::ChunkHeadGroup;
    plan.primaryWorkItemCount =
        plan.mode == PartitionMode::ChunkOnly
            ? tiling.totalChunks
            : static_cast<std::uint64_t>(tiling.totalChunks) *
                  plan.headPartitionCount;
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
    Architecture architecture, std::uint32_t groupLocalHead) noexcept
{
    return architecture == Architecture::Arch22
               ? groupLocalHead % kAivPerWorkgroup
               : groupLocalHead / kHeadsPerAiv;
}

constexpr std::uint32_t AivLocalSlotForGroupLocalHead(
    Architecture architecture, std::uint32_t groupLocalHead) noexcept
{
    return architecture == Architecture::Arch22
               ? groupLocalHead / kAivPerWorkgroup
               : groupLocalHead % kHeadsPerAiv;
}

constexpr std::uint32_t AivForGroupLocalHead(
    std::uint32_t groupLocalHead) noexcept
{
    return AivForGroupLocalHead(Architecture::Arch35, groupLocalHead);
}

constexpr std::uint32_t AivLocalSlotForGroupLocalHead(
    std::uint32_t groupLocalHead) noexcept
{
    return AivLocalSlotForGroupLocalHead(Architecture::Arch35,
                                        groupLocalHead);
}

constexpr std::uint32_t WorkspaceSlotForHead(
    Architecture architecture, std::uint32_t headId) noexcept
{
    return headId % WorkspaceSlotCountFor(architecture);
}

constexpr std::uint32_t WorkspaceSlotForHead(std::uint32_t headId) noexcept
{
    return WorkspaceSlotForHead(Architecture::Arch35, headId);
}

constexpr std::uint32_t L0cBankForGroupLocalHead(
    Architecture architecture, std::uint32_t groupLocalHead) noexcept
{
    return architecture == Architecture::Arch22
               ? groupLocalHead % kAivPerWorkgroup
               : groupLocalHead;
}

constexpr std::uint32_t L0OperandBankForGroupLocalHead(
    Architecture architecture, std::uint32_t groupLocalHead) noexcept
{
    return architecture == Architecture::Arch22
               ? groupLocalHead % kAivPerWorkgroup
               : 0U;
}

constexpr void BindL0OperandGenerations(
    HeadGroup &group, Architecture architecture,
    OwnerTicketState &tickets) noexcept
{
    // Allocation order exactly follows the AIC source order: all active C2
    // bands by head, then C4, C5, and C7 by head. Arch35 has one shared
    // operand bank; Arch22 has two independently reusable physical lanes.
    for (HeadTask &head : group.heads) {
        if (!head.active) {
            continue;
        }
        if (group.chunk.validRows > 0U) {
            head.l0OperandGenerations[static_cast<std::size_t>(
                L0OperandUse::C2Band0)] =
                tickets.l0OperandNext[head.l0OperandBankId]++;
        }
        if (group.chunk.validRows > kScoreBlockRows) {
            head.l0OperandGenerations[static_cast<std::size_t>(
                L0OperandUse::C2Band1)] =
                tickets.l0OperandNext[head.l0OperandBankId]++;
        }
        if (group.chunk.validRows > 2U * kScoreBlockRows) {
            head.l0OperandGenerations[static_cast<std::size_t>(
                L0OperandUse::C2Band2)] =
                tickets.l0OperandNext[head.l0OperandBankId]++;
        }
        if (group.chunk.validRows > 3U * kScoreBlockRows) {
            head.l0OperandGenerations[static_cast<std::size_t>(
                L0OperandUse::C2Band3)] =
                tickets.l0OperandNext[head.l0OperandBankId]++;
        }
    }
    if (group.chunk.validRows > 32U) {
        for (HeadTask &head : group.heads) {
            if (head.active) {
                head.l0OperandGenerations[static_cast<std::size_t>(
                    L0OperandUse::C4)] =
                    tickets.l0OperandNext[head.l0OperandBankId]++;
            }
        }
        for (HeadTask &head : group.heads) {
            if (head.active) {
                head.l0OperandGenerations[static_cast<std::size_t>(
                    L0OperandUse::C5)] =
                    tickets.l0OperandNext[head.l0OperandBankId]++;
            }
        }
    }
    for (HeadTask &head : group.heads) {
        if (head.active) {
            head.l0OperandGenerations[static_cast<std::size_t>(
                L0OperandUse::C7)] =
                tickets.l0OperandNext[head.l0OperandBankId]++;
        }
    }
    (void)architecture;
}

constexpr HeadGroup BuildHeadGroup(const ChunkTask &chunk,
                                   std::uint32_t headGroupId,
                                   std::uint32_t headCount,
                                   Architecture architecture,
                                   OwnerTicketState &tickets,
                                   std::uint32_t qkHeadCount = 0U) noexcept
{
    HeadGroup group{};
    group.chunk = chunk;
    group.headGroupId = headGroupId;
    group.wavefront =
        headGroupId / HeadGroupsInFlightFor(architecture);
    const std::uint32_t headBegin = headGroupId * kHeadsPerGroup;
    group.activeHeads = headBegin < headCount
                             ? std::min(kHeadsPerGroup, headCount - headBegin)
                             : 0;
    if (architecture == Architecture::Arch22) {
        for (std::uint32_t pair = 0; pair < kArch22PairCount; ++pair) {
            if (PairHasActiveHead(group, pair)) {
                group.pairCollectiveGenerations[pair] =
                    tickets.collectiveNext[pair]++;
            }
        }
    }
    for (std::uint32_t local = 0; local < kHeadsPerGroup; ++local) {
        HeadTask &task = group.heads[local];
        task.headId = headBegin + local;
        task.qkHeadId = QkHeadForValueHead(task.headId, headCount,
                                           qkHeadCount);
        task.groupLocalHead = local;
        task.workspaceSlot = WorkspaceSlotForHead(architecture, task.headId);
        task.aivId = AivForGroupLocalHead(architecture, local);
        task.aivLocalSlot =
            AivLocalSlotForGroupLocalHead(architecture, local);
        task.localBankId = task.aivId * kHeadsPerAiv + task.aivLocalSlot;
        task.sharedArenaId = task.aivId;
        task.l1BankId = local;
        task.l0cBankId = L0cBankForGroupLocalHead(architecture, local);
        task.l0OperandBankId =
            L0OperandBankForGroupLocalHead(architecture, local);
        task.l0cPairWave = architecture == Architecture::Arch22
                               ? local / kAivPerWorkgroup
                               : 0;
        task.active = local < group.activeHeads;
        if (task.active) {
            BindQkCacheTask(task, chunk, headCount, qkHeadCount,
                            architecture, tickets);
            task.workspaceGeneration = tickets.workspaceNext[task.workspaceSlot]++;
            task.localGeneration = tickets.localNext[task.localBankId]++;
            task.l1Generation = tickets.l1Next[task.l1BankId]++;
            if (architecture == Architecture::Arch35) {
                task.l0cGeneration = tickets.l0cNext[task.l0cBankId]++;
                for (std::size_t use = 0; use < kL0cStageUseCount; ++use) {
                    task.l0cStageGenerations[use] = task.l0cGeneration;
                }
            }
        }
    }

    BindL0OperandGenerations(group, architecture, tickets);

    if (architecture == Architecture::Arch22) {
        // Arch22 owns one shared 40 KiB arena per AIV. V0 and V1 retain one
        // ticket across their same-head sequence; V3 stays entirely private,
        // and V6 acquires the only later shared ticket.
        for (std::size_t use = 0; use < kSharedArenaUseCount; ++use) {
            for (std::uint32_t localSlot = 0; localSlot < kHeadsPerAiv;
                 ++localSlot) {
                for (std::uint32_t aiv = 0; aiv < kAivPerWorkgroup; ++aiv) {
                    const std::uint32_t local =
                        localSlot * kAivPerWorkgroup + aiv;
                    HeadTask &task = group.heads[local];
                    if (task.active) {
                        task.sharedGenerations[use] =
                            tickets.sharedNext[task.sharedArenaId]++;
                    }
                }
            }
        }

        // Arch22 has two physical 64 KiB L0C lanes. Every Cube stage releases
        // its lane before the next pair/stage reuses it, so generations are
        // stage-use tickets rather than an A5 transaction-long ticket.
        for (std::size_t use = 0; use < kL0cStageUseCount; ++use) {
            for (std::uint32_t pair = 0; pair < kArch22PairCount; ++pair) {
                for (std::uint32_t lane = 0; lane < kAivPerWorkgroup; ++lane) {
                    const std::uint32_t local =
                        pair * kAivPerWorkgroup + lane;
                    HeadTask &task = group.heads[local];
                    if (task.active) {
                        task.l0cStageGenerations[use] =
                            tickets.l0cNext[task.l0cBankId]++;
                    }
                }
            }
        }
    }
    return group;
}

constexpr HeadGroup BuildHeadGroupRange(
    const ChunkTask &chunk, std::uint32_t headGroupId,
    std::uint32_t headBegin, std::uint32_t headEnd,
    std::uint32_t valueHeadCount, Architecture architecture,
    OwnerTicketState &tickets, std::uint32_t qkHeadCount = 0U) noexcept
{
    HeadGroup group{};
    group.chunk = chunk;
    group.headGroupId = headGroupId;
    group.wavefront =
        headGroupId / HeadGroupsInFlightFor(architecture);
    group.activeHeads =
        headBegin < headEnd
            ? std::min(kHeadsPerGroup, headEnd - headBegin)
            : 0U;

    // The zero-head helper above allocates no owner tickets. Populate the
    // explicit range using the same ownership rules as BuildHeadGroup while
    // allowing cohort boundaries that are not multiples of four.
    if (architecture == Architecture::Arch22) {
        for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
            if (PairHasActiveHead(group, pair)) {
                group.pairCollectiveGenerations[pair] =
                    tickets.collectiveNext[pair]++;
            }
        }
    }
    for (std::uint32_t local = 0U; local < kHeadsPerGroup; ++local) {
        HeadTask &task = group.heads[local];
        task.headId = headBegin + local;
        task.qkHeadId = QkHeadForValueHead(task.headId, valueHeadCount,
                                           qkHeadCount);
        task.groupLocalHead = local;
        task.workspaceSlot = WorkspaceSlotForHead(architecture, task.headId);
        task.aivId = AivForGroupLocalHead(architecture, local);
        task.aivLocalSlot =
            AivLocalSlotForGroupLocalHead(architecture, local);
        task.localBankId = task.aivId * kHeadsPerAiv + task.aivLocalSlot;
        task.sharedArenaId = task.aivId;
        task.l1BankId = local;
        task.l0cBankId = L0cBankForGroupLocalHead(architecture, local);
        task.l0OperandBankId =
            L0OperandBankForGroupLocalHead(architecture, local);
        task.l0cPairWave = architecture == Architecture::Arch22
                               ? local / kAivPerWorkgroup
                               : 0U;
        task.active = local < group.activeHeads;
        if (task.active) {
            BindQkCacheTask(task, chunk, valueHeadCount, qkHeadCount,
                            architecture, tickets);
            task.workspaceGeneration =
                tickets.workspaceNext[task.workspaceSlot]++;
            task.localGeneration = tickets.localNext[task.localBankId]++;
            task.l1Generation = tickets.l1Next[task.l1BankId]++;
            if (architecture == Architecture::Arch35) {
                task.l0cGeneration = tickets.l0cNext[task.l0cBankId]++;
                for (std::size_t use = 0U; use < kL0cStageUseCount; ++use) {
                    task.l0cStageGenerations[use] = task.l0cGeneration;
                }
            }
        }
    }
    BindL0OperandGenerations(group, architecture, tickets);
    if (architecture == Architecture::Arch22) {
        for (std::size_t use = 0U; use < kSharedArenaUseCount; ++use) {
            for (std::uint32_t localSlot = 0U;
                 localSlot < kHeadsPerAiv; ++localSlot) {
                for (std::uint32_t aiv = 0U; aiv < kAivPerWorkgroup; ++aiv) {
                    const std::uint32_t local =
                        localSlot * kAivPerWorkgroup + aiv;
                    HeadTask &task = group.heads[local];
                    if (task.active) {
                        task.sharedGenerations[use] =
                            tickets.sharedNext[task.sharedArenaId]++;
                    }
                }
            }
        }
        for (std::size_t use = 0U; use < kL0cStageUseCount; ++use) {
            for (std::uint32_t pair = 0U; pair < kArch22PairCount; ++pair) {
                for (std::uint32_t lane = 0U; lane < kAivPerWorkgroup;
                     ++lane) {
                    const std::uint32_t local =
                        pair * kAivPerWorkgroup + lane;
                    HeadTask &task = group.heads[local];
                    if (task.active) {
                        task.l0cStageGenerations[use] =
                            tickets.l0cNext[task.l0cBankId]++;
                    }
                }
            }
        }
    }
    return group;
}

constexpr WorkItem BuildWorkItemRange(
    const CorePlan &plan, std::uint64_t ordinal, const ChunkTask &chunk,
    std::uint32_t headGroupOrdinal, std::uint32_t headBegin,
    std::uint32_t headEnd, std::uint32_t valueHeadCount,
    OwnerTicketState &tickets, std::uint32_t qkHeadCount = 0U) noexcept
{
    WorkItem item{};
    item.ordinal = ordinal;
    item.chunkOrdinal = chunk.globalChunk;
    item.headGroupOrdinal = headGroupOrdinal;
    item.group = BuildHeadGroupRange(
        chunk, headGroupOrdinal, headBegin, headEnd, valueHeadCount,
        plan.architecture, tickets, qkHeadCount);
    return item;
}

constexpr WorkItem BuildWorkItem(const CorePlan &plan, std::uint64_t ordinal,
                                 const ChunkTask &chunk,
                                 std::uint32_t headGroupOrdinal,
                                 std::uint32_t headCount,
                                 OwnerTicketState &tickets,
                                 std::uint32_t qkHeadCount = 0U) noexcept
{
    WorkItem item{};
    item.ordinal = ordinal;
    item.chunkOrdinal = chunk.globalChunk;
    item.headGroupOrdinal = headGroupOrdinal;
    item.group = BuildHeadGroup(chunk, headGroupOrdinal, headCount,
                                plan.architecture, tickets, qkHeadCount);
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

constexpr void DecodeChunkHeadPartitionOrdinal(
    const CorePlan &plan, std::uint64_t ordinal,
    std::uint32_t &chunkOrdinal,
    std::uint32_t &headPartitionOrdinal) noexcept
{
    if (plan.mode == PartitionMode::ChunkOnly) {
        chunkOrdinal = static_cast<std::uint32_t>(ordinal);
        headPartitionOrdinal = 0U;
        return;
    }
    chunkOrdinal = plan.headPartitionCount == 0U
                       ? 0U
                       : static_cast<std::uint32_t>(
                             ordinal / plan.headPartitionCount);
    headPartitionOrdinal = plan.headPartitionCount == 0U
                               ? 0U
                               : static_cast<std::uint32_t>(
                                     ordinal % plan.headPartitionCount);
}

constexpr bool HasEightStageContract() noexcept
{
    return static_cast<std::uint32_t>(Stage::C7) + 1 == 8;
}

constexpr bool CheckPartitionModeBoundary() noexcept
{
    RuntimeTiling tiling{};
    tiling.headCount = 5U;
    tiling.aicWorkgroupCount = 3U;
    tiling.totalChunks = 3U;
    if (BuildCorePlan(tiling, 0U).mode != PartitionMode::ChunkOnly) {
        return false;
    }
    tiling.totalChunks = 2U;
    return BuildCorePlan(tiling, 0U).mode ==
           PartitionMode::ChunkHeadGroup;
}

constexpr bool CheckTicketedGroup(const HeadGroup &group,
                                  Architecture architecture,
                                  OwnerTicketState &expected) noexcept
{
    for (const HeadTask &head : group.heads) {
        if (!head.active) {
            continue;
        }
        if (head.aivId !=
                AivForGroupLocalHead(architecture, head.groupLocalHead) ||
            head.aivLocalSlot != AivLocalSlotForGroupLocalHead(
                                     architecture, head.groupLocalHead) ||
            head.l0cBankId != L0cBankForGroupLocalHead(
                                     architecture, head.groupLocalHead) ||
            head.l0cPairWave !=
                (architecture == Architecture::Arch22
                     ? head.groupLocalHead / kAivPerWorkgroup
                     : 0) ||
            head.workspaceGeneration !=
                expected.workspaceNext[head.workspaceSlot]++ ||
            head.localGeneration != expected.localNext[head.localBankId]++ ||
            head.l1Generation != expected.l1Next[head.l1BankId]++) {
            return false;
        }
        if (architecture == Architecture::Arch35 &&
            head.l0cGeneration != expected.l0cNext[head.l0cBankId]++) {
            return false;
        }
    }

    if (architecture == Architecture::Arch22) {
        for (std::uint32_t pair = 0; pair < kArch22PairCount; ++pair) {
            if (PairHasActiveHead(group, pair) &&
                PairCollectiveGenerationFor(group, pair) !=
                    expected.collectiveNext[pair]++) {
                return false;
            }
        }
        for (std::size_t use = 0; use < kSharedArenaUseCount; ++use) {
            for (std::uint32_t localSlot = 0; localSlot < kHeadsPerAiv;
                 ++localSlot) {
                for (std::uint32_t aiv = 0; aiv < kAivPerWorkgroup; ++aiv) {
                    const std::uint32_t local =
                        localSlot * kAivPerWorkgroup + aiv;
                    const HeadTask &head = group.heads[local];
                    if (head.active &&
                        (head.sharedArenaId != aiv ||
                         head.sharedGenerations[use] !=
                             expected.sharedNext[aiv]++)) {
                        return false;
                    }
                }
            }
        }
        for (std::size_t use = 0; use < kL0cStageUseCount; ++use) {
            for (std::uint32_t pair = 0; pair < kArch22PairCount; ++pair) {
                for (std::uint32_t lane = 0; lane < kAivPerWorkgroup; ++lane) {
                    const std::uint32_t local =
                        pair * kAivPerWorkgroup + lane;
                    const HeadTask &head = group.heads[local];
                    if (head.active &&
                        head.l0cStageGenerations[use] !=
                            expected.l0cNext[lane]++) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

constexpr bool CheckOwnerTicketsForPlan(std::uint32_t totalChunks,
                                         std::uint32_t headCount,
                                         std::uint32_t workgroupCount,
                                         std::uint32_t workgroupId,
                                         Architecture architecture) noexcept
{
    RuntimeTiling tiling{};
    tiling.totalChunks = totalChunks;
    tiling.headCount = headCount;
    tiling.aicWorkgroupCount = workgroupCount;
    tiling.architecture = architecture;
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
                if (!CheckTicketedGroup(item.group, architecture, expected)) {
                    return false;
                }
            }
        } else {
            const WorkItem item = BuildWorkItem(
                plan, ordinal, chunk, groupOrdinal, headCount, actual);
            if (!CheckTicketedGroup(item.group, architecture, expected)) {
                return false;
            }
        }
    }
    return true;
}

constexpr bool CheckOwnerTicketMatrix() noexcept
{
    for (std::uint32_t heads = 1; heads <= 17; ++heads) {
        for (std::uint32_t architectureValue = 0; architectureValue < 2;
             ++architectureValue) {
            const Architecture architecture =
                architectureValue == 0 ? Architecture::Arch22
                                       : Architecture::Arch35;
            for (std::uint32_t workgroup = 0; workgroup < 3; ++workgroup) {
                if (!CheckOwnerTicketsForPlan(7, heads, 3, workgroup,
                                              architecture)) {
                    return false;
                }
            }
            for (std::uint32_t workgroup = 0; workgroup < 4; ++workgroup) {
                if (!CheckOwnerTicketsForPlan(2, heads, 4, workgroup,
                                              architecture)) {
                    return false;
                }
            }
        }
    }
    return true;
}

constexpr bool CheckArch22PairTicketShapeForHeads(
    std::uint32_t heads) noexcept
{
    OwnerTicketState actual{};
    std::array<std::uint64_t, kArch22PairCount> expectedCurrent{};
    const std::uint32_t groups = static_cast<std::uint32_t>(
        CeilDiv(heads, kHeadsPerGroup));
    // Three consecutive chunk transactions exercise current/next SlotFree
    // generations instead of checking only the initially seeded credit.
    for (std::uint32_t chunkId = 0; chunkId < 3U; ++chunkId) {
        const ChunkTask chunk{0, chunkId, chunkId, kChunkRows};
        for (std::uint32_t groupId = 0; groupId < groups; ++groupId) {
            const HeadGroup group = BuildHeadGroup(
                chunk, groupId, heads, Architecture::Arch22, actual);
            for (std::uint32_t pair = 0; pair < kArch22PairCount; ++pair) {
                const std::uint32_t pairBegin = pair * kAivPerWorkgroup;
                if (!PairHasActiveHead(group, pair)) {
                    continue;
                }
                const std::uint64_t generation =
                    PairCollectiveGenerationFor(group, pair);
                if (generation != expectedCurrent[pair]) {
                    return false;
                }

                const std::uint32_t activeInPair =
                    std::min<std::uint32_t>(kAivPerWorkgroup,
                                            group.activeHeads - pairBegin);
                const std::uint32_t dummyAivCount =
                    kAivPerWorkgroup - activeInPair;
                // The runtime trace test verifies actual stage helper calls.
                // This constexpr layer only proves the partial-pair shape and
                // its owner-ticket progression for every supported head tail.
                if (activeInPair + dummyAivCount != kAivPerWorkgroup ||
                    (dummyAivCount != 0U && dummyAivCount != 1U)) {
                    return false;
                }

                // C7 publishes SlotFree(generation+1); the next V0 pair
                // invocation must wait exactly that ticket.
                expectedCurrent[pair] = generation + 1U;
            }
        }
    }
    for (std::uint32_t pair = 0; pair < kArch22PairCount; ++pair) {
        if (actual.collectiveNext[pair] != expectedCurrent[pair]) {
            return false;
        }
    }
    return true;
}

static_assert(HasEightStageContract(), "Prepare must retain eight physical stages");
static_assert(CheckPartitionModeBoundary(),
              "head grouping is allowed only when chunks cannot fill AIC workgroups");
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
static_assert(AivForGroupLocalHead(Architecture::Arch35, 0) == 0 &&
                  AivForGroupLocalHead(Architecture::Arch35, 1) == 0 &&
                  AivForGroupLocalHead(Architecture::Arch35, 2) == 1 &&
                  AivForGroupLocalHead(Architecture::Arch35, 3) == 1,
              "AIV ownership must be contiguous: {0,1} and {2,3}");
static_assert(AivLocalSlotForGroupLocalHead(Architecture::Arch35, 0) == 0 &&
                  AivLocalSlotForGroupLocalHead(Architecture::Arch35, 1) == 1 &&
                  AivLocalSlotForGroupLocalHead(Architecture::Arch35, 2) == 0 &&
                  AivLocalSlotForGroupLocalHead(Architecture::Arch35, 3) == 1,
              "each AIV must use local slots zero and one");
static_assert(AivForGroupLocalHead(Architecture::Arch22, 0) == 0 &&
                  AivForGroupLocalHead(Architecture::Arch22, 1) == 1 &&
                  AivForGroupLocalHead(Architecture::Arch22, 2) == 0 &&
                  AivForGroupLocalHead(Architecture::Arch22, 3) == 1 &&
                  AivLocalSlotForGroupLocalHead(Architecture::Arch22, 0) == 0 &&
                  AivLocalSlotForGroupLocalHead(Architecture::Arch22, 1) == 0 &&
                  AivLocalSlotForGroupLocalHead(Architecture::Arch22, 2) == 1 &&
                  AivLocalSlotForGroupLocalHead(Architecture::Arch22, 3) == 1,
              "Arch22 AIV ownership must interleave {0,2} and {1,3}");
static_assert(L0cBankForGroupLocalHead(Architecture::Arch22, 0) == 0 &&
                  L0cBankForGroupLocalHead(Architecture::Arch22, 1) == 1 &&
                  L0cBankForGroupLocalHead(Architecture::Arch22, 2) == 0 &&
                  L0cBankForGroupLocalHead(Architecture::Arch22, 3) == 1,
              "Arch22 must reuse two L0C lanes in pair waves");
static_assert(WorkspaceSlotForHead(Architecture::Arch22, 0) == 0 &&
                  WorkspaceSlotForHead(Architecture::Arch22, 3) == 3 &&
                  WorkspaceSlotForHead(Architecture::Arch22, 4) == 0 &&
                  WorkspaceSlotForHead(Architecture::Arch35, 7) == 7 &&
                  WorkspaceSlotForHead(Architecture::Arch35, 8) == 0,
              "workspace slots must follow the architecture wave depth");
static_assert(CheckOwnerTicketMatrix(),
              "owner tickets must be contiguous for H=1..17 in both partition modes");
#define CHECK_ARCH22_PAIR_TICKET_SHAPE(HEADS)                              \
    static_assert(CheckArch22PairTicketShapeForHeads(HEADS),               \
                  "Arch22 partial-pair shape and SlotFree tickets must close")
CHECK_ARCH22_PAIR_TICKET_SHAPE(1U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(2U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(3U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(4U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(5U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(6U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(7U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(8U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(9U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(10U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(11U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(12U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(13U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(14U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(15U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(16U);
CHECK_ARCH22_PAIR_TICKET_SHAPE(17U);
#undef CHECK_ARCH22_PAIR_TICKET_SHAPE

} // namespace kda_prepare_pseudocode

#endif // PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_UTILS_H
