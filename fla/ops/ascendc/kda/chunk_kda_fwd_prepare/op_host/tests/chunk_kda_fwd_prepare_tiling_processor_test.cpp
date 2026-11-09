/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "../chunk_kda_fwd_prepare_tiling_processor.h"
#include "../chunk_kda_fwd_prepare_output_mask.h"

#include <cstdint>
#include <limits>

namespace {

bool CheckOutputMaskContract()
{
    using namespace optiling;
    constexpr uint32_t expectedRequired =
        PREPARE_OUTPUT_MASK_GK | PREPARE_OUTPUT_MASK_AQK |
        PREPARE_OUTPUT_MASK_W | PREPARE_OUTPUT_MASK_U |
        PREPARE_OUTPUT_MASK_KG | PREPARE_OUTPUT_MASK_QG_SCALED;
    constexpr uint32_t expectedOptional =
        PREPARE_OUTPUT_MASK_AKK | PREPARE_OUTPUT_MASK_QG |
        PREPARE_OUTPUT_MASK_Q_HAT | PREPARE_OUTPUT_MASK_K_HAT |
        PREPARE_OUTPUT_MASK_Q_RSTD | PREPARE_OUTPUT_MASK_K_RSTD |
        PREPARE_OUTPUT_MASK_BETA_EFF;
    constexpr uint32_t expectedRecompute =
        expectedRequired | (expectedOptional & ~PREPARE_OUTPUT_MASK_QG);
    constexpr uint32_t expectedSave = expectedRequired | expectedOptional;
    return PREPARE_OUTPUT_COUNT == 13 &&
        PREPARE_REQUIRED_OUTPUT_MASK == expectedRequired &&
        PREPARE_OPTIONAL_OUTPUT_MASK == expectedOptional &&
        PREPARE_RECOMPUTE_OUTPUT_MASK == expectedRecompute &&
        PREPARE_SAVE_OUTPUT_MASK == expectedSave;
}

bool CheckSingleElementOptionalOutputBits()
{
    using namespace optiling;
    // q_rstd/k_rstd/beta_eff 允许 shape 的元素数为 1，不能据此清除输出位。
    constexpr uint32_t scalarOutputBits =
        PREPARE_OUTPUT_MASK_Q_RSTD | PREPARE_OUTPUT_MASK_K_RSTD |
        PREPARE_OUTPUT_MASK_BETA_EFF;
    return (PREPARE_OPTIONAL_OUTPUT_MASK & scalarOutputBits) ==
        scalarOutputBits;
}

bool CheckChunkOnlySchedule()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        2, 4, 8, 16, 0, 24, 1024, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule) &&
        schedule.chunkWorkItems == 32 && schedule.totalWorkItems == 32 &&
        schedule.usedCoreNum == 24 && schedule.headsPerPartition == 8 &&
        schedule.workspaceBytes ==
            1024 + 24 *
                optiling::CHUNK_KDA_FWD_PREPARE_ARCH22_WORKGROUP_BYTES;
}

bool CheckArchSpecificWorkspace()
{
    optiling::ChunkKdaFwdPrepareScheduleContext arch22Context{
        2, 4, 8, 16, 0, 24, 1024, false};
    auto arch35Context = arch22Context;
    arch35Context.workspaceSlotBytes =
        optiling::CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES;
    optiling::ChunkKdaFwdPrepareSchedule arch22Schedule;
    optiling::ChunkKdaFwdPrepareSchedule arch35Schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(arch22Context)
               .Process(arch22Schedule) &&
        optiling::ChunkKdaFwdPrepareTilingProcessor(arch35Context)
            .Process(arch35Schedule) &&
        arch22Schedule.workspaceBytes ==
            1024 + 24 *
                optiling::CHUNK_KDA_FWD_PREPARE_ARCH22_WORKGROUP_BYTES &&
        arch35Schedule.workspaceBytes ==
            1024 + 24 *
                optiling::CHUNK_KDA_FWD_PREPARE_ARCH35_WORKGROUP_BYTES;
}

bool CheckGvaHeadSplitSchedule()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        1, 4, 8, 1, 0, 24, 0, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule) &&
        schedule.chunkWorkItems == 1 && schedule.totalWorkItems == 4 &&
        schedule.usedCoreNum == 4 && schedule.headsPerPartition == 2;
}

bool CheckVarLenSchedule()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        1, 3, 12, 0, 7, 6, 4096, true};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule) &&
        schedule.chunkWorkItems == 7 && schedule.totalWorkItems == 7 &&
        schedule.usedCoreNum == 6 && schedule.headsPerPartition == 12;
}

bool CheckInvalidGva()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        1, 3, 8, 1, 0, 24, 0, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return !optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule);
}

bool CheckChunkWorkItemOverflow()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
        1, 1, 2, 0, 24, 0, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return !optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule);
}

bool CheckHeadPartitionWorkItemOverflow()
{
    const uint64_t maxUint32 = std::numeric_limits<uint32_t>::max();
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        1, 3, 3, maxUint32 / 2 + 1, 0, maxUint32, 0, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return !optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule);
}

bool CheckPlatformTraits()
{
    using namespace optiling;
    constexpr auto arch22 = GetPlatformTraits(PrepareKernelTier::Arch22);
    constexpr auto arch35 = GetPlatformTraits(PrepareKernelTier::Arch35);
    constexpr auto arch92 = GetPlatformTraits(PrepareKernelTier::Arch92);
    constexpr auto unknown = GetPlatformTraits(PrepareKernelTier::Unknown);
    return arch22.tier == PrepareKernelTier::Arch22 && arch22.arch22TargetEligible &&
        !arch22.useLegacyTilingKey &&
        arch22.workspaceSlotBytes == CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES &&
        arch22.slotsPerWorkgroup == CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP &&
        arch35.tier == PrepareKernelTier::Arch35 && !arch35.arch22TargetEligible &&
        arch35.useLegacyTilingKey &&
        arch35.workspaceSlotBytes == CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES &&
        arch35.slotsPerWorkgroup == CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP &&
        arch92.tier == PrepareKernelTier::Arch92 && !arch92.arch22TargetEligible &&
        arch92.useLegacyTilingKey &&
        arch92.workspaceSlotBytes == CHUNK_KDA_FWD_PREPARE_ARCH92_SLOT_BYTES &&
        arch92.slotsPerWorkgroup == 3 &&
        unknown.tier == PrepareKernelTier::Unknown && unknown.workspaceSlotBytes == 0;
}

bool CheckTierForNpuArchId()
{
    using namespace optiling;
    return TierForNpuArchId(2201) == PrepareKernelTier::Arch22 &&
        TierForNpuArchId(3510) == PrepareKernelTier::Arch35 &&
        TierForNpuArchId(9201) == PrepareKernelTier::Arch92 &&
        TierForNpuArchId(9999) == PrepareKernelTier::Unknown;
}

bool CheckArch92Workspace()
{
    optiling::ChunkKdaFwdPrepareScheduleContext arch92Context{
        2, 4, 8, 16, 0, 24, 1024, false};
    arch92Context.workspaceSlotBytes =
        optiling::CHUNK_KDA_FWD_PREPARE_ARCH92_SLOT_BYTES;
    arch92Context.slotsPerWorkgroup = 3;
    optiling::ChunkKdaFwdPrepareSchedule arch92Schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(arch92Context)
               .Process(arch92Schedule) &&
        arch92Schedule.usedCoreNum == 24 &&
        arch92Schedule.workspaceBytes ==
            1024 + 24 * 3 * optiling::CHUNK_KDA_FWD_PREPARE_ARCH92_SLOT_BYTES;
}

} // namespace

int main()
{
    return CheckOutputMaskContract() && CheckSingleElementOptionalOutputBits() &&
                   CheckChunkOnlySchedule() &&
                   CheckArchSpecificWorkspace() &&
                   CheckGvaHeadSplitSchedule() &&
                   CheckVarLenSchedule() && CheckInvalidGva() &&
                   CheckChunkWorkItemOverflow() &&
                   CheckHeadPartitionWorkItemOverflow() &&
                   CheckPlatformTraits() && CheckTierForNpuArchId() &&
                   CheckArch92Workspace()
               ? 0
               : 1;
}
