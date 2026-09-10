/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "../chunk_kda_fwd_prepare_tiling_processor.h"

#include <cstdint>
#include <limits>

namespace {

bool CheckChunkOnlySchedule()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        2, 4, 8, 16, 0, 24, 1024, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule) &&
        schedule.chunkWorkItems == 32 && schedule.totalWorkItems == 32 &&
        schedule.usedCoreNum == 24 && schedule.headsPerPartition == 8 &&
        schedule.workspaceBytes ==
            1024 + 24 * optiling::CHUNK_KDA_FWD_PREPARE_WORKGROUP_BYTES;
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

} // namespace

int main()
{
    return CheckChunkOnlySchedule() && CheckGvaHeadSplitSchedule() &&
                   CheckVarLenSchedule() && CheckInvalidGva() &&
                   CheckChunkWorkItemOverflow() &&
                   CheckHeadPartitionWorkItemOverflow()
               ? 0
               : 1;
}
