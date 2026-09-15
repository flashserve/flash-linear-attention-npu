#include "../chunk_kda_fwd_finalize_tiling_processor.h"

#include <cstdint>
#include <limits>

namespace {

bool CheckChunkFirst()
{
    const optiling::ChunkKdaFwdFinalizeScheduleContext context{
        2, 8, 16, 0, 24, 1024, false, true};
    optiling::ChunkKdaFwdFinalizeSchedule schedule;
    return optiling::ChunkKdaFwdFinalizeTilingProcessor(context)
               .Process(schedule) &&
           schedule.chunkWorkItems == 32 &&
           schedule.totalWorkItems == 32 &&
           schedule.usedCoreNum == 24 &&
           schedule.headsPerPartition == 8 &&
           schedule.workspaceBytes ==
               1024 + 24 *
                   optiling::FINALIZE_ARCH22_WORKSPACE_BYTES_PER_CORE;
}

bool CheckHeadSplitWhenChunksInsufficient()
{
    const optiling::ChunkKdaFwdFinalizeScheduleContext context{
        1, 8, 1, 0, 24, 0, false, true};
    optiling::ChunkKdaFwdFinalizeSchedule schedule;
    return optiling::ChunkKdaFwdFinalizeTilingProcessor(context)
               .Process(schedule) &&
           schedule.chunkWorkItems == 1 &&
           schedule.totalWorkItems == 8 &&
           schedule.usedCoreNum == 8 &&
           schedule.headsPerPartition == 1;
}

bool CheckVarLenAndArch35()
{
    const optiling::ChunkKdaFwdFinalizeScheduleContext context{
        1, 12, 0, 7, 6, 4096, true, false};
    optiling::ChunkKdaFwdFinalizeSchedule schedule;
    return optiling::ChunkKdaFwdFinalizeTilingProcessor(context)
               .Process(schedule) &&
           schedule.chunkWorkItems == 7 &&
           schedule.totalWorkItems == 7 &&
           schedule.usedCoreNum == 6 &&
           schedule.headsPerPartition == 12 &&
           schedule.workspaceBytes == 4096;
}

bool CheckChunkCountOverflow()
{
    const optiling::ChunkKdaFwdFinalizeScheduleContext context{
        std::numeric_limits<uint32_t>::max(), 1, 2, 0, 24, 0,
        false, true};
    optiling::ChunkKdaFwdFinalizeSchedule schedule;
    return !optiling::ChunkKdaFwdFinalizeTilingProcessor(context)
                .Process(schedule);
}

bool CheckHeadPartitionOverflow()
{
    const uint64_t maxUint32 = std::numeric_limits<uint32_t>::max();
    const optiling::ChunkKdaFwdFinalizeScheduleContext context{
        1, 3, maxUint32 / 2 + 1, 0, maxUint32, 0, false, true};
    optiling::ChunkKdaFwdFinalizeSchedule schedule;
    return !optiling::ChunkKdaFwdFinalizeTilingProcessor(context)
                .Process(schedule);
}

bool CheckWorkspaceOverflow()
{
    const optiling::ChunkKdaFwdFinalizeScheduleContext context{
        1, 1, 1, 0, 1, std::numeric_limits<uint64_t>::max(), false, true};
    optiling::ChunkKdaFwdFinalizeSchedule schedule;
    return !optiling::ChunkKdaFwdFinalizeTilingProcessor(context)
                .Process(schedule);
}

} // namespace

int main()
{
    return CheckChunkFirst() &&
                   CheckHeadSplitWhenChunksInsufficient() &&
                   CheckVarLenAndArch35() &&
                   CheckChunkCountOverflow() &&
                   CheckHeadPartitionOverflow() &&
                   CheckWorkspaceOverflow()
               ? 0
               : 1;
}
