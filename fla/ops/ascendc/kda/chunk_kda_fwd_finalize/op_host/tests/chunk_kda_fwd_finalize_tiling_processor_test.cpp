#include <cassert>
#include <cstdint>
#include <limits>
#include "../chunk_kda_fwd_finalize_tiling_processor.h"

using namespace optiling;

int main()
{
    ChunkKdaFwdFinalizeScheduleContext context;
    context.batch = 1;
    context.valueHeadNum = 96;
    context.chunksPerSequence = 128;
    context.aicCoreNum = 20;
    context.libApiWorkspaceBytes = 4096;
    ChunkKdaFwdFinalizeSchedule schedule;

    // The default is the existing A5 chunk-first schedule, including small inputs.
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 96 && schedule.totalWorkItems == 128);
    context.chunksPerSequence = 1;
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 1 && schedule.totalWorkItems == 96);

    // The selected A2/A3 schedule distributes every chunk/head pair independently.
    context.headGroupSize = 1;
    context.chunksPerSequence = 128;
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 1 && schedule.totalWorkItems == 12288);
    assert(schedule.usedCoreNum == 20 && schedule.workspaceBytes == 4096);
    context.valueHeadNum = 5;
    context.batch = 2;
    context.chunksPerSequence = 2;
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 1);
    assert(schedule.chunkWorkItems == 4 && schedule.totalWorkItems == 20);
    assert(schedule.usedCoreNum == 20);
    context.isVarLen = true;
    context.totalVarLenChunks = 21;
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 1);
    assert(schedule.chunkWorkItems == 21 && schedule.totalWorkItems == 105);
    context.valueHeadNum = 6;
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 1 && schedule.totalWorkItems == 126);
    context.valueHeadNum = 32;
    context.totalVarLenChunks = 3;
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 1 && schedule.totalWorkItems == 96);
    context.valueHeadNum = 1;
    context.totalVarLenChunks = 21;
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 1 && schedule.totalWorkItems == 21);

    // Candidate CG4 also handles non-divisible HV without dropping the tail head.
    context.headGroupSize = 4;
    context.valueHeadNum = 5;
    context.totalVarLenChunks = 21;
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 4 && schedule.totalWorkItems == 42);

    // Reject groups beyond the four physical L1 lanes and overflowing work domains.
    context.headGroupSize = 5;
    assert(!ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    context.headGroupSize = 1;
    context.valueHeadNum = std::numeric_limits<uint32_t>::max();
    context.aicCoreNum = 1;
    context.totalVarLenChunks = 1;
    assert(ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    assert(schedule.headsPerPartition == 1);
    assert(schedule.totalWorkItems == std::numeric_limits<uint32_t>::max());
    context.totalVarLenChunks = 2;
    assert(!ChunkKdaFwdFinalizeTilingProcessor(context).Process(schedule));
    return 0;
}
