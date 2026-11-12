#ifndef CHUNK_KDA_FWD_FINALIZE_TILING_PROCESSOR_H
#define CHUNK_KDA_FWD_FINALIZE_TILING_PROCESSOR_H

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint64_t FINALIZE_CHUNK_ROWS = 64;
constexpr uint64_t FINALIZE_A5_GROUP_SIZE = 4;
constexpr uint64_t FINALIZE_A5_DENSE_MOVER_MIN_CHUNKS_PER_CORE = 4;
constexpr uint64_t FINALIZE_A5_VARLEN_MOVER_MIN_CHUNKS_PER_CORE = 8;

enum class ChunkKdaFwdFinalizeScheduleMode : uint32_t {
    A2A3_ADAPTIVE = 0,
    A5_FULL_HEAD_MOVER = 1,
    A5_GROUP4_AIC_ONLY = 2,
};

struct ChunkKdaFwdFinalizeScheduleContext {
    uint64_t batch = 0;
    uint64_t valueHeadNum = 0;
    uint64_t chunksPerSequence = 0;
    uint64_t completeChunksPerSequence = 0;
    uint64_t totalVarLenChunks = 0;
    uint64_t aicCoreNum = 0;
    uint64_t libApiWorkspaceBytes = 0;
    bool isVarLen = false;
    bool isA5 = false;
};

struct ChunkKdaFwdFinalizeSchedule {
    uint32_t usedCoreNum = 0;
    uint32_t headsPerPartition = 0;
    uint32_t groupSize = 0;
    ChunkKdaFwdFinalizeScheduleMode scheduleMode =
        ChunkKdaFwdFinalizeScheduleMode::A2A3_ADAPTIVE;
    uint64_t chunkWorkItems = 0;
    uint64_t totalWorkItems = 0;
    uint64_t workspaceBytes = 0;
    bool useAivInputMover = false;
};

class ChunkKdaFwdFinalizeTilingProcessor {
public:
    explicit ChunkKdaFwdFinalizeTilingProcessor(
        const ChunkKdaFwdFinalizeScheduleContext &context) : context_(context)
    {}

    bool Process(ChunkKdaFwdFinalizeSchedule &schedule) const
    {
        if (context_.batch == 0 || context_.valueHeadNum == 0 ||
            context_.aicCoreNum == 0 ||
            context_.valueHeadNum > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        uint64_t chunkItems = context_.totalVarLenChunks;
        if (!context_.isVarLen &&
            !CheckedMultiply(context_.batch, context_.chunksPerSequence,
                             chunkItems)) {
            return false;
        }
        if (chunkItems == 0 ||
            chunkItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        uint64_t headsPerPartition = 0;
        uint64_t groupSize = 1;
        ChunkKdaFwdFinalizeScheduleMode scheduleMode =
            ChunkKdaFwdFinalizeScheduleMode::A2A3_ADAPTIVE;
        bool useAivInputMover = false;
        if (!context_.isA5) {
            headsPerPartition = chunkItems < context_.aicCoreNum
                ? 1 : context_.valueHeadNum;
        } else {
            groupSize = FINALIZE_A5_GROUP_SIZE;
            const uint64_t fullHeadCores = std::min(context_.aicCoreNum, chunkItems);
            // Dense mover threshold counts complete chunks; a partial tail must
            // not promote a below-threshold workload to the MIX path.
            uint64_t eligibleChunks = chunkItems;
            if (!context_.isVarLen &&
                !CheckedMultiply(context_.batch,
                                 context_.completeChunksPerSequence,
                                 eligibleChunks)) {
                return false;
            }
            const uint64_t chunksPerCore = eligibleChunks / fullHeadCores;
            const uint64_t moverThreshold = context_.isVarLen
                ? FINALIZE_A5_VARLEN_MOVER_MIN_CHUNKS_PER_CORE
                : FINALIZE_A5_DENSE_MOVER_MIN_CHUNKS_PER_CORE;
            useAivInputMover = chunksPerCore >= moverThreshold;
            if (useAivInputMover) {
                headsPerPartition = context_.valueHeadNum;
                scheduleMode =
                    ChunkKdaFwdFinalizeScheduleMode::A5_FULL_HEAD_MOVER;
            } else {
                headsPerPartition =
                    std::min(FINALIZE_A5_GROUP_SIZE, context_.valueHeadNum);
                scheduleMode =
                    ChunkKdaFwdFinalizeScheduleMode::A5_GROUP4_AIC_ONLY;
            }
        }
        const uint64_t partitions = context_.valueHeadNum / headsPerPartition +
            (context_.valueHeadNum % headsPerPartition != 0);
        uint64_t workItems = 0;
        if (!CheckedMultiply(chunkItems, partitions, workItems) ||
            workItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const uint64_t cores = std::min(context_.aicCoreNum, workItems);
        if (cores > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        schedule.usedCoreNum = static_cast<uint32_t>(cores);
        schedule.headsPerPartition = static_cast<uint32_t>(headsPerPartition);
        schedule.groupSize = static_cast<uint32_t>(groupSize);
        schedule.scheduleMode = scheduleMode;
        schedule.chunkWorkItems = chunkItems;
        schedule.totalWorkItems = workItems;
        schedule.workspaceBytes = context_.libApiWorkspaceBytes;
        schedule.useAivInputMover = useAivInputMover;
        return true;
    }

private:
    static bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &out)
    {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
            return false;
        }
        out = lhs * rhs;
        return true;
    }

    const ChunkKdaFwdFinalizeScheduleContext &context_;
};

} // namespace optiling

#endif // CHUNK_KDA_FWD_FINALIZE_TILING_PROCESSOR_H
