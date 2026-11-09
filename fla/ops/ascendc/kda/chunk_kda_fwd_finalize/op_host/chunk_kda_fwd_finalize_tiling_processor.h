#ifndef CHUNK_KDA_FWD_FINALIZE_TILING_PROCESSOR_H
#define CHUNK_KDA_FWD_FINALIZE_TILING_PROCESSOR_H

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint64_t FINALIZE_CHUNK_ROWS = 64;

// 平台 tier 与 kernel 侧的三段门控一一对应：
//   Arch22 = A2/A3（DAV_2201），Arch35 = A5（DAV_3510），Arch92 = A6（DAV_9201）。
// 只增不改是扩展契约：新增平台只在这里加一个 tier 值 + tiling.cpp 的 NpuArch 映射
// 行 + 一条 traits 表记录，共享调度逻辑不受影响。
enum class FinalizeKernelTier : uint32_t {
    Arch22 = 0,
    Arch35 = 1,
    Arch92 = 2,
    Unknown = 3,
};

// 平台描述表的一行。所有平台差异收敛到这张表，替代 isAscend950 / isAscend9201 /
// isNonArch22 这类组合布尔。字段与 kernel 侧调度/layout 契约一一对应。
struct FinalizePlatformTraits {
    FinalizeKernelTier tier = FinalizeKernelTier::Unknown;
    // A2/A3 用 bounded head group（headGroupSize=1 走 bounded 分核）；A5/A6 保留
    // chunk-first 全 head（headGroupSize=0）。
    uint64_t headGroupSize = 0;
    // 仅 A5/A6 允许 AIV input mover 路径。
    bool allowAivInputMover = false;
};

// tier → 平台参数。Unknown 返回全零 sentinel，调用方须据此显式失败。
constexpr FinalizePlatformTraits GetFinalizePlatformTraits(FinalizeKernelTier tier)
{
    switch (tier) {
    case FinalizeKernelTier::Arch22:
        return {FinalizeKernelTier::Arch22, 1, false};
    case FinalizeKernelTier::Arch35:
        return {FinalizeKernelTier::Arch35, 0, true};
    case FinalizeKernelTier::Arch92:
        return {FinalizeKernelTier::Arch92, 0, true};
    default:
        return {};
    }
}

// NpuArch 枚举到 tier 的显式映射。用裸 uint32_t 接收，保持本头纯 C++、可被 host
// 单测直接覆盖；tiling.cpp 以 static_cast<uint32_t>(GetCurNpuArch()) 调用。
// 未知平台返回 Unknown 以显式失败。函数名带 Finalize 前缀，避免与 prepare 侧
// 同在 optiling 命名空间的 TierForNpuArchId 重定义。
constexpr FinalizeKernelTier FinalizeTierForNpuArchId(uint32_t archId)
{
    switch (archId) {
    case 2201: // DAV_2201：A2/A3（910B/910C）共用 arch22。
        return FinalizeKernelTier::Arch22;
    case 3510: // DAV_3510：A5（Ascend950）。
        return FinalizeKernelTier::Arch35;
    case 9201: // DAV_9201：A6（Ascend960DT）。
        return FinalizeKernelTier::Arch92;
    default:
        return FinalizeKernelTier::Unknown;
    }
}

struct ChunkKdaFwdFinalizeScheduleContext {
    uint64_t batch = 0;
    uint64_t valueHeadNum = 0;
    uint64_t chunksPerSequence = 0;
    uint64_t totalVarLenChunks = 0;
    uint64_t aicCoreNum = 0;
    uint64_t libApiWorkspaceBytes = 0;
    // 0 retains the A5 chunk-first schedule; A2/A3 use bounded head groups.
    uint64_t headGroupSize = 0;
    bool isVarLen = false;
};

struct ChunkKdaFwdFinalizeSchedule {
    uint32_t usedCoreNum = 0;
    uint32_t headsPerPartition = 0;
    uint64_t chunkWorkItems = 0;
    uint64_t totalWorkItems = 0;
    uint64_t workspaceBytes = 0;
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
            context_.headGroupSize > 4 ||
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

        const uint64_t headsPerPartition = context_.headGroupSize != 0
            ? std::min(context_.headGroupSize, context_.valueHeadNum)
            : (chunkItems < context_.aicCoreNum ? 1 : context_.valueHeadNum);
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
        schedule.chunkWorkItems = chunkItems;
        schedule.totalWorkItems = workItems;
        schedule.workspaceBytes = context_.libApiWorkspaceBytes;
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
