#ifndef CHUNK_KDA_FWD_PLAN_H
#define CHUNK_KDA_FWD_PLAN_H

#include <cstddef>
#include <cstdint>

namespace KdaForward {

constexpr uint32_t KDA_COMPACT_PLAN_MAGIC = 0x4B444150U;
constexpr uint32_t KDA_COMPACT_PLAN_VERSION = 6;
// 所有阶段都按同一个运行时窗口上限遍历head；余数由headCnt处理，
// 不再为1/2/3/4个head分别生成模板实例。这里的4只是任务窗口，
// L0C槽数和workspace队列深度必须按各自的流水生命周期独立确定。
constexpr uint32_t KDA_HEADS_PER_TASK = 4;

enum class CompactPlanKind : uint32_t {
    DENSE_AFFINE = 0,
    VARLEN_INDEXED = 1,
};

struct CompactSequencePlanHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t kind;
    uint32_t sequenceCount;
    uint32_t totalChunks;
    uint32_t totalFullChunks;
    uint32_t totalTailChunks;
    uint32_t chunkSize;
    uint32_t alignedSequenceCount;
    uint32_t tailedSequenceCount;
    // 非空chunk阶段游标数，上限为物理AIC核数。编号不小于该值的物理核
    // 必须在任何核间flag操作前跳过本阶段，但仍需到达外层阶段同步点。
    uint32_t chunkUsedCoreNum;
    uint32_t headGroupCount;
    // 版本6保留字段，host固定写0。head窗口完全由H/HV运行时推导，
    // 不再通过plan编码pair/single等模板协议。
    uint32_t chunkStageFlags;
    uint32_t fwdUsedCoreNum;
    uint32_t denseFullChunksPerSequence;
    uint32_t denseTailTokens;
    uint32_t seqChunkOffsetsOffset;
    uint32_t alignedSequenceIdsOffset;
    uint32_t tailedSequenceIdsOffset;
    uint32_t chunkCoreCursorsOffset;
    uint32_t payloadBytes;
};

struct ChunkCoreCursor {
    // 版本6在headGroupCount==1时保存chunk序号，否则保存展平后的
    // (chunk序号, head组)任务序号。host先划分统一的[完整块任务, 尾块任务]
    // owner区间，再保存与两个阶段的交集，使kernel阶段无需额外分支，
    // 同时避免完整块和尾块都集中到同一批核上。
    uint32_t fullBegin;
    uint32_t fullEnd;
    uint32_t fullStartSequence;
    uint32_t fullStartLocalChunk;
    uint32_t tailBegin;
    uint32_t tailEnd;
};

static_assert(sizeof(CompactSequencePlanHeader) == 21 * sizeof(uint32_t),
              "compact plan v6 header layout changed");
static_assert(offsetof(CompactSequencePlanHeader, payloadBytes) ==
                  20 * sizeof(uint32_t),
              "compact plan v6 header offsets changed");
static_assert(sizeof(ChunkCoreCursor) == 6 * sizeof(uint32_t),
              "compact plan v6 cursor layout changed");
static_assert(offsetof(ChunkCoreCursor, fullStartSequence) ==
                  2 * sizeof(uint32_t),
              "compact plan v6 cursor offsets changed");
static_assert(offsetof(ChunkCoreCursor, tailBegin) ==
                  4 * sizeof(uint32_t),
              "compact plan v6 cursor offsets changed");

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#define KDA_PLAN_INLINE __aicore__ inline
#else
#define KDA_PLAN_INLINE constexpr
#endif

// 该公式覆盖任意合法整数ratio，不按具体比例枚举分支。ratio不大于4时，
// 一个窗口容纳尽可能多的完整Hq组；ratio大于4时，同一个Hq按每4个Hv
// 拆窗且不跨Hq边界，例如ratio 7得到[4,3]，ratio 19得到[4,4,4,4,3]。
KDA_PLAN_INLINE uint32_t HeadWindowCount(
    uint32_t queryHeadCount, uint32_t valueHeadCount)
{
    if (queryHeadCount == 0 || valueHeadCount < queryHeadCount ||
        valueHeadCount % queryHeadCount != 0) {
        return 0;
    }
    const uint32_t headRatio = valueHeadCount / queryHeadCount;
    if (headRatio <= KDA_HEADS_PER_TASK) {
        const uint32_t headsPerWindow =
            (KDA_HEADS_PER_TASK / headRatio) * headRatio;
        return valueHeadCount / headsPerWindow +
            static_cast<uint32_t>(valueHeadCount % headsPerWindow != 0);
    }
    const uint32_t windowsPerQuery =
        headRatio / KDA_HEADS_PER_TASK +
        static_cast<uint32_t>(headRatio % KDA_HEADS_PER_TASK != 0);
    return queryHeadCount * windowsPerQuery;
}

// 将窗口序号转换成Hv起点；windowOrdinal==HeadWindowCount时返回HV，
// 便于同一个函数同时表达owner区间的左、右边界。
KDA_PLAN_INLINE uint32_t HeadWindowBegin(
    uint32_t windowOrdinal, uint32_t queryHeadCount,
    uint32_t valueHeadCount)
{
    const uint32_t windowCount =
        HeadWindowCount(queryHeadCount, valueHeadCount);
    if (windowCount == 0) {
        return 0;
    }
    if (windowOrdinal >= windowCount) {
        return windowOrdinal == windowCount ? valueHeadCount : 0;
    }
    const uint32_t headRatio = valueHeadCount / queryHeadCount;
    if (headRatio <= KDA_HEADS_PER_TASK) {
        const uint32_t headsPerWindow =
            (KDA_HEADS_PER_TASK / headRatio) * headRatio;
        return windowOrdinal * headsPerWindow;
    }
    const uint32_t windowsPerQuery =
        headRatio / KDA_HEADS_PER_TASK +
        static_cast<uint32_t>(headRatio % KDA_HEADS_PER_TASK != 0);
    const uint32_t queryHead = windowOrdinal / windowsPerQuery;
    const uint32_t localWindow = windowOrdinal % windowsPerQuery;
    return queryHead * headRatio + localWindow * KDA_HEADS_PER_TASK;
}

// hvBase必须是HeadWindowBegin返回的边界。最后一个窗口只返回真实Hv余数，
// AIC和两个AIV必须使用同一个headCnt，不能用虚拟head补足到4。
KDA_PLAN_INLINE uint32_t HeadWindowHeadCount(
    uint32_t hvBase, uint32_t queryHeadCount, uint32_t valueHeadCount)
{
    if (hvBase >= valueHeadCount ||
        HeadWindowCount(queryHeadCount, valueHeadCount) == 0) {
        return 0;
    }
    const uint32_t headRatio = valueHeadCount / queryHeadCount;
    uint32_t headCount = 0;
    if (headRatio <= KDA_HEADS_PER_TASK) {
        headCount = (KDA_HEADS_PER_TASK / headRatio) * headRatio;
    } else {
        const uint32_t queryHeadEnd =
            (hvBase / headRatio + 1) * headRatio;
        const uint32_t headsUntilQueryEnd = queryHeadEnd - hvBase;
        headCount = headsUntilQueryEnd < KDA_HEADS_PER_TASK
            ? headsUntilQueryEnd
            : KDA_HEADS_PER_TASK;
    }
    const uint32_t remainingHeads = valueHeadCount - hvBase;
    return headCount < remainingHeads ? headCount : remainingHeads;
}

KDA_PLAN_INLINE uint32_t ComputeChunkHeadGroupCount(
    uint32_t totalFullChunks, uint32_t totalTailChunks,
    uint32_t physicalCoreCount,
    uint32_t queryHeadCount, uint32_t valueHeadCount)
{
    // physicalCoreCount是host tiling传入的运行时AIC核数，不能替换成
    // 与SoC或测试绑定的字面量。完整块和尾块虽使用不同计算模板，
    // 但共享同一个host owner区间，两个本地阶段之间也没有核间barrier，
    // 因此两类任务都参与填满物理核。
    const uint64_t groupingChunkCount =
        static_cast<uint64_t>(totalFullChunks) + totalTailChunks;
    const uint32_t headWindowCount =
        HeadWindowCount(queryHeadCount, valueHeadCount);
    if (groupingChunkCount == 0 || physicalCoreCount == 0 ||
        headWindowCount == 0) {
        return 0;
    }
    if (groupingChunkCount >= physicalCoreCount) {
        return 1;
    }
    const uint32_t fillGroups = static_cast<uint32_t>(
        (static_cast<uint64_t>(physicalCoreCount) + groupingChunkCount - 1) /
        groupingChunkCount);
    return fillGroups < headWindowCount ? fillGroups : headWindowCount;
}

// 调度组按window ordinal均分，组边界因此始终落在完整runtime窗口上。
KDA_PLAN_INLINE uint32_t HeadGroupBegin(
    uint32_t group, uint32_t groupCount, uint32_t queryHeadCount,
    uint32_t valueHeadCount)
{
    const uint32_t windowCount =
        HeadWindowCount(queryHeadCount, valueHeadCount);
    if (groupCount == 0 || group > groupCount || windowCount == 0) {
        return 0;
    }
    const uint32_t windowOrdinal = static_cast<uint32_t>(
        static_cast<uint64_t>(group) * windowCount / groupCount);
    return HeadWindowBegin(
        windowOrdinal, queryHeadCount, valueHeadCount);
}

KDA_PLAN_INLINE uint32_t HeadGroupEnd(
    uint32_t group, uint32_t groupCount, uint32_t queryHeadCount,
    uint32_t valueHeadCount)
{
    return HeadGroupBegin(
        group + 1, groupCount, queryHeadCount, valueHeadCount);
}

#undef KDA_PLAN_INLINE

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__) || \
    defined(KDA_ENABLE_COMPACT_PLAN_VIEW)
class CompactSequencePlanView {
public:
    __aicore__ inline explicit CompactSequencePlanView(GM_ADDR plan)
        : base_(reinterpret_cast<const __gm__ uint8_t *>(plan))
    {
    }

    __aicore__ inline bool IsValid() const
    {
        return base_ != nullptr &&
            LoadHeaderField(offsetof(CompactSequencePlanHeader, magic)) ==
                KDA_COMPACT_PLAN_MAGIC &&
            LoadHeaderField(offsetof(CompactSequencePlanHeader, version)) ==
                KDA_COMPACT_PLAN_VERSION;
    }

    __aicore__ inline uint32_t SequenceCount() const
    {
        return LoadHeaderField(offsetof(CompactSequencePlanHeader, sequenceCount));
    }

    __aicore__ inline uint32_t AlignedSequenceCount() const
    {
        return LoadHeaderField(
            offsetof(CompactSequencePlanHeader, alignedSequenceCount));
    }

    __aicore__ inline uint32_t TailedSequenceCount() const
    {
        return LoadHeaderField(
            offsetof(CompactSequencePlanHeader, tailedSequenceCount));
    }

    __aicore__ inline bool IsDenseAffine() const
    {
        return LoadHeaderField(offsetof(CompactSequencePlanHeader, kind)) ==
            static_cast<uint32_t>(CompactPlanKind::DENSE_AFFINE);
    }

    __aicore__ inline uint32_t DenseFullChunksPerSequence() const
    {
        return LoadHeaderField(
            offsetof(CompactSequencePlanHeader, denseFullChunksPerSequence));
    }

    __aicore__ inline uint32_t DenseTailTokens() const
    {
        return LoadHeaderField(
            offsetof(CompactSequencePlanHeader, denseTailTokens));
    }

    __aicore__ inline uint32_t ChunkUsedCoreNum() const
    {
        return LoadHeaderField(offsetof(CompactSequencePlanHeader, chunkUsedCoreNum));
    }

    __aicore__ inline uint32_t HeadGroupCount() const
    {
        return LoadHeaderField(offsetof(CompactSequencePlanHeader, headGroupCount));
    }

    __aicore__ inline uint32_t HeadGroupBegin(
        uint32_t group, uint32_t queryHeadCount,
        uint32_t valueHeadCount) const
    {
        return KdaForward::HeadGroupBegin(
            group, HeadGroupCount(), queryHeadCount, valueHeadCount);
    }

    __aicore__ inline uint32_t HeadGroupEnd(
        uint32_t group, uint32_t queryHeadCount,
        uint32_t valueHeadCount) const
    {
        return KdaForward::HeadGroupEnd(
            group, HeadGroupCount(), queryHeadCount, valueHeadCount);
    }

    __aicore__ inline bool DecodeChunkHeadGroupTask(
        uint32_t task, uint32_t queryHeadCount, uint32_t valueHeadCount,
        uint32_t &chunkOrdinal, uint32_t &headBegin,
        uint32_t &headEnd) const
    {
        const uint32_t groupCount = HeadGroupCount();
        if (groupCount == 0) {
            return false;
        }
        const uint32_t group = task % groupCount;
        chunkOrdinal = task / groupCount;
        headBegin = HeadGroupBegin(
            group, queryHeadCount, valueHeadCount);
        headEnd = HeadGroupEnd(
            group, queryHeadCount, valueHeadCount);
        return headBegin < headEnd && headEnd <= valueHeadCount;
    }

    __aicore__ inline uint32_t FwdUsedCoreNum() const
    {
        return LoadHeaderField(offsetof(CompactSequencePlanHeader, fwdUsedCoreNum));
    }

    __aicore__ inline uint32_t SequenceChunkOffsetsOffset() const
    {
        return LoadHeaderField(
            offsetof(CompactSequencePlanHeader, seqChunkOffsetsOffset));
    }

    __aicore__ inline uint32_t SequenceChunkOffset(uint32_t sequence) const
    {
        if (IsDenseAffine()) {
            const uint32_t fullChunks = DenseFullChunksPerSequence();
            const uint32_t tailChunks = DenseTailTokens() != 0;
            return sequence * (fullChunks + tailChunks);
        }
        const uint32_t arrayOffset = SequenceChunkOffsetsOffset();
        return LoadU32(arrayOffset + sequence * sizeof(uint32_t));
    }

    __aicore__ inline uint32_t AlignedSequenceId(uint32_t ordinal) const
    {
        if (IsDenseAffine()) {
            return ordinal;
        }
        const uint32_t arrayOffset = LoadHeaderField(
            offsetof(CompactSequencePlanHeader, alignedSequenceIdsOffset));
        return LoadU16(arrayOffset + ordinal * sizeof(uint16_t));
    }

    __aicore__ inline uint32_t TailedSequenceId(uint32_t tailOrdinal) const
    {
        if (IsDenseAffine()) {
            return tailOrdinal;
        }
        const uint32_t arrayOffset =
            LoadHeaderField(offsetof(CompactSequencePlanHeader, tailedSequenceIdsOffset));
        return LoadU16(arrayOffset + tailOrdinal * sizeof(uint16_t));
    }

    __aicore__ inline bool LoadChunkCoreCursor(
        uint32_t coreIdx, ChunkCoreCursor &cursor) const
    {
        if (!IsValid() || coreIdx >= ChunkUsedCoreNum()) {
            return false;
        }
        const uint32_t arrayOffset =
            LoadHeaderField(offsetof(CompactSequencePlanHeader, chunkCoreCursorsOffset));
        const uint32_t cursorOffset = arrayOffset + coreIdx * sizeof(ChunkCoreCursor);
        cursor.fullBegin = LoadU32(
            cursorOffset + offsetof(ChunkCoreCursor, fullBegin));
        cursor.fullEnd = LoadU32(
            cursorOffset + offsetof(ChunkCoreCursor, fullEnd));
        cursor.fullStartSequence = LoadU32(
            cursorOffset + offsetof(ChunkCoreCursor, fullStartSequence));
        cursor.fullStartLocalChunk = LoadU32(
            cursorOffset + offsetof(ChunkCoreCursor, fullStartLocalChunk));
        cursor.tailBegin = LoadU32(
            cursorOffset + offsetof(ChunkCoreCursor, tailBegin));
        cursor.tailEnd = LoadU32(
            cursorOffset + offsetof(ChunkCoreCursor, tailEnd));
        return true;
    }

private:
    __aicore__ inline uint32_t LoadHeaderField(uint32_t offset) const
    {
        return LoadU32(offset);
    }

    __aicore__ inline uint32_t LoadU32(uint32_t offset) const
    {
        return *reinterpret_cast<const __gm__ uint32_t *>(base_ + offset);
    }

    __aicore__ inline uint16_t LoadU16(uint32_t offset) const
    {
        return *reinterpret_cast<const __gm__ uint16_t *>(base_ + offset);
    }

    const __gm__ uint8_t *base_ = nullptr;
};
#endif

} // 命名空间 KdaForward

#endif
