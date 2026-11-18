#ifndef GDN_BWD_FINALIZE_A2_WORKSPACE_H
#define GDN_BWD_FINALIZE_A2_WORKSPACE_H

#include <cstdint>
#include "dq/chunk_bwd_dqkwg_schedule.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#include "chunk_gated_delta_rule_bwd_finalize_struct.h"
#else
#include "../chunk_gated_delta_rule_bwd_finalize_struct.h"
#endif

#if defined(__NPU_ARCH__)
#define GDN_A2_HOST_DEVICE __aicore__ inline
#else
#define GDN_A2_HOST_DEVICE inline
#endif

namespace GDN {

// 固定 SDK 模板字段宽度为 8+8+1+1+1；私有第六个 bool 占 bit19。
// Host 保持原五字段 GET，确认旧 key 不占该位后才合入此标志。
constexpr uint64_t A2_FINALIZE_MHA_OWNER_KEY_MASK = uint64_t{1} << 19;

// 与原 device 调度共用整数 peak 评分，确保 Host 分配和实际 head 分区一致。
GDN_A2_HOST_DEVICE bool A2FinalizeDqHkPartitionPeak(
    uint64_t coreLoops, uint64_t HV, uint64_t coreNum, uint64_t partitions, uint64_t &peak)
{
    if (coreLoops == 0 || coreLoops > UINT32_MAX || coreNum == 0 ||
        partitions == 0 || partitions > UINT32_MAX || HV == 0 || HV > UINT32_MAX || HV % partitions != 0) {
        return false;
    }
    // 两个 U32 因子的乘积可由 U64 表示，因此可在乘后检查原 INT64 上界。
    const uint64_t tasks = coreLoops * partitions;
    if (tasks > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    const uint64_t waves = tasks / coreNum + static_cast<uint64_t>(tasks % coreNum != 0);
    const uint64_t heads = HV / partitions;
    // waves<=tasks，且 P 整除 HV，所以 waves*heads<=coreLoops*HV<=UINT32_MAX^2。
    peak = waves * heads;
    return true;
}

GDN_A2_HOST_DEVICE uint64_t A2FinalizeSelectDqHkPartitions(
    uint64_t B, uint64_t numChunks, uint64_t HV, uint64_t HK,
    uint64_t coreNum, uint64_t parentPartitions, bool allowBalance)
{
    if (!allowBalance || !DqkwgUseU32Schedule(B, numChunks, HV, HK, coreNum, parentPartitions)) {
        return parentPartitions;
    }
    // 原 U32 资格先除后乘界定了 B*numChunks，此处才形成 N。
    const uint64_t coreLoops = B * numChunks;
    if (coreLoops < coreNum) {
        return parentPartitions;
    }
    uint64_t parentPeak = 0;
    if (!A2FinalizeDqHkPartitionPeak(coreLoops, HV, coreNum, parentPartitions, parentPeak)) {
        return parentPartitions;
    }
    uint64_t bestPeak = parentPeak;
    uint64_t bestPartitions = parentPartitions;
    for (uint64_t divisor = 1; divisor <= HK / divisor; ++divisor) {
        if (HK % divisor != 0) {
            continue;
        }
        const uint64_t other = HK / divisor;
        for (uint32_t candidate = 0; candidate < 2; ++candidate) {
            const uint64_t partitions = candidate == 0 ? divisor : other;
            if ((candidate != 0 && other == divisor) || (HK / partitions) % 2 != 0) {
                continue;
            }
            uint64_t peak = 0;
            if (A2FinalizeDqHkPartitionPeak(coreLoops, HV, coreNum, partitions, peak) &&
                (peak < bestPeak || (peak == bestPeak && partitions < bestPartitions))) {
                bestPeak = peak;
                bestPartitions = partitions;
            }
        }
    }
    return bestPeak < parentPeak ? bestPartitions : parentPartitions;
}

GDN_A2_HOST_DEVICE uint64_t A2FinalizeDqPartitionCount(
    const ChunkGatedDeltaRuleBwdFinalizeTilingData &t, uint64_t blocks)
{
    const uint64_t numChunks = t.isVariable != 0 ? t.totalChunkNum : t.chunkNumForT;
    return A2FinalizeSelectDqHkPartitions(
        t.B, numChunks, t.HV, t.HK, blocks, 1,
        t.isVariable != 0 || t.T % t.chunkSize == 0);
}

GDN_A2_HOST_DEVICE bool A2FinalizeUseOutputReuse(const ChunkGatedDeltaRuleBwdFinalizeTilingData &t)
{
    return t.K == 128 && t.V == 128;
}

// 两槽严格省空间才批处理，否则保留单槽；0 表示完整沿原全 T 路径。
GDN_A2_HOST_DEVICE uint32_t A2FinalizeChunkDkScratchSlots(
    const ChunkGatedDeltaRuleBwdFinalizeTilingData &t, int64_t blocks)
{
    if (t.K != 128 || (t.V != 128 && t.V != 256) ||
        (t.chunkSize != 64 && t.chunkSize != 128) ||
        t.B <= 0 || t.T <= 0 || t.HK <= 0 || t.HV < t.HK ||
        t.HV % t.HK != 0 || t.headRatio != t.HV / t.HK ||
        blocks <= 0 || blocks > UINT32_MAX || t.totalChunkNum <= 0 || t.totalChunkNum > UINT32_MAX) {
        return 0;
    }
    constexpr uint64_t limit = static_cast<uint64_t>(INT64_MAX) - 511U;
    const uint64_t fullFactors[4] = {static_cast<uint64_t>(t.B), static_cast<uint64_t>(t.HK),
                                    static_cast<uint64_t>(t.T), static_cast<uint64_t>(t.K)};
    const uint64_t compactFactors[4] = {static_cast<uint64_t>(blocks), static_cast<uint64_t>(t.HK),
                                       static_cast<uint64_t>(t.chunkSize), static_cast<uint64_t>(t.K)};
    uint64_t full = 2;
    uint64_t compact = 2;
    for (uint32_t i = 0; i < 4; ++i) {
        if (full > limit / fullFactors[i] || compact > limit / compactFactors[i]) {
            return 0;
        }
        full *= fullFactors[i];
        compact *= compactFactors[i];
    }
    const uint64_t fullBytes = (full + 511U) / 512U * 512U;
    const uint64_t compactBytes = (compact + 511U) / 512U * 512U;
    // K128、BT64/128 使每 core 块本身 512 对齐；总量与按 core 对齐之和相同。
    if (compactBytes >= fullBytes) {
        return 0;
    }
    return compactBytes <= limit / 2U && compactBytes * 2U < fullBytes ? 2U : 1U;
}

// 所选槽的全部 core 总字节数；Slots 已逐因子证明乘积及额外乘2均不溢出。
GDN_A2_HOST_DEVICE int64_t A2FinalizeChunkDkScratchBytes(
    const ChunkGatedDeltaRuleBwdFinalizeTilingData &t, int64_t blocks)
{
    const uint32_t slots = A2FinalizeChunkDkScratchSlots(t, blocks);
    if (slots == 0) {
        return 0;
    }
    return blocks * t.HK * t.chunkSize * t.K * 2 * slots;
}

GDN_A2_HOST_DEVICE bool A2FinalizeUseChunkDkScratch(
    const ChunkGatedDeltaRuleBwdFinalizeTilingData &t, int64_t blocks)
{
    return A2FinalizeChunkDkScratchBytes(t, blocks) != 0;
}

// Host 和 device 共用同一份字节布局。DQ/WY 的临时区只在全核 barrier 后复用。
struct A2FinalizeWorkspaceLayout {
    int64_t gOffset;
    int64_t betaOffset;
    int64_t hOffset;
    int64_t dhOffset;
    int64_t dqOffset;
    int64_t dkOffset;
    int64_t dkWyOffset;
    int64_t dwOffset;
    int64_t dgDqOffset;
    int64_t dgWyOffset;
    int64_t dbetaOffset;
    int64_t stageOffset;
    int64_t dqWorkspaceBytes;
    int64_t wyWorkspaceBytes;
    int64_t totalBytes;

    GDN_A2_HOST_DEVICE static int64_t Align(int64_t bytes)
    {
        return (bytes + 511) / 512 * 512;
    }

    GDN_A2_HOST_DEVICE static int64_t DqWorkspaceHeads(
        const ChunkGatedDeltaRuleBwdFinalizeTilingData &t, int64_t blocks, uint64_t partitions)
    {
        const int64_t compactHeads = t.HV / static_cast<int64_t>(partitions);
        // K=V128的私有slot按分区heads消费，DQ/WY后续仍取容量最大值复用。
        if (A2FinalizeUseOutputReuse(t)) {
            return compactHeads;
        }
        const int64_t btk = blocks * compactHeads * t.chunkSize * t.K * 2;
        const int64_t btb = blocks * compactHeads * t.chunkSize * t.chunkSize * 2;
        const int64_t compactDqBytes = 4 * btk + 2 * btb + Align(blocks * t.HV * 4);
        const int64_t kBytes = t.chunkSize * t.K * 2;
        const int64_t vBytes = t.chunkSize * t.V * 2;
        const int64_t mBytes = t.chunkSize * t.chunkSize * 2;
        const int64_t wyBytes = blocks * 4 * (5 * kBytes + 2 * vBytes + 7 * mBytes);
        return compactDqBytes >= wyBytes ? compactHeads : t.HV;
    }

    GDN_A2_HOST_DEVICE A2FinalizeWorkspaceLayout(
        const ChunkGatedDeltaRuleBwdFinalizeTilingData &t, int64_t blocks, bool fp32Dbeta)
    {
        const int64_t scalar = t.B * t.HV * t.T;
        const int64_t qBytes = t.B * t.HK * t.T * t.K * 2;
        const int64_t stateBytes = t.stateVFirst != 0 ?
            t.totalChunkNum * t.HV * t.K * t.V * 2 : 0;
        const bool reuseOutputs = A2FinalizeUseOutputReuse(t);
        int64_t offset = 0;
        gOffset = offset; offset += Align(scalar * 4);
        betaOffset = offset; offset += Align(scalar * 4);
        hOffset = offset; offset += Align(stateBytes);
        dhOffset = offset; offset += Align(stateBytes);
        dqOffset = offset;
        if (!reuseOutputs) {
            offset += Align(qBytes);
        }
        dkOffset = offset;
        if (!reuseOutputs) {
            offset += Align(qBytes);
        }
        const int64_t chunkDkBytes = A2FinalizeChunkDkScratchBytes(t, blocks);
        dkWyOffset = offset; offset += chunkDkBytes != 0 ? chunkDkBytes : Align(qBytes);
        dwOffset = offset;
        if (!reuseOutputs) {
            offset += Align(scalar * t.K * 2);
        }
        dgDqOffset = offset;
        if (!fp32Dbeta) {
            offset += Align(scalar * 4);
        }
        dgWyOffset = offset; offset += Align(scalar * 4);
        dbetaOffset = offset;
        if (!fp32Dbeta) {
            offset += Align(scalar * 4);
        }
        stageOffset = offset;

        const int64_t dqHeads = DqWorkspaceHeads(t, blocks, A2FinalizeDqPartitionCount(t, blocks));
        const int64_t btk = blocks * dqHeads * t.chunkSize * t.K * 2;
        const int64_t btb = blocks * dqHeads * t.chunkSize * t.chunkSize * 2;
        dqWorkspaceBytes = 4 * btk + 2 * btb + Align(blocks * t.HV * 4);
        const int64_t kBytes = t.chunkSize * t.K * 2;
        const int64_t vBytes = t.chunkSize * t.V * 2;
        const int64_t mBytes = t.chunkSize * t.chunkSize * 2;
        wyWorkspaceBytes = blocks * 4 * (5 * kBytes + 2 * vBytes + 7 * mBytes);
        offset += Align(dqWorkspaceBytes > wyWorkspaceBytes ? dqWorkspaceBytes : wyWorkspaceBytes);
        totalBytes = offset;
    }
};

} // namespace GDN

#undef GDN_A2_HOST_DEVICE
#endif
