/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_TILING_PROCESSOR_H
#define CHUNK_KDA_FWD_PREPARE_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "../op_kernel/arch22/chunk_kda_fwd_prepare_target_config.h"
#include "../op_kernel/arch22/chunk_kda_fwd_prepare_relay_config.h"

namespace optiling {

constexpr uint64_t CHUNK_KDA_FWD_PREPARE_CHUNK_ROWS = 64;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES = 0x1A400;
// A6(9201) 的 workspace 单槽字节数与 arch35 相同（kernel 侧 kSlotStride=0x1A400），
// 但语义独立：A6 为 3 槽、arch35 为 4 槽。单独成常量，避免 host 侧用 isAscend950
// 笼统包含 DAV_9201 产生架构歧义。
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH92_SLOT_BYTES = 0x1A400;
// Arch22：105 KiB 公共区 + 20 KiB Cube relay + 64 KiB FP32 RHS + 16 KiB FP32 Akk。
// 该值必须与 kernel 侧 Workspace::kArch22SlotStride 保持一致。
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES =
    KdaPrepare::Arch22RelayConfig::kSlotBytes;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP =
    KdaPrepare::Arch22RelayConfig::kSlotsPerWorkgroup;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH35_WORKGROUP_BYTES =
    CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES *
    CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH22_WORKGROUP_BYTES =
    CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES *
    CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP;

// 平台 tier 与 kernel 侧的三段门控一一对应：
//   Arch22 = A2/A3（DAV_2201），Arch35 = A5（DAV_3510），Arch92 = A6（DAV_9201）。
// 只增不改是扩展契约：新增平台（如 A7）只在这里加一个 tier 值 + tiling.cpp 的
// NpuArch 映射行 + 一条 traits 表记录，共享调度逻辑不受影响。
enum class PrepareKernelTier : uint32_t {
    Arch22 = 0,
    Arch35 = 1,
    Arch92 = 2,
    Unknown = 3,
};

// 平台描述表的一行。所有平台差异收敛到这张表，替代 isNonArch22 / isArch35Like
// 这类组合布尔值。字段与 kernel 侧布局/调度契约一一对应。
struct PreparePlatformTraits {
    PrepareKernelTier tier = PrepareKernelTier::Unknown;
    // 仅 Arch22 可能命中 target 快速变体；A5/A6 恒 false。
    bool arch22TargetEligible = false;
    // Arch35/Arch92 用 legacy tiling key；Arch22 用 runtime key。
    bool useLegacyTilingKey = false;
    // 通用（非 target）workspace 单槽字节数与每 workgroup 槽数。
    uint64_t workspaceSlotBytes = 0;
    uint64_t slotsPerWorkgroup = 0;
};

// tier → 平台参数。Unknown 返回全零 sentinel，调用方须据此显式失败。
constexpr PreparePlatformTraits GetPlatformTraits(PrepareKernelTier tier)
{
    switch (tier) {
    case PrepareKernelTier::Arch22:
        return {PrepareKernelTier::Arch22, true, false,
                CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES,
                CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP};
    case PrepareKernelTier::Arch35:
        return {PrepareKernelTier::Arch35, false, true,
                CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES,
                CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP};
    case PrepareKernelTier::Arch92:
        // A6(1:1) 每 workgroup 只 3 个在途 head，收敛为 3；必须与 kernel 侧
        // Workspace::kArch92SlotCount 保持一致（见 op_kernel policy.h）。
        return {PrepareKernelTier::Arch92, false, true,
                CHUNK_KDA_FWD_PREPARE_ARCH92_SLOT_BYTES, 3};
    default:
        return {};
    }
}

// NpuArch 枚举（见 platform/soc_spec.h）到 tier 的显式映射。用裸 uint32_t 接收，
// 保持本头纯 C++、可被 host 单测直接覆盖；tiling.cpp 以
// static_cast<uint32_t>(GetCurNpuArch()) 调用。未知平台返回 Unknown 以显式失败。
constexpr PrepareKernelTier TierForNpuArchId(uint32_t archId)
{
    switch (archId) {
    case 2201: // DAV_2201：A2/A3（910B/910C）共用 arch22。
        return PrepareKernelTier::Arch22;
    case 3510: // DAV_3510：A5（Ascend950）。
        return PrepareKernelTier::Arch35;
    case 9201: // DAV_9201：A6（Ascend960DT）。
        return PrepareKernelTier::Arch92;
    default:
        return PrepareKernelTier::Unknown;
    }
}

struct ChunkKdaFwdPrepareScheduleContext {
    uint64_t batch = 0;
    uint64_t qkHeadNum = 0;
    uint64_t valueHeadNum = 0;
    uint64_t chunksPerSequence = 0;
    uint64_t totalVarLenChunks = 0;
    uint64_t aicCoreNum = 0;
    uint64_t libApiWorkspaceBytes = 0;
    bool isVarLen = false;
    // 调用方未显式识别平台时按较大的 Arch22 slot 保守分配。
    uint64_t workspaceSlotBytes = CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES;
    // A6(9201) 每 workgroup 只 3 个在途 head，收敛为 3；其余架构保持 4。
    // 该值必须与 kernel 侧 Workspace::kArch92SlotCount / kArch35SlotCount 一致。
    uint64_t slotsPerWorkgroup = CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP;
    bool targetPipeline = false;
};

struct ChunkKdaFwdPrepareSchedule {
    uint32_t usedCoreNum = 0;
    uint32_t headsPerPartition = 0;
    uint64_t chunkWorkItems = 0;
    uint64_t totalWorkItems = 0;
    uint64_t workspaceBytes = 0;
};

class ChunkKdaFwdPrepareTilingProcessor {
public:
    explicit ChunkKdaFwdPrepareTilingProcessor(
        const ChunkKdaFwdPrepareScheduleContext &context) : context_(context)
    {}

    bool Process(ChunkKdaFwdPrepareSchedule &schedule) const
    {
        if (context_.batch == 0 || context_.qkHeadNum == 0 ||
            context_.valueHeadNum < context_.qkHeadNum ||
            context_.valueHeadNum % context_.qkHeadNum != 0 ||
            context_.aicCoreNum == 0 ||
            (context_.workspaceSlotBytes !=
                 CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES &&
             context_.workspaceSlotBytes !=
                 CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES &&
             context_.workspaceSlotBytes !=
                 KdaPrepare::Arch22TargetConfig::kSlotBytes) ||
            context_.targetPipeline !=
                (context_.workspaceSlotBytes ==
                 KdaPrepare::Arch22TargetConfig::kSlotBytes)) {
            return false;
        }

        uint64_t chunkWorkItems = context_.totalVarLenChunks;
        if (!context_.isVarLen) {
            if (!CheckedMultiply(context_.batch, context_.chunksPerSequence,
                                 chunkWorkItems)) {
                return false;
            }
        }
        if (chunkWorkItems == 0) {
            return false;
        }
        if (chunkWorkItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        // 先仅按 chunk 分核。chunk 数不足时，才以完整 GVA QK 头组为单位分 head。
        uint64_t headsPerPartition = context_.valueHeadNum;
        uint64_t partitionCount = 1;
        if (chunkWorkItems < context_.aicCoreNum && context_.qkHeadNum > 1) {
            headsPerPartition = context_.valueHeadNum / context_.qkHeadNum;
            partitionCount = context_.qkHeadNum;
        }
        if (context_.targetPipeline) {
            const uint64_t qkGroup =
                context_.valueHeadNum / context_.qkHeadNum;
            uint64_t bestCost = std::numeric_limits<uint64_t>::max();
            for (uint64_t p = 1; p <= context_.qkHeadNum && p <= 4; ++p) {
                if (context_.valueHeadNum % p != 0 ||
                    (context_.valueHeadNum / p) % qkGroup != 0) {
                    continue;
                }
                const uint64_t items = chunkWorkItems * p;
                const uint64_t cores = std::min(items, context_.aicCoreNum);
                const uint64_t cost =
                    ((items + cores - 1) / cores) * (context_.valueHeadNum / p);
                if (cost < bestCost) {
                    bestCost = cost;
                    partitionCount = p;
                }
            }
            headsPerPartition = context_.valueHeadNum / partitionCount;
        }

        uint64_t totalWorkItems = 0;
        if (!CheckedMultiply(chunkWorkItems, partitionCount, totalWorkItems)) {
            return false;
        }
        if (totalWorkItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const uint64_t usedCoreNum = std::min(context_.aicCoreNum, totalWorkItems);
        uint64_t workgroupBytes = 0;
        uint64_t userWorkspaceBytes = 0;
        if (!CheckedMultiply(context_.workspaceSlotBytes,
                             context_.slotsPerWorkgroup,
                             workgroupBytes) ||
            !CheckedMultiply(usedCoreNum, workgroupBytes,
                             userWorkspaceBytes) ||
            context_.libApiWorkspaceBytes >
                std::numeric_limits<uint64_t>::max() - userWorkspaceBytes ||
            usedCoreNum > std::numeric_limits<uint32_t>::max() ||
            headsPerPartition > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        schedule.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
        schedule.headsPerPartition = static_cast<uint32_t>(headsPerPartition);
        schedule.chunkWorkItems = chunkWorkItems;
        schedule.totalWorkItems = totalWorkItems;
        schedule.workspaceBytes = context_.libApiWorkspaceBytes + userWorkspaceBytes;
        return true;
    }

private:
    static bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result)
    {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
            return false;
        }
        result = lhs * rhs;
        return true;
    }

    const ChunkKdaFwdPrepareScheduleContext &context_;
};

} // namespace optiling

#endif // CHUNK_KDA_FWD_PREPARE_TILING_PROCESSOR_H
