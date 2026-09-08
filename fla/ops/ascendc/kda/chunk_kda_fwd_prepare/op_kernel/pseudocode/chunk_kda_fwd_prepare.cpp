/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_kda_fwd_prepare_tiling_key.h"
#include "chunk_kda_fwd_prepare_struct.h"
#include "chunk_kda_fwd_prepare_utils.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_fwd_prepare_cube.h"
#include "arch35/chunk_kda_fwd_prepare_vec.h"
#else
#include "arch22/chunk_kda_fwd_prepare_cube.h"
#include "arch22/chunk_kda_fwd_prepare_vec.h"
#endif

#ifndef KDA_PREPARE_PSEUDOCODE_ENTRY_NAME
#define KDA_PREPARE_PSEUDOCODE_ENTRY_NAME RunChunkKdaFwdPreparePseudocode
#endif

namespace kda_prepare_pseudocode {

using ResolveChunk = ChunkTask (*)(std::uint32_t chunkOrdinal);

namespace {

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
constexpr Architecture kCompileTimeArchitecture = Architecture::Arch35;
#else
constexpr Architecture kCompileTimeArchitecture = Architecture::Arch22;
#endif

bool BindWorkspaceView(const WorkspaceSizing &sizing,
                       WorkspaceView &workspace) noexcept
{
    if (!sizing.valid || workspace.backingBytes < sizing.totalBytes) {
        return false;
    }
    workspace.workgroupBase = sizing.workgroupBase;
    workspace.contextOffsetBytes = 0U;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    workspace.slotStrideBytes = WorkspacePolicy::kSlotStride;
    workspace.contextBytes = WorkspacePolicy::kStagePayload.offset;
    workspace.payloadOffsetBytes = WorkspacePolicy::kStagePayload.offset;
    workspace.payloadBytes = WorkspacePolicy::kStagePayload.size;
#else
    workspace.slotStrideBytes =
        arch22_policy::WorkspacePolicy::kSlotStride;
    workspace.contextBytes =
        arch22_policy::WorkspacePolicy::kStagePayload.offset;
    workspace.payloadOffsetBytes =
        arch22_policy::WorkspacePolicy::kStagePayload.offset;
    workspace.payloadBytes =
        arch22_policy::WorkspacePolicy::kStagePayload.size;
#endif
    return true;
}

ChunkTask ResolveDenseChunk(std::uint32_t chunkOrdinal) noexcept
{
    return {0, chunkOrdinal, chunkOrdinal, kChunkRows};
}

class ChunkKdaFwdPrepareKernel {
public:
    bool Init(const RuntimeTiling &tiling, std::uint32_t workgroupId,
              CoreRole role, std::uint32_t aivId, WorkspaceView &workspace,
              SyncLedger &sync, VectorOps &vectorOps, CubeOps &cubeOps,
              ResolveChunk resolveChunk) noexcept
    {
        if (tiling.aicWorkgroupCount == 0U || tiling.totalChunks == 0U ||
            !IsValidHeadMapping(tiling.headCount, tiling.qkHeadCount) ||
            workgroupId >= tiling.aicWorkgroupCount) {
            return false;
        }
        const WorkspaceSizing sizing = CheckedWorkspaceSizing(
            kCompileTimeArchitecture, tiling.aicWorkgroupCount, workgroupId);
        if (!BindWorkspaceView(sizing, workspace)) {
            return false;
        }

        tiling_ = &tiling;
        workspace_ = &workspace;
        sync_ = &sync;
        vectorOps_ = &vectorOps;
        cubeOps_ = &cubeOps;
        plan_ = BuildCorePlan(tiling, workgroupId,
                              kCompileTimeArchitecture);
        resolveChunk_ = resolveChunk == nullptr ? ResolveDenseChunk
                                                : resolveChunk;
        workgroupId_ = workgroupId;
        role_ = role;
        aivId_ = aivId;
        return true;
    }

    void Process() noexcept
    {
        // 与真实 MIX 核一致，每颗物理核只进入一种角色的 Process；角色判断
        // 不进入分块/头主循环。
        if (role_ == CoreRole::Aiv) {
            ProcessAiv();
        } else if (role_ == CoreRole::Aic) {
            ProcessAic();
        }
    }

private:
    void ProcessAiv() noexcept
    {
        if (aivId_ >= kAivPerWorkgroup) {
            return;
        }
        OwnerTicketState ownerTickets{};
        for (std::uint64_t ordinal = plan_.begin; ordinal < plan_.end;
             ++ordinal) {
            std::uint32_t chunkOrdinal = 0U;
            std::uint32_t partitionOrdinal = 0U;
            DecodeChunkHeadPartitionOrdinal(
                plan_, ordinal, chunkOrdinal, partitionOrdinal);
            const ChunkTask chunk = resolveChunk_(chunkOrdinal);
            // 主机分块配置必须忽略空序列，并拒绝不在 [1,64] 范围内的描述符。
            // 符号核同样不让非法描述符进入 V6 的 validRows - 1 计算。
            if (chunk.validRows == 0U || chunk.validRows > kChunkRows) {
                continue;
            }

            // 分块优先时遍历该分块的全部 HK 同源头分区；只有分块数量
            // 不足时，才使用展平任务中携带的单个头分区。
            const std::uint32_t partitionBegin =
                plan_.mode == PartitionMode::ChunkOnly ? 0U
                                                       : partitionOrdinal;
            const std::uint32_t partitionEnd =
                plan_.mode == PartitionMode::ChunkOnly
                    ? plan_.headPartitionCount
                    : partitionOrdinal + 1U;
            for (std::uint32_t partition = partitionBegin;
                 partition < partitionEnd; ++partition) {
                const std::uint32_t headBegin = PartitionValueHeadBegin(
                    partition, tiling_->headCount, tiling_->qkHeadCount);
                const std::uint32_t headEnd = PartitionValueHeadEnd(
                    partition, tiling_->headCount, tiling_->qkHeadCount);
                std::uint32_t groupOrdinal =
                    partition * HeadGroupsPerPartition(
                                    tiling_->headCount,
                                    tiling_->qkHeadCount);

                // 每组最多四个 HV，且边界不切开同一 HK 的共享 Q/K 头组。
                for (std::uint32_t groupBegin = headBegin;
                     groupBegin < headEnd;
                     groupBegin += kHeadsPerGroup, ++groupOrdinal) {
                    const WorkItem item = BuildWorkItemRange(
                        plan_, ordinal, chunk, groupOrdinal, groupBegin,
                        headEnd, tiling_->headCount, ownerTickets,
                        tiling_->qkHeadCount);
                    if (item.group.activeHeads == 0U) {
                        continue;
                    }
                    VectorStageArgs args = {
                        &item,
                        workspace_,
                        sync_,
                        vectorOps_,
                        tiling_->key,
                        tiling_->epsilon,
                        tiling_->lowerBound,
                        tiling_->scale,
                        tiling_->hasDtBias,
                        workgroupId_,
                        aivId_,
                        kAllGroupLocalHeads};

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    // 每个调用就是一个物理向量阶段，函数名同时给出该次
                    // VF 的结果。V3/V6 在阶段内等待 AIC，源码相邻不表示
                    // 两类核之间存在隐式串行关系。
                    arch35::StageV0_AivNormalizeQkGateCumsumBeta(args);
                    arch35::StageV1_AivBuildS4ScoreOperands(args);
                    arch35::StageV3_AivBuildAqkAndAkkFactors(args);
                    arch35::StageV6_AivBuildPostWuOperands(args);
#else
                    const std::uint32_t pairWaves =
                        static_cast<std::uint32_t>(CeilDiv(
                            item.group.activeHeads, kAivPerWorkgroup));
                    // Arch22 只有一块 40 KiB 共享区，因此每个头配对必须
                    // 先完成 V0 -> V1，才能切换到下一块 72 KiB 私有区。
                    for (std::uint32_t pair = 0U; pair < pairWaves; ++pair) {
                        args.selectedGroupLocalHead =
                            pair * kAivPerWorkgroup + aivId_;
                        arch22::StageV0_AivNormalizeQkGateCumsumBeta(args);
                        arch22::StageV1_AivBuildS4ScoreOperands(args);
                    }
                    for (std::uint32_t pair = 0U; pair < pairWaves; ++pair) {
                        args.selectedGroupLocalHead =
                            pair * kAivPerWorkgroup + aivId_;
                        arch22::StageV3_AivBuildAqkAndAkkFactors(args);
                    }
                    for (std::uint32_t pair = 0U; pair < pairWaves; ++pair) {
                        args.selectedGroupLocalHead =
                            pair * kAivPerWorkgroup + aivId_;
                        arch22::StageV6_AivBuildPostWuOperands(args);
                    }
#endif
                }
            }
        }
    }

    void ProcessAic() noexcept
    {
        OwnerTicketState ownerTickets{};
        for (std::uint64_t ordinal = plan_.begin; ordinal < plan_.end;
             ++ordinal) {
            std::uint32_t chunkOrdinal = 0U;
            std::uint32_t partitionOrdinal = 0U;
            DecodeChunkHeadPartitionOrdinal(
                plan_, ordinal, chunkOrdinal, partitionOrdinal);
            const ChunkTask chunk = resolveChunk_(chunkOrdinal);
            if (chunk.validRows == 0U || chunk.validRows > kChunkRows) {
                continue;
            }

            const std::uint32_t partitionBegin =
                plan_.mode == PartitionMode::ChunkOnly ? 0U
                                                       : partitionOrdinal;
            const std::uint32_t partitionEnd =
                plan_.mode == PartitionMode::ChunkOnly
                    ? plan_.headPartitionCount
                    : partitionOrdinal + 1U;
            for (std::uint32_t partition = partitionBegin;
                 partition < partitionEnd; ++partition) {
                const std::uint32_t headBegin = PartitionValueHeadBegin(
                    partition, tiling_->headCount, tiling_->qkHeadCount);
                const std::uint32_t headEnd = PartitionValueHeadEnd(
                    partition, tiling_->headCount, tiling_->qkHeadCount);
                std::uint32_t groupOrdinal =
                    partition * HeadGroupsPerPartition(
                                    tiling_->headCount,
                                    tiling_->qkHeadCount);

                for (std::uint32_t groupBegin = headBegin;
                     groupBegin < headEnd;
                     groupBegin += kHeadsPerGroup, ++groupOrdinal) {
                    const WorkItem item = BuildWorkItemRange(
                        plan_, ordinal, chunk, groupOrdinal, groupBegin,
                        headEnd, tiling_->headCount, ownerTickets,
                        tiling_->qkHeadCount);
                    if (item.group.activeHeads == 0U) {
                        continue;
                    }
                    CubeStageArgs args = {
                        &item,          workspace_,       sync_,
                        cubeOps_,       tiling_->key,      tiling_->epsilon,
                        tiling_->lowerBound,
                        tiling_->scale, workgroupId_};

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    // C2 现场展示 S=4 行堆叠分数 MMAD；后续三段函数名
                    // 直接给出 T、Akk_q10 以及 W/U 的矩阵公式。
                    arch35::StageC2_AicComputeRawAqkAkk(args);
                    arch35::StageC4_AicComputeTEqualsBMatmulX0(args);
                    arch35::StageC5_AicComputeAkkQ10EqualsNegX1MatmulT(
                        args);
                    arch35::StageC7_AicComputeWAndU(args);
#else
                    arch22::StageC2_AicComputeRawAqkAkk(args);
                    arch22::StageC4_AicComputeTEqualsBMatmulX0(args);
                    arch22::StageC5_AicComputeAkkQ10EqualsNegX1MatmulT(
                        args);
                    arch22::StageC7_AicComputeWAndU(args);
#endif
                }
            }
        }
    }

    const RuntimeTiling *tiling_ = nullptr;
    WorkspaceView *workspace_ = nullptr;
    SyncLedger *sync_ = nullptr;
    VectorOps *vectorOps_ = nullptr;
    CubeOps *cubeOps_ = nullptr;
    ResolveChunk resolveChunk_ = ResolveDenseChunk;
    CorePlan plan_{};
    std::uint32_t workgroupId_ = 0U;
    CoreRole role_ = CoreRole::Aic;
    std::uint32_t aivId_ = 0U;
};

} // namespace

// 仅为待实现的控制入口。此处刻意不实现为 __global__ 核函数，也不声明 GM ABI、
// TilingKey 注册、任务混合宏或设备 API；入口与参考
// chunk_gated_delta_rule_fwd_prepare 核函数一样只保留 Init/Process，任务遍历和
// 八个 Stage 的顺序由 Kernel 类统一表达。
void KDA_PREPARE_PSEUDOCODE_ENTRY_NAME(
    const RuntimeTiling &tiling, std::uint32_t workgroupId, CoreRole role,
    std::uint32_t aivId, WorkspaceView &workspace, SyncLedger &sync,
    VectorOps &vectorOps, CubeOps &cubeOps,
    ResolveChunk resolveChunk) noexcept
{
    ChunkKdaFwdPrepareKernel kernel;
    if (!kernel.Init(tiling, workgroupId, role, aivId, workspace, sync,
                     vectorOps, cubeOps, resolveChunk)) {
        return;
    }
    kernel.Process();
}

} // namespace kda_prepare_pseudocode
