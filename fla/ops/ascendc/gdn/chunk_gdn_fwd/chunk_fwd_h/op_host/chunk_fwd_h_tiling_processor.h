/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_h_tiling_processor.h
 * \brief Tiling computation decoupled from gert::TilingContext.
 *
 * The caller is responsible for resolving framework-specific information (shapes, dtypes,
 * platform core number, lib-api workspace size) into the plain context struct below. The
 * processor then fills ChunkFwdHPlainTilingData together with the block
 * dim and the total workspace size, mirroring exactly the original Tiling4ChunkFwdH.
 */

#ifndef CHUNK_FWD_H_TILING_PROCESSOR_H
#define CHUNK_FWD_H_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../op_kernel/chunk_fwd_h_struct.h"

namespace optiling {

static constexpr size_t CHUNK_FWD_H_WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
static constexpr size_t CHUNK_FWD_H_GM_ALIGN = 512;
static constexpr int64_t CHUNK_FWD_H_ROUND_HEAD_SLOTS = 4;
static constexpr int64_t CHUNK_FWD_H_CHUNK_SIZE = 64;
static constexpr int64_t CHUNK_FWD_H_K_DIM = 128;
static constexpr int64_t CHUNK_FWD_H_V_DIM = 128;
// P1 均衡分区的序列数上限：超此值回退均分（大 N 下各核窗口覆盖上百序列，
// 长度按大数定律自然平均化，均衡收益趋零；同时封顶 attr 序列化与 host tiling 延迟）
static constexpr int64_t CHUNK_FWD_H_MAX_SEQ_COUNTS = 65536;

// Plain, framework-agnostic inputs needed to compute the tiling.
struct ChunkFwdHTilingContext {
    // shapes
    int64_t seqlen;        // k.dim(2)
    int64_t kNumHead;      // k.dim(1)
    int64_t kHeadDim;      // k.dim(3)
    int64_t vNumHead;      // u.dim(1)
    int64_t vHeadDim;      // u.dim(3)
    int64_t shapeBatchDim; // k.dim(0)
    // variable length
    bool hasCuSeqlens;
    int64_t cuSeqlensDim0; // length of cu_seqlens (only used when hasCuSeqlens)
    bool useInitialState;
    // attrs
    bool storeFinalState;
    int64_t chunkSize;
    // P1: per-sequence chunk counts read from cu_seqlens on host (aclnn/l0op layer);
    // nullptr / zero count means dense path — kernel keeps uniform head partition.
    const int64_t *seqChunkCounts;
    int64_t seqCount;
    // platform
    uint32_t aicCoreNum;
    size_t libApiWorkSpaceSize;
    // gated 复用路径：gate/state dtype token（gated tiling 字段值：1=bf16, 2=fp32）
    int64_t gatedGateDtype = 1;
    int64_t gatedStateDtype = 2;
    // gated 复用 eligible（与 kernel 侧 FwdHGatedReusable 同条件）：
    // g 模式 + fp32 gate + fp32 state + storeFinal + useExp2 + vfirst=0。
    // eligible 时 blockDim 恒 aicCoreNum（gated 调度器要求全核参与 mode2
    // 配对——收缩 launch 会导致 flag 配对残缺损）。
    bool gatedEligible = false;
    // 链摊满 soc 门控（2026-10-29，用户约束"A2 的修改不能影响到 A5"+白名单裁定）：
    // tiling.cpp 注入 GetCurNpuArch()==DAV_2201（A2/A3 家族白名单）——A5/950 恒
    // false，spread 分支不触发，blockDim 保持原公式（A5 行为不变）。默认 false
    // （保守：未显式注入则不开）。
    bool spreadChainsAllowed = false;
};

class ChunkFwdHTilingProcessor {
public:
    explicit ChunkFwdHTilingProcessor(const ChunkFwdHTilingContext &ctx) : ctx_(ctx) {}

    // Fills the plain tiling struct, the block dim and the total workspace size.
    void Process(::ChunkFwdHPlainTilingData &tiling,
                 uint32_t &blockDim, size_t &workspaceSize) const
    {
        int64_t isVariedLen;
        int64_t shapeBatch;
        int64_t tokenBatch;
        int64_t batch;

        if (!ctx_.hasCuSeqlens) {
            isVariedLen = 0;
            shapeBatch = ctx_.shapeBatchDim;
            tokenBatch = 1;
            batch = shapeBatch;
        } else {
            isVariedLen = 1;
            shapeBatch = 1;
            tokenBatch = ctx_.cuSeqlensDim0 - 1;
            batch = tokenBatch;
        }

        const uint64_t effectiveSequenceCount = static_cast<uint64_t>(batch);
        const uint64_t totalHeadTasks =
            effectiveSequenceCount * static_cast<uint64_t>(ctx_.vNumHead);
        const uint64_t physicalAicCores = static_cast<uint64_t>(ctx_.aicCoreNum);
        const uint64_t headsPerCore =
            (totalHeadTasks + physicalAicCores - 1) / physicalAicCores;
        // ——链摊满试验（2026-10-29，§9.10.10 后续）：少任务形态（任务数 < 2×物理核）
        // 放开 blockDim 到 min(cores, tasks)——pair 核每核 AIC 串行服务 2 链指令流，
        // 若 AIC 段是 chunk 周期主体（画像 AIV vec 仅 52.6% 佐证），单链核把 AIC 段
        // 砍半可缩短链周期。gated 路径不参与（blockDim 恒全核）；dense 不受影响
        // （headsPerCore≥2 时走原公式）。1 任务/核的正确性有 fn_ 小 head 用例先例。
        // soc 门控（A5 隔离，用户约束）：spreadChainsAllowed 恒 false 于 A5/950。
        // 可行性预判前置（守卫回退空核修复）：只有 任务数>核数 的区间 blockDim 才
        // 偏离原公式（≤核数时原公式本就每核 1 链）；该区间若链摊满表写不成（dense
        // 无权重表 / 对齐守卫不满足），抬高的 blockDim 只会带尾部空核（gk 定长实测
        // +2.4%）——预判不过则回原公式。预判与 FillBalancedCoreBounds 内守卫同条件
        // （r=任务-核、s=2核-任务，须 s 为 vNumHead 倍数且 2r≤vNumHead）。
        bool spreadOk = ctx_.spreadChainsAllowed &&
            !ctx_.gatedEligible && totalHeadTasks < 2 * physicalAicCores;
        if (spreadOk && totalHeadTasks > physicalAicCores) {
            const int64_t r = static_cast<int64_t>(totalHeadTasks) -
                static_cast<int64_t>(physicalAicCores);
            const int64_t s = 2 * static_cast<int64_t>(physicalAicCores) -
                static_cast<int64_t>(totalHeadTasks);
            spreadOk = ctx_.seqChunkCounts != nullptr && ctx_.seqCount > 0 &&
                s >= 0 && s % ctx_.vNumHead == 0 && 2 * r <= ctx_.vNumHead;
        }
        if (spreadOk) {
            blockDim = static_cast<uint32_t>(
                totalHeadTasks < physicalAicCores ? totalHeadTasks : physicalAicCores);
        } else {
            blockDim = ctx_.gatedEligible
                ? static_cast<uint32_t>(physicalAicCores)
                : static_cast<uint32_t>((totalHeadTasks + headsPerCore - 1) / headsPerCore);
        }
        const int64_t activeAicCoreNum = static_cast<int64_t>(blockDim);
        constexpr int64_t chunkSize = CHUNK_FWD_H_CHUNK_SIZE;
        constexpr int64_t kHeadDim = CHUNK_FWD_H_K_DIM;
        constexpr int64_t vHeadDim = CHUNK_FWD_H_V_DIM;

        FillBalancedCoreBounds(tiling, blockDim, totalHeadTasks);

        size_t workspaceOffset = ctx_.libApiWorkSpaceSize;
        workspaceOffset += CHUNK_FWD_H_WORKSPACE_RSV_BYTE;

        tiling.vWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        chunkSize * vHeadDim * sizeof(float)));

        tiling.vUpdateWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        chunkSize * vHeadDim * sizeof(uint16_t)));

        tiling.kDecayWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        kHeadDim * vHeadDim * sizeof(float)));

        tiling.hWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        kHeadDim * vHeadDim * sizeof(float)));

        // ===== gated 复用路径 workspace 追加段 =====
        // gated tiling struct 镜像（kernel 现场构造后由 gated kernel 消费；
        // 尺寸按 ChunkGatedDeltaRuleFwdHTilingData 上界 256B）+ numSeq/numChunks
        // 前缀表（int64 × (tokenBatch+1)）——gated scheduler varlen 建表用。
        tiling.gatedTilingOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += 256;
        tiling.gatedNumSeqOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>((tokenBatch + 2) * sizeof(int64_t)));
        tiling.gatedNumChunksOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>((tokenBatch + 2) * sizeof(int64_t)));

        workspaceOffset += CHUNK_FWD_H_WORKSPACE_RSV_BYTE;
        workspaceSize = workspaceOffset;

        tiling.gatedGateDataType = ctx_.gatedGateDtype;
        tiling.gatedStateDataType = ctx_.gatedStateDtype;

        tiling.batch = batch;
        tiling.seqlen = ctx_.seqlen;
        tiling.kNumHead = ctx_.kNumHead;
        tiling.vNumHead = ctx_.vNumHead;
        tiling.kHeadDim = ctx_.kHeadDim;
        tiling.vHeadDim = ctx_.vHeadDim;
        tiling.chunkSize = chunkSize;
        tiling.useInitialState = ctx_.useInitialState;
        tiling.storeFinalState = ctx_.storeFinalState;
        tiling.isVariedLen = isVariedLen;
        tiling.shapeBatch = shapeBatch;
        tiling.tokenBatch = tokenBatch;
    }

private:
    // P1 varlen balanced partition: cut the seq-major head-task sequence into
    // blockDim contiguous spans of near-equal total chunk weight (capacity
    // binary search + greedy fill). blockDim itself is unchanged, so workspace
    // size and core launch count are untouched — only per-core [begin, end)
    // move. Falls back to the uniform partition (coreBoundCount=0) when the
    // sequence weights are unavailable or degenerate (dense / fewer tasks
    // than cores / greedy produced fewer spans than cores).
    //
    // ——A 轮 pair 原子化（2026-10-29）：切分最小单元改为 (seq, head-pair)——
    // 单任务粒度的贪心在"少任务 × 长序列"形态（case32 类：每核 ~≤2 任务）会跨
    // 序列切 span，核内 round 配到跨序列 head 对，破坏 MIX_AIC_1_2 双 AIV 并行
    // 前提（pair slot 需同序列 2 head 各扛一链，画像铁证：vector1 空转 +12.9%）。
    // pair 原子化后 span 边界恒对齐 pair 边界——核内连续 2 任务恒同序列，每
    // round 配对自然成立。奇 vNumHead 的末 head 退化为单任务单元（该 round 单
    // AIV，仅奇 head 形态受影响）。大任务数形态（case2 类）均衡粒度 2 倍化，
    // 段长大数定律下均衡收益不变。GreedySpanCount 只看权重序列，签名不变。
    void FillBalancedCoreBounds(::ChunkFwdHPlainTilingData &tiling,
                                uint32_t blockDim, uint64_t totalHeadTasks) const
    {
        tiling.coreBoundCount = 0;
        for (int64_t &bound : tiling.coreTaskBounds) {
            bound = 0;
        }
        if (ctx_.seqChunkCounts == nullptr || ctx_.seqCount <= 0 ||
            ctx_.seqCount > CHUNK_FWD_H_MAX_SEQ_COUNTS ||
            blockDim == 0 || totalHeadTasks < blockDim ||
            static_cast<int64_t>(blockDim) > CHUNK_FWD_H_MAX_CORE_BOUNDS) {
            return;
        }
        struct Unit {
            int64_t weight;
            uint64_t taskBegin;
        };
        std::vector<Unit> units;
        std::vector<int64_t> unitWeights;
        units.reserve(totalHeadTasks / 2 + static_cast<uint64_t>(ctx_.seqCount));
        uint64_t taskCursor = 0;
        for (int64_t seq = 0; seq < ctx_.seqCount; ++seq) {
            const int64_t chunks = ctx_.seqChunkCounts[seq];
            if (chunks <= 0) {
                return; // malformed host table — fall back to uniform partition
            }
            for (int64_t head = 0; head < ctx_.vNumHead; head += 2) {
                const int64_t pairHeads = (head + 1 < ctx_.vNumHead) ? 2 : 1;
                units.push_back({pairHeads * chunks, taskCursor});
                unitWeights.push_back(pairHeads * chunks);
                taskCursor += static_cast<uint64_t>(pairHeads);
            }
        }
        if (taskCursor != totalHeadTasks || units.empty()) {
            return;
        }
        // ——链摊满分配（核多于 pair 数时，与 blockDim 放开分支配套）——
        // r = 双任务核数（尾段），s = 单任务核数：s + r = blockDim 且 s + 2r = total。
        // 双核区间 [s, s+2r) 须整段在一个序列内（连续 2 任务同序列 pair）——对齐守卫：
        // s 为 vNumHead 倍数且 2r ≤ vNumHead；不满足则回退 pair 贪心（安全）。
        // 尾段 = seq-major 最后序列区间，case32 恰为最短序列（双核扛轻链最优）。
        if (static_cast<int64_t>(blockDim) > static_cast<int64_t>(units.size())) {
            const int64_t totalT = static_cast<int64_t>(totalHeadTasks);
            const int64_t r = totalT - static_cast<int64_t>(blockDim);
            const int64_t s = 2 * static_cast<int64_t>(blockDim) - totalT;
            if (r > 0 && s >= 0 && s % ctx_.vNumHead == 0 && 2 * r <= ctx_.vNumHead) {
                for (int64_t core = 0; core < static_cast<int64_t>(blockDim); ++core) {
                    const int64_t end = core < s ? (core + 1) : (s + 2 * (core - s + 1));
                    tiling.coreTaskBounds[static_cast<size_t>(core)] = end;
                }
                tiling.coreBoundCount = static_cast<int64_t>(blockDim);
            }
            return; // 对齐不满足（跨界风险）→ 无表 = kernel 均分回退，安全
        }
        int64_t maxWeight = 0;
        int64_t totalWeight = 0;
        for (const int64_t w : unitWeights) {
            maxWeight = w > maxWeight ? w : maxWeight;
            totalWeight += w;
        }
        // Binary search the minimal capacity whose greedy span count <= blockDim.
        int64_t lo = maxWeight;
        int64_t hi = totalWeight;
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (GreedySpanCount(unitWeights, mid) <= blockDim) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        // Greedy cut at the minimal capacity; force exactly blockDim spans by
        // opening a new span whenever the remaining units equal the remaining
        // spans to place (keeps every span non-empty). Span bounds are unit
        // boundaries — a pair never straddles two cores.
        const int64_t capacity = lo;
        std::vector<int64_t> bounds(static_cast<size_t>(blockDim), 0);
        int64_t spanCount = 1;
        int64_t sum = 0;
        for (size_t u = 0; u < units.size(); ++u) {
            const uint64_t remainingUnits = units.size() - u;
            const int64_t remainingSpans = static_cast<int64_t>(blockDim) - spanCount;
            const bool mustSplit = sum + units[u].weight > capacity ||
                remainingUnits <= static_cast<uint64_t>(remainingSpans);
            if (mustSplit && spanCount < static_cast<int64_t>(blockDim)) {
                bounds[static_cast<size_t>(spanCount - 1)] = static_cast<int64_t>(units[u].taskBegin);
                spanCount += 1;
                sum = 0;
            }
            sum += units[u].weight;
        }
        bounds[static_cast<size_t>(spanCount - 1)] = static_cast<int64_t>(totalHeadTasks);
        if (spanCount != static_cast<int64_t>(blockDim)) {
            return; // degenerate — uniform partition is equivalent here
        }
        for (uint32_t core = 0; core < blockDim; ++core) {
            tiling.coreTaskBounds[core] = bounds[core];
        }
        tiling.coreBoundCount = static_cast<int64_t>(blockDim);
    }

    static int64_t GreedySpanCount(const std::vector<int64_t> &weights, int64_t capacity)
    {
        int64_t spanCount = 1;
        int64_t sum = 0;
        for (const int64_t w : weights) {
            if (sum + w > capacity) {
                spanCount += 1;
                sum = 0;
            }
            sum += w;
        }
        return spanCount;
    }

    // Mirrors the original "(x + GM_ALIGN) / GM_ALIGN * GM_ALIGN" alignment.
    static size_t AlignUp(size_t x)
    {
        return (x + CHUNK_FWD_H_GM_ALIGN - 1) / CHUNK_FWD_H_GM_ALIGN * CHUNK_FWD_H_GM_ALIGN;
    }

    const ChunkFwdHTilingContext &ctx_;
};

} // namespace optiling

#endif // CHUNK_FWD_H_TILING_PROCESSOR_H
