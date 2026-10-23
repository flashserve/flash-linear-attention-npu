/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH22_TARGET_CHUNK_KDA_FWD_PREPARE_CUBE_H
#define ARCH22_TARGET_CHUNK_KDA_FWD_PREPARE_CUBE_H

#include <cstdint>
#include "kernel_operator.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_utils.h"
#include "chunk_kda_fwd_prepare_target_common.h"

namespace KdaPrepare::Arch22Target {

constexpr AscendC::FixpipeConfig kFixpipeNz = {AscendC::CO2Layout::NZ, false};


// Tensor primitives receive explicit operands; stages own their event lifecycle.
template <typename CompilePolicy>
__aicore__ inline uint32_t ActiveSubChunks(uint32_t validRows);

template <typename CompilePolicy>
__aicore__ inline void LoadFp32Nz32(AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src,
                                    uint32_t srcPitch = 32);

template <typename CompilePolicy>
__aicore__ inline void LoadL0A32(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src);

template <typename CompilePolicy>
__aicore__ inline void LoadL0B32(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src);

template <typename CompilePolicy>
__aicore__ inline void FixpipeFp32NzRaw(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> l0C,
                                        uint32_t slot);

template <typename CompilePolicy>
__aicore__ inline void FixpipeAkkRowMajorRaw(AscendC::GlobalTensor<bfloat16_t> dst, AscendC::LocalTensor<float> l0C,
                                             uint32_t rows, uint32_t slot);

template <typename CompilePolicy>
__aicore__ inline uint32_t ActiveSubChunks(uint32_t validRows)
{
    const uint32_t count = CeilDiv(validRows, Shape::kSubChunkRows);
    return count > Shape::kSubChunkCount ? Shape::kSubChunkCount : count;
}

template <typename CompilePolicy>
__aicore__ inline void LoadFp32Nz32(AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src,
                                    uint32_t srcPitch)
{
    AscendC::Nd2NzParams copy{};
    copy.ndNum = 1;
    copy.nValue = 32;
    copy.dValue = 32;
    copy.srcDValue = static_cast<uint16_t>(srcPitch);
    copy.srcNdMatrixStride = 0;
    copy.dstNzNStride = 1;
    copy.dstNzC0Stride = 32;
    copy.dstNzMatrixStride = 0;
    AscendC::DataCopy(dst, src, copy);
}

template <typename CompilePolicy>
__aicore__ inline void LoadL0A32(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src)
{
    constexpr uint32_t kFp32FractalElements = 16 * 8;
    constexpr uint32_t kColumnFractals = 4;
    AscendC::LoadData2DParams load{};
    load.startIndex = 0;
    load.repeatTimes = kColumnFractals;
    load.srcStride = 2;
    load.dstGap = 0;
    load.ifTranspose = false;
    load.sid = 0;
    load.addrMode = 0;
    for (uint32_t rowFractal = 0; rowFractal < 2; ++rowFractal) {
        AscendC::LoadData(dst[rowFractal * kColumnFractals * kFp32FractalElements],
                          src[rowFractal * kFp32FractalElements], load);
    }
}

template <typename CompilePolicy>
__aicore__ inline void LoadL0B32(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src)
{
    AscendC::LoadData3DParamsV2<float> load{};
    load.l1H = 1;
    load.l1W = 32;
    load.channelSize = 32;
    load.kExtension = 32;
    load.mExtension = 32;
    load.kStartPt = 0;
    load.mStartPt = 0;
    load.strideW = 1;
    load.strideH = 1;
    load.filterW = 1;
    load.filterH = 1;
    load.dilationFilterW = 1;
    load.dilationFilterH = 1;
    load.enTranspose = true;
    load.enSmallK = false;
    load.filterSizeW = false;
    load.filterSizeH = false;
    load.fMatrixCtrl = false;
    AscendC::LoadData(dst, src, load);
}

template <typename CompilePolicy>
__aicore__ inline void FixpipeFp32NzRaw(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> l0C,
                                        uint32_t slot)
{
    auto fix = AscendC::FixpipeParamsV220(32, 32, 32, 32, false);
    fix.quantPre = QuantMode_t::NoQuant;
    fix.isChannelSplit = true;
    AscendC::Fixpipe<float, float, kFixpipeNz>(dst, l0C[slot * 1024], fix);
}

template <typename CompilePolicy>
__aicore__ inline void FixpipeAkkRowMajorRaw(AscendC::GlobalTensor<bfloat16_t> dst, AscendC::LocalTensor<float> l0C,
                                             uint32_t rows, uint32_t slot)
{
    auto fix = AscendC::FixpipeParamsV220(32, rows, 32, Shape::kChunkRows, false);
    fix.quantPre = QuantMode_t::F322BF16;
    AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(dst, l0C[slot * 1024], fix);
}

template <typename CompilePolicy>
struct TargetCubeContext {
    static_assert(CompilePolicy::arch22CubeLeaf, "The target pipeline requires the dedicated leaf policy");
    static constexpr uint32_t kSlotStride =
        CompilePolicy::arch22CubeLeaf ? Workspace::kArch22TargetSlotStride : Workspace::kArch22BaseSlotStride;
    static constexpr uint32_t kWorkgroupStride = Workspace::kArch22SlotCount * kSlotStride;

    PrepareKernelArgs args_{};
    AscendC::TPipe *pipe_ = nullptr;
    uint32_t workgroup_ = 0;
    uint32_t coreCount_ = 0;
    AscendC::TBuf<AscendC::TPosition::A1> l1Buf_{};
    AscendC::TBuf<AscendC::TPosition::A2> l0ABuf_{};
    AscendC::TBuf<AscendC::TPosition::B2> l0BBuf_{};
    AscendC::TBuf<AscendC::TPosition::CO1> l0CBuf_{};
    AscendC::TEventID mte2ToMte1_{};
    AscendC::TEventID mte1ToM_{};
    AscendC::TEventID mToMte1_{};
    AscendC::TEventID mToFix_{};
    AscendC::TEventID fixToM_{};
    AscendC::TEventID c3MToFix_[2]{};
    AscendC::TEventID c3FixToM_[2]{};
    AscendC::TEventID c3MToMte1_[2]{};
    AscendC::TEventID c3Mte1ToM_[2]{};
    AscendC::TEventID fixToMte2_[Shape::kHeadsPerGroup]{};
    AscendC::TEventID tReady_[Shape::kHeadsPerGroup]{};
    AscendC::TEventID fixToMte1_[Shape::kHeadsPerGroup]{};
    AscendC::GlobalTensor<bfloat16_t> wGm_{};
    AscendC::GlobalTensor<bfloat16_t> uGm_{};
};

template <typename CompilePolicy>
__aicore__ inline void InitTargetCube(TargetCubeContext<CompilePolicy> &ctx, const PrepareKernelArgs &args,
                                      AscendC::TPipe *pipe);

template <typename CompilePolicy>
__aicore__ inline void ProcessTargetCube(TargetCubeContext<CompilePolicy> &ctx);

template <typename CompilePolicy>
__aicore__ inline bool FirstGroup(TargetCubeContext<CompilePolicy> &ctx, uint32_t workBegin, uint32_t workEnd,
                                  ChunkRange &chunk, uint32_t &groupBegin, uint32_t &activeHeads);

template <typename CompilePolicy>
__aicore__ inline bool NextGroup(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &currentChunk,
                                 uint32_t work, uint32_t groupBegin,
                                 uint32_t activeHeads, uint32_t headEnd, uint32_t workEnd, ChunkRange &chunk,
                                 uint32_t &nextGroup, uint32_t &nextActive);

template <typename CompilePolicy>
__aicore__ inline void StageNormSums(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk,
                                     uint32_t groupBegin, uint32_t activeHeads);

template <typename CompilePolicy>
__aicore__ inline void Mmad32(TargetCubeContext<CompilePolicy> &ctx, AscendC::LocalTensor<float> l0A,
                              AscendC::LocalTensor<float> l0B, AscendC::LocalTensor<float> l0C,
                              AscendC::LocalTensor<float> srcA, AscendC::LocalTensor<float> srcB, uint32_t slot,
                              bool accumulate, bool last);

template <typename CompilePolicy>
__aicore__ inline void FixpipeFp32Nz(TargetCubeContext<CompilePolicy> &ctx, AscendC::GlobalTensor<float> dst,
                                     AscendC::LocalTensor<float> l0C, uint32_t slot);

template <typename CompilePolicy>
__aicore__ inline void StageC3(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t groupBegin,
                               uint32_t activeHeads);

template <typename CompilePolicy>
__aicore__ inline void StageC2(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t localHead);

template <typename CompilePolicy>
__aicore__ inline void StageC4(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t groupBegin,
                               uint32_t localHead);

template <typename CompilePolicy>
__aicore__ inline void StageC5(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t valueHead,
                               uint32_t localHead);

template <typename CompilePolicy>
__aicore__ inline void StageC7(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t groupBegin,
                               uint32_t activeHeads, uint32_t pair, uint16_t freeFlagId);

template <typename CompilePolicy>
__aicore__ inline void ReleaseEvents(TargetCubeContext<CompilePolicy> &ctx);

template <typename CompilePolicy>
__aicore__ inline void InitTargetCube(TargetCubeContext<CompilePolicy> &ctx, const PrepareKernelArgs &args,
                                      AscendC::TPipe *pipe)
{
    ctx.args_ = args;
    ctx.pipe_ = pipe;
    ctx.workgroup_ = WorkgroupId();
    ctx.coreCount_ = ctx.args_.tiling.usedCoreNum;
    if (ctx.coreCount_ == 0) {
        return;
    }
    ctx.wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.w));
    ctx.uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.u));
    ctx.pipe_->InitBuffer(ctx.l1Buf_, L1::kPeak);
    ctx.pipe_->InitBuffer(ctx.l0ABuf_, 0x10000);
    ctx.pipe_->InitBuffer(ctx.l0BBuf_, 0x10000);
    ctx.pipe_->InitBuffer(ctx.l0CBuf_, 0x10000);

    // 每种 HardEvent 有独立 ID 池。保留 Alloc/Release，并把 CANN 9.1
    // 分配器的预期返回值直接列在申请现场，便于逐项核对。
    ctx.mte2ToMte1_ = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE2_MTE1>(); // ID 0
    ctx.mte1ToM_ = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE1_M>();       // ID 0
    // AIC 的 TPipe::Init 已在 M_MTE1 池占用 0、1、2。
    ctx.mToMte1_ = ctx.pipe_->template AllocEventID<AscendC::HardEvent::M_MTE1>();        // ID 3
    ctx.mToFix_ = ctx.pipe_->template AllocEventID<AscendC::HardEvent::M_FIX>();          // ID 0
    ctx.fixToM_ = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_M>();          // ID 0
    ctx.fixToMte2_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_MTE2>(); // ID 0
    ctx.fixToMte2_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_MTE2>(); // ID 1
    ctx.fixToMte2_[2] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_MTE2>(); // ID 2
    ctx.fixToMte2_[3] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_MTE2>(); // ID 3
    ctx.tReady_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE2_MTE1>();   // ID 1
    ctx.tReady_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE2_MTE1>();   // ID 2
    ctx.tReady_[2] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE2_MTE1>();   // ID 3
    ctx.tReady_[3] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE2_MTE1>();   // ID 4
    ctx.fixToMte1_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_MTE1>(); // ID 0
    ctx.fixToMte1_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_MTE1>(); // ID 1
    ctx.fixToMte1_[2] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_MTE1>(); // ID 2
    ctx.fixToMte1_[3] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_MTE1>(); // ID 3
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
    if constexpr (CompilePolicy::arch22CubeLeaf) {
        ctx.c3MToFix_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::M_FIX>();
        ctx.c3MToFix_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::M_FIX>();
        ctx.c3FixToM_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_M>();
        ctx.c3FixToM_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::FIX_M>();
        ctx.c3MToMte1_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::M_MTE1>();
        ctx.c3MToMte1_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::M_MTE1>();
        ctx.c3Mte1ToM_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE1_M>();
        ctx.c3Mte1ToM_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE1_M>();
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[0]);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[1]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[0]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[1]);
    }
}

template <typename CompilePolicy>
__aicore__ inline void ProcessTargetCube(TargetCubeContext<CompilePolicy> &ctx)
{
    if (ctx.coreCount_ == 0) {
        return;
    }
    // mode 0x2 的核间 flag 是固定物理编号，不经过 EventID 分配器。
    // pair0/pair1: ready=0/1，free=2/3。
    constexpr uint16_t kReadyFlagId[2] = {0, 1};
    constexpr uint16_t kFreeFlagId[2] = {2, 3};
    const uint32_t total = TargetTotalWorkItems(ctx.args_.tiling);
    const uint32_t workBegin = WorkBegin(total, ctx.workgroup_, ctx.coreCount_);
    const uint32_t workEnd = WorkEnd(total, ctx.workgroup_, ctx.coreCount_);
    if (ctx.workgroup_ >= ctx.coreCount_ || workBegin >= workEnd) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
        if constexpr (CompilePolicy::arch22CubeLeaf) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[0]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[1]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[0]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[1]);
        }
        ReleaseEvents<CompilePolicy>(ctx);
        return;
    }

    // V0 的 L2 norm 行平方和由 AIC 预算：进入循环前先算第一个 group，
    // 之后每个 group 在 C2 的等待窗口里算下一个 group，AIV 侧不再做
    // 平方/折半/归约三遍。必须放在初始 free 旗标之前发布，否则 AIV 的
    // 第一个 V0 会读到未写入的 relay。
    if constexpr (CompilePolicy::normMode == QkNormMode::L2) {
        ChunkRange firstChunk{};
        uint32_t firstGroup = 0;
        uint32_t firstActive = 0;
        if (FirstGroup<CompilePolicy>(ctx, workBegin, workEnd, firstChunk, firstGroup, firstActive)) {
            StageNormSums<CompilePolicy>(ctx, firstChunk, firstGroup, firstActive);
        }
    }

    // 两个 pair 的初始许可由 AIC 发布；后续许可由 C7 写回完成后发布。
    for (uint32_t pair = 0; pair < 2; ++pair) {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFreeFlagId[pair]);
    }

    for (uint32_t work = workBegin; work < workEnd; ++work) {
        uint32_t globalChunk = 0;
        uint32_t headPartition = 0;
        TargetDecodeWorkItem(ctx.args_.tiling, work, globalChunk, headPartition);
        ChunkRange chunk{};
        if (!ResolveChunk(ctx.args_, globalChunk, chunk)) {
            continue;
        }
        uint32_t headBegin = 0;
        uint32_t headEnd = 0;
        TargetHeadRange(ctx.args_.tiling, headPartition, headBegin, headEnd);
        for (uint32_t groupBegin = headBegin; groupBegin < headEnd;) {
            uint32_t activeHeads = headEnd - groupBegin;
            if (activeHeads > Shape::kHeadsPerGroup) {
                activeHeads = Shape::kHeadsPerGroup;
            }
            // C2：一次 pair wait 汇聚两个 AIV，再消费该 pair 的两个 local head。
            for (uint32_t pair = 0; pair < 2; ++pair) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kReadyFlagId[pair]);
                for (uint32_t headInPair = 0; headInPair < 2; ++headInPair) {
                    const uint32_t localHead = pair * 2 + headInPair;
                    if (localHead < activeHeads) {
                        StageC2<CompilePolicy>(ctx, chunk, localHead);
                    }
                }
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFreeFlagId[pair]);
            }

            // C4 会把同一 L1 head lane 从 score 输入换义为 B；先确认
            // C2 的 MTE2 搬运全部结束，避免两批异步写访问重叠。
            AscendC::PipeBarrier<PIPE_MTE2>();

            if constexpr (CompilePolicy::normMode == QkNormMode::L2) {
                // 时序：本组的 relay 已在 V0 读完（C2 等到了 V0+V1 的
                // ready），下一组的 relay 必须在 V0(下一组) 之前写完。
                // 放在这里正好用 C2 之后、AIV 跑 V3 的空转窗口。
                ChunkRange nextChunk{};
                uint32_t nextGroup = 0;
                uint32_t nextActive = 0;
                if (NextGroup<CompilePolicy>(ctx, chunk, work, groupBegin, activeHeads, headEnd, workEnd, nextChunk, nextGroup,
                                             nextActive)) {
                    StageNormSums<CompilePolicy>(ctx, nextChunk, nextGroup, nextActive);
                }
            }

            if constexpr (CompilePolicy::arch22CubeLeaf) {
                // C3 consumes all four heads together, so gather both V3
                // pairs before releasing their AIV slots.
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kReadyFlagId[pair]);
                }
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(kFreeFlagId[pair]);
                }
                StageC3<CompilePolicy>(ctx, chunk, groupBegin, activeHeads);
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    for (uint32_t headInPair = 0; headInPair < 2; ++headInPair) {
                        const uint32_t localHead = pair * 2 + headInPair;
                        if (localHead < activeHeads) {
                            StageC4<CompilePolicy>(ctx, chunk, groupBegin, localHead);
                        }
                    }
                }
            } else {
                // Original AIV-leaf schedule: consume and release each
                // pair around C4, preserving low-occupancy latency.
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kReadyFlagId[pair]);
                    for (uint32_t headInPair = 0; headInPair < 2; ++headInPair) {
                        const uint32_t localHead = pair * 2 + headInPair;
                        if (localHead < activeHeads) {
                            StageC4<CompilePolicy>(ctx, chunk, groupBegin, localHead);
                        }
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(kFreeFlagId[pair]);
                }
            }

            // C5：只在下半块存在时计算 Akk[32:M,0:32]=negX1@T。
            for (uint32_t localHead = 0; localHead < Shape::kHeadsPerGroup; ++localHead) {
                if (localHead < activeHeads) {
                    const uint32_t valueHead = groupBegin + localHead;
                    StageC5<CompilePolicy>(ctx, chunk, valueHead, localHead);
                }
            }

            // C7：每个 pair 一次调用；先搬完两个 head 的 RHS 并归还
            // workspace，再分别计算 W 与 U。
            for (uint32_t pair = 0; pair < 2; ++pair) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kReadyFlagId[pair]);
                StageC7<CompilePolicy>(ctx, chunk, groupBegin, activeHeads, pair, kFreeFlagId[pair]);
            }
            groupBegin += activeHeads;
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
    if constexpr (CompilePolicy::arch22CubeLeaf) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[1]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[1]);
    }
    ReleaseEvents<CompilePolicy>(ctx);
}

// 本核第一个 group（与主循环同样的解码规则：跳过无效 chunk 与空 head 段）。
template <typename CompilePolicy>
__aicore__ inline bool FirstGroup(TargetCubeContext<CompilePolicy> &ctx, uint32_t workBegin, uint32_t workEnd,
                                  ChunkRange &chunk, uint32_t &groupBegin, uint32_t &activeHeads)
{
    for (uint32_t work = workBegin; work < workEnd; ++work) {
        uint32_t globalChunk = 0;
        uint32_t headPartition = 0;
        TargetDecodeWorkItem(ctx.args_.tiling, work, globalChunk, headPartition);
        ChunkRange candidate{};
        if (!ResolveChunk(ctx.args_, globalChunk, candidate)) {
            continue;
        }
        uint32_t headBegin = 0;
        uint32_t headEnd = 0;
        TargetHeadRange(ctx.args_.tiling, headPartition, headBegin, headEnd);
        if (headEnd <= headBegin) {
            continue;
        }
        chunk = candidate;
        groupBegin = headBegin;
        const uint32_t count = headEnd - headBegin;
        activeHeads = count > Shape::kHeadsPerGroup ? Shape::kHeadsPerGroup : count;
        return true;
    }
    return false;
}

// 当前 group 之后的下一个 group；跨 work item 时按主循环规则重新解码。
template <typename CompilePolicy>
__aicore__ inline bool NextGroup(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &currentChunk,
                                 uint32_t work, uint32_t groupBegin,
                                 uint32_t activeHeads, uint32_t headEnd, uint32_t workEnd, ChunkRange &chunk,
                                 uint32_t &nextGroup, uint32_t &nextActive)
{
    uint32_t cursor = groupBegin + activeHeads;
    uint32_t cursorWork = work;
    // The output chunk starts empty. A head-group lookahead within the same
    // work item must retain the current chunk, including its valid row count.
    ChunkRange cursorChunk = currentChunk;
    uint32_t cursorEnd = headEnd;
    while (true) {
        if (cursor < cursorEnd) {
            chunk = cursorChunk;
            nextGroup = cursor;
            const uint32_t count = cursorEnd - cursor;
            nextActive = count > Shape::kHeadsPerGroup ? Shape::kHeadsPerGroup : count;
            return true;
        }
        ++cursorWork;
        if (cursorWork >= workEnd) {
            return false;
        }
        uint32_t globalChunk = 0;
        uint32_t headPartition = 0;
        TargetDecodeWorkItem(ctx.args_.tiling, cursorWork, globalChunk, headPartition);
        ChunkRange candidate{};
        if (!ResolveChunk(ctx.args_, globalChunk, candidate)) {
            cursor = 0;
            cursorEnd = 0;
            continue;
        }
        uint32_t headBegin = 0;
        uint32_t headEndNext = 0;
        TargetHeadRange(ctx.args_.tiling, headPartition, headBegin, headEndNext);
        cursorChunk = candidate;
        cursor = headBegin;
        cursorEnd = headEndNext;
    }
}

// q/k 的行平方和：每个 32 行块计算 x @ x^T，L0C 的对角元就是
// 每行的 |x|^2，以行主序写入 relay。
// AIV 只读对角线，因此这里不做任何归约。
template <typename CompilePolicy>
__aicore__ inline void StageNormSums(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk,
                                     uint32_t groupBegin, uint32_t activeHeads)
{
    // 尾 chunk（validRows < 64）与 sequence-major 布局走 AIV 原路径：
    // 前者避免越界读，后者 Nd2Nz 表达不了行间跨 stride。
    if (chunk.validRows != Shape::kChunkRows || ctx.args_.tiling.inputSequenceMajor) {
        return;
    }
    auto l1Bytes = ctx.l1Buf_.template Get<uint8_t>();
    auto l0A = ctx.l0ABuf_.template Get<bfloat16_t>();
    auto l0B = ctx.l0BBuf_.template Get<bfloat16_t>();
    auto l0C = ctx.l0CBuf_.template Get<float>();
    AscendC::GlobalTensor<bfloat16_t> qGm;
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.q));
    AscendC::GlobalTensor<bfloat16_t> kGm;
    kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.k));
    constexpr uint32_t kFractalElements = 16 * 16;
    constexpr uint32_t kBlocks = Shape::kChunkRows / 32;   // 2
    constexpr uint32_t kBlockElems = 32 * Shape::kHeadDim; // 4096 BF16
    // 每个张量 2 个 32x32 块 = 2048 个 FP32 = 8 KiB。
    constexpr uint32_t kSumBlockElems = 32 * 32;
    constexpr uint32_t kSumTensorElems = kBlocks * kSumBlockElems;
    for (uint32_t h = 0; h < activeHeads; ++h) {
        const uint32_t valueHead = groupBegin + h;
        const uint64_t slot = WorkspaceSlotBase(ctx.workgroup_, h, ctx.kWorkgroupStride, ctx.kSlotStride);
        const uint64_t qkOffset =
            QkInputOffset(ctx.args_.tiling, chunk, QkHeadForValueHead(ctx.args_.tiling, valueHead));
        AscendC::GlobalTensor<float> normRelay;
        normRelay.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kArch22NormSum),
            Workspace::kArch22NormSumBytes / sizeof(float));
        auto lane = l1Bytes[L1::kHeadLane[h]].template ReinterpretCast<bfloat16_t>();
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.dValue = Shape::kHeadDim;
        copy.srcDValue = Shape::kHeadDim;
        copy.srcNdMatrixStride = 0;
        copy.dstNzNStride = 1;
        copy.dstNzMatrixStride = 0;
        copy.nValue = Shape::kChunkRows;
        copy.dstNzC0Stride = Shape::kChunkRows;
        AscendC::DataCopy(lane, qGm[qkOffset], copy);
        AscendC::DataCopy(lane[Shape::kBf16MatrixBytes / sizeof(bfloat16_t)], kGm[qkOffset], copy);
        // B 侧必须与“Kminus 的 n=32 独立矩阵”同构，直接从 64 行矩阵里
        // 跨分形取会和 L0B 的 Zn 序不一致。这里按 32 行块单独搬一份 NZ，
        // A 侧仍走 C2 已验证的 64 行 stride=4 路径。
        auto laneB = lane[0x8000 / sizeof(bfloat16_t)];
        AscendC::Nd2NzParams copyBlk = copy;
        copyBlk.nValue = 32;
        copyBlk.dstNzC0Stride = 32;
        for (uint32_t tensor = 0; tensor < 2; ++tensor) {
            AscendC::GlobalTensor<bfloat16_t> src = tensor == 0 ? qGm : kGm;
            for (uint32_t block = 0; block < kBlocks; ++block) {
                AscendC::DataCopy(laneB[(tensor * kBlocks + block) * kBlockElems],
                                  src[qkOffset + block * 32 * Shape::kHeadDim], copyBlk);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);
        for (uint32_t tensor = 0; tensor < 2; ++tensor) {
            auto srcL1 = lane[tensor == 0 ? 0 : (Shape::kBf16MatrixBytes / sizeof(bfloat16_t))];
            for (uint32_t block = 0; block < kBlocks; ++block) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
                AscendC::LoadData2DParams loadA{};
                loadA.startIndex = 0;
                loadA.repeatTimes = Shape::kHeadDim / 16;
                loadA.srcStride = Shape::kChunkRows / 16;
                loadA.dstGap = 0;
                loadA.ifTranspose = false;
                loadA.sid = 0;
                loadA.addrMode = 0;
                // m=32：两个 16 行块分别落在 l0A 的 [0] 与 [8 分形]。
                AscendC::LoadData(l0A, srcL1[block * 2 * kFractalElements], loadA);
                AscendC::LoadData(l0A[Shape::kHeadDim / 16 * kFractalElements],
                                  srcL1[(block * 2 + 1) * kFractalElements], loadA);
                AscendC::LoadData2DParams loadB{};
                loadB.startIndex = 0;
                loadB.repeatTimes = (Shape::kHeadDim / 16) * 2;
                loadB.srcStride = 1;
                loadB.dstGap = 0;
                loadB.ifTranspose = false;
                loadB.sid = 0;
                loadB.addrMode = 0;
                AscendC::LoadData(l0B, laneB[(tensor * kBlocks + block) * kBlockElems], loadB);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
                AscendC::MmadParams mmad{};
                mmad.m = 32;
                mmad.n = 32;
                mmad.k = Shape::kHeadDim;
                mmad.cmatrixInitVal = true;
                mmad.cmatrixSource = false;
                mmad.unitFlag = 0;
                AscendC::Mmad(l0C, l0A, l0B, mmad);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
                auto fix = AscendC::FixpipeParamsV220(32, 32, 32, 32, false);
                fix.quantPre = QuantMode_t::NoQuant;
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    normRelay[tensor * kSumTensorElems + block * kSumBlockElems], l0C, fix);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
            }
        }
    }
}

template <typename CompilePolicy>
__aicore__ inline void Mmad32(TargetCubeContext<CompilePolicy> &ctx, AscendC::LocalTensor<float> l0A,
                              AscendC::LocalTensor<float> l0B, AscendC::LocalTensor<float> l0C,
                              AscendC::LocalTensor<float> srcA, AscendC::LocalTensor<float> srcB, uint32_t slot,
                              bool accumulate, bool last)
{
    auto result = l0C[slot * 1024];
    auto aBuf = l0A[slot * 1024];
    auto bBuf = l0B[slot * 1024];
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[slot]);
    LoadL0A32<CompilePolicy>(aBuf, srcA);
    LoadL0B32<CompilePolicy>(bBuf, srcB);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ctx.c3Mte1ToM_[slot]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ctx.c3Mte1ToM_[slot]);
    if (!accumulate) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[slot]);
    }
    AscendC::SetHF32Mode(false);
    AscendC::MmadParams mmad{};
    mmad.m = 32;
    mmad.n = 32;
    mmad.k = 32;
    mmad.cmatrixInitVal = !accumulate;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    AscendC::Mmad(result, aBuf, bBuf, mmad);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[slot]);
    if (last) {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ctx.c3MToFix_[slot]);
    }
}

template <typename CompilePolicy>
__aicore__ inline void FixpipeFp32Nz(TargetCubeContext<CompilePolicy> &ctx, AscendC::GlobalTensor<float> dst,
                                     AscendC::LocalTensor<float> l0C, uint32_t slot)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ctx.c3MToFix_[slot]);
    FixpipeFp32NzRaw<CompilePolicy>(dst, l0C, slot);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[slot]);
}

template <typename CompilePolicy>
__aicore__ inline void StageC3(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t groupBegin,
                               uint32_t activeHeads)
{
    using namespace Workspace;
    constexpr uint32_t kRounds = CompilePolicy::arch22FastLeaf ? 1 : 2;
    auto l1Bytes = ctx.l1Buf_.template Get<uint8_t>();
    auto l0A = ctx.l0ABuf_.template Get<float>();
    auto l0B = ctx.l0BBuf_.template Get<float>();
    auto l0C = ctx.l0CBuf_.template Get<float>();

    AscendC::LocalTensor<float> a0L1[Shape::kHeadsPerGroup];
    AscendC::LocalTensor<float> y0L1[Shape::kHeadsPerGroup];
    AscendC::LocalTensor<float> a1L1[Shape::kHeadsPerGroup];
    AscendC::LocalTensor<float> y1L1[Shape::kHeadsPerGroup];
    AscendC::LocalTensor<float> idL1[Shape::kHeadsPerGroup];
    AscendC::GlobalTensor<float> aux[Shape::kHeadsPerGroup];
    for (uint32_t h = 0; h < activeHeads; ++h) {
        a0L1[h] = l1Bytes[L1::kHeadLane[h]].template ReinterpretCast<float>();
        y0L1[h] = l1Bytes[L1::kX0 + h * L1::kQuadrantStride].template ReinterpretCast<float>();
        a1L1[h] = l1Bytes[L1::kNegX1 + h * L1::kQuadrantStride].template ReinterpretCast<float>();
        y1L1[h] = l1Bytes[L1::kT + h * L1::kQuadrantStride].template ReinterpretCast<float>();
        idL1[h] = l1Bytes[L1::kAkk + h * L1::kAkkStride].template ReinterpretCast<float>();
        const uint64_t slot = WorkspaceSlotBase(ctx.workgroup_, h, ctx.kWorkgroupStride, ctx.kSlotStride);
        aux[h].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + kArch22InvAux));
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(aux[h]);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        LoadFp32Nz32<CompilePolicy>(a0L1[h], aux[h][kArch22InvL00 / sizeof(float)], 64);
        LoadFp32Nz32<CompilePolicy>(y0L1[h], aux[h][kArch22InvY00 / sizeof(float)]);
        LoadFp32Nz32<CompilePolicy>(a1L1[h], aux[h][kArch22InvL11 / sizeof(float) + 32], 64);
        LoadFp32Nz32<CompilePolicy>(y1L1[h], aux[h][kArch22InvY11 / sizeof(float)]);
        LoadFp32Nz32<CompilePolicy>(idL1[h], aux[h][kArch22InvIdentity / sizeof(float)]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.tReady_[h]);
    }
    for (uint32_t h = 0; h < activeHeads; ++h) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.tReady_[h]);
    }

    // Convert A=-L into A=L^2 once. The remaining four rounds then consume
    // the product factors (I+L^2), (I+L^4), (I+L^8), (I+L^16).
    for (uint32_t h = 0; h < activeHeads; ++h) {
        Mmad32<CompilePolicy>(ctx, l0A, l0B, l0C, a0L1[h], a0L1[h], 0, false, true);
        Mmad32<CompilePolicy>(ctx, l0A, l0B, l0C, a1L1[h], a1L1[h], 1, false, true);
        FixpipeFp32Nz<CompilePolicy>(ctx, aux[h][kArch22InvL00 / sizeof(float)], l0C, 0);
        FixpipeFp32Nz<CompilePolicy>(ctx, aux[h][kArch22InvL11 / sizeof(float)], l0C, 1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(ctx.fixToMte2_[h]);
    }
    for (uint32_t h = 0; h < activeHeads; ++h) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(ctx.fixToMte2_[h]);
        AscendC::DataCopy(a0L1[h], aux[h][kArch22InvL00 / sizeof(float)], AscendC::DataCopyParams(1, 128, 0, 0));
        AscendC::DataCopy(a1L1[h], aux[h][kArch22InvL11 / sizeof(float)], AscendC::DataCopyParams(1, 128, 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.tReady_[h]);
    }
    for (uint32_t h = 0; h < activeHeads; ++h) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.tReady_[h]);
    }

    for (uint32_t round = 0; round < kRounds; ++round) {
        for (uint32_t h = 0; h < activeHeads; ++h) {
            for (uint32_t leaf = 0; leaf < 2; ++leaf) {
                auto a = leaf == 0 ? a0L1[h] : a1L1[h];
                auto y = leaf == 0 ? y0L1[h] : y1L1[h];
                const uint32_t aOffset = leaf == 0 ? kArch22InvL00 : kArch22InvL11;
                const uint32_t yOffset = leaf == 0 ? kArch22InvY00 : kArch22InvY11;
                if (round + 1 < kRounds) {
                    // Independent A^2 work fills the second L0C slot while
                    // the Y result is ready for Fixpipe. Its L0A/L0B load
                    // fills the bubble before Y's accumulate MMAD.
                    Mmad32<CompilePolicy>(ctx, l0A, l0B, l0C, y, idL1[h], 0, false, false);
                    Mmad32<CompilePolicy>(ctx, l0A, l0B, l0C, a, a, 1, false, true);
                    Mmad32<CompilePolicy>(ctx, l0A, l0B, l0C, y, a, 0, true, true);
                } else {
                    Mmad32<CompilePolicy>(ctx, l0A, l0B, l0C, y, idL1[h], 0, false, false);
                    Mmad32<CompilePolicy>(ctx, l0A, l0B, l0C, y, a, 0, true, true);
                }

                if (round + 1 == kRounds) {
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ctx.c3MToFix_[0]);
                    FixpipeFp32NzRaw<CompilePolicy>(aux[h][yOffset / sizeof(float)], l0C, 0);
                    AscendC::GlobalTensor<bfloat16_t> payloadAkk;
                    payloadAkk.SetGlobalBuffer(
                        reinterpret_cast<__gm__ bfloat16_t *>(
                            ctx.args_.workspace +
                            WorkspaceSlotBase(ctx.workgroup_, h, ctx.kWorkgroupStride, ctx.kSlotStride) + kPayload +
                            kAkk),
                        Shape::kChunkRows * Shape::kChunkRows);
                    const uint32_t blockOffset = leaf == 0 ? 0 : 32 * Shape::kChunkRows + 32;
                    constexpr bool readFullAkkFromOutput = CompilePolicy::outputAkk;
                    if (!readFullAkkFromOutput || chunk.validRows != Shape::kChunkRows) {
                        FixpipeAkkRowMajorRaw<CompilePolicy>(payloadAkk[blockOffset], l0C, 32, 0);
                    }
                    if constexpr (CompilePolicy::outputAkk) {
                        const uint32_t valueHead = groupBegin + h;
                        const uint32_t rows = leaf == 0 ? (chunk.validRows < 32 ? chunk.validRows : 32) :
                                                          (chunk.validRows > 32 ? chunk.validRows - 32 : 0);
                        if (rows != 0) {
                            AscendC::GlobalTensor<bfloat16_t> out;
                            const uint64_t outOffset = AOutputOffset(ctx.args_.tiling, chunk, valueHead) + blockOffset;
                            out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.akk) + outOffset,
                                                rows * Shape::kChunkRows);
                            FixpipeAkkRowMajorRaw<CompilePolicy>(out, l0C, rows, 0);
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[0]);
                } else {
                    FixpipeFp32Nz<CompilePolicy>(ctx, aux[h][yOffset / sizeof(float)], l0C, 0);
                    FixpipeFp32Nz<CompilePolicy>(ctx, aux[h][aOffset / sizeof(float)], l0C, 1);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(ctx.fixToMte2_[h]);
        }
        for (uint32_t h = 0; h < activeHeads; ++h) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(ctx.fixToMte2_[h]);
        }

        if (round + 1 < kRounds) {
            for (uint32_t h = 0; h < activeHeads; ++h) {
                AscendC::DataCopy(a0L1[h], aux[h][kArch22InvL00 / sizeof(float)],
                                  AscendC::DataCopyParams(1, 128, 0, 0));
                AscendC::DataCopy(y0L1[h], aux[h][kArch22InvY00 / sizeof(float)],
                                  AscendC::DataCopyParams(1, 128, 0, 0));
                AscendC::DataCopy(a1L1[h], aux[h][kArch22InvL11 / sizeof(float)],
                                  AscendC::DataCopyParams(1, 128, 0, 0));
                AscendC::DataCopy(y1L1[h], aux[h][kArch22InvY11 / sizeof(float)],
                                  AscendC::DataCopyParams(1, 128, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.tReady_[h]);
            }
            for (uint32_t h = 0; h < activeHeads; ++h) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.tReady_[h]);
            }
        }
    }
}

template <typename CompilePolicy>
__aicore__ inline void StageC2(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t localHead)
{
    const uint64_t slot = WorkspaceSlotBase(ctx.workgroup_, localHead, ctx.kWorkgroupStride, ctx.kSlotStride);
    AscendC::GlobalTensor<bfloat16_t> payload;
    payload.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kPayload));
    auto l1Bytes = ctx.l1Buf_.template Get<uint8_t>();
    auto scoreL1 = l1Bytes[L1::kHeadLane[localHead]].template ReinterpretCast<bfloat16_t>();
    auto l0A = ctx.l0ABuf_.template Get<bfloat16_t>();
    auto l0B = ctx.l0BBuf_.template Get<bfloat16_t>();
    auto l0C = ctx.l0CBuf_.template Get<float>();

    // 每个源矩阵只搬一次；MTE2 在搬运时把 ND 转为 L1 NZ。
    AscendC::Nd2NzParams copy{};
    copy.ndNum = 1;
    copy.dValue = Shape::kHeadDim;
    copy.srcDValue = Shape::kHeadDim;
    copy.srcNdMatrixStride = 0;
    copy.dstNzNStride = 1;
    copy.dstNzMatrixStride = 0;
    copy.nValue = Shape::kChunkRows;
    copy.dstNzC0Stride = Shape::kChunkRows;
    AscendC::DataCopy(scoreL1[ScorePayload::kQPlus / sizeof(bfloat16_t)],
                      payload[ScorePayload::kQPlus / sizeof(bfloat16_t)], copy);
    AscendC::DataCopy(scoreL1[ScorePayload::kKPlus / sizeof(bfloat16_t)],
                      payload[ScorePayload::kKPlus / sizeof(bfloat16_t)], copy);
    for (uint32_t s = 0; s < Shape::kSubChunkCount; ++s) {
        copy.nValue = Shape::kPrefixRows[s];
        copy.dstNzC0Stride = Shape::kPrefixRows[s];
        AscendC::DataCopy(scoreL1[ScorePayload::kKMinus[s] / sizeof(bfloat16_t)],
                          payload[ScorePayload::kKMinus[s] / sizeof(bfloat16_t)], copy);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);

    const uint32_t active = ActiveSubChunks<CompilePolicy>(chunk.validRows);
    AscendC::GlobalTensor<float> rawScoreRelay;
    rawScoreRelay.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kArch22CubeRelay),
        Workspace::kArch22RawScoreBytes / sizeof(float));
    for (uint32_t s = 0; s < active; ++s) {
        const uint32_t n = Shape::kPrefixRows[s];
        const uint32_t stackedBand = Shape::kSubChunkRows * Shape::kSubChunkRows * s * (s + 1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);

        // L1 zN 的同一 16 行在 8 个 K 分形间隔 4 个分形；
        // L0A zZ 中同一行块的 K 分形连续排列。
        constexpr uint32_t kBf16FractalElements = 16 * 16;
        AscendC::LoadData2DParams loadQ{};
        loadQ.startIndex = 0;
        loadQ.repeatTimes = Shape::kHeadDim / 16;
        loadQ.srcStride = Shape::kChunkRows / 16;
        loadQ.dstGap = 0;
        loadQ.ifTranspose = false;
        loadQ.sid = 0;
        loadQ.addrMode = 0;
        AscendC::LoadData(l0A, scoreL1[ScorePayload::kQPlus / sizeof(bfloat16_t) + s * kBf16FractalElements], loadQ);

        // c220 的 L0A 是 zZ，Kplus 的 16 行紧跟完整的 Qplus 行块。
        AscendC::LoadData(l0A[(Shape::kHeadDim / 16) * kBf16FractalElements],
                          scoreL1[ScorePayload::kKPlus / sizeof(bfloat16_t) + s * kBf16FractalElements], loadQ);

        // Kminus 的 ND [n,128] 经 Nd2Nz 后，分形顺序已经与
        // L0B 中数学上的 [128,n] 一致；不能再转置 16x16 分形。
        AscendC::LoadData2DParams loadKMinus{};
        loadKMinus.startIndex = 0;
        loadKMinus.repeatTimes = (Shape::kHeadDim / 16) * (n / 16);
        loadKMinus.srcStride = 1;
        loadKMinus.dstGap = 0;
        loadKMinus.ifTranspose = false;
        loadKMinus.sid = 0;
        loadKMinus.addrMode = 0;
        AscendC::LoadData(l0B, scoreL1[ScorePayload::kKMinus[s] / sizeof(bfloat16_t)], loadKMinus);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);

        AscendC::MmadParams mmad{};
        mmad.m = 32;
        mmad.n = n;
        mmad.k = Shape::kHeadDim;
        mmad.cmatrixInitVal = true;
        mmad.cmatrixSource = false;
        mmad.unitFlag = 0;
        // 一次 MMAD 同时得到上半 rawAqk 和下半 rawAkk。
        AscendC::Mmad(l0C, l0A, l0B, mmad);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
        uint32_t rows = chunk.validRows - s * Shape::kSubChunkRows;
        if (rows > Shape::kSubChunkRows) {
            rows = Shape::kSubChunkRows;
        }
        if (rows == Shape::kSubChunkRows) {
            auto fix = AscendC::FixpipeParamsV220(n, 2 * Shape::kSubChunkRows, 2 * Shape::kSubChunkRows, n, false);
            fix.quantPre = QuantMode_t::NoQuant;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(rawScoreRelay[stackedBand], l0C, fix);
        } else {
            auto fix = AscendC::FixpipeParamsV220(n, rows, 2 * Shape::kSubChunkRows, n, false);
            fix.quantPre = QuantMode_t::NoQuant;
            // 尾 sub-chunk 只写有效的 Qplus/Kplus 行。FP32 L0C
            // 的基础分形为 16x16，下半 16 行从第二个 M1 分形开始。
            constexpr uint32_t kLowerM1Offset = 16 * 16;
            const uint32_t relayElements = rows * n;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(rawScoreRelay[stackedBand], l0C, fix);
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(rawScoreRelay[stackedBand + relayElements],
                                                                   l0C[kLowerM1Offset], fix);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
    }
}

template <typename CompilePolicy>
__aicore__ inline void StageC4(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t groupBegin,
                               uint32_t localHead)
{
    const uint64_t slot = WorkspaceSlotBase(ctx.workgroup_, localHead, ctx.kWorkgroupStride, ctx.kSlotStride);
    auto l1Bytes = ctx.l1Buf_.template Get<uint8_t>();
    auto akkL1 = l1Bytes[L1::kAkk + localHead * L1::kAkkStride].template ReinterpretCast<bfloat16_t>();
    AscendC::GlobalTensor<bfloat16_t> akkSource;
    if constexpr (CompilePolicy::outputAkk) {
        if (chunk.validRows == Shape::kChunkRows) {
            const uint32_t valueHead = groupBegin + localHead;
            akkSource.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.akk) +
                                          AOutputOffset(ctx.args_.tiling, chunk, valueHead),
                                      Shape::kChunkRows * Shape::kChunkRows);
        } else {
            akkSource.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot +
                                                                            Workspace::kPayload + Workspace::kAkk));
        }
    } else {
        akkSource.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kPayload + Workspace::kAkk));
    }
    AscendC::Nd2NzParams akkCopy{};
    akkCopy.ndNum = 1;
    akkCopy.nValue = Shape::kChunkRows;
    akkCopy.dValue = Shape::kChunkRows;
    akkCopy.srcDValue = Shape::kChunkRows;
    akkCopy.srcNdMatrixStride = 0;
    akkCopy.dstNzNStride = 1;
    akkCopy.dstNzC0Stride = Shape::kChunkRows;
    akkCopy.dstNzMatrixStride = 0;
    AscendC::DataCopy(akkL1, akkSource, akkCopy);

    if (chunk.validRows <= 32) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);
        return;
    }

    AscendC::GlobalTensor<float> payload;
    payload.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kPayload));
    AscendC::GlobalTensor<float> invAux;
    invAux.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kArch22InvAux),
                           Workspace::kArch22InvAuxBytes / sizeof(float));
    auto bL1 = l1Bytes[L1::kHeadLane[localHead]].template ReinterpretCast<float>();
    auto x0L1 = l1Bytes[L1::kX0 + localHead * L1::kQuadrantStride].template ReinterpretCast<float>();
    auto negX1L1 = l1Bytes[L1::kNegX1 + localHead * L1::kQuadrantStride].template ReinterpretCast<float>();
    auto tL1 = l1Bytes[L1::kT + localHead * L1::kQuadrantStride].template ReinterpretCast<float>();
    AscendC::GlobalTensor<float> tRelay;
    tRelay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kArch22TRelay),
                           Shape::kQuadrantFp32Bytes / sizeof(float));
    AscendC::Nd2NzParams copy{};
    copy.ndNum = 1;
    copy.nValue = 32;
    copy.dValue = 32;
    copy.srcDValue = 32;
    copy.srcNdMatrixStride = 0;
    copy.dstNzNStride = 1;
    copy.dstNzC0Stride = 32;
    copy.dstNzMatrixStride = 0;
    AscendC::DataCopy(bL1, payload[Workspace::kB / sizeof(float)], copy);
    if constexpr (CompilePolicy::arch22CubeLeaf) {
        AscendC::DataCopy(x0L1, invAux[Workspace::kArch22InvY00 / sizeof(float)],
                          AscendC::DataCopyParams(1, 128, 0, 0));
        AscendC::DataCopy(negX1L1, invAux[Workspace::kArch22InvY11 / sizeof(float)],
                          AscendC::DataCopyParams(1, 128, 0, 0));
    } else {
        AscendC::DataCopy(x0L1, payload[Workspace::kX0 / sizeof(float)], copy);
        AscendC::DataCopy(negX1L1, payload[Workspace::kNegX1 / sizeof(float)], copy);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);

    // T[32,32] = B[32,32] @ X0[32,32]。
    auto l0A = ctx.l0ABuf_.template Get<float>();
    auto l0B = ctx.l0BBuf_.template Get<float>();
    auto l0C = ctx.l0CBuf_.template Get<float>();
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
    constexpr uint32_t kFp32FractalElements = 16 * 8;
    constexpr uint32_t kFp32RowFractals = 2;
    constexpr uint32_t kFp32ColumnFractals = 4;
    AscendC::LoadData2DParams loadA{};
    loadA.startIndex = 0;
    loadA.repeatTimes = kFp32ColumnFractals;
    loadA.srcStride = kFp32RowFractals;
    loadA.dstGap = 0;
    loadA.ifTranspose = false;
    loadA.sid = 0;
    loadA.addrMode = 0;
    for (uint32_t rowFractal = 0; rowFractal < kFp32RowFractals; ++rowFractal) {
        AscendC::LoadData(l0A[rowFractal * kFp32ColumnFractals * kFp32FractalElements],
                          bL1[rowFractal * kFp32FractalElements], loadA);
    }

    // c220 的 FP32 L1 zN 到 L0B Zn 使用 3Dv2。参数完整描述
    // [32,32] 矩阵，默认 LoadData 同时设置 FMatrix 和 padding。
    AscendC::LoadData3DParamsV2<float> loadB{};
    loadB.l1H = 1;
    loadB.l1W = 32;
    loadB.channelSize = 32;
    loadB.kExtension = 32;
    loadB.mExtension = 32;
    loadB.kStartPt = 0;
    loadB.mStartPt = 0;
    loadB.strideW = 1;
    loadB.strideH = 1;
    loadB.filterW = 1;
    loadB.filterH = 1;
    loadB.dilationFilterW = 1;
    loadB.dilationFilterH = 1;
    loadB.enTranspose = true;
    loadB.enSmallK = false;
    loadB.filterSizeW = false;
    loadB.filterSizeH = false;
    loadB.fMatrixCtrl = false;
    AscendC::LoadData(l0B, x0L1, loadB);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
    AscendC::SetHF32Mode(false);
    AscendC::MmadParams tMmad{};
    tMmad.m = 32;
    tMmad.n = 32;
    tMmad.k = 32;
    tMmad.cmatrixInitVal = true;
    tMmad.cmatrixSource = false;
    tMmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, tMmad);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);

    auto fix = AscendC::FixpipeParamsV220(32, 32, 32, 32, false);
    fix.quantPre = QuantMode_t::NoQuant;
    fix.isChannelSplit = true;
    // C220 不支持 FP32 L0C 直写 L1：先按标准 FP32 16x8 NZ 分形写 GM，
    // 再原样搬回 L1 供 C5 消费。
    AscendC::Fixpipe<float, float, kFixpipeNz>(tRelay, l0C, fix);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(ctx.fixToMte2_[localHead]);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(ctx.fixToMte2_[localHead]);
    AscendC::DataCopy(tL1, tRelay, AscendC::DataCopyParams(1, 128, 0, 0));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.tReady_[localHead]);
}

template <typename CompilePolicy>
__aicore__ inline void StageC5(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t valueHead,
                               uint32_t localHead)
{
    if (chunk.validRows <= 32) {
        return;
    }
    auto l1Bytes = ctx.l1Buf_.template Get<uint8_t>();
    auto negX1L1 = l1Bytes[L1::kNegX1 + localHead * L1::kQuadrantStride].template ReinterpretCast<float>();
    auto tL1 = l1Bytes[L1::kT + localHead * L1::kQuadrantStride].template ReinterpretCast<float>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.tReady_[localHead]);

    // Akk[32:M,0:32] = negX1[32:M,0:32] @ T[32,32]。
    auto l0A = ctx.l0ABuf_.template Get<float>();
    auto l0B = ctx.l0BBuf_.template Get<float>();
    auto l0C = ctx.l0CBuf_.template Get<float>();
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
    constexpr uint32_t kFp32FractalElements = 16 * 8;
    constexpr uint32_t kFp32RowFractals = 2;
    constexpr uint32_t kFp32ColumnFractals = 4;
    AscendC::LoadData2DParams loadA{};
    loadA.startIndex = 0;
    loadA.repeatTimes = kFp32ColumnFractals;
    loadA.srcStride = kFp32RowFractals;
    loadA.dstGap = 0;
    loadA.ifTranspose = false;
    loadA.sid = 0;
    loadA.addrMode = 0;
    for (uint32_t rowFractal = 0; rowFractal < kFp32RowFractals; ++rowFractal) {
        AscendC::LoadData(l0A[rowFractal * kFp32ColumnFractals * kFp32FractalElements],
                          negX1L1[rowFractal * kFp32FractalElements], loadA);
    }

    AscendC::LoadData3DParamsV2<float> loadB{};
    loadB.l1H = 1;
    loadB.l1W = 32;
    loadB.channelSize = 32;
    loadB.kExtension = 32;
    loadB.mExtension = 32;
    loadB.kStartPt = 0;
    loadB.mStartPt = 0;
    loadB.strideW = 1;
    loadB.strideH = 1;
    loadB.filterW = 1;
    loadB.filterH = 1;
    loadB.dilationFilterW = 1;
    loadB.dilationFilterH = 1;
    loadB.enTranspose = true;
    loadB.enSmallK = false;
    loadB.filterSizeW = false;
    loadB.filterSizeH = false;
    loadB.fMatrixCtrl = false;
    AscendC::LoadData(l0B, tL1, loadB);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
    AscendC::SetHF32Mode(false);
    AscendC::MmadParams akkMmad{};
    akkMmad.m = 32;
    akkMmad.n = 32;
    akkMmad.k = 32;
    akkMmad.cmatrixInitVal = true;
    akkMmad.cmatrixSource = false;
    akkMmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, akkMmad);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);

    const uint32_t bottomRows = chunk.validRows - 32;
    auto l1Fix = AscendC::FixpipeParamsV220(32, bottomRows, 32, Shape::kChunkRows, false);
    l1Fix.quantPre = QuantMode_t::F322BF16;
    auto akkL1 = l1Bytes[L1::kAkk + localHead * L1::kAkkStride].template ReinterpretCast<bfloat16_t>();
    // q10 在两字节 NZ 中从 (N1=0,M1=2) 开始，即 32*16 个元素；
    // dstStride=64 个 datablock 跨过完整 64 行 M 轴。
    AscendC::Fixpipe<bfloat16_t, float, kFixpipeNz>(akkL1[L1::kAkkQ10Elements], l0C, l1Fix);
    if constexpr (CompilePolicy::outputAkk) {
        AscendC::GlobalTensor<bfloat16_t> akkOutput;
        akkOutput.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.akk) +
                                      AOutputOffset(ctx.args_.tiling, chunk, valueHead) + 32 * Shape::kChunkRows,
                                  bottomRows * Shape::kChunkRows);
        auto outputFix = AscendC::FixpipeParamsV220(32, bottomRows, 32, Shape::kChunkRows, false);
        outputFix.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(akkOutput, l0C, outputFix);
    }
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(ctx.fixToMte1_[localHead]);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
}

template <typename CompilePolicy>
__aicore__ inline void StageC7(TargetCubeContext<CompilePolicy> &ctx, const ChunkRange &chunk, uint32_t groupBegin,
                               uint32_t activeHeads, uint32_t pair, uint16_t freeFlagId)
{
    const bool hasBottom = chunk.validRows > 32;
    const uint32_t m = hasBottom ? 64 : 32;
    auto l1Bytes = ctx.l1Buf_.template Get<uint8_t>();

    // 先把同一 pair 两个有效 head 的 K_beta_g/V_beta 全部搬入独立 L1 lane。
    AscendC::Nd2NzParams rhsCopy{};
    rhsCopy.ndNum = 1;
    rhsCopy.nValue = m;
    rhsCopy.dValue = Shape::kHeadDim;
    rhsCopy.srcDValue = Shape::kHeadDim;
    rhsCopy.srcNdMatrixStride = 0;
    rhsCopy.dstNzNStride = 1;
    rhsCopy.dstNzC0Stride = m;
    rhsCopy.dstNzMatrixStride = 0;
    for (uint32_t headInPair = 0; headInPair < 2; ++headInPair) {
        const uint32_t localHead = pair * 2 + headInPair;
        if (localHead >= activeHeads) {
            continue;
        }
        const uint32_t valueHead = groupBegin + localHead;
        const uint64_t slot = WorkspaceSlotBase(ctx.workgroup_, localHead, ctx.kWorkgroupStride, ctx.kSlotStride);
        auto kBetaL1 = l1Bytes[L1::kHeadLane[localHead]].template ReinterpretCast<bfloat16_t>();
        auto vBetaL1 =
            l1Bytes[L1::kHeadLane[localHead] + Shape::kBf16MatrixBytes].template ReinterpretCast<bfloat16_t>();
        AscendC::GlobalTensor<bfloat16_t> kBetaRelay;
        AscendC::GlobalTensor<bfloat16_t> vBetaRelay;
        kBetaRelay.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kArch22RhsKBetaG));
        vBetaRelay.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kArch22RhsVBeta));
        AscendC::DataCopy(kBetaL1, kBetaRelay, rhsCopy);
        AscendC::DataCopy(vBetaL1, vBetaRelay, rhsCopy);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);

    // RHS 已全部离开 workspace；一次 collective free 同时归还 pair 的两个 slot。
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(freeFlagId);

    // 再逐 head 消费 L1 常驻的 Akk 与两个 RHS，分别计算 W、U。
    for (uint32_t headInPair = 0; headInPair < 2; ++headInPair) {
        const uint32_t localHead = pair * 2 + headInPair;
        if (localHead >= activeHeads) {
            continue;
        }
        const uint32_t valueHead = groupBegin + localHead;
        if (hasBottom) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(ctx.fixToMte1_[localHead]);
        }

        auto akkL1 = l1Bytes[L1::kAkk + localHead * L1::kAkkStride].template ReinterpretCast<bfloat16_t>();
        auto kBetaL1 = l1Bytes[L1::kHeadLane[localHead]].template ReinterpretCast<bfloat16_t>();
        auto vBetaL1 =
            l1Bytes[L1::kHeadLane[localHead] + Shape::kBf16MatrixBytes].template ReinterpretCast<bfloat16_t>();
        const uint64_t outputOffset = HeadTensorOffset(ctx.args_.tiling, chunk, valueHead, Shape::kHeadDim);

        // W = Akk @ K_beta_g。
        auto l0A = ctx.l0ABuf_.template Get<bfloat16_t>();
        auto l0BForW = ctx.l0BBuf_.template Get<bfloat16_t>();
        auto l0C = ctx.l0CBuf_.template Get<float>();
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
        constexpr uint32_t kBf16FractalElements = 16 * 16;
        const uint32_t mFractals = m / 16;
        AscendC::LoadData2DParams loadA{};
        loadA.startIndex = 0;
        loadA.repeatTimes = mFractals;
        // Akk 在 L1 中始终按 64x64 zN 常驻。
        loadA.srcStride = Shape::kChunkRows / 16;
        loadA.dstGap = 0;
        loadA.ifTranspose = false;
        loadA.sid = 0;
        loadA.addrMode = 0;
        for (uint32_t rowFractal = 0; rowFractal < mFractals; ++rowFractal) {
            AscendC::LoadData(l0A[rowFractal * mFractals * kBf16FractalElements],
                              akkL1[rowFractal * kBf16FractalElements], loadA);
        }

        AscendC::LoadData2DParams loadW{};
        loadW.startIndex = 0;
        loadW.repeatTimes = Shape::kHeadDim / 16;
        loadW.srcStride = mFractals;
        loadW.dstGap = 0;
        loadW.ifTranspose = true;
        loadW.sid = 0;
        loadW.addrMode = 0;
        for (uint32_t rowFractal = 0; rowFractal < mFractals; ++rowFractal) {
            AscendC::LoadData(l0BForW[rowFractal * (Shape::kHeadDim / 16) * kBf16FractalElements],
                              kBetaL1[rowFractal * kBf16FractalElements], loadW);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
        AscendC::MmadParams wMmad{};
        wMmad.m = m;
        wMmad.n = Shape::kHeadDim;
        wMmad.k = m;
        wMmad.cmatrixInitVal = true;
        wMmad.cmatrixSource = false;
        wMmad.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0BForW, wMmad);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
        auto wFix = AscendC::FixpipeParamsV220(Shape::kHeadDim, chunk.validRows, m, Shape::kHeadDim, false);
        wFix.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(ctx.wGm_[outputOffset], l0C, wFix);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);

        // U = Akk @ V_beta；复用 L0 前等待 W 的 reader 与 Fixpipe 完成。
        auto l0BForU = ctx.l0BBuf_.template Get<bfloat16_t>();
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
        for (uint32_t rowFractal = 0; rowFractal < mFractals; ++rowFractal) {
            AscendC::LoadData(l0A[rowFractal * mFractals * kBf16FractalElements],
                              akkL1[rowFractal * kBf16FractalElements], loadA);
            AscendC::LoadData(l0BForU[rowFractal * (Shape::kHeadDim / 16) * kBf16FractalElements],
                              vBetaL1[rowFractal * kBf16FractalElements], loadW);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
        AscendC::MmadParams uMmad{};
        uMmad.m = m;
        uMmad.n = Shape::kHeadDim;
        uMmad.k = m;
        uMmad.cmatrixInitVal = true;
        uMmad.cmatrixSource = false;
        uMmad.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0BForU, uMmad);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
        auto uFix = AscendC::FixpipeParamsV220(Shape::kHeadDim, chunk.validRows, m, Shape::kHeadDim, false);
        uFix.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(ctx.uGm_[outputOffset], l0C, uFix);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
    }
}

template <typename CompilePolicy>
__aicore__ inline void ReleaseEvents(TargetCubeContext<CompilePolicy> &ctx)
{
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(ctx.mte2ToMte1_);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::MTE1_M>(ctx.mte1ToM_);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::M_MTE1>(ctx.mToMte1_);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::M_FIX>(ctx.mToFix_);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::FIX_M>(ctx.fixToM_);
    if constexpr (CompilePolicy::arch22CubeLeaf) {
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::M_FIX>(ctx.c3MToFix_[0]);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::M_FIX>(ctx.c3MToFix_[1]);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[0]);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::FIX_M>(ctx.c3FixToM_[1]);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[0]);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::M_MTE1>(ctx.c3MToMte1_[1]);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::MTE1_M>(ctx.c3Mte1ToM_[0]);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::MTE1_M>(ctx.c3Mte1ToM_[1]);
    }
    for (uint32_t head = 0; head < Shape::kHeadsPerGroup; ++head) {
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::FIX_MTE2>(ctx.fixToMte2_[head]);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(ctx.tReady_[head]);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::FIX_MTE1>(ctx.fixToMte1_[head]);
    }
}


} // namespace KdaPrepare::Arch22Target

#endif // ARCH22_TARGET_CHUNK_KDA_FWD_PREPARE_CUBE_H
