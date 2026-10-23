/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H
#define ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H

#include <cstdint>
#include "kernel_operator.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace KdaPrepare::Arch22 {

constexpr AscendC::FixpipeConfig kFixpipeNz = {
    AscendC::CO2Layout::NZ, false};

template <typename CompilePolicy>
class ChunkKdaFwdPrepareCube {
public:
    __aicore__ inline void Init(const PrepareKernelArgs &args, AscendC::TPipe *pipe)
    {
        args_ = args;
        pipe_ = pipe;
        workgroup_ = WorkgroupId();
        coreCount_ = args_.tiling.usedCoreNum;
        if (coreCount_ == 0) {
            return;
        }
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.w));
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.u));

        pipe_->InitBuffer(l1Buf_, L1::kPeak);
        pipe_->InitBuffer(l0ABuf_, 0x10000);
        pipe_->InitBuffer(l0BBuf_, 0x10000);
        pipe_->InitBuffer(l0CBuf_, 0x10000);

        // 每种 HardEvent 有独立 ID 池。保留 Alloc/Release，并把 CANN 9.1
        // 分配器的预期返回值直接列在申请现场，便于逐项核对。
        mte2ToMte1_ = pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE1>(); // ID 0
        mte1ToM_ = pipe_->AllocEventID<AscendC::HardEvent::MTE1_M>(); // ID 0
        // AIC 的 TPipe::Init 已在 M_MTE1 池占用 0、1、2。
        mToMte1_ = pipe_->AllocEventID<AscendC::HardEvent::M_MTE1>(); // ID 3
        mToFix_ = pipe_->AllocEventID<AscendC::HardEvent::M_FIX>(); // ID 0
        fixToM_ = pipe_->AllocEventID<AscendC::HardEvent::FIX_M>(); // ID 0
        fixToMte2_[0] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE2>(); // ID 0
        fixToMte2_[1] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE2>(); // ID 1
        fixToMte2_[2] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE2>(); // ID 2
        fixToMte2_[3] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE2>(); // ID 3
        tReady_[0] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE1>(); // ID 1
        tReady_[1] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE1>(); // ID 2
        tReady_[2] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE1>(); // ID 3
        tReady_[3] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE1>(); // ID 4
        fixToMte1_[0] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE1>(); // ID 0
        fixToMte1_[1] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE1>(); // ID 1
        fixToMte1_[2] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE1>(); // ID 2
        fixToMte1_[3] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE1>(); // ID 3
        // RHS 专用 L1 lane 的写后读事件（MTE1 → MTE2），独立 ID 池，ID 0。
        rhsLaneFree_ = pipe_->AllocEventID<AscendC::HardEvent::MTE1_MTE2>(); // ID 0
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixToM_);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(rhsLaneFree_);
    }

    __aicore__ inline void Process()
    {
        if (coreCount_ == 0) {
            return;
        }
        // mode 0x2 的核间 flag 是固定物理编号，不经过 EventID 分配器。
        // pair0/pair1: ready=0/1，free=2/3。
        constexpr uint16_t kReadyFlagId[2] = {0, 1};
        constexpr uint16_t kFreeFlagId[2] = {2, 3};
        const uint32_t total = TotalWorkItems(args_.tiling);
        const uint32_t workBegin = WorkBegin(total, workgroup_, coreCount_);
        const uint32_t workEnd = WorkEnd(total, workgroup_, coreCount_);
        if (workgroup_ >= coreCount_ || workBegin >= workEnd) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(fixToM_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(rhsLaneFree_);
            ReleaseEvents();
            return;
        }

        // 两个 pair 的初始许可由 AIC 发布；后续许可由 C7 写回完成后发布。
        for (uint32_t pair = 0; pair < 2; ++pair) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                kFreeFlagId[pair]);
        }

        for (uint32_t work = workBegin; work < workEnd; ++work) {
            uint32_t globalChunk = 0;
            uint32_t headPartition = 0;
            DecodeWorkItem(args_.tiling, work, globalChunk, headPartition);
            ChunkRange chunk{};
            if (!ResolveChunk(args_, globalChunk, chunk)) {
                continue;
            }
            uint32_t headBegin = 0;
            uint32_t headEnd = 0;
            HeadRange(args_.tiling, headPartition, headBegin, headEnd);
            for (uint32_t groupBegin = headBegin; groupBegin < headEnd;) {
                uint32_t activeHeads = headEnd - groupBegin;
                if (activeHeads > Shape::kHeadsPerGroup) {
                    activeHeads = Shape::kHeadsPerGroup;
                }
                // C2：一次 pair wait 汇聚两个 AIV，再消费该 pair 的两个 local head。
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                        kReadyFlagId[pair]);
                    for (uint32_t headInPair = 0; headInPair < 2; ++headInPair) {
                        const uint32_t localHead = pair * 2 + headInPair;
                        if (localHead < activeHeads) {
                            StageC2(chunk, localHead);
                        }
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        kFreeFlagId[pair]);
                }

                // C4 会把同一 L1 head lane 从 score 输入换义为 B；先确认
                // C2 的 MTE2 搬运全部结束，避免两批异步写访问重叠。
                AscendC::PipeBarrier<PIPE_MTE2>();

                // C4：读取 V3 的 B/X0/negX1，计算 T=B@X0 并常驻 L1。
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                        kReadyFlagId[pair]);
                    for (uint32_t headInPair = 0; headInPair < 2; ++headInPair) {
                        const uint32_t localHead = pair * 2 + headInPair;
                        if (localHead < activeHeads) {
                            StageC4(chunk, localHead);
                        }
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                        kFreeFlagId[pair]);
                }

                // C5：只在下半块存在时计算 Akk[32:M,0:32]=negX1@T。
                for (uint32_t localHead = 0; localHead < Shape::kHeadsPerGroup;
                     ++localHead) {
                    if (localHead < activeHeads) {
                        const uint32_t valueHead = groupBegin + localHead;
                        StageC5(chunk, valueHead, localHead);
                    }
                }

                // C7：每个 pair 一次调用；先搬完两个 head 的 RHS 并归还
                // workspace，再分别计算 W 与 U。
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                        kReadyFlagId[pair]);
                    StageC7(chunk, groupBegin, activeHeads, pair,
                            kFreeFlagId[pair]);
                }
                groupBegin += activeHeads;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(fixToM_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(rhsLaneFree_);
        ReleaseEvents();
    }

private:
    __aicore__ inline uint32_t ActiveSubChunks(uint32_t validRows) const
    {
        const uint32_t count = CeilDiv(validRows, Shape::kSubChunkRows);
        return count > Shape::kSubChunkCount ? Shape::kSubChunkCount : count;
    }

    __aicore__ inline void StageC2(const ChunkRange &chunk,
                                   uint32_t localHead)
    {
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride,
            Workspace::kArch22SlotStride);
        AscendC::GlobalTensor<bfloat16_t> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload));
        auto l1Bytes = l1Buf_.Get<uint8_t>();
        auto scoreL1 = l1Bytes[L1::kHeadLane[localHead]].template ReinterpretCast<bfloat16_t>();
        auto l0A = l0ABuf_.Get<bfloat16_t>();
        auto l0B = l0BBuf_.Get<bfloat16_t>();
        auto l0C = l0CBuf_.Get<float>();

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
        // C2 读入的 V1 payload 同样来自 AIV，按同一方式加固。
        AscendC::DataCacheCleanAndInvalid<bfloat16_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(payload);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
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
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(mte2ToMte1_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(mte2ToMte1_);

        const uint32_t active = ActiveSubChunks(chunk.validRows);
        AscendC::GlobalTensor<float> rawScoreRelay;
        rawScoreRelay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + slot + Workspace::kArch22CubeRelay),
            Workspace::kArch22RawScoreBytes / sizeof(float));
        for (uint32_t s = 0; s < active; ++s) {
            const uint32_t n = Shape::kPrefixRows[s];
            const uint32_t stackedBand = Shape::kSubChunkRows *
                                         Shape::kSubChunkRows * s * (s + 1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);

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
            AscendC::LoadData(
                l0A,
                scoreL1[ScorePayload::kQPlus / sizeof(bfloat16_t) +
                        s * kBf16FractalElements],
                loadQ);

            // c220 的 L0A 是 zZ，Kplus 的 16 行紧跟完整的 Qplus 行块。
            AscendC::LoadData(
                l0A[(Shape::kHeadDim / 16) * kBf16FractalElements],
                scoreL1[ScorePayload::kKPlus / sizeof(bfloat16_t) +
                        s * kBf16FractalElements],
                loadQ);

            // Kminus 的 ND [n,128] 经 Nd2Nz 后，分形顺序已经与
            // L0B 中数学上的 [128,n] 一致；不能再转置 16x16 分形。
            AscendC::LoadData2DParams loadKMinus{};
            loadKMinus.startIndex = 0;
            loadKMinus.repeatTimes =
                (Shape::kHeadDim / 16) * (n / 16);
            loadKMinus.srcStride = 1;
            loadKMinus.dstGap = 0;
            loadKMinus.ifTranspose = false;
            loadKMinus.sid = 0;
            loadKMinus.addrMode = 0;
            AscendC::LoadData(l0B,
                scoreL1[ScorePayload::kKMinus[s] / sizeof(bfloat16_t)], loadKMinus);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1ToM_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1ToM_);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(fixToM_);

            AscendC::MmadParams mmad{};
            mmad.m = 32;
            mmad.n = n;
            mmad.k = Shape::kHeadDim;
            mmad.cmatrixInitVal = true;
            mmad.cmatrixSource = false;
            mmad.unitFlag = 0;
            // 一次 MMAD 同时得到上半 rawAqk 和下半 rawAkk。
            AscendC::Mmad(l0C, l0A, l0B, mmad);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mToFix_);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mToFix_);
            uint32_t rows = chunk.validRows - s * Shape::kSubChunkRows;
            if (rows > Shape::kSubChunkRows) {
                rows = Shape::kSubChunkRows;
            }
            if (rows == Shape::kSubChunkRows) {
                auto fix = AscendC::FixpipeParamsV220(
                    n, 2 * Shape::kSubChunkRows,
                    2 * Shape::kSubChunkRows, n, false);
                fix.quantPre = QuantMode_t::NoQuant;
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    rawScoreRelay[stackedBand], l0C, fix);
            } else {
                auto fix = AscendC::FixpipeParamsV220(
                    n, rows, 2 * Shape::kSubChunkRows, n, false);
                fix.quantPre = QuantMode_t::NoQuant;
                // 尾 sub-chunk 只写有效的 Qplus/Kplus 行。FP32 L0C
                // 的基础分形为 16x16，下半 16 行从第二个 M1 分形开始。
                constexpr uint32_t kLowerM1Offset = 16 * 16;
                const uint32_t relayElements = rows * n;
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    rawScoreRelay[stackedBand], l0C, fix);
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    rawScoreRelay[stackedBand + relayElements],
                    l0C[kLowerM1Offset], fix);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixToM_);
        }
    }

    __aicore__ inline void StageC4(const ChunkRange &chunk,
                                   uint32_t localHead)
    {
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride,
            Workspace::kArch22SlotStride);
        auto l1Bytes = l1Buf_.Get<uint8_t>();
        auto akkL1 = l1Bytes[L1::kHeadLane[localHead] + Arch22L1::kAkkInHeadLane]
                         .template ReinterpretCast<float>();
        AscendC::GlobalTensor<float> akkSource;
        akkSource.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + slot + Workspace::kArch22AkkFp32));
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
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(mte2ToMte1_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(mte2ToMte1_);
            return;
        }

        AscendC::GlobalTensor<float> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + slot + Workspace::kPayload));
        auto bL1 = l1Bytes[L1::kHeadLane[localHead]].template ReinterpretCast<float>();
        auto x0L1 = l1Bytes[L1::kX0 + localHead * L1::kQuadrantStride]
                        .template ReinterpretCast<float>();
        auto negX1L1 = l1Bytes[L1::kNegX1 + localHead * L1::kQuadrantStride]
                           .template ReinterpretCast<float>();
        auto tL1 = l1Bytes[L1::kT + localHead * L1::kQuadrantStride]
                       .template ReinterpretCast<float>();
        AscendC::GlobalTensor<float> tRelay;
        tRelay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + slot + Workspace::kArch22TRelay),
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
        // C4 读入的 V3 产物（B/X0/negX1）同样来自 AIV，按同一方式加固。
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(payload);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::DataCopy(bL1, payload[Workspace::kB / sizeof(float)], copy);
        AscendC::DataCopy(x0L1, payload[Workspace::kX0 / sizeof(float)], copy);
        AscendC::DataCopy(negX1L1, payload[Workspace::kNegX1 / sizeof(float)], copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(mte2ToMte1_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(mte2ToMte1_);

        // T[32,32] = B[32,32] @ X0[32,32]。
        auto l0A = l0ABuf_.Get<float>();
        auto l0B = l0BBuf_.Get<float>();
        auto l0C = l0CBuf_.Get<float>();
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
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
        for (uint32_t rowFractal = 0; rowFractal < kFp32RowFractals;
             ++rowFractal) {
            AscendC::LoadData(
                l0A[rowFractal * kFp32ColumnFractals *
                     kFp32FractalElements],
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
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(fixToM_);
        AscendC::SetHF32Mode(false);
        AscendC::MmadParams tMmad{};
        tMmad.m = 32;
        tMmad.n = 32;
        tMmad.k = 32;
        tMmad.cmatrixInitVal = true;
        tMmad.cmatrixSource = false;
        tMmad.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, tMmad);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mToFix_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mToFix_);

        auto fix = AscendC::FixpipeParamsV220(32, 32, 32, 32, false);
        fix.quantPre = QuantMode_t::NoQuant;
        fix.isChannelSplit = true;
        // C220 不支持 FP32 L0C 直写 L1：先按标准 FP32 16x8 NZ 分形写 GM，
        // 再原样搬回 L1 供 C5 消费。
        AscendC::Fixpipe<float, float, kFixpipeNz>(
            tRelay, l0C, fix);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixToM_);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(fixToMte2_[localHead]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(fixToMte2_[localHead]);
        AscendC::DataCopy(tL1, tRelay, AscendC::DataCopyParams(1, 128, 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(tReady_[localHead]);
    }

    __aicore__ inline void StageC5(const ChunkRange &chunk, uint32_t valueHead,
                                   uint32_t localHead)
    {
        if (chunk.validRows <= 32) {
            return;
        }
        auto l1Bytes = l1Buf_.Get<uint8_t>();
        auto negX1L1 = l1Bytes[L1::kNegX1 + localHead * L1::kQuadrantStride]
                           .template ReinterpretCast<float>();
        auto tL1 = l1Bytes[L1::kT + localHead * L1::kQuadrantStride]
                       .template ReinterpretCast<float>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(tReady_[localHead]);

        // Akk[32:M,0:32] = negX1[32:M,0:32] @ T[32,32]。
        auto l0A = l0ABuf_.Get<float>();
        auto l0B = l0BBuf_.Get<float>();
        auto l0C = l0CBuf_.Get<float>();
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
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
        for (uint32_t rowFractal = 0; rowFractal < kFp32RowFractals;
             ++rowFractal) {
            AscendC::LoadData(
                l0A[rowFractal * kFp32ColumnFractals *
                     kFp32FractalElements],
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
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1ToM_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(fixToM_);
        AscendC::SetHF32Mode(false);
        AscendC::MmadParams akkMmad{};
        akkMmad.m = 32;
        akkMmad.n = 32;
        akkMmad.k = 32;
        akkMmad.cmatrixInitVal = true;
        akkMmad.cmatrixSource = false;
        akkMmad.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, akkMmad);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mToFix_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mToFix_);

        const uint32_t bottomRows = chunk.validRows - 32;
        // FP32 结果经 GM ND 转为 L1 NZ，保留 V3 已写入的两个对角块。
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride,
            Workspace::kArch22SlotStride);
        AscendC::GlobalTensor<float> akkSource;
        akkSource.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + slot + Workspace::kArch22AkkFp32));
        auto l1Fix = AscendC::FixpipeParamsV220(32, 32, 32, 64, false);
        auto akkL1 = l1Bytes[L1::kHeadLane[localHead] + Arch22L1::kAkkInHeadLane]
                         .template ReinterpretCast<float>();
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
            akkSource[32 * 64], l0C, l1Fix);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(fixToMte2_[localHead]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(fixToMte2_[localHead]);
        AscendC::Nd2NzParams akkCopy{};
        akkCopy.ndNum = 1;
        akkCopy.nValue = 64;
        akkCopy.dValue = 64;
        akkCopy.srcDValue = 64;
        akkCopy.dstNzNStride = 1;
        akkCopy.dstNzC0Stride = 64;
        AscendC::DataCopy(akkL1, akkSource, akkCopy);
        if constexpr (CompilePolicy::outputAkk) {
            AscendC::GlobalTensor<bfloat16_t> akkOutput;
            akkOutput.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t *>(args_.akk) +
                    AOutputOffset(args_.tiling, chunk, valueHead) +
                    32 * Shape::kChunkRows,
                bottomRows * Shape::kChunkRows);
            auto outputFix = AscendC::FixpipeParamsV220(
                32, bottomRows, 32, Shape::kChunkRows, false);
            outputFix.quantPre = QuantMode_t::F322BF16;
            AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(
                akkOutput, l0C, outputFix);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(tReady_[localHead]);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixToM_);
    }

    __aicore__ inline void StageC7(const ChunkRange &chunk,
                                   uint32_t groupBegin,
                                   uint32_t activeHeads,
                                   uint32_t pair,
                                   uint16_t freeFlagId)
    {
        const uint32_t m = chunk.validRows > 32 ? 64 : 32;
        auto l1Bytes = l1Buf_.Get<uint8_t>();
        // 两个 pair 复用两条 FP32 lane；等待上一 pair 的最后一次 MTE1 读取。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(rhsLaneFree_);
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = m;
        copy.dValue = Shape::kHeadDim;
        copy.srcDValue = Shape::kHeadDim;
        copy.dstNzNStride = 1;
        copy.dstNzC0Stride = m;
        for (uint32_t headInPair = 0; headInPair < 2; ++headInPair) {
            const uint32_t localHead = pair * 2 + headInPair;
            if (localHead >= activeHeads) {
                continue;
            }
            const uint64_t slot = WorkspaceSlotBase(
                workgroup_, localHead, Workspace::kArch22WorkgroupStride,
                Workspace::kArch22SlotStride);
            auto kBetaL1 = l1Bytes[Arch22L1::kRhsLane[headInPair]]
                               .template ReinterpretCast<float>();
            auto vBetaL1 = l1Bytes[Arch22L1::kRhsLane[headInPair] + Arch22L1::kRhsVBeta]
                               .template ReinterpretCast<float>();
            AscendC::GlobalTensor<float> kBetaRelay, vBetaRelay;
            kBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                args_.workspace + slot + Workspace::kArch22RhsKBetaG));
            vBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                args_.workspace + slot + Workspace::kArch22RhsVBeta));
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                              AscendC::DcciDst::CACHELINE_OUT>(kBetaRelay);
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                              AscendC::DcciDst::CACHELINE_OUT>(vBetaRelay);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            AscendC::DataCopy(kBetaL1, kBetaRelay, copy);
            AscendC::DataCopy(vBetaL1, vBetaRelay, copy);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(mte2ToMte1_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(mte2ToMte1_);
        // 两个 RHS 均已离开 GM，维持原有的提前归还协议。
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(freeFlagId);

        for (uint32_t headInPair = 0; headInPair < 2; ++headInPair) {
            const uint32_t localHead = pair * 2 + headInPair;
            if (localHead >= activeHeads) {
                continue;
            }
            if (chunk.validRows > 32) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(tReady_[localHead]);
            }
            auto akkL1 = l1Bytes[L1::kHeadLane[localHead] + Arch22L1::kAkkInHeadLane]
                             .template ReinterpretCast<float>();
            auto l0A = l0ABuf_.Get<float>();
            auto l0B = l0BBuf_.Get<float>();
            auto l0C = l0CBuf_.Get<float>();
            const uint64_t out = HeadTensorOffset(
                args_.tiling, chunk, groupBegin + localHead, Shape::kHeadDim);
            // FP32 NZ 的基本块为 16x8；A 的 K 维为 32/64，M 维同样为 32/64。
            AscendC::LoadData2DParams loadA{};
            loadA.repeatTimes = m / 8;
            loadA.srcStride = Shape::kChunkRows / 16;
            AscendC::LoadData3DParamsV2<float> loadB{};
            loadB.l1H = 1;
            loadB.l1W = m;
            loadB.channelSize = Shape::kHeadDim;
            // Load3D 的 K 指源 channel（本次 GEMM 的 N），M 指源 HW（GEMM 的 K）。
            loadB.kExtension = Shape::kHeadDim;
            loadB.mExtension = m;
            loadB.strideW = 1;
            loadB.strideH = 1;
            loadB.filterW = 1;
            loadB.filterH = 1;
            loadB.dilationFilterW = 1;
            loadB.dilationFilterH = 1;
            loadB.enTranspose = true;
            AscendC::SetHF32Mode(false);
            for (uint32_t rhs = 0; rhs < 2; ++rhs) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
                for (uint32_t row = 0; row < m / 16; ++row) {
                    AscendC::LoadData(l0A[row * (m / 8) * 128],
                                      akkL1[row * 128], loadA);
                }
                auto rhsL1 = l1Bytes[Arch22L1::kRhsLane[headInPair] + rhs * Arch22L1::kRhsVBeta]
                                 .template ReinterpretCast<float>();
                AscendC::LoadData(l0B, rhsL1, loadB);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1ToM_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1ToM_);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(fixToM_);
                AscendC::MmadParams mm{};
                mm.m = m;
                mm.n = Shape::kHeadDim;
                mm.k = m;
                mm.cmatrixInitVal = true;
                AscendC::Mmad(l0C, l0A, l0B, mm);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mToMte1_);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mToFix_);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mToFix_);
                auto fix = AscendC::FixpipeParamsV220(
                    Shape::kHeadDim, chunk.validRows, m, Shape::kHeadDim, false);
                fix.quantPre = QuantMode_t::F322BF16;
                if (rhs == 0) {
                    AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(wGm_[out], l0C, fix);
                } else {
                    AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(uGm_[out], l0C, fix);
                }
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixToM_);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(rhsLaneFree_);
    }

    __aicore__ inline void ReleaseEvents()
    {
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(mte2ToMte1_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_M>(mte1ToM_);
        pipe_->ReleaseEventID<AscendC::HardEvent::M_MTE1>(mToMte1_);
        pipe_->ReleaseEventID<AscendC::HardEvent::M_FIX>(mToFix_);
        pipe_->ReleaseEventID<AscendC::HardEvent::FIX_M>(fixToM_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(rhsLaneFree_);
        for (uint32_t head = 0; head < Shape::kHeadsPerGroup; ++head) {
            pipe_->ReleaseEventID<AscendC::HardEvent::FIX_MTE2>(fixToMte2_[head]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(tReady_[head]);
            pipe_->ReleaseEventID<AscendC::HardEvent::FIX_MTE1>(fixToMte1_[head]);
        }
    }

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
    AscendC::TEventID fixToMte2_[Shape::kHeadsPerGroup]{};
    AscendC::TEventID tReady_[Shape::kHeadsPerGroup]{};
    AscendC::TEventID fixToMte1_[Shape::kHeadsPerGroup]{};
    AscendC::TEventID rhsLaneFree_{};
    AscendC::GlobalTensor<bfloat16_t> wGm_{};
    AscendC::GlobalTensor<bfloat16_t> uGm_{};
};

} // namespace KdaPrepare::Arch22

#endif // ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H
