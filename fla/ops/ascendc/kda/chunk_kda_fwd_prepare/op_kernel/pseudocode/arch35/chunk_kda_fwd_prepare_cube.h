/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_ARCH35_CHUNK_KDA_FWD_PREPARE_CUBE_H
#define PSEUDOCODE_ARCH35_CHUNK_KDA_FWD_PREPARE_CUBE_H

#include <cstdint>
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"

#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace KdaPrepare::Arch35 {

constexpr AscendC::FixpipeConfig kFixpipeRowMajorUb = {
    AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig kFixpipeNzL1 = {
    AscendC::CO2Layout::NZ, false};

class ChunkKdaFwdPrepareCube {
public:
    __aicore__ inline void Init(const PrepareKernelArgs &args)
    {
        args_ = args;
        workgroup_ = WorkgroupId();
        akkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.akk));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.w));
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.u));
    }

    __aicore__ inline void Process()
    {
        if (args_.tiling.usedCoreNum == 0) {
            return;
        }

        // AIC 视角下四个 local head 的固定核间编号。AIV1 的两个本地
        // ready 0/1、free 4/5 在 AIC 侧映射为 16/17、20/21。
        constexpr uint16_t kAivToAicPayloadReadyFlagId[4] = {
            0, 1, 16, 17};
        constexpr uint16_t kAicToAivSlotReusableFlagId[4] = {
            4, 5, 20, 21};
        // free[localHead] 初始只发布一次。以后每一组的 V0 会消费上一组
        // C7 发布的 free，不能在组首重复 set 同一个计数器。
        bool freeInitialized[Shape::kHeadsPerGroup] = {false, false, false, false};
        if (workgroup_ >= args_.tiling.usedCoreNum) {
            return;
        }
        const uint32_t total = TotalWorkItems(args_.tiling);
        const uint32_t workBegin = WorkBegin(
            total, workgroup_, args_.tiling.usedCoreNum);
        const uint32_t workEnd = WorkEnd(
            total, workgroup_, args_.tiling.usedCoreNum);
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
            for (uint32_t groupBegin = headBegin; groupBegin < headEnd;
                 groupBegin += Shape::kHeadsPerGroup) {
                for (uint32_t localHead = 0;
                     localHead < Shape::kHeadsPerGroup; ++localHead) {
                    if (groupBegin + localHead >= headEnd ||
                        freeInitialized[localHead]) {
                        continue;
                    }
                    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
                        kAicToAivSlotReusableFlagId[localHead]);
                    freeInitialized[localHead] = true;
                }

                // C2：逐 HEAD 等 V1 的 72 KiB 分数操作数。一次装入 L1 后，
                // 尾块只提交有效 sub-chunk 的 32x128 @ 128xN MMAD。
                for (uint32_t localHead = 0;
                     localHead < Shape::kHeadsPerGroup; ++localHead) {
                    if (groupBegin + localHead >= headEnd) {
                        continue;
                    }
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                        kAivToAicPayloadReadyFlagId[localHead]);
                    StageC2(chunk, localHead);
                    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
                        kAicToAivSlotReusableFlagId[localHead]);
                }

                // 先为所有有效 HEAD 提交 C4，再统一提交 C5。若逐 HEAD 交替
                // 提交 C4/C5，C5 的 MTE1 会等待同 HEAD 的 C4 Fixpipe 写完 T，
                // 并阻塞后续 HEAD 的独立 C4；两轮提交允许下一 HEAD 的
                // MTE1/MMAD 与上一 HEAD 的 Fixpipe 排空重叠。
                // C4 一次性读完 B/X0/negX1/Akk 后立即归还 workspace payload，
                // 后续 C4/C5 只访问每 HEAD 独立的 L1 常驻数据。
                for (uint32_t localHead = 0;
                     localHead < Shape::kHeadsPerGroup; ++localHead) {
                    const uint32_t valueHead = groupBegin + localHead;
                    if (valueHead >= headEnd) {
                        continue;
                    }
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                        kAivToAicPayloadReadyFlagId[localHead]);
                    StageC4(chunk, localHead,
                            kAicToAivSlotReusableFlagId[localHead]);
                }
                for (uint32_t localHead = 0;
                     localHead < Shape::kHeadsPerGroup; ++localHead) {
                    const uint32_t valueHead = groupBegin + localHead;
                    if (valueHead >= headEnd) {
                        continue;
                    }
                    StageC5(chunk, valueHead, localHead);
                }

                // C7：V6 已将两个 RHS 平面写入 workspace。Akk 继续使用
                // C4/C5 的 L1 常驻副本，分别计算 W 和 U，最后归还槽位。
                for (uint32_t localHead = 0;
                     localHead < Shape::kHeadsPerGroup; ++localHead) {
                    const uint32_t valueHead = groupBegin + localHead;
                    if (valueHead >= headEnd) {
                        continue;
                    }
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                        kAivToAicPayloadReadyFlagId[localHead]);
                    StageC7(chunk, valueHead, localHead,
                            kAicToAivSlotReusableFlagId[localHead]);
                }
            }
        }
    }

private:
    __aicore__ inline void StageC2(const ChunkRange &chunk,
                                   uint32_t localHead)
    {
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch35WorkgroupStride);
        const uint32_t lane = L1::kHeadLane[localHead];
        const uint8_t l1Mutex = static_cast<uint8_t>(localHead); // 0..3
        const uint8_t operandMutex = 4;
        const uint8_t l0cMutex = static_cast<uint8_t>(5 + localHead); // 5..8
        const uint8_t ownerAiv = static_cast<uint8_t>(localHead / 2);
        const uint32_t localSlot = localHead % 2;
        const uint32_t l0cLane = localHead * 64 * 1024;

        AscendC::GlobalTensor<bfloat16_t> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload));
        auto scoreL1 =
            resource_.l1Buf.template GetBufferByByte<bfloat16_t>(lane);

        // Stage 入口把 Qplus、Kplus 和四个 Kminus 前缀一次性搬完，共 72 KiB。
        // 这里的 DataCopy 是 GM(ND)->L1(NZ)，后面的四次 MMAD 不再读 GM。
        AscendC::Mutex::Lock<PIPE_MTE2>(l1Mutex);
        AscendC::Nd2NzParams qkCopy{};
        qkCopy.ndNum = 1;
        qkCopy.nValue = Shape::kChunkRows;
        qkCopy.dValue = Shape::kHeadDim;
        qkCopy.srcDValue = Shape::kHeadDim;
        qkCopy.srcNdMatrixStride = 0;
        qkCopy.dstNzNStride = 1;
        qkCopy.dstNzC0Stride = Shape::kChunkRows;
        qkCopy.dstNzMatrixStride = 0;
        // TODO：用目标 CANN 9.1.0 头文件确认 BF16 的 NZ C0 与
        // dstNzC0Stride 单位；确认前不得把下列参数复制到正式 kernel。
        AscendC::DataCopy(
            scoreL1[ScorePayload::kQPlus / sizeof(bfloat16_t)],
            payload[ScorePayload::kQPlus / sizeof(bfloat16_t)], qkCopy);
        AscendC::DataCopy(
            scoreL1[ScorePayload::kKPlus / sizeof(bfloat16_t)],
            payload[ScorePayload::kKPlus / sizeof(bfloat16_t)], qkCopy);
        for (uint32_t s = 0; s < Shape::kSubChunkCount; ++s) {
            AscendC::Nd2NzParams prefixCopy = qkCopy;
            prefixCopy.nValue = Shape::kPrefixRows[s];
            prefixCopy.dstNzC0Stride = Shape::kPrefixRows[s];
            AscendC::DataCopy(
                scoreL1[ScorePayload::kKMinus[s] / sizeof(bfloat16_t)],
                payload[ScorePayload::kKMinus[s] / sizeof(bfloat16_t)],
                prefixCopy);
        }
        AscendC::Mutex::Unlock<PIPE_MTE2>(l1Mutex);

        auto stackedQkL0 =
            resource_.l0ABuf.template GetBufferByByte<bfloat16_t>(0);
        auto kMinusL0 =
            resource_.l0BBuf.template GetBufferByByte<bfloat16_t>(0);
        // 四个 HEAD 各占一条 64 KiB L0C 通道，和 Mutex 5..8/9..12
        // 一一对应；不同 Mutex 保护的语义不能落到同一物理地址。
        // TODO：正式实现前必须用目标 A5 头文件和最小 kernel 确认该资源
        // 确实暴露 256 KiB L0C；容量不足时要改为同一 Mutex 下顺序复用。
        auto rawL0c =
            resource_.l0CBuf.template GetBufferByByte<float>(l0cLane);
        auto rawScoreUb = resource_.ubBuf.template GetBufferByByte<float>(
            Arch35Ub::kComputeSlotBase[localSlot] + Arch35Ub::kRawScore);

        uint32_t stackedElements = 0;
        const uint32_t active = CeilDiv(
            chunk.validRows, Shape::kSubChunkRows);
        for (uint32_t s = 0; s < active; ++s) {
            const uint32_t n = Shape::kPrefixRows[s];

            AscendC::Mutex::Lock<PIPE_MTE1>(l1Mutex);
            AscendC::Mutex::Lock<PIPE_MTE1>(operandMutex);
            AscendC::LoadData2DParamsV2 loadQ{};
            loadQ.mStartPosition = s;
            loadQ.kStartPosition = 0;
            loadQ.mStep = 1;
            loadQ.kStep = Shape::kHeadDim / 16;
            loadQ.srcStride = Shape::kChunkRows / 16;
            loadQ.dstStride = 2;
            loadQ.ifTranspose = false;
            loadQ.sid = 0;
            AscendC::LoadData(
                stackedQkL0,
                scoreL1[ScorePayload::kQPlus / sizeof(bfloat16_t)], loadQ);
            // TODO：目标版本最小编译确认 zN L0A 的下半 16 行物理偏移；
            // 语义必须是 [Qplus_s; Kplus_s]，不能在 L1 内重排。
            AscendC::LoadData(
                stackedQkL0[Shape::kSubChunkRows * Shape::kHeadDim],
                scoreL1[ScorePayload::kKPlus / sizeof(bfloat16_t)], loadQ);

            AscendC::LoadData2DParamsV2 loadKMinus{};
            loadKMinus.mStartPosition = 0;
            loadKMinus.kStartPosition = 0;
            loadKMinus.mStep = Shape::kHeadDim / 16;
            loadKMinus.kStep = n / 16;
            loadKMinus.srcStride = n / 16;
            loadKMinus.dstStride = Shape::kHeadDim / 16;
            loadKMinus.ifTranspose = true;
            loadKMinus.sid = 0;
            // TODO：确认 Kminus 的 L1 NZ 到 L0B 转置装载中 mStep、
            // kStep、srcStride、dstStride 的分形单位。
            AscendC::LoadData(
                kMinusL0,
                scoreL1[ScorePayload::kKMinus[s] / sizeof(bfloat16_t)],
                loadKMinus);
            AscendC::Mutex::Unlock<PIPE_MTE1>(operandMutex);
            AscendC::Mutex::Unlock<PIPE_MTE1>(l1Mutex);

            AscendC::Mutex::Lock<PIPE_M>(operandMutex);
            AscendC::Mutex::Lock<PIPE_M>(l0cMutex);
            AscendC::MmadParams mmad{};
            mmad.m = 2 * Shape::kSubChunkRows;
            mmad.n = n;
            mmad.k = Shape::kHeadDim;
            mmad.cmatrixInitVal = true;
            mmad.cmatrixSource = false;
            mmad.unitFlag = 0;
            // 一次 API 提交包含两个数学乘积：
            // 上 16 行 Qplus_s@Kminus_s^T，下 16 行 Kplus_s@Kminus_s^T。
            AscendC::Mmad(rawL0c, stackedQkL0, kMinusL0, mmad);
            AscendC::Mutex::Unlock<PIPE_M>(l0cMutex);
            AscendC::Mutex::Unlock<PIPE_M>(operandMutex);

            AscendC::Mutex::Lock<PIPE_FIX>(l0cMutex);
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fix{};
            fix.nSize = n;
            fix.mSize = 2 * Shape::kSubChunkRows;
            fix.srcStride = 2 * Shape::kSubChunkRows;
            fix.dstStride = n;
            fix.quantPre = QuantMode_t::NoQuant;
            fix.reluEn = false;
            fix.unitFlag = 0;
            fix.dualDstCtl = 0;
            fix.subBlockId = ownerAiv;
            AscendC::Fixpipe<float, float, kFixpipeRowMajorUb>(
                rawScoreUb[stackedElements], rawL0c, fix);
            AscendC::Mutex::Unlock<PIPE_FIX>(l0cMutex);

            stackedElements += 2 * Shape::kSubChunkRows * n;
        }
    }

    __aicore__ inline void StageC4(const ChunkRange &chunk,
                                   uint32_t localHead,
                                   uint16_t slotReusableFlagId)
    {
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch35WorkgroupStride);
        const uint32_t lane = L1::kHeadLane[localHead];
        const uint8_t l1Mutex = static_cast<uint8_t>(localHead); // 0..3
        const uint8_t operandMutex = 4;
        const uint8_t l0cMutex = static_cast<uint8_t>(5 + localHead); // 5..8
        const uint32_t l0cLane = localHead * 64 * 1024;

        AscendC::GlobalTensor<float> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + slot + Workspace::kPayload));
        AscendC::GlobalTensor<bfloat16_t> akkSource;
        akkSource.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload + Workspace::kAkk));

        auto bL1 = resource_.l1Buf.template GetBufferByByte<float>(lane);
        auto x0L1 = resource_.l1Buf.template GetBufferByByte<float>(
            L1::kX0 + localHead * L1::kQuadrantStride);
        auto negX1L1 = resource_.l1Buf.template GetBufferByByte<float>(
            L1::kNegX1 + localHead * L1::kQuadrantStride);
        auto tL1 = resource_.l1Buf.template GetBufferByByte<float>(
            L1::kT + localHead * L1::kQuadrantStride);
        auto akkL1 = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(
            L1::kAkk + localHead * L1::kAkkStride);

        // V3 写出的所有后续 Cube 输入只在这里搬一次。即使 M<=32，Akk
        // 仍须进入最终 L1 常驻地址，C7 直接复用；B/X0/negX1 则无需搬入。
        AscendC::Mutex::Lock<PIPE_MTE2>(l1Mutex);
        AscendC::Nd2NzParams akkCopy{};
        akkCopy.ndNum = 1;
        akkCopy.nValue = Shape::kChunkRows;
        akkCopy.dValue = Shape::kChunkRows;
        akkCopy.srcDValue = Shape::kChunkRows;
        akkCopy.srcNdMatrixStride = 0;
        akkCopy.dstNzNStride = 1;
        akkCopy.dstNzC0Stride = Shape::kChunkRows;
        akkCopy.dstNzMatrixStride = 0;
        // TODO：确认 64x64 BF16 ND->NZ 后四个 32x32 象限的物理次序，
        // C5 必须能直接写 q10，C7 必须能直接按 64x64 装入 L0A。
        AscendC::DataCopy(akkL1, akkSource, akkCopy);

        if (chunk.validRows > 32) {
            AscendC::Nd2NzParams quadrantCopy{};
            quadrantCopy.ndNum = 1;
            quadrantCopy.nValue = 32;
            quadrantCopy.dValue = 32;
            quadrantCopy.srcDValue = 32;
            quadrantCopy.srcNdMatrixStride = 0;
            quadrantCopy.dstNzNStride = 1;
            quadrantCopy.dstNzC0Stride = 32;
            quadrantCopy.dstNzMatrixStride = 0;
            AscendC::DataCopy(
                bL1, payload[Workspace::kB / sizeof(float)], quadrantCopy);
            AscendC::DataCopy(
                x0L1, payload[Workspace::kX0 / sizeof(float)], quadrantCopy);
            AscendC::DataCopy(
                negX1L1,
                payload[Workspace::kNegX1 / sizeof(float)], quadrantCopy);
        }
        AscendC::Mutex::Unlock<PIPE_MTE2>(l1Mutex);

        // 载荷已完全离开 workspace，立即允许 AIV 在同一地址生成 V6 RHS。
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE2>(
            slotReusableFlagId);
        if (chunk.validRows <= 32) {
            return;
        }

        auto l0A = resource_.l0ABuf.template GetBufferByByte<float>(0);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<float>(0);
        auto l0C =
            resource_.l0CBuf.template GetBufferByByte<float>(l0cLane);

        AscendC::Mutex::Lock<PIPE_MTE1>(l1Mutex);
        AscendC::Mutex::Lock<PIPE_MTE1>(operandMutex);
        AscendC::LoadData2DParamsV2 load{};
        load.mStartPosition = 0;
        load.kStartPosition = 0;
        load.mStep = 2;
        load.kStep = 4;
        load.srcStride = 2;
        load.dstStride = 2;
        load.ifTranspose = false;
        load.sid = 0;
        // TODO：确认 Arch35 FP32 NZ 的 C0=8 以及四个步长字段的单位。
        AscendC::LoadData(l0A, bL1, load);
        load.ifTranspose = true;
        AscendC::LoadData(l0B, x0L1, load);
        AscendC::Mutex::Unlock<PIPE_MTE1>(operandMutex);
        AscendC::Mutex::Unlock<PIPE_MTE1>(l1Mutex);

        AscendC::Mutex::Lock<PIPE_M>(operandMutex);
        AscendC::Mutex::Lock<PIPE_M>(l0cMutex);
        AscendC::SetHF32Mode(false);
        AscendC::MmadParams mmad{};
        mmad.m = 32;
        mmad.n = 32;
        mmad.k = 32;
        mmad.cmatrixInitVal = true;
        mmad.cmatrixSource = false;
        mmad.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, mmad); // 计算 T=B@X0。
        AscendC::Mutex::Unlock<PIPE_M>(l0cMutex);
        AscendC::Mutex::Unlock<PIPE_M>(operandMutex);

        AscendC::Mutex::Lock<PIPE_FIX>(l0cMutex);
        AscendC::Mutex::Lock<PIPE_FIX>(l1Mutex);
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fix{};
        fix.nSize = 32;
        fix.mSize = 32;
        fix.srcStride = 32;
        fix.dstStride = 32 * 16;
        fix.quantPre = QuantMode_t::NoQuant;
        fix.isChannelSplit = false;
        // TODO：用目标头文件确认 NZ 到 L1 的 dstStride 分形单位。
        AscendC::Fixpipe<float, float, kFixpipeNzL1>(tL1, l0C, fix);
        AscendC::Mutex::Unlock<PIPE_FIX>(l1Mutex);
        AscendC::Mutex::Unlock<PIPE_FIX>(l0cMutex);
    }

    __aicore__ inline void StageC5(const ChunkRange &chunk,
                                    uint32_t valueHead,
                                    uint32_t localHead)
    {
        if (chunk.validRows <= 32) {
            return;
        }

        const uint8_t l1Mutex = static_cast<uint8_t>(localHead); // 0..3
        const uint8_t operandMutex = 4;
        const uint8_t l0cMutex = static_cast<uint8_t>(5 + localHead); // 5..8
        const uint32_t l0cLane = localHead * 64 * 1024;
        auto negX1L1 = resource_.l1Buf.template GetBufferByByte<float>(
            L1::kNegX1 + localHead * L1::kQuadrantStride);
        auto tL1 = resource_.l1Buf.template GetBufferByByte<float>(
            L1::kT + localHead * L1::kQuadrantStride);
        auto akkL1 = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(
            L1::kAkk + localHead * L1::kAkkStride);
        auto l0A = resource_.l0ABuf.template GetBufferByByte<float>(0);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<float>(0);
        // C5 使用当前 HEAD 通道的第二个 4 KiB 区域，避免和 C4 的 T
        // Fixpipe 源发生物理重叠。
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(
            l0cLane + 4 * 1024);

        AscendC::Mutex::Lock<PIPE_MTE1>(l1Mutex);
        AscendC::Mutex::Lock<PIPE_MTE1>(operandMutex);
        AscendC::LoadData2DParamsV2 load{};
        load.mStartPosition = 0;
        load.kStartPosition = 0;
        load.mStep = 2;
        load.kStep = 4;
        load.srcStride = 2;
        load.dstStride = 2;
        load.ifTranspose = false;
        load.sid = 0;
        AscendC::LoadData(l0A, negX1L1, load);
        load.ifTranspose = true;
        AscendC::LoadData(l0B, tL1, load);
        AscendC::Mutex::Unlock<PIPE_MTE1>(operandMutex);
        AscendC::Mutex::Unlock<PIPE_MTE1>(l1Mutex);

        AscendC::Mutex::Lock<PIPE_M>(operandMutex);
        AscendC::Mutex::Lock<PIPE_M>(l0cMutex);
        AscendC::SetHF32Mode(false);
        AscendC::MmadParams mmad{};
        mmad.m = 32;
        mmad.n = 32;
        mmad.k = 32;
        mmad.cmatrixInitVal = true;
        mmad.cmatrixSource = false;
        mmad.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, mmad); // 计算 Akk 的 q10=negX1@T。
        AscendC::Mutex::Unlock<PIPE_M>(l0cMutex);
        AscendC::Mutex::Unlock<PIPE_M>(operandMutex);

        AscendC::Mutex::Lock<PIPE_FIX>(l0cMutex);
        AscendC::Mutex::Lock<PIPE_FIX>(l1Mutex);
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> l1Fix{};
        l1Fix.nSize = 32;
        l1Fix.mSize = 32;
        l1Fix.srcStride = 32;
        l1Fix.dstStride = Shape::kChunkRows * 16;
        l1Fix.isChannelSplit = false;
        l1Fix.quantPre = QuantMode_t::F322BF16;
        // q10 在两字节 NZ 中从 (N1=0,M1=2) 开始，即 32*16 个元素。
        // dstStride=64*16 个元素跨过完整 64 行 M 轴。
        AscendC::Fixpipe<bfloat16_t, float, kFixpipeNzL1>(
            akkL1[L1::kAkkQ10Elements], l0C, l1Fix);
        AscendC::Mutex::Unlock<PIPE_FIX>(l1Mutex);

        const uint32_t bottomRows = chunk.validRows - 32;
        auto gmFix = AscendC::FixpipeParamsV220(
            32, bottomRows, 32, Shape::kChunkRows, false);
        gmFix.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(
            akkGm_[AOutputOffset(args_.tiling, chunk, valueHead) +
                   32 * Shape::kChunkRows],
            l0C, gmFix);
        // V3 已写回 q00/q01/q11；本次 q10 Fixpipe 完成后，公开 Akk
        // 的四个象限全部为最终值，不需要再从 GM 回读或重复 MMAD。
        AscendC::Mutex::Unlock<PIPE_FIX>(l0cMutex);
    }

    __aicore__ inline void StageC7(const ChunkRange &chunk,
                                    uint32_t valueHead,
                                    uint32_t localHead,
                                    uint16_t slotReusableFlagId)
    {
        const uint32_t m = chunk.validRows > 32 ? 64 : 32;
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch35WorkgroupStride);
        const uint32_t lane = L1::kHeadLane[localHead];
        const uint8_t l1Mutex = static_cast<uint8_t>(localHead); // 0..3
        const uint8_t operandMutex = 4;
        const uint8_t wL0cMutex =
            static_cast<uint8_t>(5 + localHead); // 5..8
        const uint8_t uL0cMutex =
            static_cast<uint8_t>(9 + localHead); // 9..12
        const uint32_t l0cLane = localHead * 64 * 1024;

        AscendC::GlobalTensor<bfloat16_t> kBetaRelay;
        AscendC::GlobalTensor<bfloat16_t> vBetaRelay;
        kBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload +
            Workspace::kKBetaG));
        vBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload +
            Workspace::kVBeta));

        auto akkL1 = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(
            L1::kAkk + localHead * L1::kAkkStride);
        auto kBetaL1 =
            resource_.l1Buf.template GetBufferByByte<bfloat16_t>(lane);
        auto vBetaL1 = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(
            lane + Shape::kBf16MatrixBytes);

        // 两个 RHS 平面各搬一次；它们在 L1 中保持独立平面，不拼成
        // [M,256]，也不在 L1 内移动。
        AscendC::Mutex::Lock<PIPE_MTE2>(l1Mutex);
        AscendC::Nd2NzParams rhsCopy{};
        rhsCopy.ndNum = 1;
        rhsCopy.nValue = m;
        rhsCopy.dValue = Shape::kHeadDim;
        rhsCopy.srcDValue = Shape::kHeadDim;
        rhsCopy.srcNdMatrixStride = 0;
        rhsCopy.dstNzNStride = 1;
        rhsCopy.dstNzC0Stride = m;
        rhsCopy.dstNzMatrixStride = 0;
        // 两个 RHS 均固定为 BF16，第二平面按一个完整 BF16
        // 矩阵的字节数偏移。
        AscendC::DataCopy(kBetaL1, kBetaRelay, rhsCopy);
        AscendC::DataCopy(vBetaL1, vBetaRelay, rhsCopy);
        AscendC::Mutex::Unlock<PIPE_MTE2>(l1Mutex);
        // 两个 RHS 已完整进入 L1，后续 Cube 不再读取 workspace；立即归还
        // 当前 slot，使下一组 V0/V1 与本组 C7 的 MMAD/Fixpipe 重叠。
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE2>(
            slotReusableFlagId);

        auto akkL0 = resource_.l0ABuf.template GetBufferByByte<bfloat16_t>(0);
        auto kBetaL0 =
            resource_.l0BBuf.template GetBufferByByte<bfloat16_t>(0);
        auto vBetaL0 = resource_.l0BBuf.template GetBufferByByte<bfloat16_t>(
            Shape::kBf16MatrixBytes);
        auto wL0c =
            resource_.l0CBuf.template GetBufferByByte<float>(l0cLane);
        auto uL0c = resource_.l0CBuf.template GetBufferByByte<float>(
            l0cLane + 32 * 1024);

        AscendC::Mutex::Lock<PIPE_MTE1>(l1Mutex);
        AscendC::Mutex::Lock<PIPE_MTE1>(operandMutex);
        AscendC::LoadData2DParamsV2 loadA{};
        loadA.mStartPosition = 0;
        loadA.kStartPosition = 0;
        loadA.mStep = m / 16;
        loadA.kStep = m / 16;
        loadA.srcStride = Shape::kChunkRows / 16;
        loadA.dstStride = m / 16;
        loadA.ifTranspose = false;
        loadA.sid = 0;
        // TODO：确认 C4/C5 的 q00/q01/q10/q11 NZ 常驻布局可由这一次
        // LoadData 直接组装为 m x m L0A，禁止在 L1 内重排。
        AscendC::LoadData(akkL0, akkL1, loadA);

        AscendC::LoadData2DParamsV2 loadRhs{};
        loadRhs.mStartPosition = 0;
        loadRhs.kStartPosition = 0;
        loadRhs.mStep = m / 16;
        loadRhs.kStep = Shape::kHeadDim / 16;
        loadRhs.srcStride = m / 16;
        loadRhs.dstStride = m / 16;
        loadRhs.ifTranspose = true;
        loadRhs.sid = 0;
        AscendC::LoadData(kBetaL0, kBetaL1, loadRhs);
        AscendC::LoadData(vBetaL0, vBetaL1, loadRhs);
        AscendC::Mutex::Unlock<PIPE_MTE1>(operandMutex);
        AscendC::Mutex::Unlock<PIPE_MTE1>(l1Mutex);

        AscendC::MmadParams mmad{};
        mmad.m = m;
        mmad.n = Shape::kHeadDim;
        mmad.k = m;
        mmad.cmatrixInitVal = true;
        mmad.cmatrixSource = false;
        mmad.unitFlag = 0;

        AscendC::Mutex::Lock<PIPE_M>(operandMutex);
        AscendC::Mutex::Lock<PIPE_M>(wL0cMutex);
        AscendC::Mmad(wL0c, akkL0, kBetaL0, mmad); // 计算 W=Akk@K_beta_g。
        AscendC::Mutex::Unlock<PIPE_M>(wL0cMutex);
        AscendC::Mutex::Unlock<PIPE_M>(operandMutex);

        AscendC::Mutex::Lock<PIPE_M>(operandMutex);
        AscendC::Mutex::Lock<PIPE_M>(uL0cMutex);
        AscendC::Mmad(uL0c, akkL0, vBetaL0, mmad); // 计算 U=Akk@V_beta。
        AscendC::Mutex::Unlock<PIPE_M>(uL0cMutex);
        AscendC::Mutex::Unlock<PIPE_M>(operandMutex);

        const uint64_t outputOffset = HeadTensorOffset(
            args_.tiling, chunk, valueHead, Shape::kHeadDim);
        AscendC::Mutex::Lock<PIPE_FIX>(wL0cMutex);
        auto wFix = AscendC::FixpipeParamsV220(
            Shape::kHeadDim, chunk.validRows, m,
            Shape::kHeadDim, false);
        wFix.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(
            wGm_[outputOffset], wL0c, wFix);
        AscendC::Mutex::Unlock<PIPE_FIX>(wL0cMutex);

        AscendC::Mutex::Lock<PIPE_FIX>(uL0cMutex);
        auto uFix = AscendC::FixpipeParamsV220(
            Shape::kValueDim, chunk.validRows, m,
            Shape::kValueDim, false);
        uFix.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(
            uGm_[outputOffset], uL0c, uFix);
        AscendC::Mutex::Unlock<PIPE_FIX>(uL0cMutex);
    }

    PrepareKernelArgs args_{};
    uint32_t workgroup_ = 0;
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource_{};
    AscendC::GlobalTensor<bfloat16_t> akkGm_{};
    AscendC::GlobalTensor<bfloat16_t> wGm_{};
    AscendC::GlobalTensor<bfloat16_t> uGm_{};
};

} // namespace KdaPrepare::Arch35

#endif // PSEUDOCODE_ARCH35_CHUNK_KDA_FWD_PREPARE_CUBE_H
