/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H
#define ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "chunk_kda_fwd_prepare_coarse_score.h"
#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_runtime_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace KdaPrepare::Arch22 {

// softplus 的 log1p 补偿：u=exp(-|x|)，log1p(u)=Ln(1+u)+r/(1+u)，
// r=u-(fl(1+u)-1)。两个 FP32 临时区固定复用共享 scratch 的前 7 KiB（与 V1Vf
// 同一约定）：scratch 之后紧邻第二份 private 槽，0x1F00 起是 sequence-major 的
// beta gather 偏移表，因此临时区不得越过 0x1C00。

template <typename GateT, typename BetaT, typename CompilePolicy, bool BF16_RELAY = false>
class ChunkKdaFwdPrepareVec {
    using RelayT = std::conditional_t<BF16_RELAY, bfloat16_t, float>;

    using Domain = std::conditional_t<CompilePolicy::commonMath,
        ExpDomainTraits<true>, RuntimeExpDomain>;
    Domain domain_{};

public:
    __aicore__ inline void Init(const PrepareKernelArgs &args, AscendC::TPipe *pipe)
    {
        args_ = args;
        // This launch-wide predicate is identical for both engines. Keep its
        // math constants folded and do not reload modes for each head/stage.
        coarseEligible_ = BF16_RELAY && CoarseScore::Eligible<CompilePolicy>(args_);
        if constexpr (!CompilePolicy::commonMath) {
            domain_.Init(args_.tiling.useExp2);
        }
        pipe_ = pipe;
        workgroup_ = WorkgroupId();
        aiv_ = AscendC::GetSubBlockIdx();
        coreCount_ = args_.tiling.usedCoreNum;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.v));
        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(args_.rawGate));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(args_.beta));
        if (args_.dtBias != nullptr) {
            dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_DT_BIAS *>(args_.dtBias));
        }
        if (args_.aLog != nullptr) {
            aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_A_LOG *>(args_.aLog));
        }
        if (RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save) {
            qgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.qg));
        }
        qgScaledGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(args_.qgScaled));
        kgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.kg));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.gk));
        aqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.aqk));
        if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) != OutputMode::None)) {
            akkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.akk));
        }
        if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
            qHatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.qHat));
        }
        if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
            kHatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.kHat));
        }
        if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
            qRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.qRstd));
        }
        if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
            kRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.kRstd));
        }
        if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
            betaEffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.betaEff));
        }
        if (coreCount_ == 0) {
            return;
        }
        // Generic policy reserves184KiB; head-major tile16 explicitly needs192KiB.
        pipe_->InitBuffer(ubBuf_, args_.tiling.inputSequenceMajor
            ? Arch22Ub::kUsableBytes : Arch22Ub::kHardwareBytes);
        // 每种 HardEvent 有独立 ID 池。这里保留 Alloc/Release，让后续
        // 基础 API 能看到占用状态；注释给出 CANN 9.1 分配器的预期返回值。
        scalarRead_ = pipe_->AllocEventID<AscendC::HardEvent::V_S>(); // ID 0
        scalarWrite_ = pipe_->AllocEventID<AscendC::HardEvent::S_V>(); // ID 0
        if (args_.tiling.inputSequenceMajor) {
            auto offsets = ubBuf_.Get<uint8_t>()[Arch22Ub::kBetaGatherOffsets]
                               .template ReinterpretCast<uint32_t>();
            for (uint32_t row = 0; row < Shape::kChunkRows; ++row) {
                offsets.SetValue(row, row * 32U);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarWrite_);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarWrite_);
        }
        sharedFree_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>(); // ID 0
        // 两个 pair 分时复用共享 G/scratch；初始许可只发布一次。
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        ioFree_[0] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>(); // ID 0
        ioFree_[1] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>(); // ID 1
        inputReady_[0] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>(); // ID 0
        inputReady_[1] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>(); // ID 1
        outputReady_[0] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>(); // ID 0
        outputReady_[1] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>(); // ID 1
        mte3ToV_[0] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>(); // ID 0
        mte3ToV_[1] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>(); // ID 1
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[1]);
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
        const uint32_t workBegin = TokenBalancedWorkBoundary(
            args_, total, workgroup_, coreCount_);
        const uint32_t workEnd = TokenBalancedWorkBoundary(
            args_, total, workgroup_ + 1, coreCount_);
        if (workgroup_ >= coreCount_ || workBegin >= workEnd) {
            for (uint32_t pair = 0; pair < 2; ++pair) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
                ReleasePairEvents(pair);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(sharedFree_);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_S>(scalarRead_);
            pipe_->ReleaseEventID<AscendC::HardEvent::S_V>(scalarWrite_);
            return;
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
                // AIV0 处理 0/2，AIV1 处理 1/3；两个 pair 分时复用共享 G 区。
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t localHead = pair * 2 + aiv_;
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                        kFreeFlagId[pair]);
                    if (localHead < activeHeads) {
                        const uint32_t valueHead = groupBegin + localHead;
                        StageV0(chunk, valueHead, localHead, pair);
                        StageV1(chunk, localHead, pair);
                    }
                    // 尾部无任务的 AIV 仍参加集合，但不计算地址或访问 GM。
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        kReadyFlagId[pair]);
                }
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t localHead = pair * 2 + aiv_;
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                        kFreeFlagId[pair]);
                    if (localHead < activeHeads) {
                        const uint32_t valueHead = groupBegin + localHead;
                        StageV3(chunk, valueHead, localHead, pair);
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        kReadyFlagId[pair]);
                }
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t localHead = pair * 2 + aiv_;
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                        kFreeFlagId[pair]);
                    if (localHead < activeHeads) {
                        const uint32_t valueHead = groupBegin + localHead;
                        StageV6(chunk, valueHead, localHead, pair);
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        kReadyFlagId[pair]);
                }
                groupBegin += activeHeads;
            }
        }
        // 消费最后一次 C7 发布，保证每次 set 都有对应 wait。
        for (uint32_t pair = 0; pair < 2; ++pair) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                kFreeFlagId[pair]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
            ReleasePairEvents(pair);
        }
        // 消费最后一轮 V6（或从未使用时的初始）共享区许可后再释放事件。
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(sharedFree_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_S>(scalarRead_);
        pipe_->ReleaseEventID<AscendC::HardEvent::S_V>(scalarWrite_);
    }

private:
    // V1 and the BF16 V6 path use this phase-local tail sequentially.
    static constexpr uint32_t kV6BatchStage = Arch22Ub::kUsableBytes;
    static constexpr uint32_t kV6BatchBetaBrcb = 0x10000;
    static constexpr uint32_t kV6BatchBetaRepeat = 0x10800;

    __aicore__ inline float ReadScalar(AscendC::LocalTensor<float> tensor,
                                       uint32_t index)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(scalarRead_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalarRead_);
        const float value = tensor.GetValue(index);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarWrite_);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarWrite_);
        return value;
    }

    __aicore__ inline void ReleasePairEvents(uint32_t pair)
    {
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
    }

    __aicore__ inline void StageV0(const ChunkRange &chunk, uint32_t valueHead,
                                   uint32_t localHead, uint32_t pair)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
        const uint32_t base = Arch22Ub::kPrivateBase[pair];
        auto ub = ubBuf_.Get<uint8_t>();
        auto q = ub[base + Arch22Ub::kQ].template ReinterpretCast<bfloat16_t>();
        auto k = ub[base + Arch22Ub::kK].template ReinterpretCast<bfloat16_t>();
        const uint32_t gateBase = sizeof(GateT) == 4 ? Arch22Ub::kSharedG
                                                     : base + Arch22Ub::kGateOrKMinus;
        auto gate = ub[gateBase].template ReinterpretCast<GateT>();
        auto betaStrided = ub[base + Arch22Ub::kBetaRawStrided]
                               .template ReinterpretCast<BetaT>();
        auto beta = ub[base + Arch22Ub::kBetaRaw].template ReinterpretCast<BetaT>();
        auto betaEff = ub[base + Arch22Ub::kBetaEff].template ReinterpretCast<float>();
        auto dtBias = ub[base + Arch22Ub::kDtBias].template ReinterpretCast<float>();
        auto aLog = ub[base + Arch22Ub::kALog].template ReinterpretCast<float>();
        // Private gaps: beta uses at most 256 bytes; scalar log uses one DMA block.
        auto biasInput = ub[base + (sizeof(DTYPE_DT_BIAS) == 2 ? Arch22Ub::kBetaRaw + 256 : Arch22Ub::kDtBias)]
                             .template ReinterpretCast<DTYPE_DT_BIAS>();
        auto logInput = ub[base + Arch22Ub::kALog + (sizeof(DTYPE_A_LOG) == 2 ? 32 : 0)]
                            .template ReinterpretCast<DTYPE_A_LOG>();
        auto qRstd = ub[base + Arch22Ub::kQRstd].template ReinterpretCast<float>();
        auto kRstd = ub[base + Arch22Ub::kKRstd].template ReinterpretCast<float>();
        auto g = ub[Arch22Ub::kSharedG].template ReinterpretCast<float>();
        auto scratch = ub[Arch22Ub::kSharedScratch].template ReinterpretCast<float>();
        const uint32_t qkHead = QkHeadForValueHead(args_.tiling, valueHead);
        const uint64_t qkOffset = QkInputOffset(args_.tiling, chunk, qkHead);
        const uint64_t gateOffset =
            RawGateInputOffset(args_.tiling, chunk, valueHead);
        const uint64_t betaOffset =
            BetaInputOffset(args_.tiling, chunk, valueHead);
        const uint64_t headOutputOffset =
            HeadTensorOffset(args_.tiling, chunk, valueHead, Shape::kHeadDim);
        const uint32_t qkStride = args_.tiling.inputSequenceMajor
                                      ? static_cast<uint32_t>(
                                            static_cast<uint64_t>(
                                                args_.tiling.qkHeadNum - 1) *
                                            Shape::kHeadDim * sizeof(bfloat16_t))
                                      : 0;
        const uint32_t gateStride = args_.tiling.inputSequenceMajor
                                        ? static_cast<uint32_t>(
                                              static_cast<uint64_t>(
                                                  args_.tiling.valueHeadNum - 1) *
                                              Shape::kHeadDim * sizeof(GateT))
                                        : 0;
        // 获得共享 G/scratch 的独占许可；V1 完成最后一次 V 读取后归还。
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::DataCopyExtParams qkCopy{static_cast<uint16_t>(chunk.validRows),
            static_cast<uint32_t>(Shape::kHeadDim * sizeof(bfloat16_t)),
            qkStride, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> qkPad{false, 0, 0, 0};
        AscendC::DataCopyPad(q, qGm_[qkOffset], qkCopy, qkPad);
        AscendC::DataCopyPad(k, kGm_[qkOffset], qkCopy, qkPad);
        AscendC::DataCopyExtParams gateCopy{static_cast<uint16_t>(chunk.validRows),
            static_cast<uint32_t>(Shape::kHeadDim * sizeof(GateT)),
            gateStride, 0, 0};
        AscendC::DataCopyPadExtParams<GateT> gatePad{false, 0, 0, 0};
        AscendC::DataCopyPad(gate, gateGm_[gateOffset], gateCopy, gatePad);
        AscendC::DataCopyPadExtParams<BetaT> betaPad{false, 0, 0, 0};
        if (args_.tiling.inputSequenceMajor) {
            const uint32_t betaStride =
                static_cast<uint32_t>(static_cast<uint64_t>(
                    args_.tiling.valueHeadNum - 1) * sizeof(BetaT));
            AscendC::DataCopyPad(betaStrided, betaGm_[betaOffset],
                AscendC::DataCopyExtParams{
                    static_cast<uint16_t>(chunk.validRows),
                    static_cast<uint32_t>(sizeof(BetaT)), betaStride, 0, 0},
                betaPad);
        } else {
            AscendC::DataCopyPad(beta, betaGm_[betaOffset],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(chunk.validRows * sizeof(BetaT)),
                    0, 0, 0}, betaPad);
        }
        if (static_cast<GateMode>(args_.tiling.gateMode) != GateMode::PrecomputedStep) {
            if (args_.tiling.hasDtBias) {
                // 公开接口将 dt_bias 固定为展平的 [HV,K]。
                AscendC::DataCopyPad(biasInput, dtBiasGm_[valueHead * Shape::kHeadDim],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(Shape::kHeadDim * sizeof(DTYPE_DT_BIAS)),
                        0, 0, 0}, AscendC::DataCopyPadExtParams<DTYPE_DT_BIAS>{false, 0, 0, 0});
            }
            if (args_.aLog != nullptr) {
                // 公开接口将 a_log 固定为 [HV]。
                AscendC::DataCopyPad(logInput, aLogGm_[valueHead],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(sizeof(DTYPE_A_LOG)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<DTYPE_A_LOG>{false, 0, 0, 0});
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        if constexpr (sizeof(DTYPE_DT_BIAS) == 2) {
            if (args_.tiling.hasDtBias) AscendC::Cast(dtBias, biasInput, AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
        }
        if constexpr (sizeof(DTYPE_A_LOG) == 2) {
            if (static_cast<GateMode>(args_.tiling.gateMode) != GateMode::PrecomputedStep)
            AscendC::Cast(aLog, logInput, AscendC::RoundMode::CAST_NONE, 1);
        }
        if constexpr (sizeof(DTYPE_A_LOG) == 2 || sizeof(DTYPE_DT_BIAS) == 2) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (args_.tiling.inputSequenceMajor) {
            auto offsets = ub[Arch22Ub::kBetaGatherOffsets]
                               .template ReinterpretCast<uint32_t>();
            AscendC::Gather(beta, betaStrided, offsets,
                            static_cast<uint32_t>(0), chunk.validRows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // 本 Stage 的一次向量计算完成 Q/K 可选 L2 norm、beta 变换、
        // gate 变换和逐 token cumsum，并生成后续 Stage 所需中间量。
        // tileBuf 复用 kMinus 所在的私有区（V1 之前未被使用；BF16 gate
        // 场景下 gate 自身也在此区，但已整体 Cast 到 g，可以安全复用）。
        V0Vf(q, k, qRstd, kRstd, gate, beta, betaEff, dtBias, aLog, g,
             scratch, ub[base + Arch22Ub::kGateOrKMinus]
                          .template ReinterpretCast<float>(),
             chunk.validRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride,
            Workspace::kArch22SlotStride);
        AscendC::GlobalTensor<bfloat16_t> qhatContext;
        AscendC::GlobalTensor<bfloat16_t> khatContext;
        AscendC::GlobalTensor<float> betaContext;
        qhatContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace + slot + Workspace::kQHat));
        khatContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace + slot + Workspace::kKHat));
        betaContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + slot + Workspace::kBetaEff));
        AscendC::DataCopy(qhatContext, q, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(khatContext, k, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopyPad(betaContext, betaEff,
            AscendC::DataCopyExtParams{
                1, static_cast<uint32_t>(chunk.validRows * sizeof(float)),
                0, 0, 0});
        // Q/K 保存量按 HK 写回。GVA 中只有 QK 头组的第一个 HV 是 owner，
        // 其余 HV 仍保留各自 workspace context，供本 kernel 的 V1/V6 使用。
        if (IsQkOutputOwner(args_.tiling, valueHead)) {
            const uint64_t qkOutputOffset = QkHeadTensorOffset(
                args_.tiling, chunk, qkHead, Shape::kHeadDim);
            const uint64_t rstdOutputOffset =
                QkHeadScalarOffset(args_.tiling, chunk, qkHead);
            if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
                AscendC::DataCopy(qHatGm_[qkOutputOffset], q,
                                  chunk.validRows * Shape::kHeadDim);
            }
            if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
                AscendC::DataCopy(kHatGm_[qkOutputOffset], k,
                                  chunk.validRows * Shape::kHeadDim);
            }
            if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
                AscendC::DataCopyPad(qRstdGm_[rstdOutputOffset], qRstd,
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(chunk.validRows * sizeof(float)),
                        0, 0, 0});
            }
            if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
                AscendC::DataCopyPad(kRstdGm_[rstdOutputOffset], kRstd,
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(chunk.validRows * sizeof(float)),
                        0, 0, 0});
            }
        }
        if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Recompute || RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save)) {
            AscendC::DataCopyPad(
                betaEffGm_[HeadScalarOffset(args_.tiling, chunk, valueHead)],
                betaEff, AscendC::DataCopyExtParams{
                             1, static_cast<uint32_t>(
                                    chunk.validRows * sizeof(float)),
                             0, 0, 0});
        }
        AscendC::DataCopy(gkGm_[headOutputOffset], g,
                          chunk.validRows * Shape::kHeadDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
    }

    __aicore__ inline void StageV1(const ChunkRange &chunk, uint32_t localHead,
                                   uint32_t pair)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
        const uint32_t base = Arch22Ub::kPrivateBase[pair];
        auto ub = ubBuf_.Get<uint8_t>();
        auto qHat = ub[base + Arch22Ub::kQ].template ReinterpretCast<bfloat16_t>();
        auto kHat = ub[base + Arch22Ub::kK].template ReinterpretCast<bfloat16_t>();
        auto qPlus = ub[base + Arch22Ub::kQ].template ReinterpretCast<bfloat16_t>();
        auto kPlus = ub[base + Arch22Ub::kK].template ReinterpretCast<bfloat16_t>();
        auto kMinus = ub[base + Arch22Ub::kGateOrKMinus].template ReinterpretCast<bfloat16_t>();
        auto g = ub[Arch22Ub::kSharedG].template ReinterpretCast<float>();
        auto scratch = ub[Arch22Ub::kSharedScratch].template ReinterpretCast<float>();
        // 唯一一次 VF：按四个 16 行中点广播 Gref，生成 Qplus/Kplus 和
        // 16/32/48/64 行 Kminus；Gref 不物化为矩阵。
        // SIMD 路径按 USE_EXP2 选择等价截断域；仅 log2 分支乘 ln(2)，
        // 两个分支最终都调用自然底 Exp。
        V1Vf(qHat, kHat, qPlus, kPlus, kMinus, g, scratch,
             chunk.validRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::GlobalTensor<bfloat16_t> payload;
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride,
            Workspace::kArch22SlotStride);
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace + slot + Workspace::kPayload));
        const uint32_t payloadBytes = BF16_RELAY && coarseEligible_
            ? CoarseScore::kPayloadBytes : Shape::kScorePayloadBytes;
        AscendC::DataCopy(payload, qPlus, payloadBytes / sizeof(bfloat16_t));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
    }

    __aicore__ inline void StageV3(const ChunkRange &chunk, uint32_t valueHead,
                                   uint32_t localHead, uint32_t pair)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
        const uint32_t base = Arch22Ub::kPrivateBase[pair];
        auto ub = ubBuf_.Get<uint8_t>();
        auto raw = ub[base + Arch22Ub::kV3CompactRaw].template ReinterpretCast<float>();
        auto aqk = ub[base + Arch22Ub::kV3Aqk].template ReinterpretCast<bfloat16_t>();
        auto lkk = ub[base + Arch22Ub::kV3Lkk].template ReinterpretCast<float>();
        auto b = ub[base + Arch22Ub::kV3B].template ReinterpretCast<float>();
        auto x0 = ub[base + Arch22Ub::kV3X0].template ReinterpretCast<float>();
        auto x1 = ub[base + Arch22Ub::kV3X1].template ReinterpretCast<float>();
        auto negX1 = ub[base + Arch22Ub::kV3NegX1].template ReinterpretCast<float>();
        auto betaEff = ub[Arch22Ub::kV3BetaEff].template ReinterpretCast<float>();
        auto akkPack = ub[base + Arch22Ub::kV3AkkPack]
                           .template ReinterpretCast<bfloat16_t>();
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride,
            Workspace::kArch22SlotStride);
        AscendC::GlobalTensor<float> betaContext;
        betaContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + slot + Workspace::kBetaEff));
        const uint32_t active = CeilDiv(chunk.validRows, Shape::kSubChunkRows);
        uint32_t compactElements = 0;
        for (uint32_t s = 0; s < active; ++s) {
            const uint32_t remaining =
                chunk.validRows - s * Shape::kSubChunkRows;
            const uint32_t rows = remaining < Shape::kSubChunkRows
                                      ? remaining
                                      : Shape::kSubChunkRows;
            const uint32_t bandElements =
                2 * rows * Shape::kPrefixRows[s];
            compactElements += bandElements;
        }
        AscendC::GlobalTensor<float> rawScoreRelay;
        rawScoreRelay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + slot + Workspace::kArch22CubeRelay),
            Workspace::kArch22RawScoreBytes / sizeof(float));
        // C2 按四段连续写入独占 relay；尾 band 只包含有效行。
        AscendC::DataCopy(raw, rawScoreRelay, compactElements);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(betaEff, betaContext,
            AscendC::DataCopyExtParams{
                1, static_cast<uint32_t>(chunk.validRows * sizeof(float)),
                0, 0, 0}, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        // 本 Stage 一次完成因果 mask、scale、beta 和两个 32x32 叶逆，
        // 生成 Aqk/B/X0/X1/negX1/稳定 Akk；negX1 供 C5 做普通 Mmad。
        // Leaf0 区在 V3 内不被其它 buffer 使用，借给 Gather 的偏移表。
        auto coeffOffsetInt =
            ub[base + Arch22Ub::kV3Leaf0].template ReinterpretCast<int32_t>();
        V3Vf(raw, betaEff, aqk, lkk, b, x0, x1, negX1, akkPack,
             coeffOffsetInt, chunk.validRows, args_.tiling.scale);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::DataCopy(aqkGm_[AOutputOffset(args_.tiling, chunk, valueHead)],
                          aqk, chunk.validRows * Shape::kChunkRows);
        // C4 固定读取完整 64x64 矩阵，因此补零后的中转矩阵始终写入
        // 工作空间；公开 Akk 只有 T 行，尾 chunk 只能写有效行。
        AscendC::GlobalTensor<bfloat16_t> akkRelay;
        akkRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload + Workspace::kAkk));
        AscendC::DataCopy(akkRelay, akkPack,
                          Shape::kChunkRows * Shape::kChunkRows);
        if ((RuntimeMath<CompilePolicy>::Output(args_.tiling) != OutputMode::None)) {
            const uint64_t outputOffset =
                AOutputOffset(args_.tiling, chunk, valueHead);
            const uint32_t topRows = chunk.validRows < 32
                                         ? chunk.validRows
                                         : 32;
            AscendC::DataCopyPad(
                akkGm_[outputOffset], akkPack,
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(
                           topRows * Shape::kChunkRows * sizeof(bfloat16_t)),
                    0, 0, 0});
            if (chunk.validRows > 32) {
                const uint32_t bottomRows = chunk.validRows - 32;
                constexpr uint32_t kBottomRightOffset =
                    32 * Shape::kChunkRows + 32;
                AscendC::DataCopyPad(
                    akkGm_[outputOffset + kBottomRightOffset],
                    akkPack[kBottomRightOffset],
                    AscendC::DataCopyExtParams{
                        static_cast<uint16_t>(bottomRows),
                        32 * sizeof(bfloat16_t), 2, 64, 0});
            }
        }
        if (chunk.validRows > 32) {
            AscendC::GlobalTensor<float> payload;
            payload.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                args_.workspace + slot + Workspace::kPayload));
            AscendC::DataCopy(payload[Workspace::kX0 / sizeof(float)], x0, 1024);
            AscendC::DataCopy(payload[Workspace::kNegX1 / sizeof(float)], negX1, 1024);
            AscendC::DataCopy(payload[Workspace::kB / sizeof(float)], b, 1024);
        }
        // 公开 Akk 的 BF16 store 完成后，复用已死亡的 raw score 区构造内部 FP32 Akk。
        // [0xD000,0x11000) 位于当前 private slot 内，不覆盖 x0/x1。
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
        auto akkFp32 = ub[base + Arch22Ub::kV3CompactRaw].template ReinterpretCast<float>();
        AscendC::Duplicate(akkFp32, 0.0f, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(akkFp32, x0, AscendC::DataCopyParams(32, 4, 0, 4));
        AscendC::DataCopy(akkFp32[32 * 64 + 32], x1,
                          AscendC::DataCopyParams(32, 4, 0, 4));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::GlobalTensor<float> akkFp32Relay;
        akkFp32Relay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + slot + Workspace::kArch22AkkFp32));
        AscendC::DataCopy(akkFp32Relay, akkFp32, 4096);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
    }

    __aicore__ inline void StageV6(const ChunkRange &chunk, uint32_t valueHead,
                                   uint32_t localHead, uint32_t pair)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
        const uint32_t base = Arch22Ub::kPrivateBase[pair];
        auto ub = ubBuf_.Get<uint8_t>();
        auto qg = ub[base + Arch22Ub::kV6Qg].template ReinterpretCast<bfloat16_t>();
        auto qgScaled = ub[Arch22Ub::kV6QgScaled].template ReinterpretCast<bfloat16_t>();
        auto kg = ub[base + Arch22Ub::kV6Kg].template ReinterpretCast<bfloat16_t>();
        auto vBeta = ub[base + Arch22Ub::kV6VBeta].template ReinterpretCast<bfloat16_t>();
        auto kBetaG = ub[base + Arch22Ub::kV6KBetaG].template ReinterpretCast<bfloat16_t>();
        auto g = ub[Arch22Ub::kSharedG].template ReinterpretCast<float>();
        auto scratch = ub[Arch22Ub::kSharedScratch].template ReinterpretCast<float>();
        auto betaEff = ub[base + Arch22Ub::kBetaEff].template ReinterpretCast<float>();
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride,
            Workspace::kArch22SlotStride);
        AscendC::GlobalTensor<bfloat16_t> qhat;
        AscendC::GlobalTensor<bfloat16_t> khat;
        AscendC::GlobalTensor<float> betaContext;
        qhat.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace + slot + Workspace::kQHat));
        khat.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace + slot + Workspace::kKHat));
        betaContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + slot + Workspace::kBetaEff));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::DataCopy(qg, qhat, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kg, khat, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(betaEff, betaContext,
            AscendC::DataCopyExtParams{
                1, static_cast<uint32_t>(chunk.validRows * sizeof(float)),
                0, 0, 0}, pad);
        AscendC::DataCopy(g,
                          gkGm_[HeadTensorOffset(args_.tiling, chunk, valueHead,
                                                Shape::kHeadDim)],
                          chunk.validRows * Shape::kHeadDim);
        const uint32_t vStride = args_.tiling.inputSequenceMajor
                                     ? static_cast<uint32_t>(
                                           static_cast<uint64_t>(
                                               args_.tiling.valueHeadNum - 1) *
                                           Shape::kValueDim * sizeof(bfloat16_t))
                                     : 0;
        AscendC::DataCopyPad(vBeta,
            vGm_[ValueInputOffset(args_.tiling, chunk, valueHead)],
            AscendC::DataCopyExtParams{static_cast<uint16_t>(chunk.validRows),
                static_cast<uint32_t>(Shape::kValueDim * sizeof(bfloat16_t)),
                vStride, 0, 0},
            AscendC::DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        AscendC::GlobalTensor<RelayT> kBetaRelay;
        AscendC::GlobalTensor<RelayT> vBetaRelay;
        kBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ RelayT *>(
            args_.workspace + slot + Workspace::kArch22RhsKBetaG));
        vBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ RelayT *>(
            args_.workspace + slot + Workspace::kArch22RhsVBeta));
        // Gate/factor/beta arithmetic stays FP32 in both relay variants.
        if constexpr (BF16_RELAY) {
            V6BatchVf(qg, qgScaled, kg, vBeta, kBetaG, g, betaEff, scratch,
                      chunk.validRows, args_.tiling.scale, pair);
        } else {
            V6Vf(qg, qgScaled, kg, vBeta, kBetaG, g, betaEff, scratch,
                 chunk.validRows, args_.tiling.scale, kBetaRelay, vBetaRelay, pair);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        if constexpr (BF16_RELAY) {
            const uint32_t rhsRows = chunk.validRows > 32 ? 64 : 32;
            AscendC::DataCopy(kBetaRelay, kBetaG, rhsRows * Shape::kHeadDim);
            AscendC::DataCopy(vBetaRelay, vBeta, rhsRows * Shape::kHeadDim);
            // Both packed RHS arrays remain live until this whole-head drain.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const uint64_t out =
            HeadTensorOffset(args_.tiling, chunk, valueHead, Shape::kHeadDim);
        // 共享区只有 qgScaled 仍被 MTE3 读取，优先搬出并单独发布完成事件；
        // 后续私有区输出可继续在 MTE3 流水中执行。
        AscendC::DataCopy(qgScaledGm_[out], qgScaled,
                          chunk.validRows * Shape::kHeadDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
        if (RuntimeMath<CompilePolicy>::Output(args_.tiling) == OutputMode::Save) {
            AscendC::DataCopy(qgGm_[out], qg,
                              chunk.validRows * Shape::kHeadDim);
        }
        AscendC::DataCopy(kgGm_[out], kg, chunk.validRows * Shape::kHeadDim);
        // 写侧加固（保守）：这两份 relay 马上会被 AIC 的 C7 跨核读回。只靠
        // ready 跨核 flag 时，MTE3 写缓冲/L2 里的数据可能还没对读侧可见，读侧
        // 就会拿到 relay 的旧内容（现场表现为单个 work item 的 U 整块错、同一
        // Stage 的 W 正常）。先把两份 relay clean 出去再放行下一步。
        AscendC::DataCacheCleanAndInvalid<RelayT, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(kBetaRelay);
        AscendC::DataCacheCleanAndInvalid<RelayT, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(vBetaRelay);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
        // qgScaled 与共享 G 复用地址；必须等 MTE3 读完，才能把共享区
        // 通过 V_MTE2 许可交给下一个 pair 的 MTE2。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
    }

    // 行广播二元运算：把 128 lane 的行张量广播到 rows 行。
    // fp32 一次 repeat 只有 64 lane，因此按低/高半行各下发一条指令，
    // 广播侧用 repeatStride=0，一次 replace 内 8 个 block 连续读取。
    __aicore__ inline void SubTileMinusRow(AscendC::LocalTensor<float> dst,
                                           AscendC::LocalTensor<float> tile,
                                           AscendC::LocalTensor<float> row,
                                           uint32_t rows)
    {
        const uint8_t rowStride =
            static_cast<uint8_t>(Shape::kHeadDim * sizeof(float) / 32);
        AscendC::Sub(dst, tile, row, 64, static_cast<uint8_t>(rows),
                     {1, 1, 1, rowStride, rowStride, 0});
        AscendC::Sub(dst[64], tile[64], row[64], 64,
                     static_cast<uint8_t>(rows),
                     {1, 1, 1, rowStride, rowStride, 0});
    }

    __aicore__ inline void SubRowMinusTile(AscendC::LocalTensor<float> dst,
                                           AscendC::LocalTensor<float> row,
                                           AscendC::LocalTensor<float> tile,
                                           uint32_t rows)
    {
        const uint8_t rowStride =
            static_cast<uint8_t>(Shape::kHeadDim * sizeof(float) / 32);
        AscendC::Sub(dst, row, tile, 64, static_cast<uint8_t>(rows),
                     {1, 1, 1, rowStride, 0, rowStride});
        AscendC::Sub(dst[64], row[64], tile[64], 64,
                     static_cast<uint8_t>(rows),
                     {1, 1, 1, rowStride, 0, rowStride});
    }

    __aicore__ inline void AddRowToTile(AscendC::LocalTensor<float> dst,
                                        AscendC::LocalTensor<float> tile,
                                        AscendC::LocalTensor<float> row,
                                        uint32_t rows)
    {
        const uint8_t rowStride =
            static_cast<uint8_t>(Shape::kHeadDim * sizeof(float) / 32);
        AscendC::Add(dst, tile, row, 64, static_cast<uint8_t>(rows),
                     {1, 1, 1, rowStride, rowStride, 0});
        AscendC::Add(dst[64], tile[64], row[64], 64,
                     static_cast<uint8_t>(rows),
                     {1, 1, 1, rowStride, rowStride, 0});
    }

    // Keep one eight-lane block per row. Mul reuses it across the row
    // with src1BlkStride=0; src1RepStride=1 selects the next row.
    __aicore__ inline void CopyRowScalarBlocks(
        AscendC::LocalTensor<float> dstTile,
        AscendC::LocalTensor<float> rowScalars, uint32_t rows)
    {
        AscendC::DataCopy(dstTile, rowScalars, rows * 8);
    }

    __aicore__ inline void MulTileByRowScalars(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> tile,
        AscendC::LocalTensor<float> rowScalarBlocks, uint32_t rows)
    {
        const uint8_t rowStride =
            static_cast<uint8_t>(Shape::kHeadDim * sizeof(float) / 32);
        AscendC::Mul(dst, tile, rowScalarBlocks, 64,
                     static_cast<uint8_t>(rows),
                     {1, 1, 0, rowStride, rowStride, 1});
        AscendC::Mul(dst[64], tile[64], rowScalarBlocks, 64,
                     static_cast<uint8_t>(rows),
                     {1, 1, 0, rowStride, rowStride, 1});
    }

    // 单位下三角逆的右看消元：对 leafRows 阶单位下三角块求 X=(I+L)^-1。
    // 每轮需要“L 的第 k 列按行复制成 4 个 block”的系数张量。这里用带元素
    // 偏移的 Gather 直接从 lkk 取：dst 的第 j 行 32 个 lane 全取
    // lkk[lkkBase + (k+1+j)*rowStride + k]，等价于该列复制 4 个 block，
    // 一条指令替代原来的 1 次 DataCopy + 1 次 Brcb + 4 次 DataCopy。
    // 偏移张量与 k 无关（只随行变化），k 由 srcBaseAddr 吸收。
    __aicore__ inline void SolveLeafRightLooking(
        AscendC::LocalTensor<float> x, AscendC::LocalTensor<float> lkk,
        AscendC::LocalTensor<uint32_t> coeffOffset,
        AscendC::LocalTensor<float> coeffTile,
        AscendC::LocalTensor<float> prod, uint32_t leafRows, uint32_t lkkBase,
        uint32_t rowStride)
    {
        constexpr uint32_t kLeaf = 32;
        constexpr uint8_t kLeafRepStride =
            static_cast<uint8_t>(kLeaf * sizeof(float) / 32);
        for (uint32_t k = 0; k + 1 < leafRows; ++k) {
            const uint32_t rows = leafRows - k - 1;
            // (k+1+j)*rowStride + k = (lkkBase + (rowStride+1)*k) + rowStride*(j+1)
            const uint32_t baseElems = lkkBase + (rowStride + 1) * k;
            AscendC::Gather(coeffTile, lkk, coeffOffset,
                            static_cast<uint32_t>(baseElems * sizeof(float)),
                            rows * kLeaf);
            AscendC::PipeBarrier<PIPE_V>();
            // src0 用 repStride=0 反复读取 X[k,:]，src1 是逐行不同的系数，
            // dst 是 k 以下各行。
            AscendC::Mul(prod, x[k * kLeaf], coeffTile, kLeaf,
                         static_cast<uint8_t>(rows),
                         {1, 1, 1, kLeafRepStride, 0, kLeafRepStride});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(x[(k + 1) * kLeaf], x[(k + 1) * kLeaf], prod,
                         static_cast<int32_t>(rows * kLeaf));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void SolveTwoLeavesRightLooking(
        AscendC::LocalTensor<float> x0, AscendC::LocalTensor<float> x1,
        AscendC::LocalTensor<float> lkk,
        AscendC::LocalTensor<uint32_t> offsets,
        AscendC::LocalTensor<float> raw, uint32_t topRows, uint32_t bottomRows)
    {
        constexpr uint32_t kLeaf = 32;
        constexpr uint32_t kStride = Shape::kChunkRows;
        constexpr uint32_t kBottom = 32 * kStride + 32;
        constexpr uint32_t kCoeffLanes = 8;
        constexpr uint8_t kRepStride = kLeaf * sizeof(float) / 32;
        static_assert(Arch22Ub::kPrivateBytes - Arch22Ub::kV3CompactRaw >= 4096 * sizeof(float));
        auto coeffA = raw;
        auto coeffB = raw[1024];
        auto prodA = raw[2048];
        auto prodB = raw[3072];
        const uint32_t maxRows = topRows > bottomRows ? topRows : bottomRows;
        if (topRows > 1) {
            AscendC::Gather(coeffA, lkk, offsets, uint32_t{0}, (topRows - 1) * kCoeffLanes);
        }
        if (bottomRows > 1) {
            AscendC::Gather(coeffB, lkk, offsets, uint32_t{kBottom * sizeof(float)}, (bottomRows - 1) * kCoeffLanes);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column + 1 < maxRows; ++column) {
            const uint32_t rowsA = column + 1 < topRows ? topRows - column - 1 : 0;
            const uint32_t rowsB = column + 1 < bottomRows ? bottomRows - column - 1 : 0;
            if (rowsA > 0) {
                AscendC::Mul(prodA, x0[column * kLeaf], coeffA, kLeaf,
                             static_cast<uint8_t>(rowsA), {1, 1, 0, kRepStride, 0, 1});
            }
            if (rowsB > 0) {
                AscendC::Mul(prodB, x1[column * kLeaf], coeffB, kLeaf,
                             static_cast<uint8_t>(rowsB), {1, 1, 0, kRepStride, 0, 1});
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (rowsA > 0) {
                AscendC::Sub(x0[(column + 1) * kLeaf], x0[(column + 1) * kLeaf], prodA,
                             static_cast<int32_t>(rowsA * kLeaf));
            }
            if (rowsB > 0) {
                AscendC::Sub(x1[(column + 1) * kLeaf], x1[(column + 1) * kLeaf], prodB,
                             static_cast<int32_t>(rowsB * kLeaf));
            }
            if (column + 2 < topRows) {
                AscendC::Gather(coeffA, lkk, offsets,
                    static_cast<uint32_t>((kStride + 1) * (column + 1) * sizeof(float)),
                    (topRows - column - 2) * kCoeffLanes);
            }
            if (column + 2 < bottomRows) {
                AscendC::Gather(coeffB, lkk, offsets,
                    static_cast<uint32_t>((kBottom + (kStride + 1) * (column + 1)) * sizeof(float)),
                    (bottomRows - column - 2) * kCoeffLanes);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // FP32 gate lives in shared G, so the private Kminus region is free
    // throughout norm. Two32-row FP32 tensors fill the private32 KiB region;
    // small scalar/broadcast buffers use the already-owned shared scratch.
    // Keep the original FP32 square/reduce/epsilon/sqrt/div sequence; scales
    // remain on Vector rather than crossing V_S/S_V for every token.
    __aicore__ inline void L2NormRowsBatched(
        AscendC::LocalTensor<bfloat16_t> src,
        AscendC::LocalTensor<float> rstd,
        AscendC::LocalTensor<float> workspace, uint32_t validRows)
    {
        constexpr uint32_t kRows = 32;
        constexpr uint32_t kElems = kRows * Shape::kHeadDim;
        static_assert(2 * kElems * sizeof(float) <=
                      Arch22Ub::kBetaRawStrided - Arch22Ub::kGateOrKMinus);
        auto values = workspace;
        auto squares = workspace[kElems];
        // Installed CANN9.2 ReduceSum AR/isReuseSource computes in squares;
        // its external temporary is unused. Keep a valid aligned scratch base.
        auto shared = ubBuf_.Get<uint8_t>()[Arch22Ub::kSharedScratch]
                          .template ReinterpretCast<float>();
        auto reduceTmp = shared[1024].template ReinterpretCast<uint8_t>();
        auto repeatScalars = shared;
        auto blockScalars = shared[256];
        auto ones = shared[512];
        static_assert((1024 + 64) * sizeof(float) <= 0x1F00);
        // Brcb reads sixteen scalars even for a final partial tile.
        AscendC::Duplicate(rstd, 0.0F, Shape::kChunkRows);
        AscendC::Duplicate(ones, 1.0F, kRows);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t begin = 0; begin < validRows; begin += kRows) {
            const uint32_t rows = validRows - begin < kRows ? validRows - begin : kRows;
            const uint32_t elems = rows * Shape::kHeadDim;
            AscendC::Cast(values, src[begin * Shape::kHeadDim],
                          AscendC::RoundMode::CAST_NONE, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(squares, values, values, elems);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t shape[2] = {rows, Shape::kHeadDim};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                rstd[begin], squares, reduceTmp, shape, true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(rstd[begin], rstd[begin], args_.tiling.epsilon, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(rstd[begin], rstd[begin], rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(rstd[begin], ones, rstd[begin], rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(blockScalars, rstd[begin], kRows / 8, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            CopyRowScalarBlocks(repeatScalars, blockScalars, rows);
            AscendC::PipeBarrier<PIPE_V>();
            MulTileByRowScalars(values, values, repeatScalars, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(src[begin * Shape::kHeadDim], values,
                          AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void V0Vf(
        AscendC::LocalTensor<bfloat16_t> q, AscendC::LocalTensor<bfloat16_t> k,
        AscendC::LocalTensor<float> qRstd, AscendC::LocalTensor<float> kRstd,
        AscendC::LocalTensor<GateT> gate, AscendC::LocalTensor<BetaT> beta,
        AscendC::LocalTensor<float> betaEff, AscendC::LocalTensor<float> dtBias,
        AscendC::LocalTensor<float> aLog, AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> scratch,
        AscendC::LocalTensor<float> tileBuf, uint32_t validRows)
    {
        const uint32_t count = validRows * Shape::kHeadDim;
        if (RuntimeMath<CompilePolicy>::Norm(args_.tiling) == QkNormMode::L2 &&
                      std::is_same_v<GateT, float>) {
            L2NormRowsBatched(q, qRstd, tileBuf, validRows);
            L2NormRowsBatched(k, kRstd, tileBuf, validRows);
        } else if (RuntimeMath<CompilePolicy>::Norm(args_.tiling) == QkNormMode::L2) {
            // 每行按冻结语义执行 x * rsqrt(sum(x^2) + epsilon)。AR
            // ReduceSum 以 isReuseSource=true 调用，归约复用平方结果区；
            // 传入的临时区从 1 KiB 对齐地址开始，不额外占用 UB。
            uint32_t reduceShape[2] = {1, Shape::kHeadDim};
            for (uint32_t row = 0; row < validRows; ++row) {
                AscendC::Cast(scratch, q[row * Shape::kHeadDim],
                              AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(scratch[Shape::kHeadDim], scratch, scratch,
                             Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                    betaEff, scratch[Shape::kHeadDim],
                    scratch[2 * Shape::kHeadDim]
                        .template ReinterpretCast<uint8_t>(),
                    reduceShape, true);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(betaEff, betaEff, args_.tiling.epsilon, 1);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sqrt(betaEff, betaEff, 1);
                auto qOne = scratch[3 * Shape::kHeadDim];
                AscendC::Duplicate(qOne, 1.0F, 1);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(betaEff, qOne, betaEff, 1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(scalarRead_);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalarRead_);
                const float qScale = betaEff.GetValue(0);
                qRstd.SetValue(row, qScale);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarWrite_);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarWrite_);
                AscendC::Muls(scratch, scratch, qScale, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(q[row * Shape::kHeadDim], scratch,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Cast(scratch, k[row * Shape::kHeadDim],
                              AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(scratch[Shape::kHeadDim], scratch, scratch,
                             Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                    betaEff, scratch[Shape::kHeadDim],
                    scratch[2 * Shape::kHeadDim]
                        .template ReinterpretCast<uint8_t>(),
                    reduceShape, true);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(betaEff, betaEff, args_.tiling.epsilon, 1);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sqrt(betaEff, betaEff, 1);
                auto kOne = scratch[3 * Shape::kHeadDim];
                AscendC::Duplicate(kOne, 1.0F, 1);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(betaEff, kOne, betaEff, 1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(scalarRead_);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalarRead_);
                const float kScale = betaEff.GetValue(0);
                kRstd.SetValue(row, kScale);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarWrite_);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarWrite_);
                AscendC::Muls(scratch, scratch, kScale, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(k[row * Shape::kHeadDim], scratch,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            // 固定输出合同下 Identity 仍写确定值；反向关闭 L2Norm 时不会
            // 使用 rstd，但不能留下未初始化输出。
            AscendC::Duplicate(qRstd, 1.0F, validRows);
            AscendC::Duplicate(kRstd, 1.0F, validRows);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (RuntimeMath<CompilePolicy>::Beta(args_.tiling) == BetaMode::Raw) {
            // Raw 模式只统一转成 FP32，公开 betaEff 不做 sigmoid 变换。
            if constexpr (std::is_same_v<BetaT, float>) {
                AscendC::Adds(betaEff, beta, 0.0F, validRows);
            } else {
                AscendC::Cast(betaEff, beta, AscendC::RoundMode::CAST_NONE,
                              validRows);
            }
        } else {
            if constexpr (std::is_same_v<BetaT, float>) {
                AscendC::Muls(scratch, beta, -1.0F, validRows);
            } else {
                AscendC::Cast(scratch, beta, AscendC::RoundMode::CAST_NONE,
                              validRows);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(scratch, scratch, -1.0F, validRows);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(scratch, scratch, validRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(scratch, scratch, 1.0F, validRows);
            AscendC::Duplicate(betaEff, 1.0F, validRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(betaEff, betaEff, scratch, validRows);
            if (RuntimeMath<CompilePolicy>::Beta(args_.tiling) == BetaMode::TwoSigmoid) {
                // betaEff 原址读写，必须等待前一条 Div 完成。
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(betaEff, betaEff, 2.0F, validRows);
            }
        }

        if constexpr (std::is_same_v<GateT, float>) {
            AscendC::Adds(g, gate, 0.0F, count);
        } else {
            AscendC::Cast(g, gate, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (static_cast<GateMode>(args_.tiling.gateMode) != GateMode::PrecomputedStep) {
            float gateA = 1.0F;
            if (args_.aLog != nullptr) {
                // a_h = exp(A_log[h])，不能直接把 A_log 当成乘数。
                AscendC::Exp(aLog, aLog, 1);
                AscendC::PipeBarrier<PIPE_V>();
                gateA = ReadScalar(aLog, 0);
            }
            if (args_.tiling.hasDtBias) {
                // dtBias 是 [K] 行向量，整块广播下发，避免逐行一次 Add。
                AddRowToTile(g, g, dtBias, validRows);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (!(static_cast<GateMode>(args_.tiling.gateMode) == GateMode::SafeSigmoid) &&
                          static_cast<GateMode>(args_.tiling.gateMode) == GateMode::Softplus) {
                // deltaG=-a*(max(x,0)+log(1+exp(-abs(x))))，其中
                // log1p(u)=Ln(1+u)+r/(1+u)，r=u-(fl(1+u)-1)。直接取 Ln(1+u)
                // 会把 1+u 的舍入按 1/u 放大（u=exp(-|x|) 越小越明显），双标杆
                // L1 下会超限，故用补偿项抵消。
                //
                // 两个 FP32 临时区只能落在共享 scratch 的前 7 KiB：scratch 之后
                // 紧邻第二份 private 槽（q/k/gate/kMinus/score），0x1F00 起又是
                // sequence-major 的 beta gather 偏移表，因此按 7 行一个 tile
                // 下发，与 V1Vf 使用同一段 scratch 的约定保持一致。
                const float factor = -gateA;
                constexpr uint32_t kLog1pRows = 7;
                constexpr uint32_t kLog1pElems = kLog1pRows * Shape::kHeadDim;
                auto gateSum = scratch;
                auto gateResidual = scratch[kLog1pElems];
                for (uint32_t rowBegin = 0; rowBegin < validRows;
                     rowBegin += kLog1pRows) {
                    const uint32_t rows = validRows - rowBegin < kLog1pRows
                                              ? validRows - rowBegin
                                              : kLog1pRows;
                    const uint32_t elems = rows * Shape::kHeadDim;
                    const uint32_t offset = rowBegin * Shape::kHeadDim;
                    AscendC::Abs(tileBuf, g[offset], elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(tileBuf, tileBuf, -1.0F, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Exp(tileBuf, tileBuf, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(gateSum, tileBuf, 1.0F, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(gateResidual, gateSum, -1.0F, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    // r = u - (fl(1+u) - 1)：补偿 1+u 的舍入残差。
                    AscendC::Sub(gateResidual, tileBuf, gateResidual, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(tileBuf, gateSum, 0.0F, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln(tileBuf, tileBuf, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    // log1p(u) = Ln(1+u) + r/(1+u)
                    AscendC::Div(gateResidual, gateResidual, gateSum, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(tileBuf, tileBuf, gateResidual, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Maxs(g[offset], g[offset], 0.0F, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(tileBuf, tileBuf, g[offset], elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(g[offset], tileBuf, factor, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            } else {
                // deltaG=lower_bound/(1+exp(-a*x))。
                // 整块一次下发，替代原先每行 5 条指令。
                const float negativeA = -gateA;
                AscendC::Muls(tileBuf, g, negativeA, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(tileBuf, tileBuf, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(tileBuf, tileBuf, 1.0F, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate(g, args_.tiling.lowerBound, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(g, g, tileBuf, count);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (domain_.useExp2) {
            AscendC::Muls(g, g, domain_.stepScale, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // token 维前缀和保留 headDim 向量宽度，不跨 chunk 传播。
        // 这里刻意保留逐行累加：保证 gk（FP32 公开输出）与改动前逐位一致，
        // 63 步累加相对整块指令链的开销很小。
        for (uint32_t row = 1; row < validRows; ++row) {
            AscendC::Add(g[row * Shape::kHeadDim],
                         g[row * Shape::kHeadDim],
                         g[(row - 1) * Shape::kHeadDim], Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (validRows < Shape::kChunkRows) {
            const uint32_t tail =
                (Shape::kChunkRows - validRows) * Shape::kHeadDim;
            AscendC::Duplicate(q[validRows * Shape::kHeadDim],
                               static_cast<bfloat16_t>(0), tail);
            AscendC::Duplicate(k[validRows * Shape::kHeadDim],
                               static_cast<bfloat16_t>(0), tail);
            AscendC::Duplicate(g[validRows * Shape::kHeadDim], 0.0F, tail);
        }
    }

    __aicore__ inline void V1CoarseFactorsVf(
        AscendC::LocalTensor<bfloat16_t> qHat,
        AscendC::LocalTensor<bfloat16_t> kHat,
        AscendC::LocalTensor<bfloat16_t> qPlus,
        AscendC::LocalTensor<bfloat16_t> kPlus,
        AscendC::LocalTensor<bfloat16_t> kMinus,
        AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> scratch, uint32_t validRows)
    {
        const float clampMin = domain_.StoredBound(ExpDomain::kV1Bf16LowerBase2);
        const float clampMax = domain_.StoredBound(ExpDomain::kV1Bf16UpperBase2);
        constexpr uint32_t kRows = 16;
        constexpr uint32_t kTailStage = Arch22Ub::kUsableBytes;
        static_assert(kTailStage + kRows * Shape::kHeadDim * sizeof(float)
                      <= Arch22Ub::kHardwareBytes);
        auto factor = scratch;
        auto stage = ubBuf_.Get<uint8_t>()[kTailStage].template ReinterpretCast<float>();
        const uint32_t active = CeilDiv(validRows, CoarseScore::kAnchorRows);
        auto references = kMinus[(CoarseScore::kReferences - Arch22Ub::kGateOrKMinus)
                                / sizeof(bfloat16_t)].template ReinterpretCast<float>();
        // A short coarse band's midpoint can be in the preceding source tile.
        // Preserve references before any G tile is repurposed (tails25/57 etc.).
        for (uint32_t s = 0; s < active; ++s) {
            const uint32_t begin = s * CoarseScore::kAnchorRows;
            const uint32_t end = validRows < begin + CoarseScore::kAnchorRows
                ? validRows : begin + CoarseScore::kAnchorRows;
            const uint32_t midpoint = (begin + end) / 2;
            AscendC::Copy(references[s * Shape::kHeadDim], g[midpoint * Shape::kHeadDim],
                          static_cast<uint64_t>(64), 2, {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < CoarseScore::kAnchors; ++s) {
            auto dst = kMinus[(CoarseScore::kMinusOffset[s] - Arch22Ub::kGateOrKMinus)
                              / sizeof(bfloat16_t)];
            AscendC::Duplicate(dst, static_cast<bfloat16_t>(0),
                               CoarseScore::kPrefixRows[s] * Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t rowBegin = 0; rowBegin < validRows; rowBegin += kRows) {
            const uint32_t rows = validRows - rowBegin < kRows ? validRows - rowBegin : kRows;
            const uint32_t elems = rows * Shape::kHeadDim;
            const uint32_t sourceBand = rowBegin / CoarseScore::kAnchorRows;
            AscendC::Cast(stage, kHat[rowBegin * Shape::kHeadDim],
                          AscendC::RoundMode::CAST_NONE, elems);
            AscendC::PipeBarrier<PIPE_V>();
            // Own band is last, preserving its unrounded Exp for Q/Kplus.
            for (uint32_t remaining = active; remaining > sourceBand; --remaining) {
                const uint32_t s = remaining - 1;
                SubRowMinusTile(factor, references[s * Shape::kHeadDim],
                                g[rowBegin * Shape::kHeadDim], rows);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(factor, factor, clampMin, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(factor, factor, clampMax, elems);
                AscendC::PipeBarrier<PIPE_V>();
                if (domain_.useExp2) {
                    AscendC::Muls(factor, factor, domain_.expInputScale, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Exp(factor, factor, elems);
                AscendC::PipeBarrier<PIPE_V>();
                // All readers of this G tile finish in the final own band.
                // Future source rows stay untouched; references were saved above.
                auto product = s == sourceBand ? g[rowBegin * Shape::kHeadDim] : factor;
                AscendC::Mul(product, stage, factor, elems);
                AscendC::PipeBarrier<PIPE_V>();
                auto dst = kMinus[(CoarseScore::kMinusOffset[s] - Arch22Ub::kGateOrKMinus)
                                  / sizeof(bfloat16_t)];
                AscendC::Cast(dst[rowBegin * Shape::kHeadDim], product,
                              AscendC::RoundMode::CAST_RINT, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
            // Reciprocal is approximate on A2. One Newton step restores FP32
            // accuracy: y <- y * (2 - E*y). E is the own-band unrounded Exp.
            // The current G tile is now dead; original public gk was saved by V0.
            auto reciprocal = g[rowBegin * Shape::kHeadDim];
            AscendC::Reciprocal(reciprocal, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(factor, factor, reciprocal, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(factor, factor, -1.0F, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(factor, factor, 2.0F, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(factor, factor, reciprocal, elems);
            AscendC::PipeBarrier<PIPE_V>();
            // This tile's last Kminus reader finished; Khat may be overwritten.
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kPlus[rowBegin * Shape::kHeadDim], stage,
                          AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(stage, qHat[rowBegin * Shape::kHeadDim],
                          AscendC::RoundMode::CAST_NONE, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qPlus[rowBegin * Shape::kHeadDim], stage,
                          AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void V1Vf(
        AscendC::LocalTensor<bfloat16_t> qHat,
        AscendC::LocalTensor<bfloat16_t> kHat,
        AscendC::LocalTensor<bfloat16_t> qPlus,
        AscendC::LocalTensor<bfloat16_t> kPlus,
        AscendC::LocalTensor<bfloat16_t> kMinus,
        AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> scratch, uint32_t validRows)
    {
        if constexpr (BF16_RELAY) {
            if (coarseEligible_) {
                V1CoarseFactorsVf(qHat, kHat, qPlus, kPlus, kMinus, g, scratch, validRows);
                return;
            }
        }
        constexpr float base2Min = ExpDomain::kV1Bf16LowerBase2;
        constexpr float base2Max = ExpDomain::kV1Bf16UpperBase2;
        const float clampMin = domain_.StoredBound(base2Min);
        const float clampMax = domain_.StoredBound(base2Max);
        // Head-major Init explicitly expands the generic allocation by8KiB.
        // Tail stage leaves both private payload slots independently owned.
        // Sequence-major preserves tile7 and its scratch+0x1F00 gather table.
        constexpr uint32_t kTailStage = Arch22Ub::kPrivateBase[1] + Arch22Ub::kPrivateBytes;
        static_assert(kTailStage == Arch22Ub::kUsableBytes);
        static_assert(kTailStage + 16 * Shape::kHeadDim * sizeof(float)
                      <= Arch22Ub::kHardwareBytes);
        const uint32_t kTileRows = args_.tiling.inputSequenceMajor ? 7U : 16U;
        const uint32_t kTileElems = kTileRows * Shape::kHeadDim;
        auto factor = scratch;
        auto stage = args_.tiling.inputSequenceMajor ? scratch[kTileElems]
            : ubBuf_.Get<uint8_t>()[kTailStage].template ReinterpretCast<float>();
        const uint32_t active = CeilDiv(validRows, Shape::kSubChunkRows);
        const uint32_t blockEnds[Shape::kSubChunkCount] = {
            validRows < 16 ? validRows : 16,
            validRows < 32 ? validRows : 32,
            validRows < 48 ? validRows : 48,
            validRows,
        };

        // Kplus 与 Khat 原位复用，所以必须先生成完四个 Kminus 前缀。
        // 否则后一个参考块会错误读取已经舍入成 Kplus 的数据。
        for (uint32_t s = 0; s < Shape::kSubChunkCount; ++s) {
            auto kMinusBlock = kMinus[
                (Arch22Ub::kKMinus[s] - Arch22Ub::kGateOrKMinus) /
                sizeof(bfloat16_t)];
            AscendC::Duplicate(kMinusBlock, static_cast<bfloat16_t>(0),
                               Shape::kPrefixRows[s] * Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // Kminus 前缀整体按 8 行一块下发：参考行差值用行广播指令生成，
        // 之后 clamp/缩放/Exp/Cast/Mul/Cast 全部是整块指令。
        for (uint32_t s = 0; s < active; ++s) {
            const uint32_t blockBegin = s * Shape::kSubChunkRows;
            const uint32_t blockEnd = blockEnds[s];
            auto kMinusBlock = kMinus[
                (Arch22Ub::kKMinus[s] - Arch22Ub::kGateOrKMinus) /
                sizeof(bfloat16_t)];
            // 半开区间 [begin,end) 的中点取 floor((begin+end)/2)。
            const uint32_t midpoint = (blockBegin + blockEnd) / 2;
            const uint32_t prefix = blockEnd;
            for (uint32_t rowBegin = 0; rowBegin < prefix;
                 rowBegin += kTileRows) {
                const uint32_t rows = prefix - rowBegin < kTileRows
                                          ? prefix - rowBegin
                                          : kTileRows;
                const uint32_t elems = rows * Shape::kHeadDim;
                SubRowMinusTile(factor, g[midpoint * Shape::kHeadDim],
                                g[rowBegin * Shape::kHeadDim], rows);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(factor, factor, clampMin, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(factor, factor, clampMax, elems);
                AscendC::PipeBarrier<PIPE_V>();
                if (domain_.useExp2) {
                    AscendC::Muls(factor, factor, domain_.expInputScale,
                                  elems);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Exp(factor, factor, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(stage, kHat[rowBegin * Shape::kHeadDim],
                              AscendC::RoundMode::CAST_NONE, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(stage, stage, factor, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(kMinusBlock[rowBegin * Shape::kHeadDim], stage,
                              AscendC::RoundMode::CAST_RINT, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        // 四个 Kminus 都已完成，此时可以把 Qhat/Khat 原位改写为
        // Qplus/Kplus；无效行沿用 V0 写入的零。同样按 8 行一块下发。
        for (uint32_t s = 0; s < active; ++s) {
            const uint32_t blockBegin = s * Shape::kSubChunkRows;
            const uint32_t blockEnd = blockEnds[s];
            const uint32_t midpoint = (blockBegin + blockEnd) / 2;
            for (uint32_t rowBegin = blockBegin; rowBegin < blockEnd;
                 rowBegin += kTileRows) {
                const uint32_t rows = blockEnd - rowBegin < kTileRows
                                          ? blockEnd - rowBegin
                                          : kTileRows;
                const uint32_t elems = rows * Shape::kHeadDim;
                SubTileMinusRow(factor, g[rowBegin * Shape::kHeadDim],
                                g[midpoint * Shape::kHeadDim], rows);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(factor, factor, clampMin, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(factor, factor, clampMax, elems);
                AscendC::PipeBarrier<PIPE_V>();
                // SIMD 侧统一调用自然 Exp；log2 域显式换算，ln 域直接计算。
                if (domain_.useExp2) {
                    AscendC::Muls(factor, factor, domain_.expInputScale,
                                  elems);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Exp(factor, factor, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(stage, qHat[rowBegin * Shape::kHeadDim],
                              AscendC::RoundMode::CAST_NONE, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(stage, stage, factor, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(qPlus[rowBegin * Shape::kHeadDim], stage,
                              AscendC::RoundMode::CAST_RINT, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(stage, kHat[rowBegin * Shape::kHeadDim],
                              AscendC::RoundMode::CAST_NONE, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(stage, stage, factor, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(kPlus[rowBegin * Shape::kHeadDim], stage,
                              AscendC::RoundMode::CAST_RINT, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void V3Vf(
        AscendC::LocalTensor<float> raw, AscendC::LocalTensor<float> betaEff,
        AscendC::LocalTensor<bfloat16_t> aqk,
        AscendC::LocalTensor<float> lkk,
        AscendC::LocalTensor<float> b, AscendC::LocalTensor<float> x0,
        AscendC::LocalTensor<float> x1, AscendC::LocalTensor<float> negX1,
        AscendC::LocalTensor<bfloat16_t> akkPack,
        AscendC::LocalTensor<int32_t> coeffOffsetInt, uint32_t validRows,
        float scale)
    {
        AscendC::Duplicate(aqk, static_cast<bfloat16_t>(0),
                           Shape::kChunkRows * Shape::kChunkRows);
        AscendC::Duplicate(b, 0.0F, 1024);
        // C2 按 s 堆叠 [rawAqk, rawAkk]，s 之前的 FP32 元素数为
        // subChunkRows^2*s*(s+1)。按全局行解包后，
        // 每个循环不再需要根据行号分支。
        // rawAqk 段的 scale 整体下发一次，随后逐行 Cast 进 Aqk；
        // Akk 段不乘 scale，只能按 s 分别处理。
        const uint32_t activeBands = CeilDiv(validRows, Shape::kSubChunkRows);
        for (uint32_t s = 0; s < activeBands; ++s) {
            const uint32_t remaining = validRows - s * Shape::kSubChunkRows;
            const uint32_t bandRows = remaining < Shape::kSubChunkRows
                                          ? remaining
                                          : Shape::kSubChunkRows;
            const uint32_t band = Shape::kSubChunkRows *
                                  Shape::kSubChunkRows * s * (s + 1);
            AscendC::Muls(raw[band], raw[band], scale,
                          bandRows * Shape::kPrefixRows[s]);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t globalRow = 0; globalRow < validRows; ++globalRow) {
            const uint32_t s = globalRow / Shape::kSubChunkRows;
            const uint32_t row = globalRow % Shape::kSubChunkRows;
            const uint32_t n = Shape::kPrefixRows[s];
            const uint32_t stackedBand = Shape::kSubChunkRows *
                                         Shape::kSubChunkRows * s * (s + 1);
            auto rawAqk = raw[stackedBand + row * n];
            const uint32_t aqkCount = globalRow + 1;
            AscendC::Cast(aqk[globalRow * Shape::kChunkRows], rawAqk,
                          AscendC::RoundMode::CAST_RINT, aqkCount);
        }

        const uint32_t topRows = validRows < 32 ? validRows : 32;
        // StageV3 owns sharedFree through V3Vf. Keep broadcasts separate
        // from both the live beta input and the leaf Gather offset table.
        auto betaRepeat = ubBuf_.Get<uint8_t>()[Arch22Ub::kSharedScratch]
                              .template ReinterpretCast<float>();
        auto betaBrcb = betaRepeat[1024];
        auto betaPadded = betaRepeat[1152];
        static_assert(Arch22Ub::kSharedScratch + 0x1300 <=
                      Arch22Ub::kBetaGatherOffsets);
        AscendC::Duplicate(betaPadded, 0.0F, Shape::kChunkRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Copy(betaPadded, betaEff, static_cast<uint64_t>(validRows),
                      1, {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < activeBands; ++s) {
            const uint32_t rowBegin = s * Shape::kSubChunkRows;
            const uint32_t remaining = validRows - rowBegin;
            const uint32_t rows = remaining < Shape::kSubChunkRows
                                      ? remaining : Shape::kSubChunkRows;
            const uint32_t n = Shape::kPrefixRows[s];
            const uint32_t stackedBand = Shape::kSubChunkRows *
                                         Shape::kSubChunkRows * s * (s + 1);
            auto rawAkk = raw[stackedBand + rows * n];
            AscendC::Brcb(betaBrcb, betaPadded[rowBegin], 2, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            CopyRowScalarBlocks(betaRepeat, betaBrcb, rows);
            AscendC::PipeBarrier<PIPE_V>();
            const uint8_t stride = static_cast<uint8_t>(n / 8);
            AscendC::Mul(rawAkk, rawAkk, betaRepeat, n,
                         static_cast<uint8_t>(rows),
                         {1, 1, 0, stride, stride, 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(lkk[rowBegin * Shape::kChunkRows], rawAkk,
                AscendC::DataCopyParams(static_cast<uint16_t>(rows),
                    static_cast<uint16_t>(n / 8), 0,
                    static_cast<uint16_t>((Shape::kChunkRows - n) / 8)));
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (validRows > 32) {
            AscendC::DataCopy(b, lkk[32 * Shape::kChunkRows],
                AscendC::DataCopyParams(static_cast<uint16_t>(validRows - 32),
                                        4, 4, 0));
        }

        AscendC::PipeBarrier<PIPE_V>();
        // I+Lkk=[[A00,0],[B,A11]]；在同一次 VF 中求两个 32 阶单位下三角逆。
        AscendC::Duplicate(x0, 0.0F, 1024);
        AscendC::Duplicate(x1, 0.0F, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t bottomRows = validRows > 32 ? validRows - 32 : 0;
        // FP32 向量指令要求 UB 首地址按 32 Byte 对齐，不能直接从
        // x[row, row] 发起长度为 1 的 Duplicate。按对角元素在 32 Byte
        // block 内的 lane 分组：同一 lane 相邻两个对角元素跨 8 行，
        // repeat stride 为 8 * 4 + 1 = 33 个 block。
        constexpr uint32_t kFp32PerBlock = 8;
        constexpr uint16_t kDiagonalBlockStride = 1;
        constexpr uint8_t kDiagonalRepeatStride = 33;
        const uint32_t topDiagonalLanes =
            topRows < kFp32PerBlock ? topRows : kFp32PerBlock;
        for (uint32_t lane = 0; lane < topDiagonalLanes; ++lane) {
            uint64_t diagonalMask[2] = {1ULL << lane, 0};
            const uint8_t repeat = static_cast<uint8_t>(
                CeilDiv(topRows - lane, kFp32PerBlock));
            AscendC::Duplicate(x0[lane * 32], 1.0F, diagonalMask, repeat,
                               kDiagonalBlockStride, kDiagonalRepeatStride);
        }
        const uint32_t bottomDiagonalLanes =
            bottomRows < kFp32PerBlock ? bottomRows : kFp32PerBlock;
        for (uint32_t lane = 0; lane < bottomDiagonalLanes; ++lane) {
            uint64_t diagonalMask[2] = {1ULL << lane, 0};
            const uint8_t repeat = static_cast<uint8_t>(
                CeilDiv(bottomRows - lane, kFp32PerBlock));
            AscendC::Duplicate(x1[lane * 32], 1.0F, diagonalMask, repeat,
                               kDiagonalBlockStride, kDiagonalRepeatStride);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // 单位下三角逆：X[i,:]=-sum(k<i,L[i,k]*X[k,:])，X[i,i]=1。
        // 系数靠 Gather 每次只取一列，偏移张量与 k 无关，只随行号变化：
        // offsets[i] = 256 * (i / 8 + 1) 字节，用 4 条矢量指令生成本次调用的
        // 偏移表（Leaf0 区在 V3 内空闲）；系数区与乘积区放在已消费的
        // compact raw 低地址。
        // Eight equal lanes form one block; Mul reuses that block four times.
        constexpr uint32_t kLeafLanes = 8;
        const uint32_t coeffCount =
            (topRows > bottomRows ? topRows : bottomRows) * kLeafLanes;
        AscendC::CreateVecIndex(coeffOffsetInt, 0, coeffCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(coeffOffsetInt, coeffOffsetInt, 3, coeffCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(coeffOffsetInt, coeffOffsetInt, 1, coeffCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(coeffOffsetInt, coeffOffsetInt,
                      static_cast<int32_t>(Shape::kChunkRows * sizeof(float)),
                      coeffCount);
        AscendC::PipeBarrier<PIPE_V>();
        auto coeffOffset = coeffOffsetInt.template ReinterpretCast<uint32_t>();
        SolveTwoLeavesRightLooking(x0, x1, lkk, coeffOffset, raw, topRows, bottomRows);
        // B 已在解包 rawAkk 时直接写入；其余行保持零。
        AscendC::PipeBarrier<PIPE_V>();
        // C5 直接执行普通 MMAD：negX1@T，不依赖不存在的 negate 参数。
        AscendC::Muls(negX1, x1, -1.0F, 1024);
        AscendC::PipeBarrier<PIPE_V>();

        // q00=X0、q01=0、q11=X1；q10 留给 C5 的 negX1@T。
        AscendC::Duplicate(akkPack, static_cast<bfloat16_t>(0), 4096);
        AscendC::PipeBarrier<PIPE_V>();
        // The coefficient offsets are dead after the solve. Pack two full,
        // initialized leaves there before copying their valid rows into Akk.
        auto cast0 = coeffOffsetInt.template ReinterpretCast<bfloat16_t>();
        auto cast1 = cast0[2048];
        AscendC::Cast(cast0, x0, AscendC::RoundMode::CAST_RINT, 1024);
        AscendC::Cast(cast1, x1, AscendC::RoundMode::CAST_RINT, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        if (topRows > 0) {
            AscendC::DataCopy(akkPack, cast0,
                AscendC::DataCopyParams(static_cast<uint16_t>(topRows), 2, 0, 2));
        }
        if (bottomRows > 0) {
            AscendC::DataCopy(akkPack[32 * Shape::kChunkRows + 32], cast1,
                AscendC::DataCopyParams(static_cast<uint16_t>(bottomRows), 2, 0, 2));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void V6BatchVf(
        AscendC::LocalTensor<bfloat16_t> qg,
        AscendC::LocalTensor<bfloat16_t> qgScaled,
        AscendC::LocalTensor<bfloat16_t> kg,
        AscendC::LocalTensor<bfloat16_t> vBeta,
        AscendC::LocalTensor<bfloat16_t> kBetaG,
        AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> betaEff,
        AscendC::LocalTensor<float> scratch, uint32_t validRows, float scale,
        uint32_t pair)
    {
        const uint32_t rhsRows = validRows > 32 ? 64 : 32;
        if (validRows == 0) {
            return;
        }
        const uint32_t last = (validRows - 1) * Shape::kHeadDim;
        const float directMin =
            domain_.StoredBound(ExpDomain::kV6LowerBase2);
        const float directMax =
            domain_.StoredBound(ExpDomain::kV6UpperBase2);
        // Borrow only after the other pair's last MTE3 read has retired.
        // Reissue its existing token; the next input load also waits sharedFree,
        // published after this VF's final Vector use of the borrowed Q/K region.
        const bool wide = !args_.tiling.inputSequenceMajor;
        const uint32_t otherPair = pair ^ 1U;
        if (wide) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[otherPair]);
            // The borrower writes with Vector, so bridge MTE2 completion to V;
            // the MTE3->MTE2 wait alone cannot order a Vector overwrite.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[otherPair]);
        }
        const uint32_t kTileRows = wide ? 32U : 16U;
        static_assert(Arch22Ub::kHardwareBytes - Arch22Ub::kUsableBytes >= 8192);
        static_assert(Arch22Ub::kV6KBetaG + 16384 <= 0x10000);
        static_assert(kV6BatchBetaRepeat + 512 <= Arch22Ub::kBetaEff);
        static_assert(kV6BatchBetaBrcb + 2048 <= kV6BatchBetaRepeat);
        auto ub = ubBuf_.Get<uint8_t>();
        auto factor = wide
            ? ub[Arch22Ub::kPrivateBase[otherPair]].template ReinterpretCast<float>()
            : scratch;
        auto stage = wide
            ? ub[Arch22Ub::kPrivateBase[otherPair] + 0x4000].template ReinterpretCast<float>()
            : ub[kV6BatchStage].template ReinterpretCast<float>();
        static_assert(2 * 32 * Shape::kHeadDim * sizeof(float) <= Arch22Ub::kGateOrKMinus);
        auto betaBrcb = ub[Arch22Ub::kPrivateBase[pair] + kV6BatchBetaBrcb]
                            .template ReinterpretCast<float>();
        auto betaRepeat = ub[Arch22Ub::kPrivateBase[pair] + kV6BatchBetaRepeat]
                              .template ReinterpretCast<float>();
        AscendC::Brcb(betaBrcb, betaEff,
                      static_cast<uint8_t>(Shape::kChunkRows / 8), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t rowBegin = 0; rowBegin < validRows;
             rowBegin += kTileRows) {
            const uint32_t rows = validRows - rowBegin < kTileRows
                                      ? validRows - rowBegin
                                      : kTileRows;
            const uint32_t elems = rows * Shape::kHeadDim;
            const uint32_t offset = rowBegin * Shape::kHeadDim;
            // E(G)=exp(clamp(g[row]))：这里刻意不减 g[last]，与 arch35 的
            // V6 语义（qg/qBeta 用行内累计量，kg 才用 g[last]-g[row]）一致。
            AscendC::Maxs(factor, g[offset], directMin, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(factor, factor, directMax, elems);
            AscendC::PipeBarrier<PIPE_V>();
            if (domain_.useExp2) {
                AscendC::Muls(factor, factor, domain_.expInputScale, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Exp(factor, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(stage, qg[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qg[offset], stage, AscendC::RoundMode::CAST_RINT,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            // Khat*E(G)*beta 连续在 FP32 域计算，不经过 BF16 中间量。
            AscendC::Cast(stage, kg[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            CopyRowScalarBlocks(betaRepeat, betaBrcb[8 * rowBegin], rows);
            AscendC::PipeBarrier<PIPE_V>();
            MulTileByRowScalars(stage, stage, betaRepeat, rows);
            // Both BF16 RHS buffers stay live until StageV6's final drain.
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kBetaG[offset], stage, AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
            // E'(G)=exp(clamp(g[last]-g[row]))
            SubRowMinusTile(factor, g[last], g[offset], rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(factor, factor, directMin, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(factor, factor, directMax, elems);
            AscendC::PipeBarrier<PIPE_V>();
            if (domain_.useExp2) {
                AscendC::Muls(factor, factor, domain_.expInputScale, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Exp(factor, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            // Overlap factor-only arithmetic with MTE3's last stage read.
            AscendC::Cast(stage, kg[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kg[offset], stage, AscendC::RoundMode::CAST_RINT,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            // qgScaled 从已舍入的 BF16 qg 中间量回读。正序处理时，BF16 输出
            // 只覆盖已经消费完的 FP32 G 低地址，不会覆盖后续 G 行。
            AscendC::Cast(stage, qg[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(stage, stage, scale, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qgScaled[offset], stage,
                          AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
            // Side04: betaRepeat still describes these rows. K's MTE3
            // was consumed before kg; V finishes before packed/stage reuse.
            AscendC::Cast(stage, vBeta[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            MulTileByRowScalars(stage, stage, betaRepeat, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vBeta[offset], stage, AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (validRows < rhsRows) {
            const uint32_t padded = (rhsRows - validRows) * Shape::kHeadDim;
            AscendC::Duplicate(kBetaG[validRows * Shape::kHeadDim],
                              static_cast<bfloat16_t>(0.0f), padded);
            AscendC::Duplicate(vBeta[validRows * Shape::kHeadDim],
                              static_cast<bfloat16_t>(0.0f), padded);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void V6Vf(
        AscendC::LocalTensor<bfloat16_t> qg,
        AscendC::LocalTensor<bfloat16_t> qgScaled,
        AscendC::LocalTensor<bfloat16_t> kg,
        AscendC::LocalTensor<bfloat16_t> vBeta,
        AscendC::LocalTensor<bfloat16_t> kBetaG,
        AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> betaEff,
        AscendC::LocalTensor<float> scratch, uint32_t validRows, float scale,
        AscendC::GlobalTensor<float> kBetaRelay,
        AscendC::GlobalTensor<float> vBetaRelay, uint32_t pair)
    {
        const uint32_t rhsRows = validRows > 32 ? 64 : 32;
        if (validRows == 0) {
            return;
        }
        const uint32_t last = (validRows - 1) * Shape::kHeadDim;
        const float directMin =
            domain_.StoredBound(ExpDomain::kV6LowerBase2);
        const float directMax =
            domain_.StoredBound(ExpDomain::kV6UpperBase2);
        // 同样的公式按整块（8 行）下发，逐行只影响行列步长参数，
        // 数学与舍入顺序与逐行版本完全一致。
        // Head-major has no live beta gather table. Use all 8 KiB scratch for
        // the factor and the dead private KBetaG tail for the 8 KiB stage.
        // Sequence-major retains the original 7-row scratch layout.
        const uint32_t kTileRows = args_.tiling.inputSequenceMajor ? 7U : 16U;
        const uint32_t kTileElems = kTileRows * Shape::kHeadDim;
        auto factor = scratch;
        auto stage = args_.tiling.inputSequenceMajor
                         ? scratch[kTileElems]
                         : kBetaG.template ReinterpretCast<float>()[1536];
        // 原 BF16 K_beta_g 缓冲不再作为输出；用其前 4 KiB 保存 beta 广播。
        auto betaBrcb = kBetaG.template ReinterpretCast<float>();
        auto betaRepeat = betaBrcb[512];
        AscendC::Brcb(betaBrcb, betaEff,
                      static_cast<uint8_t>(Shape::kChunkRows / 8), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t rowBegin = 0; rowBegin < validRows;
             rowBegin += kTileRows) {
            const uint32_t rows = validRows - rowBegin < kTileRows
                                      ? validRows - rowBegin
                                      : kTileRows;
            const uint32_t elems = rows * Shape::kHeadDim;
            const uint32_t offset = rowBegin * Shape::kHeadDim;
            // E(G)=exp(clamp(g[row]))：这里刻意不减 g[last]，与 arch35 的
            // V6 语义（qg/qBeta 用行内累计量，kg 才用 g[last]-g[row]）一致。
            AscendC::Maxs(factor, g[offset], directMin, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(factor, factor, directMax, elems);
            AscendC::PipeBarrier<PIPE_V>();
            if (domain_.useExp2) {
                AscendC::Muls(factor, factor, domain_.expInputScale, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Exp(factor, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(stage, qg[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qg[offset], stage, AscendC::RoundMode::CAST_RINT,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            // Khat*E(G)*beta 连续在 FP32 域计算，不经过 BF16 中间量。
            AscendC::Cast(stage, kg[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            CopyRowScalarBlocks(betaRepeat, betaBrcb[8 * rowBegin], rows);
            AscendC::PipeBarrier<PIPE_V>();
            MulTileByRowScalars(stage, stage, betaRepeat, rows);
            StartRhsFp32(kBetaRelay[offset], stage, elems, pair);
            // E'(G)=exp(clamp(g[last]-g[row]))
            SubRowMinusTile(factor, g[last], g[offset], rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(factor, factor, directMin, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(factor, factor, directMax, elems);
            AscendC::PipeBarrier<PIPE_V>();
            if (domain_.useExp2) {
                AscendC::Muls(factor, factor, domain_.expInputScale, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Exp(factor, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            // Overlap factor-only arithmetic with MTE3's last stage read.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
            AscendC::Cast(stage, kg[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kg[offset], stage, AscendC::RoundMode::CAST_RINT,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            // qgScaled 从已舍入的 BF16 qg 中间量回读。正序处理时，BF16 输出
            // 只覆盖已经消费完的 FP32 G 低地址，不会覆盖后续 G 行。
            AscendC::Cast(stage, qg[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(stage, stage, scale, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qgScaled[offset], stage,
                          AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // V_beta 同样只做一次 FP32 缩放，复用已经建立的 beta 广播。
        for (uint32_t rowBegin = 0; rowBegin < validRows;
             rowBegin += kTileRows) {
            const uint32_t rows = validRows - rowBegin < kTileRows
                                      ? validRows - rowBegin
                                      : kTileRows;
            const uint32_t elems = rows * Shape::kHeadDim;
            const uint32_t offset = rowBegin * Shape::kHeadDim;
            CopyRowScalarBlocks(betaRepeat, betaBrcb[8 * rowBegin],
                                        rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(stage, vBeta[offset], AscendC::RoundMode::CAST_NONE,
                          elems);
            AscendC::PipeBarrier<PIPE_V>();
            MulTileByRowScalars(stage, stage, betaRepeat, rows);
            AscendC::PipeBarrier<PIPE_V>();
            StoreRhsFp32(vBetaRelay[offset], stage, elems, pair);
        }
        // C7 按 32/64 行计算，尾行的两个 RHS 必须明确补零。
        AscendC::Duplicate(stage, 0.0f, kTileElems);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = validRows; row < rhsRows; row += kTileRows) {
            const uint32_t rows = rhsRows - row < kTileRows ? rhsRows - row : kTileRows;
            StoreRhsFp32(kBetaRelay[row * Shape::kHeadDim], stage,
                         rows * Shape::kHeadDim, pair);
            StoreRhsFp32(vBetaRelay[row * Shape::kValueDim], stage,
                         rows * Shape::kValueDim, pair);
        }
    }

    __aicore__ inline void StoreRhsFp32(AscendC::GlobalTensor<float> dst,
                                        AscendC::LocalTensor<float> src,
                                        uint32_t count, uint32_t pair)
    {
        StartRhsFp32(dst, src, count, pair);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
    }

    __aicore__ inline void StartRhsFp32(AscendC::GlobalTensor<float> dst,
                                      AscendC::LocalTensor<float> src,
                                      uint32_t count, uint32_t pair)
    {
        // Caller must consume MTE3_V before src is overwritten or reused.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::DataCopy(dst, src, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
    }

    PrepareKernelArgs args_{};
    bool coarseEligible_ = false;
    AscendC::TPipe *pipe_ = nullptr;
    uint32_t workgroup_ = 0;
    uint32_t aiv_ = 0;
    uint32_t coreCount_ = 0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_{};
    AscendC::TEventID ioFree_[2]{};
    AscendC::TEventID inputReady_[2]{};
    AscendC::TEventID outputReady_[2]{};
    AscendC::TEventID mte3ToV_[2]{};
    AscendC::TEventID scalarRead_{};
    AscendC::TEventID scalarWrite_{};
    AscendC::TEventID sharedFree_{};
    AscendC::GlobalTensor<bfloat16_t> qGm_{};
    AscendC::GlobalTensor<bfloat16_t> kGm_{};
    AscendC::GlobalTensor<bfloat16_t> vGm_{};
    AscendC::GlobalTensor<GateT> gateGm_{};
    AscendC::GlobalTensor<BetaT> betaGm_{};
    AscendC::GlobalTensor<DTYPE_DT_BIAS> dtBiasGm_{};
    AscendC::GlobalTensor<DTYPE_A_LOG> aLogGm_{};
    AscendC::GlobalTensor<bfloat16_t> qgGm_{};
    AscendC::GlobalTensor<bfloat16_t> qgScaledGm_{};
    AscendC::GlobalTensor<bfloat16_t> kgGm_{};
    AscendC::GlobalTensor<float> gkGm_{};
    AscendC::GlobalTensor<bfloat16_t> aqkGm_{};
    AscendC::GlobalTensor<bfloat16_t> akkGm_{};
    AscendC::GlobalTensor<bfloat16_t> qHatGm_{};
    AscendC::GlobalTensor<bfloat16_t> kHatGm_{};
    AscendC::GlobalTensor<float> qRstdGm_{};
    AscendC::GlobalTensor<float> kRstdGm_{};
    AscendC::GlobalTensor<float> betaEffGm_{};
};

} // namespace KdaPrepare::Arch22

#endif // ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H
