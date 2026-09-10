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
#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace KdaPrepare::Arch22 {

template <typename GateT, typename BetaT, typename CompilePolicy>
class ChunkKdaFwdPrepareVec {
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;

public:
    __aicore__ inline void Init(const PrepareKernelArgs &args, AscendC::TPipe *pipe)
    {
        args_ = args;
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
            dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.dtBias));
        }
        if (args_.aLog != nullptr) {
            aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.aLog));
        }
        qgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.qg));
        qgScaledGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(args_.qgScaled));
        kgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.kg));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.gk));
        aqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.aqk));
        akkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.akk));
        qHatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.qHat));
        kHatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.kHat));
        qRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.qRstd));
        kRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.kRstd));
        betaEffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.betaEff));
        if (coreCount_ == 0) {
            return;
        }
        pipe_->InitBuffer(ubBuf_, Arch22Ub::kUsableBytes);
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
        const uint32_t workBegin = WorkBegin(total, workgroup_, coreCount_);
        const uint32_t workEnd = WorkEnd(total, workgroup_, coreCount_);
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
        AscendC::DataCopyPadExtParams<float> fp32Pad{false, 0, 0, 0};
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
        if constexpr (CompilePolicy::gateMode != GateMode::PrecomputedStep) {
            if (args_.tiling.hasDtBias) {
                // 公开接口将 dt_bias 固定为展平的 [HV,K]。
                AscendC::DataCopyPad(dtBias, dtBiasGm_[valueHead * Shape::kHeadDim],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(Shape::kHeadDim * sizeof(float)),
                        0, 0, 0}, fp32Pad);
            }
            if (args_.aLog != nullptr) {
                // 公开接口将 a_log 固定为 [HV]。
                AscendC::DataCopyPad(aLog, aLogGm_[valueHead],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
                    fp32Pad);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        if (args_.tiling.inputSequenceMajor) {
            auto offsets = ub[Arch22Ub::kBetaGatherOffsets]
                               .template ReinterpretCast<uint32_t>();
            AscendC::Gather(beta, betaStrided, offsets,
                            static_cast<uint32_t>(0), chunk.validRows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // 本 Stage 的一次向量计算完成 Q/K 可选 L2 norm、beta 变换、
        // gate 变换和逐 token cumsum，并生成全部公开中间量。
        V0Vf(q, k, qRstd, kRstd, gate, beta, betaEff, dtBias, aLog, g,
             scratch, chunk.validRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride);
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
            AscendC::DataCopy(qHatGm_[qkOutputOffset], q,
                              chunk.validRows * Shape::kHeadDim);
            AscendC::DataCopy(kHatGm_[qkOutputOffset], k,
                              chunk.validRows * Shape::kHeadDim);
            AscendC::DataCopyPad(qRstdGm_[rstdOutputOffset], qRstd,
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(chunk.validRows * sizeof(float)),
                    0, 0, 0});
            AscendC::DataCopyPad(kRstdGm_[rstdOutputOffset], kRstd,
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(chunk.validRows * sizeof(float)),
                    0, 0, 0});
        }
        AscendC::DataCopyPad(
            betaEffGm_[HeadScalarOffset(args_.tiling, chunk, valueHead)],
            betaEff, AscendC::DataCopyExtParams{
                         1, static_cast<uint32_t>(
                                chunk.validRows * sizeof(float)),
                         0, 0, 0});
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
            workgroup_, localHead, Workspace::kArch22WorkgroupStride);
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace + slot + Workspace::kPayload));
        AscendC::DataCopy(payload, qPlus, Shape::kScorePayloadBytes / sizeof(bfloat16_t));
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
            workgroup_, localHead, Workspace::kArch22WorkgroupStride);
        AscendC::GlobalTensor<float> payload;
        AscendC::GlobalTensor<float> betaContext;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + slot + Workspace::kPayload));
        betaContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + slot + Workspace::kBetaEff));
        const uint32_t active = CeilDiv(chunk.validRows, Shape::kSubChunkRows);
        uint32_t compactElements = 0;
        for (uint32_t s = 0; s < active; ++s) {
            compactElements +=
                2 * Shape::kSubChunkRows * Shape::kPrefixRows[s];
        }
        // C2 只写有效 sub-chunk；尾块仍保持一次搬运，但不读取未写 payload。
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::DataCopy(raw, payload, compactElements);
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(betaEff, betaContext,
            AscendC::DataCopyExtParams{
                1, static_cast<uint32_t>(chunk.validRows * sizeof(float)),
                0, 0, 0}, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        // 本 Stage 一次完成因果 mask、scale、beta 和两个 32x32 叶逆，
        // 生成 Aqk/B/X0/X1/negX1/稳定 Akk；negX1 供 C5 做普通 Mmad。
        V3Vf(raw, betaEff, aqk, lkk, b, x0, x1, negX1, akkPack,
             chunk.validRows, args_.tiling.scale);
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
        AscendC::DataCopy(
            akkGm_[AOutputOffset(args_.tiling, chunk, valueHead)],
            akkPack, chunk.validRows * Shape::kChunkRows);
        if (chunk.validRows > 32) {
            AscendC::DataCopy(payload[Workspace::kX0 / sizeof(float)], x0, 1024);
            AscendC::DataCopy(payload[Workspace::kNegX1 / sizeof(float)], negX1, 1024);
            AscendC::DataCopy(payload[Workspace::kB / sizeof(float)], b, 1024);
        }
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
            workgroup_, localHead, Workspace::kArch22WorkgroupStride);
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
        // 本 Stage 一次生成 qg、qgScaled、kg、两次舍入的 K_beta_g 和
        // V_beta；两条编译路径先在对应域截断，再统一调用自然底 Exp。
        V6Vf(qg, qgScaled, kg, vBeta, kBetaG, g, betaEff, scratch,
             chunk.validRows, args_.tiling.scale);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        const uint64_t out =
            HeadTensorOffset(args_.tiling, chunk, valueHead, Shape::kHeadDim);
        // 共享区只有 qgScaled 仍被 MTE3 读取，优先搬出并单独发布完成事件；
        // 后续私有区输出可继续在 MTE3 流水中执行。
        AscendC::DataCopy(qgScaledGm_[out], qgScaled,
                          chunk.validRows * Shape::kHeadDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
        AscendC::DataCopy(qgGm_[out], qg, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kgGm_[out], kg, chunk.validRows * Shape::kHeadDim);
        const uint32_t rhsRows = chunk.validRows > 32 ? 64 : 32;
        AscendC::GlobalTensor<bfloat16_t> kBetaRelay;
        AscendC::GlobalTensor<bfloat16_t> vBetaRelay;
        kBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload +
            Workspace::kKBetaG));
        vBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload +
            Workspace::kVBeta));
        AscendC::DataCopy(kBetaRelay, kBetaG, rhsRows * Shape::kHeadDim);
        AscendC::DataCopy(vBetaRelay, vBeta, rhsRows * Shape::kValueDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
        // qgScaled 与共享 G 复用地址；必须等 MTE3 读完，才能把共享区
        // 通过 V_MTE2 许可交给下一个 pair 的 MTE2。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[pair]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
    }

    __aicore__ inline void V0Vf(
        AscendC::LocalTensor<bfloat16_t> q, AscendC::LocalTensor<bfloat16_t> k,
        AscendC::LocalTensor<float> qRstd, AscendC::LocalTensor<float> kRstd,
        AscendC::LocalTensor<GateT> gate, AscendC::LocalTensor<BetaT> beta,
        AscendC::LocalTensor<float> betaEff, AscendC::LocalTensor<float> dtBias,
        AscendC::LocalTensor<float> aLog, AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> scratch, uint32_t validRows)
    {
        const uint32_t count = validRows * Shape::kHeadDim;
        if constexpr (CompilePolicy::normMode == QkNormMode::L2) {
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

        if constexpr (CompilePolicy::betaMode == BetaMode::Raw) {
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
            if constexpr (CompilePolicy::betaMode == BetaMode::TwoSigmoid) {
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
        if constexpr (CompilePolicy::gateMode != GateMode::PrecomputedStep) {
            float gateA = 1.0F;
            if (args_.aLog != nullptr) {
                // a_h = exp(A_log[h])，不能直接把 A_log 当成乘数。
                AscendC::Exp(aLog, aLog, 1);
                AscendC::PipeBarrier<PIPE_V>();
                gateA = ReadScalar(aLog, 0);
            }
            if (args_.tiling.hasDtBias) {
                for (uint32_t row = 0; row < validRows; ++row) {
                    AscendC::Add(g[row * Shape::kHeadDim],
                                 g[row * Shape::kHeadDim], dtBias,
                                 Shape::kHeadDim);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            if constexpr (!CompilePolicy::safeGate &&
                          CompilePolicy::gateMode == GateMode::Softplus) {
                // deltaG=-a*(max(x,0)+log(1+exp(-abs(x))))。
                const float factor = -gateA;
                for (uint32_t row = 0; row < validRows; ++row) {
                    auto gateRow = g[row * Shape::kHeadDim];
                    AscendC::Abs(scratch, gateRow, Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(scratch, scratch, -1.0F, Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Exp(scratch, scratch, Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(scratch, scratch, 1.0F, Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln(scratch, scratch, Shape::kHeadDim);
                    AscendC::Maxs(gateRow, gateRow, 0.0F, Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(gateRow, gateRow, scratch, Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(gateRow, gateRow, factor, Shape::kHeadDim);
                }
            } else {
                // deltaG=lower_bound/(1+exp(-a*x))。
                const float negativeA = -gateA;
                for (uint32_t row = 0; row < validRows; ++row) {
                    auto gateRow = g[row * Shape::kHeadDim];
                    AscendC::Muls(scratch, gateRow, negativeA,
                                  Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Exp(scratch, scratch, Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(scratch, scratch, 1.0F, Shape::kHeadDim);
                    AscendC::Duplicate(gateRow, args_.tiling.lowerBound,
                                       Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Div(gateRow, gateRow, scratch, Shape::kHeadDim);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        // true 保存 log2 累计量，false 保存自然对数累计量。
        if constexpr (Domain::useExp2) {
            AscendC::Muls(g, g, Domain::stepScale, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // token 维前缀和保留 headDim 向量宽度，不跨 chunk 传播。
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

    __aicore__ inline void V1Vf(
        AscendC::LocalTensor<bfloat16_t> qHat,
        AscendC::LocalTensor<bfloat16_t> kHat,
        AscendC::LocalTensor<bfloat16_t> qPlus,
        AscendC::LocalTensor<bfloat16_t> kPlus,
        AscendC::LocalTensor<bfloat16_t> kMinus,
        AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> scratch, uint32_t validRows)
    {
        constexpr float base2Min = ExpDomain::kV1Bf16LowerBase2;
        constexpr float base2Max = ExpDomain::kV1Bf16UpperBase2;
        constexpr float clampMin = Domain::StoredBound(base2Min);
        constexpr float clampMax = Domain::StoredBound(base2Max);
        auto work = scratch[Shape::kHeadDim];
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
        for (uint32_t s = 0; s < active; ++s) {
            const uint32_t blockBegin = s * Shape::kSubChunkRows;
            const uint32_t blockEnd = blockEnds[s];
            auto kMinusBlock = kMinus[
                (Arch22Ub::kKMinus[s] - Arch22Ub::kGateOrKMinus) /
                sizeof(bfloat16_t)];
            // 半开区间 [begin,end) 的中点取 floor((begin+end)/2)。
            const uint32_t midpoint = (blockBegin + blockEnd) / 2;
            const uint32_t prefix = blockEnd;
            for (uint32_t row = 0; row < prefix; ++row) {
                AscendC::Sub(scratch, g[midpoint * Shape::kHeadDim],
                             g[row * Shape::kHeadDim], Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(scratch, scratch, clampMin, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(scratch, scratch, clampMax, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (Domain::useExp2) {
                    AscendC::Muls(scratch, scratch, Domain::expInputScale,
                                  Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Exp(scratch, scratch, Shape::kHeadDim);
                AscendC::Cast(work, kHat[row * Shape::kHeadDim],
                              AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(work, work, scratch, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(kMinusBlock[row * Shape::kHeadDim], work,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        // 四个 Kminus 都已完成，此时可以把 Qhat/Khat 原位改写为
        // Qplus/Kplus；无效行沿用 V0 写入的零。
        for (uint32_t s = 0; s < active; ++s) {
            const uint32_t blockBegin = s * Shape::kSubChunkRows;
            const uint32_t blockEnd = blockEnds[s];
            const uint32_t midpoint = (blockBegin + blockEnd) / 2;
            for (uint32_t row = blockBegin; row < blockEnd; ++row) {
                AscendC::Sub(scratch, g[row * Shape::kHeadDim],
                             g[midpoint * Shape::kHeadDim], Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(scratch, scratch, clampMin, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(scratch, scratch, clampMax, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                // SIMD 侧统一调用自然 Exp；log2 域显式换算，ln 域直接计算。
                if constexpr (Domain::useExp2) {
                    AscendC::Muls(scratch, scratch, Domain::expInputScale,
                                  Shape::kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Exp(scratch, scratch, Shape::kHeadDim);
                AscendC::Cast(work, qHat[row * Shape::kHeadDim],
                              AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(work, work, scratch, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(qPlus[row * Shape::kHeadDim], work,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(work, kHat[row * Shape::kHeadDim],
                              AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(work, work, scratch, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(kPlus[row * Shape::kHeadDim], work,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
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
        AscendC::LocalTensor<bfloat16_t> akkPack, uint32_t validRows,
        float scale)
    {
        AscendC::Duplicate(aqk, static_cast<bfloat16_t>(0),
                           Shape::kChunkRows * Shape::kChunkRows);
        AscendC::Duplicate(b, 0.0F, 1024);
        // C2 按 s 堆叠 [rawAqk, rawAkk]，s 之前的 FP32 元素数为
        // subChunkRows^2*s*(s+1)。按全局行解包后，
        // 每个循环不再需要根据行号分支。
        for (uint32_t globalRow = 0; globalRow < validRows; ++globalRow) {
            const uint32_t s = globalRow / Shape::kSubChunkRows;
            const uint32_t row = globalRow % Shape::kSubChunkRows;
            const uint32_t n = Shape::kPrefixRows[s];
            const uint32_t stackedBand = Shape::kSubChunkRows *
                                         Shape::kSubChunkRows * s * (s + 1);
            auto rawAqk = raw[stackedBand + row * n];
            const uint32_t aqkCount = globalRow + 1;
            AscendC::Muls(rawAqk, rawAqk, scale, aqkCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(aqk[globalRow * Shape::kChunkRows], rawAqk,
                          AscendC::RoundMode::CAST_RINT, aqkCount);
        }

        const uint32_t topRows = validRows < 32 ? validRows : 32;
        // A00 的第 0 行没有严格下三角元素，从第 1 行开始写。
        for (uint32_t globalRow = 1; globalRow < topRows; ++globalRow) {
            const uint32_t s = globalRow / Shape::kSubChunkRows;
            const uint32_t row = globalRow % Shape::kSubChunkRows;
            const uint32_t n = Shape::kPrefixRows[s];
            const uint32_t stackedBand = Shape::kSubChunkRows *
                                         Shape::kSubChunkRows * s * (s + 1);
            auto rawAkk = raw[stackedBand + Shape::kSubChunkRows * n +
                              row * n];
            const float betaValue = ReadScalar(betaEff, globalRow);
            AscendC::Muls(lkk[globalRow * Shape::kChunkRows], rawAkk,
                          betaValue, globalRow);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (validRows > 32) {
            constexpr uint32_t firstBottomRow = 32;
            constexpr uint32_t firstBottomSubChunk =
                firstBottomRow / Shape::kSubChunkRows;
            constexpr uint32_t firstBottomN =
                Shape::kPrefixRows[firstBottomSubChunk];
            constexpr uint32_t firstBottomBand =
                Shape::kSubChunkRows * Shape::kSubChunkRows *
                firstBottomSubChunk * (firstBottomSubChunk + 1);
            auto firstBottomRawAkk =
                raw[firstBottomBand + Shape::kSubChunkRows * firstBottomN];
            const float firstBottomBeta = ReadScalar(betaEff, firstBottomRow);
            AscendC::Muls(b, firstBottomRawAkk, firstBottomBeta, 32);
            AscendC::PipeBarrier<PIPE_V>();

            // 第 32 行的 L11 长度为 0，单独处理后，余下行同时写 B 和 L11。
            for (uint32_t globalRow = firstBottomRow + 1;
                 globalRow < validRows; ++globalRow) {
                const uint32_t s = globalRow / Shape::kSubChunkRows;
                const uint32_t row = globalRow % Shape::kSubChunkRows;
                const uint32_t n = Shape::kPrefixRows[s];
                const uint32_t stackedBand = Shape::kSubChunkRows *
                                             Shape::kSubChunkRows * s *
                                             (s + 1);
                auto rawAkk = raw[stackedBand + Shape::kSubChunkRows * n +
                                  row * n];
                const float betaValue = ReadScalar(betaEff, globalRow);
                AscendC::Muls(b[(globalRow - 32) * 32], rawAkk,
                              betaValue, 32);
                AscendC::Muls(lkk[globalRow * Shape::kChunkRows + 32],
                              rawAkk[32], betaValue, globalRow - 32);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        AscendC::PipeBarrier<PIPE_V>();
        // I+Lkk=[[A00,0],[B,A11]]；在同一次 VF 中求两个 32 阶单位下三角逆。
        AscendC::Duplicate(x0, 0.0F, 1024);
        AscendC::Duplicate(x1, 0.0F, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t bottomRows = validRows > 32 ? validRows - 32 : 0;
        // 单位下三角逆逐行前代：X[i,:]=-sum(k<i,L[i,k]*X[k,:])，X[i,i]=1。
        // raw 已全部消费，其低地址在本段作为一行 FP32 临时区，不发生 UB 搬位。
        for (uint32_t row = 0; row < topRows; ++row) {
            AscendC::Duplicate(x0[row * 32 + row], 1.0F, 1);
            for (uint32_t kIndex = 0; kIndex < row; ++kIndex) {
                const float coefficient =
                    -ReadScalar(lkk, row * Shape::kChunkRows + kIndex);
                AscendC::Muls(raw, x0[kIndex * 32], coefficient, kIndex + 1);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(x0[row * 32], x0[row * 32], raw, kIndex + 1);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        for (uint32_t row = 0; row < bottomRows; ++row) {
            AscendC::Duplicate(x1[row * 32 + row], 1.0F, 1);
            for (uint32_t kIndex = 0; kIndex < row; ++kIndex) {
                const float coefficient = -ReadScalar(
                    lkk, (row + 32) * Shape::kChunkRows + 32 + kIndex);
                AscendC::Muls(raw, x1[kIndex * 32], coefficient, kIndex + 1);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(x1[row * 32], x1[row * 32], raw, kIndex + 1);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        // B 已在解包 rawAkk 时直接写入；其余行保持零。
        AscendC::PipeBarrier<PIPE_V>();
        // C5 直接执行普通 MMAD：negX1@T，不依赖不存在的 negate 参数。
        AscendC::Muls(negX1, x1, -1.0F, 1024);
        AscendC::PipeBarrier<PIPE_V>();

        // q00=X0、q01=0、q11=X1；q10 留给 C5 的 negX1@T。
        AscendC::Duplicate(akkPack, static_cast<bfloat16_t>(0), 4096);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < topRows; ++row) {
            AscendC::Cast(akkPack[row * Shape::kChunkRows],
                          x0[row * 32], AscendC::RoundMode::CAST_RINT, 32);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t row = 0; row < bottomRows; ++row) {
            AscendC::Cast(
                akkPack[(row + 32) * Shape::kChunkRows + 32],
                x1[row * 32], AscendC::RoundMode::CAST_RINT, 32);
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
        AscendC::LocalTensor<float> scratch, uint32_t validRows, float scale)
    {
        const uint32_t rhsRows = validRows > 32 ? 64 : 32;
        if (validRows == 0) {
            AscendC::Duplicate(qg, static_cast<bfloat16_t>(0),
                               rhsRows * Shape::kHeadDim);
            AscendC::Duplicate(kg, static_cast<bfloat16_t>(0),
                               rhsRows * Shape::kHeadDim);
            AscendC::Duplicate(kBetaG, static_cast<bfloat16_t>(0),
                               rhsRows * Shape::kHeadDim);
            AscendC::Duplicate(vBeta, static_cast<bfloat16_t>(0),
                               rhsRows * Shape::kValueDim);
            return;
        }
        const uint32_t last = (validRows - 1) * Shape::kHeadDim;
        for (uint32_t row = 0; row < validRows; ++row) {
            const uint32_t offset = row * Shape::kHeadDim;
            auto work = scratch[Shape::kHeadDim];
            constexpr float directMin =
                Domain::StoredBound(ExpDomain::kV6LowerBase2);
            constexpr float directMax =
                Domain::StoredBound(ExpDomain::kV6UpperBase2);
            AscendC::Maxs(scratch, g[offset], directMin, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(scratch, scratch, directMax, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (Domain::useExp2) {
                AscendC::Muls(scratch, scratch, Domain::expInputScale,
                              Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Exp(scratch, scratch, Shape::kHeadDim);
            AscendC::Cast(work, qg[offset], AscendC::RoundMode::CAST_NONE,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(work, work, scratch, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qg[offset], work, AscendC::RoundMode::CAST_RINT,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            // 第一次舍入：Khat*E(G) 先写入 kBetaG 的 BF16 存储。
            AscendC::Cast(work, kg[offset], AscendC::RoundMode::CAST_NONE,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(work, work, scratch, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kBetaG[offset], work,
                          AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(scratch, kBetaG[offset],
                          AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
            const float betaValue = ReadScalar(betaEff, row);
            AscendC::Muls(scratch, scratch, betaValue,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            // 第二次舍入：betaEff*(round(Khat*E(G)))。
            AscendC::Cast(kBetaG[offset], scratch,
                          AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub(scratch, g[last], g[offset], Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(scratch, scratch, directMin, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(scratch, scratch, directMax, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (Domain::useExp2) {
                AscendC::Muls(scratch, scratch, Domain::expInputScale,
                              Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Exp(scratch, scratch, Shape::kHeadDim);
            AscendC::Cast(work, kg[offset], AscendC::RoundMode::CAST_NONE,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(work, work, scratch, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kg[offset], work, AscendC::RoundMode::CAST_RINT,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(work, vBeta[row * Shape::kValueDim],
                          AscendC::RoundMode::CAST_NONE, Shape::kValueDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(work, work, betaValue, Shape::kValueDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vBeta[row * Shape::kValueDim], work,
                          AscendC::RoundMode::CAST_RINT, Shape::kValueDim);
            AscendC::PipeBarrier<PIPE_V>();
            // qgScaled 从已舍入的公开 qg 回读。正序处理时，BF16 输出
            // 只覆盖已经消费完的 FP32 G 低地址，不会覆盖后续 G 行。
            AscendC::Cast(work, qg[offset], AscendC::RoundMode::CAST_NONE,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(work, work, scale, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qgScaled[offset], work,
                          AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
        }
        // 有效行与补零行分开，VF 循环体内不做 runtime 分支。
        for (uint32_t row = validRows; row < rhsRows; ++row) {
            const uint32_t offset = row * Shape::kHeadDim;
            AscendC::Duplicate(qg[offset], static_cast<bfloat16_t>(0),
                               Shape::kHeadDim);
            AscendC::Duplicate(kg[offset], static_cast<bfloat16_t>(0),
                               Shape::kHeadDim);
            AscendC::Duplicate(kBetaG[offset], static_cast<bfloat16_t>(0),
                               Shape::kHeadDim);
            AscendC::Duplicate(vBeta[row * Shape::kValueDim],
                               static_cast<bfloat16_t>(0), Shape::kValueDim);
        }
    }

    PrepareKernelArgs args_{};
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
    AscendC::GlobalTensor<float> dtBiasGm_{};
    AscendC::GlobalTensor<float> aLogGm_{};
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
