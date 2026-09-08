/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H
#define PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace KdaPrepare::Arch22 {

template <typename InputT, typename ValueT, typename GateT, typename BetaT,
          typename ScoreT, typename Policy>
class ChunkKdaFwdPrepareVec {
    using Domain = ExpDomainTraits<Policy::useExp2>;

public:
    __aicore__ inline void Init(const PrepareKernelArgs &args, AscendC::TPipe *pipe)
    {
        args_ = args;
        pipe_ = pipe;
        workgroup_ = WorkgroupId();
        aiv_ = AscendC::GetSubBlockIdx();
        coreCount_ = args_.tiling.usedCoreNum;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ValueT *>(args_.v));
        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(args_.rawGate));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(args_.beta));
        if (args_.dtBias != nullptr) {
            dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.dtBias));
        }
        if (args_.aLog != nullptr) {
            aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.aLog));
        }
        qgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.qg));
        kgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.kg));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.gk));
        aqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.aqk));
        if constexpr (Policy::abi == PrepareAbi::Current) {
            akkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.akk));
        }
        if (coreCount_ == 0) {
            return;
        }
        pipe_->InitBuffer(ubBuf_, Arch22Ub::kUsableBytes);
        scalarRead_ = pipe_->AllocEventID<AscendC::HardEvent::V_S>();
        scalarWrite_ = pipe_->AllocEventID<AscendC::HardEvent::S_V>();
        sharedFree_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        // 两个 pair 分时复用共享 G/scratch；初始许可只发布一次。
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        for (uint32_t pair = 0; pair < 2; ++pair) {
            ioFree_[pair] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            inputReady_[pair] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            outputReady_[pair] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            v0StoreDone_[pair] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreCount_ == 0) {
            return;
        }
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
            for (uint32_t groupBegin = headBegin; groupBegin < headEnd;
                 groupBegin += Shape::kHeadsPerGroup) {
                // AIV0 处理 0/2，AIV1 处理 1/3；两个 pair 分时复用共享 G 区。
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t localHead = pair * 2 + aiv_;
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                        Arch22FlagId(Arch22CrossCore::kFreeBase, pair));
                    const uint32_t valueHead = groupBegin + localHead;
                    if (valueHead < headEnd) {
                        StageV0(chunk, valueHead, localHead, pair);
                        StageV1(chunk, localHead, pair);
                    }
                    // 尾部无任务的 AIV 仍参加集合，但不计算地址或访问 GM。
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        Arch22FlagId(Arch22CrossCore::kReadyBase, pair));
                }
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t localHead = pair * 2 + aiv_;
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                        Arch22FlagId(Arch22CrossCore::kFreeBase, pair));
                    const uint32_t valueHead = groupBegin + localHead;
                    if (valueHead < headEnd) {
                        StageV3(chunk, valueHead, localHead, pair);
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        Arch22FlagId(Arch22CrossCore::kReadyBase, pair));
                }
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t localHead = pair * 2 + aiv_;
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                        Arch22FlagId(Arch22CrossCore::kFreeBase, pair));
                    const uint32_t valueHead = groupBegin + localHead;
                    if (valueHead < headEnd) {
                        StageV6(chunk, valueHead, localHead, pair);
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        Arch22FlagId(Arch22CrossCore::kReadyBase, pair));
                }
            }
        }
        // 消费最后一次 C7 发布，保证每次 set 都有对应 wait。
        for (uint32_t pair = 0; pair < 2; ++pair) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                Arch22FlagId(Arch22CrossCore::kFreeBase, pair));
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
    template <typename OutputT>
    __aicore__ inline void ClampFp32BeforeCast(
        AscendC::LocalTensor<float> tensor, uint32_t count)
    {
        if constexpr (std::is_same_v<OutputT, half>) {
            AscendC::Mins(tensor, tensor, 65504.0F, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(tensor, tensor, -65504.0F, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

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
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(v0StoreDone_[pair]);
    }

    __aicore__ inline void StageV0(const ChunkRange &chunk, uint32_t valueHead,
                                   uint32_t localHead, uint32_t pair)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
        const uint32_t base = Arch22Ub::kPrivateBase[pair];
        auto ub = ubBuf_.Get<uint8_t>();
        auto q = ub[base + Arch22Ub::kQ].template ReinterpretCast<InputT>();
        auto k = ub[base + Arch22Ub::kK].template ReinterpretCast<InputT>();
        const uint32_t gateBase = sizeof(GateT) == 4 ? Arch22Ub::kSharedG
                                                     : base + Arch22Ub::kGateOrKMinus;
        auto gate = ub[gateBase].template ReinterpretCast<GateT>();
        auto beta = ub[base + Arch22Ub::kBetaRaw].template ReinterpretCast<BetaT>();
        auto betaEff = ub[base + Arch22Ub::kBetaEff].template ReinterpretCast<float>();
        auto dtBias = ub[base + Arch22Ub::kDtBias].template ReinterpretCast<float>();
        auto aLog = ub[base + Arch22Ub::kALog].template ReinterpretCast<float>();
        auto g = ub[Arch22Ub::kSharedG].template ReinterpretCast<float>();
        auto scratch = ub[Arch22Ub::kSharedScratch].template ReinterpretCast<float>();
        const uint64_t qkOffset = QkInputOffset(
            args_.tiling, chunk, QkHeadForValueHead(args_.tiling, valueHead));
        const uint64_t gateOffset =
            RawGateInputOffset(args_.tiling, chunk, valueHead);
        const uint64_t headOutputOffset =
            HeadTensorOffset(args_.tiling, chunk, valueHead, Shape::kHeadDim);
        const uint32_t qkStride = args_.tiling.inputSequenceMajor
                                      ? (args_.tiling.qkHeadNum - 1) *
                                            Shape::kHeadDim * sizeof(InputT)
                                      : 0;
        const uint32_t gateStride = args_.tiling.inputSequenceMajor
                                        ? (args_.tiling.valueHeadNum - 1) *
                                              Shape::kHeadDim * sizeof(GateT)
                                        : 0;
        // 获得共享 G/scratch 的独占许可；V1 完成最后一次 V 读取后归还。
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::DataCopyExtParams qkCopy{static_cast<uint16_t>(chunk.validRows),
            Shape::kHeadDim * sizeof(InputT), qkStride, 0, 0};
        AscendC::DataCopyPadExtParams<InputT> qkPad{false, 0, 0, 0};
        AscendC::DataCopyPad(q, qGm_[qkOffset], qkCopy, qkPad);
        AscendC::DataCopyPad(k, kGm_[qkOffset], qkCopy, qkPad);
        AscendC::DataCopyExtParams gateCopy{static_cast<uint16_t>(chunk.validRows),
            Shape::kHeadDim * sizeof(GateT), gateStride, 0, 0};
        AscendC::DataCopyPadExtParams<GateT> gatePad{false, 0, 0, 0};
        AscendC::DataCopyPad(gate, gateGm_[gateOffset], gateCopy, gatePad);
        AscendC::DataCopyExtParams betaCopy{
            1, chunk.validRows * sizeof(BetaT), 0, 0, 0};
        AscendC::DataCopyPadExtParams<BetaT> betaPad{false, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> fp32Pad{false, 0, 0, 0};
        AscendC::DataCopyPad(beta,
            betaGm_[KdaPrepare::HeadScalarOffset(args_.tiling, chunk, valueHead)],
            betaCopy, betaPad);
        if constexpr (Policy::gateMode != GateMode::PrecomputedStep) {
            if (args_.tiling.hasDtBias) {
                // TODO：dt_bias 的 batch/head 排列需由公开 ABI 冻结。
                AscendC::DataCopyPad(dtBias, dtBiasGm_[valueHead * Shape::kHeadDim],
                    AscendC::DataCopyExtParams{1, Shape::kHeadDim * sizeof(float), 0, 0, 0}, fp32Pad);
            }
            if (args_.aLog != nullptr) {
                // TODO：A_log 的 head 索引需由公开 ABI 冻结。
                AscendC::DataCopyPad(aLog, aLogGm_[valueHead],
                    AscendC::DataCopyExtParams{1, sizeof(float), 0, 0, 0},
                    fp32Pad);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        // 唯一一次 VF：Q/K 可选 L2 norm，beta 变换，gate 变换和逐 token cumsum；
        // 生成 Qhat/Khat/G/Glast/betaEff，并清零所有无效行。
        // TODO：按目标 c220 头文件补齐寄存器、mask、归约和 repeat/stride 参数。
        V0Vf(q, k, gate, beta, betaEff, dtBias, aLog, g, scratch, chunk.validRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride);
        AscendC::GlobalTensor<InputT> qhatContext;
        AscendC::GlobalTensor<InputT> khatContext;
        AscendC::GlobalTensor<float> betaContext;
        qhatContext.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.workspace + slot + Workspace::kQHat));
        khatContext.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.workspace + slot + Workspace::kKHat));
        betaContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + slot + Workspace::kBetaEff));
        AscendC::DataCopy(qhatContext, q, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(khatContext, k, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopyPad(betaContext, betaEff,
            AscendC::DataCopyExtParams{1, chunk.validRows * sizeof(float), 0, 0, 0});
        if constexpr (Policy::abi == PrepareAbi::Current) {
            AscendC::DataCopy(gkGm_[headOutputOffset], g,
                              chunk.validRows * Shape::kHeadDim);
        } else {
            AscendC::GlobalTensor<float> gContext;
            gContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + slot + Workspace::kG));
            AscendC::DataCopy(gContext, g, chunk.validRows * Shape::kHeadDim);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(v0StoreDone_[pair]);
    }

    __aicore__ inline void StageV1(const ChunkRange &chunk, uint32_t localHead,
                                   uint32_t pair)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(v0StoreDone_[pair]);
        const uint32_t base = Arch22Ub::kPrivateBase[pair];
        auto ub = ubBuf_.Get<uint8_t>();
        auto qHat = ub[base + Arch22Ub::kQ].template ReinterpretCast<InputT>();
        auto kHat = ub[base + Arch22Ub::kK].template ReinterpretCast<InputT>();
        auto qPlus = ub[base + Arch22Ub::kQ].template ReinterpretCast<ScoreT>();
        auto kPlus = ub[base + Arch22Ub::kK].template ReinterpretCast<ScoreT>();
        auto kMinus = ub[base + Arch22Ub::kGateOrKMinus].template ReinterpretCast<ScoreT>();
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
        AscendC::GlobalTensor<ScoreT> payload;
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride);
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ ScoreT *>(args_.workspace + slot + Workspace::kPayload));
        AscendC::DataCopy(payload, qPlus, Shape::kScorePayloadBytes / sizeof(ScoreT));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
    }

    __aicore__ inline void StageV3(const ChunkRange &chunk, uint32_t valueHead,
                                   uint32_t localHead, uint32_t pair)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
        const uint32_t base = Arch22Ub::kPrivateBase[pair];
        auto ub = ubBuf_.Get<uint8_t>();
        auto raw = ub[base + Arch22Ub::kV3CompactRaw].template ReinterpretCast<float>();
        auto aqk = ub[base + Arch22Ub::kV3Aqk].template ReinterpretCast<InputT>();
        auto lkk = ub[base + Arch22Ub::kV3Lkk].template ReinterpretCast<float>();
        auto b = ub[base + Arch22Ub::kV3B].template ReinterpretCast<float>();
        auto x0 = ub[base + Arch22Ub::kV3X0].template ReinterpretCast<float>();
        auto x1 = ub[base + Arch22Ub::kV3X1].template ReinterpretCast<float>();
        auto negX1 = ub[base + Arch22Ub::kV3NegX1].template ReinterpretCast<float>();
        auto betaEff = ub[Arch22Ub::kV3BetaEff].template ReinterpretCast<float>();
        auto akkPack = ub[base + Arch22Ub::kV3AkkPack]
                           .template ReinterpretCast<InputT>();
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
            AscendC::DataCopyExtParams{1, chunk.validRows * sizeof(float), 0, 0, 0}, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        // 唯一一次 VF：因果 mask、scale、beta、两个 32x32 叶逆，以及
        // Aqk/B/X0/X1/negX1/稳定 Akk；negX1 供 C5 做普通 Mmad。
        // TODO：按目标 c220 头文件补齐叶逆寄存器分块和谓词参数。
        V3Vf(raw, betaEff, aqk, lkk, b, x0, x1, negX1, akkPack,
             chunk.validRows, args_.tiling.scale);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::DataCopy(aqkGm_[AOutputOffset(args_.tiling, chunk, valueHead)],
                          aqk, chunk.validRows * Shape::kChunkRows);
        // C4 固定读取完整 64x64 矩阵，因此补零后的中转矩阵始终写入
        // 工作空间；公开 Akk 只有 T 行，尾 chunk 只能写有效行。
        AscendC::GlobalTensor<InputT> akkRelay;
        akkRelay.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(
            args_.workspace + slot + Workspace::kPayload + Workspace::kAkk));
        AscendC::DataCopy(akkRelay, akkPack,
                          Shape::kChunkRows * Shape::kChunkRows);
        if constexpr (Policy::abi == PrepareAbi::Current) {
            AscendC::DataCopy(
                akkGm_[AOutputOffset(args_.tiling, chunk, valueHead)],
                akkPack, chunk.validRows * Shape::kChunkRows);
        }
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
        auto qg = ub[base + Arch22Ub::kV6Qg].template ReinterpretCast<InputT>();
        auto kg = ub[base + Arch22Ub::kV6Kg].template ReinterpretCast<InputT>();
        auto vBeta = ub[base + Arch22Ub::kV6VBeta].template ReinterpretCast<ValueT>();
        auto kBetaG = ub[base + Arch22Ub::kV6KBetaG].template ReinterpretCast<InputT>();
        auto g = ub[Arch22Ub::kSharedG].template ReinterpretCast<float>();
        auto scratch = ub[Arch22Ub::kSharedScratch].template ReinterpretCast<float>();
        auto betaEff = ub[base + Arch22Ub::kBetaEff].template ReinterpretCast<float>();
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch22WorkgroupStride);
        AscendC::GlobalTensor<InputT> qhat;
        AscendC::GlobalTensor<InputT> khat;
        AscendC::GlobalTensor<float> betaContext;
        qhat.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.workspace + slot + Workspace::kQHat));
        khat.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.workspace + slot + Workspace::kKHat));
        betaContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + slot + Workspace::kBetaEff));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::DataCopy(qg, qhat, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kg, khat, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(betaEff, betaContext,
            AscendC::DataCopyExtParams{1, chunk.validRows * sizeof(float), 0, 0, 0}, pad);
        if constexpr (Policy::abi == PrepareAbi::Current) {
            AscendC::DataCopy(g,
                              gkGm_[HeadTensorOffset(args_.tiling, chunk, valueHead,
                                                    Shape::kHeadDim)],
                              chunk.validRows * Shape::kHeadDim);
        } else {
            AscendC::GlobalTensor<float> gContext;
            gContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + slot + Workspace::kG));
            AscendC::DataCopy(g, gContext, chunk.validRows * Shape::kHeadDim);
        }
        const uint32_t vStride = args_.tiling.inputSequenceMajor
                                     ? (args_.tiling.valueHeadNum - 1) *
                                           Shape::kValueDim * sizeof(ValueT)
                                     : 0;
        AscendC::DataCopyPad(vBeta,
            vGm_[ValueInputOffset(args_.tiling, chunk, valueHead)],
            AscendC::DataCopyExtParams{static_cast<uint16_t>(chunk.validRows),
                Shape::kValueDim * sizeof(ValueT), vStride, 0, 0},
            AscendC::DataCopyPadExtParams<ValueT>{false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[pair]);
        // 唯一一次 VF：Qg、kg、两次舍入的 K_beta_g 和 V_beta；
        // 两条编译路径使用等价的 base-2/自然对数截断范围。
        // TODO：按目标 c220 头文件补齐 Exp 的饱和和舍入参数。
        V6Vf(qg, kg, vBeta, kBetaG, g, betaEff, scratch,
             chunk.validRows, args_.tiling.scale);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(sharedFree_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[pair]);
        const uint64_t out =
            HeadTensorOffset(args_.tiling, chunk, valueHead, Shape::kHeadDim);
        AscendC::DataCopy(qgGm_[out], qg, chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kgGm_[out], kg, chunk.validRows * Shape::kHeadDim);
        const uint32_t rhsRows = chunk.validRows > 32 ? 64 : 32;
        AscendC::GlobalTensor<InputT> kBetaRelay;
        AscendC::GlobalTensor<ValueT> vBetaRelay;
        kBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(args_.workspace + slot + Workspace::kPayload + Workspace::kKBetaG));
        vBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ ValueT *>(args_.workspace + slot + Workspace::kPayload + Workspace::kVBeta));
        AscendC::DataCopy(kBetaRelay, kBetaG, rhsRows * Shape::kHeadDim);
        AscendC::DataCopy(vBetaRelay, vBeta, rhsRows * Shape::kValueDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
    }

    __aicore__ inline void V0Vf(
        AscendC::LocalTensor<InputT> q, AscendC::LocalTensor<InputT> k,
        AscendC::LocalTensor<GateT> gate, AscendC::LocalTensor<BetaT> beta,
        AscendC::LocalTensor<float> betaEff, AscendC::LocalTensor<float> dtBias,
        AscendC::LocalTensor<float> aLog, AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> scratch, uint32_t validRows)
    {
        const uint32_t count = validRows * Shape::kHeadDim;
        if constexpr (Policy::normMode == QkNormMode::L2) {
            // 每行按冻结语义执行 x * rsqrt(sum(x^2) + epsilon)。
            // TODO：确认 c220 ReduceSum 临时区大小和地址对齐。
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
                AscendC::Muls(scratch, scratch, ReadScalar(betaEff, 0),
                              Shape::kHeadDim);
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
                AscendC::Muls(scratch, scratch, ReadScalar(betaEff, 0),
                              Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(k[row * Shape::kHeadDim], scratch,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        if constexpr (Policy::betaMode == BetaMode::Raw) {
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
            if constexpr (Policy::betaMode == BetaMode::TwoSigmoid) {
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
        if constexpr (Policy::gateMode != GateMode::PrecomputedStep) {
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
            if constexpr (!Policy::safeGate &&
                          Policy::gateMode == GateMode::Softplus) {
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
                               static_cast<InputT>(0), tail);
            AscendC::Duplicate(k[validRows * Shape::kHeadDim],
                               static_cast<InputT>(0), tail);
            AscendC::Duplicate(g[validRows * Shape::kHeadDim], 0.0F, tail);
        }
    }

    __aicore__ inline void V1Vf(
        AscendC::LocalTensor<InputT> qHat, AscendC::LocalTensor<InputT> kHat,
        AscendC::LocalTensor<ScoreT> qPlus, AscendC::LocalTensor<ScoreT> kPlus,
        AscendC::LocalTensor<ScoreT> kMinus, AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> scratch, uint32_t validRows)
    {
        constexpr float base2Min = std::is_same_v<ScoreT, half>
                                       ? ExpDomain::kV1Fp16LowerBase2
                                       : ExpDomain::kV1Bf16LowerBase2;
        constexpr float base2Max = std::is_same_v<ScoreT, half>
                                       ? ExpDomain::kV1Fp16UpperBase2
                                       : ExpDomain::kV1Bf16UpperBase2;
        constexpr float clampMin = Domain::StoredBound(base2Min);
        constexpr float clampMax = Domain::StoredBound(base2Max);
        auto work = scratch[Shape::kHeadDim];

        // Kplus 与 Khat 原位复用，所以必须先生成完四个 Kminus 前缀。
        // 否则后一个参考块会错误读取已经舍入成 Kplus 的数据。
        for (uint32_t s = 0; s < Shape::kSubChunkCount; ++s) {
            const uint32_t blockBegin = s * Shape::kSubChunkRows;
            auto kMinusBlock = kMinus[
                (Arch22Ub::kKMinus[s] - Arch22Ub::kGateOrKMinus) /
                sizeof(ScoreT)];
            AscendC::Duplicate(kMinusBlock, static_cast<ScoreT>(0),
                               Shape::kPrefixRows[s] * Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            if (blockBegin >= validRows) {
                continue;
            }
            const uint32_t blockEnd = blockBegin + Shape::kSubChunkRows < validRows
                                          ? blockBegin + Shape::kSubChunkRows
                                          : validRows;
            // 半开区间 [begin,end) 的中点取 floor((begin+end)/2)。
            const uint32_t midpoint = (blockBegin + blockEnd) / 2;
            const uint32_t prefix = Shape::kPrefixRows[s] < validRows
                                        ? Shape::kPrefixRows[s]
                                        : validRows;
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
                ClampFp32BeforeCast<ScoreT>(work, Shape::kHeadDim);
                AscendC::Cast(kMinusBlock[row * Shape::kHeadDim], work,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        // 四个 Kminus 都已完成，此时可以把 Qhat/Khat 原位改写为
        // Qplus/Kplus；无效行沿用 V0 写入的零。
        for (uint32_t s = 0; s < Shape::kSubChunkCount; ++s) {
            const uint32_t blockBegin = s * Shape::kSubChunkRows;
            if (blockBegin >= validRows) {
                continue;
            }
            const uint32_t blockEnd = blockBegin + Shape::kSubChunkRows < validRows
                                          ? blockBegin + Shape::kSubChunkRows
                                          : validRows;
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
                ClampFp32BeforeCast<ScoreT>(work, Shape::kHeadDim);
                AscendC::Cast(qPlus[row * Shape::kHeadDim], work,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(work, kHat[row * Shape::kHeadDim],
                              AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(work, work, scratch, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                ClampFp32BeforeCast<ScoreT>(work, Shape::kHeadDim);
                AscendC::Cast(kPlus[row * Shape::kHeadDim], work,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void V3Vf(
        AscendC::LocalTensor<float> raw, AscendC::LocalTensor<float> betaEff,
        AscendC::LocalTensor<InputT> aqk, AscendC::LocalTensor<float> lkk,
        AscendC::LocalTensor<float> b, AscendC::LocalTensor<float> x0,
        AscendC::LocalTensor<float> x1, AscendC::LocalTensor<float> negX1,
        AscendC::LocalTensor<InputT> akkPack, uint32_t validRows, float scale)
    {
        AscendC::Duplicate(aqk, static_cast<InputT>(0),
                           Shape::kChunkRows * Shape::kChunkRows);
        AscendC::Duplicate(b, 0.0F, 1024);
        uint32_t stackedBand = 0;
        const uint32_t active = CeilDiv(validRows, Shape::kSubChunkRows);
        for (uint32_t s = 0; s < active; ++s) {
            const uint32_t n = Shape::kPrefixRows[s];
            const uint32_t rows = validRows - s * Shape::kSubChunkRows <
                                          Shape::kSubChunkRows
                                      ? validRows - s * Shape::kSubChunkRows
                                      : Shape::kSubChunkRows;
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t globalRow = s * Shape::kSubChunkRows + row;
                const uint32_t aqkCount = globalRow + 1 < n ? globalRow + 1 : n;
                const uint32_t akkCount = globalRow < n ? globalRow : n;
                auto rawAqk = raw[stackedBand + row * n];
                auto rawAkk = raw[stackedBand + Shape::kSubChunkRows * n +
                                  row * n];
                AscendC::Muls(rawAqk, rawAqk, scale, aqkCount);
                AscendC::PipeBarrier<PIPE_V>();
                ClampFp32BeforeCast<InputT>(rawAqk, aqkCount);
                AscendC::Cast(aqk[globalRow * Shape::kChunkRows],
                              rawAqk, AscendC::RoundMode::CAST_RINT,
                              aqkCount);
                const float betaValue = ReadScalar(betaEff, globalRow);
                if (globalRow < 32) {
                    if (akkCount != 0) {
                        AscendC::Muls(
                            lkk[globalRow * Shape::kChunkRows], rawAkk,
                            betaValue, akkCount);
                    }
                } else {
                    // rawAkk 的低 32 列直接落到最终 B，其余列只写 L11。
                    AscendC::Muls(b[(globalRow - 32) * 32], rawAkk,
                                  betaValue, 32);
                    const uint32_t l11Count = globalRow - 32;
                    if (l11Count != 0) {
                        AscendC::Muls(
                            lkk[globalRow * Shape::kChunkRows + 32],
                            rawAkk[32], betaValue, l11Count);
                    }
                }
                if (akkCount != 0) {
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
            stackedBand += 2 * Shape::kSubChunkRows * n;
        }

        AscendC::PipeBarrier<PIPE_V>();
        // I+Lkk=[[A00,0],[B,A11]]；在同一次 VF 中求两个 32 阶单位下三角逆。
        AscendC::Duplicate(x0, 0.0F, 1024);
        AscendC::Duplicate(x1, 0.0F, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t topRows = validRows < 32 ? validRows : 32;
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
        AscendC::Duplicate(akkPack, static_cast<InputT>(0), 4096);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < topRows; ++row) {
            if constexpr (std::is_same_v<InputT, half>) {
                // FP16 写出需要有限值裁剪，直接以裁剪结果生成目标数据，
                // 不先把 x0 搬到另一个 UB 地址。
                AscendC::Mins(raw, x0[row * 32], 65504.0F, 32);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(raw, raw, -65504.0F, 32);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(akkPack[row * Shape::kChunkRows], raw,
                              AscendC::RoundMode::CAST_RINT, 32);
            } else {
                AscendC::Cast(akkPack[row * Shape::kChunkRows],
                              x0[row * 32], AscendC::RoundMode::CAST_RINT,
                              32);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t row = 0; row < bottomRows; ++row) {
            if constexpr (std::is_same_v<InputT, half>) {
                // FP16 写出需要有限值裁剪，直接以裁剪结果生成目标数据，
                // 不先把 x1 搬到另一个 UB 地址。
                AscendC::Mins(raw, x1[row * 32], 65504.0F, 32);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(raw, raw, -65504.0F, 32);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(
                    akkPack[(row + 32) * Shape::kChunkRows + 32], raw,
                    AscendC::RoundMode::CAST_RINT, 32);
            } else {
                AscendC::Cast(
                    akkPack[(row + 32) * Shape::kChunkRows + 32],
                    x1[row * 32], AscendC::RoundMode::CAST_RINT, 32);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void V6Vf(
        AscendC::LocalTensor<InputT> qg, AscendC::LocalTensor<InputT> kg,
        AscendC::LocalTensor<ValueT> vBeta,
        AscendC::LocalTensor<InputT> kBetaG, AscendC::LocalTensor<float> g,
        AscendC::LocalTensor<float> betaEff,
        AscendC::LocalTensor<float> scratch, uint32_t validRows, float scale)
    {
        const uint32_t rhsRows = validRows > 32 ? 64 : 32;
        if (validRows == 0) {
            AscendC::Duplicate(qg, static_cast<InputT>(0),
                               rhsRows * Shape::kHeadDim);
            AscendC::Duplicate(kg, static_cast<InputT>(0),
                               rhsRows * Shape::kHeadDim);
            AscendC::Duplicate(kBetaG, static_cast<InputT>(0),
                               rhsRows * Shape::kHeadDim);
            AscendC::Duplicate(vBeta, static_cast<ValueT>(0),
                               rhsRows * Shape::kValueDim);
            return;
        }
        const uint32_t last = (validRows - 1) * Shape::kHeadDim;
        for (uint32_t row = 0; row < rhsRows; ++row) {
            const uint32_t offset = row * Shape::kHeadDim;
            auto work = scratch[Shape::kHeadDim];
            if (row >= validRows) {
                AscendC::Duplicate(qg[offset], static_cast<InputT>(0),
                                   Shape::kHeadDim);
                AscendC::Duplicate(kg[offset], static_cast<InputT>(0),
                                   Shape::kHeadDim);
                AscendC::Duplicate(kBetaG[offset], static_cast<InputT>(0),
                                   Shape::kHeadDim);
                AscendC::Duplicate(vBeta[offset], static_cast<ValueT>(0),
                                   Shape::kValueDim);
                continue;
            }

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
            ClampFp32BeforeCast<InputT>(work, Shape::kHeadDim);
            AscendC::Cast(qg[offset], work, AscendC::RoundMode::CAST_RINT,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            // 第一次舍入：Khat*E(G) 先写入 kBetaG 的 2 字节存储。
            AscendC::Cast(work, kg[offset], AscendC::RoundMode::CAST_NONE,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(work, work, scratch, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            ClampFp32BeforeCast<InputT>(work, Shape::kHeadDim);
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
            ClampFp32BeforeCast<InputT>(scratch, Shape::kHeadDim);
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
            ClampFp32BeforeCast<InputT>(work, Shape::kHeadDim);
            AscendC::Cast(kg[offset], work, AscendC::RoundMode::CAST_RINT,
                          Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(work, vBeta[row * Shape::kValueDim],
                          AscendC::RoundMode::CAST_NONE, Shape::kValueDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(work, work, betaValue, Shape::kValueDim);
            AscendC::PipeBarrier<PIPE_V>();
            ClampFp32BeforeCast<ValueT>(work, Shape::kValueDim);
            AscendC::Cast(vBeta[row * Shape::kValueDim], work,
                          AscendC::RoundMode::CAST_RINT, Shape::kValueDim);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (Policy::abi == PrepareAbi::Fused) {
                AscendC::Cast(work, qg[offset], AscendC::RoundMode::CAST_NONE,
                              Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(work, work, scale, Shape::kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                ClampFp32BeforeCast<InputT>(work, Shape::kHeadDim);
                AscendC::Cast(qg[offset], work,
                              AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
            }
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
    AscendC::TEventID v0StoreDone_[2]{};
    AscendC::TEventID scalarRead_{};
    AscendC::TEventID scalarWrite_{};
    AscendC::TEventID sharedFree_{};
    AscendC::GlobalTensor<InputT> qGm_{};
    AscendC::GlobalTensor<InputT> kGm_{};
    AscendC::GlobalTensor<ValueT> vGm_{};
    AscendC::GlobalTensor<GateT> gateGm_{};
    AscendC::GlobalTensor<BetaT> betaGm_{};
    AscendC::GlobalTensor<float> dtBiasGm_{};
    AscendC::GlobalTensor<float> aLogGm_{};
    AscendC::GlobalTensor<InputT> qgGm_{};
    AscendC::GlobalTensor<InputT> kgGm_{};
    AscendC::GlobalTensor<float> gkGm_{};
    AscendC::GlobalTensor<InputT> aqkGm_{};
    AscendC::GlobalTensor<InputT> akkGm_{};
};

} // namespace KdaPrepare::Arch22

#endif // PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H
