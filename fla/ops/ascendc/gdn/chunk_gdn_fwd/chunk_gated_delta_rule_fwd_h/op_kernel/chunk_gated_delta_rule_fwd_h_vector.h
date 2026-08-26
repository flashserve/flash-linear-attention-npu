/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h_vector.h
 * \brief A2/A3 vector stages for chunk_gated_delta_rule_fwd_h.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_VECTOR_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_VECTOR_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "chunk_gated_delta_rule_fwd_h_policy.h"
#include "gemm/block/block_scheduler_gdn_fwd_h.hpp"

namespace GDN::FwdHStandalone {

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
class ChunkGatedDeltaRuleFwdHVector {
public:
    using ArchTag = Catlass::Arch::AtlasA2;
    using VecScheduler = Catlass::Gemm::Block::BlockSchedulerGdnFwdHVec;
    using Offsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    static constexpr bool kGOnly = GateMode == GDN_FWD_H_GATE_G;
    static constexpr bool kGkOnly = GateMode == GDN_FWD_H_GATE_GK;
    static constexpr bool kUseExp2 = ExpMode == GDN_FWD_H_EXP_2;
    static constexpr bool kSeparateRollingState = !std::is_same<StateT, InputT>::value;
    static constexpr float kLn2 = 0.6931471805599453f;
    static constexpr uint32_t kPongEventBase = 4;
    static constexpr uint32_t kRowTile = 16;

    static_assert(kGOnly || kGkOnly, "unsupported FwdH gate mode");
    static_assert(ExpMode == GDN_FWD_H_EXP_E || ExpMode == GDN_FWD_H_EXP_2,
                  "unsupported FwdH exponent mode");
    static_assert(std::is_same<WorkspaceT, float>::value,
                  "regular FwdH vector workspace must be FP32");

    __aicore__ inline ChunkGatedDeltaRuleFwdHVector() = default;

    __aicore__ inline void Init(
        GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
        GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR user, GM_ADDR tiling)
    {
        auto tilingData = reinterpret_cast<
            __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch_ = tilingData->batch;
        seqlen_ = tilingData->seqlen;
        kNumHead_ = tilingData->kNumHead;
        vNumHead_ = tilingData->vNumHead;
        kHeadDim_ = tilingData->kHeadDim;
        vHeadDim_ = tilingData->vHeadDim;
        chunkSize_ = tilingData->chunkSize;
        useInitialState_ = tilingData->useInitialState;
        storeFinalState_ = tilingData->storeFinalState;
        isVariedLen_ = tilingData->isVariedLen;
        shapeBatch_ = tilingData->shapeBatch;
        tokenBatch_ = tilingData->tokenBatch;
        numChunksWorkspaceOffset_ = tilingData->numChunksWorkspaceOffset;

        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(k));
        gmW_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(w));
        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(u));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        gmGk_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(gk));
        gmInitialState_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(initialState));
        gmH_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(h));
        gmVNew_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(vNew));
        gmFinalState_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(finalState));
        if constexpr (kSeparateRollingState) {
            if (!storeFinalState_) {
                constexpr uint64_t kGmAlign = 512;
                uint64_t numChunksBytes =
                    (static_cast<uint64_t>(tokenBatch_) + 1) * sizeof(int64_t);
                uint64_t hiddenOffset = numChunksWorkspaceOffset_ +
                    (numChunksBytes + kGmAlign) / kGmAlign * kGmAlign;
                gmFinalState_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ StateT *>(user + hiddenOffset));
                storeFinalState_ = true;
            }
        }
        gmVWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ WorkspaceT *>(
            user + tilingData->vWorkspaceOffset));
        gmVUpdateWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(
            user + tilingData->vUpdateWorkspaceOffset));
        gmHWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ WorkspaceT *>(
            user + tilingData->hWorkspaceOffset));

        scheduler_.Init(cuSeqlens, chunkIndices, tiling, user);
        BindUb();
    }

    __aicore__ inline void Process()
    {
        if (isVariedLen_) {
            AscendC::SyncAll<false>();
        }
        InitState();
        AscendC::SyncAll<false>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec2Done[0]);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec2Done[1]);
        InitEvents();

        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        while (scheduler_.isRunning) {
            scheduler_.InitTasks();
            if (!scheduler_.isRunning) {
                break;
            }
            const uint32_t windowId = scheduler_.GetWindowId();

            // Stage1 consumes Stage0 in cube order. AIV0 owns heads 0/2 and
            // AIV1 owns heads 1/3, matching the single-cube production order.
            for (uint32_t i = 0; i < scheduler_.GetHeadsInRound(); ++i) {
                const auto& headTask = scheduler_.GetHeadTask(i);
                if (scheduler_.HeadTaskIsDone(headTask)) {
                    continue;
                }
                Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube1Done[windowId]);
                if (i % subBlockNum == subBlockIdx) {
                    const Offsets& offsets = scheduler_.GetCurTaskOffsets(headTask);
                    const uint32_t localSlot = i / subBlockNum;
                    const bool isPing = localSlot == 0;
                    if (offsets.blockTokens < 16) {
                        Stage0Tail(offsets, EventId(EVENT_ID3, isPing));
                    }
                    Stage1(offsets, isPing, localSlot);
                }
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                    scheduler_.vec1Done[windowId]);
            }

            // Stage3 consumes Stage2 for the same round before the scheduler
            // advances to another chunk or four-head window.
            for (uint32_t i = 0; i < scheduler_.GetHeadsInRound(); ++i) {
                const auto& headTask = scheduler_.GetHeadTask(i);
                if (scheduler_.HeadTaskIsDone(headTask)) {
                    continue;
                }
                Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube2Done[windowId]);
                if (i % subBlockNum == subBlockIdx &&
                    scheduler_.NeedProcessStage2(headTask)) {
                    const Offsets& offsets = scheduler_.GetCurTaskOffsets(headTask);
                    const uint32_t localSlot = i / subBlockNum;
                    const bool isPing = localSlot == 0;
                    if (offsets.blockTokens < 16) {
                        Stage2Tail(offsets, EventId(EVENT_ID3, isPing));
                    }
                    Stage3(offsets, isPing, localSlot);
                }
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                scheduler_.vec2Done[windowId]);
        }
        DrainEvents();
    }

private:
    __aicore__ inline uint32_t EventId(uint32_t eventId, bool isPing) const
    {
        return eventId + (isPing ? 0 : kPongEventBase);
    }

    template <typename Element>
    __aicore__ inline void CopyGmToUb(
        AscendC::LocalTensor<Element> dst, AscendC::GlobalTensor<Element> src,
        uint32_t rows, uint32_t cols, uint32_t srcStride)
    {
        if (cols == srcStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(Element)),
            static_cast<uint32_t>((srcStride - cols) * sizeof(Element)), 0, 0};
        AscendC::DataCopyPadExtParams<Element> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    template <typename Element>
    __aicore__ inline void CopyUbToGm(
        AscendC::GlobalTensor<Element> dst, AscendC::LocalTensor<Element> src,
        uint32_t rows, uint32_t cols, uint32_t dstStride)
    {
        if (cols == dstStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(Element)), 0,
            static_cast<uint32_t>((dstStride - cols) * sizeof(Element)), 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void BindUb()
    {
        calcUb_ = resource_.ubBuf.template GetBufferByByte<float>(0);
        stage1WsUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(32 * 1024);
        stage1WsUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(96 * 1024);
        stage1InputUb_[0] = resource_.ubBuf.template GetBufferByByte<InputT>(64 * 1024);
        stage1InputUb_[1] = resource_.ubBuf.template GetBufferByByte<InputT>(128 * 1024);
        stage1GateUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(160 * 1024);
        stage1GateUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(161 * 1024);
        stage1GateLastUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(162 * 1024);
        stage1GateLastUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(163 * 1024);
        stage1GateInputUb_[0] = resource_.ubBuf.template GetBufferByByte<GateT>(162 * 1024);
        stage1GateInputUb_[1] = resource_.ubBuf.template GetBufferByByte<GateT>(163 * 1024);

        hUpdateUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(32 * 1024);
        hUpdateUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(96 * 1024);
        gateBroadcastUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(48 * 1024);
        gateBroadcastUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(112 * 1024);
        hInputUb_[0] = resource_.ubBuf.template GetBufferByByte<InputT>(80 * 1024);
        hInputUb_[1] = resource_.ubBuf.template GetBufferByByte<InputT>(144 * 1024);
        stateOutputUb_[0] = resource_.ubBuf.template GetBufferByByte<StateT>(80 * 1024);
        stateOutputUb_[1] = resource_.ubBuf.template GetBufferByByte<StateT>(144 * 1024);
        gateLastScalarUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(164 * 1024);
        gateLastScalarUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(165 * 1024);
        gateKLastUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(162 * 1024);
        gateKLastUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(163 * 1024);
        gateKInputUb_[0] = resource_.ubBuf.template GetBufferByByte<GateT>(164 * 1024);
        gateKInputUb_[1] = resource_.ubBuf.template GetBufferByByte<GateT>(165 * 1024);
        broadcastScratch_ = resource_.ubBuf.template GetBufferByByte<uint8_t>(166 * 1024);
    }

    __aicore__ inline void InitState()
    {
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t sequenceCount = isVariedLen_ ? scheduler_.tokenBatch : shapeBatch_;
        const uint32_t taskCount = sequenceCount * scheduler_.headWindowNum;
        const uint32_t hRowsPerTile = (32 * 1024) / (vHeadDim_ * sizeof(InputT));
        const uint32_t stateRowsPerTile = (64 * 1024) / (vHeadDim_ * sizeof(StateT));
        const uint32_t rowsPerTile = ::Min(hRowsPerTile, stateRowsPerTile);
        const uint32_t totalChunks = isVariedLen_ ? scheduler_.totalChunks :
            ((seqlen_ + chunkSize_ - 1) / chunkSize_);
        const uint32_t stateBlockSize = kHeadDim_ * vHeadDim_;
        auto stateUbPing = resource_.ubBuf.template GetBufferByByte<StateT>(0);
        auto stateUbPong = resource_.ubBuf.template GetBufferByByte<StateT>(96 * 1024);
        auto hUbPing = resource_.ubBuf.template GetBufferByByte<InputT>(64 * 1024);
        auto hUbPong = resource_.ubBuf.template GetBufferByByte<InputT>(160 * 1024);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t taskIdx = coreIdx; taskIdx < taskCount; taskIdx += coreNum) {
            const uint32_t batchIdx = taskIdx / scheduler_.headWindowNum;
            const uint32_t headBase = scheduler_.GetCoreHeadBase();
            for (uint32_t headOffset = 0; headOffset < scheduler_.GetHeadsPerCore(); ++headOffset) {
                const uint32_t vHeadIdx = headBase + headOffset;
                if (vHeadIdx >= vNumHead_ || headOffset % subBlockNum != subBlockIdx) {
                    continue;
                }
                uint32_t ping = ((headOffset / subBlockNum) & 1U) == 0 ? 1 : 0;
                const uint32_t chunkOffset =
                    isVariedLen_ ? scheduler_.GetVarlenChunkOffset(batchIdx) : 0;
                const uint32_t shapeBatchIdx = isVariedLen_ ? 0 : batchIdx;
                const uint32_t hBaseOffset =
                    (shapeBatchIdx * vNumHead_ * totalChunks + vHeadIdx * totalChunks + chunkOffset) *
                    stateBlockSize;
                const uint32_t initialBase =
                    (batchIdx * vNumHead_ + vHeadIdx) * stateBlockSize;
                for (uint32_t row = 0; row < kHeadDim_; row += rowsPerTile) {
                    const uint32_t rows = ::Min(rowsPerTile, kHeadDim_ - row);
                    const uint32_t elems = rows * vHeadDim_;
                    auto stateUb = ping ? stateUbPing : stateUbPong;
                    auto hUb = ping ? hUbPing : hUbPong;
                    const uint32_t eventId = ping ? EVENT_ID1 : EVENT_ID0;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    if (useInitialState_) {
                        const uint32_t initialOffset = initialBase + row * vHeadDim_;
                        if constexpr (std::is_same<StateT, InputT>::value) {
                            AscendC::DataCopy(stateUb, gmInitialState_[initialOffset], elems);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                            AscendC::DataCopy(gmH_[hBaseOffset + row * vHeadDim_], stateUb, elems);
                        } else if constexpr (std::is_same<StateT, bfloat16_t>::value &&
                                             std::is_same<InputT, half>::value) {
                            auto stateLoad = hUb.template ReinterpretCast<StateT>();
                            auto stateFp32 = stateUb.template ReinterpretCast<float>();
                            AscendC::DataCopy(stateLoad, gmInitialState_[initialOffset], elems);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                            AscendC::Cast(stateFp32, stateLoad, AscendC::RoundMode::CAST_NONE, elems);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Cast(hUb, stateFp32, AscendC::RoundMode::CAST_RINT, elems);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::DataCopy(gmH_[hBaseOffset + row * vHeadDim_], hUb, elems);
                        } else {
                            AscendC::DataCopy(stateUb, gmInitialState_[initialOffset], elems);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                            AscendC::Cast(hUb, stateUb, AscendC::RoundMode::CAST_RINT, elems);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::DataCopy(gmH_[hBaseOffset + row * vHeadDim_], hUb, elems);
                        }
                    } else {
                        AscendC::Duplicate(hUb, static_cast<InputT>(0), elems);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                        AscendC::DataCopy(gmH_[hBaseOffset + row * vHeadDim_], hUb, elems);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    ping = 1 - ping;
                }
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void InitEvents()
    {
        if (storeFinalState_ && std::is_same<StateT, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + kPongEventBase);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + kPongEventBase);
            event2FromMte3_[0] = false;
            event2FromMte3_[1] = false;
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + kPongEventBase);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + kPongEventBase);
            event2FromMte3_[0] = true;
            event2FromMte3_[1] = true;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + kPongEventBase);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + kPongEventBase);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + kPongEventBase);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + kPongEventBase);
    }

    __aicore__ inline void DrainEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t pong = slot == 0 ? 0 : kPongEventBase;
            if (event0FromMte3_[slot]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pong);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pong);
            }
            if (event2FromMte3_[slot]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pong);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pong);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pong);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pong);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pong);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pong);
        }
    }

    __aicore__ inline float LoadScalarAsFloat(
        AscendC::GlobalTensor<InputT> tensor, uint32_t offset) const
    {
        InputT value = tensor.GetValue(offset);
        if constexpr (std::is_same<InputT, bfloat16_t>::value) {
            return AscendC::ToFloat(value);
        }
        return static_cast<float>(value);
    }

    __aicore__ inline void Stage0Tail(const Offsets& offsets, uint32_t eventId)
    {
        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
        auto inputUb = resource_.ubBuf.template GetBufferByByte<InputT>(166 * 1024);
        auto floatUb = resource_.ubBuf.template GetBufferByByte<float>(167 * 1024);
        auto accumUb = resource_.ubBuf.template GetBufferByByte<float>(168 * 1024);
        for (uint32_t tokenRow = 0; tokenRow < offsets.blockTokens; ++tokenRow) {
            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t kIdx = 0; kIdx < kHeadDim_; ++kIdx) {
                AscendC::DataCopy(inputUb,
                    gmH_[offsets.hSrcOffset + kIdx * vHeadDim_], offsets.vBlockDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::Cast(floatUb, inputUb, AscendC::RoundMode::CAST_NONE,
                              offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                float weight = LoadScalarAsFloat(
                    gmW_, offsets.wOffset + tokenRow * kHeadDim_ + kIdx);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::DataCopy(
                gmVWorkspace_[offsets.vWorkOffset + tokenRow * offsets.vBlockDim],
                accumUb, offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void Stage2Tail(const Offsets& offsets, uint32_t eventId)
    {
        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
        auto inputUb = resource_.ubBuf.template GetBufferByByte<InputT>(166 * 1024);
        auto floatUb = resource_.ubBuf.template GetBufferByByte<float>(167 * 1024);
        auto accumUb = resource_.ubBuf.template GetBufferByByte<float>(168 * 1024);
        for (uint32_t kRow = 0; kRow < kHeadDim_; ++kRow) {
            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t tokenRow = 0; tokenRow < offsets.blockTokens; ++tokenRow) {
                AscendC::DataCopy(inputUb,
                    gmVUpdateWorkspace_[offsets.vWorkOffset + tokenRow * offsets.vBlockDim],
                    offsets.vBlockDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::Cast(floatUb, inputUb, AscendC::RoundMode::CAST_NONE,
                              offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                float weight = LoadScalarAsFloat(
                    gmK_, offsets.wkOffset + tokenRow * kHeadDim_ + kRow);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::DataCopy(
                gmHWorkspace_[offsets.hWorkOffset + kRow * offsets.vBlockDim],
                accumUb, offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void PrepareScalarGate(
        uint32_t gateOffset, uint32_t tokenCount, bool isPing)
    {
        const uint32_t slot = isPing ? 0 : 1;
        const uint32_t eventId = EventId(EVENT_ID3, isPing);
        auto gate = stage1GateUb_[slot];
        auto gateLast = stage1GateLastUb_[slot];
        auto gateInput = stage1GateInputUb_[slot];
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
        if (tokenCount == 1) {
            AscendC::Duplicate(gate, 1.0f, 1);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        if constexpr (std::is_same<GateT, float>::value) {
            AscendC::DataCopyParams params{1,
                static_cast<uint16_t>(tokenCount * sizeof(float)), 0, 0};
            AscendC::DataCopyPadParams pad{false, 0, 0, 0};
            AscendC::DataCopyPad(gate, gmG_[gateOffset], params, pad);
        } else {
            AscendC::DataCopyParams params{1,
                static_cast<uint16_t>(tokenCount * sizeof(GateT)), 0, 0};
            AscendC::DataCopyPadParams pad{false, 0, 0, 0};
            AscendC::DataCopyPad(gateInput, gmG_[gateOffset], params, pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        if constexpr (!std::is_same<GateT, float>::value) {
            AscendC::Cast(gate, gateInput, AscendC::RoundMode::CAST_NONE, tokenCount);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
        const float last = gate.GetValue(tokenCount - 1);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
        AscendC::Duplicate(gateLast, last, tokenCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(gate, gateLast, gate, tokenCount);
        AscendC::PipeBarrier<PIPE_V>();
        ApplyExp(gate, tokenCount);
    }

    __aicore__ inline void ApplyExp(AscendC::LocalTensor<float> tensor, uint32_t count)
    {
        if constexpr (kUseExp2) {
            AscendC::Muls(tensor, tensor, kLn2, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Exp(tensor, tensor, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool ScalarGate>
    __aicore__ inline void Stage1Impl(
        const Offsets& offsets, bool isPing, uint32_t localSlot)
    {
        const uint32_t slot = isPing ? 0 : 1;
        const uint32_t event0 = EventId(EVENT_ID0, isPing);
        const uint32_t event1 = EventId(EVENT_ID1, isPing);
        const uint32_t event3 = EventId(EVENT_ID3, isPing);
        const uint32_t n = offsets.vBlockDim;
        if constexpr (ScalarGate) {
            PrepareScalarGate(offsets.gOffset, offsets.blockTokens, isPing);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event3);
        }

        bool waitWorkspaceFromMte3 =
            storeFinalState_ && std::is_same<StateT, float>::value &&
            event0FromMte3_[localSlot];
        for (uint32_t row = 0; row < offsets.blockTokens;) {
            const uint32_t alignExtra = row & 7U;
            const uint32_t maxRows = kRowTile - alignExtra;
            const uint32_t rows = ::Min(maxRows, offsets.blockTokens - row);
            const uint32_t gateBase = row & ~7U;
            const uint32_t gateRows = alignExtra + rows;
            uint32_t dstShape[2] = {gateRows, n};
            uint32_t srcShape[2] = {gateRows, 1};

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event1);
            CopyGmToUb(stage1InputUb_[slot], gmU_[offsets.uvOffset + row * vHeadDim_],
                       rows, n, vHeadDim_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event1);
            AscendC::Cast(calcUb_, stage1InputUb_[slot], AscendC::RoundMode::CAST_NONE,
                          rows * n);
            AscendC::PipeBarrier<PIPE_V>();

            if (waitWorkspaceFromMte3) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event0);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event0);
            }
            AscendC::DataCopy(stage1WsUb_[slot],
                gmVWorkspace_[offsets.vWorkOffset + row * n], rows * n);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event0);
            waitWorkspaceFromMte3 = false;
            AscendC::Sub(stage1WsUb_[slot], calcUb_, stage1WsUb_[slot], rows * n);
            AscendC::PipeBarrier<PIPE_V>();

            uint32_t decayOffset = 0;
            if constexpr (ScalarGate) {
                AscendC::Broadcast<float, 2, 1>(
                    calcUb_, stage1GateUb_[slot][gateBase], dstShape, srcShape,
                    broadcastScratch_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(calcUb_[alignExtra * n], stage1WsUb_[slot],
                             calcUb_[alignExtra * n], rows * n);
                decayOffset = alignExtra * n;
            } else {
                AscendC::Adds(calcUb_, stage1WsUb_[slot], 0.0f, rows * n);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(stage1InputUb_[slot], calcUb_[decayOffset],
                          AscendC::RoundMode::CAST_RINT, rows * n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event1);
            AscendC::DataCopy(
                gmVUpdateWorkspace_[offsets.vWorkOffset + row * n],
                stage1InputUb_[slot], rows * n);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event1);
            AscendC::Cast(stage1InputUb_[slot], stage1WsUb_[slot],
                          AscendC::RoundMode::CAST_RINT, rows * n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event1);
            CopyUbToGm(gmVNew_[offsets.uvOffset + row * vHeadDim_],
                       stage1InputUb_[slot], rows, n, vHeadDim_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event1);
            row += rows;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event3);
        event0FromMte3_[localSlot] = false;
    }

    __aicore__ inline void Stage1GOnly(
        const Offsets& offsets, bool isPing, uint32_t localSlot)
    {
        Stage1Impl<true>(offsets, isPing, localSlot);
    }

    __aicore__ inline void Stage1GkOnly(
        const Offsets& offsets, bool isPing, uint32_t localSlot)
    {
        Stage1Impl<false>(offsets, isPing, localSlot);
    }

    __aicore__ inline void Stage1(
        const Offsets& offsets, bool isPing, uint32_t localSlot)
    {
        if constexpr (kGOnly) {
            Stage1GOnly(offsets, isPing, localSlot);
        } else {
            Stage1GkOnly(offsets, isPing, localSlot);
        }
    }

    __aicore__ inline float LoadGateLastScalar(const Offsets& offsets, bool isPing)
    {
        const uint32_t slot = isPing ? 0 : 1;
        const uint32_t event3 = EventId(EVENT_ID3, isPing);
        GateT value = gmG_[offsets.gOffset + offsets.blockTokens - 1].GetValue(0);
        float valueFp32 = 0.0f;
        if constexpr (std::is_same<GateT, bfloat16_t>::value) {
            valueFp32 = AscendC::ToFloat(value);
        } else {
            valueFp32 = static_cast<float>(value);
        }
        gateLastScalarUb_[slot].SetValue(0, valueFp32);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(event3);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(event3);
        ApplyExp(gateLastScalarUb_[slot], 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(event3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(event3);
        const float result = gateLastScalarUb_[slot].GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(event3);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(event3);
        return result;
    }

    template <bool KGate>
    __aicore__ inline void ApplyStateGate(
        AscendC::LocalTensor<float> state, const Offsets& offsets,
        uint32_t row, uint32_t rows, bool isPing, float scalarGate,
        bool& firstKGateTile)
    {
        if constexpr (!KGate) {
            AscendC::Muls(state, state, scalarGate, rows * offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            const uint32_t slot = isPing ? 0 : 1;
            const uint32_t event1 = EventId(EVENT_ID1, isPing);
            const uint32_t event2 = EventId(EVENT_ID2, isPing);
            if (firstKGateTile) {
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event2);
                firstKGateTile = false;
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event1);
            }
            auto gateInput = gmGk_[offsets.gkOffset +
                (offsets.blockTokens - 1) * kHeadDim_ + row];
            if constexpr (std::is_same<GateT, float>::value) {
                AscendC::DataCopy(gateKLastUb_[slot], gateInput, rows);
            } else {
                AscendC::DataCopy(gateKInputUb_[slot], gateInput, rows);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event2);
            if constexpr (!std::is_same<GateT, float>::value) {
                AscendC::Cast(gateKLastUb_[slot], gateKInputUb_[slot],
                              AscendC::RoundMode::CAST_NONE, rows);
                AscendC::PipeBarrier<PIPE_V>();
            }
            ApplyExp(gateKLastUb_[slot], rows);
            const uint32_t repeat = (rows + 7) / 8;
            uint32_t dstShape[2] = {repeat * 8, offsets.vBlockDim};
            uint32_t srcShape[2] = {repeat * 8, 1};
            AscendC::Broadcast<float, 2, 1>(gateBroadcastUb_[slot],
                gateKLastUb_[slot], dstShape, srcShape, broadcastScratch_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(state, state, gateBroadcastUb_[slot],
                         rows * offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event1);
        }
    }

    template <bool KGate>
    __aicore__ inline void Stage3Impl(
        const Offsets& offsets, bool isPing, uint32_t localSlot)
    {
        const uint32_t slot = isPing ? 0 : 1;
        const uint32_t event0 = EventId(EVENT_ID0, isPing);
        const uint32_t event2 = EventId(EVENT_ID2, isPing);
        const uint32_t event3 = EventId(EVENT_ID3, isPing);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event2);
        if constexpr (KGate) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event2);
        }

        const bool useStateRecurrence = storeFinalState_ && kSeparateRollingState &&
            (!offsets.isInitialState || useInitialState_);
        const bool useFp32Recurrence = useStateRecurrence &&
            std::is_same<StateT, float>::value;
        const float scalarGate = KGate ? 1.0f : LoadGateLastScalar(offsets, isPing);
        bool firstKGateTile = true;
        bool waitHFromV = storeFinalState_ && offsets.isInitialState &&
            std::is_same<StateT, float>::value;
        bool waitUpdateFromMte3 = false;
        const uint32_t updateReadyEvent = event3;

        for (uint32_t row = 0; row < kHeadDim_; row += kRowTile) {
            const uint32_t rows = ::Min(kRowTile, kHeadDim_ - row);
            const uint32_t elems = rows * offsets.vBlockDim;
            auto hInput = gmH_[offsets.hSrcOffset + row * vHeadDim_];
            auto hOutput = gmH_[offsets.hDstOffset + row * vHeadDim_];
            auto hUpdate = gmHWorkspace_[offsets.hWorkOffset + row * offsets.vBlockDim];
            auto finalState = gmFinalState_[offsets.finalStateOffset + row * vHeadDim_];

            if (waitHFromV) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event2);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
            }
            if constexpr (std::is_same<StateT, float>::value) {
                if (useFp32Recurrence) {
                    if (offsets.isInitialState) {
                        CopyGmToUb(calcUb_,
                            gmInitialState_[offsets.initialStateOffset + row * vHeadDim_],
                            rows, offsets.vBlockDim, vHeadDim_);
                    } else {
                        CopyGmToUb(calcUb_, finalState, rows,
                                   offsets.vBlockDim, vHeadDim_);
                    }
                } else {
                    CopyGmToUb(hInputUb_[slot], hInput, rows,
                               offsets.vBlockDim, vHeadDim_);
                }
            } else if constexpr (kSeparateRollingState) {
                if (useStateRecurrence) {
                    if (offsets.isInitialState) {
                        CopyGmToUb(stateOutputUb_[slot],
                            gmInitialState_[offsets.initialStateOffset + row * vHeadDim_],
                            rows, offsets.vBlockDim, vHeadDim_);
                    } else {
                        CopyGmToUb(stateOutputUb_[slot], finalState, rows,
                                   offsets.vBlockDim, vHeadDim_);
                    }
                } else {
                    CopyGmToUb(hInputUb_[slot], hInput, rows,
                               offsets.vBlockDim, vHeadDim_);
                }
            } else {
                CopyGmToUb(hInputUb_[slot], hInput, rows,
                           offsets.vBlockDim, vHeadDim_);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event2);
            if constexpr (kSeparateRollingState) {
                if (!useFp32Recurrence) {
                    if (useStateRecurrence) {
                        AscendC::Cast(calcUb_, stateOutputUb_[slot],
                                      AscendC::RoundMode::CAST_NONE, elems);
                    } else {
                        AscendC::Cast(calcUb_, hInputUb_[slot],
                                      AscendC::RoundMode::CAST_NONE, elems);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
            } else {
                AscendC::Cast(calcUb_, hInputUb_[slot],
                              AscendC::RoundMode::CAST_NONE, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
            ApplyStateGate<KGate>(calcUb_, offsets, row, rows, isPing,
                                  scalarGate, firstKGateTile);

            if (waitUpdateFromMte3) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event0);
            }
            AscendC::DataCopy(hUpdateUb_[slot], hUpdate, elems);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event0);
            AscendC::Add(hUpdateUb_[slot], calcUb_, hUpdateUb_[slot], elems);
            AscendC::PipeBarrier<PIPE_V>();
            waitHFromV = false;

            if constexpr (std::is_same<StateT, float>::value) {
                if (storeFinalState_) {
                    if (!offsets.isFinalState) {
                        AscendC::Cast(hInputUb_[slot], hUpdateUb_[slot],
                                      AscendC::RoundMode::CAST_RINT, elems);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event2);
                        waitHFromV = true;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event0);
                    CopyUbToGm(finalState, hUpdateUb_[slot], rows,
                               offsets.vBlockDim, vHeadDim_);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(updateReadyEvent);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(updateReadyEvent);
                    waitUpdateFromMte3 = true;
                    if (!offsets.isFinalState) {
                        CopyUbToGm(hOutput, hInputUb_[slot], rows,
                                   offsets.vBlockDim, vHeadDim_);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
                    }
                } else {
                    AscendC::Cast(hInputUb_[slot], hUpdateUb_[slot],
                                  AscendC::RoundMode::CAST_RINT, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event0);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event2);
                    CopyUbToGm(hOutput, hInputUb_[slot], rows,
                               offsets.vBlockDim, vHeadDim_);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
                }
            } else if constexpr (kSeparateRollingState) {
                AscendC::Cast(stateOutputUb_[slot], hUpdateUb_[slot],
                              AscendC::RoundMode::CAST_RINT, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event2);
                CopyUbToGm(finalState, stateOutputUb_[slot], rows,
                           offsets.vBlockDim, vHeadDim_);
                if (!offsets.isFinalState) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event2);
                    AscendC::Cast(calcUb_, stateOutputUb_[slot],
                                  AscendC::RoundMode::CAST_NONE, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(hInputUb_[slot], calcUb_,
                                  AscendC::RoundMode::CAST_RINT, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event2);
                    CopyUbToGm(hOutput, hInputUb_[slot], rows,
                               offsets.vBlockDim, vHeadDim_);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
                waitUpdateFromMte3 = false;
            } else {
                if (storeFinalState_ && offsets.isFinalState) {
                    AscendC::Cast(stateOutputUb_[slot], hUpdateUb_[slot],
                                  AscendC::RoundMode::CAST_RINT, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event0);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event2);
                    CopyUbToGm(finalState, stateOutputUb_[slot], rows,
                               offsets.vBlockDim, vHeadDim_);
                } else {
                    AscendC::Cast(hInputUb_[slot], hUpdateUb_[slot],
                                  AscendC::RoundMode::CAST_RINT, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event0);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event2);
                    CopyUbToGm(hOutput, hInputUb_[slot], rows,
                               offsets.vBlockDim, vHeadDim_);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
                waitUpdateFromMte3 = false;
            }
        }

        if (storeFinalState_ && std::is_same<StateT, float>::value) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event0);
            event0FromMte3_[localSlot] = true;
            if (!offsets.isFinalState) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
                event2FromMte3_[localSlot] = true;
            } else {
                event2FromMte3_[localSlot] = false;
            }
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
            event0FromMte3_[localSlot] = false;
            event2FromMte3_[localSlot] = true;
        }
        if constexpr (KGate) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EventId(EVENT_ID1, isPing));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event2);
    }

    __aicore__ inline void Stage3GOnly(
        const Offsets& offsets, bool isPing, uint32_t localSlot)
    {
        Stage3Impl<false>(offsets, isPing, localSlot);
    }

    __aicore__ inline void Stage3GkOnly(
        const Offsets& offsets, bool isPing, uint32_t localSlot)
    {
        Stage3Impl<true>(offsets, isPing, localSlot);
    }

    __aicore__ inline void Stage3(
        const Offsets& offsets, bool isPing, uint32_t localSlot)
    {
        if constexpr (kGOnly) {
            Stage3GOnly(offsets, isPing, localSlot);
        } else {
            Stage3GkOnly(offsets, isPing, localSlot);
        }
    }

    uint32_t batch_ = 0;
    uint32_t seqlen_ = 0;
    uint32_t kNumHead_ = 0;
    uint32_t vNumHead_ = 0;
    uint32_t kHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t chunkSize_ = 0;
    bool useInitialState_ = false;
    bool storeFinalState_ = false;
    uint32_t isVariedLen_ = 0;
    uint32_t shapeBatch_ = 0;
    uint32_t tokenBatch_ = 0;
    uint64_t numChunksWorkspaceOffset_ = 0;

    AscendC::GlobalTensor<InputT> gmK_;
    AscendC::GlobalTensor<InputT> gmW_;
    AscendC::GlobalTensor<InputT> gmU_;
    AscendC::GlobalTensor<GateT> gmG_;
    AscendC::GlobalTensor<GateT> gmGk_;
    AscendC::GlobalTensor<StateT> gmInitialState_;
    AscendC::GlobalTensor<InputT> gmH_;
    AscendC::GlobalTensor<InputT> gmVNew_;
    AscendC::GlobalTensor<StateT> gmFinalState_;
    AscendC::GlobalTensor<WorkspaceT> gmVWorkspace_;
    AscendC::GlobalTensor<InputT> gmVUpdateWorkspace_;
    AscendC::GlobalTensor<WorkspaceT> gmHWorkspace_;

    VecScheduler scheduler_;
    Catlass::Arch::Resource<ArchTag> resource_;

    AscendC::LocalTensor<float> calcUb_;
    AscendC::LocalTensor<float> stage1WsUb_[2];
    AscendC::LocalTensor<InputT> stage1InputUb_[2];
    AscendC::LocalTensor<float> stage1GateUb_[2];
    AscendC::LocalTensor<float> stage1GateLastUb_[2];
    AscendC::LocalTensor<GateT> stage1GateInputUb_[2];
    AscendC::LocalTensor<float> hUpdateUb_[2];
    AscendC::LocalTensor<float> gateBroadcastUb_[2];
    AscendC::LocalTensor<InputT> hInputUb_[2];
    AscendC::LocalTensor<StateT> stateOutputUb_[2];
    AscendC::LocalTensor<float> gateLastScalarUb_[2];
    AscendC::LocalTensor<float> gateKLastUb_[2];
    AscendC::LocalTensor<GateT> gateKInputUb_[2];
    AscendC::LocalTensor<uint8_t> broadcastScratch_;
    bool event0FromMte3_[2] = {false, false};
    bool event2FromMte3_[2] = {false, false};
};

} // namespace GDN::FwdHStandalone

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_VECTOR_H
