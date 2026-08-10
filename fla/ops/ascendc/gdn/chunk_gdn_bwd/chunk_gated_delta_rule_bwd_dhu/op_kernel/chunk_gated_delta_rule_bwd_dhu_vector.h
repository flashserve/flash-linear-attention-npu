/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_vector.h
 * \brief A2/A3 vector path for chunk_gated_delta_rule_bwd_dhu.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_VECTOR_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_VECTOR_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "adv_api/utils/init_global_memory.h"
#include "chunk_gated_delta_rule_bwd_dhu_common.h"
#include "chunk_gated_delta_rule_bwd_dhu_struct.h"

namespace GDN {

template <typename DT, typename GT, int USE_GK>
class ChunkGatedDeltaRuleBwdDhuVector {
public:
    __aicore__ inline ChunkGatedDeltaRuleBwdDhuVector() = default;

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR gate, GM_ADDR dv, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dh,
                                GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace,
                                const ChunkGatedDeltaRuleBwdDhuTilingData *__restrict tilingData,
                                AscendC::TPipe *pipe)
    {
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(q));
        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(gate));
        dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dv));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh));
        if (tilingData->hasDh0 != 0 && dh0 != nullptr) {
            dh0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh0));
            dh0Addr_ = dh0;
            hasDh0_ = true;
        }
        dv2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dv2));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace));
        workspaceStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        tiling_ = tilingData;
        pipe_ = pipe;
        B_ = tiling_->B;
        HK_ = tiling_->HK;
        HV_ = tiling_->HV;
        T_ = tiling_->T;
        K_ = tiling_->K;
        V_ = tiling_->V;
        HRatio_ = tiling_->HRatio;
        chunkSize_ = tiling_->chunkSize;
        totalChunkNum_ = tiling_->totalChunkNum;
        headWindowNum_ = tiling_->headWindowNum;
        taskNum_ = tiling_->taskNum;
        isVariable_ = tiling_->isVariable;
        scale_ = tiling_->scale;
        qgWorkspaceOffset_ = tiling_->qgWorkspaceOffset;
        stateWorkspaceOffset_ = tiling_->stateWorkspaceOffset;
        dvStateWorkspaceOffset_ = tiling_->dvStateWorkspaceOffset;
        termQWorkspaceOffset_ = tiling_->termQWorkspaceOffset;
        termWWorkspaceOffset_ = tiling_->termWWorkspaceOffset;
        workspaceElemsPerSubBlock_ = tiling_->workspaceElemsPerSubBlock;
        dh0ClearCoreNum_ = tiling_->dh0ClearCoreNum;
        dh0ClearElemsPerCore_ = tiling_->dh0ClearElemsPerCore;
        dh0ClearTailElems_ = tiling_->dh0ClearTailElems;
        vecRow_ = tiling_->vecRow > 0 ? tiling_->vecRow : 8;
        gateElems_ = K_ > chunkSize_ ? K_ : chunkSize_;
        subBlockNum_ = static_cast<int64_t>(AscendC::GetSubBlockNum());
        if (subBlockNum_ <= 0) {
            subBlockNum_ = 1;
        }
        subBlockIdx_ = static_cast<int64_t>(AscendC::GetSubBlockIdx());
        if (subBlockIdx_ < 0 || subBlockIdx_ >= subBlockNum_) {
            subBlockIdx_ = 0;
        }

        const int64_t inputElems = vecRow_ * (K_ > V_ ? K_ : V_);
        const int64_t brcbElems = vecRow_ * BRCB_ROW_FLOAT_ELEMS;
        pipe_->InitBuffer(qInputPing_, inputElems * static_cast<int64_t>(sizeof(DT)));
        pipe_->InitBuffer(qInputPong_, inputElems * static_cast<int64_t>(sizeof(DT)));
        pipe_->InitBuffer(gInputPing_, gateElems_ * static_cast<int64_t>(sizeof(GT)));
        pipe_->InitBuffer(gInputPong_, gateElems_ * static_cast<int64_t>(sizeof(GT)));
        pipe_->InitBuffer(outputPing_, inputElems * static_cast<int64_t>(sizeof(DT)));
        pipe_->InitBuffer(outputPong_, inputElems * static_cast<int64_t>(sizeof(DT)));
        pipe_->InitBuffer(statePing_, vecRow_ * V_ * static_cast<int64_t>(sizeof(float)));
        pipe_->InitBuffer(statePong_, vecRow_ * V_ * static_cast<int64_t>(sizeof(float)));
        pipe_->InitBuffer(qFp32Buf_, inputElems * static_cast<int64_t>(sizeof(float)));
        pipe_->InitBuffer(gateFactorAllFp32_, HEADS_PER_TASK * gateElems_ * static_cast<int64_t>(sizeof(float)));
        if constexpr (USE_GK == 0) {
            pipe_->InitBuffer(gRawAllFp32_, HEADS_PER_TASK * gateElems_ * static_cast<int64_t>(sizeof(float)));
            pipe_->InitBuffer(dvGateFactorAllFp32_,
                              HEADS_PER_TASK * gateElems_ * static_cast<int64_t>(sizeof(float)));
        }
        pipe_->InitBuffer(gBrcbBuf_, brcbElems * static_cast<int64_t>(sizeof(float)));
        pipe_->InitBuffer(outFp32Buf_, inputElems * static_cast<int64_t>(sizeof(float)));

        qInputBuf_[0] = qInputPing_.template Get<DT>();
        qInputBuf_[1] = qInputPong_.template Get<DT>();
        gateInputBuf_[0] = gInputPing_.template Get<GT>();
        gateInputBuf_[1] = gInputPong_.template Get<GT>();
        outputBuf_[0] = outputPing_.template Get<DT>();
        outputBuf_[1] = outputPong_.template Get<DT>();
        stateBuf_[0] = statePing_.template Get<float>();
        stateBuf_[1] = statePong_.template Get<float>();

        InitVectorEvents();
    }

    __aicore__ inline void Process()
    {
        if (hasDh0_) {
            const int64_t vecBlockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
            if (vecBlockIdx >= 0 && vecBlockIdx < dh0ClearCoreNum_) {
                int64_t clearOffset = vecBlockIdx * dh0ClearElemsPerCore_;
                int64_t clearElems = dh0ClearElemsPerCore_;
                if (vecBlockIdx + 1 == dh0ClearCoreNum_) {
                    clearOffset = (dh0ClearCoreNum_ - 1) * dh0ClearElemsPerCore_;
                    clearElems = dh0ClearTailElems_;
                }
                if (clearElems > 0) {
                    if constexpr (sizeof(DT) == sizeof(uint16_t)) {
                        AscendC::GlobalTensor<uint16_t> dh0ClearGm;
                        dh0ClearGm.SetGlobalBuffer(
                            reinterpret_cast<__gm__ uint16_t *>(dh0Addr_) + clearOffset);
                        AscendC::Fill(dh0ClearGm, static_cast<uint64_t>(clearElems),
                                      static_cast<uint16_t>(0));
                    } else {
                        AscendC::GlobalTensor<uint32_t> dh0ClearGm;
                        dh0ClearGm.SetGlobalBuffer(
                            reinterpret_cast<__gm__ uint32_t *>(dh0Addr_) + clearOffset);
                        AscendC::Fill(dh0ClearGm, static_cast<uint64_t>(clearElems),
                                      static_cast<uint32_t>(0));
                    }
                }
            }
            AscendC::SyncAll<true>();
        }

        const int64_t coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx() / subBlockNum_);
        const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());

        for (int64_t taskIdx = coreIdx; taskIdx < taskNum_; taskIdx += blockNum) {
            const int64_t seqIdx = taskIdx / headWindowNum_;
            const int64_t headWindowIdx = taskIdx - seqIdx * headWindowNum_;
            const int64_t hvBase = headWindowIdx * HEADS_PER_TASK;
            const int64_t headCnt = Min(HEADS_PER_TASK, HV_ - hvBase);
            const int64_t taskRound = (taskIdx - coreIdx) / blockNum;
            const int64_t windowStartSlot = (taskRound & 1) * HEADS_PER_TASK;
            if (headCnt <= 0) {
                continue;
            }

            SeqInfo seqInfo;
            GetSeqInfo(cuSeqlens_, *tiling_, seqIdx, seqInfo);
            if (!seqInfo.valid) {
                continue;
            }

            for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                if (headOffset % subBlockNum_ != subBlockIdx_) {
                    continue;
                }
                const int64_t workspaceBase = WorkspaceBase(coreIdx, windowStartSlot + headOffset);
                for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
                    const int64_t curRows = Min(vecRow_, K_ - rowOffset);
                    const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                    const uint32_t stateIdx = curStatePingPong_;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[stateIdx]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[stateIdx]);
                    AscendC::LocalTensor<float> stateFp32 = stateBuf_[stateIdx];
                    AscendC::Duplicate(stateFp32, 0.0f, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    CopyOutStateRows(stateIdx, stateFp32, StateWorkspaceFloatOffset(workspaceBase, rowOffset), elems);
                    curStatePingPong_ ^= 1U;
                }
            }

            for (int64_t chunkIdx = seqInfo.chunkCnt - 1; chunkIdx >= 0; --chunkIdx) {
                ChunkInfo chunkInfo;
                GetChunkInfoBySeqChunk(chunkIndices_, *tiling_, seqInfo, chunkIdx, chunkInfo);
                if (!chunkInfo.valid) {
                    continue;
                }

                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    const int64_t hv = hvBase + headOffset;
                    const int64_t hq = hv / HRatio_;
                    const int64_t workspaceBase = WorkspaceBase(coreIdx, workspaceSlot);
                    const int64_t dhBase = DhOffset(chunkInfo.bIdx, hv, chunkInfo.outputChunkIdx);
                    if (headOffset % subBlockNum_ != subBlockIdx_) {
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                        continue;
                    }
                    AscendC::LocalTensor<float> gateFactor =
                        gateFactorAllFp32_.template Get<float>()[headOffset * gateElems_];
                    AscendC::LocalTensor<float> gBrcb = gBrcbBuf_.template Get<float>();
                    if constexpr (USE_GK == 0) {
                        AscendC::LocalTensor<float> gateRaw =
                            gRawAllFp32_.template Get<float>()[headOffset * gateElems_];
                        const uint32_t gateIdx = CopyInGateRows(
                            gateGm_, gateInputBuf_[curGateInputPingPong_],
                            GOffset(chunkInfo.bIdx, hv, chunkInfo.tokenStart),
                            static_cast<uint32_t>(chunkInfo.chunkLen));
                        CastGateInputRows(gateRaw, gateInputBuf_[gateIdx],
                                          static_cast<uint32_t>(chunkInfo.chunkLen), gateIdx);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Exp(gateFactor, gateRaw, static_cast<uint32_t>(chunkInfo.chunkLen));
                        AscendC::PipeBarrier<PIPE_V>();
                        const int64_t lastRow = chunkInfo.chunkLen - 1;
                        const int64_t lastRowBase = (lastRow / BRCB_GROUP_ROWS) * BRCB_GROUP_ROWS;
                        AscendC::Brcb(gBrcb, gateFactor[lastRowBase], 1, {1, 8});
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        const int64_t lastToken = chunkInfo.tokenStart + chunkInfo.chunkLen - 1;
                        const uint32_t gateIdx = CopyInGateRows(
                            gateGm_, gateInputBuf_[curGateInputPingPong_],
                            ((chunkInfo.bIdx * HV_ + hv) * T_ + lastToken) * K_,
                            static_cast<uint32_t>(K_));
                        CastGateInputRows(gateFactor, gateInputBuf_[gateIdx], static_cast<uint32_t>(K_), gateIdx);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(gateFactor, gateFactor, LN2, static_cast<uint32_t>(K_));
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Exp(gateFactor, gateFactor, static_cast<uint32_t>(K_));
                        AscendC::PipeBarrier<PIPE_V>();
                    }

                    for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
                        const int64_t curRows = Min(vecRow_, K_ - rowOffset);
                        const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                        const uint32_t stateIdx = CopyInStateRows(
                            stateBuf_[curStatePingPong_], StateWorkspaceFloatOffset(workspaceBase, rowOffset), elems);
                        AscendC::LocalTensor<float> stateFp32 = stateBuf_[stateIdx];
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[stateIdx]);
                        CopyOutFp32Rows(dhGm_, stateFp32, dhBase + rowOffset * V_, elems);
                        if constexpr (USE_GK == 0) {
                            const int64_t lastRow = chunkInfo.chunkLen - 1;
                            const int64_t lastLane = lastRow - (lastRow / BRCB_GROUP_ROWS) * BRCB_GROUP_ROWS;
                            const uint8_t repeatStride = static_cast<uint8_t>(V_ * sizeof(float) / 32);
                            for (int64_t col = 0; col < V_; col += VECTOR_REPEAT_FLOAT_ELEMS) {
                                const uint64_t cur = static_cast<uint64_t>(
                                    V_ - col > VECTOR_REPEAT_FLOAT_ELEMS ? VECTOR_REPEAT_FLOAT_ELEMS : V_ - col);
                                AscendC::Mul(stateFp32[col], stateFp32[col],
                                             gBrcb[lastLane * BRCB_ROW_FLOAT_ELEMS], cur,
                                             static_cast<uint8_t>(curRows),
                                             {1, 1, 0, repeatStride, repeatStride, 0});
                            }
                        } else {
                            AscendC::Brcb(gBrcb, gateFactor[rowOffset],
                                          static_cast<uint8_t>(CeilDiv(curRows, BRCB_GROUP_ROWS)), {1, 8});
                            AscendC::PipeBarrier<PIPE_V>();
                            const uint8_t repeatStride = static_cast<uint8_t>(V_ * sizeof(float) / 32);
                            for (int64_t col = 0; col < V_; col += VECTOR_REPEAT_FLOAT_ELEMS) {
                                const uint64_t cur = static_cast<uint64_t>(
                                    V_ - col > VECTOR_REPEAT_FLOAT_ELEMS ? VECTOR_REPEAT_FLOAT_ELEMS : V_ - col);
                                AscendC::Mul(stateFp32[col], stateFp32[col], gBrcb, cur,
                                             static_cast<uint8_t>(curRows),
                                             {1, 1, 0, repeatStride, repeatStride, 1});
                            }
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        CopyOutStateRows(stateIdx, stateFp32, StateWorkspaceFloatOffset(workspaceBase, rowOffset),
                                         elems);
                    }

                    for (int64_t rowOffset = 0; rowOffset < chunkInfo.chunkLen; rowOffset += vecRow_) {
                        const int64_t curRows = Min(vecRow_, chunkInfo.chunkLen - rowOffset);
                        const int64_t token = chunkInfo.tokenStart + rowOffset;
                        const uint32_t qIdx = CopyInRows(
                            qGm_, qInputBuf_[curQInputPingPong_],
                            ((chunkInfo.bIdx * HK_ + hq) * T_ + token) * K_,
                            static_cast<uint32_t>(curRows * K_));
                        AscendC::LocalTensor<float> qFp32 = qFp32Buf_.template Get<float>();
                        CastInputRows(qFp32, qInputBuf_[qIdx], static_cast<uint32_t>(curRows * K_), qIdx);
                        AscendC::PipeBarrier<PIPE_V>();
                        if constexpr (USE_GK == 0) {
                            AscendC::Brcb(gBrcb, gateFactor[rowOffset],
                                          static_cast<uint8_t>(CeilDiv(curRows, BRCB_GROUP_ROWS)), {1, 8});
                            AscendC::PipeBarrier<PIPE_V>();
                            const uint8_t repeatStride = static_cast<uint8_t>(K_ * sizeof(float) / 32);
                            for (int64_t col = 0; col < K_; col += VECTOR_REPEAT_FLOAT_ELEMS) {
                                const uint64_t cur = static_cast<uint64_t>(
                                    K_ - col > VECTOR_REPEAT_FLOAT_ELEMS ? VECTOR_REPEAT_FLOAT_ELEMS : K_ - col);
                                AscendC::Mul(qFp32[col], qFp32[col], gBrcb, cur,
                                             static_cast<uint8_t>(curRows),
                                             {1, 1, 0, repeatStride, repeatStride, 1});
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        CopyOutFp32Rows(workspaceGm_, qFp32, workspaceBase + qgWorkspaceOffset_ + rowOffset * K_,
                                        static_cast<uint32_t>(curRows * K_));
                    }
                    if constexpr (USE_GK == 0) {
                        const int64_t lastRow = chunkInfo.chunkLen - 1;
                        const int64_t lastRowBase = (lastRow / BRCB_GROUP_ROWS) * BRCB_GROUP_ROWS;
                        const int64_t lastLane = lastRow - lastRowBase;
                        AscendC::LocalTensor<float> gateRaw =
                            gRawAllFp32_.template Get<float>()[headOffset * gateElems_];
                        AscendC::LocalTensor<float> dvGateFactor =
                            dvGateFactorAllFp32_.template Get<float>()[headOffset * gateElems_];
                        AscendC::Brcb(gBrcb, gateRaw[lastRowBase], 1, {1, 8});
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(dvGateFactor, gateRaw, -1.0f, static_cast<uint32_t>(chunkInfo.chunkLen));
                        AscendC::PipeBarrier<PIPE_V>();
                        for (int64_t offset = 0; offset < chunkInfo.chunkLen; offset += VECTOR_REPEAT_FLOAT_ELEMS) {
                            const uint64_t cur = static_cast<uint64_t>(
                                chunkInfo.chunkLen - offset > VECTOR_REPEAT_FLOAT_ELEMS ?
                                    VECTOR_REPEAT_FLOAT_ELEMS :
                                    chunkInfo.chunkLen - offset);
                            AscendC::Add(dvGateFactor[offset], dvGateFactor[offset],
                                         gBrcb[lastLane * BRCB_ROW_FLOAT_ELEMS], cur, 1, {1, 1, 0, 8, 8, 1});
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Exp(dvGateFactor, dvGateFactor, static_cast<uint32_t>(chunkInfo.chunkLen));
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                }

                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                    if (headOffset % subBlockNum_ != subBlockIdx_) {
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                        continue;
                    }
                    const int64_t hv = hvBase + headOffset;
                    const int64_t workspaceBase = WorkspaceBase(coreIdx, workspaceSlot);
                    AscendC::LocalTensor<float> gBrcb = gBrcbBuf_.template Get<float>();
                    for (int64_t rowOffset = 0; rowOffset < chunkInfo.chunkLen; rowOffset += vecRow_) {
                        const int64_t curRows = Min(vecRow_, chunkInfo.chunkLen - rowOffset);
                        const int64_t token = chunkInfo.tokenStart + rowOffset;
                        AscendC::LocalTensor<float> outFp32 = outFp32Buf_.template Get<float>();
                        const uint32_t dvStateIdx = CopyInRows(
                            workspaceGm_, qInputBuf_[curQInputPingPong_],
                            workspaceBase + dvStateWorkspaceOffset_ + rowOffset * V_,
                            static_cast<uint32_t>(curRows * V_));
                        CastInputRows(outFp32, qInputBuf_[dvStateIdx], static_cast<uint32_t>(curRows * V_),
                                      dvStateIdx);
                        AscendC::PipeBarrier<PIPE_V>();
                        if constexpr (USE_GK == 0) {
                            AscendC::LocalTensor<float> dvGateFactor =
                                dvGateFactorAllFp32_.template Get<float>()[headOffset * gateElems_];
                            AscendC::Brcb(gBrcb, dvGateFactor[rowOffset],
                                          static_cast<uint8_t>(CeilDiv(curRows, BRCB_GROUP_ROWS)), {1, 8});
                            AscendC::PipeBarrier<PIPE_V>();
                            const uint8_t repeatStride = static_cast<uint8_t>(V_ * sizeof(float) / 32);
                            for (int64_t col = 0; col < V_; col += VECTOR_REPEAT_FLOAT_ELEMS) {
                                const uint64_t cur = static_cast<uint64_t>(
                                    V_ - col > VECTOR_REPEAT_FLOAT_ELEMS ? VECTOR_REPEAT_FLOAT_ELEMS : V_ - col);
                                AscendC::Mul(outFp32[col], outFp32[col], gBrcb, cur,
                                             static_cast<uint8_t>(curRows),
                                             {1, 1, 0, repeatStride, repeatStride, 1});
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        const uint32_t dvIdx = CopyInRows(dvGm_, qInputBuf_[curQInputPingPong_],
                                                          Dv2Offset(chunkInfo.bIdx, hv, token),
                                                          static_cast<uint32_t>(curRows * V_));
                        AscendC::LocalTensor<float> dvFp32 = qFp32Buf_.template Get<float>();
                        CastInputRows(dvFp32, qInputBuf_[dvIdx], static_cast<uint32_t>(curRows * V_), dvIdx);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(outFp32, outFp32, dvFp32, static_cast<uint32_t>(curRows * V_));
                        AscendC::PipeBarrier<PIPE_V>();
                        CopyOutFp32Rows(dv2Gm_, outFp32, Dv2Offset(chunkInfo.bIdx, hv, token),
                                        static_cast<uint32_t>(curRows * V_));
                    }
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                }

                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                    if (headOffset % subBlockNum_ != subBlockIdx_) {
                        continue;
                    }
                    const int64_t workspaceBase = WorkspaceBase(coreIdx, workspaceSlot);
                    AscendC::LocalTensor<float> termQFp32 = qFp32Buf_.template Get<float>();
                    AscendC::LocalTensor<float> outFp32 = outFp32Buf_.template Get<float>();

                    for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
                        const int64_t curRows = Min(vecRow_, K_ - rowOffset);
                        const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                        const uint32_t termQIdx = CopyInRows(
                            workspaceGm_, qInputBuf_[curQInputPingPong_],
                            workspaceBase + termQWorkspaceOffset_ + rowOffset * V_,
                            elems);
                        const uint32_t termWIdx = CopyInRows(
                            workspaceGm_, qInputBuf_[curQInputPingPong_],
                            workspaceBase + termWWorkspaceOffset_ + rowOffset * V_,
                            elems);
                        CastInputRows(termQFp32, qInputBuf_[termQIdx], elems, termQIdx);
                        CastInputRows(outFp32, qInputBuf_[termWIdx], elems, termWIdx);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(termQFp32, termQFp32, scale_, elems);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Sub(termQFp32, termQFp32, outFp32, elems);
                        AscendC::PipeBarrier<PIPE_V>();
                        const uint32_t stateIdx = CopyInStateRows(
                            stateBuf_[curStatePingPong_], StateWorkspaceFloatOffset(workspaceBase, rowOffset), elems);
                        AscendC::LocalTensor<float> stateFp32 = stateBuf_[stateIdx];
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[stateIdx]);
                        AscendC::Add(stateFp32, stateFp32, termQFp32, elems);
                        AscendC::PipeBarrier<PIPE_V>();
                        CopyOutStateRows(stateIdx, stateFp32, StateWorkspaceFloatOffset(workspaceBase, rowOffset),
                                         elems);
                    }
                }
            }

            if (hasDh0_) {
                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    if (headOffset % subBlockNum_ != subBlockIdx_) {
                        continue;
                    }
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    const int64_t workspaceBase = WorkspaceBase(coreIdx, workspaceSlot);
                    const int64_t hv = hvBase + headOffset;
                    const int64_t b = isVariable_ != 0 ? 0 : seqIdx;
                    int64_t outputChunkIdx = 0;
                    if (isVariable_ != 0) {
                        outputChunkIdx = seqInfo.outputChunkBase;
                        if (outputChunkIdx >= totalChunkNum_ ||
                            !ChunkIndexMatches(chunkIndices_, outputChunkIdx, seqIdx, 0)) {
                            outputChunkIdx = FindVarlenChunkOutputIdx(chunkIndices_, *tiling_, seqIdx, 0);
                        }
                        if (outputChunkIdx < 0) {
                            continue;
                        }
                    }

                    const int64_t dh0Base = DhOffset(b, hv, outputChunkIdx);
                    for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
                        const int64_t curRows = Min(vecRow_, K_ - rowOffset);
                        const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                        const uint32_t stateIdx = CopyInStateRows(
                            stateBuf_[curStatePingPong_], StateWorkspaceFloatOffset(workspaceBase, rowOffset), elems);
                        AscendC::LocalTensor<float> stateFp32 = stateBuf_[stateIdx];
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[stateIdx]);
                        CopyOutFp32Rows(dh0Gm_, stateFp32, dh0Base + rowOffset * V_, elems);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[stateIdx]);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[stateIdx]);
                    }
                }
            }

            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        }

        ReleaseVectorEvents();
    }

private:
    static constexpr uint32_t BUFFER_COUNT = 2;
    static constexpr int64_t VECTOR_REPEAT_FLOAT_ELEMS = 64;
    static constexpr int64_t BRCB_GROUP_ROWS = 8;
    static constexpr int64_t BRCB_ROW_FLOAT_ELEMS = 8;
    static constexpr float LN2 = 0.69314718055994530942f;

    __aicore__ inline void InitVectorEvents()
    {
        for (uint32_t eventIdx = 0; eventIdx < BUFFER_COUNT; ++eventIdx) {
            qMte2ToVEvent_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            qVToMte2Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
            gateMte2ToVEvent_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            gateVToMte2Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
            vToMte3Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToVEvent_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
            stateMte2ToVEvent_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            stateVToMte2Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
            stateVToMte3Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            stateMte3ToMte2Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[eventIdx]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[eventIdx]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[eventIdx]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[eventIdx]);
        }
    }

    __aicore__ inline void ReleaseVectorEvents()
    {
        for (uint32_t eventIdx = 0; eventIdx < BUFFER_COUNT; ++eventIdx) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[eventIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[eventIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[eventIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(qMte2ToVEvent_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(gateMte2ToVEvent_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(stateVToMte3Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[eventIdx]);
        }
    }

    template <typename CopyType>
    __aicore__ inline uint32_t CopyInRows(AscendC::GlobalTensor<CopyType> &inputTensor,
                                          AscendC::LocalTensor<CopyType> dstTensor, int64_t inputOffset,
                                          uint32_t elements)
    {
        const uint32_t inputIdx = curQInputPingPong_;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[inputIdx]);
        AscendC::DataCopy(dstTensor, inputTensor[inputOffset], elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(qMte2ToVEvent_[inputIdx]);
        curQInputPingPong_ ^= 1U;
        return inputIdx;
    }

    __aicore__ inline void CastInputRows(AscendC::LocalTensor<float> dstTensor, AscendC::LocalTensor<DT> srcTensor,
                                         uint32_t elements, uint32_t inputIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(qMte2ToVEvent_[inputIdx]);
        AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_NONE, elements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[inputIdx]);
    }

    template <typename CopyType>
    __aicore__ inline uint32_t CopyInGateRows(AscendC::GlobalTensor<CopyType> &inputTensor,
                                              AscendC::LocalTensor<CopyType> dstTensor, int64_t inputOffset,
                                              uint32_t elements)
    {
        const uint32_t inputIdx = curGateInputPingPong_;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[inputIdx]);
        AscendC::DataCopyPad(dstTensor, inputTensor[inputOffset],
                             {1, elements * static_cast<uint32_t>(sizeof(CopyType)), 0, 0, 0},
                             {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(gateMte2ToVEvent_[inputIdx]);
        curGateInputPingPong_ ^= 1U;
        return inputIdx;
    }

    template <typename CopyType>
    __aicore__ inline void CastGateInputRows(AscendC::LocalTensor<float> dstTensor,
                                             AscendC::LocalTensor<CopyType> srcTensor, uint32_t elements,
                                             uint32_t inputIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(gateMte2ToVEvent_[inputIdx]);
        if constexpr (std::is_same<CopyType, float>::value) {
            AscendC::Adds(dstTensor, srcTensor, 0.0f, elements);
        } else {
            AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_NONE, elements);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[inputIdx]);
    }

    __aicore__ inline void CopyOutFp32Rows(AscendC::GlobalTensor<DT> &outTensor,
                                           AscendC::LocalTensor<float> srcTensor, int64_t outOffset,
                                           uint32_t elements)
    {
        const uint32_t outputIdx = curOutputPingPong_;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx]);
        AscendC::Cast(outputBuf_[outputIdx], srcTensor, AscendC::RoundMode::CAST_RINT, elements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx]);
        AscendC::DataCopy(outTensor[outOffset], outputBuf_[outputIdx], elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx]);
        curOutputPingPong_ ^= 1U;
    }

    __aicore__ inline uint32_t CopyInStateRows(AscendC::LocalTensor<float> dstTensor, int64_t inputOffset,
                                               uint32_t elements)
    {
        const uint32_t inputIdx = curStatePingPong_;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[inputIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[inputIdx]);
        AscendC::DataCopy(dstTensor, workspaceStateGm_[inputOffset], elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[inputIdx]);
        curStatePingPong_ ^= 1U;
        return inputIdx;
    }

    __aicore__ inline void CopyOutStateRows(uint32_t stateIdx, AscendC::LocalTensor<float> srcTensor,
                                            int64_t outOffset, uint32_t elements)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(stateVToMte3Event_[stateIdx]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[stateIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(stateVToMte3Event_[stateIdx]);
        AscendC::DataCopy(workspaceStateGm_[outOffset], srcTensor, elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[stateIdx]);
    }

    __aicore__ inline int64_t GOffset(int64_t b, int64_t hv, int64_t token) const
    {
        return (b * HV_ + hv) * T_ + token;
    }

    __aicore__ inline int64_t Dv2Offset(int64_t b, int64_t hv, int64_t token) const
    {
        return ((b * HV_ + hv) * T_ + token) * V_;
    }

    __aicore__ inline int64_t DhOffset(int64_t b, int64_t hv, int64_t chunkIdx) const
    {
        return ((b * HV_ + hv) * totalChunkNum_ + chunkIdx) * K_ * V_;
    }

    __aicore__ inline int64_t WorkspaceBase(int64_t coreIdx, int64_t workspaceSlot) const
    {
        return (coreIdx * WORKSPACE_BUFFER_COUNT + workspaceSlot) * workspaceElemsPerSubBlock_;
    }

    __aicore__ inline int64_t StateWorkspaceFloatOffset(int64_t workspaceBase, int64_t rowOffset) const
    {
        return ((workspaceBase + stateWorkspaceOffset_) * static_cast<int64_t>(sizeof(DT))) /
                   static_cast<int64_t>(sizeof(float)) +
               rowOffset * V_;
    }

    AscendC::GlobalTensor<DT> qGm_;
    AscendC::GlobalTensor<GT> gateGm_;
    AscendC::GlobalTensor<DT> dvGm_;
    AscendC::GlobalTensor<DT> dhGm_;
    AscendC::GlobalTensor<DT> dh0Gm_;
    AscendC::GlobalTensor<DT> dv2Gm_;
    AscendC::GlobalTensor<DT> workspaceGm_;
    AscendC::GlobalTensor<float> workspaceStateGm_;

    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qInputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qInputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gInputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gInputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> statePing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> statePong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gRawAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateFactorAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dvGateFactorAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gBrcbBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outFp32Buf_;

    AscendC::LocalTensor<DT> qInputBuf_[BUFFER_COUNT];
    AscendC::LocalTensor<GT> gateInputBuf_[BUFFER_COUNT];
    AscendC::LocalTensor<DT> outputBuf_[BUFFER_COUNT];
    AscendC::LocalTensor<float> stateBuf_[BUFFER_COUNT];
    Catlass::Arch::CrossCoreFlag vecToCubeFlag_{VEC_TO_CUBE_FLAG_READY};
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CUBE_TO_VEC_FLAG_READY};

    AscendC::TEventID qMte2ToVEvent_[BUFFER_COUNT];
    AscendC::TEventID qVToMte2Event_[BUFFER_COUNT];
    AscendC::TEventID gateMte2ToVEvent_[BUFFER_COUNT];
    AscendC::TEventID gateVToMte2Event_[BUFFER_COUNT];
    AscendC::TEventID vToMte3Event_[BUFFER_COUNT];
    AscendC::TEventID mte3ToVEvent_[BUFFER_COUNT];
    AscendC::TEventID stateMte2ToVEvent_[BUFFER_COUNT];
    AscendC::TEventID stateVToMte2Event_[BUFFER_COUNT];
    AscendC::TEventID stateVToMte3Event_[BUFFER_COUNT];
    AscendC::TEventID stateMte3ToMte2Event_[BUFFER_COUNT];
    uint32_t curQInputPingPong_ = 0;
    uint32_t curGateInputPingPong_ = 0;
    uint32_t curOutputPingPong_ = 0;
    uint32_t curStatePingPong_ = 0;

    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR dh0Addr_ = nullptr;
    const ChunkGatedDeltaRuleBwdDhuTilingData *tiling_ = nullptr;
    int64_t B_ = 0;
    int64_t HK_ = 0;
    int64_t HV_ = 0;
    int64_t T_ = 0;
    int64_t K_ = 0;
    int64_t V_ = 0;
    int64_t HRatio_ = 0;
    int64_t chunkSize_ = 0;
    int64_t vecRow_ = 8;
    int64_t gateElems_ = 0;
    int64_t totalChunkNum_ = 0;
    int64_t headWindowNum_ = 0;
    int64_t taskNum_ = 0;
    int64_t subBlockNum_ = 1;
    int64_t subBlockIdx_ = 0;
    int64_t isVariable_ = 0;
    float scale_ = 1.0f;
    bool hasDh0_ = false;
    int64_t dh0ClearCoreNum_ = 0;
    int64_t dh0ClearElemsPerCore_ = 0;
    int64_t dh0ClearTailElems_ = 0;
    int64_t workspaceElemsPerSubBlock_ = 0;
    int64_t qgWorkspaceOffset_ = 0;
    int64_t stateWorkspaceOffset_ = 0;
    int64_t dvStateWorkspaceOffset_ = 0;
    int64_t termQWorkspaceOffset_ = 0;
    int64_t termWWorkspaceOffset_ = 0;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_VECTOR_H
