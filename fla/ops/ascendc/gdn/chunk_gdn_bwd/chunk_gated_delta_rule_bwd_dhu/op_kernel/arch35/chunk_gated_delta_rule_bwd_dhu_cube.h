/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_cube.h
 * \brief A5 cube side process for chunk_gated_delta_rule_bwd_dhu.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_CUBE_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif
#include <type_traits>

#include "chunk_gated_delta_rule_bwd_dhu_common.h"
#include "chunk_gated_delta_rule_bwd_dhu_struct.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "kernel_operator.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GDN {

template <typename DT, int V_DIM>
class ChunkGatedDeltaRuleBwdDhuCube {
public:
    __aicore__ inline ChunkGatedDeltaRuleBwdDhuCube() = default;

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR dO, GM_ADDR dh, GM_ADDR dv2,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
                                const ChunkGatedDeltaRuleBwdDhuTilingData *__restrict tilingData)
    {
        k_ = k;
        w_ = w;
        dO_ = dO;
        dh_ = dh;
        dv2_ = dv2;
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        workspace_ = workspace;

        tiling_ = tilingData;
        B_ = tiling_->B;
        HK_ = tiling_->HK;
        HV_ = tiling_->HV;
        T_ = tiling_->T;
        K_ = tiling_->K;
        V_ = tiling_->V;
        HRatio_ = tiling_->HRatio;
        hasGk_ = tiling_->hasGk != 0;
        chunkSize_ = tiling_->chunkSize;
        vecRow_ = tiling_->vecRow > 0 ? tiling_->vecRow : 8;
        totalChunkNum_ = tiling_->totalChunkNum;
        headsPerTask_ = tiling_->headsPerTask;
        headWindowNum_ = tiling_->headWindowNum;
        taskNum_ = tiling_->taskNum;
        workspaceElemsPerSubBlock_ = tiling_->workspaceElemsPerSubBlock;
        dvStateWorkspaceOffset_ = tiling_->dvStateWorkspaceOffset;
        termQWorkspaceOffset_ = tiling_->termQWorkspaceOffset;
        termWWorkspaceOffset_ = tiling_->termWWorkspaceOffset;

        curL1B_ = 0;
        curL0_ = 0;
        curL0C_ = 0;
        nextKResidentSlot_ = 0;
        cachedKResidentValid_ = false;
        cachedKResidentBase_ = 0;
        cachedKResidentSlot_ = 0;
    }

    __aicore__ inline void Process()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        // S2：K/WT 共享聚合区基址（K(loop1)/WT(loop2) 相位交替复用同一 4×32KiB 区）
        AscendC::LocalTensor<DT> kwAggL1 = resource.l1Buf.template GetBufferByByte<DT>(KW_AGG_OFFSET);
        AscendC::LocalTensor<DT> l1AScratch[L1A_SCRATCH_BUFFER_COUNT] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1A_SCRATCH_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1A_SCRATCH_OFFSET + L1A_SCRATCH_TILE_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1A_SCRATCH_OFFSET + 2 * L1A_SCRATCH_TILE_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1A_SCRATCH_OFFSET + 3 * L1A_SCRATCH_TILE_BYTES)};
        AscendC::LocalTensor<DT> l1BScratch[L1B_SCRATCH_BUFFER_COUNT] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1B_SCRATCH_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1B_SCRATCH_OFFSET + L1B_SCRATCH_TILE_BYTES)};
        // S2：dO 聚合 ping-pong 双缓冲基址（仅 DO_AGG_ENABLED 时承载数据；否则纯地址包装、无占用）
        AscendC::LocalTensor<DT> doAggL1[DO_AGG_BUFFER_COUNT] = {
            resource.l1Buf.template GetBufferByByte<DT>(DO_AGG_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(DO_AGG_OFFSET + DO_AGG_BUF_BYTES)};
        AscendC::LocalTensor<DT> l0A[L0_BUFFER_COUNT] = {
            resource.l0ABuf.template GetBufferByByte<DT>(0),
            resource.l0ABuf.template GetBufferByByte<DT>(L0A_TILE_BYTES)};
        AscendC::LocalTensor<DT> l0B[L0_BUFFER_COUNT] = {
            resource.l0BBuf.template GetBufferByByte<DT>(0),
            resource.l0BBuf.template GetBufferByByte<DT>(L0B_TILE_BYTES)};
        AscendC::LocalTensor<ElementAccumulator> l0C[L0C_BUFFER_COUNT];
        l0C[0] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        if constexpr (L0C_BUFFER_COUNT > 1) {
            l0C[1] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_BYTES);
        }
        const uint32_t cvStrideBytes =
            static_cast<uint32_t>(vecRow_ * V_ * static_cast<int64_t>(sizeof(DT)));
        AscendC::LocalTensor<DT> matrixCvBuf[CV_BUFFER_COUNT] = {
            resource.ubBuf.template GetBufferByByte<DT>(0),
            resource.ubBuf.template GetBufferByByte<DT>(cvStrideBytes)};
        InitPipeFlags();

        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());

        for (int64_t taskIdx = blockIdx; taskIdx < taskNum_; taskIdx += blockNum) {
            const int64_t seqIdx = taskIdx / headWindowNum_;
            const int64_t headWindowIdx = taskIdx - seqIdx * headWindowNum_;
            const int64_t hvBase = headWindowIdx * headsPerTask_;
            const int64_t headCnt = Min(headsPerTask_, HV_ - hvBase);
            const int64_t taskRound = (taskIdx - blockIdx) / blockNum;
            const int64_t windowStartSlot = (taskRound & 1) * HEADS_PER_TASK;
            if (headCnt <= 0) {
                continue;
            }

            SeqInfo seqInfo;
            GetSeqInfo(cuSeqlens_, *tiling_, seqIdx, seqInfo);
            if (!seqInfo.valid) {
                continue;
            }

            for (int64_t chunkIdx = seqInfo.chunkCnt - 1; chunkIdx >= 0; --chunkIdx) {
                ChunkInfo chunkInfo;
                GetChunkInfoBySeqChunk(chunkIndices_, *tiling_, seqInfo, chunkIdx, chunkInfo);
                if (!chunkInfo.valid) {
                    continue;
                }

                cachedKResidentValid_ = false;
                nextKResidentSlot_ = 0;
                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    const int64_t hv = hvBase + headOffset;
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    const bool releaseKAfterUse = headOffset + 1 == headCnt;
                    const int64_t hq = hv / HRatio_;
                    const int64_t kBase = ((chunkInfo.bIdx * HK_ + hq) * T_ + chunkInfo.tokenStart) * K_;
                    const int64_t dOBase = ((chunkInfo.bIdx * HV_ + hv) * T_ + chunkInfo.tokenStart) * V_;
                    const int64_t dhBase =
                        ((chunkInfo.bIdx * HV_ + hv) * totalChunkNum_ + chunkInfo.outputChunkIdx) * K_ * V_;
                    const int64_t slotBase = WorkspaceBase(blockIdx, workspaceSlot);

                    LayoutTagK tagK = LayoutTagK::MakeLayout<DT>(chunkSize_, K_);
                    LayoutTagState tagState = LayoutTagState::MakeLayout<DT>(K_, V_DIM);
                    LayoutTagDvState tagDvState = LayoutTagDvState::MakeLayout<DT>(chunkSize_, V_DIM);
                    LayoutTagDO tagDO = LayoutTagDO::MakeLayout<DT>(chunkSize_, V_DIM);
                    LayoutTagTermQ tagTermQ = LayoutTagTermQ::MakeLayout<DT>(K_, V_DIM);

                    auto layoutK = tla::MakeLayoutFromTag(tagK);
                    auto layoutState = tla::MakeLayoutFromTag(tagState);
                    auto layoutDvState = tla::MakeLayoutFromTag(tagDvState);
                    auto layoutDO = tla::MakeLayoutFromTag(tagDO);
                    auto layoutTermQ = tla::MakeLayoutFromTag(tagTermQ);

                    AscendC::GlobalTensor<DT> gmK;
                    AscendC::GlobalTensor<DT> gmState;
                    AscendC::GlobalTensor<DT> gmDvState;
                    AscendC::GlobalTensor<DT> gmDO;
                    AscendC::GlobalTensor<DT> gmTermQ;
                    gmK.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(k_) + kBase);
                    gmState.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh_) + dhBase);
                    gmDvState.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace_) + slotBase +
                                              dvStateWorkspaceOffset_);
                    gmDO.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dO_) + dOBase);
                    gmTermQ.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace_) + slotBase +
                                            termQWorkspaceOffset_);

                    auto tensorK = tla::MakeTensor(gmK, layoutK, Catlass::Arch::PositionGM{});
                    auto blockK = tla::GetTile(
                        tensorK, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(K_)));
                    CopyGmToL1A_DvState<decltype(blockK)> copyGmToL1A_K;
                    // S2：K/WT 输入流 ndNum 头聚合（R-Akk LoadGroupA 同构）。HRatio==1 时 K 逐头独立
                    //（头间 stride 恒 T_*K_ elem，varlen 同式），head0 一条 ndNum=headCnt 聚合装载
                    // 本 task 全部头的 K；HRatio>1（GVA）组内共享同一 K，聚合=重复读字节负收益，
                    // 回退逐头 ndNum=1 + kBase 去重（位级无差）。K(loop1)/WT(loop2) 相位交替共享
                    // 4×16384 elem 区；区信用 EVENT_KW_AGG 双向单 ID（MTE1_MTE2=区空闲、
                    // MTE2_MTE1=数据就绪）：K 相 head0 wait free → 末头 dvState GEMM 末 K set
                    // free；WT 相 head0 wait free → 末头 termW GEMM 末 K set free，逐 chunk 严格交替。
                    const bool kAggActive = HRatio_ == 1;
                    bool waitKReady;
                    uint32_t kMatrixSlot;
                    if (headOffset == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_KW_AGG);
                        if (kAggActive) {
                            auto tensorL1KAgg =
                                tla::MakeTensor(kwAggL1, L1A_LAYOUT_K, Catlass::Arch::PositionL1{});
                            copyGmToL1A_K(tensorL1KAgg, blockK, static_cast<uint32_t>(headCnt),
                                          static_cast<uint32_t>(T_ * K_), KW_AGG_MATRIX_ELEMS);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_KW_AGG);
                        }
                    }
                    if (kAggActive) {
                        kMatrixSlot = static_cast<uint32_t>(headOffset);
                        waitKReady = headOffset == 0;
                    } else {
                        const bool needLoadKResident =
                            !cachedKResidentValid_ || cachedKResidentBase_ != kBase;
                        if (needLoadKResident) {
                            kMatrixSlot = nextKResidentSlot_;
                            nextKResidentSlot_ += 1U;
                            cachedKResidentBase_ = kBase;
                            cachedKResidentSlot_ = kMatrixSlot;
                            cachedKResidentValid_ = true;
                            auto tensorL1KLoad = tla::MakeTensor(
                                kwAggL1[kMatrixSlot * KW_AGG_MATRIX_ELEMS], L1A_LAYOUT_K,
                                Catlass::Arch::PositionL1{});
                            copyGmToL1A_K(tensorL1KLoad, blockK);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_KW_AGG);
                        } else {
                            kMatrixSlot = cachedKResidentSlot_;
                        }
                        waitKReady = needLoadKResident;
                    }
                    const int32_t kResidentEvent = EVENT_KW_AGG;
                    auto tensorL1K = tla::MakeTensor(
                        kwAggL1[kMatrixSlot * KW_AGG_MATRIX_ELEMS], L1A_LAYOUT_K,
                        Catlass::Arch::PositionL1{});

                    auto tensorDO = tla::MakeTensor(gmDO, layoutDO, Catlass::Arch::PositionGM{});
                    auto blockDO = tla::GetTile(
                        tensorDO, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(V_DIM)));
                    CopyGmToL1B_TermQ<decltype(blockDO)> copyGmToL1B_DO;

                    // S1-b/S2：dO 装载 ndNum 头聚合 A/B，S2 升级 ping-pong 双缓冲（buf=chunkIdx&1，
                    // 下一同奇偶 chunk 的聚合装载与本 chunk GEMM 消费经双 buf 重叠）。dO 逐头独立
                    //（hv 直引、不经 HRatio，GVA 组内无共享、聚合零重复读）；头间 stride 恒 T_*V_ elem
                    //（varlen 同式）。事件 {EVENT_DO_AGG_PING/PONG} 每 buf 一个双向 ID：
                    // 装载 wait free/set ready，head0 GEMM 首 K wait ready、末头 GEMM 末 K set free。
                    // 门控 headCnt∈(1,DO_AGG_HEADS]：headCnt==1 回退逐头 scratch（S1-b 教训：
                    // 组级信用协议 headCnt==1 回退档必须逐拍=原路径）；headCnt>DO_AGG_HEADS 超预算档
                    // 同路径回退（位级等价，R5 纪律）。
                    bool waitDoReady = true;
                    bool releaseDoAfter = true;
                    int32_t doL1BEvent = 0;
                    AscendC::LocalTensor<DT> doL1Tensor = l1BScratch[0];
                    bool doAggActive = false;
                    if constexpr (DO_AGG_ENABLED) {
                        doAggActive = headCnt > 1 && headCnt <= DO_AGG_HEADS;
                        if (doAggActive) {
                            const uint32_t doAggBuf = static_cast<uint32_t>(chunkIdx & 1);
                            const int32_t doAggEvent =
                                doAggBuf == 0 ? EVENT_DO_AGG_PING : EVENT_DO_AGG_PONG;
                            if (headOffset == 0) {
                                // 单条 ndNum=headCnt 聚合装载本 task 全部头（尾窗口运行时 ndNum=headCnt），
                                // 位置与现状逐头装载相同（保持在 flag3 等待阴影内、termQ GEMM 之前）
                                const int64_t dOAggBase =
                                    ((chunkInfo.bIdx * HV_ + hvBase) * T_ + chunkInfo.tokenStart) * V_;
                                AscendC::GlobalTensor<DT> gmDOAgg;
                                gmDOAgg.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dO_) + dOAggBase);
                                auto tensorDOAgg =
                                    tla::MakeTensor(gmDOAgg, layoutDO, Catlass::Arch::PositionGM{});
                                auto blockDOAgg = tla::GetTile(
                                    tensorDOAgg, tla::MakeCoord(0, 0),
                                    tla::MakeShape(static_cast<uint32_t>(chunkInfo.chunkLen),
                                                   static_cast<uint32_t>(V_DIM)));
                                auto tensorL1DOAggDst = tla::MakeTensor(doAggL1[doAggBuf], L1B_LAYOUT_DO,
                                                                        Catlass::Arch::PositionL1{});
                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(doAggEvent);
                                copyGmToL1B_DO(tensorL1DOAggDst, blockDOAgg, static_cast<uint32_t>(headCnt),
                                               static_cast<uint32_t>(T_ * V_), DO_AGG_MATRIX_ELEMS);
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(doAggEvent);
                            }
                            curL1B_ ^= 1U; // 空翻转：保持 dh-state 的 L1B scratch 槽位序列与现状逐拍一致
                            doL1Tensor = doAggL1[doAggBuf][headOffset * DO_AGG_MATRIX_ELEMS];
                            // 组级信用（R-Akk 同款）：head0 GEMM 首 K 等 ready；组内末头 GEMM 末 K 释放 free
                            waitDoReady = headOffset == 0;
                            releaseDoAfter = headOffset + 1 == headCnt;
                            doL1BEvent = doAggEvent;
                        }
                    }
                    if (!doAggActive) {
                        const uint32_t doScratchSlot = curL1B_;
                        curL1B_ ^= 1U;
                        const int32_t doScratchEvent = L1BScratchEvent(doScratchSlot);
                        doL1Tensor = l1BScratch[doScratchSlot];
                        doL1BEvent = doScratchEvent;
                        auto tensorL1DOLoad = tla::MakeTensor(doL1Tensor, L1B_LAYOUT_DO,
                                                              Catlass::Arch::PositionL1{});
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(doScratchEvent);
                        copyGmToL1B_DO(tensorL1DOLoad, blockDO);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(doScratchEvent);
                    }
                    auto tensorL1DO =
                        tla::MakeTensor(doL1Tensor, L1B_LAYOUT_DO, Catlass::Arch::PositionL1{});

                    // C2a：loop1 头只 gate dh（GM→L1B，:252 与 dvState GEMM），改等 flag3(dhReady)；
                    // qg 的就绪等待移到 termQ GEMM 前（见下方 flag2 wait）
                    Catlass::Arch::CrossCoreWaitFlag(vecToCubeDhFlag_);

                    auto tensorState = tla::MakeTensor(gmState, layoutState, Catlass::Arch::PositionGM{});
                    auto tensorDvState = tla::MakeTensor(gmDvState, layoutDvState, Catlass::Arch::PositionGM{});
                    auto blockState = tla::GetTile(
                        tensorState, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM)));
                    auto blockDvState = tla::GetTile(
                        tensorDvState, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(V_DIM)));
                    CopyGmToL1B_DvState<decltype(blockState)> copyGmToL1B_State;
                    CopyL1ToL0A_DvState copyL1ToL0A_DvState;
                    CopyL1ToL0B_DvState copyL1ToL0B_DvState;
                    TileMmadDvState tileMmadDvState;

                    const uint32_t stateScratchSlot = curL1B_;
                    curL1B_ ^= 1U;
                    const int32_t stateScratchEvent = L1BScratchEvent(stateScratchSlot);
                    auto tensorL1State =
                        tla::MakeTensor(l1BScratch[stateScratchSlot], L1B_LAYOUT_STATE, Catlass::Arch::PositionL1{});
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(stateScratchEvent);
                    copyGmToL1B_State(tensorL1State, blockState);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(stateScratchEvent);
                    if constexpr (std::is_same<DT, bfloat16_t>::value) {
                        const bool useGmDvState = V_DIM == 256 && chunkInfo.chunkLen > 64;
                        if (useGmDvState) {
                            CopyL0CToGm_DvState<decltype(blockDvState)> copyL0CToGm_DvState;
                            RunResidentMmad<LayoutTagL0A_DvState, LayoutTagL0B_DvState>(
                                copyL1ToL0A_DvState, copyL1ToL0B_DvState, tileMmadDvState, copyL0CToGm_DvState,
                                tensorL1K, tensorL1State, blockDvState, l0A, l0B, l0C,
                                waitKReady, releaseKAfterUse, kResidentEvent, true, true, stateScratchEvent,
                                static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(V_DIM),
                                static_cast<uint32_t>(K_));
                        } else {
                            uint32_t mActual = static_cast<uint32_t>(chunkInfo.chunkLen);
                            if (mActual == 1) {
                                mActual = 16;
                            }
                            const uint32_t l0CSlot = curL0C_;
                            const int32_t l0CEvent = L0CEvent(l0CSlot);
                            auto layoutL0C = tla::MakeLayoutL0C(mActual, static_cast<uint32_t>(V_DIM));
                            auto tensorL0C =
                                tla::MakeTensor(l0C[l0CSlot], layoutL0C, Catlass::Arch::PositionL0C{});
                            auto tensorTileL0C = tla::GetTile(
                                tensorL0C, tla::MakeCoord(0, 0),
                                tla::MakeShape(mActual, static_cast<uint32_t>(V_DIM)));

                            bool waitStateReady = true;
                            for (uint32_t kOffset = 0; kOffset < static_cast<uint32_t>(K_); kOffset += L0_K_TILE) {
                                const uint32_t curK = kOffset + L0_K_TILE > static_cast<uint32_t>(K_) ?
                                                          static_cast<uint32_t>(K_) - kOffset :
                                                          L0_K_TILE;
                                const bool firstK = kOffset == 0;
                                const bool lastK = kOffset + curK >= static_cast<uint32_t>(K_);
                                const uint32_t l0Slot = curL0_;
                                const int32_t l0AEvent = L0AEvent(l0Slot);
                                const int32_t l0BEvent = L0BEvent(l0Slot);
                                const int32_t l0ReadyEvent = L0ReadyEvent(l0Slot);

                                auto layoutL0A = tla::MakeLayout<DT, LayoutTagL0A_DvState>(mActual, curK);
                                auto tensorL0A =
                                    tla::MakeTensor(l0A[l0Slot], layoutL0A, Catlass::Arch::PositionL0A{});
                                auto tensorTileL1A = tla::GetTile(
                                    tensorL1K, tla::MakeCoord(0, kOffset), tla::MakeShape(mActual, curK));
                                if (waitKReady) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kResidentEvent);
                                    waitKReady = false;
                                }
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
                                copyL1ToL0A_DvState(tensorL0A, tensorTileL1A);
                                if (lastK && releaseKAfterUse) {
                                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kResidentEvent);
                                }

                                auto layoutL0B = tla::MakeLayout<DT, LayoutTagL0B_DvState>(
                                    curK, static_cast<uint32_t>(V_DIM));
                                auto tensorL0B =
                                    tla::MakeTensor(l0B[l0Slot], layoutL0B, Catlass::Arch::PositionL0B{});
                                auto tensorTileL1B = tla::GetTile(
                                    tensorL1State, tla::MakeCoord(kOffset, 0),
                                    tla::MakeShape(curK, static_cast<uint32_t>(V_DIM)));
                                if (waitStateReady) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(stateScratchEvent);
                                    waitStateReady = false;
                                }
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
                                copyL1ToL0B_DvState(tensorL0B, tensorTileL1B);
                                if (lastK) {
                                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(stateScratchEvent);
                                }
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
                                curL0_ ^= 1U;

                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
                                if (firstK) {
                                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEvent);
                                }
                                const uint8_t unitFlag = lastK ? 0b11 : 0b10;
                                tileMmadDvState(tensorTileL0C, tensorL0A, tensorL0B, firstK, unitFlag);
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
                                if (lastK) {
                                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEvent);
                                }
                            }

                            SwitchL0C();
                            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEvent);
                            // C5b：dvState CV 片逐片目标子块由 CvTargetSubBlock（limit=chunkLen，劈分头按
                            // DhuSplitHalf 16 对齐半界）判定，cvRows 在半界处截断（片不跨界，如 64 行单片→2×32）；
                            // 非劈分头=headOffset&1 与现状相同。cvListId 按目标子块独立 ping-pong，
                            // 与各 AIV phase2 局部 cvListId（各自从 0 起逐片翻转）逐片配对（沿用 C5-a 模式）
                            uint32_t cvListId[2] = {0U, 0U};
                            uint32_t rowIdx = 0;
                            const uint32_t tokenHalf = IsDhuSplitHead(headCnt, headOffset)
                                                           ? static_cast<uint32_t>(DhuSplitHalf(chunkInfo.chunkLen))
                                                           : 0U;
                            while (rowIdx < static_cast<uint32_t>(chunkInfo.chunkLen)) {
                                const uint32_t leftRows = static_cast<uint32_t>(chunkInfo.chunkLen) - rowIdx;
                                uint32_t cvRows = leftRows > static_cast<uint32_t>(vecRow_) ?
                                                      static_cast<uint32_t>(vecRow_) :
                                                      leftRows;
                                if (tokenHalf != 0 && rowIdx < tokenHalf && rowIdx + cvRows > tokenHalf) {
                                    cvRows = tokenHalf - rowIdx;
                                }
                                const uint32_t cvTarget =
                                    CvTargetSubBlock(headCnt, headOffset, rowIdx, chunkInfo.chunkLen);
                                const uint64_t subBlockFlagOffset =
                                    cvTarget == 0 ? 0 : CV_SUBBLOCK_FLAG_STRIDE;
                                auto tensorCv = tla::MakeTensor(
                                    matrixCvBuf[cvListId[cvTarget]], UB_LAYOUT_DVSTATE_CV,
                                    Catlass::Arch::PositionUB{});
                                auto blockCv = tla::GetTile(
                                    tensorCv, tla::MakeCoord(0, 0),
                                    tla::MakeShape(cvRows, static_cast<uint32_t>(V_DIM)));
                                auto blockL0C = tla::GetTile(
                                    tensorL0C, tla::MakeCoord(rowIdx, 0),
                                    tla::MakeShape(cvRows, static_cast<uint32_t>(V_DIM)));
                                CopyL0CToUB_DvState<decltype(blockCv)> copyL0CToUB;
                                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(
                                    MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN + subBlockFlagOffset + cvListId[cvTarget]);
                                copyL0CToUB(blockCv, blockL0C, cvRows,
                                              static_cast<uint8_t>(cvTarget), 1, 0b11);
                                AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
                                    MATRIX_CV_AIC_TO_AIV_FLAG_BEGIN + subBlockFlagOffset + cvListId[cvTarget]);
                                rowIdx += cvRows;
                                cvListId[cvTarget] ^= 1U;
                            }
                            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent);
                        }
                    } else {
                        CopyL0CToGm_DvState<decltype(blockDvState)> copyL0CToGm_DvState;
                        RunResidentMmad<LayoutTagL0A_DvState, LayoutTagL0B_DvState>(
                            copyL1ToL0A_DvState, copyL1ToL0B_DvState, tileMmadDvState, copyL0CToGm_DvState,
                            tensorL1K, tensorL1State, blockDvState, l0A, l0B, l0C,
                            waitKReady, releaseKAfterUse, kResidentEvent, true, true, stateScratchEvent,
                            static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(V_DIM),
                            static_cast<uint32_t>(K_));
                    }

                    auto tensorTermQ = tla::MakeTensor(gmTermQ, layoutTermQ, Catlass::Arch::PositionGM{});
                    auto blockTermQ = tla::GetTile(
                        tensorTermQ, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM)));
                    CopyL0CToGm_TermQ<decltype(blockTermQ)> copyL0CToGm_TermQ;
                    CopyL1ToL0A_TermQ copyL1ToL0A_TermQ;
                    CopyL1ToL0B_TermQ copyL1ToL0B_TermQ;
                    TileMmadTermQ tileMmadTermQ;

                    uint32_t qgScratchSlot = static_cast<uint32_t>(headOffset);
                    if (hasGk_) {
                        const int64_t groupStartHv = hq * HRatio_;
                        qgScratchSlot = static_cast<uint32_t>(groupStartHv > hvBase ? groupStartHv - hvBase : 0);
                    }
                    auto tensorL1QGT =
                        tla::MakeTensor(l1AScratch[qgScratchSlot], L1A_LAYOUT_QGT, Catlass::Arch::PositionL1{});
                    // C2a：qgReady（flag2）wait 插在 qgScratchSlot 计算之后、首次 L1→L0A 拷 qg（RunResidentMmad
                    // 内 copyL1ToL0A 读 tensorL1QGT）之前；GVA 组跟随头（produceQG=false）的 qg 由组首头更早
                    // 写入，AIC 按头序 wait、AIV 按头序 set（PIPE_MTE3），传递覆盖
                    Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
                    RunResidentMmad<LayoutTagL0A_TermQ, LayoutTagL0B_TermQ>(
                        copyL1ToL0A_TermQ, copyL1ToL0B_TermQ, tileMmadTermQ, copyL0CToGm_TermQ,
                        tensorL1QGT, tensorL1DO, blockTermQ, l0A, l0B, l0C,
                        false, false, 0, waitDoReady, releaseDoAfter, doL1BEvent,
                        static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM),
                        static_cast<uint32_t>(chunkInfo.chunkLen));

                    // C2-b：loop1 末 flag4 set 仅 GM-dvState 路径执行（与 vector phase2 头 wait 共用
                    // IsGmDvStatePath 逐 chunk 推导）；CV 路径 phase2 不读 loop1 GM 产物，set 省略。
                    // loop2 的 flag4 set（GM/CV/fp16 三分支）无条件保留——phase3 的 termQ 就绪由
                    // loop2 的 PIPE_FIX set 传递覆盖（FIX 管保序，CV 分支 set 位于 termW CV 流之前）
                    if (IsGmDvStatePath<DT>(V_DIM, chunkInfo.chunkLen)) {
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                    }
                }
                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    const int64_t hv = hvBase + headOffset;
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    const int64_t wBase = ((chunkInfo.bIdx * HV_ + hv) * T_ + chunkInfo.tokenStart) * K_;
                    const int64_t dv2Base = ((chunkInfo.bIdx * HV_ + hv) * T_ + chunkInfo.tokenStart) * V_;
                    const int64_t slotBase = WorkspaceBase(blockIdx, workspaceSlot);
                    bool waitWReady = headOffset == 0;
                    const bool releaseWAfterUse = headOffset + 1 == headCnt;

                    LayoutTagWT tagWT = LayoutTagWT::MakeLayout<DT>(K_, chunkSize_);
                    LayoutTagDv2 tagDv2 = LayoutTagDv2::MakeLayout<DT>(chunkSize_, V_DIM);
                    LayoutTagTermW tagTermW = LayoutTagTermW::MakeLayout<DT>(K_, V_DIM);

                    auto layoutWT = tla::MakeLayoutFromTag(tagWT);
                    auto layoutDv2 = tla::MakeLayoutFromTag(tagDv2);
                    auto layoutTermW = tla::MakeLayoutFromTag(tagTermW);

                    AscendC::GlobalTensor<DT> gmWT;
                    AscendC::GlobalTensor<DT> gmDv2;
                    AscendC::GlobalTensor<DT> gmTermW;
                    gmWT.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(w_) + wBase);
                    gmDv2.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dv2_) + dv2Base);
                    gmTermW.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace_) + slotBase +
                                            termWWorkspaceOffset_);

                    auto tensorWT = tla::MakeTensor(gmWT, layoutWT, Catlass::Arch::PositionGM{});
                    auto tensorDv2 = tla::MakeTensor(gmDv2, layoutDv2, Catlass::Arch::PositionGM{});
                    auto tensorTermW = tla::MakeTensor(gmTermW, layoutTermW, Catlass::Arch::PositionGM{});
                    auto blockWT = tla::GetTile(
                        tensorWT, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(K_), static_cast<uint32_t>(chunkInfo.chunkLen)));
                    auto blockDv2 = tla::GetTile(
                        tensorDv2, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(V_DIM)));
                    auto blockTermW = tla::GetTile(
                        tensorTermW, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM)));
                    CopyGmToL1A_TermW<decltype(blockWT)> copyGmToL1A_WT;
                    CopyGmToL1B_TermW<decltype(blockDv2)> copyGmToL1B_Dv2;
                    CopyL1ToL0A_TermW copyL1ToL0A_TermW;
                    CopyL1ToL0B_TermW copyL1ToL0B_TermW;
                    TileMmadTermW tileMmadTermW;

                    // S2：WT 流 ndNum 头聚合（w 逐头独立、头间 stride 恒 T_*K_ elem，varlen 同式）：
                    // head0 一条 ndNum=headCnt 聚合装载进 KW 区（与 loop1 的 K 相位交替复用，
                    // 区信用 EVENT_KW_AGG 双向单 ID 见 loop1 K 块注释）；headCnt==1 时 ndNum=1，
                    // 事件序列与原逐头路径同构。
                    const int32_t wEvent = EVENT_KW_AGG;
                    if (headOffset == 0) {
                        auto tensorL1WTAgg =
                            tla::MakeTensor(kwAggL1, L1A_LAYOUT_WT, Catlass::Arch::PositionL1{});
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(wEvent);
                        copyGmToL1A_WT(tensorL1WTAgg, blockWT, static_cast<uint32_t>(headCnt),
                                       static_cast<uint32_t>(T_ * K_), KW_AGG_MATRIX_ELEMS);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(wEvent);
                    }
                    auto tensorL1WT = tla::MakeTensor(
                        kwAggL1[headOffset * KW_AGG_MATRIX_ELEMS], L1A_LAYOUT_WT,
                        Catlass::Arch::PositionL1{});

                    Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);

                    const uint32_t dv2ScratchSlot = curL1B_;
                    curL1B_ ^= 1U;
                    const int32_t dv2ScratchEvent = L1BScratchEvent(dv2ScratchSlot);
                    auto tensorL1Dv2 =
                        tla::MakeTensor(l1BScratch[dv2ScratchSlot], L1B_LAYOUT_DV2, Catlass::Arch::PositionL1{});
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dv2ScratchEvent);
                    copyGmToL1B_Dv2(tensorL1Dv2, blockDv2);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dv2ScratchEvent);

                    if constexpr (std::is_same<DT, bfloat16_t>::value) {
                        const bool useGmTermW = V_DIM == 256 && chunkInfo.chunkLen > 64;
                        if (useGmTermW) {
                            CopyL0CToGm_TermW<decltype(blockTermW)> copyL0CToGm_TermW;
                            RunResidentMmad<LayoutTagL0A_TermW, LayoutTagL0B_TermW>(
                                copyL1ToL0A_TermW, copyL1ToL0B_TermW, tileMmadTermW, copyL0CToGm_TermW,
                                tensorL1WT, tensorL1Dv2, blockTermW, l0A, l0B, l0C,
                                waitWReady, releaseWAfterUse, wEvent, true, true, dv2ScratchEvent,
                                static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM),
                                static_cast<uint32_t>(chunkInfo.chunkLen));
                            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                        } else {
                            const uint32_t l0CSlot = curL0C_;
                            const int32_t l0CEvent = L0CEvent(l0CSlot);
                            auto layoutL0C = tla::MakeLayoutL0C(static_cast<uint32_t>(K_),
                                                                static_cast<uint32_t>(V_DIM));
                            auto tensorL0C =
                                tla::MakeTensor(l0C[l0CSlot], layoutL0C, Catlass::Arch::PositionL0C{});
                            auto tensorTileL0C = tla::GetTile(
                                tensorL0C, tla::MakeCoord(0, 0),
                                tla::MakeShape(static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM)));
                            bool waitDv2Ready = true;

                            for (uint32_t kOffset = 0; kOffset < static_cast<uint32_t>(chunkInfo.chunkLen);
                                 kOffset += L0_K_TILE) {
                                const uint32_t leftK = static_cast<uint32_t>(chunkInfo.chunkLen) - kOffset;
                                const uint32_t curK = leftK > L0_K_TILE ? L0_K_TILE : leftK;
                                const bool firstK = kOffset == 0;
                                const bool lastK = kOffset + curK >= static_cast<uint32_t>(chunkInfo.chunkLen);
                                const uint32_t l0Slot = curL0_;
                                const int32_t l0AEvent = L0AEvent(l0Slot);
                                const int32_t l0BEvent = L0BEvent(l0Slot);
                                const int32_t l0ReadyEvent = L0ReadyEvent(l0Slot);

                                auto layoutL0A = tla::MakeLayout<DT, LayoutTagL0A_TermW>(
                                    static_cast<uint32_t>(K_), curK);
                                auto tensorL0A =
                                    tla::MakeTensor(l0A[l0Slot], layoutL0A, Catlass::Arch::PositionL0A{});
                                auto tensorTileL1A = tla::GetTile(
                                    tensorL1WT, tla::MakeCoord(0, kOffset),
                                    tla::MakeShape(static_cast<uint32_t>(K_), curK));
                                if (waitWReady) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(wEvent);
                                    waitWReady = false;
                                }
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
                                copyL1ToL0A_TermW(tensorL0A, tensorTileL1A);
                                if (lastK && releaseWAfterUse) {
                                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(wEvent);
                                }

                                auto layoutL0B = tla::MakeLayout<DT, LayoutTagL0B_TermW>(
                                    curK, static_cast<uint32_t>(V_DIM));
                                auto tensorL0B =
                                    tla::MakeTensor(l0B[l0Slot], layoutL0B, Catlass::Arch::PositionL0B{});
                                auto tensorTileL1B = tla::GetTile(
                                    tensorL1Dv2, tla::MakeCoord(kOffset, 0),
                                    tla::MakeShape(curK, static_cast<uint32_t>(V_DIM)));
                                if (waitDv2Ready) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dv2ScratchEvent);
                                    waitDv2Ready = false;
                                }
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
                                copyL1ToL0B_TermW(tensorL0B, tensorTileL1B);
                                if (lastK) {
                                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dv2ScratchEvent);
                                }
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
                                curL0_ ^= 1U;

                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
                                if (firstK) {
                                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEvent);
                                }
                                const uint8_t unitFlag = lastK ? 0b11 : 0b10;
                                tileMmadTermW(tensorTileL0C, tensorL0A, tensorL0B, firstK, unitFlag);
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
                                if (lastK) {
                                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEvent);
                                }
                            }

                            SwitchL0C();
                            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEvent);
                            uint32_t rowIdx = 0;
                            // C5a：termW CV 片逐片目标子块由 CvTargetSubBlock 判定（劈分头按 K 半段，
                            // 非劈分头=headOffset&1 与现状相同）；cvListId 按目标子块独立 ping-pong，
                            // 与各 AIV phase3 局部 cvListId（各自从 0 起逐片翻转）逐片配对（R-C5-1 收敛点）
                            uint32_t cvListId[2] = {0U, 0U};
                            while (rowIdx < static_cast<uint32_t>(K_)) {
                                const uint32_t leftRows = static_cast<uint32_t>(K_) - rowIdx;
                                const uint32_t cvRows = leftRows > static_cast<uint32_t>(vecRow_) ?
                                                            static_cast<uint32_t>(vecRow_) :
                                                            leftRows;
                                const uint32_t cvTarget =
                                    CvTargetSubBlock(headCnt, headOffset, rowIdx, K_);
                                const uint64_t subBlockFlagOffset =
                                    cvTarget == 0 ? 0 : CV_SUBBLOCK_FLAG_STRIDE;
                                auto tensorCv = tla::MakeTensor(
                                    matrixCvBuf[cvListId[cvTarget]], UB_LAYOUT_TERMW_CV,
                                    Catlass::Arch::PositionUB{});
                                auto blockCv = tla::GetTile(
                                    tensorCv, tla::MakeCoord(0, 0),
                                    tla::MakeShape(cvRows, static_cast<uint32_t>(V_DIM)));
                                auto blockL0C = tla::GetTile(
                                    tensorL0C, tla::MakeCoord(rowIdx, 0),
                                    tla::MakeShape(cvRows, static_cast<uint32_t>(V_DIM)));
                                CopyL0CToUB_TermW<decltype(blockCv)> copyL0CToUB;
                                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(
                                    MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN + subBlockFlagOffset + cvListId[cvTarget]);
                                copyL0CToUB(blockCv, blockL0C, cvRows,
                                              static_cast<uint8_t>(cvTarget), 1, 0b11);
                                AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
                                    MATRIX_CV_AIC_TO_AIV_FLAG_BEGIN + subBlockFlagOffset + cvListId[cvTarget]);
                                rowIdx += cvRows;
                                cvListId[cvTarget] ^= 1U;
                            }
                            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent);
                        }
                    } else {
                        CopyL0CToGm_TermW<decltype(blockTermW)> copyL0CToGm_TermW;
                        RunResidentMmad<LayoutTagL0A_TermW, LayoutTagL0B_TermW>(
                            copyL1ToL0A_TermW, copyL1ToL0B_TermW, tileMmadTermW, copyL0CToGm_TermW,
                            tensorL1WT, tensorL1Dv2, blockTermW, l0A, l0B, l0C,
                            waitWReady, releaseWAfterUse, wEvent, true, true, dv2ScratchEvent,
                            static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM),
                            static_cast<uint32_t>(chunkInfo.chunkLen));
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                    }
                }
            }
        }

        DrainPipeFlags();
    }

private:
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutTagK = Catlass::layout::RowMajor;
    using LayoutTagState = Catlass::layout::RowMajor;
    using LayoutTagDvState = Catlass::layout::RowMajor;
    using LayoutTagQGT = Catlass::layout::ColumnMajor;
    using LayoutTagDO = Catlass::layout::RowMajor;
    using LayoutTagTermQ = Catlass::layout::RowMajor;
    using LayoutTagWT = Catlass::layout::ColumnMajor;
    using LayoutTagDv2 = Catlass::layout::RowMajor;
    using LayoutTagTermW = Catlass::layout::RowMajor;
    using TileCopyDvState =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutTagK, DT, LayoutTagState, DT, LayoutTagDvState>;
    using TileCopyTermQ =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutTagQGT, DT, LayoutTagDO, DT, LayoutTagTermQ>;
    using TileCopyTermW =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DT, LayoutTagWT, DT, LayoutTagDv2, DT, LayoutTagTermW>;
    using TileCopyDvStateToUB =
        Common::Tile::PackedTileCopyTlaToUB<ArchTag, DT, LayoutTagK, DT, LayoutTagState, DT, LayoutTagDvState>;
    using TileCopyTermWToUB =
        Common::Tile::PackedTileCopyTlaToUB<ArchTag, DT, LayoutTagWT, DT, LayoutTagDv2, DT, LayoutTagTermW>;
    using ElementAccumulator = typename TileCopyDvState::ElementAccumulator;
    using CopyL1ToL0A_DvState = typename TileCopyDvState::CopyL1ToL0A;
    using CopyL1ToL0B_DvState = typename TileCopyDvState::CopyL1ToL0B;
    using CopyL1ToL0A_TermQ = typename TileCopyTermQ::CopyL1ToL0A;
    using CopyL1ToL0B_TermQ = typename TileCopyTermQ::CopyL1ToL0B;
    using CopyL1ToL0A_TermW = typename TileCopyTermW::CopyL1ToL0A;
    using CopyL1ToL0B_TermW = typename TileCopyTermW::CopyL1ToL0B;

    using LayoutTagL1A_DvState = typename TileCopyDvState::LayoutTagL1A;
    using LayoutTagL1B_DvState = typename TileCopyDvState::LayoutTagL1B;
    using LayoutTagL0A_DvState = typename TileCopyDvState::LayoutTagL0A;
    using LayoutTagL0B_DvState = typename TileCopyDvState::LayoutTagL0B;
    using LayoutTagL1A_TermQ = typename TileCopyTermQ::LayoutTagL1A;
    using LayoutTagL1B_TermQ = typename TileCopyTermQ::LayoutTagL1B;
    using LayoutTagL0A_TermQ = typename TileCopyTermQ::LayoutTagL0A;
    using LayoutTagL0B_TermQ = typename TileCopyTermQ::LayoutTagL0B;
    using LayoutTagL1A_TermW = typename TileCopyTermW::LayoutTagL1A;
    using LayoutTagL1B_TermW = typename TileCopyTermW::LayoutTagL1B;
    using LayoutTagL0A_TermW = typename TileCopyTermW::LayoutTagL0A;
    using LayoutTagL0B_TermW = typename TileCopyTermW::LayoutTagL0B;

    using TileMmadDvState = Catlass::Gemm::Tile::TileMmadTla<ArchTag, DT, LayoutTagL1A_DvState>;
    using TileMmadTermQ = Catlass::Gemm::Tile::TileMmadTla<ArchTag, DT, LayoutTagL1A_TermQ>;
    using TileMmadTermW = Catlass::Gemm::Tile::TileMmadTla<ArchTag, DT, LayoutTagL1A_TermW>;

    template <typename Tensor>
    using CopyGmToL1A_DvState = typename TileCopyDvState::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_DvState = typename TileCopyDvState::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_DvState = typename TileCopyDvState::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyL0CToUB_DvState = typename TileCopyDvStateToUB::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyL0CToUB_TermW = typename TileCopyTermWToUB::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_TermQ = typename TileCopyTermQ::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_TermQ = typename TileCopyTermQ::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_TermW = typename TileCopyTermW::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_TermW = typename TileCopyTermW::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_TermW = typename TileCopyTermW::template CopyL0CToDst<Tensor>;

    static constexpr uint32_t BUFFER_COUNT_2 = 2;
    static constexpr uint32_t K_DIM = 128;
    static constexpr uint32_t CHUNK_MAX = 128;
    static constexpr uint32_t L0_K_TILE = V_DIM == 256 ? 64 : K_DIM;

    static constexpr auto L1A_LAYOUT_K =
        tla::MakeLayout<DT, LayoutTagL1A_DvState>(tla::Int<CHUNK_MAX>{}, tla::Int<K_DIM>{});
    static constexpr auto L1B_LAYOUT_STATE =
        tla::MakeLayout<DT, LayoutTagL1B_DvState>(tla::Int<K_DIM>{}, tla::Int<V_DIM>{});
    static constexpr auto L1A_LAYOUT_QGT =
        tla::MakeLayout<DT, LayoutTagL1A_TermQ>(tla::Int<K_DIM>{}, tla::Int<CHUNK_MAX>{});
    static constexpr auto L1B_LAYOUT_DO =
        tla::MakeLayout<DT, LayoutTagL1B_TermQ>(tla::Int<CHUNK_MAX>{}, tla::Int<V_DIM>{});
    static constexpr auto L1A_LAYOUT_WT =
        tla::MakeLayout<DT, LayoutTagL1A_TermW>(tla::Int<K_DIM>{}, tla::Int<CHUNK_MAX>{});
    static constexpr auto L1B_LAYOUT_DV2 =
        tla::MakeLayout<DT, LayoutTagL1B_TermW>(tla::Int<CHUNK_MAX>{}, tla::Int<V_DIM>{});
    static constexpr auto UB_LAYOUT_DVSTATE_CV =
        tla::MakeLayout<DT, LayoutTagDvState>(tla::Int<CHUNK_MAX>{}, tla::Int<V_DIM>{});
    static constexpr auto UB_LAYOUT_TERMW_CV =
        tla::MakeLayout<DT, LayoutTagTermW>(tla::Int<K_DIM>{}, tla::Int<V_DIM>{});
    static constexpr uint32_t L1A_SCRATCH_BUFFER_COUNT = HEADS_PER_TASK;
    static constexpr uint32_t L1B_SCRATCH_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t L1A_SCRATCH_TILE_BYTES = CHUNK_MAX * K_DIM * sizeof(DT);
    static constexpr uint32_t L1B_STATE_TILE_BYTES = K_DIM * V_DIM * sizeof(DT);
    static constexpr uint32_t L1B_TOKEN_TILE_BYTES = CHUNK_MAX * V_DIM * sizeof(DT);
    static constexpr uint32_t L1B_SCRATCH_TILE_BYTES = L1B_STATE_TILE_BYTES > L1B_TOKEN_TILE_BYTES ?
                                                           L1B_STATE_TILE_BYTES :
                                                           L1B_TOKEN_TILE_BYTES;
    // Dhu AIC-S2：K/WT 共享聚合区（K(loop1)/WT(loop2) 相位交替复用，padded 步长 16384 elem
    //（=CHUNK_MAX×K_DIM）；4 矩阵容纳 headCnt≤HEADS_PER_TASK 全部档位，K-agg 的 GVA 回退档
    // 逐头 ndNum=1 亦落入本区，协议统一）。占用原 K_RESIDENT+W_RESIDENT 的 [0,128) KiB 窗口，
    // 其后各区偏移（L1A_SCRATCH @128K 与 AIV qg 落点、L1B_SCRATCH @256K、DO_AGG @320K）全部保持。
    static constexpr uint32_t KW_AGG_MATRIX_ELEMS = CHUNK_MAX * K_DIM;
    static constexpr uint32_t KW_AGG_TILE_BYTES = KW_AGG_MATRIX_ELEMS * sizeof(DT);
    static constexpr uint32_t KW_AGG_BUFFER_COUNT = HEADS_PER_TASK;
    static constexpr uint32_t KW_AGG_BYTES = KW_AGG_TILE_BYTES * KW_AGG_BUFFER_COUNT;
    static constexpr uint32_t KW_AGG_OFFSET = 0;
    static constexpr uint32_t L1A_SCRATCH_OFFSET = KW_AGG_OFFSET + KW_AGG_BYTES;
    static constexpr uint32_t L1B_SCRATCH_OFFSET =
        L1A_SCRATCH_OFFSET + L1A_SCRATCH_TILE_BYTES * L1A_SCRATCH_BUFFER_COUNT;
    // Dhu AIC-S1/S2：dO 装载 ndNum 头聚合 A/B（R-Akk LoadGroupA 同构，
    // chunk_kda_bwd_recompute_cube.h:263-273 先例）。聚合集 padded 步长 16384 elem（=CHUNK_MAX×V_DIM，
    // 与现 32KiB L1B 槽的 NZ 格式同构）；S2 升级为 ping-pong 双缓冲（2 buf × DO_AGG_HEADS 矩阵）。
    // 仅 bf16-V128 启用（预算与收益口径均为该档，其余路径逐拍保持现状）。
    static constexpr bool DO_AGG_ENABLED = std::is_same<DT, bfloat16_t>::value && V_DIM == 128;
    static constexpr uint32_t DO_AGG_MATRIX_ELEMS = CHUNK_MAX * V_DIM;
    static constexpr uint32_t DO_AGG_TILE_BYTES = DO_AGG_MATRIX_ELEMS * sizeof(DT);
    static constexpr uint32_t DO_AGG_HEADS = 3;
    static constexpr uint32_t DO_AGG_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t DO_AGG_BUF_BYTES = DO_AGG_TILE_BYTES * DO_AGG_HEADS;
    static constexpr uint32_t DO_AGG_BYTES = DO_AGG_ENABLED ? DO_AGG_BUF_BYTES * DO_AGG_BUFFER_COUNT : 0;
    static constexpr uint32_t DO_AGG_OFFSET =
        L1B_SCRATCH_OFFSET + L1B_SCRATCH_TILE_BYTES * L1B_SCRATCH_BUFFER_COUNT;
    static constexpr uint32_t L1_TOTAL_BYTES = 512 * 1024;
    static constexpr uint32_t L1_USED_BYTES = DO_AGG_OFFSET + DO_AGG_BYTES;
    static_assert(L1_USED_BYTES <= L1_TOTAL_BYTES, "chunk_gated_delta_rule_bwd_dhu cube L1 usage exceeds 512KB.");

    static constexpr uint32_t L0_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t L0A_TILE_BYTES = CHUNK_MAX * L0_K_TILE * sizeof(DT);
    static constexpr uint32_t L0B_TILE_BYTES = L0_K_TILE * V_DIM * sizeof(DT);
    static constexpr uint32_t L0C_MAX_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t L0C_TILE_BYTES = K_DIM * V_DIM * sizeof(ElementAccumulator);
    static constexpr bool ENABLE_L0C_DOUBLE_BUFFER = L0C_TILE_BYTES * L0C_MAX_BUFFER_COUNT <= ArchTag::L0C_SIZE;
    static constexpr uint32_t L0C_BUFFER_COUNT = ENABLE_L0C_DOUBLE_BUFFER ? L0C_MAX_BUFFER_COUNT : 1;
    static_assert(L0C_TILE_BYTES * L0C_BUFFER_COUNT <= ArchTag::L0C_SIZE,
                  "chunk_gated_delta_rule_bwd_dhu cube L0C usage exceeds arch L0C size.");

    static constexpr int32_t EVENT_L1B_SCRATCH_PING = 2;
    static constexpr int32_t EVENT_L1B_SCRATCH_PONG = 3;
    // S2：K/WT 共享聚合区信用（双向单 ID：MTE1_MTE2=区空闲、MTE2_MTE1=数据就绪）。
    // 信用配对（每 chunk 两相）：K 相 head0 wait free + 末头 dvState 末 K set free；
    // WT 相 head0 wait free + 末头 termW 末 K set free——两相严格交替，深度恒 ≤1。
    static constexpr int32_t EVENT_KW_AGG = 4;
    // S2：dO 聚合 ping-pong 双缓冲事件对（buf=chunkIdx&1，每 buf 一个双向 ID，语义同上）。
    // 信用配对（每 buf 每 chunk）：装载 wait free 1 / 末头 termQ GEMM 末 K set free 1；
    // 装载 set ready 1 / head0 termQ GEMM 首 K wait ready 1。
    static constexpr int32_t EVENT_DO_AGG_PING = 0;
    static constexpr int32_t EVENT_DO_AGG_PONG = 1;
    // {5,6,7}：S2 释放的 K_RESIDENT/W_RESIDENT 旧 ID，留作 S3（dh L1 交接）储备。
    static constexpr int32_t EVENT_L0A_PING = 0;
    static constexpr int32_t EVENT_L0B_PING = 1;
    static constexpr int32_t EVENT_L0A_PONG = 2;
    static constexpr int32_t EVENT_L0B_PONG = 3;
    static constexpr int32_t EVENT_L0_READY_PING = 0;
    static constexpr int32_t EVENT_L0_READY_PONG = 1;
    static constexpr int32_t EVENT_L0C_PING = 0;
    static constexpr int32_t EVENT_L0C_PONG = 1;

    __aicore__ inline int64_t WorkspaceBase(int64_t coreIdx, int64_t workspaceSlot) const
    {
        return (coreIdx * WORKSPACE_BUFFER_COUNT + workspaceSlot) * workspaceElemsPerSubBlock_;
    }

    __aicore__ inline int32_t L1BScratchEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_L1B_SCRATCH_PING : EVENT_L1B_SCRATCH_PONG;
    }

    __aicore__ inline int32_t L0AEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
    }

    __aicore__ inline int32_t L0BEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
    }

    __aicore__ inline int32_t L0ReadyEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
    }

    __aicore__ inline int32_t L0CEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_L0C_PING : EVENT_L0C_PONG;
    }

    __aicore__ inline void SwitchL0C()
    {
        if constexpr (L0C_BUFFER_COUNT > 1) {
            curL0C_ ^= 1U;
        }
    }

    __aicore__ inline void InitPipeFlags()
    {
        if constexpr (DO_AGG_ENABLED) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO_AGG_PING);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO_AGG_PONG);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_SCRATCH_PING);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_SCRATCH_PONG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_KW_AGG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PING);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PING);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PONG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PONG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PING);
        if constexpr (L0C_BUFFER_COUNT > 1) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PONG);
        }
    }

    __aicore__ inline void DrainPipeFlags()
    {
        if constexpr (DO_AGG_ENABLED) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO_AGG_PING);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO_AGG_PONG);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_SCRATCH_PING);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_SCRATCH_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_KW_AGG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PING);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PING);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PING);
        if constexpr (L0C_BUFFER_COUNT > 1) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PONG);
        }
        for (uint32_t cvIdx = 0; cvIdx < CV_BUFFER_COUNT; ++cvIdx) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN + cvIdx);
            AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(
                MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN + CV_SUBBLOCK_FLAG_STRIDE + cvIdx);
        }
    }

    template <typename LayoutTagL0A, typename LayoutTagL0B, typename CopyL1ToL0A, typename CopyL1ToL0B,
              typename TileMmad, typename CopyL0CToGm, typename TensorL1A, typename TensorL1B, typename TensorC>
    __aicore__ inline void RunResidentMmad(CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B,
                                           TileMmad &tileMmad, CopyL0CToGm &copyL0CToGm,
                                           TensorL1A &tensorL1A, TensorL1B &tensorL1B, TensorC &tensorBlockC,
                                           AscendC::LocalTensor<DT> (&l0A)[L0_BUFFER_COUNT],
                                           AscendC::LocalTensor<DT> (&l0B)[L0_BUFFER_COUNT],
                                           AscendC::LocalTensor<ElementAccumulator> (&l0C)[L0C_BUFFER_COUNT],
                                           bool waitL1AReady, bool releaseL1AAfterUse, int32_t l1AEvent,
                                           bool waitL1BReady, bool releaseL1BAfterUse, int32_t l1BEvent,
                                           uint32_t m, uint32_t n, uint32_t k)
    {
        uint32_t mActual = m;
        if (mActual == 1) {
            mActual = 16;
        }

        const uint32_t l0CSlot = curL0C_;
        const int32_t l0CEvent = L0CEvent(l0CSlot);
        auto layoutL0C = tla::MakeLayoutL0C(mActual, n);
        auto tensorL0C = tla::MakeTensor(l0C[l0CSlot], layoutL0C, Catlass::Arch::PositionL0C{});
        auto tensorTileL0C = tla::GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, n));

        for (uint32_t kOffset = 0; kOffset < k; kOffset += L0_K_TILE) {
            const uint32_t curK = kOffset + L0_K_TILE > k ? k - kOffset : L0_K_TILE;
            const bool firstK = kOffset == 0;
            const bool lastK = kOffset + curK >= k;
            const uint32_t l0Slot = curL0_;
            const int32_t l0AEvent = L0AEvent(l0Slot);
            const int32_t l0BEvent = L0BEvent(l0Slot);
            const int32_t l0ReadyEvent = L0ReadyEvent(l0Slot);

            auto layoutL0A = tla::MakeLayout<DT, LayoutTagL0A>(mActual, curK);
            auto tensorL0A = tla::MakeTensor(l0A[l0Slot], layoutL0A, Catlass::Arch::PositionL0A{});
            auto tensorTileL1A = tla::GetTile(tensorL1A, tla::MakeCoord(0, kOffset),
                                              tla::MakeShape(mActual, curK));
            if (waitL1AReady) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvent);
                waitL1AReady = false;
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
            copyL1ToL0A(tensorL0A, tensorTileL1A);
            if (lastK && releaseL1AAfterUse) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent);
            }

            auto layoutL0B = tla::MakeLayout<DT, LayoutTagL0B>(curK, n);
            auto tensorL0B = tla::MakeTensor(l0B[l0Slot], layoutL0B, Catlass::Arch::PositionL0B{});
            auto tensorTileL1B = tla::GetTile(tensorL1B, tla::MakeCoord(kOffset, 0),
                                              tla::MakeShape(curK, n));
            if (waitL1BReady) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvent);
                waitL1BReady = false;
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
            copyL1ToL0B(tensorL0B, tensorTileL1B);
            if (lastK && releaseL1BAfterUse) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
            curL0_ ^= 1U;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
            if (firstK) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEvent);
            }
            const uint8_t mmadUnitFlag = lastK ? 0b11 : 0b10;
            tileMmad(tensorTileL0C, tensorL0A, tensorL0B, firstK, mmadUnitFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
            if (lastK) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEvent);
            }
        }

        SwitchL0C();
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEvent);
        copyL0CToGm(tensorBlockC, tensorL0C, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent);
    }

    GM_ADDR k_ = nullptr;
    GM_ADDR w_ = nullptr;
    GM_ADDR dO_ = nullptr;
    GM_ADDR dh_ = nullptr;
    GM_ADDR dv2_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    Catlass::Arch::CrossCoreFlag vecToCubeFlag_{VEC_TO_CUBE_FLAG_READY};
    Catlass::Arch::CrossCoreFlag vecToCubeDhFlag_{VEC_TO_CUBE_DH_FLAG_READY};
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CUBE_TO_VEC_FLAG_READY};
    const ChunkGatedDeltaRuleBwdDhuTilingData *tiling_ = nullptr;
    int64_t B_ = 0;
    int64_t HK_ = 0;
    int64_t HV_ = 0;
    int64_t T_ = 0;
    int64_t K_ = 0;
    int64_t V_ = 0;
    int64_t HRatio_ = 0;
    bool hasGk_ = false;
    int64_t chunkSize_ = 0;
    int64_t vecRow_ = 8;
    int64_t totalChunkNum_ = 0;
    int64_t headsPerTask_ = 0;
    int64_t headWindowNum_ = 0;
    int64_t taskNum_ = 0;
    int64_t workspaceElemsPerSubBlock_ = 0;
    int64_t dvStateWorkspaceOffset_ = 0;
    int64_t termQWorkspaceOffset_ = 0;
    int64_t termWWorkspaceOffset_ = 0;
    uint32_t curL1B_ = 0;
    uint32_t curL0_ = 0;
    uint32_t curL0C_ = 0;
    uint32_t nextKResidentSlot_ = 0;
    bool cachedKResidentValid_ = false;
    int64_t cachedKResidentBase_ = 0;
    uint32_t cachedKResidentSlot_ = 0;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_CUBE_H
