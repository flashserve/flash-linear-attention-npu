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
 * \brief A2/A3 cube side process for chunk_gated_delta_rule_bwd_dhu.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_CUBE_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_CUBE_H

#include "chunk_gated_delta_rule_bwd_dhu_common.h"
#include "chunk_gated_delta_rule_bwd_dhu_struct.h"
#define CATLASS_ARCH 2201
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
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
        chunkSize_ = tiling_->chunkSize;
        totalChunkNum_ = tiling_->totalChunkNum;
        headsPerTask_ = tiling_->headsPerTask;
        headWindowNum_ = tiling_->headWindowNum;
        taskNum_ = tiling_->taskNum;
        workspaceElemsPerSubBlock_ = tiling_->workspaceElemsPerSubBlock;
        qgWorkspaceOffset_ = tiling_->qgWorkspaceOffset;
        dvStateWorkspaceOffset_ = tiling_->dvStateWorkspaceOffset;
        termQWorkspaceOffset_ = tiling_->termQWorkspaceOffset;
        termWWorkspaceOffset_ = tiling_->termWWorkspaceOffset;

        curL1A_ = 0;
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
        AscendC::LocalTensor<DT> kResident[K_RESIDENT_BUFFER_COUNT] = {
            resource.l1Buf.template GetBufferByByte<DT>(K_RESIDENT_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(K_RESIDENT_OFFSET + K_RESIDENT_TILE_BYTES)};
        AscendC::LocalTensor<DT> wResident[W_RESIDENT_BUFFER_COUNT] = {
            resource.l1Buf.template GetBufferByByte<DT>(W_RESIDENT_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(W_RESIDENT_OFFSET + W_RESIDENT_TILE_BYTES)};
        AscendC::LocalTensor<DT> l1AScratch[L1A_SCRATCH_BUFFER_COUNT] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1A_SCRATCH_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1A_SCRATCH_OFFSET + L1A_SCRATCH_TILE_BYTES)};
        AscendC::LocalTensor<DT> l1BScratch[L1B_SCRATCH_BUFFER_COUNT] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1B_SCRATCH_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1B_SCRATCH_OFFSET + L1B_SCRATCH_TILE_BYTES)};
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
                    const bool nextHeadUsesSameK =
                        headOffset + 1 < headCnt && (hv / HRatio_) == ((hv + 1) / HRatio_);
                    const bool releaseKAfterUse = !nextHeadUsesSameK;
                    const int64_t hq = hv / HRatio_;
                    const int64_t kBase = ((chunkInfo.bIdx * HK_ + hq) * T_ + chunkInfo.tokenStart) * K_;
                    const int64_t dOBase = ((chunkInfo.bIdx * HV_ + hv) * T_ + chunkInfo.tokenStart) * V_;
                    const int64_t dhBase =
                        ((chunkInfo.bIdx * HV_ + hv) * totalChunkNum_ + chunkInfo.outputChunkIdx) * K_ * V_;
                    const int64_t slotBase = WorkspaceBase(blockIdx, workspaceSlot);

                    LayoutTagK tagK = LayoutTagK::MakeLayout<DT>(chunkSize_, K_);
                    LayoutTagState tagState = LayoutTagState::MakeLayout<DT>(K_, V_DIM);
                    LayoutTagDvState tagDvState = LayoutTagDvState::MakeLayout<DT>(chunkSize_, V_DIM);
                    LayoutTagQGT tagQGT = LayoutTagQGT::MakeLayout<DT>(K_, chunkSize_);
                    LayoutTagDO tagDO = LayoutTagDO::MakeLayout<DT>(chunkSize_, V_DIM);
                    LayoutTagTermQ tagTermQ = LayoutTagTermQ::MakeLayout<DT>(K_, V_DIM);

                    auto layoutK = tla::MakeLayoutFromTag(tagK);
                    auto layoutState = tla::MakeLayoutFromTag(tagState);
                    auto layoutDvState = tla::MakeLayoutFromTag(tagDvState);
                    auto layoutQGT = tla::MakeLayoutFromTag(tagQGT);
                    auto layoutDO = tla::MakeLayoutFromTag(tagDO);
                    auto layoutTermQ = tla::MakeLayoutFromTag(tagTermQ);

                    AscendC::GlobalTensor<DT> gmK;
                    AscendC::GlobalTensor<DT> gmState;
                    AscendC::GlobalTensor<DT> gmDvState;
                    AscendC::GlobalTensor<DT> gmQGT;
                    AscendC::GlobalTensor<DT> gmDO;
                    AscendC::GlobalTensor<DT> gmTermQ;
                    gmK.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(k_) + kBase);
                    gmState.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh_) + dhBase);
                    gmDvState.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace_) + slotBase +
                                              dvStateWorkspaceOffset_);
                    gmQGT.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace_) + slotBase + qgWorkspaceOffset_);
                    gmDO.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dO_) + dOBase);
                    gmTermQ.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace_) + slotBase +
                                            termQWorkspaceOffset_);

                    auto tensorK = tla::MakeTensor(gmK, layoutK, Catlass::Arch::PositionGM{});
                    const bool needLoadKResident = !cachedKResidentValid_ || cachedKResidentBase_ != kBase;
                    if (needLoadKResident) {
                        if (cachedKResidentValid_) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KResidentEvent(cachedKResidentSlot_));
                            cachedKResidentValid_ = false;
                        }
                        cachedKResidentBase_ = kBase;
                        cachedKResidentSlot_ = nextKResidentSlot_;
                        cachedKResidentValid_ = true;
                        nextKResidentSlot_ ^= 1U;
                    }
                    const uint32_t kResidentSlot = cachedKResidentSlot_;
                    const int32_t kResidentEvent = KResidentEvent(kResidentSlot);
                    auto tensorL1K =
                        tla::MakeTensor(kResident[kResidentSlot], L1A_LAYOUT_K, Catlass::Arch::PositionL1{});
                    auto blockK = tla::GetTile(
                        tensorK, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(K_)));
                    CopyGmToL1A_DvState<decltype(blockK)> copyGmToL1A_K;
                    if (needLoadKResident) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kResidentEvent);
                        copyGmToL1A_K(tensorL1K, blockK);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kResidentEvent);
                    }

                    auto tensorDO = tla::MakeTensor(gmDO, layoutDO, Catlass::Arch::PositionGM{});
                    auto blockDO = tla::GetTile(
                        tensorDO, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(V_DIM)));
                    CopyGmToL1B_TermQ<decltype(blockDO)> copyGmToL1B_DO;

                    const uint32_t doScratchSlot = curL1B_;
                    curL1B_ ^= 1U;
                    const int32_t doScratchEvent = L1BScratchEvent(doScratchSlot);
                    auto tensorL1DO =
                        tla::MakeTensor(l1BScratch[doScratchSlot], L1B_LAYOUT_DO, Catlass::Arch::PositionL1{});
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(doScratchEvent);
                    copyGmToL1B_DO(tensorL1DO, blockDO);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(doScratchEvent);

                    Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);

                    auto tensorState = tla::MakeTensor(gmState, layoutState, Catlass::Arch::PositionGM{});
                    auto tensorDvState = tla::MakeTensor(gmDvState, layoutDvState, Catlass::Arch::PositionGM{});
                    auto blockState = tla::GetTile(
                        tensorState, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM)));
                    auto blockDvState = tla::GetTile(
                        tensorDvState, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(V_DIM)));
                    CopyGmToL1B_DvState<decltype(blockState)> copyGmToL1B_State;
                    CopyL0CToGm_DvState<decltype(blockDvState)> copyL0CToGm_DvState;
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
                    RunResidentMmad<LayoutTagL0A_DvState, LayoutTagL0B_DvState>(
                        copyL1ToL0A_DvState, copyL1ToL0B_DvState, tileMmadDvState, copyL0CToGm_DvState,
                        tensorL1K, tensorL1State, blockDvState, l0A, l0B, l0C,
                        needLoadKResident, releaseKAfterUse, kResidentEvent, true, true, stateScratchEvent,
                        static_cast<uint32_t>(chunkInfo.chunkLen), static_cast<uint32_t>(V_DIM),
                        static_cast<uint32_t>(K_));
                    if (releaseKAfterUse) {
                        cachedKResidentValid_ = false;
                    }

                    auto tensorQGT = tla::MakeTensor(gmQGT, layoutQGT, Catlass::Arch::PositionGM{});
                    auto tensorTermQ = tla::MakeTensor(gmTermQ, layoutTermQ, Catlass::Arch::PositionGM{});
                    auto blockQGT = tla::GetTile(
                        tensorQGT, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(K_), static_cast<uint32_t>(chunkInfo.chunkLen)));
                    auto blockTermQ = tla::GetTile(
                        tensorTermQ, tla::MakeCoord(0, 0),
                        tla::MakeShape(static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM)));
                    CopyGmToL1A_TermQ<decltype(blockQGT)> copyGmToL1A_QGT;
                    CopyL0CToGm_TermQ<decltype(blockTermQ)> copyL0CToGm_TermQ;
                    CopyL1ToL0A_TermQ copyL1ToL0A_TermQ;
                    CopyL1ToL0B_TermQ copyL1ToL0B_TermQ;
                    TileMmadTermQ tileMmadTermQ;

                    const uint32_t qgScratchSlot = curL1A_;
                    curL1A_ ^= 1U;
                    const int32_t qgScratchEvent = L1AScratchEvent(qgScratchSlot);
                    auto tensorL1QGT =
                        tla::MakeTensor(l1AScratch[qgScratchSlot], L1A_LAYOUT_QGT, Catlass::Arch::PositionL1{});
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qgScratchEvent);
                    copyGmToL1A_QGT(tensorL1QGT, blockQGT);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(qgScratchEvent);
                    RunResidentMmad<LayoutTagL0A_TermQ, LayoutTagL0B_TermQ>(
                        copyL1ToL0A_TermQ, copyL1ToL0B_TermQ, tileMmadTermQ, copyL0CToGm_TermQ,
                        tensorL1QGT, tensorL1DO, blockTermQ, l0A, l0B, l0C,
                        true, true, qgScratchEvent, true, true, doScratchEvent,
                        static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM),
                        static_cast<uint32_t>(chunkInfo.chunkLen));

                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                }
                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    const int64_t hv = hvBase + headOffset;
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    const int64_t wBase = ((chunkInfo.bIdx * HV_ + hv) * T_ + chunkInfo.tokenStart) * K_;
                    const int64_t dv2Base = ((chunkInfo.bIdx * HV_ + hv) * T_ + chunkInfo.tokenStart) * V_;
                    const int64_t slotBase = WorkspaceBase(blockIdx, workspaceSlot);
                    const uint32_t residentSlot = static_cast<uint32_t>(workspaceSlot) & 1U;

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
                    CopyL0CToGm_TermW<decltype(blockTermW)> copyL0CToGm_TermW;
                    CopyL1ToL0A_TermW copyL1ToL0A_TermW;
                    CopyL1ToL0B_TermW copyL1ToL0B_TermW;
                    TileMmadTermW tileMmadTermW;

                    const int32_t wEvent = WResidentEvent(residentSlot);
                    auto tensorL1WT =
                        tla::MakeTensor(wResident[residentSlot], L1A_LAYOUT_WT, Catlass::Arch::PositionL1{});
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(wEvent);
                    copyGmToL1A_WT(tensorL1WT, blockWT);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(wEvent);

                    Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);

                    const uint32_t dv2ScratchSlot = curL1B_;
                    curL1B_ ^= 1U;
                    const int32_t dv2ScratchEvent = L1BScratchEvent(dv2ScratchSlot);
                    auto tensorL1Dv2 =
                        tla::MakeTensor(l1BScratch[dv2ScratchSlot], L1B_LAYOUT_DV2, Catlass::Arch::PositionL1{});
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dv2ScratchEvent);
                    copyGmToL1B_Dv2(tensorL1Dv2, blockDv2);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dv2ScratchEvent);

                    RunResidentMmad<LayoutTagL0A_TermW, LayoutTagL0B_TermW>(
                        copyL1ToL0A_TermW, copyL1ToL0B_TermW, tileMmadTermW, copyL0CToGm_TermW,
                        tensorL1WT, tensorL1Dv2, blockTermW, l0A, l0B, l0C,
                        true, true, wEvent, true, true, dv2ScratchEvent,
                        static_cast<uint32_t>(K_), static_cast<uint32_t>(V_DIM),
                        static_cast<uint32_t>(chunkInfo.chunkLen));

                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                }
            }
        }

        DrainPipeFlags();
    }

private:
    using ArchTag = Catlass::Arch::AtlasA2;
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
    using CopyL0CToGm_DvState = typename TileCopyDvState::template CopyL0CToGm<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_TermQ = typename TileCopyTermQ::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_TermQ = typename TileCopyTermQ::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_TermQ = typename TileCopyTermQ::template CopyL0CToGm<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_TermW = typename TileCopyTermW::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_TermW = typename TileCopyTermW::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_TermW = typename TileCopyTermW::template CopyL0CToGm<Tensor>;

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

    static constexpr uint32_t K_RESIDENT_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t W_RESIDENT_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t L1A_SCRATCH_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t L1B_SCRATCH_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t K_RESIDENT_TILE_BYTES = CHUNK_MAX * K_DIM * sizeof(DT);
    static constexpr uint32_t W_RESIDENT_TILE_BYTES = CHUNK_MAX * K_DIM * sizeof(DT);
    static constexpr uint32_t L1A_SCRATCH_TILE_BYTES = CHUNK_MAX * K_DIM * sizeof(DT);
    static constexpr uint32_t L1B_STATE_TILE_BYTES = K_DIM * V_DIM * sizeof(DT);
    static constexpr uint32_t L1B_TOKEN_TILE_BYTES = CHUNK_MAX * V_DIM * sizeof(DT);
    static constexpr uint32_t L1B_SCRATCH_TILE_BYTES = L1B_STATE_TILE_BYTES > L1B_TOKEN_TILE_BYTES ?
                                                           L1B_STATE_TILE_BYTES :
                                                           L1B_TOKEN_TILE_BYTES;
    static constexpr uint32_t K_RESIDENT_OFFSET = 0;
    static constexpr uint32_t W_RESIDENT_OFFSET = K_RESIDENT_OFFSET + K_RESIDENT_TILE_BYTES * K_RESIDENT_BUFFER_COUNT;
    static constexpr uint32_t L1A_SCRATCH_OFFSET = W_RESIDENT_OFFSET + W_RESIDENT_TILE_BYTES * W_RESIDENT_BUFFER_COUNT;
    static constexpr uint32_t L1B_SCRATCH_OFFSET =
        L1A_SCRATCH_OFFSET + L1A_SCRATCH_TILE_BYTES * L1A_SCRATCH_BUFFER_COUNT;
    static constexpr uint32_t L1_TOTAL_BYTES = 512 * 1024;
    static constexpr uint32_t L1_USED_BYTES =
        L1B_SCRATCH_OFFSET + L1B_SCRATCH_TILE_BYTES * L1B_SCRATCH_BUFFER_COUNT;
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

    static constexpr int32_t EVENT_L1A_SCRATCH_PING = 0;
    static constexpr int32_t EVENT_L1A_SCRATCH_PONG = 1;
    static constexpr int32_t EVENT_L1B_SCRATCH_PING = 2;
    static constexpr int32_t EVENT_L1B_SCRATCH_PONG = 3;
    static constexpr int32_t EVENT_K_RESIDENT_PING = 4;
    static constexpr int32_t EVENT_K_RESIDENT_PONG = 5;
    static constexpr int32_t EVENT_W_RESIDENT_PING = 6;
    static constexpr int32_t EVENT_W_RESIDENT_PONG = 7;
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

    __aicore__ inline int32_t L1AScratchEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_L1A_SCRATCH_PING : EVENT_L1A_SCRATCH_PONG;
    }

    __aicore__ inline int32_t L1BScratchEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_L1B_SCRATCH_PING : EVENT_L1B_SCRATCH_PONG;
    }

    __aicore__ inline int32_t KResidentEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_K_RESIDENT_PING : EVENT_K_RESIDENT_PONG;
    }

    __aicore__ inline int32_t WResidentEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_W_RESIDENT_PING : EVENT_W_RESIDENT_PONG;
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
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A_SCRATCH_PING);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A_SCRATCH_PONG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_SCRATCH_PING);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_SCRATCH_PONG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PING);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PONG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_W_RESIDENT_PING);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_W_RESIDENT_PONG);
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
        if (cachedKResidentValid_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KResidentEvent(cachedKResidentSlot_));
            cachedKResidentValid_ = false;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A_SCRATCH_PING);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A_SCRATCH_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_SCRATCH_PING);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_SCRATCH_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PING);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_W_RESIDENT_PING);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_W_RESIDENT_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PING);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PING);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PING);
        if constexpr (L0C_BUFFER_COUNT > 1) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PONG);
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
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CUBE_TO_VEC_FLAG_READY};
    const ChunkGatedDeltaRuleBwdDhuTilingData *tiling_ = nullptr;
    int64_t B_ = 0;
    int64_t HK_ = 0;
    int64_t HV_ = 0;
    int64_t T_ = 0;
    int64_t K_ = 0;
    int64_t V_ = 0;
    int64_t HRatio_ = 0;
    int64_t chunkSize_ = 0;
    int64_t totalChunkNum_ = 0;
    int64_t headsPerTask_ = 0;
    int64_t headWindowNum_ = 0;
    int64_t taskNum_ = 0;
    int64_t workspaceElemsPerSubBlock_ = 0;
    int64_t qgWorkspaceOffset_ = 0;
    int64_t dvStateWorkspaceOffset_ = 0;
    int64_t termQWorkspaceOffset_ = 0;
    int64_t termWWorkspaceOffset_ = 0;
    uint32_t curL1A_ = 0;
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
