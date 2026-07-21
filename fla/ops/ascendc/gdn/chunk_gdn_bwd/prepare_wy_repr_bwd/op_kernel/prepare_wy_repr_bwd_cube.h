/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_cube.h
 * \brief Cube side process for fused prepare_wy_repr_bwd A2/A3.
 */

#ifndef PREPARE_WY_REPR_BWD_CUBE_H
#define PREPARE_WY_REPR_BWD_CUBE_H

#define CATLASS_ARCH 2201
#include "prepare_wy_repr_bwd_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace tla;

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
class PrepareWyReprBwdCubeProcess {
public:
    __aicore__ inline PrepareWyReprBwdCubeProcess(GM_ADDR k, GM_ADDR A, GM_ADDR dw, GM_ADDR du, GM_ADDR cuSeqlens,
                                                  GM_ADDR chunkIndices, GM_ADDR workspace, GM_ADDR debugDkbg,
                                                  GM_ADDR debugDvb, GM_ADDR debugKkt);
    __aicore__ inline void Init(const GDN::PrepareWyReprBwdTilingData &tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitPipeFlags();
    __aicore__ inline void ProcessImpl();

private:
    using ArchTag = Arch::AtlasA2;
    using LayoutTagAT = layout::ColumnMajor;
    using LayoutTagDw = layout::RowMajor;
    using LayoutTagDu = layout::RowMajor;
    using LayoutTagK = layout::RowMajor;
    using LayoutTagKT = layout::ColumnMajor;
    using LayoutTagDkbg = layout::RowMajor;
    using LayoutTagDvb = layout::RowMajor;
    using LayoutTagKkt = layout::RowMajor;

    using TileCopyDkbg =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDw, kType, LayoutTagDkbg>;
    using TileCopyDvb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDu, kType, LayoutTagDvb>;
    using TileCopyKkt =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagK, kType, LayoutTagKT, kType, LayoutTagKkt>;

    using ElementAccumulator = typename TileCopyDkbg::ElementAccumulator;
    using CopyL1ToL0A_Dkbg = typename TileCopyDkbg::CopyL1ToL0A;
    using CopyL1ToL0B_Dkbg = typename TileCopyDkbg::CopyL1ToL0B;
    using CopyL1ToL0A_Dvb = typename TileCopyDvb::CopyL1ToL0A;
    using CopyL1ToL0B_Dvb = typename TileCopyDvb::CopyL1ToL0B;
    using CopyL1ToL0A_Kkt = typename TileCopyKkt::CopyL1ToL0A;
    using CopyL1ToL0B_Kkt = typename TileCopyKkt::CopyL1ToL0B;

    using LayoutTagL1A_Dkbg = typename TileCopyDkbg::LayoutTagL1A;
    using LayoutTagL1B_Dkbg = typename TileCopyDkbg::LayoutTagL1B;
    using LayoutTagL0A_Dkbg = typename TileCopyDkbg::LayoutTagL0A;
    using LayoutTagL0B_Dkbg = typename TileCopyDkbg::LayoutTagL0B;
    using LayoutTagL1A_Dvb = typename TileCopyDvb::LayoutTagL1A;
    using LayoutTagL1B_Dvb = typename TileCopyDvb::LayoutTagL1B;
    using LayoutTagL0A_Dvb = typename TileCopyDvb::LayoutTagL0A;
    using LayoutTagL0B_Dvb = typename TileCopyDvb::LayoutTagL0B;
    using LayoutTagL1A_Kkt = typename TileCopyKkt::LayoutTagL1A;
    using LayoutTagL1B_Kkt = typename TileCopyKkt::LayoutTagL1B;
    using LayoutTagL0A_Kkt = typename TileCopyKkt::LayoutTagL0A;
    using LayoutTagL0B_Kkt = typename TileCopyKkt::LayoutTagL0B;

    using TileMmadDkbg = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Dkbg>;
    using TileMmadDvb = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Dvb>;
    using TileMmadKkt = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Kkt>;

    template <typename Tensor>
    using CopyGmToL1A_Dkbg = typename TileCopyDkbg::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_Dkbg = typename TileCopyDkbg::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_Dkbg = typename TileCopyDkbg::template CopyL0CToGm<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_Dvb = typename TileCopyDvb::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_Dvb = typename TileCopyDvb::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_Dvb = typename TileCopyDvb::template CopyL0CToGm<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_Kkt = typename TileCopyKkt::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_Kkt = typename TileCopyKkt::template CopyL0CToGm<Tensor>;

    static constexpr auto L1A_LAYOUT_AT =
        tla::MakeLayout<kType, LayoutTagL1A_Dkbg>(tla::Int<CHUNK_SIZE>{}, tla::Int<CHUNK_SIZE>{});
    static constexpr auto L1B_LAYOUT_DW =
        tla::MakeLayout<kType, LayoutTagL1B_Dkbg>(tla::Int<CHUNK_SIZE>{}, tla::Int<K_DIM>{});
    static constexpr auto L1A_LAYOUT_AT_FOR_DU =
        tla::MakeLayout<kType, LayoutTagL1A_Dvb>(tla::Int<CHUNK_SIZE>{}, tla::Int<CHUNK_SIZE>{});
    static constexpr auto L1B_LAYOUT_DU =
        tla::MakeLayout<kType, LayoutTagL1B_Dvb>(tla::Int<CHUNK_SIZE>{}, tla::Int<V_DIM>{});
    static constexpr auto L1A_LAYOUT_K =
        tla::MakeLayout<kType, LayoutTagL1A_Kkt>(tla::Int<CHUNK_SIZE>{}, tla::Int<K_DIM>{});
    static constexpr auto L1B_LAYOUT_KT =
        tla::MakeLayout<kType, LayoutTagL1B_Kkt>(tla::Int<K_DIM>{}, tla::Int<CHUNK_SIZE>{});

    static constexpr uint32_t L1A_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t L1B_DU_TILE_BYTES = CHUNK_SIZE * V_DIM * sizeof(kType);
    static constexpr uint32_t L1B_K_TILE_BYTES = K_DIM * CHUNK_SIZE * sizeof(kType);
    static constexpr uint32_t L1B_TILE_BYTES =
        L1B_DU_TILE_BYTES > L1B_K_TILE_BYTES ? L1B_DU_TILE_BYTES : L1B_K_TILE_BYTES;
    static constexpr uint32_t L1_SLOT_BYTES = L1A_TILE_BYTES + L1B_TILE_BYTES;
    static constexpr uint32_t L1_BUFFER_COUNT = 2;
    static constexpr uint32_t K_RESIDENT_BUFFER_COUNT = 2;
    static constexpr uint32_t K_RESIDENT_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t K_RESIDENT_OFFSET = L1_SLOT_BYTES * L1_BUFFER_COUNT;
    static constexpr uint32_t L0_DVB_K_TILE = V_DIM == 256 ? 64 : CHUNK_SIZE;
    static constexpr uint32_t L0_DVB_N_TILE = V_DIM;
    static constexpr uint32_t L0A_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t L0B_DVB_TILE_BYTES = L0_DVB_K_TILE * L0_DVB_N_TILE * sizeof(kType);
    static constexpr uint32_t L0B_K_TILE_BYTES = K_DIM * CHUNK_SIZE * sizeof(kType);
    static constexpr uint32_t L0B_TILE_BYTES =
        L0B_DVB_TILE_BYTES > L0B_K_TILE_BYTES ? L0B_DVB_TILE_BYTES : L0B_K_TILE_BYTES;
    static constexpr uint32_t L0_BUFFER_COUNT = 2;
    static constexpr int32_t EVENT_L1A_PING = 0;
    static constexpr int32_t EVENT_L1B_PING = 1;
    static constexpr int32_t EVENT_L1A_PONG = 2;
    static constexpr int32_t EVENT_L1B_PONG = 3;
    static constexpr int32_t EVENT_K_RESIDENT_PING = 4;
    static constexpr int32_t EVENT_K_RESIDENT_PONG = 5;
    static constexpr int32_t EVENT_L0A_PING = 0;
    static constexpr int32_t EVENT_L0B_PING = 1;
    static constexpr int32_t EVENT_L0A_PONG = 2;
    static constexpr int32_t EVENT_L0B_PONG = 3;
    static constexpr int32_t EVENT_L0_READY_PING = 0;
    static constexpr int32_t EVENT_L0_READY_PONG = 1;
    static constexpr int32_t EVENT_L0C = 0;

    GM_ADDR k_ = nullptr;
    GM_ADDR A_ = nullptr;
    GM_ADDR dw_ = nullptr;
    GM_ADDR du_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    GM_ADDR debugDkbg_ = nullptr;
    GM_ADDR debugDvb_ = nullptr;
    GM_ADDR debugKkt_ = nullptr;
    GDN::PrepareWyReprBwdTilingData tiling_{};
    uint32_t curSlot_ = 0;
    uint32_t nextKResidentSlot_ = 0;
    uint32_t cachedKResidentSlot_ = 0;
    uint64_t cachedKResidentHk_ = static_cast<uint64_t>(-1);
    uint32_t nextKktSlot_ = 0;
    uint32_t cachedKktSlot_ = 0;
    uint64_t cachedKktHk_ = static_cast<uint64_t>(-1);
    uint32_t kktSlotForSlot_[PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT] = {0, 0};
    uint32_t curL1_ = 0;
    uint32_t curL0_ = 0;
    Arch::CrossCoreFlagWithReverse<> vecToCubeFlag_{PREPARE_WY_REPR_BWD_VEC_TO_CUBE_FLAG_READY,
                                                    PREPARE_WY_REPR_BWD_VEC_TO_CUBE_FLAG_REVERSE};
    Arch::CrossCoreFlagWithReverse<> cubeToVecFlag_{PREPARE_WY_REPR_BWD_CUBE_TO_VEC_FLAG_READY,
                                                    PREPARE_WY_REPR_BWD_CUBE_TO_VEC_FLAG_REVERSE};
};

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline PrepareWyReprBwdCubeProcess<kType, gType, V_DIM, CHUNK_SIZE>::PrepareWyReprBwdCubeProcess(
    GM_ADDR k, GM_ADDR A, GM_ADDR dw, GM_ADDR du, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
    GM_ADDR debugDkbg, GM_ADDR debugDvb, GM_ADDR debugKkt)
    : k_(k), A_(A), dw_(dw), du_(du), cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices), workspace_(workspace),
      debugDkbg_(debugDkbg), debugDvb_(debugDvb), debugKkt_(debugKkt)
{
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void
PrepareWyReprBwdCubeProcess<kType, gType, V_DIM, CHUNK_SIZE>::Init(const GDN::PrepareWyReprBwdTilingData &tiling)
{
    tiling_ = tiling;
    curSlot_ = 0;
    nextKResidentSlot_ = 0;
    cachedKResidentSlot_ = 0;
    cachedKResidentHk_ = static_cast<uint64_t>(-1);
    nextKktSlot_ = 0;
    cachedKktSlot_ = 0;
    cachedKktHk_ = static_cast<uint64_t>(-1);
    kktSlotForSlot_[0] = 0;
    kktSlotForSlot_[1] = 0;
    curL1_ = 0;
    curL0_ = 0;
    InitPipeFlags();
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdCubeProcess<kType, gType, V_DIM, CHUNK_SIZE>::InitPipeFlags()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A_PONG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_PONG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PONG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PING);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PING);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PONG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PONG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdCubeProcess<kType, gType, V_DIM, CHUNK_SIZE>::Process()
{
    ProcessImpl();
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdCubeProcess<kType, gType, V_DIM, CHUNK_SIZE>::ProcessImpl()
{
    LayoutTagAT tagAT = LayoutTagAT::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagDw tagDw = LayoutTagDw::MakeLayout<kType>(CHUNK_SIZE, K_DIM);
    LayoutTagDu tagDu = LayoutTagDu::MakeLayout<kType>(CHUNK_SIZE, V_DIM);
    LayoutTagK tagK = LayoutTagK::MakeLayout<kType>(CHUNK_SIZE, K_DIM);
    LayoutTagDkbg tagDkbg = LayoutTagDkbg::MakeLayout<kType>(CHUNK_SIZE, K_DIM);
    LayoutTagDvb tagDvb = LayoutTagDvb::MakeLayout<kType>(CHUNK_SIZE, V_DIM);
    LayoutTagKkt tagKkt = LayoutTagKkt::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);

    auto layoutAT = MakeLayoutFromTag(tagAT);
    auto layoutDw = MakeLayoutFromTag(tagDw);
    auto layoutDu = MakeLayoutFromTag(tagDu);
    auto layoutK = MakeLayoutFromTag(tagK);
    auto layoutDkbg = MakeLayoutFromTag(tagDkbg);
    auto layoutDvb = MakeLayoutFromTag(tagDvb);
    auto layoutKkt = MakeLayoutFromTag(tagKkt);

    Arch::Resource<ArchTag> resource;
    AscendC::LocalTensor<kType> l1A[L1_BUFFER_COUNT] = {resource.l1Buf.template GetBufferByByte<kType>(0),
                                                        resource.l1Buf.template GetBufferByByte<kType>(L1_SLOT_BYTES)};
    AscendC::LocalTensor<kType> l1B[L1_BUFFER_COUNT] = {
        resource.l1Buf.template GetBufferByByte<kType>(L1A_TILE_BYTES),
        resource.l1Buf.template GetBufferByByte<kType>(L1_SLOT_BYTES + L1A_TILE_BYTES)};
    AscendC::LocalTensor<kType> kResident[K_RESIDENT_BUFFER_COUNT] = {
        resource.l1Buf.template GetBufferByByte<kType>(K_RESIDENT_OFFSET),
        resource.l1Buf.template GetBufferByByte<kType>(K_RESIDENT_OFFSET + K_RESIDENT_TILE_BYTES)};
    AscendC::LocalTensor<kType> l0A[L0_BUFFER_COUNT] = {
        resource.l0ABuf.template GetBufferByByte<kType>(0),
        resource.l0ABuf.template GetBufferByByte<kType>(L0A_TILE_BYTES)};
    AscendC::LocalTensor<kType> l0B[L0_BUFFER_COUNT] = {
        resource.l0BBuf.template GetBufferByByte<kType>(0),
        resource.l0BBuf.template GetBufferByByte<kType>(L0B_TILE_BYTES)};
    auto l0C = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);

    CopyL1ToL0A_Dkbg copyL1ToL0A_Dkbg;
    CopyL1ToL0B_Dkbg copyL1ToL0B_Dkbg;
    CopyL1ToL0A_Dvb copyL1ToL0A_Dvb;
    CopyL1ToL0B_Dvb copyL1ToL0B_Dvb;
    CopyL1ToL0A_Kkt copyL1ToL0A_Kkt;
    CopyL1ToL0B_Kkt copyL1ToL0B_Kkt;
    TileMmadDkbg tileMmadDkbg;
    TileMmadDvb tileMmadDvb;
    TileMmadKkt tileMmadKkt;

    AscendC::GlobalTensor<kType> gmAT;
    AscendC::GlobalTensor<kType> gmDw;
    AscendC::GlobalTensor<kType> gmDu;
    AscendC::GlobalTensor<kType> gmK;
    AscendC::GlobalTensor<kType> gmDkbg;
    AscendC::GlobalTensor<kType> gmDvb;
    AscendC::GlobalTensor<kType> gmKkt;
    AscendC::GlobalTensor<kType> gmDebugDkbg;
    AscendC::GlobalTensor<kType> gmDebugDvb;
    AscendC::GlobalTensor<kType> gmDebugKkt;

    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreNum = AscendC::GetBlockNum();
    uint64_t groupSize = PrepareWyReprBwdGetGroupSize(tiling_);

    for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(tiling_.chunkNum); taskIdx += coreNum) {
        PrepareWyReprBwdTaskInfo task;
        PrepareWyReprBwdGetTaskInfo(cuSeqlens_, chunkIndices_, tiling_, taskIdx, task);
        GemmCoord shapeK{task.curChunkSize, K_DIM, task.curChunkSize};
        GemmCoord shapeV{task.curChunkSize, V_DIM, task.curChunkSize};
        GemmCoord shapeKkt{task.curChunkSize, task.curChunkSize, K_DIM};
        nextKResidentSlot_ = 0;
        cachedKResidentSlot_ = 0;
        cachedKResidentHk_ = static_cast<uint64_t>(-1);
        nextKktSlot_ = 0;
        cachedKktSlot_ = 0;
        cachedKktHk_ = static_cast<uint64_t>(-1);
        kktSlotForSlot_[0] = 0;
        kktSlotForSlot_[1] = 0;

        uint64_t hvTotal = static_cast<uint64_t>(tiling_.HV);
        for (uint64_t hvBase = 0; hvBase < hvTotal; hvBase += PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT) {
            uint32_t headCnt = hvBase + PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT <= hvTotal ?
                                   PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT :
                                   static_cast<uint32_t>(hvTotal - hvBase);
            uint32_t windowStartSlot = curSlot_;

            // Stage0 fills both workspace slots before later stages consume the first head.
            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                uint64_t hk = hv / groupSize;
                uint64_t valueBase = hv * tiling_.T + task.valueBos;
                uint64_t keyBase = hk * tiling_.T + task.keyBos;
                uint64_t debugLine = static_cast<uint64_t>(taskIdx) * static_cast<uint64_t>(tiling_.HV) + hv;
                uint64_t debugKktLine = static_cast<uint64_t>(taskIdx) * static_cast<uint64_t>(tiling_.HK) + hk;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
                bool needComputeKkt = cachedKktHk_ != hk;
                bool needLoadKResident = cachedKResidentHk_ != hk;
                if (needLoadKResident) {
                    cachedKResidentHk_ = hk;
                    cachedKResidentSlot_ = nextKResidentSlot_;
                    nextKResidentSlot_ ^= 1U;
                }
                if (needComputeKkt) {
                    cachedKktHk_ = hk;
                    cachedKktSlot_ = nextKktSlot_;
                    nextKktSlot_ ^= 1U;
                }
                uint32_t kktSlot = cachedKktSlot_;
                kktSlotForSlot_[curSlot_] = kktSlot;
                GM_ADDR kktBase = PrepareWyReprBwdGetKktBase(workspace_, coreIdx, kktSlot, tiling_);

                gmAT.SetGlobalBuffer((__gm__ kType *)A_ + valueBase * CHUNK_SIZE);
                gmDw.SetGlobalBuffer((__gm__ kType *)dw_ + valueBase * K_DIM);
                gmDu.SetGlobalBuffer((__gm__ kType *)du_ + valueBase * V_DIM);
                gmK.SetGlobalBuffer((__gm__ kType *)k_ + keyBase * K_DIM);
                gmDkbg.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dkbgOffset));
                gmDvb.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dvbOffset));
                gmKkt.SetGlobalBuffer((__gm__ kType *)kktBase);
                gmDebugDkbg.SetGlobalBuffer((__gm__ kType *)debugDkbg_ + debugLine * CHUNK_SIZE * K_DIM);
                gmDebugDvb.SetGlobalBuffer((__gm__ kType *)debugDvb_ + debugLine * CHUNK_SIZE * V_DIM);
                gmDebugKkt.SetGlobalBuffer((__gm__ kType *)debugKkt_ + debugKktLine * CHUNK_SIZE * CHUNK_SIZE);

                auto tensorAT = tla::MakeTensor(gmAT, layoutAT, Arch::PositionGM{});
                auto tensorDw = tla::MakeTensor(gmDw, layoutDw, Arch::PositionGM{});
                auto tensorDu = tla::MakeTensor(gmDu, layoutDu, Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK, layoutK, Arch::PositionGM{});
                auto tensorDkbg = tla::MakeTensor(gmDkbg, layoutDkbg, Arch::PositionGM{});
                auto tensorDvb = tla::MakeTensor(gmDvb, layoutDvb, Arch::PositionGM{});
                auto tensorKkt = tla::MakeTensor(gmKkt, layoutKkt, Arch::PositionGM{});
                auto tensorDebugDkbg = tla::MakeTensor(gmDebugDkbg, layoutDkbg, Arch::PositionGM{});
                auto tensorDebugDvb = tla::MakeTensor(gmDebugDvb, layoutDvb, Arch::PositionGM{});
                auto tensorDebugKkt = tla::MakeTensor(gmDebugKkt, layoutKkt, Arch::PositionGM{});
                uint8_t fixpipeUnitFlag = 0b11;

                auto blockATForDkbg = GetTile(tensorAT, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.m(), shapeK.k()));
                auto blockDw = GetTile(tensorDw, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.k(), shapeK.n()));
                auto blockATForDvb = GetTile(tensorAT, tla::MakeCoord(0, 0), tla::MakeShape(shapeV.m(), shapeV.k()));
                auto blockDu = GetTile(tensorDu, tla::MakeCoord(0, 0), tla::MakeShape(shapeV.k(), shapeV.n()));
                auto blockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(shapeKkt.m(), shapeKkt.k()));
                auto blockDebugDkbg =
                    GetTile(tensorDebugDkbg, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.m(), shapeK.n()));
                auto blockDebugKkt =
                    GetTile(tensorDebugKkt, tla::MakeCoord(0, 0), tla::MakeShape(shapeKkt.m(), shapeKkt.n()));

                CopyGmToL1A_Dkbg<decltype(blockATForDkbg)> copyGmToL1A_AT;
                CopyGmToL1B_Dkbg<decltype(blockDw)> copyGmToL1B_DW;
                CopyGmToL1A_Dvb<decltype(blockATForDvb)> copyGmToL1A_ATForDU;
                CopyGmToL1B_Dvb<decltype(blockDu)> copyGmToL1B_DU;
                CopyGmToL1A_Kkt<decltype(blockK)> copyGmToL1A_K;
                CopyL0CToGm_Dkbg<decltype(blockDebugDkbg)> copyL0CToGm_DebugDkbg;
                CopyL0CToGm_Kkt<decltype(blockDebugKkt)> copyL0CToGm_DebugKkt;

                uint32_t dkbgL1Idx = curL1_;
                int32_t dkbgL1AEvent = dkbgL1Idx == 0 ? EVENT_L1A_PING : EVENT_L1A_PONG;
                int32_t dkbgL1BEvent = dkbgL1Idx == 0 ? EVENT_L1B_PING : EVENT_L1B_PONG;
                auto tensorL1A_AT = tla::MakeTensor(l1A[dkbgL1Idx], L1A_LAYOUT_AT, Arch::PositionL1{});
                auto tensorL1B_DW = tla::MakeTensor(l1B[dkbgL1Idx], L1B_LAYOUT_DW, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dkbgL1AEvent);
                copyGmToL1A_AT(tensorL1A_AT, blockATForDkbg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dkbgL1AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dkbgL1BEvent);
                copyGmToL1B_DW(tensorL1B_DW, blockDw);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dkbgL1BEvent);
                curL1_ ^= 1U;

                uint32_t mActualDkbg = shapeK.m();
                if (mActualDkbg == 1) {
                    mActualDkbg = 16;
                }
                uint32_t dkbgL0Idx = curL0_;
                int32_t dkbgL0AEvent = dkbgL0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                int32_t dkbgL0BEvent = dkbgL0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                int32_t dkbgL0ReadyEvent = dkbgL0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                auto layoutL0A_AT = tla::MakeLayout<kType, LayoutTagL0A_Dkbg>(mActualDkbg, shapeK.k());
                auto tensorL0A_AT = tla::MakeTensor(l0A[dkbgL0Idx], layoutL0A_AT, Arch::PositionL0A{});
                auto tensorTileL1A_AT =
                    GetTile(tensorL1A_AT, tla::MakeCoord(0, 0), tla::MakeShape(mActualDkbg, shapeK.k()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dkbgL1AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbgL0AEvent);
                copyL1ToL0A_Dkbg(tensorL0A_AT, tensorTileL1A_AT);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dkbgL1AEvent);

                auto layoutL0B_DW = tla::MakeLayout<kType, LayoutTagL0B_Dkbg>(shapeK.k(), shapeK.n());
                auto tensorL0B_DW = tla::MakeTensor(l0B[dkbgL0Idx], layoutL0B_DW, Arch::PositionL0B{});
                auto tensorTileL1B_DW =
                    GetTile(tensorL1B_DW, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.k(), shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dkbgL1BEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbgL0BEvent);
                copyL1ToL0B_Dkbg(tensorL0B_DW, tensorTileL1B_DW);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dkbgL1BEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dkbgL0ReadyEvent);
                curL0_ ^= 1U;

                uint32_t dvbL1Idx = curL1_;
                int32_t dvbL1AEvent = dvbL1Idx == 0 ? EVENT_L1A_PING : EVENT_L1A_PONG;
                int32_t dvbL1BEvent = dvbL1Idx == 0 ? EVENT_L1B_PING : EVENT_L1B_PONG;
                auto tensorL1A_ATForDU = tla::MakeTensor(l1A[dvbL1Idx], L1A_LAYOUT_AT_FOR_DU, Arch::PositionL1{});
                auto tensorL1B_DU = tla::MakeTensor(l1B[dvbL1Idx], L1B_LAYOUT_DU, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dvbL1AEvent);
                copyGmToL1A_ATForDU(tensorL1A_ATForDU, blockATForDvb);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dvbL1AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dvbL1BEvent);
                copyGmToL1B_DU(tensorL1B_DU, blockDu);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dvbL1BEvent);
                curL1_ ^= 1U;

                auto layoutL0C_Dkbg = tla::MakeLayoutL0C(mActualDkbg, shapeK.n());
                auto tensorL0C_Dkbg = tla::MakeTensor(l0C, layoutL0C_Dkbg, Arch::PositionL0C{});
                auto tensorTileL0C_Dkbg =
                    GetTile(tensorL0C_Dkbg, tla::MakeCoord(0, 0), tla::MakeShape(mActualDkbg, shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dkbgL0ReadyEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                tileMmadDkbg(tensorTileL0C_Dkbg, tensorL0A_AT, tensorL0B_DW, true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbgL0AEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbgL0BEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);

                uint32_t mActualDvb = shapeV.m();
                if (mActualDvb == 1) {
                    mActualDvb = 16;
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                copyL0CToGm_DebugDkbg(blockDebugDkbg, tensorL0C_Dkbg, fixpipeUnitFlag);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                bool dvbL1AReadyConsumed = false;
                bool dvbL1BReadyConsumed = false;
                for (uint32_t nOffset = 0; nOffset < shapeV.n(); nOffset += L0_DVB_N_TILE) {
                    uint32_t curN = nOffset + L0_DVB_N_TILE > shapeV.n() ? shapeV.n() - nOffset : L0_DVB_N_TILE;
                    auto layoutL0C_Dvb = tla::MakeLayoutL0C(mActualDvb, curN);
                    auto tensorL0C_Dvb = tla::MakeTensor(l0C, layoutL0C_Dvb, Arch::PositionL0C{});
                    auto tensorTileL0C_Dvb =
                        GetTile(tensorL0C_Dvb, tla::MakeCoord(0, 0), tla::MakeShape(mActualDvb, curN));
                    auto blockDebugDvbN =
                        GetTile(tensorDebugDvb, tla::MakeCoord(0, nOffset), tla::MakeShape(shapeV.m(), curN));
                    CopyL0CToGm_Dvb<decltype(blockDebugDvbN)> copyL0CToGm_DebugDvbN;

                    for (uint32_t kOffset = 0; kOffset < shapeV.k(); kOffset += L0_DVB_K_TILE) {
                        uint32_t curK = kOffset + L0_DVB_K_TILE > shapeV.k() ? shapeV.k() - kOffset : L0_DVB_K_TILE;
                        bool lastK = kOffset + curK >= shapeV.k();
                        bool lastN = nOffset + curN >= shapeV.n();
                        uint32_t dvbL0Idx = curL0_;
                        int32_t dvbL0AEvent = dvbL0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                        int32_t dvbL0BEvent = dvbL0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                        int32_t dvbL0ReadyEvent = dvbL0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                        auto layoutL0A_ATForDU = tla::MakeLayout<kType, LayoutTagL0A_Dvb>(mActualDvb, curK);
                        auto tensorL0A_ATForDU = tla::MakeTensor(l0A[dvbL0Idx], layoutL0A_ATForDU, Arch::PositionL0A{});
                        auto tensorTileL1A_ATForDU =
                            GetTile(tensorL1A_ATForDU, tla::MakeCoord(0, kOffset), tla::MakeShape(mActualDvb, curK));
                        if (!dvbL1AReadyConsumed) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dvbL1AEvent);
                            dvbL1AReadyConsumed = true;
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dvbL0AEvent);
                        copyL1ToL0A_Dvb(tensorL0A_ATForDU, tensorTileL1A_ATForDU);
                        if (lastK && lastN) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dvbL1AEvent);
                        }

                        auto layoutL0B_DU = tla::MakeLayout<kType, LayoutTagL0B_Dvb>(curK, curN);
                        auto tensorL0B_DU = tla::MakeTensor(l0B[dvbL0Idx], layoutL0B_DU, Arch::PositionL0B{});
                        auto tensorTileL1B_DU =
                            GetTile(tensorL1B_DU, tla::MakeCoord(kOffset, nOffset), tla::MakeShape(curK, curN));
                        if (!dvbL1BReadyConsumed) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dvbL1BEvent);
                            dvbL1BReadyConsumed = true;
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dvbL0BEvent);
                        copyL1ToL0B_Dvb(tensorL0B_DU, tensorTileL1B_DU);
                        if (lastK && lastN) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dvbL1BEvent);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dvbL0ReadyEvent);
                        curL0_ ^= 1U;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dvbL0ReadyEvent);
                        if (kOffset == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                        }
                        uint8_t dvbMmadUnitFlag = lastK ? 0b11 : 0b10;
                        tileMmadDvb(tensorTileL0C_Dvb, tensorL0A_ATForDU, tensorL0B_DU, kOffset == 0,
                                    dvbMmadUnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dvbL0AEvent);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dvbL0BEvent);
                        if (lastK) {
                            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                        }
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    copyL0CToGm_DebugDvbN(blockDebugDvbN, tensorL0C_Dvb, fixpipeUnitFlag);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                }

                if (needComputeKkt) {
                    uint32_t kResidentSlot = cachedKResidentSlot_;
                    int32_t kResidentEvent =
                        kResidentSlot == 0 ? EVENT_K_RESIDENT_PING : EVENT_K_RESIDENT_PONG;
                    auto tensorL1ResidentK =
                        tla::MakeTensor(kResident[kResidentSlot], L1A_LAYOUT_K, Arch::PositionL1{});
                    auto tensorL1ResidentKT =
                        tla::MakeTensor(kResident[kResidentSlot], L1B_LAYOUT_KT, Arch::PositionL1{});
                    if (needLoadKResident) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kResidentEvent);
                        copyGmToL1A_K(tensorL1ResidentK, blockK);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kResidentEvent);
                    }

                    uint32_t mActualKkt = shapeKkt.m();
                    if (mActualKkt == 1) {
                        mActualKkt = 16;
                    }
                    uint32_t kktL0Idx = curL0_;
                    int32_t kktL0AEvent = kktL0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                    int32_t kktL0BEvent = kktL0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                    int32_t kktL0ReadyEvent = kktL0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                    auto layoutL0A_K = tla::MakeLayout<kType, LayoutTagL0A_Kkt>(mActualKkt, shapeKkt.k());
                    auto tensorL0A_K = tla::MakeTensor(l0A[kktL0Idx], layoutL0A_K, Arch::PositionL0A{});
                    auto tensorTileL1A_K =
                        GetTile(tensorL1ResidentK, tla::MakeCoord(0, 0), tla::MakeShape(mActualKkt, shapeKkt.k()));
                    if (needLoadKResident) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kResidentEvent);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kktL0AEvent);
                    copyL1ToL0A_Kkt(tensorL0A_K, tensorTileL1A_K);

                    auto layoutL0B_KT = tla::MakeLayout<kType, LayoutTagL0B_Kkt>(shapeKkt.k(), shapeKkt.n());
                    auto tensorL0B_KT = tla::MakeTensor(l0B[kktL0Idx], layoutL0B_KT, Arch::PositionL0B{});
                    auto tensorTileL1B_KT =
                        GetTile(tensorL1ResidentKT, tla::MakeCoord(0, 0), tla::MakeShape(shapeKkt.k(), shapeKkt.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kktL0BEvent);
                    copyL1ToL0B_Kkt(tensorL0B_KT, tensorTileL1B_KT);
                    if (needLoadKResident) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kResidentEvent);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kktL0ReadyEvent);
                    curL0_ ^= 1U;

                    auto layoutL0C_Kkt = tla::MakeLayoutL0C(mActualKkt, shapeKkt.n());
                    auto tensorL0C_Kkt = tla::MakeTensor(l0C, layoutL0C_Kkt, Arch::PositionL0C{});
                    auto tensorTileL0C_Kkt =
                        GetTile(tensorL0C_Kkt, tla::MakeCoord(0, 0), tla::MakeShape(mActualKkt, shapeKkt.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kktL0ReadyEvent);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    tileMmadKkt(tensorTileL0C_Kkt, tensorL0A_K, tensorL0B_KT, true, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kktL0AEvent);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kktL0BEvent);
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    copyL0CToGm_DebugKkt(blockDebugKkt, tensorL0C_Kkt, fixpipeUnitFlag);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                }

                curSlot_ ^= 1U;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PING);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PING);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PONG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PING);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PING);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PONG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PONG);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PING);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PING);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
}

#endif // PREPARE_WY_REPR_BWD_CUBE_H
