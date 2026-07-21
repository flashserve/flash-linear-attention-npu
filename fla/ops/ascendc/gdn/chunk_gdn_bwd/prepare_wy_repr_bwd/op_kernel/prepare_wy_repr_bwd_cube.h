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
    using LayoutTagKbgT = layout::ColumnMajor;
    using LayoutTagVbT = layout::ColumnMajor;
    using LayoutTagDkbg = layout::RowMajor;
    using LayoutTagDvb = layout::RowMajor;
    using LayoutTagKkt = layout::RowMajor;
    using LayoutTagDA1 = layout::RowMajor;
    using LayoutTagDA2 = layout::RowMajor;

    using TileCopyDkbg =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDw, kType, LayoutTagDkbg>;
    using TileCopyDvb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDu, kType, LayoutTagDvb>;
    using TileCopyKkt =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagK, kType, LayoutTagKT, kType, LayoutTagKkt>;
    using TileCopyDA1 =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDw, kType, LayoutTagKbgT, kType, LayoutTagDA1>;
    using TileCopyDA2 =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDu, kType, LayoutTagVbT, kType, LayoutTagDA2>;

    using ElementAccumulator = typename TileCopyDkbg::ElementAccumulator;
    using CopyL1ToL0A_Dkbg = typename TileCopyDkbg::CopyL1ToL0A;
    using CopyL1ToL0B_Dkbg = typename TileCopyDkbg::CopyL1ToL0B;
    using CopyL1ToL0A_Dvb = typename TileCopyDvb::CopyL1ToL0A;
    using CopyL1ToL0B_Dvb = typename TileCopyDvb::CopyL1ToL0B;
    using CopyL1ToL0A_Kkt = typename TileCopyKkt::CopyL1ToL0A;
    using CopyL1ToL0B_Kkt = typename TileCopyKkt::CopyL1ToL0B;
    using CopyL1ToL0A_DA1 = typename TileCopyDA1::CopyL1ToL0A;
    using CopyL1ToL0B_DA1 = typename TileCopyDA1::CopyL1ToL0B;
    using CopyL1ToL0A_DA2 = typename TileCopyDA2::CopyL1ToL0A;
    using CopyL1ToL0B_DA2 = typename TileCopyDA2::CopyL1ToL0B;

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
    using LayoutTagL1A_DA1 = typename TileCopyDA1::LayoutTagL1A;
    using LayoutTagL1B_DA1 = typename TileCopyDA1::LayoutTagL1B;
    using LayoutTagL0A_DA1 = typename TileCopyDA1::LayoutTagL0A;
    using LayoutTagL0B_DA1 = typename TileCopyDA1::LayoutTagL0B;
    using LayoutTagL1A_DA2 = typename TileCopyDA2::LayoutTagL1A;
    using LayoutTagL1B_DA2 = typename TileCopyDA2::LayoutTagL1B;
    using LayoutTagL0A_DA2 = typename TileCopyDA2::LayoutTagL0A;
    using LayoutTagL0B_DA2 = typename TileCopyDA2::LayoutTagL0B;

    using TileMmadDkbg = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Dkbg>;
    using TileMmadDvb = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Dvb>;
    using TileMmadKkt = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Kkt>;
    using TileMmadDA1 = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_DA1>;
    using TileMmadDA2 = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_DA2>;

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
    template <typename Tensor>
    using CopyGmToL1A_DA1 = typename TileCopyDA1::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_DA1 = typename TileCopyDA1::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_DA1 = typename TileCopyDA1::template CopyL0CToGm<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_DA2 = typename TileCopyDA2::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_DA2 = typename TileCopyDA2::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_DA2 = typename TileCopyDA2::template CopyL0CToGm<Tensor>;

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
    static constexpr auto L1A_LAYOUT_DW_FOR_DA1 =
        tla::MakeLayout<kType, LayoutTagL1A_DA1>(tla::Int<CHUNK_SIZE>{}, tla::Int<K_DIM>{});
    static constexpr auto L1B_LAYOUT_KBG_T =
        tla::MakeLayout<kType, LayoutTagL1B_DA1>(tla::Int<K_DIM>{}, tla::Int<CHUNK_SIZE>{});
    static constexpr auto L1A_LAYOUT_DU_FOR_DA2 =
        tla::MakeLayout<kType, LayoutTagL1A_DA2>(tla::Int<CHUNK_SIZE>{}, tla::Int<V_DIM>{});
    static constexpr auto L1B_LAYOUT_VB_T =
        tla::MakeLayout<kType, LayoutTagL1B_DA2>(tla::Int<V_DIM>{}, tla::Int<CHUNK_SIZE>{});

    static constexpr uint32_t L1_SCRATCH_BUFFER_COUNT = 2;
    static constexpr uint32_t L1_SCRATCH_K_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t L1_SCRATCH_V_TILE_BYTES = CHUNK_SIZE * V_DIM * sizeof(kType);
    static constexpr uint32_t L1_SCRATCH_TILE_BYTES = L1_SCRATCH_V_TILE_BYTES > L1_SCRATCH_K_TILE_BYTES ?
                                                          L1_SCRATCH_V_TILE_BYTES :
                                                          L1_SCRATCH_K_TILE_BYTES;
    static constexpr uint32_t K_RESIDENT_BUFFER_COUNT = 2;
    static constexpr uint32_t K_RESIDENT_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t K_RESIDENT_OFFSET = L1_SCRATCH_TILE_BYTES * L1_SCRATCH_BUFFER_COUNT;
    static constexpr uint32_t DW_RESIDENT_BUFFER_COUNT = PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT;
    static constexpr uint32_t DW_RESIDENT_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t DW_RESIDENT_OFFSET = K_RESIDENT_OFFSET + K_RESIDENT_TILE_BYTES * K_RESIDENT_BUFFER_COUNT;
    static constexpr uint32_t DU_RESIDENT_BUFFER_COUNT = PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT;
    static constexpr uint32_t DU_RESIDENT_TILE_BYTES = CHUNK_SIZE * V_DIM * sizeof(kType);
    static constexpr uint32_t DU_RESIDENT_OFFSET = DW_RESIDENT_OFFSET + DW_RESIDENT_TILE_BYTES * DW_RESIDENT_BUFFER_COUNT;
    static constexpr uint32_t L1_REUSABLE_BYTES = 384 * 1024;
    static constexpr uint32_t L1_USED_BYTES = DU_RESIDENT_OFFSET + DU_RESIDENT_TILE_BYTES * DU_RESIDENT_BUFFER_COUNT;
    static_assert(L1_USED_BYTES <= L1_REUSABLE_BYTES, "prepare_wy_repr_bwd cube L1 reuse area exceeds 384KB.");
    static constexpr uint32_t L0_DVB_K_TILE = V_DIM == 256 ? 64 : CHUNK_SIZE;
    static constexpr uint32_t L0_DVB_N_TILE = V_DIM;
    static constexpr uint32_t L0_DA2_K_TILE = V_DIM == 256 ? 64 : V_DIM;
    static constexpr uint32_t L0A_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t L0B_DVB_TILE_BYTES = L0_DVB_K_TILE * L0_DVB_N_TILE * sizeof(kType);
    static constexpr uint32_t L0B_K_TILE_BYTES = K_DIM * CHUNK_SIZE * sizeof(kType);
    static constexpr uint32_t L0B_TILE_BYTES =
        L0B_DVB_TILE_BYTES > L0B_K_TILE_BYTES ? L0B_DVB_TILE_BYTES : L0B_K_TILE_BYTES;
    static constexpr uint32_t L0_BUFFER_COUNT = 2;
    static constexpr int32_t EVENT_L1_SCRATCH_PING = 0;
    static constexpr int32_t EVENT_DU_RESIDENT_PING = 1;
    static constexpr int32_t EVENT_L1_SCRATCH_PONG = 2;
    static constexpr int32_t EVENT_DU_RESIDENT_PONG = 3;
    static constexpr int32_t EVENT_K_RESIDENT_PING = 4;
    static constexpr int32_t EVENT_K_RESIDENT_PONG = 5;
    static constexpr int32_t EVENT_DW_RESIDENT_PING = 6;
    static constexpr int32_t EVENT_DW_RESIDENT_PONG = 7;
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
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1_SCRATCH_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DU_RESIDENT_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1_SCRATCH_PONG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DU_RESIDENT_PONG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PONG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DW_RESIDENT_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DW_RESIDENT_PONG);
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
    LayoutTagKbgT tagKbgT = LayoutTagKbgT::MakeLayout<kType>(K_DIM, CHUNK_SIZE);
    LayoutTagVbT tagVbT = LayoutTagVbT::MakeLayout<kType>(V_DIM, CHUNK_SIZE);
    LayoutTagDkbg tagDkbg = LayoutTagDkbg::MakeLayout<kType>(CHUNK_SIZE, K_DIM);
    LayoutTagDvb tagDvb = LayoutTagDvb::MakeLayout<kType>(CHUNK_SIZE, V_DIM);
    LayoutTagKkt tagKkt = LayoutTagKkt::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagDA1 tagDA1 = LayoutTagDA1::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagDA2 tagDA2 = LayoutTagDA2::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);

    auto layoutAT = MakeLayoutFromTag(tagAT);
    auto layoutDw = MakeLayoutFromTag(tagDw);
    auto layoutDu = MakeLayoutFromTag(tagDu);
    auto layoutK = MakeLayoutFromTag(tagK);
    auto layoutKbgT = MakeLayoutFromTag(tagKbgT);
    auto layoutVbT = MakeLayoutFromTag(tagVbT);
    auto layoutDkbg = MakeLayoutFromTag(tagDkbg);
    auto layoutDvb = MakeLayoutFromTag(tagDvb);
    auto layoutKkt = MakeLayoutFromTag(tagKkt);
    auto layoutDA1 = MakeLayoutFromTag(tagDA1);
    auto layoutDA2 = MakeLayoutFromTag(tagDA2);

    Arch::Resource<ArchTag> resource;
    AscendC::LocalTensor<kType> l1Scratch[L1_SCRATCH_BUFFER_COUNT] = {
        resource.l1Buf.template GetBufferByByte<kType>(0),
        resource.l1Buf.template GetBufferByByte<kType>(L1_SCRATCH_TILE_BYTES)};
    AscendC::LocalTensor<kType> kResident[K_RESIDENT_BUFFER_COUNT] = {
        resource.l1Buf.template GetBufferByByte<kType>(K_RESIDENT_OFFSET),
        resource.l1Buf.template GetBufferByByte<kType>(K_RESIDENT_OFFSET + K_RESIDENT_TILE_BYTES)};
    AscendC::LocalTensor<kType> dwResident[DW_RESIDENT_BUFFER_COUNT] = {
        resource.l1Buf.template GetBufferByByte<kType>(DW_RESIDENT_OFFSET),
        resource.l1Buf.template GetBufferByByte<kType>(DW_RESIDENT_OFFSET + DW_RESIDENT_TILE_BYTES)};
    AscendC::LocalTensor<kType> duResident[DU_RESIDENT_BUFFER_COUNT] = {
        resource.l1Buf.template GetBufferByByte<kType>(DU_RESIDENT_OFFSET),
        resource.l1Buf.template GetBufferByByte<kType>(DU_RESIDENT_OFFSET + DU_RESIDENT_TILE_BYTES)};
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
    CopyL1ToL0A_DA1 copyL1ToL0A_DA1;
    CopyL1ToL0B_DA1 copyL1ToL0B_DA1;
    CopyL1ToL0A_DA2 copyL1ToL0A_DA2;
    CopyL1ToL0B_DA2 copyL1ToL0B_DA2;
    TileMmadDkbg tileMmadDkbg;
    TileMmadDvb tileMmadDvb;
    TileMmadKkt tileMmadKkt;
    TileMmadDA1 tileMmadDA1;
    TileMmadDA2 tileMmadDA2;

    AscendC::GlobalTensor<kType> gmAT;
    AscendC::GlobalTensor<kType> gmDw;
    AscendC::GlobalTensor<kType> gmDu;
    AscendC::GlobalTensor<kType> gmK;
    AscendC::GlobalTensor<kType> gmKbg;
    AscendC::GlobalTensor<kType> gmVb;
    AscendC::GlobalTensor<kType> gmDkbg;
    AscendC::GlobalTensor<kType> gmDvb;
    AscendC::GlobalTensor<kType> gmKkt;
    AscendC::GlobalTensor<kType> gmDA1;
    AscendC::GlobalTensor<kType> gmDA2;

    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreNum = AscendC::GetBlockNum();
    uint64_t groupSize = PrepareWyReprBwdGetGroupSize(tiling_);

    for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(tiling_.chunkNum); taskIdx += coreNum) {
        PrepareWyReprBwdTaskInfo task;
        PrepareWyReprBwdGetTaskInfo(cuSeqlens_, chunkIndices_, tiling_, taskIdx, task);
        GemmCoord shapeK{task.curChunkSize, K_DIM, task.curChunkSize};
        GemmCoord shapeV{task.curChunkSize, V_DIM, task.curChunkSize};
        GemmCoord shapeKkt{task.curChunkSize, task.curChunkSize, K_DIM};
        GemmCoord shapeDA1{task.curChunkSize, task.curChunkSize, K_DIM};
        GemmCoord shapeDA2{task.curChunkSize, task.curChunkSize, V_DIM};
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

                auto tensorAT = tla::MakeTensor(gmAT, layoutAT, Arch::PositionGM{});
                auto tensorDw = tla::MakeTensor(gmDw, layoutDw, Arch::PositionGM{});
                auto tensorDu = tla::MakeTensor(gmDu, layoutDu, Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK, layoutK, Arch::PositionGM{});
                auto tensorDkbg = tla::MakeTensor(gmDkbg, layoutDkbg, Arch::PositionGM{});
                auto tensorDvb = tla::MakeTensor(gmDvb, layoutDvb, Arch::PositionGM{});
                auto tensorKkt = tla::MakeTensor(gmKkt, layoutKkt, Arch::PositionGM{});
                uint8_t fixpipeUnitFlag = 0b11;

                auto blockATForDkbg = GetTile(tensorAT, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.m(), shapeK.k()));
                auto blockDw = GetTile(tensorDw, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.k(), shapeK.n()));
                auto blockDu = GetTile(tensorDu, tla::MakeCoord(0, 0), tla::MakeShape(shapeV.k(), shapeV.n()));
                auto blockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(shapeKkt.m(), shapeKkt.k()));
                auto blockDkbg =
                    GetTile(tensorDkbg, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.m(), shapeK.n()));
                auto blockKkt =
                    GetTile(tensorKkt, tla::MakeCoord(0, 0), tla::MakeShape(shapeKkt.m(), shapeKkt.n()));

                CopyGmToL1A_Dkbg<decltype(blockATForDkbg)> copyGmToL1A_AT;
                CopyGmToL1B_Dkbg<decltype(blockDw)> copyGmToL1B_DW;
                CopyGmToL1B_Dvb<decltype(blockDu)> copyGmToL1B_DU;
                CopyGmToL1A_Kkt<decltype(blockK)> copyGmToL1A_K;
                CopyL0CToGm_Dkbg<decltype(blockDkbg)> copyL0CToGm_Dkbg;
                CopyL0CToGm_Kkt<decltype(blockKkt)> copyL0CToGm_Kkt;

                uint32_t stage0AIdx = curL1_;
                int32_t stage0AEvent = stage0AIdx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                uint32_t dwResidentIdx = curSlot_;
                int32_t dwResidentEvent =
                    dwResidentIdx == 0 ? EVENT_DW_RESIDENT_PING : EVENT_DW_RESIDENT_PONG;
                auto tensorL1A_AT = tla::MakeTensor(l1Scratch[stage0AIdx], L1A_LAYOUT_AT, Arch::PositionL1{});
                auto tensorL1ResidentDWForDkbg =
                    tla::MakeTensor(dwResident[dwResidentIdx], L1B_LAYOUT_DW, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(stage0AEvent);
                copyGmToL1A_AT(tensorL1A_AT, blockATForDkbg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(stage0AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dwResidentEvent);
                copyGmToL1B_DW(tensorL1ResidentDWForDkbg, blockDw);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dwResidentEvent);

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
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(stage0AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbgL0AEvent);
                copyL1ToL0A_Dkbg(tensorL0A_AT, tensorTileL1A_AT);

                auto layoutL0B_DW = tla::MakeLayout<kType, LayoutTagL0B_Dkbg>(shapeK.k(), shapeK.n());
                auto tensorL0B_DW = tla::MakeTensor(l0B[dkbgL0Idx], layoutL0B_DW, Arch::PositionL0B{});
                auto tensorTileL1B_DW =
                    GetTile(tensorL1ResidentDWForDkbg, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.k(), shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dwResidentEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbgL0BEvent);
                copyL1ToL0B_Dkbg(tensorL0B_DW, tensorTileL1B_DW);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dkbgL0ReadyEvent);
                curL0_ ^= 1U;

                uint32_t duResidentIdx = curSlot_;
                int32_t duResidentEvent =
                    duResidentIdx == 0 ? EVENT_DU_RESIDENT_PING : EVENT_DU_RESIDENT_PONG;
                auto tensorL1A_ATForDU =
                    tla::MakeTensor(l1Scratch[stage0AIdx], L1A_LAYOUT_AT_FOR_DU, Arch::PositionL1{});
                auto tensorL1ResidentDUForDvb =
                    tla::MakeTensor(duResident[duResidentIdx], L1B_LAYOUT_DU, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(duResidentEvent);
                copyGmToL1B_DU(tensorL1ResidentDUForDvb, blockDu);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(duResidentEvent);

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
                copyL0CToGm_Dkbg(blockDkbg, tensorL0C_Dkbg, fixpipeUnitFlag);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                bool dvbL1BReadyConsumed = false;
                for (uint32_t nOffset = 0; nOffset < shapeV.n(); nOffset += L0_DVB_N_TILE) {
                    uint32_t curN = nOffset + L0_DVB_N_TILE > shapeV.n() ? shapeV.n() - nOffset : L0_DVB_N_TILE;
                    auto layoutL0C_Dvb = tla::MakeLayoutL0C(mActualDvb, curN);
                    auto tensorL0C_Dvb = tla::MakeTensor(l0C, layoutL0C_Dvb, Arch::PositionL0C{});
                    auto tensorTileL0C_Dvb =
                        GetTile(tensorL0C_Dvb, tla::MakeCoord(0, 0), tla::MakeShape(mActualDvb, curN));
                    auto blockDvbN =
                        GetTile(tensorDvb, tla::MakeCoord(0, nOffset), tla::MakeShape(shapeV.m(), curN));
                    CopyL0CToGm_Dvb<decltype(blockDvbN)> copyL0CToGm_DvbN;

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
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dvbL0AEvent);
                        copyL1ToL0A_Dvb(tensorL0A_ATForDU, tensorTileL1A_ATForDU);
                        if (lastK && lastN) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(stage0AEvent);
                        }

                        auto layoutL0B_DU = tla::MakeLayout<kType, LayoutTagL0B_Dvb>(curK, curN);
                        auto tensorL0B_DU = tla::MakeTensor(l0B[dvbL0Idx], layoutL0B_DU, Arch::PositionL0B{});
                        auto tensorTileL1B_DU =
                            GetTile(tensorL1ResidentDUForDvb, tla::MakeCoord(kOffset, nOffset),
                                    tla::MakeShape(curK, curN));
                        if (!dvbL1BReadyConsumed) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(duResidentEvent);
                            dvbL1BReadyConsumed = true;
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dvbL0BEvent);
                        copyL1ToL0B_Dvb(tensorL0B_DU, tensorTileL1B_DU);
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
                    copyL0CToGm_DvbN(blockDvbN, tensorL0C_Dvb, fixpipeUnitFlag);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                }
                curL1_ ^= 1U;

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
                    copyL0CToGm_Kkt(blockKkt, tensorL0C_Kkt, fixpipeUnitFlag);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                }

                curSlot_ ^= 1U;
            }

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                uint64_t valueBase = hv * tiling_.T + task.valueBos;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);

                gmKbg.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.kbgOffset));
                gmVb.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.vbOffset));
                gmDA1.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da1Offset));
                gmDA2.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da2Offset));

                auto tensorKbgT = tla::MakeTensor(gmKbg, layoutKbgT, Arch::PositionGM{});
                auto tensorVbT = tla::MakeTensor(gmVb, layoutVbT, Arch::PositionGM{});
                auto tensorDA1 = tla::MakeTensor(gmDA1, layoutDA1, Arch::PositionGM{});
                auto tensorDA2 = tla::MakeTensor(gmDA2, layoutDA2, Arch::PositionGM{});
                uint8_t fixpipeUnitFlag = 0b11;

                auto blockKbgT =
                    GetTile(tensorKbgT, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA1.k(), shapeDA1.n()));
                auto blockDA1 =
                    GetTile(tensorDA1, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA1.m(), shapeDA1.n()));
                auto blockVbT =
                    GetTile(tensorVbT, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA2.k(), shapeDA2.n()));
                auto blockDA2 =
                    GetTile(tensorDA2, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA2.m(), shapeDA2.n()));

                CopyGmToL1B_DA1<decltype(blockKbgT)> copyGmToL1B_KbgT;
                CopyL0CToGm_DA1<decltype(blockDA1)> copyL0CToGm_DA1;
                CopyGmToL1B_DA2<decltype(blockVbT)> copyGmToL1B_VbT;
                CopyL0CToGm_DA2<decltype(blockDA2)> copyL0CToGm_DA2;

                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(vecToCubeFlag_);

                uint32_t da1L1Idx = curL1_;
                int32_t da1L1AEvent = da1L1Idx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                uint32_t dwResidentIdx = curSlot_;
                int32_t dwResidentEvent =
                    dwResidentIdx == 0 ? EVENT_DW_RESIDENT_PING : EVENT_DW_RESIDENT_PONG;
                auto tensorL1ResidentDWForDA1 =
                    tla::MakeTensor(dwResident[dwResidentIdx], L1A_LAYOUT_DW_FOR_DA1, Arch::PositionL1{});
                auto tensorL1A_KbgT =
                    tla::MakeTensor(l1Scratch[da1L1Idx], L1B_LAYOUT_KBG_T, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(da1L1AEvent);
                copyGmToL1B_KbgT(tensorL1A_KbgT, blockKbgT);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(da1L1AEvent);
                curL1_ ^= 1U;

                uint32_t mActualDA1 = shapeDA1.m();
                if (mActualDA1 == 1) {
                    mActualDA1 = 16;
                }
                uint32_t da1L0Idx = curL0_;
                int32_t da1L0AEvent = da1L0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                int32_t da1L0BEvent = da1L0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                int32_t da1L0ReadyEvent = da1L0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                auto layoutL0A_DWForDA1 = tla::MakeLayout<kType, LayoutTagL0A_DA1>(mActualDA1, shapeDA1.k());
                auto tensorL0A_DWForDA1 =
                    tla::MakeTensor(l0A[da1L0Idx], layoutL0A_DWForDA1, Arch::PositionL0A{});
                auto tensorTileL1A_DWForDA1 =
                    GetTile(tensorL1ResidentDWForDA1, tla::MakeCoord(0, 0), tla::MakeShape(mActualDA1, shapeDA1.k()));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(da1L0AEvent);
                copyL1ToL0A_DA1(tensorL0A_DWForDA1, tensorTileL1A_DWForDA1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dwResidentEvent);

                auto layoutL0B_KbgT = tla::MakeLayout<kType, LayoutTagL0B_DA1>(shapeDA1.k(), shapeDA1.n());
                auto tensorL0B_KbgT = tla::MakeTensor(l0B[da1L0Idx], layoutL0B_KbgT, Arch::PositionL0B{});
                auto tensorTileL1B_KbgT =
                    GetTile(tensorL1A_KbgT, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA1.k(), shapeDA1.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(da1L1AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(da1L0BEvent);
                copyL1ToL0B_DA1(tensorL0B_KbgT, tensorTileL1B_KbgT);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(da1L1AEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(da1L0ReadyEvent);
                curL0_ ^= 1U;

                auto layoutL0C_DA1 = tla::MakeLayoutL0C(mActualDA1, shapeDA1.n());
                auto tensorL0C_DA1 = tla::MakeTensor(l0C, layoutL0C_DA1, Arch::PositionL0C{});
                auto tensorTileL0C_DA1 =
                    GetTile(tensorL0C_DA1, tla::MakeCoord(0, 0), tla::MakeShape(mActualDA1, shapeDA1.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(da1L0ReadyEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                tileMmadDA1(tensorTileL0C_DA1, tensorL0A_DWForDA1, tensorL0B_KbgT, true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da1L0AEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da1L0BEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                copyL0CToGm_DA1(blockDA1, tensorL0C_DA1, fixpipeUnitFlag);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);

                uint32_t da2L1Idx = curL1_;
                int32_t da2L1AEvent = da2L1Idx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                uint32_t duResidentIdxForDA2 = curSlot_;
                int32_t duResidentEventForDA2 =
                    duResidentIdxForDA2 == 0 ? EVENT_DU_RESIDENT_PING : EVENT_DU_RESIDENT_PONG;
                auto tensorL1ResidentDUForDA2 =
                    tla::MakeTensor(duResident[duResidentIdxForDA2], L1A_LAYOUT_DU_FOR_DA2, Arch::PositionL1{});
                auto tensorL1A_VbT =
                    tla::MakeTensor(l1Scratch[da2L1Idx], L1B_LAYOUT_VB_T, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(da2L1AEvent);
                copyGmToL1B_VbT(tensorL1A_VbT, blockVbT);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(da2L1AEvent);
                curL1_ ^= 1U;

                uint32_t mActualDA2 = shapeDA2.m();
                if (mActualDA2 == 1) {
                    mActualDA2 = 16;
                }
                auto layoutL0C_DA2 = tla::MakeLayoutL0C(mActualDA2, shapeDA2.n());
                auto tensorL0C_DA2 = tla::MakeTensor(l0C, layoutL0C_DA2, Arch::PositionL0C{});
                auto tensorTileL0C_DA2 =
                    GetTile(tensorL0C_DA2, tla::MakeCoord(0, 0), tla::MakeShape(mActualDA2, shapeDA2.n()));
                bool da2VbTReadyConsumed = false;
                for (uint32_t kOffset = 0; kOffset < shapeDA2.k(); kOffset += L0_DA2_K_TILE) {
                    uint32_t curK =
                        kOffset + L0_DA2_K_TILE > shapeDA2.k() ? shapeDA2.k() - kOffset : L0_DA2_K_TILE;
                    bool lastK = kOffset + curK >= shapeDA2.k();
                    uint32_t da2L0Idx = curL0_;
                    int32_t da2L0AEvent = da2L0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                    int32_t da2L0BEvent = da2L0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                    int32_t da2L0ReadyEvent = da2L0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                    auto layoutL0A_DUForDA2 = tla::MakeLayout<kType, LayoutTagL0A_DA2>(mActualDA2, curK);
                    auto tensorL0A_DUForDA2 =
                        tla::MakeTensor(l0A[da2L0Idx], layoutL0A_DUForDA2, Arch::PositionL0A{});
                    auto tensorTileL1A_DUForDA2 =
                        GetTile(tensorL1ResidentDUForDA2, tla::MakeCoord(0, kOffset),
                                tla::MakeShape(mActualDA2, curK));
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(da2L0AEvent);
                    copyL1ToL0A_DA2(tensorL0A_DUForDA2, tensorTileL1A_DUForDA2);
                    if (lastK) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(duResidentEventForDA2);
                    }

                    auto layoutL0B_VbT = tla::MakeLayout<kType, LayoutTagL0B_DA2>(curK, shapeDA2.n());
                    auto tensorL0B_VbT = tla::MakeTensor(l0B[da2L0Idx], layoutL0B_VbT, Arch::PositionL0B{});
                    auto tensorTileL1B_VbT =
                        GetTile(tensorL1A_VbT, tla::MakeCoord(kOffset, 0), tla::MakeShape(curK, shapeDA2.n()));
                    if (!da2VbTReadyConsumed) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(da2L1AEvent);
                        da2VbTReadyConsumed = true;
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(da2L0BEvent);
                    copyL1ToL0B_DA2(tensorL0B_VbT, tensorTileL1B_VbT);
                    if (lastK) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(da2L1AEvent);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(da2L0ReadyEvent);
                    curL0_ ^= 1U;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(da2L0ReadyEvent);
                    if (kOffset == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    }
                    uint8_t da2MmadUnitFlag = lastK ? 0b11 : 0b10;
                    tileMmadDA2(tensorTileL0C_DA2, tensorL0A_DUForDA2, tensorL0B_VbT, kOffset == 0,
                                da2MmadUnitFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da2L0AEvent);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da2L0BEvent);
                    if (lastK) {
                        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    }
                }

                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                copyL0CToGm_DA2(blockDA2, tensorL0C_DA2, fixpipeUnitFlag);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(cubeToVecFlag_);
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
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1_SCRATCH_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DU_RESIDENT_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1_SCRATCH_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DU_RESIDENT_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_RESIDENT_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DW_RESIDENT_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DW_RESIDENT_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PING);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PING);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B_PONG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
}

#endif // PREPARE_WY_REPR_BWD_CUBE_H
