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
 * \brief Cube side process for fused prepare_wy_repr_bwd A5.
 */

#ifndef PREPARE_WY_REPR_BWD_CUBE_H
#define PREPARE_WY_REPR_BWD_CUBE_H

#define CATLASS_ARCH 3510
#include "prepare_wy_repr_bwd_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
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
                                                  GM_ADDR chunkIndices, GM_ADDR dv, GM_ADDR workspace);
    __aicore__ inline void Init(const GDN::PrepareWyReprBwdTilingData &tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitPipeFlags();
    __aicore__ inline int32_t GetL0CEvent(uint32_t l0CIdx) const;
    __aicore__ inline void ProcessImpl();

private:
    using ArchTag = Arch::Ascend950;
    using LayoutTagAT = layout::ColumnMajor;
    using LayoutTagDw = layout::RowMajor;
    using LayoutTagDu = layout::RowMajor;
    using LayoutTagK = layout::RowMajor;
    using LayoutTagKT = layout::ColumnMajor;
    using LayoutTagKbgT = layout::ColumnMajor;
    using LayoutTagVbT = layout::ColumnMajor;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagDkbg = layout::RowMajor;
    using LayoutTagDvb = layout::RowMajor;
    using LayoutTagKkt = layout::RowMajor;
    using LayoutTagD = layout::RowMajor;
    using LayoutTagDT = layout::ColumnMajor;
    using LayoutTagKbeta = layout::RowMajor;
    using LayoutTagDkb = layout::RowMajor;
    using LayoutTagDK = layout::RowMajor;
    using LayoutTagDA1 = layout::RowMajor;
    using LayoutTagDA2 = layout::RowMajor;
    using LayoutTagDA4 = layout::RowMajor;
    using LayoutTagDA5 = layout::RowMajor;
    using LayoutTagDA5T = layout::ColumnMajor;
    using LayoutTagDA6T = layout::RowMajor;

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
    using TileCopyDA5 =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDA4, kType, LayoutTagAT, kType, LayoutTagDA5>;
    using TileCopyDA6T =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDA5T, kType, LayoutTagA, kType, LayoutTagDA6T>;
    using TileCopyDkb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDT, kType, LayoutTagK, kType, LayoutTagDkb>;
    using TileCopyDK =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagD, kType, LayoutTagKbeta, kType, LayoutTagDK>;

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
    using CopyL1ToL0A_DA5 = typename TileCopyDA5::CopyL1ToL0A;
    using CopyL1ToL0B_DA5 = typename TileCopyDA5::CopyL1ToL0B;
    using CopyL1ToL0A_DA6T = typename TileCopyDA6T::CopyL1ToL0A;
    using CopyL1ToL0B_DA6T = typename TileCopyDA6T::CopyL1ToL0B;
    using CopyL1ToL0A_Dkb = typename TileCopyDkb::CopyL1ToL0A;
    using CopyL1ToL0B_Dkb = typename TileCopyDkb::CopyL1ToL0B;
    using CopyL1ToL0A_DK = typename TileCopyDK::CopyL1ToL0A;
    using CopyL1ToL0B_DK = typename TileCopyDK::CopyL1ToL0B;

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
    using LayoutTagL1A_DA5 = typename TileCopyDA5::LayoutTagL1A;
    using LayoutTagL1B_DA5 = typename TileCopyDA5::LayoutTagL1B;
    using LayoutTagL0A_DA5 = typename TileCopyDA5::LayoutTagL0A;
    using LayoutTagL0B_DA5 = typename TileCopyDA5::LayoutTagL0B;
    using LayoutTagL1A_DA6T = typename TileCopyDA6T::LayoutTagL1A;
    using LayoutTagL1B_DA6T = typename TileCopyDA6T::LayoutTagL1B;
    using LayoutTagL0A_DA6T = typename TileCopyDA6T::LayoutTagL0A;
    using LayoutTagL0B_DA6T = typename TileCopyDA6T::LayoutTagL0B;
    using LayoutTagL1A_Dkb = typename TileCopyDkb::LayoutTagL1A;
    using LayoutTagL1B_Dkb = typename TileCopyDkb::LayoutTagL1B;
    using LayoutTagL0A_Dkb = typename TileCopyDkb::LayoutTagL0A;
    using LayoutTagL0B_Dkb = typename TileCopyDkb::LayoutTagL0B;
    using LayoutTagL1A_DK = typename TileCopyDK::LayoutTagL1A;
    using LayoutTagL1B_DK = typename TileCopyDK::LayoutTagL1B;
    using LayoutTagL0A_DK = typename TileCopyDK::LayoutTagL0A;
    using LayoutTagL0B_DK = typename TileCopyDK::LayoutTagL0B;

    using TileMmadDkbg = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Dkbg>;
    using TileMmadDvb = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Dvb>;
    using TileMmadKkt = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Kkt>;
    using TileMmadDA1 = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_DA1>;
    using TileMmadDA2 = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_DA2>;
    using TileMmadDA5 = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_DA5>;
    using TileMmadDA6T = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_DA6T>;
    using TileMmadDkb = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_Dkb>;
    using TileMmadDK = Gemm::Tile::TileMmadTla<ArchTag, kType, LayoutTagL1A_DK>;

    template <typename Tensor>
    using CopyGmToL1A_Dkbg = typename TileCopyDkbg::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_Dkbg = typename TileCopyDkbg::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_Dkbg = typename TileCopyDkbg::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_Dvb = typename TileCopyDvb::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_Dvb = typename TileCopyDvb::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_Dvb = typename TileCopyDvb::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_Kkt = typename TileCopyKkt::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_Kkt = typename TileCopyKkt::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_DA1 = typename TileCopyDA1::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_DA1 = typename TileCopyDA1::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_DA1 = typename TileCopyDA1::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_DA2 = typename TileCopyDA2::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_DA2 = typename TileCopyDA2::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_DA2 = typename TileCopyDA2::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_DA5 = typename TileCopyDA5::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_DA5 = typename TileCopyDA5::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_DA5 = typename TileCopyDA5::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_DA6T = typename TileCopyDA6T::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_DA6T = typename TileCopyDA6T::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_DA6T = typename TileCopyDA6T::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_Dkb = typename TileCopyDkb::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_Dkb = typename TileCopyDkb::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyGmToL1A_DK = typename TileCopyDK::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1B_DK = typename TileCopyDK::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGm_DK = typename TileCopyDK::template CopyL0CToDst<Tensor>;

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
    static constexpr auto L1A_LAYOUT_DA4 =
        tla::MakeLayout<kType, LayoutTagL1A_DA5>(tla::Int<CHUNK_SIZE>{}, tla::Int<CHUNK_SIZE>{});
    static constexpr auto L1B_LAYOUT_AT_FOR_DA5 =
        tla::MakeLayout<kType, LayoutTagL1B_DA5>(tla::Int<CHUNK_SIZE>{}, tla::Int<CHUNK_SIZE>{});
    static constexpr auto L1A_LAYOUT_DA5_T =
        tla::MakeLayout<kType, LayoutTagL1A_DA6T>(tla::Int<CHUNK_SIZE>{}, tla::Int<CHUNK_SIZE>{});
    static constexpr auto L1B_LAYOUT_A_FOR_DA6T =
        tla::MakeLayout<kType, LayoutTagL1B_DA6T>(tla::Int<CHUNK_SIZE>{}, tla::Int<CHUNK_SIZE>{});
    static constexpr auto L1A_LAYOUT_D_T =
        tla::MakeLayout<kType, LayoutTagL1A_Dkb>(tla::Int<CHUNK_SIZE>{}, tla::Int<CHUNK_SIZE>{});
    static constexpr auto L1B_LAYOUT_K_FOR_DKB =
        tla::MakeLayout<kType, LayoutTagL1B_Dkb>(tla::Int<CHUNK_SIZE>{}, tla::Int<K_DIM>{});
    static constexpr auto L1A_LAYOUT_D_FOR_DK =
        tla::MakeLayout<kType, LayoutTagL1A_DK>(tla::Int<CHUNK_SIZE>{}, tla::Int<CHUNK_SIZE>{});
    static constexpr auto L1B_LAYOUT_KBETA =
        tla::MakeLayout<kType, LayoutTagL1B_DK>(tla::Int<CHUNK_SIZE>{}, tla::Int<K_DIM>{});

    static constexpr uint32_t L1_SCRATCH_BUFFER_COUNT = 2;
    static constexpr uint32_t L1_SCRATCH_K_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t L1_SCRATCH_V_TILE_BYTES = CHUNK_SIZE * V_DIM * sizeof(kType);
    static constexpr uint32_t L1_SCRATCH_TILE_BYTES = L1_SCRATCH_V_TILE_BYTES > L1_SCRATCH_K_TILE_BYTES ?
                                                          L1_SCRATCH_V_TILE_BYTES :
                                                          L1_SCRATCH_K_TILE_BYTES;
    static constexpr uint32_t K_RESIDENT_BUFFER_COUNT = 2;
    static constexpr uint32_t K_RESIDENT_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t K_RESIDENT_OFFSET = L1_SCRATCH_TILE_BYTES * L1_SCRATCH_BUFFER_COUNT;
    static constexpr uint32_t DW_RESIDENT_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t DW_RESIDENT_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t DW_RESIDENT_OFFSET = K_RESIDENT_OFFSET + K_RESIDENT_TILE_BYTES * K_RESIDENT_BUFFER_COUNT;
    static constexpr uint32_t DU_RESIDENT_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t DU_RESIDENT_TILE_BYTES = CHUNK_SIZE * V_DIM * sizeof(kType);
    static constexpr uint32_t DU_RESIDENT_OFFSET = DW_RESIDENT_OFFSET + DW_RESIDENT_TILE_BYTES * DW_RESIDENT_BUFFER_COUNT;
    static constexpr uint32_t A_RESIDENT_BUFFER_COUNT = BUFFER_COUNT_2;
    static constexpr uint32_t A_RESIDENT_TILE_BYTES = CHUNK_SIZE * CHUNK_SIZE * sizeof(kType);
    static constexpr uint32_t A_RESIDENT_OFFSET = DU_RESIDENT_OFFSET + DU_RESIDENT_TILE_BYTES * DU_RESIDENT_BUFFER_COUNT;
    static constexpr uint32_t L1_TOTAL_BYTES = 512 * 1024;
    static constexpr uint32_t L1_USED_BYTES = A_RESIDENT_OFFSET + A_RESIDENT_TILE_BYTES * A_RESIDENT_BUFFER_COUNT;
    static_assert(L1_USED_BYTES <= L1_TOTAL_BYTES, "prepare_wy_repr_bwd cube L1 usage exceeds 512KB.");
    static constexpr uint32_t L0_DVB_K_TILE = V_DIM == 256 ? 64 : CHUNK_SIZE;
    static constexpr uint32_t L0_DVB_N_TILE = V_DIM;
    static constexpr uint32_t L0_DA2_K_TILE = V_DIM == 256 ? 64 : V_DIM;
    static constexpr uint32_t L0A_TILE_BYTES = CHUNK_SIZE * K_DIM * sizeof(kType);
    static constexpr uint32_t L0B_DVB_TILE_BYTES = L0_DVB_K_TILE * L0_DVB_N_TILE * sizeof(kType);
    static constexpr uint32_t L0B_K_TILE_BYTES = K_DIM * CHUNK_SIZE * sizeof(kType);
    static constexpr uint32_t L0B_TILE_BYTES =
        L0B_DVB_TILE_BYTES > L0B_K_TILE_BYTES ? L0B_DVB_TILE_BYTES : L0B_K_TILE_BYTES;
    static constexpr uint32_t L0_BUFFER_COUNT = 2;
    static constexpr uint32_t L0C_BUFFER_COUNT = 2;
    static constexpr uint32_t L0C_TILE_COLS = V_DIM > K_DIM ? V_DIM : K_DIM;
    static constexpr uint32_t L0C_TILE_BYTES = CHUNK_SIZE * L0C_TILE_COLS * sizeof(ElementAccumulator);
    static constexpr uint32_t L0C_TOTAL_BYTES = 256 * 1024;
    static_assert(L0C_TILE_BYTES * L0C_BUFFER_COUNT <= L0C_TOTAL_BYTES,
                  "prepare_wy_repr_bwd A5 cube L0C double buffer exceeds 256KB.");
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
    static constexpr int32_t EVENT_L0C_PING = 0;
    static constexpr int32_t EVENT_L0C_PONG = 1;
    static constexpr int32_t EVENT_FIX_TO_MTE2_PING = 0;
    static constexpr int32_t EVENT_FIX_TO_MTE2_PONG = 1;

    GM_ADDR k_ = nullptr;
    GM_ADDR A_ = nullptr;
    GM_ADDR dw_ = nullptr;
    GM_ADDR du_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR dv_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    GDN::PrepareWyReprBwdTilingData tiling_{};
    uint32_t curSlot_ = 0;
    uint32_t nextKResidentSlot_ = 0;
    uint32_t cachedKResidentSlot_ = 0;
    uint64_t cachedKResidentHk_ = static_cast<uint64_t>(-1);
    uint32_t nextKktSlot_ = 0;
    uint32_t cachedKktSlot_ = 0;
    uint64_t cachedKktHk_ = static_cast<uint64_t>(-1);
    uint32_t kResidentSlotForSlot_[BUFFER_COUNT_4] = {0, 0, 0, 0};
    uint32_t kktSlotForSlot_[BUFFER_COUNT_4] = {0, 0, 0, 0};
    uint32_t curL1_ = 0;
    uint32_t curL0_ = 0;
    uint32_t curL0C_ = 0;
    Arch::CrossCoreFlag vecToCubeFlag_{PREPARE_WY_REPR_BWD_VEC_TO_CUBE_FLAG_READY};
    Arch::CrossCoreFlag cubeToVecFlag_{PREPARE_WY_REPR_BWD_CUBE_TO_VEC_FLAG_READY};
};

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline PrepareWyReprBwdCubeProcess<kType, gType, V_DIM, CHUNK_SIZE>::PrepareWyReprBwdCubeProcess(
    GM_ADDR k, GM_ADDR A, GM_ADDR dw, GM_ADDR du, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dv,
    GM_ADDR workspace)
    : k_(k), A_(A), dw_(dw), du_(du), cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices), dv_(dv),
      workspace_(workspace)
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
    for (uint32_t slot = 0; slot < BUFFER_COUNT_4; ++slot) {
        kResidentSlotForSlot_[slot] = 0;
        kktSlotForSlot_[slot] = 0;
    }
    curL1_ = 0;
    curL0_ = 0;
    curL0C_ = 0;
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
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PING);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PONG);
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline int32_t
PrepareWyReprBwdCubeProcess<kType, gType, V_DIM, CHUNK_SIZE>::GetL0CEvent(uint32_t l0CIdx) const
{
    return l0CIdx == 0 ? EVENT_L0C_PING : EVENT_L0C_PONG;
}

template <typename kType, typename gType, uint32_t V_DIM, uint32_t CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdCubeProcess<kType, gType, V_DIM, CHUNK_SIZE>::Process()
{
    AscendC::SetMMLayoutTransform(true);
    ProcessImpl();
    AscendC::SetMMLayoutTransform(false);
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
    LayoutTagD tagD = LayoutTagD::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagDT tagDT = LayoutTagDT::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagKbeta tagKbeta = LayoutTagKbeta::MakeLayout<kType>(CHUNK_SIZE, K_DIM);
    LayoutTagDkb tagDkb = LayoutTagDkb::MakeLayout<kType>(CHUNK_SIZE, K_DIM);
    LayoutTagDK tagDK = LayoutTagDK::MakeLayout<kType>(CHUNK_SIZE, K_DIM);
    LayoutTagDA1 tagDA1 = LayoutTagDA1::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagDA2 tagDA2 = LayoutTagDA2::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagDA1 tagDebugDA1 = LayoutTagDA1::MakeLayout<kType>(CHUNK_SIZE, V_DIM);
    LayoutTagDA2 tagDebugDA2 = LayoutTagDA2::MakeLayout<kType>(CHUNK_SIZE, V_DIM);
    LayoutTagDA4 tagDA4 = LayoutTagDA4::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagDA5 tagDA5 = LayoutTagDA5::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagDA5T tagDA5T = LayoutTagDA5T::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);
    LayoutTagDA6T tagDA6T = LayoutTagDA6T::MakeLayout<kType>(CHUNK_SIZE, CHUNK_SIZE);

    auto layoutAT = MakeLayoutFromTag(tagAT);
    auto layoutDw = MakeLayoutFromTag(tagDw);
    auto layoutDu = MakeLayoutFromTag(tagDu);
    auto layoutK = MakeLayoutFromTag(tagK);
    auto layoutKbgT = MakeLayoutFromTag(tagKbgT);
    auto layoutVbT = MakeLayoutFromTag(tagVbT);
    auto layoutDkbg = MakeLayoutFromTag(tagDkbg);
    auto layoutDvb = MakeLayoutFromTag(tagDvb);
    auto layoutKkt = MakeLayoutFromTag(tagKkt);
    auto layoutD = MakeLayoutFromTag(tagD);
    auto layoutDT = MakeLayoutFromTag(tagDT);
    auto layoutKbeta = MakeLayoutFromTag(tagKbeta);
    auto layoutDkb = MakeLayoutFromTag(tagDkb);
    auto layoutDK = MakeLayoutFromTag(tagDK);
    auto layoutDA1 = MakeLayoutFromTag(tagDA1);
    auto layoutDA2 = MakeLayoutFromTag(tagDA2);
    auto layoutDebugDA1 = MakeLayoutFromTag(tagDebugDA1);
    auto layoutDebugDA2 = MakeLayoutFromTag(tagDebugDA2);
    auto layoutDA4 = MakeLayoutFromTag(tagDA4);
    auto layoutDA5 = MakeLayoutFromTag(tagDA5);
    auto layoutDA5T = MakeLayoutFromTag(tagDA5T);
    auto layoutDA6T = MakeLayoutFromTag(tagDA6T);

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
    AscendC::LocalTensor<kType> aResident[A_RESIDENT_BUFFER_COUNT] = {
        resource.l1Buf.template GetBufferByByte<kType>(A_RESIDENT_OFFSET),
        resource.l1Buf.template GetBufferByByte<kType>(A_RESIDENT_OFFSET + A_RESIDENT_TILE_BYTES)};
    AscendC::LocalTensor<kType> l0A[L0_BUFFER_COUNT] = {
        resource.l0ABuf.template GetBufferByByte<kType>(0),
        resource.l0ABuf.template GetBufferByByte<kType>(L0A_TILE_BYTES)};
    AscendC::LocalTensor<kType> l0B[L0_BUFFER_COUNT] = {
        resource.l0BBuf.template GetBufferByByte<kType>(0),
        resource.l0BBuf.template GetBufferByByte<kType>(L0B_TILE_BYTES)};
    AscendC::LocalTensor<ElementAccumulator> l0C[L0C_BUFFER_COUNT] = {
        resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0),
        resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_BYTES)};

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
    CopyL1ToL0A_DA5 copyL1ToL0A_DA5;
    CopyL1ToL0B_DA5 copyL1ToL0B_DA5;
    CopyL1ToL0A_DA6T copyL1ToL0A_DA6T;
    CopyL1ToL0B_DA6T copyL1ToL0B_DA6T;
    CopyL1ToL0A_Dkb copyL1ToL0A_Dkb;
    CopyL1ToL0B_Dkb copyL1ToL0B_Dkb;
    CopyL1ToL0A_DK copyL1ToL0A_DK;
    CopyL1ToL0B_DK copyL1ToL0B_DK;
    TileMmadDkbg tileMmadDkbg;
    TileMmadDvb tileMmadDvb;
    TileMmadKkt tileMmadKkt;
    TileMmadDA1 tileMmadDA1;
    TileMmadDA2 tileMmadDA2;
    TileMmadDA5 tileMmadDA5;
    TileMmadDA6T tileMmadDA6T;
    TileMmadDkb tileMmadDkb;
    TileMmadDK tileMmadDK;

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
    AscendC::GlobalTensor<kType> gmDA4;
    AscendC::GlobalTensor<kType> gmDA5;
    AscendC::GlobalTensor<kType> gmDA6T;
    AscendC::GlobalTensor<kType> gmD;
    AscendC::GlobalTensor<kType> gmKbeta;
    AscendC::GlobalTensor<kType> gmDkb;
    AscendC::GlobalTensor<kType> gmDK;
    AscendC::GlobalTensor<kType> gmDebugDv;

    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreNum = AscendC::GetBlockNum();
    uint64_t groupSize = PrepareWyReprBwdGetGroupSize(tiling_);
    uint64_t windowIdx = 0;

    for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(tiling_.chunkNum); taskIdx += coreNum) {
        PrepareWyReprBwdTaskInfo task;
        PrepareWyReprBwdGetTaskInfo(cuSeqlens_, chunkIndices_, tiling_, taskIdx, task);
        GemmCoord shapeK{task.curChunkSize, K_DIM, task.curChunkSize};
        GemmCoord shapeV{task.curChunkSize, V_DIM, task.curChunkSize};
        GemmCoord shapeKkt{task.curChunkSize, task.curChunkSize, K_DIM};
        GemmCoord shapeDA1{task.curChunkSize, task.curChunkSize, K_DIM};
        GemmCoord shapeDA2{task.curChunkSize, task.curChunkSize, V_DIM};
        GemmCoord shapeM{task.curChunkSize, task.curChunkSize, task.curChunkSize};
        nextKResidentSlot_ = 0;
        cachedKResidentSlot_ = 0;
        cachedKResidentHk_ = static_cast<uint64_t>(-1);
        nextKktSlot_ = 0;
        cachedKktSlot_ = 0;
        cachedKktHk_ = static_cast<uint64_t>(-1);
        for (uint32_t slot = 0; slot < BUFFER_COUNT_4; ++slot) {
            kResidentSlotForSlot_[slot] = 0;
            kktSlotForSlot_[slot] = 0;
        }

        uint64_t hvTotal = static_cast<uint64_t>(tiling_.HV);
        for (uint64_t hvBase = 0; hvBase < hvTotal; hvBase += BUFFER_COUNT_2) {
            uint32_t headCnt = hvBase + BUFFER_COUNT_2 <= hvTotal ?
                                   BUFFER_COUNT_2 :
                                   static_cast<uint32_t>(hvTotal - hvBase);
            uint32_t windowStartSlot = static_cast<uint32_t>((windowIdx & 1U) * BUFFER_COUNT_2);
            nextKktSlot_ = windowStartSlot;
            cachedKktSlot_ = windowStartSlot;
            cachedKktHk_ = static_cast<uint64_t>(-1);

            // Stage0 fills both workspace slots before later stages consume the first head.
            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint32_t residentSlot = curSlot_ & 1U;
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
                    ++nextKktSlot_;
                }
                uint32_t kktSlot = cachedKktSlot_;
                kResidentSlotForSlot_[curSlot_] = cachedKResidentSlot_;
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

                uint32_t stage0AIdx = residentSlot;
                int32_t stage0AEvent = stage0AIdx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                uint32_t dwResidentIdx = residentSlot;
                int32_t dwResidentEvent =
                    dwResidentIdx == 0 ? EVENT_DW_RESIDENT_PING : EVENT_DW_RESIDENT_PONG;
                auto tensorL1ResidentAT = tla::MakeTensor(aResident[stage0AIdx], L1A_LAYOUT_AT, Arch::PositionL1{});
                auto tensorL1ResidentDWForDkbg =
                    tla::MakeTensor(dwResident[dwResidentIdx], L1B_LAYOUT_DW, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(stage0AEvent);
                copyGmToL1A_AT(tensorL1ResidentAT, blockATForDkbg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(stage0AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dwResidentEvent);
                copyGmToL1B_DW(tensorL1ResidentDWForDkbg, blockDw);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dwResidentEvent);

                uint32_t mActualDkbg = shapeK.m();
                uint32_t dkbgL0Idx = curL0_;
                int32_t dkbgL0AEvent = dkbgL0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                int32_t dkbgL0BEvent = dkbgL0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                int32_t dkbgL0ReadyEvent = dkbgL0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                auto layoutL0A_AT = tla::MakeLayout<kType, LayoutTagL0A_Dkbg>(mActualDkbg, shapeK.k());
                auto tensorL0A_AT = tla::MakeTensor(l0A[dkbgL0Idx], layoutL0A_AT, Arch::PositionL0A{});
                auto tensorTileL1A_AT =
                    GetTile(tensorL1ResidentAT, tla::MakeCoord(0, 0), tla::MakeShape(mActualDkbg, shapeK.k()));
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
#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dwResidentEvent);
#endif
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dkbgL0ReadyEvent);
                curL0_ ^= 1U;

                uint32_t duResidentIdx = residentSlot;
                int32_t duResidentEvent =
                    duResidentIdx == 0 ? EVENT_DU_RESIDENT_PING : EVENT_DU_RESIDENT_PONG;
                auto tensorL1A_ATForDU =
                    tla::MakeTensor(aResident[stage0AIdx], L1A_LAYOUT_AT_FOR_DU, Arch::PositionL1{});
                auto tensorL1ResidentDUForDvb =
                    tla::MakeTensor(duResident[duResidentIdx], L1B_LAYOUT_DU, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(duResidentEvent);
                copyGmToL1B_DU(tensorL1ResidentDUForDvb, blockDu);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(duResidentEvent);

                uint32_t dkbgL0CIdx = curL0C_;
                int32_t dkbgL0CEvent = GetL0CEvent(dkbgL0CIdx);
                auto layoutL0C_Dkbg = tla::MakeLayoutL0C(mActualDkbg, shapeK.n());
                auto tensorL0C_Dkbg = tla::MakeTensor(l0C[dkbgL0CIdx], layoutL0C_Dkbg, Arch::PositionL0C{});
                auto tensorTileL0C_Dkbg =
                    GetTile(tensorL0C_Dkbg, tla::MakeCoord(0, 0), tla::MakeShape(mActualDkbg, shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dkbgL0ReadyEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dkbgL0CEvent);
                tileMmadDkbg(tensorTileL0C_Dkbg, tensorL0A_AT, tensorL0B_DW, mActualDkbg, shapeK.n(),
                              shapeK.k(), true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbgL0AEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbgL0BEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dkbgL0CEvent);
                curL0C_ ^= 1U;

                uint32_t mActualDvb = shapeV.m();
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dkbgL0CEvent);
                copyL0CToGm_Dkbg(blockDkbg, tensorL0C_Dkbg, fixpipeUnitFlag);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dkbgL0CEvent);
                bool dvbL1BReadyConsumed = false;
                for (uint32_t nOffset = 0; nOffset < shapeV.n(); nOffset += L0_DVB_N_TILE) {
                    uint32_t curN = nOffset + L0_DVB_N_TILE > shapeV.n() ? shapeV.n() - nOffset : L0_DVB_N_TILE;
                    uint32_t dvbL0CIdx = curL0C_;
                    int32_t dvbL0CEvent = GetL0CEvent(dvbL0CIdx);
                    auto layoutL0C_Dvb = tla::MakeLayoutL0C(mActualDvb, curN);
                    auto tensorL0C_Dvb = tla::MakeTensor(l0C[dvbL0CIdx], layoutL0C_Dvb, Arch::PositionL0C{});
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
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dvbL0CEvent);
                        }
                        uint8_t dvbMmadUnitFlag = lastK ? 0b11 : 0b10;
                        tileMmadDvb(tensorTileL0C_Dvb, tensorL0A_ATForDU, tensorL0B_DU, mActualDvb, curN, curK,
                                    kOffset == 0, dvbMmadUnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dvbL0AEvent);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dvbL0BEvent);
                        if (lastK) {
                            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dvbL0CEvent);
                        }
                    }

                    curL0C_ ^= 1U;
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dvbL0CEvent);
                    copyL0CToGm_DvbN(blockDvbN, tensorL0C_Dvb, fixpipeUnitFlag);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dvbL0CEvent);
                }
#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(duResidentEvent);
#endif
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

                    uint32_t kktL0CIdx = curL0C_;
                    int32_t kktL0CEvent = GetL0CEvent(kktL0CIdx);
                    auto layoutL0C_Kkt = tla::MakeLayoutL0C(mActualKkt, shapeKkt.n());
                    auto tensorL0C_Kkt = tla::MakeTensor(l0C[kktL0CIdx], layoutL0C_Kkt, Arch::PositionL0C{});
                    auto tensorTileL0C_Kkt =
                        GetTile(tensorL0C_Kkt, tla::MakeCoord(0, 0), tla::MakeShape(mActualKkt, shapeKkt.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kktL0ReadyEvent);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kktL0CEvent);
                    tileMmadKkt(tensorTileL0C_Kkt, tensorL0A_K, tensorL0B_KT, mActualKkt, shapeKkt.n(),
                                shapeKkt.k(), true, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kktL0AEvent);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kktL0BEvent);
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kktL0CEvent);
                    curL0C_ ^= 1U;
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kktL0CEvent);
                    copyL0CToGm_Kkt(blockKkt, tensorL0C_Kkt, fixpipeUnitFlag);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kktL0CEvent);
                }

                curSlot_ ^= 1U;
            }

#if PREPARE_WY_REPR_BWD_DEBUG_STAGE1_VECTOR
            ++windowIdx;
            continue;
#endif

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint32_t residentSlot = curSlot_ & 1U;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE
                uint64_t hv = hvBase + headIdx;
                uint64_t valueBase = hv * tiling_.T + task.valueBos;
                gmDebugDv.SetGlobalBuffer((__gm__ kType *)dv_ + valueBase * V_DIM);
#endif

                gmKbg.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.kbgOffset));
                gmVb.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.vbOffset));
                gmDA1.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da1Offset));
                gmDA2.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da2Offset));

                auto tensorKbgT = tla::MakeTensor(gmKbg, layoutKbgT, Arch::PositionGM{});
                auto tensorVbT = tla::MakeTensor(gmVb, layoutVbT, Arch::PositionGM{});
                auto tensorDA1 = tla::MakeTensor(gmDA1, layoutDA1, Arch::PositionGM{});
                auto tensorDA2 = tla::MakeTensor(gmDA2, layoutDA2, Arch::PositionGM{});
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE
                auto tensorDebugDA1 = tla::MakeTensor(gmDebugDv, layoutDebugDA1, Arch::PositionGM{});
                auto tensorDebugDA2 = tla::MakeTensor(gmDebugDv, layoutDebugDA2, Arch::PositionGM{});
#endif
                uint8_t fixpipeUnitFlag = 0b11;

                auto blockKbgT =
                    GetTile(tensorKbgT, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA1.k(), shapeDA1.n()));
                auto blockDA1 =
                    GetTile(tensorDA1, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA1.m(), shapeDA1.n()));
                auto blockVbT =
                    GetTile(tensorVbT, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA2.k(), shapeDA2.n()));
                auto blockDA2 =
                    GetTile(tensorDA2, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA2.m(), shapeDA2.n()));
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE
                auto blockDebugDA1 =
                    GetTile(tensorDebugDA1, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA1.m(), shapeDA1.n()));
                auto blockDebugDA2 =
                    GetTile(tensorDebugDA2, tla::MakeCoord(0, 0), tla::MakeShape(shapeDA2.m(), shapeDA2.n()));
#endif

                CopyGmToL1B_DA1<decltype(blockKbgT)> copyGmToL1B_KbgT;
                CopyL0CToGm_DA1<decltype(blockDA1)> copyL0CToGm_DA1;
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE == 1
                CopyL0CToGm_DA1<decltype(blockDebugDA1)> copyL0CToGm_DebugDA1;
#endif
                CopyGmToL1B_DA2<decltype(blockVbT)> copyGmToL1B_VbT;
                CopyL0CToGm_DA2<decltype(blockDA2)> copyL0CToGm_DA2;
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE == 2
                CopyL0CToGm_DA2<decltype(blockDebugDA2)> copyL0CToGm_DebugDA2;
#endif

                Arch::CrossCoreWaitFlag(vecToCubeFlag_);

                uint32_t da1L1Idx = curL1_;
                int32_t da1L1AEvent = da1L1Idx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                uint32_t dwResidentIdx = residentSlot;
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

                uint32_t da1L0CIdx = curL0C_;
                int32_t da1L0CEvent = GetL0CEvent(da1L0CIdx);
                auto layoutL0C_DA1 = tla::MakeLayoutL0C(mActualDA1, shapeDA1.n());
                auto tensorL0C_DA1 = tla::MakeTensor(l0C[da1L0CIdx], layoutL0C_DA1, Arch::PositionL0C{});
                auto tensorTileL0C_DA1 =
                    GetTile(tensorL0C_DA1, tla::MakeCoord(0, 0), tla::MakeShape(mActualDA1, shapeDA1.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(da1L0ReadyEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(da1L0CEvent);
                tileMmadDA1(tensorTileL0C_DA1, tensorL0A_DWForDA1, tensorL0B_KbgT, mActualDA1, shapeDA1.n(),
                             shapeDA1.k(), true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da1L0AEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da1L0BEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(da1L0CEvent);
                curL0C_ ^= 1U;
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(da1L0CEvent);
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE == 1
                copyL0CToGm_DebugDA1(blockDebugDA1, tensorL0C_DA1, fixpipeUnitFlag);
#elif PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE == 0
                copyL0CToGm_DA1(blockDA1, tensorL0C_DA1, fixpipeUnitFlag);
#endif
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(da1L0CEvent);
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE == 1
                int32_t duResidentEventForSkipDA2 =
                    residentSlot == 0 ? EVENT_DU_RESIDENT_PING : EVENT_DU_RESIDENT_PONG;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(duResidentEventForSkipDA2);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                curSlot_ ^= 1U;
                continue;
#endif

                uint32_t da2L1Idx = curL1_;
                int32_t da2L1AEvent = da2L1Idx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                uint32_t duResidentIdxForDA2 = residentSlot;
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
                uint32_t da2L0CIdx = curL0C_;
                int32_t da2L0CEvent = GetL0CEvent(da2L0CIdx);
                auto layoutL0C_DA2 = tla::MakeLayoutL0C(mActualDA2, shapeDA2.n());
                auto tensorL0C_DA2 = tla::MakeTensor(l0C[da2L0CIdx], layoutL0C_DA2, Arch::PositionL0C{});
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
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(da2L0CEvent);
                    }
                    uint8_t da2MmadUnitFlag = lastK ? 0b11 : 0b10;
                    tileMmadDA2(tensorTileL0C_DA2, tensorL0A_DUForDA2, tensorL0B_VbT, mActualDA2, shapeDA2.n(),
                                curK, kOffset == 0, da2MmadUnitFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da2L0AEvent);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da2L0BEvent);
                    if (lastK) {
                        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(da2L0CEvent);
                    }
                }

                curL0C_ ^= 1U;
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(da2L0CEvent);
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE == 2
                copyL0CToGm_DebugDA2(blockDebugDA2, tensorL0C_DA2, fixpipeUnitFlag);
#else
                copyL0CToGm_DA2(blockDA2, tensorL0C_DA2, fixpipeUnitFlag);
#endif
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(da2L0CEvent);
#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                curSlot_ ^= 1U;
                continue;
#endif
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                curSlot_ ^= 1U;
            }

#if PREPARE_WY_REPR_BWD_DEBUG_DA12_CUBE
            ++windowIdx;
            continue;
#endif

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint32_t residentSlot = curSlot_ & 1U;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);

                gmDA4.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da4Offset));
                gmDA5.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da5Offset));

                auto tensorDA4 = tla::MakeTensor(gmDA4, layoutDA4, Arch::PositionGM{});
                auto tensorDA5 = tla::MakeTensor(gmDA5, layoutDA5, Arch::PositionGM{});
                uint8_t fixpipeUnitFlag = 0b11;

                auto blockDA4 =
                    GetTile(tensorDA4, tla::MakeCoord(0, 0), tla::MakeShape(shapeM.m(), shapeM.k()));
                auto blockDA5 =
                    GetTile(tensorDA5, tla::MakeCoord(0, 0), tla::MakeShape(shapeM.m(), shapeM.n()));

                CopyGmToL1A_DA5<decltype(blockDA4)> copyGmToL1A_DA4;
                CopyL0CToGm_DA5<decltype(blockDA5)> copyL0CToGm_DA5;

                Arch::CrossCoreWaitFlag(vecToCubeFlag_);
                int32_t da5FixToMte2Event =
                    residentSlot == 0 ? EVENT_FIX_TO_MTE2_PING : EVENT_FIX_TO_MTE2_PONG;

                uint32_t da5L1AIdx = curL1_;
                int32_t da5L1AEvent = da5L1AIdx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                auto tensorL1A_DA4 = tla::MakeTensor(l1Scratch[da5L1AIdx], L1A_LAYOUT_DA4, Arch::PositionL1{});
                uint32_t da5AResidentIdx = residentSlot;
                auto tensorL1ResidentATForDA5 =
                    tla::MakeTensor(aResident[da5AResidentIdx], L1B_LAYOUT_AT_FOR_DA5, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(da5L1AEvent);
                copyGmToL1A_DA4(tensorL1A_DA4, blockDA4);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(da5L1AEvent);

                uint32_t mActualDA5 = shapeM.m();
                uint32_t da5L0Idx = curL0_;
                int32_t da5L0AEvent = da5L0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                int32_t da5L0BEvent = da5L0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                int32_t da5L0ReadyEvent = da5L0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                auto layoutL0A_DA4 = tla::MakeLayout<kType, LayoutTagL0A_DA5>(mActualDA5, shapeM.k());
                auto tensorL0A_DA4 = tla::MakeTensor(l0A[da5L0Idx], layoutL0A_DA4, Arch::PositionL0A{});
                auto tensorTileL1A_DA4 =
                    GetTile(tensorL1A_DA4, tla::MakeCoord(0, 0), tla::MakeShape(mActualDA5, shapeM.k()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(da5L1AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(da5L0AEvent);
                copyL1ToL0A_DA5(tensorL0A_DA4, tensorTileL1A_DA4);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(da5L1AEvent);

                auto layoutL0B_ATForDA5 = tla::MakeLayout<kType, LayoutTagL0B_DA5>(shapeM.k(), shapeM.n());
                auto tensorL0B_ATForDA5 =
                    tla::MakeTensor(l0B[da5L0Idx], layoutL0B_ATForDA5, Arch::PositionL0B{});
                auto tensorTileL1B_ATForDA5 =
                    GetTile(tensorL1ResidentATForDA5, tla::MakeCoord(0, 0), tla::MakeShape(shapeM.k(), shapeM.n()));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(da5L0BEvent);
                copyL1ToL0B_DA5(tensorL0B_ATForDA5, tensorTileL1B_ATForDA5);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(da5L0ReadyEvent);
                curL0_ ^= 1U;

                uint32_t da5L0CIdx = curL0C_;
                int32_t da5L0CEvent = GetL0CEvent(da5L0CIdx);
                auto layoutL0C_DA5 = tla::MakeLayoutL0C(mActualDA5, shapeM.n());
                auto tensorL0C_DA5 = tla::MakeTensor(l0C[da5L0CIdx], layoutL0C_DA5, Arch::PositionL0C{});
                auto tensorTileL0C_DA5 =
                    GetTile(tensorL0C_DA5, tla::MakeCoord(0, 0), tla::MakeShape(mActualDA5, shapeM.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(da5L0ReadyEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(da5L0CEvent);
                tileMmadDA5(tensorTileL0C_DA5, tensorL0A_DA4, tensorL0B_ATForDA5, mActualDA5, shapeM.n(),
                             shapeM.k(), true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da5L0AEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da5L0BEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(da5L0CEvent);
                curL0C_ ^= 1U;
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(da5L0CEvent);
                copyL0CToGm_DA5(blockDA5, tensorL0C_DA5, fixpipeUnitFlag);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(da5L0CEvent);
                AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(da5FixToMte2Event);
                curSlot_ ^= 1U;
            }

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint32_t residentSlot = curSlot_ & 1U;
                uint64_t hv = hvBase + headIdx;
                uint64_t valueBase = hv * tiling_.T + task.valueBos;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);

                gmDA5.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da5Offset));
                gmDA6T.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.da6Offset));

                auto tensorDA5T = tla::MakeTensor(gmDA5, layoutDA5T, Arch::PositionGM{});
                auto tensorDA6T = tla::MakeTensor(gmDA6T, layoutDA6T, Arch::PositionGM{});
                uint8_t fixpipeUnitFlag = 0b11;

                auto blockDA5T =
                    GetTile(tensorDA5T, tla::MakeCoord(0, 0), tla::MakeShape(shapeM.m(), shapeM.k()));
                auto blockDA6T =
                    GetTile(tensorDA6T, tla::MakeCoord(0, 0), tla::MakeShape(shapeM.m(), shapeM.n()));

                CopyGmToL1A_DA6T<decltype(blockDA5T)> copyGmToL1A_DA5T;
                CopyL0CToGm_DA6T<decltype(blockDA6T)> copyL0CToGm_DA6T;
                int32_t da5FixToMte2Event =
                    residentSlot == 0 ? EVENT_FIX_TO_MTE2_PING : EVENT_FIX_TO_MTE2_PONG;

                uint32_t da6L1AIdx = curL1_;
                int32_t da6L1AEvent = da6L1AIdx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                auto tensorL1A_DA5T =
                    tla::MakeTensor(l1Scratch[da6L1AIdx], L1A_LAYOUT_DA5_T, Arch::PositionL1{});
                uint32_t da6AResidentIdx = residentSlot;
                auto tensorL1ResidentAForDA6T =
                    tla::MakeTensor(aResident[da6AResidentIdx], L1B_LAYOUT_A_FOR_DA6T, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(da5FixToMte2Event);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(da6L1AEvent);
                copyGmToL1A_DA5T(tensorL1A_DA5T, blockDA5T);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(da6L1AEvent);

                uint32_t mActualDA6T = shapeM.m();
                uint32_t da6L0Idx = curL0_;
                int32_t da6L0AEvent = da6L0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                int32_t da6L0BEvent = da6L0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                int32_t da6L0ReadyEvent = da6L0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                auto layoutL0A_DA5T = tla::MakeLayout<kType, LayoutTagL0A_DA6T>(mActualDA6T, shapeM.k());
                auto tensorL0A_DA5T = tla::MakeTensor(l0A[da6L0Idx], layoutL0A_DA5T, Arch::PositionL0A{});
                auto tensorTileL1A_DA5T =
                    GetTile(tensorL1A_DA5T, tla::MakeCoord(0, 0), tla::MakeShape(mActualDA6T, shapeM.k()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(da6L1AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(da6L0AEvent);
                copyL1ToL0A_DA6T(tensorL0A_DA5T, tensorTileL1A_DA5T);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(da6L1AEvent);

                auto layoutL0B_AForDA6T = tla::MakeLayout<kType, LayoutTagL0B_DA6T>(shapeM.k(), shapeM.n());
                auto tensorL0B_AForDA6T =
                    tla::MakeTensor(l0B[da6L0Idx], layoutL0B_AForDA6T, Arch::PositionL0B{});
                auto tensorTileL1B_AForDA6T =
                    GetTile(tensorL1ResidentAForDA6T, tla::MakeCoord(0, 0), tla::MakeShape(shapeM.k(), shapeM.n()));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(da6L0BEvent);
                copyL1ToL0B_DA6T(tensorL0B_AForDA6T, tensorTileL1B_AForDA6T);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(da6L0ReadyEvent);
                curL0_ ^= 1U;

                uint32_t da6L0CIdx = curL0C_;
                int32_t da6L0CEvent = GetL0CEvent(da6L0CIdx);
                auto layoutL0C_DA6T = tla::MakeLayoutL0C(mActualDA6T, shapeM.n());
                auto tensorL0C_DA6T = tla::MakeTensor(l0C[da6L0CIdx], layoutL0C_DA6T, Arch::PositionL0C{});
                auto tensorTileL0C_DA6T =
                    GetTile(tensorL0C_DA6T, tla::MakeCoord(0, 0), tla::MakeShape(mActualDA6T, shapeM.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(da6L0ReadyEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(da6L0CEvent);
                tileMmadDA6T(tensorTileL0C_DA6T, tensorL0A_DA5T, tensorL0B_AForDA6T, mActualDA6T, shapeM.n(),
                              shapeM.k(), true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da6L0AEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(da6L0BEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(da6L0CEvent);
                curL0C_ ^= 1U;
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(da6L0CEvent);
                copyL0CToGm_DA6T(blockDA6T, tensorL0C_DA6T, fixpipeUnitFlag);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(da6L0CEvent);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                curSlot_ ^= 1U;
            }

            curSlot_ = windowStartSlot;
            for (uint32_t headIdx = 0; headIdx < headCnt; ++headIdx) {
                uint64_t hv = hvBase + headIdx;
                uint64_t hk = hv / groupSize;
                GM_ADDR slotBase = PrepareWyReprBwdGetSlotBase(workspace_, coreIdx, curSlot_, tiling_);

                gmD.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dOffset));
                gmKbeta.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.kbetaOffset));
                gmDkb.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dkbOffset));
                gmDK.SetGlobalBuffer((__gm__ kType *)(slotBase + tiling_.dkOffset));

                auto tensorD = tla::MakeTensor(gmD, layoutD, Arch::PositionGM{});
                auto tensorDT = tla::MakeTensor(gmD, layoutDT, Arch::PositionGM{});
                auto tensorKbeta = tla::MakeTensor(gmKbeta, layoutKbeta, Arch::PositionGM{});
                auto tensorDkb = tla::MakeTensor(gmDkb, layoutDkb, Arch::PositionGM{});
                auto tensorDK = tla::MakeTensor(gmDK, layoutDK, Arch::PositionGM{});
                uint8_t fixpipeUnitFlag = 0b11;

                auto blockDT = GetTile(tensorDT, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.m(), shapeK.k()));
                auto blockKbeta =
                    GetTile(tensorKbeta, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.k(), shapeK.n()));
                auto blockDkb =
                    GetTile(tensorDkb, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.m(), shapeK.n()));
                auto blockD = GetTile(tensorD, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.m(), shapeK.k()));
                auto blockDK = GetTile(tensorDK, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.m(), shapeK.n()));

                CopyGmToL1A_Dkb<decltype(blockDT)> copyGmToL1A_DT;
                CopyL0CToGm_Dkb<decltype(blockDkb)> copyL0CToGm_Dkb;
                CopyGmToL1A_DK<decltype(blockD)> copyGmToL1A_D;
                CopyGmToL1B_DK<decltype(blockKbeta)> copyGmToL1B_Kbeta;
                CopyL0CToGm_DK<decltype(blockDK)> copyL0CToGm_DK;

                Arch::CrossCoreWaitFlag(vecToCubeFlag_);

                uint32_t dkbL1AIdx = curL1_;
                int32_t dkbL1AEvent = dkbL1AIdx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                uint32_t dkbKResidentIdx = kResidentSlotForSlot_[curSlot_];
                auto tensorL1A_DT = tla::MakeTensor(l1Scratch[dkbL1AIdx], L1A_LAYOUT_D_T, Arch::PositionL1{});
                auto tensorL1ResidentKForDkb =
                    tla::MakeTensor(kResident[dkbKResidentIdx], L1B_LAYOUT_K_FOR_DKB, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dkbL1AEvent);
                copyGmToL1A_DT(tensorL1A_DT, blockDT);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dkbL1AEvent);

                uint32_t mActualDkb = shapeK.m();
                uint32_t dkbL0Idx = curL0_;
                int32_t dkbL0AEvent = dkbL0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                int32_t dkbL0BEvent = dkbL0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                int32_t dkbL0ReadyEvent = dkbL0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                auto layoutL0A_DT = tla::MakeLayout<kType, LayoutTagL0A_Dkb>(mActualDkb, shapeK.k());
                auto tensorL0A_DT = tla::MakeTensor(l0A[dkbL0Idx], layoutL0A_DT, Arch::PositionL0A{});
                auto tensorTileL1A_DT =
                    GetTile(tensorL1A_DT, tla::MakeCoord(0, 0), tla::MakeShape(mActualDkb, shapeK.k()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dkbL1AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbL0AEvent);
                copyL1ToL0A_Dkb(tensorL0A_DT, tensorTileL1A_DT);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dkbL1AEvent);

                auto layoutL0B_KForDkb = tla::MakeLayout<kType, LayoutTagL0B_Dkb>(shapeK.k(), shapeK.n());
                auto tensorL0B_KForDkb =
                    tla::MakeTensor(l0B[dkbL0Idx], layoutL0B_KForDkb, Arch::PositionL0B{});
                auto tensorTileL1B_KForDkb =
                    GetTile(tensorL1ResidentKForDkb, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.k(), shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbL0BEvent);
                copyL1ToL0B_Dkb(tensorL0B_KForDkb, tensorTileL1B_KForDkb);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dkbL0ReadyEvent);
                curL0_ ^= 1U;

                uint32_t dkbL0CIdx = curL0C_;
                int32_t dkbL0CEvent = GetL0CEvent(dkbL0CIdx);
                auto layoutL0C_Dkb = tla::MakeLayoutL0C(mActualDkb, shapeK.n());
                auto tensorL0C_Dkb = tla::MakeTensor(l0C[dkbL0CIdx], layoutL0C_Dkb, Arch::PositionL0C{});
                auto tensorTileL0C_Dkb =
                    GetTile(tensorL0C_Dkb, tla::MakeCoord(0, 0), tla::MakeShape(mActualDkb, shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dkbL0ReadyEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dkbL0CEvent);
                tileMmadDkb(tensorTileL0C_Dkb, tensorL0A_DT, tensorL0B_KForDkb, mActualDkb, shapeK.n(),
                             shapeK.k(), true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbL0AEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbL0BEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dkbL0CEvent);
                curL0C_ ^= 1U;
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dkbL0CEvent);
                copyL0CToGm_Dkb(blockDkb, tensorL0C_Dkb, fixpipeUnitFlag);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dkbL0CEvent);

                uint32_t dkL1AIdx = curL1_;
                uint32_t dkL1BIdx = curL1_ ^ 1U;
                int32_t dkL1AEvent = dkL1AIdx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                int32_t dkL1BEvent = dkL1BIdx == 0 ? EVENT_L1_SCRATCH_PING : EVENT_L1_SCRATCH_PONG;
                auto tensorL1A_D = tla::MakeTensor(l1Scratch[dkL1AIdx], L1A_LAYOUT_D_FOR_DK, Arch::PositionL1{});
                auto tensorL1B_Kbeta =
                    tla::MakeTensor(l1Scratch[dkL1BIdx], L1B_LAYOUT_KBETA, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dkL1AEvent);
                copyGmToL1A_D(tensorL1A_D, blockD);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dkL1AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dkL1BEvent);
                copyGmToL1B_Kbeta(tensorL1B_Kbeta, blockKbeta);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dkL1BEvent);

                uint32_t mActualDK = shapeK.m();
                uint32_t dkL0Idx = curL0_;
                int32_t dkL0AEvent = dkL0Idx == 0 ? EVENT_L0A_PING : EVENT_L0A_PONG;
                int32_t dkL0BEvent = dkL0Idx == 0 ? EVENT_L0B_PING : EVENT_L0B_PONG;
                int32_t dkL0ReadyEvent = dkL0Idx == 0 ? EVENT_L0_READY_PING : EVENT_L0_READY_PONG;
                auto layoutL0A_D = tla::MakeLayout<kType, LayoutTagL0A_DK>(mActualDK, shapeK.k());
                auto tensorL0A_D = tla::MakeTensor(l0A[dkL0Idx], layoutL0A_D, Arch::PositionL0A{});
                auto tensorTileL1A_D =
                    GetTile(tensorL1A_D, tla::MakeCoord(0, 0), tla::MakeShape(mActualDK, shapeK.k()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dkL1AEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkL0AEvent);
                copyL1ToL0A_DK(tensorL0A_D, tensorTileL1A_D);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dkL1AEvent);

                auto layoutL0B_Kbeta = tla::MakeLayout<kType, LayoutTagL0B_DK>(shapeK.k(), shapeK.n());
                auto tensorL0B_Kbeta = tla::MakeTensor(l0B[dkL0Idx], layoutL0B_Kbeta, Arch::PositionL0B{});
                auto tensorTileL1B_Kbeta =
                    GetTile(tensorL1B_Kbeta, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.k(), shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dkL1BEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkL0BEvent);
                copyL1ToL0B_DK(tensorL0B_Kbeta, tensorTileL1B_Kbeta);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dkL1BEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dkL0ReadyEvent);
                curL0_ ^= 1U;

                uint32_t dkL0CIdx = curL0C_;
                int32_t dkL0CEvent = GetL0CEvent(dkL0CIdx);
                auto layoutL0C_DK = tla::MakeLayoutL0C(mActualDK, shapeK.n());
                auto tensorL0C_DK = tla::MakeTensor(l0C[dkL0CIdx], layoutL0C_DK, Arch::PositionL0C{});
                auto tensorTileL0C_DK =
                    GetTile(tensorL0C_DK, tla::MakeCoord(0, 0), tla::MakeShape(mActualDK, shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dkL0ReadyEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dkL0CEvent);
                tileMmadDK(tensorTileL0C_DK, tensorL0A_D, tensorL0B_Kbeta, mActualDK, shapeK.n(), shapeK.k(), true,
                            0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkL0AEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkL0BEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dkL0CEvent);
                curL0C_ ^= 1U;
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dkL0CEvent);
                copyL0CToGm_DK(blockDK, tensorL0C_DK, fixpipeUnitFlag);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dkL0CEvent);

                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                curSlot_ ^= 1U;
            }
            ++windowIdx;
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
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PING);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_PONG);
}

#endif // PREPARE_WY_REPR_BWD_CUBE_H
