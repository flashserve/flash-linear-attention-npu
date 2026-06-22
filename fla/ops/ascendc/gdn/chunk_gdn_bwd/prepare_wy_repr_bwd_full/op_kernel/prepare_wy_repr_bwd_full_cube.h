/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_full.h
 * \brief
 */

#ifndef PREPARE_WY_REPR_BWD_FULL_CUBE_H
#define PREPARE_WY_REPR_BWD_FULL_CUBE_H
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#include "prepare_wy_repr_bwd_full_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/arch/cross_core_sync.hpp"
using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <class ArchTag_, class L1TileShape_, class L0TileShape_, class L1TileShapeDvb_, class L0TileShapeDvb_,
          class TileCopyDk_, class TileCopyDkb_, class TileCopyDkbg_, class TileCopyDvb_, class TileCopyKKT_>
class PrepareWyReprBwdFullTla {
public:
    using ArchTag = ArchTag_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using L1TileShapeDvb = L1TileShapeDvb_;
    using L0TileShapeDvb = L0TileShapeDvb_;
    using TileCopyDk = TileCopyDk_;
    using TileCopyDkb = TileCopyDkb_;
    using TileCopyDkbg = TileCopyDkbg_;
    using TileCopyDvb = TileCopyDvb_;
    using TileCopyKKT = TileCopyKKT_;

    using ElementDA = typename TileCopyDk::ElementA;
    using LayoutDA = typename TileCopyDk::LayoutA;
    using ElementKbeta = typename TileCopyDk::ElementB;
    using LayoutKbeta = typename TileCopyDk::LayoutB;
    using ElementDk = ElementDA;
    using LayoutDk = typename TileCopyDk::LayoutC;

    using ElementDAT = typename TileCopyDkb::ElementA;
    using LayoutDAT = typename TileCopyDkb::LayoutA;
    using ElementK = typename TileCopyDkb::ElementB;
    using LayoutK = typename TileCopyDkb::LayoutB;
    using ElementDkb = ElementK;
    using LayoutDkb = typename TileCopyDkb::LayoutC;

    using ElementAT = typename TileCopyDkbg::ElementA;
    using LayoutAT = typename TileCopyDkbg::LayoutA;
    using ElementDw = typename TileCopyDkbg::ElementB;
    using LayoutDw = typename TileCopyDkbg::LayoutB;
    using ElementDkbg = ElementDw;
    using LayoutDkbg = typename TileCopyDkbg::LayoutC;

    using ElementDu = typename TileCopyDvb::ElementB;
    using LayoutDu = typename TileCopyDvb::LayoutB;
    using ElementDvb = ElementDu;
    using LayoutDvb = typename TileCopyDvb::LayoutC;

    using ElementKT = typename TileCopyKKT::ElementB;
    using LayoutKT = typename TileCopyKKT::LayoutB;
    using ElementKKT = ElementK;
    using LayoutKKT = typename TileCopyKKT::LayoutC;
    using ElementAccumulator = typename TileCopyDk::ElementAccumulator;

    using CopyL1ToL0A_DK = typename TileCopyDk::CopyL1ToL0A;
    using CopyL1ToL0B_DK = typename TileCopyDk::CopyL1ToL0B;
    using CopyL1ToL0A_Dkb = typename TileCopyDkb::CopyL1ToL0A;
    using CopyL1ToL0B_Dkb = typename TileCopyDkb::CopyL1ToL0B;
    using CopyL1ToL0A_Dkbg = typename TileCopyDkbg::CopyL1ToL0A;
    using CopyL1ToL0B_Dkbg = typename TileCopyDkbg::CopyL1ToL0B;
    using CopyL1ToL0A_Dvb = typename TileCopyDvb::CopyL1ToL0A;
    using CopyL1ToL0B_Dvb = typename TileCopyDvb::CopyL1ToL0B;
    using CopyL1ToL0A_KKT = typename TileCopyKKT::CopyL1ToL0A;
    using CopyL1ToL0B_KKT = typename TileCopyKKT::CopyL1ToL0B;

    using LayoutTagL1A_DK = typename TileCopyDk::LayoutTagL1A;
    using LayoutTagL1B_DK = typename TileCopyDk::LayoutTagL1B;
    using LayoutTagL0A_DK = typename TileCopyDk::LayoutTagL0A;
    using LayoutTagL0B_DK = typename TileCopyDk::LayoutTagL0B;
    using LayoutTagL1A_Dkb = typename TileCopyDkb::LayoutTagL1A;
    using LayoutTagL1B_Dkb = typename TileCopyDkb::LayoutTagL1B;
    using LayoutTagL0A_Dkb = typename TileCopyDkb::LayoutTagL0A;
    using LayoutTagL0B_Dkb = typename TileCopyDkb::LayoutTagL0B;
    using LayoutTagL1A_Dkbg = typename TileCopyDkbg::LayoutTagL1A;
    using LayoutTagL1B_Dkbg = typename TileCopyDkbg::LayoutTagL1B;
    using LayoutTagL0A_Dkbg = typename TileCopyDkbg::LayoutTagL0A;
    using LayoutTagL0B_Dkbg = typename TileCopyDkbg::LayoutTagL0B;
    using LayoutTagL1A_Dvb = typename TileCopyDvb::LayoutTagL1A;
    using LayoutTagL1B_Dvb = typename TileCopyDvb::LayoutTagL1B;
    using LayoutTagL0A_Dvb = typename TileCopyDvb::LayoutTagL0A;
    using LayoutTagL0B_Dvb = typename TileCopyDvb::LayoutTagL0B;
    using LayoutTagL1A_KKT = typename TileCopyKKT::LayoutTagL1A;
    using LayoutTagL1B_KKT = typename TileCopyKKT::LayoutTagL1B;
    using LayoutTagL0A_KKT = typename TileCopyKKT::LayoutTagL0A;
    using LayoutTagL0B_KKT = typename TileCopyKKT::LayoutTagL0B;

    static constexpr uint32_t TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t TILE_K = tla::get<2>(L0TileShape{});
    static constexpr uint32_t TILE_M_DVB = tla::get<0>(L0TileShapeDvb{});
    static constexpr uint32_t TILE_N_DVB = tla::get<1>(L0TileShapeDvb{});
    static constexpr uint32_t TILE_K_DVB = tla::get<2>(L0TileShapeDvb{});
    static constexpr auto L1A_LAYOUT_DK =
        tla::MakeLayout<ElementDA, LayoutTagL1A_DK>(tla::Int<TILE_M>{}, tla::Int<TILE_K>{});
    static constexpr auto L1B_LAYOUT_DK =
        tla::MakeLayout<ElementKbeta, LayoutTagL1B_DK>(tla::Int<TILE_K>{}, tla::Int<TILE_N>{});
    static constexpr auto L1A_LAYOUT_Dkb =
        tla::MakeLayout<ElementDAT, LayoutTagL1A_Dkb>(tla::Int<TILE_M>{}, tla::Int<TILE_K>{});
    static constexpr auto L1B_LAYOUT_Dkb =
        tla::MakeLayout<ElementK, LayoutTagL1B_Dkb>(tla::Int<TILE_K>{}, tla::Int<TILE_N>{});
    static constexpr auto L1A_LAYOUT_Dkbg =
        tla::MakeLayout<ElementAT, LayoutTagL1A_Dkbg>(tla::Int<TILE_M>{}, tla::Int<TILE_K>{});
    static constexpr auto L1B_LAYOUT_Dkbg =
        tla::MakeLayout<ElementDw, LayoutTagL1B_Dkbg>(tla::Int<TILE_K>{}, tla::Int<TILE_N>{});
    static constexpr auto L1A_LAYOUT_Dvb =
        tla::MakeLayout<ElementAT, LayoutTagL1A_Dvb>(tla::Int<TILE_M_DVB>{}, tla::Int<TILE_K_DVB>{});
    static constexpr auto L1B_LAYOUT_Dvb =
        tla::MakeLayout<ElementDu, LayoutTagL1B_Dvb>(tla::Int<TILE_K_DVB>{}, tla::Int<TILE_N_DVB>{});
    static constexpr auto L1A_LAYOUT_KKT =
        tla::MakeLayout<ElementK, LayoutTagL1A_KKT>(tla::Int<TILE_M>{}, tla::Int<TILE_K>{});
    static constexpr auto L1B_LAYOUT_KKT =
        tla::MakeLayout<ElementKT, LayoutTagL1B_KKT>(tla::Int<TILE_K>{}, tla::Int<TILE_N>{});

    static constexpr uint32_t L1A_TILE_SIZE_DK = TILE_M * TILE_K * sizeof(ElementDA);
    static constexpr uint32_t L1B_TILE_SIZE_DK = TILE_K * TILE_N * sizeof(ElementKbeta);
    static constexpr uint32_t L1A_TILE_SIZE_Dkb = TILE_M * TILE_K * sizeof(ElementDAT);
    static constexpr uint32_t L1B_TILE_SIZE_Dkb = TILE_K * TILE_N * sizeof(ElementK);
    static constexpr uint32_t L1A_TILE_SIZE_Dkbg = TILE_M * TILE_K * sizeof(ElementAT);
    static constexpr uint32_t L1B_TILE_SIZE_Dkbg = TILE_K * TILE_N * sizeof(ElementDw);
    static constexpr uint32_t L1A_TILE_SIZE_Dvb = TILE_M_DVB * TILE_K_DVB * sizeof(ElementAT);
    static constexpr uint32_t L1B_TILE_SIZE_Dvb = TILE_K_DVB * TILE_N_DVB * sizeof(ElementDu);
    static constexpr uint32_t L1A_TILE_SIZE_KKT = TILE_M * TILE_K * sizeof(ElementK);
    static constexpr uint32_t L1B_TILE_SIZE_KKT = TILE_K * TILE_N * sizeof(ElementKT);

    using TileMmadDK = Tile::TileMmadTla<ArchTag, ElementDA, LayoutTagL1A_DK>;
    using TileMmadDkb = Tile::TileMmadTla<ArchTag, ElementDAT, LayoutTagL1A_Dkb>;
    using TileMmadDkbg = Tile::TileMmadTla<ArchTag, ElementAT, LayoutTagL1A_Dkbg>;
    using TileMmadDvb = Tile::TileMmadTla<ArchTag, ElementAT, LayoutTagL1A_Dvb>;
    using TileMmadKKT = Tile::TileMmadTla<ArchTag, ElementK, LayoutTagL1A_KKT>;

    // AIC -> AIV has five ready events in one (chunk, head):
    //   1. Dkb  = dA^T @ K
    //   2. Dkbg = A^T @ dw
    //   3. DK   = dA @ Kbeta
    //   4. Dvb  = A^T @ du
    //   5. KKT  = K @ K^T
    // AIV -> AIC has one event: Kbeta = K * beta[:, None] is ready.
    Arch::CrossCoreFlagWithReverse<5> flagAicFinishStore{SYNC_FLAG_2, SYNC_FLAG_3};
    Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_FLAG_4, SYNC_FLAG_5};
    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR ptrKbeta;
        LayoutKbeta layoutKbeta;
        GM_ADDR ptrDA;
        LayoutDA layoutDA;
        GM_ADDR ptrDk;
        LayoutDk layoutDk;
        GM_ADDR ptrDAT;
        LayoutDAT layoutDAT;
        GM_ADDR ptrK;
        LayoutK layoutK;
        GM_ADDR ptrDkb;
        LayoutDkb layoutDkb;
        GM_ADDR ptrAT;
        LayoutAT layoutAT;
        GM_ADDR ptrDw;
        LayoutDw layoutDw;
        GM_ADDR ptrDkbg;
        LayoutDkbg layoutDkbg;
        GM_ADDR ptrDu;
        LayoutDu layoutDu;
        GM_ADDR ptrDvb;
        LayoutDvb layoutDvb;
        GM_ADDR ptrKT;
        LayoutKT layoutKT;
        GM_ADDR ptrKKT;
        LayoutKKT layoutKKT;
        GM_ADDR ptrCuSeqLens;
        GM_ADDR ptrChunkIndices;
        uint64_t chunkNum;
        uint64_t B = 1;
        uint64_t T = 32768;
        uint64_t HV = 32;
        uint64_t HK = 32;
        uint64_t K = 128;
        uint64_t V = 128;
        uint64_t chunkSize = 64;
        uint64_t stage = 2;
        uint64_t fusedKVecRow = 0;
        uint64_t fusedVVecRow = 0;
        uint64_t fusedKktVecRow = 0;
        uint64_t workspaceSlotSize = 0;
        uint64_t workspaceBufferCount = 1;
        uint64_t usedCoreNum = 0;

        // Methods
        CATLASS_DEVICE
        Params()
        {
        }

        CATLASS_DEVICE
        Params(GM_ADDR ptrptrKbeta_, LayoutKbeta layoutKbeta_, GM_ADDR ptrDA_, LayoutDA layoutDA_, GM_ADDR ptrDk_,
               LayoutDk layoutDk_, GM_ADDR ptrDAT_, LayoutDAT layoutDAT_, GM_ADDR ptrK_, LayoutK layoutK_,
               GM_ADDR ptrDkb_, LayoutDkb layoutDkb_, GM_ADDR ptrAT_, LayoutAT layoutAT_, GM_ADDR ptrDw_,
               LayoutDw layoutDw_, GM_ADDR ptrDkbg_, LayoutDkbg layoutDkbg_, GM_ADDR ptrDu_, LayoutDu layoutDu_,
               GM_ADDR ptrDvb_, LayoutDvb layoutDvb_, GM_ADDR ptrKT_, LayoutKT layoutKT_, GM_ADDR ptrKKT_,
               LayoutKKT layoutKKT_, GM_ADDR ptrCuSeqLens_, GM_ADDR ptrChunkIndices_, uint64_t chunkNum_, uint64_t B_,
               uint64_t T_, uint64_t HV_, uint64_t HK_, uint64_t K_, uint64_t V_, uint64_t BT_, uint64_t stage_,
               uint64_t fusedKVecRow_, uint64_t fusedVVecRow_, uint64_t fusedKktVecRow_,
               uint64_t workspaceSlotSize_, uint64_t workspaceBufferCount_, uint64_t usedCoreNum_)
            : ptrKbeta(ptrptrKbeta_), layoutKbeta(layoutKbeta_), ptrDA(ptrDA_), layoutDA(layoutDA_), ptrDk(ptrDk_),
              layoutDk(layoutDk_), ptrDAT(ptrDAT_), layoutDAT(layoutDAT_), ptrK(ptrK_), layoutK(layoutK_),
              ptrDkb(ptrDkb_), layoutDkb(layoutDkb_), ptrAT(ptrAT_), layoutAT(layoutAT_), ptrDw(ptrDw_),
              layoutDw(layoutDw_), ptrDkbg(ptrDkbg_), layoutDkbg(layoutDkbg_), ptrDu(ptrDu_), layoutDu(layoutDu_),
               ptrDvb(ptrDvb_), layoutDvb(layoutDvb_), ptrKT(ptrKT_), layoutKT(layoutKT_), ptrKKT(ptrKKT_),
               layoutKKT(layoutKKT_), ptrCuSeqLens(ptrCuSeqLens_), ptrChunkIndices(ptrChunkIndices_),
               chunkNum(chunkNum_), B(B_), T(T_), HV(HV_), HK(HK_), K(K_), V(V_), chunkSize(BT_), stage(stage_),
               fusedKVecRow(fusedKVecRow_), fusedVVecRow(fusedVVecRow_), fusedKktVecRow(fusedKktVecRow_),
               workspaceSlotSize(workspaceSlotSize_), workspaceBufferCount(workspaceBufferCount_),
              usedCoreNum(usedCoreNum_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    PrepareWyReprBwdFullTla()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Arch::Resource<ArchTag> resource;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreLoops = params.chunkNum;
        uint32_t bos = 0;
        uint32_t eos = 0;

        // These byte sizes define the same workspace slot layout as tiling.cpp
        // and vector.h.  Every AIC core owns workspaceBufferCount slots; the
        // slots are ping-ponged by taskIdx so AIC can start the next task while
        // AIV still consumes the previous task's intermediates.
        uint64_t kBytes = ((params.chunkSize * params.K * sizeof(ElementK) + ONE_BLOCK_32 - 1) / ONE_BLOCK_32) * ONE_BLOCK_32;
        uint64_t vBytes = ((params.chunkSize * params.V * sizeof(ElementK) + ONE_BLOCK_32 - 1) / ONE_BLOCK_32) * ONE_BLOCK_32;
        uint64_t maxKVBytes = kBytes > vBytes ? kBytes : vBytes;
        uint64_t kktBytes = ((params.chunkSize * params.chunkSize * sizeof(ElementK) + ONE_BLOCK_32 - 1) / ONE_BLOCK_32) * ONE_BLOCK_32;
        (void)kktBytes;

        AscendC::GlobalTensor<ElementDA> gmDA;
        AscendC::GlobalTensor<ElementDAT> gmDAT;
        AscendC::GlobalTensor<ElementAT> gmAT;
        AscendC::GlobalTensor<ElementK> gmK;
        AscendC::GlobalTensor<ElementKT> gmKT;
        AscendC::GlobalTensor<ElementKbeta> gmKbeta;
        AscendC::GlobalTensor<ElementDw> gmDw;
        AscendC::GlobalTensor<ElementDu> gmDu;
        AscendC::GlobalTensor<ElementDk> gmDK;
        AscendC::GlobalTensor<ElementDkb> gmDkb;
        AscendC::GlobalTensor<ElementDkbg> gmDkbg;
        AscendC::GlobalTensor<ElementDvb> gmDvb;
        AscendC::GlobalTensor<ElementKKT> gmKKT;
        uint64_t l1Offset = 0;
        auto l1ADkb = resource.l1Buf.template GetBufferByByte<ElementDAT>(l1Offset);
        l1Offset += L1A_TILE_SIZE_Dkb;
        auto l1BDkb = resource.l1Buf.template GetBufferByByte<ElementK>(l1Offset);
        l1Offset += L1B_TILE_SIZE_Dkb;
        auto l1ADkbg = resource.l1Buf.template GetBufferByByte<ElementAT>(l1Offset);
        l1Offset += L1A_TILE_SIZE_Dkbg;
        auto l1BDkbg = resource.l1Buf.template GetBufferByByte<ElementDw>(l1Offset);
        l1Offset += L1B_TILE_SIZE_Dkbg;
        auto l1ADK = resource.l1Buf.template GetBufferByByte<ElementDA>(l1Offset);
        l1Offset += L1A_TILE_SIZE_DK;
        auto l1BDK = resource.l1Buf.template GetBufferByByte<ElementKbeta>(l1Offset);
        l1Offset += L1B_TILE_SIZE_DK;
        auto l1ADvb = resource.l1Buf.template GetBufferByByte<ElementAT>(l1Offset);
        l1Offset += L1A_TILE_SIZE_Dvb;
        auto l1BDvb = resource.l1Buf.template GetBufferByByte<ElementDu>(l1Offset);
        l1Offset += L1B_TILE_SIZE_Dvb;
        auto l1AKKT = resource.l1Buf.template GetBufferByByte<ElementK>(l1Offset);
        l1Offset += L1A_TILE_SIZE_KKT;
        auto l1BKKT = resource.l1Buf.template GetBufferByByte<ElementKT>(l1Offset);

        auto l0ADkb = resource.l0ABuf.template GetBufferByByte<ElementDAT>(0);
        auto l0BDkb = resource.l0BBuf.template GetBufferByByte<ElementK>(0);
        auto l0ADkbg = resource.l0ABuf.template GetBufferByByte<ElementAT>(0);
        auto l0BDkbg = resource.l0BBuf.template GetBufferByByte<ElementDw>(0);
        auto l0ADK = resource.l0ABuf.template GetBufferByByte<ElementDA>(0);
        auto l0BDK = resource.l0BBuf.template GetBufferByByte<ElementKbeta>(0);
        auto l0ADvb = resource.l0ABuf.template GetBufferByByte<ElementAT>(0);
        auto l0BDvb = resource.l0BBuf.template GetBufferByByte<ElementDu>(0);
        auto l0AKKT = resource.l0ABuf.template GetBufferByByte<ElementK>(0);
        auto l0BKKT = resource.l0BBuf.template GetBufferByByte<ElementKT>(0);
        auto l0CBuf = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);

        CopyL1ToL0A_Dkb copyL1ToL0A_Dkb;
        CopyL1ToL0B_Dkb copyL1ToL0B_Dkb;
        CopyL1ToL0A_Dkbg copyL1ToL0A_Dkbg;
        CopyL1ToL0B_Dkbg copyL1ToL0B_Dkbg;
        CopyL1ToL0A_DK copyL1ToL0A_DK;
        CopyL1ToL0B_DK copyL1ToL0B_DK;
        CopyL1ToL0A_Dvb copyL1ToL0A_Dvb;
        CopyL1ToL0B_Dvb copyL1ToL0B_Dvb;
        CopyL1ToL0A_KKT copyL1ToL0A_KKT;
        CopyL1ToL0B_KKT copyL1ToL0B_KKT;
        TileMmadDkb tileMmadDkb;
        TileMmadDkbg tileMmadDkbg;
        TileMmadDK tileMmadDK;
        TileMmadDvb tileMmadDvb;
        TileMmadKKT tileMmadKKT;
        constexpr int32_t EVENT_L1A = 0;
        constexpr int32_t EVENT_L1B = 1;
        constexpr int32_t EVENT_L0A = 0;
        constexpr int32_t EVENT_L0B = 1;
        constexpr int32_t EVENT_L0C = 0;
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);

        // Outer loop slices chunks across AIC cores.  Inner loop traverses all
        // heads for the current chunk, so all AIC/AIV stages are completed for
        // one (chunk, head) before moving to the next head.
        uint64_t taskIdx = 0;
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.HV, params.T,
                           params.chunkSize, loopIdx, bos, eos);
            uint32_t curChunkSize = eos - bos;
            GemmCoord shapeK{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
            GemmCoord shapeV{curChunkSize, static_cast<uint32_t>(params.V), curChunkSize};
            GemmCoord shapeKKT{curChunkSize, curChunkSize, static_cast<uint32_t>(params.K)};
            // GQA: value heads are grouped onto key heads, so k/dk use HK layout
            // while A/dA/dw/du/beta/g/dv/dbeta/dg keep HV layout.
            uint64_t keyBos = bos;
            if (params.ptrCuSeqLens == nullptr && params.HV != params.HK) {
                uint64_t batchIdx = bos / (params.HV * params.T);
                uint64_t timeBos = bos - batchIdx * params.HV * params.T;
                keyBos = batchIdx * params.HK * params.T + timeBos;
            }
            uint64_t headRatio = params.HV / params.HK;

            for (int h = 0; h < params.HV; h++) {
                uint64_t kh = h / headRatio;
                uint64_t valueBase = h * params.T + bos;
                uint64_t keyBase = kh * params.T + keyBos;
                // Per-task workspace layout:
                //   slotDK        : DK   = dA @ Kbeta
                //   slotDkb       : Dkb  = dA^T @ K
                //   slotDkbg      : Dkbg = A^T @ dw
                //   slotKbetaDvb  : first Kbeta, later reused as Dvb = A^T @ du
                //   slotKKT       : KKT  = K @ K^T
                uint64_t workspaceBufferIdx = taskIdx % params.workspaceBufferCount;
                uint64_t slotBaseBytes =
                    (coreIdx * params.workspaceBufferCount + workspaceBufferIdx) * params.workspaceSlotSize;
                uint64_t slotDK = slotBaseBytes / sizeof(ElementK);
                uint64_t slotDkb = (slotBaseBytes + kBytes) / sizeof(ElementK);
                uint64_t slotDkbg = (slotBaseBytes + 2 * kBytes) / sizeof(ElementK);
                uint64_t slotKbetaDvb = (slotBaseBytes + 3 * kBytes) / sizeof(ElementK);
                uint64_t slotKKT = (slotBaseBytes + 3 * kBytes + maxKVBytes) / sizeof(ElementK);
                ++taskIdx;

                gmDK.SetGlobalBuffer((__gm__ ElementDk *)params.ptrKbeta + slotDK);
                gmDkb.SetGlobalBuffer((__gm__ ElementDkb *)params.ptrKbeta + slotDkb);
                gmDkbg.SetGlobalBuffer((__gm__ ElementDkbg *)params.ptrKbeta + slotDkbg);
                gmKbeta.SetGlobalBuffer((__gm__ ElementKbeta *)params.ptrKbeta + slotKbetaDvb);
                gmDvb.SetGlobalBuffer((__gm__ ElementDvb *)params.ptrKbeta + slotKbetaDvb);
                gmKKT.SetGlobalBuffer((__gm__ ElementKKT *)params.ptrKbeta + slotKKT);

                gmDA.SetGlobalBuffer((__gm__ ElementDA *)params.ptrDA + valueBase * params.chunkSize);
                gmDAT.SetGlobalBuffer((__gm__ ElementDAT *)params.ptrDAT + valueBase * params.chunkSize);
                gmAT.SetGlobalBuffer((__gm__ ElementAT *)params.ptrAT + valueBase * params.chunkSize);
                gmK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK + keyBase * params.K);
                gmKT.SetGlobalBuffer((__gm__ ElementKT *)params.ptrKT + keyBase * params.K);
                gmDw.SetGlobalBuffer((__gm__ ElementDw *)params.ptrDw + keyBase * params.K);
                gmDu.SetGlobalBuffer((__gm__ ElementDu *)params.ptrDu + valueBase * params.V);

                auto tensorDA = tla::MakeTensor(gmDA, params.layoutDA, Arch::PositionGM{});
                auto tensorDAT = tla::MakeTensor(gmDAT, params.layoutDAT, Arch::PositionGM{});
                auto tensorAT = tla::MakeTensor(gmAT, params.layoutAT, Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
                auto tensorKT = tla::MakeTensor(gmKT, params.layoutKT, Arch::PositionGM{});
                auto tensorKbeta = tla::MakeTensor(gmKbeta, params.layoutKbeta, Arch::PositionGM{});
                auto tensorDw = tla::MakeTensor(gmDw, params.layoutDw, Arch::PositionGM{});
                auto tensorDu = tla::MakeTensor(gmDu, params.layoutDu, Arch::PositionGM{});
                auto tensorDK = tla::MakeTensor(gmDK, params.layoutDk, Arch::PositionGM{});
                auto tensorDkb = tla::MakeTensor(gmDkb, params.layoutDkb, Arch::PositionGM{});
                auto tensorDkbg = tla::MakeTensor(gmDkbg, params.layoutDkbg, Arch::PositionGM{});
                auto tensorDvb = tla::MakeTensor(gmDvb, params.layoutDvb, Arch::PositionGM{});
                auto tensorKKT = tla::MakeTensor(gmKKT, params.layoutKKT, Arch::PositionGM{});

                // Stage C1 prepares the first two independent GEMMs:
                //   Dkb  = dA^T @ K
                //   Dkbg = A^T @ dw
                // They do not depend on Kbeta, so AIC can compute them while
                // AIV is generating Kbeta.
                auto tensorBlockDAT = GetTile(tensorDAT, tla::MakeCoord(0, 0),
                                               tla::MakeShape(shapeK.m(), shapeK.k()));
                auto tensorBlockKForDkb = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(shapeK.k(), shapeK.n()));
                auto tensorBlockDkb = GetTile(tensorDkb, tla::MakeCoord(0, 0),
                                               tla::MakeShape(shapeK.m(), shapeK.n()));
                auto tensorBlockATForDkbg = GetTile(tensorAT, tla::MakeCoord(0, 0),
                                                   tla::MakeShape(shapeK.m(), shapeK.k()));
                auto tensorBlockDw = GetTile(tensorDw, tla::MakeCoord(0, 0),
                                             tla::MakeShape(shapeK.k(), shapeK.n()));
                auto tensorBlockDkbg = GetTile(tensorDkbg, tla::MakeCoord(0, 0),
                                               tla::MakeShape(shapeK.m(), shapeK.n()));
                auto tensorBlockDA = GetTile(tensorDA, tla::MakeCoord(0, 0),
                                             tla::MakeShape(shapeK.m(), shapeK.k()));
                auto tensorBlockKbeta = GetTile(tensorKbeta, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(shapeK.k(), shapeK.n()));
                auto tensorBlockDK = GetTile(tensorDK, tla::MakeCoord(0, 0),
                                               tla::MakeShape(shapeK.m(), shapeK.n()));
                using CopyGmToL1A_Dkb = typename TileCopyDkb::template CopyGmToL1A<decltype(tensorBlockDAT)>;
                using CopyGmToL1B_Dkb = typename TileCopyDkb::template CopyGmToL1B<decltype(tensorBlockKForDkb)>;
                using CopyGmToL1A_DK = typename TileCopyDk::template CopyGmToL1A<decltype(tensorBlockDA)>;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
                using CopyL0CToGm_Dkb = typename TileCopyDkb::template CopyL0CToDst<decltype(tensorBlockDkb)>;
#else
                using CopyL0CToGm_Dkb = typename TileCopyDkb::template CopyL0CToGm<decltype(tensorBlockDkb)>;
#endif
                using CopyGmToL1A_Dkbg = typename TileCopyDkbg::template CopyGmToL1A<decltype(tensorBlockATForDkbg)>;
                using CopyGmToL1B_Dkbg = typename TileCopyDkbg::template CopyGmToL1B<decltype(tensorBlockDw)>;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
                using CopyL0CToGm_Dkbg = typename TileCopyDkbg::template CopyL0CToDst<decltype(tensorBlockDkbg)>;
#else
                using CopyL0CToGm_Dkbg = typename TileCopyDkbg::template CopyL0CToGm<decltype(tensorBlockDkbg)>;
#endif
                CopyGmToL1A_Dkb copyGmToL1A_Dkb;
                CopyGmToL1B_Dkb copyGmToL1B_Dkb;
                CopyL0CToGm_Dkb copyL0CToGm_Dkb;
                CopyGmToL1A_DK copyGmToL1A_DK;
                CopyGmToL1A_Dkbg copyGmToL1A_Dkbg;
                CopyGmToL1B_Dkbg copyGmToL1B_Dkbg;
                CopyL0CToGm_Dkbg copyL0CToGm_Dkbg;
                auto tensorL1A_Dkb = tla::MakeTensor(l1ADkb, L1A_LAYOUT_Dkb, Arch::PositionL1{});
                auto tensorL1B_Dkb = tla::MakeTensor(l1BDkb, L1B_LAYOUT_Dkb, Arch::PositionL1{});
                auto tensorL1A_Dkbg = tla::MakeTensor(l1ADkbg, L1A_LAYOUT_Dkbg, Arch::PositionL1{});
                auto tensorL1B_Dkbg = tla::MakeTensor(l1BDkbg, L1B_LAYOUT_Dkbg, Arch::PositionL1{});

                // Dkb dataflow:
                //   GM(dA^T, K) -> L1 -> L0A/L0B -> MMAD -> L0C -> GM(slotDkb)
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                copyGmToL1A_Dkb(tensorL1A_Dkb, tensorBlockDAT);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                copyGmToL1B_Dkb(tensorL1B_Dkb, tensorBlockKForDkb);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                uint32_t mActualDkb = shapeK.m();
                if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
                    if (mActualDkb == 1) {
                        mActualDkb = 16;
                    }
                }
                auto layoutL0A_Dkb = tla::MakeLayout<ElementDAT, LayoutTagL0A_Dkb>(mActualDkb, shapeK.k());
                auto tensorL0A_Dkb = tla::MakeTensor(l0ADkb, layoutL0A_Dkb, Arch::PositionL0A{});
                auto tensorTileL1A_Dkb = GetTile(tensorL1A_Dkb, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(mActualDkb, shapeK.k()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                copyL1ToL0A_Dkb(tensorL0A_Dkb, tensorTileL1A_Dkb);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                auto layoutL0B_Dkb = tla::MakeLayout<ElementK, LayoutTagL0B_Dkb>(shapeK.k(), shapeK.n());
                auto tensorL0B_Dkb = tla::MakeTensor(l0BDkb, layoutL0B_Dkb, Arch::PositionL0B{});
                auto tensorTileL1B_Dkb = GetTile(tensorL1B_Dkb, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(shapeK.k(), shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                copyL1ToL0B_Dkb(tensorL0B_Dkb, tensorTileL1B_Dkb);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                copyGmToL1A_Dkbg(tensorL1A_Dkbg, tensorBlockATForDkbg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                copyGmToL1B_Dkbg(tensorL1B_Dkbg, tensorBlockDw);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                auto layoutL0C_Dkb = tla::MakeLayoutL0C(mActualDkb, shapeK.n());
                auto tensorL0C_Dkb = tla::MakeTensor(l0CBuf, layoutL0C_Dkb, Arch::PositionL0C{});
                auto tensorTileL0C_Dkb = GetTile(tensorL0C_Dkb, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(mActualDkb, shapeK.n()));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                tileMmadDkb(tensorTileL0C_Dkb, tensorL0A_Dkb, tensorL0B_Dkb, true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);

                {
                    // While Dkb is in MMAD/copyout, load A^T and dw for Dkbg.
                    // Dkbg dataflow:
                    //   GM(A^T, dw) -> L1 -> L0A/L0B -> MMAD -> L0C -> GM(slotDkbg)
                    uint32_t mActual = shapeK.m();
                    if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
                        if (mActual == 1) {
                            mActual = 16;
                        }
                    }
                    auto layoutL0A = tla::MakeLayout<ElementAT, LayoutTagL0A_Dkbg>(mActual, shapeK.k());
                    auto tensorL0A = tla::MakeTensor(l0ADkbg, layoutL0A, Arch::PositionL0A{});
                    auto tensorTileL1A = GetTile(tensorL1A_Dkbg, tla::MakeCoord(0, 0), tla::MakeShape(mActual, shapeK.k()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                    copyL1ToL0A_Dkbg(tensorL0A, tensorTileL1A);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                    auto layoutL0B = tla::MakeLayout<ElementDw, LayoutTagL0B_Dkbg>(shapeK.k(), shapeK.n());
                    auto tensorL0B = tla::MakeTensor(l0BDkbg, layoutL0B, Arch::PositionL0B{});
                    auto tensorTileL1B = GetTile(tensorL1B_Dkbg, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.k(), shapeK.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    copyL1ToL0B_Dkbg(tensorL0B, tensorTileL1B);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    copyL0CToGm_Dkb(tensorBlockDkb, tensorL0C_Dkb, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    // Ready event #1: AIV can start Dkb-related dbeta and
                    // keep Dkb*beta in UB while AIC computes Dkbg.
                    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                    auto layoutL0C = tla::MakeLayoutL0C(mActual, shapeK.n());
                    auto tensorL0C = tla::MakeTensor(l0CBuf, layoutL0C, Arch::PositionL0C{});
                    auto tensorTileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, shapeK.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    tileMmadDkbg(tensorTileL0C, tensorL0A, tensorL0B, true, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    copyL0CToGm_Dkbg(tensorBlockDkbg, tensorL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    // Ready event #2: AIV can now consume Dkbg and finish the
                    // Dkbg-related dk/dbeta/dg partials.
                    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                }

                // Wait for AIV to produce Kbeta = K * beta[:, None] into
                // slotKbetaDvb.  DK depends on this vector-side result.
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(flagAivFinishStore);

                auto tensorBlockATForDvb = GetTile(tensorAT, tla::MakeCoord(0, 0),
                                                   tla::MakeShape(shapeV.m(), shapeV.k()));
                auto tensorBlockDu = GetTile(tensorDu, tla::MakeCoord(0, 0),
                                              tla::MakeShape(shapeV.k(), shapeV.n()));
                auto tensorBlockDvb = GetTile(tensorDvb, tla::MakeCoord(0, 0),
                                              tla::MakeShape(shapeV.m(), shapeV.n()));
                using CopyGmToL1A_Dvb = typename TileCopyDvb::template CopyGmToL1A<decltype(tensorBlockATForDvb)>;
                using CopyGmToL1B_Dvb = typename TileCopyDvb::template CopyGmToL1B<decltype(tensorBlockDu)>;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
                using CopyL0CToGm_Dvb = typename TileCopyDvb::template CopyL0CToDst<decltype(tensorBlockDvb)>;
#else
                using CopyL0CToGm_Dvb = typename TileCopyDvb::template CopyL0CToGm<decltype(tensorBlockDvb)>;
#endif
                CopyGmToL1A_Dvb copyGmToL1A_Dvb;
                CopyGmToL1B_Dvb copyGmToL1B_Dvb;
                CopyL0CToGm_Dvb copyL0CToGm_Dvb;
                auto tensorL1A_Dvb = tla::MakeTensor(l1ADvb, L1A_LAYOUT_Dvb, Arch::PositionL1{});
                auto tensorL1B_Dvb = tla::MakeTensor(l1BDvb, L1B_LAYOUT_Dvb, Arch::PositionL1{});
                {
                    // Stage C2a:
                    //   DK = dA @ Kbeta
                    // Dataflow: GM(dA, Kbeta workspace) -> L1 -> L0 -> MMAD
                    // -> GM(slotDK).  This is the first dk term.
                    using CopyGmToL1B_DK = typename TileCopyDk::template CopyGmToL1B<decltype(tensorBlockKbeta)>;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
                    using CopyL0CToGm_DK = typename TileCopyDk::template CopyL0CToDst<decltype(tensorBlockDK)>;
#else
                    using CopyL0CToGm_DK = typename TileCopyDk::template CopyL0CToGm<decltype(tensorBlockDK)>;
#endif
                    CopyGmToL1B_DK copyGmToL1B_DK;
                    CopyL0CToGm_DK copyL0CToGm_DK;
                    auto tensorL1A_DK = tla::MakeTensor(l1ADK, L1A_LAYOUT_DK, Arch::PositionL1{});
                    auto tensorL1B = tla::MakeTensor(l1BDK, L1B_LAYOUT_DK, Arch::PositionL1{});
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                    copyGmToL1A_DK(tensorL1A_DK, tensorBlockDA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                    copyGmToL1B_DK(tensorL1B, tensorBlockKbeta);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                    uint32_t mActual = shapeK.m();
                    if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
                        if (mActual == 1) {
                            mActual = 16;
                        }
                    }
                    auto layoutL0A = tla::MakeLayout<ElementDA, LayoutTagL0A_DK>(mActual, shapeK.k());
                    auto tensorL0A = tla::MakeTensor(l0ADK, layoutL0A, Arch::PositionL0A{});
                    auto tensorTileL1A = GetTile(tensorL1A_DK, tla::MakeCoord(0, 0), tla::MakeShape(mActual, shapeK.k()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                    copyL1ToL0A_DK(tensorL0A, tensorTileL1A);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                    auto layoutL0B = tla::MakeLayout<ElementKbeta, LayoutTagL0B_DK>(shapeK.k(), shapeK.n());
                    auto tensorL0B = tla::MakeTensor(l0BDK, layoutL0B, Arch::PositionL0B{});
                    auto tensorTileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(shapeK.k(), shapeK.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    copyL1ToL0B_DK(tensorL0B, tensorTileL1B);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                    auto layoutL0C = tla::MakeLayoutL0C(mActual, shapeK.n());
                    auto tensorL0C = tla::MakeTensor(l0CBuf, layoutL0C, Arch::PositionL0C{});
                    auto tensorTileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, shapeK.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    tileMmadDK(tensorTileL0C, tensorL0A, tensorL0B, true, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    copyL0CToGm_DK(tensorBlockDK, tensorL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                }

                // Ready event #3: DK is available; AIV can finish dk writeback.
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);

                auto tensorBlockKForKKT = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(shapeKKT.m(), shapeKKT.k()));
                auto tensorBlockKT = GetTile(tensorKT, tla::MakeCoord(0, 0),
                                             tla::MakeShape(shapeKKT.k(), shapeKKT.n()));
                auto tensorBlockKKT = GetTile(tensorKKT, tla::MakeCoord(0, 0),
                                             tla::MakeShape(shapeKKT.m(), shapeKKT.n()));
                using CopyGmToL1A_KKT = typename TileCopyKKT::template CopyGmToL1A<decltype(tensorBlockKForKKT)>;
                using CopyGmToL1B_KKT = typename TileCopyKKT::template CopyGmToL1B<decltype(tensorBlockKT)>;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
                using CopyL0CToGm_KKT = typename TileCopyKKT::template CopyL0CToDst<decltype(tensorBlockKKT)>;
#else
                using CopyL0CToGm_KKT = typename TileCopyKKT::template CopyL0CToGm<decltype(tensorBlockKKT)>;
#endif
                CopyGmToL1A_KKT copyGmToL1A_KKT;
                CopyGmToL1B_KKT copyGmToL1B_KKT;
                CopyL0CToGm_KKT copyL0CToGm_KKT;
                auto tensorL1A_KKT = tla::MakeTensor(l1AKKT, L1A_LAYOUT_KKT, Arch::PositionL1{});
                auto tensorL1B_KKT = tla::MakeTensor(l1BKKT, L1B_LAYOUT_KKT, Arch::PositionL1{});
                uint32_t mActualDvb = shapeV.m();
                if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
                    if (mActualDvb == 1) {
                        mActualDvb = 16;
                    }
                }
                auto layoutL0C_Dvb = tla::MakeLayoutL0C(mActualDvb, shapeV.n());
                auto tensorL0C_Dvb = tla::MakeTensor(l0CBuf, layoutL0C_Dvb, Arch::PositionL0C{});
                {
                    // Stage C2b starts Dvb and preloads KKT operands while Dvb
                    // MMAD is running:
                    //   Dvb = A^T @ du
                    // AIV later computes dv = Dvb * beta[:, None] and
                    // dbeta += reduce(Dvb * V).
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                    copyGmToL1A_Dvb(tensorL1A_Dvb, tensorBlockATForDvb);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                    copyGmToL1B_Dvb(tensorL1B_Dvb, tensorBlockDu);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                    auto layoutL0A = tla::MakeLayout<ElementAT, LayoutTagL0A_Dvb>(mActualDvb, shapeV.k());
                    auto tensorL0A = tla::MakeTensor(l0ADvb, layoutL0A, Arch::PositionL0A{});
                    auto tensorTileL1A = GetTile(tensorL1A_Dvb, tla::MakeCoord(0, 0), tla::MakeShape(mActualDvb, shapeV.k()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                    copyL1ToL0A_Dvb(tensorL0A, tensorTileL1A);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                    auto layoutL0B = tla::MakeLayout<ElementDu, LayoutTagL0B_Dvb>(shapeV.k(), shapeV.n());
                    auto tensorL0B = tla::MakeTensor(l0BDvb, layoutL0B, Arch::PositionL0B{});
                    auto tensorTileL1B = GetTile(tensorL1B_Dvb, tla::MakeCoord(0, 0), tla::MakeShape(shapeV.k(), shapeV.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    copyL1ToL0B_Dvb(tensorL0B, tensorTileL1B);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                    copyGmToL1A_KKT(tensorL1A_KKT, tensorBlockKForKKT);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                    copyGmToL1B_KKT(tensorL1B_KKT, tensorBlockKT);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                    auto tensorTileL0C = GetTile(tensorL0C_Dvb, tla::MakeCoord(0, 0), tla::MakeShape(mActualDvb, shapeV.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    tileMmadDvb(tensorTileL0C, tensorL0A, tensorL0B, true, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                }

                {
                    // Stage C2c:
                    //   KKT = K @ K^T
                    // Dvb is copied out first, then KKT is computed into the
                    // dedicated KKT slot for AIV's dg row/column reductions.
                    uint32_t mActual = shapeKKT.m();
                    if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
                        if (mActual == 1) {
                            mActual = 16;
                        }
                    }
                    auto layoutL0A = tla::MakeLayout<ElementK, LayoutTagL0A_KKT>(mActual, shapeKKT.k());
                    auto tensorL0A = tla::MakeTensor(l0AKKT, layoutL0A, Arch::PositionL0A{});
                    auto tensorTileL1A = GetTile(tensorL1A_KKT, tla::MakeCoord(0, 0), tla::MakeShape(mActual, shapeKKT.k()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1A);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                    copyL1ToL0A_KKT(tensorL0A, tensorTileL1A);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
                    auto layoutL0B = tla::MakeLayout<ElementKT, LayoutTagL0B_KKT>(shapeKKT.k(), shapeKKT.n());
                    auto tensorL0B = tla::MakeTensor(l0BKKT, layoutL0B, Arch::PositionL0B{});
                    auto tensorTileL1B = GetTile(tensorL1B_KKT, tla::MakeCoord(0, 0), tla::MakeShape(shapeKKT.k(), shapeKKT.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_L1B);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    copyL1ToL0B_KKT(tensorL0B, tensorTileL1B);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    copyL0CToGm_Dvb(tensorBlockDvb, tensorL0C_Dvb, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    // Ready event #4: Dvb is available.
                    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                    auto layoutL0C = tla::MakeLayoutL0C(mActual, shapeKKT.n());
                    auto tensorL0C = tla::MakeTensor(l0CBuf, layoutL0C, Arch::PositionL0C{});
                    auto tensorTileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, shapeKKT.n()));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    tileMmadKKT(tensorTileL0C, tensorL0A, tensorL0B, true, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
                    copyL0CToGm_KKT(tensorBlockKKT, tensorL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
                }

                // Ready event #5: KKT is available; AIV can finish dg.
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
    }
};
} // namespace Catlass::Gemm::Kernel

template <typename kType, typename betaType>
class PrepareWyReprBwdFullProcess {
public:
    /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdFullProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_,
                                                  GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                  GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_,
                                                  GM_ADDR dg_, GM_ADDR workspace_);

    __aicore__ inline void Process();

    __aicore__ inline void Init(const PrepareWyReprBwdFullTilingData &tiling);

private:
    template <class L1TileShape, class L0TileShape, class L1TileShapeDvb, class L0TileShapeDvb>
    __aicore__ inline void ProcessImpl();

    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t HV = 0;
    uint64_t HK = 0;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum;
    uint64_t fusedKVecRow = 0;
    uint64_t fusedVVecRow = 0;
    uint64_t fusedKktVecRow = 0;
    uint64_t workspaceSlotSize = 0;
    uint64_t workspaceBufferCount = 1;
    uint64_t usedCoreNum = 0;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR dA;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR dbeta;
    GM_ADDR dg;
    GM_ADDR workspace;
};

template <typename kType, typename betaType>
__aicore__ inline PrepareWyReprBwdFullProcess<kType, betaType>::PrepareWyReprBwdFullProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), dA(dA_), dw(dw_), du(du_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), dk(dk_), dv(dv_), dbeta(dbeta_), dg(dg_), workspace(workspace_){};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Init(const PrepareWyReprBwdFullTilingData &tiling)
{
    B = tiling.B;
    T = tiling.T;
    HV = tiling.HV;
    HK = tiling.HK;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    fusedKVecRow = tiling.fusedKVecRow;
    fusedVVecRow = tiling.fusedVVecRow;
    fusedKktVecRow = tiling.fusedKktVecRow;
    workspaceSlotSize = tiling.workspaceSlotSize;
    workspaceBufferCount = tiling.workspaceBufferCount;
    usedCoreNum = tiling.usedCoreNum;
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Process()
{
    if (V == 256) {
        ProcessImpl<Shape<_128, _128, _256>, Shape<_128, _128, _128>, Shape<_128, _256, _256>,
                    Shape<_128, _256, _128>>();
    } else {
        ProcessImpl<Shape<_128, _128, _256>, Shape<_128, _128, _128>, Shape<_128, _128, _256>,
                    Shape<_128, _128, _128>>();
    }
}

template <typename kType, typename betaType>
template <class L1TileShape, class L0TileShape, class L1TileShapeDvb, class L0TileShapeDvb>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::ProcessImpl()
{
    // GM tensor layouts used by tile_mmad.  A and dA have both row-major and
    // column-major views because different GEMMs need A^T/dA^T without
    // materializing a transpose in this kernel.
    using LayoutTagA = layout::RowMajor;
    using LayoutTagAT = layout::ColumnMajor;
    using LayoutTagDW = layout::RowMajor;
    using LayoutTagDA = layout::RowMajor;
    using LayoutTagDAT = layout::ColumnMajor;
    using LayoutTagBeta = layout::RowMajor;
    using LayoutTagK = layout::RowMajor;
    using LayoutTagV = layout::RowMajor;
    using LayoutTagKT = layout::ColumnMajor;
    using LayoutTagDu = layout::RowMajor;
    using LayoutTagDvb = layout::RowMajor;


    // Input logical matrices for one (chunk, head):
    //   A/dA : chunkSize x chunkSize
    //   K/V  : chunkSize x K/V
    //   dw/du: chunkSize x K/V
    // ColumnMajor tags reinterpret A/dA/K as transposed operands for MMAD.
    LayoutTagA tagA = LayoutTagA::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagAT tagAT = LayoutTagAT::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagDW tagDW = LayoutTagDW::MakeLayout<kType>(chunkSize, K);
    LayoutTagDA tagDA = LayoutTagDA::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagDAT tagDAT = LayoutTagDAT::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagK tagK = LayoutTagK::MakeLayout<kType>(chunkSize, K);
    LayoutTagV tagV = LayoutTagV::MakeLayout<kType>(chunkSize, V);
    LayoutTagKT tagKT = LayoutTagKT::MakeLayout<kType>(K, chunkSize);
    LayoutTagDu tagDu = LayoutTagDu::MakeLayout<kType>(chunkSize, V);

    // Workspace intermediate layouts.  These matrices are scoped to the
    // ping-pong slot for the current (chunk, head):
    //   Kbeta = K * beta[:, None]
    //   Dkb   = dA^T @ K
    //   Dkbg  = A^T @ dw
    //   Dvb   = A^T @ du
    //   KKT   = K @ K^T
    using LayoutTagKbeta = layout::RowMajor;
    LayoutTagKbeta tagKbeta = LayoutTagKbeta::MakeLayout<kType>(chunkSize, K);

    using LayoutTagDkb = layout::RowMajor;
    LayoutTagDkb tagDkb = LayoutTagDkb::MakeLayout<kType>(chunkSize, K);

    using LayoutTagDkbg = layout::RowMajor;
    LayoutTagDkbg tagDkbg = LayoutTagDkbg::MakeLayout<kType>(chunkSize, K);

    using LayoutTagDvb = layout::RowMajor;
    LayoutTagDvb tagDvb = LayoutTagDvb::MakeLayout<kType>(chunkSize, V);

    using LayoutTagKKT = layout::RowMajor;
    LayoutTagKKT tagKKT = LayoutTagKKT::MakeLayout<kType>(chunkSize, chunkSize);

    // Output dk is written directly to GM after AIV adds all dk terms.
    using LayoutTagDk = layout::RowMajor;
    LayoutTagDk tagDk = LayoutTagDk::MakeLayout<kType>(chunkSize, K);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Arch::Ascend950;
#else
    using ArchTag = Arch::AtlasA2;
#endif
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;

    // Tile copy policies bind each MMAD's operand/result layouts:
    //   TileCopyDk   : DK   = dA @ Kbeta
    //   TileCopyDkb  : Dkb  = dA^T @ K
    //   TileCopyDkbg : Dkbg = A^T @ dw
    //   TileCopyDvb  : Dvb  = A^T @ du
    //   TileCopyKKT  : KKT  = K @ K^T
    using TileCopyDk =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDA, kType, LayoutTagKbeta, kType, LayoutTagDk>;

    using TileCopyDkb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDAT, kType, LayoutTagK, kType, LayoutTagDkb>;

    using TileCopyDkbg =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDW, kType, LayoutTagDkbg>;

    using TileCopyDvb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagAT, kType, LayoutTagDu, kType, LayoutTagDvb>;

    using TileCopyKKT =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagK, kType, LayoutTagKT, kType, LayoutTagKKT>;

    auto layoutKbeta = MakeLayoutFromTag(tagKbeta);
    auto layoutDA = MakeLayoutFromTag(tagDA);
    auto layoutDK = MakeLayoutFromTag(tagDk);
    auto layoutDAT = MakeLayoutFromTag(tagDAT);
    auto layoutK = MakeLayoutFromTag(tagK);
    auto layoutDkb = MakeLayoutFromTag(tagDkb);
    auto layoutAT = MakeLayoutFromTag(tagAT);
    auto layoutDw = MakeLayoutFromTag(tagDW);
    auto layoutDkbg = MakeLayoutFromTag(tagDkbg);
    auto layoutDu = MakeLayoutFromTag(tagDu);
    auto layoutDvb = MakeLayoutFromTag(tagDvb);
    auto layoutKT = MakeLayoutFromTag(tagKT);
    auto layoutKKT = MakeLayoutFromTag(tagKKT);
    // kernel level
    using MatmulKernel =
        Gemm::Kernel::PrepareWyReprBwdFullTla<ArchTag, L1TileShape, L0TileShape, L1TileShapeDvb, L0TileShapeDvb,
                                              TileCopyDk, TileCopyDkb, TileCopyDkbg, TileCopyDvb, TileCopyKKT>;

    MatmulKernel kernel;

    // params.ptrKbeta is the workspace base.  The kernel computes per-core
    // slot offsets internally and aliases the same workspace base as Kbeta,
    // Dkb, Dkbg, Dvb and KKT according to the slot layout.
    typename MatmulKernel::Params param{
        workspace, layoutKbeta, dA, layoutDA, dk,        layoutDK,  dA,         layoutDAT,     k,        layoutK,
        workspace, layoutDkb,   A,  layoutAT, dw,        layoutDw,  workspace,  layoutDkbg,    du,       layoutDu,
        workspace, layoutDvb,   k,  layoutKT, workspace, layoutKKT, cu_seqlens, chunk_indices, chunkNum, B,
        T,         HV,          HK, K,        V,          chunkSize, 4, fusedKVecRow, fusedVVecRow, fusedKktVecRow,
        workspaceSlotSize, workspaceBufferCount, usedCoreNum};
    kernel(param);
}


#endif // PREPARE_WY_REPR_BWD_FULL_CUBE_H
