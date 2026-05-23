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
#define CATLASS_ARCH 3510
#include "prepare_wy_repr_bwd_full_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/arch/cross_core_sync.hpp"
using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <class BlockMmadBdk_, class BlockMmadBdkb_, class BlockMmadBdkbg_, class BlockMmadBdvb_, class BlockMmadBkkT_>
class PrepareWyReprBwdFullTla {
public:
    using BlockMmadBdk = BlockMmadBdk_;
    using BlockMmadBdkb = BlockMmadBdkb_;
    using BlockMmadBdkbg = BlockMmadBdkbg_;
    using BlockMmadBkkT = BlockMmadBkkT_;
    using BlockMmadBdvb = BlockMmadBdvb_;
    using ArchTag = typename BlockMmadBdkb::ArchTag;
    using BdkL1TileShape = typename BlockMmadBdk::L1TileShape;
    using BdkbL1TileShape = typename BlockMmadBdkb::L1TileShape;
    using ElementDA = typename BlockMmadBdk::ElementA;
    using LayoutDA = typename BlockMmadBdk::LayoutA;
    using ElementKbeta = typename BlockMmadBdk::ElementB;
    using LayoutKbeta = typename BlockMmadBdk::LayoutB;
    using ElementDk = typename BlockMmadBdk::ElementC;
    using LayoutDk = typename BlockMmadBdk::LayoutC;

    using ElementDAT = typename BlockMmadBdkb::ElementA;
    using LayoutDAT = typename BlockMmadBdkb::LayoutA;
    using ElementK = typename BlockMmadBdkb::ElementB;
    using LayoutK = typename BlockMmadBdkb::LayoutB;
    using ElementDkb = typename BlockMmadBdkb::ElementC;
    using LayoutDkb = typename BlockMmadBdkb::LayoutC;

    using ElementAT = typename BlockMmadBdkbg::ElementA;
    using LayoutAT = typename BlockMmadBdkbg::LayoutA;
    using ElementDw = typename BlockMmadBdkbg::ElementB;
    using LayoutDw = typename BlockMmadBdkbg::LayoutB;
    using ElementDkbg = typename BlockMmadBdkbg::ElementC;
    using LayoutDkbg = typename BlockMmadBdkbg::LayoutC;

    using ElementDu = typename BlockMmadBdvb::ElementB;
    using LayoutDu = typename BlockMmadBdvb::LayoutB;
    using ElementDvb = typename BlockMmadBdvb::ElementC;
    using LayoutDvb = typename BlockMmadBdvb::LayoutC;

    using ElementKT = typename BlockMmadBkkT::ElementB;
    using LayoutKT = typename BlockMmadBkkT::LayoutB;
    using ElementKKT = typename BlockMmadBkkT::ElementC;
    using LayoutKKT = typename BlockMmadBkkT::LayoutC;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{SYNC_FLAG_2, SYNC_FLAG_3};
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
        LayoutDw layoutDkbg;
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
        uint64_t K = 128;
        uint64_t V = 128;
        uint64_t chunkSize = 64;
        uint64_t stage = 2;
        uint64_t kBeteVecRow = 0;
        uint64_t dkbVecRow = 0;
        uint64_t dkbgVecRow = 0;
        uint64_t dvbVecRow = 0;
        uint64_t kktVecRow = 0;
        uint64_t kBetaCVNum = 0;
        uint64_t dkbCVNum = 0;
        uint64_t dkbgCVNum = 0;
        uint64_t dvbCVNum = 0;
        uint64_t kktCVNum = 0;

        // Methods
        CATLASS_DEVICE
        Params()
        {
        }

        CATLASS_DEVICE
        Params(GM_ADDR ptrptrKbeta_, LayoutKbeta layoutKbeta_, GM_ADDR ptrDA_, LayoutDA layoutDA_, GM_ADDR ptrDk_,
               LayoutKbeta layoutDk_, GM_ADDR ptrDAT_, LayoutDAT layoutDAT_, GM_ADDR ptrK_, LayoutK layoutK_,
               GM_ADDR ptrDkb_, LayoutDkb layoutDkb_, GM_ADDR ptrAT_, LayoutAT layoutAT_, GM_ADDR ptrDw_,
               LayoutDw layoutDw_, GM_ADDR ptrDkbg_, LayoutDkbg layoutDkbg_, GM_ADDR ptrDu_, LayoutDkbg layoutDu_,
               GM_ADDR ptrDvb_, LayoutDkbg layoutDvb_, GM_ADDR ptrKT_, LayoutKT layoutKT_, GM_ADDR ptrKKT_,
               LayoutKKT layoutKKT_, GM_ADDR ptrCuSeqLens_, GM_ADDR ptrChunkIndices_, uint64_t chunkNum_, uint64_t B_,
               uint64_t T_, uint64_t H_, uint64_t K_, uint64_t V_, uint64_t BT_, uint64_t stage_,
               uint64_t kBeteVecRow_, uint64_t dkbVecRow_, uint64_t dkbgVecRow_, uint64_t dvbVecRow_, uint64_t kktVecRow_,
               uint64_t kBetaCVNum_, uint64_t dkbCVNum_, uint64_t dkbgCVNum_, uint64_t dvbCVNum_, uint64_t kktCVNum_)
            : ptrKbeta(ptrptrKbeta_), layoutKbeta(layoutKbeta_), ptrDA(ptrDA_), layoutDA(layoutDA_), ptrDk(ptrDk_),
              layoutDk(layoutDk_), ptrDAT(ptrDAT_), layoutDAT(layoutDAT_), ptrK(ptrK_), layoutK(layoutK_),
              ptrDkb(ptrDkb_), layoutDkb(layoutDkb_), ptrAT(ptrAT_), layoutAT(layoutAT_), ptrDw(ptrDw_),
              layoutDw(layoutDw_), ptrDkbg(ptrDkbg_), layoutDkbg(layoutDkbg_), ptrDu(ptrDu_), layoutDu(layoutDu_),
              ptrDvb(ptrDvb_), layoutDvb(layoutDvb_), ptrKT(ptrKT_), layoutKT(layoutKT_), ptrKKT(ptrKKT_),
              layoutKKT(layoutKKT_), ptrCuSeqLens(ptrCuSeqLens_), ptrChunkIndices(ptrChunkIndices_),
              chunkNum(chunkNum_), B(B_), T(T_), HV(H_), K(K_), V(V_), chunkSize(BT_), stage(stage_),
              kBeteVecRow(kBeteVecRow_), dkbVecRow(dkbVecRow_), dkbgVecRow(dkbgVecRow_), dvbVecRow(dvbVecRow_), kktVecRow(kktVecRow_),
              kBetaCVNum(kBetaCVNum_), dkbCVNum(dkbCVNum_), dkbgCVNum(dkbgCVNum_), dvbCVNum(dvbCVNum_), kktCVNum(kktCVNum_)
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
        { //处理第一部分cube DA @ Kbeta     V->C
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_FLAG_5);
            BlockMmadBdk blockMmadBdk(resource);
            AscendC::GlobalTensor<ElementDA> gmDA;
            AscendC::GlobalTensor<ElementKbeta> gmKbeta;
            AscendC::GlobalTensor<ElementDk> gmDk;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.HV, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                for (int h = 0; h < params.HV; h++) {
                    // Represent the full gm
                    gmDA.SetGlobalBuffer((__gm__ ElementDA *)params.ptrDA + (h * params.T + bos) * params.chunkSize);
                    gmKbeta.SetGlobalBuffer((__gm__ ElementKbeta *)params.ptrKbeta + (h * params.T + bos) * params.K);
                    gmDk.SetGlobalBuffer((__gm__ ElementDk *)params.ptrDk + (h * params.T + bos) * params.K);

                    // Represent the full tensors
                    auto tensorDA = tla::MakeTensor(gmDA, params.layoutDA, Arch::PositionGM{});
                    auto tensorKbeta = tla::MakeTensor(gmKbeta, params.layoutKbeta, Arch::PositionGM{});
                    auto tensorDk = tla::MakeTensor(gmDk, params.layoutDk, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockDA = GetTile(tensorDA, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockKbeta = GetTile(tensorKbeta, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockDk = GetTile(tensorDk, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdk(tensorBlockDA, tensorBlockKbeta, tensorBlockDk, actualBlockShape);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_FLAG_5);
                }
            }
        }
        AscendC::SyncAll<false>();
        { //处理第二部分 DAT@K -> DKB
            BlockMmadBdkb blockMmadBdkb(resource);
            AscendC::GlobalTensor<ElementDAT> gmDAT;
            AscendC::GlobalTensor<ElementK> gmK;

            uint32_t ubOffset = 0;
            uint32_t ubListId = 0;
            AscendC::LocalTensor<ElementDkb> ubDkbList[MAX_CUBE_VEC_SYNC_NUM];
            for(int i = 0; i < params.dkbCVNum; i++) {
                ubDkbList[i] = resource.ubBuf.template GetBufferByByte<ElementDkb>(ubOffset);
                ubOffset += params.dkbVecRow * params.K * sizeof(ElementDkb);
            }
            uint8_t beginSubBlockIdx = 1;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.HV, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                for (int h = 0; h < params.HV; h++) {
                    // Represent the full gm
                    gmDAT.SetGlobalBuffer((__gm__ ElementDAT *)params.ptrDAT + (h * params.T + bos) * params.chunkSize);
                    gmK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK + (h * params.T + bos) * params.K);
                    
                    // Represent the full tensors
                    auto tensorDAT = tla::MakeTensor(gmDAT, params.layoutDAT, Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});

                    // Make tiled views
                    auto tensorBlockDAT = GetTile(tensorDAT, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    using UBTensor = tla::Tensor<AscendC::LocalTensor<ElementDkb>, LayoutDkb, tuple<int, int>, AscendC::TPosition::VECCALC>;
                    UBTensor tensorBlockDkbList[MAX_CUBE_VEC_SYNC_NUM];
                    for (uint32_t i = 0; i < params.dkbCVNum; i++) {
                        auto tensorDkb = tla::MakeTensor(ubDkbList[i], params.layoutDkb, Arch::PositionUB{});
                        tensorBlockDkbList[i] = GetTile(tensorDkb, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    }
                    
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdkb(tensorBlockDAT, tensorBlockK, tensorBlockDkbList, actualBlockShape, params.dkbVecRow,
                        beginSubBlockIdx, SYNC_AIV_AIC_FLAG_BEGIN, SYNC_AIC_AIV_FLAG_BEGIN, ubListId, 2, params.dkbCVNum);
                    beginSubBlockIdx += CeilDiv(curChunkSize, params.dkbVecRow);
                    beginSubBlockIdx = beginSubBlockIdx % 2;
                }
            }
            for (uint32_t i = 0; i < params.dkbCVNum; i++) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_BEGIN + i);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_BEGIN + FLAG_ID_MAX + i);
            }
        }
        AscendC::SyncAll<false>();
        { //处理第三部分 AT@dw -> DKBG
            BlockMmadBdkbg blockMmadBdkbg(resource);
            AscendC::GlobalTensor<ElementAT> gmAT;
            AscendC::GlobalTensor<ElementDw> gmDw;

            uint32_t ubOffset = 0;
            uint32_t ubListId = 0;
            AscendC::LocalTensor<ElementDkbg> ubDkbgList[MAX_CUBE_VEC_SYNC_NUM];
            for (uint32_t i = 0; i < params.dkbgCVNum; i++) {
                ubDkbgList[i] = resource.ubBuf.template GetBufferByByte<ElementDkbg>(ubOffset);
                ubOffset += params.dkbgVecRow * params.K * sizeof(ElementDkbg);
            }
            uint8_t beginSubBlockIdx = 1;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.HV, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                for (int h = 0; h < params.HV; h++) {
                    // Represent the full gm
                    gmAT.SetGlobalBuffer((__gm__ ElementAT *)params.ptrAT + (h * params.T + bos) * params.chunkSize);
                    gmDw.SetGlobalBuffer((__gm__ ElementDw *)params.ptrDw + (h * params.T + bos) * params.K);

                    // Represent the full tensors
                    auto tensorAT = tla::MakeTensor(gmAT, params.layoutAT, Arch::PositionGM{});
                    auto tensorDw = tla::MakeTensor(gmDw, params.layoutDw, Arch::PositionGM{});

                    using UBTensor =
                        tla::Tensor<AscendC::LocalTensor<ElementDkbg>, LayoutDkbg, tuple<int, int>, AscendC::TPosition::VECCALC>;
                    UBTensor tensorBlockDkbgList[MAX_CUBE_VEC_SYNC_NUM];
                    for (uint32_t i = 0; i < params.dkbgCVNum; i++) {
                        auto ubDkbg = tla::MakeTensor(ubDkbgList[i], params.layoutDkbg, Arch::PositionUB{});
                        tensorBlockDkbgList[i] = GetTile(ubDkbg, tla::MakeCoord(0, 0),
                                                         tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    }
                    // Make tiled views
                    auto tensorBlockAT = GetTile(tensorAT, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockDw = GetTile(tensorDw, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmadBdkbg(tensorBlockAT, tensorBlockDw, tensorBlockDkbgList, actualBlockShape, params.dkbgVecRow,
                                   beginSubBlockIdx, SYNC_AIV_AIC_FLAG_BEGIN, SYNC_AIC_AIV_FLAG_BEGIN, ubListId, 2, params.dkbgCVNum);
                    beginSubBlockIdx += CeilDiv(curChunkSize, params.dkbgVecRow);
                    beginSubBlockIdx = beginSubBlockIdx % 2;
                }
            }
            for (uint32_t i = 0; i < params.dkbgCVNum; i++) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_BEGIN + i);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_BEGIN + FLAG_ID_MAX + i);
            }
        }
        AscendC::SyncAll<false>();
        { //处理第四部分 AT@du -> dvb
            BlockMmadBdvb blockMmadBdvb(resource);
            AscendC::GlobalTensor<ElementAT> gmAT;
            AscendC::GlobalTensor<ElementDu> gmDu;

            uint32_t ubOffset = 0;
            uint32_t ubListId = 0;
            AscendC::LocalTensor<ElementDvb> ubDvbList[MAX_CUBE_VEC_SYNC_NUM];
            for (uint32_t i = 0; i < params.dvbCVNum; i++) {
                ubDvbList[i] = resource.ubBuf.template GetBufferByByte<ElementDvb>(ubOffset);
                ubOffset += params.dvbVecRow * params.V * sizeof(ElementDvb);
            }
            uint8_t beginSubBlockIdx = 1;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.HV, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.V), curChunkSize};
                for (int h = 0; h < params.HV; h++) {
                    // Represent the full gm
                    gmAT.SetGlobalBuffer((__gm__ ElementAT *)params.ptrAT + (h * params.T + bos) * params.chunkSize);
                    gmDu.SetGlobalBuffer((__gm__ ElementDu *)params.ptrDu + (h * params.T + bos) * params.V);

                    // Represent the full tensors
                    auto tensorAT = tla::MakeTensor(gmAT, params.layoutAT, Arch::PositionGM{});
                    auto tensorDu = tla::MakeTensor(gmDu, params.layoutDu, Arch::PositionGM{});

                    // Make tiled views
                    auto tensorBlockAT = GetTile(tensorAT, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockDu = GetTile(tensorDu, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    using UBTensor = tla::Tensor<AscendC::LocalTensor<ElementDvb>, LayoutDvb, tuple<int, int>, AscendC::TPosition::VECCALC>;
                    UBTensor tensorBlockDvbList[MAX_CUBE_VEC_SYNC_NUM];
                    for (uint32_t i = 0; i < params.dvbCVNum; i++) {
                        auto ubDvb = tla::MakeTensor(ubDvbList[i], params.layoutDvb, Arch::PositionUB{});
                        tensorBlockDvbList[i] = GetTile(ubDvb, tla::MakeCoord(0, 0),
                                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    }

                    // Compute block-scoped matrix multiply-add
                    blockMmadBdvb(tensorBlockAT, tensorBlockDu, tensorBlockDvbList, actualBlockShape, params.dvbVecRow,
                                  beginSubBlockIdx, SYNC_AIV_AIC_FLAG_BEGIN, SYNC_AIC_AIV_FLAG_BEGIN, ubListId, 2, params.dvbCVNum);
                    beginSubBlockIdx += CeilDiv(curChunkSize, params.dvbVecRow);
                    beginSubBlockIdx = beginSubBlockIdx % 2;
                }
            }
            for (uint32_t i = 0; i < params.dvbCVNum; i++) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_BEGIN + i);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_BEGIN + FLAG_ID_MAX + i);
            }
        }
        AscendC::SyncAll<false>();
        { //处理第五部分 K@KT -> kkT
            BlockMmadBkkT blockMmadkkT(resource);
            AscendC::GlobalTensor<ElementK> gmK;
            AscendC::GlobalTensor<ElementKT> gmKT;

            uint32_t ubOffset = 0;
            uint32_t ubIdList[2] = {0};
            AscendC::LocalTensor<ElementKKT> ubKKTList[MAX_CUBE_VEC_SYNC_NUM];
            for (uint32_t i = 0; i < params.kktCVNum; i++) {
                ubKKTList[i] = resource.ubBuf.template GetBufferByByte<ElementKKT>(ubOffset);
                ubOffset += params.kktVecRow * params.chunkSize * sizeof(ElementKKT);
            }
            uint8_t beginSubBlockIdx = 1;
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.HV, params.T,
                               params.chunkSize, loopIdx, bos, eos);
                uint32_t curChunkSize = eos - bos;
                GemmCoord blockCoord{0, 0, 0};
                GemmCoord actualBlockShape{curChunkSize, curChunkSize, static_cast<uint32_t>(params.K)};
                for (int h = 0; h < params.HV; h++) {
                    // Represent the full gm
                    gmK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK + (h * params.T + bos) * params.K);
                    gmKT.SetGlobalBuffer((__gm__ ElementKT *)params.ptrKT + (h * params.T + bos) * params.K);

                    // Represent the full tensors
                    auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
                    auto tensorKT = tla::MakeTensor(gmKT, params.layoutKT, Arch::PositionGM{});

                    // Make tiled views
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockKT = GetTile(tensorKT, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    using UBTensor = tla::Tensor<AscendC::LocalTensor<ElementKKT>, LayoutKKT, tuple<int, int>, AscendC::TPosition::VECCALC>;
                    UBTensor tensorBlockKKTList[MAX_CUBE_VEC_SYNC_NUM];
                    for (uint32_t i = 0; i < params.kktCVNum; i++) {
                        auto tensorKKT = tla::MakeTensor(ubKKTList[i], params.layoutKKT, Arch::PositionUB{});
                        tensorBlockKKTList[i] = GetTile(tensorKKT, tla::MakeCoord(0, 0),
                                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    }
                    auto& ubListId = ubIdList[beginSubBlockIdx];
                    // Compute block-scoped matrix multiply-add
                    blockMmadkkT(tensorBlockK, tensorBlockKT, tensorBlockKKTList, actualBlockShape, params.kktVecRow,
                                 beginSubBlockIdx, SYNC_AIV_AIC_FLAG_BEGIN, SYNC_AIC_AIV_FLAG_BEGIN, ubListId, 1, params.kktCVNum);
                    beginSubBlockIdx = (beginSubBlockIdx + 1 < 2) ? (beginSubBlockIdx + 1) : 0;
                }
            }
            for (uint32_t i = 0; i < params.kktCVNum; i++) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_BEGIN + i);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(SYNC_AIV_AIC_FLAG_BEGIN + FLAG_ID_MAX + i);
            }
        }
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

    __aicore__ inline void Init(const PrepareWyReprBwdFullTilingDataA5 &tiling);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t HV = 0;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum;
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
    uint64_t kBeteVecRow = 0;
    uint64_t dkbVecRow = 0;
    uint64_t dkbgVecRow = 0;
    uint64_t dvbVecRow = 0;
    uint64_t kktVecRow = 0;
    uint64_t kBetaCVNum = 0;
    uint64_t dkbCVNum = 0;
    uint64_t dkbgCVNum = 0;
    uint64_t dvbCVNum = 0;
    uint64_t kktCVNum = 0;
};

template <typename kType, typename betaType>
__aicore__ inline PrepareWyReprBwdFullProcess<kType, betaType>::PrepareWyReprBwdFullProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), dA(dA_), dw(dw_), du(du_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), dk(dk_), dv(dv_), dbeta(dbeta_), dg(dg_), workspace(workspace_){};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Init(const PrepareWyReprBwdFullTilingDataA5 &tiling)
{
    B = tiling.B;
    T = tiling.T;
    HV = tiling.HV;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    kBeteVecRow = tiling.kBeteVecRow;
    dkbVecRow = tiling.dkbVecRow;
    dkbgVecRow = tiling.dkbgVecRow;
    dvbVecRow = tiling.dvbVecRow;
    kktVecRow = tiling.kktVecRow;
    kBetaCVNum = tiling.kBetaCVNum;
    dkbCVNum = tiling.dkbCVNum;
    dkbgCVNum = tiling.dkbgCVNum;
    dvbCVNum = tiling.dvbCVNum;
    kktCVNum = tiling.kktCVNum;
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Process()
{
    //输入
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


    //输入
    LayoutTagA tagA = LayoutTagA::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagAT tagAT = LayoutTagAT::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagDW tagDW = LayoutTagDW::MakeLayout<kType>(chunkSize, K);
    LayoutTagDA tagDA = LayoutTagDA::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagDAT tagDAT = LayoutTagDAT::MakeLayout<kType>(chunkSize, chunkSize);
    LayoutTagK tagK = LayoutTagK::MakeLayout<kType>(chunkSize, K);
    LayoutTagV tagV = LayoutTagV::MakeLayout<kType>(chunkSize, V);
    LayoutTagKT tagKT = LayoutTagKT::MakeLayout<kType>(K, chunkSize);
    LayoutTagDu tagDu = LayoutTagDu::MakeLayout<kType>(chunkSize, V);

    //输出
    using LayoutTagDk = layout::RowMajor;
    LayoutTagDk tagDk = LayoutTagDk::MakeLayout<kType>(chunkSize, K);

    //中间结果
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

    using L1TileShape = Shape<_128, _128, _256>;
    using L0TileShape = Shape<_128, _128, _128>;

    using ArchTag = Arch::Ascend950;
    using DispatchPolicy = Common::MmadPingpong<ArchTag, false, false, 2>;

    //计算dk第一部分, dA @ Kbeta
    using TileCopyDk =
        Common::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDA, kType, LayoutTagKbeta, kType, LayoutTagDk>;
    using BlockMmadDk =
        Common::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDk>;

    using TileCopyDkb =
        Common::Tile::PackedTileCopyTlaToUB<ArchTag, kType, LayoutTagDAT, kType, LayoutTagK, kType, LayoutTagDkb, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadDkb =
        Common::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDkb>;

    using TileCopyDkbg =
        Common::Tile::PackedTileCopyTlaToUB<ArchTag, kType, LayoutTagAT, kType, LayoutTagDW, kType, LayoutTagDkbg, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadDkbg =
        Common::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDkbg>;

    using TileCopyDvb =
        Common::Tile::PackedTileCopyTlaToUB<ArchTag, kType, LayoutTagAT, kType, LayoutTagDu, kType, LayoutTagDvb, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadDvb =
        Common::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDvb>;

    using TileCopyKKT =
        Common::Tile::PackedTileCopyTlaToUB<ArchTag, kType, LayoutTagK, kType, LayoutTagKT, kType, LayoutTagKKT, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadKKT =
        Common::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyKKT>;

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
        Gemm::Kernel::PrepareWyReprBwdFullTla<BlockMmadDk, BlockMmadDkb, BlockMmadDkbg, BlockMmadDvb, BlockMmadKKT>;

    MatmulKernel kernel;

    typename MatmulKernel::Params param{
        workspace, layoutKbeta, dA, layoutDA, dk,        layoutDK,  dA,         layoutDAT,     k,        layoutK,
        workspace, layoutDkb,   A,  layoutAT, dw,        layoutDw,  workspace,  layoutDkbg,    du,       layoutDu,
        workspace, layoutDvb,   k,  layoutKT, workspace, layoutKKT, cu_seqlens, chunk_indices, chunkNum, B,
        T,         HV,           K,  V,        chunkSize, 4,
        kBeteVecRow, dkbVecRow, dkbgVecRow, dvbVecRow, kktVecRow, kBetaCVNum, dkbCVNum, dkbgCVNum, dvbCVNum, kktCVNum};
    kernel(param);
}


#endif // PREPARE_WY_REPR_BWD_FULL_CUBE_H
