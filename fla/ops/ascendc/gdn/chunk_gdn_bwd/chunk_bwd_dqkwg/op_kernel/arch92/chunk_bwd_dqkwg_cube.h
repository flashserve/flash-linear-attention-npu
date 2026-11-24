/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CHUNK_BWD_DQKWG_CUBE_H
#define CHUNK_BWD_DQKWG_CUBE_H

#define CATLASS_ARCH 9201
#include "chunk_bwd_dqkwg_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
using _128 = tla::Int<128>;
using _256 = tla::Int<256>;
using namespace tla;
using namespace Catlass;
using namespace AscendC;

namespace Catlass::Gemm::Kernel {

template <class BlockMmadPart1_,
          class BlockMmadPart2_,
          class BlockMmadPart3_,
          class BlockMmadPart4_,
          class BlockMmadPart5_,
          class BlockMmadPart6_,
          class BlockMmadPart7_
          >
class ChunkBwdDqkwgTla {
public:
    using BlockMmadPart1 = BlockMmadPart1_;
    using BlockMmadPart2 = BlockMmadPart2_;
    using BlockMmadPart3 = BlockMmadPart3_;
    using BlockMmadPart4 = BlockMmadPart4_;
    using BlockMmadPart5 = BlockMmadPart5_;
    using BlockMmadPart6 = BlockMmadPart6_;
    using BlockMmadPart7 = BlockMmadPart7_;

    using ArchTag = typename BlockMmadPart1::ArchTag;

    using ElementA = typename BlockMmadPart2::ElementA;
    using ElementB = typename BlockMmadPart2::ElementB;
    using ElementC = typename BlockMmadPart2::ElementC;

    using LayoutRowMajor = layout::RowMajor;
    using LayoutColMajor = layout::ColumnMajor;

    struct Params {

        GM_ADDR ptrQ;
        GM_ADDR ptrK;
        GM_ADDR ptrV;
        GM_ADDR ptrH;
        GM_ADDR ptrDo;
        GM_ADDR ptrDh;
        GM_ADDR ptrDv;

        GM_ADDR ptrCuSeqLens;
        GM_ADDR ptrChunkIndices;

        GM_ADDR ptrDq;
        GM_ADDR ptrDk;
        GM_ADDR ptrDw;

        GM_ADDR ptrWorkspace;

        uint64_t wsMm3Offset;
        uint64_t wsMm4Offset;
        uint64_t wsMm6Offset;
        uint64_t wsMm5Offset;
        uint64_t wsMm7Offset;
        uint64_t wsDsTempOffset;

        uint64_t B;
        uint64_t HV;
        uint64_t HK;
        uint64_t T;
        uint64_t K;
        uint64_t V;
        uint64_t BT;
        uint64_t numChunks;
        uint64_t n_ratio;
        uint64_t aicCoreNum = 0;
        bool isVarLen = false;

        CATLASS_DEVICE
        Params(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen,
               GM_ADDR chunk_indices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dw,GM_ADDR workspace, uint64_t B, uint64_t HV, uint64_t HK,
               uint64_t T, uint64_t K, uint64_t V, uint64_t BT, uint64_t numChunks, uint64_t n_ratio, bool isVarLen,
               uint64_t wsMm3, uint64_t wsMm4, uint64_t wsMm6, uint64_t wsMm5, uint64_t wsMm7, uint64_t wsDsTemp)
            : ptrQ(q), ptrK(k), ptrV(v), ptrH(h), ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLens(cu_seqlen),
              ptrChunkIndices(chunk_indices), ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrWorkspace(workspace),
              wsMm3Offset(wsMm3), wsMm4Offset(wsMm4), wsMm6Offset(wsMm6), wsMm5Offset(wsMm5),
              wsMm7Offset(wsMm7), wsDsTempOffset(wsDsTemp), B(B), HV(HV), HK(HK), T(T), K(K), V(V), BT(BT),
              numChunks(numChunks), n_ratio(n_ratio), isVarLen(isVarLen)
        {
        }
    };

    GlobalTensor<ElementA> gmQ;
    GlobalTensor<ElementA> gmK;
    GlobalTensor<ElementA> gmV;
    GlobalTensor<ElementA> gmH;
    GlobalTensor<ElementA> gmDo;
    GlobalTensor<ElementA> gmDh;
    GlobalTensor<ElementA> gmDv;
    GlobalTensor<ElementC> gmDq;
    GlobalTensor<ElementC> gmDk;
    GlobalTensor<ElementC> gmDw;
    GlobalTensor<ElementC> gmMm3;
    GlobalTensor<ElementC> gmMm4;
    GlobalTensor<ElementC> gmMm6;
    GlobalTensor<ElementC> gmMm5;
    GlobalTensor<ElementC> gmMm7;
    GlobalTensor<ElementC> gmDs;
    GlobalTensor<ElementA> gmDsTemp;
    GlobalTensor<uint64_t> cuSeqlensTensor;
    GlobalTensor<uint64_t> chunkIndicesTensor;

    CATLASS_DEVICE
    ChunkBwdDqkwgTla()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Arch::Resource<ArchTag> resource;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = params.aicCoreNum;
        uint32_t coreLoops = params.B * params.numChunks;
        uint32_t K = static_cast<uint32_t>(params.K);
        uint32_t V = static_cast<uint32_t>(params.V);

        gmQ.SetGlobalBuffer((__gm__ ElementA *)params.ptrQ);
        gmK.SetGlobalBuffer((__gm__ ElementA *)params.ptrK);
        gmV.SetGlobalBuffer((__gm__ ElementA *)params.ptrV);
        gmH.SetGlobalBuffer((__gm__ ElementA *)params.ptrH);
        gmDo.SetGlobalBuffer((__gm__ ElementA *)params.ptrDo);
        gmDh.SetGlobalBuffer((__gm__ ElementA *)params.ptrDh);
        gmDv.SetGlobalBuffer((__gm__ ElementA *)params.ptrDv);
        gmDq.SetGlobalBuffer((__gm__ ElementC *)params.ptrDq);
        gmDk.SetGlobalBuffer((__gm__ ElementC *)params.ptrDk);
        gmDw.SetGlobalBuffer((__gm__ ElementC *)params.ptrDw);
        gmMm3.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm3Offset));
        gmMm4.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm4Offset));
        gmMm6.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm6Offset));
        gmMm5.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm5Offset));
        gmMm7.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm7Offset));
        gmDs.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsDsTempOffset));
        gmDsTemp.SetGlobalBuffer((__gm__ ElementA *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsDsTempOffset));
        if (params.isVarLen) {
            cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)params.ptrCuSeqLens);
            chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)params.ptrChunkIndices);
        }

        auto layoutBTxK = LayoutRowMajor::MakeLayout<ElementA>(params.BT, params.K);
        auto layoutKxBT = LayoutColMajor::MakeLayout<ElementA>(params.K, params.BT);
        auto layoutBTxV = LayoutRowMajor::MakeLayout<ElementA>(params.BT, params.V);
        auto layoutVxBT = LayoutColMajor::MakeLayout<ElementA>(params.V, params.BT);
        auto layoutBTxBT = LayoutRowMajor::MakeLayout<ElementA>(params.BT, params.BT);
        auto layoutBTxBT_T = LayoutColMajor::MakeLayout<ElementA>(params.BT, params.BT);
        auto layoutVxK = LayoutColMajor::MakeLayout<ElementA>(params.V, params.K);

        uint32_t bos = 0;
        uint32_t eos = 0;

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GetChunkOffset(cuSeqlensTensor, chunkIndicesTensor, params.B, params.HV, params.T, params.BT,
                            loopIdx, bos, eos, params.isVarLen);
            uint32_t actual_chunk_len = eos - bos;
            uint32_t bIdx = loopIdx / params.numChunks;
            uint32_t chunkIdx = loopIdx % params.numChunks;
            uint64_t bos_hk =
                        bos - static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(params.HV - params.HK) * params.T;

            GemmCoord actualBlockShape1{actual_chunk_len, K, V};
            GemmCoord actualBlockShape2{actual_chunk_len, actual_chunk_len, K};
            GemmCoord actualBlockShape3{actual_chunk_len, actual_chunk_len, V};
            GemmCoord actualBlockShape4{actual_chunk_len, K, V};
            GemmCoord actualBlockShape6{actual_chunk_len, K, actual_chunk_len};
            GemmCoord actualBlockShape5{actual_chunk_len, K, V};
            GemmCoord actualBlockShape7{actual_chunk_len, K, actual_chunk_len};

            {
                BlockMmadPart1 blockMmadPart1(resource);
                for (uint32_t h = 0; h < params.HV; h++) {
                    uint64_t dvOffset = (h * params.T + bos) * params.V;
                    uint64_t hOffset = ((bIdx * params.numChunks + chunkIdx) * params.HV + h) * params.K * params.V;
                    uint64_t dwOffset = (h * params.T + bos) * params.K;

                    auto tensorDv = tla::MakeTensor(gmDv[dvOffset], MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[hOffset], MakeLayoutFromTag(layoutVxK), Arch::PositionGM{});
                    auto tensorDw = tla::MakeTensor(gmDw[dwOffset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});

                    auto tensorBlockDv = GetTile(tensorDv, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape1.m(), actualBlockShape1.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape1.k(), actualBlockShape1.n()));
                    auto tensorBlockDw = GetTile(tensorDw, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape1.m(), actualBlockShape1.n()));
                    blockMmadPart1(tensorBlockDv, tensorBlockH, tensorBlockDw, actualBlockShape1);
                }
            }
            SetCubeReady();
            WaitClear();
            {

                BlockMmadPart2 blockMmadPart2(resource);
                for (uint32_t h = 0; h < params.HV; h++) {

                    uint32_t hk_idx = h / params.n_ratio;
                    uint64_t qkOffset = (hk_idx * params.T + bos_hk) * params.K;
                    uint64_t mm3Offset = DqkwgBtbElemOffset(coreIdx, h, params.HV, params.BT);

                    auto tensorQ = tla::MakeTensor(gmQ[qkOffset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK[qkOffset], MakeLayoutFromTag(layoutKxBT), Arch::PositionGM{});
                    auto tensorMm3 = tla::MakeTensor(gmMm3[mm3Offset], MakeLayoutFromTag(layoutBTxBT), Arch::PositionGM{});

                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape2.m(), actualBlockShape2.k()));
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape2.k(), actualBlockShape2.n()));
                    auto tensorBlockMm3 = GetTile(tensorMm3, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape2.m(), actualBlockShape2.n()));

                    blockMmadPart2(tensorBlockQ, tensorBlockK, tensorBlockMm3, actualBlockShape2);
                }
            }

            {
                BlockMmadPart3 blockMmadPart3(resource);
                for (uint32_t h = 0; h < params.HV; h++) {
                    uint64_t dvOffset = (h * params.T + bos) * params.V;
                    uint64_t dsOffset = DqkwgBtbElemOffset(coreIdx, h, params.HV, params.BT);

                    auto tensorDo = tla::MakeTensor(gmDo[dvOffset], MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(gmV[dvOffset], MakeLayoutFromTag(layoutVxBT), Arch::PositionGM{});
                    auto tensorDs = tla::MakeTensor(gmDs[dsOffset], MakeLayoutFromTag(layoutBTxBT), Arch::PositionGM{});

                    auto tensorBlockDo = GetTile(tensorDo, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape3.m(), actualBlockShape3.k()));
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape3.k(), actualBlockShape3.n()));
                    auto tensorBlockDs = GetTile(tensorDs, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape3.m(), actualBlockShape3.n()));
                    blockMmadPart3(tensorBlockDo, tensorBlockV, tensorBlockDs, actualBlockShape3);
                }
            }
            SetCubeReady();
            WaitClear();
            {
                BlockMmadPart4 blockMmadPart4(resource);
                for (uint32_t h = 0; h < params.HV; h++) {

                    uint64_t doOffset = (h * params.T + bos) * params.V;
                    uint64_t hOffset = ((bIdx * params.numChunks + chunkIdx) * params.HV + h) * params.K * params.V;
                    uint64_t mm4Offset = DqkwgBtxKElemOffset(coreIdx, h, params.HV, params.BT, params.K);

                    auto tensorDo = tla::MakeTensor(gmDo[doOffset], MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[hOffset], MakeLayoutFromTag(layoutVxK), Arch::PositionGM{});
                    auto tensorMm4 = tla::MakeTensor(gmMm4[mm4Offset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});

                    auto tensorBlockDo = GetTile(tensorDo, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape4.m(), actualBlockShape4.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape4.k(), actualBlockShape4.n()));
                    auto tensorBlockMm4 = GetTile(tensorMm4, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape4.m(), actualBlockShape4.n()));
                    blockMmadPart4(tensorBlockDo, tensorBlockH, tensorBlockMm4, actualBlockShape4);
                }
            }
            WaitCredit();
            {
                BlockMmadPart6 blockMmadPart6(resource);
                for (uint32_t h = 0; h < params.HV; h++) {

                    uint32_t hk_idx = h / params.n_ratio;
                    uint64_t dsOffset = DqkwgBtbElemOffset(coreIdx, h, params.HV, params.BT);
                    uint64_t kOffset = (hk_idx * params.T + bos_hk) * params.K;
                    uint64_t mm6Offset = DqkwgBtxKElemOffset(coreIdx, h, params.HV, params.BT, params.K);

                    auto tensorDsTemp = tla::MakeTensor(gmDsTemp[dsOffset], MakeLayoutFromTag(layoutBTxBT), Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK[kOffset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                    auto tensorMm6 = tla::MakeTensor(gmMm6[mm6Offset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});

                    auto tensorBlockDsTemp = GetTile(tensorDsTemp, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape6.m(), actualBlockShape6.k()));
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape6.k(), actualBlockShape6.n()));
                    auto tensorBlockMm6 = GetTile(tensorMm6, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape6.m(), actualBlockShape6.n()));

                    blockMmadPart6(tensorBlockDsTemp, tensorBlockK, tensorBlockMm6, actualBlockShape6);
                }
            }
            SetCubeReady();
            WaitClear();
            {
                BlockMmadPart5 blockMmadPart5(resource);
                for (uint32_t h = 0; h < params.HV; h++) {

                    uint64_t vOffset = (h * params.T + bos) * params.V;
                    uint64_t dhOffset = ((bIdx * params.numChunks + chunkIdx) * params.HV + h) * params.K * params.V;
                    uint64_t mm5Offset = DqkwgBtxKElemOffset(coreIdx, h, params.HV, params.BT, params.K);

                    auto tensorV = tla::MakeTensor(gmV[vOffset], MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                    auto tensorDh = tla::MakeTensor(gmDh[dhOffset], MakeLayoutFromTag(layoutVxK), Arch::PositionGM{});
                    auto tensorMm5 = tla::MakeTensor(gmMm5[mm5Offset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});

                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape5.m(), actualBlockShape5.k()));
                    auto tensorBlockDh = GetTile(tensorDh, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape5.k(), actualBlockShape5.n()));
                    auto tensorBlockMm5 = GetTile(tensorMm5, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape5.m(), actualBlockShape5.n()));
                    blockMmadPart5(tensorBlockV, tensorBlockDh, tensorBlockMm5, actualBlockShape5);
                }
            }
            {
                BlockMmadPart7 blockMmadPart7(resource);
                for (uint32_t h = 0; h < params.HV; h++) {

                    uint32_t hk_idx = h / params.n_ratio;
                    uint64_t dsOffset = DqkwgBtbElemOffset(coreIdx, h, params.HV, params.BT);
                    uint64_t qOffset = (hk_idx * params.T + bos_hk) * params.K;
                    uint64_t mm7Offset = DqkwgBtxKElemOffset(coreIdx, h, params.HV, params.BT, params.K);

                    auto tensorDsTemp =tla::MakeTensor(gmDsTemp[dsOffset], MakeLayoutFromTag(layoutBTxBT_T), Arch::PositionGM{});
                    auto tensorQ = tla::MakeTensor(gmQ[qOffset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                    auto tensorMm7 = tla::MakeTensor(gmMm7[mm7Offset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});

                    auto tensorBlockDsTemp = GetTile(tensorDsTemp, tla::MakeCoord(0, 0),
                                                        tla::MakeShape(actualBlockShape7.m(), actualBlockShape7.k()));
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape7.k(), actualBlockShape7.n()));
                    auto tensorBlockMm7 = GetTile(tensorMm7, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape7.m(), actualBlockShape7.n()));
                    blockMmadPart7(tensorBlockDsTemp, tensorBlockQ, tensorBlockMm7, actualBlockShape7);
                }
            }
            SetCubeReady();
        }

        WaitClear();
        WaitClear();
        WaitClear();
    }
};

}

template <typename DataType, typename GType>
class ChunkBwdDqkwgCubeProcess {
public:
    __aicore__ inline ChunkBwdDqkwgCubeProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR do_, GM_ADDR dh,
                                               GM_ADDR dv, GM_ADDR cu_seqlen, GM_ADDR chunk_indices, GM_ADDR dq,
                                               GM_ADDR dk, GM_ADDR dw,GM_ADDR workspace)
        : ptrQ(q), ptrK(k), ptrV(v), ptrH(h), ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLen(cu_seqlen),
          ptrChunkIndices(chunk_indices), ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrWorkspace(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkBwdDqkwgTilingData &tiling);
    __aicore__ inline void Process();

private:

    GM_ADDR ptrQ;
    GM_ADDR ptrK;
    GM_ADDR ptrV;
    GM_ADDR ptrH;
    GM_ADDR ptrDo;
    GM_ADDR ptrDh;
    GM_ADDR ptrDv;
    GM_ADDR ptrCuSeqLen;
    GM_ADDR ptrChunkIndices;
    GM_ADDR ptrDq;
    GM_ADDR ptrDk;
    GM_ADDR ptrDw;
    GM_ADDR ptrWorkspace;

    uint64_t B;
    uint64_t HV;
    uint64_t HK;
    uint64_t n_ratio = 1;
    uint64_t T;
    uint64_t K;
    uint64_t V;
    uint64_t BT;
    uint64_t numChunks;
    uint64_t aicCoreNum;
    bool isVarLen;

    uint64_t wsMm3Offset;
    uint64_t wsMm4Offset;
    uint64_t wsMm6Offset;
    uint64_t wsMm5Offset;
    uint64_t wsMm7Offset;
    uint64_t wsDsTempOffset;
};

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgCubeProcess<DataType, GType>::Init(const ChunkBwdDqkwgTilingData &tiling)
{
    B = tiling.B;
    HV = tiling.HV;
    HK = tiling.HK;
    n_ratio = (HK > 0) ? (HV / HK) : 1;
    T = tiling.T;
    K = tiling.K;
    V = tiling.V;
    BT = tiling.BT;
    numChunks = tiling.numChunks;
    aicCoreNum = tiling.aicCoreNum;
    isVarLen = (tiling.isVarLen == 1);
    wsMm3Offset = tiling.wsMm3Offset;
    wsMm4Offset = tiling.wsMm4Offset;
    wsMm6Offset = tiling.wsMm6Offset;
    wsMm5Offset = tiling.wsMm5Offset;
    wsMm7Offset = tiling.wsMm7Offset;
    wsDsTempOffset = tiling.wsDsTempOffset;
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgCubeProcess<DataType, GType>::Process()
{
    using namespace Catlass;
    using namespace Catlass::Gemm;

    using LayoutRowMajor = layout::RowMajor;
    using LayoutColMajor = layout::ColumnMajor;

    using ArchTag = Arch::Ascend960DT;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = tla::Shape<_128, _128, _256>;
    using L0TileShape = tla::Shape<_128, _128, _128>;

    using TileCopyPart1Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart1Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart1Tiled>;
    using TileCopyPart2Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart2Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart2Tiled>;
    using TileCopyPart3Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart3Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart3Tiled>;
    using TileCopyPart4Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart4Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart4Tiled>;
    using TileCopyPart5Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart5Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart5Tiled>;
    using TileCopyPart6Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutRowMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart6Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart6Tiled>;
    using TileCopyPart7Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutColMajor, DataType,
                                                             LayoutRowMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart7Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart7Tiled>;
    using MatmulKernelTiled =
        Kernel::ChunkBwdDqkwgTla<BlockMmadPart1Tiled, BlockMmadPart2Tiled, BlockMmadPart3Tiled, BlockMmadPart4Tiled,
                                 BlockMmadPart5Tiled, BlockMmadPart6Tiled, BlockMmadPart7Tiled>;

    MatmulKernelTiled kernel;
    typename MatmulKernelTiled::Params params(ptrQ, ptrK, ptrV, ptrH, ptrDo, ptrDh, ptrDv, ptrCuSeqLen,
                                                ptrChunkIndices, ptrDq, ptrDk, ptrDw, ptrWorkspace, B, HV, HK, T, K, V, BT,
                                                numChunks, n_ratio, isVarLen, wsMm3Offset, wsMm4Offset, wsMm6Offset,
                                                wsMm5Offset, wsMm7Offset, wsDsTempOffset);
    params.aicCoreNum = aicCoreNum;
    kernel(params);
}

#endif
