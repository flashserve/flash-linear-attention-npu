/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PREPARE_WY_REPR_BWD_FULL_TLA_HPP
#define PREPARE_WY_REPR_BWD_FULL_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmadBdkb_,
    class BlockMmadBdkbg_,
    class BlockMmadBkkT_,
    class BlockMmadBdvb_,
    class BlockMmadBdk_,
    class BlockScheduler_,
    typename ElementBeta,
    typename LayoutBeta
>
class PrepareWyReprBwdFullTla {
public:
    using BlockMmadBdkb = BlockMmadBdkb_;
    using BlockMmadBdkbg = BlockMmadBdkbg_;
    using BlockMmadBkkT = BlockMmadBkkT_;
    using BlockMmadBdvb = BlockMmadBdvb_;
    using BlockMmadBdk = BlockMmadBdk_;
    using ArchTag = typename BlockMmadBdkb::ArchTag;
    using BdkbL1TileShape = typename BlockMmadBdkb::L1TileShape;
    using ElementA = typename BlockMmadBdkb::ElementA;
    using LayoutDA = typename BlockMmadBdkb::LayoutA;
    using ElementK = typename BlockMmadBdkb::ElementB;
    using LayoutK = typename BlockMmadBdkb::LayoutB;
    using ElementC = typename BlockMmadBdkb::ElementC;
    using LayoutDkb = typename BlockMmadBdkb::LayoutC;
    using ElementAccumulator = typename BlockMmadBdkb::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_BDKB_M = tla::get<0>(BdkbL1TileShape{});
    static constexpr uint32_t L1_TILE_BDKB_N = tla::get<1>(BdkbL1TileShape{});
    static constexpr uint32_t L1_TILE_BDKB_K = tla::get<2>(BdkbL1TileShape{});
    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR ptrDA; LayoutDA layoutDA;
        GM_ADDR ptrK; LayoutK layoutK;
        GM_ADDR ptrBeta; LayoutBeta layoutBeta;
        GM_ADDR ptrWorkspace; LayoutDkb layoutDkb;
        uint64_t B = 1;
        uint64_t T = 32768;
        uint64_t H = 32;
        uint64_t K = 128;
        uint64_t V = 128;
        uint64_t BT = 64;
        uint64_t stage = 4;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptrDA_,const LayoutDA &layoutDA_,GM_ADDR ptrK_,const LayoutK &layoutK_,GM_ADDR ptrBeta_,const LayoutBeta &layoutBeta_,GM_ADDR ptrWorkspace_,const LayoutDkb &layoutDkb_,
        uint64_t B_, uint64_t T_,uint64_t H_,uint64_t K_,uint64_t V_,uint64_t BT_, uint64_t stage_)
            : ptrDA(ptrDA_), layoutDA(layoutDA_), ptrK(ptrK_), layoutK(layoutK_), ptrBeta(ptrBeta_), layoutBeta(layoutBeta_), ptrWorkspace(ptrWorkspace_),
              layoutDkb(layoutDkb_), B(B_), T(T_), H(H_), K(K_), V(V_), BT(BT_), stage(stage_){}
    };

    struct Arguments {
        GM_ADDR ptrDA; LayoutDA layoutDA;
        GM_ADDR ptrK; LayoutK layoutK;
        GM_ADDR ptrBeta; LayoutBeta layoutBeta;
        GM_ADDR ptrWorkspace; LayoutDkb layoutDkb;
        uint64_t B = 1;
        uint64_t T = 32768;
        uint64_t H = 32;
        uint64_t K = 128;
        uint64_t V = 128;
        uint64_t BT = 64;
        uint64_t stage = 4;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        // AscendC::print("in GetWorkspaceSize\n");
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {

        Params params{args.ptrDA, args.layoutDA, args.ptrK, args.layoutK,args.ptrBeta, args.layoutBeta,args.ptrWorkspace, args.layoutDkb,args.B, args.T, args.H, args.K, args.V, args.BT, args.stage};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    PrepareWyReprBwdFullTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {
        GemmCoord ProblemShapeDkb{static_cast<uint32_t>(params.T),static_cast<uint32_t>(params.K), static_cast<uint32_t>(params.BT)};
        AscendC::printf("ProblemShapeDkb.m:%u ProblemShapeDkb.n:%u ProblemShapeDkb.k:%u\n", ProblemShapeDkb.m(), ProblemShapeDkb.n(), ProblemShapeDkb.k());
        BlockScheduler matmulBlockSchedulerDkb(ProblemShapeDkb, MakeCoord(L1_TILE_BDKB_M, L1_TILE_BDKB_N));
        uint32_t coreLoopsInB = matmulBlockSchedulerDkb.GetCoreLoops();
        uint32_t coreLoops = params.B * CeilDiv(params.T, params.BT);
        // uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
        Arch::Resource<ArchTag> resource;
        BlockMmadBdkb blockMmadBdkb(resource);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t bIdx = loopIdx / coreLoopsInB;
            GemmCoord blockCoord = matmulBlockSchedulerDkb.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockSchedulerDkb.GetActualBlockShape(blockCoord);
            for (int h = 0; h < params.H; h++) {
                // Represent the full gm
                AscendC::GlobalTensor<ElementA> gmDA;
                gmDA.SetGlobalBuffer((__gm__ ElementA *)params.ptrDA + ((bIdx * params.H + h) * params.T * params.BT));
                AscendC::GlobalTensor<ElementK> gmK;
                gmK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK + ((bIdx * params.H + h) * params.T * params.K));
                AscendC::GlobalTensor<ElementC> gmWorkspaceDkb;
                gmWorkspaceDkb.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace + ((bIdx * params.H + h) * params.T * params.K));

                // Represent the full tensors
                auto tensorDA = tla::MakeTensor(gmDA, params.layoutDA, Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
                auto tensorDkb = tla::MakeTensor(gmWorkspaceDkb, params.layoutDkb, Arch::PositionGM{});


                // Make tiled views
                auto tensorBlockDA = GetTile(tensorDA,
                                            tla::MakeCoord(blockCoord.m() * L1_TILE_BDKB_M, blockCoord.k() * L1_TILE_BDKB_K),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockK = GetTile(tensorK,
                                            tla::MakeCoord(blockCoord.k() * L1_TILE_BDKB_K, blockCoord.n() * L1_TILE_BDKB_N),
                                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockDkb = GetTile(tensorDkb,
                                            tla::MakeCoord(blockCoord.m() * L1_TILE_BDKB_M, blockCoord.n() * L1_TILE_BDKB_N),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                // Compute block-scoped matrix multiply-add
                blockMmadBdkb(tensorBlockDA, tensorBlockK, tensorBlockDkb, actualBlockShape);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
                AscendC::printf("CrossCoreSetFlag\n");
            }
        }

        // AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}
};

} // namespace Catlass::Gemm::Kernel

#endif // PREPARE_WY_REPR_BWD_FULL_TLA_HPP