/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_w_u_fwd.h
 * \brief
 */


#ifndef RECOMPUTE_W_U_FWD_CUBE_H
#define RECOMPUTE_W_U_FWD_CUBE_H

#include "recompute_w_u_fwd_struct.h"
#include <type_traits>

using GDN::GdnMegaArch22RecomputeWUTilingData;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
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
template <class BlockMmadU_, class BlockMmadW_, bool kFlattenHeadTasks_ = false,
          bool kAbcTaskOrder_ = false, int kStaticV = 0, int kStaticBT = 0>
class RecomputeWUFwdTla {
public:
    using BlockMmadU = BlockMmadU_;
    using BlockMmadW = BlockMmadW_;

    using ArchTag = typename BlockMmadU::ArchTag;
    using BdkL1TileShape = typename BlockMmadU::L1TileShape;
    using BdkbL1TileShape = typename BlockMmadU::L1TileShape;

    using ElementA = typename BlockMmadU::ElementA;
    using LayoutA = typename BlockMmadU::LayoutA;
    using ElementVb = typename BlockMmadU::ElementB;
    using LayoutVb = typename BlockMmadU::LayoutB;
    using ElementU = typename BlockMmadU::ElementC;
    using LayoutU = typename BlockMmadU::LayoutC;

    using ElementKbgExp = typename BlockMmadW::ElementB;
    using LayoutKbgExp = typename BlockMmadW::LayoutB;
    using ElementW = typename BlockMmadW::ElementC;
    using LayoutW = typename BlockMmadW::LayoutC;
    static constexpr bool kFlattenHeadTasks = kFlattenHeadTasks_;
    static constexpr bool kAbcTaskOrder = kAbcTaskOrder_;
    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrVb;
        LayoutVb layoutVb;
        GM_ADDR ptrU;
        LayoutU layoutU;
        GM_ADDR ptrKbgExp;
        LayoutKbgExp layoutKbgExp;
        GM_ADDR ptrW;
        LayoutW layoutW;
        GM_ADDR ptrCuSeqLens;
        GM_ADDR ptrChunkIndices;
        uint64_t chunkNum;
        uint64_t B = 1;
        uint64_t Hk = 1;
        uint64_t Hv = 1;
        uint64_t hvPerHk = 1;
        uint64_t T = 32768;
        uint64_t K = 128;
        __aicore__ inline uint64_t ValueDim() const
        {
            if constexpr (kStaticV != 0) {
                return static_cast<uint64_t>(kStaticV);
            } else {
                return VRuntime_;
            }
        }

        uint64_t VRuntime_ = 128;
        __aicore__ inline uint64_t ChunkSize() const
        {
            if constexpr (kStaticBT != 0) {
                return static_cast<uint64_t>(kStaticBT);
            } else {
                return chunkSizeRuntime_;
            }
        }

        uint64_t chunkSizeRuntime_ = 64;
        const GDN::RecomputeTaskRange *taskRange = nullptr;

        // Methods
        CATLASS_DEVICE
        Params()
        {
        }

        CATLASS_DEVICE
        Params(GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrVb_, LayoutVb layoutVb_, GM_ADDR ptrU_,
               LayoutU layoutU_, GM_ADDR ptrKbgExp_, LayoutKbgExp layoutKbgExp_, GM_ADDR ptrW_, LayoutW layoutW_,
               GM_ADDR ptrCuSeqLens_, GM_ADDR ptrChunkIndices_, uint64_t chunkNum_, uint64_t B_,
               uint64_t Hk_, uint64_t Hv_, uint64_t hvPerHk_, uint64_t T_, uint64_t K_, uint64_t V_, uint64_t BT_,
               const GDN::RecomputeTaskRange *range_ = nullptr)
            : ptrA(ptrA_), layoutA(layoutA_), ptrVb(ptrVb_), layoutVb(layoutVb_), ptrU(ptrU_),
              layoutU(layoutU_), ptrKbgExp(ptrKbgExp_), layoutKbgExp(layoutKbgExp_), ptrW(ptrW_), layoutW(layoutW_),
              ptrCuSeqLens(ptrCuSeqLens_), ptrChunkIndices(ptrChunkIndices_),
              chunkNum(chunkNum_), B(B_), Hk(Hk_), Hv(Hv_), hvPerHk(hvPerHk_), T(T_), K(K_), VRuntime_(V_), chunkSizeRuntime_(BT_), taskRange(range_)
        {
        }
    };

    CATLASS_DEVICE void RunInterleaved(Arch::Resource<ArchTag> &resource, Params const &params)
    {
        uint32_t bos = 0;
        uint32_t eos = 0;
        uint32_t taskIdx = 0;
        BlockMmadU blockMmad(resource);
        blockMmad.preSetFlags();
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementVb> gmVb;
        AscendC::GlobalTensor<ElementU> gmU;
        AscendC::GlobalTensor<ElementKbgExp> gmKbgExp;
        AscendC::GlobalTensor<ElementW> gmW;
        const uint32_t loopBegin = static_cast<uint32_t>(params.taskRange->begin);
        const uint32_t loopEnd = static_cast<uint32_t>(params.taskRange->end);
        for (uint32_t loopIdx = loopBegin; loopIdx < loopEnd; ++loopIdx) {
            uint32_t chunkIdx = 0;
            uint32_t hBegin = 0;
            uint32_t hEnd = 0;
            DecodeRecomputeTask<kFlattenHeadTasks, kAbcTaskOrder>(
                loopIdx, params.ptrCuSeqLens, params.Hv, params.T, params.ChunkSize(),
                params.chunkNum, params.hvPerHk, chunkIdx, hBegin, hEnd);
            GetChunkOffset(params.ptrCuSeqLens, params.ptrChunkIndices, params.B, params.Hv, params.T,
                            params.ChunkSize(), chunkIdx, bos, eos);
            uint32_t curChunkSize = eos - bos;
            GemmCoord blockCoord{0, 0, 0};
            for (uint32_t h = hBegin; h < hEnd; ++h) {
                ++taskIdx;
                // 设置 A 矩阵（U 和 W 共享同一个 A）
                gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA + (h * params.T + bos) * params.ChunkSize());
                auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});

                // === 第一部分：U = A @ Vb ===
                if ((taskIdx & 0b11) == 1) {
                    AscendC::CrossCoreWaitFlag<0x2>(SYNC_AIV_AIC_FLAG_3);
                    if (taskIdx > GDN::RECOMPUTE_WU_SYNC_WINDOW_TASKS) {
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG_5);
                    }
                }
                uint32_t tileN = tla::get<1>(BdkL1TileShape{});
                for (uint32_t nOffset = 0; nOffset < params.ValueDim(); nOffset += tileN) {
                    uint32_t curN = (nOffset + tileN > params.ValueDim()) ? (params.ValueDim() - nOffset) : tileN;
                    GemmCoord actualBlockShape{curChunkSize, curN, curChunkSize};
                    const uint64_t vbOffset = (loopIdx - params.taskRange->begin) * params.ChunkSize() * params.ValueDim();
                    gmVb.SetGlobalBuffer((__gm__ ElementVb *)params.ptrVb + vbOffset + nOffset);
                    gmU.SetGlobalBuffer((__gm__ ElementU *)params.ptrU + (h * params.T + bos) * params.ValueDim() + nOffset);

                    auto tensorVb = tla::MakeTensor(gmVb, params.layoutVb, Arch::PositionGM{});
                    auto tensorU = tla::MakeTensor(gmU, params.layoutU, Arch::PositionGM{});
                    // Make tiled views
                    auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockVb = GetTile(tensorVb, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockU = GetTile(tensorU, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add
                    blockMmad(tensorBlockA, tensorBlockVb, tensorBlockU, actualBlockShape);
                }

                // W follows U on the same MMAD object; two L1A slots may still reload A.
                // 窗口通知已覆盖本窗口全部任务的 Vb+KbgExp，无需再等待。
                GemmCoord actualBlockShape{curChunkSize, static_cast<uint32_t>(params.K), curChunkSize};
                const uint64_t kbgOffset = (loopIdx - params.taskRange->begin) * params.ChunkSize() * params.K;
                gmKbgExp.SetGlobalBuffer((__gm__ ElementKbgExp *)params.ptrKbgExp + kbgOffset);
                gmW.SetGlobalBuffer((__gm__ ElementW *)params.ptrW + (h * params.T + bos) * params.K);

                auto tensorKbgExp = tla::MakeTensor(gmKbgExp, params.layoutKbgExp, Arch::PositionGM{});
                auto tensorW = tla::MakeTensor(gmW, params.layoutW, Arch::PositionGM{});

                auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockKbgExp = GetTile(tensorKbgExp, tla::MakeCoord(0, 0),
                                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                blockMmad(tensorBlockA, tensorBlockKbgExp, tensorBlockW, actualBlockShape);
                // Parent batches may reuse the GM temporary addresses; invalidate per task.
                blockMmad.RestoreStatus();
            }
        }
        blockMmad.finalWaitFlags();
    }

    // Methods
    CATLASS_DEVICE
    RecomputeWUFwdTla()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Arch::Resource<ArchTag> &resource, Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Arch::Resource<ArchTag> &resource, Params const &params)
    {
        RunInterleaved(resource, params);
    }
};
} // namespace Catlass::Gemm::Kernel

template <class... Dims>
using GemmCubeTileShape = tla::Shape<Dims...>;

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape,
          bool kFlattenHeadTasks = false, bool kAbcTaskOrder = false,
          int kStaticV = 0, int kStaticBT = 0>
class RecomputeWUFwdProcess {
public:
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using Resource = Arch::Resource<Arch::Ascend950>;
#else
    using Resource = Arch::Resource<Arch::AtlasA2>;
#endif
    /** @brief constructor */
    __aicore__ inline RecomputeWUFwdProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
                                                        GM_ADDR workspace_);

    __aicore__ inline void Process(Resource &resource, const GDN::RecomputeTaskRange *taskRange = nullptr);

    __aicore__ inline void Init(const GdnMegaArch22RecomputeWUTilingData &tiling);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t Hv = 1;
    uint64_t Hk = 1;
    uint64_t hvPerHk = 1;
    uint64_t K = 0;
    __aicore__ inline uint64_t ValueDim() const
    {
        if constexpr (kStaticV != 0) {
            return static_cast<uint64_t>(kStaticV);
        } else {
            return VRuntime_;
        }
    }

    uint64_t VRuntime_ = 0;
    __aicore__ inline uint64_t ChunkSize() const
    {
        if constexpr (kStaticBT != 0) {
            return static_cast<uint64_t>(kStaticBT);
        } else {
            return chunkSizeRuntime_;
        }
    }

    uint64_t chunkSizeRuntime_ = 0;
    uint64_t chunkNum;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR workspace;
};

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape,
          bool kFlattenHeadTasks, bool kAbcTaskOrder, int kStaticV, int kStaticBT>
__aicore__ inline RecomputeWUFwdProcess<kType, betaType, L1TileShape, L0TileShape,
                                        kFlattenHeadTasks, kAbcTaskOrder, kStaticV, kStaticBT>::RecomputeWUFwdProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_){};

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape,
          bool kFlattenHeadTasks, bool kAbcTaskOrder, int kStaticV, int kStaticBT>
__aicore__ void inline RecomputeWUFwdProcess<kType, betaType, L1TileShape, L0TileShape,
                                             kFlattenHeadTasks, kAbcTaskOrder, kStaticV, kStaticBT>::Init(
    const GdnMegaArch22RecomputeWUTilingData &tiling)
{
    B = tiling.B;
    T = tiling.T;
    Hv = tiling.Hv;
    Hk = tiling.Hk;
    hvPerHk = tiling.hvPerHk;
    K = tiling.K;
    if constexpr (kStaticV == 0) {
        VRuntime_ = tiling.V;
    }
    if constexpr (kStaticBT == 0) {
        chunkSizeRuntime_ = tiling.chunkSize;
    }
    chunkNum = tiling.chunkNum;
    return;
}

template <typename kType, typename betaType, typename L1TileShape, typename L0TileShape,
          bool kFlattenHeadTasks, bool kAbcTaskOrder, int kStaticV, int kStaticBT>
__aicore__ void inline RecomputeWUFwdProcess<kType, betaType, L1TileShape, L0TileShape,
                                             kFlattenHeadTasks, kAbcTaskOrder, kStaticV, kStaticBT>::Process(
    Resource &resource, const GDN::RecomputeTaskRange *taskRange)
{
    //输入
    using LayoutTagA = layout::RowMajor;
    using LayoutTagKbgExp = layout::RowMajor;
    using LayoutTagVb = layout::RowMajor;
    using LayoutTagW = layout::RowMajor;
    using LayoutTagU = layout::RowMajor;

    //输入
    LayoutTagA tagA = LayoutTagA::MakeLayout<kType>(ChunkSize(), ChunkSize());
    LayoutTagKbgExp tagKbgExp = LayoutTagKbgExp::MakeLayout<kType>(ChunkSize(), K);
    LayoutTagVb tagVb = LayoutTagVb::MakeLayout<kType>(ChunkSize(), ValueDim());

    //输出
    LayoutTagW tagW = LayoutTagW::MakeLayout<kType>(ChunkSize(), K);
    LayoutTagU tagU = LayoutTagU::MakeLayout<kType>(ChunkSize(), ValueDim());

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Arch::Ascend950;
    // 950: UnitFlag=true 时 L0C_STAGES 必须为 1（见 block_mmad_pingpong_tla.hpp static_assert）
    using DispatchPolicy = Gemm::MmadPingpongTlaMulti<ArchTag, true, false, 1, true>;
#else
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpongTlaMulti<ArchTag, true, false, 1, true>;
#endif

    //计算U
    using TileCopyU =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagA, kType, LayoutTagVb, kType, LayoutTagU>;
    using BlockMmadU =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyU>;
    //计算W
    using TileCopyW =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagA, kType, LayoutTagKbgExp, kType, LayoutTagW>;
    using BlockMmadW =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyW>;

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutVb = MakeLayoutFromTag(tagVb);
    auto layoutU = MakeLayoutFromTag(tagU);

    auto layoutKbgExp = MakeLayoutFromTag(tagKbgExp);
    auto layoutW = MakeLayoutFromTag(tagW);
    
    // kernel level
    using MatmulKernel = Gemm::Kernel::RecomputeWUFwdTla<
        BlockMmadU, BlockMmadW, kFlattenHeadTasks, kAbcTaskOrder, kStaticV, kStaticBT>;

    MatmulKernel kernel;

    GM_ADDR vb = workspace;
    GM_ADDR kbgExp = taskRange == nullptr ? workspace :
        workspace + taskRange->capacity * ChunkSize() * ValueDim() * sizeof(kType);
    typename MatmulKernel::Params param{
        A, layoutA, vb, layoutVb, u,        layoutU,  
        kbgExp, layoutKbgExp, w,        layoutW,
        cu_seqlens, chunk_indices, chunkNum, B,
        Hk, Hv, hvPerHk, T, K, ValueDim(), ChunkSize(), taskRange};
    kernel(resource, param);
}


#endif // RECOMPUTE_W_U_FWD_CUBE_H
