/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*!
 * \file recurrent_kda_cube.h
 * \brief Ascend 950 Cube path for the recurrent KDA mixed kernel.
 */

#ifndef RECURRENT_KDA_CUBE_H
#define RECURRENT_KDA_CUBE_H

#define CATLASS_ARCH 3510
#include "kernel_operator.h"
#include "recurrent_kda_common.h"
#include "recurrent_kda_tiling_data_apt.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace RecurrentKda {

using namespace Catlass;
using namespace tla;

class RKDACube {
public:
    __aicore__ inline RKDACube(GM_ADDR cuSeqlens, GM_ADDR workspace,
                              const RecurrentKdaTilingDataA5 *tilingData)
        : cuSeqlens_(cuSeqlens), workspace_(workspace)
    {
        const RecurrentKdaTilingData &base = tilingData->base;
        B_ = base.b;
        T_ = base.t;
        seqLen_ = base.seqLen;
        NV_ = base.nv;
        realK_ = base.dk;
        realV_ = base.dv;
        alignK_ = (realK_ + 15) / 16 * 16;
        vStep_ = tilingData->cubeVecRows;
        workspaceStride_ = tilingData->workspaceStride;
        hasCuSeqlens_ = base.hasCuSeqlens == 1;
        cuSeqlensDtype_ = base.cuSeqlensDtype;
        blockIdx_ = AscendC::GetBlockIdx();
        cuSeqlensInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens_);
        cuSeqlensInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens_);
    }

    __aicore__ inline void Process()
    {
        using ArchTag = Catlass::Arch::Ascend950;
        using LayoutTagQk = Catlass::layout::RowMajor;
        using LayoutTagState = Catlass::layout::ColumnMajor;
        using LayoutTagOut = Catlass::layout::RowMajor;
        using DispatchPolicy = Common::MmadPingpong<ArchTag, false, false, 2>;
        using L1TileShape = Shape<_128, _128, _128>;
        using L0TileShape = Shape<_128, _128, _128>;
        using TileCopy = Common::Tile::PackedTileCopyTlaToUB<
            ArchTag, bfloat16_t, LayoutTagQk, bfloat16_t, LayoutTagState, float, LayoutTagOut, void,
            Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
        using BlockMmad = Common::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape, bfloat16_t, bfloat16_t, float, void, TileCopy>;

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        LayoutTagQk tagQk = LayoutTagQk::MakeLayout<bfloat16_t>(RKDA_CUBE_M, alignK_);
        LayoutTagState tagState = LayoutTagState::MakeLayout<bfloat16_t>(alignK_, vStep_);
        LayoutTagOut tagOut = LayoutTagOut::MakeLayout<float>(RKDA_CUBE_M, vStep_);
        auto layoutQk = MakeLayoutFromTag(tagQk);
        auto layoutState = MakeLayoutFromTag(tagState);
        auto layoutOut = MakeLayoutFromTag(tagOut);

        __gm__ bfloat16_t *coreWorkspace =
            reinterpret_cast<__gm__ bfloat16_t *>(workspace_) +
            blockIdx_ * workspaceStride_ / sizeof(bfloat16_t);
        uint32_t ubListId = 0;
        AscendC::LocalTensor<float> ubOut = resource.ubBuf.GetBufferByByte<float>(0);
        auto tensorOut = tla::MakeTensor(ubOut, layoutOut, Arch::PositionUB{});
        auto tensorOutTile =
            GetTile(tensorOut, tla::MakeCoord(0, 0), tla::MakeShape(RKDA_CUBE_M, vStep_));
        using UBTensor = decltype(tensorOutTile);
        UBTensor tensorOutList[RKDA_CV_BUFFER_NUM] = {tensorOutTile};

        AscendC::GlobalTensor<bfloat16_t> gmQk;
        gmQk.SetGlobalBuffer(coreWorkspace + 2 * vStep_ * alignK_);
        auto tensorQk = tla::MakeTensor(gmQk, layoutQk, Arch::PositionGM{});

        for (uint64_t batch = 0; batch < B_; ++batch) {
            const int64_t seq0 = SequenceStart(batch);
            const int64_t seq1 = SequenceEnd(batch);
            if (seq1 <= seq0) {
                continue;
            }
            for (uint64_t head = 0; head < NV_; ++head) {
                if (!IsCurrentTask(batch, head)) {
                    continue;
                }
                for (uint32_t vBase = 0; vBase < realV_; vBase += 2 * vStep_) {
                    for (int64_t token = seq0; token < seq1; ++token) {
                        (void)token;
                        Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(stateReadyFlag_);
                        for (uint8_t subBlock = 0; subBlock < 2; ++subBlock) {
                            const uint32_t vOffset = vBase + subBlock * vStep_;
                            if (vOffset >= realV_) {
                                continue;
                            }
                            const uint32_t rows = min(vStep_, realV_ - vOffset);
                            AscendC::GlobalTensor<bfloat16_t> gmState;
                            gmState.SetGlobalBuffer(coreWorkspace + subBlock * vStep_ * alignK_);
                            auto tensorState = tla::MakeTensor(gmState, layoutState, Arch::PositionGM{});
                            GemmCoord actualShape{RKDA_CUBE_M, rows, alignK_};
                            auto tensorQkTile =
                                GetTile(tensorQk, tla::MakeCoord(0, 0),
                                        tla::MakeShape(RKDA_CUBE_M, alignK_));
                            auto tensorStateTile =
                                GetTile(tensorState, tla::MakeCoord(0, 0), tla::MakeShape(alignK_, rows));
                            blockMmad(tensorQkTile, tensorStateTile, tensorOutList, actualShape, RKDA_CUBE_M, subBlock,
                                      RKDA_L0C_FREE_FLAG, RKDA_L0C_READY_FLAG, ubListId, 1,
                                      RKDA_CV_BUFFER_NUM);
                        }
                        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(RKDA_STATE_FREE_FLAG);
                        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
                            RKDA_STATE_FREE_FLAG + RKDA_FLAG_ID_MAX);
                    }
                }
            }
        }

        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(RKDA_L0C_FREE_FLAG);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(RKDA_L0C_FREE_FLAG + RKDA_FLAG_ID_MAX);
    }

private:
    __aicore__ inline int64_t LoadCuSeqlens(uint64_t index) const
    {
        return cuSeqlensDtype_ == 0 ? static_cast<int64_t>(cuSeqlensInt32Gm_.GetValue(index)) :
                                     cuSeqlensInt64Gm_.GetValue(index);
    }

    __aicore__ inline int64_t SequenceStart(uint64_t batch) const
    {
        return hasCuSeqlens_ ? LoadCuSeqlens(batch) : static_cast<int64_t>(batch * seqLen_);
    }

    __aicore__ inline int64_t SequenceEnd(uint64_t batch) const
    {
        return hasCuSeqlens_ ? LoadCuSeqlens(batch + 1) :
                               static_cast<int64_t>((batch + 1) * seqLen_);
    }

    __aicore__ inline bool IsCurrentTask(uint64_t batch, uint64_t head) const
    {
        return ((batch * NV_ + head) % AscendC::GetBlockNum()) == blockIdx_;
    }



    AscendC::GlobalTensor<int32_t> cuSeqlensInt32Gm_;
    AscendC::GlobalTensor<int64_t> cuSeqlensInt64Gm_;
    GM_ADDR cuSeqlens_;
    Arch::CrossCoreFlagWithReverse<> stateReadyFlag_{RKDA_STATE_READY_FLAG, RKDA_STATE_READY_REVERSE_FLAG};
    GM_ADDR workspace_;
    uint32_t B_;
    uint32_t T_;
    uint32_t seqLen_;
    uint32_t NV_;
    uint32_t realK_;
    uint32_t realV_;
    uint32_t alignK_;
    uint32_t vStep_;
    uint32_t workspaceStride_;
    uint64_t blockIdx_;
    uint32_t cuSeqlensDtype_;
    bool hasCuSeqlens_;
};

} // namespace RecurrentKda

#endif // RECURRENT_KDA_CUBE_H
